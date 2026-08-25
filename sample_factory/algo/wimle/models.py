from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace


class SquashedNormal:
    """Diagonal Gaussian transformed by tanh, as used by the WIMLE actor."""

    def __init__(self, mean: Tensor, log_std: Tensor):
        self.mean = mean
        self.log_std = -10.0 + 12.0 * 0.5 * (1.0 + torch.tanh(log_std))
        self.std = self.log_std.exp()
        self.base = torch.distributions.Normal(self.mean, self.std)

    @property
    def means(self) -> Tensor:
        return torch.tanh(self.mean)

    def rsample(self, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(
            self.mean.shape,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        return torch.tanh(self.mean + self.std * noise)

    def log_prob(self, actions: Tensor) -> Tensor:
        clipped = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(clipped)
        correction = torch.log(1.0 - clipped.square() + 1e-6)
        return (self.base.log_prob(pre_tanh) - correction).sum(dim=-1)


def _init_linear(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class _BroNet(nn.Module):
    """BRO residual MLP used by the actor and quantile critics."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, output_dim: int | None = None):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            for _ in range(depth)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim) if output_dim is not None else None
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _init_linear(module)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.input_norm(self.input_layer(x)))
        for first, first_norm, second, second_norm in self.blocks:
            residual = torch.relu(first_norm(first(x)))
            residual = second_norm(second(residual))
            x = x + residual
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x


class _PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.trunk = _BroNet(obs_dim, 256, depth=1)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        _init_linear(self.mean)
        _init_linear(self.log_std, gain=1.0)

    def forward(self, obs: Tensor) -> Tensor:
        features = self.trunk(obs)
        return torch.cat((self.mean(features), self.log_std(features)), dim=-1)


class WIMLEActorCritic(ActorCritic):
    """SF policy contract backed by WIMLE's tanh Gaussian actor."""

    def __init__(self, model_factory, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
        super().__init__(obs_space, action_space, cfg)
        obs_dim = obs_space["obs"].shape[0]
        action_dim = action_space.shape[0]
        self.actor = _PolicyNet(obs_dim, action_dim)
        self.action_dim = action_dim
        self.warmup_steps = 2500
        self.collection_steps = self.warmup_steps
        self.action_generator = None

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return next(self.parameters()).device

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32

    def forward(self, normalized_obs_dict, rnn_states, values_only=False):
        obs = normalized_obs_dict["obs"].float().flatten(start_dim=1)
        params = self.actor(obs)
        mean, log_std = params.chunk(2, dim=-1)
        self.last_action_distribution = SquashedNormal(mean, log_std)
        values = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)

        if values_only:
            return {"values": values}

        if self.collection_steps < self.warmup_steps:
            actions = torch.empty_like(mean).uniform_(-1.0, 1.0, generator=self.action_generator)
            log_prob = torch.zeros(actions.shape[0], device=actions.device)
        else:
            actions = self.last_action_distribution.rsample(self.action_generator)
            log_prob = self.last_action_distribution.log_prob(actions)

        return {
            "actions": actions,
            "action_logits": torch.cat((mean, log_std), dim=-1),
            "log_prob_actions": log_prob,
            "values": values,
            "new_rnn_states": rnn_states,
        }


class QuantileCritic(nn.Module):
    """BRO quantile critic used by WIMLE's double-Q learner."""

    def __init__(self, obs_dim: int, action_dim: int, num_quantiles: int = 100):
        super().__init__()
        self.net = _BroNet(obs_dim + action_dim, 512, depth=2, output_dim=num_quantiles)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat((obs, action), dim=-1))


class _Scaler(nn.Module):
    def __init__(self, dim: int, init: float, scale: float):
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), scale))
        self.forward_scaler = init / scale

    def forward(self, x: Tensor) -> Tensor:
        return self.scaler * self.forward_scaler * x


class _HyperDense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        _init_linear(self.linear, gain=1.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def _l2normalize(x: Tensor) -> Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)


class _HyperMLP(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, scaler_init: float, scaler_scale: float):
        super().__init__()
        self.w1 = _HyperDense(hidden_dim, hidden_dim * 2)
        self.scaler = _Scaler(hidden_dim * 2, scaler_init, scaler_scale)
        self.w2 = _HyperDense(hidden_dim * 2, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        x = torch.relu(x) + 1e-8
        return _l2normalize(self.w2(x))


class _HyperLERPBlock(nn.Module):
    def __init__(self, hidden_dim: int, scaler_init: float, scaler_scale: float, alpha_init: float, alpha_scale: float):
        super().__init__()
        self.mlp = _HyperMLP(
            hidden_dim,
            hidden_dim,
            scaler_init / math.sqrt(2.0),
            scaler_scale / math.sqrt(2.0),
        )
        self.alpha_scaler = _Scaler(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.mlp(x)
        return _l2normalize(residual + self.alpha_scaler(x - residual))


class IMLEWorldModel(nn.Module):
    """Stochastic HyperLERP world model predicting reward and state delta."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = obs_dim + 1
        self.input_mean = nn.Parameter(torch.zeros(obs_dim + action_dim))
        self.input_std = nn.Parameter(torch.ones(obs_dim + action_dim))

        hidden_dim = 512
        depth = 3
        scaler_scale = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (depth + 1)
        alpha_scale = 1.0 / math.sqrt(hidden_dim)
        self.pre_encoder = _HyperDense(obs_dim + action_dim + latent_dim, hidden_dim)
        self.encoder = nn.Sequential(
            *(
                _HyperLERPBlock(
                    hidden_dim,
                    scaler_scale,
                    scaler_scale,
                    alpha_init,
                    alpha_scale,
                )
                for _ in range(depth)
            )
        )
        self.means = _HyperDense(hidden_dim, self.output_dim)
        self.log_stds = _HyperDense(hidden_dim, self.output_dim)

    def forward(self, obs: Tensor, action: Tensor, latent: Tensor, return_params: bool = False):
        inputs = (torch.cat((obs, action), dim=-1) - self.input_mean) / (self.input_std + 1e-6)
        encoded = self.pre_encoder(torch.cat((inputs, latent), dim=-1))
        encoded = self.encoder(encoded)
        mean = self.means(encoded)
        raw_log_std = self.log_stds(encoded)
        log_std = -10.0 + 11.0 * 0.5 * (1.0 + torch.tanh(raw_log_std))
        if return_params:
            return mean, log_std
        return mean[..., 1:], mean[..., 0]
