from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sample_factory.algo.utils.action_distributions import ContinuousActionDistribution
from sample_factory.model.actor_critic import ActorCritic


ACTOR_HIDDEN_DIMS = (512, 256, 128)
CRITIC_HIDDEN_DIMS = (1024, 512, 256)
NUM_ATOMS = 101
ACTOR_INIT_SCALE = 0.01
EXPLORATION_STD_MIN = 0.001
EXPLORATION_STD_MAX = 0.4


class EmpiricalNormalization(nn.Module):
    """The empirical mean/std normalizer used by the upstream FastTD3 agent."""

    def __init__(self, shape, device, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(shape, device=device).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape, device=device).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape, device=device).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))

    @property
    def mean(self) -> Tensor:
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> Tensor:
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(self, x: Tensor, update_stats: bool = True) -> Tensor:
        if self.training and update_stats:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: Tensor) -> None:
        batch_size = x.shape[0]
        batch_mean = x.mean(dim=0, keepdim=True)
        batch_var = x.var(dim=0, keepdim=True, unbiased=False)

        new_count = self.count + batch_size
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_size / new_count))
        m_a = self._var * self.count
        m_b = batch_var * batch_size
        m2 = m_a + m_b + delta.square() * (self.count * batch_size / new_count)
        self._var.copy_(m2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)


class Actor(nn.Module):
    """Official FastTD3 512/256/128 ReLU-tanh deterministic actor."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        std_min: float = EXPLORATION_STD_MIN,
        std_max: float = EXPLORATION_STD_MAX,
        init_scale: float = ACTOR_INIT_SCALE,
        device: torch.device | None = None,
    ):
        super().__init__()
        layers = []
        in_features = n_obs
        for hidden_dim in ACTOR_HIDDEN_DIMS:
            layers.extend((nn.Linear(in_features, hidden_dim, device=device), nn.ReLU()))
            in_features = hidden_dim
        layers.extend((nn.Linear(in_features, n_act, device=device), nn.Tanh()))
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-2].weight, 0.0, init_scale)
        nn.init.constant_(self.net[-2].bias, 0.0)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)

class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        device: torch.device | None = None,
    ):
        super().__init__()
        layers = []
        in_features = n_obs + n_act
        for hidden_dim in CRITIC_HIDDEN_DIMS:
            layers.extend((nn.Linear(in_features, hidden_dim, device=device), nn.ReLU()))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, num_atoms, device=device))
        self.net = nn.Sequential(*layers)
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        return self.net(torch.cat((obs, actions), dim=1))

    def projection(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        bootstrap: Tensor,
        discount: Tensor,
        q_support: Tensor,
    ) -> Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()
        # Explicit exact-bin weights prevent compiled FP32 FMA from creating negative mass.
        lower_weight = torch.where(l == u, 1.0, u.float() - b)
        upper_weight = torch.where(l == u, 0.0, b - l.float())

        next_dist = F.softmax(self(obs, actions), dim=1)
        projected = torch.zeros_like(next_dist)
        offsets = (
            torch.arange(rewards.shape[0], device=rewards.device)
            .mul(self.num_atoms)
            .unsqueeze(1)
            .expand(-1, self.num_atoms)
        )
        projected.view(-1).index_add_(
            0, (l + offsets).reshape(-1), (next_dist * lower_weight).reshape(-1)
        )
        projected.view(-1).index_add_(
            0, (u + offsets).reshape(-1), (next_dist * upper_weight).reshape(-1)
        )
        return projected


class Critic(nn.Module):
    """Twin C51 critics with the upstream 1024/512/256/101 architecture."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int = NUM_ATOMS,
        v_min: float = -250.0,
        v_max: float = 250.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, device)
        self.qnet2 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, device)
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def forward(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        bootstrap: Tensor,
        discount: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return (
            self.qnet1.projection(obs, actions, rewards, bootstrap, discount, self.q_support),
            self.qnet2.projection(obs, actions, rewards, bootstrap, discount, self.q_support),
        )

    def get_value(self, probs: Tensor) -> Tensor:
        return torch.sum(probs * self.q_support, dim=1)


class FastTD3ActorCritic(ActorCritic):
    """Sample Factory policy wrapper around the FastTD3 deterministic actor."""

    def __init__(self, obs_space, action_space, cfg):
        super().__init__(obs_space, action_space, cfg)
        self.physical_obs_dim = math.prod(obs_space["obs"].shape)
        self.observation_keys = (
            ("obs", "scheduled_chunk", "scheduled_index")
            if cfg.fasttd3_actor_chunk_context else ("obs",)
        )
        obs_dim = sum(math.prod(obs_space[key].shape) for key in self.observation_keys)
        action_dim = action_space.shape[0]
        self.actor = Actor(self.physical_obs_dim, action_dim)
        if cfg.fasttd3_actor_chunk_context:
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                original = self.actor.net[0]
                expanded = nn.Linear(obs_dim, original.out_features, device=original.weight.device)
                expanded.weight[:, :self.physical_obs_dim].copy_(original.weight)
                expanded.weight[:, self.physical_obs_dim:].zero_()
                expanded.bias.copy_(original.bias)
                self.actor.net[0] = expanded
        self.empirical_obs_normalizer = EmpiricalNormalization(self.physical_obs_dim, torch.device("cpu"))
        self.obs_normalizer = nn.Identity()

    def model_to_device(self, device):
        self.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return next(self.parameters()).device

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32

    def observation_tensor(self, obs: Dict[str, Tensor]) -> Tensor:
        return torch.cat(
            [obs[key].float().flatten(start_dim=1) for key in self.observation_keys], dim=1
        )

    def normalize_observation_tensor(self, obs: Tensor, update_stats: bool = True) -> Tensor:
        physical = self.empirical_obs_normalizer(obs[:, :self.physical_obs_dim], update_stats)
        return torch.cat((physical, obs[:, self.physical_obs_dim:]), dim=1)

    def normalize_obs(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        obs = dict(obs)
        obs["obs"] = self.normalize_observation_tensor(self.observation_tensor(obs))
        return obs

    def summaries(self) -> Dict:
        return {
            "obs_mean": self.empirical_obs_normalizer.mean.mean(),
            "obs_std": self.empirical_obs_normalizer.std.mean(),
        }

    def forward(self, normalized_obs_dict, rnn_states, values_only: bool = False):
        obs = normalized_obs_dict["obs"].float().flatten(start_dim=1)
        means = self.actor(obs)
        noise_scales = rnn_states.reshape(obs.shape[0], -1)[:, :1]
        fresh_scales = torch.rand_like(noise_scales) * (self.actor.std_max - self.actor.std_min) + self.actor.std_min
        noise_scales = torch.where(noise_scales > 0.0, noise_scales, fresh_scales)
        action_logits = torch.cat((means, torch.zeros_like(means)), dim=1)
        self.last_action_distribution = ContinuousActionDistribution(action_logits)
        result = {"values": torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)}
        if values_only:
            return result
        actions = (means + torch.randn_like(means) * noise_scales).clamp(-1.0, 1.0)
        result.update(
            {
                "actions": actions,
                "action_logits": action_logits,
                "log_prob_actions": self.last_action_distribution.log_prob(actions),
                "new_rnn_states": noise_scales,
            }
        )
        return result
