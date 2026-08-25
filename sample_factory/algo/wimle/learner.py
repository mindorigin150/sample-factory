from __future__ import annotations

import os
from os.path import join
from typing import Dict

import torch
from torch import Tensor

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, STATS_KEY, TRAIN_STATS
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.wimle.models import IMLEWorldModel, QuantileCritic, SquashedNormal, WIMLEActorCritic
from sample_factory.algo.wimle.replay import ReplayBuffer
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.utils.utils import ensure_dir_exists, experiment_dir


class WIMLELearner:
    """WIMLE optimization and model replay behind Sample Factory's learner-worker contract."""

    _REAL_REPLAY_CAPACITY = 1_000_000
    _RECENT_MODEL_SAMPLES = 100_000
    _SCALER_SAMPLES = 300_000
    _BATCH_SIZE = 128
    _MODEL_BATCH_SIZE = 512
    _MODEL_ROLLOUTS = 200
    _MODEL_UPDATES = 100
    _MODEL_CANDIDATES = 4
    _STALE_CODE_UPDATES = 4
    _ROLLOUT_LATENTS = 10
    _UPDATES_PER_SOURCE = 10
    _SUMMARY_INTERVAL = 1000
    _RESET_INDICES = (15_001, 50_001, 250_001)

    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        self.cfg = cfg
        self.env_info = env_info
        self.policy_id = policy_id
        self.policy_versions = policy_versions
        self.param_server = param_server
        self.timing = Timing(name=f"WIMLELearner {policy_id} profile")
        self.train_step = 0
        self.env_steps = 0
        self.best_performance = -float("inf")
        self.device = policy_device(cfg, policy_id)
        self.learner_generator = torch.Generator(device=self.device)
        self.replay_generator = torch.Generator()
        if cfg.seed is None:
            self.learner_generator.seed()
            self.replay_generator.seed()
        else:
            self.learner_generator.manual_seed(cfg.seed)
            self.replay_generator.manual_seed(cfg.seed)

        self.actor_critic: WIMLEActorCritic | None = None
        self.critics: list[QuantileCritic] = []
        self.target_critics: list[QuantileCritic] = []
        self.world_models: list[IMLEWorldModel] = []
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.model_optimizers = []
        self.log_alpha = None
        self.alpha_optimizer = None

        self.real_replay = ReplayBuffer(self._REAL_REPLAY_CAPACITY, self.replay_generator)
        self.model_replay = ReplayBuffer(
            self._MODEL_BATCH_SIZE * self._MODEL_ROLLOUTS * cfg.wimle_rollout_horizon,
            self.replay_generator,
        )
        self._initial_actor_state = None
        self._initial_critic_states = []
        self._initial_learner_rng_state = self.learner_generator.get_state()

    def init(self):
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        self.actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
        self.actor_critic.model_to_device(self.device)
        self.actor_critic.train()

        obs_dim = self.env_info.obs_space["obs"].shape[0]
        action_dim = self.env_info.action_space.shape[0]
        self.critics = [QuantileCritic(obs_dim, action_dim).to(self.device) for _ in range(2)]
        self.target_critics = [QuantileCritic(obs_dim, action_dim).to(self.device) for _ in range(2)]
        for target, source in zip(self.target_critics, self.critics):
            target.load_state_dict(source.state_dict())
        self.world_models = [IMLEWorldModel(obs_dim, action_dim).to(self.device) for _ in range(7)]

        self.log_alpha = torch.nn.Parameter(torch.zeros((), device=self.device))
        self._make_policy_optimizers()
        self.model_optimizers = [
            torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) for model in self.world_models
        ]

        self._initial_actor_state = self._cpu_state(self.actor_critic.actor)
        self._initial_critic_states = [self._cpu_state(critic) for critic in self.critics]
        self.actor_critic.collection_steps = 0
        self.actor_critic.action_generator = self.learner_generator
        self.param_server.init(self.actor_critic, self.train_step, self.device)
        return self.policy_id, None, self.device, self.train_step

    @staticmethod
    def _cpu_state(module) -> Dict[str, Tensor]:
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}

    def _make_policy_optimizers(self) -> None:
        self.actor_optimizer = torch.optim.AdamW(
            self.actor_critic.actor.parameters(), lr=3e-4, weight_decay=1e-4
        )
        self.critic_optimizer = torch.optim.AdamW(
            [parameter for critic in self.critics for parameter in critic.parameters()],
            lr=3e-4,
            weight_decay=1e-4,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4, betas=(0.5, 0.999))

    @staticmethod
    def _distribution(actor: WIMLEActorCritic, obs: Tensor) -> SquashedNormal:
        mean, log_std = actor.actor(obs).chunk(2, dim=-1)
        return SquashedNormal(mean, log_std)

    def _actor_action(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        distribution = self._distribution(self.actor_critic, obs)
        action = distribution.rsample(self.learner_generator)
        return action, distribution.log_prob(action)

    @staticmethod
    def _quantile_huber_loss(prediction: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        td_error = target.unsqueeze(1) - prediction.unsqueeze(2)
        abs_error = td_error.abs()
        huber = torch.where(abs_error <= 1.0, 0.5 * td_error.square(), abs_error - 0.5)
        taus = (
            torch.arange(prediction.shape[1], device=prediction.device, dtype=prediction.dtype) + 0.5
        ) / prediction.shape[1]
        quantile_weight = (taus.view(1, -1, 1) - (td_error.detach() < 0).to(prediction.dtype)).abs()
        per_transition = (quantile_weight * huber).sum(dim=1).mean(dim=1)
        return (per_transition * weights).mean()

    def _update(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        with torch.no_grad():
            next_action, next_log_prob = self._actor_action(batch["next_obs"])
            next_quantiles = torch.stack(
                [critic(batch["next_obs"], next_action) for critic in self.target_critics]
            ).mean(dim=0)
            target = batch["rewards"].unsqueeze(-1) + 0.99 * batch["masks"].unsqueeze(-1) * (
                next_quantiles - self.log_alpha.exp() * next_log_prob.unsqueeze(-1)
            )

        current_quantiles = [critic(batch["obs"], batch["actions"]) for critic in self.critics]
        critic_loss = sum(
            self._quantile_huber_loss(quantiles, target, batch["weights"]) for quantiles in current_quantiles
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            for target_critic, critic in zip(self.target_critics, self.critics):
                for target_parameter, parameter in zip(target_critic.parameters(), critic.parameters()):
                    target_parameter.mul_(0.995).add_(parameter, alpha=0.005)

        for critic in self.critics:
            critic.requires_grad_(False)
        action, log_prob = self._actor_action(batch["obs"])
        q_value = torch.stack([critic(batch["obs"], action).mean(dim=-1) for critic in self.critics]).mean(dim=0)
        actor_loss = (self.log_alpha.exp().detach() * log_prob - q_value).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for critic in self.critics:
            critic.requires_grad_(True)

        entropy = -log_prob.detach().mean()
        alpha_loss = self.log_alpha.exp() * (entropy - (-self.actor_critic.action_dim / 2.0))
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.train_step += 1
        return {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "alpha": self.log_alpha.exp().detach(),
            "alpha_loss": alpha_loss.detach(),
            "entropy": entropy,
            "q": q_value.detach().mean(),
        }

    @staticmethod
    def _huber(residual: Tensor) -> Tensor:
        absolute = residual.abs()
        return torch.where(absolute <= 1.0, 0.5 * residual.square(), absolute - 0.5)

    def _train_world_models(self) -> Tensor:
        recent = self.real_replay.recent(self._SCALER_SAMPLES)
        model_inputs = torch.cat((recent["obs"], recent["actions"]), dim=-1)
        input_mean = model_inputs.mean(dim=0).to(self.device)
        input_std = model_inputs.std(dim=0, correction=0)
        input_std = torch.where(input_std < 1e-12, torch.ones_like(input_std), input_std).to(self.device)

        loss_value = torch.zeros((), device=self.device)
        for model, optimizer in zip(self.world_models, self.model_optimizers):
            with torch.no_grad():
                model.input_mean.copy_(input_mean)
                model.input_std.copy_(input_std)
            latents = torch.randn(
                self._MODEL_BATCH_SIZE,
                self._MODEL_CANDIDATES,
                model.latent_dim,
                device=self.device,
                generator=self.learner_generator,
            )
            for _ in range(self._MODEL_UPDATES):
                batch = self.real_replay.sample_recent(
                    self._RECENT_MODEL_SAMPLES, self._MODEL_BATCH_SIZE, self.device
                )
                labels = torch.cat((batch["rewards"].unsqueeze(-1), batch["next_obs"] - batch["obs"]), dim=-1)
                with torch.no_grad():
                    observations = (
                        batch["obs"].unsqueeze(1).expand(-1, self._MODEL_CANDIDATES, -1).flatten(0, 1)
                    )
                    actions = (
                        batch["actions"].unsqueeze(1).expand(-1, self._MODEL_CANDIDATES, -1).flatten(0, 1)
                    )
                    means, log_variance = model(
                        observations,
                        actions,
                        latents.flatten(0, 1),
                        return_params=True,
                    )
                    residual = means.unflatten(
                        0, (self._MODEL_BATCH_SIZE, self._MODEL_CANDIDATES)
                    ) - labels.unsqueeze(1)
                    candidate_log_variance = log_variance.unflatten(
                        0, (self._MODEL_BATCH_SIZE, self._MODEL_CANDIDATES)
                    )
                    candidate_loss = (
                        self._huber(residual) * torch.exp(-candidate_log_variance)
                    ).mean(dim=-1) + candidate_log_variance.mean(dim=-1)
                    selected = candidate_loss.argmin(dim=1)
                    selected_latents = latents[
                        torch.arange(self._MODEL_BATCH_SIZE, device=self.device), selected
                    ]

                for _ in range(self._STALE_CODE_UPDATES):
                    prediction, selected_log_variance = model(
                        batch["obs"], batch["actions"], selected_latents, return_params=True
                    )
                    residual = prediction - labels
                    loss = (self._huber(residual) * torch.exp(-selected_log_variance)).mean()
                    loss = loss + selected_log_variance.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_value = loss.detach()

        return loss_value

    def _predict_model_ensemble(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = observations.shape[0]
        reward_predictions = []
        state_predictions = []
        repeated_observations = (
            observations.unsqueeze(1).expand(-1, self._ROLLOUT_LATENTS, -1).flatten(0, 1)
        )
        repeated_actions = actions.unsqueeze(1).expand(-1, self._ROLLOUT_LATENTS, -1).flatten(0, 1)
        for model in self.world_models:
            latents = torch.randn(
                batch_size,
                self._ROLLOUT_LATENTS,
                model.latent_dim,
                device=self.device,
                generator=self.learner_generator,
            )
            state_delta, rewards = model(repeated_observations, repeated_actions, latents.flatten(0, 1))
            state_predictions.append(
                state_delta.unflatten(0, (batch_size, self._ROLLOUT_LATENTS)) + observations.unsqueeze(1)
            )
            reward_predictions.append(
                rewards.unflatten(0, (batch_size, self._ROLLOUT_LATENTS)).unsqueeze(-1)
            )

        rewards = torch.stack(reward_predictions)
        states = torch.stack(state_predictions)
        reward = rewards.mean(dim=(0, 2)).squeeze(-1)
        next_observation = states.mean(dim=(0, 2))
        reward_std = rewards.std(dim=(0, 2), correction=0)
        state_std = states.std(dim=(0, 2), correction=0)
        coefficient = 1.0 / ((reward_std + state_std).mean(dim=-1) + 1.0)
        return reward, next_observation, coefficient

    def _refresh_model_replay(self) -> Tensor:
        model_loss = self._train_world_models()
        self.model_replay.clear()

        for _ in range(self._MODEL_ROLLOUTS):
            starts = self.real_replay.sample_recent(
                self._RECENT_MODEL_SAMPLES, self._MODEL_BATCH_SIZE, self.device
            )
            observations = starts["obs"]
            masks = starts["masks"]
            for _ in range(self.cfg.wimle_rollout_horizon):
                with torch.no_grad():
                    actions, _ = self._actor_action(observations)
                    rewards, next_observations, weights = self._predict_model_ensemble(observations, actions)
                self.model_replay.add_batch(observations, actions, rewards, next_observations, masks, weights)
                observations = next_observations

        return model_loss

    def _reset_policy(self) -> None:
        self.actor_critic.actor.load_state_dict(self._initial_actor_state)
        for critic, initial_state in zip(self.critics, self._initial_critic_states):
            critic.load_state_dict(initial_state)
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())
        self.log_alpha.data.zero_()
        self._make_policy_optimizers()
        self.learner_generator.set_state(self._initial_learner_rng_state)

    def train(self, batch: TensorDict):
        with self.timing.add_time("data"):
            observations = batch["obs"]["obs"][:, :-1].flatten(0, 1).float()
            next_observations = batch["next_obs"]["obs"].flatten(0, 1).float()
            actions = batch["actions"].flatten(0, 1).float()
            rewards = batch["rewards"].flatten().float()
            dones = batch["dones"].flatten()
            timeouts = batch["time_outs"].flatten()
            masks = (~(dones & ~timeouts)).float()
            self.real_replay.add_batch(
                observations,
                actions,
                rewards,
                next_observations,
                masks,
                torch.ones_like(rewards),
            )
            self.env_steps += rewards.shape[0]
            self.actor_critic.collection_steps = self.env_steps

        env_index = self.env_steps - 1
        if env_index in self._RESET_INDICES:
            self._reset_policy()

        model_loss = None
        if env_index % 1000 == 0:
            with self.timing.add_time("model_refresh"):
                model_loss = self._refresh_model_replay()

        with self.timing.add_time("update"):
            batches = [
                self.real_replay.sample(self._BATCH_SIZE, self.device) for _ in range(self._UPDATES_PER_SOURCE)
            ]
            batches.extend(
                self.model_replay.sample(self._BATCH_SIZE, self.device) for _ in range(self._UPDATES_PER_SOURCE)
            )
            stats = {}
            for update_index in torch.randperm(len(batches), generator=self.replay_generator).tolist():
                stats = self._update(batches[update_index])

        self.param_server.update_weights(self.train_step)
        report = {
            LEARNER_ENV_STEPS: self.env_steps,
            POLICY_ID_KEY: self.policy_id,
        }
        if env_index % self._SUMMARY_INTERVAL == 0:
            report[TRAIN_STATS] = {name: value.item() for name, value in stats.items()}
            if model_loss is not None:
                report[TRAIN_STATS]["world_model_loss"] = model_loss.item()
            report[STATS_KEY] = {
                "real_replay": len(self.real_replay),
                "model_replay": len(self.model_replay),
            }
        return report

    def _checkpoint(self):
        return {"train_step": self.train_step, "env_steps": self.env_steps, "model": self.actor_critic.state_dict()}

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        return ensure_dir_exists(join(experiment_dir(cfg=cfg), f"checkpoint_p{policy_id}"))

    @staticmethod
    def _atomic_save(checkpoint, filepath: str) -> None:
        temporary = f"{filepath}.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, filepath)

    def _save_impl(self, prefix: str, suffix: str, keep: int) -> bool:
        path = self.checkpoint_dir(self.cfg, self.policy_id)
        filename = join(path, f"{prefix}_{self.train_step:09d}_{self.env_steps}{suffix}.pth")
        self._atomic_save(self._checkpoint(), filename)
        checkpoints = sorted(join(path, name) for name in os.listdir(path) if name.startswith(f"{prefix}_"))
        for old in checkpoints[:-keep]:
            os.remove(old)
        return True

    def save(self) -> bool:
        return self._save_impl("checkpoint", "", self.cfg.keep_checkpoints)

    def save_best(self, policy_id, metric, metric_value) -> bool:
        if policy_id != self.policy_id:
            return False
        if metric_value > self.best_performance:
            self.best_performance = metric_value
            return self._save_impl("best", f"_{metric}_{metric_value:.3f}", 1)
        return False

    def save_milestone(self):
        milestones = ensure_dir_exists(join(self.checkpoint_dir(self.cfg, self.policy_id), "milestones"))
        filename = join(milestones, f"checkpoint_{self.train_step:09d}_{self.env_steps}.pth")
        self._atomic_save(self._checkpoint(), filename)
