from __future__ import annotations

import math
import time
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from sample_factory.algo.fast_td3.models import (
    NUM_ATOMS,
    Critic,
    FastTD3ActorCritic,
)
from sample_factory.algo.fast_td3.replay import FlatReplayBuffer
from sample_factory.algo.learning.learner import Learner, model_initialization_data
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, STATS_KEY, TRAIN_STATS
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.torch_utils import synchronize
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import Config, PolicyID


TARGET_POLICY_NOISE = 0.001
TARGET_NOISE_CLIP = 0.5
POLICY_DELAY = 2
TAU = 0.1
LEARNING_START_TRANSITIONS = 1_280
WEIGHT_DECAY = 0.1


class FastTD3Learner(Learner):
    """FastTD3's flat-replay learner behind Sample Factory's LearnerWorker API."""

    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        super().__init__(cfg, env_info, policy_versions, policy_id, param_server)
        self.timing = Timing(name=f"FastTD3Learner {policy_id} profile")
        self.replay = None
        self.critic: Critic | None = None
        self.target_critic: Critic | None = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.update_credit = 0
        self._actor_step_for_update = None
        self._critic_step_for_update = None

    def init(self):
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        self.device = policy_device(self.cfg, self.policy_id)
        self.actor_critic = FastTD3ActorCritic(
            self.env_info.obs_space, self.env_info.action_space, self.cfg
        ).to(self.device)
        self.actor_critic.train()

        obs_dim = math.prod(self.env_info.obs_space["obs"].shape)
        action_dim = self.env_info.action_space.shape[0]
        self.critic = Critic(
            obs_dim,
            action_dim,
            NUM_ATOMS,
            self.cfg.fasttd3_v_min,
            self.cfg.fasttd3_v_max,
            self.device,
        ).to(self.device)
        self.target_critic = Critic(
            obs_dim,
            action_dim,
            NUM_ATOMS,
            self.cfg.fasttd3_v_min,
            self.cfg.fasttd3_v_max,
            self.device,
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.AdamW(
            self.actor_critic.actor.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        self.optimizer = self.critic_optimizer
        self.curr_lr = self.cfg.learning_rate
        replay_generator = torch.Generator(device=self.device)
        if self.cfg.seed is None:
            replay_generator.seed()
        else:
            replay_generator.manual_seed(self.cfg.seed)
        self.replay = FlatReplayBuffer(self.cfg.fasttd3_replay_capacity, self.device, replay_generator)

        self.load_from_checkpoint(self.policy_id)
        self.update_credit = 0

        if self.cfg.fasttd3_compile:
            self._actor_step_for_update = torch.compile(self._actor_step, mode="reduce-overhead")
            self._critic_step_for_update = torch.compile(self._critic_step, mode="reduce-overhead")
        else:
            self._actor_step_for_update = self._actor_step
            self._critic_step_for_update = self._critic_step

        self.is_initialized = True
        policy_revision = self.env_steps
        self.param_server.init(self.actor_critic, policy_revision, self.device)
        return model_initialization_data(
            self.cfg, self.policy_id, self.actor_critic, policy_revision, self.device
        )

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict["train_step"]
            self.env_steps = checkpoint_dict["env_steps"]
            self.best_performance = checkpoint_dict["best_performance"]
        self.actor_critic.load_state_dict(checkpoint_dict["model"])
        self.critic.load_state_dict(checkpoint_dict["critic"])
        self.target_critic.load_state_dict(checkpoint_dict["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint_dict["critic_optimizer"])
        self.curr_lr = checkpoint_dict["curr_lr"]

    @staticmethod
    def _flatten_obs(obs: Tensor) -> Tensor:
        return obs.reshape(obs.shape[0], -1)

    def _critic_step(self, obs, next_obs, actions, rewards, bootstrap, discount):
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            with torch.no_grad():
                target_actions = self.actor_critic.actor(next_obs)
                target_noise = torch.randn_like(target_actions).mul(TARGET_POLICY_NOISE).clamp(
                    -TARGET_NOISE_CLIP, TARGET_NOISE_CLIP
                )
                target_actions = (target_actions + target_noise).clamp(-1.0, 1.0)
                target_dist_1, target_dist_2 = self.target_critic.projection(
                    next_obs, target_actions, rewards, bootstrap, discount
                )
                target_value_1 = self.target_critic.get_value(target_dist_1)
                target_value_2 = self.target_critic.get_value(target_dist_2)
                target_dist = torch.where(
                    target_value_1.unsqueeze(1) < target_value_2.unsqueeze(1),
                    target_dist_1,
                    target_dist_2,
                )

            current_dist_1, current_dist_2 = self.critic(obs, actions)
            critic_loss = -(
                target_dist * F.log_softmax(current_dist_1, dim=1)
            ).sum(dim=1).mean()
            critic_loss = critic_loss - (
                target_dist * F.log_softmax(current_dist_2, dim=1)
            ).sum(dim=1).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.detach(), target_value_1.mean().detach()

    def _actor_step(self, obs):
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            actor_actions = self.actor_critic.actor(obs)
            actor_dist_1, actor_dist_2 = self.critic(obs, actor_actions)
            actor_values = torch.minimum(
                self.critic.get_value(F.softmax(actor_dist_1, dim=1)),
                self.critic.get_value(F.softmax(actor_dist_2, dim=1)),
            )
            actor_loss = -actor_values.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.detach()

    def _update(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        obs = self._flatten_obs(batch["obs"])
        next_obs = self._flatten_obs(batch["next_obs"])
        with self.param_server.policy_lock:
            obs = self.actor_critic.empirical_obs_normalizer(obs)
            next_obs = self.actor_critic.empirical_obs_normalizer(next_obs)
        actions = batch["actions"]
        rewards = batch["rewards"]
        bootstrap = (batch["timeouts"] | ~batch["dones"]).float()
        discount = torch.full_like(rewards, self.cfg.gamma)

        critic_loss, q_value = self._critic_step_for_update(
            obs, next_obs, actions, rewards, bootstrap, discount
        )
        critic_loss = critic_loss.clone()
        q_value = q_value.clone()

        next_train_step = self.train_step + 1
        actor_loss = torch.zeros((), device=self.device)
        if next_train_step % POLICY_DELAY == 0:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            with self.param_server.policy_lock:
                actor_loss = self._actor_step_for_update(obs).clone()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

        with torch.no_grad():
            target_parameters = [parameter.data for parameter in self.target_critic.parameters()]
            parameters = [parameter.data for parameter in self.critic.parameters()]
            torch._foreach_mul_(target_parameters, 1.0 - TAU)
            torch._foreach_add_(target_parameters, parameters, alpha=TAU)

        self.train_step = next_train_step
        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "q_value": q_value,
        }

    def train(self, batch: TensorDict):
        self.actor_critic.train()
        observations = batch["obs"]["obs"][:, :-1].flatten(0, 1).float()
        next_observations = batch["next_obs"]["obs"].flatten(0, 1).float()
        actions = batch["actions"].flatten(0, 1).float()
        rewards = batch["rewards"].flatten().float()
        dones = batch["dones"].flatten().bool()
        timeouts = batch["time_outs"].flatten().bool()

        previous_replay_size = len(self.replay)
        with self.timing.add_time("replay_add"):
            with torch.no_grad(), self.param_server.policy_lock:
                self.actor_critic.empirical_obs_normalizer(observations.to(self.device))
            self.replay.add_batch(observations, actions, rewards, next_observations, dones, timeouts)
        self.env_steps += rewards.shape[0]
        if previous_replay_size >= LEARNING_START_TRANSITIONS:
            self.update_credit += rewards.shape[0]

        stats = {}
        while (
            self.update_credit >= self.cfg.fasttd3_transitions_per_update
            and len(self.replay) >= LEARNING_START_TRANSITIONS
        ):
            with self.timing.add_time("replay_sample"):
                replay_batch = self.replay.sample(self.cfg.fasttd3_replay_batch_size)
            with self.timing.add_time("update"):
                stats = self._update(replay_batch)
            self.update_credit -= self.cfg.fasttd3_transitions_per_update

        with self.timing.add_time("publish_weights"):
            synchronize(self.cfg, self.device)
            self.param_server.update_weights(self.env_steps)
        report = {
            LEARNER_ENV_STEPS: self.env_steps,
            POLICY_ID_KEY: self.policy_id,
            STATS_KEY: {"replay": len(self.replay), "update_credit": self.update_credit},
        }
        if stats and self._should_save_summaries():
            self.last_summary_time = time.time()
            report[TRAIN_STATS] = {name: value.item() for name, value in stats.items()}
        return report

    def _get_checkpoint_dict(self):
        return {
            "train_step": self.train_step,
            "env_steps": self.env_steps,
            "best_performance": self.best_performance,
            "model": self.actor_critic.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "curr_lr": self.curr_lr,
        }
