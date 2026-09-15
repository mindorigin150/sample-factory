from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from sample_factory.algo.fast_td3.models import (
    NUM_ATOMS,
    Critic,
    FastTD3ActorCritic,
)
from sample_factory.algo.fast_td3.replay import ChunkExecutionReplayBuffer, FlatReplayBuffer
from sample_factory.algo.learning.learner import Learner, model_initialization_data
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import (
    LEARNER_ENV_STEPS,
    LEARNER_TRAIN_STEPS,
    POLICY_ID_KEY,
    STATS_KEY,
    TRAIN_STATS,
)
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
    """FastTD3 learner; chunk replay stitches issue decisions to their execution segments."""

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
        self.target_actor = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.update_credit = 0
        self._actor_step_for_update = None
        self._critic_step_for_update = None
        self.action_chunk_horizon = cfg.fasttd3_action_chunk_horizon

    def init(self):
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        self.device = policy_device(self.cfg, self.policy_id)
        self.actor_critic = FastTD3ActorCritic(
            self.env_info.obs_space, self.env_info.action_space, self.cfg
        ).to(self.device)
        self.actor_critic.train()
        if self.cfg.fasttd3_target_actor:
            self.target_actor = deepcopy(self.actor_critic.actor)
            self.target_actor.requires_grad_(False)

        obs_dim = math.prod(self.env_info.obs_space["obs"].shape)
        action_dim = self.env_info.action_space.shape[0]
        action_chunks = self.action_chunk_horizon > 1
        if action_chunks:
            obs_dim += self.action_chunk_horizon
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
        if action_chunks:
            self.replay = ChunkExecutionReplayBuffer(
                self.cfg.fasttd3_replay_capacity, self.device, replay_generator,
                num_envs=self.cfg.num_workers * self.cfg.num_envs_per_worker,
                gamma=self.cfg.gamma, horizon=self.action_chunk_horizon,
            )
        else:
            self.replay = FlatReplayBuffer(self.cfg.fasttd3_replay_capacity, self.device, replay_generator)

        if self.cfg.restart_behavior == "resume" or self.cfg.initial_model_path is None:
            self.load_from_checkpoint(self.policy_id)
        else:
            checkpoint_dict = self.load_checkpoint([self.cfg.initial_model_path], self.device)
            self.actor_critic.load_state_dict(checkpoint_dict["model"])
            self.critic.load_state_dict(checkpoint_dict["critic"])
            self.target_critic.load_state_dict(checkpoint_dict["target_critic"])
            if self.cfg.fasttd3_target_actor:
                self.target_actor.load_state_dict(self.actor_critic.actor.state_dict())
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
        if self.cfg.fasttd3_target_actor:
            self.target_actor.load_state_dict(checkpoint_dict["target_actor"])
        self.actor_optimizer.load_state_dict(checkpoint_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint_dict["critic_optimizer"])
        if self.device.type != "cuda":
            for optimizer in (self.actor_optimizer, self.critic_optimizer):
                for parameter_group in optimizer.param_groups:
                    parameter_group["capturable"] = False
        self.curr_lr = checkpoint_dict["curr_lr"]

    @staticmethod
    def _flatten_obs(obs: Tensor) -> Tensor:
        return obs.reshape(obs.shape[0], -1)

    def _chunk_critic_actions(self, actions, critic_obs):
        mask = (critic_obs[:, -self.action_chunk_horizon:] > 0).repeat_interleave(
            actions.shape[1] // self.action_chunk_horizon, dim=1
        )
        return actions * mask

    def _critic_step(self, obs, next_obs, actions, rewards, bootstrap, discount,
                     critic_obs=None, critic_next_obs=None):
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            with torch.no_grad():
                target_actor = self.target_actor if self.cfg.fasttd3_target_actor else self.actor_critic.actor
                target_actions = target_actor(next_obs)
                target_noise = torch.randn_like(target_actions).mul(TARGET_POLICY_NOISE).clamp(
                    -TARGET_NOISE_CLIP, TARGET_NOISE_CLIP
                )
                target_actions = (target_actions + target_noise).clamp(-1.0, 1.0)
                target_obs = next_obs
                if self.action_chunk_horizon > 1:
                    target_obs = critic_next_obs
                    target_actions = self._chunk_critic_actions(target_actions, critic_next_obs)
                target_dist_1, target_dist_2 = self.target_critic.projection(
                    target_obs, target_actions, rewards, bootstrap, discount
                )
                target_value_1 = self.target_critic.get_value(target_dist_1)
                target_value_2 = self.target_critic.get_value(target_dist_2)
                target_dist = torch.where(
                    target_value_1.unsqueeze(1) < target_value_2.unsqueeze(1),
                    target_dist_1,
                    target_dist_2,
                )

            if self.action_chunk_horizon > 1:
                current_dist_1, current_dist_2 = self.critic(critic_obs, self._chunk_critic_actions(actions, critic_obs))
            else:
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

    def _actor_step(self, obs, critic_obs=None):
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            actor_actions = self.actor_critic.actor(obs)
            value_obs = obs
            if self.action_chunk_horizon > 1:
                value_obs = critic_obs
                # Physical termination censors rewards, not the current actor's planned suffix.
                actor_actions = self._chunk_critic_actions(actor_actions, critic_obs)
            actor_dist_1, actor_dist_2 = self.critic(value_obs, actor_actions)
            actor_values = torch.minimum(
                self.critic.get_value(F.softmax(actor_dist_1, dim=1)),
                self.critic.get_value(F.softmax(actor_dist_2, dim=1)),
            )
            actor_q_loss = -actor_values.mean()
            actor_action_l2 = actor_actions.square().sum(dim=-1).mean()
            actor_loss = actor_q_loss + self.cfg.fasttd3_actor_action_l2 * actor_action_l2
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.detach(), actor_q_loss.detach(), actor_action_l2.detach()

    def _update(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        obs = self._flatten_obs(batch["obs"])
        next_obs = self._flatten_obs(batch["next_obs"])
        with self.param_server.policy_lock:
            obs = self.actor_critic.normalize_observation_tensor(obs)
            next_obs = self.actor_critic.normalize_observation_tensor(next_obs)
            critic_obs = critic_next_obs = None
            if self.action_chunk_horizon > 1:
                critic_obs = torch.cat((
                    self.actor_critic.empirical_obs_normalizer(
                        batch["critic_obs"][:, :-self.action_chunk_horizon], update_stats=False,
                    ),
                    batch["critic_obs"][:, -self.action_chunk_horizon:],
                ), dim=1)
                critic_next_obs = torch.cat((
                    self.actor_critic.empirical_obs_normalizer(
                        batch["critic_next_obs"][:, :-self.action_chunk_horizon], update_stats=False,
                    ),
                    batch["critic_next_obs"][:, -self.action_chunk_horizon:],
                ), dim=1)
        actions = batch["actions"]
        rewards = batch["rewards"]
        bootstrap = (batch["timeouts"] | ~batch["dones"]).float()
        discount = batch["discount"] if self.action_chunk_horizon > 1 else torch.full_like(rewards, self.cfg.gamma)

        critic_loss, q_value = self._critic_step_for_update(
            obs, next_obs, actions, rewards, bootstrap, discount, critic_obs, critic_next_obs
        )
        critic_loss = critic_loss.clone()
        q_value = q_value.clone()

        next_train_step = self.train_step + 1
        actor_loss = torch.zeros((), device=self.device)
        actor_q_loss = torch.zeros((), device=self.device)
        actor_action_l2 = torch.zeros((), device=self.device)
        if next_train_step % POLICY_DELAY == 0:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            with self.param_server.policy_lock:
                actor_loss, actor_q_loss, actor_action_l2 = self._actor_step_for_update(
                    obs, critic_obs
                )
                actor_loss = actor_loss.clone()
                actor_q_loss = actor_q_loss.clone()
                actor_action_l2 = actor_action_l2.clone()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)
            if self.cfg.fasttd3_target_actor:
                with torch.no_grad():
                    target_parameters = [parameter.data for parameter in self.target_actor.parameters()]
                    parameters = [parameter.data for parameter in self.actor_critic.actor.parameters()]
                    torch._foreach_mul_(target_parameters, 1.0 - TAU)
                    torch._foreach_add_(target_parameters, parameters, alpha=TAU)

        with torch.no_grad():
            target_parameters = [parameter.data for parameter in self.target_critic.parameters()]
            parameters = [parameter.data for parameter in self.critic.parameters()]
            torch._foreach_mul_(target_parameters, 1.0 - TAU)
            torch._foreach_add_(target_parameters, parameters, alpha=TAU)

        self.train_step = next_train_step
        stats = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "actor_q_loss": actor_q_loss,
            "actor_action_l2": actor_action_l2,
            "q_value": q_value,
        }
        if self.action_chunk_horizon > 1:
            active = batch["execution_counts"].bool()
            stats["actor_credit_fraction"] = (batch["critic_obs"][:, -self.action_chunk_horizon:] > 0).float().mean()
            stats["execution_length_mean"] = batch["execution_counts"].sum(dim=1).mean()
            for index in range(self.action_chunk_horizon):
                stats[f"execution_index_{index}_fraction"] = active[:, index].float().mean()
        return stats

    def train(self, batch: TensorDict):
        self.actor_critic.train()
        observations = self.actor_critic.observation_tensor({
            key: batch["obs"][key][:, :-1].flatten(0, 1)
            for key in self.actor_critic.observation_keys
        })
        next_observations = self.actor_critic.observation_tensor({
            key: batch["next_obs"][key].flatten(0, 1)
            for key in self.actor_critic.observation_keys
        })
        actions = batch["actions"].flatten(0, 1).float()
        rewards = batch["rewards"].flatten().float()
        dones = batch["dones"].flatten().bool()
        timeouts = batch["time_outs"].flatten().bool()

        previous_replay_size = len(self.replay)
        with self.timing.add_time("replay_add"):
            with torch.no_grad(), self.param_server.policy_lock:
                self.actor_critic.normalize_observation_tensor(observations.to(self.device))
            if self.action_chunk_horizon > 1:
                trace = batch["next_obs"]
                raw_rewards = trace["replay_rewards"].flatten(0, 1).float() * self.cfg.reward_scale
                raw_rewards = raw_rewards.clamp(-self.cfg.reward_clip, self.cfg.reward_clip)
                lengths = trace["replay_length"].flatten()
                added_transitions = self.replay.add_batch(
                    observations, actions, dones, timeouts,
                    env_ids=batch["env_ids"].flatten(),
                    raw_obs=trace["replay_obs"].flatten(0, 1).float(), rewards=raw_rewards,
                    clock=trace["replay_clock"].flatten(0, 1), lengths=lengths,
                )
                advanced_frames = lengths.sum().item()
            else:
                self.replay.add_batch(observations, actions, rewards, next_observations, dones, timeouts)
                added_transitions = advanced_frames = rewards.shape[0]
        self.env_steps += advanced_frames
        if previous_replay_size >= LEARNING_START_TRANSITIONS:
            self.update_credit += added_transitions

        stats = {}
        while (
            self.update_credit >= self.cfg.fasttd3_transitions_per_update
            and self.train_step < self.cfg.fasttd3_train_for_optimizer_steps
            and (len(self.replay) > 0 if self.action_chunk_horizon > 1 else len(self.replay) >= LEARNING_START_TRANSITIONS)
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
            LEARNER_TRAIN_STEPS: self.train_step,
            POLICY_ID_KEY: self.policy_id,
            STATS_KEY: {"replay": len(self.replay), "update_credit": self.update_credit},
        }
        if self.action_chunk_horizon > 1:
            report[STATS_KEY].update(execution_events=self.replay.execution_events,
                                   timeout_censored_segments=self.replay.censored_segments)
        if stats and self._should_save_summaries():
            self.last_summary_time = time.time()
            report[TRAIN_STATS] = {
                **{name: value.item() for name, value in stats.items()},
                "raw_frames": self.env_steps,
                "replay_size": len(self.replay),
                "update_credit": self.update_credit,
            }
        return report

    def _get_checkpoint_dict(self):
        checkpoint = {
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
        if self.cfg.fasttd3_target_actor:
            checkpoint["target_actor"] = self.target_actor.state_dict()
        return checkpoint
