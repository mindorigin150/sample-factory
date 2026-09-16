"""BC-compatible policy, independent value network, and fixed-shape SF inference."""
import math

import torch
from torch import nn
from torch.distributions import Independent, Normal

from sample_factory.algo.fast_td3.models import Actor, EmpiricalNormalization
from sample_factory.algo.mc_ppo.env import OBS_KEYS
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCritic


class PPOPolicy(nn.Module):
    """Load the existing BC actor and freeze its physical-observation normalizer."""

    def __init__(self, obs_space, action_space, settings):
        super().__init__()
        self.physical_dim = obs_space["obs"].shape[0]
        self.input_dim = sum(math.prod(obs_space[key].shape) for key in OBS_KEYS)
        self.actor = Actor(self.input_dim, action_space.shape[0])
        self.empirical_obs_normalizer = EmpiricalNormalization(self.physical_dim, torch.device("cpu"))
        action_dim = action_space.shape[0] // settings["chunk_horizon"]
        self.active = slice(settings["active_heads"][0] * action_dim, settings["active_heads"][1] * action_dim)
        self.std = settings["action_std"]

    def normalize(self, observations):
        device = next(self.parameters()).device
        parts = [torch.as_tensor(observations[key], device=device, dtype=torch.float32).flatten(1) for key in OBS_KEYS]
        parts[0] = self.empirical_obs_normalizer(parts[0], update_stats=False)
        return torch.cat(parts, dim=1)

    def logits(self, obs):
        return self.actor.net[:-1](obs)

    def distribution(self, means):
        # The tanh Jacobian cancels in the ratio for the same stored latent action.
        return Independent(Normal(means, self.std, validate_args=False), 1, validate_args=False)


class MCPPOActorCritic(ActorCritic):
    """Publish bounded actions and their pre-tanh samples through SF shared buffers."""

    def __init__(self, obs_space, action_space, cfg):
        super().__init__(obs_space, action_space, cfg)
        self.obs_normalizer = nn.Identity()
        self.policy = PPOPolicy(obs_space, action_space, cfg.ppo)
        self.value = nn.Sequential(nn.Linear(self.policy.input_dim + 1, 128), nn.ReLU(),
                                   nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def device_for_input_tensor(self, name):
        return next(self.parameters()).device

    def type_for_input_tensor(self, name):
        return torch.int64 if name == "seed" else torch.float32

    def summaries(self):
        return {}

    def policy_logits(self, observations):
        obs = self.policy.normalize(observations)
        # SF batches ready requests dynamically. Preserve the measured BC GEMM shape
        # without making ready environments wait for the slowest worker.
        width = self.cfg.ppo["inference_batch_size"]
        chunks = []
        for part in obs.split(width):
            padded = torch.nn.functional.pad(part, (0, 0, 0, width - len(part)))
            chunks.append(self.policy.logits(padded)[:len(part)])
        return obs, torch.cat(chunks)

    def forward(self, observations, rnn_states, values_only=False):
        obs, logits = self.policy_logits(observations)
        result = TensorDict(values=self.value(torch.cat((obs, observations["remaining"]), 1)).squeeze(1))
        if values_only:
            return result
        means = logits[:, self.policy.active]
        latent = logits.clone()
        noise = torch.zeros_like(means) if self.cfg.benchmark else observations["noise"]
        latent[:, self.policy.active] = means + self.policy.std * noise
        result.update(actions=latent.tanh(), latent_actions=latent,
                      action_logits=torch.cat((logits, torch.full_like(logits, math.log(self.policy.std))), 1),
                      log_prob_actions=self.policy.distribution(means).log_prob(latent[:, self.policy.active]),
                      new_rnn_states=rnn_states)
        return result

    def deterministic_actions(self, observations):
        return self.policy_logits(observations)[1].tanh()
