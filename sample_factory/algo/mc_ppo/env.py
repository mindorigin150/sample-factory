"""Decision-step environment contract for execution-aligned Monte Carlo PPO."""
import gymnasium as gym
import numpy as np


OBS_KEYS = ("obs", "scheduled_chunk", "scheduled_index")


class EpisodeEnv(gym.Wrapper):
    """Keep reset ownership with the sampler and expose exact raw-frame rewards."""

    def __init__(self, env, settings):
        super().__init__(env)
        self.stride = settings["decision_frames"]
        self.horizon = settings["episode_frames"]
        action_dim = env.action_space.shape[0] // settings["chunk_horizon"]
        self.noise_dim = (settings["active_heads"][1] - settings["active_heads"][0]) * action_dim
        self.observation_space = gym.spaces.Dict({
            **{key: env.observation_space[key] for key in OBS_KEYS},
            "remaining": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
            "seed": gym.spaces.Box(0, np.iinfo(np.int64).max, (1,), dtype=np.int64),
            "noise": gym.spaces.Box(-np.inf, np.inf, (self.noise_dim,), dtype=np.float32),
        })

    def observation(self, obs):
        return {**{key: obs[key] for key in OBS_KEYS},
                "remaining": np.array([1 - self.frames / self.horizon], dtype=np.float32),
                "seed": np.array([self.seed], dtype=np.int64),
                "noise": self.generator.standard_normal(self.noise_dim).astype(np.float32)}

    def reset(self, *, seed=None, options=None):
        self.seed = seed
        self.generator = np.random.default_rng(seed)
        self.frames, self.total = 0, 0.0
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(self, action):
        raw_reward = 0.0
        for frames in range(1, self.stride + 1):
            obs, reward, terminated, truncated, info = self.env.step(action)
            raw_reward += reward
            self.total += reward
            self.frames += 1
            if terminated or truncated:
                break
        info = {"frames": frames, "raw_reward": raw_reward}
        if terminated or truncated:
            info["episode"] = {"seed": self.seed, "length": self.frames, "return": self.total}
        return self.observation(obs), frames / self.horizon, terminated, truncated, info
