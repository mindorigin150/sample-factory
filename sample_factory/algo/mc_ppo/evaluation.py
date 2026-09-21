"""Fixed-seed, fixed-inference-shape evaluation separate from the training budget."""
import json
import time
from collections import deque
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sample_factory.algo.utils.context import set_global_context, sf_global_context
from sample_factory.envs.create_env import create_env
from sample_factory.utils.attr_dict import AttrDict


class EvaluationEnv(gym.Wrapper):
    """Stop each vector slot once its assigned evaluation seeds are exhausted."""

    def __init__(self, cfg, context, worker, workers):
        set_global_context(context)
        super().__init__(create_env(cfg.env, cfg, AttrDict(worker_index=worker, vector_index=0, env_id=worker)))
        self.worker, self.workers = worker, workers
        self.seeds = deque()
        self.active = False
        self.last_obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}
        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces,
                                                "active": gym.spaces.Box(0, 1, (1,), dtype=np.float32)})

    def assign_seeds(self, seeds):
        self.seeds = deque(seeds[self.worker::self.workers])

    def observation(self):
        return {**self.last_obs, "active": np.array([self.active], dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        self.active = bool(self.seeds)
        if self.active:
            self.last_obs, _ = self.env.reset(seed=self.seeds.popleft())
        return self.observation(), {}

    def step(self, action):
        if not self.active:
            return self.observation(), 0.0, False, False, {}
        self.last_obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(), reward, terminated, truncated, info


def evaluation_env(cfg):
    workers = cfg.ppo["workers"]
    return gym.vector.AsyncVectorEnv(
        [partial(EvaluationEnv, cfg, sf_global_context(), i, workers) for i in range(workers)],
        context="forkserver", shared_memory=True,
    )


@torch.no_grad()
def evaluate(env, model, seeds):
    started = time.perf_counter()
    env.call("assign_seeds", list(seeds))
    obs, _ = env.reset()
    episodes = []
    while obs["active"].any():
        action = model.deterministic_actions(obs).cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action)
        for index in np.flatnonzero(terminated | truncated):
            episodes.append(info["final_info"][index]["episode"])
    episodes.sort(key=lambda row: row["seed"])
    return {**episode_summary(episodes), "seconds": time.perf_counter() - started}


def episode_summary(episodes):
    lengths = np.array([episode["length"] for episode in episodes])
    return {"n": len(episodes), "mean_length": lengths.mean().item(),
            "mean_return": np.mean([episode["return"] for episode in episodes]).item(),
            "ge100": np.count_nonzero(lengths >= 100), "ge500": np.count_nonzero(lengths >= 500),
            "episodes": episodes}


def comparison(episodes, baseline):
    by_seed = {row["seed"]: row for row in baseline}
    delta = np.array([row["length"] - by_seed[row["seed"]]["length"] for row in episodes])
    samples = np.random.default_rng(42).choice(delta, (10000, len(delta))).mean(1)
    interval = np.quantile(samples, [.025, .975]).tolist()
    return {"mean_length_gain": delta.mean().item(), "paired_95_interval": interval,
            "accepted": interval[0] > 0 and episode_summary(episodes)["ge100"] >= episode_summary(baseline)["ge100"]}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
