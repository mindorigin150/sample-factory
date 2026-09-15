from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor


class FlatReplayBuffer:
    """A device-resident flat transition ring; replay is intentionally not checkpointed."""

    _fields = ("obs", "actions", "rewards", "next_obs", "dones", "timeouts")

    def __init__(self, capacity: int, device: torch.device, generator: torch.Generator | None = None):
        self.capacity = capacity
        self.device = device
        self.generator = generator
        self.storage: Dict[str, Tensor] | None = None
        self.size = 0
        self.write_index = 0

    def _allocate(self, values: tuple[Tensor, ...]) -> None:
        self.storage = {
            name: torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.device)
            for name, value in zip(self._fields, values)
        }

    def add_batch(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_obs: Tensor,
        dones: Tensor,
        timeouts: Tensor,
    ) -> None:
        values = tuple(value.detach().to(self.device) for value in (obs, actions, rewards, next_obs, dones, timeouts))
        self._add_values(values)

    def _add_values(self, values: tuple[Tensor, ...]) -> None:
        count = values[0].shape[0]
        if self.storage is None:
            self._allocate(values)
        if count >= self.capacity:
            for name, value in zip(self._fields, values):
                self.storage[name].copy_(value[-self.capacity :])
            self.size = self.capacity
            self.write_index = 0
            return
        first = min(count, self.capacity - self.write_index)
        second = count - first
        first_slice = slice(self.write_index, self.write_index + first)
        for name, value in zip(self._fields, values):
            self.storage[name][first_slice].copy_(value[:first])
            if second:
                self.storage[name][:second].copy_(value[first:])
        self.size = min(self.capacity, self.size + count)
        self.write_index = (self.write_index + count) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        indices = torch.randint(
            self.size, (batch_size,), device=self.device, generator=self.generator
        )
        return {name: self.storage[name][indices] for name in self._fields}

    def __len__(self) -> int:
        return self.size


class ChunkExecutionReplayBuffer(FlatReplayBuffer):
    """Aggregate actual rewards; Q uses the uncensored, timing-only handoff plan.

    Each segment keeps its first-execution state and planned counts, including
    heads censored by a physical terminal. The next segment supplies bootstrap's
    own plan; actual executions determine rewards and duration discount.
    """

    _fields = (*FlatReplayBuffer._fields, "discount", "critic_obs", "critic_next_obs", "execution_counts")

    def __init__(self, capacity, device, generator, *, num_envs, gamma, horizon):
        super().__init__(capacity, device, generator)
        self.gamma = gamma
        self.horizon = horizon
        self.issued = [dict() for _ in range(num_envs)]
        self.current = [None for _ in range(num_envs)]
        self.pending = [dict() for _ in range(num_envs)]
        self.next_episode = np.zeros(num_envs, dtype=np.int64)
        self.next_frame = np.zeros(num_envs, dtype=np.int64)
        self.execution_events = 0
        self.censored_segments = 0

    def _new_segment(self, actor_obs, action, critic_obs, source, reward):
        return {
            "obs": actor_obs,
            "actions": action,
            "critic_obs": critic_obs,
            "source": source,
            "rewards": np.zeros_like(reward),
            "discount": np.ones_like(reward),
            "execution_counts": np.zeros(self.horizon, dtype=np.float32),
        }

    def add_batch(self, obs, actions, dones, timeouts, *, env_ids, raw_obs, rewards, clock, lengths):
        """Consume request observations and valid rows of each padded raw trace.

        Clock rows encode episode, frame, source, head, admitted, plan begin/end.
        """
        obs, actions, raw_obs, rewards, dones, timeouts, env_ids, clock, lengths = (
            value.detach().cpu().numpy()
            for value in (obs, actions, raw_obs, rewards, dones, timeouts, env_ids, clock, lengths)
        )
        completed = []

        for row, env in enumerate(env_ids):
            for index in range(lengths[row]):
                episode, frame, source, head, admitted, start, end = clock[row, index]
                self.pending[env][episode, frame] = {
                    "issued": (obs[row].copy(), actions[row].copy()) if admitted else None,
                    "source": source,
                    "head": head,
                    "start": start,
                    "end": end,
                    "raw_obs": raw_obs[row, index].copy(),
                    "next_raw_obs": raw_obs[row, index + 1].copy(),
                    "reward": rewards[row, index].copy(),
                    "done": bool(dones[row] and index == lengths[row] - 1),
                    "timeout": timeouts[row].copy(),
                }

            while (self.next_episode[env], self.next_frame[env]) in self.pending[env]:
                event = self.pending[env].pop((self.next_episode[env], self.next_frame[env]))
                if event["issued"] is not None:
                    self.issued[env][self.next_frame[env]] = event["issued"]

                source = event["source"]
                if source >= 0:
                    current = self.current[env]
                    if current is None or source != current["source"]:
                        actor_obs, action = self.issued[env][source]
                        for key in tuple(self.issued[env]):
                            if key <= source:
                                del self.issued[env][key]
                        planned_counts = np.bincount(
                            np.minimum(np.arange(event["start"], event["end"]), self.horizon - 1),
                            minlength=self.horizon,
                        ).astype(obs.dtype)
                        critic_obs = np.concatenate((event["raw_obs"], planned_counts))
                        new_segment = self._new_segment(actor_obs, action, critic_obs, source, event["reward"])
                        if current is not None:
                            current["next_obs"] = actor_obs
                            current["critic_next_obs"] = critic_obs
                            current["dones"] = np.zeros_like(dones[row])
                            current["timeouts"] = np.zeros_like(timeouts[row])
                            completed.append(current)
                        self.current[env] = new_segment
                        current = new_segment
                        self.execution_events += 1
                    current["execution_counts"][event["head"]] += 1.0
                    current["rewards"] += current["discount"] * event["reward"]
                    current["discount"] *= self.gamma

                if event["done"]:
                    current = self.current[env]
                    if current is not None:
                        if event["timeout"]:
                            self.censored_segments += 1
                        else:
                            terminal_obs = event["next_raw_obs"]
                            terminal_actor_obs = np.zeros_like(current["obs"])
                            terminal_actor_obs[:terminal_obs.size] = terminal_obs.reshape(-1)
                            current["next_obs"] = terminal_actor_obs
                            current["critic_next_obs"] = np.concatenate((
                                terminal_obs, np.zeros(self.horizon, dtype=terminal_obs.dtype)
                            ))
                            current["dones"] = np.ones_like(dones[row])
                            current["timeouts"] = event["timeout"]
                            completed.append(current)
                    self.issued[env].clear()
                    self.current[env] = None
                    self.next_episode[env] += 1
                    self.next_frame[env] = 0
                else:
                    self.next_frame[env] += 1

        if completed:
            self._add_values(tuple(
                torch.from_numpy(np.stack([segment[name] for segment in completed]))
                for name in self._fields
            ))
        return len(completed)
