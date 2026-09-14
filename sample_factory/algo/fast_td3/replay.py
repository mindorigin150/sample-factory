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
        self.episodes = np.full(num_envs, -1, dtype=np.int64)
        self.issued = [dict() for _ in range(num_envs)]
        self.current = [None for _ in range(num_envs)]
        self.execution_events = 0
        self.censored_segments = 0

    def _new_segment(self, actor_obs, action, critic_obs, source, reward):
        return {
            "obs": actor_obs,
            "actions": action,
            "critic_obs": critic_obs,
            "source": source,
            "rewards": torch.zeros_like(reward),
            "discount": torch.ones_like(reward),
            "execution_counts": torch.zeros(
                self.horizon, dtype=torch.float32, device=self.device
            ),
        }

    def _completed_values(self, segment):
        return tuple(segment[name].unsqueeze(0) for name in self._fields)

    def add_batch(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        timeouts,
        *,
        env_ids,
        frames,
        episodes,
        applied_sources,
        applied_indices,
        applied_windows,
        admitted,
    ):
        obs, actions, rewards, next_obs, dones, timeouts = (
            value.detach().to(self.device)
            for value in (obs, actions, rewards, next_obs, dones, timeouts)
        )
        env_ids_cpu = env_ids.detach().cpu().numpy()
        frames_cpu = frames.detach().cpu().numpy()
        episodes_cpu = episodes.detach().cpu().numpy()
        sources_cpu = applied_sources.detach().cpu().numpy()
        indices_cpu = applied_indices.detach().cpu().numpy()
        windows_cpu = applied_windows.detach().cpu().numpy()
        admitted_cpu = admitted.detach().cpu().numpy()
        completed = []

        for row, env in enumerate(env_ids_cpu):
            episode = episodes_cpu[row]
            frame = frames_cpu[row]
            if episode != self.episodes[env]:
                self.issued[env].clear()
                self.current[env] = None
                self.episodes[env] = episode

            if admitted_cpu[row]:
                self.issued[env][frame] = (
                    obs[row].clone(),
                    actions[row].clone(),
                )

            source = sources_cpu[row]
            if source >= 0:
                current = self.current[env]
                if current is None or source != current["source"]:
                    actor_obs, action = self.issued[env][source]
                    for key in tuple(self.issued[env]):
                        if key <= source:
                            del self.issued[env][key]
                    start, end = windows_cpu[row]
                    planned_counts = torch.as_tensor(
                        np.bincount(np.minimum(np.arange(start, end), self.horizon - 1), minlength=self.horizon),
                        device=self.device, dtype=obs.dtype,
                    )
                    critic_obs = torch.cat((obs[row, :198], planned_counts))
                    new_segment = self._new_segment(
                        actor_obs,
                        action,
                        critic_obs,
                        source,
                        rewards[row],
                    )
                    if current is not None:
                        current["next_obs"] = actor_obs
                        current["critic_next_obs"] = critic_obs
                        current["dones"] = torch.zeros_like(dones[row])
                        current["timeouts"] = torch.zeros_like(timeouts[row])
                        completed.append(self._completed_values(current))
                    self.current[env] = new_segment
                    current = new_segment
                    self.execution_events += 1
                current["execution_counts"][indices_cpu[row]] += 1.0
                current["rewards"] += current["discount"] * rewards[row]
                current["discount"] *= self.gamma

            if dones[row].item():
                current = self.current[env]
                if current is not None:
                    if timeouts[row].item():
                        self.censored_segments += 1
                    else:
                        current["next_obs"] = next_obs[row].clone()
                        current["critic_next_obs"] = torch.cat((
                            next_obs[row, :198], torch.zeros(self.horizon, device=self.device)
                        ))
                        current["dones"] = dones[row].clone()
                        current["timeouts"] = timeouts[row].clone()
                        completed.append(self._completed_values(current))
                self.issued[env].clear()
                self.current[env] = None

        if completed:
            self._add_values(
                tuple(torch.cat(fields, dim=0) for fields in zip(*completed))
            )
        return len(completed)
