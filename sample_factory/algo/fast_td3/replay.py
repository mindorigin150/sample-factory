from __future__ import annotations

from typing import Dict

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
