from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


class ReplayBuffer:
    """CPU tensor ring buffer for WIMLE transition batches."""

    _fields = ("obs", "actions", "rewards", "next_obs", "masks", "weights")

    def __init__(self, capacity: int, generator: torch.Generator):
        self.capacity = capacity
        self.generator = generator
        self._storage: Dict[str, Tensor] | None = None
        self._size = 0
        self._write_index = 0

    def _allocate(self, values: tuple[Tensor, ...]) -> None:
        self._storage = {
            name: torch.empty(
                (self.capacity, *value.shape[1:]),
                dtype=value.dtype,
                device="cpu",
            )
            for name, value in zip(self._fields, values)
        }

    def add_batch(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_obs: Tensor,
        masks: Tensor,
        weights: Tensor,
    ) -> None:
        values = tuple(value.detach().to("cpu") for value in (obs, actions, rewards, next_obs, masks, weights))
        count = values[0].shape[0]
        if self._storage is None:
            self._allocate(values)

        if count >= self.capacity:
            for name, value in zip(self._fields, values):
                self._storage[name].copy_(value[-self.capacity:])
            self._size = self.capacity
            self._write_index = 0
            return

        first_count = min(count, self.capacity - self._write_index)
        second_count = count - first_count
        first_slice = slice(self._write_index, self._write_index + first_count)
        for name, value in zip(self._fields, values):
            self._storage[name][first_slice].copy_(value[:first_count])
            if second_count:
                self._storage[name][:second_count].copy_(value[first_count:])

        self._size = min(self.capacity, self._size + count)
        self._write_index = (self._write_index + count) % self.capacity

    def clear(self) -> None:
        self._size = 0
        self._write_index = 0

    def _recent_indices(self, count: int) -> Tensor:
        count = min(count, self._size)
        start = (self._write_index - count) % self.capacity
        return (start + torch.arange(count)) % self.capacity

    def recent(self, count: int) -> Dict[str, Tensor]:
        indices = self._recent_indices(count)
        return {name: self._storage[name][indices] for name in self._fields}

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        indices = torch.randint(self._size, (batch_size,), generator=self.generator)
        return {name: self._storage[name][indices].to(device) for name in self._fields}

    def sample_recent(self, recent_size: int, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        count = min(recent_size, self._size)
        indices = (
            self._write_index
            - count
            + torch.randint(count, (batch_size,), generator=self.generator)
        ) % self.capacity
        return {name: self._storage[name][indices].to(device) for name in self._fields}

    def __len__(self) -> int:
        return self._size
