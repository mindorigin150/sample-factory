from __future__ import annotations

from typing import Any

import torch


CudaEventHandle = Any


def record_cuda_event(
    device: torch.device | str,
    event: torch.cuda.Event | None = None,
) -> tuple[torch.cuda.Event, CudaEventHandle]:
    device = torch.device(device)
    if event is None:
        event = torch.cuda.Event(interprocess=True)
    event.record(torch.cuda.current_stream(device))
    return event, event.ipc_handle()


def wait_cuda_event(device: torch.device | str, handle: CudaEventHandle) -> torch.cuda.Event:
    device = torch.device(device)
    event = torch.cuda.Event.from_ipc_handle(device, handle)
    torch.cuda.current_stream(device).wait_event(event)
    return event
