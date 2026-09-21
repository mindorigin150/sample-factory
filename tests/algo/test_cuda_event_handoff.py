import torch
import torch.multiprocessing as mp
import pytest

from sample_factory.algo.utils.cuda_event_handoff import record_cuda_event, wait_cuda_event


def _producer(tensor_queue, handle_queue, ack_queue, device: str) -> None:
    torch.cuda.set_device(torch.device(device).index)
    tensor = torch.zeros(4, device=device)
    tensor.fill_(9.0)
    _, handle = record_cuda_event(device)
    tensor_queue.put(tensor)
    handle_queue.put(handle)
    ack_queue.get(timeout=30)


def _consumer(tensor_queue, handle_queue, result_queue, ack_queue, device: str) -> None:
    torch.cuda.set_device(torch.device(device).index)
    tensor = tensor_queue.get(timeout=30)
    handle = handle_queue.get(timeout=30)
    wait_cuda_event(device, handle)
    result_queue.put(tensor.cpu().tolist())
    del tensor
    torch.cuda.synchronize(device)
    ack_queue.put(True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for IPC event handoff")
def test_cuda_event_handoff_between_processes():
    device = "cuda:0"
    ctx = mp.get_context("spawn")
    tensor_queue = ctx.Queue()
    handle_queue = ctx.Queue()
    result_queue = ctx.Queue()
    ack_queue = ctx.Queue()

    producer = ctx.Process(target=_producer, args=(tensor_queue, handle_queue, ack_queue, device))
    consumer = ctx.Process(target=_consumer, args=(tensor_queue, handle_queue, result_queue, ack_queue, device))

    producer.start()
    consumer.start()

    result = result_queue.get(timeout=30)
    producer.join(timeout=30)
    consumer.join(timeout=30)

    assert result == [9.0, 9.0, 9.0, 9.0]
    assert producer.exitcode == 0
    assert consumer.exitcode == 0
