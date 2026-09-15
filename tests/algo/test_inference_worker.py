import torch
from faster_fifo import Empty

from sample_factory.algo.sampling.inference_worker import InferenceWorker
from sample_factory.utils.timing import Timing


class OneBatchQueue:
    def __init__(self, batch):
        self.batch = batch
        self.read = False
        self.blocking_reads = 0

    def get_many(self, timeout):
        if timeout > 0:
            self.blocking_reads += 1
        if self.read:
            raise Empty
        self.read = True
        return self.batch


def test_cpu_inference_does_not_block_after_available_batch():
    queue = OneBatchQueue(["request"])
    worker = InferenceWorker.__new__(InferenceWorker)
    worker.event_loop = None
    worker.device = torch.device("cpu")
    worker.inference_queue = queue
    worker.requests = []
    worker.timing = Timing()

    worker._get_inference_requests_async()

    assert worker.requests == ["request"]
    assert queue.blocking_reads == 1


class TwoBatchQueue:
    def __init__(self):
        self.calls = []
        self.batches = [["first"], ["second"]]

    def get_many(self, block=True, timeout=10.0):
        self.calls.append((block, timeout))
        if self.batches:
            if not block or timeout == 0.005:
                return self.batches.pop(0)
        raise Empty


def test_gpu_inference_waits_for_minimum_batch():
    queue = TwoBatchQueue()
    worker = InferenceWorker.__new__(InferenceWorker)
    worker.event_loop = None
    worker.device = torch.device("cuda")
    worker.inference_queue = queue
    worker.requests = []
    worker.timing = Timing()
    worker.min_num_requests = 2

    worker._get_inference_requests_async()

    assert worker.requests == ["first", "second"]
    assert queue.calls == [(True, 0.005), (True, 0.005)]
