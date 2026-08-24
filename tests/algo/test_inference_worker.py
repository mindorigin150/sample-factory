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
