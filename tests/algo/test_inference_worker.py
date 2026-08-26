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


def test_inference_worker_adds_sonic_reference_before_decoder(monkeypatch):
    class ActorCritic:
        training = False

        def eval(self):
            return self

        def __call__(self, normalized_obs, rnn_states):
            return {
                "actions": residual.clone(),
                "values": torch.zeros(rnn_states.shape[0]),
            }

    class Decoder:
        def __call__(self, tokens, state):
            self.tokens = tokens
            self.state = state
            return torch.zeros(tokens.shape[0], 29)

    base_token = torch.full((2, 64), 0.98)
    residual = torch.ones(2, 64)
    sonic_state = torch.zeros(2, 930)
    obs = {
        "obs": torch.zeros(2, 3),
        "base_token": base_token,
        "sonic_state": sonic_state,
    }
    decoder = Decoder()
    captured = {}

    worker = InferenceWorker.__new__(InferenceWorker)
    worker.event_loop = None
    worker.cfg = type("Config", (), {"serial_mode": True})()
    worker.device = torch.device("cpu")
    worker._batch_func = lambda timing: (obs, torch.zeros(2, 1))
    worker.param_client = type("ParamClient", (), {"actor_critic": ActorCritic(), "policy_version": 3})()
    worker.sonic_decoder = decoder
    worker.total_num_samples = 0
    worker.timing = Timing()
    worker.requests = []
    worker._prepare_policy_outputs_func = lambda num_samples, outputs, requests: captured.update(outputs) or {}
    worker.emit_many = lambda *args: None

    monkeypatch.setattr(
        "sample_factory.algo.sampling.inference_worker.prepare_and_normalize_obs",
        lambda actor_critic, observations: observations,
    )
    worker._handle_policy_steps(worker.timing)

    torch.testing.assert_close(decoder.tokens, base_token + residual)
    torch.testing.assert_close(decoder.state, sonic_state)
    torch.testing.assert_close(captured["actions"], residual)
    assert captured["env_actions"].shape == (2, 29)
