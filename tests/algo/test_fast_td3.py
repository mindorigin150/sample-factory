from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from sample_factory.algo.fast_td3.learner import FastTD3Learner
from sample_factory.algo.fast_td3.models import Critic, FastTD3ActorCritic
from sample_factory.algo.fast_td3.replay import FlatReplayBuffer
from sample_factory.algo.fast_td3.sonic import SonicCudaDecoder
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, TRAIN_STATS
from sample_factory.algo.utils.model_sharing import ParameterClientAsync, ParameterServer
from sample_factory.algo.utils.shared_buffers import alloc_policy_output_tensors
from sample_factory.cfg.arguments import default_cfg, verify_cfg
from sample_factory.utils.timing import Timing


def _spaces():
    return (
        spaces.Dict({"obs": spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)}),
        spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
    )


def test_actor_uses_sf_contract_and_carries_noise_scale():
    cfg = default_cfg("FAST_TD3", "fasttd3")
    cfg.normalize_input = False
    obs_space, action_space = _spaces()
    actor = FastTD3ActorCritic(obs_space, action_space, cfg)
    result = actor({"obs": torch.zeros(8, 3)}, torch.zeros(8, 1))

    assert [layer.out_features for layer in actor.actor.net if isinstance(layer, torch.nn.Linear)] == [512, 256, 128, 2]
    assert result["actions"].shape == (8, 2)
    assert result["new_rnn_states"].shape == (8, 1)
    assert torch.all(result["actions"].abs() <= 1.0)
    assert torch.all((result["new_rnn_states"] >= 0.001) & (result["new_rnn_states"] <= 0.4))
    assert actor.obs_normalizer({"obs": torch.ones(2, 3)})["obs"].equal(torch.ones(2, 3))


def test_sonic_policy_keeps_hand_actions_in_policy_outputs():
    cfg = default_cfg("FAST_TD3", "fasttd3_sonic")
    cfg.num_workers = 1
    cfg.num_envs_per_worker = 1
    cfg.worker_num_splits = 1
    cfg.batched_sampling = False
    cfg.fasttd3_sonic_decoder_path = "decoder.onnx"
    env_info = SimpleNamespace(
        num_agents=1,
        action_space=spaces.Box(-1.0, 1.0, (78,), dtype=np.float32),
    )

    outputs, names, _ = alloc_policy_output_tensors(
        cfg, env_info, rnn_size=1, device=torch.device("cpu"), share=False
    )

    assert outputs.shape[-1] == 78 + 156 + 1 + 1 + 1 + 29 + 1
    assert names[-2:] == ("env_actions", "new_rnn_states")


def test_sonic_decoder_uses_only_token_prefix():
    calls = []

    class Binding:
        def bind_input(self, name, device, device_id, dtype, shape, data_ptr):
            self.input_shape = shape

        def bind_output(self, *args):
            pass

        def synchronize_inputs(self):
            calls.append("inputs")

        def synchronize_outputs(self):
            calls.append("outputs")

    class Session:
        def io_binding(self):
            self.binding = Binding()
            return self.binding

        def run_with_iobinding(self, binding):
            self.binding = binding
            calls.append("run")

    decoder = SonicCudaDecoder.__new__(SonicCudaDecoder)
    decoder.device = torch.device("cpu")
    decoder.session = Session()
    decoder.input_name = "input"
    decoder.output_name = "output"
    decoder_input = torch.zeros(2, 78)
    state = torch.zeros(2, 930)

    decoder(decoder_input, state)

    assert decoder.session.binding.input_shape == (2, 64 + 930)
    assert calls == ["inputs", "run", "outputs"]


def test_c51_projection_preserves_probability_mass():
    critic = Critic(3, 2)
    projected = critic.projection(
        torch.zeros(4, 3), torch.zeros(4, 2), torch.zeros(4), torch.ones(4), torch.ones(4)
    )
    assert all(torch.allclose(dist.sum(1), torch.ones(4), atol=1e-6) for dist in projected)


def test_flat_replay_wraps_and_copies():
    replay = FlatReplayBuffer(4, torch.device("cpu"), torch.Generator().manual_seed(0))
    values = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    replay.add_batch(
        values,
        values,
        values[:, 0],
        values,
        torch.zeros(6, dtype=torch.bool),
        torch.zeros(6, dtype=torch.bool),
    )
    values.fill_(99)
    assert len(replay) == 4
    assert replay.sample(32)["obs"].min() >= 2


@pytest.mark.parametrize("unsupported_option", ["batched_sampling", "async_rl", "use_rnn"])
def test_unsupported_fast_td3_options_are_rejected(unsupported_option):
    cfg = default_cfg("FAST_TD3", "fasttd3")
    cfg.num_workers = 1
    cfg.num_envs_per_worker = 1
    cfg.worker_num_splits = 1
    cfg.rollout = 2
    cfg.num_batches_per_epoch = 1
    cfg.batch_size = 2
    cfg.batched_sampling = False
    cfg.async_rl = False
    cfg.use_rnn = False
    cfg.recurrence = 1
    assert verify_cfg(cfg, SimpleNamespace(num_agents=1))
    setattr(cfg, unsupported_option, True)
    if unsupported_option == "use_rnn":
        cfg.recurrence = cfg.rollout
    assert not verify_cfg(cfg, SimpleNamespace(num_agents=1))


def test_learner_warms_replay_without_update_debt(monkeypatch):
    monkeypatch.setattr("sample_factory.algo.fast_td3.learner.LEARNING_START_TRANSITIONS", 4)
    cfg = default_cfg("FAST_TD3", "fasttd3")
    cfg.device = "cpu"
    cfg.async_rl = False
    cfg.batched_sampling = False
    cfg.normalize_input = False
    cfg.serial_mode = True
    cfg.use_rnn = False
    cfg.fasttd3_replay_capacity = 16
    cfg.fasttd3_replay_batch_size = 2
    cfg.fasttd3_transitions_per_update = 2
    cfg.fasttd3_v_min = -10.0
    cfg.fasttd3_v_max = 10.0
    cfg.fasttd3_compile = False
    obs_space, action_space = _spaces()
    env_info = SimpleNamespace(obs_space=obs_space, action_space=action_space)
    versions = torch.zeros(1, dtype=torch.int64)
    server = ParameterServer(0, versions, True)
    learner = FastTD3Learner(cfg, env_info, versions, 0, server)
    learner.init()
    batch = {
        "obs": {"obs": torch.zeros(4, 2, 3)},
        "next_obs": {"obs": torch.zeros(4, 1, 3)},
        "actions": torch.zeros(4, 1, 2),
        "rewards": torch.ones(4, 1),
        "dones": torch.zeros(4, 1, dtype=torch.bool),
        "time_outs": torch.zeros(4, 1, dtype=torch.bool),
    }
    first_report = learner.train(batch)
    assert learner.env_steps == 4
    assert learner.train_step == 0
    assert first_report[LEARNER_ENV_STEPS] == learner.env_steps
    assert versions[0].item() == learner.env_steps

    normalizer_count = learner.actor_critic.empirical_obs_normalizer.count.item()
    learner.actor_critic.eval()
    monkeypatch.setattr(learner, "_should_save_summaries", lambda: False)
    second_report = learner.train(batch)
    assert learner.env_steps == 8
    assert learner.train_step == 2
    assert learner.actor_critic.training
    assert learner.actor_critic.empirical_obs_normalizer.count.item() > normalizer_count
    assert TRAIN_STATS not in second_report
    checkpoint = learner._get_checkpoint_dict()
    assert {
        "model",
        "critic",
        "target_critic",
        "actor_optimizer",
        "critic_optimizer",
        "train_step",
        "env_steps",
    } <= checkpoint.keys()

    resumed_versions = torch.zeros(1, dtype=torch.int64)
    resumed = FastTD3Learner(
        cfg,
        env_info,
        resumed_versions,
        0,
        ParameterServer(0, resumed_versions, True),
    )
    monkeypatch.setattr(
        resumed, "load_from_checkpoint", lambda policy_id: resumed._load_state(checkpoint)
    )
    resumed.init()
    assert (resumed.train_step, resumed.env_steps, len(resumed.replay), resumed.update_credit) == (2, 8, 0, 0)
    assert resumed_versions[0].item() == resumed.env_steps
    torch.testing.assert_close(resumed.actor_critic.state_dict(), checkpoint["model"])
    torch.testing.assert_close(resumed.critic.state_dict(), checkpoint["critic"])
    torch.testing.assert_close(resumed.target_critic.state_dict(), checkpoint["target_critic"])
    torch.testing.assert_close(resumed.actor_optimizer.state_dict(), checkpoint["actor_optimizer"])
    torch.testing.assert_close(resumed.critic_optimizer.state_dict(), checkpoint["critic_optimizer"])
    resumed.train(batch)
    assert resumed_versions[0].item() == resumed.env_steps

    monkeypatch.setattr(learner, "_should_save_summaries", lambda: True)
    assert TRAIN_STATS in learner.train(batch)

    async_cfg = default_cfg("FAST_TD3", "fasttd3")
    async_cfg.device = "cpu"
    async_cfg.async_rl = False
    async_cfg.batched_sampling = False
    async_cfg.normalize_input = False
    async_cfg.serial_mode = False
    async_cfg.use_rnn = False
    async_cfg.fasttd3_replay_capacity = 16
    async_cfg.fasttd3_replay_batch_size = 2
    async_cfg.fasttd3_transitions_per_update = 8
    async_cfg.fasttd3_v_min = -10.0
    async_cfg.fasttd3_v_max = 10.0
    async_cfg.fasttd3_compile = False
    async_versions = torch.zeros(1, dtype=torch.int64)
    async_server = ParameterServer(0, async_versions, False)
    async_learner = FastTD3Learner(async_cfg, env_info, async_versions, 0, async_server)
    init_data = async_learner.init()
    client = ParameterClientAsync(async_server, async_cfg, env_info, Timing("fasttd3 test"))
    client.on_weights_initialized(*init_data[1:])
    async_learner.train(batch)
    assert async_learner.train_step == 0
    assert async_versions[0].item() == async_learner.env_steps
    assert client.actor_critic.empirical_obs_normalizer.count.item() == 0
    async_learner.train(batch)
    assert async_learner.train_step == 0
    assert async_versions[0].item() == async_learner.env_steps
    client.ensure_weights_updated()
    assert client.policy_version == async_learner.env_steps
    assert (
        client.actor_critic.empirical_obs_normalizer.count.item()
        == async_learner.actor_critic.empirical_obs_normalizer.count.item()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compile contract")
def test_compiled_updates_keep_stats_and_optimizer_state(monkeypatch, tmp_path):
    monkeypatch.setattr("sample_factory.algo.fast_td3.learner.LEARNING_START_TRANSITIONS", 4)
    cfg = default_cfg("FAST_TD3", "fasttd3_compile")
    cfg.device = "gpu"
    cfg.train_dir = str(tmp_path)
    cfg.async_rl = False
    cfg.batched_sampling = False
    cfg.normalize_input = False
    cfg.serial_mode = True
    cfg.use_rnn = False
    cfg.fasttd3_replay_capacity = 16
    cfg.fasttd3_replay_batch_size = 8
    cfg.fasttd3_transitions_per_update = 2
    cfg.fasttd3_v_min = -10.0
    cfg.fasttd3_v_max = 10.0
    cfg.fasttd3_compile = True
    obs_space, action_space = _spaces()
    env_info = SimpleNamespace(obs_space=obs_space, action_space=action_space)
    versions = torch.zeros(1, dtype=torch.int64)
    learner = FastTD3Learner(
        cfg, env_info, versions, 0, ParameterServer(0, versions, True)
    )
    learner.init()
    monkeypatch.setattr(learner, "_should_save_summaries", lambda: True)
    batch = {
        "obs": {"obs": torch.zeros(4, 2, 3)},
        "next_obs": {"obs": torch.zeros(4, 1, 3)},
        "actions": torch.zeros(4, 1, 2),
        "rewards": torch.ones(4, 1),
        "dones": torch.zeros(4, 1, dtype=torch.bool),
        "time_outs": torch.zeros(4, 1, dtype=torch.bool),
    }
    learner.train(batch)
    reports = (learner.train(batch), learner.train(batch))

    assert learner.train_step == 4
    assert all(torch.isfinite(torch.tensor(tuple(report[TRAIN_STATS].values()))).all() for report in reports)
    assert {state["step"].item() for state in learner.critic_optimizer.state.values()} == {4}
    assert {state["step"].item() for state in learner.actor_optimizer.state.values()} == {2}
