from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from sample_factory.algo.fast_td3.learner import FastTD3Learner
from sample_factory.algo.fast_td3.models import Critic, EmpiricalNormalization, FastTD3ActorCritic
from sample_factory.algo.fast_td3.replay import ChunkExecutionReplayBuffer, FlatReplayBuffer
from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.sampling.non_batched_sampling import ActorState
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, LEARNER_TRAIN_STEPS, TRAIN_STATS
from sample_factory.algo.utils.model_sharing import ParameterClientAsync, ParameterServer
from sample_factory.cfg.arguments import default_cfg, verify_cfg
from sample_factory.utils.timing import Timing


def _spaces():
    return (
        spaces.Dict({"obs": spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)}),
        spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
    )


def test_fasttd3_inference_process_seeds_exploration(monkeypatch):
    from sample_factory.algo.sampling import inference_worker
    from sample_factory.algo.utils.context import sf_global_context

    cfg = default_cfg("FAST_TD3", "fasttd3")
    cfg.device = "cpu"
    cfg.num_workers = 1
    cfg.seed = 17
    cfg.policy_workers_per_policy = 2
    worker = SimpleNamespace(cfg=cfg, object_id="seed test", policy_id=1, worker_idx=1)
    monkeypatch.setattr(inference_worker, "init_file_logger", lambda _: None)
    monkeypatch.setattr(inference_worker, "init_torch_runtime", lambda _: None)
    monkeypatch.setattr("signal.signal", lambda *_: None)
    with torch.random.fork_rng(devices=[]):
        inference_worker.init_inference_process(sf_global_context(), worker)
        first = torch.randn_like(torch.zeros(4, 344))
        inference_worker.init_inference_process(sf_global_context(), worker)
        torch.testing.assert_close(torch.randn_like(torch.zeros(4, 344)), first, rtol=0, atol=0)
        cfg.seed += 1
        inference_worker.init_inference_process(sf_global_context(), worker)
        assert not torch.equal(torch.randn_like(torch.zeros(4, 344)), first)


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


@pytest.mark.parametrize("coefficient", [0.0, 2.5])
def test_actor_action_l2_is_weighted_and_reported(coefficient):
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action = torch.nn.Parameter(torch.tensor([0.5, -0.25]))

        def forward(self, obs):
            return self.action.expand(obs.shape[0], -1)

    class Critic:
        def __call__(self, obs, actions):
            logits = torch.cat((actions[:, :1], torch.zeros_like(actions)), dim=1)
            return logits, logits

        @staticmethod
        def get_value(probabilities):
            return probabilities[:, 0]

    learner = FastTD3Learner.__new__(FastTD3Learner)
    learner.action_chunk_horizon = 1
    learner.device = torch.device("cpu")
    learner.cfg = SimpleNamespace(fasttd3_actor_action_l2=coefficient)
    learner.actor_critic = SimpleNamespace(actor=Actor())
    learner.critic = Critic()
    learner.actor_optimizer = torch.optim.SGD(learner.actor_critic.actor.parameters(), lr=0.0)
    obs = torch.zeros(4, 3)

    actor_loss, actor_q_loss, actor_action_l2 = learner._actor_step(obs)

    torch.testing.assert_close(actor_action_l2, torch.tensor(0.3125))
    torch.testing.assert_close(actor_loss, actor_q_loss + coefficient * actor_action_l2)


@pytest.mark.parametrize(
    ("device", "compiled"),
    [
        ("cpu", False),
        pytest.param(
            "cuda",
            False,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA projection contract"),
        ),
        pytest.param(
            "cuda",
            True,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compile contract"),
        ),
    ],
)
def test_c51_projection_preserves_probability_mass(device, compiled):
    device = torch.device(device)
    critic = Critic(3, 2, device=device)

    obs = torch.zeros(128, 3, device=device)
    actions = torch.zeros(128, 2, device=device)
    case_rewards = (-300.0, -250.0, -247.5, -245.0, 0.0, 245.0, 247.5, 250.0, 300.0)
    rewards = torch.tensor((case_rewards * 15)[:128], device=device)
    bootstrap = torch.zeros(128, device=device)
    discount = torch.full((128,), 0.99, device=device)

    def project(rewards, bootstrap, discount):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            return critic.projection(obs, actions, rewards, bootstrap, discount)

    if compiled:
        project = torch.compile(project, mode="reduce-overhead")
    with torch.no_grad():
        projected = project(rewards, bootstrap, discount)

    expected_cases = torch.zeros(len(case_rewards), critic.q_support.numel(), device=device)
    expected_cases[[0, 1], 0] = 1.0
    expected_cases[2, :2] = 0.5
    expected_cases[3, 1] = 1.0
    expected_cases[4, 50] = 1.0
    expected_cases[5, 99] = 1.0
    expected_cases[6, 99:101] = 0.5
    expected_cases[[7, 8], 100] = 1.0
    expected = expected_cases.repeat((128 + len(case_rewards) - 1) // len(case_rewards), 1)[:128]

    for dist in projected:
        torch.testing.assert_close(dist, expected, atol=5e-6, rtol=0)
        assert torch.all(dist >= 0)
        assert torch.allclose(dist.sum(1), torch.ones(128, device=device), atol=1e-6)
        logits = torch.full_like(dist, -1e8)
        logits.masked_fill_(expected > 0, 0)
        cross_entropy = -(dist * torch.log_softmax(logits, dim=1)).sum(1)
        assert torch.all(cross_entropy >= 0)

    continuing_bootstrap = torch.ones(128, device=device)
    continuing_discount = torch.ones(128, device=device)
    with torch.no_grad():
        continuing = project(torch.zeros_like(rewards), continuing_bootstrap, continuing_discount)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            continuing_expected = tuple(torch.softmax(critic(obs, actions)[i], dim=1) for i in range(2))
    for dist, expected_dist in zip(continuing, continuing_expected):
        torch.testing.assert_close(dist, expected_dist, atol=5e-6, rtol=0)
        assert torch.all(dist >= 0)
        assert torch.allclose(dist.sum(1), torch.ones(128, device=device), atol=1e-6)


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


def test_fasttd3_runner_stops_on_optimizer_steps_and_uses_them_for_summaries():
    runner = SimpleNamespace(
        cfg=SimpleNamespace(
            algo="FAST_TD3",
            train_for_env_steps=int(1e10),
            train_for_seconds=int(1e10),
            fasttd3_train_for_optimizer_steps=500_000,
        ),
        env_steps={0: 160_000_000},
        train_steps={0: 499_999},
        total_train_seconds=1.0,
    )

    assert Runner._summary_step(runner, 0) == 499_999
    assert not Runner._should_end_training(runner)

    runner.train_steps[0] = 500_000
    assert Runner._should_end_training(runner)


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
    assert first_report[LEARNER_TRAIN_STEPS] == learner.train_step
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
    train_stats = learner.train(batch)[TRAIN_STATS]
    assert {
        "actor_loss", "actor_q_loss", "actor_action_l2",
        "raw_frames", "replay_size", "update_credit",
    } <= train_stats.keys()
    cfg.fasttd3_train_for_optimizer_steps = learner.train_step
    learner.train(batch)
    assert learner.train_step == cfg.fasttd3_train_for_optimizer_steps

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


@pytest.mark.parametrize("batches", [
    (torch.tensor([[0.0], [0.0]]), torch.tensor([[2.0], [2.0]])),
    (torch.tensor([[1.0, 4.0], [3.0, 2.0]]), torch.tensor([[7.0, -2.0]])),
])
def test_empirical_normalization_matches_combined_population(batches):
    combined = torch.cat(batches)
    normalizer = EmpiricalNormalization(combined.shape[1], torch.device("cpu"))
    for batch in batches:
        normalizer.update(batch)
    torch.testing.assert_close(normalizer.mean, combined.mean(dim=0))
    torch.testing.assert_close(normalizer.std.square(), combined.var(dim=0, unbiased=False))
    assert normalizer.count.item() == combined.shape[0]


@pytest.mark.parametrize("obs_dim", [3, 198])
def test_chunk_execution_replay_credits_only_real_execution_windows(obs_dim):
    replay = ChunkExecutionReplayBuffer(
        16,
        torch.device("cpu"),
        torch.Generator().manual_seed(0),
        num_envs=1,
        gamma=0.5,
        horizon=4,
    )

    def add(frame, source, index, reward, *, done=False, timeout=False, buffer=replay, window=None):
        observation = torch.full((1, obs_dim), float(frame))
        action = torch.arange(8, dtype=torch.float32).reshape(1, 8) + frame * 10
        start, end = (index, index + 2) if window is None else window
        return buffer.add_batch(
            observation, action, torch.tensor([done]), torch.tensor([timeout]),
            env_ids=torch.tensor([0]), raw_obs=torch.stack((observation, observation + 1), dim=1),
            rewards=torch.tensor([[reward]]), lengths=torch.tensor([1]),
            clock=torch.tensor([[[0, frame, source, index, 1, start, end]]]),
        )

    add(0, -1, -1, 0.0)
    add(1, -1, -1, 0.0)
    add(2, 0, 2, 1.0)
    add(3, 0, 3, 2.0)
    assert add(4, 2, 2, 4.0) == 1
    add(5, 2, 3, 8.0)
    assert add(6, 4, 2, 16.0, done=True) == 2

    assert len(replay) == 3
    torch.testing.assert_close(replay.storage["obs"][:3, 0], torch.tensor([0.0, 2.0, 4.0]))
    torch.testing.assert_close(replay.storage["critic_obs"][:3, 0], torch.tensor([2.0, 4.0, 6.0]))
    torch.testing.assert_close(replay.storage["critic_next_obs"][:3, 0], torch.tensor([4.0, 6.0, 7.0]))
    torch.testing.assert_close(replay.storage["rewards"][:3], torch.tensor([2.0, 8.0, 16.0]))
    torch.testing.assert_close(replay.storage["discount"][:3], torch.tensor([0.25, 0.25, 0.5]))
    torch.testing.assert_close(
        replay.storage["execution_counts"][:3],
        torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]]),
    )
    torch.testing.assert_close(replay.storage["critic_obs"][:3, obs_dim:], torch.tensor([[0., 0., 1., 1.]]).expand(3, -1))
    torch.testing.assert_close(replay.storage["critic_next_obs"][2, obs_dim:], torch.zeros(4))
    assert replay.storage["dones"][:2].logical_not().all()
    assert replay.storage["dones"][2]

    timeout_replay = ChunkExecutionReplayBuffer(
        4,
        torch.device("cpu"),
        torch.Generator().manual_seed(0),
        num_envs=1,
        gamma=0.5,
        horizon=4,
    )
    assert add(0, 0, 0, 1.0, done=True, timeout=True, buffer=timeout_replay) == 0
    assert timeout_replay.censored_segments == 1
    add(1, 1, 0, 2.0, buffer=timeout_replay)
    assert add(2, 2, 0, 4.0, done=True, timeout=True, buffer=timeout_replay) == 1
    assert len(timeout_replay) == 1
    assert timeout_replay.censored_segments == 2
    torch.testing.assert_close(timeout_replay.storage["rewards"][:1], torch.tensor([2.0]))
    torch.testing.assert_close(timeout_replay.storage["discount"][:1], torch.tensor([0.5]))
    assert not timeout_replay.storage["dones"][0]


    tail = ChunkExecutionReplayBuffer(8, torch.device("cpu"), torch.Generator().manual_seed(0), num_envs=1, gamma=0.5, horizon=4)
    add(0, -1, -1, 0., buffer=tail)
    add(1, -1, -1, 0., buffer=tail)
    for frame in range(2, 7):
        add(frame, 0, min(frame, 3), 2. ** (frame - 2), done=frame == 6, buffer=tail, window=(2, 7))
    torch.testing.assert_close(tail.storage["critic_obs"][0, obs_dim:], torch.tensor([0., 0., 1., 4.]))
    torch.testing.assert_close(tail.storage["execution_counts"][0], torch.tensor([0., 0., 1., 4.]))
    torch.testing.assert_close(tail.storage["rewards"][0], torch.tensor(5.))
    torch.testing.assert_close(tail.storage["discount"][0], torch.tensor(0.5 ** 5))


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("timeout", [False, True])
def test_chunk_replay_is_independent_of_raw_trace_grouping(stride, timeout):
    """Long-lived replay contract: source credit and censoring use actual raw transitions."""
    replay = ChunkExecutionReplayBuffer(16, torch.device("cpu"), None, num_envs=1, gamma=0.5, horizon=4)
    for boundary in range(0, 15, stride):
        length = min(stride, 15 - boundary)
        clock = []
        for frame in range(boundary, boundary + length):
            source = ((frame - 1) // 6) * 6 if frame else -1
            clock.append([0, frame, source, min(frame - source, 3) if source >= 0 else -1,
                          frame % 6 == 0, 1, 7])
        raw_obs = torch.arange(boundary, boundary + stride + 1).float().reshape(1, stride + 1, 1).expand(1, -1, 17)
        replay.add_batch(
            torch.full((1, 17), float(boundary)), torch.full((1, 4), float(boundary)),
            torch.tensor([boundary + length == 15]), torch.tensor([timeout and boundary + length == 15]),
            env_ids=torch.tensor([0]), raw_obs=raw_obs,
            rewards=torch.arange(boundary, boundary + stride).float()[None],
            clock=torch.tensor([clock + [[-1] * 7] * (stride - length)]), lengths=torch.tensor([length]),
        )
    count = 2 if timeout else 3
    assert len(replay) == count
    assert replay.censored_segments == timeout
    torch.testing.assert_close(replay.storage["obs"][:count, 0], torch.tensor([0., 6., 12.])[:count])
    torch.testing.assert_close(replay.storage["actions"][:count, 0], torch.tensor([0., 6., 12.])[:count])
    torch.testing.assert_close(replay.storage["critic_obs"][:count, 0], torch.tensor([1., 7., 13.])[:count])
    torch.testing.assert_close(replay.storage["critic_obs"][:count, -4:], torch.tensor([[0., 1., 1., 4.]]).expand(count, -1))
    expected_rewards = [sum(0.5 ** i * (start + i) for i in range(length)) for start, length in [(1, 6), (7, 6), (13, 2)]]
    torch.testing.assert_close(replay.storage["rewards"][:count], torch.tensor(expected_rewards)[:count])
    torch.testing.assert_close(replay.storage["discount"][:count], torch.tensor([0.5 ** 6, 0.5 ** 6, 0.5 ** 2])[:count])
    torch.testing.assert_close(replay.storage["next_obs"][:2, 0], torch.tensor([6., 12.]))
    torch.testing.assert_close(replay.storage["critic_next_obs"][:2, 0], torch.tensor([7., 13.]))
    if not timeout:
        assert replay.storage["dones"][2]
        torch.testing.assert_close(replay.storage["critic_next_obs"][2, -4:], torch.zeros(4))


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_fasttd3_sampling_preserves_physical_terminal_precedence(terminated, truncated):
    """Long-lived sampling/replay contract: a physical terminal is never timeout-censored."""
    actor = ActorState.__new__(ActorState)
    actor.cfg = SimpleNamespace(algo="FAST_TD3", summaries_use_frameskip=False)
    actor.curr_traj_buffer = {name: torch.zeros(1) for name in ("rewards", "dones", "time_outs", "policy_id")}
    actor.is_active = True
    actor.curr_policy_id = actor.agent_idx = actor.env_idx = actor.global_env_idx = 0
    actor.last_episode_duration = 0
    actor.policy_mgr = SimpleNamespace(get_policy_for_agent=lambda *_: 0)
    actor._episodic_stats = lambda _: {}
    actor._update_training_info = lambda: None
    actor.record_env_step(1., terminated, truncated, {}, 0)
    assert actor.curr_traj_buffer["dones"][0]
    assert actor.curr_traj_buffer["time_outs"][0] == (truncated and not terminated)


@pytest.mark.parametrize("coefficient", [0.0, 2.5])
@pytest.mark.parametrize("heads", [[0], [2, 3], [2, 3, 4], [3], [3, 4, 5], [7]])
@pytest.mark.parametrize("obs_dim", [3, 198])
def test_chunk_q_and_actor_credit_use_the_planned_window(monkeypatch, coefficient, heads, obs_dim):
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actions = torch.nn.Parameter(torch.full((1, 16), 0.25))

        def forward(self, obs):
            return self.actions.expand(obs.shape[0], -1)

    learner = FastTD3Learner.__new__(FastTD3Learner)
    learner.device = torch.device("cpu")
    learner.action_chunk_horizon = 8
    learner.cfg = SimpleNamespace(fasttd3_actor_action_l2=coefficient)
    planned = torch.zeros(8, dtype=torch.bool)
    planned[heads] = True
    planned = planned.repeat_interleave(2)
    learner.actor_critic = SimpleNamespace(actor=Actor())
    learner.critic = Critic(obs_dim + 8, 16, num_atoms=3)
    learner.target_critic = deepcopy(learner.critic)
    learner.actor_optimizer = torch.optim.SGD(learner.actor_critic.actor.parameters(), lr=0.0)
    learner.critic_optimizer = torch.optim.SGD(learner.critic.parameters(), lr=0.0)
    current_actions, target_actions = [], []
    learner.critic.register_forward_pre_hook(lambda module, args: current_actions.append(args[1].detach().clone()))
    learner.target_critic.qnet1.register_forward_pre_hook(lambda module, args: target_actions.append(args[1].detach().clone()))
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    obs = torch.zeros(1, obs_dim)
    expected = learner.actor_critic.actor(obs).detach() * planned
    critic_obs = torch.cat((obs, planned.reshape(8, 2)[:, 0].float()[None]), dim=1)
    next_critic_obs = critic_obs.clone()
    next_critic_obs[:, obs_dim + 2] = 0
    next_critic_obs[:, obs_dim + 4:obs_dim + 6] = 1
    next_expected = learner.actor_critic.actor(obs).detach() * (next_critic_obs[:, obs_dim:] > 0).repeat_interleave(2, dim=1)

    learner._critic_step(obs, obs, torch.full((1, 16), 0.25), torch.ones(1), torch.zeros(1), torch.full((1,), 0.99),
                         critic_obs=critic_obs, critic_next_obs=next_critic_obs)
    torch.testing.assert_close(current_actions[-1], expected)
    torch.testing.assert_close(target_actions[-1], next_expected)

    # The preceding critic update represents a terminal behavior transition;
    # its current-policy replacement still receives credit for both planned actions.
    loss, q_loss, l2 = learner._actor_step(obs, critic_obs=critic_obs)
    torch.testing.assert_close(current_actions[-1], expected)
    torch.testing.assert_close(l2, torch.tensor(2 * len(heads) * 0.25 ** 2))
    torch.testing.assert_close(loss, q_loss + coefficient * l2)
    gradient = learner.actor_critic.actor.actions.grad
    assert (gradient[:, planned] != 0).all()
    torch.testing.assert_close(gradient[:, ~planned], torch.zeros_like(gradient[:, ~planned]))


@pytest.mark.parametrize("device,compiled", [
    ("cpu", False),
    pytest.param("gpu", True, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compile contract")),
])
@pytest.mark.parametrize("obs_dim,action_dim,stride,horizon", [(3, 1, 2, 4), (17, 6, 3, 8), (198, 43, 1, 8)])
def test_chunk_learner_updates_and_resumes(monkeypatch, tmp_path, device, compiled, obs_dim, action_dim, stride, horizon):
    """Chunk updates depend on completed replay transitions, not raw trace stride."""
    monkeypatch.setattr("sample_factory.algo.fast_td3.learner.LEARNING_START_TRANSITIONS", 2)
    cfg = default_cfg("FAST_TD3", "chunk_resume")
    cfg.device = device
    cfg.train_dir = str(tmp_path)
    cfg.serial_mode = True
    cfg.async_rl = False
    cfg.batched_sampling = False
    cfg.normalize_input = False
    cfg.fasttd3_compile = compiled
    cfg.fasttd3_action_chunk_horizon = horizon
    cfg.fasttd3_replay_capacity = 16
    cfg.fasttd3_replay_batch_size = 2
    cfg.fasttd3_transitions_per_update = 1
    cfg.reward_scale = 2.0
    cfg.reward_clip = 3.0
    cfg.num_workers = 2
    cfg.num_envs_per_worker = 1
    obs_space = spaces.Dict({
        "obs": spaces.Box(-1.0, 1.0, (obs_dim,), dtype=np.float32),
        "replay_clock": spaces.Box(-1, 100, (stride, 7), dtype=np.int64),
        "replay_obs": spaces.Box(-np.inf, np.inf, (stride + 1, obs_dim), dtype=np.float32),
        "replay_rewards": spaces.Box(-np.inf, np.inf, (stride,), dtype=np.float32),
        "replay_length": spaces.Box(0, stride, (), dtype=np.int64),
    })
    action_space = spaces.Box(-1.0, 1.0, (horizon * action_dim,), dtype=np.float32)
    env_info = SimpleNamespace(obs_space=obs_space, action_space=action_space)
    versions = torch.zeros(1, dtype=torch.int64)
    learner = FastTD3Learner(cfg, env_info, versions, 0, ParameterServer(0, versions, True))
    learner.init()
    monkeypatch.setattr(learner, "_should_save_summaries", lambda: True)

    def train_frame(target, frame):
        clock = torch.tensor([
            [0, frame * stride + index, frame * stride, index, index == 0, 0, stride]
            for index in range(stride)
        ]).expand(2, 1, stride, 7).clone()
        raw_obs = (torch.arange(stride + 1).float() * 0.01 + frame * 0.1).reshape(1, 1, stride + 1, 1).expand(2, 1, stride + 1, obs_dim)
        return target.train({
            "obs": {"obs": torch.full((2, 2, obs_dim), frame * 0.1)},
            "next_obs": {"obs": torch.full((2, 1, obs_dim), (frame + 1) * 0.1), "replay_clock": clock,
                         "replay_obs": raw_obs, "replay_rewards": torch.arange(1, stride + 1).float().expand(2, 1, stride),
                         "replay_length": torch.full((2, 1), stride)},
            "actions": torch.zeros(2, 1, horizon * action_dim),
            "rewards": torch.full((2, 1), 999.),
            "dones": torch.zeros(2, 1, dtype=torch.bool),
            "time_outs": torch.zeros(2, 1, dtype=torch.bool),
            "env_ids": torch.arange(2).reshape(2, 1),
        })

    for frame in range(4):
        report = train_frame(learner, frame)
    assert learner.train_step == 4
    assert report[LEARNER_TRAIN_STEPS] == learner.train_step
    assert all(np.isfinite(value) for value in report[TRAIN_STATS].values())
    assert report[TRAIN_STATS]["actor_action_l2"] > 0
    physical = torch.zeros(2, obs_dim, device=learner.device)
    assert learner.actor_critic.actor(physical).shape == (2, horizon * action_dim)
    assert learner.replay.storage["critic_obs"].shape == (16, obs_dim + horizon)
    counts = torch.cat((torch.ones(stride), torch.zeros(horizon - stride))).to(learner.device).expand(6, -1)
    torch.testing.assert_close(learner.replay.storage["critic_obs"][:6, obs_dim:], counts)
    assert {state["step"].item() for state in learner.critic_optimizer.state.values()} == {4}
    assert {state["step"].item() for state in learner.actor_optimizer.state.values()} == {2}

    expected_reward = sum(cfg.gamma ** index * min(2.0 * (index + 1), 3.0) for index in range(stride))
    torch.testing.assert_close(learner.replay.storage["rewards"][:6], torch.full((6,), expected_reward, device=learner.device))
    torch.testing.assert_close(learner.replay.storage["discount"][:6], torch.full((6,), cfg.gamma ** stride, device=learner.device))
    checkpoint = deepcopy(learner._get_checkpoint_dict())
    cfg.restart_behavior = "resume"
    cfg.initial_model_path = str(tmp_path / "unused_teacher.pth")
    resumed = FastTD3Learner(cfg, env_info, versions, 0, ParameterServer(0, versions, True))
    monkeypatch.setattr(resumed, "load_from_checkpoint", lambda policy_id: resumed._load_state(checkpoint))
    resumed.init()
    assert (resumed.train_step, resumed.env_steps, len(resumed.replay), resumed.update_credit) == (4, 8 * stride, 0, 0)
    torch.testing.assert_close(resumed.actor_critic.state_dict(), checkpoint["model"])
    torch.testing.assert_close(resumed.critic.state_dict(), checkpoint["critic"])
    torch.testing.assert_close(resumed.target_critic.state_dict(), checkpoint["target_critic"])
    torch.testing.assert_close(resumed.actor_optimizer.state_dict(), checkpoint["actor_optimizer"])
    torch.testing.assert_close(resumed.critic_optimizer.state_dict(), checkpoint["critic_optimizer"])
    train_frame(resumed, 4)
    assert resumed.train_step == 4
    assert resumed.update_credit == 0
