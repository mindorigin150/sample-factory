from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces

from sample_factory.algo.sampling.non_batched_sampling import NonBatchedVectorEnvRunner
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.make_env import NonBatchedVecEnv
from sample_factory.algo.wimle.learner import WIMLELearner
from sample_factory.algo.wimle.models import IMLEWorldModel, WIMLEActorCritic
from sample_factory.algo.wimle.replay import ReplayBuffer
from sample_factory.cfg.arguments import default_cfg
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.train import _validate_wimle_cfg


def _cfg():
    cfg = default_cfg("WIMLE", "test_wimle")
    cfg.device = "cpu"
    cfg.normalize_input = False
    cfg.use_rnn = False
    cfg.wimle_rollout_horizon = 1
    return cfg


def _spaces():
    obs_space = spaces.Dict({"obs": spaces.Box(-1, 1, (3,), dtype=np.float32)})
    action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
    return obs_space, action_space


def test_wimle_config_keeps_training_and_inference_observations_identical():
    cfg = _cfg()
    for key, value in {
        "restart_behavior": "overwrite",
        "serial_mode": True,
        "async_rl": False,
        "num_workers": 1,
        "num_envs_per_worker": 1,
        "worker_num_splits": 1,
        "rollout": 1,
        "batch_size": 1,
        "num_batches_per_epoch": 1,
    }.items():
        setattr(cfg, key, value)
    _validate_wimle_cfg(cfg)
    WIMLELearner(cfg, None, torch.zeros(1, dtype=torch.int64), 0, None)

    cfg.obs_scale = 2.0
    with pytest.raises(ValueError, match="obs_scale=1.0"):
        _validate_wimle_cfg(cfg)


def test_squashed_actor_and_checkpoint_factory_contract(tmp_path):
    cfg = _cfg()
    obs_space, action_space = _spaces()
    actor = create_actor_critic(cfg, obs_space, action_space)
    assert isinstance(actor, WIMLEActorCritic)

    obs = {"obs": torch.zeros(16, 3)}
    actor.collection_steps = actor.warmup_steps
    output = actor(obs, torch.zeros(16, 0))
    distribution = actor.action_distribution()
    assert torch.all(output["actions"].abs() <= 1.0)
    assert torch.isfinite(output["log_prob_actions"]).all()
    assert torch.all(distribution.log_std >= -10.0)
    assert torch.all(distribution.log_std <= 2.0)
    deterministic_actions = argmax_actions(distribution)

    checkpoint_path = tmp_path / "actor.pth"
    WIMLELearner._atomic_save({"model": actor.state_dict()}, str(checkpoint_path))
    restored = create_actor_critic(cfg, obs_space, action_space)
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True)["model"])
    restored.collection_steps = restored.warmup_steps
    restored(obs, torch.zeros(16, 0))
    assert torch.allclose(argmax_actions(restored.action_distribution()), deterministic_actions)
    assert not (tmp_path / "actor.pth.tmp").exists()


def test_replay_owns_batches_wraps_and_clears():
    replay = ReplayBuffer(4, torch.Generator().manual_seed(0))
    observations = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
    replay.add_batch(
        observations,
        observations + 10,
        observations[:, 0] + 20,
        observations + 30,
        torch.ones(6),
        torch.ones(6),
    )
    observations.fill_(99)
    assert replay.recent(4)["obs"][:, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert set(replay.sample_recent(2, 64, torch.device("cpu"))["obs"][:, 0].tolist()) == {4.0, 5.0}

    zeros = torch.zeros(2, 1)
    replay.clear()
    replay.add_batch(zeros, zeros, zeros[:, 0], zeros, torch.ones(2), torch.full((2,), 0.5))
    assert len(replay) == 2
    assert replay.recent(2)["weights"].tolist() == [0.5, 0.5]


def test_quantile_confidence_weights_individual_transitions():
    prediction = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    first_only = WIMLELearner._quantile_huber_loss(prediction, target, torch.tensor([1.0, 0.0]))
    second_only = WIMLELearner._quantile_huber_loss(prediction, target, torch.tensor([0.0, 1.0]))
    assert second_only > first_only


def test_model_ensemble_averages_models_and_latents():
    class ConstantModel:
        latent_dim = 4

        def __init__(self, value):
            self.value = value

        def __call__(self, obs, actions, latents):
            assert obs.shape == (50, 3)
            assert actions.shape == (50, 2)
            assert latents.shape == (50, 4)
            return torch.full_like(obs, self.value), torch.full((obs.shape[0],), self.value)

    learner = object.__new__(WIMLELearner)
    learner.device = torch.device("cpu")
    learner.learner_generator = torch.Generator().manual_seed(0)
    learner.world_models = [ConstantModel(float(value)) for value in range(7)]
    observations = torch.zeros(5, 3)
    reward, next_observation, coefficient = learner._predict_model_ensemble(observations, torch.zeros(5, 2))
    assert torch.allclose(reward, torch.full((5,), 3.0))
    assert torch.allclose(next_observation, torch.full((5, 3), 3.0))
    assert torch.allclose(coefficient, torch.full((5,), 0.2))

    model = IMLEWorldModel(3, 2)
    state_delta, model_reward = model(torch.zeros(5, 3), torch.zeros(5, 2), torch.zeros(5, 4))
    assert state_delta.shape == (5, 3)
    assert model_reward.shape == (5,)
    assert torch.isfinite(state_delta).all() and torch.isfinite(model_reward).all()


def test_refresh_replaces_exact_official_rollout_volume():
    class RealReplay:
        def sample_recent(self, recent_size, batch_size, device):
            return {
                "obs": torch.zeros(batch_size, 3, device=device),
                "actions": torch.zeros(batch_size, 2, device=device),
                "rewards": torch.zeros(batch_size, device=device),
                "next_obs": torch.zeros(batch_size, 3, device=device),
                "masks": torch.arange(batch_size, device=device).remainder(2).float(),
                "weights": torch.ones(batch_size, device=device),
            }

    learner = object.__new__(WIMLELearner)
    learner.cfg = SimpleNamespace(wimle_rollout_horizon=1)
    learner.device = torch.device("cpu")
    learner.real_replay = RealReplay()
    learner.model_replay = ReplayBuffer(512 * 200, torch.Generator().manual_seed(0))
    learner._train_world_models = lambda: torch.tensor(1.25)
    learner._actor_action = lambda obs: (torch.zeros(obs.shape[0], 2), torch.zeros(obs.shape[0]))
    learner._predict_model_ensemble = lambda obs, action: (
        torch.ones(obs.shape[0]),
        obs + 1,
        torch.full((obs.shape[0],), 0.5),
    )

    assert learner._refresh_model_replay().item() == 1.25
    assert len(learner.model_replay) == 512 * 200
    recent = learner.model_replay.recent(512)
    assert recent["next_obs"].eq(1).all()
    assert recent["weights"].eq(0.5).all()
    assert recent["masks"].tolist() == torch.arange(512).remainder(2).float().tolist()


@pytest.mark.parametrize("truncated", [False, True])
def test_auto_reset_preserves_final_observation(truncated):
    class TerminalEnv(Env):
        observation_space = spaces.Box(-100, 100, (1,), dtype=np.float32)
        action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

        def reset(self, **kwargs):
            return np.array([10], dtype=np.float32), {}

        def step(self, action):
            return np.array([20], dtype=np.float32), 1.0, not truncated, truncated, {}

    env = NonBatchedVecEnv(TerminalEnv())
    env.reset()
    obs, _, terminated, got_truncated, infos = env.step([np.zeros(1, dtype=np.float32)])
    assert terminated == [not truncated]
    assert got_truncated == [truncated]
    assert obs[0]["obs"].item() == 10
    assert infos[0]["final_observation"]["obs"].item() == 20


def test_sampler_keeps_wimle_terminal_observation_without_changing_appo():
    class NextObservationSlot:
        value = None

        def __setitem__(self, index, value):
            self.value = value

    class ActorState:
        def __init__(self):
            self.last_episode_reward = 0.0
            self.curr_traj_buffer = {"next_obs": NextObservationSlot()}
            self.last_obs = None

        def record_env_step(self, *args):
            return None

        def update_rnn_state(self, done):
            pass

    runner = object.__new__(NonBatchedVectorEnvRunner)
    runner.num_agents = 1
    runner.rollout_step = 0
    actor_state = ActorState()
    runner.actor_states = [[actor_state]]
    runner.cfg = SimpleNamespace(algo="WIMLE", reward_scale=1.0, reward_clip=1000.0)
    final_observation = {"obs": np.array([20], dtype=np.float32)}
    runner._process_env_step(
        [{"obs": np.array([10], dtype=np.float32)}],
        [1.0],
        [True],
        [False],
        [{"final_observation": final_observation}],
        0,
    )
    assert actor_state.curr_traj_buffer["next_obs"].value == final_observation

    runner.cfg.algo = "APPO"
    runner._process_env_step(
        [{"obs": np.array([10], dtype=np.float32)}],
        [1.0],
        [True],
        [False],
        [{}],
        0,
    )


@pytest.mark.parametrize(
    ("previous_env_steps", "expected_prefix"),
    [
        (0, ["refresh", "update"]),
        (999, ["update"]),
        (1000, ["refresh", "update"]),
        (15_001, ["reset", "update"]),
    ],
)
def test_source_index_schedule(previous_env_steps, expected_prefix):
    class FakeReplay:
        def add_batch(self, *args):
            pass

        def sample(self, batch_size, device):
            return {}

        def __len__(self):
            return 1

    class FakeTiming:
        def add_time(self, name):
            return nullcontext()

    events = []
    learner = object.__new__(WIMLELearner)
    learner.timing = FakeTiming()
    learner.env_steps = previous_env_steps
    learner.train_step = 0
    learner.actor_critic = SimpleNamespace(collection_steps=0)
    learner.real_replay = FakeReplay()
    learner.model_replay = FakeReplay()
    learner.device = torch.device("cpu")
    learner.replay_generator = torch.Generator().manual_seed(0)
    learner.policy_id = 0
    learner.param_server = SimpleNamespace(update_weights=lambda step: None)
    learner._reset_policy = lambda: events.append("reset")
    learner._refresh_model_replay = lambda: events.append("refresh") or torch.tensor(0.0)
    learner._update = lambda batch: events.append(batch["source"]) or {}
    learner.real_replay.sample = lambda batch_size, device: {"source": "real"}
    learner.model_replay.sample = lambda batch_size, device: {"source": "model"}
    batch = {
        "obs": {"obs": torch.zeros(1, 2, 3)},
        "next_obs": {"obs": torch.zeros(1, 1, 3)},
        "actions": torch.zeros(1, 1, 2),
        "rewards": torch.zeros(1, 1),
        "dones": torch.zeros(1, 1, dtype=torch.bool),
        "time_outs": torch.zeros(1, 1, dtype=torch.bool),
    }

    learner.train(batch)
    compact_events = [
        event
        for index, event in enumerate(events)
        if event in {"reset", "refresh"} and (index == 0 or event != events[index - 1])
    ]
    compact_events.append("update")
    assert compact_events == expected_prefix
    assert events.count("real") == 10
    assert events.count("model") == 10
