from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from sample_factory.algo.fast_td3.learner import FastTD3Learner
from sample_factory.algo.fast_td3.models import Critic, FastTD3ActorCritic
from sample_factory.algo.fast_td3.replay import FlatReplayBuffer
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.cfg.arguments import default_cfg, verify_cfg


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


def test_batched_sampling_is_rejected():
    cfg = default_cfg("FAST_TD3", "fasttd3")
    cfg.num_workers = 1
    cfg.num_envs_per_worker = 1
    cfg.worker_num_splits = 1
    cfg.rollout = 1
    cfg.num_batches_per_epoch = 1
    cfg.batch_size = 1
    cfg.use_rnn = False
    cfg.recurrence = 1
    cfg.batched_sampling = True
    assert not verify_cfg(cfg, SimpleNamespace(num_agents=1))


def test_learner_warms_replay_without_update_debt(monkeypatch):
    monkeypatch.setattr("sample_factory.algo.fast_td3.learner.LEARNING_START_TRANSITIONS", 4)
    cfg = default_cfg("FAST_TD3", "fasttd3")
    cfg.device = "cpu"
    cfg.normalize_input = False
    cfg.fasttd3_replay_capacity = 16
    cfg.fasttd3_replay_batch_size = 2
    cfg.fasttd3_transitions_per_update = 2
    cfg.fasttd3_v_min = -10.0
    cfg.fasttd3_v_max = 10.0
    cfg.fasttd3_compile = False
    obs_space, action_space = _spaces()
    env_info = SimpleNamespace(obs_space=obs_space, action_space=action_space)
    versions = torch.zeros(1, dtype=torch.int32)
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
    learner.train(batch)
    assert learner.env_steps == 4
    assert learner.train_step == 0
    learner.train(batch)
    assert learner.env_steps == 8
    assert learner.train_step == 2
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

    resumed_versions = torch.zeros(1, dtype=torch.int32)
    resumed = FastTD3Learner(
        cfg,
        env_info,
        resumed_versions,
        0,
        ParameterServer(0, resumed_versions, True),
    )
    resumed.init()
    resumed._load_state(checkpoint)
    assert (resumed.train_step, resumed.env_steps, len(resumed.replay), resumed.update_credit) == (2, 8, 0, 0)
    torch.testing.assert_close(resumed.actor_critic.state_dict(), checkpoint["model"])
    torch.testing.assert_close(resumed.critic.state_dict(), checkpoint["critic"])
    torch.testing.assert_close(resumed.target_critic.state_dict(), checkpoint["target_critic"])
    torch.testing.assert_close(resumed.actor_optimizer.state_dict(), checkpoint["actor_optimizer"])
    torch.testing.assert_close(resumed.critic_optimizer.state_dict(), checkpoint["critic_optimizer"])
