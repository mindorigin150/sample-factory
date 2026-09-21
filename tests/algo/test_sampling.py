from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import Env, spaces

from sample_factory.algo.sampling.non_batched_sampling import NonBatchedVectorEnvRunner
from sample_factory.algo.utils.make_env import NonBatchedVecEnv
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.timing import Timing


class TerminalEnv(Env):
    observation_space = spaces.Box(-100, 100, (1,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

    def __init__(self, truncated):
        self.truncated = truncated

    def reset(self, **kwargs):
        return np.array([10], dtype=np.float32), {}

    def step(self, action):
        return np.array([20], dtype=np.float32), 1.0, not self.truncated, self.truncated, {}


@pytest.mark.parametrize("truncated", [False, True])
def test_auto_reset_preserves_final_observation(truncated):
    env = NonBatchedVecEnv(TerminalEnv(truncated))
    env.reset()
    obs, _, terminated, got_truncated, infos = env.step([np.zeros(1, dtype=np.float32)])
    assert terminated == [not truncated]
    assert got_truncated == [truncated]
    assert obs[0]["obs"].item() == 10
    assert infos[0]["final_observation"]["obs"].item() == 20


def test_samplers_store_terminal_observation_for_fast_td3_only():
    class Slot:
        value = None

        def __setitem__(self, index, value):
            self.value = value

    class ActorState:
        def __init__(self):
            self.last_episode_reward = 0.0
            self.global_env_idx = 3
            self.curr_traj_buffer = {"env_ids": Slot(), "next_obs": Slot()}
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
    runner.cfg = SimpleNamespace(algo="FAST_TD3", reward_scale=1.0, reward_clip=1000.0)
    final_observation = {"obs": np.array([20], dtype=np.float32)}
    runner._process_env_step(
        [{"obs": np.array([10], dtype=np.float32)}],
        [1.0],
        [True],
        [False],
        [{"final_observation": final_observation}],
        0,
    )
    assert actor_state.curr_traj_buffer["env_ids"].value == actor_state.global_env_idx
    assert actor_state.curr_traj_buffer["next_obs"].value == final_observation

    runner.cfg.algo = "APPO"
    actor_state.curr_traj_buffer = {}
    runner._process_env_step(
        [{"obs": np.array([10], dtype=np.float32)}],
        [1.0],
        [True],
        [False],
        [{}],
        0,
    )


def test_non_batched_policy_outputs_restore_trajectory_shapes():
    class ActorState:
        def __init__(self):
            self.is_active = True
            self.curr_policy_id = 0
            self.ready = False
            self.policy_output_names = (
                "actions",
                "action_logits",
                "log_prob_actions",
                "values",
                "policy_version",
                "new_rnn_states",
            )
            self.policy_output_indices = np.cumsum([1, 3, 1, 1, 1, 2])[:-1]
            self.policy_output_tensors = np.arange(9, dtype=np.float32)
            self.curr_traj_buffer = TensorDict(
                actions=np.zeros((1, 1), dtype=np.float32),
                action_logits=np.zeros((1, 3), dtype=np.float32),
                log_prob_actions=np.zeros(1, dtype=np.float32),
                values=np.zeros(1, dtype=np.float32),
                policy_version=np.zeros(1, dtype=np.float32),
            )

        def set_trajectory_data(self, data, rollout_step):
            self.policy_outputs = data
            self.curr_traj_buffer[rollout_step] = data

    runner = object.__new__(NonBatchedVectorEnvRunner)
    runner.num_envs = 1
    runner.num_agents = 1
    runner.rollout_step = 0
    actor_state = ActorState()
    runner.actor_states = [[actor_state]]

    assert runner._process_policy_outputs(0, Timing())

    assert actor_state.policy_outputs["actions"].shape == (1,)
    assert actor_state.policy_outputs["action_logits"].shape == (3,)
    assert actor_state.policy_outputs["log_prob_actions"].shape == ()
    assert actor_state.policy_outputs["values"].shape == ()
    assert actor_state.policy_outputs["policy_version"].shape == ()
    assert actor_state.policy_outputs["new_rnn_states"].shape == (2,)
    np.testing.assert_array_equal(actor_state.curr_traj_buffer[0]["actions"], [0])
    np.testing.assert_array_equal(actor_state.curr_traj_buffer[0]["action_logits"], [1, 2, 3])
    assert actor_state.curr_traj_buffer[0]["log_prob_actions"] == 4
    assert actor_state.curr_traj_buffer[0]["values"] == 5
    assert actor_state.curr_traj_buffer[0]["policy_version"] == 6
