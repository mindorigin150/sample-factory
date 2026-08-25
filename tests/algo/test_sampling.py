from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import Env, spaces

from sample_factory.algo.sampling.non_batched_sampling import NonBatchedVectorEnvRunner
from sample_factory.algo.utils.make_env import NonBatchedVecEnv


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
