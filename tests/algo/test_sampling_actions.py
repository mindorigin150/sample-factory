import gymnasium as gym
import numpy as np
import torch

from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.sampling.non_batched_sampling import ActorState
from sample_factory.algo.sampling.sampling_utils import _clip_actions_to_space
from sample_factory.algo.utils.env_info import EnvInfo


def _env_info(action_space, gpu_actions=True):
    return EnvInfo(
        obs_space=gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        action_space=action_space,
        num_agents=1,
        gpu_actions=gpu_actions,
        gpu_observations=False,
        action_splits=[1] if isinstance(action_space, gym.spaces.Tuple) else None,
        all_discrete=(
            all(isinstance(space, gym.spaces.Discrete) for space in action_space)
            if isinstance(action_space, gym.spaces.Tuple)
            else False
        ),
        frameskip=1,
    )


def test_numpy_box_clipping_is_out_of_place():
    action_space = gym.spaces.Box(low=np.array([-2.0, 1.0]), high=np.array([3.0, 4.0]), dtype=np.float32)
    actions = np.array([[-5.0, 7.0]], dtype=np.float32)

    clipped = _clip_actions_to_space(action_space, actions)

    np.testing.assert_array_equal(clipped, [[-2.0, 4.0]])
    np.testing.assert_array_equal(actions, [[-5.0, 7.0]])
    assert clipped is not actions


def test_torch_box_clipping_is_out_of_place():
    action_space = gym.spaces.Box(low=np.array([-2.0, 1.0]), high=np.array([3.0, 4.0]), dtype=np.float32)
    actions = torch.tensor([[-5.0, 7.0]])

    clipped = _clip_actions_to_space(action_space, actions)

    torch.testing.assert_close(clipped, torch.tensor([[-2.0, 4.0]]))
    torch.testing.assert_close(actions, torch.tensor([[-5.0, 7.0]]))
    assert clipped is not actions


def test_tuple_box_clipping_preserves_discrete_component():
    action_space = gym.spaces.Tuple(
        (
            gym.spaces.Discrete(3),
            gym.spaces.Box(low=np.array([-2.0, 1.0]), high=np.array([3.0, 4.0]), dtype=np.float32),
        )
    )
    actions = torch.tensor([[2.0, -5.0, 7.0]])

    clipped = _clip_actions_to_space(action_space, actions)

    torch.testing.assert_close(clipped, torch.tensor([[2.0, -2.0, 4.0]]))
    torch.testing.assert_close(actions, torch.tensor([[2.0, -5.0, 7.0]]))


def test_batched_preprocess_clips_box_before_env_and_keeps_discrete_processing():
    box_space = gym.spaces.Box(low=np.array([-2.0]), high=np.array([3.0]), dtype=np.float32)
    box_actions = torch.tensor([[-5.0]])
    box_result = preprocess_actions(_env_info(box_space), box_actions)
    torch.testing.assert_close(box_result, torch.tensor([[-2.0]]))

    discrete_space = gym.spaces.Discrete(3)
    discrete_actions = torch.tensor([[2.0]])
    discrete_result = preprocess_actions(_env_info(discrete_space), discrete_actions)
    torch.testing.assert_close(discrete_result, torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(discrete_actions, torch.tensor([[2.0]]))


def test_non_batched_actor_action_is_clipped_before_env_step():
    action_space = gym.spaces.Box(low=np.array([-2.0, 1.0]), high=np.array([3.0, 4.0]), dtype=np.float32)
    actor_state = ActorState.__new__(ActorState)
    actor_state.last_actions = np.array([-5.0, 7.0], dtype=np.float32)
    actor_state.env_info = _env_info(action_space, gpu_actions=False)

    clipped = actor_state.curr_actions()

    np.testing.assert_array_equal(clipped, [-2.0, 4.0])
    np.testing.assert_array_equal(actor_state.last_actions, [-5.0, 7.0])
