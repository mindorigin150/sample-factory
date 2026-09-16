"""Complete episodes with dynamic work assignment and one policy per round."""
import numpy as np
from signal_slot.signal_slot import signal

from sample_factory.algo.sampling.sampler import ParallelSampler, SerialSampler
from sample_factory.algo.utils.heartbeat import HeartbeatStoppableEventLoopObject
from sample_factory.algo.utils.misc import EPISODIC, POLICY_ID_KEY, new_trajectories_signal
from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
from sample_factory.algo.utils.tensor_dict import to_numpy
from sample_factory.cfg.configurable import Configurable
from sample_factory.envs.create_env import create_env
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.timing import Timing


class EpisodeSchedule:
    """Assign each seed once; open the next round only after the learner releases it."""

    def __init__(self, cfg):
        self.settings = cfg.ppo
        self.benchmark = cfg.benchmark
        self.state = get_mp_ctx(False).Array("q", [cfg.ppo_start_round, 0])

    def claim(self, round_index):
        with self.state.get_lock():
            if round_index > self.state[0]:
                self.state[:] = [round_index, 0]
            index = self.state[1]
            if index == self.settings["episodes_per_round"]:
                return None
            self.state[1] += 1
            offset = 0 if self.benchmark else round_index - 1
            return self.settings["train_seed_base"] + offset * self.settings["episodes_per_round"] + index


class EpisodeRolloutWorker(HeartbeatStoppableEventLoopObject, Configurable):
    """Write one masked episode buffer and request actions from SF's inference worker."""

    def __init__(self, event_loop, worker_idx, buffer_mgr, inference_queues, cfg, schedule):
        Configurable.__init__(self, cfg)
        HeartbeatStoppableEventLoopObject.__init__(self, event_loop, f"EpisodeRolloutWorker_w{worker_idx}", cfg.heartbeat_interval)
        self.worker_idx = worker_idx
        self.schedule = schedule
        self.buffer_mgr = buffer_mgr
        self.inference_queue = inference_queues[0]
        self.buffer_queue = buffer_mgr.traj_buffer_queues["cpu"]
        self.output_indices = np.cumsum(buffer_mgr.output_sizes)[:-1]
        self.round = cfg.ppo_start_round
        self.env = None
        self.timing = Timing(name=f"MC episode worker {worker_idx}")

    @signal
    def report_msg(self):
        ...

    def init(self):
        # NumPy views must be created after spawn; pickling them would copy storage.
        self.buffers = to_numpy(self.buffer_mgr.traj_tensors_torch["cpu"])
        self.outputs = to_numpy(self.buffer_mgr.policy_output_tensors_torch["cpu"])[self.worker_idx, 0, 0, 0]
        self.env = create_env(self.cfg.env, self.cfg, AttrDict(worker_index=self.worker_idx, vector_index=0, env_id=self.worker_idx))
        self._start_episode()

    def _start_episode(self):
        if self.round > self.cfg.ppo["rounds"]:
            return
        seed = self.schedule.claim(self.round)
        if seed is None:
            return
        self.buffer_idx = self.buffer_queue.get()
        self.buffer = self.buffers[self.buffer_idx]
        for _, _, array in iterate_recursively(self.buffer):
            array.fill(0)
        self.buffer["policy_id"].fill(-1)
        self.buffer["dones"].fill(True)
        self.step = 0
        self.obs, _ = self.env.reset(seed=seed)
        self._request()

    def _request(self):
        self.buffer["obs"][self.step] = self.obs
        self.inference_queue.put((self.worker_idx, 0, [(0, 0, self.buffer_idx, self.step)], "cpu", None))

    def advance_rollouts(self, split_idx, policy_id, event_handle=None):
        output = dict(zip(self.buffer_mgr.output_names, np.split(self.outputs, self.output_indices)))
        self.buffer[self.step] = {key: value for key, value in output.items() if key != "new_rnn_states"}
        with self.timing.add_time("simulation"):
            self.obs, reward, terminated, truncated, info = self.env.step(output["actions"])
        done = terminated or truncated
        self.buffer["rewards"][self.step] = reward
        self.buffer["raw_frames"][self.step] = info["frames"]
        self.buffer["raw_rewards"][self.step] = info["raw_reward"]
        self.buffer["dones"][self.step] = done
        self.buffer["time_outs"][self.step] = truncated
        self.buffer["policy_id"][self.step] = 0
        self.buffer["valids"][self.step] = True
        self.step += 1
        if done:
            self.buffer["obs"][self.step] = self.obs
            episode = info["episode"]
            self.report_msg.emit({POLICY_ID_KEY: 0, EPISODIC: {
                "reward": episode["return"], "len": episode["length"],
                "episode_extra_stats": {"episode/return": episode["return"], "episode/raw_frames": episode["length"]},
            }})
            self.emit(new_trajectories_signal(0), [{"policy_id": 0, "traj_buffer_idx": self.buffer_idx,
                                                  "length": self.step}], "cpu")
            self._start_episode()
        else:
            self._request()

    def on_trajectory_buffers_available(self, policy_id, training_iteration):
        next_round = self.cfg.ppo_start_round + training_iteration
        if next_round > self.round:
            self.round = next_round
            if self.env is not None:
                self._start_episode()

    def on_update_training_info(self, training_info):
        pass

    def on_stop(self, *args):
        if self.env is not None:
            self.env.close()
        self.stop.emit(self.object_id, {self.object_id: self.timing})
        super().on_stop(*args)


class EpisodeSampler:
    """Specialize SF's worker creation while retaining its processes and connections."""

    def __init__(self, event_loop, buffer_mgr, param_servers, cfg, env_info):
        self.schedule = EpisodeSchedule(cfg)
        super().__init__(event_loop, buffer_mgr, param_servers, cfg, env_info)

    def _make_rollout_worker(self, event_loop, worker_idx):
        return EpisodeRolloutWorker(event_loop, worker_idx, self.buffer_mgr, self.inference_queues,
                                    self.cfg, self.schedule)


class ParallelEpisodeSampler(EpisodeSampler, ParallelSampler):
    """Run episode workers and centralized SF inference in separate processes."""


class SerialEpisodeSampler(EpisodeSampler, SerialSampler):
    """Use the same episode protocol in SF's serial execution mode."""
