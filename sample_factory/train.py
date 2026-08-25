from typing import Tuple

from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.runners.runner_parallel import ParallelRunner
from sample_factory.algo.runners.runner_serial import SerialRunner
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import maybe_load_from_checkpoint
from sample_factory.pbt.population_based_training import PopulationBasedTraining
from sample_factory.utils.typing import Config


def _validate_wimle_cfg(cfg: Config) -> None:
    if cfg.algo != "WIMLE":
        return
    if cfg.restart_behavior == "resume":
        raise ValueError("WIMLE checkpoints are actor-only; use --restart_behavior=overwrite or restart")
    required = {
        "serial_mode": True,
        "async_rl": False,
        "batched_sampling": False,
        "num_workers": 1,
        "num_envs_per_worker": 1,
        "worker_num_splits": 1,
        "rollout": 1,
        "batch_size": 1,
        "num_batches_per_epoch": 1,
        "num_policies": 1,
        "use_rnn": False,
        "with_pbt": False,
        "normalize_input": False,
        "obs_subtract_mean": 0.0,
        "obs_scale": 1.0,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"WIMLE requires {key}={expected}, got {getattr(cfg, key)}")
    if cfg.wimle_rollout_horizon is None:
        raise ValueError("WIMLE requires --wimle_rollout_horizon")


def make_runner(cfg: Config) -> Tuple[Config, Runner]:
    if cfg.restart_behavior == "resume":
        # if we're resuming from checkpoint, we load all of the config parameters from the checkpoint
        # unless they're explicitly specified in the command line
        cfg = maybe_load_from_checkpoint(cfg)

    _validate_wimle_cfg(cfg)

    if cfg.serial_mode:
        runner_cls = SerialRunner
    else:
        runner_cls = ParallelRunner

    runner = runner_cls(cfg)

    if cfg.with_pbt:
        runner.register_observer(PopulationBasedTraining(cfg, runner))

    return cfg, runner


def run_rl(cfg: Config):
    cfg, runner = make_runner(cfg)

    # here we can register additional message or summary handlers
    # see sf_examples/dmlab/train_dmlab.py for example

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status
