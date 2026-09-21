from typing import Tuple

import torch

from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.runners.runner_parallel import ParallelRunner
from sample_factory.algo.runners.runner_serial import SerialRunner
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import maybe_load_from_checkpoint
from sample_factory.pbt.population_based_training import PopulationBasedTraining
from sample_factory.utils.typing import Config


def make_runner(cfg: Config) -> Tuple[Config, Runner]:
    if cfg.restart_behavior == "resume":
        # if we're resuming from checkpoint, we load all of the config parameters from the checkpoint
        # unless they're explicitly specified in the command line
        cfg = maybe_load_from_checkpoint(cfg)

    if cfg.algo == "PPO":
        cfg.ppo_start_round = 0
        if cfg.restart_behavior == "resume":
            prefix = {"latest": "checkpoint", "best": "best"}[cfg.load_checkpoint_kind]
            path = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, 0), f"{prefix}_*.pth")[-1]
            cfg.ppo_start_round = torch.load(path, map_location="cpu", weights_only=False)["round"] + 1

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
