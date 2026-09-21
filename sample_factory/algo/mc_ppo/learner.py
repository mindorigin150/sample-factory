"""Monte Carlo PPO updates and round-boundary checkpoints under SF's learner worker."""
import copy
import time
from pathlib import Path

import torch

from sample_factory.algo.learning.learner import Learner, model_initialization_data
from sample_factory.algo.mc_ppo.evaluation import comparison, episode_summary, evaluate, evaluation_env, write_json
from sample_factory.algo.mc_ppo.models import MCPPOActorCritic
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, LEARNER_TRAIN_STEPS, POLICY_ID_KEY, TRAIN_STATS
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.envs.create_env import create_env
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import experiment_dir


@torch.no_grad()
def joint_kl(policy, data):
    means = policy.logits(data["obs"])[:, policy.active]
    return ((means - data["means"]) / policy.std).square().sum(1).mean() / 2


def update(policy, value, actor_optimizer, value_optimizer, data, settings, *, epochs, actor):
    """Keep accepted actor updates inside the round's measured KL budget."""
    actor_steps, value_steps, rejected, stopped = 0, 0, 0, not actor
    rollback = {"early_stop": False, "rollback": True}[settings["kl_control"]]
    advantages = data["advantages"]
    if settings["normalize_advantages"]:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-7)
    before_kl = joint_kl(policy, data).item()
    proposed_kl = before_kl
    for _ in range(epochs):
        for indices in torch.randperm(len(data["obs"]), device=data["obs"].device).split(settings["minibatch_size"]):
            obs = data["obs"][indices]
            if not stopped:
                means = policy.logits(obs)[:, policy.active]
                if not rollback:
                    proposed_kl = (((means.detach() - data["means"][indices]) / policy.std).square().sum(1).mean() / 2).item()
                    stopped = proposed_kl > settings["target_kl"]
                if not stopped:
                    ratio = (policy.distribution(means).log_prob(data["latent"][indices]) - data["log_prob"][indices]).exp()
                    advantage = advantages[indices]
                    clipped = ratio.clamp(1 - settings["clip_ratio"], 1 + settings["clip_ratio"])
                    loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                    actor_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if rollback:
                        parameters = [parameter.detach().clone() for parameter in policy.actor.parameters()]
                        optimizer_state = copy.deepcopy(actor_optimizer.state_dict())
                    actor_optimizer.step()
                    if rollback:
                        proposed_kl = joint_kl(policy, data).item()
                        if proposed_kl > settings["target_kl"]:
                            with torch.no_grad():
                                for parameter, previous in zip(policy.actor.parameters(), parameters):
                                    parameter.copy_(previous)
                            actor_optimizer.load_state_dict(optimizer_state)
                            rejected += 1
                            stopped = True
                    if not stopped:
                        actor_steps += 1
            predictions = value(torch.cat((obs, data["remaining"][indices]), 1)).squeeze(1)
            value_loss = (predictions - data["returns"][indices]).square().mean()
            value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            value_optimizer.step()
            value_steps += 1
    with torch.no_grad():
        means = policy.logits(data["obs"])[:, policy.active]
        ratio = (policy.distribution(means).log_prob(data["latent"]) - data["log_prob"]).exp()
        predicted = value(torch.cat((data["obs"], data["remaining"]), 1)).squeeze(1)
        variance = data["returns"].var().item()
        return {
            "actor_loss": -torch.minimum(ratio * advantages, ratio.clamp(1 - settings["clip_ratio"], 1 + settings["clip_ratio"]) * advantages).mean().item(),
            "value_loss": (predicted - data["returns"]).square().mean().item(),
            "value_explained_variance": 1 - (predicted - data["returns"]).var().item() / variance if variance else None,
            "advantage_mean": data["advantages"].mean().item(), "advantage_std": data["advantages"].std().item(),
            "kl_before": before_kl, "kl_proposed": proposed_kl, "kl": joint_kl(policy, data).item(),
            "clip_fraction": ((ratio - 1).abs() > settings["clip_ratio"]).float().mean().item(),
            "actor_updates": actor_steps, "value_updates": value_steps, "rejected_updates": rejected,
        }


@torch.no_grad()
def prepare_batch(model, batch, episode_frames):
    """Exclude padding and compute undiscounted returns through the actual terminal."""
    # Dynamic worker scheduling must not change optimizer shuffling on resume.
    order = batch["obs"]["seed"][:, 0, 0].argsort()
    batch = batch[order]
    valid = batch["valids"][:, :-1]
    obs = {key: tensor[:, :-1][valid] for key, tensor in batch["obs"].items()}
    frames = batch["raw_frames"]
    returns = frames.flip(1).cumsum(1).flip(1)[valid].float() / episode_frames
    data = {"obs": model.policy.normalize(obs), "remaining": obs["remaining"],
            "latent": batch["latent_actions"][valid][:, model.policy.active],
            "means": batch["action_logits"][valid][:, :model.action_space.shape[0]][:, model.policy.active],
            "log_prob": batch["log_prob_actions"][valid], "returns": returns}
    data["advantages"] = returns - batch["values"][:, :-1][valid]
    episodes = [{"seed": seed, "length": length, "return": reward} for seed, length, reward in zip(
        batch["obs"]["seed"][:, 0, 0].tolist(), frames.sum(1).tolist(), batch["raw_rewards"].sum(1).tolist())]
    return data, episodes


class MCPPOLearner(Learner):
    """Own the MC optimizers, frozen BC reference, evaluation, and SF checkpoints."""

    def init(self):
        torch.manual_seed(self.cfg.seed)
        self.device = policy_device(self.cfg, self.policy_id)
        self.settings = self.cfg.ppo
        self.directory = Path(experiment_dir(self.cfg))
        self.actor_critic = MCPPOActorCritic(self.env_info.obs_space, self.env_info.action_space, self.cfg).to(self.device)
        self.actor_critic.policy.load_state_dict(torch.load(self.cfg.initial_model_path, map_location=self.device, weights_only=False)["model"])
        self.reference = copy.deepcopy(self.actor_critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.policy.actor.parameters(), lr=self.settings["actor_lr"])
        self.value_optimizer = torch.optim.Adam(self.actor_critic.value.parameters(), lr=self.settings["value_lr"])
        self.round, self.value_updates, self.episodes, self.best_round = -1, 0, 0, 0
        self.evaluator = None
        if self.cfg.restart_behavior == "resume":
            self.load_from_checkpoint(self.policy_id)
        if self.device.type == "cpu":
            self.actor_critic.share_memory()
        self.param_server.init(self.actor_critic, self.round + 1, self.device)
        self.is_initialized = True
        if not self.cfg.benchmark:
            self.evaluator = evaluation_env(self.cfg)
            baseline = evaluate(self.evaluator, self.reference, range(*self.settings["development_seeds"]))
            write_json(self.directory / "baseline_development.json", baseline)
            if self.round == -1:
                self.best_performance = baseline["mean_length"]
                self._save_impl("best", "_development", 1)
        self.sample_started = time.perf_counter()
        return model_initialization_data(self.cfg, self.policy_id, self.actor_critic, self.round + 1, self.device)

    def _load_state(self, state, load_progress=True):
        self.actor_critic.load_state_dict(state["model"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])
        self.round, self.train_step, self.env_steps = state["round"], state["train_step"], state["env_steps"]
        self.value_updates, self.episodes = state["value_updates"], state["episodes"]
        self.best_performance, self.best_round = state["best_performance"], state["best_round"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)

    def _get_checkpoint_dict(self):
        return {"model": self.actor_critic.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(), "round": self.round,
                "train_step": self.train_step, "env_steps": self.env_steps,
                "value_updates": self.value_updates, "episodes": self.episodes,
                "best_performance": self.best_performance, "best_round": self.best_round,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None}

    def save_best(self, policy_id, metric, metric_value):
        # SF's periodic training-return best cannot replace fixed-seed selection.
        if metric != "development":
            return False
        return super().save_best(policy_id, metric, metric_value)

    def train(self, batch):
        sampling_seconds = time.perf_counter() - self.sample_started
        started = time.perf_counter()
        data, episodes = prepare_batch(self.actor_critic, batch, self.settings["episode_frames"])
        self.round += 1
        with self.param_server.policy_lock:
            learned = update(self.actor_critic.policy, self.actor_critic.value, self.actor_optimizer, self.value_optimizer,
                             data, self.settings, epochs=0 if self.cfg.benchmark else
                             (self.settings["value_warmup_epochs"] if self.round == 0 else self.settings["epochs"]),
                             actor=self.round > 0 and not self.cfg.benchmark)
            self.train_step += learned["actor_updates"]
            self.value_updates += learned["value_updates"]
            self.env_steps += sum(row["length"] for row in episodes)
            self.episodes += len(episodes)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.param_server.update_weights(self.round + 1)
        learning_seconds = time.perf_counter() - started
        frames = sum(row["length"] for row in episodes)
        record = {"round": self.round, "sampling": {**episode_summary(episodes), "frames": frames,
                  "seconds": sampling_seconds, "frames_per_second": frames / sampling_seconds},
                  "learning": learned, "learning_seconds": learning_seconds,
                  "counters": {"round": self.round, "actor_updates": self.train_step,
                               "value_updates": self.value_updates, "env_frames": self.env_steps, "episodes": self.episodes}}
        if not self.cfg.benchmark and self.round > 0 and (self.round % self.settings["eval_every"] == 0 or self.round == self.settings["rounds"]):
            result = evaluate(self.evaluator, self.actor_critic, range(*self.settings["development_seeds"]))
            record["evaluation"] = result
            if result["mean_length"] > self.best_performance:
                self.best_round = self.round
                self.save_best(self.policy_id, "development", result["mean_length"])
        self.save()
        write_json(self.directory / f"round_{self.round:04d}.json", record)
        report = {"round": self.round, **{key: value for key, value in learned.items() if value is not None},
                  "sampling_seconds": sampling_seconds, "sampling_frames_per_second": frames / sampling_seconds,
                  "sampling_mean_return": record["sampling"]["mean_return"], "learning_seconds": learning_seconds,
                  "best_development_length": self.best_performance}
        if "evaluation" in record:
            report.update({"eval_" + key: record["evaluation"][key] for key in ("mean_return", "mean_length", "ge100", "ge500")})
        self.sample_started = time.perf_counter()
        return {POLICY_ID_KEY: self.policy_id, LEARNER_ENV_STEPS: self.env_steps,
                LEARNER_TRAIN_STEPS: self.train_step, TRAIN_STATS: report}

    def close(self):
        if self.evaluator is not None:
            self.evaluator.close(terminate=True)
            self.evaluator = None


def evaluate_checkpoint(cfg, *, final=False):
    """Evaluate immutable SF best/latest snapshots without changing learner state."""
    device = policy_device(cfg, 0)
    env = create_env(cfg.env, cfg, AttrDict(worker_index=0, vector_index=0, env_id=0))
    try:
        model = MCPPOActorCritic(env.observation_space, env.action_space, cfg).to(device)
    finally:
        env.close()
    model.policy.load_state_dict(torch.load(cfg.initial_model_path, map_location=device, weights_only=False)["model"])
    directory = Path(experiment_dir(cfg))
    output = directory if final else directory / f"eval_{cfg.load_checkpoint_kind}_{time.time_ns()}"
    output.mkdir(exist_ok=True)
    env = evaluation_env(cfg)
    try:
        baseline = evaluate(env, model, range(*cfg.ppo["test_seeds"]))
        results = {"baseline": baseline, "evaluation_completed": True}
        kinds = ("best", "latest") if final else (cfg.load_checkpoint_kind,)
        for kind in kinds:
            prefix = "best" if kind == "best" else "checkpoint"
            path = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, 0), f"{prefix}_*.pth")[-1]
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            measured = evaluate(env, model, range(*cfg.ppo["test_seeds"]))
            results[kind] = {**measured, "round": state["round"], "checkpoint": path,
                             "comparison": comparison(measured["episodes"], baseline["episodes"])}
        if final:
            results["training_completed"] = state["round"] == cfg.ppo["rounds"]
        write_json(output / "result.json", results)
    finally:
        env.close(terminate=True)
    return results
