"""Compare nominal PPO policies within a common wall-clock training budget.

Training saves post-update policies and their observation normalization. Offline
evaluation measures every snapshot on the same fixed-size nominal episode batch.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
import socket
import sys
import tarfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/saife_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import train_robust_lp_agent as training  # noqa: E402
from experiments.vectorization_convergence import nonnegative_int, positive_int  # noqa: E402

TRAJECTORY_COUNTS = (10, 100, 1000)
EXPERIMENT = "nominal_vectorization_wallclock"
SCHEMA_VERSION = 1
TRAINING_COLUMNS = (
    "rollout_update", "elapsed_seconds", "num_decision_timesteps", "num_timesteps",
    "within_budget", "mean_episode_reward", "std_episode_reward", "episode_count",
)


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a fresh policy for a time budget.")
    train.add_argument("--num-trajectories", type=positive_int, required=True)
    train.add_argument("--training-seconds", type=positive_seconds, default=3600.0)
    train.add_argument("--checkpoint-seconds", type=positive_seconds, default=300.0)
    train.add_argument("--evaluation-episodes", type=positive_int, default=1000)
    train.add_argument("--evaluation-seed", type=nonnegative_int, default=100042)
    train.add_argument("--seed", type=nonnegative_int, default=42)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument(
        "--smoke-test", action="store_true",
        help="Use 2 seconds, 1-second checkpoints, 4 simulator steps, stride 2, and 4 evaluation episodes.",
    )
    evaluate = commands.add_parser("evaluate", help="Evaluate saved snapshots; safe to rerun.")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    plot = commands.add_parser("plot", help="Validate, plot and package a completed sweep.")
    plot.add_argument("--input-dir", type=Path, required=True)
    plot.add_argument("--output-dir", type=Path)
    plot.add_argument("--trajectory-counts", type=positive_int, nargs="+", default=TRAJECTORY_COUNTS)
    return parser.parse_args(argv)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    training.save_csv(temporary, rows)
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def training_configuration(options):
    args = training.parse_args([])
    args.num_trajectories = options.num_trajectories
    args.seed = options.seed
    args.evaluation_seed = options.evaluation_seed
    args.smoke_test = options.smoke_test
    args.convergence_eval_every_rollouts = 0
    if options.smoke_test:
        args.n_steps = 4
        args.decision_stride = 2
    # Each learn() call collects and optimizes one complete rollout. These are
    # per-call sample counts, not the experiment's stopping budget.
    args.total_timesteps = args.num_trajectories * args.n_steps
    training.validate_and_derive_decision_timing(args)
    return args


class EpisodeRewards(BaseCallback):
    """Extract every episode in a rollout, independently of SB3's log window."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.episodes = []

    def _on_rollout_start(self):
        self.episodes.clear()

    def _on_step(self):
        self.episodes.extend(
            info["episode"]
            for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True)
            if done and "episode" in info
        )
        return True

    def _on_rollout_end(self):
        if (len(self.episodes) != self.args.num_trajectories
                or any(episode["l"] != self.args.n_steps for episode in self.episodes)):
            raise ValueError("Expected one complete episode per trajectory per rollout")
        returns = np.asarray([episode["r"] for episode in self.episodes], dtype=np.float64)
        if not np.isfinite(returns).all():
            raise ValueError("Non-finite training episode returns")
        self.statistics = {
            "mean_episode_reward": float(returns.mean()),
            "std_episode_reward": float(returns.std()),
            "episode_count": int(returns.size),
        }


@dataclass
class Snapshot:
    metadata: dict
    model: bytes
    normalization: bytes


def capture_snapshot(model, normalize, args, rollouts, elapsed):
    stream = io.BytesIO()
    model.save(stream)
    return Snapshot(
        metadata={
            "checkpoint_id": f"update_{rollouts:06d}",
            "elapsed_seconds": elapsed,
            "rollouts_completed": rollouts,
            "num_decision_timesteps": int(model.num_timesteps),
            "num_timesteps": int(model.num_timesteps * args.decision_stride),
        },
        model=stream.getvalue(),
        normalization=pickle.dumps(normalize, protocol=pickle.HIGHEST_PROTOCOL),
    )


def save_snapshot(run_dir, snapshot, manifest):
    checkpoint_id = snapshot.metadata["checkpoint_id"]
    if manifest and manifest[-1]["checkpoint_id"] == checkpoint_id:
        return
    directory = run_dir / "checkpoints" / checkpoint_id
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "ppo.zip"
    normalize_path = directory / "vecnormalize.pkl"
    model_path.write_bytes(snapshot.model)
    normalize_path.write_bytes(snapshot.normalization)
    manifest.append({
        **snapshot.metadata,
        "model_file": str(model_path.relative_to(run_dir)),
        "normalization_file": str(normalize_path.relative_to(run_dir)),
        "model_sha256": digest(snapshot.model),
        "normalization_sha256": digest(snapshot.normalization),
    })
    atomic_json(run_dir / "checkpoints.json", manifest)


def train_cycles(model, normalize, args, run_dir, budget, interval, *, clock=perf_counter):
    """Retain the latest policy completed by the deadline, including its stats.

Only the latest eligible snapshot is cached in memory between scheduled disk
checkpoints. This lets an overrun finish without replacing the in-budget policy.
"""
    manifest = []
    latest = capture_snapshot(model, normalize, args, 0, 0.0)
    save_snapshot(run_dir, latest, manifest)
    callback = EpisodeRewards(args)
    next_checkpoint = interval
    rollouts = 0
    started = clock()
    with (run_dir / "training_rewards.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_COLUMNS)
        writer.writeheader()
        handle.flush()
        while clock() - started < budget:
            # SB3 preserves _last_obs, optimizer and environment state with
            # reset_num_timesteps=False. Nominal learning/clip rates are constant.
            model.learn(
                total_timesteps=args.ppo_total_timesteps,
                reset_num_timesteps=False, callback=callback,
            )
            elapsed = clock() - started
            rollouts += 1
            if model.num_timesteps != rollouts * args.ppo_total_timesteps:
                raise RuntimeError("PPO did not complete exactly one rollout")
            row = {
                "rollout_update": rollouts,
                "elapsed_seconds": elapsed,
                "num_decision_timesteps": int(model.num_timesteps),
                "num_timesteps": int(model.num_timesteps * args.decision_stride),
                "within_budget": elapsed <= budget,
                **callback.statistics,
            }
            writer.writerow(row)
            handle.flush()
            if elapsed <= budget:
                latest = capture_snapshot(model, normalize, args, rollouts, elapsed)
                if elapsed >= next_checkpoint:
                    save_snapshot(run_dir, latest, manifest)
                    next_checkpoint = (math.floor(elapsed / interval) + 1) * interval
                    print(f"Saved {latest.metadata['checkpoint_id']} at {elapsed:.1f}s", flush=True)
        stopped = clock() - started
    save_snapshot(run_dir, latest, manifest)
    completion = {
        "status": "completed", "training_seconds": budget,
        "elapsed_seconds": stopped, "overrun_seconds": max(0.0, stopped - budget),
        "rollouts_completed": rollouts,
        "eligible_rollouts_completed": latest.metadata["rollouts_completed"],
        "last_checkpoint_id": latest.metadata["checkpoint_id"],
        "num_decision_timesteps": int(model.num_timesteps),
        "num_timesteps": int(model.num_timesteps * args.decision_stride),
    }
    atomic_json(run_dir / "training_completion.json", completion)
    return completion


def train_setting(options, *, clock=perf_counter):
    args = training_configuration(options)
    run_dir = options.output_dir / f"trajectories_{args.num_trajectories}"
    run_dir.mkdir(parents=True, exist_ok=False)
    params = training.DomainParameters(args.nominal_sigma, args.nominal_arrival_rate)
    env = training.make_fixed_env(args, params, seed=args.seed + 20_000)
    model, ppo_env, normalize = training.build_ppo(env, args, device="cpu")
    try:
        ppo_env.seed(args.seed + 20_000)
        # Reuse a logger across consecutive learn calls; keep PPO's useful logs.
        model.set_logger(configure(str(run_dir), ["stdout"]))
        config = {
            "schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT,
            "training": {k: v for k, v in vars(args).items() if k not in (
                "output_dir", "total_timesteps", "ppo_total_timesteps",
            )},
            "training_seconds": 2.0 if options.smoke_test else options.training_seconds,
            "checkpoint_seconds": 1.0 if options.smoke_test else options.checkpoint_seconds,
            "evaluation": {
                "episodes": 4 if options.smoke_test else options.evaluation_episodes,
                "seed": options.evaluation_seed, "deterministic": True,
            },
            "environment_seed": args.seed + 20_000,
            "environment_constants": {
                "num_ticks": training.NUM_TICKS, "initial_price": training.INITIAL_PRICE,
                "fee_tier": training.FEE_TIER,
            },
            "ppo": {
                "batch_size": model.batch_size, "n_steps": model.n_steps,
                "n_epochs": model.n_epochs, "gamma": model.gamma,
                "gae_lambda": model.gae_lambda, "policy_kwargs": model.policy_kwargs,
                "device": str(model.device), "normalize_advantage": model.normalize_advantage,
            },
            "versions": {name: version(name) for name in (
                "numpy", "torch", "stable-baselines3", "gymnasium",
            )},
            "runtime": {"hostname": socket.gethostname(), "threads": os.getenv("OMP_NUM_THREADS")},
        }
        atomic_json(run_dir / "config.json", config)
        print(f"Training {args.num_trajectories} trajectories for {config['training_seconds']}s: {run_dir}", flush=True)
        train_cycles(model, normalize, args, run_dir, config["training_seconds"],
                     config["checkpoint_seconds"], clock=clock)
    finally:
        ppo_env.close()
    return run_dir


def load_training(run_dir):
    config = json.loads((run_dir / "config.json").read_text())
    manifest = json.loads((run_dir / "checkpoints.json").read_text())
    done = json.loads((run_dir / "training_completion.json").read_text())
    if config["experiment"] != EXPERIMENT or config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unexpected wall-clock experiment schema")
    args = config["training"]
    budget = config["training_seconds"]
    size = args["num_trajectories"] * args["ppo_n_steps"]
    if (not math.isfinite(budget) or budget <= 0
            or not math.isfinite(config["checkpoint_seconds"]) or config["checkpoint_seconds"] <= 0
            or config["evaluation"]["episodes"] < 1
            or config["evaluation"]["seed"] < 0
            or config["evaluation"]["deterministic"] is not True
            or args["n_steps"] != args["ppo_n_steps"] * args["decision_stride"]
            or config["ppo"]["batch_size"] != min(max(64, size // 16), size)):
        raise ValueError("Invalid wall-clock experiment configuration")
    if (done["status"] != "completed" or done["training_seconds"] != budget
            or not math.isfinite(done["elapsed_seconds"]) or done["elapsed_seconds"] < budget
            or not math.isclose(done["overrun_seconds"], done["elapsed_seconds"] - budget)
            or done["num_decision_timesteps"] != done["rollouts_completed"] * size
            or done["num_timesteps"] != done["num_decision_timesteps"] * args["decision_stride"]
            or not manifest):
        raise ValueError("Incomplete wall-clock training")
    with (run_dir / "training_rewards.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(TRAINING_COLUMNS):
            raise ValueError("Invalid training reward columns")
        trace = list(reader)
    if len(trace) != done["rollouts_completed"]:
        raise ValueError("Missing completed rollout rows")
    last_elapsed, last_eligible = 0.0, 0
    for index, row in enumerate(trace, 1):
        elapsed = float(row["elapsed_seconds"])
        if (not math.isfinite(elapsed) or not last_elapsed < elapsed <= done["elapsed_seconds"]
                or int(row["rollout_update"]) != index
                or int(row["episode_count"]) != args["num_trajectories"]
                or int(row["num_decision_timesteps"]) != index * size
                or int(row["num_timesteps"]) != index * size * args["decision_stride"]
                or row["within_budget"] != str(elapsed <= budget)
                or not math.isfinite(float(row["mean_episode_reward"]))
                or not math.isfinite(float(row["std_episode_reward"]))
                or float(row["std_episode_reward"]) < 0):
            raise ValueError("Invalid completed rollout record")
        last_elapsed = elapsed
        if elapsed <= budget:
            last_eligible = index
    if last_eligible != done["eligible_rollouts_completed"]:
        raise ValueError("Final checkpoint is not the last eligible policy")
    previous_time, previous_rollout = -1, -1
    for item in manifest:
        rollout, elapsed = item["rollouts_completed"], item["elapsed_seconds"]
        if (not math.isfinite(elapsed) or not previous_time < elapsed <= budget
                or not previous_rollout < rollout <= done["rollouts_completed"]
                or item["checkpoint_id"] != f"update_{rollout:06d}"
                or (rollout > 0 and elapsed != float(trace[rollout - 1]["elapsed_seconds"]))
                or item["num_decision_timesteps"] != rollout * size
                or item["num_timesteps"] != rollout * args["num_trajectories"] * args["n_steps"]):
            raise ValueError("Invalid checkpoint timing or transition counts")
        for name in ("model", "normalization"):
            path = run_dir / item[f"{name}_file"]
            if not path.resolve().is_relative_to(run_dir.resolve()):
                raise ValueError("Checkpoint path must be inside its run directory")
            if digest(path.read_bytes()) != item[f"{name}_sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {path}")
        previous_time, previous_rollout = elapsed, rollout
    if (manifest[0]["elapsed_seconds"] != 0 or manifest[0]["rollouts_completed"] != 0
            or manifest[-1]["checkpoint_id"] != done["last_checkpoint_id"]
            or manifest[-1]["rollouts_completed"] != done["eligible_rollouts_completed"]):
        raise ValueError("Missing initial or final eligible checkpoint")
    return config, manifest


def evaluate_checkpoint(run_dir, item, args, params):
    model = PPO.load(run_dir / item["model_file"], device="cpu")
    if model.num_timesteps != item["num_decision_timesteps"]:
        raise ValueError("Model does not match its checkpoint transition count")
    normalize, normalization_env = None, None
    try:
        if args.normalise_obs:
            # Only the observation transform is used by the evaluator. Attach a
            # small adapter to satisfy VecNormalize's load contract without
            # allocating a second large evaluation batch.
            normalize_args = argparse.Namespace(**vars(args))
            normalize_args.num_trajectories = 1
            normalization_env = training.wrap_env(
                training.make_fixed_env(normalize_args, params, args.evaluation_seed),
                normalise_obs=False,
            )
            normalize = VecNormalize.load(run_dir / item["normalization_file"], normalization_env)
            normalize.training = False
            normalize.norm_reward = False
        return training.evaluate_ppo(
            "nominal_ppo", model, normalize, params, "convergence_validation",
            args, regime_seed=args.evaluation_seed,
        )
    finally:
        if normalization_env is not None:
            normalization_env.close()


def evaluate_setting(options):
    run_dir = options.run_dir
    config, manifest = load_training(run_dir)
    args = training.parse_args([])
    vars(args).update(config["training"])
    args.num_trajectories = config["evaluation"]["episodes"]
    args.n_eval_episodes = 1  # One native batch = exactly the requested episode count.
    args.evaluation_seed = config["evaluation"]["seed"]
    params = training.DomainParameters(args.nominal_sigma, args.nominal_arrival_rate)
    completion_path = run_dir / "evaluation_completion.json"
    completion_path.unlink(missing_ok=True)
    rows = []
    started = perf_counter()
    for item in manifest:
        evaluation_started = perf_counter()
        metrics = evaluate_checkpoint(run_dir, item, args, params)
        if metrics["n_evaluation_paths"] != args.num_trajectories:
            raise ValueError("Evaluation did not produce the configured episode count")
        rows.append({
            **{key: item[key] for key in (
                "checkpoint_id", "elapsed_seconds", "rollouts_completed",
                "num_decision_timesteps", "num_timesteps",
            )},
            "mean_episode_reward": metrics["mean_running_inventory_objective"],
            "std_episode_reward": metrics["evaluation_path_std_running_inventory_objective"],
            "episode_count": metrics["n_evaluation_paths"],
            "evaluation_seed": args.evaluation_seed,
            "evaluation_seconds": perf_counter() - evaluation_started,
        })
        atomic_csv(run_dir / "evaluation_rewards.csv", rows)
        print(f"Evaluated {item['checkpoint_id']}: mean reward {rows[-1]['mean_episode_reward']:.4f}", flush=True)
    atomic_json(completion_path, {
        "status": "completed", "checkpoint_count": len(rows),
        "evaluation": config["evaluation"], "elapsed_seconds": perf_counter() - started,
        "config_sha256": digest((run_dir / "config.json").read_bytes()),
        "checkpoints_sha256": digest((run_dir / "checkpoints.json").read_bytes()),
    })
    return run_dir


def load_evaluation(run_dir):
    config, manifest = load_training(run_dir)
    done = json.loads((run_dir / "evaluation_completion.json").read_text())
    with (run_dir / "evaluation_rewards.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if (done["status"] != "completed" or done["checkpoint_count"] != len(manifest)
            or len(rows) != len(manifest) or done["evaluation"] != config["evaluation"]
            or done["config_sha256"] != digest((run_dir / "config.json").read_bytes())
            or done["checkpoints_sha256"] != digest((run_dir / "checkpoints.json").read_bytes())):
        raise ValueError("Incomplete or stale evaluation")
    for row, item in zip(rows, manifest, strict=True):
        for key in ("rollouts_completed", "num_decision_timesteps", "num_timesteps", "episode_count", "evaluation_seed"):
            row[key] = int(row[key])
        for key in ("elapsed_seconds", "mean_episode_reward", "std_episode_reward", "evaluation_seconds"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"Non-finite evaluation value: {key}")
        if (any(row[key] != item[key] for key in (
                "checkpoint_id", "elapsed_seconds", "rollouts_completed", "num_decision_timesteps", "num_timesteps"))
                or row["episode_count"] != config["evaluation"]["episodes"]
                or row["evaluation_seed"] != config["evaluation"]["seed"]
                or row["std_episode_reward"] < 0 or row["evaluation_seconds"] < 0):
            raise ValueError("Evaluation does not match its checkpoint or protocol")
    return config, rows


def comparable_configuration(config):
    result = {key: value for key, value in config.items() if key != "runtime"}
    result["training"] = {key: value for key, value in config["training"].items() if key != "num_trajectories"}
    result["ppo"] = {key: value for key, value in config["ppo"].items() if key != "batch_size"}
    return result


def plot_sweep(options):
    counts = options.trajectory_counts
    if len(set(counts)) != len(counts):
        raise ValueError("Trajectory counts must be unique")
    runs = [load_evaluation(options.input_dir / f"trajectories_{count}") for count in counts]
    reference = comparable_configuration(runs[0][0])
    for count, (config, _) in zip(counts, runs, strict=True):
        if (config["training"]["num_trajectories"] != count
                or comparable_configuration(config) != reference):
            raise ValueError(f"Mismatched sweep configuration for {count} trajectories")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = options.output_dir or options.input_dir / "plots"
    output.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(9, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    combined = []
    for count, (config, rows) in zip(counts, runs, strict=True):
        ax.plot([row["elapsed_seconds"] / 60 for row in rows],
                [row["mean_episode_reward"] for row in rows],
                marker="o", markersize=3,
                label=f"{count:,} {'trajectory' if count == 1 else 'trajectories'}")
        combined.extend({"num_trajectories": count, "training_seed": config["training"]["seed"], **row} for row in rows)
    ax.set_xlim(0, reference["training_seconds"] / 60)
    ax.set_xlabel("Elapsed training time (minutes)")
    ax.set_ylabel("Mean evaluation episode reward")
    label = "Workflow smoke test" if reference["training"]["smoke_test"] else "Nominal PPO: reward versus training time"
    ax.set_title(f"{label}\n{reference['evaluation']['episodes']:,} held-out episodes per policy; training seed {reference['training']['seed']}")
    ax.grid(alpha=0.25)
    ax.legend()
    figure.supxlabel("Deterministic policies; offline evaluation time excluded; actual checkpoint times; no smoothing.", fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"vectorization_wallclock.{extension}", dpi=200)
    atomic_csv(output / "combined_evaluation_rewards.csv", combined)
    atomic_json(output / "comparison.json", {
        "trajectory_counts": list(counts), "protocol": reference,
        "note": "One training seed per setting; episode dispersion is not uncertainty across training seeds.",
    })
    archive = output / "vectorization_wallclock_review.tar.gz"
    temporary = archive.with_suffix(".gz.tmp")
    with tarfile.open(temporary, "w:gz") as bundle:
        for name in ("vectorization_wallclock.png", "vectorization_wallclock.pdf", "combined_evaluation_rewards.csv", "comparison.json"):
            bundle.add(output / name, arcname=name)
        for count in counts:
            directory = options.input_dir / f"trajectories_{count}"
            for name in ("config.json", "checkpoints.json", "training_completion.json", "evaluation_completion.json", "training_rewards.csv", "evaluation_rewards.csv"):
                bundle.add(directory / name, arcname=f"trajectories_{count}/{name}")
    temporary.replace(archive)
    print(f"Saved comparison and retrieval archive: {archive}", flush=True)
    return output


def main(argv=None):
    options = parse_args(argv)
    try:
        {"train": train_setting, "evaluate": evaluate_setting, "plot": plot_sweep}[options.command](options)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
