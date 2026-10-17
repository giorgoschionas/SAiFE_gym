"""Compare nominal PPO training returns at a fixed number of rollout updates.

Each point describes the episodes collected FOR that update, before its PPO
optimization. All episode returns are included, independently of SB3's rolling
logging window. Run ``train --help`` or ``plot --help`` for standalone usage.
"""

import argparse
import csv
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/saife_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import train_robust_lp_agent as training  # noqa: E402


TRAJECTORY_COUNTS = (10, 100, 1000, 10000)
SCHEMA_VERSION = 1
EXPERIMENT = "nominal_vectorization_convergence"
TRACE_COLUMNS = (
    "rollout_update",
    "mean_episode_reward",
    "std_episode_reward",
    "episode_count",
    "num_timesteps",
    "num_decision_timesteps",
    "elapsed_seconds",
)


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train one nominal vectorization setting.")
    train.add_argument("--num-trajectories", type=positive_int, required=True)
    train.add_argument("--rollout-updates", type=positive_int, default=100)
    train.add_argument("--seed", type=nonnegative_int, default=training.SEED)
    train.add_argument("--output-dir", type=Path, required=True, help="Sweep directory.")
    train.add_argument(
        "--smoke-test", action="store_true",
        help=("Use two updates, four simulator steps per episode and decision stride "
              "two. The requested trajectory count is preserved."),
    )
    plot = commands.add_parser("plot", help="Validate a completed sweep and plot its curves.")
    plot.add_argument("--input-dir", type=Path, required=True)
    plot.add_argument("--output-dir", type=Path, help="Defaults to INPUT_DIR/plots.")
    plot.add_argument(
        "--trajectory-counts", type=positive_int, nargs="+", default=TRAJECTORY_COUNTS,
        help="Expected settings; override only for a smaller smoke sweep.",
    )
    return parser.parse_args(argv)


def training_configuration(options: argparse.Namespace) -> argparse.Namespace:
    """Inherit nominal defaults and convert the update budget to SB3 timesteps."""
    args = training.parse_args([])
    args.num_trajectories = options.num_trajectories
    args.rollout_updates = 2 if options.smoke_test else options.rollout_updates
    args.seed = options.seed
    args.output_dir = str(options.output_dir)
    args.smoke_test = options.smoke_test
    args.convergence_eval_every_rollouts = 0
    if args.smoke_test:
        args.n_steps = 4
        args.decision_stride = 2
    if args.num_trajectories < 1 or args.rollout_updates < 1:
        raise ValueError("trajectory count and rollout updates must be positive")
    args.total_timesteps = args.rollout_updates * args.num_trajectories * args.n_steps
    training.validate_and_derive_decision_timing(args)
    return args


class RolloutRewardCallback(BaseCallback):
    """Persist all completed VecMonitor episode returns in each full rollout."""

    def __init__(self, args: argparse.Namespace, output_path: Path):
        super().__init__(verbose=0)
        self.args = args
        self.output_path = output_path
        self.rollouts_completed = 0
        self._returns: list[float] = []
        self._lengths: list[int] = []

    def _on_training_start(self) -> None:
        self.started_at = perf_counter()
        with self.output_path.open("x", newline="") as handle:
            csv.DictWriter(handle, fieldnames=TRACE_COLUMNS).writeheader()

    def _on_rollout_start(self) -> None:
        self._returns.clear()
        self._lengths.clear()

    def _on_step(self) -> bool:
        # SB3 exposes episode metadata as a list of dictionaries. Only extract
        # records here; the numerical reductions below use the full NumPy batch.
        episodes = [
            info["episode"]
            for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True)
            if done and "episode" in info
        ]
        self._returns.extend(episode["r"] for episode in episodes)
        self._lengths.extend(episode["l"] for episode in episodes)
        return True

    def _on_rollout_end(self) -> None:
        returns = np.asarray(self._returns, dtype=np.float64)
        if returns.size != self.args.num_trajectories:
            raise ValueError(
                "Expected one completed episode per trajectory in each rollout; "
                f"received {returns.size}, expected {self.args.num_trajectories}"
            )
        if not np.all(np.asarray(self._lengths) == self.args.n_steps):
            raise ValueError("Episode lengths do not match the simulator horizon")
        if not np.all(np.isfinite(returns)):
            raise ValueError("Non-finite training episode reward")
        self.rollouts_completed += 1
        row = {
            "rollout_update": self.rollouts_completed,
            "mean_episode_reward": float(returns.mean()),
            "std_episode_reward": float(returns.std(ddof=0)),
            "episode_count": int(returns.size),
            "num_timesteps": self.num_timesteps * self.args.decision_stride,
            "num_decision_timesteps": self.num_timesteps,
            "elapsed_seconds": perf_counter() - self.started_at,
        }
        # Close after every row so an interrupted HPC run retains its trace.
        with self.output_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=TRACE_COLUMNS).writerow(row)
        self.logger.record("rollout/all_episodes_reward_mean", row["mean_episode_reward"])
        self.logger.record("rollout/all_episodes_count", row["episode_count"])


def train_setting(options: argparse.Namespace) -> Path:
    args = training_configuration(options)
    run_dir = Path(args.output_dir) / f"trajectories_{args.num_trajectories}"
    # Never append a second attempt to an earlier run's configuration or trace.
    run_dir.mkdir(parents=True, exist_ok=False)
    environment_seed = args.seed + 20_000
    params = training.DomainParameters(
        sigma=args.nominal_sigma, arrival_rate=args.nominal_arrival_rate,
    )
    env = training.make_fixed_env(args, params, seed=environment_seed)
    model, ppo_env, vec_normalize = training.build_ppo(env, args, device="cpu")
    try:
        # The PPO builder seeds with the policy seed; restore the nominal stream.
        ppo_env.seed(environment_seed)
        config = {
            **vars(args),
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "environment_seed": environment_seed,
            "environment_constants": {
                "num_ticks": training.NUM_TICKS,
                "initial_price": training.INITIAL_PRICE,
                "fee_tier": training.FEE_TIER,
            },
            "ppo": {
                "batch_size": model.batch_size,
                "n_steps": model.n_steps,
                "n_epochs": model.n_epochs,
                "gamma": model.gamma,
                "gae_lambda": model.gae_lambda,
                "normalize_advantage": model.normalize_advantage,
                "policy_kwargs": model.policy_kwargs,
                "device": str(model.device),
            },
            "versions": {
                package: version(package)
                for package in ("numpy", "stable-baselines3", "torch", "gymnasium")
            },
        }
        training.save_json(run_dir / "config.json", config)
        callback = RolloutRewardCallback(args, run_dir / "training_rewards.csv")
        print(
            f"Training {args.num_trajectories:,} trajectories for "
            f"{args.rollout_updates} rollout updates "
            f"({args.total_timesteps:,} simulator transitions). Output: {run_dir}",
            flush=True,
        )
        started_at = perf_counter()
        model.learn(total_timesteps=args.ppo_total_timesteps, callback=callback)
        if (callback.rollouts_completed != args.rollout_updates
                or model.num_timesteps != args.ppo_total_timesteps):
            raise RuntimeError("Training did not complete the exact requested rollout budget")
        model.save(str(run_dir / "nominal_ppo"))
        if vec_normalize is not None:
            vec_normalize.save(str(run_dir / "nominal_vecnormalize.pkl"))
        completion = {
            "status": "completed",
            "rollouts_completed": callback.rollouts_completed,
            "num_decision_timesteps": model.num_timesteps,
            "num_timesteps": model.num_timesteps * args.decision_stride,
            "elapsed_seconds": perf_counter() - started_at,
        }
        temporary = run_dir / "completion.json.tmp"
        training.save_json(temporary, completion)
        temporary.replace(run_dir / "completion.json")
    finally:
        ppo_env.close()
    return run_dir


def comparable_configuration(config: dict) -> dict:
    """Only vectorization, its derived budget/batch size, and paths may vary."""
    varying = {"output_dir", "num_trajectories", "total_timesteps", "ppo_total_timesteps"}
    result = {key: value for key, value in config.items() if key not in varying}
    result["ppo"] = {key: value for key, value in config["ppo"].items() if key != "batch_size"}
    return result


def load_run(run_dir: Path, num_trajectories: int) -> tuple[dict, list[dict]]:
    config = json.loads((run_dir / "config.json").read_text())
    completion = json.loads((run_dir / "completion.json").read_text())
    if (config["schema_version"] != SCHEMA_VERSION or config["experiment"] != EXPERIMENT
            or config["num_trajectories"] != num_trajectories):
        raise ValueError(f"Unexpected experiment configuration in {run_dir}")
    updates = config["rollout_updates"]
    simulator_steps = num_trajectories * config["n_steps"]
    decision_steps = num_trajectories * config["ppo_n_steps"]
    rollout_size = config["ppo_n_steps"] * num_trajectories
    if (updates < 1 or config["n_steps"] != config["ppo_n_steps"] * config["decision_stride"]
            or config["total_timesteps"] != updates * simulator_steps
            or config["ppo_total_timesteps"] != updates * decision_steps
            or config["ppo"]["batch_size"] != min(max(64, rollout_size // 16), rollout_size)):
        raise ValueError(f"Inconsistent training budget or minibatch size in {run_dir}")
    if (completion["status"] != "completed"
            or completion["rollouts_completed"] != updates
            or completion["num_timesteps"] != config["total_timesteps"]
            or completion["num_decision_timesteps"] != config["ppo_total_timesteps"]):
        raise ValueError(f"Incomplete training in {run_dir}")
    required_artifacts = [run_dir / "nominal_ppo.zip"]
    if config["normalise_obs"]:
        required_artifacts.append(run_dir / "nominal_vecnormalize.pkl")
    for artifact in required_artifacts:
        if not artifact.is_file():
            raise ValueError(f"Missing training artifact: {artifact}")
    with (run_dir / "training_rewards.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(TRACE_COLUMNS):
            raise ValueError(f"Unexpected training reward CSV columns in {run_dir}")
        rows = list(reader)
    if len(rows) != updates:
        raise ValueError(f"Expected {updates} rollout rows in {run_dir}, found {len(rows)}")
    previous_elapsed = 0.0
    for index, row in enumerate(rows, start=1):
        for key in ("rollout_update", "episode_count", "num_timesteps", "num_decision_timesteps"):
            row[key] = int(row[key])
        for key in ("mean_episode_reward", "std_episode_reward", "elapsed_seconds"):
            row[key] = float(row[key])
            if not np.isfinite(row[key]):
                raise ValueError(f"Non-finite {key} in {run_dir}")
        if (row["rollout_update"] != index or row["episode_count"] != num_trajectories
                or row["num_timesteps"] != index * simulator_steps
                or row["num_decision_timesteps"] != index * decision_steps
                or row["std_episode_reward"] < 0
                or row["elapsed_seconds"] < previous_elapsed):
            raise ValueError(f"Invalid rollout row {index} in {run_dir}")
        previous_elapsed = row["elapsed_seconds"]
    return config, rows


def plot_sweep(options: argparse.Namespace) -> Path:
    counts = tuple(options.trajectory_counts)
    if len(set(counts)) != len(counts):
        raise ValueError("Expected trajectory counts must be unique")
    runs = [
        load_run(options.input_dir / f"trajectories_{count}", count)
        for count in counts
    ]
    reference = comparable_configuration(runs[0][0])
    for count, (config, _) in zip(counts, runs, strict=True):
        current = comparable_configuration(config)
        differences = sorted(
            key for key in reference.keys() | current.keys()
            if reference.get(key) != current.get(key)
        )
        if differences:
            raise ValueError(f"Mismatched settings for {count} trajectories: {differences}")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    output_dir = options.output_dir or options.input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(9, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    combined_rows = []
    for count, (config, rows) in zip(counts, runs, strict=True):
        ax.plot(
            [row["rollout_update"] for row in rows],
            [row["mean_episode_reward"] for row in rows],
            label=f"{count:,} {'trajectory' if count == 1 else 'trajectories'}",
            linewidth=1.7, marker="o" if len(rows) == 1 else None,
        )
        combined_rows.extend(
            {"num_trajectories": count, "training_seed": config["seed"], **row}
            for row in rows
        )
    ax.set_xlabel("Rollout update")
    ax.set_ylabel("Mean training episode reward")
    title = "workflow smoke test" if reference["smoke_test"] else "vectorization convergence"
    ax.set_title(f"Nominal PPO: {title} (seed {reference['seed']})")
    if reference["rollout_updates"] > 1:
        ax.set_xlim(1, reference["rollout_updates"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend()
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"vectorization_convergence.{extension}", dpi=200)
    training.save_csv(output_dir / "combined_training_rewards.csv", combined_rows)
    print(f"Saved comparison plot and combined CSV to: {output_dir}")
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_args(argv)
    try:
        if options.command == "train":
            train_setting(options)
        else:
            plot_sweep(options)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
