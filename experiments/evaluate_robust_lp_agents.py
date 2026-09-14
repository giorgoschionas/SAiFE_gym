"""Evaluate periodic rebalancing and optionally saved PPO policies, without training.

Configurations and observation statistics come from saved training runs. Step
resolution can change while physical decision/rebalance intervals stay fixed.
Independent vectorized episodes are the units for bootstrap confidence intervals:
trajectories within an episode share the simulator's swap-order coin flips.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/saife_matplotlib")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluation_grid import configured_evaluation_grid  # noqa: E402
from experiments.policy_behavior_diagnostics import BEHAVIOR_DIAGNOSTIC_COLUMNS  # noqa: E402
from experiments import train_robust_lp_agent as training  # noqa: E402
from experiments.fast_lp_evaluation import make_evaluation_env  # noqa: E402


POLICIES = ("periodic_rebalance", "nominal_ppo", "domain_randomized_ppo")
REWARD = "mean_running_inventory_objective"
METRICS = (REWARD, *BEHAVIOR_DIAGNOSTIC_COLUMNS)
REQUIRED_CONFIG = {
    "n_steps", "terminal_time", "decision_stride", "num_trajectories",
    "periodic_rebalance_every", "periodic_width", "n_eval_episodes",
    "evaluation_seed", "seed", "tau", "tick_stride", "alpha3",
    "arrival_alpha2", "kernel_beta", "kernel_window", "liquidity_scale",
    "trade_size_notional", "price_impact_depth_window", "price_impact_min_depth",
    "nominal_gas_cost", "initial_wealth", "inventory_phi", "normalise_obs",
    "train_sigma_range", "train_arrival_rate_range",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New diagnostic directory; existing directories are rejected.")
    parser.add_argument("--policies", choices=POLICIES, nargs="+",
                        default=["periodic_rebalance"])
    parser.add_argument("--sigma-values", type=float, nargs="+",
                        default=[.03, .045, .065, .08])
    parser.add_argument("--arrival-rate-values", type=float, nargs="+",
                        default=[300., 450., 600., 800., 1000.])
    parser.add_argument("--n-steps-values", type=int, nargs="+",
                        help="Defaults to the source resolution; intervals scale automatically.")
    parser.add_argument("--n-eval-episodes", type=int)
    parser.add_argument("--num-trajectories", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--episode-offset", type=int, default=0,
                        help="Start later in each regime's seed block for independent confirmation.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--reference-evaluator", action="store_true",
                        help="Use original full-tick accounting and state snapshots.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the same sweep from its completed episode checkpoints.")
    return parser.parse_args(argv)


def load_config(run_dir):
    config = json.loads((run_dir / "config.json").read_text())
    missing = REQUIRED_CONFIG - config.keys()
    if missing:
        raise ValueError(f"missing configuration fields in {run_dir}: {sorted(missing)}")
    configured_evaluation_grid(config)  # Validate the historical axes and bounds layout.
    for name in ("train_sigma_range", "train_arrival_rate_range"):
        bounds = config[name]
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or not 0 <= bounds[0] <= bounds[1]:
            raise ValueError(f"invalid {name} in {run_dir}")
    return config


def resolution_args(config, n_steps, cli):
    args = training.parse_args([])
    vars(args).update(config)
    if n_steps < 1 or config["n_steps"] < 1:
        raise ValueError("simulator step counts must be positive")
    for name in ("decision_stride", "periodic_rebalance_every"):
        numerator = config[name] * n_steps
        if numerator % config["n_steps"]:
            raise ValueError(f"{n_steps} steps cannot preserve the saved {name} interval exactly")
        setattr(args, name, numerator // config["n_steps"])
        if getattr(args, name) < 1:
            raise ValueError(f"scaled {name} must be positive")
    args.n_steps = n_steps
    if n_steps % args.decision_stride:
        raise ValueError("n_steps must be divisible by the scaled decision_stride")
    args.max_agent_decisions_per_episode = n_steps // args.decision_stride
    for name in ("num_trajectories", "n_eval_episodes", "evaluation_seed"):
        if getattr(cli, name) is not None:
            setattr(args, name, getattr(cli, name))
    if args.num_trajectories < 1 or args.n_eval_episodes < 2:
        raise ValueError("require positive num_trajectories and at least two evaluation episodes")
    if not 0 <= args.evaluation_seed < 2**32 - 1_002_000_000:
        raise ValueError("evaluation_seed is outside the supported seed range")
    if not 1 <= args.periodic_width <= args.tau:
        raise ValueError("periodic_width must lie between 1 and tau")
    return args


def regime_seed(config, sigma, arrival, base_seed):
    """Retain historical cell seeds; new seeds are stable across subsets/resolutions."""
    for index, (_, old_sigma, old_arrival) in enumerate(configured_evaluation_grid(config)):
        if (sigma, arrival) == (old_sigma, old_arrival):
            return base_seed + 1000 * index
    coordinate = f"{float(sigma):.17g},{float(arrival):.17g}".encode()
    digest = hashlib.sha256(coordinate).digest()
    return base_seed + 1_000_000 + 1000 * (int.from_bytes(digest[:4], "big") % 1_000_000)


def evaluation_set(config, sigma, arrival):
    sigma_inside = config["train_sigma_range"][0] <= sigma <= config["train_sigma_range"][1]
    arrival_inside = config["train_arrival_rate_range"][0] <= arrival <= config["train_arrival_rate_range"][1]
    return {
        (True, True): "in_distribution", (True, False): "arrival_only_stress",
        (False, True): "sigma_only_stress", (False, False): "stress",
    }[sigma_inside, arrival_inside]


class ArrivalDiagnostics:
    """Measure generated indicators (not necessarily executed trades) and rate*dt."""

    def __init__(self):
        self.side_steps = 0
        self.arrivals = np.zeros(2, dtype=np.int64)
        self.rate_dt_sum = 0.0
        self.rate_dt_max = 0.0
        self.large_rate_dt = 0

    def __call__(self, env):
        dynamics = env.model_dynamics
        rate_dt = dynamics.arrival_model.current_state * env.step_size
        self.side_steps += rate_dt.size
        self.arrivals += dynamics.last_arrivals.sum(axis=0)
        self.rate_dt_sum += float(rate_dt.sum())
        self.rate_dt_max = max(self.rate_dt_max, float(rate_dt.max()))
        self.large_rate_dt += int((rate_dt >= 1).sum())

    def summarize(self, num_trajectories):
        return {
            "mean_sell_arrival_indicators_per_path": float(self.arrivals[0] / num_trajectories),
            "mean_buy_arrival_indicators_per_path": float(self.arrivals[1] / num_trajectories),
            "mean_intensity_times_dt": self.rate_dt_sum / self.side_steps,
            "maximum_intensity_times_dt": self.rate_dt_max,
            "fraction_side_steps_intensity_times_dt_ge_1": self.large_rate_dt / self.side_steps,
        }


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_episode_checkpoint(job):
    path = Path(job["episode_path"])
    if not path.exists():
        return []
    text_fields = {"policy", "evaluation_set", "source_run_dir"}
    integer_fields = {
        "source_training_seed", "evaluation_seed", "n_steps", "num_trajectories",
        "n_evaluation_paths", "decision_stride", "periodic_rebalance_every",
        "regime_seed", "episode_seed", "episode_offset", "episode_index",
    }
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"incomplete checkpoint row in {path}")
            row = {
                key: value if key in text_fields or value == "" else
                int(value) if key in integer_fields else float(value)
                for key, value in raw.items()
            }
            expected_episode = job["episode_offset"] + len(rows)
            expected = dict(
                policy=job["policy"], evaluation_set=job["evaluation_set"],
                source_run_dir=job["run_dir"], sigma=job["sigma"], arrival_rate=job["arrival"],
                regime_seed=job["regime_seed"], episode_index=expected_episode,
                episode_seed=job["regime_seed"] + expected_episode,
                episode_offset=job["episode_offset"],
                source_training_seed=job["args"]["seed"] if job["policy"] != "periodic_rebalance" else "",
                n_evaluation_paths=job["args"]["num_trajectories"],
                **{key: job["args"][key] for key in (
                    "n_steps", "num_trajectories", "decision_stride", "periodic_rebalance_every",
                )},
            )
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"checkpoint parameters or seed sequence disagree in {path}")
            if not all(np.isfinite(row.get(metric, np.nan)) for metric in METRICS):
                raise ValueError(f"non-finite or missing checkpoint metrics in {path}")
            if not np.isclose(row[REWARD], row["mean_pnl_per_path"] - row["mean_inventory_penalty_per_path"], rtol=1e-9, atol=1e-7):
                raise ValueError(f"checkpoint reward decomposition disagrees in {path}")
            rows.append(row)
    if len(rows) > job["args"]["n_eval_episodes"]:
        raise ValueError(f"too many completed episodes in {path}")
    return rows


def summarize_episodes(rows, resamples, bootstrap_seed):
    result = {key: rows[0][key] for key in (
        "policy", "source_training_seed", "source_run_dir", "evaluation_set",
        "sigma", "arrival_rate", "n_steps", "decision_stride", "periodic_rebalance_every",
        "regime_seed", "episode_offset", "num_trajectories",
    )}
    count = len(rows)
    result["n_independent_episodes"] = count
    result["n_evaluation_paths"] = sum(row["n_evaluation_paths"] for row in rows)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(count, size=(resamples, count))
    for metric in METRICS:
        values = np.array([row[metric] for row in rows])
        result[metric] = float(values.mean())
        lower, upper = np.quantile(values[indices].mean(axis=1), [.025, .975])
        result[f"{metric}_ci_lower"] = float(lower)
        result[f"{metric}_ci_upper"] = float(upper)
    means = np.array([row[REWARD] for row in rows])
    variances = np.array([row["evaluation_path_std_running_inventory_objective"]**2 for row in rows])
    result["evaluation_path_std_running_inventory_objective"] = float(np.sqrt(
        np.mean(variances + (means - means.mean())**2)
    ))
    for metric in rows[0]:
        if metric.startswith(("mean_sell_arrival_", "mean_buy_arrival_", "mean_intensity_", "fraction_side_steps_")):
            result[metric] = float(np.mean([row[metric] for row in rows]))
    if "maximum_intensity_times_dt" in rows[0]:
        result["maximum_intensity_times_dt"] = max(row["maximum_intensity_times_dt"] for row in rows)
    return result


def evaluate_job(job):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    torch.set_num_threads(1)
    args = argparse.Namespace(**job["args"])
    rows = read_episode_checkpoint(job)
    if len(rows) == args.n_eval_episodes:
        result = summarize_episodes(rows, job["bootstrap_resamples"], job["bootstrap_seed"])
        result["elapsed_seconds"] = 0.0
        return result
    env_factory = training.make_fixed_env if job["reference_evaluator"] else make_evaluation_env
    params = training.DomainParameters(job["sigma"], job["arrival"])
    policy = job["policy"]
    run_dir = Path(job["run_dir"])
    normalizer = None
    normalization_env = None
    model = None
    if policy != "periodic_rebalance":
        model = PPO.load(str(run_dir / f"{policy}.zip"), device="cpu")
        if args.normalise_obs:
            label = policy.removesuffix("_ppo")
            normalization_env = training.wrap_env(
                env_factory(args, params, job["regime_seed"]), normalise_obs=False,
            )
            normalizer = VecNormalize.load(str(run_dir / f"{label}_vecnormalize.pkl"), normalization_env)
            normalizer.training = False
            normalizer.norm_reward = False
    episode_count = args.n_eval_episodes
    args.n_eval_episodes = 1
    started = time.monotonic()
    try:
        for episode in range(job["episode_offset"] + len(rows), job["episode_offset"] + episode_count):
            seed = job["regime_seed"] + episode
            if policy == "periodic_rebalance":
                diagnostics = ArrivalDiagnostics()
                row = training.evaluate_periodic_rebalance(
                    params, job["evaluation_set"], args, seed,
                    step_observer=diagnostics, env_factory=env_factory,
                )
                row.update(diagnostics.summarize(args.num_trajectories))
            else:
                row = training.evaluate_ppo(policy, model, normalizer, params, job["evaluation_set"], args, seed, env_factory=env_factory)
            row.pop("training_seed")
            row.update(
                source_training_seed=args.seed if policy != "periodic_rebalance" else "",
                source_run_dir=str(run_dir), n_steps=args.n_steps,
                num_trajectories=args.num_trajectories, decision_stride=args.decision_stride,
                periodic_rebalance_every=args.periodic_rebalance_every,
                regime_seed=job["regime_seed"], episode_seed=seed,
                episode_offset=job["episode_offset"], episode_index=episode,
            )
            rows.append(row)
            write_csv(Path(job["episode_path"]), rows)
    finally:
        if normalization_env is not None:
            normalization_env.close()
    summary = summarize_episodes(rows, job["bootstrap_resamples"], job["bootstrap_seed"])
    summary["elapsed_seconds"] = time.monotonic() - started
    return summary


def build_jobs(cli):
    if cli.workers < 1 or cli.bootstrap_resamples < 1 or cli.bootstrap_seed < 0:
        raise ValueError("workers/resamples must be positive and bootstrap seed non-negative")
    if not 0 <= cli.episode_offset < 1000:
        raise ValueError("episode-offset must be in [0, 1000)")
    for name in ("sigma_values", "arrival_rate_values"):
        values = getattr(cli, name)
        if not values or not np.all(np.isfinite(values)) or min(values) < 0 or len(set(values)) != len(values):
            raise ValueError(f"{name} requires unique, finite, non-negative values")
    configs = [(path.resolve(), load_config(path)) for path in cli.run_dirs]
    if len(set(path for path, _ in configs)) != len(configs):
        raise ValueError("run directories must be unique")
    if len(set(cli.policies)) != len(cli.policies):
        raise ValueError("policies must be unique")
    signatures = [dict(c, seed=None, output_dir=None) for _, c in configs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("source runs must have matching configurations except seed/output_dir")
    if len({c["seed"] for _, c in configs}) != len(configs):
        raise ValueError("source training seeds must be unique")
    jobs = []
    for run_index, (path, config) in enumerate(configs):
        for policy in cli.policies:
            if policy == "periodic_rebalance" and run_index > 0:
                continue  # The same fixed baseline is not another training replicate.
            if policy != "periodic_rebalance":
                required = [path / f"{policy}.zip"]
                if config["normalise_obs"]:
                    required.append(path / f"{policy.removesuffix('_ppo')}_vecnormalize.pkl")
                for artifact in required:
                    if not artifact.is_file():
                        raise FileNotFoundError(artifact)
            resolutions = cli.n_steps_values or [config["n_steps"]]
            if len(set(resolutions)) != len(resolutions):
                raise ValueError("step resolutions must be unique")
            for n_steps in resolutions:
                args = resolution_args(config, n_steps, cli)
                if cli.episode_offset + args.n_eval_episodes > 1000:
                    raise ValueError("episodes must remain within the regime's 1000-seed block")
                seeds = set()
                for sigma in cli.sigma_values:
                    for arrival in cli.arrival_rate_values:
                        seed = regime_seed(config, sigma, arrival, args.evaluation_seed)
                        if seed in seeds:
                            raise ValueError("regime seed collision; choose a different grid")
                        seeds.add(seed)
                        jobs.append(dict(
                            args=vars(args).copy(), run_dir=str(path), policy=policy,
                            sigma=sigma, arrival=arrival, evaluation_set=evaluation_set(config, sigma, arrival),
                            regime_seed=seed, episode_offset=cli.episode_offset,
                            bootstrap_resamples=cli.bootstrap_resamples, bootstrap_seed=cli.bootstrap_seed,
                            reference_evaluator=cli.reference_evaluator,
                            episode_path=str(cli.output_dir / "episodes" / f"job_{len(jobs):04d}.csv"),
                        ))
    return configs, jobs


def plot_results(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = sorted({row["sigma"] for row in rows})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4), squeeze=False, sharey=True)
    for ax, sigma in zip(axes[0], sigmas):
        groups = sorted({(r["policy"], r["source_training_seed"], r["n_steps"]) for r in rows if r["sigma"] == sigma})
        for policy, seed, steps in groups:
            data = sorted([r for r in rows if (r["sigma"], r["policy"], r["source_training_seed"], r["n_steps"]) == (sigma, policy, seed, steps)], key=lambda r: r["arrival_rate"])
            label = f"{policy.replace('_', ' ')}; {steps:,} steps"
            if seed != "":
                label += f"; seed {seed}"
            x = [r["arrival_rate"] for r in data]
            ax.plot(x, [r[REWARD] for r in data], "o-", label=label)
            ax.fill_between(x, [r[f"{REWARD}_ci_lower"] for r in data], [r[f"{REWARD}_ci_upper"] for r in data], alpha=.15)
        ax.axhline(0, color="black", linewidth=.8, linestyle="--")
        ax.set(title=f"σ = {sigma:g}", xlabel="Baseline arrival rate α₁")
        ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Mean cumulative reward per episode")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle("Saved-policy evaluation; shaded marginal 95% episode-bootstrap intervals")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"reward_by_arrival.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    cli = parse_args(argv)
    cli.output_dir = cli.output_dir.resolve()
    configs, jobs = build_jobs(cli)
    if cli.resume and not cli.output_dir.is_dir():
        raise FileNotFoundError(f"cannot resume missing directory {cli.output_dir}")
    cli.output_dir.mkdir(parents=True, exist_ok=cli.resume)
    (cli.output_dir / "episodes").mkdir(exist_ok=cli.resume)
    with (cli.output_dir / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"a sweep is already running in {cli.output_dir}") from exc
        return run_jobs(cli, configs, jobs, argv)


def run_jobs(cli, configs, jobs, argv):
    source_files = [Path(__file__), ROOT / "experiments/train_robust_lp_agent.py",
                    ROOT / "experiments/fast_lp_evaluation.py"]
    source_files += sorted((ROOT / "SAiFE_gym").rglob("*.py"))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    manifest = dict(
        status="running", command=sys.argv if argv is None else argv,
        settings={k: str(v) if isinstance(v, Path) else [str(x) if isinstance(x, Path) else x for x in v] if isinstance(v, list) else v for k, v in vars(cli).items()},
        source_configs={str(path): config for path, config in configs},
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        git_revision=revision.stdout.strip(), python=platform.python_version(), numpy=np.__version__,
        confidence_intervals="Marginal 95% percentile bootstrap of independent vectorized episode means, not training seeds; within-episode trajectories may share swap ordering.",
        resolution_note="Physical action intervals are preserved. Matching seed numbers across resolutions do not couple identical Brownian paths.",
        jobs=jobs,
    )
    manifest_path = cli.output_dir / "manifest.json"
    if cli.resume:
        previous = json.loads(manifest_path.read_text())
        if previous["jobs"] != jobs:
            raise ValueError("resume requires the same policies, grid, seeds, and evaluation settings")
        runner = str(Path(__file__).relative_to(ROOT))
        for name, digest in previous["source_sha256"].items():
            if name != runner and manifest["source_sha256"].get(name) != digest:
                raise ValueError(f"cannot resume after changing simulation source: {name}")
        previous.setdefault("resume_history", []).append({
            "command": manifest["command"], "previous_status": previous["status"],
            "runner_sha256": manifest["source_sha256"][runner],
            "time_unix": time.time(),
        })
        previous["status"] = "running"
        previous.pop("error", None)
        manifest = previous
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summaries = []
    print(f"Evaluating {len(jobs)} policy/regime/resolution combinations with {cli.workers} workers", flush=True)
    def record(row):
        summaries.append(row)
        summaries.sort(key=lambda r: (r["policy"], str(r["source_training_seed"]), r["n_steps"], r["sigma"], r["arrival_rate"]))
        write_csv(cli.output_dir / "evaluation_summary.csv", summaries)
        print(f"[{len(summaries)}/{len(jobs)}] {row['policy']} σ={row['sigma']:g} α₁={row['arrival_rate']:g} steps={row['n_steps']} reward={row[REWARD]:+.4f} CI=[{row[REWARD + '_ci_lower']:+.4f}, {row[REWARD + '_ci_upper']:+.4f}]", flush=True)
    try:
        if cli.workers == 1:
            for job in jobs:
                record(evaluate_job(job))
        else:
            with ProcessPoolExecutor(max_workers=cli.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
                futures = [executor.submit(evaluate_job, job) for job in jobs]
                for future in as_completed(futures):
                    record(future.result())
        if not cli.no_plots:
            plot_results(summaries, cli.output_dir)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc), completed_jobs=len(summaries))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    manifest.update(status="complete", completed_jobs=len(summaries))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved evaluation artifacts to {cli.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
