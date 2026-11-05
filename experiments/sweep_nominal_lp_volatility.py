"""Train separate nominal PPO policies and compare deployed width by volatility.

`train --task-index N` runs one volatility/seed pair (for a Slurm array).
`train --dry-run` only prints the job plan. `summarize` requires a complete grid.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import train_robust_lp_agent as training  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = "experiments/results/nominal_lp_volatility"
DEFAULT_HALF_WIDTHS = [2, *range(5, 51, 5)]
CONFIG_OPTIONS = {
    "total_timesteps": (int, 100_000_000),
    "num_trajectories": (int, 100),
    "n_steps": (int, 10_000),
    "terminal_time": (float, 1.0),
    "decision_stride": (int, 100),
    "tau": (int, 50),
    "initial_wealth": (float, 1_000.0),
    "background_liquidity": (float, 30_000.0),
    "trade_size_notional": (float, 250.0),
    "price_impact_depth_window": (int, 15),
    "nominal_arrival_rate": (float, 300.0),
    "nominal_gas_cost": (float, 2.0),
    "alpha3": (float, 4000.0),
    "learning_rate": (float, 3e-4),
    "n_eval_episodes": (int, 10),
    "convergence_eval_every_rollouts": (int, 5),
    "convergence_n_eval_episodes": (int, 1),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train nominal policies, one per volatility/seed.")
    train.add_argument("--sigma-values", nargs="+", type=float,
                       default=[0.03, 0.035, 0.04, 0.045, 0.05])
    train.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    train.add_argument("--half-widths", nargs="+", type=int, default=DEFAULT_HALF_WIDTHS.copy(),
                       help="Increasing half-width choices in ticks; every action recenters and rebalances.")
    train.add_argument("--task-index", type=int, default=None,
                       help="Zero-based job index: volatility first, then training seed.")
    train.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    train.add_argument("--validation-seed", type=int, default=100_042)
    train.add_argument("--evaluation-seed", type=int, default=200_042)
    train.add_argument("--dry-run", action="store_true", help="Print the plan without training or writing results.")
    for name, (kind, default) in CONFIG_OPTIONS.items():
        train.add_argument("--" + name.replace("_", "-"), type=kind, default=default)
    summarize = commands.add_parser("summarize", help="Check the completed grid and save tables and plots.")
    summarize.add_argument("--input-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def training_args(cli, sigma, seed):
    args = training.parse_args([])
    for name in CONFIG_OPTIONS:
        setattr(args, name, getattr(cli, name))
    args.seed = seed
    args.nominal_sigma = sigma
    args.inventory_phi = 0.0
    args.action_mode = "symmetric_width"
    args.half_widths = list(cli.half_widths)
    args.evaluation_seed = cli.validation_seed
    args.periodic_width = min(args.periodic_width, args.tau)
    training.validate_and_derive_decision_timing(args)
    training.validate_evaluation_configuration(args)
    return args


def job_grid(cli):
    sigmas, seeds = cli.sigma_values, cli.seeds
    if len(set(sigmas)) != len(sigmas) or not all(np.isfinite(s) and s >= 0 for s in sigmas):
        raise ValueError("sigma-values must be distinct, finite, and non-negative")
    if len(set(seeds)) != len(seeds) or not all(0 <= s < 2**32 - 10 for s in seeds):
        raise ValueError("seeds must be distinct integers in the supported RNG range")
    for name in ("num_trajectories", "tau", "n_eval_episodes"):
        if getattr(cli, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("terminal_time", "initial_wealth", "learning_rate"):
        if not np.isfinite(getattr(cli, name)) or getattr(cli, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("nominal_arrival_rate", "nominal_gas_cost", "alpha3", "trade_size_notional"):
        if not np.isfinite(getattr(cli, name)) or getattr(cli, name) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if cli.price_impact_depth_window < 1:
        raise ValueError("price_impact_depth_window must be positive")
    # The final evaluation stream must not overlap the convergence-validation
    # episode seeds or any training seed. Market components use offsets 0..4.
    validation = range(cli.validation_seed, cli.validation_seed + cli.convergence_n_eval_episodes + 4)
    evaluation = range(cli.evaluation_seed, cli.evaluation_seed + cli.n_eval_episodes + 4)
    if min(validation.start, evaluation.start) < 0 or max(validation.stop, evaluation.stop) > 2**32:
        raise ValueError("validation/evaluation seeds are outside the supported RNG range")
    if max(validation.start, evaluation.start) < min(validation.stop, evaluation.stop):
        raise ValueError("validation and final evaluation seed ranges must not overlap")
    if any(any(s + offset in stream for offset in range(5))
           for s in seeds for stream in (validation, evaluation)):
        raise ValueError("training seeds must not overlap validation or evaluation seeds")
    jobs = [(sigma, seed) for sigma in sigmas for seed in seeds]
    if cli.task_index is not None and not 0 <= cli.task_index < len(jobs):
        raise ValueError(f"task-index must be between 0 and {len(jobs) - 1}")
    args = training_args(cli, *jobs[0])
    if args.ppo_n_steps * args.num_trajectories < 2:
        raise ValueError("PPO requires at least two rollout samples")
    return jobs


def source_fingerprint():
    files = sorted((ROOT / "SAiFE_gym").rglob("*.py"))
    files += [ROOT / "experiments" / name for name in (
        "sweep_nominal_lp_volatility.py", "train_robust_lp_agent.py", "helpers.py",
        "policy_behavior_diagnostics.py", "evaluation_grid.py",
    )]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    digest.update((ROOT / "requirements.txt").read_bytes())
    return digest.hexdigest()


def run_config(cli, sigma, seed):
    return {
        "schema_version": SCHEMA_VERSION,
        "sigma_values": cli.sigma_values,
        "seeds": cli.seeds,
        "sigma": sigma,
        "seed": seed,
        "settings": {name: getattr(cli, name) for name in CONFIG_OPTIONS},
        "validation_seed": cli.validation_seed,
        "evaluation_seed": cli.evaluation_seed,
        "inventory_phi": 0.0,
        "action_mode": "symmetric_width",
        "half_widths": cli.half_widths,
        "policy_action_space": "half-width only; recenter and rebalance every decision",
        "code_sha256": source_fingerprint(),
    }


def cell_directory(root, sigma, seed):
    return root / f"sigma_{sigma}" / f"seed_{seed}"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_cell(cli, sigma, seed):
    args = training_args(cli, sigma, seed)
    config = run_config(cli, sigma, seed)
    directory = cell_directory(cli.output_dir, sigma, seed)
    # Atomic creation prevents Slurm retries or a second sweep overwriting a run.
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    params = training.DomainParameters(sigma, args.nominal_arrival_rate)
    env = training.make_fixed_env(args, params, seed)
    model = None
    try:
        model, normalization = training.train_and_save("nominal", env, args, directory)
        eval_args = argparse.Namespace(**vars(args))
        eval_args.evaluation_seed = cli.evaluation_seed
        metrics = training.evaluate_ppo(
            "nominal_ppo", model, normalization, params, "in_distribution",
            eval_args, regime_seed=cli.evaluation_seed,
        )
        if not np.isclose(metrics["mean_inventory_penalty_per_path"], 0.0, atol=1e-7):
            raise ValueError("Expected pure P&L evaluation without an inventory penalty")
        # No deployed position is an undefined width, not a zero-width position.
        width = metrics["mean_active_range_width_ticks"]
        metrics["mean_deployed_width_ticks"] = width if width > 0 else None
        metrics["equivalent_half_width_ticks"] = width / 2 if width > 0 else None
        result = {"config": config, "metrics": metrics}
        # Only a completely written result marks a finished cell.
        pending = directory / "result.pending.json"
        write_json(pending, result)
        pending.replace(directory / "result.json")
        return result
    finally:
        if model is not None:
            model.get_env().close()
        else:
            env.close()


def load_results(root):
    configs = sorted(root.glob("sigma_*/seed_*/config.json"))
    if not configs:
        raise ValueError(f"No sweep configurations found in {root}")
    results, reference = {}, None
    for path in configs:
        config = json.loads(path.read_text())
        if config["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Unsupported sweep schema in {path}")
        signature = {k: v for k, v in config.items() if k not in ("sigma", "seed")}
        if reference is None:
            reference = signature
        elif reference != signature:
            raise ValueError("Cannot combine runs with different settings, grids, or source versions")
        key = (config["sigma"], config["seed"])
        result_path = path.with_name("result.json")
        if not result_path.exists():
            raise ValueError(f"Unfinished run: {path.parent}")
        result = json.loads(result_path.read_text())
        if result["config"] != config or key in results:
            raise ValueError(f"Mismatched or duplicate result: {result_path}")
        metrics = result["metrics"]
        if (metrics["sigma"], metrics["training_seed"], metrics["evaluation_seed"]) != (
            *key, config["evaluation_seed"],
        ):
            raise ValueError(f"Evaluation metadata disagrees with config: {result_path}")
        for value in metrics.values():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"Non-finite metric in {result_path}")
        results[key] = result
    expected = {(sigma, seed) for sigma in reference["sigma_values"] for seed in reference["seeds"]}
    if results.keys() != expected:
        raise ValueError(f"Incomplete sweep; missing cells: {sorted(expected - results.keys())}")
    return reference, results


def mean_and_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None, len(values)


def summarize_results(root):
    reference, results = load_results(root)
    rows = []
    for sigma in sorted(reference["sigma_values"]):
        metrics = [results[(sigma, seed)]["metrics"] for seed in reference["seeds"]]
        width, width_std, width_count = mean_and_std([m["mean_deployed_width_ticks"] for m in metrics])
        pnl, pnl_std, _ = mean_and_std([m["mean_pnl_per_path"] for m in metrics])
        rows.append({
            "sigma": sigma, "n_training_seeds": len(metrics), "n_width_seeds": width_count,
            "mean_deployed_width_ticks": width, "seed_std_width_ticks": width_std,
            "equivalent_half_width_ticks": width / 2 if width is not None else None,
            "mean_pnl": pnl, "seed_std_pnl": pnl_std,
            "mean_never_deployed_fraction": float(np.mean([m["never_deployed_fraction"] for m in metrics])),
            "mean_agent_hold_decision_fraction": float(np.mean([m["agent_hold_decision_fraction"] for m in metrics])),
            "mean_rebalances_per_path": float(np.mean([m["mean_agent_rebalances_after_deployment_per_path"] for m in metrics])),
        })
    output = root / "summary"
    output.mkdir(exist_ok=True)
    training.save_csv(output / "width_by_volatility.csv", rows)
    raw_rows = [{"seed": seed, **result["metrics"]} for (sigma, seed), result in sorted(results.items())]
    training.save_csv(output / "per_seed.csv", raw_rows)
    write_json(output / "summary.json", {
        "configuration": reference, "results": rows,
        "width_definition": "Mean of per-policy time-weighted full widths (upper minus lower) over deployed trajectory-steps, including out-of-range positions; non-deployment is excluded.",
        "uncertainty": "Sample standard deviation across training seeds on shared evaluation paths; not a confidence interval or proof of optimality.",
    })
    plot_summary(output, rows, results, reference)
    return rows


def plot_summary(output, rows, results, reference):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.array([100 * row["sigma"] for row in rows])
    for ax, mean_key, std_key, raw_key, ylabel in (
        (axes[0], "mean_deployed_width_ticks", "seed_std_width_ticks", "mean_deployed_width_ticks", "Deployed full width (ticks)"),
        (axes[1], "mean_pnl", "seed_std_pnl", "mean_pnl_per_path", "P&L (token1)"),
    ):
        mean = np.array([r[mean_key] if r[mean_key] is not None else np.nan for r in rows])
        std = np.array([r[std_key] if r[std_key] is not None else np.nan for r in rows])
        ax.plot(x, mean, "o-", label="Mean across training seeds")
        ax.fill_between(x, mean - std, mean + std, alpha=.2, label="±1 training-seed SD")
        for seed in reference["seeds"]:
            values = [results[(r["sigma"], seed)]["metrics"][raw_key] for r in rows]
            ax.scatter(x, [v if v is not None else np.nan for v in values], s=18, alpha=.55, color="black")
        ax.set(xlabel="Fixed episode volatility (%)", ylabel=ylabel, xticks=x)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="gray", linewidth=.8)
    fig.suptitle("Nominal PPO: learned position width and pure P&L")
    fig.savefig(output / "width_by_volatility.png", dpi=180)
    fig.savefig(output / "width_by_volatility.pdf")
    plt.close(fig)


def main(argv=None):
    cli = parse_args(argv)
    if cli.command == "summarize":
        summarize_results(cli.input_dir)
        print(f"Saved tables and plots to {cli.input_dir / 'summary'}")
        return 0
    jobs = job_grid(cli)
    selected = list(enumerate(jobs)) if cli.task_index is None else [(cli.task_index, jobs[cli.task_index])]
    if cli.dry_run:
        print(json.dumps({
            "jobs": [{"task_index": i, "sigma": sigma, "seed": seed} for i, (sigma, seed) in selected],
            "settings": {name: getattr(cli, name) for name in CONFIG_OPTIONS},
            "ppo_decisions_per_episode": cli.n_steps // cli.decision_stride,
            "requested_ppo_decisions_per_policy": cli.total_timesteps // cli.decision_stride,
            "rollouts_per_policy": int(np.ceil(cli.total_timesteps / (cli.n_steps * cli.num_trajectories))),
            "inventory_penalty": 0.0,
            "action_mode": "symmetric_width",
            "half_widths": cli.half_widths,
            "n_actions": len(cli.half_widths),
        }, indent=2))
        return 0
    for index, (sigma, seed) in selected:
        print(f"Training task {index}: fixed sigma={sigma}, seed={seed}", flush=True)
        run_cell(cli, sigma, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
