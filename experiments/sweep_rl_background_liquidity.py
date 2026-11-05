"""Train nominal PPO at each background liquidity and compare position widths.

Uses train_robust_lp_agent's fixed environment, PPO builder and evaluator.
All its training options are accepted, except --background-liquidity (use
--background-values). Defaults, including the inventory penalty and structured
action space, are preserved. Domain randomization is intentionally disabled.

Examples (from the repository root)::

    python experiments/sweep_rl_background_liquidity.py --dry-run
    python experiments/sweep_rl_background_liquidity.py --seeds 42 43 44
    python experiments/sweep_rl_background_liquidity.py --smoke-test

Widths are full widths (upper tick minus lower tick), not half-widths.
Each policy uses fresh normalization statistics. Seeds and evaluation paths
are paired across liquidity levels; endogenous order flow and pool prices can
change. Training-seed standard deviations are not confidence intervals.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import train_robust_lp_agent as training  # noqa: E402


DEFAULT_BACKGROUNDS = [10_000., 15_000., 20_000., 25_000., 30_000.]
WIDTH_METRICS = {
    "deployed_width_ticks": "mean_active_range_width_ticks",
    "selected_width_ticks": "mean_agent_selected_range_width_ticks",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--background-values", nargs="+", type=float,
                        default=DEFAULT_BACKGROUNDS.copy())
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Paired training seeds; defaults to the training script's --seed.")
    parser.add_argument("--output-dir", default="experiments/results/rl_background_liquidity")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--training-help", action="store_true",
                        help="Show the inherited training options.")
    cli, remaining = parser.parse_known_args(argv)
    if cli.training_help:
        training.parse_args(["--help"])
    if any(arg.split("=", 1)[0] == "--background-liquidity" for arg in remaining):
        parser.error("Use --background-values to specify the sweep grid")
    args = training.parse_args(remaining)
    args.output_dir = cli.output_dir
    training.apply_smoke_overrides(args)
    training.validate_and_derive_decision_timing(args)
    training.validate_evaluation_configuration(args)
    cli.seeds = cli.seeds if cli.seeds is not None else [args.seed]
    backgrounds = cli.background_values
    if (len(backgrounds) < 2 or len(set(backgrounds)) != len(backgrounds)
            or not all(np.isfinite(x) and x >= 0 for x in backgrounds)):
        parser.error("Use at least two distinct, finite, non-negative background values")
    cli.background_values = sorted(backgrounds)
    if (len(set(cli.seeds)) != len(cli.seeds)
            or not all(0 <= seed < 2**32 - 5 for seed in cli.seeds)):
        parser.error("Training seeds must be distinct integers between 0 and 2**32 - 6")
    if args.num_trajectories < 1 or args.ppo_n_steps * args.num_trajectories < 2:
        parser.error("PPO needs positive num-trajectories and at least two rollout samples")
    # Evaluation uses seed + episode index; market components use offsets 0..4.
    eval_stop = args.evaluation_seed + max(
        args.n_eval_episodes, args.convergence_n_eval_episodes,
    ) + 4
    if eval_stop > 2**32:
        parser.error("Evaluation seeds exceed the supported RNG range")
    if any(seed < eval_stop and seed + 5 > args.evaluation_seed for seed in cli.seeds):
        parser.error("Training and evaluation seed streams must not overlap")
    return cli, args


def cell_args(base_args, background, seed):
    """Copy the common configuration; only liquidity and paired seed vary."""
    return argparse.Namespace(**{
        **vars(base_args), "background_liquidity": background, "seed": seed,
    })


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_cell(args, directory):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", vars(args))
    params = training.DomainParameters(args.nominal_sigma, args.nominal_arrival_rate)
    env = training.make_fixed_env(args, params, seed=args.seed)
    model = None
    try:
        model, normalization = training.train_and_save("nominal", env, args, directory)
        metrics = training.evaluate_ppo(
            "nominal_ppo", model, normalization, params, "in_distribution", args,
            regime_seed=args.evaluation_seed,
        )
        row = {"background_liquidity": args.background_liquidity, **metrics}
        for name, metric in WIDTH_METRICS.items():
            # The shared evaluator returns zero when no position/action exists.
            # Missing widths should not lower the mean or look like narrow LPs.
            row[name] = metrics[metric] if metrics[metric] > 0 else None
        write_json(directory / "metrics.json", row)
        return row
    finally:
        if model is not None:
            model.get_env().close()
        else:
            env.close()


def summarize(rows):
    summaries = []
    for background in sorted({row["background_liquidity"] for row in rows}):
        cells = [row for row in rows if row["background_liquidity"] == background]
        summary = {"background_liquidity": background, "n_training_seeds": len(cells)}
        for name in WIDTH_METRICS:
            widths = [row[name] for row in cells if row[name] is not None]
            summary[f"mean_{name}"] = float(np.mean(widths)) if widths else None
            summary[f"seed_std_{name}"] = float(np.std(widths, ddof=1)) if len(widths) > 1 else None
            summary[f"n_seeds_with_{name}"] = len(widths)
        for metric in ("never_deployed_fraction", "agent_hold_decision_fraction",
                       "mean_agent_rebalances_after_deployment_per_path",
                       "mean_pnl_per_path", "mean_running_inventory_objective"):
            summary[metric] = float(np.mean([row[metric] for row in cells]))
        summaries.append(summary)
    return summaries


def plot_summary(directory, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    backgrounds = [row["background_liquidity"] for row in summaries]
    for ax, name, title in zip(axes, WIDTH_METRICS, (
        "Time-weighted deployed position width", "Width selected on rebalance decisions",
    )):
        means = np.array([row[f"mean_{name}"] if row[f"mean_{name}"] is not None
                          else np.nan for row in summaries])
        stds = np.array([row[f"seed_std_{name}"] if row[f"seed_std_{name}"] is not None
                         else np.nan for row in summaries])
        ax.plot(backgrounds, means, "o-", label="Mean across training seeds")
        if np.any(np.isfinite(stds)):
            ax.fill_between(backgrounds, means - stds, means + stds, alpha=.2,
                            label="±1 training-seed SD")
        ax.set(title=title, xlabel="Background liquidity per tick interval",
               ylabel="Full width (ticks)", xticks=backgrounds)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(directory / "width_by_background_liquidity.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    cli, base_args = parse_args(argv)
    jobs = [(background, seed) for background in cli.background_values for seed in cli.seeds]
    configuration = {
        "background_values": cli.background_values, "seeds": cli.seeds,
        "training_settings": vars(base_args),
        "training_mode": "fresh nominal PPO per liquidity/seed; no domain randomization",
        "evaluation": "Deterministic final policies on paired evaluation seeds at each training liquidity.",
        "deployed_width": "Full width averaged over deployed trajectory-steps, including out-of-range positions.",
        "selected_width": "Full width averaged over rebalance decisions; hold decisions excluded.",
        "missing_width": "null/blank when no deployed positions or rebalance decisions exist",
        "uncertainty": "Sample standard deviation across training seeds; null for fewer than two valid seeds.",
    }
    if cli.dry_run:
        print(json.dumps({**configuration, "jobs": [
            {"background_liquidity": background, "seed": seed} for background, seed in jobs
        ]}, indent=2))
        return 0

    directory = training.make_run_dir(cli.output_dir, base_args.smoke_test)
    write_json(directory / "sweep_config.json", configuration)
    print(f"Saving {len(jobs)} policies and sweep results to {directory.resolve()}", flush=True)
    rows = []
    for background, seed in jobs:
        print(f"Training background_liquidity={background:g}, seed={seed}", flush=True)
        args = cell_args(base_args, background, seed)
        cell_directory = directory / f"background_{background}" / f"seed_{seed}"
        rows.append(run_cell(args, cell_directory))
        # Preserve completed cells if a subsequent training run fails.
        training.save_csv(directory / "per_seed.csv", rows)
    summaries = summarize(rows)
    training.save_csv(directory / "width_by_background_liquidity.csv", summaries)
    write_json(directory / "summary.json", {
        "configuration": configuration, "results": summaries,
    })
    plot_summary(directory, summaries)
    print(f"Saved width tables, plot, and trained policies to {directory.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
