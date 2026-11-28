"""Simulate fixed-width periodic LPs and plot their payoffs across one driver."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
from itertools import groupby, product, repeat
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SNAPSHOTS = {
    "volatility": HERE / "volatility/fixed-width-bg30000",
    "background-liquidity": HERE / "background-liquidity/fixed-width",
    "risk-aversion": HERE / "risk-aversion/cara",
}
PARAMETERS = {
    "volatility": "sigma", "background-liquidity": "background_liquidity",
    "risk-aversion": "risk_aversion",
}
EXTRA_LIQUIDITY_WIDTHS = (1, 3, 55, 60, 70, 85, 100)
LIQUIDITY_LEVELS = (50_000, 40_000, 30_000, 20_000, 10_000)
LIQUIDITY_WIDTHS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 85, 100, 125, 150)
PATH_FIELDS = ("sigma", "background_liquidity", "half_width_ticks", "episode_seed", "path", "pnl")
# The environment uses the episode seed and offsets through +4 for its component RNGs.
BATCH_SEED_STRIDE = 10


def sweep_defaults(driver):
    """Use the market, cadence and sample sizes of the recorded periodic runs."""
    directory = SNAPSHOTS[driver]
    if driver == "risk-aversion":
        source = json.loads((directory / "config.json").read_text())
        config = source["trainer_configuration"]
        experiment = source["experiment"]
        return config, experiment["risk_aversion_values"], experiment["half_widths"], config["n_eval_episodes"]
    source = json.loads((directory / "summary.json").read_text())
    config = next(iter(source["configurations"].values()))
    if driver == "volatility":
        return config, source["sigma_values"], source["half_widths"], 1
    return source["configurations"][str(LIQUIDITY_LEVELS[0])], LIQUIDITY_LEVELS, LIQUIDITY_WIDTHS, 1


def grid_row(driver, level, width, n_paths, mean_pnl, ce="", bootstrap_fraction=""):
    return {
        "driver": driver, "parameter": PARAMETERS[driver], "parameter_value": float(level),
        "half_width_ticks": int(width), "full_width_ticks": 2 * int(width),
        "n_paths": int(n_paths), "mean_pnl": float(mean_pnl),
        "certainty_equivalent_pnl": ce, "bootstrap_best_fraction": bootstrap_fraction,
        "peak": "",
    }


def curves(rows):
    """Order liquidity from deep to thin, other drivers from low to high."""
    direction = -1 if rows[0]["driver"] == "background-liquidity" else 1
    ordered = sorted(rows, key=lambda r: (direction * r["parameter_value"], r["half_width_ticks"]))
    for level, group in groupby(ordered, key=lambda r: r["parameter_value"]):
        yield level, list(group)


def objective(row):
    return row["certainty_equivalent_pnl"] if row["driver"] == "risk-aversion" else row["mean_pnl"]


def ranked_grid(rows):
    ordered = []
    for _, curve in curves(rows):
        # The first maximum on the ordered grid breaks exact ties toward narrower ranges.
        best = max(curve, key=objective)
        for row in curve:
            row["peak"] = ""
        if best is curve[0]:
            best["peak"] = "lower_edge"
        elif best is curve[-1]:
            best["peak"] = "upper_edge"
        else:
            best["peak"] = "interior"
        ordered.extend(curve)
    return ordered


def write_grid(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_widths(grids, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    titles = {"volatility": "Midprice volatility", "background-liquidity": "Background liquidity",
              "risk-aversion": "CARA risk aversion"}
    labels = {"volatility": "Midprice volatility (σ)", "background-liquidity": "Liquidity per tick interval",
              "risk-aversion": "Risk aversion (1/token1)"}
    if len(grids) == 1:
        fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.1), width_ratios=(1.65, 1), layout="constrained")
        axis_pairs = [(left, right)]
    else:
        fig, axes = plt.subplots(2, len(grids), figsize=(15, 8.2), height_ratios=(1.65, 1), layout="constrained")
        axis_pairs = axes.T

    for (driver, rows), (payoff_ax, width_ax) in zip(grids.items(), axis_pairs):
        groups = list(curves(rows))
        levels, widths = [], []
        for index, (level, curve) in enumerate(groups):
            color = plt.get_cmap("viridis")(0.08 + 0.77 * index / max(1, len(groups) - 1))
            label = f"{level:,.0f}" if driver == "background-liquidity" and level.is_integer() else f"{level:g}"
            risk_neutral = driver == "risk-aversion" and level == 0
            if risk_neutral:
                label = "0 (mean PnL)"
                color = "#343a43"
            payoff_ax.plot([r["half_width_ticks"] for r in curve], [objective(r) for r in curve],
                           marker=".", linestyle="--" if risk_neutral else "-", color=color,
                           markersize=4, linewidth=1.6, label=label)
            best = next(r for r in curve if r["peak"])
            width = best["half_width_ticks"]
            levels.append(level)
            widths.append(width)
            marker = "D" if best["peak"] == "interior" else "s"
            for ax, x, y in ((payoff_ax, width, objective(best)), (width_ax, level, width)):
                ax.plot(x, y, marker=marker, color=color, markeredgecolor="#202a32",
                        markersize=7, markeredgewidth=0.9, zorder=3)
        payoff_ax.set(title=titles[driver], xlabel="Position half-width (ticks)",
                      ylabel=("Certainty-equivalent PnL (token1)" if driver == "risk-aversion"
                              else "Mean net PnL (token1)"))
        payoff_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False,
                         ncols=min(3, len(groups)), fontsize=9, columnspacing=1.1, handlelength=1.6)
        width_ax.plot(levels, widths, color="#65727b", linewidth=1.2, zorder=1)
        width_ax.set(xlabel=labels[driver], ylabel="Best sampled half-width (ticks)")
        if len(set(widths)) <= 8:
            width_ax.set_yticks(sorted(set(widths)))
        else:
            width_ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        if driver == "background-liquidity":
            width_ax.invert_xaxis()
        if len(groups) <= 5:
            width_ax.set_xticks(levels, [f"{v:g}" if driver != "background-liquidity"
                                         else f"{v / 1000:g}k" for v in levels])
        for ax in (payoff_ax, width_ax):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.19)
            ax.margins(x=0.045)
    fig.suptitle("Periodic LP payoff and best sampled position width", fontsize=15)
    fig.savefig(filename, dpi=180)
    if len(grids) > 1:
        fig.savefig(filename.with_suffix(".pdf"), metadata={"CreationDate": None})
    plt.close(fig)


def cara_scores(pnl, rewards, widths, risks):
    from experiments.sweep_dense_exponential_width import summarize

    episodes, paths = pnl.shape[1:]
    if episodes * paths == 1:
        # A single observed payoff has the same empirical CE at every risk.
        return [grid_row("risk-aversion", a, width, 1, value, value)
                for width, value in zip(widths, pnl[:, 0, 0]) for a in risks]
    scores, _, _ = summarize(pnl, rewards, widths, risks)
    return [grid_row("risk-aversion", r["risk_aversion"], r["half_width_ticks"], r["n_paths"],
                     r["mean_pnl"], r["certainty_equivalent_pnl"],
                     r["bootstrap_best_fraction"] if episodes > 1 else "") for r in scores]


def simulate(driver, config, levels, widths, workers, directory):
    import numpy as np
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from experiments.sweep_dense_exponential_width import run_episode

    # A preference changes the score of an existing payoff distribution.
    # Only the two market drivers require a fresh rollout per level.
    is_risk = driver == "risk-aversion"
    markets = [None] if is_risk else levels
    risks = levels if is_risk else [0.0]
    episodes = config["n_eval_episodes"]
    jobs = list(product(widths, range(episodes)))
    job_widths, episode_ids = zip(*jobs)
    seed_stride = config.get("evaluation_seed_stride", BATCH_SEED_STRIDE)
    # run_episode adds its episode argument to the evaluation seed.
    seed_offsets = [episode * seed_stride for episode in episode_ids]
    rows = []
    executor = ProcessPoolExecutor(workers) if workers > 1 else nullcontext()
    with executor as pool, (directory / "path_pnl.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(PATH_FIELDS)
        evaluate = pool.map if pool else map
        for level in markets:
            settings = argparse.Namespace(**config)
            if not is_risk:
                setattr(settings, "nominal_sigma" if driver == "volatility" else "background_liquidity", level)
            results = evaluate(run_episode, repeat(settings, len(jobs)), job_widths, seed_offsets,
                               repeat(risks, len(jobs)))
            pnl = np.empty((len(widths), episodes, settings.num_trajectories))
            if is_risk:
                rewards = np.empty((len(widths), len(risks), episodes, settings.num_trajectories))
            for (width, episode), result in zip(jobs, results):
                wi = widths.index(width)
                pnl[wi, episode] = result["pnl"]
                if is_risk:
                    rewards[wi, :, episode] = result["total_rewards"]
                writer.writerows((settings.nominal_sigma, settings.background_liquidity, width,
                                  result["seed"], path, value) for path, value in enumerate(result["pnl"]))
                if episode == episodes - 1:
                    print(f"σ={settings.nominal_sigma:g}, background={settings.background_liquidity:g}, "
                          f"±{width} ticks, mean net PnL {pnl[wi].mean():.2f}", flush=True)
            if is_risk:
                rows.extend(cara_scores(pnl, rewards, widths, risks))
            else:
                rows.extend(grid_row(driver, level, width, values.size, values.mean())
                            for width, values in zip(widths, pnl))
    return ranked_grid(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", choices=SNAPSHOTS)
    parser.add_argument("--levels", type=float, nargs="+", help="Values of the selected driver")
    parser.add_argument("--half-widths", type=int, nargs="+", help="Symmetric offsets in ticks")
    parser.add_argument("--episodes", type=int, help="Vectorized batches per market and width")
    parser.add_argument("--paths-per-episode", type=int)
    parser.add_argument("--seed", type=int, help="First episode seed, shared across widths and levels")
    parser.add_argument("--seed-stride", type=int, default=BATCH_SEED_STRIDE,
                        help="Gap between batch seeds (default 10). Use 1 to replay the historical CARA seeds.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config, levels, widths, episodes = sweep_defaults(args.driver)
    levels = sorted(set(args.levels or levels), reverse=args.driver == "background-liquidity")
    widths = sorted(set(args.half_widths or widths))
    if not all(math.isfinite(x) and x >= 0 for x in levels):
        parser.error("driver levels must be finite and nonnegative")
    if len(widths) < 2 or widths[0] < 1:
        parser.error("use at least two distinct positive half-widths")
    if args.seed_stride < 1:
        parser.error("seed-stride must be positive")
    config.update(tau=max(config["tau"], widths[-1]), evaluation_seed_stride=args.seed_stride,
                  n_eval_episodes=episodes if args.episodes is None else args.episodes)
    if args.paths_per_episode is not None:
        config["num_trajectories"] = args.paths_per_episode
    if args.seed is not None:
        config["evaluation_seed"] = args.seed
    if min(config["n_eval_episodes"], config["num_trajectories"], args.workers) < 1 or config["evaluation_seed"] < 0:
        parser.error("episodes, paths per episode and workers must be positive, and seed nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = args.output_dir or REPO / "experiments/results/periodic_widths" / args.driver / stamp
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as temporary:
        work = Path(temporary)
        rows = simulate(args.driver, config, levels, widths, args.workers, work)
        (work / "config.json").write_text(json.dumps({
            "driver": args.driver, "levels": levels, "half_widths": widths,
            "policy": "PeriodicRebalanceAgent", "reward_function": "DenseExponentialUtility",
            "objective": "certainty_equivalent_pnl" if args.driver == "risk-aversion" else "mean_net_pnl",
            "base_environment": config,
        }, indent=2) + "\n")
        write_grid(work / "width_grid.csv", rows)
        plot_widths({args.driver: rows}, work / "width_curves.png")
        for artifact in work.iterdir():
            artifact.replace(directory / artifact.name)
    print(f"Results in {directory.resolve()}")
    for level, curve in curves(rows):
        best = next(r for r in curve if r["peak"])
        print(f"{PARAMETERS[args.driver]}={level:g}  ±{best['half_width_ticks']} ticks  "
              f"{objective(best):.2f} token1  ({best['peak']})")


if __name__ == "__main__":
    main()
