"""Evaluate periodic LP widths under dense extended CARA utility.

Uses the nominal environment factory and defaults from train_robust_lp_agent.
Additional trainer arguments can be passed directly, e.g. --nominal-sigma 0.05.
No PPO training, reward normalization, or inventory penalty is applied.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/saife_matplotlib")
# Avoid one BLAS thread pool per simulation worker.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import train_robust_lp_agent as trainer
from experiments.policy_behavior_diagnostics import PolicyBehaviorAccumulator
from SAiFE_gym.agents.BaselineAgents import PeriodicRebalanceAgent
from SAiFE_gym.gym.index_names import INITIAL_WEALTH_KEY, PORTFOLIO_VALUE_KEY
from SAiFE_gym.rewards.RewardFunctions import DenseExponentialUtility


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-widths", type=int, nargs="+",
                        default=[1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--risk-aversion-values", type=float, nargs="+",
                        default=[0.0001, 0.001, 0.003, 0.01, 0.03, 0.1],
                        help="Positive: risk averse; zero: risk neutral; negative: risk seeking")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=98765)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/results/dense_exponential_width_sweep"))
    args, remaining = parser.parse_known_args(argv)
    environment = trainer.parse_args(remaining)
    if environment.smoke_test:
        # Keep market dynamics/horizon unchanged; reduce evaluation work only.
        environment.num_trajectories = 2
        environment.n_eval_episodes = 2
        args.half_widths = [1, environment.tau]
        args.workers = 1
        args.bootstrap_resamples = 100
    args.half_widths = sorted(set(args.half_widths))
    args.risk_aversion_values = sorted(set(args.risk_aversion_values))
    if len(args.half_widths) < 2 or not all(1 <= w <= environment.tau for w in args.half_widths):
        parser.error("provide at least two half-widths in [1, tau]")
    for risk in args.risk_aversion_values:
        DenseExponentialUtility(risk)
    if environment.n_eval_episodes < 2 or environment.num_trajectories < 1:
        parser.error("use at least two independent episodes and one trajectory")
    if args.workers < 1 or args.bootstrap_resamples < 100:
        parser.error("workers must be positive; bootstrap-resamples must be >= 100")
    if environment.n_steps < 1 or environment.periodic_rebalance_every < 1:
        parser.error("n-steps and periodic-rebalance-every must be positive")
    return args, environment


def replay_dense_rewards(wealth, initial_wealth, risk_aversions):
    """Apply the production reward to every recorded transition, vectorized.

    The baseline ignores rewards, so replaying them is identical to rerunning
    the simulator for each risk parameter, without resampling market paths.
    """
    current = {PORTFOLIO_VALUE_KEY: wealth[:-1], INITIAL_WEALTH_KEY: initial_wealth}
    following = {PORTFOLIO_VALUE_KEY: wealth[1:]}
    totals = []
    errors = []
    for risk in risk_aversions:
        reward_function = DenseExponentialUtility(risk)
        increments = reward_function.calculate(current, None, following)
        total = increments.sum(axis=0, dtype=np.float64)
        # Compare the whole path with one endpoint increment. Reuse the stable
        # calculation: individual utility levels can overflow even when their
        # difference is finite, and direct division loses accuracy near a=0.
        expected = reward_function.calculate(
            {PORTFOLIO_VALUE_KEY: wealth[0], INITIAL_WEALTH_KEY: initial_wealth},
            None, {PORTFOLIO_VALUE_KEY: wealth[-1]},
        )
        # Relative to the objective's scale; absolute near zero.
        error = np.max(np.abs(total - expected) / np.maximum(1.0, np.abs(expected)))
        if not np.all(np.isfinite(total)) or error > 1e-8:
            raise FloatingPointError(f"Dense reward telescoping failed at risk_aversion={risk}: {error}")
        totals.append(total)
        errors.append(error)
    return np.stack(totals), np.asarray(errors)


def run_episode(environment, width, episode, risk_aversions):
    start = perf_counter()
    seed = environment.evaluation_seed + episode
    params = trainer.DomainParameters(environment.nominal_sigma, environment.nominal_arrival_rate)
    env = trainer.make_fixed_env(
        environment, params, seed,
        reward_function=DenseExponentialUtility(risk_aversions[0]),
    )
    agent = PeriodicRebalanceAgent(env, environment.periodic_rebalance_every, width)
    obs, _ = env.reset(seed=seed)
    initial_wealth = obs[INITIAL_WEALTH_KEY].copy()
    wealth = np.empty((env.n_steps + 1, env.num_trajectories), dtype=np.float64)
    wealth[0] = obs[PORTFOLIO_VALUE_KEY]
    actual_total = np.zeros(env.num_trajectories, dtype=np.float64)
    behavior = PolicyBehaviorAccumulator()
    behavior.begin_episode(env.num_trajectories)
    try:
        for step in range(env.n_steps):
            action = agent.get_action(obs)
            before = behavior.before_step(obs, action)
            obs, rewards, terminated, truncated, _ = env.step(action)
            wealth[step + 1] = obs[PORTFOLIO_VALUE_KEY]
            actual_total += rewards
            # This accumulator infers an inventory penalty from PnL - reward.
            # Feed PnL for accounting only and omit that inapplicable diagnostic.
            behavior.after_step(before, obs, wealth[step + 1] - wealth[step])
            if (terminated | truncated).all():
                if step != env.n_steps - 1:
                    raise RuntimeError("episode terminated before the configured horizon")
                break
        else:
            raise RuntimeError("episode did not terminate at the configured horizon")
        behavior.finish_episode(obs)
        totals, errors = replay_dense_rewards(wealth, initial_wealth, risk_aversions)
        np.testing.assert_allclose(actual_total, totals[0], rtol=1e-10, atol=1e-10)
        diagnostics = behavior.summarize()
        diagnostics.pop("mean_inventory_penalty_per_path")
        return {
            "width": width, "episode": episode, "seed": seed,
            "pnl": wealth[-1] - initial_wealth, "total_rewards": totals,
            "telescoping_errors": errors, "diagnostics": diagnostics,
            "seconds": perf_counter() - start,
        }
    finally:
        env.close()


def summarize(pnl, rewards, widths, risks, bootstrap_resamples=2000, seed=98765):
    """Paired cluster bootstrap: one cluster is a vectorized episode batch.

    Shapes: PnL (width, episode, trajectory), rewards (width, risk, episode,
    trajectory). Swap execution order is shared within each simulator batch.
    Whole batches are resampled, preserving this dependence and width pairing.
    """
    widths = np.asarray(widths)
    risks = np.asarray(risks)
    n_widths, n_episodes, n_trajectories = pnl.shape
    n_paths = n_episodes * n_trajectories
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_episodes, np.full(n_episodes, 1 / n_episodes),
                              size=bootstrap_resamples) / n_episodes
    batch_means = rewards.mean(axis=-1)
    means = batch_means.mean(axis=-1)
    bootstrap = np.einsum("be,wae->bwa", weights, batch_means)
    reward_ci = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    best = np.argmax(means, axis=0)
    bootstrap_best = np.argmax(bootstrap, axis=1)
    rows, optimum_rows, shape_rows = [], [], []
    for wi, width in enumerate(widths):
        values = pnl[wi].ravel()
        for ai, risk in enumerate(risks):
            # Exponential weights emphasize losses for a > 0, gains for a < 0,
            # and are uniform at a = 0. The neutral CE is simply mean PnL.
            log_terms = -risk * pnl[wi]
            shift = log_terms.max()
            scaled_terms = np.exp(log_terms - shift)
            if risk == 0.0:
                ce = values.mean()
                ce_bootstrap = weights @ pnl[wi].mean(axis=-1)
            elif np.max(np.abs(log_terms)) <= 0.5:
                # log1p avoids cancellation close to risk neutrality.
                mean_rewards = np.concatenate(([means[wi, ai]], bootstrap[:, wi, ai]))
                scaled_means = -risk * mean_rewards
                # Divide out a analytically to retain the mean-PnL limit even
                # when a * mean_reward is subnormal or underflows to zero.
                ratio = np.divide(
                    np.log1p(scaled_means), scaled_means,
                    out=np.ones_like(scaled_means), where=scaled_means != 0.0,
                )
                certainty_equivalents = mean_rewards * ratio
                ce, ce_bootstrap = certainty_equivalents[0], certainty_equivalents[1:]
            else:
                log_batch_means = np.logaddexp.reduce(log_terms, axis=-1) - np.log(n_trajectories)
                ce = -(np.logaddexp.reduce(log_batch_means) - np.log(n_episodes)) / risk
                # A bootstrap sample may omit the episode containing the global
                # exponential maximum. Keep its weighted mean in log space so
                # the remaining episodes cannot all underflow to zero.
                with np.errstate(divide="ignore"):
                    log_weights = np.log(weights)
                ce_bootstrap = -np.logaddexp.reduce(log_weights + log_batch_means, axis=-1) / risk
            ce_ci = np.quantile(ce_bootstrap, [0.025, 0.975])
            utility_weights = scaled_terms.ravel() / scaled_terms.sum()
            row = {
                "risk_aversion": float(risk), "half_width_ticks": int(width),
                "full_width_ticks": int(2 * width), "n_paths": n_paths,
                "mean_total_reward": float(means[wi, ai]),
                "reward_ci_low": float(reward_ci[0, wi, ai]),
                "reward_ci_high": float(reward_ci[1, wi, ai]),
                "certainty_equivalent_pnl": float(ce),
                "ce_ci_low": float(ce_ci[0]), "ce_ci_high": float(ce_ci[1]),
                "mean_pnl": float(values.mean()), "std_pnl": float(values.std(ddof=1)),
                "p05_pnl": float(np.quantile(values, 0.05)),
                "min_pnl": float(values.min()), "max_pnl": float(values.max()),
                "loss_probability": float(np.mean(values < 0)),
                "utility_weight_ess": float(1 / np.sum(utility_weights**2)),
                "largest_utility_weight": float(utility_weights.max()),
                "bootstrap_best_fraction": float(np.mean(bootstrap_best[:, ai] == wi)),
            }
            rows.append(row)
            if best[ai] == wi:
                optimum_rows.append(row.copy())
    # Concavity means non-increasing secant slopes, including uneven grids.
    slopes = np.diff(means, axis=0) / np.diff(widths)[:, None]
    slope_changes = np.diff(slopes, axis=0)
    bootstrap_slopes = np.diff(bootstrap, axis=1) / np.diff(widths)[None, :, None]
    changes_ci = np.quantile(np.diff(bootstrap_slopes, axis=1), [0.025, 0.975], axis=0)
    for ai, risk in enumerate(risks):
        shape_rows.append({
            "risk_aversion": float(risk),
            "sample_curve_is_concave": bool(np.all(slope_changes[:, ai] <= 1e-12)) if n_widths > 2 else None,
            "positive_slope_changes": int(np.sum(slope_changes[:, ai] > 1e-12)),
            "positive_slope_changes_with_pointwise_ci_above_zero": int(np.sum(changes_ci[0, :, ai] > 0)),
        })
    optimum_rows.sort(key=lambda row: row["risk_aversion"])
    summary = {
        "optimal_half_widths_nondecreasing": bool(np.all(np.diff(widths[best]) >= 0)),
        "optimal_half_widths": widths[best].tolist(),
        "curve_shape": shape_rows,
        "uncertainty": "95% pointwise percentile intervals from paired episode-batch bootstrap; not simultaneous intervals or proof of a population optimum",
    }
    return rows, optimum_rows, summary


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(directory, rows, optima, risks):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(risks)))
    fig, axes = plt.subplots((len(risks) + 1) // 2, 2, figsize=(12, 3.2 * ((len(risks) + 1) // 2)), squeeze=False)
    for ax, risk, color in zip(axes.flat, risks, colors):
        selected = [r for r in rows if r["risk_aversion"] == risk]
        x = [r["half_width_ticks"] for r in selected]
        ax.plot(x, [r["mean_total_reward"] for r in selected], "o-", color=color)
        ax.fill_between(x, [r["reward_ci_low"] for r in selected], [r["reward_ci_high"] for r in selected], color=color, alpha=0.18)
        ax.axhline(0, color="gray", linestyle=":", label="Cash")
        ax.set(title=f"risk_aversion = {risk:g}", xlabel="Half-width w (range [−w, +w])", ylabel="Mean total dense reward")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(risks):]:
        ax.set_visible(False)
    fig.suptitle("Periodic rebalancing: reward versus width\nShading: 95% pointwise episode-batch bootstrap intervals")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"reward_vs_width.{suffix}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for risk, color in zip(risks, colors):
        selected = [r for r in rows if r["risk_aversion"] == risk]
        axes[0, 0].plot([r["half_width_ticks"] for r in selected], [r["certainty_equivalent_pnl"] for r in selected], "o-", color=color, label=f"a={risk:g}")
    axes[0, 0].axhline(0, color="gray", linestyle=":")
    axes[0, 0].set(xlabel="Half-width (ticks)", ylabel="Certainty-equivalent PnL (token1)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot([r["risk_aversion"] for r in optima], [r["half_width_ticks"] for r in optima], "o-")
    nonzero_risks = [abs(risk) for risk in risks if risk != 0.0]
    if all(risk > 0.0 for risk in risks):
        axes[0, 1].set_xscale("log")
    elif nonzero_risks:
        axes[0, 1].set_xscale("symlog", linthresh=min(nonzero_risks))
    axes[0, 1].set(xlabel="risk_aversion (1 / token1)", ylabel="Best sampled half-width (ticks)")
    selected = [r for r in rows if r["risk_aversion"] == risks[0]]
    x = [r["half_width_ticks"] for r in selected]
    axes[1, 0].plot(x, [r["mean_pnl"] for r in selected], "o-", label="Mean PnL")
    axes[1, 0].plot(x, [r["p05_pnl"] for r in selected], "o-", label="5th percentile PnL")
    axes[1, 0].axhline(0, color="gray", linestyle=":")
    axes[1, 0].set(xlabel="Half-width (ticks)", ylabel="PnL (token1)")
    axes[1, 0].legend()
    axes[1, 1].plot(x, [r["std_pnl"] for r in selected], "o-")
    axes[1, 1].set(xlabel="Half-width (ticks)", ylabel="Standard deviation of PnL (token1)")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Width selection and the underlying PnL distribution")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"width_selection.{suffix}", dpi=170)
    plt.close(fig)


def main(argv=None):
    args, environment = parse_args(argv)
    start = perf_counter()
    directory = args.output_dir / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "episodes").mkdir()
    config = {
        "experiment": {**vars(args), "output_dir": str(args.output_dir)},
        "trainer_configuration": vars(environment),
        "environment_mode": "fixed nominal; no domain randomization",
        "reward": "DenseExponentialUtility (extended CARA); discount=1; no clipping or normalization",
        "utility": "U(G)=-expm1(-a*G)/a for a!=0; U(G)=G for a=0; G=wealth-initial_wealth",
        "width_definition": "half-width in ticks; full span is 2 * half_width",
        "risk_replay": "production reward applied to every transition of the same wealth paths; first risk also runs inside the environment",
        "seed_pairing": "same episode seed across widths; identical external price shocks, but endogenous swap paths can differ",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Results: {directory.resolve()}", flush=True)
    widths, risks = args.half_widths, args.risk_aversion_values
    n_episodes, n_paths = environment.n_eval_episodes, environment.num_trajectories
    pnl = np.empty((len(widths), n_episodes, n_paths))
    rewards = np.empty((len(widths), len(risks), n_episodes, n_paths))
    errors = np.zeros(len(risks))
    diagnostic_rows = []

    def record(result):
        wi, episode = widths.index(result["width"]), result["episode"]
        pnl[wi, episode] = result["pnl"]
        rewards[wi, :, episode] = result["total_rewards"]
        np.maximum(errors, result["telescoping_errors"], out=errors)
        diagnostic_rows.append({"half_width_ticks": result["width"], "episode": episode,
                                "seed": result["seed"], **result["diagnostics"]})
        np.savez_compressed(directory / "episodes" / f"width_{result['width']:03d}_episode_{episode:04d}.npz",
                            pnl=result["pnl"], total_rewards=result["total_rewards"],
                            risks=risks, seed=result["seed"], diagnostics_json=json.dumps(result["diagnostics"]))
        print(f"Completed {len(diagnostic_rows)}/{len(widths) * n_episodes}: "
              f"width=±{result['width']} episode={episode} "
              f"mean PnL={result['pnl'].mean():.3f} ({result['seconds']:.1f}s)", flush=True)

    jobs = [(width, episode) for width in widths for episode in range(n_episodes)]
    if args.workers == 1:
        for width, episode in jobs:
            record(run_episode(environment, width, episode, risks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_episode, environment, width, episode, risks) for width, episode in jobs]
            for future in as_completed(futures):
                record(future.result())

    rows, optima, summary = summarize(pnl, rewards, widths, risks, args.bootstrap_resamples, args.bootstrap_seed)
    summary["max_scaled_telescoping_error_by_risk"] = errors.tolist()
    summary["elapsed_seconds"] = perf_counter() - start
    summary["best_widths"] = optima
    write_csv(directory / "reward_grid.csv", rows)
    write_csv(directory / "best_widths.csv", optima)
    write_csv(directory / "episode_diagnostics.csv", sorted(diagnostic_rows, key=lambda r: (r["half_width_ticks"], r["episode"])))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(directory / "path_results.npz", pnl=pnl, total_rewards=rewards, half_widths=widths, risk_aversions=risks)
    plot_results(directory, rows, optima, risks)
    print(json.dumps({"best_widths": summary["optimal_half_widths"], "seconds": summary["elapsed_seconds"]}), flush=True)
    return directory


if __name__ == "__main__":
    main()
