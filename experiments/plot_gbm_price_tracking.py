"""Plot pool-price tracking under GBM midprice and liquidity-depth impact."""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/saife_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from SAiFE_gym.agents.BaselineAgents import PeriodicRebalanceAgent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    POOL_SQRT_PRICE_KEY,
    TIME_KEY,
)
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import (
    GeometricBrownianMotionMidpriceModel,
)
from SAiFE_gym.stochastic_processes.price_impact_models import (
    LiquidityDepthUniswapV3PriceImpact,
)


TERMINAL_TIME = 1.0
N_STEPS = 1000
INITIAL_PRICE = 100.0
VOLATILITY = 0.01
DEFAULT_SIGMAS = (0.01, 0.02, 0.04, 0.08, 0.10)
FEE_TIER = 0.003
TAU = 100
NUM_TICKS = 5000
EXPONENTIAL_VALUE = 1.0001
INITIAL_WEALTH = 1000.0
LIQUIDITY_SCALE = 1e5
ALPHA0 = np.array([1.0, 1.0])
ALPHA1 = np.array([150.0, 150.0])
ALPHA2 = np.array([0.0, 0.0])
ALPHA3 = np.array([4000.0, 4000.0])
ARRIVAL_BETA = 0.1
ARRIVAL_KERNEL_TICKS = 20
TRADE_SIZE_NOTIONAL = 350.0
PRICE_IMPACT_DEPTH_WINDOW = 10
DEFAULT_SEED = 42
DEFAULT_FIGURES_DIR = Path("experiments/figures")
DEFAULT_RESULTS_DIR = Path("experiments/results")
LP_POLICY_HOLD = "hold"
LP_POLICY_PERIODIC_UNIFORM = "periodic_uniform"
DEFAULT_LP_POLICIES = (LP_POLICY_HOLD, LP_POLICY_PERIODIC_UNIFORM)
PERIODIC_REBALANCE_EVERY = 10


def constant_trade_size_sampler(trade_notional: float):
    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, trade_notional, dtype=np.float64)

    return sampler


def _alpha_matrix(alpha1: float, alpha3: float) -> np.ndarray:
    return np.array(
        [
            ALPHA0,
            np.array([alpha1, alpha1], dtype=np.float64),
            ALPHA2,
            np.array([alpha3, alpha3], dtype=np.float64),
        ],
        dtype=np.float64,
    )


def build_environment(
    n_steps: int = N_STEPS,
    seed: int = DEFAULT_SEED,
    volatility: float = VOLATILITY,
    trade_size_notional: float = TRADE_SIZE_NOTIONAL,
    alpha1: float = float(ALPHA1[0]),
    alpha3: float = float(ALPHA3[0]),
) -> AMMEnvironment:
    step_size = TERMINAL_TIME / n_steps
    alpha = _alpha_matrix(alpha1, alpha3)

    midprice_model = GeometricBrownianMotionMidpriceModel(
        drift=0.0,
        volatility=volatility,
        initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME,
        step_size=step_size,
        num_trajectories=1,
        seed=seed,
    )
    arrival_model = LiquidityKernelArrivalModel(
        alpha=alpha,
        beta=ARRIVAL_BETA,
        K=ARRIVAL_KERNEL_TICKS,
        liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size,
        num_trajectories=1,
        seed=seed + 1,
    )
    price_impact_model = LiquidityDepthUniswapV3PriceImpact(
        trade_size_sampler=constant_trade_size_sampler(trade_size_notional),
        depth_window=PRICE_IMPACT_DEPTH_WINDOW,
        trade_size_unit="token1_notional",
        num_trajectories=1,
        seed=seed + 3,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        price_impact_model=price_impact_model,
        num_trajectories=1,
        fee_tier=FEE_TIER,
        tau=TAU,
        num_ticks=NUM_TICKS,
        exponential_value=EXPONENTIAL_VALUE,
        seed=seed + 2,
    )
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME,
        n_steps=n_steps,
        initial_wealth=INITIAL_WEALTH,
        reward_function=PnL(),
        model_dynamics=model_dynamics,
        num_trajectories=1,
        initial_pool_price=None,
        seed=seed,
    )


def _make_lp_policy(env: AMMEnvironment, lp_policy: str):
    if lp_policy == LP_POLICY_HOLD:
        return None
    if lp_policy == LP_POLICY_PERIODIC_UNIFORM:
        return PeriodicRebalanceAgent(
            env,
            rebalance_every=PERIODIC_REBALANCE_EVERY,
            width=env.model_dynamics.tau,
        )
    raise ValueError(f"unknown LP policy: {lp_policy}")


def _policy_action(policy, state: dict, env: AMMEnvironment) -> np.ndarray:
    if policy is None:
        return np.array([[0.0, 1.0, 1.0]], dtype=np.float32)
    return policy.get_action(state)


def _record_state(data: dict, state: dict, sell_count: int, buy_count: int) -> None:
    pool_price = float(state[POOL_SQRT_PRICE_KEY][0] ** 2)
    midprice = float(state[ASSET_PRICE_KEY][0])
    data["time"].append(float(state[TIME_KEY][0]))
    data["pool_price"].append(pool_price)
    data["midprice"].append(midprice)
    data["price_gap"].append(pool_price - midprice)
    data["cumulative_sell_arrivals"].append(sell_count)
    data["cumulative_buy_arrivals"].append(buy_count)
    data["cumulative_total_arrivals"].append(sell_count + buy_count)


def collect_price_tracking(
    n_steps: int = N_STEPS,
    seed: int = DEFAULT_SEED,
    volatility: float = VOLATILITY,
    trade_size_notional: float = TRADE_SIZE_NOTIONAL,
    alpha1: float = float(ALPHA1[0]),
    alpha3: float = float(ALPHA3[0]),
    lp_policy: str = LP_POLICY_HOLD,
) -> dict:
    env = build_environment(
        n_steps=n_steps,
        seed=seed,
        volatility=volatility,
        trade_size_notional=trade_size_notional,
        alpha1=alpha1,
        alpha3=alpha3,
    )
    state, _ = env.reset()
    policy = _make_lp_policy(env, lp_policy)
    sell_count = 0
    buy_count = 0
    data = {
        "time": [],
        "pool_price": [],
        "midprice": [],
        "price_gap": [],
        "cumulative_sell_arrivals": [],
        "cumulative_buy_arrivals": [],
        "cumulative_total_arrivals": [],
    }
    _record_state(data, state, sell_count, buy_count)

    for _ in range(n_steps):
        action = _policy_action(policy, state, env)
        state, _, terminated, _, _ = env.step(action)
        arrivals = env.model_dynamics.last_arrivals[0]
        sell_count += int(arrivals[0])
        buy_count += int(arrivals[1])
        _record_state(data, state, sell_count, buy_count)
        if terminated[0]:
            break

    return {key: np.asarray(value) for key, value in data.items()}


def summarize_tracking(data: dict) -> dict:
    pool_price = data["pool_price"]
    midprice = data["midprice"]
    gap = data["price_gap"]
    if np.std(pool_price) > 0.0 and np.std(midprice) > 0.0:
        correlation = float(np.corrcoef(pool_price, midprice)[0, 1])
    else:
        correlation = float("nan")

    return {
        "mean_abs_gap": float(np.mean(np.abs(gap))),
        "max_abs_gap": float(np.max(np.abs(gap))),
        "correlation": correlation,
        "total_sell_arrivals": int(data["cumulative_sell_arrivals"][-1]),
        "total_buy_arrivals": int(data["cumulative_buy_arrivals"][-1]),
        "total_arrivals": int(data["cumulative_total_arrivals"][-1]),
        "final_pool_price": float(pool_price[-1]),
        "final_midprice": float(midprice[-1]),
        "final_gap": float(gap[-1]),
    }


def save_csv(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        columns = list(data)
        writer.writerow(columns)
        for row in zip(*(data[column] for column in columns)):
            writer.writerow(row)


def plot_price_tracking(data: dict, summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_price, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(
        data["time"],
        data["pool_price"],
        color="black",
        linewidth=1.4,
        label="Pool Price",
    )
    ax_price.plot(
        data["time"],
        data["midprice"],
        color="0.55",
        linewidth=1.1,
        alpha=0.8,
        label="External Midprice",
    )
    ax_price.set_ylabel("Price")
    ax_price.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left")
    ax_price.set_title(
        "GBM Midprice vs Pool Price "
        f"(mean |gap|={summary['mean_abs_gap']:.4f}, "
        f"corr={summary['correlation']:.3f})"
    )

    ax_gap.plot(
        data["time"],
        data["price_gap"],
        color="#4c78a8",
        linewidth=1.0,
        label="Pool - Midprice",
    )
    ax_gap.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    ax_gap.set_xlabel("Time")
    ax_gap.set_ylabel("Gap")
    ax_gap.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _iter_sweep_results(sweep_results: dict):
    for lp_policy, policy_results in sweep_results.items():
        for sigma, output in policy_results.items():
            yield lp_policy, sigma, output


def save_sweep_csv(sweep_results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "lp_policy",
        "sigma",
        "time",
        "pool_price",
        "midprice",
        "price_gap",
        "cumulative_sell_arrivals",
        "cumulative_buy_arrivals",
        "cumulative_total_arrivals",
    ]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for lp_policy, sigma, output in _iter_sweep_results(sweep_results):
            data = output["data"]
            for row in zip(*(data[column] for column in columns[2:])):
                writer.writerow((lp_policy, sigma, *row))


def save_sweep_summary_csv(sweep_results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_keys = [
        "mean_abs_gap",
        "max_abs_gap",
        "correlation",
        "total_sell_arrivals",
        "total_buy_arrivals",
        "total_arrivals",
        "final_pool_price",
        "final_midprice",
        "final_gap",
    ]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["lp_policy", "sigma", *summary_keys])
        for lp_policy, sigma, output in _iter_sweep_results(sweep_results):
            summary = output["summary"]
            writer.writerow([lp_policy, sigma, *(summary[key] for key in summary_keys)])


def plot_sweep_tracking(sweep_results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first_policy = next(iter(sweep_results))
    sigmas = list(sweep_results[first_policy])
    policies = list(sweep_results)
    nrows = len(sigmas)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(13, 3.2 * nrows),
        sharex=True,
        squeeze=False,
        gridspec_kw={"width_ratios": [3, 2]},
    )
    policy_styles = {
        LP_POLICY_HOLD: {
            "color": "black",
            "linewidth": 1.3,
            "label": "Pool Price (hold)",
        },
        LP_POLICY_PERIODIC_UNIFORM: {
            "color": "#d95f02",
            "linewidth": 1.2,
            "label": "Pool Price (periodic uniform LP)",
        },
    }

    for row, sigma in enumerate(sigmas):
        ax_price, ax_gap = axes[row]
        midprice_data = sweep_results[policies[0]][sigma]["data"]

        ax_price.plot(
            midprice_data["time"],
            midprice_data["midprice"],
            color="0.55",
            linewidth=1.0,
            alpha=0.85,
            label="External Midprice",
        )
        for lp_policy in policies:
            output = sweep_results[lp_policy][sigma]
            data = output["data"]
            style = policy_styles.get(
                lp_policy,
                {
                    "color": None,
                    "linewidth": 1.1,
                    "label": f"Pool Price ({lp_policy})",
                },
            )
            ax_price.plot(
                data["time"],
                data["pool_price"],
                color=style["color"],
                linewidth=style["linewidth"],
                label=style["label"],
            )
        ax_price.set_ylabel(f"sigma={sigma:g}\nPrice")
        ax_price.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax_price.grid(True, alpha=0.3)
        if row == 0:
            ax_price.legend(loc="upper left")

        title_parts = []
        for lp_policy in policies:
            output = sweep_results[lp_policy][sigma]
            data = output["data"]
            summary = output["summary"]
            style = policy_styles.get(
                lp_policy,
                {
                    "color": None,
                    "linewidth": 1.0,
                    "label": f"Pool - Midprice ({lp_policy})",
                },
            )
            ax_gap.plot(
                data["time"],
                data["price_gap"],
                color=style["color"],
                linewidth=style["linewidth"],
                label=f"{lp_policy}: Pool - Midprice",
            )
            title_parts.append(
                f"{lp_policy} mean |gap|={summary['mean_abs_gap']:.4f}"
            )
        ax_gap.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
        ax_gap.set_ylabel("Gap")
        ax_gap.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax_gap.grid(True, alpha=0.3)
        ax_gap.set_title("; ".join(title_parts))
        if row == 0:
            ax_gap.legend(loc="upper left")

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    n_steps: int = N_STEPS,
    seed: int = DEFAULT_SEED,
    volatility: float = VOLATILITY,
    trade_size_notional: float = TRADE_SIZE_NOTIONAL,
    alpha1: float = float(ALPHA1[0]),
    alpha3: float = float(ALPHA3[0]),
    lp_policy: str = LP_POLICY_HOLD,
    save_plot: bool = True,
    save_csv_output: bool = True,
    figures_dir: Path | str = DEFAULT_FIGURES_DIR,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> dict:
    data = collect_price_tracking(
        n_steps=n_steps,
        seed=seed,
        volatility=volatility,
        trade_size_notional=trade_size_notional,
        alpha1=alpha1,
        alpha3=alpha3,
        lp_policy=lp_policy,
    )
    summary = summarize_tracking(data)
    output = {
        "data": data,
        "summary": summary,
        "plot_path": None,
        "csv_path": None,
    }

    if save_plot:
        plot_path = (
            Path(figures_dir)
            / (
                f"gbm_price_tracking_{lp_policy}_sigma{volatility:g}"
                f"_trade{trade_size_notional:g}_seed{seed}.png"
            )
        )
        plot_price_tracking(data, summary, plot_path)
        output["plot_path"] = plot_path
    if save_csv_output:
        csv_path = (
            Path(results_dir)
            / (
                f"gbm_price_tracking_{lp_policy}_sigma{volatility:g}"
                f"_trade{trade_size_notional:g}_seed{seed}.csv"
            )
        )
        save_csv(data, csv_path)
        output["csv_path"] = csv_path

    return output


def run_sweep(
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    lp_policies: Sequence[str] = (LP_POLICY_HOLD,),
    n_steps: int = N_STEPS,
    seed: int = DEFAULT_SEED,
    trade_size_notional: float = TRADE_SIZE_NOTIONAL,
    alpha1: float = float(ALPHA1[0]),
    alpha3: float = float(ALPHA3[0]),
    save_plot: bool = True,
    save_csv_output: bool = True,
    figures_dir: Path | str = DEFAULT_FIGURES_DIR,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> dict:
    sweep_results = {}
    for lp_policy in lp_policies:
        sweep_results[lp_policy] = {}
        for sigma in sigmas:
            sweep_results[lp_policy][float(sigma)] = run_experiment(
                n_steps=n_steps,
                seed=seed,
                volatility=float(sigma),
                trade_size_notional=trade_size_notional,
                alpha1=alpha1,
                alpha3=alpha3,
                lp_policy=lp_policy,
                save_plot=False,
                save_csv_output=False,
                figures_dir=figures_dir,
                results_dir=results_dir,
            )

    output = {
        "results": sweep_results,
        "plot_path": None,
        "csv_path": None,
        "summary_csv_path": None,
    }
    if save_plot:
        prefix = (
            "gbm_price_tracking_policy_sweep"
            if len(lp_policies) > 1
            else f"gbm_price_tracking_{lp_policies[0]}_sweep"
        )
        plot_path = (
            Path(figures_dir)
            / f"{prefix}_trade{trade_size_notional:g}_seed{seed}.png"
        )
        plot_sweep_tracking(sweep_results, plot_path)
        output["plot_path"] = plot_path
    if save_csv_output:
        prefix = (
            "gbm_price_tracking_policy_sweep"
            if len(lp_policies) > 1
            else f"gbm_price_tracking_{lp_policies[0]}_sweep"
        )
        csv_path = (
            Path(results_dir)
            / f"{prefix}_trade{trade_size_notional:g}_seed{seed}.csv"
        )
        summary_csv_path = (
            Path(results_dir)
            / f"{prefix}_summary_trade{trade_size_notional:g}_seed{seed}.csv"
        )
        save_sweep_csv(sweep_results, csv_path)
        save_sweep_summary_csv(sweep_results, summary_csv_path)
        output["csv_path"] = csv_path
        output["summary_csv_path"] = summary_csv_path

    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot external GBM midprice and pool price under liquidity-depth impact."
    )
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sigmas", nargs="+", type=float, default=list(DEFAULT_SIGMAS))
    parser.add_argument("--trade-size-notional", type=float, default=TRADE_SIZE_NOTIONAL)
    parser.add_argument("--alpha1", type=float, default=float(ALPHA1[0]))
    parser.add_argument("--alpha3", type=float, default=float(ALPHA3[0]))
    parser.add_argument(
        "--lp-policies",
        nargs="+",
        choices=list(DEFAULT_LP_POLICIES),
        default=list(DEFAULT_LP_POLICIES),
    )
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = run_sweep(
        sigmas=args.sigmas,
        lp_policies=args.lp_policies,
        n_steps=args.n_steps,
        seed=args.seed,
        trade_size_notional=args.trade_size_notional,
        alpha1=args.alpha1,
        alpha3=args.alpha3,
        save_plot=not args.no_plot,
        save_csv_output=not args.no_csv,
        figures_dir=args.figures_dir,
        results_dir=args.results_dir,
    )
    print("GBM price-tracking volatility sweep")
    print(f"  steps: {args.n_steps}")
    print(f"  seed: {args.seed}")
    print(f"  sigmas: {', '.join(f'{sigma:g}' for sigma in args.sigmas)}")
    print(f"  trade_size_notional: {args.trade_size_notional:g}")
    print(f"  alpha1: {args.alpha1:g}")
    print(f"  alpha3: {args.alpha3:g}")
    print(f"  lp_policies: {', '.join(args.lp_policies)}")
    print("\n  policy           | sigma | mean |pool-mid| | max |pool-mid| | corr | arrivals")
    print("  " + "-" * 83)
    for lp_policy, policy_results in output["results"].items():
        for sigma, result in policy_results.items():
            summary = result["summary"]
            print(
                f"  {lp_policy:<16} | "
                f"{sigma:>5g} | "
                f"{summary['mean_abs_gap']:>15.6f} | "
                f"{summary['max_abs_gap']:>14.6f} | "
                f"{summary['correlation']:>5.3f} | "
                f"{summary['total_arrivals']:>8d}"
            )
    if output["plot_path"] is not None:
        print(f"  plot: {output['plot_path']}")
    if output["csv_path"] is not None:
        print(f"  csv: {output['csv_path']}")
    if output["summary_csv_path"] is not None:
        print(f"  summary_csv: {output['summary_csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
