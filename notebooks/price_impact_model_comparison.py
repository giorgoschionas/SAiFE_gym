"""
Compare whether pool prices track the external midprice under the one-tick and
liquidity-depth Uniswap V3 price-impact models.

This script reuses the baseline-agent configuration from
notebooks/agent_comparison.py and only runs single-trajectory diagnostics.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

import agent_comparison as ac

from SAiFE_gym.agents.BaselineAgents import (
    ArrivalRebalanceAgent,
    DeployOnceAgent,
    DoNothingAgent,
    UniformAllocationAgent,
)
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import OrnsteinUhlenbeckMidpriceModel
from SAiFE_gym.stochastic_processes.price_impact_models import (
    LiquidityDepthUniswapV3PriceImpact,
)


job_id = os.environ.get("SLURM_JOB_ID", "local")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

PRICE_IMPACT_MODELS = {
    "OneTick": "one_tick",
    "LiquidityDepth": "liquidity_depth",
    "LiquidityDepthLognormal": "liquidity_depth_lognormal",
}
MODEL_COLORS = {
    "OneTick": "#4c78a8",
    "LiquidityDepth": "#f58518",
    "LiquidityDepthLognormal": "#54a24b",
}

BASELINE_AGENT_NAMES = [
    name
    for name in [
        "DoNothing",
        "Uniform",
        "DeployOnce",
        "ArrivalRebalance",
    ]
    if ac.ENABLE_AGENTS.get(name)
]

# Constant token1-notional trade size used by LiquidityDepthUniswapV3PriceImpact.
# Around price 100 and base liquidity 100_000, one tick of buy/sell input is
# about 50 token1 of notional value. Sell-side notionals are converted to
# token0 using the external midprice.
LIQUIDITY_DEPTH_TRADE_NOTIONAL = 40.0
LIQUIDITY_DEPTH_LOGNORMAL_MEAN_NOTIONAL = 40.0
LIQUIDITY_DEPTH_LOGNORMAL_SIGMA = 1.2
LIQUIDITY_DEPTH_WINDOW = 10
LIQUIDITY_DEPTH_MIN_DEPTH = 1e-12


def constant_trade_size_sampler(trade_notional: float):
    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, trade_notional, dtype=np.float64)

    return sampler


def lognormal_trade_size_sampler(mean_notional: float, sigma: float):
    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        safe_mean = max(mean_notional, 0.01)
        safe_sigma = max(sigma, 0.01)
        mu = float(np.log(safe_mean) - 0.5 * safe_sigma * safe_sigma)
        return rng.lognormal(mu, safe_sigma, size=size)

    return sampler


def create_environment(
    num_trajectories: int,
    seed: int = None,
    price_impact_kind: str = "one_tick",
):
    step_size = ac.TERMINAL_TIME / ac.N_STEPS
    alpha = np.array([ac.ALPHA0, ac.ALPHA1, ac.ALPHA2, ac.ALPHA3])

    midprice_model = OrnsteinUhlenbeckMidpriceModel(
        mean_reversion=0.4,
        long_term_mean=ac.INITIAL_PRICE,
        volatility=ac.VOLATILITY,
        num_trajectories=num_trajectories,
        seed=seed,
        initial_price=ac.INITIAL_PRICE,
        terminal_time=ac.TERMINAL_TIME,
        step_size=step_size,
    )

    arrival_model = LiquidityKernelArrivalModel(
        alpha=alpha,
        beta=0.1,
        K=20,
        liquidity_scale=ac.LIQUIDITY_SCALE,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1 if seed is not None else None,
    )

    price_impact_model = None
    if price_impact_kind in {"liquidity_depth", "liquidity_depth_lognormal"}:
        trade_size_sampler = constant_trade_size_sampler(LIQUIDITY_DEPTH_TRADE_NOTIONAL)
        if price_impact_kind == "liquidity_depth_lognormal":
            trade_size_sampler = lognormal_trade_size_sampler(
                LIQUIDITY_DEPTH_LOGNORMAL_MEAN_NOTIONAL,
                LIQUIDITY_DEPTH_LOGNORMAL_SIGMA,
            )

        price_impact_model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=trade_size_sampler,
            depth_window=LIQUIDITY_DEPTH_WINDOW,
            min_depth=LIQUIDITY_DEPTH_MIN_DEPTH,
            trade_size_unit="token1_notional",
            num_trajectories=num_trajectories,
            seed=seed + 3 if seed is not None else None,
        )
    elif price_impact_kind != "one_tick":
        raise ValueError(f"Unknown price_impact_kind={price_impact_kind!r}")

    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        price_impact_model=price_impact_model,
        num_trajectories=num_trajectories,
        fee_tier=ac.FEE_TIER,
        tau=ac.TAU,
        num_ticks=5000,
        exponential_value=ac.EXP_VALUE,
        seed=seed + 2 if seed is not None else None,
    )

    return AMMEnvironment(
        terminal_time=ac.TERMINAL_TIME,
        n_steps=ac.N_STEPS,
        initial_wealth=ac.INITIAL_WEALTH,
        reward_function=PnL(),
        model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        initial_pool_price=ac.INITIAL_POOL_PRICE,
        seed=seed,
    )


def make_agent(agent_name: str, env: AMMEnvironment):
    if agent_name == "DoNothing":
        return DoNothingAgent(env)
    if agent_name == "Uniform":
        return UniformAllocationAgent(env)
    if agent_name == "DeployOnce":
        return DeployOnceAgent(
            env,
            lower_offset=ac.DEPLOYONCE_LOWER,
            upper_offset=ac.DEPLOYONCE_UPPER,
        )
    if agent_name == "ArrivalRebalance":
        return ArrivalRebalanceAgent(
            env,
            rebalance_every=ac.ARRIVAL_REBALANCE_EVERY,
            width=ac.ARRIVAL_REBALANCE_WIDTH,
            lower_offset=ac.ARRIVAL_REBALANCE_LOWER,
            upper_offset=ac.ARRIVAL_REBALANCE_UPPER,
        )
    raise ValueError(f"Unsupported baseline agent {agent_name!r}")


def collect_price_tracking_data():
    tracking_data = {}
    single_seed_base = ac.SEED + 7777

    for model_label, price_impact_kind in PRICE_IMPACT_MODELS.items():
        tracking_data[model_label] = {}
        print(f"\nCollecting {model_label}")
        for sim_idx in range(ac.NUM_SINGLE_SIMS):
            sim_seed = single_seed_base + sim_idx
            for agent_name in BASELINE_AGENT_NAMES:
                env = create_environment(1, sim_seed, price_impact_kind=price_impact_kind)
                agent = make_agent(agent_name, env)
                trajectory = ac.collect_single_trajectory(
                    env,
                    agent.get_action,
                    max_trades=ac.MAX_TRADES_DEBUG,
                )
                tracking_data[model_label].setdefault(agent_name, []).append(trajectory)

    return tracking_data


def _tracking_stats(trajectory: dict) -> tuple[float, float, float, int]:
    pool_price = trajectory["pool_price"]
    midprice = trajectory["midprice"]
    gap = pool_price - midprice

    if np.std(pool_price) > 0.0 and np.std(midprice) > 0.0:
        correlation = np.corrcoef(pool_price, midprice)[0, 1]
    else:
        correlation = np.nan

    return (
        float(np.mean(np.abs(gap))),
        float(np.max(np.abs(gap))),
        float(correlation),
        int(trajectory["trade_count"][-1]) if len(trajectory["trade_count"]) else 0,
    )


def print_tracking_summary(tracking_data: dict):
    impact_width = max(15, max(len(label) for label in tracking_data))
    header = (
        f"{'Impact':<{impact_width}} | {'Agent':<18} | {'Mean |Pool-Mid|':>16} "
        f"| {'Max |Pool-Mid|':>15} | {'Corr':>8} | {'Avg Trades':>10}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))

    for model_label, agent_data in tracking_data.items():
        for agent_name in BASELINE_AGENT_NAMES:
            stats = [_tracking_stats(t) for t in agent_data[agent_name]]
            mean_abs_gap, max_abs_gap, correlation, trades = np.array(stats).T
            print(
                f"{model_label:<{impact_width}} | {agent_name:<18} "
                f"| {np.mean(mean_abs_gap):>16.6f} "
                f"| {np.mean(max_abs_gap):>15.6f} "
                f"| {np.nanmean(correlation):>8.3f} "
                f"| {np.mean(trades):>10.1f}"
            )

    print("-" * len(header))


def _plot_agent_price_paths(ax, tracking_data: dict, agent_name: str, sim_idx: int = None):
    plotted_midprice = False
    plotted_models = set()

    for model_label, agent_data in tracking_data.items():
        trajectories = agent_data[agent_name]
        selected = trajectories if sim_idx is None else [trajectories[sim_idx]]
        alpha = max(0.2, 0.8 / len(selected)) if sim_idx is None else 1.0

        for trajectory in selected:
            if not plotted_midprice:
                ax.plot(
                    trajectory["time"],
                    trajectory["midprice"],
                    color="0.55",
                    linewidth=1.0,
                    alpha=0.7,
                    label="External Midprice",
                )
                plotted_midprice = True

            ax.plot(
                trajectory["time"],
                trajectory["pool_price"],
                color=MODEL_COLORS[model_label],
                linewidth=1.2 if sim_idx is None else 1.5,
                alpha=alpha,
                label=model_label if model_label not in plotted_models else None,
            )
            plotted_models.add(model_label)

    ax.set_title(agent_name)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_price_tracking_overview(tracking_data: dict):
    n_agents = len(BASELINE_AGENT_NAMES)
    if n_agents == 0:
        return

    ncols = 2
    nrows = (n_agents + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for ax in axes.flat[n_agents:]:
        ax.set_visible(False)

    for ax, agent_name in zip(axes.flat, BASELINE_AGENT_NAMES):
        _plot_agent_price_paths(ax, tracking_data, agent_name)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f"price_impact_tracking_overview_{job_id}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_price_tracking_by_sim(tracking_data: dict):
    if not BASELINE_AGENT_NAMES:
        return

    n_sims = max(
        len(agent_data[agent_name])
        for agent_data in tracking_data.values()
        for agent_name in BASELINE_AGENT_NAMES
    )
    if n_sims <= 1:
        return

    n_agents = len(BASELINE_AGENT_NAMES)
    ncols = 2
    nrows = (n_agents + ncols - 1) // ncols

    for sim_idx in range(n_sims):
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
        axes = np.atleast_2d(axes)

        for ax in axes.flat[n_agents:]:
            ax.set_visible(False)

        for ax, agent_name in zip(axes.flat, BASELINE_AGENT_NAMES):
            _plot_agent_price_paths(ax, tracking_data, agent_name, sim_idx=sim_idx)

        fig.tight_layout()
        path = os.path.join(
            FIGURES_DIR,
            f"price_impact_tracking_sim{sim_idx + 1}_{job_id}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def main():
    np.random.seed(ac.SEED)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 72)
    print("Pool-price tracking under one-tick and liquidity-depth impact")
    print("=" * 72)
    print(f"Agents: {', '.join(BASELINE_AGENT_NAMES)}")
    print(f"Single sims per agent/model: {ac.NUM_SINGLE_SIMS}")
    print(
        "LiquidityDepth constant config: "
        f"trade_notional={LIQUIDITY_DEPTH_TRADE_NOTIONAL}, "
        f"depth_window={LIQUIDITY_DEPTH_WINDOW}"
    )
    print(
        "LiquidityDepth lognormal config: "
        f"mean_notional={LIQUIDITY_DEPTH_LOGNORMAL_MEAN_NOTIONAL}, "
        f"sigma={LIQUIDITY_DEPTH_LOGNORMAL_SIGMA}, "
        f"depth_window={LIQUIDITY_DEPTH_WINDOW}"
    )

    tracking_data = collect_price_tracking_data()
    print_tracking_summary(tracking_data)

    print("\nGenerating price-tracking plots")
    plot_price_tracking_overview(tracking_data)
    plot_price_tracking_by_sim(tracking_data)
    print("\nDone. Price-tracking figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
