"""
Compare baseline-agent behaviour under the one-tick and liquidity-depth
Uniswap V3 price-impact models.

This script intentionally reuses the baseline-agent configuration from
notebooks/agent_comparison.py, but does not train or evaluate RL agents.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch

import agent_comparison as ac

from SAiFE_gym.agents.BaselineAgents import (
    ArrivalRebalanceAgent,
    CarteaPLAgent,
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
}

BASELINE_AGENT_NAMES = [
    name
    for name in [
        "DoNothing",
        "Uniform",
        "DeployOnce",
        "ArrivalRebalance",
        "CarteaDrissiMonga",
    ]
    if ac.ENABLE_AGENTS.get(name)
]

# Constant token1-notional trade size used by LiquidityDepthUniswapV3PriceImpact.
# Around price 100 and base liquidity 100_000, one tick of buy/sell input is
# about 50 token1 of notional value. Sell-side notionals are converted to
# token0 using the external midprice.
LIQUIDITY_DEPTH_TRADE_NOTIONAL = 40.0
LIQUIDITY_DEPTH_WINDOW = 10
LIQUIDITY_DEPTH_MIN_DEPTH = 1e-12


def constant_trade_size_sampler(trade_notional: float):
    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, trade_notional, dtype=np.float64)

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
    if price_impact_kind == "liquidity_depth":
        price_impact_model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=constant_trade_size_sampler(LIQUIDITY_DEPTH_TRADE_NOTIONAL),
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
    if agent_name == "CarteaDrissiMonga":
        return CarteaPLAgent(env, gamma=ac.GAMMA_CARTEA, seed=ac.SEED)
    raise ValueError(f"Unsupported baseline agent {agent_name!r}")


def evaluate_model_agents(model_label: str, price_impact_kind: str):
    results = {}
    eval_seed = ac.SEED + 999
    for agent_name in BASELINE_AGENT_NAMES:
        env = create_environment(
            ac.NUM_TRAJECTORIES_EVAL,
            eval_seed,
            price_impact_kind=price_impact_kind,
        )
        agent = make_agent(agent_name, env)
        results[agent_name] = ac.evaluate_on_trajectories(env, agent.get_action)
    return results


def collect_model_single_data(model_label: str, price_impact_kind: str):
    single_data = {}
    single_seed_base = ac.SEED + 7777
    for sim_idx in range(ac.NUM_SINGLE_SIMS):
        sim_seed = single_seed_base + sim_idx
        for agent_name in BASELINE_AGENT_NAMES:
            env = create_environment(1, sim_seed, price_impact_kind=price_impact_kind)
            agent = make_agent(agent_name, env)
            single_data.setdefault(agent_name, []).append(
                ac.collect_single_trajectory(
                    env,
                    agent.get_action,
                    max_trades=ac.MAX_TRADES_DEBUG,
                )
            )
    return single_data


def print_pnl_summary(all_results):
    header = (
        f"{'Impact':<15} | {'Agent':<18} | {'Mean PnL':>10} | {'Std':>10} "
        f"| {'Median':>10} | {'Profitable':>12}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for model_label, agent_results in all_results.items():
        for agent_name in BASELINE_AGENT_NAMES:
            pnl = agent_results[agent_name]
            pct = 100 * np.mean(pnl > 0)
            print(
                f"{model_label:<15} | {agent_name:<18} | {np.mean(pnl):>+10.2f} "
                f"| {np.std(pnl):>10.2f} | {np.median(pnl):>+10.2f} "
                f"| {np.sum(pnl > 0):>4d}/{len(pnl)} ({pct:.0f}%)"
            )
    print("-" * len(header))

    if set(all_results) >= {"OneTick", "LiquidityDepth"}:
        print("\nLiquidityDepth - OneTick mean PnL deltas:")
        for agent_name in BASELINE_AGENT_NAMES:
            delta = (
                np.mean(all_results["LiquidityDepth"][agent_name])
                - np.mean(all_results["OneTick"][agent_name])
            )
            print(f"  {agent_name:<18} {delta:+.2f}")


def plot_grouped_pnl_boxplot(all_results):
    fig, ax = plt.subplots(figsize=(11, 5))
    data = []
    positions = []
    labels = []
    colors = []
    pos = 1
    palette = {"OneTick": "#4c78a8", "LiquidityDepth": "#f58518"}

    for agent_name in BASELINE_AGENT_NAMES:
        for model_label in all_results:
            data.append(all_results[model_label][agent_name])
            positions.append(pos)
            labels.append(f"{agent_name}\n{model_label}")
            colors.append(palette.get(model_label, "#999999"))
            pos += 1
        pos += 1

    bp = ax.boxplot(data, positions=positions, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Baseline PnL by Price-Impact Model")
    ax.set_ylabel("Cumulative PnL")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, f"price_impact_model_pnl_boxplot_{job_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_single_pnl_evolution(all_single_data):
    fig, axes = plt.subplots(
        len(BASELINE_AGENT_NAMES),
        1,
        figsize=(10, max(3, 3 * len(BASELINE_AGENT_NAMES))),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    colors = {"OneTick": "#4c78a8", "LiquidityDepth": "#f58518"}

    for ax, agent_name in zip(axes, BASELINE_AGENT_NAMES):
        for model_label, single_data in all_single_data.items():
            for sim_idx, trajectory in enumerate(single_data[agent_name]):
                label = model_label if sim_idx == 0 else None
                ax.plot(
                    trajectory["time"],
                    trajectory["cumulative_pnl"],
                    color=colors.get(model_label),
                    alpha=0.25,
                    linewidth=1.0,
                    label=label,
                )
        ax.set_title(agent_name)
        ax.set_ylabel("PnL")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Time")
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, f"price_impact_model_pnl_evolution_{job_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_liquidity_depth_price_evolution(liquidity_depth_single_data):
    agents = [name for name in BASELINE_AGENT_NAMES if name in liquidity_depth_single_data]
    if not agents:
        return

    n_agents = len(agents)
    ncols = 2
    nrows = (n_agents + ncols - 1) // ncols
    n_sims = max(len(v) for v in liquidity_depth_single_data.values())
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for ax in axes.flat[n_agents:]:
        ax.set_visible(False)

    for ax, agent_name in zip(axes.flat, agents):
        color = ac.AGENT_COLORS[agent_name]
        sim_alpha = max(0.15, 0.8 / len(liquidity_depth_single_data[agent_name]))
        for sim_idx, data in enumerate(liquidity_depth_single_data[agent_name]):
            pool_label = "Pool Price" if sim_idx == 0 else None
            mid_label = "Midprice" if sim_idx == 0 else None
            range_label = f"{agent_name} Range" if sim_idx == 0 else None

            ax.plot(
                data["time"],
                data["pool_price"],
                "k-",
                linewidth=1.0,
                alpha=sim_alpha,
                label=pool_label,
            )
            ax.plot(
                data["time"],
                data["midprice"],
                color="gray",
                linewidth=0.8,
                alpha=sim_alpha * 0.7,
                label=mid_label,
            )
            ax.fill_between(
                data["time"],
                data["position_lower_price"],
                data["position_upper_price"],
                alpha=sim_alpha * 0.3,
                color=color,
                label=range_label,
            )
            ax.plot(
                data["time"],
                data["position_lower_price"],
                "--",
                color=color,
                alpha=sim_alpha * 0.6,
                linewidth=0.8,
            )
            ax.plot(
                data["time"],
                data["position_upper_price"],
                "--",
                color=color,
                alpha=sim_alpha * 0.6,
                linewidth=0.8,
            )

            has_hold_flag = not np.all(data["hold_flag"] == -1.0)
            if has_hold_flag:
                rebalance_mask = data["hold_flag"] <= 0
                rebalance_times = data["time"][rebalance_mask]
                for rebalance_time in rebalance_times:
                    ax.axvline(rebalance_time, color="red", alpha=0.1, linewidth=0.5)
                if sim_idx == 0 and len(rebalance_times) > 0:
                    ax.axvline(
                        rebalance_times[0],
                        color="red",
                        alpha=0.3,
                        linewidth=0.5,
                        label="Rebalance",
                    )

        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Price", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.set_title(agent_name)
        ax.legend(fontsize=14, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(
        FIGURES_DIR,
        f"liquidity_depth_price_evolution_{job_id}.png",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    if n_sims <= 1:
        return

    for sim_idx in range(n_sims):
        fig_i, axes_i = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        axes_i = np.atleast_2d(axes_i)
        for ax in axes_i.flat[n_agents:]:
            ax.set_visible(False)

        for ax, agent_name in zip(axes_i.flat, agents):
            if sim_idx >= len(liquidity_depth_single_data[agent_name]):
                continue
            data = liquidity_depth_single_data[agent_name][sim_idx]
            color = ac.AGENT_COLORS[agent_name]
            ax.plot(data["time"], data["pool_price"], "k-", linewidth=1.5, label="Pool Price")
            ax.plot(
                data["time"],
                data["midprice"],
                color="gray",
                linewidth=1,
                alpha=0.7,
                label="Midprice",
            )
            ax.fill_between(
                data["time"],
                data["position_lower_price"],
                data["position_upper_price"],
                alpha=0.25,
                color=color,
                label=f"{agent_name} Range",
            )
            ax.plot(
                data["time"],
                data["position_lower_price"],
                "--",
                color=color,
                alpha=0.5,
                linewidth=0.8,
            )
            ax.plot(
                data["time"],
                data["position_upper_price"],
                "--",
                color=color,
                alpha=0.5,
                linewidth=0.8,
            )

            has_hold_flag = not np.all(data["hold_flag"] == -1.0)
            if has_hold_flag:
                rebalance_mask = data["hold_flag"] <= 0
                rebalance_times = data["time"][rebalance_mask]
                for rebalance_time in rebalance_times:
                    ax.axvline(rebalance_time, color="red", alpha=0.15, linewidth=0.5)
                if len(rebalance_times) > 0:
                    ax.axvline(
                        rebalance_times[0],
                        color="red",
                        alpha=0.3,
                        linewidth=0.5,
                        label="Rebalance",
                    )

            ax.set_xlabel("Time", fontsize=16)
            ax.set_ylabel("Price", fontsize=16)
            ax.tick_params(axis="both", labelsize=14)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            ax.set_title(agent_name)
            ax.legend(fontsize=14, loc="lower right")
            ax.grid(True, alpha=0.3)

        fig_i.tight_layout()
        path_i = os.path.join(
            FIGURES_DIR,
            f"liquidity_depth_price_evolution_sim{sim_idx + 1}_{job_id}.png",
        )
        fig_i.savefig(path_i, dpi=150, bbox_inches="tight")
        plt.close(fig_i)
        print(f"  Saved: {path_i}")


def main():
    torch.manual_seed(ac.SEED)
    np.random.seed(ac.SEED)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 72)
    print("Baseline comparison under one-tick vs liquidity-depth price impact")
    print("=" * 72)
    print(f"Agents: {', '.join(BASELINE_AGENT_NAMES)}")
    print(f"Eval trajectories: {ac.NUM_TRAJECTORIES_EVAL}")
    print(f"Single sims per agent/model: {ac.NUM_SINGLE_SIMS}")
    print(
        "LiquidityDepth config: "
        f"trade_notional={LIQUIDITY_DEPTH_TRADE_NOTIONAL}, "
        f"depth_window={LIQUIDITY_DEPTH_WINDOW}"
    )

    all_results = {}
    all_single_data = {}
    for model_label, price_impact_kind in PRICE_IMPACT_MODELS.items():
        print(f"\nEvaluating {model_label}")
        all_results[model_label] = evaluate_model_agents(model_label, price_impact_kind)
        all_single_data[model_label] = collect_model_single_data(
            model_label,
            price_impact_kind,
        )

    print_pnl_summary(all_results)

    print("\nGenerating comparison plots")
    plot_grouped_pnl_boxplot(all_results)
    plot_single_pnl_evolution(all_single_data)
    plot_liquidity_depth_price_evolution(all_single_data["LiquidityDepth"])
    print("\nDone. Comparison figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
