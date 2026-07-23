#!/usr/bin/env python3
"""Compare arrival-driven and explicit-arbitrage pool price tracking."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
job_id = os.environ.get("SLURM_JOB_ID", "local")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/saife_gym_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from SAiFE_gym.agents.BaselineAgents import (
    ArbitrageurAgent,
    DeployOnceAgent,
    SpeedControlArbitrageurAgent,
)
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    POOL_SQRT_PRICE_KEY,
    TIME_KEY,
)
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import OrnsteinUhlenbeckMidpriceModel


SEED = 6
TERMINAL_TIME = 1.0
N_STEPS = 1000
INITIAL_WEALTH = 1000
TAU = 20
LIQUIDITY_SCALE = 1e5

INITIAL_PRICE = 100.0
INITIAL_POOL_PRICE = None
VOLATILITY = 0.009
FEE_TIER = 0.0005
EXP_VALUE = 1.0001

ALPHA0 = np.array([1.0, 1.0])
ALPHA1 = np.array([20.0, 20.0])
ALPHA2 = np.array([0.0, 0.0])
ALPHA3_HIGH = np.array([20000.0, 20000.0])
ALPHA3_ZERO = np.array([0.0, 0.0])

DEPLOYONCE_LOWER = -10
DEPLOYONCE_UPPER = 10
MAX_ARBITRAGE_TICKS = 25
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

SCENARIOS = (
    ("HighAlpha3Arrivals", ALPHA3_HIGH, None),
    ("NoiseOnly", ALPHA3_ZERO, None),
    ("NoisePlusSizeArb", ALPHA3_ZERO, "size"),
    ("NoisePlusSpeedArb", ALPHA3_ZERO, "speed"),
)


def build_env(
    alpha3: np.ndarray,
    num_trajectories: int = 1,
    n_steps: int = N_STEPS,
    seed: int = SEED,
) -> AMMEnvironment:
    step_size = TERMINAL_TIME / n_steps
    alpha = np.array([ALPHA0, ALPHA1, ALPHA2, alpha3], dtype=np.float64)
    midprice_model = OrnsteinUhlenbeckMidpriceModel(
        mean_reversion=0.4,
        long_term_mean=INITIAL_PRICE,
        volatility=VOLATILITY,
        initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed,
    )
    arrival_model = LiquidityKernelArrivalModel(
        alpha=alpha,
        beta=0.1,
        K=20,
        liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1 if seed is not None else None,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=FEE_TIER,
        tau=TAU,
        num_ticks=5000,
        exponential_value=EXP_VALUE,
        seed=seed + 2 if seed is not None else None,
    )
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME,
        n_steps=n_steps,
        initial_wealth=INITIAL_WEALTH,
        reward_function=PnL(),
        model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        initial_pool_price=INITIAL_POOL_PRICE,
        seed=seed,
    )


def _make_arbitrageur(env: AMMEnvironment, kind: str):
    if kind == "size":
        return ArbitrageurAgent(env, max_ticks_per_trade=MAX_ARBITRAGE_TICKS)
    if kind == "speed":
        return SpeedControlArbitrageurAgent(env, max_ticks_per_step=MAX_ARBITRAGE_TICKS)
    return None


def run_scenario(
    name: str,
    alpha3: np.ndarray,
    arbitrageur_kind: str = None,
    n_steps: int = N_STEPS,
    seed: int = SEED,
) -> dict:
    env = build_env(alpha3=alpha3, n_steps=n_steps, seed=seed)
    lp_agent = DeployOnceAgent(
        env,
        lower_offset=DEPLOYONCE_LOWER,
        upper_offset=DEPLOYONCE_UPPER,
    )
    arbitrageur = _make_arbitrageur(env, arbitrageur_kind)

    state, _ = env.reset()
    data = {k: [] for k in [
        "time",
        "pool_price",
        "midprice",
        "position_lower_price",
        "position_upper_price",
        "mispricing",
        "arbitrage_action",
        "arbitrage_order",
        "unfilled_input",
        "tick_movement",
    ]}

    for _ in range(n_steps):
        lp_action = lp_agent.get_action(state)
        state, _, terminated, _, _ = env.step(lp_action)

        arb_action = np.zeros((env.num_trajectories, 1), dtype=np.float64)
        arb_info = {
            "order_input": np.zeros(env.num_trajectories, dtype=np.float64),
            "unfilled_input": np.zeros(env.num_trajectories, dtype=np.float64),
            "tick_movement": np.zeros(env.num_trajectories, dtype=np.int64),
        }
        if arbitrageur_kind == "size":
            arb_action = arbitrageur.get_action(state)
            arb_info = env.model_dynamics.execute_liquidity_taker_orders(arb_action)
            env._compute_derived_obs()
            state = env.state
        elif arbitrageur_kind == "speed":
            arb_action = arbitrageur.get_action(state)
            arb_info = env.model_dynamics.execute_liquidity_taker_speeds(arb_action)
            env._compute_derived_obs()
            state = env.state

        pool_price = state[POOL_SQRT_PRICE_KEY] ** 2
        midprice = state[ASSET_PRICE_KEY]
        data["time"].append(state[TIME_KEY].copy())
        data["pool_price"].append(pool_price.copy())
        data["midprice"].append(midprice.copy())
        data["position_lower_price"].append(EXP_VALUE ** state[LP_TICK_LOWER_KEY].copy())
        data["position_upper_price"].append(EXP_VALUE ** state[LP_TICK_UPPER_KEY].copy())
        data["mispricing"].append((midprice - pool_price).copy())
        data["arbitrage_action"].append(arb_action[:, 0].copy())
        data["arbitrage_order"].append(arb_info["order_input"].copy())
        data["unfilled_input"].append(arb_info["unfilled_input"].copy())
        data["tick_movement"].append(arb_info["tick_movement"].copy())

        if bool(terminated[0]):
            break

    out = {key: np.asarray(value) for key, value in data.items()}
    out["name"] = name
    out["arbitrageur_kind"] = arbitrageur_kind or "none"
    return out


def summarize_tracking(data: dict) -> dict:
    mispricing = np.asarray(data["mispricing"], dtype=np.float64)
    arb_action = np.asarray(data["arbitrage_action"], dtype=np.float64)
    tick_movement = np.asarray(data["tick_movement"], dtype=np.float64)
    return {
        "mean_abs_mispricing": float(np.mean(np.abs(mispricing))),
        "max_abs_mispricing": float(np.max(np.abs(mispricing))),
        "total_abs_arbitrage_action": float(np.sum(np.abs(arb_action))),
        "total_abs_tick_movement": int(np.sum(np.abs(tick_movement))),
    }


def run_experiment(
    n_steps: int = N_STEPS,
    seed: int = SEED,
    save_plots: bool = True,
    figures_dir: str = FIGURES_DIR,
) -> dict:
    results = {
        name: run_scenario(name, alpha3, kind, n_steps=n_steps, seed=seed)
        for name, alpha3, kind in SCENARIOS
    }
    summaries = {name: summarize_tracking(data) for name, data in results.items()}
    if save_plots:
        os.makedirs(figures_dir, exist_ok=True)
        plot_price_evolution(results, figures_dir=figures_dir)
    return {"results": results, "summaries": summaries}


def plot_price_evolution(results: dict, figures_dir: str = FIGURES_DIR) -> str:
    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9), sharex=True, sharey=True)
    colors = {
        "HighAlpha3Arrivals": "#1f77b4",
        "NoiseOnly": "#7f7f7f",
        "NoisePlusSizeArb": "#2ca02c",
        "NoisePlusSpeedArb": "#d62728",
    }
    for ax, (name, data) in zip(axes.flat, results.items()):
        time = data["time"][:, 0]
        pool_price = data["pool_price"][:, 0]
        midprice = data["midprice"][:, 0]
        lower = data["position_lower_price"][:, 0]
        upper = data["position_upper_price"][:, 0]
        color = colors.get(name, "#1f77b4")

        ax.plot(time, pool_price, color="black", linewidth=1.2, label="Pool Price")
        ax.plot(time, midprice, color="gray", linewidth=1.0, label="Midprice")
        ax.fill_between(time, lower, upper, color=color, alpha=0.18, label="LP Range")
        ax.plot(time, lower, "--", color=color, alpha=0.6, linewidth=0.8)
        ax.plot(time, upper, "--", color=color, alpha=0.6, linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    path = os.path.join(figures_dir, f"arbitrageur_price_evolution_{job_id}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    output = run_experiment(save_plots=True)
    for name, summary in output["summaries"].items():
        print(
            f"{name}: "
            f"mean|S-Z|={summary['mean_abs_mispricing']:.8f}, "
            f"max|S-Z|={summary['max_abs_mispricing']:.8f}, "
            f"total|arb_action|={summary['total_abs_arbitrage_action']:.6f}, "
            f"total|tick_move|={summary['total_abs_tick_movement']}"
        )
    print(f"Saved: {os.path.join(FIGURES_DIR, f'arbitrageur_price_evolution_{job_id}.png')}")


if __name__ == "__main__":
    main()
