"""
Calibration probe for LiquidityKernelArrivalModel coefficients (ALPHA2, ALPHA3).

Runs an episode under the training config with a UniformAllocationAgent LP and
LiquidityKernelArrivalModel arrival flow with ALPHA2=ALPHA3=0 (baseline only).
Logs the empirical distribution of:
    - kernel-weighted directional liquidity:  weighted_liq_sell, weighted_liq_buy
    - mispricing:                              S - Z = midprice - sqrt_price**2
and recommends ALPHA2, ALPHA3 sized so that each term contributes on the order
of ALPHA1 (the baseline intensity) on average — neither dominating nor dominated.

Caveat: with ALPHA3=0 during the probe, the AMM does not get arbitraged toward
the external midprice, so observed |mispricing| is an upper bound. Once you set
a non-zero ALPHA3, arbitrage flow will dampen mispricing — re-run to refine.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    MISPRICING_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
)
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel

# Match train_policy_gradient_agent.py
SEED              = 42
TERMINAL_TIME     = 1.0
N_STEPS           = 300
NUM_TRAJECTORIES  = 200
INITIAL_PRICE     = 200.0
DRIFT             = 1.0
VOLATILITY        = 2.0
FEE_TIER          = 0.003
TAU               = 50
INITIAL_WEALTH    = 1000.0

# Baseline intensity that we want each extra term to be comparable to
ALPHA0 = np.array([10.0,  10.0])
ALPHA1 = np.array([150.0, 150.0])

# Kernel hyperparameters (LiquidityKernelArrivalModel defaults)
BETA            = 0.5
K               = 10
LIQUIDITY_SCALE = 1e6


def build_env() -> AMMEnvironment:
    step_size = TERMINAL_TIME / N_STEPS
    midprice_model = BrownianMotionMidpriceModel(
        drift=DRIFT, volatility=VOLATILITY, initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME, step_size=step_size,
        num_trajectories=NUM_TRAJECTORIES, seed=SEED,
    )
    arrival_model = LiquidityKernelArrivalModel(
        alpha=np.stack([ALPHA0, ALPHA1, np.zeros(2), np.zeros(2)]),
        beta=BETA, K=K, liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size, num_trajectories=NUM_TRAJECTORIES, seed=SEED + 1,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model, arrival_model=arrival_model,
        num_trajectories=NUM_TRAJECTORIES, fee_tier=FEE_TIER, tau=TAU, seed=SEED + 2,
    )
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS,
        model_dynamics=model_dynamics, reward_function=PnL(),
        initial_wealth=INITIAL_WEALTH, num_trajectories=NUM_TRAJECTORIES, seed=SEED,
    )


def kernel_weighted_liquidity(
    liquidity_array: np.ndarray,    # (N, num_ticks)
    current_tick: np.ndarray,       # (N,)
    tick_lower_global: int,
    kernel_weights: np.ndarray,     # (K,)
    liquidity_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicates LiquidityKernelArrivalModel.update() kernel computation."""
    N, num_ticks = liquidity_array.shape
    K = kernel_weights.shape[0]
    cur_idx = (current_tick - tick_lower_global).astype(np.int64)
    offsets = np.arange(1, K + 1)
    sell_idx = cur_idx[:, None] - offsets[None, :]
    buy_idx  = cur_idx[:, None] + offsets[None, :]
    sell_valid = (sell_idx >= 0) & (sell_idx < num_ticks)
    buy_valid  = (buy_idx  >= 0) & (buy_idx  < num_ticks)
    sell_safe = np.clip(sell_idx, 0, num_ticks - 1)
    buy_safe  = np.clip(buy_idx,  0, num_ticks - 1)
    traj = np.arange(N)[:, None]
    sell_liq = liquidity_array[traj, sell_safe] * sell_valid
    buy_liq  = liquidity_array[traj, buy_safe]  * buy_valid
    return (sell_liq @ kernel_weights / liquidity_scale,
            buy_liq  @ kernel_weights / liquidity_scale)


def percentiles_str(arr: np.ndarray, pcts=(5, 25, 50, 75, 95)) -> str:
    return "  ".join(f"p{p:>2}={v:+.4f}" for p, v in zip(pcts, np.percentile(arr, pcts)))


# ---------------------------------------------------------------------------
# Analytical kernel-geometry analysis (no simulation needed)
# ---------------------------------------------------------------------------
#
# For a uniform LP position [-w, +w] with fixed total liquidity W and no
# background LPs, per-tick liquidity is L = W / (2w + 1). The kernel-weighted
# directional liquidity (one side, since both are symmetric for uniform LP) is:
#
#     weighted_liq(w) = L · S(min(w, K), β) = W · S(min(w, K), β) / (2w + 1)
#
# where S(n, β) = Σ_{d=1}^{n} exp(-β·d).
#
# The width w* that maximizes weighted_liq(w) is the geometric incentive the
# kernel gives the agent. Use this to pick (β, K) so that w* matches the LP
# width you want to see emerge.

def kernel_optimal_width(beta: float, K: int, max_w: int = 100) -> tuple[int, float]:
    """Find w* that maximizes weighted_liq(w) / W. Returns (w_star, peak_value)."""
    widths = np.arange(1, max_w + 1)
    weights = np.exp(-beta * np.arange(1, K + 1))
    S = np.array([weights[:min(w, K)].sum() for w in widths])
    normalized = S / (2 * widths + 1)
    idx = int(np.argmax(normalized))
    return int(widths[idx]), float(normalized[idx])


def kernel_geometry_table(beta_values, K_values, max_w: int = 100):
    print("Optimal LP width w* by (β, K) — uniform [-w, +w], fixed total liquidity")
    print(f"(scanning w ∈ [1, {max_w}], maximizing weighted_liq(w) = W·S/(2w+1))\n")
    col_w = 10
    label = "β \\ K"
    print(f"{label:<8}", end="")
    for K in K_values:
        print(f"{f'K={K}':<{col_w}}", end="")
    print()
    print("-" * (8 + col_w * len(K_values)))
    for beta in beta_values:
        print(f"{f'β={beta}':<8}", end="")
        for K in K_values:
            w, peak = kernel_optimal_width(beta, K, max_w=max_w)
            print(f"{f'{w} ({peak:.3f})':<{col_w}}", end="")
        print()
    print("\nCell format: w*  (weighted_liq/W at w*).")
    print("A higher peak value means the kernel pumps arrival intensity more strongly.\n")


def kernel_curve(beta: float, K: int,
                 widths_to_show=(1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100)):
    print(f"weighted_liq(w) / W for β={beta}, K={K}:")
    weights = np.exp(-beta * np.arange(1, K + 1))
    print(f"  {'w':<5}{'span':<8}{'S(min(w,K),β)':<18}{'weighted_liq/W':<16}")
    for w in widths_to_show:
        S = weights[:min(w, K)].sum()
        wl = S / (2 * w + 1)
        marker = "  ← peak" if w == kernel_optimal_width(beta, K, max_w=max(widths_to_show))[0] else ""
        print(f"  {w:<5}{2*w+1:<8}{S:<18.4f}{wl:<16.4f}{marker}")
    print()


def widths_for_target_betas(target_widths, K_values, beta_grid=None):
    """For each (target_width, K), find the β value whose w* is closest to target."""
    if beta_grid is None:
        beta_grid = np.concatenate([
            np.linspace(0.001, 0.1, 50),
            np.linspace(0.1, 1.0, 50)[1:],
        ])
    print("Suggested β to make w* ≈ target_width (no background LPs)\n")
    col_w = 22
    print(f"{'target w*':<12}", end="")
    for K in K_values:
        print(f"{f'K={K}':<{col_w}}", end="")
    print()
    print("-" * (12 + col_w * len(K_values)))
    for target in target_widths:
        print(f"{target:<12}", end="")
        for K in K_values:
            max_w = max(target * 3, 100)
            best_beta, best_diff = None, float("inf")
            for beta in beta_grid:
                w, _ = kernel_optimal_width(beta, K, max_w=max_w)
                diff = abs(w - target)
                if diff < best_diff:
                    best_diff, best_beta = diff, beta
            achieved = kernel_optimal_width(best_beta, K, max_w=max_w)[0]
            cell = f"β≈{best_beta:.3f} → w*={achieved}"
            print(f"{cell:<{col_w}}", end="")
        print()
    print()


def main():
    env = build_env()
    agent = UniformAllocationAgent(env)
    kernel_weights = np.exp(-BETA * np.arange(1, K + 1))
    tick_lower_global = env.model_dynamics.tick_lower_global

    print(f"Probe: {NUM_TRAJECTORIES} trajectories x {N_STEPS} steps "
          f"(probe sets ALPHA2=ALPHA3=0)")
    print(f"Kernel: beta={BETA}, K={K}, sum(weights)={kernel_weights.sum():.4f}")
    print(f"liquidity_scale={LIQUIDITY_SCALE:.0e}, baseline ALPHA1={ALPHA1.tolist()}\n")

    obs, _ = env.reset()
    if tick_lower_global is None:
        tick_lower_global = env.model_dynamics.tick_lower_global

    w_sell_log, w_buy_log, mispricing_log = [], [], []
    for _ in range(N_STEPS):
        action = agent.get_action(obs)
        obs, *_ = env.step(action)

        ws, wb = kernel_weighted_liquidity(
            obs[POOL_LIQUIDITY_ARRAY_KEY],
            obs[POOL_CURRENT_TICK_KEY],
            tick_lower_global,
            kernel_weights,
            LIQUIDITY_SCALE,
        )
        w_sell_log.append(ws)
        w_buy_log.append(wb)
        if MISPRICING_KEY in obs:
            mispricing_log.append(obs[MISPRICING_KEY])
        else:
            mispricing_log.append(obs[ASSET_PRICE_KEY] - obs[POOL_SQRT_PRICE_KEY] ** 2)

    weighted_dir = np.concatenate([np.concatenate(w_sell_log),
                                   np.concatenate(w_buy_log)])
    mispricing   = np.concatenate(mispricing_log)
    abs_mp       = np.abs(mispricing)

    print("=== Kernel-weighted directional liquidity (normalized) ===")
    print(f"  mean={weighted_dir.mean():.4f}  std={weighted_dir.std():.4f}")
    print(f"  {percentiles_str(weighted_dir)}\n")

    print("=== Mispricing (S - Z) ===")
    print(f"  mean={mispricing.mean():+.4f}  std={mispricing.std():.4f}  "
          f"abs_mean={abs_mp.mean():.4f}")
    print(f"  {percentiles_str(mispricing)}\n")

    target = ALPHA1.mean()
    mean_w = weighted_dir.mean()
    std_w  = weighted_dir.std()
    mean_m = abs_mp.mean()
    std_m  = mispricing.std()

    print("=== Suggested coefficients ===")
    print(f"(target = ALPHA1 = {target:.0f})\n")
    print(f"{'':<8}  {'mean-match (term ≈ ALPHA1 on avg)':<38}  "
          f"{'std-match (term modulates ALPHA1)':<38}")
    if mean_w > 0:
        print(f"{'ALPHA2':<8}  {target / mean_w:>10.1f}  "
              f"({target:.0f}/{mean_w:.4f})           "
              f"{(target / std_w) if std_w > 0 else float('nan'):>10.1f}  "
              f"({target:.0f}/{std_w:.4f})")
    else:
        print(f"{'ALPHA2':<8}  weighted_liq is ~0 — LP not deploying near current tick")
    if mean_m > 0:
        print(f"{'ALPHA3':<8}  {target / mean_m:>10.1f}  "
              f"({target:.0f}/{mean_m:.4f})           "
              f"{(target / std_m) if std_m > 0 else float('nan'):>10.1f}  "
              f"({target:.0f}/{std_m:.4f})")

    print()
    print("Mean-match: term contributes ~ALPHA1 on average (shifts overall rate).")
    print("Std-match:  term's *variability* matches ALPHA1 (modulates around baseline).")
    print(f"\nReference: train_policy_gradient_agent.py uses ALPHA2=[0,0], ALPHA3=[5000,5000].")

    print("\n" + "=" * 72)
    print("KERNEL GEOMETRY: which (β, K) reward which LP widths?")
    print("=" * 72 + "\n")
    print("This part is purely analytical (no simulation). It assumes a uniform")
    print("[-w, +w] LP position with fixed total liquidity and no background LPs.")
    print("It tells you which width the kernel *geometrically* incentivizes —")
    print("independent of mispricing, fees, or rebalancing costs.\n")

    kernel_geometry_table(
        beta_values=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        K_values=[5, 10, 20, 50],
        max_w=100,
    )

    print(f"Curve at current defaults (β={BETA}, K={K}):")
    kernel_curve(BETA, K)

    widths_for_target_betas(
        target_widths=[2, 5, 10, 20, 50],
        K_values=[10, 20, 50],
    )

    print("Reading the tables:")
    print("  - Larger β (faster decay) → narrower w*.")
    print("  - With β=0.5, K=10 (current defaults), w* ≈ 1 → kernel still rewards")
    print("    near-single-tick concentration.")
    print("  - To make the agent prefer width ≈ N, pick (β, K) from the second table.")
    print("  - These are *necessary* but not sufficient: gas_cost, lower TAU, and")
    print("    width-dependent ALPHA3 add further pressure against narrow positions.")


if __name__ == "__main__":
    main()
