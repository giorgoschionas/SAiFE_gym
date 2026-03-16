#!/usr/bin/env python3
"""
uniform_agent_sensitivity.py

Sensitivity analysis for the UniformAllocationAgent operating at maximum tau
(tau_max = 2000 ticks, spanning ±20% from the current price at any step).
This makes it analogous to a Uniswap v2 LP: always in range, full capital
deployed, no strategic narrowing.

Two studies:

  Study 1 – Orderflow Toxicity (alpha3 sweep)
    alpha3 in PoissonLinearArrivalModel controls how aggressively informed
    traders exploit AMM mispricings.  Formula:
        intensity_sell = max(α₀,  α₁ + α₂*L  −  α₃*(S−Z))
        intensity_buy  = max(α₀,  α₁ + α₂*L  +  α₃*(S−Z))
    where S = external price, Z = AMM price.
    When S > Z the AMM is cheap: informed traders buy heavily (α₃*(S−Z) more
    buys than sells) until the AMM price catches up.  This tracking is what
    creates Impermanent Loss for the LP.

  Study 2 – External price volatility (vol sweep)
    Higher vol → larger mispricings → more arbitrage pressure → amplified IL.
    We run both with α₃=0 (noise traders only) and α₃=500 (moderate toxicity)
    to separate the two effects.

PnL decomposition
-----------------
    PnL  =  fee_income  −  Impermanent Loss
  where:
    fee_income ≈ LP_COLLECTED_FEES0 * final_price + LP_COLLECTED_FEES1
                 (cumulative fees collected at each rebalance, converted to
                  token-1 units at final external price; reinvested fees suffer
                  their own IL, so this slightly overstates raw fee income)
    IL_estimate = fee_income − total_PnL
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from experiments.helpers import get_amm_env, INITIAL_WEALTH
from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent
from SAiFE_gym.gym.index_names import (
    LP_COLLECTED_FEES0_KEY,
    LP_COLLECTED_FEES1_KEY,
    ASSET_PRICE_KEY,
)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

# Large tau → LP always covers ±2000 ticks (~±20%) centered on current price.
# This replicates Uniswap v2 "full-range" behaviour: capital is always in range
# and fees are earned on every swap.
TAU_MAX = 2000

N_EPISODES    = 10    # Independent episodes per configuration
NUM_TRAJ      = 200   # Parallel trajectories per episode  →  2 000 samples/config
N_STEPS       = 200   # Steps per episode
TERMINAL_TIME = 1.0
BASE_SEED     = 42

# ── Study 1: Toxicity sweep ────────────────────────────────────────────────
ALPHA3_VALUES = [0, 50, 100, 200, 500, 1000, 2000, 5000]
FIXED_VOL = 2.0

# ── Study 2: Volatility sweep ──────────────────────────────────────────────
VOL_VALUES   = [0.5, 1.0, 2.0, 4.0, 8.0]
ALPHA3_NONE  = 0.0
ALPHA3_TOXIC = 500.0   # moderate toxicity  (compare to the no-toxicity case)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


# ─────────────────────────────────────────────────────────────────────────────
# Core episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_config(tau, volatility, alpha3, n_episodes, num_traj, seed):
    """Run one (volatility, alpha3) configuration and return metrics.

    Returns
    -------
    pnl   : (n_episodes * num_traj,) float  –  cumulative ΔWealth per trajectory
    fees  : same shape  –  total fee income in token-1 equivalent
    il    : same shape  –  estimated Impermanent Loss  =  fees − pnl
    """
    env = get_amm_env(
        num_trajectories=num_traj,
        tau=tau,
        volatility=volatility,
        alpha3=alpha3,
        n_steps=N_STEPS,
        terminal_time=TERMINAL_TIME,
        seed=seed,
    )
    agent = UniformAllocationAgent(env)

    all_pnl, all_fees = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        cum_reward = np.zeros(num_traj)

        for _ in range(N_STEPS):
            action = agent.get_action(obs)
            obs, rewards, _, _, _ = env.step(action)
            cum_reward += rewards

        # Cumulative fee income (token-1 equivalent) at final external price.
        # LP_COLLECTED_FEES0/1 accumulate across all rebalances in this episode.
        final_price = obs[ASSET_PRICE_KEY]
        fees0       = obs[LP_COLLECTED_FEES0_KEY]
        fees1       = obs[LP_COLLECTED_FEES1_KEY]
        fee_value   = fees0 * final_price + fees1    # token-1 value of all fees

        all_pnl.append(cum_reward)
        all_fees.append(fee_value)

    pnl  = np.concatenate(all_pnl)
    fees = np.concatenate(all_fees)
    il   = fees - pnl   # IL estimate: what fees earned minus what net PnL is
    return pnl, fees, il


# ─────────────────────────────────────────────────────────────────────────────
# Study runners
# ─────────────────────────────────────────────────────────────────────────────

def study_toxicity():
    """Sweep alpha3; fix volatility = FIXED_VOL."""
    print(f"\n{'='*60}")
    print(f"  Study 1: Toxicity sweep  (vol={FIXED_VOL}, τ={TAU_MAX})")
    print(f"  {N_EPISODES} episodes × {NUM_TRAJ} trajectories per config")
    print(f"{'='*60}")
    results = {}
    n = len(ALPHA3_VALUES)
    for i, alpha3 in enumerate(ALPHA3_VALUES, 1):
        print(f"  [{i}/{n}] alpha3={alpha3:.0f} ...", end=" ", flush=True)
        pnl, fees, il = run_config(
            TAU_MAX, FIXED_VOL, alpha3, N_EPISODES, NUM_TRAJ, BASE_SEED
        )
        results[alpha3] = dict(pnl=pnl, fees=fees, il=il)
        print(f"mean ΔW={pnl.mean():+.0f}  fees={fees.mean():.0f}  IL={il.mean():.0f}")
    return results


def study_volatility():
    """Sweep volatility; run both alpha3=0 and alpha3=ALPHA3_TOXIC."""
    print(f"\n{'='*60}")
    print(f"  Study 2: Volatility sweep  (τ={TAU_MAX})")
    print(f"  {N_EPISODES} episodes × {NUM_TRAJ} trajectories per config")
    print(f"{'='*60}")
    results = {}
    configs = [(v, a) for v in VOL_VALUES for a in [ALPHA3_NONE, ALPHA3_TOXIC]]
    n = len(configs)
    for i, (vol, alpha3) in enumerate(configs, 1):
        print(f"  [{i}/{n}] vol={vol}, alpha3={alpha3:.0f} ...", end=" ", flush=True)
        pnl, fees, il = run_config(
            TAU_MAX, vol, alpha3, N_EPISODES, NUM_TRAJ, BASE_SEED
        )
        results[(vol, alpha3)] = dict(pnl=pnl, fees=fees, il=il)
        print(f"mean ΔW={pnl.mean():+.0f}  fees={fees.mean():.0f}  IL={il.mean():.0f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(x):
    """Format as +/- integer with thousands separator."""
    return f"{x:>+12,.0f}"


def print_toxicity_table(results):
    header = (
        f"\n  {'alpha3':>8}  {'mean ΔWealth':>14}  {'std ΔW':>12}  "
        f"{'mean Fees':>12}  {'mean IL':>12}  {'win%':>6}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(f"\n{'='*60}")
    print(f"  Toxicity sweep results  (vol={FIXED_VOL}, τ={TAU_MAX})")
    print(sep)
    print(header)
    print(sep)
    for a in ALPHA3_VALUES:
        r = results[a]
        win_pct = 100.0 * np.mean(r["pnl"] > 0)
        print(
            f"  {a:>8.0f}  {_fmt(r['pnl'].mean())}  {r['pnl'].std():>12,.0f}  "
            f"{r['fees'].mean():>12,.0f}  {r['il'].mean():>12,.0f}  {win_pct:>5.1f}%"
        )
    print(sep)


def print_volatility_table(results):
    header = (
        f"\n  {'vol':>5}  {'alpha3':>8}  {'mean ΔWealth':>14}  {'std ΔW':>12}  "
        f"{'mean Fees':>12}  {'mean IL':>12}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(f"\n{'='*60}")
    print(f"  Volatility sweep results  (τ={TAU_MAX})")
    print(sep)
    print(header)
    print(sep)
    for vol in VOL_VALUES:
        for alpha3 in [ALPHA3_NONE, ALPHA3_TOXIC]:
            r = results[(vol, alpha3)]
            print(
                f"  {vol:>5.1f}  {alpha3:>8.0f}  {_fmt(r['pnl'].mean())}  "
                f"{r['pnl'].std():>12,.0f}  {r['fees'].mean():>12,.0f}  {r['il'].mean():>12,.0f}"
            )
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _add_zero_line(ax):
    ax.axhline(0, color="0.5", linestyle="--", linewidth=0.9, zorder=0)


def _k_fmt(x, _pos=None):
    """Format large numbers as k (thousands)."""
    return f"{x/1e3:.0f}k" if abs(x) >= 1000 else f"{x:.0f}"


def plot_toxicity(results, save_dir=FIGURES_DIR):
    """Three-panel figure for Study 1."""
    os.makedirs(save_dir, exist_ok=True)
    alphas = ALPHA3_VALUES

    mean_pnl  = np.array([results[a]["pnl"].mean()  for a in alphas])
    std_pnl   = np.array([results[a]["pnl"].std()   for a in alphas])
    mean_fees = np.array([results[a]["fees"].mean()  for a in alphas])
    mean_il   = np.array([results[a]["il"].mean()    for a in alphas])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"UniformAllocationAgent (τ={TAU_MAX} ≈ Uniswap v2)  –  Effect of Orderflow Toxicity (α₃)\n"
        f"vol={FIXED_VOL},  {N_EPISODES} episodes × {NUM_TRAJ} trajectories",
        fontsize=11,
    )

    # ── Panel 1: Mean ΔWealth vs alpha3 ──────────────────────────────────
    ax = axes[0]
    ax.plot(alphas, mean_pnl, "o-", color="#1f77b4", lw=2, ms=7, label="Mean ΔWealth")
    ax.fill_between(
        alphas, mean_pnl - std_pnl, mean_pnl + std_pnl,
        alpha=0.18, color="#1f77b4", label="±1 std",
    )
    _add_zero_line(ax)
    ax.set_xlabel("α₃  (orderflow toxicity)", fontsize=10)
    ax.set_ylabel("ΔWealth  (token-1 units)", fontsize=9)
    ax.set_title("Net LP Profit vs Toxicity", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k_fmt))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("symlog", linthresh=1000)

    # ── Panel 2: Fee income and IL decomposition ──────────────────────────
    ax = axes[1]
    ax.plot(alphas, mean_fees, "s-",  color="#2ca02c", lw=2, ms=7, label="Fee income")
    ax.plot(alphas, mean_il,   "^--", color="#d62728", lw=2, ms=7, label="Impermanent Loss")
    _add_zero_line(ax)
    ax.set_xlabel("α₃  (orderflow toxicity)", fontsize=10)
    ax.set_ylabel("Token-1 units", fontsize=9)
    ax.set_title("Fee Income vs Impermanent Loss", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k_fmt))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Box plots at selected toxicities ─────────────────────────
    ax = axes[2]
    selected = [0, 200, 1000, 5000]
    box_data  = [results[a]["pnl"] for a in selected]
    bp = ax.boxplot(
        box_data,
        tick_labels=[str(a) for a in selected],
        patch_artist=True,
        notch=False,
        showfliers=False,
    )
    box_colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    _add_zero_line(ax)
    ax.set_xlabel("α₃  (orderflow toxicity)", fontsize=10)
    ax.set_ylabel("ΔWealth  (token-1 units)", fontsize=9)
    ax.set_title("ΔWealth Distribution (no outliers)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k_fmt))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "uniform_toxicity_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close(fig)


def plot_volatility(results, save_dir=FIGURES_DIR):
    """Three-panel figure for Study 2."""
    os.makedirs(save_dir, exist_ok=True)
    vols = VOL_VALUES

    styles = {
        ALPHA3_NONE:  ("o-",  "#1f77b4", f"α₃={ALPHA3_NONE:.0f}  (no toxicity)"),
        ALPHA3_TOXIC: ("s--", "#d62728", f"α₃={ALPHA3_TOXIC:.0f}  (moderate toxicity)"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"UniformAllocationAgent (τ={TAU_MAX} ≈ Uniswap v2)  –  Effect of External Price Volatility\n"
        f"{N_EPISODES} episodes × {NUM_TRAJ} trajectories",
        fontsize=11,
    )

    for alpha3, (ls, color, label) in styles.items():
        mean_pnl  = np.array([results[(v, alpha3)]["pnl"].mean()  for v in vols])
        std_pnl   = np.array([results[(v, alpha3)]["pnl"].std()   for v in vols])
        mean_fees = np.array([results[(v, alpha3)]["fees"].mean()  for v in vols])
        mean_il   = np.array([results[(v, alpha3)]["il"].mean()    for v in vols])

        # Panel 1 – Net PnL
        ax = axes[0]
        ax.plot(vols, mean_pnl, ls, color=color, lw=2, ms=7, label=label)
        ax.fill_between(vols, mean_pnl - std_pnl, mean_pnl + std_pnl,
                        alpha=0.12, color=color)

        # Panel 2 – Fee income
        ax = axes[1]
        ax.plot(vols, mean_fees, ls, color=color, lw=2, ms=7, label=label)

        # Panel 3 – IL estimate
        ax = axes[2]
        ax.plot(vols, mean_il, ls, color=color, lw=2, ms=7, label=label)

    titles  = ["Net LP Profit vs Volatility",
               "Fee Income vs Volatility",
               "Impermanent Loss vs Volatility"]
    ylabels = ["ΔWealth  (token-1)", "Fee income  (token-1)", "IL estimate  (token-1)"]

    for ax, title, ylabel in zip(axes, titles, ylabels):
        _add_zero_line(ax)
        ax.set_xlabel("Volatility  σ", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k_fmt))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "uniform_volatility_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tox_results = study_toxicity()
    print_toxicity_table(tox_results)
    plot_toxicity(tox_results)

    vol_results = study_volatility()
    print_volatility_table(vol_results)
    plot_volatility(vol_results)

    print("\nDone.")
