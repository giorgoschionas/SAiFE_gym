#!/usr/bin/env python3
"""
gas_cost_impact.py

Study how different gas_cost levels eat into the PnL of UniformAllocationAgent.

UniformAllocationAgent rebalances EVERY step (200 rebalances per episode).
With gas_cost > 0 this leaks wealth on every step even when the position
doesn't meaningfully change.

Metrics per config:
  - mean / std of cumulative ΔWealth
  - mean fee income (from LP_COLLECTED_FEES0/1)
  - mean gas drag  =  gas_cost × n_steps  (deterministic lower bound)
  - % trajectories that finish above initial_wealth (win rate)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from experiments.helpers import get_amm_env, INITIAL_WEALTH, N_STEPS
from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent
from SAiFE_gym.gym.index_names import (
    LP_COLLECTED_FEES0_KEY,
    LP_COLLECTED_FEES1_KEY,
    ASSET_PRICE_KEY,
)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment parameters
# ─────────────────────────────────────────────────────────────────────────────

GAS_COSTS = [0, 10, 50, 100, 500, 1000]   # token-1 units per rebalance

TAU          = 5         # narrow range — LP earns more per trade but rebalances often
N_EPISODES   = 20        # independent episodes
NUM_TRAJ     = 200       # parallel trajectories per episode → 4 000 samples
VOLATILITY   = 2.0
ALPHA3       = 0.0       # no toxicity — isolate gas drag effect
SEED         = 42

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_config(gas_cost, n_episodes=N_EPISODES, num_traj=NUM_TRAJ, seed=SEED):
    env = get_amm_env(
        num_trajectories=num_traj,
        tau=TAU,
        volatility=VOLATILITY,
        alpha3=ALPHA3,
        n_steps=N_STEPS,
        gas_cost=gas_cost,
        swap_fee_rate=0.0,
        seed=seed,
    )
    agent = UniformAllocationAgent(env)

    all_pnl, all_fees = [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        cum_reward = np.zeros(num_traj)

        for _ in range(N_STEPS):
            action = agent.get_action(obs)
            obs, rewards, _, _, _ = env.step(action)
            cum_reward += rewards

        final_price = obs[ASSET_PRICE_KEY]
        fee_value   = obs[LP_COLLECTED_FEES0_KEY] * final_price + obs[LP_COLLECTED_FEES1_KEY]
        all_pnl.append(cum_reward)
        all_fees.append(fee_value)

    pnl  = np.concatenate(all_pnl)
    fees = np.concatenate(all_fees)
    return pnl, fees


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print(f"  Gas cost sensitivity  (τ={TAU}, vol={VOLATILITY}, α₃={ALPHA3})")
    print(f"  {N_EPISODES} episodes × {NUM_TRAJ} trajectories per config")
    print(f"  {N_STEPS} rebalances per episode")
    print(f"{'='*65}")

    results = {}
    for gc in GAS_COSTS:
        print(f"  gas_cost={gc:>5} ...", end=" ", flush=True)
        pnl, fees = run_config(gc)
        results[gc] = {"pnl": pnl, "fees": fees}
        win_pct = 100.0 * np.mean(pnl > 0)
        print(
            f"mean ΔW={pnl.mean():>+9,.0f}  "
            f"fees={fees.mean():>8,.0f}  "
            f"gas_drag={gc * N_STEPS:>7,}  "
            f"win={win_pct:.1f}%"
        )

    # ── Summary table ─────────────────────────────────────────────────────
    header = (
        f"\n  {'gas_cost':>9}  {'mean ΔW':>12}  {'std ΔW':>10}  "
        f"{'mean fees':>10}  {'gas drag':>10}  {'win%':>6}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(f"\n{'='*65}")
    print(f"  Summary Table")
    print(sep)
    print(header)
    print(sep)
    for gc in GAS_COSTS:
        r = results[gc]
        win_pct   = 100.0 * np.mean(r["pnl"] > 0)
        gas_drag  = gc * N_STEPS
        print(
            f"  {gc:>9}  {r['pnl'].mean():>+12,.0f}  {r['pnl'].std():>10,.0f}  "
            f"{r['fees'].mean():>10,.0f}  {gas_drag:>10,}  {win_pct:>5.1f}%"
        )
    print(sep)

    # ── Plot ──────────────────────────────────────────────────────────────
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"UniformAllocationAgent (τ={TAU})  –  Effect of Gas Cost per Rebalance\n"
        f"vol={VOLATILITY},  α₃={ALPHA3},  {N_STEPS} rebalances/episode,  "
        f"{N_EPISODES} episodes × {NUM_TRAJ} trajectories",
        fontsize=11,
    )

    def _k(x, _=None):
        return f"{x/1e3:.0f}k" if abs(x) >= 1000 else f"{x:.0f}"

    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(i / (len(GAS_COSTS) - 1)) for i in range(len(GAS_COSTS))]

    # Panel 1 – Mean ΔWealth vs gas_cost
    ax = axes[0]
    mean_pnl = [results[gc]["pnl"].mean()  for gc in GAS_COSTS]
    std_pnl  = [results[gc]["pnl"].std()   for gc in GAS_COSTS]
    ax.errorbar(GAS_COSTS, mean_pnl, yerr=std_pnl, fmt="o-", color="#1f77b4",
                lw=2, ms=7, capsize=5, label="Mean ΔWealth ± std")
    # Overlay deterministic gas drag
    gas_drag = [-gc * N_STEPS for gc in GAS_COSTS]
    ax.plot(GAS_COSTS, gas_drag, "r--", lw=1.5, label=f"−gas_drag (={N_STEPS}×gas_cost)")
    ax.axhline(0, color="0.5", linestyle=":", lw=0.9)
    ax.set_xlabel("gas_cost per rebalance (token-1)", fontsize=10)
    ax.set_ylabel("ΔWealth (token-1)", fontsize=9)
    ax.set_title("Net PnL vs Gas Cost", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2 – Fee income stays constant; gas drag grows
    ax = axes[1]
    mean_fees = [results[gc]["fees"].mean() for gc in GAS_COSTS]
    mean_net  = [results[gc]["pnl"].mean()  for gc in GAS_COSTS]
    ax.plot(GAS_COSTS, mean_fees, "s-",  color="#2ca02c", lw=2, ms=7, label="Fee income")
    ax.plot(GAS_COSTS, mean_net,  "o--", color="#d62728", lw=2, ms=7, label="Net ΔWealth")
    ax.axhline(0, color="0.5", linestyle=":", lw=0.9)
    ax.set_xlabel("gas_cost per rebalance (token-1)", fontsize=10)
    ax.set_ylabel("Token-1 units", fontsize=9)
    ax.set_title("Fee Income vs Net PnL", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 – Box plots per gas_cost
    ax = axes[2]
    box_data = [results[gc]["pnl"] for gc in GAS_COSTS]
    bp = ax.boxplot(
        box_data,
        tick_labels=[str(gc) for gc in GAS_COSTS],
        patch_artist=True,
        notch=False,
        showfliers=False,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color="0.5", linestyle=":", lw=0.9)
    ax.set_xlabel("gas_cost per rebalance (token-1)", fontsize=10)
    ax.set_ylabel("ΔWealth (token-1)", fontsize=9)
    ax.set_title("ΔWealth Distribution (no outliers)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "gas_cost_impact.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved figure: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
