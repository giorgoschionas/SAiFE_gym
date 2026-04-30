"""
RL Agent Evaluation: PPO vs UniformAllocationAgent

Three parameter sweeps:
  1. Toxicity  — alpha3  ∈ {0, 500, 2000, 5000, 10000}  (tau=5, cost=0)
  2. Position  — tau     ∈ {2, 5, 10, 20}               (alpha3=0, cost=0)
  3. Cost      — cost    ∈ {0, 0.0001, 0.0005, 0.001}   (tau=5, alpha3=0)

For each configuration a fresh PPO agent is trained from scratch, then
evaluated head-to-head against UniformAllocationAgent on a held-out seed.
Results are saved as PNG figures in notebooks/figures/.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from experiments.helpers import (
    INITIAL_WEALTH,
    AGENT_COLORS,
    AGENT_LABELS,
    get_amm_env,
    wrap_env,
    compare_rl_vs_uniform,
    print_comparison_table,
)

# ---------------------------------------------------------------------------
# Experiment hyperparameters
# ---------------------------------------------------------------------------

TRAIN_TRAJECTORIES = 50
N_STEPS = 200
TOTAL_TIMESTEPS = TRAIN_TRAJECTORIES * N_STEPS * 60   # 60 PPO rollouts ≈ 600k steps
EVAL_TRAJECTORIES = 50
N_EVAL_EPISODES = 20                                   # × 50 traj = 1000 eval paths

TRAIN_SEED = 42
EVAL_SEED = 999    # deliberately different from training seed

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Fixed values when a parameter is held constant
FIXED_TAU = 5
FIXED_ALPHA3 = 0.0
FIXED_COST = 0.0
FIXED_ARRIVAL_RATE = 100.0

# Sweep grids
ALPHA3_VALUES = [0.0, 500.0, 2000.0, 5000.0, 10000.0]
TAU_VALUES    = [2, 5, 10, 20]
COST_VALUES   = [0.0, 0.0001, 0.0005, 0.001]


# ---------------------------------------------------------------------------
# Core training + evaluation routine
# ---------------------------------------------------------------------------

def train_ppo(tau: int, alpha3: float, cost: float) -> PPO:
    """Train a PPO agent on a fresh environment with the given parameters."""
    env = get_amm_env(
        num_trajectories=TRAIN_TRAJECTORIES,
        n_steps=N_STEPS,
        tau=tau,
        alpha3=alpha3,
        rebalance_cost_coeff=cost,
        seed=TRAIN_SEED,
    )
    model = PPO(
        policy="MlpPolicy",
        env=wrap_env(env),
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=N_STEPS,
        batch_size=TRAIN_TRAJECTORIES * N_STEPS // 4,
        n_epochs=10,
        gae_lambda=0.95,
        gamma=1.0,
        normalize_advantage=True,
        verbose=0,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    return model


def evaluate(model: PPO, tau: int, alpha3: float, cost: float) -> dict:
    """Evaluate trained PPO vs Uniform on a held-out seed."""
    eval_env = get_amm_env(
        num_trajectories=EVAL_TRAJECTORIES,
        n_steps=N_STEPS,
        tau=tau,
        alpha3=alpha3,
        rebalance_cost_coeff=cost,
        seed=EVAL_SEED,
    )
    sb3_env = StableBaselinesAMMEnvironment(eval_env)
    return compare_rl_vs_uniform(
        model, eval_env, sb3_env, n_eval_episodes=N_EVAL_EPISODES
    )


def run_config(label: str, tau: int, alpha3: float, cost: float) -> dict:
    """Train + evaluate one configuration, print progress and timing."""
    t0 = time.time()
    print(f"  Training  {label} ...", end="", flush=True)
    model = train_ppo(tau, alpha3, cost)
    print(f" done ({time.time()-t0:.0f}s). Evaluating ...", end="", flush=True)
    results = evaluate(model, tau, alpha3, cost)
    elapsed = time.time() - t0
    rl_dw  = results["rl"]["mean"]  - INITIAL_WEALTH
    uni_dw = results["uniform"]["mean"] - INITIAL_WEALTH
    print(f" done. RL ΔW={rl_dw:+.0f}  Uniform ΔW={uni_dw:+.0f}  ({elapsed:.0f}s total)")
    return results


# ---------------------------------------------------------------------------
# Helper: build a mean±std sweep plot with two lines (RL and Uniform)
# ---------------------------------------------------------------------------

def _sweep_plot(
    ax,
    x_vals,
    rl_means, rl_stds,
    uni_means, uni_stds,
    x_label: str,
):
    x = np.array(x_vals, dtype=float)
    for key, means, stds in [
        ("rl",      rl_means,  rl_stds),
        ("uniform", uni_means, uni_stds),
    ]:
        means_arr = np.array(means)
        stds_arr  = np.array(stds)
        ax.plot(x, means_arr, marker="o", color=AGENT_COLORS[key],
                label=AGENT_LABELS[key], linewidth=1.8)
        ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr,
                        alpha=0.15, color=AGENT_COLORS[key])
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Break-even")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean ΔWealth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ===========================================================================
# SWEEP 1: alpha3 (toxicity)
# ===========================================================================

print("=" * 60)
print("SWEEP 1 — Toxicity (alpha3)   [tau=5, cost=0]")
print("=" * 60)

alpha3_results = {}
for alpha3 in ALPHA3_VALUES:
    label = f"tau={FIXED_TAU} alpha3={alpha3:.0f} cost={FIXED_COST}"
    alpha3_results[alpha3] = run_config(label, FIXED_TAU, alpha3, FIXED_COST)

# ---- Figure 1 ----
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle(
    f"Effect of order-flow toxicity (α₃) on LP profit\n"
    f"tau={FIXED_TAU},  rebalance_cost={FIXED_COST},  {EVAL_TRAJECTORIES*N_EVAL_EPISODES} eval paths",
    fontsize=12,
)

# left panel: mean ΔWealth vs alpha3
rl_means  = [alpha3_results[a]["rl"]["mean"]      - INITIAL_WEALTH for a in ALPHA3_VALUES]
rl_stds   = [alpha3_results[a]["rl"]["std"]                         for a in ALPHA3_VALUES]
uni_means = [alpha3_results[a]["uniform"]["mean"]  - INITIAL_WEALTH for a in ALPHA3_VALUES]
uni_stds  = [alpha3_results[a]["uniform"]["std"]                    for a in ALPHA3_VALUES]
_sweep_plot(axes1[0], ALPHA3_VALUES, rl_means, rl_stds, uni_means, uni_stds, "α₃ (toxicity)")
axes1[0].set_title("Mean ΔWealth vs α₃")
axes1[0].set_xscale("symlog", linthresh=1)

# right panel: RL advantage (RL mean − Uniform mean) vs alpha3
advantage = [r - u for r, u in zip(rl_means, uni_means)]
axes1[1].bar(range(len(ALPHA3_VALUES)), advantage,
             color=[AGENT_COLORS["rl"] if a >= 0 else "#d62728" for a in advantage],
             alpha=0.7)
axes1[1].set_xticks(range(len(ALPHA3_VALUES)))
axes1[1].set_xticklabels([f"{a:.0f}" for a in ALPHA3_VALUES])
axes1[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes1[1].set_xlabel("α₃ (toxicity)")
axes1[1].set_ylabel("RL advantage  (RL mean − Uniform mean)")
axes1[1].set_title("RL outperformance vs α₃")
axes1[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig1.savefig(os.path.join(FIGURES_DIR, "rl_sweep_alpha3.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: figures/rl_sweep_alpha3.png")

print("\nSummary table — toxicity sweep:")
for a in ALPHA3_VALUES:
    print_comparison_table(alpha3_results[a], tau=FIXED_TAU, alpha3=a)


# ===========================================================================
# SWEEP 2: tau (position width)
# ===========================================================================

print("\n" + "=" * 60)
print("SWEEP 2 — Position width (tau)   [alpha3=0, cost=0]")
print("=" * 60)

tau_results = {}
for tau in TAU_VALUES:
    label = f"tau={tau} alpha3={FIXED_ALPHA3:.0f} cost={FIXED_COST}"
    if tau == FIXED_TAU and FIXED_ALPHA3 == 0.0:
        # Reuse result from sweep 1 (same config)
        tau_results[tau] = alpha3_results[0.0]
        print(f"  Reusing tau={tau} result from sweep 1.")
    else:
        tau_results[tau] = run_config(label, tau, FIXED_ALPHA3, FIXED_COST)

# ---- Figure 2 ----
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(
    f"Effect of position width (τ) on LP profit\n"
    f"α₃={FIXED_ALPHA3:.0f},  rebalance_cost={FIXED_COST},  {EVAL_TRAJECTORIES*N_EVAL_EPISODES} eval paths",
    fontsize=12,
)

rl_means_t  = [tau_results[t]["rl"]["mean"]     - INITIAL_WEALTH for t in TAU_VALUES]
rl_stds_t   = [tau_results[t]["rl"]["std"]                        for t in TAU_VALUES]
uni_means_t = [tau_results[t]["uniform"]["mean"] - INITIAL_WEALTH for t in TAU_VALUES]
uni_stds_t  = [tau_results[t]["uniform"]["std"]                   for t in TAU_VALUES]

_sweep_plot(axes2[0], TAU_VALUES, rl_means_t, rl_stds_t, uni_means_t, uni_stds_t, "τ (position half-width)")
axes2[0].set_title("Mean ΔWealth vs τ")
axes2[0].set_xticks(TAU_VALUES)

advantage_t = [r - u for r, u in zip(rl_means_t, uni_means_t)]
axes2[1].bar(range(len(TAU_VALUES)), advantage_t,
             color=[AGENT_COLORS["rl"] if a >= 0 else "#d62728" for a in advantage_t],
             alpha=0.7)
axes2[1].set_xticks(range(len(TAU_VALUES)))
axes2[1].set_xticklabels([str(t) for t in TAU_VALUES])
axes2[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes2[1].set_xlabel("τ (position half-width)")
axes2[1].set_ylabel("RL advantage  (RL mean − Uniform mean)")
axes2[1].set_title("RL outperformance vs τ")
axes2[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig2.savefig(os.path.join(FIGURES_DIR, "rl_sweep_tau.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: figures/rl_sweep_tau.png")

print("\nSummary table — tau sweep:")
for t in TAU_VALUES:
    print_comparison_table(tau_results[t], tau=t, alpha3=FIXED_ALPHA3)


# ===========================================================================
# SWEEP 3: rebalancing cost
# ===========================================================================

print("\n" + "=" * 60)
print("SWEEP 3 — Rebalancing cost   [tau=5, alpha3=0]")
print("=" * 60)

cost_results = {}
for cost in COST_VALUES:
    label = f"tau={FIXED_TAU} alpha3={FIXED_ALPHA3:.0f} cost={cost}"
    if cost == FIXED_COST and FIXED_TAU == 5 and FIXED_ALPHA3 == 0.0:
        cost_results[cost] = alpha3_results[0.0]
        print(f"  Reusing cost={cost} result from sweep 1.")
    else:
        cost_results[cost] = run_config(label, FIXED_TAU, FIXED_ALPHA3, cost)

# Compute effective total cost: 1 - (1-cost)^N_STEPS  (for labeling)
def _total_cost_pct(c):
    return 100 * (1 - (1 - c) ** N_STEPS)

# ---- Figure 3 ----
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(
    f"Effect of rebalancing cost on LP profit\n"
    f"τ={FIXED_TAU},  α₃={FIXED_ALPHA3:.0f},  {EVAL_TRAJECTORIES*N_EVAL_EPISODES} eval paths",
    fontsize=12,
)

rl_means_c  = [cost_results[c]["rl"]["mean"]     - INITIAL_WEALTH for c in COST_VALUES]
rl_stds_c   = [cost_results[c]["rl"]["std"]                        for c in COST_VALUES]
uni_means_c = [cost_results[c]["uniform"]["mean"] - INITIAL_WEALTH for c in COST_VALUES]
uni_stds_c  = [cost_results[c]["uniform"]["std"]                   for c in COST_VALUES]

x_cost = np.arange(len(COST_VALUES))
cost_labels = [f"{c*100:.2f}%\n(total≈{_total_cost_pct(c):.1f}%)" for c in COST_VALUES]

for key, means, stds in [
    ("rl",      rl_means_c,  rl_stds_c),
    ("uniform", uni_means_c, uni_stds_c),
]:
    means_arr = np.array(means)
    stds_arr  = np.array(stds)
    axes3[0].plot(x_cost, means_arr, marker="o", color=AGENT_COLORS[key],
                  label=AGENT_LABELS[key], linewidth=1.8)
    axes3[0].fill_between(x_cost, means_arr - stds_arr, means_arr + stds_arr,
                          alpha=0.15, color=AGENT_COLORS[key])
axes3[0].set_xticks(x_cost)
axes3[0].set_xticklabels(cost_labels, fontsize=8)
axes3[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Break-even")
axes3[0].set_xlabel("Per-step rebalancing cost  (total cumulative cost over episode)")
axes3[0].set_ylabel("Mean ΔWealth")
axes3[0].set_title("Mean ΔWealth vs rebalancing cost")
axes3[0].legend(fontsize=8)
axes3[0].grid(True, alpha=0.3)

advantage_c = [r - u for r, u in zip(rl_means_c, uni_means_c)]
axes3[1].bar(x_cost, advantage_c,
             color=[AGENT_COLORS["rl"] if a >= 0 else "#d62728" for a in advantage_c],
             alpha=0.7)
axes3[1].set_xticks(x_cost)
axes3[1].set_xticklabels(cost_labels, fontsize=8)
axes3[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes3[1].set_xlabel("Per-step rebalancing cost")
axes3[1].set_ylabel("RL advantage  (RL mean − Uniform mean)")
axes3[1].set_title("RL outperformance vs rebalancing cost")
axes3[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig3.savefig(os.path.join(FIGURES_DIR, "rl_sweep_cost.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: figures/rl_sweep_cost.png")

print("\nSummary table — cost sweep:")
for c in COST_VALUES:
    print_comparison_table(cost_results[c], tau=FIXED_TAU, alpha3=FIXED_ALPHA3)


# ===========================================================================
# MASTER SUMMARY TABLE
# ===========================================================================

print("\n" + "=" * 70)
print("MASTER RESULTS SUMMARY")
print("=" * 70)
header = (
    f"{'Sweep':<10} | {'Param':>10} | {'RL ΔW':>12} | {'Uni ΔW':>12} "
    f"| {'RL Adv':>10} | {'RL Std':>10} | {'Uni Std':>10}"
)
sep = "-" * len(header)
print(header)
print(sep)

def _row(sweep, param, res):
    rl  = res["rl"]
    uni = res["uniform"]
    rl_dw  = rl["mean"]  - INITIAL_WEALTH
    uni_dw = uni["mean"] - INITIAL_WEALTH
    adv    = rl_dw - uni_dw
    print(
        f"{sweep:<10} | {param:>10} | {rl_dw:>+12,.0f} | {uni_dw:>+12,.0f} "
        f"| {adv:>+10,.0f} | {rl['std']:>10,.0f} | {uni['std']:>10,.0f}"
    )

print("--- alpha3 sweep (tau=5, cost=0) ---")
for a in ALPHA3_VALUES:
    _row("alpha3", f"{a:.0f}", alpha3_results[a])

print("--- tau sweep (alpha3=0, cost=0) ---")
for t in TAU_VALUES:
    _row("tau", str(t), tau_results[t])

print("--- cost sweep (tau=5, alpha3=0) ---")
for c in COST_VALUES:
    _row("cost", f"{c*100:.2f}%", cost_results[c])

print(sep)
print("\nAll figures saved to notebooks/figures/")
