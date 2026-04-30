"""
Comprehensive RL vs Uniform LP Agent Comparison.

Four parameter sweeps with config deduplication:
  1. Alpha3 ceiling — alpha3 ∈ {0, 100, 300, 500, 1000, 2000, 5000, 10000}  (tau=5, vol=2.0)
  2. Tau width      — tau    ∈ {2, 5, 10, 20}                                (alpha3=500, vol=2.0)
  3. Volatility     — vol    ∈ {0.5, 1.0, 2.0, 4.0}                         (tau=5, alpha3=500)
  4. Tau × Alpha3   — heatmap  {2,5,10,20} × {0, 500, 2000, 5000}           (vol=2.0)

All sweeps use cost=0. Results cached by (tau, alpha3, vol) key.
Incremental JSON saves after each sweep for resume on interruption.

Run:  cd SAiFE_gym && python notebooks/rl_vs_uniform_comprehensive.py
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures", "comprehensive")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(RESULTS_DIR, "comprehensive_results.json")

# ---------------------------------------------------------------------------
# Experiment hyperparameters
# ---------------------------------------------------------------------------

TRAIN_TRAJECTORIES = 50
N_STEPS = 200
TOTAL_TIMESTEPS = TRAIN_TRAJECTORIES * N_STEPS * 100   # 100 PPO rollouts = 1M steps
EVAL_TRAJECTORIES = 50
N_EVAL_EPISODES = 20                                    # × 50 traj = 1000 eval paths

TRAIN_SEED = 42
EVAL_SEED = 999

# Sweep grids
ALPHA3_VALUES = [0.0, 100.0, 300.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
TAU_VALUES    = [20, 50, 200, 400]
VOL_VALUES    = [0.5, 1.0, 2.0, 4.0]

# Fixed defaults (used when parameter is not being swept)
FIXED_TAU = 20
FIXED_ALPHA3 = 500.0
FIXED_VOL = 2.0

# Heatmap grid
HEATMAP_TAUS    = [20, 50, 200, 400]
HEATMAP_ALPHA3S = [0.0, 500.0, 2000.0, 5000.0]


# ---------------------------------------------------------------------------
# Config cache  — (tau, alpha3, vol) → results dict
# ---------------------------------------------------------------------------

_cache: dict = {}


def _cache_key(tau: int, alpha3: float, vol: float) -> str:
    return f"{tau}_{alpha3}_{vol}"


def _save_results():
    """Incrementally save cached results to JSON."""
    serializable = {}
    for key, res in _cache.items():
        serializable[key] = {
            agent: {
                "mean": res[agent]["mean"],
                "std": res[agent]["std"],
            }
            for agent in ("rl", "uniform")
        }
    with open(RESULTS_JSON, "w") as f:
        json.dump(serializable, f, indent=2)


# ---------------------------------------------------------------------------
# Core training + evaluation
# ---------------------------------------------------------------------------

def train_ppo(tau: int, alpha3: float, vol: float):
    """Train a PPO agent from scratch.

    Returns:
        (model, vec_normalize): Trained PPO model and the VecNormalize wrapper
        containing running observation statistics from training.
    """
    env = get_amm_env(
        num_trajectories=TRAIN_TRAJECTORIES,
        n_steps=N_STEPS,
        tau=tau,
        alpha3=alpha3,
        volatility=vol,
        seed=TRAIN_SEED,
    )
    vec_env = wrap_env(env)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=N_STEPS,
        batch_size=max(64, TRAIN_TRAJECTORIES * N_STEPS // 16),
        n_epochs=10,
        gae_lambda=0.95,
        gamma=1.0,
        learning_rate=3e-4,
        normalize_advantage=True,
        verbose=0,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    return model, model.get_env()


def evaluate(model: PPO, vec_normalize, tau: int, alpha3: float, vol: float) -> dict:
    """Evaluate trained PPO vs Uniform on a held-out seed."""
    eval_env = get_amm_env(
        num_trajectories=EVAL_TRAJECTORIES,
        n_steps=N_STEPS,
        tau=tau,
        alpha3=alpha3,
        volatility=vol,
        seed=EVAL_SEED,
    )
    sb3_env = StableBaselinesAMMEnvironment(eval_env)
    return compare_rl_vs_uniform(
        model, eval_env, sb3_env,
        n_eval_episodes=N_EVAL_EPISODES,
        vec_normalize=vec_normalize,
    )


def run_config(tau: int, alpha3: float, vol: float) -> dict:
    """Train + evaluate one configuration with caching."""
    key = _cache_key(tau, alpha3, vol)
    if key in _cache:
        print(f"    [cached] tau={tau} alpha3={alpha3:.0f} vol={vol}")
        return _cache[key]

    label = f"tau={tau} alpha3={alpha3:.0f} vol={vol}"
    t0 = time.time()
    print(f"    Training  {label} ...", end="", flush=True)
    model, vec_normalize = train_ppo(tau, alpha3, vol)
    print(f" done ({time.time()-t0:.0f}s). Evaluating ...", end="", flush=True)
    results = evaluate(model, vec_normalize, tau, alpha3, vol)
    elapsed = time.time() - t0

    rl_dw = results["rl"]["mean"] - INITIAL_WEALTH
    uni_dw = results["uniform"]["mean"] - INITIAL_WEALTH
    print(f" done. RL ΔW={rl_dw:+,.0f}  Uniform ΔW={uni_dw:+,.0f}  ({elapsed:.0f}s)")

    _cache[key] = results
    return results


def _dw(results: dict, agent: str) -> float:
    """Extract ΔWealth (mean - initial) for an agent."""
    return results[agent]["mean"] - INITIAL_WEALTH


def _adv(results: dict) -> float:
    """RL advantage = RL ΔW - Uniform ΔW."""
    return _dw(results, "rl") - _dw(results, "uniform")


# ---------------------------------------------------------------------------
# Helper: sweep plot with two lines + advantage bars
# ---------------------------------------------------------------------------

def sweep_plot(
    x_vals,
    all_results: list,
    x_label: str,
    title: str,
    filename: str,
    x_log: bool = False,
    x_tick_labels: list = None,
):
    """Two-panel sweep plot: mean ΔWealth + RL advantage bars."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12)

    x = np.arange(len(x_vals))

    # Left: mean ΔWealth ± std
    ax = axes[0]
    for agent_key in ("rl", "uniform"):
        means = [_dw(r, agent_key) for r in all_results]
        stds = [r[agent_key]["std"] for r in all_results]
        ax.plot(x, means, marker="o", color=AGENT_COLORS[agent_key],
                label=AGENT_LABELS[agent_key], linewidth=1.8)
        ax.fill_between(x, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.15, color=AGENT_COLORS[agent_key])
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Break-even")
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels or [str(v) for v in x_vals])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean ΔWealth")
    ax.set_title(f"Mean ΔWealth vs {x_label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: RL advantage bars
    ax = axes[1]
    advantages = [_adv(r) for r in all_results]
    colors = [AGENT_COLORS["rl"] if a >= 0 else "#d62728" for a in advantages]
    ax.bar(x, advantages, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels or [str(v) for v in x_vals])
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("RL advantage (RL − Uniform)")
    ax.set_title(f"RL outperformance vs {x_label}")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")
    return advantages


# ===========================================================================
# SWEEP 1: Alpha3 ceiling (toxicity)
# ===========================================================================

print("=" * 65)
print(f"SWEEP 1 — Alpha3 ceiling   [tau={FIXED_TAU}, vol={FIXED_VOL}]")
print("=" * 65)

alpha3_results = []
for a3 in ALPHA3_VALUES:
    alpha3_results.append(run_config(FIXED_TAU, a3, FIXED_VOL))

# Annotate the trough (minimum ΔW for Uniform)
uni_dws = [_dw(r, "uniform") for r in alpha3_results]
trough_idx = int(np.argmin(uni_dws))
trough_alpha3 = ALPHA3_VALUES[trough_idx]
trough_dw = uni_dws[trough_idx]

alpha3_advantages = sweep_plot(
    ALPHA3_VALUES, alpha3_results,
    x_label="α₃ (toxicity)",
    title=(f"Alpha3 ceiling: LP loss trough identification\n"
           f"tau={FIXED_TAU}, vol={FIXED_VOL}, cost=0, "
           f"{EVAL_TRAJECTORIES*N_EVAL_EPISODES} eval paths"),
    filename="sweep_alpha3_ceiling.png",
)

_save_results()
print(f"  LP loss trough at alpha3={trough_alpha3:.0f}, Uniform ΔW={trough_dw:+,.0f}\n")


# ===========================================================================
# SWEEP 2: Tau width
# ===========================================================================

print("=" * 65)
print(f"SWEEP 2 — Tau width   [alpha3={FIXED_ALPHA3:.0f}, vol={FIXED_VOL}]")
print("=" * 65)

tau_results = []
for tau in TAU_VALUES:
    tau_results.append(run_config(tau, FIXED_ALPHA3, FIXED_VOL))

tau_advantages = sweep_plot(
    TAU_VALUES, tau_results,
    x_label="τ (position half-width)",
    title=(f"Effect of position width (τ) on LP profit\n"
           f"α₃={FIXED_ALPHA3:.0f}, vol={FIXED_VOL}, cost=0, "
           f"{EVAL_TRAJECTORIES*N_EVAL_EPISODES} eval paths"),
    filename="sweep_tau_width.png",
)

_save_results()
print()


# ===========================================================================
# SWEEP 3: Volatility
# ===========================================================================

print("=" * 65)
print(f"SWEEP 3 — Volatility   [tau={FIXED_TAU}, alpha3={FIXED_ALPHA3:.0f}]")
print("=" * 65)

vol_results = []
for vol in VOL_VALUES:
    vol_results.append(run_config(FIXED_TAU, FIXED_ALPHA3, vol))

vol_advantages = sweep_plot(
    VOL_VALUES, vol_results,
    x_label="Volatility (σ)",
    title=(f"Effect of price volatility on LP profit\n"
           f"tau={FIXED_TAU}, α₃={FIXED_ALPHA3:.0f}, cost=0, "
           f"{EVAL_TRAJECTORIES*N_EVAL_EPISODES} eval paths"),
    filename="sweep_volatility.png",
)

_save_results()
print()


# ===========================================================================
# SWEEP 4: Tau × Alpha3 heatmap
# ===========================================================================

print("=" * 65)
print(f"SWEEP 4 — Tau × Alpha3 heatmap   [vol={FIXED_VOL}]")
print("=" * 65)

heatmap_data = {}
for tau in HEATMAP_TAUS:
    for a3 in HEATMAP_ALPHA3S:
        res = run_config(tau, a3, FIXED_VOL)
        heatmap_data[(tau, a3)] = res

_save_results()

# Build matrices for plotting
n_tau = len(HEATMAP_TAUS)
n_a3 = len(HEATMAP_ALPHA3S)
rl_matrix = np.zeros((n_tau, n_a3))
uni_matrix = np.zeros((n_tau, n_a3))
adv_matrix = np.zeros((n_tau, n_a3))

for i, tau in enumerate(HEATMAP_TAUS):
    for j, a3 in enumerate(HEATMAP_ALPHA3S):
        res = heatmap_data[(tau, a3)]
        rl_matrix[i, j] = _dw(res, "rl")
        uni_matrix[i, j] = _dw(res, "uniform")
        adv_matrix[i, j] = _adv(res)

# 3-panel heatmap figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(f"Tau × Alpha3 interaction (vol={FIXED_VOL}, cost=0)", fontsize=13)

tau_labels = [str(t) for t in HEATMAP_TAUS]
a3_labels = [f"{a:.0f}" for a in HEATMAP_ALPHA3S]

# Symmetric diverging norm centered at 0
all_vals = np.concatenate([rl_matrix.ravel(), uni_matrix.ravel()])
vmax_wealth = max(abs(all_vals.min()), abs(all_vals.max()))
wealth_norm = mcolors.TwoSlopeNorm(vmin=-vmax_wealth, vcenter=0, vmax=vmax_wealth)

vmax_adv = max(abs(adv_matrix.min()), abs(adv_matrix.max())) or 1.0
adv_norm = mcolors.TwoSlopeNorm(vmin=-vmax_adv, vcenter=0, vmax=vmax_adv)

for ax, matrix, title, norm in [
    (axes[0], rl_matrix,  "RL ΔWealth",     wealth_norm),
    (axes[1], uni_matrix, "Uniform ΔWealth", wealth_norm),
    (axes[2], adv_matrix, "RL Advantage",    adv_norm),
]:
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(range(n_a3))
    ax.set_xticklabels(a3_labels)
    ax.set_yticks(range(n_tau))
    ax.set_yticklabels(tau_labels)
    ax.set_xlabel("α₃")
    ax.set_ylabel("τ")
    ax.set_title(title)
    # Annotate cells
    for i in range(n_tau):
        for j in range(n_a3):
            ax.text(j, i, f"{matrix[i,j]:+,.0f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
path = os.path.join(FIGURES_DIR, "heatmap_tau_alpha3.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Figure saved: {path}\n")


# ===========================================================================
# MASTER SUMMARY TABLE
# ===========================================================================

print("=" * 80)
print("MASTER RESULTS SUMMARY")
print("=" * 80)

header = (
    f"{'Sweep':<12} | {'tau':>4} | {'α₃':>8} | {'vol':>5} "
    f"| {'RL ΔW':>12} | {'Uni ΔW':>12} | {'RL Adv':>10} "
    f"| {'RL Std':>10} | {'Uni Std':>10}"
)
sep = "-" * len(header)
print(header)
print(sep)

summary_lines = [header, sep]

def _print_row(sweep: str, tau: int, a3: float, vol: float, res: dict):
    rl_dw = _dw(res, "rl")
    uni_dw = _dw(res, "uniform")
    adv = rl_dw - uni_dw
    line = (
        f"{sweep:<12} | {tau:>4} | {a3:>8.0f} | {vol:>5.1f} "
        f"| {rl_dw:>+12,.0f} | {uni_dw:>+12,.0f} | {adv:>+10,.0f} "
        f"| {res['rl']['std']:>10,.0f} | {res['uniform']['std']:>10,.0f}"
    )
    print(line)
    summary_lines.append(line)

summary_lines.append("--- Alpha3 sweep ---")
print("--- Alpha3 sweep ---")
for i, a3 in enumerate(ALPHA3_VALUES):
    _print_row("alpha3", FIXED_TAU, a3, FIXED_VOL, alpha3_results[i])

summary_lines.append("--- Tau sweep ---")
print("--- Tau sweep ---")
for i, tau in enumerate(TAU_VALUES):
    _print_row("tau", tau, FIXED_ALPHA3, FIXED_VOL, tau_results[i])

summary_lines.append("--- Volatility sweep ---")
print("--- Volatility sweep ---")
for i, vol in enumerate(VOL_VALUES):
    _print_row("vol", FIXED_TAU, FIXED_ALPHA3, vol, vol_results[i])

summary_lines.append("--- Heatmap configs ---")
print("--- Heatmap configs ---")
for tau in HEATMAP_TAUS:
    for a3 in HEATMAP_ALPHA3S:
        _print_row("heatmap", tau, a3, FIXED_VOL, heatmap_data[(tau, a3)])

print(sep)
summary_lines.append(sep)

# Save summary table
summary_path = os.path.join(RESULTS_DIR, "summary_table.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))
print(f"\nSummary table saved: {summary_path}")


# ===========================================================================
# AUTOMATED OBSERVATIONS
# ===========================================================================

observations = []

# 1. Alpha3 ceiling location
observations.append(
    f"1. LP LOSS TROUGH (Alpha3 ceiling):\n"
    f"   Trough at alpha3 = {trough_alpha3:.0f}\n"
    f"   Uniform ΔWealth at trough = {trough_dw:+,.0f}\n"
    f"   (out of grid: {[f'{a:.0f}' for a in ALPHA3_VALUES]})"
)

# 2. Is RL advantage largest near the trough?
alpha3_advs = [_adv(r) for r in alpha3_results]
best_adv_idx = int(np.argmax(alpha3_advs))
observations.append(
    f"2. RL ADVANTAGE vs ALPHA3:\n"
    f"   Largest RL advantage at alpha3 = {ALPHA3_VALUES[best_adv_idx]:.0f} "
    f"(advantage = {alpha3_advs[best_adv_idx]:+,.0f})\n"
    f"   Trough is at alpha3 = {trough_alpha3:.0f} — "
    f"{'MATCH' if ALPHA3_VALUES[best_adv_idx] == trough_alpha3 else 'MISMATCH'}\n"
    f"   All advantages: {[f'{a:+,.0f}' for a in alpha3_advs]}"
)

# 3. Does RL advantage grow with volatility?
vol_advs = [_adv(r) for r in vol_results]
vol_trend = "INCREASING" if vol_advs[-1] > vol_advs[0] else "DECREASING"
observations.append(
    f"3. RL ADVANTAGE vs VOLATILITY:\n"
    f"   Trend: {vol_trend} (low vol adv = {vol_advs[0]:+,.0f}, "
    f"high vol adv = {vol_advs[-1]:+,.0f})\n"
    f"   All advantages: {[f'{a:+,.0f}' for a in vol_advs]}"
)

# 4. Is RL advantage largest at small tau?
tau_advs = [_adv(r) for r in tau_results]
best_tau_idx = int(np.argmax(tau_advs))
observations.append(
    f"4. RL ADVANTAGE vs TAU:\n"
    f"   Largest RL advantage at tau = {TAU_VALUES[best_tau_idx]} "
    f"(advantage = {tau_advs[best_tau_idx]:+,.0f})\n"
    f"   Smallest tau (tau={TAU_VALUES[0]}) advantage = {tau_advs[0]:+,.0f}\n"
    f"   All advantages: {[f'{a:+,.0f}' for a in tau_advs]}"
)

# 5. Heatmap interaction summary
best_heatmap = max(heatmap_data.items(), key=lambda x: _adv(x[1]))
worst_heatmap = min(heatmap_data.items(), key=lambda x: _adv(x[1]))
observations.append(
    f"5. TAU × ALPHA3 INTERACTION:\n"
    f"   Best RL advantage:  tau={best_heatmap[0][0]}, alpha3={best_heatmap[0][1]:.0f} "
    f"→ advantage = {_adv(best_heatmap[1]):+,.0f}\n"
    f"   Worst RL advantage: tau={worst_heatmap[0][0]}, alpha3={worst_heatmap[0][1]:.0f} "
    f"→ advantage = {_adv(worst_heatmap[1]):+,.0f}"
)

# Print and save
print("\n" + "=" * 65)
print("AUTOMATED OBSERVATIONS")
print("=" * 65)
obs_text = "\n\n".join(observations)
print(obs_text)

obs_path = os.path.join(RESULTS_DIR, "observations.txt")
with open(obs_path, "w") as f:
    f.write(obs_text)
print(f"\nObservations saved: {obs_path}")

# Final save
_save_results()

print(f"\nTotal unique configs trained: {len(_cache)}")
print(f"Results JSON: {RESULTS_JSON}")
print(f"Figures: {FIGURES_DIR}/")
print("\nDone.")
