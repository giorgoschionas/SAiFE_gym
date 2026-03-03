"""
Uniform LP: concentrated vs wide liquidity as tau increases,
with and without rebalancing costs.

UniformAllocationAgent always places [-tau, +tau] symmetric range.
As tau grows the LP covers more ticks and approaches Uniswap V2-style
full-range provision (tau = num_ticks // 2 = 1000 covers the entire
price array, the closest analogue to Uniswap V2 in this model).

Three rebalancing cost levels:
  r = 0.0000  (zero cost, baseline)
  r = 0.0001  (0.01% / step  →  ~2%  total over 200 steps)
  r = 0.0010  (0.10% / step  →  ~18% total over 200 steps)

Key thresholds:
  tau  = 50   → LP's range always fits the max intra-step move
                (max_arrivals_per_step = 50 caps any step at 50 ticks)
  tau  = 800  → covers ±4σ of the BM price path
  tau  = 1000 → covers the full liquidity array  ← "Uniswap V2 equiv."

Cost arithmetic (first-order):
  total_cost ≈ N_STEPS × r × INITIAL_WEALTH  (tau-independent)
  fee_income ∝ 1/tau                          (decreasing with tau)
  → break-even tau: fee_income(tau) = total_cost
      r=0.0001:  break-even tau ≈ 15
      r=0.0010:  break-even tau ≈  2  (nothing survives)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent
from SAiFE_gym.rewards.RewardFunctions import PnL

# ============================================================================
# Parameters
# ============================================================================
SEED              = 42
TERMINAL_TIME     = 1.0
N_STEPS           = 200
INITIAL_WEALTH    = 1e6
NUM_TRAJECTORIES  = 200
LIQUIDITY_SCALE   = 1e6

INITIAL_PRICE     = 100.0
DRIFT             = 0.0
VOLATILITY        = 2.0
FEE_TIER          = 0.003

NUM_TICKS         = 2000
V2_TAU            = NUM_TICKS // 2   # = 1000  ← "Uniswap V2 equivalent"

TAU_VALUES            = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
ALPHA3_VALUES         = [0.0, 500.0, 5000.0]
REBALANCE_COST_VALUES = [0.0, 0.0001, 0.001]

# Fixed arrival model params
ALPHA0 = np.array([10.0,  10.0])
ALPHA1 = np.array([100.0, 100.0])
ALPHA2 = np.array([0.0,   0.0])

# Subset for trajectory plots
TRAJ_ALPHA3 = 5000.0
TRAJ_TAU    = [2, 20, 200, 1000]

STEP_TIMES = np.linspace(TERMINAL_TIME / N_STEPS, TERMINAL_TIME, N_STEPS)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# Factory and episode runner
# ============================================================================

def make_env(tau, alpha3, rebalance_cost, num_trajectories, seed):
    step_size = TERMINAL_TIME / N_STEPS
    alpha = np.array([ALPHA0, ALPHA1, ALPHA2, [alpha3, alpha3]])

    midprice_model = BrownianMotionMidpriceModel(
        drift=DRIFT, volatility=VOLATILITY, initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME, step_size=step_size,
        num_trajectories=num_trajectories, seed=seed,
    )
    arrival_model = PoissonLinearArrivalModel(
        alpha=alpha, liquidity_scale=LIQUIDITY_SCALE, step_size=step_size,
        num_trajectories=num_trajectories, seed=seed + 1,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model, arrival_model=arrival_model,
        num_trajectories=num_trajectories, fee_tier=FEE_TIER, tau=tau,
        num_ticks=NUM_TICKS, exponential_value=1.0001,
        initial_wealth=INITIAL_WEALTH, rebalance_cost_coeff=rebalance_cost,
        seed=seed + 2,
    )
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS,
        model_dynamics=model_dynamics,
        reward_function=PnL(initial_wealth=INITIAL_WEALTH),
        num_trajectories=num_trajectories, seed=seed,
    )


def run_episode(env, agent):
    """Run one episode. Returns final_wealth (N,) and cumulative_pnl (N, n_steps)."""
    obs, _ = env.reset()
    step_rewards = np.zeros((env.num_trajectories, N_STEPS))

    for step in range(N_STEPS):
        action = agent.get_action(obs)
        obs, rewards, terminated, truncated, _ = env.step(action)
        step_rewards[:, step] = rewards

    cumulative = np.cumsum(step_rewards, axis=1)
    return INITIAL_WEALTH + cumulative[:, -1], cumulative


# ============================================================================
# Main simulation
# ============================================================================

results      = {}   # (tau, alpha3, cost) → {'mean', 'std'}
trajectories = {}   # (tau, alpha3, cost) → mean cumulative PnL over time (n_steps,)

total   = len(TAU_VALUES) * len(ALPHA3_VALUES) * len(REBALANCE_COST_VALUES)
run_idx = 0

print(f"Running {total} configurations "
      f"({len(TAU_VALUES)} tau × {len(ALPHA3_VALUES)} alpha3 × "
      f"{len(REBALANCE_COST_VALUES)} cost levels), "
      f"{NUM_TRAJECTORIES} trajectories each.\n")

for tau in TAU_VALUES:
    for alpha3 in ALPHA3_VALUES:
        for cost in REBALANCE_COST_VALUES:
            run_idx += 1
            v2_tag = '  [V2]' if tau == V2_TAU else ''
            print(f"[{run_idx:3d}/{total}] tau={tau:4d}{v2_tag:<6}  "
                  f"alpha3={alpha3:8.1f}  cost={cost:.4f} ...",
                  end='', flush=True)

            env   = make_env(tau, alpha3, cost, NUM_TRAJECTORIES, seed=SEED)
            agent = UniformAllocationAgent(env)
            final_wealths, cumulative = run_episode(env, agent)

            results[(tau, alpha3, cost)] = {
                'mean': np.mean(final_wealths),
                'std':  np.std(final_wealths),
            }
            trajectories[(tau, alpha3, cost)] = np.mean(cumulative, axis=0)

            delta = results[(tau, alpha3, cost)]['mean'] - INITIAL_WEALTH
            print(f"  mean ΔW = {delta:+.0f}")

print()

# Approximate total cost per episode (first-order: N * r * W)
EXPECTED_TOTAL_COST = {r: N_STEPS * r * INITIAL_WEALTH for r in REBALANCE_COST_VALUES}

print("Expected total rebalancing drag over episode (first-order  N·r·W):")
for r, drag in EXPECTED_TOTAL_COST.items():
    print(f"  r={r:.4f}  →  ~{drag:>8,.0f}  ({100*drag/INITIAL_WEALTH:.1f}% of initial wealth)")
print()

# ============================================================================
# Plot helpers
# ============================================================================

ALPHA3_COLORS  = {0.0: '#1f77b4', 500.0: '#e07b39', 5000.0: '#2ca02c'}
ALPHA3_LABELS  = {
    0.0:    'α₃ = 0  (no toxicity)',
    500.0:  'α₃ = 500  (moderate)',
    5000.0: 'α₃ = 5000  (high volume)',
}
COST_STYLES     = {0.0: '-',  0.0001: '--', 0.001: ':'}
COST_LINEWIDTHS = {0.0: 2.0, 0.0001: 1.8, 0.001: 1.8}
COST_LABELS     = {
    0.0:    'cost = 0  (baseline)',
    0.0001: 'cost = 0.01%/step  (~2% total)',
    0.001:  'cost = 0.1%/step  (~18% total)',
}
TAU_COLORS = plt.cm.plasma(np.linspace(0.1, 0.85, len(TRAJ_TAU)))


def _setup_tau_axis(ax):
    ax.set_xscale('log')
    ax.set_xlabel('τ  (log scale)')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(TAU_VALUES)
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(True, alpha=0.3)
    # V2 reference line
    ax.axvline(V2_TAU, color='gray', linestyle=':', linewidth=0.9, alpha=0.6)


def _add_v2_label(ax):
    ymin, ymax = ax.get_ylim()
    ax.text(V2_TAU, ymax - 0.04 * (ymax - ymin), 'V2',
            ha='center', fontsize=7, color='gray', va='top')


# ============================================================================
# Figure 1 — PnL vs tau: zero-cost vs non-zero-cost  (one panel per alpha3)
# ============================================================================

fig1, axes1 = plt.subplots(1, len(ALPHA3_VALUES), figsize=(18, 5), sharey=False)
fig1.suptitle(
    'Mean PnL vs τ — effect of rebalancing cost  (UniformAllocationAgent)\n'
    f'{NUM_TRAJECTORIES} trajectories  ·  rebalancing every step (N={N_STEPS})',
    fontsize=12,
)

for col, alpha3 in enumerate(ALPHA3_VALUES):
    ax = axes1[col]

    for cost in REBALANCE_COST_VALUES:
        means = [results[(tau, alpha3, cost)]['mean'] - INITIAL_WEALTH for tau in TAU_VALUES]
        stds  = [results[(tau, alpha3, cost)]['std']  for tau in TAU_VALUES]
        color = ALPHA3_COLORS[alpha3]

        ax.plot(TAU_VALUES, means,
                linestyle=COST_STYLES[cost],
                linewidth=COST_LINEWIDTHS[cost],
                color=color,
                label=COST_LABELS[cost])
        # Shaded ±1 std band only for zero-cost to avoid clutter
        if cost == 0.0:
            ax.fill_between(TAU_VALUES,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.10, color=color)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    _setup_tau_axis(ax)
    _add_v2_label(ax)
    ax.set_title(ALPHA3_LABELS[alpha3])
    ax.legend(fontsize=8)

axes1[0].set_ylabel('Mean final ΔWealth')

plt.tight_layout()
path1 = os.path.join(FIGURES_DIR, 'tau_pnl_cost_comparison.png')
fig1.savefig(path1, dpi=150, bbox_inches='tight')
print(f"Figure 1 saved to: {path1}")


# ============================================================================
# Figure 2 — Cost drag:  ΔPnL(cost) = PnL(cost) − PnL(0)  vs tau
#
# The drag is approximately tau-independent (cost ≈ N·r·W regardless of tau)
# while fee income falls as 1/tau.  This figure makes that asymmetry explicit.
# ============================================================================

fig2, axes2 = plt.subplots(1, len(ALPHA3_VALUES), figsize=(18, 5), sharey=True)
fig2.suptitle(
    'Cost drag:  ΔPnL(cost) = PnL(cost) − PnL(cost=0)  vs τ\n'
    'Dashed reference = first-order estimate  −N·r·W  (tau-independent)',
    fontsize=12,
)

for col, alpha3 in enumerate(ALPHA3_VALUES):
    ax = axes2[col]

    for cost in REBALANCE_COST_VALUES:
        if cost == 0.0:
            continue
        drag = [
            results[(tau, alpha3, cost)]['mean'] - results[(tau, alpha3, 0.0)]['mean']
            for tau in TAU_VALUES
        ]
        ax.plot(TAU_VALUES, drag,
                linestyle=COST_STYLES[cost], linewidth=COST_LINEWIDTHS[cost],
                color='steelblue', label=COST_LABELS[cost])
        # First-order theoretical estimate
        ax.axhline(-EXPECTED_TOTAL_COST[cost],
                   color='steelblue', linestyle=COST_STYLES[cost],
                   linewidth=0.8, alpha=0.45,
                   label=f'theory: −N·r·W = {-EXPECTED_TOTAL_COST[cost]:,.0f}')

    ax.axhline(0, color='black', linestyle='-', linewidth=0.7)
    _setup_tau_axis(ax)
    _add_v2_label(ax)
    ax.set_title(ALPHA3_LABELS[alpha3])
    ax.legend(fontsize=7.5)

axes2[0].set_ylabel('ΔPnL due to cost')

plt.tight_layout()
path2 = os.path.join(FIGURES_DIR, 'tau_cost_drag.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
print(f"Figure 2 saved to: {path2}")


# ============================================================================
# Figure 3 — Trajectories: cost=0 vs cost=0.001 for selected tau
#             Fixed alpha3 = TRAJ_ALPHA3
# ============================================================================

fig3, axes3 = plt.subplots(1, len(TRAJ_TAU), figsize=(5 * len(TRAJ_TAU), 5), sharey=False)
fig3.suptitle(
    f'Cumulative PnL trajectory — cost = 0 (solid) vs cost = 0.001 (dashed)\n'
    f'α₃ = {TRAJ_ALPHA3:.0f}  ·  mean over {NUM_TRAJECTORIES} trajectories',
    fontsize=12,
)

for col, tau in enumerate(TRAJ_TAU):
    ax = axes3[col]
    color = TAU_COLORS[col]

    for cost in [0.0, 0.001]:
        traj  = trajectories[(tau, TRAJ_ALPHA3, cost)]
        label = COST_LABELS[cost].split('(')[0].strip()
        ax.plot(STEP_TIMES, traj,
                color=color, linestyle=COST_STYLES[cost],
                linewidth=1.8, label=label)

    # Shade the gap between zero-cost and high-cost trajectories
    t0   = trajectories[(tau, TRAJ_ALPHA3, 0.0)]
    t001 = trajectories[(tau, TRAJ_ALPHA3, 0.001)]
    ax.fill_between(STEP_TIMES, t001, t0,
                    alpha=0.12, color=color, label='cost drag')

    ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax.set_xlabel('Time')
    ax.set_title(f'τ = {tau}' + ('  [V2]' if tau == V2_TAU else ''))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes3[0].set_ylabel('Mean cumulative PnL')

plt.tight_layout()
path3 = os.path.join(FIGURES_DIR, 'tau_trajectories_cost_comparison.png')
fig3.savefig(path3, dpi=150, bbox_inches='tight')
print(f"Figure 3 saved to: {path3}")

plt.show()

# ============================================================================
# Summary table
# ============================================================================

CW = 12
header = (f"{'tau':>6}  {'alpha3':>8}  {'cost':>8}  "
          f"{'Mean ΔW':>{CW}}  {'Std':>{CW}}  {'ΔW/Std':>8}  {'drag vs 0':>12}")
sep = '-' * len(header)
print()
print(sep)
print(header)
print(sep)

for tau in TAU_VALUES:
    v2_tag = ' [V2]' if tau == V2_TAU else ''
    for alpha3 in ALPHA3_VALUES:
        base_mean = results[(tau, alpha3, 0.0)]['mean']
        for cost in REBALANCE_COST_VALUES:
            r      = results[(tau, alpha3, cost)]
            mean_d = r['mean'] - INITIAL_WEALTH
            sharpe = mean_d / r['std'] if r['std'] > 0 else 0.0
            drag   = r['mean'] - base_mean
            drag_s = f'{drag:+,.0f}' if cost > 0 else '—'
            print(f"{tau:>6}{v2_tag:<6}  {alpha3:>8.1f}  {cost:>8.4f}  "
                  f"{mean_d:>{CW},.0f}  {r['std']:>{CW},.0f}  "
                  f"{sharpe:>8.3f}  {drag_s:>12}")
        print()
    print()

print(sep)
