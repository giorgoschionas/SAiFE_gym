"""
Mispricing dynamics as a function of α₃ in PoissonLinearArrivalModel.

Measures ε = S − Z  (external midprice minus AMM pool price) over time for
different values of the arbitrage coefficient α₃.

== Anticipated behavior (Ornstein–Uhlenbeck approximation) ==

The mispricing ε = S − Z satisfies, in continuous time:

    dε = σ dW − κ ε dt + Poisson noise

where the restoring rate κ comes from the asymmetric arrivals:
    net arrivals in direction of S ~ α₃·ε·dt   (before floor)
    each arrival moves Z by one tick ≈ price · (exp_val − 1) ≈ 0.01

Hence  κ ≈ 2·α₃·0.01 = 0.02·α₃,  and the OU equilibrium gives:

    Var(ε)  = σ² / (2κ) = 4 / (0.04·α₃) = 100 / α₃
    Std(ε)  ≈ 10 / √α₃
    E[|ε|]  ≈ 10 / √α₃ · √(2/π)  ≈ 7.98 / √α₃

Caveats:
  - Formula assumes α₃·|ε| < α₁ − α₀ = 90 (otherwise sell/buy intensity hits floor α₀).
    At equilibrium std ≈ 10/√α₃, this floor is regularly hit for α₃ ≳ 100 (α₃·std ~ 10√α₃).
    The floor creates a stronger restoring force → actual steady-state |ε| will be
    BELOW the linear OU prediction.
  - max_arrivals_per_step = 50 caps any step at 50 tick moves;
    BM moves ≈ σ√dt ≈ 0.14 price units ≈ 14 ticks per step on average, so the cap
    is rarely binding for the values tested.
  - Tick size changes with price; formula uses approximation at P ≈ 100.

== α₃ = 0 (no coupling) ==
    No directed arrivals → Z performs its own random walk independent of S.
    ε = S − Z ~ difference of two independent BM-like processes → |ε| grows as √t.

Three questions tested:
  Q1. Does |ε| converge to a stationary level for α₃ > 0?
  Q2. Does steady-state E[|ε|] decrease with α₃?  (and how does it scale?)
  Q3. Is the α₃ = 0 case visibly divergent?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.gym.index_names import ASSET_PRICE_KEY, POOL_SQRT_PRICE_KEY

# ============================================================================
# Parameters
# ============================================================================
SEED             = 42
TERMINAL_TIME    = 1.0
N_STEPS          = 500            # longer episode to see convergence
INITIAL_WEALTH   = 1e6
NUM_TRAJECTORIES = 100
LIQUIDITY_SCALE  = 1e6

INITIAL_PRICE = 100.0
DRIFT         = 0.0
VOLATILITY    = 2.0
FEE_TIER      = 0.003
TAU           = 50       # large enough LP range; LP presence doesn't affect
                          # mispricing when alpha2 = 0
NUM_TICKS     = 2000

# Arrival model — alpha2 = 0 so liquidity doesn't enter intensity
ALPHA0 = np.array([10.0,  10.0])
ALPHA1 = np.array([100.0, 100.0])
ALPHA2 = np.array([0.0,   0.0])

ALPHA3_VALUES = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

# Use last STEADY_FRAC fraction of steps to estimate steady-state statistics
STEADY_FRAC  = 0.5
STEADY_START = int(N_STEPS * (1 - STEADY_FRAC))

STEP_TIMES = np.linspace(TERMINAL_TIME / N_STEPS, TERMINAL_TIME, N_STEPS)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# Theoretical OU prediction  (valid when floor not hit, i.e. α₃ small)
# ============================================================================
# κ = 2 · α₃ · tick_size_abs,  tick_size_abs ≈ 0.01 at P ≈ 100
TICK_SIZE_ABS = INITIAL_PRICE * (1.0001 - 1.0)   # ≈ 0.01

def ou_std(alpha3: float) -> float:
    if alpha3 == 0:
        return np.nan
    kappa = 2.0 * alpha3 * TICK_SIZE_ABS
    return np.sqrt(VOLATILITY**2 / (2.0 * kappa))

def ou_mean_abs(alpha3: float) -> float:
    return ou_std(alpha3) * np.sqrt(2.0 / np.pi)

# ============================================================================
# Factory
# ============================================================================

def make_env(alpha3: float, num_trajectories: int, seed: int) -> AMMEnvironment:
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
        num_trajectories=num_trajectories, fee_tier=FEE_TIER, tau=TAU,
        num_ticks=NUM_TICKS, exponential_value=1.0001,
        initial_wealth=INITIAL_WEALTH, rebalance_cost_coeff=0.0,
        seed=seed + 2,
    )
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS,
        model_dynamics=model_dynamics,
        reward_function=PnL(initial_wealth=INITIAL_WEALTH),
        num_trajectories=num_trajectories, seed=seed,
    )


# ============================================================================
# Simulation
# ============================================================================

# mispricing[alpha3] → array of shape (num_trajectories, N_STEPS)
mispricing_series = {}

print(f"Running {len(ALPHA3_VALUES)} alpha3 values, "
      f"{NUM_TRAJECTORIES} trajectories × {N_STEPS} steps each.\n")

for alpha3 in ALPHA3_VALUES:
    env   = make_env(alpha3, NUM_TRAJECTORIES, seed=SEED)
    agent = UniformAllocationAgent(env)

    obs = env.reset()
    eps = np.zeros((NUM_TRAJECTORIES, N_STEPS))

    for step in range(N_STEPS):
        action = agent.get_action(obs)
        obs, _, _, _ = env.step(action)

        S = obs[ASSET_PRICE_KEY]                  # external midprice, shape (N,)
        Z = obs[POOL_SQRT_PRICE_KEY] ** 2         # AMM price, shape (N,)
        eps[:, step] = S - Z

    mispricing_series[alpha3] = eps

    mean_abs_ss = np.mean(np.abs(eps[:, STEADY_START:]))
    ou_pred     = ou_mean_abs(alpha3)
    ou_str      = f"{ou_pred:.4f}" if not np.isnan(ou_pred) else "∞  (diverges)"
    print(f"  alpha3={alpha3:>6}   steady-state E[|ε|] = {mean_abs_ss:.4f}   "
          f"OU theory = {ou_str}")

print()

# ============================================================================
# Derived statistics
# ============================================================================

steady_mean_abs = {}   # E[|ε|] in steady state
steady_std      = {}   # std(ε) in steady state
time_mean_abs   = {}   # mean|ε| at each step (averaged over trajectories)

for alpha3 in ALPHA3_VALUES:
    eps = mispricing_series[alpha3]
    steady_mean_abs[alpha3] = np.mean(np.abs(eps[:, STEADY_START:]))
    steady_std[alpha3]      = np.std(eps[:, STEADY_START:])
    time_mean_abs[alpha3]   = np.mean(np.abs(eps), axis=0)  # (N_STEPS,)

# ============================================================================
# Plots
# ============================================================================

ALPHA3_NONZERO = [a for a in ALPHA3_VALUES if a > 0]
cmap   = plt.cm.plasma
colors = {a: cmap(i / (len(ALPHA3_VALUES) - 1))
          for i, a in enumerate(ALPHA3_VALUES)}

# -------------------------------------------------------------------------
# Figure 1 — Time evolution of mean |ε|  (convergence check)
# -------------------------------------------------------------------------

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle(
    r'Time evolution of mean $|\varepsilon|$ = mean $|S - Z|$'
    '\n(mispricing between external midprice and AMM price)',
    fontsize=12,
)

# Left: all alpha3 including 0
ax = axes1[0]
for alpha3 in ALPHA3_VALUES:
    tma = time_mean_abs[alpha3]
    label = f'α₃={alpha3}' + (' (no coupling)' if alpha3 == 0 else '')
    ax.plot(STEP_TIMES, tma, color=colors[alpha3], linewidth=1.3, label=label)
ax.set_xlabel('Time')
ax.set_ylabel(r'Mean $|S - Z|$')
ax.set_title('All α₃ values')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Right: α₃ > 0 only  (zoom in to see convergence)
ax = axes1[1]
for alpha3 in ALPHA3_NONZERO:
    tma = time_mean_abs[alpha3]
    # Add OU steady-state reference line
    ou_ref = ou_mean_abs(alpha3)
    ax.plot(STEP_TIMES, tma, color=colors[alpha3], linewidth=1.3,
            label=f'α₃={alpha3}')
    ax.axhline(ou_ref, color=colors[alpha3], linestyle=':', linewidth=0.7, alpha=0.6)

ax.axvline(STEADY_FRAC * TERMINAL_TIME, color='gray', linestyle='--',
           linewidth=0.9, label='Steady-state window start')
ax.set_xlabel('Time')
ax.set_title('α₃ > 0 only  (dotted = OU theory)')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
path1 = os.path.join(FIGURES_DIR, 'mispricing_time_evolution.png')
fig1.savefig(path1, dpi=150, bbox_inches='tight')
print(f'Figure 1 saved to: {path1}')


# -------------------------------------------------------------------------
# Figure 2 — Steady-state E[|ε|] vs α₃  (log-log)
# -------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(
    r'Steady-state mispricing $E[|\varepsilon|]$ vs $\alpha_3$'
    '\n(last 50% of steps)',
    fontsize=12,
)

alpha3_plot   = np.array(ALPHA3_NONZERO, dtype=float)
sim_mean_abs  = np.array([steady_mean_abs[a] for a in ALPHA3_NONZERO])
sim_std       = np.array([steady_std[a]      for a in ALPHA3_NONZERO])
ou_mean_arr   = np.array([ou_mean_abs(a)     for a in ALPHA3_NONZERO])
ou_std_arr    = np.array([ou_std(a)          for a in ALPHA3_NONZERO])

# Left: log-log of E[|ε|]
ax = axes2[0]
ax.loglog(alpha3_plot, sim_mean_abs,  'o-', color='steelblue',
          linewidth=1.8, markersize=6, label='Simulated E[|ε|]')
ax.loglog(alpha3_plot, ou_mean_arr,   '--', color='coral',
          linewidth=1.6, label=r'OU theory: $7.98/\sqrt{\alpha_3}$')
ax.set_xlabel(r'$\alpha_3$')
ax.set_ylabel(r'Steady-state $E[|\varepsilon|]$')
ax.set_title(r'$E[|\varepsilon|]$ vs $\alpha_3$ (log-log)')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

# Right: log-log of Std(ε)
ax = axes2[1]
ax.loglog(alpha3_plot, sim_std,    'o-', color='steelblue',
          linewidth=1.8, markersize=6, label=r'Simulated $\sigma(\varepsilon)$')
ax.loglog(alpha3_plot, ou_std_arr, '--', color='coral',
          linewidth=1.6, label=r'OU theory: $10/\sqrt{\alpha_3}$')
ax.set_xlabel(r'$\alpha_3$')
ax.set_ylabel(r'Steady-state $\sigma(\varepsilon)$')
ax.set_title(r'$\sigma(\varepsilon)$ vs $\alpha_3$ (log-log)')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

plt.tight_layout()
path2 = os.path.join(FIGURES_DIR, 'mispricing_vs_alpha3.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
print(f'Figure 2 saved to: {path2}')


# -------------------------------------------------------------------------
# Figure 3 — Steady-state distribution of ε  (box plots per α₃)
# -------------------------------------------------------------------------

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle(
    r'Steady-state distribution of $\varepsilon = S - Z$'
    f'\n(last {int(STEADY_FRAC*100)}% of steps, all trajectories)',
    fontsize=12,
)

# Left: include α₃=0 (will have very wide distribution)
ss_data_all = [mispricing_series[a][:, STEADY_START:].ravel()
               for a in ALPHA3_VALUES]
ax = axes3[0]
ax.boxplot(ss_data_all, tick_labels=[str(a) for a in ALPHA3_VALUES],
           showfliers=False, patch_artist=True,
           boxprops=dict(facecolor='lightsteelblue', alpha=0.7))
ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
ax.set_xlabel(r'$\alpha_3$')
ax.set_ylabel(r'$\varepsilon = S - Z$')
ax.set_title('All α₃ values')
ax.tick_params(axis='x', labelrotation=45)
ax.grid(True, alpha=0.3)

# Right: α₃ > 0 only (zoom to see structure)
ss_data_nz = [mispricing_series[a][:, STEADY_START:].ravel()
              for a in ALPHA3_NONZERO]
ax = axes3[1]
ax.boxplot(ss_data_nz, tick_labels=[str(a) for a in ALPHA3_NONZERO],
           showfliers=False, patch_artist=True,
           boxprops=dict(facecolor='lightsteelblue', alpha=0.7))
ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
ax.set_xlabel(r'$\alpha_3$')
ax.set_title('α₃ > 0 only')
ax.tick_params(axis='x', labelrotation=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
path3 = os.path.join(FIGURES_DIR, 'mispricing_distribution.png')
fig3.savefig(path3, dpi=150, bbox_inches='tight')
print(f'Figure 3 saved to: {path3}')

plt.show()

# ============================================================================
# Summary table
# ============================================================================

print()
print(f"{'alpha3':>8}  {'sim E[|ε|]':>12}  {'sim std(ε)':>12}  "
      f"{'OU E[|ε|]':>12}  {'OU std(ε)':>12}  {'ratio (sim/OU)':>15}")
print('-' * 80)

for alpha3 in ALPHA3_VALUES:
    sim_m = steady_mean_abs[alpha3]
    sim_s = steady_std[alpha3]
    ou_m  = ou_mean_abs(alpha3)
    ou_s  = ou_std(alpha3)
    ratio = sim_m / ou_m if (alpha3 > 0 and not np.isnan(ou_m)) else float('nan')
    ou_m_str = f'{ou_m:.4f}' if not np.isnan(ou_m) else '  ∞'
    ou_s_str = f'{ou_s:.4f}' if not np.isnan(ou_s) else '  ∞'
    ratio_str = f'{ratio:.3f}' if not np.isnan(ratio) else '  —'
    print(f"{alpha3:>8}  {sim_m:>12.4f}  {sim_s:>12.4f}  "
          f"{ou_m_str:>12}  {ou_s_str:>12}  {ratio_str:>15}")
