"""
Baseline agent simulation: RandomAgent vs UniformAllocationAgent under varying
market conditions.

Sweeps over:
  - tau:    LP position width (2, 5, 10, 20)
  - alpha3: orderflow toxicity in PoissonLinearArrivalModel (0, 500, 2000, 5000, 10000)

Hypothesis: as alpha3 increases, LP returns decrease (adverse selection),
and the effect may differ between agents.

Figures are saved to notebooks/figures/.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.agents.BaselineAgents import RandomAgent, UniformAllocationAgent
from SAiFE_gym.rewards.RewardFunctions import PnL

# ============================================================================
# Simulation Parameters
# ============================================================================
SEED = 42
TERMINAL_TIME = 1.0
N_STEPS = 200
INITIAL_WEALTH = 1e6
NUM_TRAJECTORIES = 200

INITIAL_PRICE = 100.0
DRIFT = 0.0
VOLATILITY = 2.0
FEE_TIER = 0.003
NUM_TICKS = 1000
LIQUIDITY_SCALE = 1e6

TAU_VALUES = [2, 5, 10, 20]
ALPHA3_VALUES = [0.0, 500.0, 2000.0, 5000.0, 10000.0]

# Fixed arrival model params (alpha2 = 0 per spec)
ALPHA0 = np.array([10.0,  10.0])
ALPHA1 = np.array([100.0, 100.0])
ALPHA2 = np.array([0.0,   0.0])


# ============================================================================
# Factory and episode runner
# ============================================================================

def make_env(tau: int, alpha3: float, num_trajectories: int, seed: int) -> AMMEnvironment:
    """Build a full AMMEnvironment for the given tau and alpha3."""
    step_size = TERMINAL_TIME / N_STEPS

    alpha = np.array([
        ALPHA0,
        ALPHA1,
        ALPHA2,
        [alpha3, alpha3],
    ])

    midprice_model = BrownianMotionMidpriceModel(
        drift=DRIFT,
        volatility=VOLATILITY,
        initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed,
    )

    arrival_model = PoissonLinearArrivalModel(
        alpha=alpha,
        liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1,
    )

    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=FEE_TIER,
        tau=tau,
        num_ticks=NUM_TICKS,
        exponential_value=1.0001,
        initial_wealth=INITIAL_WEALTH,
        seed=seed + 2,
    )

    env = AMMEnvironment(
        terminal_time=TERMINAL_TIME,
        n_steps=N_STEPS,
        model_dynamics=model_dynamics,
        reward_function=PnL(),
        num_trajectories=num_trajectories,
        seed=seed,
    )

    return env


def run_episode(env: AMMEnvironment, agent, n_steps: int) -> np.ndarray:
    """Run one episode and return final wealth per trajectory.

    PnL.calculate() returns V(t+1) - V(t) each step, so
    sum(rewards) = V(T) - V(0) = V(T) - INITIAL_WEALTH.
    """
    obs, _ = env.reset()
    cumulative_reward = np.zeros(env.num_trajectories)

    for _ in range(n_steps):
        action = agent.get_action(obs)
        obs, rewards, terminated, truncated, _ = env.step(action)
        cumulative_reward += rewards

    return INITIAL_WEALTH + cumulative_reward  # final wealth, shape (num_trajectories,)


# ============================================================================
# Main simulation loop
# ============================================================================

AGENTS = [
    (UniformAllocationAgent, 'Uniform'),
    (RandomAgent,            'Random'),
]

results = {}  # key: (agent_name, tau, alpha3)

total_runs = len(TAU_VALUES) * len(ALPHA3_VALUES) * len(AGENTS)
run_idx = 0

print(f"Running {total_runs} configurations "
      f"({len(TAU_VALUES)} tau × {len(ALPHA3_VALUES)} alpha3 × {len(AGENTS)} agents), "
      f"{NUM_TRAJECTORIES} trajectories each.")
print()

for tau in TAU_VALUES:
    for alpha3 in ALPHA3_VALUES:
        env = make_env(tau, alpha3, NUM_TRAJECTORIES, seed=SEED)

        for AgentClass, name in AGENTS:
            run_idx += 1
            print(f"[{run_idx:2d}/{total_runs}] agent={name:<8s}  tau={tau:2d}  alpha3={alpha3:8.1f} ...",
                  end='', flush=True)

            agent = AgentClass(env)
            final_wealths = run_episode(env, agent, N_STEPS)

            results[(name, tau, alpha3)] = {
                'mean': np.mean(final_wealths),
                'std':  np.std(final_wealths),
            }

            delta = results[(name, tau, alpha3)]['mean'] - INITIAL_WEALTH
            print(f"  mean ΔW = {delta:+.0f}")

print()

# ============================================================================
# Plotting helpers
# ============================================================================

AGENT_COLORS = {'Uniform': '#1f77b4', 'Random': '#ff7f0e'}
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def _plot_mean_std(ax, x_vals, means, stds, label, color):
    ax.plot(x_vals, means, marker='o', label=label, color=color, linewidth=1.5)
    ax.fill_between(x_vals,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.15, color=color)


# ============================================================================
# Figure 1 — Effect of tau (one subplot per alpha3 value)
# ============================================================================

fig1, axes1 = plt.subplots(1, len(ALPHA3_VALUES), figsize=(5 * len(ALPHA3_VALUES), 5),
                           sharey=True)
fig1.suptitle('Effect of tau on final wealth\n(one panel per alpha3)', fontsize=13)

for col, alpha3 in enumerate(ALPHA3_VALUES):
    ax = axes1[col]

    for _, name in AGENTS:
        means = [results[(name, tau, alpha3)]['mean'] for tau in TAU_VALUES]
        stds  = [results[(name, tau, alpha3)]['std']  for tau in TAU_VALUES]
        _plot_mean_std(ax, TAU_VALUES, means, stds, label=name, color=AGENT_COLORS[name])

    ax.axhline(INITIAL_WEALTH, color='gray', linestyle='--', linewidth=0.8, label='Initial wealth')
    ax.set_title(f'alpha3 = {alpha3:.0f}')
    ax.set_xlabel('tau')
    if col == 0:
        ax.set_ylabel('Mean final wealth')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, 'effect_of_tau.png')
fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"Figure 1 saved to: {fig1_path}")


# ============================================================================
# Figure 2 — Effect of alpha3 (one subplot per tau value)
# ============================================================================

fig2, axes2 = plt.subplots(1, len(TAU_VALUES), figsize=(5 * len(TAU_VALUES), 5),
                           sharey=True)
fig2.suptitle('Effect of alpha3 (toxicity) on final wealth\n(one panel per tau)', fontsize=13)

for col, tau in enumerate(TAU_VALUES):
    ax = axes2[col]

    for _, name in AGENTS:
        means = [results[(name, tau, alpha3)]['mean'] for alpha3 in ALPHA3_VALUES]
        stds  = [results[(name, tau, alpha3)]['std']  for alpha3 in ALPHA3_VALUES]
        _plot_mean_std(ax, ALPHA3_VALUES, means, stds, label=name, color=AGENT_COLORS[name])

    ax.axhline(INITIAL_WEALTH, color='gray', linestyle='--', linewidth=0.8, label='Initial wealth')
    ax.set_title(f'tau = {tau}')
    ax.set_xlabel('alpha3')
    if col == 0:
        ax.set_ylabel('Mean final wealth')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'effect_of_alpha3.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Figure 2 saved to: {fig2_path}")

plt.show()

# ============================================================================
# Summary table
# ============================================================================

header = f"{'Agent':<12} | {'tau':>4} | {'alpha3':>8} | {'Mean Wealth':>15} | {'Std':>10}"
sep    = '-' * len(header)
print()
print(sep)
print(header)
print(sep)

for _, name in AGENTS:
    for tau in TAU_VALUES:
        for alpha3 in ALPHA3_VALUES:
            r = results[(name, tau, alpha3)]
            print(f"{name:<12} | {tau:>4} | {alpha3:>8.1f} | {r['mean']:>15,.2f} | {r['std']:>10,.0f}")

print(sep)
