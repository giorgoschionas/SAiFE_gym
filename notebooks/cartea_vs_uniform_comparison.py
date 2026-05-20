"""
Cartea vs Uniform Agent Comparison: Optimal vs Heuristic Liquidity Provision

This notebook compares the CarteaPLAgent (optimal liquidity provision strategy from
Cartea et al.) against the UniformAllocationAgent (full-range heuristic).

Key comparisons:
  1. PnL performance across different market conditions
  2. Evolution of optimal deltas (δₗ*, δᵘ*) vs fixed strategy
  3. Spread adaptation vs fixed spread
  4. Fee rate responsiveness
  5. Risk-adjusted performance

Parameter sweeps:
  - gamma: Risk aversion (0.001, 0.005, 0.01, 0.05, 0.1)
  - alpha3: Orderflow toxicity (0, 1000, 5000, 10000)
  - tau: Position width (5, 10, 20)
  - drift: Market drift (-0.01, 0.0, +0.01)

Hypothesis: Cartea agent should outperform Uniform agent, especially in:
  - High fee environments (accumulated trading history)
  - Trending markets (non-zero drift)
  - High toxicity environments (alpha3 > 0)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.agents.BaselineAgents import CarteaPLAgent, UniformAllocationAgent
from SAiFE_gym.rewards.RewardFunctions import PnL

# ============================================================================
# Simulation Parameters
# ============================================================================
SEED = 42
TERMINAL_TIME = 1.0
N_STEPS = 200
INITIAL_WEALTH = 1000
NUM_TRAJECTORIES = 500 

# Market parameters
INITIAL_PRICE = 100.0
VOLATILITY = 0
FEE_TIER = 0.003
NUM_TICKS = 1000
LIQUIDITY_SCALE = 1e6

# Parameter sweeps
GAMMA_VALUES = [0.0005, 0.001]  # Risk aversion
ALPHA3_VALUES = [1000.0, 5000.0, 10000.0]   # Orderflow toxicity
TAU_VALUES = [10, 20]                    # Position width
DRIFT_VALUES = [-0.05, 0.0, 0.05]                # Market drift

# Fixed arrival model params
ALPHA0 = np.array([10.0,  10.0])   # Minimum intensity
ALPHA1 = np.array([100.0, 100.0]) # Baseline intensity
ALPHA2 = np.array([0.0,   0.0])   # Liquidity coefficient (disabled)

# ============================================================================
# Detailed tracking for time-series analysis
# ============================================================================

class DetailedTracker:
    """Track detailed metrics during simulation for analysis."""

    def __init__(self, n_steps: int, num_trajectories: int):
        self.n_steps = n_steps
        self.num_trajectories = num_trajectories
        self.reset()

    def reset(self):
        """Reset all tracking arrays."""
        shape = (self.n_steps, self.num_trajectories)

        # Common metrics
        self.rewards = np.zeros(shape)
        self.prices = np.zeros(shape)
        self.cumulative_pnl = np.zeros(shape)
        self.fee_rates = np.zeros(shape)

        # Cartea-specific metrics
        self.delta_lower = np.zeros(shape)
        self.delta_upper = np.zeros(shape)
        self.spreads = np.zeros(shape)
        self.position_lower = np.zeros(shape)
        self.position_upper = np.zeros(shape)

        # Actions (offsets)
        self.action_lower = np.zeros(shape)
        self.action_upper = np.zeros(shape)

        self.step_count = 0

    def update(self, state: dict, action: np.ndarray, reward: np.ndarray,
               agent=None, **kwargs):
        """Update tracking with current step data."""
        if self.step_count >= self.n_steps:
            return

        step = self.step_count

        # Common metrics
        self.prices[step] = state['sqrt_price'] ** 2
        self.rewards[step] = reward
        if step == 0:
            self.cumulative_pnl[step] = reward
        else:
            self.cumulative_pnl[step] = self.cumulative_pnl[step-1] + reward

        # Actions
        self.action_lower[step] = action[:, 0]
        self.action_upper[step] = action[:, 1]

        # Position bounds (absolute ticks)
        current_tick = state['current_tick']
        self.position_lower[step] = current_tick + action[:, 0]
        self.position_upper[step] = current_tick + action[:, 1]

        # Cartea-specific metrics
        if hasattr(agent, 'calculate_dynamic_fee_rate'):
            self.fee_rates[step] = agent.calculate_dynamic_fee_rate(state)

            if hasattr(agent, 'compute_optimal_deltas'):
                delta_l, delta_u = agent.compute_optimal_deltas(state)
                self.delta_lower[step] = delta_l
                self.delta_upper[step] = delta_u
                self.spreads[step] = delta_l + delta_u

        # Additional metrics from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                getattr(self, key)[step] = value

        self.step_count += 1

# ============================================================================
# Environment factory
# ============================================================================

def make_env(tau: int, alpha3: float, drift: float, num_trajectories: int,
             seed: int) -> AMMEnvironment:
    """Build AMMEnvironment with specified parameters."""
    step_size = TERMINAL_TIME / N_STEPS

    alpha = np.array([
        ALPHA0,
        ALPHA1,
        ALPHA2,
        [alpha3, alpha3],
    ])

    midprice_model = BrownianMotionMidpriceModel(
        drift=drift,
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
        reward_function=PnL(initial_wealth=INITIAL_WEALTH),
        num_trajectories=num_trajectories,
        seed=seed,
    )

    return env

# ============================================================================
# Simulation runner with detailed tracking
# ============================================================================

def run_detailed_episode(env: AMMEnvironment, agent, tracker: DetailedTracker) -> Dict:
    """Run episode with detailed tracking of agent behavior."""
    obs, _ = env.reset()
    tracker.reset()

    for step in range(N_STEPS):
        action = agent.get_action(obs)
        next_obs, rewards, terminated, truncated, _ = env.step(action)

        # Track this step
        tracker.update(obs, action, rewards, agent)

        obs = next_obs

        if terminated[0]:
            break

    # Final metrics
    final_wealth = INITIAL_WEALTH + tracker.cumulative_pnl[-1]

    return {
        'final_wealth_mean': np.mean(final_wealth),
        'final_wealth_std': np.std(final_wealth),
        'total_pnl_mean': np.mean(tracker.cumulative_pnl[-1]),
        'total_pnl_std': np.std(tracker.cumulative_pnl[-1]),
        'max_drawdown_mean': np.mean(np.min(tracker.cumulative_pnl, axis=0)),
        'sharpe_ratio_mean': np.mean(np.mean(tracker.rewards, axis=0) / (np.std(tracker.rewards, axis=0) + 1e-8)),
        'tracker': tracker,
    }

# ============================================================================
# Main simulation loop
# ============================================================================

print("🚀 Starting Cartea vs Uniform Agent Comparison")
print("=" * 70)

# Results storage
results = {}  # key: (agent_name, gamma, alpha3, tau, drift)
detailed_results = {}  # Store one detailed run per configuration for plots

# Baseline configuration for detailed analysis
BASELINE_CONFIG = {
    'gamma': 0.005,   # Updated to match GAMMA_VALUES
    'alpha3': 1000.0,
    'tau': 10,
    'drift': 0.0
}

total_runs = len(GAMMA_VALUES) * len(ALPHA3_VALUES) * len(TAU_VALUES) * len(DRIFT_VALUES) * 2
run_idx = 0

print(f"Running {total_runs} configurations:")
print(f"  - Gamma values: {GAMMA_VALUES}")
print(f"  - Alpha3 values: {ALPHA3_VALUES}")
print(f"  - Tau values: {TAU_VALUES}")
print(f"  - Drift values: {DRIFT_VALUES}")
print(f"  - {NUM_TRAJECTORIES} trajectories per run")
print()

for gamma in GAMMA_VALUES:
    for alpha3 in ALPHA3_VALUES:
        for tau in TAU_VALUES:
            for drift in DRIFT_VALUES:

                # Create environment
                env = make_env(tau, alpha3, drift, NUM_TRAJECTORIES, SEED)

                # Test both agents
                agents = [
                    (CarteaPLAgent(env, gamma=gamma, seed=SEED), f"Cartea(γ={gamma})"),
                    (UniformAllocationAgent(env), "Uniform")
                ]

                for agent, agent_name in agents:
                    run_idx += 1
                    print(f"[{run_idx:3d}/{total_runs}] {agent_name:<15} "
                          f"γ={gamma:.3f} α3={alpha3:5.0f} τ={tau:2d} μ={drift:+.2f} ...",
                          end='', flush=True)

                    # Run simulation
                    tracker = DetailedTracker(N_STEPS, NUM_TRAJECTORIES)
                    result = run_detailed_episode(env, agent, tracker)

                    # Store results
                    key = (agent_name.split('(')[0], gamma, alpha3, tau, drift)
                    results[key] = result

                    # Store detailed results for baseline config
                    config_matches_baseline = (
                        gamma == BASELINE_CONFIG['gamma'] and
                        alpha3 == BASELINE_CONFIG['alpha3'] and
                        tau == BASELINE_CONFIG['tau'] and
                        drift == BASELINE_CONFIG['drift']
                    )
                    if config_matches_baseline:
                        detailed_results[agent_name.split('(')[0]] = result

                    pnl = result['total_pnl_mean']
                    print(f" PnL={pnl:+8.0f}")

print("\n✅ Simulation completed!")

# ============================================================================
# Analysis and Visualization
# ============================================================================

print("\n📊 Creating visualizations...")

# Create figures directory
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color scheme
COLORS = {
    'Cartea': '#1f77b4',   # Blue
    'Uniform': '#ff7f0e',  # Orange
}

# ============================================================================
# Figure 1: PnL comparison across risk aversion (gamma)
# ============================================================================

fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
fig1.suptitle('PnL vs Risk Aversion (γ) - Cartea vs Uniform\n(Fixed: α3=1000, τ=10, μ=0)', fontsize=14)

# Use baseline config except varying gamma
fixed_alpha3, fixed_tau, fixed_drift = BASELINE_CONFIG['alpha3'], BASELINE_CONFIG['tau'], BASELINE_CONFIG['drift']

for ax, metric in zip(axes1.flat, ['total_pnl_mean', 'total_pnl_std', 'max_drawdown_mean', 'sharpe_ratio_mean']):

    cartea_values = [results[('Cartea', g, fixed_alpha3, fixed_tau, fixed_drift)][metric] for g in GAMMA_VALUES]
    uniform_values = [results[('Uniform', g, fixed_alpha3, fixed_tau, fixed_drift)][metric] for g in GAMMA_VALUES]

    ax.plot(GAMMA_VALUES, cartea_values, 'o-', label='Cartea', color=COLORS['Cartea'], linewidth=2)
    ax.plot(GAMMA_VALUES, uniform_values, 's-', label='Uniform', color=COLORS['Uniform'], linewidth=2)

    ax.set_xlabel('Risk Aversion (γ)')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, 'pnl_vs_gamma.png')
fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"📈 Figure 1 saved: {fig1_path}")

# ============================================================================
# Figure 2: Performance across market toxicity (alpha3)
# ============================================================================

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('Performance vs Order Flow Toxicity (α3)\n(Cartea with γ=0.01 vs Uniform)', fontsize=14)

for col, tau in enumerate(TAU_VALUES):
    ax = axes2[col]

    cartea_pnl = [results[('Cartea', BASELINE_CONFIG['gamma'], a3, tau, BASELINE_CONFIG['drift'])]['total_pnl_mean']
                  for a3 in ALPHA3_VALUES]
    cartea_std = [results[('Cartea', BASELINE_CONFIG['gamma'], a3, tau, BASELINE_CONFIG['drift'])]['total_pnl_std']
                  for a3 in ALPHA3_VALUES]

    uniform_pnl = [results[('Uniform', BASELINE_CONFIG['gamma'], a3, tau, BASELINE_CONFIG['drift'])]['total_pnl_mean']
                   for a3 in ALPHA3_VALUES]
    uniform_std = [results[('Uniform', BASELINE_CONFIG['gamma'], a3, tau, BASELINE_CONFIG['drift'])]['total_pnl_std']
                   for a3 in ALPHA3_VALUES]

    ax.errorbar(ALPHA3_VALUES, cartea_pnl, yerr=cartea_std,
                label='Cartea', color=COLORS['Cartea'], marker='o', linewidth=2)
    ax.errorbar(ALPHA3_VALUES, uniform_pnl, yerr=uniform_std,
                label='Uniform', color=COLORS['Uniform'], marker='s', linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'τ = {tau}')
    ax.set_xlabel('α3 (Toxicity)')
    if col == 0:
        ax.set_ylabel('Mean Total PnL')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'pnl_vs_toxicity.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"📈 Figure 2 saved: {fig2_path}")

# ============================================================================
# Figure 3: Time-series analysis (baseline configuration)
# ============================================================================

if 'Cartea' in detailed_results and 'Uniform' in detailed_results:

    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle(f'Time Series Comparison - Baseline Configuration\n'
                  f'γ={BASELINE_CONFIG["gamma"]}, α3={BASELINE_CONFIG["alpha3"]}, '
                  f'τ={BASELINE_CONFIG["tau"]}, μ={BASELINE_CONFIG["drift"]}', fontsize=14)

    time_steps = np.arange(N_STEPS) * TERMINAL_TIME / N_STEPS

    cartea_tracker = detailed_results['Cartea']['tracker']
    uniform_tracker = detailed_results['Uniform']['tracker']

    # Plot 1: Cumulative PnL evolution
    ax = axes3[0, 0]
    cartea_pnl_mean = np.mean(cartea_tracker.cumulative_pnl, axis=1)
    cartea_pnl_std = np.std(cartea_tracker.cumulative_pnl, axis=1)
    uniform_pnl_mean = np.mean(uniform_tracker.cumulative_pnl, axis=1)
    uniform_pnl_std = np.std(uniform_tracker.cumulative_pnl, axis=1)

    ax.plot(time_steps, cartea_pnl_mean, label='Cartea', color=COLORS['Cartea'], linewidth=2)
    ax.fill_between(time_steps, cartea_pnl_mean - cartea_pnl_std, cartea_pnl_mean + cartea_pnl_std,
                    alpha=0.2, color=COLORS['Cartea'])

    ax.plot(time_steps, uniform_pnl_mean, label='Uniform', color=COLORS['Uniform'], linewidth=2)
    ax.fill_between(time_steps, uniform_pnl_mean - uniform_pnl_std, uniform_pnl_mean + uniform_pnl_std,
                    alpha=0.2, color=COLORS['Uniform'])

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Cumulative PnL')
    ax.set_title('PnL Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Price evolution
    ax = axes3[0, 1]
    price_mean = np.mean(cartea_tracker.prices, axis=1)  # Same for both agents
    price_std = np.std(cartea_tracker.prices, axis=1)

    ax.plot(time_steps, price_mean, color='black', linewidth=2, label='Price')
    ax.fill_between(time_steps, price_mean - price_std, price_mean + price_std,
                    alpha=0.2, color='black')
    ax.set_ylabel('Price')
    ax.set_title('Price Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Fee rate evolution (Cartea only)
    ax = axes3[0, 2]
    fee_rate_mean = np.mean(cartea_tracker.fee_rates, axis=1)
    fee_rate_std = np.std(cartea_tracker.fee_rates, axis=1)

    ax.plot(time_steps, fee_rate_mean * 100, color=COLORS['Cartea'], linewidth=2)  # Convert to percentage
    ax.fill_between(time_steps, (fee_rate_mean - fee_rate_std) * 100, (fee_rate_mean + fee_rate_std) * 100,
                    alpha=0.2, color=COLORS['Cartea'])
    ax.set_ylabel('Fee Rate (%)')
    ax.set_title('Dynamic Fee Rate (π_t)')
    ax.grid(True, alpha=0.3)

    # Plot 4: Optimal deltas evolution (Cartea only)
    ax = axes3[1, 0]
    delta_lower_mean = np.mean(cartea_tracker.delta_lower, axis=1)
    delta_upper_mean = np.mean(cartea_tracker.delta_upper, axis=1)

    ax.plot(time_steps, delta_lower_mean, label='δₗ*', color='red', linewidth=2)
    ax.plot(time_steps, delta_upper_mean, label='δᵤ*', color='blue', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Delta Values')
    ax.set_title('Optimal Delta Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Action comparison
    ax = axes3[1, 1]
    cartea_lower_mean = np.mean(cartea_tracker.action_lower, axis=1)
    cartea_upper_mean = np.mean(cartea_tracker.action_upper, axis=1)
    uniform_lower_mean = np.mean(uniform_tracker.action_lower, axis=1)
    uniform_upper_mean = np.mean(uniform_tracker.action_upper, axis=1)

    ax.plot(time_steps, cartea_lower_mean, '--', color=COLORS['Cartea'], label='Cartea lower', linewidth=2)
    ax.plot(time_steps, cartea_upper_mean, '-', color=COLORS['Cartea'], label='Cartea upper', linewidth=2)
    ax.plot(time_steps, uniform_lower_mean, '--', color=COLORS['Uniform'], label='Uniform lower', linewidth=1)
    ax.plot(time_steps, uniform_upper_mean, '-', color=COLORS['Uniform'], label='Uniform upper', linewidth=1)

    ax.set_ylabel('Tick Offset')
    ax.set_title('Position Bounds (Actions)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Spread comparison
    ax = axes3[1, 2]
    cartea_spread_mean = np.mean(cartea_tracker.action_upper - cartea_tracker.action_lower, axis=1)
    uniform_spread_mean = np.mean(uniform_tracker.action_upper - uniform_tracker.action_lower, axis=1)
    cartea_spread_std = np.std(cartea_tracker.action_upper - cartea_tracker.action_lower, axis=1)

    ax.plot(time_steps, cartea_spread_mean, color=COLORS['Cartea'], linewidth=2, label='Cartea')
    ax.fill_between(time_steps, cartea_spread_mean - cartea_spread_std, cartea_spread_mean + cartea_spread_std,
                    alpha=0.2, color=COLORS['Cartea'])
    ax.plot(time_steps, uniform_spread_mean, color=COLORS['Uniform'], linewidth=2, label='Uniform')

    ax.set_ylabel('Spread (ticks)')
    ax.set_xlabel('Time')
    ax.set_title('Position Width Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes3[1, :]:
        ax.set_xlabel('Time')

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, 'time_series_comparison.png')
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"📈 Figure 3 saved: {fig3_path}")

# ============================================================================
# Figure 4: Drift sensitivity analysis
# ============================================================================

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
fig4.suptitle('Sensitivity to Market Drift (μ)\n(Fixed: γ=0.01, α3=1000, τ=10)', fontsize=14)

# PnL vs drift
ax = axes4[0]
cartea_pnl = [results[('Cartea', BASELINE_CONFIG['gamma'], BASELINE_CONFIG['alpha3'],
                      BASELINE_CONFIG['tau'], drift)]['total_pnl_mean'] for drift in DRIFT_VALUES]
uniform_pnl = [results[('Uniform', BASELINE_CONFIG['gamma'], BASELINE_CONFIG['alpha3'],
                       BASELINE_CONFIG['tau'], drift)]['total_pnl_mean'] for drift in DRIFT_VALUES]

ax.plot(DRIFT_VALUES, cartea_pnl, 'o-', label='Cartea', color=COLORS['Cartea'], linewidth=2, markersize=8)
ax.plot(DRIFT_VALUES, uniform_pnl, 's-', label='Uniform', color=COLORS['Uniform'], linewidth=2, markersize=8)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Market Drift (μ)')
ax.set_ylabel('Mean Total PnL')
ax.set_title('PnL vs Market Drift')
ax.legend()
ax.grid(True, alpha=0.3)

# Sharpe ratio vs drift
ax = axes4[1]
cartea_sharpe = [results[('Cartea', BASELINE_CONFIG['gamma'], BASELINE_CONFIG['alpha3'],
                         BASELINE_CONFIG['tau'], drift)]['sharpe_ratio_mean'] for drift in DRIFT_VALUES]
uniform_sharpe = [results[('Uniform', BASELINE_CONFIG['gamma'], BASELINE_CONFIG['alpha3'],
                          BASELINE_CONFIG['tau'], drift)]['sharpe_ratio_mean'] for drift in DRIFT_VALUES]

ax.plot(DRIFT_VALUES, cartea_sharpe, 'o-', label='Cartea', color=COLORS['Cartea'], linewidth=2, markersize=8)
ax.plot(DRIFT_VALUES, uniform_sharpe, 's-', label='Uniform', color=COLORS['Uniform'], linewidth=2, markersize=8)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Market Drift (μ)')
ax.set_ylabel('Mean Sharpe Ratio')
ax.set_title('Risk-Adjusted Performance')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, 'drift_sensitivity.png')
fig4.savefig(fig4_path, dpi=150, bbox_inches='tight')
print(f"📈 Figure 4 saved: {fig4_path}")

# ============================================================================
# Summary table
# ============================================================================

print("\n📊 Performance Summary")
print("=" * 70)

# Create summary DataFrame for easy analysis
summary_data = []
for key, result in results.items():
    agent, gamma, alpha3, tau, drift = key
    summary_data.append({
        'Agent': agent,
        'Gamma': gamma,
        'Alpha3': alpha3,
        'Tau': tau,
        'Drift': drift,
        'Mean_PnL': result['total_pnl_mean'],
        'Std_PnL': result['total_pnl_std'],
        'Sharpe': result['sharpe_ratio_mean'],
        'Max_DD': result['max_drawdown_mean'],
    })

df = pd.DataFrame(summary_data)

# Best performing configurations
print("🏆 Top 10 Configurations by Mean PnL:")
top_10 = df.nlargest(10, 'Mean_PnL')
for idx, row in top_10.iterrows():
    print(f"{row['Agent']:>7} γ={row['Gamma']:5.3f} α3={row['Alpha3']:5.0f} "
          f"τ={row['Tau']:2.0f} μ={row['Drift']:+5.2f} → PnL={row['Mean_PnL']:+8.0f}")

# Cartea vs Uniform head-to-head
print(f"\n🥊 Head-to-Head Comparison (Cartea wins):")
cartea_wins = 0
total_comparisons = 0

for gamma in GAMMA_VALUES:
    for alpha3 in ALPHA3_VALUES:
        for tau in TAU_VALUES:
            for drift in DRIFT_VALUES:
                cartea_result = results[('Cartea', gamma, alpha3, tau, drift)]
                uniform_result = results[('Uniform', gamma, alpha3, tau, drift)]

                if cartea_result['total_pnl_mean'] > uniform_result['total_pnl_mean']:
                    cartea_wins += 1
                total_comparisons += 1

win_rate = cartea_wins / total_comparisons * 100
print(f"Cartea wins: {cartea_wins}/{total_comparisons} = {win_rate:.1f}%")

print(f"\n✅ Analysis completed! All figures saved to {FIGURES_DIR}/")
print("\n🎯 Key Findings:")
print("   1. Check PnL vs Risk Aversion plots for optimal γ values")
print("   2. Observe how Cartea adapts position width vs Uniform's fixed strategy")
print("   3. Analyze delta evolution to understand Cartea's decision making")
print("   4. Compare performance across different market conditions")

plt.show()