"""
Cartea vs Uniform: Detailed Single Simulation Visualization

This notebook runs a single simulation with fixed parameters and creates detailed
visualizations showing:

1. Delta boundaries (δₗ*, δᵤ*) evolution for both agents
2. Position boundaries relative to midprice
3. PnL evolution comparison
4. Price path with position overlays
5. Fee rate evolution
6. Spread dynamics

Fixed parameters chosen to show interesting behavior:
- gamma = 0.005 (moderate risk aversion)
- alpha3 = 5000 (moderate toxicity)
- tau = 20 (wide position capability)
- drift = 0.02 (slight upward trend)
- volatility = 2.0
- 500 steps for detailed evolution
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import pandas as pd

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel, GeometricBrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.agents.BaselineAgents import CarteaPLAgent, DeployOnceAgent, UniformAllocationAgent
from SAiFE_gym.rewards.RewardFunctions import PnL

# ============================================================================
# Fixed Simulation Parameters
# ============================================================================

# Simulation setup
SEED = 123
TERMINAL_TIME = 1.0
N_STEPS = 2000 # More steps for detailed evolution
NUM_TRAJECTORIES = 1  # Single trajectory for clear visualization
INITIAL_WEALTH = 1000

# Market parameters
INITIAL_PRICE = 100.0
DRIFT = 0  # Slight upward trend to see strategy differences
VOLATILITY = 0.1
FEE_TIER = 0.003
NUM_TICKS = 3000
LIQUIDITY_SCALE = 1e4

# Agent parameters
GAMMA = 0.000005  # Moderate risk aversion
TAU = 20      # Wide position capability
ALPHA3 = 5000.0  # Moderate toxicity

# Fixed arrival model params
ALPHA0 = np.array([10.0,   10.0])
ALPHA1 = np.array([150.0, 150.0])
ALPHA2 = np.array([0.0, 0.0])
ALPHA3 = np.array([5000.0, 5000.0])
#ALPHA3 = np.array([10000.0,   0.0])

print(" Cartea vs Uniform: Detailed Visualization")
print("=" * 60)
print(f"Parameters: γ={GAMMA}, α3={ALPHA3}, τ={TAU}, μ={DRIFT}")
print(f"Simulation: {N_STEPS} steps, {TERMINAL_TIME} time units")
print()

# ============================================================================
# Environment Setup
# ============================================================================

def create_environment():
    """Create the simulation environment with fixed parameters."""
    step_size = TERMINAL_TIME / N_STEPS

    alpha = np.array([
        ALPHA0,
        ALPHA1,
        ALPHA2,
        ALPHA3,
    ])

    midprice_model = GeometricBrownianMotionMidpriceModel(
        drift=DRIFT,
        volatility=VOLATILITY,
        initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME,
        step_size=step_size,
        num_trajectories=NUM_TRAJECTORIES,
        seed=SEED,
    )

    arrival_model = PoissonLinearArrivalModel(
        alpha=alpha,
        liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size,
        num_trajectories=NUM_TRAJECTORIES,
        seed=SEED + 1,
    )

    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=NUM_TRAJECTORIES,
        fee_tier=FEE_TIER,
        tau=TAU,
        num_ticks=3000,
        exponential_value=1.0001,
        initial_wealth=INITIAL_WEALTH,
        seed=SEED + 2,
    )

    env = AMMEnvironment(
        terminal_time=TERMINAL_TIME,
        n_steps=N_STEPS,
        model_dynamics=model_dynamics,
        reward_function=PnL(initial_wealth=INITIAL_WEALTH),
        num_trajectories=NUM_TRAJECTORIES,
        seed=SEED,
    )

    return env

# ============================================================================
# Detailed Data Collection
# ============================================================================

class DetailedDataCollector:
    """Collect detailed step-by-step data for visualization."""

    def __init__(self, n_steps: int):
        self.n_steps = n_steps
        self.reset()

    def reset(self):
        """Initialize all tracking arrays."""
        # Common data
        self.time = np.zeros(self.n_steps)
        self.prices = np.zeros(self.n_steps)
        self.midprices = np.zeros(self.n_steps)
        self.rewards = np.zeros(self.n_steps)
        self.cumulative_pnl = np.zeros(self.n_steps)

        # Position data (convert ticks to prices for visualization)
        self.position_lower_prices = np.zeros(self.n_steps)
        self.position_upper_prices = np.zeros(self.n_steps)

        # Action data (tick offsets)
        self.action_lower = np.zeros(self.n_steps)
        self.action_upper = np.zeros(self.n_steps)

        # Cartea-specific data
        self.delta_lower = np.zeros(self.n_steps)
        self.delta_upper = np.zeros(self.n_steps)
        self.fee_rates = np.zeros(self.n_steps)

        self.step_count = 0

    def update(self, step_time, state, action, reward, agent, next_state=None):
        """Update tracking with current step data."""
        if self.step_count >= self.n_steps:
            return

        i = self.step_count

        # Common data
        self.time[i] = step_time
        self.prices[i] = state['sqrt_price'][0] ** 2
        self.midprices[i] = state['midprice'][0]
        self.rewards[i] = reward[0]

        # Cumulative PnL
        if i == 0:
            self.cumulative_pnl[i] = reward[0]
        else:
            self.cumulative_pnl[i] = self.cumulative_pnl[i-1] + reward[0]

        # Actions
        self.action_lower[i] = action[0, 0]
        self.action_upper[i] = action[0, 1]

        # Use actual LP position from next_state (reflects hold vs rebalance correctly)
        exponential_value = 1.0001
        post_state = next_state if next_state is not None else state
        lower_tick = post_state['lp_tick_lower'][0]
        upper_tick = post_state['lp_tick_upper'][0]

        self.position_lower_prices[i] = exponential_value ** lower_tick
        self.position_upper_prices[i] = exponential_value ** upper_tick

        # Cartea-specific data
        if hasattr(agent, 'calculate_dynamic_fee_rate'):
            self.fee_rates[i] = agent.calculate_dynamic_fee_rate(state)[0]

            if hasattr(agent, 'compute_optimal_deltas'):
                delta_l, delta_u = agent.compute_optimal_deltas(state)
                self.delta_lower[i] = delta_l[0]
                self.delta_upper[i] = delta_u[0]

        self.step_count += 1

def run_simulation_with_tracking(env, agent, agent_name):
    """Run simulation and collect detailed tracking data."""
    print(f" Running {agent_name} simulation...")

    collector = DetailedDataCollector(N_STEPS)
    state, _ = env.reset(seed=SEED)

    for step in range(N_STEPS):
        step_time = step * TERMINAL_TIME / N_STEPS
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Collect data (next_state has the actual LP position after hold/rebalance)
        collector.update(step_time, state, action, reward, agent, next_state=next_state)

        state = next_state
        if terminated[0]:
            break

    print(f" {agent_name} simulation completed")
    return collector

# ============================================================================
# Run Simulations
# ============================================================================

# Create environment
env = create_environment()

# Create agents
cartea_agent = CarteaPLAgent(env, gamma=GAMMA, seed=SEED)
uniform_agent = UniformAllocationAgent(env)
deploy_once_agent = DeployOnceAgent(env)

# Run simulations
cartea_data = run_simulation_with_tracking(env, cartea_agent, "Cartea")

# Reset environment for uniform agent
env.reset(seed=SEED)
uniform_data = run_simulation_with_tracking(env, uniform_agent, "Uniform")

# Reset environment for deploy-once agent
env.reset(seed=SEED)
deploy_once_data = run_simulation_with_tracking(env, deploy_once_agent, "DeployOnce")

print(f"\n Simulation Results Summary:")
print(f"Cartea     final PnL: {cartea_data.cumulative_pnl[-1]:+.0f}")
print(f"Uniform    final PnL: {uniform_data.cumulative_pnl[-1]:+.0f}")
print(f"DeployOnce final PnL: {deploy_once_data.cumulative_pnl[-1]:+.0f}")

# ============================================================================
# Visualization
# ============================================================================

print(f"Creating detailed visualizations...")

# Find the threshold condition: π_t > σ²/8
threshold = VOLATILITY**2 / 8
threshold_met = cartea_data.fee_rates > threshold
start_idx = 0

if np.any(threshold_met):
    start_idx = np.argmax(threshold_met)
    print(f"Threshold π_t > σ²/8 = {threshold:.6f} met at step {start_idx} (t={cartea_data.time[start_idx]:.3f})")
else:
    print(f"Threshold π_t > σ²/8 = {threshold:.6f} never met, showing full simulation")

# Set up the plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Color scheme
COLORS = {
    'cartea': '#1f77b4',      # Blue
    'uniform': '#ff7f0e',     # Orange
    'deploy_once': '#9467bd', # Purple
    'price': '#2ca02c',       # Green
    'midprice': '#d62728',    # Red
    'background': '#f0f0f0',  # Light gray
}

# Create figure with subplots
fig = plt.figure(figsize=(20, 16))

# Create a grid layout: 3 rows, 2 columns with different sizes
gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1.5, 1.5, 1.5], hspace=0.3, wspace=0.2)

# ============================================================================
# Plot 1: Price Evolution with Position Bands (Top, spanning both columns)
# ============================================================================

ax1 = fig.add_subplot(gs[0, 0])

# Plot price paths from threshold onwards
ax1.plot(cartea_data.time[start_idx:], cartea_data.prices[start_idx:], color=COLORS['price'], linewidth=2, label='Pool Price', alpha=0.8)
ax1.plot(cartea_data.time[start_idx:], cartea_data.midprices[start_idx:], color=COLORS['midprice'], linewidth=2, label='Market Midprice', alpha=0.8)

# Plot position bands using fill_between from threshold onwards
ax1.fill_between(cartea_data.time[start_idx:],
                cartea_data.position_lower_prices[start_idx:], cartea_data.position_upper_prices[start_idx:],
                alpha=0.15, color=COLORS['cartea'], label='Cartea Position Range')

ax1.fill_between(uniform_data.time[start_idx:],
                uniform_data.position_lower_prices[start_idx:], uniform_data.position_upper_prices[start_idx:],
                alpha=0.15, color=COLORS['uniform'], label='Uniform Position Range')

#ax1.fill_between(deploy_once_data.time[start_idx:],
#                deploy_once_data.position_lower_prices[start_idx:], deploy_once_data.position_upper_prices[start_idx:],
#                alpha=0.15, color=COLORS['deploy_once'], label='DeployOnce Position Range')

# Add position boundary lines from threshold onwards
ax1.plot(cartea_data.time[start_idx:], cartea_data.position_lower_prices[start_idx:], '--', color=COLORS['cartea'], alpha=0.6, linewidth=1)
ax1.plot(cartea_data.time[start_idx:], cartea_data.position_upper_prices[start_idx:], '--', color=COLORS['cartea'], alpha=0.6, linewidth=1)
ax1.plot(uniform_data.time[start_idx:], uniform_data.position_lower_prices[start_idx:], '--', color=COLORS['uniform'], alpha=0.6, linewidth=1)
ax1.plot(uniform_data.time[start_idx:], uniform_data.position_upper_prices[start_idx:], '--', color=COLORS['uniform'], alpha=0.6, linewidth=1)
#ax1.plot(deploy_once_data.time[start_idx:], deploy_once_data.position_lower_prices[start_idx:], '--', color=COLORS['deploy_once'], alpha=0.6, linewidth=1)
#ax1.plot(deploy_once_data.time[start_idx:], deploy_once_data.position_upper_prices[start_idx:], '--', color=COLORS['deploy_once'], alpha=0.6, linewidth=1)

ax1.set_ylabel('Price')
ax1.set_title(f'Price Evolution with LP Position Ranges\n(γ={GAMMA}, α3={ALPHA3}, τ={TAU}, μ={DRIFT})', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# ============================================================================
# Plot 2: PnL Evolution (Bottom left)
# ============================================================================

ax2 = fig.add_subplot(gs[0, 1])

ax2.plot(cartea_data.time[start_idx:], cartea_data.cumulative_pnl[start_idx:], color=COLORS['cartea'], linewidth=2.5, label='Cartea')
ax2.plot(uniform_data.time[start_idx:], uniform_data.cumulative_pnl[start_idx:], color=COLORS['uniform'], linewidth=2.5, label='Uniform')
ax2.plot(deploy_once_data.time[start_idx:], deploy_once_data.cumulative_pnl[start_idx:], color=COLORS['deploy_once'], linewidth=2.5, label='DeployOnce')

ax2.axhline(0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
ax2.set_xlabel('Time')
ax2.set_ylabel('Cumulative PnL')
ax2.set_title('PnL Evolution Comparison', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============================================================================
# Plot 3: Delta Evolution (Bottom right) - Cartea Only
# ============================================================================

ax3 = fig.add_subplot(gs[1, 0])

ax3.plot(cartea_data.time[start_idx:], cartea_data.delta_lower[start_idx:], label='δₗ* (Lower)', color='red', linewidth=2, alpha=0.5)
ax3.plot(cartea_data.time[start_idx:], cartea_data.delta_upper[start_idx:], label='δᵤ* (Upper)', color='blue', linewidth=2, alpha=0.5)
ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)

ax3.set_xlabel('Time')
ax3.set_ylabel('Delta Values')
ax3.set_title('Cartea Optimal Deltas Evolution', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============================================================================
# Plot 4: Position Width (Spread) Comparison
# ============================================================================

ax4 = fig.add_subplot(gs[1, 1])

cartea_spreads = cartea_data.action_upper - cartea_data.action_lower
uniform_spreads = uniform_data.action_upper - uniform_data.action_lower
deploy_once_spreads = deploy_once_data.action_upper - deploy_once_data.action_lower

ax4.plot(cartea_data.time[start_idx:], cartea_spreads[start_idx:], color=COLORS['cartea'], linewidth=2.5, label='Cartea')
ax4.plot(uniform_data.time[start_idx:], uniform_spreads[start_idx:], color=COLORS['uniform'], linewidth=2.5, label='Uniform')
ax4.plot(deploy_once_data.time[start_idx:], deploy_once_spreads[start_idx:], color=COLORS['deploy_once'], linewidth=2.5, label='DeployOnce')

ax4.set_xlabel('Time')
ax4.set_ylabel('Position Width (ticks)')
ax4.set_title('Position Spread Evolution', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ============================================================================
# Plot 5: Fee Rate Evolution (Cartea)
# ============================================================================

ax5 = fig.add_subplot(gs[2, 0])

ax5.plot(cartea_data.time, cartea_data.fee_rates * 100, color=COLORS['cartea'], linewidth=2.5)  # Convert to %
ax5.set_xlabel('Time')
ax5.set_ylabel('Fee Rate (%)')
ax5.set_title('Dynamic Fee Rate (π_t) Evolution', fontweight='bold')
ax5.grid(True, alpha=0.3)

# ============================================================================
# Plot 6: Position Offsets from Current Price
# ============================================================================

ax6 = fig.add_subplot(gs[2, 1])

# Show how far above/below current price the positions are
ax6.plot(cartea_data.time, cartea_data.action_lower, label='Cartea Lower Offset', color=COLORS['cartea'], linestyle='--', linewidth=2)
ax6.plot(cartea_data.time, cartea_data.action_upper, label='Cartea Upper Offset', color=COLORS['cartea'], linestyle='-', linewidth=2)
#ax6.plot(uniform_data.time, uniform_data.action_lower, label='Uniform Lower Offset', color=COLORS['uniform'], linestyle='--', linewidth=1.5, alpha=0.7)
#ax6.plot(uniform_data.time, uniform_data.action_upper, label='Uniform Upper Offset', color=COLORS['uniform'], linestyle='-', linewidth=1.5, alpha=0.7)

ax6.axhline(0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
ax6.set_xlabel('Time')
ax6.set_ylabel('Tick Offset from Current Price')
ax6.set_title('Position Boundaries Relative to Current Price', fontweight='bold')
ax6.legend(ncol=2)
ax6.grid(True, alpha=0.3)

# ============================================================================
# Add summary statistics box
# ============================================================================

# Create text summary
summary_text = f"""Simulation Summary:
• Cartea Final PnL: {cartea_data.cumulative_pnl[-1]:+.0f}
• Uniform Final PnL: {uniform_data.cumulative_pnl[-1]:+.0f}
• DeployOnce Final PnL: {deploy_once_data.cumulative_pnl[-1]:+.0f}

• Price Range: {np.min(cartea_data.prices):.1f} - {np.max(cartea_data.prices):.1f}
• Final Fee Rate: {cartea_data.fee_rates[-1]*100:.4f}%

• Avg Cartea Spread: {np.mean(cartea_spreads):.1f} ticks
• Avg Uniform Spread: {np.mean(uniform_spreads):.1f} ticks
• Avg DeployOnce Spread: {np.mean(deploy_once_spreads):.1f} ticks"""

# Add text box to upper right of first plot
#ax1.text(0.98, 0.98, summary_text, transform=ax1.transAxes, fontsize=9,
#         verticalalignment='top', horizontalalignment='right',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================================
# Save and display
# ============================================================================

plt.suptitle('Cartea vs Uniform: Detailed Single Simulation Analysis',
             fontsize=16, fontweight='bold', y=0.98)

# Create figures directory if it doesn't exist
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Save figure
figure_path = os.path.join(FIGURES_DIR, 'cartea_detailed_single_simulation.png')
plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f" Detailed visualization saved: {figure_path}")

plt.tight_layout()
#plt.show()

# ============================================================================
# Additional Analysis
# ============================================================================

print(f"\n Detailed Analysis:")
print("=" * 40)

print(f"Performance Metrics:")
final_pnl_diff = cartea_data.cumulative_pnl[-1] - uniform_data.cumulative_pnl[-1]
cartea_volatility = np.std(cartea_data.rewards)
uniform_volatility = np.std(uniform_data.rewards)
deploy_once_volatility = np.std(deploy_once_data.rewards)

print(f"  Cartea vs Uniform: {final_pnl_diff:+.0f} ({final_pnl_diff/abs(uniform_data.cumulative_pnl[-1])*100:+.1f}%)")
print(f"  Cartea PnL volatility: {cartea_volatility:.1f}")
print(f"  Uniform PnL volatility: {uniform_volatility:.1f}")
print(f"  DeployOnce PnL volatility: {deploy_once_volatility:.1f}")

print(f"\n Position Strategy Analysis:")
print(f"  Cartea avg spread: {np.mean(cartea_spreads):.1f} ± {np.std(cartea_spreads):.1f} ticks")
print(f"  Uniform avg spread: {np.mean(uniform_spreads):.1f} ± {np.std(uniform_spreads):.1f} ticks")
print(f"  DeployOnce avg spread: {np.mean(deploy_once_spreads):.1f} ± {np.std(deploy_once_spreads):.1f} ticks")

print(f"\n Fee Analysis:")
initial_fee_rate = cartea_data.fee_rates[0] * 100
final_fee_rate = cartea_data.fee_rates[-1] * 100
print(f"  Initial fee rate: {initial_fee_rate:.6f}%")
print(f"  Final fee rate: {final_fee_rate:.6f}%")
print(f"  Fee rate growth: {final_fee_rate - initial_fee_rate:+.6f}%")

print(f"\n Delta Strategy Analysis:")
avg_delta_lower = np.mean(cartea_data.delta_lower)
avg_delta_upper = np.mean(cartea_data.delta_upper)
print(f"  Average δₗ*: {avg_delta_lower:+.4f}")
print(f"  Average δᵤ*: {avg_delta_upper:+.4f}")
print(f"  Average spread (δₗ* + δᵤ*): {avg_delta_lower + avg_delta_upper:+.4f}")

print(f"Detailed visualization analysis complete!")