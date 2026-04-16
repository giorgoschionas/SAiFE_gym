"""
Training script for PolicyGradientAgent on SAiFE_gym AMM environment.

This script demonstrates how to train a policy gradient agent for liquidity provision
in a Uniswap V3-style concentrated liquidity AMM using SAiFE_gym's vectorized environment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.agents.PolicyGradientAgent import PolicyGradientAgent
from SAiFE_gym.rewards.RewardFunctions import PnL, RunningInventoryPenalty

# ============================================================================
# Training Configuration
# ============================================================================

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Environment parameters
TERMINAL_TIME = 1.0
N_STEPS = 300
NUM_TRAJECTORIES = 300  # Train on multiple parallel trajectories
INITIAL_WEALTH = 1000
TAU = 50  # LP position width

# Market parameters
INITIAL_PRICE = 200.0
DRIFT = 1  # Small positive drift to encourage asymmetric strategies
VOLATILITY = 2.0
FEE_TIER = 0.003

# Arrival model parameters (state-dependent)
ALPHA0 = np.array([10.0, 10.0])    # Minimum intensity
ALPHA1 = np.array([150.0, 150.0]) # Baseline intensity
ALPHA2 = np.array([0.0, 0.0])     # Liquidity coefficient (disabled)
ALPHA3 = np.array([5000.0, 5000.0])  # Arbitrage coefficient

# Training parameters
NUM_EPOCHS = 250
LEARNING_RATE = 2e-4
ACTION_STD_INIT = 1.8  # Higher std for exploration
ACTION_STD_DECAY = lambda t: max(0.3, ACTION_STD_INIT * (0.998 ** (t * 200)))  # Much slower decay

# ============================================================================
# Neural Network Architecture
# ============================================================================

def create_policy_network(input_size: int, hidden_size: int = 256, action_size: int = 2):
    """
    Create a deeper feedforward policy network for more sophisticated strategies.

    Args:
        input_size: Number of flattened state features
        hidden_size: Hidden layer size
        action_size: Output action dimensions (2 for [lower_offset, upper_offset])
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),  # Regularization to prevent overfitting

        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(hidden_size, hidden_size // 2),  # Gradually reduce size
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(hidden_size // 2, hidden_size // 4),  # Further reduction
        nn.ReLU(),

        nn.Linear(hidden_size // 4, action_size)
        # No final activation - sigmoid parameterization expects raw outputs in [-inf, +inf]
    )

class UnbiasedTwoStagePolicyNetwork(nn.Module):
    """
    UNBIASED policy network using two-stage sampling to eliminate width bias.

    Stage 1: Sample width uniformly from [1, 2*tau]
    Stage 2: Sample center uniformly from valid range given width

    IMPORTANT: Center range is constrained to ensure positions STRADDLE the current tick.
    This guarantees lower_offset < 0 < upper_offset, making the agent a true liquidity
    provider rather than allowing positions entirely above or below the current price.

    This ensures all valid position sizes have equal probability density,
    eliminating the bias toward wide strategies present in the original
    SigmoidConstrainedPolicyNetwork.
    """
    def __init__(self, base_network, tau: int = 100):
        super().__init__()
        self.base_network = base_network  # Expects outputs in [-inf, +inf] (no final activation)
        self.tau = tau

    def forward(self, x):
        # Base network outputs raw values in [-inf, +inf]
        raw_output = self.base_network(x)
        sigmoid_output = torch.sigmoid(raw_output)

        # Stage 1: Uniform width sampling [1, 2*tau]
        # Maps sigmoid[0] ∈ [0,1] → width ∈ [1, 2*tau]
        width = sigmoid_output[:, 0] * (2 * self.tau - 1) + 1

        # Stage 2: CORRECTED center sampling to ensure position straddles current tick
        # Ensure lower_offset < 0 < upper_offset (position straddles current price)
        half_width = width / 2

        # Calculate bounds ensuring position crosses current tick (offset 0)
        min_center = torch.maximum(
            torch.full_like(half_width, -self.tau) + half_width,  # Respect tau bound
            -half_width + 0.001                                   # Ensure lower < 0
        )
        max_center = torch.minimum(
            torch.full_like(half_width, self.tau) - half_width,   # Respect tau bound
            half_width - 0.001                                    # Ensure upper > 0
        )

        center_range = max_center - min_center

        # Ensure we have a valid range (handle edge cases for very wide positions)
        valid_range = center_range > 0
        center_range = torch.where(valid_range, center_range, torch.ones_like(center_range))

        # Maps sigmoid[1] ∈ [0,1] → center ∈ [min_center, max_center]
        center = min_center + sigmoid_output[:, 1] * center_range

        # Convert to offsets
        lower_offset = center - half_width  # Will be < 0
        upper_offset = center + half_width  # Will be > 0

        return torch.stack([lower_offset, upper_offset], dim=1)

# ============================================================================
# Environment Setup
# ============================================================================

def create_environment(num_trajectories: int, tau: int, seed: int = None):
    """Create SAiFE_gym environment with specified configuration."""

    # Stochastic processes
    midprice_model = BrownianMotionMidpriceModel(
        drift=DRIFT,
        volatility=VOLATILITY,
        initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME,
        step_size=TERMINAL_TIME / N_STEPS,
        num_trajectories=num_trajectories,
        seed=seed
    )

    arrival_model = PoissonLinearArrivalModel(
        alpha=np.column_stack([ALPHA0, ALPHA1, ALPHA2, ALPHA3]).T,
        step_size=TERMINAL_TIME / N_STEPS,
        num_trajectories=num_trajectories,
        seed=seed + 1 if seed else None
    )

    # Model dynamics
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        tau=tau,
        fee_tier=FEE_TIER,
        initial_wealth=INITIAL_WEALTH,
        seed=seed
    )

    # Reward function (basic PnL - change in portfolio value)
    reward_function = PnL(
        exponential_value=1.0001,
        initial_wealth=INITIAL_WEALTH
    )

    # Environment
    env = AMMEnvironment(
        terminal_time=TERMINAL_TIME,
        n_steps=N_STEPS,
        reward_function=reward_function,
        model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        seed=seed
    )

    return env

# ============================================================================
# Training Loop
# ============================================================================

def main():
    """Main training loop."""

    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create environment
    env = create_environment(NUM_TRAJECTORIES, TAU, SEED)

    # Get state size by creating dummy state
    dummy_state, _ = env.reset()

    # Create dummy agent to get input size
    dummy_policy = nn.Linear(7, 2)  # Temporary network
    temp_agent = PolicyGradientAgent(dummy_policy, env)
    input_size = temp_agent.input_size

    print(f"State input size: {input_size}")
    print(f"Action size: {env.action_space.shape[0]}")
    print(f"Training on {NUM_TRAJECTORIES} parallel trajectories")

    # Create policy network with UNBIASED two-stage parameterization
    base_network = create_policy_network(input_size, hidden_size=256, action_size=2)
    policy_network = UnbiasedTwoStagePolicyNetwork(base_network, tau=TAU)

    # Alternative: Use original biased network
    # policy_network = SigmoidConstrainedPolicyNetwork(base_network, tau=TAU)
    policy_network = policy_network.to(DEVICE)

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    # Create agent
    agent = PolicyGradientAgent(
        policy=policy_network,
        env=env,
        action_std=ACTION_STD_DECAY,
        optimizer=optimizer,
        lr_scheduler=scheduler
    )

    print("Starting training...")

    # Train agent
    losses, rewards = agent.train(num_epochs=NUM_EPOCHS, reporting_freq=50)

    print(f"Training completed!")
    print(f"Final average reward: {rewards[-1]:.4f}")

    # ========================================================================
    # Results Visualization
    # ========================================================================

    # Create plots directory
    os.makedirs("figures", exist_ok=True)

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Learning curve
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Policy Gradient Loss')
    ax1.grid(True)

    # Reward curve
    window = 50
    smoothed_rewards = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]

    ax2.plot(rewards, alpha=0.3, color='blue', label='Raw')
    ax2.plot(smoothed_rewards, color='red', label=f'Smoothed ({window})')
    ax2.set_title('Training Rewards')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Reward')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('figures/policy_gradient_training.png', dpi=150)
    #plt.show()

    # ========================================================================
    # Comprehensive Evaluation
    # ========================================================================

    print("Running comprehensive evaluation...")

    # Test 1: Large-scale evaluation with 1000 trajectories
    print("Testing on 1000 trajectories...")
    large_eval_env = create_environment(1000, TAU, SEED + 999)

    # Run single episode to completion
    state, _ = large_eval_env.reset()
    total_rewards = []
    terminated = np.zeros(1000, dtype=bool)

    while not np.any(terminated):
        action = agent.get_action(state, deterministic=True)
        state, reward, terminated, _, _ = large_eval_env.step(action)
        total_rewards.append(reward)

    # Calculate PnL statistics
    final_rewards = np.array(total_rewards)  # Shape: (num_steps, 1000)
    cumulative_pnl = np.sum(final_rewards, axis=0)  # Sum over time for each trajectory

    print(f"\n=== Large Scale Evaluation Results (1000 trajectories) ===")
    print(f"Episode length: {len(total_rewards)} steps")
    print(f"Average PnL: {np.mean(cumulative_pnl):.2f}")
    print(f"Std PnL: {np.std(cumulative_pnl):.2f}")
    print(f"Median PnL: {np.median(cumulative_pnl):.2f}")
    print(f"Min PnL: {np.min(cumulative_pnl):.2f}")
    print(f"Max PnL: {np.max(cumulative_pnl):.2f}")
    print(f"Profitable trajectories: {np.sum(cumulative_pnl > 0)}/1000 ({100*np.mean(cumulative_pnl > 0):.1f}%)")

    # Test 2: Detailed single trajectory analysis
    print("\nRunning detailed single trajectory analysis...")

    # Create single trajectory environment for detailed tracking
    single_env = create_environment(1, TAU, SEED + 1234)

    # Track detailed trajectory data
    trajectory_data = {
        'time': [],
        'midprice': [],
        'pool_price': [],  # sqrt_price^2
        'pool_sqrt_price': [],
        'current_tick': [],
        'lp_lower_tick': [],
        'lp_upper_tick': [],
        'lp_liquidity': [],
        'actions': [],
        'rewards': [],
        'cumulative_pnl': []
    }

    state, _ = single_env.reset()
    terminated = np.zeros(1, dtype=bool)
    cumulative_pnl_single = 0.0

    # Import key names for state access
    from SAiFE_gym.gym.index_names import (
        POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, ASSET_PRICE_KEY, TIME_KEY,
        LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY
    )

    while not np.any(terminated):
        # Record current state
        trajectory_data['time'].append(state[TIME_KEY][0])
        trajectory_data['midprice'].append(state[ASSET_PRICE_KEY][0])
        trajectory_data['pool_sqrt_price'].append(state[POOL_SQRT_PRICE_KEY][0])
        trajectory_data['pool_price'].append(state[POOL_SQRT_PRICE_KEY][0] ** 2)
        trajectory_data['current_tick'].append(state[POOL_CURRENT_TICK_KEY][0])
        trajectory_data['lp_lower_tick'].append(state[LP_TICK_LOWER_KEY][0])
        trajectory_data['lp_upper_tick'].append(state[LP_TICK_UPPER_KEY][0])
        trajectory_data['lp_liquidity'].append(state[LP_LIQUIDITY_KEY][0])

        # Get action
        action = agent.get_action(state, deterministic=True)
        trajectory_data['actions'].append(action[0].copy())  # First (and only) trajectory

        # Step environment
        state, reward, terminated, _, _ = single_env.step(action)

        # Record reward and cumulative PnL
        step_reward = reward[0]  # First (and only) trajectory
        cumulative_pnl_single += step_reward
        trajectory_data['rewards'].append(step_reward)
        trajectory_data['cumulative_pnl'].append(cumulative_pnl_single)

    print(f"Single trajectory final PnL: {cumulative_pnl_single:.2f}")

    # ========================================================================
    # Visualization
    # ========================================================================

    # Convert to arrays for easier plotting
    times = np.array(trajectory_data['time'])
    midprices = np.array(trajectory_data['midprice'])
    pool_prices = np.array(trajectory_data['pool_price'])
    current_ticks = np.array(trajectory_data['current_tick'])
    lp_lower_ticks = np.array(trajectory_data['lp_lower_tick'])
    lp_upper_ticks = np.array(trajectory_data['lp_upper_tick'])
    lp_liquidities = np.array(trajectory_data['lp_liquidity'])
    cumulative_pnls = np.array(trajectory_data['cumulative_pnl'])

    # Convert ticks to prices for visualization
    exponential_value = 1.0001
    lp_lower_prices = exponential_value ** lp_lower_ticks
    lp_upper_prices = exponential_value ** lp_upper_ticks

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Price evolution and LP bounds
    ax1 = axes[0, 0]
    ax1.plot(times, midprices, 'b-', label='External Midprice', alpha=0.8, linewidth=2)
    ax1.plot(times, pool_prices, 'r-', label='Pool Price', alpha=0.8, linewidth=2)
    ax1.fill_between(times, lp_lower_prices, lp_upper_prices,
                     alpha=0.3, color='green', label='LP Position Range')
    ax1.plot(times, lp_lower_prices, 'g--', alpha=0.7, label='LP Lower Bound')
    ax1.plot(times, lp_upper_prices, 'g--', alpha=0.7, label='LP Upper Bound')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.set_title('Price Evolution and LP Strategy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: LP liquidity over time
    ax2 = axes[0, 1]
    ax2.plot(times, lp_liquidities, 'purple', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('LP Liquidity')
    ax2.set_title('LP Liquidity Deployment')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative PnL
    ax3 = axes[1, 0]
    ax3.plot(times, cumulative_pnls, 'orange', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Cumulative PnL')
    ax3.set_title('Cumulative Profit & Loss')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Action visualization (position width)
    ax4 = axes[1, 1]
    actions_array = np.array(trajectory_data['actions'])
    position_widths = actions_array[:, 1] - actions_array[:, 0]  # upper - lower
    ax4.plot(times, position_widths, 'brown', linewidth=2)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Position Width (ticks)')
    ax4.set_title('LP Position Width Over Time')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/policy_gradient_detailed_analysis.png', dpi=150, bbox_inches='tight')
    #plt.show()

    # Print strategy summary
    print(f"\n=== Strategy Analysis ===")
    print(f"Average position width: {np.mean(position_widths):.1f} ticks")
    print(f"Position width std: {np.std(position_widths):.1f} ticks")
    print(f"Average liquidity deployed: {np.mean(lp_liquidities):.0f}")
    print(f"Price range covered: {np.min(lp_lower_prices):.2f} - {np.max(lp_upper_prices):.2f}")

    # Save model
    torch.save({
        'policy_state_dict': policy_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'rewards': rewards,
        'evaluation_results': {
            'large_scale_pnl': cumulative_pnl,
            'single_trajectory_pnl': cumulative_pnl_single,
            'trajectory_data': trajectory_data
        },
        'config': {
            'input_size': input_size,
            'tau': TAU,
            'num_trajectories': NUM_TRAJECTORIES,
            'learning_rate': LEARNING_RATE,
        }
    }, 'policy_gradient_agent.pth')

    print("Model and evaluation results saved to policy_gradient_agent.pth")

if __name__ == "__main__":
    main()