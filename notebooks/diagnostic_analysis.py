"""
Diagnostic script to analyze why the agent isn't learning sophisticated strategies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.agents.PolicyGradientAgent import PolicyGradientAgent
from SAiFE_gym.rewards.RewardFunctions import PnL

# Load trained model
checkpoint = torch.load('policy_gradient_agent.pth')

print("=== Diagnostic Analysis ===")

# 1. Analyze action distribution
print("\n1. ACTION DISTRIBUTION ANALYSIS:")
print("Loading model and running test episodes...")

# Create environment (same as training)
from train_policy_gradient_agent import create_environment, ConstrainedPolicyNetwork, create_policy_network

TAU = 100
env = create_environment(10, TAU, 42)  # 10 trajectories for analysis

# Recreate network
dummy_state, _ = env.reset()
temp_agent = PolicyGradientAgent(torch.nn.Linear(9, 2), env)
input_size = temp_agent.input_size

base_network = create_policy_network(input_size, hidden_size=256, action_size=2)
policy_network = ConstrainedPolicyNetwork(base_network, tau=TAU)
policy_network.load_state_dict(checkpoint['policy_state_dict'])
policy_network.eval()

# Create agent
agent = PolicyGradientAgent(policy_network, env)

# Collect actions from multiple episodes
all_actions = []
all_rewards = []
all_states = []

for episode in range(5):  # Multiple episodes
    state, _ = env.reset()
    terminated = np.zeros(10, dtype=bool)
    episode_actions = []
    episode_rewards = []
    episode_states = []

    while not np.any(terminated):
        action = agent.get_action(state, deterministic=True)
        episode_actions.append(action.copy())
        episode_states.append(agent._flatten_state(state).copy())

        state, reward, terminated, _, _ = env.step(action)
        episode_rewards.append(reward.copy())

    all_actions.extend(episode_actions)
    all_rewards.extend(episode_rewards)
    all_states.extend(episode_states)

# Convert to arrays
actions_array = np.array(all_actions)  # Shape: (total_steps, 10, 2)
rewards_array = np.array(all_rewards)  # Shape: (total_steps, 10)
states_array = np.array(all_states)    # Shape: (total_steps, 10, features)

# Flatten across trajectories
actions_flat = actions_array.reshape(-1, 2)  # Shape: (total_steps * 10, 2)
rewards_flat = rewards_array.reshape(-1)     # Shape: (total_steps * 10,)
states_flat = states_array.reshape(-1, states_array.shape[-1])  # Shape: (total_steps * 10, features)

print(f"Collected {len(actions_flat)} action samples")

# Analyze action patterns
lower_offsets = actions_flat[:, 0]
upper_offsets = actions_flat[:, 1]
position_widths = upper_offsets - lower_offsets
position_centers = (upper_offsets + lower_offsets) / 2

print(f"\nACTION STATISTICS:")
print(f"Lower offset - Mean: {np.mean(lower_offsets):.1f}, Std: {np.std(lower_offsets):.1f}")
print(f"Upper offset - Mean: {np.mean(upper_offsets):.1f}, Std: {np.std(upper_offsets):.1f}")
print(f"Width - Mean: {np.mean(position_widths):.1f}, Std: {np.std(position_widths):.1f}")
print(f"Center - Mean: {np.mean(position_centers):.1f}, Std: {np.std(position_centers):.1f}")

print(f"\nACTION RANGES:")
print(f"Lower offset range: [{np.min(lower_offsets):.1f}, {np.max(lower_offsets):.1f}]")
print(f"Upper offset range: [{np.min(upper_offsets):.1f}, {np.max(upper_offsets):.1f}]")
print(f"Width range: [{np.min(position_widths):.1f}, {np.max(position_widths):.1f}]")
print(f"Center range: [{np.min(position_centers):.1f}, {np.max(position_centers):.1f}]")

# 2. Check if actions correlate with state features
print(f"\n2. STATE-ACTION CORRELATION ANALYSIS:")
mispricing_idx = 4  # Mispricing feature index
mispricing = states_flat[:, mispricing_idx]

correlation_center = np.corrcoef(mispricing, position_centers)[0, 1]
correlation_width = np.corrcoef(mispricing, position_widths)[0, 1]

print(f"Mispricing vs Center correlation: {correlation_center:.3f}")
print(f"Mispricing vs Width correlation: {correlation_width:.3f}")

# 3. Analyze reward variance
print(f"\n3. REWARD ANALYSIS:")
print(f"Reward - Mean: {np.mean(rewards_flat):.4f}, Std: {np.std(rewards_flat):.4f}")
print(f"Reward range: [{np.min(rewards_flat):.4f}, {np.max(rewards_flat):.4f}]")

# Check if different actions lead to different rewards
# Bin actions by width and see reward differences
width_bins = np.percentile(position_widths, [0, 33, 66, 100])
bin_indices = np.digitize(position_widths, width_bins) - 1
bin_indices = np.clip(bin_indices, 0, 2)

print(f"\nREWARD BY WIDTH BINS:")
for i in range(3):
    mask = bin_indices == i
    if np.sum(mask) > 0:
        bin_rewards = rewards_flat[mask]
        bin_widths = position_widths[mask]
        print(f"Width bin {i}: Mean width {np.mean(bin_widths):.1f}, Mean reward {np.mean(bin_rewards):.4f}")

# 4. Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Action distributions
axes[0, 0].hist(lower_offsets, bins=20, alpha=0.7, label='Lower Offset')
axes[0, 0].hist(upper_offsets, bins=20, alpha=0.7, label='Upper Offset')
axes[0, 0].set_title('Action Offset Distributions')
axes[0, 0].legend()

axes[0, 1].hist(position_widths, bins=20, alpha=0.7)
axes[0, 1].set_title('Position Width Distribution')

axes[1, 0].hist(position_centers, bins=20, alpha=0.7)
axes[1, 0].set_title('Position Center Distribution')

axes[1, 1].hist(rewards_flat, bins=20, alpha=0.7)
axes[1, 1].set_title('Reward Distribution')

plt.tight_layout()
plt.savefig('figures/diagnostic_analysis.png', dpi=150)
plt.show()

print(f"\nDiagnostic plots saved to figures/diagnostic_analysis.png")