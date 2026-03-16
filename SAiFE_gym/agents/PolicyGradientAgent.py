from typing import Union, Callable, Tuple

import gymnasium
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm

from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment


def generate_trajectory(env, agent, include_log_probs=False):
    """
    Generate a full trajectory using SAiFE_gym's vectorized environment.

    Returns:
        observations: List of flattened states for each step
        actions: List of actions (num_trajectories, 2) for each step
        rewards: Array of rewards (num_trajectories,) for each step
        log_probs: (Optional) List of log probabilities for each step
    """
    observations = []
    actions = []
    rewards = []
    log_probs = [] if include_log_probs else None

    state, _ = env.reset()
    terminated = np.zeros(env.num_trajectories, dtype=bool)

    while not np.any(terminated):
        # Flatten state for neural network
        flat_state = agent._flatten_state(state)
        observations.append(flat_state)

        # Get action (with log probs if needed)
        if include_log_probs:
            action, log_prob = agent.get_action(state, include_log_probs=True)
            log_probs.append(log_prob)
        else:
            action = agent.get_action(state)
        actions.append(action)

        # Step environment
        state, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)

    # Convert to arrays - rewards should be (num_steps, num_trajectories)
    rewards = np.array(rewards)  # Shape: (num_steps, num_trajectories)

    if include_log_probs:
        # Stack log probs - shape: (num_steps, num_trajectories, action_dim)
        log_probs = torch.stack(log_probs) if log_probs else None
        return observations, actions, rewards, log_probs

    return observations, actions, rewards


class PolicyGradientAgent(Agent):
    def __init__(
        self,
        policy: torch.nn.Module,
        env: gymnasium.Env,
        action_std: Union[float, Callable] = 0.1,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: _LRScheduler = None,
    ):
        self.env = env
        self.num_trajectories = env.num_trajectories
        self.action_size = env.action_space.shape[0]  # Should be 2 for [lower_offset, upper_offset]

        # Get input size by creating a dummy state and flattening it
        dummy_state, _ = env.reset()
        dummy_flat = self._flatten_state(dummy_state)
        self.input_size = dummy_flat.shape[1]  # Shape should be (num_trajectories, features)

        self.policy_net = policy
        self.action_std = action_std
        self.optimizer = optimizer or torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.lr_scheduler = lr_scheduler or StepLR(self.optimizer, step_size=100, gamma=0.99)
        self.proportion_completed: float = 0.0

    def _flatten_state(self, state: dict) -> np.ndarray:
        """
        Convert SAiFE_gym's dict state to flattened array for neural network.

        Args:
            state: Dict with keys like 'sqrt_price', 'current_tick', etc.

        Returns:
            Flattened state array of shape (num_trajectories, features)
        """
        from SAiFE_gym.gym.index_names import (
            POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, ASSET_PRICE_KEY, TIME_KEY,
            LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY
        )

        # Select key features for the policy (avoid using full liquidity array for now)
        features = []

        # Pool state
        #features.append(state[POOL_SQRT_PRICE_KEY])  # (num_trajectories,)
        features.append(state[POOL_CURRENT_TICK_KEY].astype(np.float32))  # (num_trajectories,)

        # Market state

        #################
        #features.append(state[ASSET_PRICE_KEY])  # (num_trajectories,)
        #features.append(state[TIME_KEY])  # (num_trajectories,)

        # Mispricing signal - key for asymmetric strategies!
        #pool_price = state[POOL_SQRT_PRICE_KEY] ** 2
        #mispricing = state[ASSET_PRICE_KEY] - pool_price  # positive = AMM underpriced
        #features.append(mispricing)  # (num_trajectories,)
        ############

        # LP position state
        features.append(state[LP_LIQUIDITY_KEY])  # (num_trajectories,)
        features.append(state[LP_TICK_LOWER_KEY].astype(np.float32))  # (num_trajectories,)
        features.append(state[LP_TICK_UPPER_KEY].astype(np.float32))  # (num_trajectories,)

        # Stack features: shape (num_features, num_trajectories) -> (num_trajectories, num_features)
        return np.column_stack(features)

    def get_action(
        self, state: dict, deterministic: bool = False, include_log_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, torch.tensor]]:
        assert not (deterministic and include_log_probs), "include_log_probs only available for non-deterministic output"

        # Flatten state for neural network input
        flat_state = self._flatten_state(state)
        state_tensor = torch.tensor(flat_state, dtype=torch.float32, requires_grad=False)

        # Get action means from policy network: shape (num_trajectories, action_dim)
        mean_actions = self.policy_net(state_tensor)

        # Get current action std (potentially decaying)
        std = self.action_std(self.proportion_completed) if callable(self.action_std) else self.action_std

        if deterministic:
            # Return deterministic action (no noise)
            return mean_actions.detach().numpy()

        # Sample from Gaussian policy
        action_dist = torch.distributions.Normal(loc=mean_actions, scale=std)
        sampled_actions = action_dist.sample()

        if include_log_probs:
            # Get log probs for the sampled actions
            log_probs = action_dist.log_prob(sampled_actions).sum(dim=-1)  # Sum over action dimensions
            return sampled_actions.detach().numpy(), log_probs

        return sampled_actions.detach().numpy()


    def train(self, num_epochs: int = 1, reporting_freq: int = 100):
        learning_losses = []
        learning_rewards = []
        self.proportion_completed = 0.0

        for epoch in tqdm(range(num_epochs)):
            # Generate trajectory with log probabilities
            observations, actions, rewards, log_probs = generate_trajectory(
                self.env, self, include_log_probs=True
            )

            # rewards shape: (num_steps, num_trajectories)
            # log_probs shape: (num_steps, num_trajectories)

            # Average reward across all trajectories and steps for monitoring
            mean_reward = np.mean(rewards)
            learning_rewards.append(mean_reward)

            # Convert to tensors
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

            # Calculate future rewards for each trajectory separately
            # Shape: (num_steps, num_trajectories)
            future_rewards = self._calculate_future_rewards(rewards_tensor)

            # Policy gradient loss: -E[log π(a|s) * R]
            # Sum over action dimensions, mean over trajectories and time
            policy_loss = -torch.mean(log_probs * future_rewards)

            # Optimize policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            # Logging
            if epoch % reporting_freq == 0:
                tqdm.write(f"Epoch {epoch}: Loss = {policy_loss.item():.4f}, Mean Reward = {mean_reward:.4f}")

            learning_losses.append(policy_loss.item())
            self.proportion_completed = epoch / max(num_epochs - 1, 1)
            self.lr_scheduler.step()

        return learning_losses, learning_rewards

    @staticmethod
    def _calculate_future_rewards(rewards: torch.Tensor) -> torch.Tensor:
        """
        Calculate future rewards (returns) for REINFORCE algorithm.

        Args:
            rewards: Shape (num_steps, num_trajectories)

        Returns:
            future_rewards: Shape (num_steps, num_trajectories)
        """
        # Flip along time dimension, compute cumsum, then flip back
        flipped_rewards = torch.flip(rewards, dims=(0,))  # Flip along time axis
        cumulative_flipped = torch.cumsum(flipped_rewards, dim=0)
        return torch.flip(cumulative_flipped, dims=(0,))
