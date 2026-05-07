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
        # Stack log probs - shape: (num_steps, num_trajectories)
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
        max_grad_norm: float = 1.0,
        gamma: float = 0.99,
    ):
        self.env = env
        self.num_trajectories = env.num_trajectories
        self.action_size = env.action_space.shape[0]  # 3 for [center_offset, half_width, hold_flag]

        # Get input size by creating a dummy state and flattening it
        dummy_state, _ = env.reset()
        dummy_flat = self._flatten_state(dummy_state)
        self.input_size = dummy_flat.shape[1]  # Shape should be (num_trajectories, features)

        self.policy_net = policy
        self.action_std = action_std
        self.optimizer = optimizer or torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.lr_scheduler = lr_scheduler or StepLR(self.optimizer, step_size=100, gamma=0.99)
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.proportion_completed: float = 0.0

        # Check if policy supports squashed Gaussian (transform + log_prob_correction)
        self._squashed = (
            hasattr(self.policy_net, 'transform')
            and hasattr(self.policy_net, 'log_prob_correction')
        )

    def _flatten_state(self, state: dict) -> np.ndarray:
        """
        Convert SAiFE_gym's dict state to flattened array for neural network.

        All features are normalized to roughly similar scales:
        - Relative mispricing (asset - pool) / pool: dimensionless, O(0.01)
        - LP offsets / tau: normalized to ~[-1, 1]
        - Time remaining: [0, 1]
        - Normalized LP liquidity: lp_liq / initial_wealth

        Args:
            state: Dict with keys like 'sqrt_price', 'current_tick', etc.

        Returns:
            Flattened state array of shape (num_trajectories, features)
        """
        from SAiFE_gym.gym.index_names import (
            POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, ASSET_PRICE_KEY, TIME_KEY,
            LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY
        )

        tau = self.env.model_dynamics.tau
        features = []

        # Relative mispricing: (asset_price - pool_price) / pool_price
        pool_price = state[POOL_SQRT_PRICE_KEY] ** 2
        mispricing = (state[ASSET_PRICE_KEY] - pool_price) / np.maximum(pool_price, 1e-8)
        features.append(mispricing.astype(np.float32))

        # LP position offsets normalized by tau → roughly [-1, 1]
        current_tick = state[POOL_CURRENT_TICK_KEY]
        lower_offset = (current_tick - state[LP_TICK_LOWER_KEY]).astype(np.float32) / tau
        upper_offset = (state[LP_TICK_UPPER_KEY] - current_tick).astype(np.float32) / tau
        features.append(lower_offset)
        features.append(upper_offset)

        # Time remaining (normalized to [0, 1])
        time_remaining = 1.0 - state[TIME_KEY] / self.env.terminal_time
        features.append(time_remaining.astype(np.float32))

        # Normalized LP liquidity
        initial_wealth = getattr(self.env.model_dynamics, 'initial_wealth', 1.0)
        norm_liq = state[LP_LIQUIDITY_KEY] / max(initial_wealth, 1.0)
        features.append(norm_liq.astype(np.float32))

        return np.column_stack(features)

    def get_action(
        self, state: dict, deterministic: bool = False, include_log_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, torch.tensor]]:
        assert not (deterministic and include_log_probs), "include_log_probs only available for non-deterministic output"

        # Flatten state for neural network input
        flat_state = self._flatten_state(state)
        state_tensor = torch.tensor(flat_state, dtype=torch.float32, requires_grad=False)

        # Get raw output from policy network: shape (num_trajectories, action_dim)
        raw_output = self.policy_net(state_tensor)

        # Get current action std (potentially decaying)
        std = self.action_std(self.proportion_completed) if callable(self.action_std) else self.action_std

        if self._squashed:
            # --- Squashed Gaussian: noise in raw space, then transform ---
            if deterministic:
                return self.policy_net.transform(raw_output).detach().numpy()

            dist = torch.distributions.Normal(raw_output, std)
            raw_samples = dist.sample()
            actions = self.policy_net.transform(raw_samples)

            if include_log_probs:
                log_probs = dist.log_prob(raw_samples).sum(dim=-1)
                log_probs = log_probs - self.policy_net.log_prob_correction(raw_samples)
                return actions.detach().numpy(), log_probs

            return actions.detach().numpy()
        else:
            # --- Legacy: noise directly on bounded outputs ---
            if deterministic:
                return raw_output.detach().numpy()

            dist = torch.distributions.Normal(loc=raw_output, scale=std)
            sampled_actions = dist.sample()

            if include_log_probs:
                log_probs = dist.log_prob(sampled_actions).sum(dim=-1)
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

            # Calculate discounted future rewards (returns) for each trajectory
            # Shape: (num_steps, num_trajectories)
            future_rewards = self._calculate_future_rewards(rewards_tensor)

            # Baseline: mean return across trajectories at each timestep
            # This reduces variance without introducing bias
            baseline = future_rewards.mean(dim=1, keepdim=True)
            advantages = future_rewards - baseline

            # Normalize advantages (zero-mean, unit-variance) for stable gradients
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss: -E[log π(a|s) * A(s,a)]
            policy_loss = -torch.mean(log_probs * advantages)

            # Optimize policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            # Logging
            if epoch % reporting_freq == 0:
                tqdm.write(f"Epoch {epoch}: Loss = {policy_loss.item():.4f}, Mean Reward = {mean_reward:.4f}")

            learning_losses.append(policy_loss.item())
            self.proportion_completed = epoch / max(num_epochs - 1, 1)
            self.lr_scheduler.step()

        return learning_losses, learning_rewards

    def _calculate_future_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Calculate discounted future rewards (returns) for REINFORCE.

        G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … + γ^(T-t)·r_T

        Args:
            rewards: Shape (num_steps, num_trajectories)

        Returns:
            future_rewards: Shape (num_steps, num_trajectories)
        """
        T = rewards.shape[0]
        future = torch.zeros_like(rewards)
        future[-1] = rewards[-1]
        for t in range(T - 2, -1, -1):
            future[t] = rewards[t] + self.gamma * future[t + 1]
        return future
