import gymnasium
import numpy as np
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment

from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    POOL_CURRENT_TICK_KEY,
    POOL_SQRT_PRICE_KEY,
    ASSET_PRICE_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_EVER_DEPLOYED_KEY,
    TIME_KEY,
    FEES0_KEY,
    FEES1_KEY
)


class RandomAgent(Agent):
    """
    Randomly samples LP position center and half-width.

    Action format: [center_offset, half_width, hold_flag].
    """
    def __init__(self, env: gymnasium.Env, seed: int = None):
        self.tau = env.model_dynamics.tau
        self.num_trajectories = env.num_trajectories
        self.rng = np.random.default_rng(seed)

    def get_action(self, state: dict) -> np.ndarray:
        center = self.rng.uniform(-self.tau, self.tau, size=self.num_trajectories)
        half_width = self.rng.uniform(1.0, self.tau, size=self.num_trajectories)
        hold_flag = np.full(self.num_trajectories, -1.0)
        return np.column_stack([center, half_width, hold_flag]).astype(np.float32)


class UniformAllocationAgent(Agent):
    """
    Allocates capital across the full active tick range around the current price.

    Action format: [center_offset, half_width, hold_flag] = [0, tau, -1.0]
    This covers 2*tau ticks centered on the current price. Always rebalances.
    """
    def __init__(self, env: AMMEnvironment):
        self.env = env
        self.tau = env.model_dynamics.tau


    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        center = np.zeros(n, dtype=np.float32)
        half_width = np.full(n, self.tau, dtype=np.float32)
        hold_flag = np.full(n, -1.0, dtype=np.float32)
        return np.column_stack([center, half_width, hold_flag])

class DeployOnceAgent(Agent):
    """
    Deploys liquidity once at the full active tick range and holds for the entire episode.

    On the first step (LP not yet deployed), emits hold_flag = -1 to trigger
    deployment at [-tau, +tau]. On all subsequent steps, emits hold_flag = +1
    to hold the existing position without rebalancing.
    """
    def __init__(self, env: AMMEnvironment):
        self.env = env
        self.tau = env.model_dynamics.tau

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        center = np.zeros(n, dtype=np.float32)
        half_width = np.full(n, self.tau, dtype=np.float32)
        # hold_flag: -1 (rebalance) if never deployed, +1 (hold) otherwise
        ever_deployed = state[LP_EVER_DEPLOYED_KEY]
        hold_flag = np.where(ever_deployed, 1.0, -1.0).astype(np.float32)

        return np.column_stack([center, half_width, hold_flag])


class CarteaPLAgent(Agent):
    """
    Cartea-Drissi-Monga Optimal Liquidity Provision Agent.

    Implements the optimal position strategy from Cartea et al. research,
    using dynamic fee rates and stochastic drift/volatility parameters
    to compute optimal position boundaries.

    Key formulas:
    - δₜˡ* = (2γ + μₜ²σ²)/(8πₜ - σ² + 2μₜ(μₜ - σ²/2)) - μₜ
    - δₜᵘ* = (2γ + μₜ²σ²)/(8πₜ - σ² + 2μₜ(μₜ - σ²/2)) + μₜ
    """

    def __init__(self, env: AMMEnvironment, gamma: float = 0.005, seed: int = None):
        """
        Initialize Cartea agent.

        Args:
            env: SAiFE_gym AMM environment
            gamma: Risk aversion parameter (default 0.005)
            seed: Random seed for reproducibility
        """
        self.env = env
        self.gamma = gamma
        self.tau = env.model_dynamics.tau
        self.num_trajectories = env.num_trajectories
        self.rng = np.random.default_rng(seed)

        # Cache model dynamics for parameter access
        self.model_dynamics = env.model_dynamics

        # Cache midprice model parameters
        self.drift = self.model_dynamics.midprice_model.drift
        self.volatility = self.model_dynamics.midprice_model.volatility
        self.sigma_squared = self.volatility ** 2

    def calculate_dynamic_fee_rate(self, state: dict) -> np.ndarray:
        """
        Calculate dynamic pool fee rate π_t as cumulative fees / pool size.

        π_t = (total_fee_income) / (2 × κ × √price)

        Returns:
            π_t values for each trajectory, shape (num_trajectories,)
        """
        # Get active liquidity (κ) and current price
        _, active_liquidity = self.model_dynamics._get_current_tick_liquidity()
        sqrt_price = state[POOL_SQRT_PRICE_KEY]

        # Pool size: 2 × κ × √price
        pool_size = 2 * active_liquidity * sqrt_price + 1e-8  # Small epsilon to avoid division by zero

        # Total cumulative fees across all ticks
        total_fees_0 = np.sum(state[FEES0_KEY], axis=1)  # Shape: (num_trajectories,)
        total_fees_1 = np.sum(state[FEES1_KEY], axis=1)  # Shape: (num_trajectories,)

        # Convert to token0 units: fees_1 × price
        price = sqrt_price ** 2
        total_fee_income = total_fees_0 + total_fees_1 * price

        # Calculate percentage fee rate
        pi_t = total_fee_income / pool_size

        return pi_t

    def compute_optimal_deltas(self, state: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal boundary controls δₜˡ* and δₜᵘ* using Cartea formulas.

        Returns:
            (delta_lower, delta_upper): Optimal boundary controls, each shape (num_trajectories,)
        """
        # Get dynamic fee rate
        pi_t = self.calculate_dynamic_fee_rate(state)

        # Extract parameters
        mu = self.drift
        sigma_sq = self.sigma_squared
        gamma = self.gamma

        # Compute common terms
        numerator = 2 * gamma + mu**2 * sigma_sq
        denominator = 8 * pi_t - sigma_sq + 2 * mu * (mu - sigma_sq/2)

        # Handle edge case: very small or negative denominator
        # This can happen when fees are very low or drift/volatility parameters are extreme
        denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)

        # Compute base spread term
        base_term = numerator / denominator

        # Optimal deltas from equations (27)
        delta_lower = base_term - mu  # δₗ* = base_term - μ

        # Clip delta_lower to valid bounds [0.0001, 2.0]
        delta_lower = np.clip(delta_lower, 0.0001, 2.0)

        delta_upper = base_term + mu  # δᵤ* = base_term + μ

        # Clip delta_upper to valid bounds [0.0, 1.9999]
        delta_upper = np.clip(delta_upper, 0.0, 1.9999) 

        return delta_lower, delta_upper

    def deltas_to_tick_offsets(self, delta_lower: np.ndarray, delta_upper: np.ndarray,
                               sqrt_price: np.ndarray, state: dict) -> np.ndarray:
        """
        Convert continuous δ controls to discrete tick offsets.

        Academic formulas give δ in continuous price space, but SAiFE_gym
        requires integer tick offsets relative to current tick.

        Args:
            delta_lower: δₗ* values, shape (num_trajectories,)
            delta_upper: δᵤ* values, shape (num_trajectories,)
            sqrt_price: Current pool √price, shape (num_trajectories,)

        Returns:
            actions: [center_offset, half_width], shape (num_trajectories, 2)
        """
        # Convert δ to actual price boundaries using Cartea formulas (equation 6)
        # (Zₜˡ)^(1/2) = √price * (1 - δₗ/2)
        # (Zₜᵘ)^(1/2) = √price / (1 - δᵤ/2)

        sqrt_price_lower = sqrt_price * (1 - delta_lower/2)
        sqrt_price_upper = sqrt_price / (1 - delta_upper/2)

        # Convert √price to actual prices
        price = sqrt_price ** 2
        price_lower = sqrt_price_lower ** 2
        price_upper = sqrt_price_upper ** 2

        # Convert prices to ticks using SAiFE_gym's exponential spacing
        exponential_value = self.model_dynamics.exponential_value
        current_tick = state[POOL_CURRENT_TICK_KEY]

        tick_lower = np.log(price_lower) / np.log(exponential_value)
        tick_upper = np.log(price_upper) / np.log(exponential_value)

        # Compute offsets relative to current tick
        lower_offset = tick_lower - current_tick
        upper_offset = tick_upper - current_tick

        # Convert to (center, half_width) parameterization
        center = (lower_offset + upper_offset) / 2.0
        half_width = (upper_offset - lower_offset) / 2.0

        # Clip to action-space bounds
        center = np.clip(center, -self.tau, self.tau)
        half_width = np.clip(half_width, 1.0, self.tau)

        return np.column_stack([center, half_width]).astype(np.float32)

    def get_action(self, state: dict) -> np.ndarray:
        """
        Compute optimal liquidity provision action using Cartea strategy.

        Args:
            state: Current environment state dictionary

        Returns:
            action: [center_offset, half_width] for each trajectory,
                   shape (num_trajectories, 2). update_state treats a
                   2-element action as "always rebalance" (no hold_flag).
        """
        # Compute optimal boundary controls
        delta_lower, delta_upper = self.compute_optimal_deltas(state)

        # Convert to tick offsets
        sqrt_price = state[POOL_SQRT_PRICE_KEY]
        actions = self.deltas_to_tick_offsets(delta_lower, delta_upper, sqrt_price, state)

        return actions


