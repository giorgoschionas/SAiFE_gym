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
    TIME_KEY,
    FEES0_KEY,
    FEES1_KEY
)


class RandomAgent(Agent):
    """
    Randomly samples LP position bounds uniformly from [-tau, tau].

    Uses order statistics: samples two points, sorts them to ensure lower < upper.
    This guarantees valid actions where lower_offset < upper_offset.
    """
    def __init__(self, env: gymnasium.Env, seed: int = None):
        self.tau = env.model_dynamics.tau
        self.num_trajectories = env.num_trajectories
        self.rng = np.random.default_rng(seed)

    def get_action(self, state: dict) -> np.ndarray:
        # Sample two points uniformly from [-tau, tau] for each trajectory
        # Shape: (num_trajectories, 2)
        samples = self.rng.uniform(-self.tau, self.tau, size=(self.num_trajectories, 2))

        # Sort along axis=1 so that [:, 0] < [:, 1]
        actions = np.sort(samples, axis=1)

        # Clip to valid action space bounds: lower ∈ [-tau, tau-1], upper ∈ [-tau+1, tau]
        actions[:, 0] = np.clip(actions[:, 0], -self.tau, self.tau - 1)
        actions[:, 1] = np.clip(actions[:, 1], -self.tau + 1, self.tau)

        # Ensure minimum width of 1 tick (lower < upper)
        too_close = actions[:, 1] <= actions[:, 0]
        actions[too_close, 1] = actions[too_close, 0] + 1

        return actions.astype(np.float32)
    

class UniformAllocationAgent(Agent):
    """
    Allocates capital across the full active tick range around the current price.

    Action format: [lower_offset, upper_offset] = [-tau, +tau]
    This covers 2*tau+1 ticks centered on the current price.
    """
    def __init__(self, env: AMMEnvironment):
        self.env = env
        self.tau = env.model_dynamics.tau


    def get_action(self, state: dict) -> np.ndarray:

        action = np.array([[-self.tau, self.tau]])
        action = np.array([[-5, +5]])
        return np.repeat(action, self.env.num_trajectories, axis=0)

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
            actions: [lower_offset, upper_offset], shape (num_trajectories, 2)
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

        # Round to integers (tick offsets must be whole numbers)
        lower_offset = np.round(lower_offset)
        upper_offset = np.round(upper_offset)

        # Ensure offsets are within tau bounds and satisfy lower < upper
        lower_offset = np.clip(lower_offset, -self.tau, self.tau - 1)
        upper_offset = np.clip(upper_offset, -self.tau + 1, self.tau)

        # Ensure minimum width of 1 tick
        invalid = lower_offset >= upper_offset
        upper_offset = np.where(invalid, lower_offset + 1, upper_offset)
        upper_offset = np.clip(upper_offset, -self.tau + 1, self.tau)

        # Stack into action format
        actions = np.column_stack([lower_offset, upper_offset])
        return actions.astype(np.float32)

    def get_action(self, state: dict) -> np.ndarray:
        """
        Compute optimal liquidity provision action using Cartea strategy.

        Args:
            state: Current environment state dictionary

        Returns:
            action: [lower_offset, upper_offset] for each trajectory,
                   shape (num_trajectories, 2)
        """
        # Compute optimal boundary controls
        delta_lower, delta_upper = self.compute_optimal_deltas(state)

        # Convert to tick offsets
        sqrt_price = state[POOL_SQRT_PRICE_KEY]
        actions = self.deltas_to_tick_offsets(delta_lower, delta_upper, sqrt_price, state)

        return actions


