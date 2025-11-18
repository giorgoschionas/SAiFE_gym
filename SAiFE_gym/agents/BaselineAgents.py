import gym
from copy import deepcopy
import numpy as np
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.index_names import (
    V3_LIQUIDITY_INDEX, V3_SQRT_PRICE_INDEX, V3_TICK_INDEX,
    V3_TICK_LOWER_INDEX, V3_TICK_UPPER_INDEX, V3_FEES_INDEX,
    V3_MIDPRICE_INDEX, V3_TIME_INDEX)
from SAiFE_gym.gym.helpers.AMM_utils import (
    price_to_tick, tick_to_price, calculate_liquidity_amounts,
    calculate_position_amounts, swap_v3_single_tick, is_position_in_range,
    create_liquidity_range_buckets
)   
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics



class RandomAgent(Agent):
    def __init__(self, env: gym.Env, seed: int = None):
        self.action_space = deepcopy(env.action_space)
        self.action_space.seed(seed)
        self.num_trajectories = env.num_trajectories

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.action_space.sample().reshape(1, -1), self.num_trajectories, axis=0)
    

class UniformAllocationAgent(Agent):
    """
    Allocates capital uniformly across all available buckets.
    Returns a uniform probability distribution where each bucket receives equal weight.

    This is a simple baseline that doesn't consider market conditions or price dynamics.
    """
    def __init__(self, env: AMMEnvironment):
        self.num_buckets = env.model_dynamics.num_buckets
        self.num_trajectories = getattr(env, 'num_trajectories', 1)
        # Pre-compute the uniform distribution
        self.uniform_action = np.ones(self.num_buckets) / self.num_buckets

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns uniform probability distribution over all buckets.

        Args:
            state: Current environment state (not used by this agent)

        Returns:
            action: Shape (num_trajectories, num_buckets) with uniform probabilities
        """
        # Return the same uniform distribution for all trajectories
        return np.repeat(self.uniform_action.reshape(1, -1), self.num_trajectories, axis=0)


class SingleBucketAgent(Agent):
    """
    Allocates all capital to a single specified bucket.
    Useful as a baseline for testing individual bucket performance.
    """
    def __init__(self, env: AMMEnvironment, bucket_index: int = 0):
        """
        Args:
            env: The AMM environment
            bucket_index: Which bucket to allocate to (default: 0, the first bucket)
        """
        self.num_buckets = env.model_dynamics.num_buckets
        self.num_trajectories = getattr(env, 'num_trajectories', 1)
        self.bucket_index = bucket_index % self.num_buckets  # Ensure valid index

        # Pre-compute the single-bucket action (one-hot encoded)
        self.action = np.zeros(self.num_buckets)
        self.action[self.bucket_index] = 1.0

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns action that allocates 100% to the specified bucket.

        Args:
            state: Current environment state (not used by this agent)

        Returns:
            action: Shape (num_trajectories, num_buckets) with all mass on one bucket
        """
        return np.repeat(self.action.reshape(1, -1), self.num_trajectories, axis=0)


class CurrentPriceBucketAgent(Agent):
    """
    Allocates all capital to the bucket containing the current price.
    This represents a concentrated liquidity strategy around the current market price.
    """
    def __init__(self, env: AMMEnvironment):
        self.num_buckets = env.model_dynamics.num_buckets
        self.num_trajectories = getattr(env, 'num_trajectories', 1)
        self.buckets = env.model_dynamics.buckets
        self.exponential_value = getattr(env.model_dynamics, 'exponential_value', 1.0001)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns action that allocates to bucket containing current price.

        Args:
            state: Current environment state (uses midprice to find bucket)

        Returns:
            action: Shape (num_trajectories, num_buckets) concentrated on current price bucket
        """
        # Extract current midprice from state
        current_price = state[0, V3_MIDPRICE_INDEX]

        # Find which bucket contains this price
        bucket_idx = self._find_bucket_containing_price(current_price)

        # Create one-hot action
        action = np.zeros(self.num_buckets)
        action[bucket_idx] = 1.0

        return np.repeat(action.reshape(1, -1), self.num_trajectories, axis=0)

    def _find_bucket_containing_price(self, price: float) -> int:
        """Find the bucket index that contains the given price."""
        for idx, bucket in enumerate(self.buckets):
            if bucket['p_low'] <= price < bucket['p_high']:
                return idx
        # If price is above all buckets, return last bucket
        # If price is below all buckets, return first bucket
        return self.num_buckets - 1 if price >= self.buckets[-1]['p_high'] else 0


class PassiveLPAgent(Agent):
    """
    Provides liquidity proportionally across the whole price range.
    """
    pass

class ActiveLPAgent(Agent):
    def __init__(self, env: AMMEnvironment, seed: int = None):
        self.env = AMMEnvironment() or env
        self.volatility = self.env.model_dynamics.midprice_model.volatility
        

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Rebalance LP position to new price range based on action.
    
        Args:
            state: The current state of the environment.

        Returns:
            action: a probability distribution over predefined price ranges.

        """

        liquidity = state[:, V3_LIQUIDITY_INDEX]
        time = state[:, V3_TIME_INDEX]
        


class MovingAverageLPAgent(Agent):
    pass