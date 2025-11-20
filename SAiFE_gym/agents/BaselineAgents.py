import gym
from copy import deepcopy
import numpy as np
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.index_names import (
    V3_LIQUIDITY_INDEX, V3_SQRT_PRICE_INDEX, V3_TICK_INDEX,
    V3_TICK_LOWER_INDEX, V3_TICK_UPPER_INDEX, V3_FEES_INDEX,
    V3_MIDPRICE_INDEX, V3_TIME_INDEX
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
    Allocates capital uniformly across all active buckets (2*tau+1 buckets around current price).
    Returns a uniform probability distribution where each active bucket receives equal weight.

    This is a simple baseline that doesn't consider market conditions beyond the current price.
    """
    def __init__(self, env: AMMEnvironment):
        self.num_active_buckets = env.model_dynamics.num_active_buckets
        self.num_trajectories = getattr(env, 'num_trajectories', 1)
        # Pre-compute the uniform distribution over active buckets
        self.uniform_action = np.ones(self.num_active_buckets) / self.num_active_buckets

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns uniform probability distribution over 2*tau+1 active buckets.

        Args:
            state: Current environment state (not used by this agent)

        Returns:
            action: Shape (num_trajectories, num_active_buckets) with uniform probabilities
        """
        # Return the same uniform distribution for all trajectories
        return np.repeat(self.uniform_action.reshape(1, -1), self.num_trajectories, axis=0)


class SingleBucketAgent(Agent):
    """
    Allocates all capital to a single bucket within the active bucket range.
    Useful as a baseline for testing individual bucket strategies.

    The bucket_index refers to a position in the 2*tau+1 active buckets:
    - 0 = leftmost (tau buckets below current price)
    - tau = center (bucket containing current price)
    - 2*tau = rightmost (tau buckets above current price)
    """
    def __init__(self, env: AMMEnvironment, bucket_index: int = None):
        """
        Args:
            env: The AMM environment
            bucket_index: Which active bucket to allocate to.
                         If None, defaults to center bucket (tau).
                         Valid range: 0 to 2*tau
        """
        self.num_active_buckets = env.model_dynamics.num_active_buckets
        self.num_trajectories = getattr(env, 'num_trajectories', 1)
        self.tau = env.model_dynamics.tau

        # Default to center bucket if not specified
        if bucket_index is None:
            bucket_index = self.tau  # Center bucket

        self.bucket_index = bucket_index % self.num_active_buckets  # Ensure valid index

        # Pre-compute the single-bucket action (one-hot encoded)
        self.action = np.zeros(self.num_active_buckets)
        self.action[self.bucket_index] = 1.0

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns action that allocates 100% to the specified active bucket.

        Args:
            state: Current environment state (not used by this agent)

        Returns:
            action: Shape (num_trajectories, num_active_buckets) with all mass on one bucket
        """
        return np.repeat(self.action.reshape(1, -1), self.num_trajectories, axis=0)


class CurrentPriceBucketAgent(Agent):
    """
    Allocates all capital to the bucket containing the current price.
    In the dynamic action space, this is always the center bucket at index tau.

    This represents a maximally concentrated liquidity strategy around the current market price.
    """
    def __init__(self, env: AMMEnvironment):
        self.num_active_buckets = env.model_dynamics.num_active_buckets
        self.num_trajectories = getattr(env, 'num_trajectories', 1)
        self.tau = env.model_dynamics.tau

        # Pre-compute action: center bucket is at index tau
        self.action = np.zeros(self.num_active_buckets)
        self.action[self.tau] = 1.0  # Center bucket contains current price

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns action that allocates to the center bucket (current price bucket).

        Args:
            state: Current environment state (not used - center bucket is always at tau)

        Returns:
            action: Shape (num_trajectories, num_active_buckets) with all mass on center bucket
        """
        return np.repeat(self.action.reshape(1, -1), self.num_trajectories, axis=0)


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