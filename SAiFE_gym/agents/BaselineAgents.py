import gym
from copy import deepcopy
import numpy as np
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.index_names import (
    LIQUIDITY_INDEX, AMM_PRICE_INDEX, ASSET_PRICE_INDEX, TIME_INDEX
)
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.helpers.AMM_utils import is_out_of_range, find_bucket_id, get_buckets_given_center_bucket_id


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

    Only rebalances when price moves outside the current active bucket range.
    When price stays in range, returns the cached previous action.
    """
    def __init__(self, env: AMMEnvironment):
        self.num_active_buckets = env.model_dynamics.num_active_buckets
        self.num_trajectories = getattr(env, 'num_trajectories', 1)
        self.tau = env.model_dynamics.tau
        self.exponential_value = env.model_dynamics.exponential_value

        # Pre-compute the uniform distribution over active buckets
        self.uniform_action = np.ones(self.num_active_buckets) / self.num_active_buckets

        # State tracking for "sticky" rebalancing
        self.previous_action = None
        self.current_center_bucket = None

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns uniform probability distribution over 2*tau+1 active buckets.
        Only rebalances when price moves out of current bucket range.

        Args:
            state: Current environment state

        Returns:
            action: Shape (num_trajectories, num_active_buckets) with uniform probabilities
        """
        current_price = state[0, AMM_PRICE_INDEX]
        new_center_bucket = find_bucket_id(current_price, self.exponential_value)

        # First call OR price moved out of range
        if self.current_center_bucket is None or \
           is_out_of_range(current_price, self.current_center_bucket, self.tau):
            # Rebalance: compute new action
            action = np.repeat(self.uniform_action.reshape(1, -1), self.num_trajectories, axis=0)
            # Update cached state
            self.previous_action = action.copy()
            self.current_center_bucket = new_center_bucket
            return action
        else:
            # Price in range: return cached action
            return self.previous_action

    def reset(self):
        """Reset cached state for new episode"""
        self.previous_action = None
        self.current_center_bucket = None


class ActiveLPAgent(Agent):
    """
    Active LP agent that implements a more sophisticated rebalancing strategy.
    Only rebalances when price moves outside the current active bucket range.
    When price stays in range, returns the cached previous action.
    """
    def __init__(self, env: AMMEnvironment, seed: int = None):
        self.env = env
        self.volatility = self.env.model_dynamics.midprice_model.volatility
        self.num_active_buckets = env.model_dynamics.num_active_buckets
        self.num_trajectories = getattr(env, 'num_trajectories', 1)
        self.tau = env.model_dynamics.tau
        self.exponential_value = env.model_dynamics.exponential_value

        # State tracking for "sticky" rebalancing
        self.previous_action = None
        self.current_center_bucket = None

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Rebalance LP position to new price range based on action.
        Only rebalances when price moves out of current bucket range.

        Args:
            state: The current state of the environment.

        Returns:
            action: a probability distribution over predefined price ranges.
        """
        current_price = state[0, AMM_PRICE_INDEX]
        new_center_bucket = find_bucket_id(current_price, self.exponential_value)

        # First call OR price moved out of range
        if self.current_center_bucket is None or \
           is_out_of_range(current_price, self.current_center_bucket, self.tau):
            # Rebalance: compute new action using strategy
            liquidity = state[:, LIQUIDITY_INDEX]
            time = state[:, TIME_INDEX]
            action = self._get_action(liquidity, time, current_price, state)
            # Update cached state
            self.previous_action = action.copy()
            self.current_center_bucket = new_center_bucket
            return action
        else:
            # Price in range: return cached action
            return self.previous_action

    def _get_action(self, liquidity: np.ndarray, time: np.ndarray, current_price: float, state: np.ndarray) -> np.ndarray:
        """
        Implement your active LP strategy here.

        Args:
            liquidity: Current liquidity in the position
            time: Current time in the simulation
            current_price: Current market price
            state: Full state array for additional features

        Returns:
            action: Probability distribution over active buckets
        """
        # TODO: Implement your active LP strategy
        # For now, return uniform distribution as placeholder
        uniform = np.ones(self.num_active_buckets) / self.num_active_buckets
        return np.repeat(uniform.reshape(1, -1), self.num_trajectories, axis=0)

    def reset(self):
        """Reset cached state for new episode"""
        self.previous_action = None
        self.current_center_bucket = None


class MovingAverageLPAgent(Agent):
    pass