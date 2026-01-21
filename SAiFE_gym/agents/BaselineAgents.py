import gym
from copy import deepcopy
import numpy as np
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment

from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    POOL_CURRENT_TICK_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    TIME_KEY
)


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
        self.env = env
        self.num_trajectories = env.num_trajectories
        self.tau = env.model_dynamics.tau

        # Internal state tracking
        self._initialized = False
        self._last_action = None

        # Pre-compute uniform action: [-tau, +tau, 1.0]
        self._uniform_action = np.array([-self.tau, self.tau, 1.0], dtype=np.float32)

    def reset(self):
        """Reset agent internal state. Call when environment resets."""
        self._initialized = False
        self._last_action = None

    def get_action(self, state: dict) -> np.ndarray:
        # Auto-reset detection: if time is 0, reset agent state
        if TIME_KEY in state and np.all(state[TIME_KEY] == 0):
            if self._initialized:
                self.reset()

        # Extract state components
        current_tick = state[POOL_CURRENT_TICK_KEY]
        lp_tick_lower = state[LP_TICK_LOWER_KEY]
        lp_tick_upper = state[LP_TICK_UPPER_KEY]

        # First call: always rebalance
        if not self._initialized:
            self._initialized = True
            self._last_action = np.tile(self._uniform_action, (self.num_trajectories, 1))
            return self._last_action.copy()

        # Rebalance when price is OUTSIDE the LP position range
        needs_rebalance = (current_tick < lp_tick_lower) | (current_tick > lp_tick_upper)

        # Early exit: no rebalancing needed
        if not np.any(needs_rebalance):
            return self._last_action.copy()

        # Selective update: only rebalance out-of-range trajectories
        action = self._last_action.copy()
        action[needs_rebalance, :] = self._uniform_action

        # Update cache
        self._last_action = action.copy()

        return action


class ActiveLPAgent(Agent):
    """
    Active LP agent that implements a more sophisticated rebalancing strategy.
    Only rebalances when price moves outside the current active bucket range.
    When price stays in range, returns the cached previous action.
    """
    def __init__(self, env: AMMEnvironment, seed: int = None):
        pass

    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass


class MovingAverageLPAgent(Agent):
    def __init__(self, env: AMMEnvironment, window_size: int = 5, seed: int = None):
        pass
    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass
