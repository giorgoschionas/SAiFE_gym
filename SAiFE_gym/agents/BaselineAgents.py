import gym
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
    """
    Randomly samples LP position bounds uniformly from [-tau, tau].

    Uses order statistics: samples two points, sorts them to ensure lower < upper.
    This guarantees valid actions where lower_offset < upper_offset.
    """
    def __init__(self, env: gym.Env, seed: int = None):
        self.tau = env.model_dynamics.tau
        self.num_trajectories = env.num_trajectories
        self.rng = np.random.default_rng(seed)

    def get_action(self, state: dict) -> np.ndarray:
        # Sample two points uniformly from [-tau, tau] for each trajectory
        # Shape: (num_trajectories, 2)
        samples = self.rng.uniform(-self.tau, self.tau, size=(self.num_trajectories, 2))

        # Sort along axis=1 so that [:, 0] < [:, 1]
        # This ensures lower_offset < upper_offset
        actions = np.sort(samples, axis=1)

        return actions.astype(np.float32)
    

class UniformAllocationAgent(Agent):
    """
    Allocates capital across the full active tick range around the current price.

    Action format: [lower_offset, upper_offset] = [-tau, +tau]
    This covers 2*tau+1 ticks centered on the current price.

    Only rebalances when price moves outside the current LP position range.
    When price stays in range, returns the cached previous action.
    """
    def __init__(self, env: AMMEnvironment):
        self.env = env
        self.num_trajectories = env.num_trajectories
        self.tau = env.model_dynamics.tau

        # Internal state tracking
        self._initialized = False
        self._last_action = None

        # Pre-compute uniform action: [-tau, +tau] (full width around current tick)
        self._uniform_action = np.array([-self.tau, self.tau], dtype=np.float32)

    def get_action(self, state: dict) -> np.ndarray:
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

        # Selective update: only rebalance out-of-range trajectories
        action = self._last_action.copy()
        action[needs_rebalance, :] = self._uniform_action

        return action


class PassiveAgent(Agent):

    def __init__(self, env: AMMEnvironment, seed: int = None):
        pass

    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass


