import gymnasium
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

        # Append hold_flag = -1.0 (always rebalance)
        hold_col = np.full((self.num_trajectories, 1), -1.0, dtype=np.float32)
        return np.concatenate([actions.astype(np.float32), hold_col], axis=1)
    

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

        action = np.array([[-self.tau, self.tau, -1.0]])
        return np.repeat(action, self.env.num_trajectories, axis=0)

class CarteaPLAgent(Agent):
    pass


