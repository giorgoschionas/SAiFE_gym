import gym
from copy import deepcopy
import numpy as np
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment

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

    Only rebalances when price moves outside the current active bucket range.
    When price stays in range, returns the cached previous action.
    """
    def __init__(self, env: AMMEnvironment):
        pass 

    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass


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
