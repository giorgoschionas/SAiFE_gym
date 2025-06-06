import gym
from copy import deepcopy
import numpy as np
from SAiFE_gym.agents.Agent import Agent

class RandomAgent(Agent):
    def __init__(self, env: gym.Env, seed: int = None):
        self.action_space = deepcopy(env.action_space)
        self.action_space.seed(seed)
        self.num_trajectories = env.num_trajectories

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.action_space.sample().reshape(1, -1), self.num_trajectories, axis=0)
    

class PassiveLPAgent(Agent):
    pass

class ActiveLPAgent(Agent):
    pass

class MovingAverageLPAgent(Agent):
    pass