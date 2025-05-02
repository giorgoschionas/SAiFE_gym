import gym
import numpy as np

from gym.spaces import Box



class LiquidityProvisionEnvironment(gym.Env):
    def __init__(self, seed: int = None):
        if seed:
            self.seed(seed)
            self.rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed)


    def seed(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        for i, process in enumerate(self.stochastic_processes.values()):
            process.seed(seed + i + 1)
