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