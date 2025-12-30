
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import (
    LIQUIDITY_INDEX, AMM_PRICE_INDEX, FEES_TOKEN_A_INDEX, FEES_TOKEN_B_INDEX, TIME_INDEX
)

from SAiFE_gym.gym.helpers.AMM_utils import (
   bucket_bounds_from_center_ids, find_bucket_id_vec
)


from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import MidpriceModel
from SAiFE_gym.stochastic_processes.price_impact_models import PriceImpactModel

class ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        price_impact_model: PriceImpactModel = None,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.price_impact_model = price_impact_model
        self.num_trajectories = num_trajectories
        self.rng = default_rng(seed)
        self.seed_ = seed

        self.state = None 

    def update_state(self, arrivals: np.ndarray, action: np.ndarray):
        pass

    
    def get_arrivals(self, action: np.ndarray):
        return None, None 


    def get_action_space(self) -> gym.spaces.Space:
        pass

    def get_required_stochastic_processes(self):
        pass

    @property
    def midprice(self):
        return self.midprice_model.current_state[:, 0].reshape(-1, 1)



class UniswapV3ModelDynamics(ModelDynamics):
    """
    Uniswap V3 Model Dynamics with concentrated liquidity.

    The agent (LP) can choose to allocate liquidity in different price ranges (buckets).
    The action space is DYNAMIC - only buckets within tau of the current price are "active".
    Agents return probability distributions over 2*tau+1 active buckets.
    """

    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        num_trajectories: int = 1,
        fee_tier: float = 0.003,           # 0.3% fee tier
        tau: int = 5,                      # Number of buckets on each side of current price
        exponential_value: float = 1.0001, # Base for exponential bucket spacing (Uniswap V3 tick spacing)
        seed: int = None,
    ):
        super().__init__(midprice_model = midprice_model, 
                         arrival_model = arrival_model, 
                         num_trajectories = num_trajectories, 
                         seed = seed)

        self.initial_price = midprice_model.initial_state[0, 0] if midprice_model else 100.0
        self.fee_tier = fee_tier
        self.tau = tau  # Hyperparameter for active bucket window
        self.exponential_value = exponential_value

        # Dynamic action space: 2*tau + 1 active buckets around current price
        self.num_active_buckets = 2 * tau + 1

    def get_action_space(self):
        """
        Return the action space for the agent.
        Action is a probability distribution over 2*tau+1 active buckets around current price.
        """

        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_active_buckets,), dtype=np.float32)

def update_state(self, arrivals: np.ndarray, action: np.ndarray):
    """
    update_state using a unified orderflow that captures 
    baseline, depth-dependent and arbitrage flow
    """
    pass 


    def get_arrivals(self):
        """Get arrivals from arrival model, passing current state."""
        arrivals = self.arrival_model.get_arrivals(self.state)
        return arrivals

