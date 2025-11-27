
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import (
    LIQUIDITY_INDEX, AMM_PRICE_INDEX, ASSET_PRICE_INDEX, TIME_INDEX
)

from SAiFE_gym.gym.helpers.AMM_utils import (
    get_buckets_given_center_bucket_id, find_bucket_id, is_out_of_range
)


from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import MidpriceModel
from SAiFE_gym.stochastic_processes.price_impact_models import PriceImpactModel

class ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: PriceImpactModel = None,
        price_impact_model: PriceImpactModel = None,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model
        self.rng = default_rng(seed)
        self.seed_ = seed

        self.state = None 

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass

    def get_fills(self, action: np.ndarray):
        pass
    
    def get_arrivals_and_fills(self, action: np.ndarray):
        return None, None 


    def get_action_space(self) -> gym.spaces.Space:
        pass

    def get_required_stochastic_processes(self):
        pass



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
        fill_probability_model: Optional[PriceImpactModel] = None,
        price_impact_model: Optional[PriceImpactModel] = None,
        initial_capital: float = 10000.0,  # Total initial capital to provide as liquidity
        fee_tier: float = 0.003,           # 0.3% fee tier
        tau: int = 5,                      # Number of buckets on each side of current price
        exponential_value: float = 1.0001, # Base for exponential bucket spacing (Uniswap V3 tick spacing)
        seed: int = None,
    ):
        super().__init__(midprice_model, arrival_model, fill_probability_model, price_impact_model, seed)

        self.initial_capital = initial_capital
        self.initial_price = midprice_model.initial_state[0, 0] if midprice_model else 100.0
        self.fee_tier = fee_tier
        self.tau = tau  # Hyperparameter for active bucket window
        self.exponential_value = exponential_value
        self.use_mixed_strategy = True

        # Dynamic action space: 2*tau + 1 active buckets around current price
        self.num_active_buckets = 2 * tau + 1

    def get_action_space(self):
        """
        Return the action space for the agent.
        Action is a probability distribution over 2*tau+1 active buckets around current price.
        """
        if self.use_mixed_strategy:
            # Output probability distribution over 2*tau+1 active buckets
            return gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_active_buckets,), dtype=np.float32)
        else:
            # Output single bucket index (from 0 to 2*tau)
            return gym.spaces.Discrete(self.num_active_buckets)



    def update_state(self, arrivals: np.ndarray, action: np.ndarray):
        """
        Update state based on arrivals (orderflow) and LP's action (bucket selection).

        Args:
            arrivals: Array of shape (2,) representing [buy_arrivals, sell_arrivals]
                     These are order sizes in token amounts
            action: Integer action representing which bucket to allocate liquidity to

        The function:
        1. Processes buy and sell arrivals as swaps
        2. Updates LP reserves and fees when swaps occur in their range
        """

        pass

    def get_arrivals_and_fills(self, action: np.ndarray):
        """
        Generate arrivals from the arrival model.
        In V3, fills = arrivals (all orders are filled by the AMM).
        """
        if self.arrival_model is not None:
            arrivals = self.arrival_model.get_next_state()
            # For AMMs, all arrivals are filled (no order book)
            fills = arrivals.copy()
            return arrivals[0], fills[0]  # Return first trajectory
        else:
            return np.zeros(2), np.zeros(2)
