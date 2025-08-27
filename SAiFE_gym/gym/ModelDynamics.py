
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import X, Y


from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import MidpriceModel
from SAiFE_gym.stochastic_processes.price_impact_models import PriceImpactModel

class ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model = None,
        arrival_model = None,
        fill_probability_model = None,
        price_impact_model = None,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model
        self.rng = default_rng(seed)
        self.seed_ = seed

        self.state = None 
        self.spot_price = None

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass

    def get_fills(self, action: np.ndarray):
        pass
    
    def get_arrivals_and_fills(self, action: np.ndarray):
        return None, None 


class UniswapV2ModelDynamics(ModelDynamics):
    # the state in UniV2 is the reserves of the two assets in the pool - the midprice (spot price) is derived from these reserves
    # the fees are a percentage of the trade amount, and are applied to the fills
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        num_trajectories: int = 1,
        seed: int = None,
        FEE: float = 0.003,  # 0.3% fee
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        num_trajectories = num_trajectories,
                        seed = seed)

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        # There is no action in Uniswap V2, the action is implicitly defined by the fills
        # For now, the orderflow is handled as a batch - this actually does not impact the revenue of LP from fees
        
        self.state[:, X] += np.sum(arrivals, axis=1)
        self.state[:, Y] -= np.sum(fills, axis=1) * (1 + self.FEE)  # LP pays the fee to the pool

    def get_action_space(self):
        # agent does not do anything - LPing is passive in UniV2
        pass



class UniswapV3ModelDynamics (ModelDynamics):
    # the state here should be more complex, as Uniswap V3 allows for concentrated liquidity
    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: Optional[PriceImpactModel] = None,
        price_impact_model: Optional[PriceImpactModel] = None,
        seed: int = None,
    ):
        super().__init__(midprice_model, arrival_model, fill_probability_model, price_impact_model, seed)
    
    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass

    def get_action_space(self):
        # agent reallocates his capital 
        pass
