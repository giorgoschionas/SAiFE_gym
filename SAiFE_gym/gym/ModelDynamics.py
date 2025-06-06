
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

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

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass

    def get_fills(self, action: np.ndarray):
        pass
    
    def get_arrivals_and_fills(self, action: np.ndarray):
        return None, None 


class UniswapV2ModelDynamics(ModelDynamics):
    # the state in UniV2 is the reserves of the two assets in the pool - the midprice (spot price) is derived from these reserves
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        num_trajectories = num_trajectories,
                        seed = seed)

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
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