
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

    The agent (LP) can choose to allocate liquidity in specific price ranges.
    The action space is DISCRETE - agents specify position bounds and liquidity fraction.

    """

    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        num_trajectories: int = 1,
        fee_tier: float = 0.003,           # 0.3% fee tier
        tau: int = 5,                      # Number of ticks around current tick
        num_ticks: int = 1000,             # Total ticks to track in liquidity array
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
        self.num_ticks = num_ticks  # Total number of ticks to track
        self.exponential_value = exponential_value

        # Track the center of the liquidity array (set during state initialization)
        self.tick_lower_global = None

        # Dynamic action space: 2*tau + 1 active buckets around current price (DEPRECATED)
        self.num_active_buckets = 2 * tau + 1

    def get_action_space(self):
        """
        Return the action space for the agent.

        Action format: [lower_offset, upper_offset, liquidity_fraction]
        - lower_offset: Tick offset from current tick (range: -tau to tau-1)
        - upper_offset: Tick offset from current tick (range: -tau+1 to tau)
        - liquidity_fraction: Fraction of available capital (range: 0.0 to 1.0)

        Constraint: lower_offset < upper_offset (enforced by validate_action)
        """

        return gym.spaces.Box(
            low=np.array([-self.tau, -self.tau + 1, 0.0], dtype=np.float32),
            high=np.array([self.tau - 1, self.tau, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Validate and clip action to ensure constraints.

        Args:
            action: (num_trajectories, 3) array of actions

        Returns:
            Validated action with same shape

        Ensures:
        - All values are within box bounds
        - lower_offset < upper_offset (minimum width of 1 tick)
        """
        action = action.copy()

        # Clip to box bounds
        action[:, 0] = np.clip(action[:, 0], -self.tau, self.tau - 1)
        action[:, 1] = np.clip(action[:, 1], -self.tau + 1, self.tau)
        action[:, 2] = np.clip(action[:, 2], 0.0, 1.0)

        # Ensure lower < upper (add minimum width of 1 tick if violated)
        invalid = action[:, 0] >= action[:, 1]
        action[invalid, 1] = action[invalid, 0] + 1

        # Re-clip upper after adjustment
        action[:, 1] = np.clip(action[:, 1], -self.tau + 1, self.tau)

        return action

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

