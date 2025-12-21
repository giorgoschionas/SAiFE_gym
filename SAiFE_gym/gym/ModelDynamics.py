
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
    get_buckets_given_center_bucket_id, find_bucket_id, is_out_of_range, transaction_fee_one_step_vec
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
        initial_capital: float = 10000.0,  # Total initial capital to provide as liquidity
        fee_tier: float = 0.003,           # 0.3% fee tier
        tau: int = 5,                      # Number of buckets on each side of current price
        exponential_value: float = 1.0001, # Base for exponential bucket spacing (Uniswap V3 tick spacing)
        non_arb_lambda: float = 0.00005,
        seed: int = None,
    ):
        super().__init__(midprice_model = midprice_model, 
                         arrival_model = arrival_model, 
                         num_trajectories = num_trajectories, 
                         seed = seed)

        self.initial_capital = initial_capital
        self.initial_price = midprice_model.initial_state[0, 0] if midprice_model else 100.0
        self.fee_tier = fee_tier
        self.tau = tau  # Hyperparameter for active bucket window
        self.exponential_value = exponential_value
        self.non_arb_lambda = non_arb_lambda #Price movement of noisy trades

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
    Vectorized update_state using:
      - vectorized bucket selection (center_ids -> p_low/p_high arrays)
      - vectorized fee collection across trajectories (loop only over buckets)
      - transaction_fee_one_step_vec directly (no scalar fallback)
    """
    N = self.state.shape[0]
    B = action.shape[1]  # expected: 2*self.tau + 1

    # ====================================================================
    # PHASE 0: Determine active buckets (vectorized) and store initial price
    # ====================================================================
    price_sqrt_0 = self.state[:, AMM_PRICE_INDEX].copy()
    price_0 = price_sqrt_0 ** 2

    center_ids = find_bucket_id_vec(price_0, self.exponential_value)  # (N,)
    p_low, p_high = bucket_bounds_from_center_ids(
        center_ids, self.tau, self.exponential_value
    )  # both (N, B)

    # ====================================================================
    # PHASE 1: Arbitrage Trades (vectorized)
    # ====================================================================
    midprice = self.midprice.reshape(-1)  # (N,)
    lower_bound = np.sqrt((1.0 - self.fee_tier) * midprice)
    upper_bound = np.sqrt(midprice / (1.0 - self.fee_tier))

    price_sqrt_1 = np.clip(price_sqrt_0, lower_bound, upper_bound)
    self.state[:, AMM_PRICE_INDEX] = price_sqrt_1
    price_1 = price_sqrt_1 ** 2

    # ====================================================================
    # PHASE 2: Noisy Trader Orders (vectorized)
    # ====================================================================
    sell_mask = arrivals[:, 0].astype(bool)
    buy_mask = arrivals[:, 1].astype(bool)

    price_sqrt_2 = price_sqrt_1.copy()
    price_sqrt_2[sell_mask] *= (1.0 - self.non_arb_lambda)
    price_sqrt_2[buy_mask] *= (1.0 + self.non_arb_lambda)

    price_sqrt_2 = np.maximum(price_sqrt_2, 1e-8)
    self.state[:, AMM_PRICE_INDEX] = price_sqrt_2
    price_2 = price_sqrt_2 ** 2

    # ====================================================================
    # PHASE 3: Fee Collection (vectorized over trajectories; loop over buckets)
    # ====================================================================
    move01 = ~np.isclose(price_0, price_1)
    move12 = ~np.isclose(price_1, price_2)

    for b in range(B):
        probs = action[:, b]                 # (N,)
        active = probs > 1e-10
        if not np.any(active):
            continue

        L = probs * self.initial_capital     # (N,)

        # Step 0 -> 1
        mask = active & move01
        if np.any(mask):
            fa, fb = transaction_fee_one_step_vec(
                p_low[mask, b],
                p_high[mask, b],
                price_0[mask],
                price_1[mask],
                self.fee_tier
            )
            self.state[mask, FEES_TOKEN_A_INDEX] += fa * L[mask]
            self.state[mask, FEES_TOKEN_B_INDEX] += fb * L[mask]

        # Step 1 -> 2
        mask = active & move12
        if np.any(mask):
            fa, fb = transaction_fee_one_step_vec(
                p_low[mask, b],
                p_high[mask, b],
                price_1[mask],
                price_2[mask],
                self.fee_tier
            )
            self.state[mask, FEES_TOKEN_A_INDEX] += fa * L[mask]
            self.state[mask, FEES_TOKEN_B_INDEX] += fb * L[mask]


    def get_arrivals(self):
        arrivals = self.arrival_model.get_arrivals()
        return arrivals

