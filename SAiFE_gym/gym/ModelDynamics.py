
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import (
    V3_LIQUIDITY_INDEX, V3_SQRT_PRICE_INDEX, V3_TICK_INDEX,
    V3_TICK_LOWER_INDEX, V3_TICK_UPPER_INDEX, V3_FEES_INDEX,
    V3_MIDPRICE_INDEX, V3_TIME_INDEX
)

from SAiFE_gym.gym.helpers.AMM_utils import (
    price_to_tick, tick_to_price, calculate_liquidity_amounts,
    calculate_position_amounts, swap_v3_single_tick, is_position_in_range,
    create_liquidity_range_buckets
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
    The update_state function processes orderflow (arrivals) and updates the LP's reserves
    based on swaps that occur within their liquidity range.
    """

    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: Optional[PriceImpactModel] = None,
        price_impact_model: Optional[PriceImpactModel] = None,
        initial_capital: float = 10000.0,  # Total initial capital to provide as liquidity
        fee_tier: float = 0.003,           # 0.3% fee tier
        tick_spacing: int = 60,            # Tick spacing for fee tier
        num_buckets: int = 5,              # Number of discrete price range options
        bucket_width_pct: float = 0.10,   # Width of each bucket as % of price (e.g., 10%)
        seed: int = None,
    ):
        super().__init__(midprice_model, arrival_model, fill_probability_model, price_impact_model, seed)

        self.initial_capital = initial_capital
        self.fee_tier = fee_tier
        self.tick_spacing = tick_spacing
        self.num_buckets = num_buckets
        self.bucket_width_pct = bucket_width_pct

        # Define price range buckets (action space)
        # Each bucket represents a different concentration level around current price
        self.bucket_ranges = create_liquidity_range_buckets(self.num_buckets, self.bucket_width_pct)

    def get_action_space(self):
        """
        Return the action space for the agent.
        Action is a discrete choice of which bucket (price range) to allocate liquidity.
        """
        return gym.spaces.Discrete(self.num_buckets)

    def get_position_amounts(self) -> tuple:
        """
        Compute token amounts (amount0, amount1) on-demand from state.
        Uses the (P, L) parameterization stored in state.

        Returns:
            tuple: (amount0, amount1) token amounts in the LP position
        """
        liquidity = self.state[0, V3_LIQUIDITY_INDEX]
        sqrt_price_current = self.state[0, V3_SQRT_PRICE_INDEX]
        tick_lower = int(self.state[0, V3_TICK_LOWER_INDEX])
        tick_upper = int(self.state[0, V3_TICK_UPPER_INDEX])

        # Convert ticks to sqrt prices
        sqrt_price_lower = np.sqrt(tick_to_price(tick_lower))
        sqrt_price_upper = np.sqrt(tick_to_price(tick_upper))

        # Calculate amounts from liquidity and price
        amount0, amount1 = calculate_position_amounts(
            liquidity, sqrt_price_current, sqrt_price_lower, sqrt_price_upper
        )

        return amount0, amount1

    def update_state(self, arrivals: np.ndarray, action: np.ndarray):
        """
        Update state based on arrivals (orderflow) and LP's action (bucket selection).

        Args:
            arrivals: Array of shape (2,) representing [buy_arrivals, sell_arrivals]
                     These are order sizes in token amounts
            action: Integer action representing which bucket to allocate liquidity to

        The function:
        1. Rebalances LP position to new range based on action
        2. Processes buy and sell arrivals as swaps
        3. Updates LP reserves and fees when swaps occur in their range
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
