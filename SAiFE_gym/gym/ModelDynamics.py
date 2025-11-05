
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import (
    X, Y,
    V3_AMOUNT0_INDEX, V3_AMOUNT1_INDEX, V3_LIQUIDITY_INDEX,
    V3_SQRT_PRICE_INDEX, V3_TICK_INDEX, V3_TICK_LOWER_INDEX,
    V3_TICK_UPPER_INDEX, V3_FEES_INDEX,
    V3_MIDPRICE_INDEX, V3_TIME_INDEX
)
from SAiFE_gym.gym.helpers.AMM_utils import (
    price_to_tick, tick_to_price, calculate_liquidity_amounts,
    calculate_position_amounts, swap_v3_single_tick, is_position_in_range
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
        self.spot_price = None

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass

    def get_fills(self, action: np.ndarray):
        pass
    
    def get_arrivals_and_fills(self, action: np.ndarray):
        return None, None 



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
        initial_price: float = 2000.0,     # Initial price (e.g., ETH/USDC)
        fee_tier: float = 0.003,           # 0.3% fee tier
        tick_spacing: int = 60,            # Tick spacing for fee tier
        num_buckets: int = 5,              # Number of discrete price range options
        bucket_width_pct: float = 0.10,   # Width of each bucket as % of price (e.g., 10%)
        seed: int = None,
    ):
        super().__init__(midprice_model, arrival_model, fill_probability_model, price_impact_model, seed)

        self.initial_capital = initial_capital
        self.initial_price = initial_price
        self.fee_tier = fee_tier
        self.tick_spacing = tick_spacing
        self.num_buckets = num_buckets
        self.bucket_width_pct = bucket_width_pct

        # Define price range buckets (action space)
        # Each bucket represents a different concentration level around current price
        self.bucket_ranges = self._define_buckets()

        # Initialize state
        self._initialize_state()

    def _define_buckets(self):
        """
        Define discrete buckets for action space.
        Each bucket is a price range defined as [lower_pct, upper_pct] around current price.

        Examples:
        - Bucket 0: [-5%, +5%] (very concentrated, high capital efficiency)
        - Bucket 1: [-10%, +10%] (moderate concentration)
        - Bucket 2: [-20%, +20%] (wider range, lower IL risk)
        - etc.
        """
        buckets = []
        base_widths = np.linspace(0.05, self.bucket_width_pct * self.num_buckets, self.num_buckets)

        for width in base_widths:
            buckets.append({
                'lower_pct': 1 - width,  # e.g., 0.95 for -5%
                'upper_pct': 1 + width,  # e.g., 1.05 for +5%
                'width': width * 2
            })

        return buckets

    def _initialize_state(self):
        """Initialize the state vector with LP position and pool information."""
        # State shape: (num_trajectories, state_dim)
        # For simplicity, we'll start with 1 trajectory
        state_dim = 10  # As defined in index_names
        self.state = np.zeros((1, state_dim))

        # Set initial price and tick
        self.state[0, V3_SQRT_PRICE_INDEX] = np.sqrt(self.initial_price)
        self.state[0, V3_TICK_INDEX] = price_to_tick(self.initial_price, self.tick_spacing)
        self.state[0, V3_MIDPRICE_INDEX] = self.initial_price

        # Initialize with no position (agent will set position with first action)
        self.state[0, V3_LIQUIDITY_INDEX] = 0.0
        self.state[0, V3_AMOUNT0_INDEX] = 0.0
        self.state[0, V3_AMOUNT1_INDEX] = 0.0
        self.state[0, V3_FEES_INDEX] = 0.0
        self.state[0, V3_TIME_INDEX] = 0.0

        # Set initial position ticks (will be updated by first action)
        self.state[0, V3_TICK_LOWER_INDEX] = 0
        self.state[0, V3_TICK_UPPER_INDEX] = 0

        # Track total capital (starts as uninvested)
        self.uninvested_capital = self.initial_capital

    def get_action_space(self):
        """
        Return the action space for the agent.
        Action is a discrete choice of which bucket (price range) to allocate liquidity.
        """
        return gym.spaces.Discrete(self.num_buckets)

    def _rebalance_position(self, action: int, current_price: float):
        """
        Rebalance LP position to new price range based on action.

        Args:
            action: Integer index of bucket to use
            current_price: Current pool price
        """
        # Get target price range from action
        bucket = self.bucket_ranges[action]
        price_lower = current_price * bucket['lower_pct']
        price_upper = current_price * bucket['upper_pct']

        # Convert to ticks
        tick_lower = price_to_tick(price_lower, self.tick_spacing)
        tick_upper = price_to_tick(price_upper, self.tick_spacing)

        # If we have existing position, remove it first
        if self.state[0, V3_LIQUIDITY_INDEX] > 0:
            # Return capital from old position
            self.uninvested_capital += (
                self.state[0, V3_AMOUNT0_INDEX] +
                self.state[0, V3_AMOUNT1_INDEX] * current_price
            )

        # Calculate how much of each token to deposit
        # For simplicity, we'll split capital 50/50 in value terms
        value_per_token = self.uninvested_capital / 2
        amount0_desired = value_per_token  # USDC
        amount1_desired = value_per_token / current_price  # ETH

        # Calculate liquidity and actual amounts
        sqrt_price_current = np.sqrt(current_price)
        sqrt_price_lower = np.sqrt(price_lower)
        sqrt_price_upper = np.sqrt(price_upper)

        liquidity = calculate_liquidity_amounts(
            sqrt_price_current, sqrt_price_lower, sqrt_price_upper,
            amount0_desired, amount1_desired
        )

        amount0_actual, amount1_actual = calculate_position_amounts(
            liquidity, sqrt_price_current, sqrt_price_lower, sqrt_price_upper
        )

        # Update state
        self.state[0, V3_LIQUIDITY_INDEX] = liquidity
        self.state[0, V3_AMOUNT0_INDEX] = amount0_actual
        self.state[0, V3_AMOUNT1_INDEX] = amount1_actual
        self.state[0, V3_TICK_LOWER_INDEX] = tick_lower
        self.state[0, V3_TICK_UPPER_INDEX] = tick_upper

        # Update uninvested capital
        capital_used = amount0_actual + amount1_actual * current_price
        self.uninvested_capital -= capital_used

    def _process_swap(self, amount_in: float, zero_for_one: bool):
        """
        Process a single swap through the pool.

        Args:
            amount_in: Amount of input token
            zero_for_one: True if swapping token0 for token1, False otherwise

        Returns:
            dict with swap results (amount_out, new_sqrt_price, fees)
        """
        current_sqrt_price = self.state[0, V3_SQRT_PRICE_INDEX]
        tick_lower = self.state[0, V3_TICK_LOWER_INDEX]
        tick_upper = self.state[0, V3_TICK_UPPER_INDEX]
        current_tick = self.state[0, V3_TICK_INDEX]
        liquidity = self.state[0, V3_LIQUIDITY_INDEX]

        # Check if position is in range for this swap
        in_range = tick_lower <= current_tick <= tick_upper and liquidity > 0

        if not in_range:
            # LP doesn't participate in this swap
            return {
                'amount_out': 0,
                'sqrt_price_next': current_sqrt_price,
                'fee_amount': 0,
                'lp_participated': False
            }

        # Execute swap using LP's liquidity
        swap_result = swap_v3_single_tick(
            amount_in=amount_in,
            sqrt_price_current=current_sqrt_price,
            liquidity=liquidity,
            fee_tier=self.fee_tier,
            zero_for_one=zero_for_one
        )

        swap_result['lp_participated'] = True
        return swap_result

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
        4. Updates pool price based on swaps
        """
        # Extract current state
        current_price = self.state[0, V3_SQRT_PRICE_INDEX] ** 2

        # Step 1: Rebalance position based on action
        action_idx = int(action[0]) if isinstance(action, np.ndarray) else int(action)
        action_idx = np.clip(action_idx, 0, self.num_buckets - 1)
        self._rebalance_position(action_idx, current_price)

        # Step 2: Process arrivals (orderflow)
        buy_amount = arrivals[0]   # Amount of token0 (USDC) to buy token1 (ETH)
        sell_amount = arrivals[1]  # Amount of token1 (ETH) to sell for token0 (USDC)

        # Process buy orders (token0 -> token1, zero_for_one=True)
        if buy_amount > 0:
            swap_result = self._process_swap(buy_amount, zero_for_one=True)

            if swap_result['lp_participated']:
                # Update pool price
                self.state[0, V3_SQRT_PRICE_INDEX] = swap_result['sqrt_price_next']
                new_price = swap_result['sqrt_price_next'] ** 2
                self.state[0, V3_TICK_INDEX] = price_to_tick(new_price, self.tick_spacing)

                # Update LP position amounts (as price changed, token amounts change)
                liquidity = self.state[0, V3_LIQUIDITY_INDEX]
                sqrt_price_lower = np.sqrt(tick_to_price(int(self.state[0, V3_TICK_LOWER_INDEX])))
                sqrt_price_upper = np.sqrt(tick_to_price(int(self.state[0, V3_TICK_UPPER_INDEX])))

                amount0_new, amount1_new = calculate_position_amounts(
                    liquidity, swap_result['sqrt_price_next'], sqrt_price_lower, sqrt_price_upper
                )

                self.state[0, V3_AMOUNT0_INDEX] = amount0_new
                self.state[0, V3_AMOUNT1_INDEX] = amount1_new

                # Accumulate fees in token0 (fee from token0->token1 swap is in token0)
                self.state[0, V3_FEES_INDEX] += swap_result['fee_amount']

        # Process sell orders (token1 -> token0, zero_for_one=False)
        if sell_amount > 0:
            swap_result = self._process_swap(sell_amount, zero_for_one=False)

            if swap_result['lp_participated']:
                # Update pool price
                self.state[0, V3_SQRT_PRICE_INDEX] = swap_result['sqrt_price_next']
                new_price = swap_result['sqrt_price_next'] ** 2
                self.state[0, V3_TICK_INDEX] = price_to_tick(new_price, self.tick_spacing)

                # Update LP position amounts
                liquidity = self.state[0, V3_LIQUIDITY_INDEX]
                sqrt_price_lower = np.sqrt(tick_to_price(int(self.state[0, V3_TICK_LOWER_INDEX])))
                sqrt_price_upper = np.sqrt(tick_to_price(int(self.state[0, V3_TICK_UPPER_INDEX])))

                amount0_new, amount1_new = calculate_position_amounts(
                    liquidity, swap_result['sqrt_price_next'], sqrt_price_lower, sqrt_price_upper
                )

                self.state[0, V3_AMOUNT0_INDEX] = amount0_new
                self.state[0, V3_AMOUNT1_INDEX] = amount1_new

                # Accumulate fees in token0 equivalent (fee from token1->token0 swap is in token1)
                # Convert to token0 by multiplying by current price
                current_price = swap_result['sqrt_price_next'] ** 2
                self.state[0, V3_FEES_INDEX] += swap_result['fee_amount'] * current_price

        # Step 3: Update midprice from stochastic process (if available)
        if self.midprice_model is not None:
            # Get next midprice from the model
            midprice_trajectory = self.midprice_model.get_next_state()
            self.state[0, V3_MIDPRICE_INDEX] = midprice_trajectory[0]

        # Step 4: Update time
        if self.midprice_model is not None and hasattr(self.midprice_model, 'step_size'):
            self.state[0, V3_TIME_INDEX] += self.midprice_model.step_size

        return self.state

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
