
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, ASSET_PRICE_KEY, TIME_KEY
)


from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import MidpriceModel

class ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.num_trajectories = num_trajectories
        self.rng = default_rng(seed)
        self.seed = seed

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
    Uniswap V3 Model Dynamics with Concentrated Liquidity.

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
        exponential_value: float = 1.0001, # Base for exponential tick spacing
    
        seed: int = None,
    ):
        super().__init__(midprice_model = midprice_model,
                         arrival_model = arrival_model,
                         num_trajectories = num_trajectories,
                         seed = seed)

        self.initial_price = midprice_model.initial_state[0, 0] if midprice_model else 100.0
        self.fee_tier = fee_tier
        self.tau = tau  # Hyperparameter for active tick window
        self.num_ticks = num_ticks  # Total number of ticks to track
        self.exponential_value = exponential_value
        self.tick_factor = np.sqrt(exponential_value) - 1.0

        # Track the center of the liquidity array (set during state initialization)
        self.tick_lower_global = None

        # Liquidity-dependent trade sizes (computed lazily)
        self.xi_sell = None  # Minimum trade size for sell (token0 in)
        self.xi_buy = None   # Minimum trade size for buy (token1 in)
        self._xi_stale = True  # Flag to trigger recomputation


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

    def _compute_xi(self):
        """
        Compute liquidity-dependent trade sizes (xi_sell, xi_buy).

        Trade size is the minimum amount that crosses at most one tick across all
        ticks in the liquidity array. This ensures consistent trade sizes across
        varying liquidity depths.

        Formula:
            xi_sell = min_i { L_i * tick_factor / sqrt(p_i) }
            xi_buy  = min_i { L_i * tick_factor * sqrt(p_i) }

        Where tick_factor = sqrt(1.0001) - 1 ≈ 0.00005

        Note: Only considers ticks with non-zero liquidity.
        """
        if self.state is None:
            raise ValueError("State not initialized. Call reset() first.")

        liquidity_array = self.state[POOL_LIQUIDITY_ARRAY_KEY]  # (num_traj, num_ticks)

        # Compute sqrt prices for all ticks
        tick_indices = np.arange(self.num_ticks)
        absolute_ticks = self.tick_lower_global + tick_indices  # (num_ticks,)
        sqrt_prices = np.sqrt(self.exponential_value ** absolute_ticks)  # (num_ticks,)

        # Compute capacity for each tick (vectorized across ticks)
        # x_capacity = L * tick_factor / sqrt_p (for sell, token0 in)
        # y_capacity = L * tick_factor * sqrt_p (for buy, token1 in)
        x_capacity = liquidity_array * self.tick_factor / sqrt_prices  # (num_traj, num_ticks)
        y_capacity = liquidity_array * self.tick_factor * sqrt_prices  # (num_traj, num_ticks)

        # Mask out zero-liquidity ticks (set to inf so they don't affect min)
        zero_liq_mask = liquidity_array <= 0
        x_capacity_masked = np.where(zero_liq_mask, np.inf, x_capacity)
        y_capacity_masked = np.where(zero_liq_mask, np.inf, y_capacity)

        # Take global minimum across all ticks for each trajectory
        # If all ticks have zero liquidity, result is inf (handled in update_state)
        self.xi_sell = np.min(x_capacity_masked, axis=1)  # (num_traj,)
        self.xi_buy = np.min(y_capacity_masked, axis=1)   # (num_traj,)

        self._xi_stale = False

    def mark_xi_stale(self):
        """Mark xi values as needing recomputation (call after liquidity changes)."""
        self._xi_stale = True

    def update_state(self, arrivals: np.ndarray, action: np.ndarray):
        """
        Liquidity-dependent state update with price impact inversely proportional to depth.

        Trade size (xi) is computed as the minimum trade that crosses at most one tick.
        Price impact depends on liquidity:
        - High liquidity → small price change within tick
        - Low liquidity → price crosses tick boundary

        Args:
            arrivals: Binary array of shape (num_trajectories, 2)
                      Column 0: sell_token0 arrivals (token0 into pool, price decreases)
                      Column 1: buy_token0 arrivals (token0 out of pool, price increases)
            action: Agent action array (for LP positioning, not used in swap)

        Price Update Logic:
            SELL (price decreases):
                No crossing: 1/sqrt_p_new = 1/sqrt_p_c + xi_sell/L_current
                Crossing: 1/sqrt_p_new = 1/sqrt_p_low + xi_remaining/L_prev, tick -= 1

            BUY (price increases):
                No crossing: sqrt_p_new = sqrt_p_c + xi_buy/L_current
                Crossing: sqrt_p_new = sqrt_p_high + yi_remaining/L_next, tick += 1
        """
        if self.state is None:
            raise ValueError("State not initialized. Call reset() first.")

        # Step 0: Compute xi if stale
        if self._xi_stale:
            self._compute_xi()

        num_traj = self.num_trajectories

        # Step 1: Extract current state
        current_tick = self.state[POOL_CURRENT_TICK_KEY].copy()
        sqrt_p_c = self.state[POOL_SQRT_PRICE_KEY].copy()
        liquidity_array = self.state[POOL_LIQUIDITY_ARRAY_KEY]

        # Step 2: Compute tick boundaries
        sqrt_p_low = np.sqrt(self.exponential_value ** current_tick)
        sqrt_p_high = np.sqrt(self.exponential_value ** (current_tick + 1))

        # Step 3: Get liquidity at current, previous, and next ticks
        tick_array_idx = (current_tick - self.tick_lower_global).astype(np.int64)
        tick_array_idx = np.clip(tick_array_idx, 0, self.num_ticks - 1)

        # Current tick liquidity
        L_current = liquidity_array[np.arange(num_traj), tick_array_idx]

        # Previous tick liquidity (for sell crossing)
        prev_tick_idx = np.clip(tick_array_idx - 1, 0, self.num_ticks - 1)
        L_prev = liquidity_array[np.arange(num_traj), prev_tick_idx]

        # Next tick liquidity (for buy crossing)
        next_tick_idx = np.clip(tick_array_idx + 1, 0, self.num_ticks - 1)
        L_next = liquidity_array[np.arange(num_traj), next_tick_idx]

        # Step 4: Compute capacity to boundary
        # Handle zero liquidity: treat as zero capacity (will always cross)
        L_current_safe = np.where(L_current > 0, L_current, 1.0)  # Avoid division by zero

        # Capacity to reach lower boundary (for sell)
        # x_to_boundary = L * (1/sqrt_p_low - 1/sqrt_p_c)
        x_to_boundary = np.where(
            L_current > 0,
            L_current * (1.0 / sqrt_p_low - 1.0 / sqrt_p_c),
            0.0  # Zero liquidity → zero capacity → always crosses
        )

        # Capacity to reach upper boundary (for buy)
        # y_to_boundary = L * (sqrt_p_high - sqrt_p_c)
        y_to_boundary = np.where(
            L_current > 0,
            L_current * (sqrt_p_high - sqrt_p_c),
            0.0  # Zero liquidity → zero capacity → always crosses
        )

        # Step 5: Determine arrival types
        is_sell = arrivals[:, 0].astype(bool)
        is_buy = arrivals[:, 1].astype(bool)

        # Step 6: Determine tick crossings (vectorized)
        # Sell crosses if xi_sell > x_to_boundary
        sell_crosses = is_sell & (self.xi_sell > x_to_boundary)

        # Buy crosses if xi_buy > y_to_boundary
        buy_crosses = is_buy & (self.xi_buy > y_to_boundary)

        # Step 7: Compute new sqrt_price for all 4 cases

        # Case 1: Sell, no crossing
        # 1/sqrt_p_new = 1/sqrt_p_c + xi_sell/L_current
        inv_sqrt_p_sell_no_cross = np.where(
            L_current > 0,
            1.0 / sqrt_p_c + self.xi_sell / L_current_safe,
            1.0 / sqrt_p_low  # Zero liquidity: jump to boundary
        )
        sqrt_p_sell_no_cross = 1.0 / inv_sqrt_p_sell_no_cross

        # Case 2: Sell, crossing
        # Remaining amount after crossing: xi_remaining = xi_sell - x_to_boundary
        xi_remaining_sell = np.maximum(self.xi_sell - x_to_boundary, 0)
        L_prev_safe = np.where(L_prev > 0, L_prev, 1.0)

        # 1/sqrt_p_new = 1/sqrt_p_low + xi_remaining/L_prev
        # Note: sqrt_p_low of current tick becomes sqrt_p_high of prev tick
        sqrt_p_prev_tick_low = np.sqrt(self.exponential_value ** (current_tick - 1))
        inv_sqrt_p_sell_cross = np.where(
            L_prev > 0,
            1.0 / sqrt_p_low + xi_remaining_sell / L_prev_safe,
            1.0 / sqrt_p_prev_tick_low  # Zero liquidity in prev: jump to its boundary
        )
        sqrt_p_sell_cross = 1.0 / inv_sqrt_p_sell_cross

        # Case 3: Buy, no crossing
        # sqrt_p_new = sqrt_p_c + xi_buy/L_current
        sqrt_p_buy_no_cross = np.where(
            L_current > 0,
            sqrt_p_c + self.xi_buy / L_current_safe,
            sqrt_p_high  # Zero liquidity: jump to boundary
        )

        # Case 4: Buy, crossing
        # Remaining amount after crossing: yi_remaining = xi_buy - y_to_boundary
        yi_remaining_buy = np.maximum(self.xi_buy - y_to_boundary, 0)
        L_next_safe = np.where(L_next > 0, L_next, 1.0)

        # sqrt_p_new = sqrt_p_high + yi_remaining/L_next
        sqrt_p_next_tick_high = np.sqrt(self.exponential_value ** (current_tick + 2))
        sqrt_p_buy_cross = np.where(
            L_next > 0,
            sqrt_p_high + yi_remaining_buy / L_next_safe,
            sqrt_p_next_tick_high  # Zero liquidity in next: jump to its boundary
        )

        # Step 8: Combine results using np.where (no branching)
        # Priority: sell_cross > sell_no_cross > buy_cross > buy_no_cross > no_change
        new_sqrt_price = sqrt_p_c.copy()

        # Apply buy no crossing
        new_sqrt_price = np.where(is_buy & ~buy_crosses, sqrt_p_buy_no_cross, new_sqrt_price)

        # Apply buy crossing
        new_sqrt_price = np.where(buy_crosses, sqrt_p_buy_cross, new_sqrt_price)

        # Apply sell no crossing (overwrites buy if both, but arrivals should be mutually exclusive)
        new_sqrt_price = np.where(is_sell & ~sell_crosses, sqrt_p_sell_no_cross, new_sqrt_price)

        # Apply sell crossing
        new_sqrt_price = np.where(sell_crosses, sqrt_p_sell_cross, new_sqrt_price)

        # Step 9: Update tick based on crossing
        # tick -= 1 for sell crossing, tick += 1 for buy crossing
        tick_change = np.zeros(num_traj, dtype=np.int64)
        tick_change = np.where(sell_crosses, -1, tick_change)
        tick_change = np.where(buy_crosses, 1, tick_change)

        self.state[POOL_CURRENT_TICK_KEY] = current_tick + tick_change
        self.state[POOL_SQRT_PRICE_KEY] = new_sqrt_price

        # Step 10: Calculate fees proportional to xi
        # fee = fee_rate / (1 - fee_rate) * xi
        fee_multiplier = self.fee_tier / (1.0 - self.fee_tier)

        fee_token0 = np.where(is_sell, fee_multiplier * self.xi_sell, 0.0)
        fee_token1 = np.where(is_buy, fee_multiplier * self.xi_buy, 0.0)

        self.state[FEES0_KEY] += fee_token0
        self.state[FEES1_KEY] += fee_token1

        # Step 11: Update time
        step_size = self.midprice_model.step_size if self.midprice_model else 0.005
        self.state[TIME_KEY] += step_size

    def get_arrivals(self) -> np.ndarray:
        """
        Get arrivals from arrival model (uses model's internal state).

        Returns:
            np.ndarray: Boolean arrivals array, shape (num_trajectories, 2)
                        Column 0: sell_token0 arrivals
                        Column 1: buy_token0 arrivals
        """
        if self.arrival_model is None:
            # Return no arrivals if no arrival model
            return np.zeros((self.num_trajectories, 2), dtype=bool)

        # No arguments! Uses arrival model's internal state
        return self.arrival_model.get_arrivals()

    def _update_arrival_model(self, arrivals: np.ndarray, action: np.ndarray):
        """
        Update arrival model's internal state after processing arrivals.

        Following mbt_gym pattern, this is called after state updates to prepare
        the arrival model for the next timestep.

        Args:
            arrivals: Binary array of shape (num_trajectories, 2)
            action: Agent action array
        """
        if self.arrival_model is None:
            return

        # Build state dict for arrival model
        state_for_arrival = self._build_arrival_context()
        self.arrival_model.update(arrivals, None, action, state_for_arrival)

    def _build_arrival_context(self) -> dict:
        """
        Pre-compute values for state-dependent arrival models.

        Returns:
            dict: Context with keys:
                - 'active_liquidity': Liquidity at current tick, shape (num_trajectories,)
                - 'amm_price': AMM price (sqrt_price**2), shape (num_trajectories,)
                - 'midprice': External market midprice, shape (num_trajectories,)
            Returns None if state is not initialized.
        """
        if self.state is None:
            return None

        # Get active liquidity at current tick (same pattern as update_state)
        current_tick = self.state[POOL_CURRENT_TICK_KEY]
        tick_array_idx = (current_tick - self.tick_lower_global).astype(np.int64)
        tick_array_idx = np.clip(tick_array_idx, 0, self.num_ticks - 1)
        active_liquidity = self.state[POOL_LIQUIDITY_ARRAY_KEY][
            np.arange(self.num_trajectories), tick_array_idx
        ]

        # Get prices
        sqrt_price = self.state[POOL_SQRT_PRICE_KEY]
        amm_price = sqrt_price ** 2
        midprice = self.state[ASSET_PRICE_KEY]

        return {
            'active_liquidity': active_liquidity,
            'amm_price': amm_price,
            'midprice': midprice,
        }
