
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY
)
from SAiFE_gym.gym.helpers.AMM_utils import get_position_value_vec


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
        initial_wealth: float = 1e6,       # LP's initial wealth for first rebalance
        rebalance_cost_coeff: float = 0.0, # Proportional cost per rebalance (e.g. 0.01 = 1%)
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
        self.initial_wealth = initial_wealth
        self.rebalance_cost_coeff = rebalance_cost_coeff

        # Track the center of the liquidity array (set during state initialization)
        self.tick_lower_global = None

        # Liquidity-dependent trade sizes (computed lazily)
        self.xi_sell = None  # Minimum trade size for sell (token0 in)
        self.xi_buy = None   # Minimum trade size for buy (token1 in)
        self._xi_stale = True  # Flag to trigger recomputation


    def get_action_space(self):
        """
        Return the action space for the agent.

        Action format: [lower_offset, upper_offset]
        - lower_offset: Tick offset from current tick (range: -tau to tau-1)
        - upper_offset: Tick offset from current tick (range: -tau+1 to tau)

        Constraint: lower_offset < upper_offset (enforced by validate_action)
<<<<<<< HEAD
        Capital is always fully deployed (no liquidity_fraction parameter).
=======
        The LP always deploys all available wealth into the specified range.
>>>>>>> main
        """

        return gym.spaces.Box(
            low=np.array([-self.tau, -self.tau + 1], dtype=np.float32),
            high=np.array([self.tau - 1, self.tau], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Validate and clip action to ensure constraints.

        Args:
<<<<<<< HEAD
            action: (num_trajectories, 2) array of actions
=======
            action: (num_trajectories, 2) array of [lower_offset, upper_offset]
>>>>>>> main

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

    def _process_sell(self, is_sell: np.ndarray) -> None:
        """
        Process sell arrivals and update state in place.

        Args:
            is_sell: Boolean array of shape (num_trajectories,) indicating sell arrivals.
        """
        if not np.any(is_sell):
            return

        num_traj = self.num_trajectories
        current_tick = self.state[POOL_CURRENT_TICK_KEY].copy()
        sqrt_p_c = self.state[POOL_SQRT_PRICE_KEY].copy()
        liquidity_array = self.state[POOL_LIQUIDITY_ARRAY_KEY]

        # Compute tick boundaries
        sqrt_p_low = np.sqrt(self.exponential_value ** current_tick)

        # Get liquidity at current and previous ticks
        tick_array_idx = (current_tick - self.tick_lower_global).astype(np.int64)
        tick_array_idx = np.clip(tick_array_idx, 0, self.num_ticks - 1)
        L_current = liquidity_array[np.arange(num_traj), tick_array_idx]

        prev_tick_idx = np.clip(tick_array_idx - 1, 0, self.num_ticks - 1)
        L_prev = liquidity_array[np.arange(num_traj), prev_tick_idx]

        # Handle zero liquidity: treat as zero capacity (will always cross)
        L_current_safe = np.where(L_current > 0, L_current, 1.0)

        # Capacity to reach lower boundary (for sell)
        x_to_boundary = np.where(
            L_current > 0,
            L_current * (1.0 / sqrt_p_low - 1.0 / sqrt_p_c),
            0.0  # Zero liquidity → zero capacity → always crosses
        )

        # Determine tick crossings
        sell_crosses = is_sell & (self.xi_sell > x_to_boundary)

        # Case 1: Sell, no crossing
        inv_sqrt_p_sell_no_cross = np.where(
            L_current > 0,
            1.0 / sqrt_p_c + self.xi_sell / L_current_safe,
            1.0 / sqrt_p_low  # Zero liquidity: jump to boundary
        )
        sqrt_p_sell_no_cross = 1.0 / inv_sqrt_p_sell_no_cross

        # Case 2: Sell, crossing
        xi_remaining_sell = np.maximum(self.xi_sell - x_to_boundary, 0)
        L_prev_safe = np.where(L_prev > 0, L_prev, 1.0)

        sqrt_p_prev_tick_low = np.sqrt(self.exponential_value ** (current_tick - 1))
        inv_sqrt_p_sell_cross = np.where(
            L_prev > 0,
            1.0 / sqrt_p_low + xi_remaining_sell / L_prev_safe,
            1.0 / sqrt_p_prev_tick_low  # Zero liquidity in prev: jump to its boundary
        )
        sqrt_p_sell_cross = 1.0 / inv_sqrt_p_sell_cross

        # Combine results: crossing takes priority over no crossing
        new_sqrt_p = np.where(sell_crosses, sqrt_p_sell_cross, sqrt_p_sell_no_cross)

        # Update state only where is_sell is True
        self.state[POOL_SQRT_PRICE_KEY] = np.where(is_sell, new_sqrt_p, sqrt_p_c)
        self.state[POOL_CURRENT_TICK_KEY] = np.where(
            sell_crosses, current_tick - 1,
            np.where(is_sell, current_tick, self.state[POOL_CURRENT_TICK_KEY])
        )

        # Collect sell fees, split between ticks on crossing
        fee_multiplier = self.fee_tier / (1.0 - self.fee_tier)

        # Fee at current tick: full xi for non-crossing, x_to_boundary for crossing
        fee_at_current = np.where(
            sell_crosses,
            fee_multiplier * x_to_boundary,
            fee_multiplier * self.xi_sell
        )
        self.state[FEES0_KEY][np.arange(num_traj), tick_array_idx] += np.where(
            is_sell, fee_at_current, 0.0
        )

        # Fee at prev tick: only for crossing sells
        fee_at_prev = fee_multiplier * xi_remaining_sell
        self.state[FEES0_KEY][np.arange(num_traj), prev_tick_idx] += np.where(
            sell_crosses, fee_at_prev, 0.0
        )

    def _process_buy(self, is_buy: np.ndarray) -> None:
        """
        Process buy arrivals and update state in place.

        Reads from CURRENT state (may have been updated by _process_sell).

        Args:
            is_buy: Boolean array of shape (num_trajectories,) indicating buy arrivals.
        """
        if not np.any(is_buy):
            return

        num_traj = self.num_trajectories
        current_tick = self.state[POOL_CURRENT_TICK_KEY].copy()
        sqrt_p_c = self.state[POOL_SQRT_PRICE_KEY].copy()
        liquidity_array = self.state[POOL_LIQUIDITY_ARRAY_KEY]

        # Compute tick boundaries from CURRENT tick (may be updated by sell)
        sqrt_p_high = np.sqrt(self.exponential_value ** (current_tick + 1))

        # Get liquidity at CURRENT and next ticks
        tick_array_idx = (current_tick - self.tick_lower_global).astype(np.int64)
        tick_array_idx = np.clip(tick_array_idx, 0, self.num_ticks - 1)
        L_current = liquidity_array[np.arange(num_traj), tick_array_idx]

        next_tick_idx = np.clip(tick_array_idx + 1, 0, self.num_ticks - 1)
        L_next = liquidity_array[np.arange(num_traj), next_tick_idx]

        # Handle zero liquidity
        L_current_safe = np.where(L_current > 0, L_current, 1.0)

        # Capacity to reach upper boundary (for buy)
        y_to_boundary = np.where(
            L_current > 0,
            L_current * (sqrt_p_high - sqrt_p_c),
            0.0  # Zero liquidity → zero capacity → always crosses
        )

        # Determine tick crossings
        buy_crosses = is_buy & (self.xi_buy > y_to_boundary)

        # Case 1: Buy, no crossing
        sqrt_p_buy_no_cross = np.where(
            L_current > 0,
            sqrt_p_c + self.xi_buy / L_current_safe,
            sqrt_p_high  # Zero liquidity: jump to boundary
        )

        # Case 2: Buy, crossing
        yi_remaining_buy = np.maximum(self.xi_buy - y_to_boundary, 0)
        L_next_safe = np.where(L_next > 0, L_next, 1.0)

        sqrt_p_next_tick_high = np.sqrt(self.exponential_value ** (current_tick + 2))
        sqrt_p_buy_cross = np.where(
            L_next > 0,
            sqrt_p_high + yi_remaining_buy / L_next_safe,
            sqrt_p_next_tick_high  # Zero liquidity in next: jump to its boundary
        )

        # Combine results: crossing takes priority over no crossing
        new_sqrt_p = np.where(buy_crosses, sqrt_p_buy_cross, sqrt_p_buy_no_cross)

        # Update state only where is_buy is True
        self.state[POOL_SQRT_PRICE_KEY] = np.where(is_buy, new_sqrt_p, sqrt_p_c)
        self.state[POOL_CURRENT_TICK_KEY] = np.where(
            buy_crosses, current_tick + 1,
            np.where(is_buy, current_tick, self.state[POOL_CURRENT_TICK_KEY])
        )

        # Collect buy fees, split between ticks on crossing
        fee_multiplier = self.fee_tier / (1.0 - self.fee_tier)

        # Fee at current tick: full xi for non-crossing, y_to_boundary for crossing
        fee_at_current = np.where(
            buy_crosses,
            fee_multiplier * y_to_boundary,
            fee_multiplier * self.xi_buy
        )
        self.state[FEES1_KEY][np.arange(num_traj), tick_array_idx] += np.where(
            is_buy, fee_at_current, 0.0
        )

        # Fee at next tick: only for crossing buys
        fee_at_next = fee_multiplier * yi_remaining_buy
        self.state[FEES1_KEY][np.arange(num_traj), next_tick_idx] += np.where(
            buy_crosses, fee_at_next, 0.0
        )

    def _collect_lp_fees(self):
        """
        Collect LP's share of pool fees from the LP's current position range.

        For each trajectory with an active position:
        - LP's share at each tick = lp_liquidity / total_liquidity_at_tick
        - Only considers ticks in LP's range [lower, upper)
        - Subtracts LP's share from pool fee arrays

        Returns:
            (fee0_per_traj, fee1_per_traj): Arrays of shape (num_trajectories,)
        """
        lp_liq = self.state[LP_LIQUIDITY_KEY]
        lp_lower = self.state[LP_TICK_LOWER_KEY].astype(np.int64)
        lp_upper = self.state[LP_TICK_UPPER_KEY].astype(np.int64)
        pool_liq = self.state[POOL_LIQUIDITY_ARRAY_KEY]

        # Build mask for LP's range: (num_traj, num_ticks)
        tick_indices = np.arange(self.num_ticks)  # (num_ticks,)
        absolute_ticks = self.tick_lower_global + tick_indices  # (num_ticks,)

        in_range = (absolute_ticks[None, :] >= lp_lower[:, None]) & \
                   (absolute_ticks[None, :] < lp_upper[:, None])  # (num_traj, num_ticks)

        # LP's share at each tick: lp_liq / total_liq (0 where total_liq is 0)
        total_liq_safe = np.where(pool_liq > 0, pool_liq, 1.0)
        lp_share = np.where(pool_liq > 0, lp_liq[:, None] / total_liq_safe, 0.0)

        # Only count fees in LP's range
        lp_share_in_range = lp_share * in_range  # (num_traj, num_ticks)

        # Compute LP's fee share
        lp_fee0 = np.sum(self.state[FEES0_KEY] * lp_share_in_range, axis=1)  # (num_traj,)
        lp_fee1 = np.sum(self.state[FEES1_KEY] * lp_share_in_range, axis=1)  # (num_traj,)

        # Subtract LP's share from pool fee arrays
        self.state[FEES0_KEY] -= self.state[FEES0_KEY] * lp_share_in_range
        self.state[FEES1_KEY] -= self.state[FEES1_KEY] * lp_share_in_range

        return lp_fee0, lp_fee1

    def _rebalance(self, action: np.ndarray):
        """
        Rebalance LP position: withdraw old position + fees, deploy into new range.

        Called at the start of update_state() before xi computation and swaps.

        Args:
            action: (num_trajectories, 2) validated action [lower_offset, upper_offset]
        """
        num_traj = self.num_trajectories
        sqrt_p = self.state[POOL_SQRT_PRICE_KEY]
        external_price = self.state[ASSET_PRICE_KEY]
        current_tick = self.state[POOL_CURRENT_TICK_KEY].astype(np.int64)
        lp_liq = self.state[LP_LIQUIDITY_KEY]

        has_position = lp_liq > 0

        # --- Phase 1: Compute wealth ---
        wealth = np.full(num_traj, self.initial_wealth, dtype=np.float64)

        if np.any(has_position):
            # Compute position value for trajectories with existing positions
            lp_lower = self.state[LP_TICK_LOWER_KEY].astype(np.int64)
            lp_upper = self.state[LP_TICK_UPPER_KEY].astype(np.int64)
            sqrt_p_lower = np.sqrt(self.exponential_value ** lp_lower.astype(np.float64))
            sqrt_p_upper = np.sqrt(self.exponential_value ** lp_upper.astype(np.float64))

            pos_value = get_position_value_vec(lp_liq, external_price, sqrt_p, sqrt_p_lower, sqrt_p_upper)

            # Collect LP's share of fees
            fee0, fee1 = self._collect_lp_fees()

            # Accumulate in LP_COLLECTED_FEES for reward tracking
            self.state[LP_COLLECTED_FEES0_KEY] += fee0
            self.state[LP_COLLECTED_FEES1_KEY] += fee1

            # Convert fee0 (token0) to token1 value using external price
            fee_value = fee0 * external_price + fee1

            wealth_with_pos = pos_value + fee_value

            # Apply rebalancing cost (proportional to total position value)
            wealth_with_pos *= (1.0 - self.rebalance_cost_coeff)

            wealth = np.where(has_position, wealth_with_pos, wealth)

            # Remove LP's liquidity from pool at old range
            tick_indices = np.arange(self.num_ticks)
            absolute_ticks = self.tick_lower_global + tick_indices
            old_in_range = (absolute_ticks[None, :] >= lp_lower[:, None]) & \
                           (absolute_ticks[None, :] < lp_upper[:, None])
            # Only remove for trajectories that have a position
            remove_mask = old_in_range & has_position[:, None]
            self.state[POOL_LIQUIDITY_ARRAY_KEY] -= remove_mask * lp_liq[:, None]

        # --- Phase 2: Deploy new position ---
        new_lower = current_tick + np.round(action[:, 0]).astype(np.int64)
        new_upper = current_tick + np.round(action[:, 1]).astype(np.int64)

        sqrt_p_new_lower = np.sqrt(self.exponential_value ** new_lower.astype(np.float64))
        sqrt_p_new_upper = np.sqrt(self.exponential_value ** new_upper.astype(np.float64))

        # Value per unit liquidity at new range
        value_per_L = get_position_value_vec(
            np.ones(num_traj), external_price, sqrt_p, sqrt_p_new_lower, sqrt_p_new_upper
        )

        # Compute new liquidity (handle zero value_per_L)
        new_L = np.where(value_per_L > 0, wealth / value_per_L, 0.0)

        # Add new liquidity to pool at new range
        tick_indices = np.arange(self.num_ticks)
        absolute_ticks = self.tick_lower_global + tick_indices
        new_in_range = (absolute_ticks[None, :] >= new_lower[:, None]) & \
                       (absolute_ticks[None, :] < new_upper[:, None])
        self.state[POOL_LIQUIDITY_ARRAY_KEY] += new_in_range * new_L[:, None]

        # --- Phase 3: Update LP state ---
        self.state[LP_LIQUIDITY_KEY] = new_L
        self.state[LP_TICK_LOWER_KEY] = new_lower.astype(np.float64)
        self.state[LP_TICK_UPPER_KEY] = new_upper.astype(np.float64)

        # Mark xi as stale since liquidity changed
        self.mark_xi_stale()

    def update_state(self, arrivals: np.ndarray, action: np.ndarray):
        """
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

        # Sync external price from midprice model (updated by AMMEnvironment before this call)
        if self.midprice_model is not None:
            self.state[ASSET_PRICE_KEY] = self.midprice_model.current_state[:, 0].copy()

        # Phase 0: Rebalance LP position (before xi computation and swaps)
        if action is not None:
            validated = self.validate_action(action)
            self._rebalance(validated)

        # Phase 1: Compute xi if stale (triggered by rebalance or external changes)
        if self._xi_stale:
            self._compute_xi()

        # Determine arrival types
        is_sell = arrivals[:, 0].astype(bool)
        is_buy = arrivals[:, 1].astype(bool)

        # Phase 2: Process sells (updates state in place)
        self._process_sell(is_sell)

        # Phase 3: Process buys from updated state
        self._process_buy(is_buy)

        # Phase 4: Update time
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
