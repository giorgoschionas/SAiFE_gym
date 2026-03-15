
import abc

import gymnasium
import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, ASSET_PRICE_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
    LP_FEE_SNAPSHOT0_KEY, LP_FEE_SNAPSHOT1_KEY
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


    def get_action_space(self) -> gymnasium.spaces.Space:
        pass

    def get_required_stochastic_processes(self):
        pass

    @property
    def midprice(self):
        return self.midprice_model.current_state[:, 0].reshape(-1, 1)

    @property
    def initial_price(self):
        if self.midprice_model is not None:
            return self.midprice_model.initial_state[0, 0]
        raise AttributeError("initial_price requires a midprice_model")



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
        gas_cost: float = 0.0,            # Fixed cost per rebalance in token1 units
        swap_fee_rate: float = 0.0,        # Fee rate on imbalanced swap amount
        seed: int = None,
    ):
        super().__init__(midprice_model = midprice_model,
                         arrival_model = arrival_model,
                         num_trajectories = num_trajectories,
                         seed = seed)

        self.fee_tier = fee_tier
        self.fee_multiplier = fee_tier / (1.0 - fee_tier)
        self.tau = tau
        self.num_ticks = num_ticks
        self.exponential_value = exponential_value
        self.tick_factor = np.sqrt(exponential_value) - 1.0
        self.exp_quarter = exponential_value ** 0.25
        self.initial_wealth = initial_wealth
        self.gas_cost = gas_cost
        self.swap_fee_rate = swap_fee_rate

        # Track the center of the liquidity array (set during state initialization)
        self.tick_lower_global = None

        # Safety cap for maximum arrivals per step (prevents runaway loops)
        self.max_arrivals_per_step = 50


    def get_action_space(self):
        """
        Return the action space for the agent.

        Action format: [lower_offset, upper_offset]
        - lower_offset: Tick offset from current tick (range: -tau to tau-1)
        - upper_offset: Tick offset from current tick (range: -tau+1 to tau)

        Constraint: lower_offset < upper_offset (enforced by validate_action)
        The LP always deploys all available wealth into the specified range.
        """

        return gymnasium.spaces.Box(
            low=np.array([-self.tau, -self.tau + 1], dtype=np.float32),
            high=np.array([self.tau - 1, self.tau], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Validate and clip action to ensure constraints.

        Args:
            action: (num_trajectories, 2) array of [lower_offset, upper_offset]

        Returns:
            Validated action with same shape, rounded to integers.

        Ensures:
        - Values are rounded to integers (tick offsets must be whole numbers)
        - All values are within box bounds
        - lower_offset < upper_offset (minimum width of 1 tick)
        """
        action = action.copy()

        # Round to integers first — continuous actions from PPO must snap to tick grid
        # before the width constraint is applied (avoids zero-width positions)
        action[:, 0] = np.round(action[:, 0])
        action[:, 1] = np.round(action[:, 1])

        # Clip to box bounds
        action[:, 0] = np.clip(action[:, 0], -self.tau, self.tau - 1)
        action[:, 1] = np.clip(action[:, 1], -self.tau + 1, self.tau)

        # Ensure lower < upper (add minimum width of 1 tick if violated)
        invalid = action[:, 0] >= action[:, 1]
        action[invalid, 1] = action[invalid, 0] + 1

        # Re-clip upper after adjustment
        action[:, 1] = np.clip(action[:, 1], -self.tau + 1, self.tau)

        return action

    def _get_current_tick_liquidity(self):
        """
        Look up the liquidity at the current tick for each trajectory.

        Returns:
            (tick_array_idx, L_current):
                tick_array_idx: Clipped array index, shape (num_trajectories,)
                L_current: Liquidity at current tick, shape (num_trajectories,)
        """
        current_tick = self.state[POOL_CURRENT_TICK_KEY]
        tick_array_idx = (current_tick - self.tick_lower_global).astype(np.int64)
        tick_array_idx = np.clip(tick_array_idx, 0, self.num_ticks - 1)
        L_current = self.state[POOL_LIQUIDITY_ARRAY_KEY][
            np.arange(self.num_trajectories), tick_array_idx
        ]
        return tick_array_idx, L_current

    def _compute_local_xi(self):
        """
        Compute trade size from current tick's liquidity (one tick's capacity).

        Each trade moves the price by at most one tick. Trade size is determined
        by the liquidity at the current tick only, not a global minimum.

        Returns:
            (xi_sell, xi_buy): Each shape (num_trajectories,)
                xi_sell: token0 amount to traverse current tick downward
                xi_buy: token1 amount to traverse current tick upward
                Returns 0.0 where liquidity is zero.
        """
        if self.state is None:
            raise ValueError("State not initialized. Call reset() first.")

        current_tick = self.state[POOL_CURRENT_TICK_KEY]
        _, L = self._get_current_tick_liquidity()

        sqrt_p_low = np.sqrt(self.exponential_value ** current_tick)
        sqrt_p_high = np.sqrt(self.exponential_value ** (current_tick + 1))

        xi_sell = np.where(L > 0, L * self.tick_factor / sqrt_p_low, 0.0)
        xi_buy = np.where(L > 0, L * self.tick_factor * sqrt_p_high, 0.0)
        return xi_sell, xi_buy

    def _process_sell_single(self, active: np.ndarray, xi_sell: np.ndarray) -> None:
        """
        Process a single sell arrival per trajectory and update state in place.

        Sells push price downward. Trade size xi_sell is the token0 amount that
        traverses the full current tick. If capacity to the lower boundary is less
        than xi_sell (or liquidity is zero), the trade crosses into the previous tick
        and price snaps to the midpoint of that tick.

        Args:
            active: Boolean mask, shape (num_trajectories,) -- which trajectories have a sell.
            xi_sell: Trade size, shape (num_trajectories,) from _compute_local_xi().
        """
        if not np.any(active):
            return

        current_tick = self.state[POOL_CURRENT_TICK_KEY].copy()
        sqrt_p_c = self.state[POOL_SQRT_PRICE_KEY].copy()

        sqrt_p_low = np.sqrt(self.exponential_value ** current_tick)
        tick_array_idx, L_current = self._get_current_tick_liquidity()
        L_safe = np.where(L_current > 0, L_current, 1.0)

        # Token0 capacity from current price to lower tick boundary
        x_to_boundary = np.where(
            L_current > 0,
            L_current * (1.0 / sqrt_p_low - 1.0 / sqrt_p_c),
            0.0
        )

        crosses = active & ((xi_sell > x_to_boundary) | (L_current <= 0))

        # No crossing: new price from 1/sqrt_p formula
        inv_sqrt_p_no_cross = np.where(
            L_current > 0,
            1.0 / sqrt_p_c + xi_sell / L_safe,
            1.0 / sqrt_p_low
        )
        sqrt_p_no_cross = 1.0 / inv_sqrt_p_no_cross

        # Crossing: snap to geometric midpoint of previous tick
        # midpoint(tick T-1) = sqrt(exp_val^(T-0.5)) = sqrt_p_low / exp_val^0.25
        sqrt_p_cross = sqrt_p_low / self.exp_quarter

        new_sqrt_p = np.where(crosses, sqrt_p_cross, sqrt_p_no_cross)

        self.state[POOL_SQRT_PRICE_KEY] = np.where(active, new_sqrt_p, sqrt_p_c)
        self.state[POOL_CURRENT_TICK_KEY] = np.where(
            crosses, current_tick - 1,
            np.where(active, current_tick, self.state[POOL_CURRENT_TICK_KEY])
        )

        # Fees in old tick: full xi_sell if no crossing, x_to_boundary if crossing
        fee_volume = np.where(crosses, x_to_boundary, xi_sell)
        self.state[FEES0_KEY][np.arange(self.num_trajectories), tick_array_idx] += np.where(
            active, self.fee_multiplier * fee_volume, 0.0
        )

        # Fees in new tick (Uniswap V3 split): x_from_boundary executes in tick-1 after crossing
        # x_from_boundary = L_{i-1} * (1/sqrt_p_cross - 1/sqrt_p_low)
        prev_tick_array_idx = np.clip(tick_array_idx - 1, 0, self.num_ticks - 1)
        L_prev = self.state[POOL_LIQUIDITY_ARRAY_KEY][np.arange(self.num_trajectories), prev_tick_array_idx]
        x_from_boundary = L_prev * (1.0 / sqrt_p_cross - 1.0 / sqrt_p_low)
        self.state[FEES0_KEY][np.arange(self.num_trajectories), prev_tick_array_idx] += np.where(
            crosses, self.fee_multiplier * x_from_boundary, 0.0
        )

    def _process_buy_single(self, active: np.ndarray, xi_buy: np.ndarray) -> None:
        """
        Process a single buy arrival per trajectory and update state in place.

        Buys push price upward. Trade size xi_buy is the token1 amount that
        traverses the full current tick. If capacity to the upper boundary is less
        than xi_buy (or liquidity is zero), the trade crosses into the next tick
        and price snaps to the midpoint of that tick.

        Args:
            active: Boolean mask, shape (num_trajectories,) -- which trajectories have a buy.
            xi_buy: Trade size, shape (num_trajectories,) from _compute_local_xi().
        """
        if not np.any(active):
            return

        current_tick = self.state[POOL_CURRENT_TICK_KEY].copy()
        sqrt_p_c = self.state[POOL_SQRT_PRICE_KEY].copy()

        sqrt_p_high = np.sqrt(self.exponential_value ** (current_tick + 1))
        tick_array_idx, L_current = self._get_current_tick_liquidity()
        L_safe = np.where(L_current > 0, L_current, 1.0)

        # Token1 capacity from current price to upper tick boundary
        y_to_boundary = np.where(
            L_current > 0,
            L_current * (sqrt_p_high - sqrt_p_c),
            0.0
        )

        crosses = active & ((xi_buy > y_to_boundary) | (L_current <= 0))

        # No crossing: new price from sqrt_p formula
        sqrt_p_no_cross = np.where(
            L_current > 0,
            sqrt_p_c + xi_buy / L_safe,
            sqrt_p_high
        )

        # Crossing: snap to geometric midpoint of next tick
        # midpoint(tick T+1) = sqrt(exp_val^(T+1.5)) = sqrt_p_high * exp_val^0.25
        sqrt_p_cross = sqrt_p_high * self.exp_quarter

        new_sqrt_p = np.where(crosses, sqrt_p_cross, sqrt_p_no_cross)

        self.state[POOL_SQRT_PRICE_KEY] = np.where(active, new_sqrt_p, sqrt_p_c)
        self.state[POOL_CURRENT_TICK_KEY] = np.where(
            crosses, current_tick + 1,
            np.where(active, current_tick, self.state[POOL_CURRENT_TICK_KEY])
        )

        # Fees in old tick: full xi_buy if no crossing, y_to_boundary if crossing
        fee_volume = np.where(crosses, y_to_boundary, xi_buy)
        self.state[FEES1_KEY][np.arange(self.num_trajectories), tick_array_idx] += np.where(
            active, self.fee_multiplier * fee_volume, 0.0
        )

        # Fees in new tick (Uniswap V3 split): y_from_boundary executes in tick+1 after crossing
        # y_from_boundary = L_{i+1} * (sqrt_p_cross - sqrt_p_high)
        next_tick_array_idx = np.clip(tick_array_idx + 1, 0, self.num_ticks - 1)
        L_next = self.state[POOL_LIQUIDITY_ARRAY_KEY][np.arange(self.num_trajectories), next_tick_array_idx]
        y_from_boundary = L_next * (sqrt_p_cross - sqrt_p_high)
        self.state[FEES1_KEY][np.arange(self.num_trajectories), next_tick_array_idx] += np.where(
            crosses, self.fee_multiplier * y_from_boundary, 0.0
        )

    def _compute_gross_lp_fees(self):
        """
        Compute LP's gross share of pool fees (read-only, does not modify state).

        Returns:
            (gross_fee0, gross_fee1, lp_share_in_range):
                gross_fee0: shape (num_trajectories,)
                gross_fee1: shape (num_trajectories,)
                lp_share_in_range: shape (num_trajectories, num_ticks) — for pool subtraction
        """
        lp_liq = self.state[LP_LIQUIDITY_KEY]
        lp_lower = self.state[LP_TICK_LOWER_KEY].astype(np.int64)
        lp_upper = self.state[LP_TICK_UPPER_KEY].astype(np.int64)
        pool_liq = self.state[POOL_LIQUIDITY_ARRAY_KEY]

        tick_indices = np.arange(self.num_ticks)
        absolute_ticks = self.tick_lower_global + tick_indices

        in_range = (absolute_ticks[None, :] >= lp_lower[:, None]) & \
                   (absolute_ticks[None, :] < lp_upper[:, None])

        total_liq_safe = np.where(pool_liq > 0, pool_liq, 1.0)
        lp_share = np.where(pool_liq > 0, lp_liq[:, None] / total_liq_safe, 0.0)
        lp_share_in_range = lp_share * in_range

        gross_fee0 = np.sum(self.state[FEES0_KEY] * lp_share_in_range, axis=1)
        gross_fee1 = np.sum(self.state[FEES1_KEY] * lp_share_in_range, axis=1)

        return gross_fee0, gross_fee1, lp_share_in_range

    def _collect_lp_fees(self):
        """
        Collect LP's share of pool fees, excluding pre-entry fees via snapshots.

        net = max(gross - snapshot, 0)
        Only the net portion is subtracted from pool fee arrays.

        Returns:
            (net_fee0, net_fee1): Arrays of shape (num_trajectories,)
        """
        gross_fee0, gross_fee1, lp_share_in_range = self._compute_gross_lp_fees()

        snapshot0 = self.state[LP_FEE_SNAPSHOT0_KEY]
        snapshot1 = self.state[LP_FEE_SNAPSHOT1_KEY]

        net_fee0 = np.maximum(gross_fee0 - snapshot0, 0.0)
        net_fee1 = np.maximum(gross_fee1 - snapshot1, 0.0)

        # Compute ratio of net to gross (guarded for zero gross)
        safe_gross0 = np.where(gross_fee0 > 0, gross_fee0, 1.0)
        safe_gross1 = np.where(gross_fee1 > 0, gross_fee1, 1.0)
        net_ratio0 = np.where(gross_fee0 > 0, net_fee0 / safe_gross0, 0.0)
        net_ratio1 = np.where(gross_fee1 > 0, net_fee1 / safe_gross1, 0.0)

        # Subtract only the net portion from pool fee arrays
        self.state[FEES0_KEY] -= self.state[FEES0_KEY] * lp_share_in_range * net_ratio0[:, None]
        self.state[FEES1_KEY] -= self.state[FEES1_KEY] * lp_share_in_range * net_ratio1[:, None]

        # Zero out snapshots after use
        self.state[LP_FEE_SNAPSHOT0_KEY] = np.zeros(self.num_trajectories, dtype=np.float64)
        self.state[LP_FEE_SNAPSHOT1_KEY] = np.zeros(self.num_trajectories, dtype=np.float64)

        return net_fee0, net_fee1

    def _compute_token0_fraction_vec(self, sqrt_p, external_price, sqrt_p_lower, sqrt_p_upper):
        """
        Compute fraction of position value held in token0 (vectorized).

        Returns α ∈ [0, 1]:
        - price >= upper: 0.0 (all token1)
        - price <= lower: 1.0 (all token0)
        - in range: P*(1/√P - 1/√P_U) / (P*(1/√P - 1/√P_U) + (√P - √P_L))
          where P = sqrt_p**2 (pool price)
        """
        above = sqrt_p >= sqrt_p_upper
        below = sqrt_p <= sqrt_p_lower
        in_range = ~above & ~below

        numerator = np.where(in_range, sqrt_p**2 * (1.0 / sqrt_p - 1.0 / sqrt_p_upper), 0.0)
        denominator = numerator + np.where(in_range, sqrt_p - sqrt_p_lower, 0.0)
        safe_denom = np.where(denominator > 0, denominator, 1.0)
        alpha_in_range = np.where(denominator > 0, numerator / safe_denom, 0.5)

        return np.where(above, 0.0, np.where(below, 1.0, alpha_in_range))

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
        alpha_current = np.zeros(num_traj)

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

            alpha_pos = self._compute_token0_fraction_vec(sqrt_p, external_price, sqrt_p_lower, sqrt_p_upper)
            token0_value = alpha_pos * pos_value + fee0 * external_price
            alpha_current = np.where(wealth_with_pos > 0, token0_value / wealth_with_pos, 0.0)

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

        # Apply decomposed rebalancing cost (only for existing positions)
        if np.any(has_position):
            alpha_new = self._compute_token0_fraction_vec(sqrt_p, external_price, sqrt_p_new_lower, sqrt_p_new_upper)
            swap_cost = self.swap_fee_rate * np.abs(alpha_new - alpha_current) * wealth
            total_cost = np.where(has_position, self.gas_cost + swap_cost, 0.0)
            wealth = np.maximum(wealth - total_cost, 0.0)

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

        # --- Phase 4: Snapshot pre-existing fees in new range ---
        snapshot0, snapshot1, _ = self._compute_gross_lp_fees()
        self.state[LP_FEE_SNAPSHOT0_KEY] = snapshot0
        self.state[LP_FEE_SNAPSHOT1_KEY] = snapshot1

    def _process_arrivals(self, counts: np.ndarray, xi_index: int, process_fn) -> None:
        """
        Iterate over arrival counts, processing one trade per iteration.

        Each trade changes price/tick, so xi is recomputed from the new tick's
        liquidity before each trade. Trajectories drop out as their count is
        exhausted.

        Args:
            counts: Integer array, shape (num_trajectories,) -- per-trajectory arrival count.
            xi_index: Index into _compute_local_xi() return tuple (0=sell, 1=buy).
            process_fn: One of _process_sell_single or _process_buy_single.
        """
        max_count = int(np.max(counts)) if np.any(counts > 0) else 0
        for i in range(max_count):
            active = counts > i
            if not np.any(active):
                break
            xi = self._compute_local_xi()[xi_index]
            process_fn(active, xi)

    def _process_arrivals_alternating(self, sell_counts: np.ndarray, buy_counts: np.ndarray) -> None:
        """
        Process sell and buy arrivals in interleaved rounds with randomized intra-round ordering.

        Each round i:
        - Trajectories with sell_counts > i execute one sell
        - Trajectories with buy_counts > i execute one buy
        - The within-round order (sell-first or buy-first) is chosen randomly each round

        Once one side is exhausted, remaining trades of the other side continue alone.
        Using self.rng ensures reproducibility via the seed parameter.

        Args:
            sell_counts: Integer array, shape (num_trajectories,)
            buy_counts:  Integer array, shape (num_trajectories,)
        """
        max_count = int(np.max(np.maximum(sell_counts, buy_counts))) \
            if np.any((sell_counts > 0) | (buy_counts > 0)) else 0
        for i in range(max_count):
            active_sell = sell_counts > i
            active_buy  = buy_counts  > i
            sell_first  = bool(self.rng.integers(0, 2))
            if sell_first:
                if np.any(active_sell):
                    self._process_sell_single(active_sell, self._compute_local_xi()[0])
                if np.any(active_buy):
                    self._process_buy_single(active_buy,  self._compute_local_xi()[1])
            else:
                if np.any(active_buy):
                    self._process_buy_single(active_buy,  self._compute_local_xi()[1])
                if np.any(active_sell):
                    self._process_sell_single(active_sell, self._compute_local_xi()[0])

    def update_state(self, arrivals: np.ndarray, action: np.ndarray):
        """
        Process one timestep: rebalance LP, execute swaps, advance time.

        Args:
            arrivals: Integer array of shape (num_trajectories, 2)
                      Column 0: sell_token0 arrival counts (token0 into pool, price decreases)
                      Column 1: buy_token0 arrival counts (token0 out of pool, price increases)
            action: Agent action array (for LP positioning, not used in swap)

        Each arrival count triggers that many individual trades. Each trade uses
        local xi (current tick's capacity) and may cross one tick boundary,
        after which xi is recomputed from the new tick's liquidity.
        """
        if self.state is None:
            raise ValueError("State not initialized. Call reset() first.")

        if action is not None:
            self._rebalance(self.validate_action(action))

        # Cap arrival counts to prevent runaway loops
        sell_counts = np.minimum(arrivals[:, 0].astype(np.int64), self.max_arrivals_per_step)
        buy_counts = np.minimum(arrivals[:, 1].astype(np.int64), self.max_arrivals_per_step)

        self._process_arrivals_alternating(sell_counts, buy_counts)

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


