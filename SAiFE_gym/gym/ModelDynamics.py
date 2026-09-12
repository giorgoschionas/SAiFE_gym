
import abc

import gymnasium
import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, ASSET_PRICE_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
    LP_UNCLAIMED_FEES0_KEY, LP_UNCLAIMED_FEES1_KEY,
    LP_FEE_SNAPSHOT0_KEY, LP_FEE_SNAPSHOT1_KEY,
    LP_EVER_DEPLOYED_KEY, INITIAL_WEALTH_KEY, GAS_COST_KEY,
)
from SAiFE_gym.gym.helpers.AMM_utils import get_position_value_vec
from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel
from SAiFE_gym.stochastic_processes.fee_accounting_models import FeeAccountingModel, UniswapV3FeeAccounting
from SAiFE_gym.stochastic_processes.midprice_models import MidpriceModel
from SAiFE_gym.stochastic_processes.price_impact_models import (
    PriceImpactModel,
    OneTickUniswapV3PriceImpact,
)

class ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        price_impact_model: PriceImpactModel = None,
        fee_accounting_model: FeeAccountingModel = None,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.price_impact_model = price_impact_model
        self.fee_accounting_model = fee_accounting_model
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
    Uniswap V3 Model Dynamics: Constant Product Market Maker with Concentrated Liquidity.

    """

    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        price_impact_model: PriceImpactModel = None,
        fee_accounting_model: FeeAccountingModel = None,
        num_trajectories: int = 1,
        fee_tier: float = 0.003,           # 0.3% fee tier
        tau: int = 5,                      # Number of ticks around current tick
        num_ticks: int = 2000,             # Total ticks to track in liquidity array
        exponential_value: float = 1.0001, # Base for exponential tick spacing
        gas_cost: float = 20,      # Fixed cost per rebalance in token1 units
        swap_fee_rate: float = 0.0,        # Fee rate on imbalanced swap amount
        seed: int = None,
    ):
        super().__init__(midprice_model = midprice_model,
                         arrival_model = arrival_model,
                         price_impact_model = price_impact_model,
                         fee_accounting_model = fee_accounting_model,
                         num_trajectories = num_trajectories,
                         seed = seed)

        self.price_impact_model = self.price_impact_model or OneTickUniswapV3PriceImpact(
            num_trajectories=num_trajectories,
            seed=seed,
        )
        self.fee_accounting_model = self.fee_accounting_model or UniswapV3FeeAccounting()
        self.fee_tier = fee_tier
        self.fee_multiplier = fee_tier / (1.0 - fee_tier)
        self.tau = tau
        self.num_ticks = num_ticks
        self.exponential_value = exponential_value
        self.gas_cost = gas_cost
        self.swap_fee_rate = swap_fee_rate

        # Track the center of the liquidity array (set during state initialization)
        self.tick_lower_global = None
        # Precomputed lattice AMM[i] = sqrt(exponential_value^i); built in _build_sqrt_grid()
        # once tick_lower_global is known. Shape: (num_ticks + 1,) so AMM[i+1] at the last
        # interior tick is a safe lookup.
        self.sqrt_grid = None
        # Cached arrivals from the most recent get_arrivals() call. Read by diagnostic
        # agents that key off realised order flow. Shape: (num_trajectories, 2) [sell, buy].
        self.last_arrivals = np.zeros((num_trajectories, 2), dtype=bool)

    def _build_sqrt_grid(self):
        """Precompute the lattice `AMM[i] = sqrt(r^i)` for every reachable absolute tick.

        Must be called after `tick_lower_global` is assigned (see
        `AMMEnvironment._build_initial_state`).
        """
        absolute_ticks = self.tick_lower_global + np.arange(
            self.num_ticks + 1, dtype=np.float64
        )
        self.sqrt_grid = np.sqrt(self.exponential_value ** absolute_ticks)

    def get_action_space(self):
        """
        Return the low-level command action space for the model dynamics.

        This raw Box is the simulator execution format, not the canonical
        discrete LP decision space. For agents that should choose one unique
        no-op plus finite rebalance ranges, wrap the environment with
        SAiFE_gym.wrappers.DiscreteActionWrapper or DiscreteActionVecEnv.

        Action format: [lower_offset, upper_offset, hold_flag]
        - lower_offset: Tick offset from current tick (range: -tau to tau-1)
        - upper_offset: Tick offset from current tick (range: -tau+1 to tau)
        - hold_flag: <= 0 triggers rebalance, > 0 holds current position

        Constraint: lower_offset < upper_offset (enforced by validate_action)
        The LP always deploys all available wealth into the specified range.
        """
        return gymnasium.spaces.Box(
            low=np.array([-self.tau, -self.tau + 1, -1.0], dtype=np.float32),
            high=np.array([self.tau - 1, self.tau, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Validate and clip action to ensure constraints.

        Args:
            action: (num_trajectories, 2) array of [lower_offset, upper_offset]

        Returns:
            Validated action with same shape, rounded to integers.
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

    def _process_swap(self, active: np.ndarray, direction: int) -> None:
        """Process one swap side through price impact and fee accounting."""
        swap_result = self.price_impact_model.process_swap(
            self.state,
            active,
            direction,
            tick_lower_global=self.tick_lower_global,
            sqrt_grid=self.sqrt_grid,
            num_ticks=self.num_ticks,
            fee_multiplier=self.fee_multiplier,
        )
        self.fee_accounting_model.apply_fees(
            self.state,
            swap_result,
            self.fee_multiplier,
        )

    def execute_liquidity_taker_orders(self, order_sizes: np.ndarray) -> dict:
        """
        Execute signed liquidity-taker order sizes against the current v3 pool.

        Positive orders are gross token1 input used to buy token0 from the pool,
        moving the pool price up. Negative orders are gross token0 input sold to
        the pool, moving the pool price down. Fees are paid from the gross input
        budget. This first version executes only complete tick intervals; any
        remainder that cannot cross the next full interval is reported as
        unfilled.
        """
        if self.state is None:
            raise ValueError("State not initialized. Call reset() first.")
        if self.sqrt_grid is None or self.tick_lower_global is None:
            raise ValueError("sqrt_grid is not initialized. Build/reset the environment first.")

        price_before = (self.state[POOL_SQRT_PRICE_KEY] ** 2).copy()
        execution = self.price_impact_model.process_order_sizes(
            self.state,
            order_sizes,
            tick_lower_global=self.tick_lower_global,
            sqrt_grid=self.sqrt_grid,
            num_ticks=self.num_ticks,
            fee_multiplier=self.fee_multiplier,
        )

        for swap_result in execution.swap_results:
            self.fee_accounting_model.apply_fees(
                self.state,
                swap_result,
                self.fee_multiplier,
            )

        if LP_LIQUIDITY_KEY in self.state:
            self._accrue_lp_fees()

        return {
            "order_input": execution.order_input.copy(),
            "executed_input": execution.executed_input.copy(),
            "unfilled_input": execution.unfilled_input.copy(),
            "curve_input": execution.curve_input.copy(),
            "tick_movement": execution.tick_movement.copy(),
            "direction": execution.direction.copy(),
            "token0_delta": execution.token0_delta.copy(),
            "token1_delta": execution.token1_delta.copy(),
            "fee_input": execution.fee_input.copy(),
            "pool_price_before": price_before,
            "pool_price_after": (self.state[POOL_SQRT_PRICE_KEY] ** 2).copy(),
        }

    def execute_liquidity_taker_speeds(
        self,
        trading_speeds: np.ndarray,
        step_size: float = None,
    ) -> dict:
        """
        Execute signed liquidity-taker trading speeds over one time step.

        Speeds are gross input-token amount per unit simulation time. Positive
        speeds are token1 input used to buy token0; negative speeds are token0
        input sold to the pool. Fees are paid from the gross input budget. The
        executed order request is ``speed * step_size``.
        """
        if step_size is None:
            if self.midprice_model is None or self.midprice_model.step_size is None:
                raise ValueError(
                    "step_size must be provided when midprice_model.step_size is unavailable"
                )
            step_size = self.midprice_model.step_size
        step_size = float(step_size)
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")

        speeds = np.asarray(trading_speeds, dtype=np.float64).reshape(
            self.num_trajectories, -1
        )
        if speeds.shape[1] != 1:
            raise ValueError(
                "trading_speeds must have shape (num_trajectories, 1) or "
                "(num_trajectories,)"
            )
        speeds = speeds[:, 0]
        diagnostics = self.execute_liquidity_taker_orders(
            (speeds * step_size).reshape(self.num_trajectories, 1)
        )
        diagnostics["trading_speed"] = speeds.copy()
        diagnostics["step_size"] = step_size
        return diagnostics

    def _state_or_current(self, state: dict = None) -> dict:
        return self.state if state is None else state

    def get_pool_sqrt_price(self, state: dict = None) -> np.ndarray:
        """Return the pool sqrt price implied by the current tick lattice."""
        state = self._state_or_current(state)
        current_tick = state[POOL_CURRENT_TICK_KEY].astype(np.int64)

        if self.sqrt_grid is not None and self.tick_lower_global is not None:
            idx = current_tick - self.tick_lower_global
            if idx.min() >= 0 and idx.max() < len(self.sqrt_grid):
                return self.sqrt_grid[idx]

        return state[POOL_SQRT_PRICE_KEY]

    def _position_sqrt_bounds(self, state: dict = None):
        state = self._state_or_current(state)
        sqrt_p_lower = np.sqrt(
            self.exponential_value ** state[LP_TICK_LOWER_KEY].astype(np.float64)
        )
        sqrt_p_upper = np.sqrt(
            self.exponential_value ** state[LP_TICK_UPPER_KEY].astype(np.float64)
        )
        return sqrt_p_lower, sqrt_p_upper

    def _compute_gross_lp_fees(self, state: dict = None):
        """
        Compute LP's gross share of pool fees (read-only, does not modify state).

        Returns:
            (gross_fee0, gross_fee1, lp_share_in_range):
                gross_fee0: shape (num_trajectories,)
                gross_fee1: shape (num_trajectories,)
                lp_share_in_range: shape (num_trajectories, num_ticks) — for pool subtraction
        """
        state = self._state_or_current(state)
        lp_liq = state[LP_LIQUIDITY_KEY]
        lp_lower = state[LP_TICK_LOWER_KEY].astype(np.int64)
        lp_upper = state[LP_TICK_UPPER_KEY].astype(np.int64)
        pool_liq = state[POOL_LIQUIDITY_ARRAY_KEY]

        tick_indices = np.arange(self.num_ticks)
        absolute_ticks = self.tick_lower_global + tick_indices

        in_range = (absolute_ticks[None, :] >= lp_lower[:, None]) & \
                   (absolute_ticks[None, :] < lp_upper[:, None])

        total_liq_safe = np.where(pool_liq > 0, pool_liq, 1.0)
        lp_share = np.where(pool_liq > 0, lp_liq[:, None] / total_liq_safe, 0.0)
        lp_share_in_range = lp_share * in_range

        gross_fee0 = np.sum(state[FEES0_KEY] * lp_share_in_range, axis=1)
        gross_fee1 = np.sum(state[FEES1_KEY] * lp_share_in_range, axis=1)

        return gross_fee0, gross_fee1, lp_share_in_range

    def _claimable_lp_fees(self, state: dict = None, include_gross: bool = False):
        state = self._state_or_current(state)
        gross0, gross1, lp_share_in_range = self._compute_gross_lp_fees(state)
        claimable0 = np.maximum(gross0 - state[LP_FEE_SNAPSHOT0_KEY], 0.0)
        claimable1 = np.maximum(gross1 - state[LP_FEE_SNAPSHOT1_KEY], 0.0)
        if include_gross:
            return claimable0, claimable1, gross0, gross1, lp_share_in_range
        return claimable0, claimable1

    def compute_unclaimed_fees(self, state: dict = None):
        """Return LP claimable fees since entry snapshots without mutating state."""
        return self._claimable_lp_fees(state)

    def compute_lp_alpha(self, state: dict = None) -> np.ndarray:
        """Return the LP's token0 value fraction for observation/reward features."""
        state = self._state_or_current(state)
        has_position = state[LP_LIQUIDITY_KEY] > 0
        sqrt_p_lower, sqrt_p_upper = self._position_sqrt_bounds(state)
        alpha = self._compute_token0_fraction_vec(
            self.get_pool_sqrt_price(state),
            sqrt_p_lower,
            sqrt_p_upper,
        )
        return np.where(has_position, alpha, 0.0)

    def compute_lp_token0_amount(self, state: dict = None) -> np.ndarray:
        """Return absolute token0 holdings represented by the active LP position."""
        state = self._state_or_current(state)
        lp_liq = state[LP_LIQUIDITY_KEY]
        has_position = lp_liq > 0
        sqrt_p = self.get_pool_sqrt_price(state)
        sqrt_p_lower, sqrt_p_upper = self._position_sqrt_bounds(state)

        x_in = lp_liq * (1.0 / sqrt_p - 1.0 / sqrt_p_upper)
        x_below = lp_liq * (1.0 / sqrt_p_lower - 1.0 / sqrt_p_upper)
        above = sqrt_p >= sqrt_p_upper
        below = sqrt_p <= sqrt_p_lower
        token0_amount = np.where(above, 0.0, np.where(below, x_below, x_in))
        return np.where(has_position, token0_amount, 0.0)

    def compute_portfolio_value(self, state: dict = None) -> np.ndarray:
        """Return LP mark-to-market wealth including currently unclaimed fees."""
        state = self._state_or_current(state)
        lp_liq = state[LP_LIQUIDITY_KEY]
        has_position = lp_liq > 0
        sqrt_p_lower, sqrt_p_upper = self._position_sqrt_bounds(state)

        pos_value = get_position_value_vec(
            lp_liq,
            state[ASSET_PRICE_KEY],
            self.get_pool_sqrt_price(state),
            sqrt_p_lower,
            sqrt_p_upper,
        )
        unclaimed0, unclaimed1 = self.compute_unclaimed_fees(state)
        unclaimed_value = unclaimed0 * state[ASSET_PRICE_KEY] + unclaimed1
        no_pos_value = np.where(
            state[LP_EVER_DEPLOYED_KEY], 0.0, state[INITIAL_WEALTH_KEY]
        )
        return np.where(has_position, pos_value + unclaimed_value, no_pos_value)

    def _collect_lp_fees(self):
        """
        Collect LP's share of pool fees, excluding pre-entry fees via snapshots.

        net = max(gross - snapshot, 0)
        Only the net portion is subtracted from pool fee arrays.

        Returns:
            (net_fee0, net_fee1): Arrays of shape (num_trajectories,)
        """
        net_fee0, net_fee1, gross_fee0, gross_fee1, lp_share_in_range = (
            self._claimable_lp_fees(include_gross=True)
        )

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

    def _accrue_lp_fees(self):
        """Refresh LP_UNCLAIMED_FEES and roll the per-step delta into LP_COLLECTED_FEES.

        Called at the end of every update_state (after swaps). Read-only with respect
        to pool fee arrays and snapshots — this is pure bookkeeping so the reward
        function can see fee income on hold steps, without interfering with the
        rebalance-time collection path.

        Delta is clamped at zero so that a rebalance (which drops unclaimed back to 0)
        does not subtract from the cumulative lifetime counter.
        """
        new_unclaimed0, new_unclaimed1 = self._claimable_lp_fees()
        delta0 = new_unclaimed0 - self.state[LP_UNCLAIMED_FEES0_KEY]
        delta1 = new_unclaimed1 - self.state[LP_UNCLAIMED_FEES1_KEY]
        self.state[LP_COLLECTED_FEES0_KEY] += np.maximum(delta0, 0.0)
        self.state[LP_COLLECTED_FEES1_KEY] += np.maximum(delta1, 0.0)
        self.state[LP_UNCLAIMED_FEES0_KEY] = new_unclaimed0
        self.state[LP_UNCLAIMED_FEES1_KEY] = new_unclaimed1

    def _compute_token0_fraction_vec(self, sqrt_p, sqrt_p_lower, sqrt_p_upper):
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

    def _rebalance(self, action: np.ndarray, rebalance_mask: np.ndarray = None):
        """
        Rebalance LP position: withdraw old position + fees, deploy into new range.

        Called at the start of update_state() before xi computation and swaps.

        Args:
            action: (num_trajectories, 2) validated action [lower_offset, upper_offset]
            rebalance_mask: Optional boolean mask (num_trajectories,). If provided,
                only trajectories where mask is True are rebalanced; others are held.
        """
        if rebalance_mask is not None and not np.any(rebalance_mask):
            return

        # Save held trajectories' state so it can be restored after computation.
        # When all trajectories rebalance (rebalance_mask is None or all True),
        # no save/restore is needed.
        has_held = rebalance_mask is not None and not np.all(rebalance_mask)
        if has_held:
            hold_mask = ~rebalance_mask
            _SAVE_KEYS = [
                LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
                LP_EVER_DEPLOYED_KEY, LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
                LP_UNCLAIMED_FEES0_KEY, LP_UNCLAIMED_FEES1_KEY,
                LP_FEE_SNAPSHOT0_KEY, LP_FEE_SNAPSHOT1_KEY,
                POOL_LIQUIDITY_ARRAY_KEY, FEES0_KEY, FEES1_KEY,
            ]
            saved = {k: self.state[k][hold_mask].copy() for k in _SAVE_KEYS}

        num_traj = self.num_trajectories
        sqrt_p = self.state[POOL_SQRT_PRICE_KEY]
        external_price = self.state[ASSET_PRICE_KEY]
        current_tick = self.state[POOL_CURRENT_TICK_KEY].astype(np.int64)
        lp_liq = self.state[LP_LIQUIDITY_KEY]

        has_position = lp_liq > 0

        # --- Phase 1: Compute wealth ---
        # Trajectories that have never deployed get initial_wealth as starting capital.
        # Trajectories that were previously deployed but are now bankrupt (lp_liq == 0
        # after gas costs drained their wealth) correctly start at 0, not initial_wealth.
        ever_deployed = self.state[LP_EVER_DEPLOYED_KEY]
        wealth = np.where(ever_deployed, 0.0, self.state[INITIAL_WEALTH_KEY])
        alpha_current = np.zeros(num_traj)

        if np.any(has_position):
            # Compute position value for trajectories with existing positions
            lp_lower = self.state[LP_TICK_LOWER_KEY].astype(np.int64)
            lp_upper = self.state[LP_TICK_UPPER_KEY].astype(np.int64)
            sqrt_p_lower = np.sqrt(self.exponential_value ** lp_lower.astype(np.float64))
            sqrt_p_upper = np.sqrt(self.exponential_value ** lp_upper.astype(np.float64))

            pos_value = get_position_value_vec(lp_liq, external_price, sqrt_p, sqrt_p_lower, sqrt_p_upper)

            # Collect LP's share of fees (subtracts net portion from pool arrays,
            # zeros snapshots). Cumulative LP_COLLECTED_FEES is updated per-step by
            # _accrue_lp_fees, so we do NOT re-add here — the delta between pre- and
            # post-rebalance unclaimed is already baked into the cumulative counter.
            fee0, fee1 = self._collect_lp_fees()

            # Convert fee0 (token0) to token1 value using external price
            fee_value = fee0 * external_price + fee1

            wealth_with_pos = pos_value + fee_value

            alpha_pos = self._compute_token0_fraction_vec(sqrt_p, sqrt_p_lower, sqrt_p_upper)
            token0_value = alpha_pos * pos_value + fee0 * external_price
            alpha_current = np.zeros(num_traj, dtype=np.float64)
            np.divide(
                token0_value,
                wealth_with_pos,
                out=alpha_current,
                where=wealth_with_pos > 0,
            )

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
            alpha_new = self._compute_token0_fraction_vec(sqrt_p, sqrt_p_new_lower, sqrt_p_new_upper)
            swap_cost = self.swap_fee_rate * np.abs(alpha_new - alpha_current) * wealth
            episode_gas_cost = self.state.get(GAS_COST_KEY, self.gas_cost)
            total_cost = np.where(has_position, episode_gas_cost + swap_cost, 0.0)
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
        self.state[LP_EVER_DEPLOYED_KEY] |= (new_L > 0)

        # --- Phase 4: Snapshot pre-existing fees in new range ---
        snapshot0, snapshot1, _ = self._compute_gross_lp_fees()
        self.state[LP_FEE_SNAPSHOT0_KEY] = snapshot0
        self.state[LP_FEE_SNAPSHOT1_KEY] = snapshot1

        # Unclaimed bucket is absorbed into the new position; reset to zero so the next
        # per-step accrual measures only post-rebalance fee earnings.
        self.state[LP_UNCLAIMED_FEES0_KEY] = np.zeros(num_traj, dtype=np.float64)
        self.state[LP_UNCLAIMED_FEES1_KEY] = np.zeros(num_traj, dtype=np.float64)

        # Restore held trajectories after computation
        if has_held:
            for k, v in saved.items():
                self.state[k][hold_mask] = v

    def update_state(self, arrivals: np.ndarray, action: np.ndarray):
        """
        Process one timestep: rebalance LP, execute swaps, advance time.

        Arrivals are Bernoulli trials: at most one sell and one buy per step.
        When both arrive simultaneously, execution order is randomized.

        Args:
            arrivals: Boolean array of shape (num_trajectories, 2)
                      Column 0: sell_token0 arrival (token0 into pool, price decreases)
                      Column 1: buy_token0 arrival (token0 out of pool, price increases)
            action: Agent action array (for LP positioning, not used in swap)
        """
        if self.state is None:
            raise ValueError("State not initialized. Call reset() first.")

        if action is not None:
            tick_action = self.validate_action(action[:, :2])
            rebalance_mask = action[:, 2] <= 0 if action.shape[1] >= 3 else None
            self._rebalance(tick_action, rebalance_mask)

        sell_active = arrivals[:, 0].astype(bool)
        buy_active = arrivals[:, 1].astype(bool)

        both = sell_active & buy_active
        sell_first = bool(self.rng.integers(0, 2)) if np.any(both) else True

        if sell_first:
            self._process_swap(sell_active, -1)
            self._process_swap(buy_active, 1)
        else:
            self._process_swap(buy_active, 1)
            self._process_swap(sell_active, -1)

        self._accrue_lp_fees()

    def get_arrivals(self) -> np.ndarray:
        """
        Get arrivals from arrival model (uses model's internal state).

        Returns:
            np.ndarray: Boolean arrivals array, shape (num_trajectories, 2)
                        Column 0: sell_token0 arrivals
                        Column 1: buy_token0 arrivals
        """
        if self.arrival_model is None:
            arrivals = np.zeros((self.num_trajectories, 2), dtype=bool)
        else:
            arrivals = self.arrival_model.get_arrivals()
        self.last_arrivals = arrivals
        return arrivals
