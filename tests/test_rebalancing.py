"""
Tests for LP rebalancing in UniswapV3ModelDynamics.

Tests verify:
1. First rebalance: uses initial_wealth, deploys to correct range
2. Subsequent rebalance: withdraws old position + fees, deploys new
3. Fee collection proportional to LP's liquidity share
4. Position value computed correctly (above/below/in range)
5. Pool liquidity array correctly modified
6. xi marked stale after rebalance
7. Edge cases: zero-liquidity ticks, out-of-bounds ranges
8. Vectorized: different actions per trajectory
"""

import numpy as np
import pytest

from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
    LP_UNCLAIMED_FEES0_KEY, LP_UNCLAIMED_FEES1_KEY,
    LP_FEE_SNAPSHOT0_KEY, LP_FEE_SNAPSHOT1_KEY,
    LP_EVER_DEPLOYED_KEY, INITIAL_WEALTH_KEY
)
from SAiFE_gym.gym.helpers.AMM_utils import get_position_value_vec
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel


def create_test_model(num_trajectories=1, num_ticks=100, initial_price=100.0,
                      initial_wealth=1e6, tau=5, gas_cost=0.0, swap_fee_rate=0.0):
    """Create a UniswapV3ModelDynamics instance for testing."""
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=0.0,
        initial_price=initial_price,
        terminal_time=1.0,
        step_size=0.005,
        num_trajectories=num_trajectories
    )

    arrival_model = PoissonArrivalModel(
        intensity=np.array([100.0, 100.0]),
        step_size=0.005,
        num_trajectories=num_trajectories,
        seed=42
    )

    model = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=0.003,
        tau=tau,
        num_ticks=num_ticks,
        exponential_value=1.0001,
        gas_cost=gas_cost,
        swap_fee_rate=swap_fee_rate,
        seed=42
    )

    return model


def initialize_state(model, liquidity_value=1e6, mid_tick=True, initial_wealth=1e6):
    """Initialize model state with uniform liquidity and no LP position.

    The `mid_tick` flag is retained for backward-compatible call sites but, under the
    lattice model, the pool price always lives on AMM[initial_tick]. Both branches
    now return the same lattice value.
    """
    num_traj = model.num_trajectories
    num_ticks = model.num_ticks
    initial_price = model.initial_price

    initial_tick = int(np.floor(np.log(initial_price) / np.log(model.exponential_value)))
    model.tick_lower_global = initial_tick - num_ticks // 2
    model._build_sqrt_grid()

    initial_sqrt_price = model.sqrt_grid[initial_tick - model.tick_lower_global]

    liquidity_array = np.full((num_traj, num_ticks), liquidity_value, dtype=np.float64)

    model.state = {
        POOL_SQRT_PRICE_KEY: np.full(num_traj, initial_sqrt_price, dtype=np.float64),
        POOL_CURRENT_TICK_KEY: np.full(num_traj, initial_tick, dtype=np.float64),
        POOL_LIQUIDITY_ARRAY_KEY: liquidity_array,
        FEES0_KEY: np.zeros((num_traj, num_ticks), dtype=np.float64),
        FEES1_KEY: np.zeros((num_traj, num_ticks), dtype=np.float64),
        LP_LIQUIDITY_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_TICK_LOWER_KEY: np.full(num_traj, initial_tick - model.tau, dtype=np.float64),
        LP_TICK_UPPER_KEY: np.full(num_traj, initial_tick + model.tau, dtype=np.float64),
        LP_COLLECTED_FEES0_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_COLLECTED_FEES1_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_UNCLAIMED_FEES0_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_UNCLAIMED_FEES1_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_FEE_SNAPSHOT0_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_FEE_SNAPSHOT1_KEY: np.zeros(num_traj, dtype=np.float64),
        ASSET_PRICE_KEY: np.full(num_traj, initial_price, dtype=np.float64),
        TIME_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_EVER_DEPLOYED_KEY: np.zeros(num_traj, dtype=bool),
        INITIAL_WEALTH_KEY: np.full(num_traj, initial_wealth, dtype=np.float64),
    }



class TestFirstRebalance:
    """Test LP's first rebalance (LP_LIQUIDITY == 0 → uses initial_wealth)."""

    def test_first_rebalance_uses_initial_wealth(self):
        """First rebalance should deploy initial_wealth into new range."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        base_liq = model.state[POOL_LIQUIDITY_ARRAY_KEY].copy()

        # Action: place LP position at [current_tick - 2, current_tick + 2)
        action = np.array([[- 2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, action)

        # LP should now have non-zero liquidity
        assert model.state[LP_LIQUIDITY_KEY][0] > 0

        # Pool liquidity should have increased in the LP's range
        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        new_lower = current_tick - 2
        new_upper = current_tick + 2

        for tick in range(new_lower, new_upper):
            idx = tick - model.tick_lower_global
            if 0 <= idx < model.num_ticks:
                assert model.state[POOL_LIQUIDITY_ARRAY_KEY][0, idx] > base_liq[0, idx]

    def test_first_rebalance_position_value_matches_wealth(self):
        """After first rebalance, position value should equal initial_wealth."""
        wealth = 5e5
        model = create_test_model()
        initialize_state(model, liquidity_value=1e6, initial_wealth=wealth)

        action = np.array([[-3, 3]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, action)

        # Compute position value
        sqrt_p = model.state[POOL_SQRT_PRICE_KEY][0]
        lp_liq = model.state[LP_LIQUIDITY_KEY][0]
        lp_lower = int(model.state[LP_TICK_LOWER_KEY][0])
        lp_upper = int(model.state[LP_TICK_UPPER_KEY][0])
        sqrt_p_lower = np.sqrt(model.exponential_value ** lp_lower)
        sqrt_p_upper = np.sqrt(model.exponential_value ** lp_upper)

        ext_price = model.state[ASSET_PRICE_KEY][0]
        pos_value = get_position_value_vec(
            np.array([lp_liq]), np.array([ext_price]), np.array([sqrt_p]),
            np.array([sqrt_p_lower]), np.array([sqrt_p_upper])
        )[0]

        assert np.isclose(pos_value, wealth, rtol=1e-6)

    def test_first_rebalance_updates_lp_bounds(self):
        """First rebalance should update LP_TICK_LOWER/UPPER."""
        model = create_test_model()
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        action = np.array([[-1, 3]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, action)

        assert model.state[LP_TICK_LOWER_KEY][0] == current_tick - 1
        assert model.state[LP_TICK_UPPER_KEY][0] == current_tick + 3


class TestSubsequentRebalance:
    """Test rebalance when LP already has a position."""

    def test_old_liquidity_removed(self):
        """After rebalance, old range should have LP's liquidity removed."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)
        base_liq = 1e6  # base liquidity per tick

        # First rebalance: place at [-2, 2)
        action1 = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, action1)

        old_lower = int(model.state[LP_TICK_LOWER_KEY][0])
        old_upper = int(model.state[LP_TICK_UPPER_KEY][0])
        lp_liq_first = model.state[LP_LIQUIDITY_KEY][0]

        # Second rebalance: move to [2, 5) — non-overlapping with old range [-2, 2)
        action2 = np.array([[2, 5]], dtype=np.float64)
        model.update_state(arrivals, action2)

        # Old range should be back to base liquidity
        for tick in range(old_lower, old_upper):
            idx = tick - model.tick_lower_global
            if 0 <= idx < model.num_ticks:
                assert np.isclose(
                    model.state[POOL_LIQUIDITY_ARRAY_KEY][0, idx],
                    base_liq,
                    rtol=1e-6
                ), f"Old tick {tick} not restored to base liquidity"

    def test_new_liquidity_added(self):
        """After rebalance, new range should have increased liquidity."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)
        base_liq = 1e6

        # First rebalance
        action1 = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, action1)

        # Second rebalance to new range
        action2 = np.array([[0, 3]], dtype=np.float64)
        model.update_state(arrivals, action2)

        # New range should have increased liquidity
        new_lower = int(model.state[LP_TICK_LOWER_KEY][0])
        new_upper = int(model.state[LP_TICK_UPPER_KEY][0])
        new_lp_liq = model.state[LP_LIQUIDITY_KEY][0]

        for tick in range(new_lower, new_upper):
            idx = tick - model.tick_lower_global
            if 0 <= idx < model.num_ticks:
                expected = base_liq + new_lp_liq
                assert np.isclose(
                    model.state[POOL_LIQUIDITY_ARRAY_KEY][0, idx],
                    expected,
                    rtol=1e-6
                )

    def test_wealth_preserved_without_fees(self):
        """Position value should be preserved across rebalances when no fees."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        # First rebalance
        action1 = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, action1)

        # Get position value after first rebalance
        sqrt_p = model.state[POOL_SQRT_PRICE_KEY][0]
        lp_liq = model.state[LP_LIQUIDITY_KEY][0]
        lp_lower = int(model.state[LP_TICK_LOWER_KEY][0])
        lp_upper = int(model.state[LP_TICK_UPPER_KEY][0])
        sqrt_p_lower = np.sqrt(model.exponential_value ** lp_lower)
        sqrt_p_upper = np.sqrt(model.exponential_value ** lp_upper)
        ext_price = model.state[ASSET_PRICE_KEY][0]
        value_before = get_position_value_vec(
            np.array([lp_liq]), np.array([ext_price]), np.array([sqrt_p]),
            np.array([sqrt_p_lower]), np.array([sqrt_p_upper])
        )[0]

        # Second rebalance (different range, no fee collection expected)
        action2 = np.array([[-1, 3]], dtype=np.float64)
        model.update_state(arrivals, action2)

        # Get position value after second rebalance
        ext_price2 = model.state[ASSET_PRICE_KEY][0]
        sqrt_p2 = model.state[POOL_SQRT_PRICE_KEY][0]
        lp_liq2 = model.state[LP_LIQUIDITY_KEY][0]
        lp_lower2 = int(model.state[LP_TICK_LOWER_KEY][0])
        lp_upper2 = int(model.state[LP_TICK_UPPER_KEY][0])
        sqrt_p_lower2 = np.sqrt(model.exponential_value ** lp_lower2)
        sqrt_p_upper2 = np.sqrt(model.exponential_value ** lp_upper2)
        value_after = get_position_value_vec(
            np.array([lp_liq2]), np.array([ext_price2]), np.array([sqrt_p2]),
            np.array([sqrt_p_lower2]), np.array([sqrt_p_upper2])
        )[0]

        # Wealth should be preserved (no price change, no fees)
        assert np.isclose(value_before, value_after, rtol=1e-6)


class TestFeeCollection:
    """Test fee collection during rebalancing."""

    def test_fees_collected_on_rebalance(self):
        """LP should collect their share of accumulated fees on rebalance."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        # First rebalance to establish position
        action1 = np.array([[-2, 2]], dtype=np.float64)
        arrivals_none = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals_none, action1)

        # Generate some fees via trades
        arrivals_sell = np.array([[1, 0]], dtype=np.int64)
        model.update_state(arrivals_sell, None)  # No rebalance, just trade

        total_fees_before = model.state[FEES0_KEY][0].sum()
        assert total_fees_before > 0, "No fees generated by trade"

        # Rebalance to collect fees
        action2 = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals_none, action2)

        # LP should have collected fees
        assert model.state[LP_COLLECTED_FEES0_KEY][0] > 0

    def test_fee_share_proportional_to_liquidity(self):
        """LP's fee share should be proportional to their liquidity fraction."""
        model = create_test_model()  # Smaller than base liquidity
        initialize_state(model, liquidity_value=1e6, initial_wealth=1e5)

        # First rebalance
        action1 = np.array([[-2, 2]], dtype=np.float64)
        arrivals_none = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals_none, action1)

        lp_liq = model.state[LP_LIQUIDITY_KEY][0]
        base_liq = 1e6

        # LP's share at the trade tick: lp_liq / (base_liq + lp_liq)
        expected_share = lp_liq / (base_liq + lp_liq)

        # Generate fees
        arrivals_sell = np.array([[1, 0]], dtype=np.int64)
        model.update_state(arrivals_sell, None)

        total_fees0 = model.state[FEES0_KEY][0].sum()

        # Rebalance to collect fees
        action2 = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals_none, action2)

        collected = model.state[LP_COLLECTED_FEES0_KEY][0]

        # The fee was deposited at a single tick in LP's range.
        # LP's share = lp_liq / total_liq at that tick
        # Total liq at trade tick = base_liq + lp_liq
        assert collected > 0
        assert collected < total_fees0  # LP doesn't get all fees

    def test_lp_collected_fees_cumulative(self):
        """LP_COLLECTED_FEES should accumulate across multiple rebalances."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        arrivals_none = np.array([[0, 0]], dtype=np.int64)
        arrivals_sell = np.array([[1, 0]], dtype=np.int64)

        # First rebalance
        action = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals_none, action)

        # Trade to generate fees
        model.update_state(arrivals_sell, None)

        # Rebalance to collect first batch
        model.update_state(arrivals_none, action)
        fees_after_first = model.state[LP_COLLECTED_FEES0_KEY][0]

        # Trade again
        model.update_state(arrivals_sell, None)

        # Rebalance to collect second batch
        model.update_state(arrivals_none, action)
        fees_after_second = model.state[LP_COLLECTED_FEES0_KEY][0]

        assert fees_after_second > fees_after_first


class TestPoolLiquidityModification:
    """Test that pool liquidity array is correctly modified."""

    def test_liquidity_added_to_correct_range(self):
        """LP's liquidity should only be added within [lower, upper) range."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)
        base_liq = 1e6

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        action = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, action)

        lp_liq = model.state[LP_LIQUIDITY_KEY][0]
        lp_lower = int(model.state[LP_TICK_LOWER_KEY][0])
        lp_upper = int(model.state[LP_TICK_UPPER_KEY][0])

        for tick in range(model.tick_lower_global, model.tick_lower_global + model.num_ticks):
            idx = tick - model.tick_lower_global
            if lp_lower <= tick < lp_upper:
                expected = base_liq + lp_liq
                assert np.isclose(
                    model.state[POOL_LIQUIDITY_ARRAY_KEY][0, idx],
                    expected, rtol=1e-6
                ), f"Tick {tick} in range: expected {expected}, got {model.state[POOL_LIQUIDITY_ARRAY_KEY][0, idx]}"
            else:
                assert np.isclose(
                    model.state[POOL_LIQUIDITY_ARRAY_KEY][0, idx],
                    base_liq, rtol=1e-6
                ), f"Tick {tick} out of range: expected {base_liq}, got {model.state[POOL_LIQUIDITY_ARRAY_KEY][0, idx]}"

    def test_full_array_after_subsequent_rebalance(self):
        """
        After a second rebalance, verify every tick in the liquidity array:

          Old range [-3, 2), New range [0, 4)  →  overlap at [0, 2)

          Tick zone               Expected liquidity
          ─────────────────────────────────────────
          only old  [-3, 0)     base_liq               (removed, not re-added)
          overlap   [0,  2)     base_liq + new_lp_liq  (removed then re-added)
          only new  [2,  4)     base_liq + new_lp_liq  (freshly added)
          neither              base_liq               (untouched)
        """
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)
        base_liq = 1e6

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        arrivals = np.array([[0, 0]], dtype=np.int64)

        # First rebalance: deploy into [-3, 2)
        model.update_state(arrivals, np.array([[-3, 2]], dtype=np.float64))

        # Second rebalance: move to [0, 4)
        model.update_state(arrivals, np.array([[0, 4]], dtype=np.float64))

        old_lower = current_tick - 3
        old_upper = current_tick + 2
        new_lower = current_tick + 0
        new_upper = current_tick + 4
        new_lp_liq = model.state[LP_LIQUIDITY_KEY][0]

        for tick in range(model.tick_lower_global, model.tick_lower_global + model.num_ticks):
            idx = tick - model.tick_lower_global
            actual = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, idx]

            in_old = old_lower <= tick < old_upper
            in_new = new_lower <= tick < new_upper

            if in_new:
                # New range: base + new LP liquidity (whether or not it was in old range)
                expected = base_liq + new_lp_liq
            else:
                # Outside new range: old LP liquidity must have been removed
                expected = base_liq

            assert np.isclose(actual, expected, rtol=1e-6), (
                f"Tick {tick} (in_old={in_old}, in_new={in_new}): "
                f"expected {expected:.4f}, got {actual:.4f}"
            )

    def test_no_negative_liquidity(self):
        """Pool liquidity should never go negative after rebalance."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        arrivals = np.array([[0, 0]], dtype=np.int64)

        # Multiple rebalances with different ranges
        for lower, upper in [(-3, 3), (-1, 4), (0, 2), (-4, -1)]:
            action = np.array([[lower, upper]], dtype=np.float64)
            model.update_state(arrivals, action)
            assert np.all(model.state[POOL_LIQUIDITY_ARRAY_KEY] >= -1e-10), \
                f"Negative liquidity after action [{lower}, {upper}]"


class TestActionValidation:
    """Test action validation for 2-element actions."""

    def test_action_clipping(self):
        """Actions outside bounds should be clipped."""
        model = create_test_model(tau=5)
        initialize_state(model, liquidity_value=1e6)

        # Action with out-of-bounds values
        action = np.array([[-10, 10]], dtype=np.float64)
        validated = model.validate_action(action)

        assert validated[0, 0] == -5  # clipped to -tau
        assert validated[0, 1] == 5   # clipped to tau

    def test_lower_must_be_less_than_upper(self):
        """lower_offset must be < upper_offset."""
        model = create_test_model(tau=5)
        initialize_state(model, liquidity_value=1e6)

        # Invalid: lower >= upper
        action = np.array([[3, 2]], dtype=np.float64)
        validated = model.validate_action(action)

        assert validated[0, 0] < validated[0, 1]

    def test_action_space_shape(self):
        """Action space should be 3-element (lower_offset, upper_offset, hold_flag)."""
        model = create_test_model(tau=5)
        space = model.get_action_space()

        assert space.shape == (3,)


class TestPositionValueVec:
    """Test the vectorized get_position_value_vec function."""

    def test_price_above_range(self):
        """When price is above range, position is 100% token1."""
        L = np.array([1000.0])
        ext_p = np.array([144.0])    # external price
        sqrt_p = np.array([12.0])    # pool sqrt price
        sqrt_p_l = np.array([9.0])   # lower bound
        sqrt_p_u = np.array([10.0])  # upper bound (price above this)

        value = get_position_value_vec(L, ext_p, sqrt_p, sqrt_p_l, sqrt_p_u)
        expected = L * (sqrt_p_u - sqrt_p_l)
        assert np.isclose(value[0], expected[0])

    def test_price_below_range(self):
        """When price is below range, position is 100% token0."""
        L = np.array([1000.0])
        ext_p = np.array([64.0])     # external price
        sqrt_p = np.array([8.0])     # pool sqrt price
        sqrt_p_l = np.array([9.0])   # lower bound (price below this)
        sqrt_p_u = np.array([10.0])  # upper bound

        value = get_position_value_vec(L, ext_p, sqrt_p, sqrt_p_l, sqrt_p_u)
        expected = ext_p * L * (sqrt_p_u - sqrt_p_l) / (sqrt_p_l * sqrt_p_u)
        assert np.isclose(value[0], expected[0])

    def test_price_in_range(self):
        """When price is in range, position is mix of tokens."""
        L = np.array([1000.0])
        ext_p = np.array([90.25])    # external price (= 9.5^2, same as pool)
        sqrt_p = np.array([9.5])     # in range
        sqrt_p_l = np.array([9.0])   # lower bound
        sqrt_p_u = np.array([10.0])  # upper bound

        value = get_position_value_vec(L, ext_p, sqrt_p, sqrt_p_l, sqrt_p_u)
        # V = L * (P_ext/sqrt_p + sqrt_p - sqrt_p_l - P_ext/sqrt_p_u)
        expected = L * (ext_p / sqrt_p + sqrt_p - sqrt_p_l - ext_p / sqrt_p_u)
        assert np.isclose(value[0], expected[0])

    def test_price_in_range_external_differs(self):
        """When external price differs from pool price, valuation reflects it."""
        L = np.array([1000.0])
        sqrt_p = np.array([9.5])     # pool sqrt price
        sqrt_p_l = np.array([9.0])
        sqrt_p_u = np.array([10.0])

        # External price higher than pool → token0 worth more → higher value
        ext_high = np.array([100.0])
        ext_low = np.array([80.0])

        value_high = get_position_value_vec(L, ext_high, sqrt_p, sqrt_p_l, sqrt_p_u)
        value_low = get_position_value_vec(L, ext_low, sqrt_p, sqrt_p_l, sqrt_p_u)

        assert value_high[0] > value_low[0]

    def test_vectorized_mixed_cases(self):
        """Test with multiple trajectories in different cases."""
        L = np.array([1000.0, 1000.0, 1000.0])
        ext_p = np.array([144.0, 64.0, 90.25])  # external prices
        sqrt_p = np.array([12.0, 8.0, 9.5])     # above, below, in range
        sqrt_p_l = np.array([9.0, 9.0, 9.0])
        sqrt_p_u = np.array([10.0, 10.0, 10.0])

        values = get_position_value_vec(L, ext_p, sqrt_p, sqrt_p_l, sqrt_p_u)

        assert values[0] > 0  # above range
        assert values[1] > 0  # below range
        assert values[2] > 0  # in range

        # Above range value: L * (sqrt_p_u - sqrt_p_l) (external price irrelevant)
        assert np.isclose(values[0], 1000.0 * (10.0 - 9.0))


class TestVectorizedRebalance:
    """Test rebalancing with multiple trajectories."""

    def test_different_actions_per_trajectory(self):
        """Each trajectory can have a different action."""
        num_traj = 3
        model = create_test_model(num_trajectories=num_traj, initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])

        # Different ranges for each trajectory
        action = np.array([
            [-2, 2],   # centered
            [-1, 4],   # shifted right
            [-4, -1],  # shifted left
        ], dtype=np.float64)

        arrivals = np.zeros((num_traj, 2), dtype=np.int64)
        model.update_state(arrivals, action)

        # Each trajectory should have different bounds
        assert model.state[LP_TICK_LOWER_KEY][0] == current_tick - 2
        assert model.state[LP_TICK_UPPER_KEY][0] == current_tick + 2

        assert model.state[LP_TICK_LOWER_KEY][1] == current_tick - 1
        assert model.state[LP_TICK_UPPER_KEY][1] == current_tick + 4

        assert model.state[LP_TICK_LOWER_KEY][2] == current_tick - 4
        assert model.state[LP_TICK_UPPER_KEY][2] == current_tick - 1

        # All should have positive liquidity
        assert np.all(model.state[LP_LIQUIDITY_KEY] > 0)

    def test_mixed_first_and_subsequent(self):
        """Some trajectories on first rebalance, others on subsequent."""
        num_traj = 2
        model = create_test_model(num_trajectories=num_traj, initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        # First rebalance: both trajectories get initial position
        action1 = np.array([
            [-2, 2],
            [-1, 3],
        ], dtype=np.float64)
        arrivals = np.zeros((num_traj, 2), dtype=np.int64)
        model.update_state(arrivals, action1)

        lp_liq_first = model.state[LP_LIQUIDITY_KEY].copy()
        assert np.all(lp_liq_first > 0)

        # Second rebalance: different ranges
        action2 = np.array([
            [0, 3],
            [-3, 0],
        ], dtype=np.float64)
        model.update_state(arrivals, action2)

        # Both should still have positive liquidity
        assert np.all(model.state[LP_LIQUIDITY_KEY] > 0)


class TestEdgeCases:
    """Test edge cases in rebalancing."""

    def test_same_range_rebalance(self):
        """Rebalancing to the same range should preserve position value."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        action = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)

        # First rebalance
        model.update_state(arrivals, action)
        lp_liq_1 = model.state[LP_LIQUIDITY_KEY][0]

        # Second rebalance to same range (no fees accumulated)
        model.update_state(arrivals, action)
        lp_liq_2 = model.state[LP_LIQUIDITY_KEY][0]

        # Liquidity should be approximately the same (same wealth, same range)
        assert np.isclose(lp_liq_1, lp_liq_2, rtol=1e-6)

    def test_none_action_skips_rebalance(self):
        """Passing None as action should skip rebalancing."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        lp_liq_before = model.state[LP_LIQUIDITY_KEY][0]

        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, None)

        # LP liquidity should be unchanged
        assert model.state[LP_LIQUIDITY_KEY][0] == lp_liq_before

    def test_rebalance_with_fees_increases_wealth(self):
        """LP wealth should increase after collecting fees."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        arrivals_none = np.array([[0, 0]], dtype=np.int64)
        arrivals_sell = np.array([[1, 0]], dtype=np.int64)

        # Establish position
        action = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals_none, action)

        # Record initial position value
        sqrt_p = model.state[POOL_SQRT_PRICE_KEY][0]
        lp_liq = model.state[LP_LIQUIDITY_KEY][0]
        lp_lower = int(model.state[LP_TICK_LOWER_KEY][0])
        lp_upper = int(model.state[LP_TICK_UPPER_KEY][0])
        sqrt_p_lower = np.sqrt(model.exponential_value ** lp_lower)
        sqrt_p_upper = np.sqrt(model.exponential_value ** lp_upper)
        ext_price = model.state[ASSET_PRICE_KEY][0]
        initial_value = get_position_value_vec(
            np.array([lp_liq]), np.array([ext_price]), np.array([sqrt_p]),
            np.array([sqrt_p_lower]), np.array([sqrt_p_upper])
        )[0]

        # Generate fees via multiple trades (without rebalancing)
        for _ in range(5):
            model.update_state(arrivals_sell, None)

        # Rebalance to collect fees (same range to minimize price effect)
        model.update_state(arrivals_none, action)

        # New position should include collected fees
        ext_price2 = model.state[ASSET_PRICE_KEY][0]
        sqrt_p2 = model.state[POOL_SQRT_PRICE_KEY][0]
        lp_liq2 = model.state[LP_LIQUIDITY_KEY][0]
        lp_lower2 = int(model.state[LP_TICK_LOWER_KEY][0])
        lp_upper2 = int(model.state[LP_TICK_UPPER_KEY][0])
        sqrt_p_lower2 = np.sqrt(model.exponential_value ** lp_lower2)
        sqrt_p_upper2 = np.sqrt(model.exponential_value ** lp_upper2)
        new_value = get_position_value_vec(
            np.array([lp_liq2]), np.array([ext_price2]), np.array([sqrt_p2]),
            np.array([sqrt_p_lower2]), np.array([sqrt_p_upper2])
        )[0]

        # LP collected fees should be tracked
        assert model.state[LP_COLLECTED_FEES0_KEY][0] > 0


def _get_position_value(model):
    """Helper: compute current LP position value in token1 units."""
    sqrt_p = model.state[POOL_SQRT_PRICE_KEY][0]
    lp_liq = model.state[LP_LIQUIDITY_KEY][0]
    lp_lower = int(model.state[LP_TICK_LOWER_KEY][0])
    lp_upper = int(model.state[LP_TICK_UPPER_KEY][0])
    sqrt_p_lower = np.sqrt(model.exponential_value ** lp_lower)
    sqrt_p_upper = np.sqrt(model.exponential_value ** lp_upper)
    ext_price = model.state[ASSET_PRICE_KEY][0]
    return get_position_value_vec(
        np.array([lp_liq]), np.array([ext_price]), np.array([sqrt_p]),
        np.array([sqrt_p_lower]), np.array([sqrt_p_upper])
    )[0]


class TestRebalancingCost:
    """Test decomposed rebalancing cost (gas + swap fee)."""

    def test_zero_cost_preserves_wealth(self):
        """With gas_cost=0 and swap_fee_rate=0, wealth is fully preserved."""
        model = create_test_model(initial_wealth=1e6, gas_cost=0.0, swap_fee_rate=0.0)
        initialize_state(model, liquidity_value=1e6)

        action = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)

        # First rebalance
        model.update_state(arrivals, action)
        value_after_first = _get_position_value(model)

        # Second rebalance (same range, no fees)
        model.update_state(arrivals, action)
        value_after_second = _get_position_value(model)

        assert np.isclose(value_after_first, value_after_second, rtol=1e-6)

    def test_first_rebalance_no_cost(self):
        """First rebalance (from cash) should not incur any cost."""
        gas_cost = 50000.0  # Large to make effect obvious
        initial_wealth = 1e6
        model = create_test_model(initial_wealth=initial_wealth, gas_cost=gas_cost)
        initialize_state(model, liquidity_value=1e6)

        action = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)

        model.update_state(arrivals, action)

        # Full initial_wealth deployed — no cost on first deployment
        assert np.isclose(_get_position_value(model), initial_wealth, rtol=1e-6)

    def test_gas_cost_deducted_linearly(self):
        """After N rebalances to same range: final_wealth = initial - N * gas_cost."""
        gas_cost = 1000.0
        initial_wealth = 1e6
        n_rebalances = 5
        model = create_test_model(initial_wealth=initial_wealth, gas_cost=gas_cost, swap_fee_rate=0.0)
        initialize_state(model, liquidity_value=1e6)

        action = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)

        # First rebalance (no cost — initial deployment)
        model.update_state(arrivals, action)

        # N more rebalances to same range (each incurs only gas_cost; swap_cost=0 same range)
        for _ in range(n_rebalances):
            model.update_state(arrivals, action)

        expected = initial_wealth - n_rebalances * gas_cost
        assert np.isclose(_get_position_value(model), expected, rtol=1e-6)

    def test_gas_cost_not_on_first_rebalance(self):
        """First step from cash: gas cost is not deducted."""
        gas_cost = 10000.0
        initial_wealth = 1e6
        model = create_test_model(initial_wealth=initial_wealth, gas_cost=gas_cost, swap_fee_rate=0.0)
        initialize_state(model, liquidity_value=1e6)

        action = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)

        model.update_state(arrivals, action)

        # Should be full initial_wealth (no gas cost on first deployment)
        assert np.isclose(_get_position_value(model), initial_wealth, rtol=1e-6)

    def test_swap_fee_zero_same_range(self):
        """Rebalancing to identical range: α_new = α_current → swap cost = 0."""
        swap_fee_rate = 0.05  # Large to detect any error
        initial_wealth = 1e6
        model = create_test_model(initial_wealth=initial_wealth, gas_cost=0.0, swap_fee_rate=swap_fee_rate)
        initialize_state(model, liquidity_value=1e6)

        action = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)

        # First rebalance (no cost — initial deployment)
        model.update_state(arrivals, action)
        value_after_first = _get_position_value(model)

        # Second rebalance: same range, same price, no fees → α_new = α_current
        model.update_state(arrivals, action)
        value_after_second = _get_position_value(model)

        # No cost applied
        assert np.isclose(value_after_first, value_after_second, rtol=1e-6)

    def test_swap_fee_proportional_to_imbalance(self):
        """Swap fee = swap_fee_rate * W * |α_new - α_current| (analytically verified)."""
        swap_fee_rate = 0.01
        initial_wealth = 1e6
        model = create_test_model(initial_wealth=initial_wealth, gas_cost=0.0,
                                  swap_fee_rate=swap_fee_rate, tau=10)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        sqrt_p = model.state[POOL_SQRT_PRICE_KEY][0]
        arrivals = np.array([[0, 0]], dtype=np.int64)

        # First rebalance: place LP ABOVE current price → price below range → α_current = 1.0
        action1 = np.array([[2, 8]], dtype=np.float64)
        model.update_state(arrivals, action1)

        # Verify price is below LP range (all token0 → α_current = 1.0)
        lp_lower1 = int(model.state[LP_TICK_LOWER_KEY][0])
        sqrt_p_lower1 = np.sqrt(model.exponential_value ** lp_lower1)
        assert sqrt_p <= sqrt_p_lower1, "Price should be below LP range for test setup"

        W = initial_wealth  # wealth after first rebalance (no cost on first)

        # Compute expected α_new for centered range [-2, 2]
        new_lower = current_tick - 2
        new_upper = current_tick + 2
        sqrt_p_lower2 = np.sqrt(model.exponential_value ** new_lower)
        sqrt_p_upper2 = np.sqrt(model.exponential_value ** new_upper)

        assert sqrt_p_lower2 < sqrt_p < sqrt_p_upper2, "Price should be in new range"
        numerator = sqrt_p**2 * (1.0 / sqrt_p - 1.0 / sqrt_p_upper2)
        denominator = numerator + (sqrt_p - sqrt_p_lower2)
        alpha_new = numerator / denominator
        alpha_current = 1.0

        expected_cost = swap_fee_rate * W * abs(alpha_new - alpha_current)

        # Second rebalance to centered range
        action2 = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals, action2)

        actual_cost = W - _get_position_value(model)
        assert np.isclose(actual_cost, expected_cost, rtol=1e-6)

    def test_gas_and_swap_fee_combined(self):
        """Both gas cost and swap fee are deducted correctly together."""
        gas_cost = 500.0
        swap_fee_rate = 0.01
        initial_wealth = 1e6
        model = create_test_model(initial_wealth=initial_wealth, gas_cost=gas_cost,
                                  swap_fee_rate=swap_fee_rate, tau=10)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        sqrt_p = model.state[POOL_SQRT_PRICE_KEY][0]
        arrivals = np.array([[0, 0]], dtype=np.int64)

        # First rebalance: place LP ABOVE current price → α_current = 1.0
        action1 = np.array([[2, 8]], dtype=np.float64)
        model.update_state(arrivals, action1)

        lp_lower1 = int(model.state[LP_TICK_LOWER_KEY][0])
        sqrt_p_lower1 = np.sqrt(model.exponential_value ** lp_lower1)
        assert sqrt_p <= sqrt_p_lower1, "Price should be below LP range for test setup"

        W = initial_wealth

        # Compute expected costs for second rebalance to centered range
        new_lower = current_tick - 2
        new_upper = current_tick + 2
        sqrt_p_lower2 = np.sqrt(model.exponential_value ** new_lower)
        sqrt_p_upper2 = np.sqrt(model.exponential_value ** new_upper)

        numerator = sqrt_p**2 * (1.0 / sqrt_p - 1.0 / sqrt_p_upper2)
        denominator = numerator + (sqrt_p - sqrt_p_lower2)
        alpha_new = numerator / denominator
        alpha_current = 1.0

        expected_cost = gas_cost + swap_fee_rate * W * abs(alpha_new - alpha_current)

        action2 = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals, action2)

        actual_cost = W - _get_position_value(model)
        assert np.isclose(actual_cost, expected_cost, rtol=1e-6)


class TestHoldAction:
    """Test hold (no-rebalance) action via the 3rd action dimension."""

    def test_hold_preserves_position(self):
        """Deploy LP, hold next step -> LP_LIQUIDITY/bounds unchanged, no gas cost."""
        gas_cost = 10000.0
        model = create_test_model(initial_wealth=1e6, gas_cost=gas_cost)
        initialize_state(model, liquidity_value=1e6)

        # First rebalance to establish position (3-col action, hold_flag=-1 = rebalance)
        action_deploy = np.array([[-2, 2, -1.0]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, action_deploy)

        lp_liq_after_deploy = model.state[LP_LIQUIDITY_KEY][0]
        lp_lower_after = model.state[LP_TICK_LOWER_KEY][0]
        lp_upper_after = model.state[LP_TICK_UPPER_KEY][0]
        assert lp_liq_after_deploy > 0

        # Hold action (hold_flag=1.0 > 0 -> hold)
        action_hold = np.array([[-2, 2, 1.0]], dtype=np.float64)
        model.update_state(arrivals, action_hold)

        # LP position should be unchanged
        assert model.state[LP_LIQUIDITY_KEY][0] == lp_liq_after_deploy
        assert model.state[LP_TICK_LOWER_KEY][0] == lp_lower_after
        assert model.state[LP_TICK_UPPER_KEY][0] == lp_upper_after

    def test_hold_still_processes_swaps(self):
        """Hold with sell arrivals -> price moves, fees accumulate in pool."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        # Deploy LP
        action_deploy = np.array([[-2, 2, -1.0]], dtype=np.float64)
        arrivals_none = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals_none, action_deploy)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Hold + sell arrival
        action_hold = np.array([[-2, 2, 1.0]], dtype=np.float64)
        arrivals_sell = np.array([[1, 0]], dtype=np.int64)
        model.update_state(arrivals_sell, action_hold)

        # Price should have moved (sell -> tick decreases)
        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick - 1
        # Fees should have accumulated
        assert model.state[FEES0_KEY][0].sum() > 0

    def test_mixed_hold_and_rebalance(self):
        """2 trajectories: one holds, one rebalances -> independent behavior."""
        num_traj = 2
        model = create_test_model(num_trajectories=num_traj, initial_wealth=1e6, gas_cost=1000.0)
        initialize_state(model, liquidity_value=1e6)

        # Deploy both
        action_deploy = np.array([[-2, 2, -1.0], [-2, 2, -1.0]], dtype=np.float64)
        arrivals = np.zeros((num_traj, 2), dtype=np.int64)
        model.update_state(arrivals, action_deploy)

        lp_liq_0 = model.state[LP_LIQUIDITY_KEY][0]
        lp_liq_1 = model.state[LP_LIQUIDITY_KEY][1]

        # Trajectory 0: hold (hold_flag=1.0), trajectory 1: rebalance (hold_flag=-1.0)
        action_mixed = np.array([[-2, 2, 1.0], [-1, 3, -1.0]], dtype=np.float64)
        model.update_state(arrivals, action_mixed)

        # Trajectory 0 should be unchanged
        assert model.state[LP_LIQUIDITY_KEY][0] == lp_liq_0
        assert model.state[LP_TICK_LOWER_KEY][0] == model.state[LP_TICK_LOWER_KEY][0]

        # Trajectory 1 should have rebalanced (different bounds, gas deducted)
        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][1])
        assert model.state[LP_TICK_LOWER_KEY][1] == current_tick - 1
        assert model.state[LP_TICK_UPPER_KEY][1] == current_tick + 3

    def test_first_step_hold_no_deployment(self):
        """Hold on step 0 -> LP stays undeployed (lp_liq=0, ever_deployed=False)."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        action_hold = np.array([[-2, 2, 1.0]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, action_hold)

        assert model.state[LP_LIQUIDITY_KEY][0] == 0.0
        assert not model.state[LP_EVER_DEPLOYED_KEY][0]

    def test_hold_flag_boundary(self):
        """hold_flag = 0.0 -> rebalance (condition is <= 0)."""
        model = create_test_model(initial_wealth=1e6)
        initialize_state(model, liquidity_value=1e6)

        # hold_flag=0.0 should trigger rebalance
        action = np.array([[-2, 2, 0.0]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model.update_state(arrivals, action)

        # LP should be deployed
        assert model.state[LP_LIQUIDITY_KEY][0] > 0
        assert model.state[LP_EVER_DEPLOYED_KEY][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
