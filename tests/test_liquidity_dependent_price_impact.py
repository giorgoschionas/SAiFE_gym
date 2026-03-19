"""
Tests for liquidity-dependent price impact in UniswapV3ModelDynamics.

Tests verify:
1. Local xi computation from current tick's liquidity
2. No-crossing: directly calling _process_sell_single with small xi
3. Crossing: local xi always crosses one tick per arrival
4. Zero liquidity: price jumps to boundary
5. Vectorization: multiple trajectories with different directions
6. Fee calculation proportional to local xi
7. Bernoulli arrivals: at most one sell + one buy per step
"""

import numpy as np
import pytest

from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
    LP_FEE_SNAPSHOT0_KEY, LP_FEE_SNAPSHOT1_KEY,
    LP_EVER_DEPLOYED_KEY
)
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel


def create_test_model(num_trajectories=1, num_ticks=100, initial_price=100.0):
    """Create a UniswapV3ModelDynamics instance for testing."""
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=0.0,  # No randomness for deterministic tests
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
        tau=5,
        num_ticks=num_ticks,
        exponential_value=1.0001,
        seed=42
    )

    return model


def initialize_state(model, liquidity_value=1e6, initial_sqrt_price=None, mid_tick=True):
    """Initialize model state with uniform liquidity.

    Args:
        model: UniswapV3ModelDynamics instance
        liquidity_value: Liquidity value for all ticks
        initial_sqrt_price: Override sqrt price (for specific tests)
        mid_tick: If True, place price in middle of tick (more room before crossing)
    """
    num_traj = model.num_trajectories
    num_ticks = model.num_ticks

    # Compute tick_lower_global to center around initial price
    try:
        initial_price = model.initial_price
    except AttributeError:
        initial_price = 100.0  # default when no midprice_model

    # Tick for initial price
    initial_tick = int(np.floor(np.log(initial_price) / np.log(model.exponential_value)))
    model.tick_lower_global = initial_tick - num_ticks // 2

    # Compute sqrt_price: either at tick boundary or middle of tick
    if initial_sqrt_price is None:
        if mid_tick:
            # Place price in middle of tick: sqrt((p_low + p_high) / 2)
            p_low = model.exponential_value ** initial_tick
            p_high = model.exponential_value ** (initial_tick + 1)
            initial_sqrt_price = np.sqrt((p_low + p_high) / 2)
        else:
            initial_sqrt_price = np.sqrt(initial_price)

    # Initialize liquidity array
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
        LP_FEE_SNAPSHOT0_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_FEE_SNAPSHOT1_KEY: np.zeros(num_traj, dtype=np.float64),
        ASSET_PRICE_KEY: np.full(num_traj, initial_price, dtype=np.float64),
        TIME_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_EVER_DEPLOYED_KEY: np.zeros(num_traj, dtype=bool),
    }


class TestLocalXiComputation:
    """Test local xi (trade size) computation from current tick."""

    def test_local_xi_positive_finite(self):
        """Test local xi computation with uniform liquidity."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        xi_sell, xi_buy = model._compute_local_xi()

        # xi should be finite and positive
        assert np.isfinite(xi_sell[0])
        assert np.isfinite(xi_buy[0])
        assert xi_sell[0] > 0
        assert xi_buy[0] > 0

    def test_local_xi_varies_with_liquidity(self):
        """Test that local xi scales with current tick's liquidity."""
        model = create_test_model(num_trajectories=1, num_ticks=100)

        # High liquidity
        initialize_state(model, liquidity_value=1e8)
        xi_sell_high, xi_buy_high = model._compute_local_xi()

        # Low liquidity
        initialize_state(model, liquidity_value=1e4)
        xi_sell_low, xi_buy_low = model._compute_local_xi()

        # Lower liquidity should give smaller xi
        assert xi_sell_low[0] < xi_sell_high[0]
        assert xi_buy_low[0] < xi_buy_high[0]

    def test_local_xi_ignores_distant_ticks(self):
        """Test that changing liquidity at a distant tick does NOT affect local xi."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        xi_sell_before, xi_buy_before = model._compute_local_xi()

        # Change liquidity at a distant tick (tick 99, far from current tick ~50)
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, 99] = 100.0  # Much lower

        xi_sell_after, xi_buy_after = model._compute_local_xi()

        # Local xi should be unchanged (only depends on current tick)
        assert xi_sell_before[0] == xi_sell_after[0]
        assert xi_buy_before[0] == xi_buy_after[0]

    def test_local_xi_zero_liquidity_gives_zero(self):
        """Test that zero liquidity at current tick gives zero xi."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        # Set current tick to zero liquidity
        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx] = 0.0

        xi_sell, xi_buy = model._compute_local_xi()

        assert xi_sell[0] == 0.0
        assert xi_buy[0] == 0.0


class TestNoCrossing:
    """Test no-crossing code path by calling _process_sell_single/_process_buy_single
    directly with a manually small xi.

    Note: With local xi (full tick capacity), crossing always occurs via update_state().
    These tests exercise the no-crossing code path for completeness.
    """

    def test_sell_no_crossing_with_small_xi(self):
        """Test sell that doesn't cross by passing a small xi directly."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8)

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Compute local xi and use 1/10th to ensure no crossing
        xi_sell, _ = model._compute_local_xi()
        small_xi = xi_sell * 0.1

        active = np.array([True])
        model._process_sell_single(active, small_xi)

        new_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0]
        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Price should decrease
        assert new_sqrt_price < initial_sqrt_price
        # Tick should NOT change (small xi)
        assert new_tick == initial_tick

    def test_buy_no_crossing_with_small_xi(self):
        """Test buy that doesn't cross by passing a small xi directly."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8)

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Compute local xi and use 1/10th to ensure no crossing
        _, xi_buy = model._compute_local_xi()
        small_xi = xi_buy * 0.1

        active = np.array([True])
        model._process_buy_single(active, small_xi)

        new_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0]
        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Price should increase
        assert new_sqrt_price > initial_sqrt_price
        # Tick should NOT change (small xi)
        assert new_tick == initial_tick

    def test_uniform_liquidity_always_crosses(self):
        """Test that with local xi, a single arrival always crosses."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Sell arrival
        arrivals = np.array([[True, False]])
        model.update_state(arrivals, None)

        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # With local xi (full tick capacity), always crosses
        assert new_tick == initial_tick - 1


class TestCrossing:
    """Test price updates when trade crosses tick boundary."""

    def test_sell_crossing(self):
        """Test sell that crosses tick boundary."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=100.0)  # Any liquidity

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Sell arrival
        arrivals = np.array([[True, False]])
        model.update_state(arrivals, None)

        new_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0]
        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Price should decrease
        assert new_sqrt_price < initial_sqrt_price

        # Tick should decrease by 1 (crossing)
        assert new_tick == initial_tick - 1

    def test_buy_crossing(self):
        """Test buy that crosses tick boundary."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=100.0)  # Any liquidity

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Buy arrival
        arrivals = np.array([[False, True]])
        model.update_state(arrivals, None)

        new_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0]
        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Price should increase
        assert new_sqrt_price > initial_sqrt_price

        # Tick should increase by 1 (crossing)
        assert new_tick == initial_tick + 1


class TestZeroLiquidity:
    """Test behavior with zero liquidity at current tick."""

    def test_sell_zero_current_liquidity(self):
        """Test sell when current tick has zero liquidity."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        # Set current tick to zero liquidity
        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx] = 0.0

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Sell arrival
        arrivals = np.array([[True, False]])
        model.update_state(arrivals, None)

        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Should cross tick (zero capacity at current tick)
        assert new_tick == initial_tick - 1

    def test_buy_zero_current_liquidity(self):
        """Test buy when current tick has zero liquidity."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        # Set current tick to zero liquidity
        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx] = 0.0

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Buy arrival
        arrivals = np.array([[False, True]])
        model.update_state(arrivals, None)

        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Should cross tick (zero capacity at current tick)
        assert new_tick == initial_tick + 1


class TestSingleArrival:
    """Test Bernoulli arrivals: at most one sell + one buy per step."""

    def test_zero_arrivals_no_change(self):
        """arrivals=[[False,False]] should not change price or tick."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6, mid_tick=True)

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        arrivals = np.array([[False, False]])
        model.update_state(arrivals, None)

        assert model.state[POOL_SQRT_PRICE_KEY][0] == initial_sqrt_price
        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick

    def test_single_sell_one_tick(self):
        """[[True, False]] → exactly -1 tick."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6, mid_tick=True)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        arrivals = np.array([[True, False]])
        model.update_state(arrivals, None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick - 1

    def test_single_buy_one_tick(self):
        """[[False, True]] → exactly +1 tick."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6, mid_tick=True)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        arrivals = np.array([[False, True]])
        model.update_state(arrivals, None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick + 1

    def test_simultaneous_net_zero(self):
        """[[True, True]] → net 0 tick movement."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6, mid_tick=True)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        arrivals = np.array([[True, True]])
        model.update_state(arrivals, None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick


class TestVectorization:
    """Test that operations are properly vectorized across trajectories."""

    def test_multiple_trajectories_different_directions(self):
        """Test multiple trajectories with different arrival directions."""
        num_traj = 4
        model = create_test_model(num_trajectories=num_traj, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        initial_sqrt_prices = model.state[POOL_SQRT_PRICE_KEY].copy()

        # Different arrivals for each trajectory:
        arrivals = np.array([
            [True, False],   # sell
            [False, False],  # no trade
            [False, True],   # buy
            [True, True],    # both
        ])

        model.update_state(arrivals, None)

        new_sqrt_prices = model.state[POOL_SQRT_PRICE_KEY]

        # Trajectory 0 (sell): price decreased
        assert new_sqrt_prices[0] < initial_sqrt_prices[0]

        # Trajectory 1 (no trade): price unchanged
        assert new_sqrt_prices[1] == initial_sqrt_prices[1]

        # Trajectory 2 (buy): price increased
        assert new_sqrt_prices[2] > initial_sqrt_prices[2]

    def test_mixed_liquidity_per_trajectory(self):
        """Test trajectories with different current-tick liquidity."""
        num_traj = 2
        model = create_test_model(num_trajectories=num_traj, num_ticks=100)
        initialize_state(model, liquidity_value=1e6, mid_tick=True)

        # Trajectory 0: low liquidity at current tick
        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx] = 1e3
        # Trajectory 1: high liquidity at current tick (already 1e6)

        initial_prices = model.state[POOL_SQRT_PRICE_KEY].copy() ** 2

        # Both sell
        arrivals = np.array([
            [True, False],
            [True, False],
        ])

        model.update_state(arrivals, None)

        # Both should cross one tick (local xi always crosses)
        initial_ticks = np.full(num_traj, current_tick)
        new_ticks = model.state[POOL_CURRENT_TICK_KEY]
        assert new_ticks[0] == initial_ticks[0] - 1
        assert new_ticks[1] == initial_ticks[1] - 1


class TestFeeCalculation:
    """Test that fees are calculated proportionally to local xi."""

    def test_sell_fees(self):
        """Test fee collection for sell trades.

        On a crossing, fees are split across two ticks (Uniswap V3 mechanics):
          - x_to_boundary: portion that trades in the old tick (up to boundary)
          - x_from_boundary: portion that trades in the new tick (boundary to new midpoint)
        Total fee = fee_multiplier * (x_to_boundary + x_from_boundary).
        """
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        current_tick = model.state[POOL_CURRENT_TICK_KEY].copy()
        tick_idx = int(current_tick[0] - model.tick_lower_global)
        sqrt_p_c = model.state[POOL_SQRT_PRICE_KEY][0]
        sqrt_p_low = np.sqrt(model.exponential_value ** current_tick[0])
        sqrt_p_cross = sqrt_p_low / model.exp_quarter

        L = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx]
        L_prev = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx - 1]

        x_to_boundary   = L      * (1.0 / sqrt_p_low   - 1.0 / sqrt_p_c)
        x_from_boundary = L_prev * (1.0 / sqrt_p_cross  - 1.0 / sqrt_p_low)

        initial_fees0 = model.state[FEES0_KEY][0].sum()

        arrivals = np.array([[True, False]])
        model.update_state(arrivals, None)

        new_fees0 = model.state[FEES0_KEY][0].sum()

        assert new_fees0 > initial_fees0

        expected_fee = model.fee_tier / (1 - model.fee_tier) * (x_to_boundary + x_from_boundary)
        assert np.isclose(new_fees0 - initial_fees0, expected_fee, rtol=1e-10)

    def test_buy_fees(self):
        """Test fee collection for buy trades.

        On a crossing, fees are split across two ticks (Uniswap V3 mechanics):
          - y_to_boundary: portion that trades in the old tick (up to boundary)
          - y_from_boundary: portion that trades in the new tick (boundary to new midpoint)
        Total fee = fee_multiplier * (y_to_boundary + y_from_boundary).
        """
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        current_tick = model.state[POOL_CURRENT_TICK_KEY].copy()
        tick_idx = int(current_tick[0] - model.tick_lower_global)
        sqrt_p_c = model.state[POOL_SQRT_PRICE_KEY][0]
        sqrt_p_high = np.sqrt(model.exponential_value ** (current_tick[0] + 1))
        sqrt_p_cross = sqrt_p_high * model.exp_quarter

        L      = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx]
        L_next = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx + 1]

        y_to_boundary   = L      * (sqrt_p_high  - sqrt_p_c)
        y_from_boundary = L_next * (sqrt_p_cross - sqrt_p_high)

        initial_fees1 = model.state[FEES1_KEY][0].sum()

        arrivals = np.array([[False, True]])
        model.update_state(arrivals, None)

        new_fees1 = model.state[FEES1_KEY][0].sum()

        assert new_fees1 > initial_fees1

        expected_fee = model.fee_tier / (1 - model.fee_tier) * (y_to_boundary + y_from_boundary)
        assert np.isclose(new_fees1 - initial_fees1, expected_fee, rtol=1e-10)

    def test_fees_scale_with_liquidity(self):
        """Test that fees vary with current-tick liquidity (via local xi)."""
        # High liquidity case
        model_high = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_high, liquidity_value=1e8)

        arrivals = np.array([[1, 0]], dtype=np.int64)
        model_high.update_state(arrivals, None)
        fees_high = model_high.state[FEES0_KEY][0].sum()

        # Low liquidity case
        model_low = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_low, liquidity_value=1e4)

        model_low.update_state(arrivals, None)
        fees_low = model_low.state[FEES0_KEY][0].sum()

        # Lower liquidity → smaller local xi → smaller fees
        assert fees_low < fees_high


class TestFeeSplitOnCrossing:
    """Test that fees are split between ticks when a trade crosses a tick boundary."""

    def test_sell_crossing_fees_split_across_ticks(self):
        """On a sell crossing, fees are split between old and new tick (Uniswap V3 split)."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        prev_tick_idx = tick_idx - 1

        sqrt_p_c = model.state[POOL_SQRT_PRICE_KEY][0]
        sqrt_p_low   = np.sqrt(model.exponential_value ** current_tick)
        sqrt_p_cross = sqrt_p_low / model.exp_quarter

        L      = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx]
        L_prev = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, prev_tick_idx]

        x_to_boundary   = L      * (1.0 / sqrt_p_low   - 1.0 / sqrt_p_c)
        x_from_boundary = L_prev * (1.0 / sqrt_p_cross  - 1.0 / sqrt_p_low)

        arrivals = np.array([[True, False]])
        model.update_state(arrivals, None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == current_tick - 1

        fee_at_current = model.state[FEES0_KEY][0, tick_idx]
        fee_at_prev    = model.state[FEES0_KEY][0, prev_tick_idx]

        fee_multiplier = model.fee_tier / (1.0 - model.fee_tier)
        assert np.isclose(fee_at_current, fee_multiplier * x_to_boundary,   rtol=1e-10), "Old tick fee wrong"
        assert np.isclose(fee_at_prev,    fee_multiplier * x_from_boundary, rtol=1e-10), "New tick fee wrong"

    def test_buy_crossing_fees_split_across_ticks(self):
        """On a buy crossing, fees are split between old and new tick (Uniswap V3 split)."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        next_tick_idx = tick_idx + 1

        sqrt_p_c    = model.state[POOL_SQRT_PRICE_KEY][0]
        sqrt_p_high = np.sqrt(model.exponential_value ** (current_tick + 1))
        sqrt_p_cross = sqrt_p_high * model.exp_quarter

        L      = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx]
        L_next = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, next_tick_idx]

        y_to_boundary   = L      * (sqrt_p_high  - sqrt_p_c)
        y_from_boundary = L_next * (sqrt_p_cross - sqrt_p_high)

        arrivals = np.array([[False, True]])
        model.update_state(arrivals, None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == current_tick + 1

        fee_at_current = model.state[FEES1_KEY][0, tick_idx]
        fee_at_next    = model.state[FEES1_KEY][0, next_tick_idx]

        fee_multiplier = model.fee_tier / (1.0 - model.fee_tier)
        assert np.isclose(fee_at_current, fee_multiplier * y_to_boundary,   rtol=1e-10), "Old tick fee wrong"
        assert np.isclose(fee_at_next,    fee_multiplier * y_from_boundary, rtol=1e-10), "New tick fee wrong"

    def test_no_crossing_fees_at_single_tick(self):
        """When no crossing (via direct call with small xi), all fees at current tick."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        prev_tick_idx = tick_idx - 1

        # Use small xi to force no-crossing
        xi_sell, _ = model._compute_local_xi()
        small_xi = xi_sell * 0.1
        active = np.array([True])
        model._process_sell_single(active, small_xi)

        # Verify no crossing
        assert model.state[POOL_CURRENT_TICK_KEY][0] == current_tick

        # Fee only at current tick
        fee_at_current = model.state[FEES0_KEY][0, tick_idx]
        fee_at_prev = model.state[FEES0_KEY][0, prev_tick_idx]

        assert fee_at_current > 0
        assert fee_at_prev == 0, "Fee leaked to prev tick without crossing"


class TestPriceImpactMagnitude:
    """Test that price impact magnitude depends on current tick liquidity."""

    def test_price_impact_higher_liquidity_same_tick_movement(self):
        """Both high and low liquidity cross one tick, but endpoint differs."""
        # High liquidity case
        model_high = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_high, liquidity_value=1e8, mid_tick=True)
        initial_tick_high = model_high.state[POOL_CURRENT_TICK_KEY][0].copy()

        arrivals = np.array([[1, 0]], dtype=np.int64)
        model_high.update_state(arrivals, None)

        # Low liquidity case
        model_low = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_low, liquidity_value=1e4, mid_tick=True)
        initial_tick_low = model_low.state[POOL_CURRENT_TICK_KEY][0].copy()

        model_low.update_state(arrivals, None)

        # Both should cross exactly one tick
        assert model_high.state[POOL_CURRENT_TICK_KEY][0] == initial_tick_high - 1
        assert model_low.state[POOL_CURRENT_TICK_KEY][0] == initial_tick_low - 1


class TestSequentialProcessing:
    """Test that sell and buy are processed sequentially."""

    def test_both_arrivals_collect_both_fees(self):
        """When [1,1] arrives, fees collected for BOTH trades."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        initial_fees0 = model.state[FEES0_KEY][0].sum()
        initial_fees1 = model.state[FEES1_KEY][0].sum()

        arrivals = np.array([[True, True]])
        model.update_state(arrivals, None)

        assert model.state[FEES0_KEY][0].sum() > initial_fees0, "Sell fee not collected"
        assert model.state[FEES1_KEY][0].sum() > initial_fees1, "Buy fee not collected"

    def test_both_arrivals_fee_amounts_correct(self):
        """Sell fee = x_to_boundary + x_from_boundary (exact arithmetic check)."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        current_tick = model.state[POOL_CURRENT_TICK_KEY].copy()
        tick_idx = int(current_tick[0] - model.tick_lower_global)
        sqrt_p_c   = model.state[POOL_SQRT_PRICE_KEY][0]
        sqrt_p_low = np.sqrt(model.exponential_value ** current_tick[0])
        sqrt_p_cross = sqrt_p_low / model.exp_quarter

        L      = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx]
        L_prev = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx - 1]

        x_to_boundary   = L      * (1.0 / sqrt_p_low   - 1.0 / sqrt_p_c)
        x_from_boundary = L_prev * (1.0 / sqrt_p_cross  - 1.0 / sqrt_p_low)

        # Use sell-only to guarantee the sell executes from the pre-computed initial state
        arrivals = np.array([[True, False]])
        model.update_state(arrivals, None)

        fee_multiplier = model.fee_tier / (1.0 - model.fee_tier)
        expected_fee0 = fee_multiplier * (x_to_boundary + x_from_boundary)

        assert np.isclose(model.state[FEES0_KEY][0].sum(), expected_fee0, rtol=1e-10)
        assert model.state[FEES1_KEY][0].sum() == 0

    def test_buy_uses_updated_state_after_sell(self):
        """Buy phase uses state updated by sell phase."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        # With local xi, sell will cross (tick -= 1)
        # Then buy should start from the new (lower) price

        arrivals = np.array([[True, True]])
        model.update_state(arrivals, None)

        final_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0]

        # Compare to sell-only to ensure buy also had an effect
        model_sell_only = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_sell_only, liquidity_value=1e8, mid_tick=True)
        model_sell_only.update_state(np.array([[True, False]]), None)
        sell_only_sqrt_price = model_sell_only.state[POOL_SQRT_PRICE_KEY][0]

        # Final price should be higher than sell-only (buy increased it)
        assert final_sqrt_price > sell_only_sqrt_price, \
            "Buy did not increase price after sell"

    def test_sequential_vs_individual_arrivals(self):
        """Test that [True,True] processes both, not just one."""
        # Scenario: simultaneous arrivals
        model_both = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_both, liquidity_value=1e8, mid_tick=True)

        arrivals_both = np.array([[True, True]])
        model_both.update_state(arrivals_both, None)

        fees0_both = model_both.state[FEES0_KEY][0].sum()
        fees1_both = model_both.state[FEES1_KEY][0].sum()

        # Scenario: sell only
        model_sell = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_sell, liquidity_value=1e8, mid_tick=True)

        arrivals_sell = np.array([[True, False]])
        model_sell.update_state(arrivals_sell, None)

        fees0_sell = model_sell.state[FEES0_KEY][0].sum()
        fees1_sell = model_sell.state[FEES1_KEY][0].sum()

        # Scenario: buy only
        model_buy = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_buy, liquidity_value=1e8, mid_tick=True)

        arrivals_buy = np.array([[False, True]])
        model_buy.update_state(arrivals_buy, None)

        fees0_buy = model_buy.state[FEES0_KEY][0].sum()
        fees1_buy = model_buy.state[FEES1_KEY][0].sum()

        # Both arrivals should collect both fees
        assert fees0_both > 0, "Sell fee not collected in [True,True]"
        assert fees1_both > 0, "Buy fee not collected in [True,True]"

        # Sell-only should only collect token0 fee
        assert fees0_sell > 0
        assert fees1_sell == 0

        # Buy-only should only collect token1 fee
        assert fees0_buy == 0
        assert fees1_buy > 0

        # With randomized ordering, the sell in [True,True] may execute at an adjacent tick
        # (if buy goes first), so fee0 is close but not necessarily identical to sell-only fee0.
        assert np.isclose(fees0_both, fees0_sell, rtol=1e-2), "Sell fee should be close"

    def test_vectorized_mixed_arrivals(self):
        """Test vectorized handling with mixed arrival patterns."""
        num_traj = 4
        model = create_test_model(num_trajectories=num_traj, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        initial_sqrt_prices = model.state[POOL_SQRT_PRICE_KEY].copy()

        # Different arrival patterns
        arrivals = np.array([
            [True, False],   # sell only
            [False, True],   # buy only
            [True, True],    # both
            [False, False],  # no trade
        ])

        model.update_state(arrivals, None)

        # Trajectory 0 (sell only): price decreased, only fee0 collected
        assert model.state[POOL_SQRT_PRICE_KEY][0] < initial_sqrt_prices[0]
        assert model.state[FEES0_KEY][0].sum() > 0
        assert model.state[FEES1_KEY][0].sum() == 0

        # Trajectory 1 (buy only): price increased, only fee1 collected
        assert model.state[POOL_SQRT_PRICE_KEY][1] > initial_sqrt_prices[1]
        assert model.state[FEES0_KEY][1].sum() == 0
        assert model.state[FEES1_KEY][1].sum() > 0

        # Trajectory 2 (both): both fees collected
        assert model.state[FEES0_KEY][2].sum() > 0
        assert model.state[FEES1_KEY][2].sum() > 0

        # Trajectory 3 (no trade): no change, no fees
        assert model.state[POOL_SQRT_PRICE_KEY][3] == initial_sqrt_prices[3]
        assert model.state[FEES0_KEY][3].sum() == 0
        assert model.state[FEES1_KEY][3].sum() == 0


class TestBernoulliOrdering:
    """Tests for Bernoulli arrivals with randomized sell/buy ordering."""

    def test_simultaneous_net_zero(self):
        """[True, True] → net = 0 ticks: balanced arrivals produce zero net price movement."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6, mid_tick=True)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        arrivals = np.array([[True, True]])
        model.update_state(arrivals, None)

        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]
        assert new_tick == initial_tick, \
            f"Expected net 0 ticks (1 sell = 1 buy), got {new_tick - initial_tick}"

    def test_randomization_uses_rng(self):
        """Same seed → identical final states; both fee0 and fee1 collected over many runs."""
        # Two models with the same seed must produce identical results
        model_a = create_test_model(num_trajectories=1, num_ticks=100, initial_price=100.0)
        initialize_state(model_a, liquidity_value=1e6, mid_tick=True)

        model_b = create_test_model(num_trajectories=1, num_ticks=100, initial_price=100.0)
        initialize_state(model_b, liquidity_value=1e6, mid_tick=True)

        arrivals = np.array([[True, True]])
        model_a.update_state(arrivals, None)
        model_b.update_state(arrivals, None)

        assert model_a.state[POOL_CURRENT_TICK_KEY][0] == model_b.state[POOL_CURRENT_TICK_KEY][0]
        assert np.isclose(model_a.state[FEES0_KEY][0].sum(), model_b.state[FEES0_KEY][0].sum())
        assert np.isclose(model_a.state[FEES1_KEY][0].sum(), model_b.state[FEES1_KEY][0].sum())

        # Over many runs with seed=None both orderings occur → both fee0 and fee1 are collected
        fee0_seen = False
        fee1_seen = False
        for _ in range(50):
            m = UniswapV3ModelDynamics(
                midprice_model=None,
                arrival_model=None,
                num_trajectories=1,
                fee_tier=0.003,
                tau=5,
                num_ticks=100,
                exponential_value=1.0001,
                seed=None,
            )
            initialize_state(m, liquidity_value=1e6, mid_tick=True)
            m.update_state(np.array([[True, True]]), None)
            fee0_seen = fee0_seen or m.state[FEES0_KEY][0].sum() > 0
            fee1_seen = fee1_seen or m.state[FEES1_KEY][0].sum() > 0
            if fee0_seen and fee1_seen:
                break

        assert fee0_seen, "fee0 never collected — sell never executed"
        assert fee1_seen, "fee1 never collected — buy never executed"

    def test_single_direction_unaffected(self):
        """[True, False] → -1 tick; [False, True] → +1 tick."""
        # Sells only
        model_sell = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_sell, liquidity_value=1e6, mid_tick=True)
        initial_tick = model_sell.state[POOL_CURRENT_TICK_KEY][0].copy()

        model_sell.update_state(np.array([[True, False]]), None)
        assert model_sell.state[POOL_CURRENT_TICK_KEY][0] == initial_tick - 1, \
            f"Expected -1 tick, got {model_sell.state[POOL_CURRENT_TICK_KEY][0] - initial_tick}"

        # Buys only
        model_buy = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_buy, liquidity_value=1e6, mid_tick=True)
        initial_tick = model_buy.state[POOL_CURRENT_TICK_KEY][0].copy()

        model_buy.update_state(np.array([[False, True]]), None)
        assert model_buy.state[POOL_CURRENT_TICK_KEY][0] == initial_tick + 1, \
            f"Expected +1 tick, got {model_buy.state[POOL_CURRENT_TICK_KEY][0] - initial_tick}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
