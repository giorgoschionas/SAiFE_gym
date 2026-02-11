"""
Tests for liquidity-dependent price impact in UniswapV3ModelDynamics.

Tests verify:
1. xi computation from liquidity array
2. No-crossing: small trades stay within tick
3. Crossing: large trades cross tick boundary
4. Zero liquidity: price jumps to boundary
5. Vectorization: multiple trajectories with different directions
6. Fee calculation proportional to xi
"""

import numpy as np
import pytest

from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY
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
    initial_price = model.initial_price

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
        ASSET_PRICE_KEY: np.full(num_traj, initial_price, dtype=np.float64),
        TIME_KEY: np.zeros(num_traj, dtype=np.float64),
    }

    # Mark xi as stale to force recomputation
    model._xi_stale = True


class TestXiComputation:
    """Test xi (trade size) computation."""

    def test_xi_computation_uniform_liquidity(self):
        """Test xi computation with uniform liquidity."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        model._compute_xi()

        # xi should be finite and positive
        assert np.isfinite(model.xi_sell[0])
        assert np.isfinite(model.xi_buy[0])
        assert model.xi_sell[0] > 0
        assert model.xi_buy[0] > 0

    def test_xi_computation_varies_with_liquidity(self):
        """Test that xi is smaller with lower liquidity."""
        model = create_test_model(num_trajectories=1, num_ticks=100)

        # High liquidity
        initialize_state(model, liquidity_value=1e8)
        model._compute_xi()
        xi_sell_high = model.xi_sell[0]
        xi_buy_high = model.xi_buy[0]

        # Low liquidity
        initialize_state(model, liquidity_value=1e4)
        model._compute_xi()
        xi_sell_low = model.xi_sell[0]
        xi_buy_low = model.xi_buy[0]

        # Lower liquidity should give smaller xi
        assert xi_sell_low < xi_sell_high
        assert xi_buy_low < xi_buy_high

    def test_xi_minimum_across_ticks(self):
        """Test that xi takes minimum across all ticks."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        # Set one tick to very low liquidity
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, 50] = 100.0  # Much lower
        model._xi_stale = True
        model._compute_xi()

        xi_sell_with_low = model.xi_sell[0]

        # Reset to uniform high liquidity
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, 50] = 1e6
        model._xi_stale = True
        model._compute_xi()

        xi_sell_uniform = model.xi_sell[0]

        # xi with one low-liquidity tick should be smaller
        assert xi_sell_with_low < xi_sell_uniform

    def test_xi_zero_liquidity_gives_inf(self):
        """Test that all-zero liquidity gives infinite xi (no trades possible)."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=0.0)  # All zero

        model._compute_xi()

        assert np.isinf(model.xi_sell[0])
        assert np.isinf(model.xi_buy[0])


class TestNoCrossing:
    """Test price updates when trade stays within tick.

    Note: With uniform liquidity, xi (global minimum) is ~2x the mid-tick boundary capacity,
    so crossing always occurs. To test no-crossing, we need non-uniform liquidity where
    the xi-determining tick has low liquidity (small xi) while current tick has high
    liquidity (large boundary capacity).
    """

    def test_sell_no_crossing_non_uniform_liquidity(self):
        """Test sell that doesn't cross with non-uniform liquidity."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8)  # High base liquidity

        # Set the highest-price tick (which determines xi_sell) to low liquidity
        # This makes xi_sell small while keeping current tick's capacity high
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, 99] = 1e4  # Low at tick 99
        model._xi_stale = True

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Sell arrival
        arrivals = np.array([[1, 0]], dtype=np.int64)
        model.update_state(arrivals, None)

        new_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0]
        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Price should decrease
        assert new_sqrt_price < initial_sqrt_price

        # Tick should NOT change (xi is small due to low liquidity at tick 99)
        assert new_tick == initial_tick

    def test_buy_no_crossing_non_uniform_liquidity(self):
        """Test buy that doesn't cross with non-uniform liquidity."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8)  # High base liquidity

        # Set the lowest-price tick (which determines xi_buy) to low liquidity
        # This makes xi_buy small while keeping current tick's capacity high
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, 0] = 1e4  # Low at tick 0
        model._xi_stale = True

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Buy arrival
        arrivals = np.array([[0, 1]], dtype=np.int64)
        model.update_state(arrivals, None)

        new_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0]
        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Price should increase
        assert new_sqrt_price > initial_sqrt_price

        # Tick should NOT change (xi is small due to low liquidity at tick 0)
        assert new_tick == initial_tick

    def test_uniform_liquidity_always_crosses(self):
        """Test that uniform liquidity causes crossing (by design)."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Sell arrival
        arrivals = np.array([[1, 0]], dtype=np.int64)
        model.update_state(arrivals, None)

        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # With uniform liquidity, xi/capacity ratio ~2, so always crosses
        assert new_tick == initial_tick - 1


class TestCrossing:
    """Test price updates when trade crosses tick boundary."""

    def test_sell_crossing(self):
        """Test sell that crosses tick boundary due to low liquidity."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=100.0)  # Very low liquidity

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Sell arrival
        arrivals = np.array([[1, 0]], dtype=np.int64)
        model.update_state(arrivals, None)

        new_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0]
        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Price should decrease
        assert new_sqrt_price < initial_sqrt_price

        # Tick should decrease by 1 (crossing)
        assert new_tick == initial_tick - 1

    def test_buy_crossing(self):
        """Test buy that crosses tick boundary due to low liquidity."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=100.0)  # Very low liquidity

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # Buy arrival
        arrivals = np.array([[0, 1]], dtype=np.int64)
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
        arrivals = np.array([[1, 0]], dtype=np.int64)
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
        arrivals = np.array([[0, 1]], dtype=np.int64)
        model.update_state(arrivals, None)

        new_tick = model.state[POOL_CURRENT_TICK_KEY][0]

        # Should cross tick (zero capacity at current tick)
        assert new_tick == initial_tick + 1


class TestVectorization:
    """Test that operations are properly vectorized across trajectories."""

    def test_multiple_trajectories_different_directions(self):
        """Test multiple trajectories with different arrival directions."""
        num_traj = 4
        model = create_test_model(num_trajectories=num_traj, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        initial_sqrt_prices = model.state[POOL_SQRT_PRICE_KEY].copy()

        # Different arrivals for each trajectory:
        # [sell, no trade, buy, both (net zero)]
        arrivals = np.array([
            [1, 0],  # sell
            [0, 0],  # no trade
            [0, 1],  # buy
            [1, 1],  # both (should prioritize sell in current impl)
        ], dtype=np.int64)

        model.update_state(arrivals, None)

        new_sqrt_prices = model.state[POOL_SQRT_PRICE_KEY]

        # Trajectory 0 (sell): price decreased
        assert new_sqrt_prices[0] < initial_sqrt_prices[0]

        # Trajectory 1 (no trade): price unchanged
        assert new_sqrt_prices[1] == initial_sqrt_prices[1]

        # Trajectory 2 (buy): price increased
        assert new_sqrt_prices[2] > initial_sqrt_prices[2]

    def test_mixed_crossing_no_crossing_per_trajectory(self):
        """Test trajectories with different xi (per-trajectory) causing mixed behavior.

        Note: xi is computed per-trajectory based on each trajectory's liquidity array.
        Trajectory 0 has non-uniform liquidity (low at tick 99) → small xi → no crossing
        Trajectory 1 has uniform liquidity → xi ~2x boundary capacity → crosses
        """
        num_traj = 2
        model = create_test_model(num_trajectories=num_traj, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        # Trajectory 0: low liquidity at tick 99 (determines xi_sell) → small xi
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, 99] = 1e3
        # Trajectory 1: uniform high liquidity → xi ~2x boundary, will cross
        # (already set to 1e8)

        model._xi_stale = True

        initial_ticks = model.state[POOL_CURRENT_TICK_KEY].copy()

        # Both sell
        arrivals = np.array([
            [1, 0],
            [1, 0],
        ], dtype=np.int64)

        model.update_state(arrivals, None)

        new_ticks = model.state[POOL_CURRENT_TICK_KEY]

        # Trajectory 0 (small xi due to low liq at tick 99): should NOT cross
        assert new_ticks[0] == initial_ticks[0]

        # Trajectory 1 (uniform liquidity, large xi): should cross
        assert new_ticks[1] == initial_ticks[1] - 1


class TestFeeCalculation:
    """Test that fees are calculated proportionally to xi."""

    def test_sell_fees(self):
        """Test fee collection for sell trades."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        initial_fees0 = model.state[FEES0_KEY][0].sum()

        # Sell arrival
        arrivals = np.array([[1, 0]], dtype=np.int64)
        model.update_state(arrivals, None)

        new_fees0 = model.state[FEES0_KEY][0].sum()

        # Fees should increase
        assert new_fees0 > initial_fees0

        # Fee should be proportional to xi_sell
        expected_fee = model.fee_tier / (1 - model.fee_tier) * model.xi_sell[0]
        assert np.isclose(new_fees0 - initial_fees0, expected_fee, rtol=1e-10)

    def test_buy_fees(self):
        """Test fee collection for buy trades."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        initial_fees1 = model.state[FEES1_KEY][0].sum()

        # Buy arrival
        arrivals = np.array([[0, 1]], dtype=np.int64)
        model.update_state(arrivals, None)

        new_fees1 = model.state[FEES1_KEY][0].sum()

        # Fees should increase
        assert new_fees1 > initial_fees1

        # Fee should be proportional to xi_buy
        expected_fee = model.fee_tier / (1 - model.fee_tier) * model.xi_buy[0]
        assert np.isclose(new_fees1 - initial_fees1, expected_fee, rtol=1e-10)

    def test_fees_scale_with_liquidity(self):
        """Test that fees vary with liquidity (via xi)."""
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

        # Lower liquidity → smaller xi → smaller fees
        assert fees_low < fees_high


class TestMarkXiStale:
    """Test the xi staleness mechanism."""

    def test_mark_xi_stale(self):
        """Test that mark_xi_stale triggers recomputation."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        # Force initial computation
        model._compute_xi()
        xi_sell_initial = model.xi_sell[0]

        # Change liquidity
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, :] = 1e4  # Much lower

        # xi should still be the old value (stale)
        assert model.xi_sell[0] == xi_sell_initial

        # Mark stale and update state (which will recompute)
        model.mark_xi_stale()
        arrivals = np.array([[1, 0]], dtype=np.int64)
        model.update_state(arrivals, None)

        # Now xi should be different (lower due to lower liquidity)
        assert model.xi_sell[0] < xi_sell_initial


class TestPriceImpactMagnitude:
    """Test that price impact magnitude depends on liquidity distribution."""

    def test_price_impact_with_crossing_vs_no_crossing(self):
        """Test price impact differs between crossing and no-crossing cases.

        With non-uniform liquidity we can control whether crossing occurs:
        - No crossing (small xi): smaller price change, stays within tick
        - Crossing (large xi): larger price change, moves to next tick
        """
        # No-crossing case: low liquidity at tick 99 makes xi_sell small
        model_no_cross = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_no_cross, liquidity_value=1e8, mid_tick=True)
        model_no_cross.state[POOL_LIQUIDITY_ARRAY_KEY][0, 99] = 1e3  # Small xi
        model_no_cross._xi_stale = True

        initial_price_no_cross = model_no_cross.state[POOL_SQRT_PRICE_KEY][0] ** 2
        initial_tick_no_cross = model_no_cross.state[POOL_CURRENT_TICK_KEY][0]

        arrivals = np.array([[1, 0]], dtype=np.int64)
        model_no_cross.update_state(arrivals, None)

        final_price_no_cross = model_no_cross.state[POOL_SQRT_PRICE_KEY][0] ** 2
        final_tick_no_cross = model_no_cross.state[POOL_CURRENT_TICK_KEY][0]
        impact_no_cross = abs(final_price_no_cross - initial_price_no_cross) / initial_price_no_cross

        # Crossing case: uniform high liquidity makes xi large (crosses)
        model_cross = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_cross, liquidity_value=1e8, mid_tick=True)

        initial_price_cross = model_cross.state[POOL_SQRT_PRICE_KEY][0] ** 2
        initial_tick_cross = model_cross.state[POOL_CURRENT_TICK_KEY][0]

        model_cross.update_state(arrivals, None)

        final_price_cross = model_cross.state[POOL_SQRT_PRICE_KEY][0] ** 2
        final_tick_cross = model_cross.state[POOL_CURRENT_TICK_KEY][0]
        impact_cross = abs(final_price_cross - initial_price_cross) / initial_price_cross

        # Verify crossing behavior
        assert final_tick_no_cross == initial_tick_no_cross, "No-cross should stay in tick"
        assert final_tick_cross == initial_tick_cross - 1, "Cross should move down one tick"

        # Crossing case should have larger price impact (moved further)
        assert impact_cross > impact_no_cross

    def test_price_impact_scales_with_xi(self):
        """Test that price impact is proportional to xi (trade size)."""
        # Create two scenarios with different xi
        # Scenario 1: small xi (low liquidity at determining tick)
        model_small_xi = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_small_xi, liquidity_value=1e8, mid_tick=True)
        model_small_xi.state[POOL_LIQUIDITY_ARRAY_KEY][0, 99] = 1e2  # Very small xi
        model_small_xi._xi_stale = True

        initial_price_small = model_small_xi.state[POOL_SQRT_PRICE_KEY][0] ** 2

        arrivals = np.array([[1, 0]], dtype=np.int64)
        model_small_xi.update_state(arrivals, None)
        xi_small = model_small_xi.xi_sell[0]

        final_price_small = model_small_xi.state[POOL_SQRT_PRICE_KEY][0] ** 2
        impact_small = abs(final_price_small - initial_price_small)

        # Scenario 2: larger xi (higher liquidity at determining tick)
        model_large_xi = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_large_xi, liquidity_value=1e8, mid_tick=True)
        model_large_xi.state[POOL_LIQUIDITY_ARRAY_KEY][0, 99] = 1e5  # Larger xi
        model_large_xi._xi_stale = True

        initial_price_large = model_large_xi.state[POOL_SQRT_PRICE_KEY][0] ** 2

        model_large_xi.update_state(arrivals, None)
        xi_large = model_large_xi.xi_sell[0]

        final_price_large = model_large_xi.state[POOL_SQRT_PRICE_KEY][0] ** 2
        impact_large = abs(final_price_large - initial_price_large)

        # Verify xi relationship
        assert xi_large > xi_small, "Larger liquidity should give larger xi"

        # Larger xi should cause larger price impact
        assert impact_large > impact_small


class TestSequentialProcessing:
    """Test that sell and buy are processed sequentially."""

    def test_both_arrivals_collect_both_fees(self):
        """When [1,1] arrives, fees collected for BOTH trades."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        initial_fees0 = model.state[FEES0_KEY][0].sum()
        initial_fees1 = model.state[FEES1_KEY][0].sum()

        arrivals = np.array([[1, 1]], dtype=np.int64)
        model.update_state(arrivals, None)

        assert model.state[FEES0_KEY][0].sum() > initial_fees0, "Sell fee not collected"
        assert model.state[FEES1_KEY][0].sum() > initial_fees1, "Buy fee not collected"

    def test_both_arrivals_fee_amounts_correct(self):
        """When [1,1] arrives, fee amounts match expected values."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        # Force xi computation
        model._compute_xi()
        xi_sell = model.xi_sell[0]
        xi_buy = model.xi_buy[0]

        arrivals = np.array([[1, 1]], dtype=np.int64)
        model.update_state(arrivals, None)

        fee_multiplier = model.fee_tier / (1.0 - model.fee_tier)
        expected_fee0 = fee_multiplier * xi_sell
        expected_fee1 = fee_multiplier * xi_buy

        assert np.isclose(model.state[FEES0_KEY][0].sum(), expected_fee0, rtol=1e-10)
        assert np.isclose(model.state[FEES1_KEY][0].sum(), expected_fee1, rtol=1e-10)

    def test_buy_uses_updated_state_after_sell(self):
        """Buy phase uses state updated by sell phase."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        # With uniform liquidity and mid_tick, sell will cross (tick -= 1)
        # Then buy should start from the new (lower) price

        arrivals = np.array([[1, 1]], dtype=np.int64)
        model.update_state(arrivals, None)

        final_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0]

        # Sequential processing: sell decreases price, then buy increases from new state
        # The result should be different from just applying sell OR buy individually

        # Compare to sell-only to ensure buy also had an effect
        model_sell_only = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_sell_only, liquidity_value=1e8, mid_tick=True)
        model_sell_only.update_state(np.array([[1, 0]], dtype=np.int64), None)
        sell_only_sqrt_price = model_sell_only.state[POOL_SQRT_PRICE_KEY][0]

        # Final price should be higher than sell-only (buy increased it)
        assert final_sqrt_price > sell_only_sqrt_price, \
            "Buy did not increase price after sell"

    def test_sequential_vs_individual_arrivals(self):
        """Test that [1,1] processes both, not just one."""
        # Scenario: simultaneous arrivals
        model_both = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_both, liquidity_value=1e8, mid_tick=True)

        arrivals_both = np.array([[1, 1]], dtype=np.int64)
        model_both.update_state(arrivals_both, None)

        fees0_both = model_both.state[FEES0_KEY][0].sum()
        fees1_both = model_both.state[FEES1_KEY][0].sum()

        # Scenario: sell only
        model_sell = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_sell, liquidity_value=1e8, mid_tick=True)

        arrivals_sell = np.array([[1, 0]], dtype=np.int64)
        model_sell.update_state(arrivals_sell, None)

        fees0_sell = model_sell.state[FEES0_KEY][0].sum()
        fees1_sell = model_sell.state[FEES1_KEY][0].sum()

        # Scenario: buy only
        model_buy = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_buy, liquidity_value=1e8, mid_tick=True)

        arrivals_buy = np.array([[0, 1]], dtype=np.int64)
        model_buy.update_state(arrivals_buy, None)

        fees0_buy = model_buy.state[FEES0_KEY][0].sum()
        fees1_buy = model_buy.state[FEES1_KEY][0].sum()

        # Both arrivals should collect both fees
        assert fees0_both > 0, "Sell fee not collected in [1,1]"
        assert fees1_both > 0, "Buy fee not collected in [1,1]"

        # Sell-only should only collect token0 fee
        assert fees0_sell > 0
        assert fees1_sell == 0

        # Buy-only should only collect token1 fee
        assert fees0_buy == 0
        assert fees1_buy > 0

        # Both should collect approximately the sum (not exactly due to sequential state changes)
        assert fees0_both == fees0_sell, "Sell fee should match"
        # Note: fees1_both may differ from fees1_buy because buy starts from post-sell state

    def test_vectorized_mixed_arrivals(self):
        """Test vectorized handling with mixed arrival patterns."""
        num_traj = 4
        model = create_test_model(num_trajectories=num_traj, num_ticks=100)
        initialize_state(model, liquidity_value=1e8, mid_tick=True)

        initial_sqrt_prices = model.state[POOL_SQRT_PRICE_KEY].copy()

        # Different arrival patterns
        arrivals = np.array([
            [1, 0],  # sell only
            [0, 1],  # buy only
            [1, 1],  # both
            [0, 0],  # no trade
        ], dtype=np.int64)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
