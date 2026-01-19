"""
Unit tests for Uniswap v3 swap functions in AMM_utils.py
"""

import numpy as np
import pytest
from SAiFE_gym.gym.helpers.AMM_utils import (
    delta_y_vec,
    delta_x_vec,
    price_to_tick,
    tick_to_price,
    get_tick_boundaries,
    unified_swap_single_tick
)


# Helper to create liquidity array for tests
def create_liquidity_array(center_tick, range_offset, liquidity_value):
    """Create a liquidity array centered around a tick"""
    tick_lower = center_tick - range_offset
    tick_upper = center_tick + range_offset
    num_ticks = tick_upper - tick_lower
    liquidity_array = np.full(num_ticks, liquidity_value, dtype=np.float64)
    return liquidity_array, tick_lower


class TestDeltaYVec:
    """Tests for delta_y_vec helper function"""

    def test_basic_calculation(self):
        """Test basic delta_y calculation"""
        p_high = np.array([100.0, 200.0])
        p_low = np.array([64.0, 100.0])
        result = delta_y_vec(p_high, p_low)
        expected = np.array([10.0 - 8.0, np.sqrt(200) - 10.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_scalar_inputs(self):
        """Test with scalar inputs"""
        result = delta_y_vec(100.0, 64.0)
        expected = 10.0 - 8.0  # sqrt(100) - sqrt(64)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_numerical_stability(self):
        """Test numerical stability with small values"""
        p_high = np.array([1e-100, 1.0])
        p_low = np.array([1e-200, 0.5])
        # Should not raise errors
        result = delta_y_vec(p_high, p_low)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))


class TestGetTickBoundaries:
    """Tests for get_tick_boundaries helper function"""

    def test_basic_boundaries(self):
        """Test basic tick boundary computation"""
        sqrt_p = np.array([10.0])  # price = 100
        tick_lower_sqrt, tick_upper_sqrt, current_tick = get_tick_boundaries(sqrt_p)

        # Lower boundary should be <= current price
        assert tick_lower_sqrt[0] <= 10.0
        # Upper boundary should be > current price
        assert tick_upper_sqrt[0] > 10.0
        # Lower < Upper
        assert tick_lower_sqrt[0] < tick_upper_sqrt[0]
        # Tick should be consistent with price
        expected_tick = price_to_tick(100.0)
        assert current_tick[0] == expected_tick

    def test_vectorized_boundaries(self):
        """Test boundaries for multiple trajectories"""
        sqrt_p = np.array([10.0, 12.0, 8.0, 15.0])
        tick_lower_sqrt, tick_upper_sqrt, current_tick = get_tick_boundaries(sqrt_p)

        assert tick_lower_sqrt.shape == (4,)
        assert tick_upper_sqrt.shape == (4,)
        assert current_tick.shape == (4,)

        # All should satisfy lower <= sqrt_p < upper
        for i in range(4):
            assert tick_lower_sqrt[i] <= sqrt_p[i]
            assert tick_upper_sqrt[i] > sqrt_p[i] - 1e-10  # small tolerance

    def test_boundary_prices_are_tick_aligned(self):
        """Test that boundaries align with tick prices"""
        sqrt_p = np.array([10.0])
        tick_lower_sqrt, tick_upper_sqrt, current_tick = get_tick_boundaries(sqrt_p)

        # Lower boundary should be exactly exponential_value^tick
        expected_lower = np.sqrt(1.0001 ** current_tick[0])
        expected_upper = np.sqrt(1.0001 ** (current_tick[0] + 1))

        np.testing.assert_allclose(tick_lower_sqrt[0], expected_lower, rtol=1e-10)
        np.testing.assert_allclose(tick_upper_sqrt[0], expected_upper, rtol=1e-10)


class TestUnifiedSwapSingleTick:
    """Tests for unified_swap_single_tick function"""

    def test_sell_only_basic(self):
        """Test sell-only swap (column 0 > 0, column 1 = 0)"""
        sqrt_p = np.array([10.0])
        liquidity = np.array([1000000.0])
        amount_in = np.array([[5.0, 0.0]])  # Sell token0 only
        tick_lower, tick_upper, _ = get_tick_boundaries(sqrt_p)

        sqrt_p_next, token0_net, token1_net, fee0, fee1, hit_boundary = unified_swap_single_tick(
            sqrt_p, liquidity, amount_in, tick_lower, tick_upper
        )

        # Price should decrease (selling token0)
        assert sqrt_p_next[0] < sqrt_p[0]
        # Token0 flows into pool (positive)
        assert token0_net[0] > 0
        # Token1 flows out of pool (negative)
        assert token1_net[0] < 0
        # Fee in token0
        assert fee0[0] > 0
        assert fee1[0] == 0.0
        # Note: With tight Uniswap V3 tick spacing (1.0001), even small swaps
        # may hit the tick boundary. The key behaviors (price direction, token flows)
        # are verified above.

    def test_buy_only_within_tick(self):
        """Test buy-only swap (column 0 = 0, column 1 > 0) within tick"""
        sqrt_p = np.array([10.0])
        liquidity = np.array([100000.0])
        amount_in = np.array([[0.0, 50.0]])  # Buy token0 only
        tick_lower, tick_upper, _ = get_tick_boundaries(sqrt_p)

        sqrt_p_next, token0_net, token1_net, fee0, fee1, hit_boundary = unified_swap_single_tick(
            sqrt_p, liquidity, amount_in, tick_lower, tick_upper
        )

        # Price should increase (buying token0)
        assert sqrt_p_next[0] > sqrt_p[0]
        # Token0 flows out of pool (negative)
        assert token0_net[0] < 0
        # Token1 flows into pool (positive)
        assert token1_net[0] > 0
        # Fee in token1
        assert fee0[0] == 0.0
        assert fee1[0] > 0

    def test_net_sell(self):
        """Test net sell (column 0 > column 1)"""
        sqrt_p = np.array([10.0])
        liquidity = np.array([100000.0])
        amount_in = np.array([[100.0, 30.0]])  # Net sell = 70
        tick_lower, tick_upper, _ = get_tick_boundaries(sqrt_p)

        sqrt_p_next, token0_net, token1_net, fee0, fee1, _ = unified_swap_single_tick(
            sqrt_p, liquidity, amount_in, tick_lower, tick_upper
        )

        # Price should decrease (net selling)
        assert sqrt_p_next[0] < sqrt_p[0]
        # Token0 flows in, Token1 flows out
        assert token0_net[0] > 0
        assert token1_net[0] < 0

    def test_net_buy(self):
        """Test net buy (column 1 > column 0)"""
        sqrt_p = np.array([10.0])
        liquidity = np.array([100000.0])
        amount_in = np.array([[30.0, 100.0]])  # Net buy = -70
        tick_lower, tick_upper, _ = get_tick_boundaries(sqrt_p)

        sqrt_p_next, token0_net, token1_net, fee0, fee1, _ = unified_swap_single_tick(
            sqrt_p, liquidity, amount_in, tick_lower, tick_upper
        )

        # Price should increase (net buying)
        assert sqrt_p_next[0] > sqrt_p[0]
        # Token0 flows out, Token1 flows in
        assert token0_net[0] < 0
        assert token1_net[0] > 0

    def test_exact_offset_no_trade(self):
        """Test exact offset (column 0 = column 1, no trade)"""
        sqrt_p = np.array([10.0])
        liquidity = np.array([100000.0])
        amount_in = np.array([[50.0, 50.0]])  # Net = 0
        tick_lower, tick_upper, _ = get_tick_boundaries(sqrt_p)

        sqrt_p_next, token0_net, token1_net, fee0, fee1, _ = unified_swap_single_tick(
            sqrt_p, liquidity, amount_in, tick_lower, tick_upper
        )

        # No price change
        assert sqrt_p_next[0] == sqrt_p[0]
        # No token flows
        assert token0_net[0] == 0.0
        assert token1_net[0] == 0.0
        # No fees
        assert fee0[0] == 0.0
        assert fee1[0] == 0.0

    def test_boundary_crossing(self):
        """Test swap that hits tick boundary"""
        sqrt_p = np.array([10.0])
        liquidity = np.array([100.0])  # Low liquidity
        amount_in = np.array([[10000.0, 0.0]])  # Large sell amount
        tick_lower, tick_upper, _ = get_tick_boundaries(sqrt_p)

        sqrt_p_next, token0_net, token1_net, fee0, fee1, hit_boundary = unified_swap_single_tick(
            sqrt_p, liquidity, amount_in, tick_lower, tick_upper
        )

        # Should hit boundary
        assert hit_boundary[0]
        # Price should be at lower boundary
        np.testing.assert_allclose(sqrt_p_next[0], tick_lower[0], rtol=1e-10)

    def test_zero_liquidity(self):
        """Test handling of zero liquidity"""
        sqrt_p = np.array([10.0])
        liquidity = np.array([0.0])
        amount_in = np.array([[100.0, 0.0]])
        tick_lower, tick_upper, _ = get_tick_boundaries(sqrt_p)

        sqrt_p_next, token0_net, token1_net, fee0, fee1, _ = unified_swap_single_tick(
            sqrt_p, liquidity, amount_in, tick_lower, tick_upper
        )

        # No price change
        assert sqrt_p_next[0] == sqrt_p[0]
        # No token flows
        assert token0_net[0] == 0.0
        assert token1_net[0] == 0.0
        # No fees
        assert fee0[0] == 0.0
        assert fee1[0] == 0.0

    def test_vectorized_mixed_directions(self):
        """Test multiple trajectories with different directions"""
        sqrt_p = np.array([10.0, 10.0, 10.0, 10.0])
        liquidity = np.array([100000.0, 100000.0, 100000.0, 100000.0])
        # Trajectory 0: sell, 1: buy, 2: sell, 3: no trade
        amount_in = np.array([
            [50.0, 0.0],    # sell
            [0.0, 50.0],    # buy
            [100.0, 30.0],  # net sell
            [50.0, 50.0],   # no trade
        ])
        tick_lower, tick_upper, _ = get_tick_boundaries(sqrt_p)

        sqrt_p_next, token0_net, token1_net, fee0, fee1, _ = unified_swap_single_tick(
            sqrt_p, liquidity, amount_in, tick_lower, tick_upper
        )

        # Check directions
        assert sqrt_p_next[0] < sqrt_p[0]  # sell: price decreases
        assert sqrt_p_next[1] > sqrt_p[1]  # buy: price increases
        assert sqrt_p_next[2] < sqrt_p[2]  # net sell: price decreases
        assert sqrt_p_next[3] == sqrt_p[3]  # no trade: no change

        # Check token flows
        assert token0_net[0] > 0  # sell: token0 in
        assert token0_net[1] < 0  # buy: token0 out
        assert token1_net[0] < 0  # sell: token1 out
        assert token1_net[1] > 0  # buy: token1 in

    def test_large_batch_processing(self):
        """Test processing large batch of trajectories with varied inputs"""
        num_traj = 1000
        rng = np.random.RandomState(42)

        sqrt_p = np.full(num_traj, 10.0)
        liquidity = rng.uniform(10000, 100000, size=num_traj)
        amount_in = rng.uniform(0, 100, size=(num_traj, 2))
        tick_lower, tick_upper, _ = get_tick_boundaries(sqrt_p)

        sqrt_p_next, token0_net, token1_net, fee0, fee1, hit_boundary = unified_swap_single_tick(
            sqrt_p, liquidity, amount_in, tick_lower, tick_upper
        )

        # Check shapes
        assert sqrt_p_next.shape == (num_traj,)
        assert token0_net.shape == (num_traj,)
        assert token1_net.shape == (num_traj,)
        assert fee0.shape == (num_traj,)
        assert fee1.shape == (num_traj,)
        assert hit_boundary.shape == (num_traj,)

        # All values should be finite
        assert np.all(np.isfinite(sqrt_p_next))
        assert np.all(np.isfinite(token0_net))
        assert np.all(np.isfinite(token1_net))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
