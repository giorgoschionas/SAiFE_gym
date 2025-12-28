"""
Unit tests for Uniswap v3 swap functions in AMM_utils.py
"""

import numpy as np
import pytest
from SAiFE_gym.gym.helpers.AMM_utils import (
    delta_y_vec,
    swap_step_within_tick_vec,
    execute_swap_vec_array,
    delta_x_vec,
    price_to_tick,
    tick_to_price
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


class TestSwapStepWithinTick:
    """Tests for swap_step_within_tick_vec function"""

    def test_zero_for_one_within_tick(self):
        """Test selling Token 0, staying within tick"""
        sqrt_p_current = 10.0  # price = 100
        sqrt_p_target = 9.5  # price ~ 90.25
        liquidity = 100000.0
        amount_remaining = 50.0
        fee_rate = 0.003

        sqrt_p_next, amt_in, amt_out, fee = swap_step_within_tick_vec(
            sqrt_p_current, sqrt_p_target, liquidity, amount_remaining, fee_rate, zero_for_one=True
        )

        # Price should decrease but not reach target
        assert sqrt_p_next < sqrt_p_current
        assert sqrt_p_next > sqrt_p_target

        # Should consume all remaining amount
        assert abs(amt_in - amount_remaining) < 1e-10

        # Output should be positive
        assert amt_out > 0

        # Fee should be positive
        assert fee > 0

    def test_zero_for_one_hit_boundary(self):
        """Test selling Token 0, hitting tick boundary"""
        sqrt_p_current = 10.0
        sqrt_p_target = 9.999  # Very close boundary
        liquidity = 100.0
        amount_remaining = 1000.0  # Large amount
        fee_rate = 0.003

        sqrt_p_next, amt_in, amt_out, fee = swap_step_within_tick_vec(
            sqrt_p_current, sqrt_p_target, liquidity, amount_remaining, fee_rate, zero_for_one=True
        )

        # Should hit target exactly
        assert abs(sqrt_p_next - sqrt_p_target) < 1e-10

        # Should consume less than remaining
        assert amt_in < amount_remaining

    def test_not_zero_for_one_within_tick(self):
        """Test buying Token 0, staying within tick"""
        sqrt_p_current = 10.0
        sqrt_p_target = 10.5
        liquidity = 100000.0
        amount_remaining = 50.0
        fee_rate = 0.003

        sqrt_p_next, amt_in, amt_out, fee = swap_step_within_tick_vec(
            sqrt_p_current, sqrt_p_target, liquidity, amount_remaining, fee_rate, zero_for_one=False
        )

        # Price should increase but not reach target
        assert sqrt_p_next > sqrt_p_current
        assert sqrt_p_next < sqrt_p_target

        # Should consume all remaining amount
        assert abs(amt_in - amount_remaining) < 1e-10

    def test_zero_liquidity(self):
        """Test handling of zero liquidity"""
        sqrt_p_current = 10.0
        sqrt_p_target = 9.5
        liquidity = 0.0
        amount_remaining = 100.0
        fee_rate = 0.003

        sqrt_p_next, amt_in, amt_out, fee = swap_step_within_tick_vec(
            sqrt_p_current, sqrt_p_target, liquidity, amount_remaining, fee_rate, zero_for_one=True
        )

        # Should return zeros
        assert sqrt_p_next == sqrt_p_current
        assert amt_in == 0.0
        assert amt_out == 0.0
        assert fee == 0.0


class TestArrayBasedSwap:
    """Tests specific to the array-based vectorized implementation"""

    def test_sparse_liquidity_handling(self):
        """Test that zero liquidity ticks are handled gracefully"""
        sqrt_p = np.array([10.0] * 10)
        current_tick = price_to_tick(100.0)

        # Create liquidity with gaps (every 5th tick is zero)
        liquidity_array, tick_lower = create_liquidity_array(current_tick, 100, 10000.0)
        # Set every 5th tick to zero
        for i in range(0, len(liquidity_array), 5):
            liquidity_array[i] = 0.0

        amount_in = np.full(10, 20.0)

        result = execute_swap_vec_array(
            sqrt_p, liquidity_array, tick_lower, amount_in, True
        )

        # Should skip zero-liquidity ticks gracefully
        assert result[0].shape == (10,)
        assert np.all(result[0] < 10.0)  # Prices should decrease
        assert np.all(result[1] > 0)  # Should still produce output

    def test_variable_completion_times(self):
        """Test when trajectories complete at very different times"""
        # Small amounts complete quickly, large amounts take many ticks
        amount_in = np.array([1.0, 10.0, 50.0, 100.0])
        sqrt_p = np.full(4, 10.0)

        current_tick = price_to_tick(100.0)
        liquidity_array, tick_lower = create_liquidity_array(current_tick, 100, 50000.0)

        result = execute_swap_vec_array(
            sqrt_p, liquidity_array, tick_lower, amount_in, True
        )

        # All should complete successfully
        assert result[0].shape == (4,)
        # Larger amounts should have greater or equal price impact (decreasing sqrt prices)
        assert result[0][0] >= result[0][1] >= result[0][2] >= result[0][3]
        # Prices should decrease from initial
        assert np.all(result[0] <= sqrt_p)
        # Larger amounts should cross more or equal ticks
        assert result[3][0] <= result[3][1] <= result[3][2] <= result[3][3]

    def test_large_batch_processing(self):
        """Test processing a large batch of trajectories"""
        num_traj = 100
        sqrt_p = np.full(num_traj, 10.0)
        current_tick = price_to_tick(100.0)
        liquidity_array, tick_lower = create_liquidity_array(current_tick, 100, 100000.0)
        amount_in = np.random.RandomState(42).uniform(5, 50, size=num_traj)

        result = execute_swap_vec_array(
            sqrt_p, liquidity_array, tick_lower, amount_in, True
        )

        # Check output shapes
        assert result[0].shape == (num_traj,)
        assert result[1].shape == (num_traj,)
        assert result[2].shape == (num_traj,)
        assert result[3].shape == (num_traj,)

        # All prices should decrease
        assert np.all(result[0] < sqrt_p)
        # All outputs should be positive
        assert np.all(result[1] > 0)

    def test_out_of_bounds_handling(self):
        """Test behavior when swap would go out of liquidity array bounds"""
        sqrt_p = np.array([10.0])
        # Very limited liquidity range
        liquidity_array = np.full(10, 10000.0)
        tick_lower = price_to_tick(100.0) - 5
        # Large swap amount that would cross many ticks
        amount_in = np.array([1000.0])

        result = execute_swap_vec_array(
            sqrt_p, liquidity_array, tick_lower, amount_in, True
        )

        # Should handle out-of-bounds gracefully
        assert result[0].shape == (1,)
        # May not consume all input if runs out of liquidity
        assert result[1] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
