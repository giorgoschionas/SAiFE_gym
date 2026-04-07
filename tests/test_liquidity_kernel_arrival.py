"""
Test script for LiquidityKernelArrivalModel.

Tests:
1. Internal state management
2. Directional liquidity asymmetry
3. Exponential decay (closer ticks weighted more)
4. Boundary safety (current tick near array edges)
5. Floor enforcement
6. Analytical formula verification
7. Multi-trajectory independence
8. Mispricing composability
"""
import numpy as np
import sys
import pytest

sys.path.insert(0, '/home/gchionas/Programming/Blockchain/Ethereum/defi-trading/SAiFE_gym/.trees/new-arrivals')

from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(liquidity_array, current_tick, tick_lower_global,
                amm_price=100.0, midprice=100.0):
    """Build a context dict for update()."""
    N = liquidity_array.shape[0]
    return {
        'liquidity_array': liquidity_array,
        'current_tick': np.atleast_1d(current_tick).astype(float),
        'tick_lower_global': tick_lower_global,
        'amm_price': np.full(N, amm_price),
        'midprice': np.full(N, midprice),
    }


class TestLiquidityKernelArrivalModelStateful:

    def test_internal_state_updates(self):
        """update() should modify current_state with non-trivial liquidity."""
        N = 5
        model = LiquidityKernelArrivalModel(num_trajectories=N, K=5, seed=42)
        initial = model.current_state.copy()

        num_ticks = 100
        liq = np.ones((N, num_ticks)) * 1e6
        state = _make_state(liq, current_tick=np.full(N, 50), tick_lower_global=0)
        model.update(None, None, None, state)

        assert not np.allclose(model.current_state, initial)

    def test_get_arrivals_shape_and_dtype(self):
        """get_arrivals() returns (N, 2) boolean array."""
        N = 10
        model = LiquidityKernelArrivalModel(num_trajectories=N, seed=42)
        arrivals = model.get_arrivals()
        assert arrivals.shape == (N, 2)
        assert arrivals.dtype == bool

    def test_reset_restores_baseline(self):
        """reset() should restore current_state to alpha[1]."""
        N = 5
        model = LiquidityKernelArrivalModel(num_trajectories=N, K=3, seed=42)

        # Perturb state
        liq = np.ones((N, 50)) * 2e6
        state = _make_state(liq, current_tick=np.full(N, 25), tick_lower_global=0)
        model.update(None, None, None, state)
        assert not np.allclose(model.current_state, model.alpha[1])

        model.reset()
        expected = np.ones((N, 2)) * model.alpha[1]
        np.testing.assert_array_equal(model.current_state, expected)

    def test_initial_state_is_baseline(self):
        """Fresh model should have current_state == alpha[1]."""
        N = 3
        model = LiquidityKernelArrivalModel(num_trajectories=N, seed=42)
        expected = np.ones((N, 2)) * model.alpha[1]
        np.testing.assert_array_equal(model.current_state, expected)

    def test_update_with_none_preserves_state(self):
        """update(state=None) should not change current_state."""
        N = 3
        model = LiquidityKernelArrivalModel(num_trajectories=N, K=3, seed=42)

        liq = np.ones((N, 50)) * 1e6
        state = _make_state(liq, current_tick=np.full(N, 25), tick_lower_global=0)
        model.update(None, None, None, state)
        after = model.current_state.copy()

        model.update(None, None, None, None)
        np.testing.assert_array_equal(model.current_state, after)

    def test_kernel_weights_precomputed(self):
        """kernel_weights should be exp(-beta * [1..K])."""
        beta, K = 0.3, 7
        model = LiquidityKernelArrivalModel(beta=beta, K=K, seed=42)
        expected = np.exp(-beta * np.arange(1, K + 1))
        np.testing.assert_allclose(model.kernel_weights, expected)


class TestParameterValidation:

    def test_alpha_shape(self):
        with pytest.raises(AssertionError, match="alpha must have shape"):
            LiquidityKernelArrivalModel(alpha=np.ones((3, 2)))

    def test_floor_nonnegative(self):
        alpha = np.array([[-1.0, -1.0], [100, 100], [50, 50], [5, 5]])
        with pytest.raises(AssertionError, match="non-negative"):
            LiquidityKernelArrivalModel(alpha=alpha)

    def test_beta_positive(self):
        with pytest.raises(AssertionError, match="beta must be positive"):
            LiquidityKernelArrivalModel(beta=0.0)

    def test_K_at_least_one(self):
        with pytest.raises(AssertionError, match="K must be >= 1"):
            LiquidityKernelArrivalModel(K=0)


class TestDirectionalLiquidityEffect:

    def test_more_right_liquidity_increases_buy(self):
        """Heavy liquidity on RIGHT only -> buy intensity > sell intensity."""
        N = 1
        alpha = np.array([
            [10.0, 10.0],
            [100.0, 100.0],
            [50.0, 50.0],
            [0.0, 0.0],      # no mispricing
        ])
        model = LiquidityKernelArrivalModel(
            alpha=alpha, K=5, beta=0.5, liquidity_scale=1e6,
            num_trajectories=N, seed=42,
        )

        num_ticks = 100
        liq = np.zeros((N, num_ticks))
        current_idx = 50
        # Place liquidity only to the right
        liq[0, current_idx + 1: current_idx + 6] = 2e6

        state = _make_state(liq, current_tick=np.array([current_idx]), tick_lower_global=0)
        model.update(None, None, None, state)

        assert model.current_state[0, 1] > model.current_state[0, 0], \
            f"Buy ({model.current_state[0,1]:.2f}) should exceed sell ({model.current_state[0,0]:.2f})"

    def test_more_left_liquidity_increases_sell(self):
        """Heavy liquidity on LEFT only -> sell intensity > buy intensity."""
        N = 1
        alpha = np.array([
            [10.0, 10.0],
            [100.0, 100.0],
            [50.0, 50.0],
            [0.0, 0.0],
        ])
        model = LiquidityKernelArrivalModel(
            alpha=alpha, K=5, beta=0.5, liquidity_scale=1e6,
            num_trajectories=N, seed=42,
        )

        num_ticks = 100
        liq = np.zeros((N, num_ticks))
        current_idx = 50
        # Place liquidity only to the left
        liq[0, current_idx - 5: current_idx] = 2e6

        state = _make_state(liq, current_tick=np.array([current_idx]), tick_lower_global=0)
        model.update(None, None, None, state)

        assert model.current_state[0, 0] > model.current_state[0, 1], \
            f"Sell ({model.current_state[0,0]:.2f}) should exceed buy ({model.current_state[0,1]:.2f})"

    def test_symmetric_liquidity_gives_symmetric_intensity(self):
        """Uniform liquidity + no mispricing -> equal sell/buy intensities."""
        N = 1
        alpha = np.array([
            [10.0, 10.0],
            [100.0, 100.0],
            [50.0, 50.0],
            [0.0, 0.0],
        ])
        model = LiquidityKernelArrivalModel(
            alpha=alpha, K=5, beta=0.5, liquidity_scale=1e6,
            num_trajectories=N, seed=42,
        )

        num_ticks = 100
        liq = np.ones((N, num_ticks)) * 1e6

        state = _make_state(liq, current_tick=np.array([50]), tick_lower_global=0)
        model.update(None, None, None, state)

        np.testing.assert_allclose(
            model.current_state[0, 0], model.current_state[0, 1],
            rtol=1e-10,
            err_msg="Symmetric liquidity should give equal sell/buy intensity",
        )

    def test_exponential_decay_closer_ticks_matter_more(self):
        """Liquidity at d=1 should produce higher intensity than same amount at d=K."""
        N = 1
        K = 10
        alpha = np.array([
            [0.0, 0.0],
            [0.0, 0.0],       # zero baseline to isolate kernel effect
            [100.0, 100.0],
            [0.0, 0.0],
        ])
        model_near = LiquidityKernelArrivalModel(
            alpha=alpha, K=K, beta=0.5, liquidity_scale=1e6,
            num_trajectories=N, seed=42,
        )
        model_far = LiquidityKernelArrivalModel(
            alpha=alpha, K=K, beta=0.5, liquidity_scale=1e6,
            num_trajectories=N, seed=42,
        )

        num_ticks = 100
        current_idx = 50

        # Near: liquidity at d=1 only (right side for buy)
        liq_near = np.zeros((N, num_ticks))
        liq_near[0, current_idx + 1] = 1e6

        # Far: same amount at d=K only
        liq_far = np.zeros((N, num_ticks))
        liq_far[0, current_idx + K] = 1e6

        state_near = _make_state(liq_near, np.array([current_idx]), 0)
        state_far = _make_state(liq_far, np.array([current_idx]), 0)

        model_near.update(None, None, None, state_near)
        model_far.update(None, None, None, state_far)

        # Buy intensity (column 1) should be higher for near case
        assert model_near.current_state[0, 1] > model_far.current_state[0, 1], \
            f"Near ({model_near.current_state[0,1]:.4f}) should exceed far ({model_far.current_state[0,1]:.4f})"


class TestEdgeCases:

    def test_current_tick_near_left_boundary(self):
        """Current tick at array index 2 with K=10 should not crash."""
        N = 1
        K = 10
        model = LiquidityKernelArrivalModel(K=K, num_trajectories=N, seed=42)

        num_ticks = 50
        liq = np.ones((N, num_ticks)) * 1e6
        # Current tick at index 2: only 2 sell-side neighbors in bounds
        state = _make_state(liq, current_tick=np.array([2]), tick_lower_global=0)
        model.update(None, None, None, state)

        # Should not crash, intensities should be valid
        assert np.all(np.isfinite(model.current_state))
        # Buy side has 10 valid neighbors, sell side has only 2
        # -> buy intensity should be higher (with no mispricing, alpha_3=0 by default... actually default alpha_3=5)
        # Let's just check it didn't crash and values are reasonable
        assert np.all(model.current_state >= model.alpha[0])

    def test_current_tick_near_right_boundary(self):
        """Current tick near right edge should not crash."""
        N = 1
        K = 10
        model = LiquidityKernelArrivalModel(K=K, num_trajectories=N, seed=42)

        num_ticks = 50
        liq = np.ones((N, num_ticks)) * 1e6
        # Current tick at index num_ticks-3: only 2 buy-side neighbors
        state = _make_state(liq, current_tick=np.array([num_ticks - 3]),
                            tick_lower_global=0)
        model.update(None, None, None, state)

        assert np.all(np.isfinite(model.current_state))
        assert np.all(model.current_state >= model.alpha[0])

    def test_zero_liquidity_everywhere(self):
        """With zero liquidity, intensity = max(alpha_0, alpha_1)."""
        N = 1
        alpha = np.array([
            [10.0, 10.0],
            [100.0, 100.0],
            [50.0, 50.0],
            [0.0, 0.0],       # no mispricing
        ])
        model = LiquidityKernelArrivalModel(
            alpha=alpha, K=5, num_trajectories=N, seed=42,
        )

        num_ticks = 50
        liq = np.zeros((N, num_ticks))
        state = _make_state(liq, current_tick=np.array([25]), tick_lower_global=0)
        model.update(None, None, None, state)

        # Weighted liq is 0, so intensity = max(10, 100 + 0 + 0) = 100
        np.testing.assert_allclose(model.current_state, 100.0)

    def test_intensity_floor_prevents_negative(self):
        """Large negative linear part should be clamped to alpha_0."""
        N = 1
        alpha = np.array([
            [50.0, 50.0],     # high floor
            [10.0, 10.0],     # low baseline
            [0.0, 0.0],       # no liquidity effect
            [100.0, 100.0],   # strong mispricing to drive sell negative
        ])
        model = LiquidityKernelArrivalModel(
            alpha=alpha, K=3, num_trajectories=N, seed=42,
        )

        num_ticks = 50
        liq = np.zeros((N, num_ticks))
        # S > Z: mispricing = +10 -> sell linear = 10 + 0 - 100*10 = -990
        state = _make_state(liq, current_tick=np.array([25]),
                            tick_lower_global=0, amm_price=90.0, midprice=100.0)
        model.update(None, None, None, state)

        # Sell should be floored at 50
        assert model.current_state[0, 0] == 50.0, \
            f"Sell intensity should be floored at 50, got {model.current_state[0,0]}"


class TestFormulaVerification:

    def test_intensity_values_match_analytical(self):
        """Hand-compute expected intensity and verify."""
        N = 1
        K = 3
        beta = 0.5
        liq_scale = 1e6
        alpha = np.array([
            [0.0, 0.0],
            [100.0, 100.0],
            [50.0, 50.0],
            [20.0, 20.0],
        ])
        model = LiquidityKernelArrivalModel(
            alpha=alpha, beta=beta, K=K, liquidity_scale=liq_scale,
            num_trajectories=N, seed=42,
        )

        num_ticks = 20
        liq = np.zeros((N, num_ticks))
        current_idx = 10
        # Left ticks (sell direction): indices 9, 8, 7
        liq[0, 9] = 3e6   # d=1
        liq[0, 8] = 1e6   # d=2
        liq[0, 7] = 2e6   # d=3
        # Right ticks (buy direction): indices 11, 12, 13
        liq[0, 11] = 1e6  # d=1
        liq[0, 12] = 4e6  # d=2
        liq[0, 13] = 0.0  # d=3

        amm_price = 95.0
        midprice = 100.0
        state = _make_state(liq, current_tick=np.array([current_idx]),
                            tick_lower_global=0, amm_price=amm_price, midprice=midprice)
        model.update(None, None, None, state)

        # Manual calculation
        w = np.exp(-beta * np.array([1, 2, 3]))
        wl_sell = (w[0] * 3e6 + w[1] * 1e6 + w[2] * 2e6) / liq_scale
        wl_buy = (w[0] * 1e6 + w[1] * 4e6 + w[2] * 0.0) / liq_scale
        mispricing = midprice - amm_price  # +5

        expected_sell = max(0.0, 100.0 + 50.0 * wl_sell - 20.0 * mispricing)
        expected_buy = max(0.0, 100.0 + 50.0 * wl_buy + 20.0 * mispricing)

        np.testing.assert_allclose(model.current_state[0, 0], expected_sell, rtol=1e-10)
        np.testing.assert_allclose(model.current_state[0, 1], expected_buy, rtol=1e-10)

    def test_mispricing_composability_alpha2_zero(self):
        """With alpha_2=0 (no kernel effect), only baseline + mispricing."""
        N = 5
        alpha = np.array([
            [10.0, 10.0],
            [100.0, 100.0],
            [0.0, 0.0],       # no kernel effect
            [20.0, 20.0],
        ])
        model = LiquidityKernelArrivalModel(
            alpha=alpha, K=5, num_trajectories=N, seed=42,
        )

        num_ticks = 50
        liq = np.ones((N, num_ticks)) * 5e6  # liquidity irrelevant
        amm_price = 95.0
        midprice = 100.0
        state = _make_state(liq, current_tick=np.full(N, 25),
                            tick_lower_global=0, amm_price=amm_price, midprice=midprice)
        model.update(None, None, None, state)

        mispricing = midprice - amm_price  # 5.0
        expected_sell = max(10.0, 100.0 + 0.0 - 20.0 * 5.0)  # max(10, 0) = 10
        expected_buy = max(10.0, 100.0 + 0.0 + 20.0 * 5.0)   # max(10, 200) = 200

        np.testing.assert_allclose(model.current_state[:, 0], expected_sell, rtol=1e-10)
        np.testing.assert_allclose(model.current_state[:, 1], expected_buy, rtol=1e-10)


class TestMultiTrajectory:

    def test_different_ticks_per_trajectory(self):
        """Each trajectory's intensity depends on its own tick neighborhood."""
        N = 3
        K = 3
        alpha = np.array([
            [0.0, 0.0],
            [0.0, 0.0],       # zero baseline
            [100.0, 100.0],
            [0.0, 0.0],       # no mispricing
        ])
        model = LiquidityKernelArrivalModel(
            alpha=alpha, beta=0.5, K=K, liquidity_scale=1e6,
            num_trajectories=N, seed=42,
        )

        num_ticks = 50
        liq = np.zeros((N, num_ticks))
        # Trajectory 0: current at 10, liquidity at 11
        liq[0, 11] = 5e6
        # Trajectory 1: current at 25, no nearby liquidity
        # Trajectory 2: current at 40, liquidity at 41
        liq[2, 41] = 5e6

        current_ticks = np.array([10, 25, 40])
        state = _make_state(liq, current_ticks, tick_lower_global=0)
        model.update(None, None, None, state)

        # Traj 0 and 2 should have same buy intensity (same liquidity pattern)
        np.testing.assert_allclose(
            model.current_state[0, 1], model.current_state[2, 1], rtol=1e-10)
        # Traj 1 should have zero buy intensity (no nearby liquidity, zero baseline)
        np.testing.assert_allclose(model.current_state[1, 1], 0.0)

    def test_different_liquidity_per_trajectory(self):
        """Trajectories with different liquidity get different intensities."""
        N = 2
        K = 3
        alpha = np.array([
            [0.0, 0.0],
            [100.0, 100.0],
            [50.0, 50.0],
            [0.0, 0.0],
        ])
        model = LiquidityKernelArrivalModel(
            alpha=alpha, beta=0.5, K=K, liquidity_scale=1e6,
            num_trajectories=N, seed=42,
        )

        num_ticks = 30
        liq = np.zeros((N, num_ticks))
        current_idx = 15
        # Traj 0: heavy liquidity on right
        liq[0, current_idx + 1: current_idx + 4] = 3e6
        # Traj 1: light liquidity on right
        liq[1, current_idx + 1: current_idx + 4] = 0.5e6

        state = _make_state(liq, current_tick=np.full(N, current_idx),
                            tick_lower_global=0)
        model.update(None, None, None, state)

        # Traj 0 should have higher buy intensity
        assert model.current_state[0, 1] > model.current_state[1, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
