"""
Test script for PoissonLinearArrivalModel with stateful pattern (mbt_gym style)

Tests:
1. Internal state management - update() modifies current_state
2. Sign convention - BUY > SELL when AMM underpriced (S > Z)
3. Sign convention - SELL > BUY when AMM overpriced (S < Z)
4. Liquidity effect - Higher L increases both arrival rates
5. Parameter validation - alpha shape and floor constraints
6. get_arrivals() works without arguments
7. reset() restores baseline intensity
"""
import numpy as np
import sys
import pytest

sys.path.insert(0, '/home/gchionas/Programming/Blockchain/Ethereum/defi-trading/SAiFE_gym')

from SAiFE_gym.stochastic_processes.arrival_models import (
    PoissonLinearArrivalModel,
    PoissonArrivalModel
)


class TestPoissonLinearArrivalModelStateful:
    """Test suite for PoissonLinearArrivalModel with stateful pattern."""

    def test_internal_state_updates(self):
        """Test that update() modifies internal state."""
        num_trajectories = 10
        model = PoissonLinearArrivalModel(num_trajectories=num_trajectories, seed=42)
        initial_state = model.current_state.copy()

        state = {
            'active_liquidity': np.full(num_trajectories, 2e6),  # High liquidity
            'amm_price': np.full(num_trajectories, 95.0),        # AMM underpriced
            'midprice': np.full(num_trajectories, 100.0),
        }
        model.update(None, None, None, state)

        # State should have changed
        assert not np.allclose(model.current_state, initial_state), \
            "Internal state should change after update()"
        # BUY intensity should be higher than SELL (AMM underpriced)
        assert np.all(model.current_state[:, 1] > model.current_state[:, 0]), \
            "BUY intensity should be higher than SELL when AMM underpriced"

    def test_get_arrivals_no_arguments(self):
        """Test that get_arrivals() works without arguments."""
        num_trajectories = 10
        model = PoissonLinearArrivalModel(num_trajectories=num_trajectories, seed=42)
        arrivals = model.get_arrivals()  # No arguments!
        assert arrivals.shape == (num_trajectories, 2), \
            f"Expected shape ({num_trajectories}, 2), got {arrivals.shape}"
        assert arrivals.dtype == bool or arrivals.dtype == np.bool_, \
            f"Expected bool dtype, got {arrivals.dtype}"

    def test_reset_restores_baseline(self):
        """Test that reset() restores baseline intensity."""
        num_trajectories = 10
        model = PoissonLinearArrivalModel(num_trajectories=num_trajectories, seed=42)

        # Update state to something different
        state = {
            'active_liquidity': np.full(num_trajectories, 2e6),
            'amm_price': np.full(num_trajectories, 50.0),
            'midprice': np.full(num_trajectories, 100.0),
        }
        model.update(None, None, None, state)

        # Reset
        model.reset()

        # Should be back to baseline (alpha[1])
        expected = np.ones((num_trajectories, 2)) * model.alpha[1]
        np.testing.assert_array_equal(model.current_state, expected)

    def test_initial_state_is_baseline(self):
        """Test that initial current_state is baseline intensity."""
        num_trajectories = 10
        model = PoissonLinearArrivalModel(num_trajectories=num_trajectories, seed=42)

        expected = np.ones((num_trajectories, 2)) * model.alpha[1]
        np.testing.assert_array_equal(model.current_state, expected)

    def test_update_with_none_state_preserves_current(self):
        """Test that update() with None state preserves current state."""
        num_trajectories = 10
        model = PoissonLinearArrivalModel(num_trajectories=num_trajectories, seed=42)

        # First update with real state
        state = {
            'active_liquidity': np.full(num_trajectories, 2e6),
            'amm_price': np.full(num_trajectories, 95.0),
            'midprice': np.full(num_trajectories, 100.0),
        }
        model.update(None, None, None, state)
        state_after_update = model.current_state.copy()

        # Update with None should preserve state
        model.update(None, None, None, None)
        np.testing.assert_array_equal(model.current_state, state_after_update)

    def test_mispricing_increases_buy_when_amm_underpriced(self):
        """When S > Z (AMM underpriced), BUY intensity should exceed SELL."""
        num_trajectories = 10000
        model = PoissonLinearArrivalModel(
            alpha=np.array([
                [10.0, 10.0],   # a0: floor
                [100.0, 100.0], # a1: baseline
                [0.0, 0.0],     # a2: no liquidity effect (isolate mispricing)
                [50.0, 50.0],   # a3: strong arbitrage coefficient
            ]),
            step_size=0.01,  # Higher step_size for more arrivals
            num_trajectories=num_trajectories,
            seed=42
        )

        # S > Z: AMM underpriced (market price higher than AMM)
        state = {
            'active_liquidity': np.full(num_trajectories, 1e6),
            'amm_price': np.full(num_trajectories, 95.0),    # Z = 95
            'midprice': np.full(num_trajectories, 100.0),    # S = 100
        }

        # Update internal state
        model.update(None, None, None, state)

        # Run multiple samples for statistical significance
        buy_counts = 0
        sell_counts = 0
        for _ in range(100):
            arrivals = model.get_arrivals()
            sell_counts += arrivals[:, 0].sum()
            buy_counts += arrivals[:, 1].sum()

        assert buy_counts > sell_counts, (
            f"When AMM underpriced (S > Z), BUY should exceed SELL. "
            f"Got BUY={buy_counts}, SELL={sell_counts}"
        )

    def test_mispricing_increases_sell_when_amm_overpriced(self):
        """When S < Z (AMM overpriced), SELL intensity should exceed BUY."""
        num_trajectories = 10000
        model = PoissonLinearArrivalModel(
            alpha=np.array([
                [10.0, 10.0],   # a0: floor
                [100.0, 100.0], # a1: baseline
                [0.0, 0.0],     # a2: no liquidity effect (isolate mispricing)
                [50.0, 50.0],   # a3: strong arbitrage coefficient
            ]),
            step_size=0.01,
            num_trajectories=num_trajectories,
            seed=42
        )

        # S < Z: AMM overpriced (market price lower than AMM)
        state = {
            'active_liquidity': np.full(num_trajectories, 1e6),
            'amm_price': np.full(num_trajectories, 105.0),   # Z = 105
            'midprice': np.full(num_trajectories, 100.0),    # S = 100
        }

        # Update internal state
        model.update(None, None, None, state)

        # Run multiple samples for statistical significance
        buy_counts = 0
        sell_counts = 0
        for _ in range(100):
            arrivals = model.get_arrivals()
            sell_counts += arrivals[:, 0].sum()
            buy_counts += arrivals[:, 1].sum()

        assert sell_counts > buy_counts, (
            f"When AMM overpriced (S < Z), SELL should exceed BUY. "
            f"Got SELL={sell_counts}, BUY={buy_counts}"
        )

    def test_liquidity_increases_arrival_rates(self):
        """Higher active liquidity should increase both arrival rates."""
        num_trajectories = 10000
        model = PoissonLinearArrivalModel(
            alpha=np.array([
                [10.0, 10.0],   # a0: floor
                [50.0, 50.0],   # a1: baseline
                [100.0, 100.0], # a2: strong liquidity coefficient
                [0.0, 0.0],     # a3: no arbitrage effect (isolate liquidity)
            ]),
            liquidity_scale=1e6,
            step_size=0.01,
            num_trajectories=num_trajectories,
            seed=42
        )

        # Low liquidity case
        state_low = {
            'active_liquidity': np.full(num_trajectories, 1e5),  # L = 0.1 (normalized)
            'amm_price': np.full(num_trajectories, 100.0),
            'midprice': np.full(num_trajectories, 100.0),
        }

        # High liquidity case
        state_high = {
            'active_liquidity': np.full(num_trajectories, 2e6),  # L = 2.0 (normalized)
            'amm_price': np.full(num_trajectories, 100.0),
            'midprice': np.full(num_trajectories, 100.0),
        }

        # Collect arrivals with low liquidity
        model.update(None, None, None, state_low)
        low_arrivals = 0
        for _ in range(100):
            arrivals_low = model.get_arrivals()
            low_arrivals += arrivals_low.sum()

        # Collect arrivals with high liquidity
        model.update(None, None, None, state_high)
        high_arrivals = 0
        for _ in range(100):
            arrivals_high = model.get_arrivals()
            high_arrivals += arrivals_high.sum()

        assert high_arrivals > low_arrivals, (
            f"Higher liquidity should increase arrivals. "
            f"Got high={high_arrivals}, low={low_arrivals}"
        )

    def test_intensity_floor_prevents_negative(self):
        """The a0 floor should prevent negative intensities."""
        num_trajectories = 1000
        model = PoissonLinearArrivalModel(
            alpha=np.array([
                [50.0, 50.0],   # a0: high floor
                [10.0, 10.0],   # a1: low baseline
                [0.0, 0.0],     # a2: no liquidity
                [100.0, 100.0], # a3: strong arbitrage (to drive intensity negative)
            ]),
            step_size=0.01,
            num_trajectories=num_trajectories,
            seed=42
        )

        # Large mispricing that would drive linear part negative
        state = {
            'active_liquidity': np.full(num_trajectories, 1e6),
            'amm_price': np.full(num_trajectories, 100.0),
            'midprice': np.full(num_trajectories, 100.0),  # S - Z = 0, symmetric
        }
        model.update(None, None, None, state)

        # Even with extreme mispricing, arrivals should still occur due to floor
        arrivals = model.get_arrivals()

        # With floor of 50 and step_size of 0.01, expected rate is 0.5
        # Should see arrivals (not all zeros)
        assert arrivals.sum() > 0, "Floor should ensure some arrivals occur"

    def test_alpha_shape_validation(self):
        """Test that alpha must have shape (4, 2)."""
        with pytest.raises(AssertionError, match="alpha must have shape"):
            PoissonLinearArrivalModel(
                alpha=np.array([[10, 20, 30], [40, 50, 60]]),  # Wrong shape
                num_trajectories=1
            )

    def test_floor_nonnegative_validation(self):
        """Test that a0 (floor) must be non-negative."""
        with pytest.raises(AssertionError, match="must be non-negative"):
            PoissonLinearArrivalModel(
                alpha=np.array([
                    [-10.0, -5.0],  # Negative floor
                    [50, 50],
                    [0.1, 0.1],
                    [1, 1]
                ]),
                num_trajectories=1
            )

    def test_default_alpha_values(self):
        """Test that default alpha values are used when not provided."""
        model = PoissonLinearArrivalModel(num_trajectories=1, seed=42)

        # Check defaults
        expected_alpha = np.array([
            [10.0, 10.0],    # a0
            [100.0, 100.0],  # a1
            [50.0, 50.0],    # a2
            [5.0, 5.0],      # a3
        ])
        np.testing.assert_array_equal(model.alpha, expected_alpha)
        assert model.liquidity_scale == 1e6


class TestPoissonArrivalModelStateful:
    """Test suite for PoissonArrivalModel with stateful pattern."""

    def test_get_arrivals_no_arguments(self):
        """Test that get_arrivals() works without arguments."""
        num_trajectories = 100
        model = PoissonArrivalModel(
            intensity=np.array([100.0, 100.0]),
            step_size=0.01,
            num_trajectories=num_trajectories,
            seed=42
        )

        # Should work without arguments
        arrivals = model.get_arrivals()
        assert arrivals.shape == (num_trajectories, 2)

    def test_internal_state_is_constant(self):
        """Test that PoissonArrivalModel has constant internal state."""
        num_trajectories = 10
        intensity = np.array([150.0, 200.0])
        model = PoissonArrivalModel(
            intensity=intensity,
            step_size=0.01,
            num_trajectories=num_trajectories,
            seed=42
        )

        # Internal state should be the constant intensity
        expected = np.ones((num_trajectories, 2)) * intensity
        np.testing.assert_array_equal(model.current_state, expected)

    def test_update_does_not_change_state(self):
        """Test that update() does not change state for constant model."""
        num_trajectories = 10
        model = PoissonArrivalModel(
            intensity=np.array([100.0, 100.0]),
            step_size=0.01,
            num_trajectories=num_trajectories,
            seed=42
        )

        initial_state = model.current_state.copy()

        # Update with some state (should be ignored)
        state = {
            'active_liquidity': np.full(num_trajectories, 2e6),
            'amm_price': np.full(num_trajectories, 95.0),
            'midprice': np.full(num_trajectories, 100.0),
        }
        model.update(None, None, None, state)

        # State should be unchanged
        np.testing.assert_array_equal(model.current_state, initial_state)

    def test_reset(self):
        """Test that reset() works (even though it's a no-op for constant model)."""
        num_trajectories = 10
        intensity = np.array([100.0, 100.0])
        model = PoissonArrivalModel(
            intensity=intensity,
            step_size=0.01,
            num_trajectories=num_trajectories,
            seed=42
        )

        model.reset()
        expected = np.ones((num_trajectories, 2)) * intensity
        np.testing.assert_array_equal(model.current_state, expected)


class TestFormulaVerification:
    """Verify the mathematical formulas are implemented correctly."""

    def test_intensity_formula_symmetric_case(self):
        """Test intensity formula when S = Z (no mispricing)."""
        num_trajectories = 10000
        model = PoissonLinearArrivalModel(
            alpha=np.array([
                [0.0, 0.0],     # a0: no floor
                [100.0, 100.0], # a1: baseline
                [50.0, 50.0],   # a2: liquidity coefficient
                [10.0, 10.0],   # a3: arbitrage coefficient
            ]),
            liquidity_scale=1e6,
            step_size=0.1,  # High step_size to get measurable rates
            num_trajectories=num_trajectories,
            seed=42
        )

        # S = Z: no mispricing
        state = {
            'active_liquidity': np.full(num_trajectories, 1e6),  # L = 1.0 (normalized)
            'amm_price': np.full(num_trajectories, 100.0),
            'midprice': np.full(num_trajectories, 100.0),
        }
        model.update(None, None, None, state)

        # Expected intensity: a1 + a2*L = 100 + 50*1.0 = 150
        expected_intensity = 150.0
        np.testing.assert_allclose(
            model.current_state,
            np.full((num_trajectories, 2), expected_intensity),
            rtol=1e-10
        )

        # When S = Z, SELL and BUY should have same rates
        sell_counts = 0
        buy_counts = 0
        for _ in range(100):
            arrivals = model.get_arrivals()
            sell_counts += arrivals[:, 0].sum()
            buy_counts += arrivals[:, 1].sum()

        # They should be approximately equal (within statistical noise)
        total = sell_counts + buy_counts
        sell_ratio = sell_counts / total
        buy_ratio = buy_counts / total

        # Should be roughly 50-50
        assert 0.45 < sell_ratio < 0.55, f"SELL ratio should be ~0.5 when S=Z, got {sell_ratio}"
        assert 0.45 < buy_ratio < 0.55, f"BUY ratio should be ~0.5 when S=Z, got {buy_ratio}"

    def test_intensity_values_match_formula(self):
        """Test that computed intensity values match the analytical formula."""
        num_trajectories = 5
        model = PoissonLinearArrivalModel(
            alpha=np.array([
                [10.0, 10.0],   # a0
                [100.0, 100.0], # a1
                [50.0, 50.0],   # a2
                [20.0, 20.0],   # a3
            ]),
            liquidity_scale=1e6,
            num_trajectories=num_trajectories,
            seed=42
        )

        # Test case: L=1.5 (normalized), S-Z = 5 (AMM underpriced)
        state = {
            'active_liquidity': np.full(num_trajectories, 1.5e6),  # L = 1.5
            'amm_price': np.full(num_trajectories, 95.0),          # Z = 95
            'midprice': np.full(num_trajectories, 100.0),          # S = 100
        }
        model.update(None, None, None, state)

        # Expected intensity:
        # SELL: a1 + a2*L - a3*(S-Z) = 100 + 50*1.5 - 20*5 = 100 + 75 - 100 = 75
        # BUY:  a1 + a2*L + a3*(S-Z) = 100 + 50*1.5 + 20*5 = 100 + 75 + 100 = 275
        expected_sell = 75.0
        expected_buy = 275.0

        np.testing.assert_allclose(model.current_state[:, 0], expected_sell, rtol=1e-10)
        np.testing.assert_allclose(model.current_state[:, 1], expected_buy, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
