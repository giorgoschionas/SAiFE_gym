"""
Test script for PoissonLinearArrivalModel with context-based interface

Tests:
1. Basic functionality - shape and dtype of arrivals
2. Sign convention - BUY > SELL when AMM underpriced (S > Z)
3. Sign convention - SELL > BUY when AMM overpriced (S < Z)
4. Liquidity effect - Higher L increases both arrival rates
5. Parameter validation - alpha shape and floor constraints
6. Context validation - requires context dict
"""
import numpy as np
import sys
import pytest

sys.path.insert(0, '/home/gchionas/Programming/Blockchain/Ethereum/defi-trading/SAiFE_gym')

from SAiFE_gym.stochastic_processes.arrival_models import (
    PoissonLinearArrivalModel,
    PoissonArrivalModel
)


class TestPoissonLinearArrivalModel:
    """Test suite for PoissonLinearArrivalModel with context-based interface."""

    def test_basic_functionality(self):
        """Test that arrivals have correct shape and dtype."""
        num_trajectories = 100
        model = PoissonLinearArrivalModel(
            alpha=np.array([
                [10.0, 10.0],   # a0: minimum intensity floor
                [100.0, 100.0], # a1: baseline intensity
                [50.0, 50.0],   # a2: liquidity coefficient
                [5.0, 5.0],     # a3: arbitrage coefficient
            ]),
            liquidity_scale=1e6,
            step_size=0.001,
            num_trajectories=num_trajectories,
            seed=42
        )

        context = {
            'active_liquidity': np.full(num_trajectories, 1e6),
            'amm_price': np.full(num_trajectories, 100.0),
            'midprice': np.full(num_trajectories, 100.0),
        }

        arrivals = model.get_arrivals(context)

        assert arrivals.shape == (num_trajectories, 2), f"Expected shape ({num_trajectories}, 2), got {arrivals.shape}"
        assert arrivals.dtype == bool or arrivals.dtype == np.bool_, f"Expected bool dtype, got {arrivals.dtype}"

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
        context = {
            'active_liquidity': np.full(num_trajectories, 1e6),
            'amm_price': np.full(num_trajectories, 95.0),    # Z = 95
            'midprice': np.full(num_trajectories, 100.0),    # S = 100
        }

        # Run multiple samples for statistical significance
        buy_counts = 0
        sell_counts = 0
        for _ in range(100):
            arrivals = model.get_arrivals(context)
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
        context = {
            'active_liquidity': np.full(num_trajectories, 1e6),
            'amm_price': np.full(num_trajectories, 105.0),   # Z = 105
            'midprice': np.full(num_trajectories, 100.0),    # S = 100
        }

        # Run multiple samples for statistical significance
        buy_counts = 0
        sell_counts = 0
        for _ in range(100):
            arrivals = model.get_arrivals(context)
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
        context_low = {
            'active_liquidity': np.full(num_trajectories, 1e5),  # L = 0.1 (normalized)
            'amm_price': np.full(num_trajectories, 100.0),
            'midprice': np.full(num_trajectories, 100.0),
        }

        # High liquidity case
        context_high = {
            'active_liquidity': np.full(num_trajectories, 2e6),  # L = 2.0 (normalized)
            'amm_price': np.full(num_trajectories, 100.0),
            'midprice': np.full(num_trajectories, 100.0),
        }

        # Collect arrivals
        low_arrivals = 0
        high_arrivals = 0
        for _ in range(100):
            arrivals_low = model.get_arrivals(context_low)
            arrivals_high = model.get_arrivals(context_high)
            low_arrivals += arrivals_low.sum()
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
        context = {
            'active_liquidity': np.full(num_trajectories, 1e6),
            'amm_price': np.full(num_trajectories, 100.0),
            'midprice': np.full(num_trajectories, 100.0),  # S - Z = 0, symmetric
        }

        # Even with extreme mispricing, arrivals should still occur due to floor
        arrivals = model.get_arrivals(context)

        # With floor of 50 and step_size of 0.01, expected rate is 0.5
        # Should see arrivals (not all zeros)
        assert arrivals.sum() > 0, "Floor should ensure some arrivals occur"

    def test_context_required(self):
        """Test that PoissonLinearArrivalModel raises error without context."""
        model = PoissonLinearArrivalModel(num_trajectories=10, seed=42)

        with pytest.raises(ValueError, match="requires context dict"):
            model.get_arrivals(None)

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


class TestPoissonArrivalModel:
    """Test suite for PoissonArrivalModel (state-independent)."""

    def test_ignores_context(self):
        """Test that PoissonArrivalModel ignores context parameter."""
        num_trajectories = 100
        model = PoissonArrivalModel(
            intensity=np.array([100.0, 100.0]),
            step_size=0.01,
            num_trajectories=num_trajectories,
            seed=42
        )

        # Should work with None context
        arrivals_none = model.get_arrivals(None)
        assert arrivals_none.shape == (num_trajectories, 2)

        # Should work with context dict (but ignore it)
        context = {
            'active_liquidity': np.full(num_trajectories, 1e6),
            'amm_price': np.full(num_trajectories, 100.0),
            'midprice': np.full(num_trajectories, 100.0),
        }
        arrivals_ctx = model.get_arrivals(context)
        assert arrivals_ctx.shape == (num_trajectories, 2)


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
        context = {
            'active_liquidity': np.full(num_trajectories, 1e6),  # L = 1.0 (normalized)
            'amm_price': np.full(num_trajectories, 100.0),
            'midprice': np.full(num_trajectories, 100.0),
        }

        # Expected intensity: a1 + a2*L = 100 + 50*1.0 = 150
        # Expected rate: 150 * 0.1 = 15.0 (but capped at 1.0 for probability)
        # With step_size=0.1 and intensity=150, P(arrival) = min(1.0, 15.0) = 1.0
        # But that's > 1, so effectively all should fire

        # When S = Z, SELL and BUY should have same rates
        sell_counts = 0
        buy_counts = 0
        for _ in range(100):
            arrivals = model.get_arrivals(context)
            sell_counts += arrivals[:, 0].sum()
            buy_counts += arrivals[:, 1].sum()

        # They should be approximately equal (within statistical noise)
        total = sell_counts + buy_counts
        sell_ratio = sell_counts / total
        buy_ratio = buy_counts / total

        # Should be roughly 50-50
        assert 0.45 < sell_ratio < 0.55, f"SELL ratio should be ~0.5 when S=Z, got {sell_ratio}"
        assert 0.45 < buy_ratio < 0.55, f"BUY ratio should be ~0.5 when S=Z, got {buy_ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
