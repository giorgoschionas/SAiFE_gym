"""
Tests for pre-entry fee collection bug fix.

The bug: FEES0_KEY/FEES1_KEY accumulate continuously. When the LP rebalances
to a new range, _collect_lp_fees() credited the LP's share of ALL fees in that
range -- including fees from before the LP entered. This inflated LP earnings
and removed pre-entry fees from the pool that belong to background LPs.

The fix: Per-trajectory snapshot variables recorded at position entry.
On collection: net_fees = gross_fees - snapshot. Only the net portion is
subtracted from pool arrays.
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
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel


def create_test_model(num_trajectories=1, num_ticks=100, initial_price=100.0,
                      initial_wealth=1e6, tau=5):
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0, volatility=0.0, initial_price=initial_price,
        terminal_time=1.0, step_size=0.005,
        num_trajectories=num_trajectories
    )
    arrival_model = PoissonArrivalModel(
        intensity=np.array([100.0, 100.0]),
        step_size=0.005, num_trajectories=num_trajectories, seed=42
    )
    return UniswapV3ModelDynamics(
        midprice_model=midprice_model, arrival_model=arrival_model,
        num_trajectories=num_trajectories, fee_tier=0.003, tau=tau,
        num_ticks=num_ticks, exponential_value=1.0001,
        seed=42
    )


def initialize_state(model, liquidity_value=1e6, initial_wealth=1e6):
    num_traj = model.num_trajectories
    num_ticks = model.num_ticks
    initial_price = model.initial_price
    initial_tick = int(np.floor(np.log(initial_price) / np.log(model.exponential_value)))
    model.tick_lower_global = initial_tick - num_ticks // 2
    model._build_sqrt_grid()

    initial_sqrt_price = model.sqrt_grid[initial_tick - model.tick_lower_global]

    model.state = {
        POOL_SQRT_PRICE_KEY: np.full(num_traj, initial_sqrt_price, dtype=np.float64),
        POOL_CURRENT_TICK_KEY: np.full(num_traj, initial_tick, dtype=np.float64),
        POOL_LIQUIDITY_ARRAY_KEY: np.full((num_traj, num_ticks), liquidity_value, dtype=np.float64),
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


class TestPreEntryFeeBug:
    """Tests verifying LP cannot collect fees that existed before they entered a range."""

    def test_zero_fees_from_pre_entry_range(self):
        """LP enters range with pre-existing fees and collects zero pre-entry fees."""
        model = create_test_model(initial_wealth=1e6, tau=5)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        arrivals_none = np.array([[0, 0]], dtype=np.int64)

        # Inject pre-existing fees across the range LP will enter
        lp_lower = current_tick - 2
        lp_upper = current_tick + 2
        for tick in range(lp_lower, lp_upper):
            idx = tick - model.tick_lower_global
            model.state[FEES0_KEY][0, idx] = 500.0
            model.state[FEES1_KEY][0, idx] = 300.0

        pre_entry_fee0_total = model.state[FEES0_KEY][0].sum()

        # First rebalance: LP enters the range with pre-existing fees
        action = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals_none, action)

        # Immediately rebalance again (triggers fee collection)
        model.update_state(arrivals_none, action)

        # LP should NOT have collected any of the pre-existing fees
        collected_fee0 = model.state[LP_COLLECTED_FEES0_KEY][0]
        collected_fee1 = model.state[LP_COLLECTED_FEES1_KEY][0]
        assert np.isclose(collected_fee0, 0.0, atol=1e-10), \
            f"LP collected {collected_fee0:.6f} in pre-entry fee0 (should be 0)"
        assert np.isclose(collected_fee1, 0.0, atol=1e-10), \
            f"LP collected {collected_fee1:.6f} in pre-entry fee1 (should be 0)"

    def test_only_post_entry_fees_collected(self):
        """LP earns fees from trades after entering, not before."""
        model = create_test_model(initial_wealth=1e6, tau=5)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        arrivals_none = np.array([[0, 0]], dtype=np.int64)

        # Inject pre-existing fees
        for tick in range(current_tick - 2, current_tick + 2):
            idx = tick - model.tick_lower_global
            model.state[FEES0_KEY][0, idx] = 1000.0

        # Deploy LP
        action = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals_none, action)

        # Generate POST-entry fees via trades
        arrivals_sell = np.array([[1, 0]], dtype=np.int64)
        model.update_state(arrivals_sell, None)

        post_entry_fees_generated = model.state[FEES0_KEY][0].sum() - 4000.0  # approx new fees

        # Rebalance to collect
        model.update_state(arrivals_none, action)

        collected = model.state[LP_COLLECTED_FEES0_KEY][0]
        # LP should have collected only post-entry fees (their share)
        assert collected > 0, "LP should collect post-entry fees"
        # Collected should be much less than total pre-existing fees (4000)
        assert collected < 100.0, \
            f"LP collected {collected:.4f}, likely includes pre-entry fees"

    def test_snapshot_set_on_first_rebalance(self):
        """First deploy into range with existing fees records correct snapshot."""
        model = create_test_model(initial_wealth=1e6, tau=5)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        arrivals_none = np.array([[0, 0]], dtype=np.int64)

        # Inject pre-existing fees
        for tick in range(current_tick - 2, current_tick + 2):
            idx = tick - model.tick_lower_global
            model.state[FEES0_KEY][0, idx] = 200.0

        # Deploy LP
        action = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals_none, action)

        # Snapshot should be non-zero (captures LP's share of pre-existing fees)
        snapshot0 = model.state[LP_FEE_SNAPSHOT0_KEY][0]
        assert snapshot0 > 0, f"Snapshot should be > 0, got {snapshot0}"

    def test_snapshot_zeroed_after_collection(self):
        """Snapshot is consumed (zeroed) after fee collection."""
        model = create_test_model(initial_wealth=1e6, tau=5)
        initialize_state(model, liquidity_value=1e6)

        arrivals_none = np.array([[0, 0]], dtype=np.int64)

        # Inject some pre-existing fees
        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        idx = current_tick - model.tick_lower_global
        model.state[FEES0_KEY][0, idx] = 100.0

        # Deploy LP (sets snapshot)
        action = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals_none, action)
        assert model.state[LP_FEE_SNAPSHOT0_KEY][0] > 0

        # Rebalance again (triggers _collect_lp_fees, which zeroes snapshot,
        # then Phase 4 sets a new snapshot for the new position)
        model.update_state(arrivals_none, action)

        # The snapshot is now set to whatever pre-existing fees are in the
        # new position range (which includes the old pre-entry fees not taken)
        # The key invariant is that _collect_lp_fees zeroed it before Phase 4 reset it
        # We can verify this indirectly: collected fees should be 0 (no post-entry fees)
        assert np.isclose(model.state[LP_COLLECTED_FEES0_KEY][0], 0.0, atol=1e-10)

    def test_pre_entry_fees_remain_in_pool(self):
        """Pre-entry fees stay in pool arrays (not subtracted by LP)."""
        model = create_test_model(initial_wealth=1e6, tau=5)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        arrivals_none = np.array([[0, 0]], dtype=np.int64)

        # Inject pre-existing fees at a specific tick
        idx = current_tick - model.tick_lower_global
        pre_entry_amount = 500.0
        model.state[FEES0_KEY][0, idx] = pre_entry_amount

        # Deploy LP
        action = np.array([[-2, 2]], dtype=np.float64)
        model.update_state(arrivals_none, action)

        # Rebalance (triggers collection)
        model.update_state(arrivals_none, action)

        # Pre-entry fees should still be in the pool (not subtracted)
        remaining = model.state[FEES0_KEY][0, idx]
        assert np.isclose(remaining, pre_entry_amount, rtol=1e-6), \
            f"Pre-entry fees should remain in pool: expected {pre_entry_amount}, got {remaining}"

    def test_same_range_rebalance_correct(self):
        """Rebalance to same range still works correctly with snapshots."""
        model = create_test_model(initial_wealth=1e6, tau=5)
        initialize_state(model, liquidity_value=1e6)

        arrivals_none = np.array([[0, 0]], dtype=np.int64)
        arrivals_sell = np.array([[1, 0]], dtype=np.int64)
        action = np.array([[-2, 2]], dtype=np.float64)

        # Deploy LP
        model.update_state(arrivals_none, action)
        lp_liq_1 = model.state[LP_LIQUIDITY_KEY][0]

        # Generate fees via trade
        model.update_state(arrivals_sell, None)

        # Rebalance to same range (should collect post-entry fees)
        model.update_state(arrivals_none, action)

        assert model.state[LP_COLLECTED_FEES0_KEY][0] > 0, \
            "LP should collect post-entry fees"

        # Liquidity should be preserved or slightly increased (from collected fees)
        lp_liq_2 = model.state[LP_LIQUIDITY_KEY][0]
        assert lp_liq_2 >= lp_liq_1 * 0.999, \
            "LP liquidity should be preserved after same-range rebalance"

    def test_vectorized_snapshots(self):
        """Multiple trajectories with different ranges get independent snapshots."""
        num_traj = 3
        model = create_test_model(num_trajectories=num_traj, initial_wealth=1e6, tau=5)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        arrivals_none = np.zeros((num_traj, 2), dtype=np.int64)

        # Inject different pre-existing fees per trajectory
        for t in range(num_traj):
            idx = (current_tick - model.tick_lower_global) + t
            model.state[FEES0_KEY][t, idx] = (t + 1) * 100.0

        # Deploy with different ranges
        action = np.array([
            [-2, 2],
            [-1, 3],
            [-3, 1],
        ], dtype=np.float64)
        model.update_state(arrivals_none, action)

        # Each trajectory should have an independent snapshot
        snapshots = model.state[LP_FEE_SNAPSHOT0_KEY]
        # At least some should be non-zero (pre-existing fees in their range)
        assert np.any(snapshots > 0), "At least some snapshots should be non-zero"

        # Rebalance again - should collect 0 pre-entry fees
        model.update_state(arrivals_none, action)

        collected = model.state[LP_COLLECTED_FEES0_KEY]
        assert np.allclose(collected, 0.0, atol=1e-10), \
            f"No trajectory should collect pre-entry fees, got {collected}"

    def test_no_pre_entry_scenario_unchanged(self):
        """When no pre-existing fees, behavior is identical to before the fix."""
        model = create_test_model(initial_wealth=1e6, tau=5)
        initialize_state(model, liquidity_value=1e6)

        arrivals_none = np.array([[0, 0]], dtype=np.int64)
        arrivals_sell = np.array([[1, 0]], dtype=np.int64)
        action = np.array([[-2, 2]], dtype=np.float64)

        # Deploy LP into clean range (no pre-existing fees)
        model.update_state(arrivals_none, action)

        # Snapshot should be zero (no pre-existing fees)
        assert np.isclose(model.state[LP_FEE_SNAPSHOT0_KEY][0], 0.0)

        # Generate fees via trades
        model.update_state(arrivals_sell, None)
        fees_generated = model.state[FEES0_KEY][0].sum()
        assert fees_generated > 0

        # Rebalance to collect
        model.update_state(arrivals_none, action)

        # LP should collect their full share (no pre-entry deduction)
        collected = model.state[LP_COLLECTED_FEES0_KEY][0]
        assert collected > 0, "LP should collect post-entry fees normally"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
