"""
Tests for fee accumulation correctness vs Uniswap V3 mechanics.

In Uniswap V3, when a swap crosses a tick boundary the fee is split
between the two ticks proportionally to the input amount traded in each:

  Buy (price goes up), crossing tick i:
    - Tick (i-1): fee on y_to_boundary   = L_{i-1} * (sqrt_p_boundary - sqrt_p_current)
    - Tick (i):   fee on y_from_boundary = L_i     * (sqrt_p_new       - sqrt_p_boundary)

  Sell (price goes down), crossing tick i:
    - Tick (i):   fee on x_to_boundary   = L_i     * (1/sqrt_p_boundary - 1/sqrt_p_current)
    - Tick (i-1): fee on x_from_boundary = L_{i-1} * (1/sqrt_p_new      - 1/sqrt_p_boundary)

These tests check whether the current implementation satisfies this invariant.
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
    LP_EVER_DEPLOYED_KEY
)
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel


# ---------------------------------------------------------------------------
# Helpers (identical to test_liquidity_dependent_price_impact.py)
# ---------------------------------------------------------------------------

def create_test_model(num_trajectories=1, num_ticks=100, initial_price=100.0):
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
    return UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=0.003,
        tau=5,
        num_ticks=num_ticks,
        exponential_value=1.0001,
        seed=42
    )


def initialize_state_geometric_midpoint(model, liquidity_value=1e6):
    """
    Place price at the true geometric midpoint of the initial tick:
      sqrt_p = sqrt(exp_val^(T + 0.5)) = sqrt_p_low * exp_quarter
    This matches the snap target used by _process_buy/sell_single on crossing,
    ensuring the test starts from a clean steady-state position.
    """
    num_traj = model.num_trajectories
    num_ticks = model.num_ticks
    initial_price = model.initial_price
    initial_tick = int(np.floor(np.log(initial_price) / np.log(model.exponential_value)))
    model.tick_lower_global = initial_tick - num_ticks // 2

    sqrt_p_low = np.sqrt(model.exponential_value ** initial_tick)
    initial_sqrt_price = sqrt_p_low * model.exp_quarter   # geometric midpoint

    model.state = {
        POOL_SQRT_PRICE_KEY:        np.full(num_traj, initial_sqrt_price),
        POOL_CURRENT_TICK_KEY:      np.full(num_traj, float(initial_tick)),
        POOL_LIQUIDITY_ARRAY_KEY:   np.full((num_traj, num_ticks), liquidity_value),
        FEES0_KEY:                  np.zeros((num_traj, num_ticks)),
        FEES1_KEY:                  np.zeros((num_traj, num_ticks)),
        LP_LIQUIDITY_KEY:           np.zeros(num_traj),
        LP_TICK_LOWER_KEY:          np.full(num_traj, float(initial_tick - model.tau)),
        LP_TICK_UPPER_KEY:          np.full(num_traj, float(initial_tick + model.tau)),
        LP_COLLECTED_FEES0_KEY:     np.zeros(num_traj),
        LP_COLLECTED_FEES1_KEY:     np.zeros(num_traj),
        LP_UNCLAIMED_FEES0_KEY:     np.zeros(num_traj),
        LP_UNCLAIMED_FEES1_KEY:     np.zeros(num_traj),
        LP_FEE_SNAPSHOT0_KEY:       np.zeros(num_traj),
        LP_FEE_SNAPSHOT1_KEY:       np.zeros(num_traj),
        ASSET_PRICE_KEY:            np.full(num_traj, initial_price),
        TIME_KEY:                   np.zeros(num_traj),
        LP_EVER_DEPLOYED_KEY:       np.zeros(num_traj, dtype=bool),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeeSplitUniswapV3:
    """
    Verify that fee accumulation on tick-crossing matches Uniswap V3 mechanics:
    fees are split proportionally between the two ticks that participate in the trade.
    """

    def test_buy_crossing_fee_at_current_tick_matches_uniswap_v3(self):
        """
        Tick (i-1) fee on a buy crossing must equal fee_multiplier * y_to_boundary.

        y_to_boundary = L_{i-1} * (sqrt_p_boundary - sqrt_p_current)

        This is the portion of the trade that executes in tick (i-1) before
        the price reaches the boundary — Uniswap V3 whitepaper Eq. 6.13.
        """
        model = create_test_model()
        initialize_state_geometric_midpoint(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx     = current_tick - model.tick_lower_global

        sqrt_p_c        = model.state[POOL_SQRT_PRICE_KEY][0]
        sqrt_p_boundary = np.sqrt(model.exponential_value ** (current_tick + 1))
        L               = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx]

        y_to_boundary    = L * (sqrt_p_boundary - sqrt_p_c)
        expected_fee_i_minus_1 = model.fee_multiplier * y_to_boundary

        model.update_state(np.array([[0, 1]]), None)

        fee_at_current_tick = model.state[FEES1_KEY][0, tick_idx]
        assert np.isclose(fee_at_current_tick, expected_fee_i_minus_1, rtol=1e-10), (
            f"Tick (i-1) fee mismatch: expected {expected_fee_i_minus_1:.6e}, "
            f"got {fee_at_current_tick:.6e}"
        )

    def test_buy_crossing_fee_at_next_tick_matches_uniswap_v3(self):
        """
        Tick (i) fee on a buy crossing must equal fee_multiplier * y_from_boundary.

        y_from_boundary = L_i * (sqrt_p_new - sqrt_p_boundary)
        sqrt_p_new = sqrt_p_boundary * exp_quarter  (the midpoint snap target)

        This is the portion of the trade that executes in tick (i) after crossing.
        Per Uniswap V3, this portion ALSO generates fees — deposited in tick (i).

        Current implementation deposits ZERO fees here (missing split).
        """
        model = create_test_model()
        initialize_state_geometric_midpoint(model, liquidity_value=1e6)

        current_tick  = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx      = current_tick - model.tick_lower_global
        next_tick_idx = tick_idx + 1

        sqrt_p_boundary = np.sqrt(model.exponential_value ** (current_tick + 1))
        sqrt_p_new      = sqrt_p_boundary * model.exp_quarter   # snap target in next tick
        L_next          = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, next_tick_idx]

        y_from_boundary      = L_next * (sqrt_p_new - sqrt_p_boundary)
        expected_fee_tick_i  = model.fee_multiplier * y_from_boundary

        model.update_state(np.array([[0, 1]]), None)

        fee_at_next_tick = model.state[FEES1_KEY][0, next_tick_idx]
        assert np.isclose(fee_at_next_tick, expected_fee_tick_i, rtol=1e-10), (
            f"Tick (i) fee mismatch: expected {expected_fee_tick_i:.6e} (Uniswap V3), "
            f"got {fee_at_next_tick:.6e}  ← missing split in current implementation"
        )

    def test_sell_crossing_fee_at_current_tick_matches_uniswap_v3(self):
        """
        Tick (i) fee on a sell crossing must equal fee_multiplier * x_to_boundary.

        x_to_boundary = L_i * (1/sqrt_p_boundary - 1/sqrt_p_current)

        This is the portion of the sell that executes in tick (i) before reaching
        the lower boundary — Uniswap V3 whitepaper Eq. 6.14.
        """
        model = create_test_model()
        initialize_state_geometric_midpoint(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx     = current_tick - model.tick_lower_global

        sqrt_p_c        = model.state[POOL_SQRT_PRICE_KEY][0]
        sqrt_p_boundary = np.sqrt(model.exponential_value ** current_tick)   # lower boundary of current tick
        L               = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx]

        x_to_boundary          = L * (1.0 / sqrt_p_boundary - 1.0 / sqrt_p_c)
        expected_fee_tick_i    = model.fee_multiplier * x_to_boundary

        model.update_state(np.array([[1, 0]]), None)

        fee_at_current_tick = model.state[FEES0_KEY][0, tick_idx]
        assert np.isclose(fee_at_current_tick, expected_fee_tick_i, rtol=1e-10), (
            f"Tick (i) fee mismatch: expected {expected_fee_tick_i:.6e}, "
            f"got {fee_at_current_tick:.6e}"
        )

    def test_sell_crossing_fee_at_prev_tick_matches_uniswap_v3(self):
        """
        Tick (i-1) fee on a sell crossing must equal fee_multiplier * x_from_boundary.

        x_from_boundary = L_{i-1} * (1/sqrt_p_new - 1/sqrt_p_boundary)
        sqrt_p_new = sqrt_p_boundary / exp_quarter  (the midpoint snap target)

        This is the portion of the sell that executes in tick (i-1) after crossing.
        Per Uniswap V3, this portion ALSO generates fees — deposited in tick (i-1).

        Current implementation deposits ZERO fees here (missing split).
        """
        model = create_test_model()
        initialize_state_geometric_midpoint(model, liquidity_value=1e6)

        current_tick  = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx      = current_tick - model.tick_lower_global
        prev_tick_idx = tick_idx - 1

        sqrt_p_boundary = np.sqrt(model.exponential_value ** current_tick)
        sqrt_p_new      = sqrt_p_boundary / model.exp_quarter   # snap target in prev tick
        L_prev          = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, prev_tick_idx]

        x_from_boundary     = L_prev * (1.0 / sqrt_p_new - 1.0 / sqrt_p_boundary)
        expected_fee_tick_i_minus_1 = model.fee_multiplier * x_from_boundary

        model.update_state(np.array([[1, 0]]), None)

        fee_at_prev_tick = model.state[FEES0_KEY][0, prev_tick_idx]
        assert np.isclose(fee_at_prev_tick, expected_fee_tick_i_minus_1, rtol=1e-10), (
            f"Tick (i-1) fee mismatch: expected {expected_fee_tick_i_minus_1:.6e} (Uniswap V3), "
            f"got {fee_at_prev_tick:.6e}  ← missing split in current implementation"
        )

    def test_total_fee_volume_buy_crossing(self):
        """
        Total fees collected across all ticks on a buy crossing must equal
        fee_multiplier * (y_to_boundary + y_from_boundary).

        This is the full fee on the complete trade, regardless of how it is
        split between ticks. Confirms the total is also wrong when the split
        is missing (some fees are simply not recorded).
        """
        model = create_test_model()
        initialize_state_geometric_midpoint(model, liquidity_value=1e6)

        current_tick  = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx      = current_tick - model.tick_lower_global
        next_tick_idx = tick_idx + 1

        sqrt_p_c        = model.state[POOL_SQRT_PRICE_KEY][0]
        sqrt_p_boundary = np.sqrt(model.exponential_value ** (current_tick + 1))
        sqrt_p_new      = sqrt_p_boundary * model.exp_quarter
        L               = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx]

        total_trade_volume = L * (sqrt_p_boundary - sqrt_p_c) + L * (sqrt_p_new - sqrt_p_boundary)
        expected_total_fee = model.fee_multiplier * total_trade_volume

        model.update_state(np.array([[0, 1]]), None)

        actual_total_fee = model.state[FEES1_KEY][0].sum()
        assert np.isclose(actual_total_fee, expected_total_fee, rtol=1e-10), (
            f"Total fee mismatch: expected {expected_total_fee:.6e}, "
            f"got {actual_total_fee:.6e}  "
            f"(missing {expected_total_fee - actual_total_fee:.6e})"
        )
