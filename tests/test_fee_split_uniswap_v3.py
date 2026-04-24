"""
Tests for fee accumulation correctness under the lattice swap model.

Under the simplified lattice dynamics:
  - Price always lives on `AMM[i] = sqrt(exponential_value^i)`.
  - A buy at tick i uses liquidity L[i] and moves price AMM[i] -> AMM[i+1];
    fees go to FEES1[i] with dy = L[i] * (AMM[i+1] - AMM[i]).
  - A sell at tick i uses liquidity L[i-1] and moves price AMM[i] -> AMM[i-1];
    fees go to FEES0[i-1] with dx = L[i-1] * (1/AMM[i-1] - 1/AMM[i]).
  - No two-tick fee split; each trade deposits fees in exactly one tick.
"""

import numpy as np

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


def initialize_state_at_lattice(model, liquidity_value=1e6):
    """Place the pool price on the lattice: sqrt_p = sqrt(r^T) = AMM[T]."""
    num_traj = model.num_trajectories
    num_ticks = model.num_ticks
    initial_price = model.initial_price
    initial_tick = int(np.floor(np.log(initial_price) / np.log(model.exponential_value)))
    model.tick_lower_global = initial_tick - num_ticks // 2
    model._build_sqrt_grid()

    initial_sqrt_price = model.sqrt_grid[initial_tick - model.tick_lower_global]

    model.state = {
        POOL_SQRT_PRICE_KEY:        np.full(num_traj, initial_sqrt_price),
        POOL_CURRENT_TICK_KEY:      np.full(num_traj, initial_tick, dtype=np.int64),
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


class TestFeeRoutingLattice:
    """Each trade deposits fees in exactly one tick under the lattice model."""

    def test_buy_deposits_fee_at_current_tick_index(self):
        """Buy at tick i: FEES1[i] += fee_multiplier * L[i] * (AMM[i+1] - AMM[i])."""
        model = create_test_model()
        initialize_state_at_lattice(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global

        L_i = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx]
        sqrt_p_i = model.sqrt_grid[tick_idx]
        sqrt_p_next = model.sqrt_grid[tick_idx + 1]

        dy = L_i * (sqrt_p_next - sqrt_p_i)
        expected_fee = model.fee_multiplier * dy

        model.update_state(np.array([[0, 1]]), None)

        assert np.isclose(model.state[FEES1_KEY][0, tick_idx], expected_fee, rtol=1e-10)
        # No fee deposited at i+1 (no two-tick split anymore)
        assert model.state[FEES1_KEY][0, tick_idx + 1] == 0.0
        # Sell side untouched
        assert model.state[FEES0_KEY][0].sum() == 0.0

    def test_sell_deposits_fee_at_prev_tick_index(self):
        """Sell at tick i: FEES0[i-1] += fee_multiplier * L[i-1] * (1/AMM[i-1] - 1/AMM[i])."""
        model = create_test_model()
        initialize_state_at_lattice(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        prev_idx = tick_idx - 1

        L_prev = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, prev_idx]
        sqrt_p_i = model.sqrt_grid[tick_idx]
        sqrt_p_prev = model.sqrt_grid[prev_idx]

        dx = L_prev * (1.0 / sqrt_p_prev - 1.0 / sqrt_p_i)
        expected_fee = model.fee_multiplier * dx

        model.update_state(np.array([[1, 0]]), None)

        assert np.isclose(model.state[FEES0_KEY][0, prev_idx], expected_fee, rtol=1e-10)
        # No fee deposited at current tick index (no two-tick split anymore)
        assert model.state[FEES0_KEY][0, tick_idx] == 0.0
        # Buy side untouched
        assert model.state[FEES1_KEY][0].sum() == 0.0

    def test_buy_fee_scales_with_liquidity_at_current_tick(self):
        """Buy uses L[i] (tick i), not L[i-1] — changing L[i-1] leaves buy fees unchanged."""
        model = create_test_model()
        initialize_state_at_lattice(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global

        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx - 1] = 100.0  # irrelevant to a buy

        model.update_state(np.array([[0, 1]]), None)
        fee_with_changed_prev = model.state[FEES1_KEY][0, tick_idx]

        model2 = create_test_model()
        initialize_state_at_lattice(model2, liquidity_value=1e6)
        model2.update_state(np.array([[0, 1]]), None)
        fee_uniform = model2.state[FEES1_KEY][0, tick_idx]

        assert np.isclose(fee_with_changed_prev, fee_uniform, rtol=1e-12)

    def test_sell_fee_scales_with_liquidity_at_prev_tick(self):
        """Sell uses L[i-1], not L[i] — changing L[i] leaves sell fees unchanged."""
        model = create_test_model()
        initialize_state_at_lattice(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global

        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx] = 100.0  # irrelevant to a sell

        model.update_state(np.array([[1, 0]]), None)
        fee_with_changed_curr = model.state[FEES0_KEY][0, tick_idx - 1]

        model2 = create_test_model()
        initialize_state_at_lattice(model2, liquidity_value=1e6)
        model2.update_state(np.array([[1, 0]]), None)
        fee_uniform = model2.state[FEES0_KEY][0, tick_idx - 1]

        assert np.isclose(fee_with_changed_curr, fee_uniform, rtol=1e-12)

    def test_lattice_invariant_after_trade(self):
        """After any trade, sqrt_price == AMM[current_tick] exactly."""
        model = create_test_model()
        initialize_state_at_lattice(model, liquidity_value=1e6)

        for arrival in [[1, 0], [0, 1], [1, 1]]:
            model.update_state(np.array([arrival]), None)
            tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
            sp = float(model.state[POOL_SQRT_PRICE_KEY][0])
            expected = model.sqrt_grid[tick - model.tick_lower_global]
            assert np.isclose(sp, expected, rtol=1e-12), (
                f"invariant broken after {arrival}: sp={sp}, AMM[tick]={expected}"
            )
