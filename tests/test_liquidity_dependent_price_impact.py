"""
Tests for the lattice swap dynamics in UniswapV3ModelDynamics.

Under the lattice model:
  - Price always lives on AMM[i] = sqrt(exponential_value^i).
  - Every trade moves the price by exactly one lattice step (buy +1, sell -1).
  - Buy at tick i uses L[i]; sell at tick i uses L[i-1].
  - No no-crossing branch, no two-tick fee split.
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


def initialize_state(model, liquidity_value=1e6):
    """Initialize state with the pool price on the lattice (sqrt_p = AMM[initial_tick])."""
    num_traj = model.num_trajectories
    num_ticks = model.num_ticks

    try:
        initial_price = model.initial_price
    except AttributeError:
        initial_price = 100.0

    initial_tick = int(np.floor(np.log(initial_price) / np.log(model.exponential_value)))
    model.tick_lower_global = initial_tick - num_ticks // 2
    model._build_sqrt_grid()

    initial_sqrt_price = model.sqrt_grid[initial_tick - model.tick_lower_global]

    liquidity_array = np.full((num_traj, num_ticks), liquidity_value, dtype=np.float64)

    model.state = {
        POOL_SQRT_PRICE_KEY: np.full(num_traj, initial_sqrt_price, dtype=np.float64),
        POOL_CURRENT_TICK_KEY: np.full(num_traj, initial_tick, dtype=np.int64),
        POOL_LIQUIDITY_ARRAY_KEY: liquidity_array,
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
    }


class TestCrossing:
    """Every arrival moves the price by exactly one lattice step."""

    def test_sell_moves_down_one_tick(self):
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=100.0)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()
        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()

        model.update_state(np.array([[True, False]]), None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick - 1
        assert model.state[POOL_SQRT_PRICE_KEY][0] < initial_sqrt_price
        expected = model.sqrt_grid[(initial_tick - 1) - model.tick_lower_global]
        assert np.isclose(model.state[POOL_SQRT_PRICE_KEY][0], expected, rtol=1e-12)

    def test_buy_moves_up_one_tick(self):
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=100.0)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()
        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()

        model.update_state(np.array([[False, True]]), None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick + 1
        assert model.state[POOL_SQRT_PRICE_KEY][0] > initial_sqrt_price
        expected = model.sqrt_grid[(initial_tick + 1) - model.tick_lower_global]
        assert np.isclose(model.state[POOL_SQRT_PRICE_KEY][0], expected, rtol=1e-12)


class TestZeroLiquidity:
    """Zero liquidity still moves the price by one tick; only fees are zero."""

    def test_sell_zero_prev_tick_liquidity_still_moves(self):
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        prev_idx = tick_idx - 1
        # A sell uses L[i-1], so zero that out.
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, prev_idx] = 0.0

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()
        model.update_state(np.array([[True, False]]), None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick - 1
        assert model.state[FEES0_KEY][0, prev_idx] == 0.0

    def test_buy_zero_current_tick_liquidity_still_moves(self):
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        # A buy uses L[i], so zero that out.
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx] = 0.0

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()
        model.update_state(np.array([[False, True]]), None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick + 1
        assert model.state[FEES1_KEY][0, tick_idx] == 0.0


class TestSingleArrival:
    """Bernoulli arrivals: at most one sell and one buy per step."""

    def test_zero_arrivals_no_change(self):
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()
        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()

        model.update_state(np.array([[False, False]]), None)

        assert model.state[POOL_SQRT_PRICE_KEY][0] == initial_sqrt_price
        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick

    def test_single_sell_one_tick(self):
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()
        model.update_state(np.array([[True, False]]), None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick - 1

    def test_single_buy_one_tick(self):
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()
        model.update_state(np.array([[False, True]]), None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick + 1

    def test_simultaneous_net_zero(self):
        """Exactly one sell + one buy returns the tick to its starting value."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        initial_tick = model.state[POOL_CURRENT_TICK_KEY][0].copy()
        initial_sqrt_price = model.state[POOL_SQRT_PRICE_KEY][0].copy()

        model.update_state(np.array([[True, True]]), None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_tick
        # Round-trip should be exact on the lattice
        assert np.isclose(model.state[POOL_SQRT_PRICE_KEY][0], initial_sqrt_price, rtol=1e-12)


class TestVectorization:
    """Operations vectorize across trajectories."""

    def test_multiple_trajectories_different_directions(self):
        num_traj = 4
        model = create_test_model(num_trajectories=num_traj, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        initial_sqrt_prices = model.state[POOL_SQRT_PRICE_KEY].copy()
        initial_ticks = model.state[POOL_CURRENT_TICK_KEY].copy()

        arrivals = np.array([
            [True, False],   # sell
            [False, False],  # no trade
            [False, True],   # buy
            [True, True],    # both
        ])
        model.update_state(arrivals, None)

        assert model.state[POOL_CURRENT_TICK_KEY][0] == initial_ticks[0] - 1
        assert model.state[POOL_SQRT_PRICE_KEY][1] == initial_sqrt_prices[1]
        assert model.state[POOL_CURRENT_TICK_KEY][2] == initial_ticks[2] + 1
        assert model.state[POOL_CURRENT_TICK_KEY][3] == initial_ticks[3]

    def test_mixed_liquidity_per_trajectory(self):
        num_traj = 2
        model = create_test_model(num_trajectories=num_traj, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        # Sells use L[i-1], so change the prev-tick liquidity for traj 0
        model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx - 1] = 1e3

        arrivals = np.array([[True, False], [True, False]])
        model.update_state(arrivals, None)

        # Both trajectories still move one tick down
        assert model.state[POOL_CURRENT_TICK_KEY][0] == current_tick - 1
        assert model.state[POOL_CURRENT_TICK_KEY][1] == current_tick - 1
        # But traj 0 earned less fees (lower L[i-1])
        assert model.state[FEES0_KEY][0, tick_idx - 1] < model.state[FEES0_KEY][1, tick_idx - 1]


class TestFeeCalculation:
    """Fees match the Uniswap-v3 full-tick formulas."""

    def test_sell_fee_matches_formula(self):
        """Sell fee = fee_multiplier * L[i-1] * (1/AMM[i-1] - 1/AMM[i])."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global
        prev_idx = tick_idx - 1

        L_prev = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, prev_idx]
        sqrt_p_i = model.sqrt_grid[tick_idx]
        sqrt_p_prev = model.sqrt_grid[prev_idx]

        dx = L_prev * (1.0 / sqrt_p_prev - 1.0 / sqrt_p_i)
        expected_total_fee = model.fee_multiplier * dx

        model.update_state(np.array([[True, False]]), None)

        total_fees0 = model.state[FEES0_KEY][0].sum()
        assert np.isclose(total_fees0, expected_total_fee, rtol=1e-10)

    def test_buy_fee_matches_formula(self):
        """Buy fee = fee_multiplier * L[i] * (AMM[i+1] - AMM[i])."""
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e6)

        current_tick = int(model.state[POOL_CURRENT_TICK_KEY][0])
        tick_idx = current_tick - model.tick_lower_global

        L_i = model.state[POOL_LIQUIDITY_ARRAY_KEY][0, tick_idx]
        sqrt_p_i = model.sqrt_grid[tick_idx]
        sqrt_p_next = model.sqrt_grid[tick_idx + 1]

        dy = L_i * (sqrt_p_next - sqrt_p_i)
        expected_total_fee = model.fee_multiplier * dy

        model.update_state(np.array([[False, True]]), None)

        total_fees1 = model.state[FEES1_KEY][0].sum()
        assert np.isclose(total_fees1, expected_total_fee, rtol=1e-10)

    def test_fees_scale_with_liquidity(self):
        model_high = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_high, liquidity_value=1e8)
        model_high.update_state(np.array([[1, 0]]), None)
        fees_high = model_high.state[FEES0_KEY][0].sum()

        model_low = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_low, liquidity_value=1e4)
        model_low.update_state(np.array([[1, 0]]), None)
        fees_low = model_low.state[FEES0_KEY][0].sum()

        assert fees_low < fees_high


class TestSequentialProcessing:
    """Sell and buy run sequentially with randomized order on [True, True]."""

    def test_both_arrivals_collect_both_fees(self):
        model = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model, liquidity_value=1e8)

        model.update_state(np.array([[True, True]]), None)

        assert model.state[FEES0_KEY][0].sum() > 0
        assert model.state[FEES1_KEY][0].sum() > 0

    def test_single_direction_fees_one_sided(self):
        model_sell = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_sell, liquidity_value=1e8)
        model_sell.update_state(np.array([[True, False]]), None)

        assert model_sell.state[FEES0_KEY][0].sum() > 0
        assert model_sell.state[FEES1_KEY][0].sum() == 0

        model_buy = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_buy, liquidity_value=1e8)
        model_buy.update_state(np.array([[False, True]]), None)

        assert model_buy.state[FEES0_KEY][0].sum() == 0
        assert model_buy.state[FEES1_KEY][0].sum() > 0


class TestBernoulliOrdering:
    """Randomized ordering of simultaneous arrivals."""

    def test_randomization_reproducible_with_seed(self):
        model_a = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_a, liquidity_value=1e6)
        model_b = create_test_model(num_trajectories=1, num_ticks=100)
        initialize_state(model_b, liquidity_value=1e6)

        arrivals = np.array([[True, True]])
        model_a.update_state(arrivals, None)
        model_b.update_state(arrivals, None)

        assert model_a.state[POOL_CURRENT_TICK_KEY][0] == model_b.state[POOL_CURRENT_TICK_KEY][0]
        assert np.isclose(model_a.state[FEES0_KEY][0].sum(), model_b.state[FEES0_KEY][0].sum())
        assert np.isclose(model_a.state[FEES1_KEY][0].sum(), model_b.state[FEES1_KEY][0].sum())

    def test_both_orderings_occur_without_seed(self):
        """Across many runs without a seed, both sell-first and buy-first should occur."""
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
            # Manually initialize lattice since no midprice_model
            m.tick_lower_global = 46054 - 50  # any valid anchor
            m._build_sqrt_grid()
            initial_tick = 46054
            num_traj = 1
            num_ticks = 100
            m.state = {
                POOL_SQRT_PRICE_KEY: np.full(num_traj, m.sqrt_grid[initial_tick - m.tick_lower_global]),
                POOL_CURRENT_TICK_KEY: np.full(num_traj, initial_tick, dtype=np.int64),
                POOL_LIQUIDITY_ARRAY_KEY: np.full((num_traj, num_ticks), 1e6),
                FEES0_KEY: np.zeros((num_traj, num_ticks)),
                FEES1_KEY: np.zeros((num_traj, num_ticks)),
                LP_LIQUIDITY_KEY: np.zeros(num_traj),
                LP_TICK_LOWER_KEY: np.full(num_traj, float(initial_tick - 5)),
                LP_TICK_UPPER_KEY: np.full(num_traj, float(initial_tick + 5)),
                LP_COLLECTED_FEES0_KEY: np.zeros(num_traj),
                LP_COLLECTED_FEES1_KEY: np.zeros(num_traj),
                LP_UNCLAIMED_FEES0_KEY: np.zeros(num_traj),
                LP_UNCLAIMED_FEES1_KEY: np.zeros(num_traj),
                LP_FEE_SNAPSHOT0_KEY: np.zeros(num_traj),
                LP_FEE_SNAPSHOT1_KEY: np.zeros(num_traj),
                ASSET_PRICE_KEY: np.full(num_traj, 100.0),
                TIME_KEY: np.zeros(num_traj),
                LP_EVER_DEPLOYED_KEY: np.zeros(num_traj, dtype=bool),
            }
            m.update_state(np.array([[True, True]]), None)
            fee0_seen = fee0_seen or m.state[FEES0_KEY][0].sum() > 0
            fee1_seen = fee1_seen or m.state[FEES1_KEY][0].sum() > 0
            if fee0_seen and fee1_seen:
                break

        assert fee0_seen
        assert fee1_seen


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
