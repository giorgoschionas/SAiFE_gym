import numpy as np
import pytest

from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    FEES0_KEY,
    FEES1_KEY,
    LP_COLLECTED_FEES0_KEY,
    LP_COLLECTED_FEES1_KEY,
    LP_EVER_DEPLOYED_KEY,
    LP_FEE_SNAPSHOT0_KEY,
    LP_FEE_SNAPSHOT1_KEY,
    LP_LIQUIDITY_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_UNCLAIMED_FEES0_KEY,
    LP_UNCLAIMED_FEES1_KEY,
    INITIAL_WEALTH_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
)
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.fee_accounting_models import UniswapV3FeeAccounting
from SAiFE_gym.stochastic_processes.price_impact_models import (
    LiquidityDepthUniswapV3PriceImpact,
    OneTickUniswapV3PriceImpact,
    SwapResult,
)


EXPONENTIAL_VALUE = 1.0001


def _make_state(
    num_trajectories=1,
    num_ticks=10,
    tick_lower_global=100,
    current_tick=105,
    asset_price=100.0,
):
    sqrt_grid = np.sqrt(
        EXPONENTIAL_VALUE ** (tick_lower_global + np.arange(num_ticks + 1, dtype=np.float64))
    )
    current_tick = np.full(num_trajectories, current_tick, dtype=np.float64)
    return {
        POOL_CURRENT_TICK_KEY: current_tick,
        POOL_SQRT_PRICE_KEY: sqrt_grid[current_tick.astype(np.int64) - tick_lower_global],
        POOL_LIQUIDITY_ARRAY_KEY: np.full((num_trajectories, num_ticks), 1e6, dtype=np.float64),
        FEES0_KEY: np.zeros((num_trajectories, num_ticks), dtype=np.float64),
        FEES1_KEY: np.zeros((num_trajectories, num_ticks), dtype=np.float64),
        ASSET_PRICE_KEY: np.full(num_trajectories, asset_price, dtype=np.float64),
    }, sqrt_grid


def _buy_capacity(state, sqrt_grid, trajectory, fee_idx):
    liquidity = state[POOL_LIQUIDITY_ARRAY_KEY][trajectory, fee_idx]
    return liquidity * (sqrt_grid[fee_idx + 1] - sqrt_grid[fee_idx])


def _sell_capacity(state, sqrt_grid, trajectory, fee_idx):
    liquidity = state[POOL_LIQUIDITY_ARRAY_KEY][trajectory, fee_idx]
    return liquidity * (1.0 / sqrt_grid[fee_idx] - 1.0 / sqrt_grid[fee_idx + 1])


def _gross_input(curve_input, fee_multiplier):
    return curve_input * (1.0 + fee_multiplier)


def _capacity_weighted_amounts(capacities, curve_input):
    capacities = np.asarray(capacities, dtype=np.float64)
    total = capacities.sum()
    if total <= 0.0:
        return np.full(capacities.shape, curve_input / capacities.size)
    return curve_input * capacities / total


def _fixed_sampler(values):
    values = np.asarray(values, dtype=np.float64)

    def sampler(rng, size):
        if values.size == 1:
            return np.full(size, values.item(), dtype=np.float64)
        assert values.shape == (size,)
        return values.copy()

    return sampler


class TestOneTickUniswapV3PriceImpact:
    def test_inactive_mask_leaves_state_unchanged(self):
        state, sqrt_grid = _make_state(num_trajectories=2)
        before = {key: value.copy() for key, value in state.items()}
        model = OneTickUniswapV3PriceImpact(num_trajectories=2)

        model.process_swap(
            state,
            np.array([False, False]),
            -1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)

    def test_sell_moves_down_one_tick_and_returns_swap_result(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        model = OneTickUniswapV3PriceImpact()

        result = model.process_swap(
            state,
            np.array([True]),
            -1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        idx = 105 - 100
        fee_idx = idx - 1
        expected_amount = 1e6 * (1.0 / sqrt_grid[idx - 1] - 1.0 / sqrt_grid[idx])
        assert state[POOL_CURRENT_TICK_KEY][0] == 104
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[4]
        assert result.fee_key == FEES0_KEY
        assert result.direction == -1
        np.testing.assert_array_equal(result.trajectories, np.array([0]))
        np.testing.assert_array_equal(result.fee_indices, np.array([fee_idx]))
        np.testing.assert_allclose(result.amounts, np.array([expected_amount]))
        assert np.sum(state[FEES0_KEY]) == 0.0
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_buy_moves_up_one_tick_and_returns_swap_result(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        model = OneTickUniswapV3PriceImpact()

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        idx = 105 - 100
        expected_amount = 1e6 * (sqrt_grid[idx + 1] - sqrt_grid[idx])
        assert state[POOL_CURRENT_TICK_KEY][0] == 106
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[6]
        assert result.fee_key == FEES1_KEY
        assert result.direction == 1
        np.testing.assert_array_equal(result.trajectories, np.array([0]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx]))
        np.testing.assert_allclose(result.amounts, np.array([expected_amount]))
        assert np.sum(state[FEES0_KEY]) == 0.0
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_invalid_direction_raises(self):
        state, sqrt_grid = _make_state()
        model = OneTickUniswapV3PriceImpact()

        with pytest.raises(ValueError, match="direction must be -1 for sell or 1 for buy"):
            model.process_swap(
                state,
                np.array([True]),
                0,
                tick_lower_global=100,
                sqrt_grid=sqrt_grid,
                num_ticks=10,
            )

    def test_boundary_check_raises_for_sell_outside_window(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=100)
        model = OneTickUniswapV3PriceImpact()

        with pytest.raises(AssertionError, match="tick out of array window"):
            model.process_swap(
                state,
                np.array([True]),
                -1,
                tick_lower_global=100,
                sqrt_grid=sqrt_grid,
                num_ticks=10,
            )

    def test_boundary_check_raises_for_buy_outside_window(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=110)
        model = OneTickUniswapV3PriceImpact()

        with pytest.raises(AssertionError, match="tick out of array window"):
            model.process_swap(
                state,
                np.array([True]),
                1,
                tick_lower_global=100,
                sqrt_grid=sqrt_grid,
                num_ticks=10,
            )


class TestLiquidityDepthUniswapV3PriceImpact:
    def test_explicit_positive_order_moves_up_and_returns_token1_fee_result(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        amount = _buy_capacity(state, sqrt_grid, 0, idx)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(0.0),
        )

        result = model.process_order_sizes(
            state,
            np.array([[amount]]),
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 106
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[6]
        assert result.tick_movement[0] == 1
        assert result.executed_input[0] == pytest.approx(amount)
        assert result.unfilled_input[0] == pytest.approx(0.0)
        assert result.curve_input[0] == pytest.approx(amount)
        assert len(result.swap_results) == 1
        swap_result = result.swap_results[0]
        assert swap_result.fee_key == FEES1_KEY
        assert swap_result.direction == 1
        np.testing.assert_array_equal(swap_result.trajectories, np.array([0]))
        np.testing.assert_array_equal(swap_result.fee_indices, np.array([idx]))
        np.testing.assert_allclose(swap_result.amounts, np.array([amount]))

    def test_explicit_negative_order_moves_down_and_returns_token0_fee_result(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        amount = _sell_capacity(state, sqrt_grid, 0, idx - 1)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(0.0),
        )

        result = model.process_order_sizes(
            state,
            np.array([[-amount]]),
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 104
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[4]
        assert result.tick_movement[0] == -1
        assert result.executed_input[0] == pytest.approx(-amount)
        assert result.unfilled_input[0] == pytest.approx(0.0)
        assert result.curve_input[0] == pytest.approx(-amount)
        assert len(result.swap_results) == 1
        swap_result = result.swap_results[0]
        assert swap_result.fee_key == FEES0_KEY
        assert swap_result.direction == -1
        np.testing.assert_array_equal(swap_result.trajectories, np.array([0]))
        np.testing.assert_array_equal(swap_result.fee_indices, np.array([idx - 1]))
        np.testing.assert_allclose(swap_result.amounts, np.array([amount]))

    def test_explicit_zero_order_leaves_state_unchanged(self):
        state, sqrt_grid = _make_state()
        before = {key: value.copy() for key, value in state.items()}
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(100.0),
        )

        result = model.process_order_sizes(
            state,
            np.array([[0.0]]),
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)
        assert result.executed_input[0] == 0.0
        assert result.unfilled_input[0] == 0.0
        assert result.curve_input[0] == 0.0
        assert result.tick_movement[0] == 0
        assert result.swap_results == ()

    def test_explicit_vectorized_positive_negative_and_zero_orders(self):
        state, sqrt_grid = _make_state(
            num_trajectories=3,
            tick_lower_global=100,
            current_tick=105,
        )
        idx = 105 - 100
        buy_amount = _buy_capacity(state, sqrt_grid, 0, idx)
        sell_amount = _sell_capacity(state, sqrt_grid, 1, idx - 1)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(0.0),
            num_trajectories=3,
        )

        result = model.process_order_sizes(
            state,
            np.array([[buy_amount], [-sell_amount], [0.0]]),
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        np.testing.assert_array_equal(state[POOL_CURRENT_TICK_KEY], np.array([106, 104, 105]))
        np.testing.assert_array_equal(result.tick_movement, np.array([1, -1, 0]))
        np.testing.assert_allclose(result.executed_input, np.array([buy_amount, -sell_amount, 0.0]))
        np.testing.assert_allclose(result.unfilled_input, np.zeros(3))
        np.testing.assert_allclose(result.curve_input, np.array([buy_amount, -sell_amount, 0.0]))
        assert len(result.swap_results) == 2
        assert result.swap_results[0].fee_key == FEES1_KEY
        assert result.swap_results[1].fee_key == FEES0_KEY

    def test_explicit_oversized_order_stops_at_boundary_and_reports_unfilled(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=108)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(0.0),
        )

        result = model.process_order_sizes(
            state,
            np.array([[1e12]]),
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 110
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[10]
        assert result.tick_movement[0] == 2
        assert result.executed_input[0] > 0.0
        assert result.unfilled_input[0] > 0.0

    def test_explicit_positive_order_fees_are_paid_from_gross_budget(self):
        fee_multiplier = 0.25
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        curve_input = _buy_capacity(state, sqrt_grid, 0, idx)
        gross_input = _gross_input(curve_input, fee_multiplier)
        expected_token0_out = state[POOL_LIQUIDITY_ARRAY_KEY][0, idx] * (
            1.0 / sqrt_grid[idx] - 1.0 / sqrt_grid[idx + 1]
        )
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(0.0),
        )

        too_small = model.process_order_sizes(
            state,
            np.array([[curve_input]]),
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 105
        assert too_small.tick_movement[0] == 0
        assert too_small.executed_input[0] == pytest.approx(0.0)
        assert too_small.unfilled_input[0] == pytest.approx(curve_input)
        assert too_small.curve_input[0] == pytest.approx(0.0)

        result = model.process_order_sizes(
            state,
            np.array([[gross_input]]),
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 106
        assert result.tick_movement[0] == 1
        assert result.executed_input[0] == pytest.approx(gross_input)
        assert result.unfilled_input[0] == pytest.approx(0.0)
        assert result.curve_input[0] == pytest.approx(curve_input)
        assert result.fee_input[0] == pytest.approx(fee_multiplier * curve_input)
        assert result.token0_delta[0] == pytest.approx(expected_token0_out)
        assert result.token1_delta[0] == pytest.approx(-gross_input)
        np.testing.assert_allclose(result.swap_results[0].amounts, np.array([curve_input]))

    def test_explicit_negative_order_fees_are_paid_from_gross_budget(self):
        fee_multiplier = 0.25
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        curve_input = _sell_capacity(state, sqrt_grid, 0, idx - 1)
        gross_input = _gross_input(curve_input, fee_multiplier)
        expected_token1_out = state[POOL_LIQUIDITY_ARRAY_KEY][0, idx - 1] * (
            sqrt_grid[idx] - sqrt_grid[idx - 1]
        )
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(0.0),
        )

        too_small = model.process_order_sizes(
            state,
            np.array([[-curve_input]]),
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 105
        assert too_small.tick_movement[0] == 0
        assert too_small.executed_input[0] == pytest.approx(0.0)
        assert too_small.unfilled_input[0] == pytest.approx(-curve_input)
        assert too_small.curve_input[0] == pytest.approx(0.0)

        result = model.process_order_sizes(
            state,
            np.array([[-gross_input]]),
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 104
        assert result.tick_movement[0] == -1
        assert result.executed_input[0] == pytest.approx(-gross_input)
        assert result.unfilled_input[0] == pytest.approx(0.0)
        assert result.curve_input[0] == pytest.approx(-curve_input)
        assert result.fee_input[0] == pytest.approx(fee_multiplier * curve_input)
        assert result.token0_delta[0] == pytest.approx(-gross_input)
        assert result.token1_delta[0] == pytest.approx(expected_token1_out)
        np.testing.assert_allclose(result.swap_results[0].amounts, np.array([curve_input]))

    def test_explicit_order_sizes_do_not_call_trade_size_sampler(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        amount = _buy_capacity(state, sqrt_grid, 0, 105 - 100)

        def raising_sampler(rng, size):
            raise AssertionError("trade_size_sampler should not be used")

        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=raising_sampler,
        )

        result = model.process_order_sizes(
            state,
            np.array([[amount]]),
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert result.executed_input[0] == pytest.approx(amount)

    def test_inactive_mask_leaves_state_unchanged(self):
        state, sqrt_grid = _make_state(num_trajectories=2)
        before = {key: value.copy() for key, value in state.items()}
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(100.0),
            num_trajectories=2,
        )

        result = model.process_swap(
            state,
            np.array([False, False]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert result is None
        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)

    def test_buy_moves_exact_integer_ticks_and_returns_crossed_fee_entries(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        capacities = np.array([
            _buy_capacity(state, sqrt_grid, 0, idx),
            _buy_capacity(state, sqrt_grid, 0, idx + 1),
        ])
        average_depth = np.mean(capacities)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(2.0 * average_depth),
            depth_window=2,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 107
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[7]
        assert result.fee_key == FEES1_KEY
        assert result.direction == 1
        np.testing.assert_array_equal(result.trajectories, np.array([0, 0]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx + 1]))
        np.testing.assert_allclose(
            result.amounts,
            _capacity_weighted_amounts(capacities, 2.0 * average_depth),
        )
        assert np.sum(state[FEES0_KEY]) == 0.0
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_sell_moves_exact_integer_ticks_and_returns_crossed_fee_entries(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        capacities = np.array([
            _sell_capacity(state, sqrt_grid, 0, idx - 1),
            _sell_capacity(state, sqrt_grid, 0, idx - 2),
        ])
        average_depth = np.mean(capacities)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(2.0 * average_depth),
            depth_window=2,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            -1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 103
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[3]
        assert result.fee_key == FEES0_KEY
        assert result.direction == -1
        np.testing.assert_array_equal(result.trajectories, np.array([0, 0]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx - 1, idx - 2]))
        np.testing.assert_allclose(
            result.amounts,
            _capacity_weighted_amounts(capacities, 2.0 * average_depth),
        )
        assert np.sum(state[FEES0_KEY]) == 0.0
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_buy_multi_tick_fee_allocation_is_capacity_weighted(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        first_capacity = _buy_capacity(state, sqrt_grid, 0, idx)
        state[POOL_LIQUIDITY_ARRAY_KEY][0, idx + 1] *= (
            3.0 * first_capacity / _buy_capacity(state, sqrt_grid, 0, idx + 1)
        )
        capacities = np.array([
            _buy_capacity(state, sqrt_grid, 0, idx),
            _buy_capacity(state, sqrt_grid, 0, idx + 1),
        ])
        curve_input = capacities.sum()
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(curve_input),
            depth_window=2,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 107
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx + 1]))
        np.testing.assert_allclose(result.amounts, np.array([0.25, 0.75]) * curve_input)

    def test_sell_multi_tick_fee_allocation_is_capacity_weighted(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        first_capacity = _sell_capacity(state, sqrt_grid, 0, idx - 1)
        state[POOL_LIQUIDITY_ARRAY_KEY][0, idx - 2] *= (
            3.0 * first_capacity / _sell_capacity(state, sqrt_grid, 0, idx - 2)
        )
        capacities = np.array([
            _sell_capacity(state, sqrt_grid, 0, idx - 1),
            _sell_capacity(state, sqrt_grid, 0, idx - 2),
        ])
        curve_input = capacities.sum()
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(curve_input),
            depth_window=2,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            -1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 103
        np.testing.assert_array_equal(result.fee_indices, np.array([idx - 1, idx - 2]))
        np.testing.assert_allclose(result.amounts, np.array([0.25, 0.75]) * curve_input)

    def test_fractional_eta_uses_seeded_stochastic_rounding(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        depth = _buy_capacity(state, sqrt_grid, 0, idx)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(1.999 * depth),
            depth_window=1,
            seed=7,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 107
        capacities = np.array([
            _buy_capacity(state, sqrt_grid, 0, idx),
            _buy_capacity(state, sqrt_grid, 0, idx + 1),
        ])
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx + 1]))
        np.testing.assert_allclose(
            result.amounts,
            _capacity_weighted_amounts(capacities, 1.999 * depth),
        )

    def test_multi_trajectory_active_mask_and_liquidity_depths_are_vectorized(self):
        state, sqrt_grid = _make_state(num_trajectories=3, tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        state[POOL_LIQUIDITY_ARRAY_KEY][2] *= 2.0
        depth0 = _buy_capacity(state, sqrt_grid, 0, idx)
        depth2 = _buy_capacity(state, sqrt_grid, 2, idx)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(np.array([depth0, 2.0 * depth2])),
            depth_window=1,
            num_trajectories=3,
        )

        result = model.process_swap(
            state,
            np.array([True, False, True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        np.testing.assert_array_equal(state[POOL_CURRENT_TICK_KEY], np.array([106, 105, 107]))
        capacities2 = np.array([
            _buy_capacity(state, sqrt_grid, 2, idx),
            _buy_capacity(state, sqrt_grid, 2, idx + 1),
        ])
        np.testing.assert_array_equal(result.trajectories, np.array([0, 2, 2]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx, idx + 1]))
        np.testing.assert_allclose(
            result.amounts,
            np.concatenate(
                [
                    np.array([depth0]),
                    _capacity_weighted_amounts(capacities2, 2.0 * depth2),
                ]
            ),
        )

    def test_tick_move_is_clipped_at_array_boundary(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=108)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(1e18),
            depth_window=3,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 110
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[10]
        capacities = np.array([
            _buy_capacity(state, sqrt_grid, 0, 8),
            _buy_capacity(state, sqrt_grid, 0, 9),
        ])
        np.testing.assert_array_equal(result.fee_indices, np.array([8, 9]))
        np.testing.assert_allclose(
            result.amounts,
            _capacity_weighted_amounts(capacities, 1e18),
        )

    @pytest.mark.parametrize("direction", [1, -1], ids=["buy", "sell"])
    @pytest.mark.parametrize("zero_liquidity", [False, True], ids=["nonuniform", "zero"])
    def test_mixed_trajectories_at_array_boundary(self, direction, zero_liquidity):
        num_ticks = 10
        tick_lower_global = 100
        fee_multiplier = 0.25
        state, sqrt_grid = _make_state(num_trajectories=5)
        # Trajectories 0 and 2 move one and five ticks, respectively; the others
        # are inactive, have a zero-sized order, or are already at the boundary.
        indices = np.array([9, 7, 5, 9, 10])
        if direction == -1:
            indices = num_ticks - indices
        state[POOL_CURRENT_TICK_KEY] = tick_lower_global + indices
        state[POOL_SQRT_PRICE_KEY] = sqrt_grid[indices]
        state[POOL_LIQUIDITY_ARRAY_KEY] = (
            1e6 * np.arange(1, 6)[:, None] * np.arange(1, num_ticks + 1)[None, :]
        )
        if zero_liquidity:
            state[POOL_LIQUIDITY_ARRAY_KEY].fill(0.0)
        liquidity_before = state[POOL_LIQUIDITY_ARRAY_KEY].copy()
        gross_inputs = np.array([1e9, 2e9, 0.0, 3e9])
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(gross_inputs),
            depth_window=3,
            num_trajectories=5,
            seed=7,
        )

        result = model.process_swap(
            state,
            np.array([True, False, True, True, True]),
            direction,
            tick_lower_global=tick_lower_global,
            sqrt_grid=sqrt_grid,
            num_ticks=num_ticks,
            fee_multiplier=fee_multiplier,
        )

        expected_indices = indices + direction * np.array([1, 0, 5, 0, 0])
        np.testing.assert_array_equal(
            state[POOL_CURRENT_TICK_KEY], tick_lower_global + expected_indices
        )
        np.testing.assert_array_equal(state[POOL_SQRT_PRICE_KEY], sqrt_grid[expected_indices])
        np.testing.assert_array_equal(state[POOL_LIQUIDITY_ARRAY_KEY], liquidity_before)
        np.testing.assert_array_equal(state[FEES0_KEY], 0.0)
        np.testing.assert_array_equal(state[FEES1_KEY], 0.0)

        expected_traj = np.array([0, 2, 2, 2, 2, 2])
        if direction == 1:
            expected_fee_indices = np.array([9, 5, 6, 7, 8, 9])
            capacities = _buy_capacity(state, sqrt_grid, 2, expected_fee_indices[1:])
            fee_key, other_fee_key = FEES1_KEY, FEES0_KEY
        else:
            expected_fee_indices = np.array([0, 4, 3, 2, 1, 0])
            capacities = _sell_capacity(state, sqrt_grid, 2, expected_fee_indices[1:])
            fee_key, other_fee_key = FEES0_KEY, FEES1_KEY
        curve_inputs = gross_inputs[:2] / (1.0 + fee_multiplier)
        expected_amounts = np.concatenate([
            curve_inputs[:1],
            _capacity_weighted_amounts(capacities, curve_inputs[1]),
        ])
        assert result.fee_key == fee_key
        assert result.direction == direction
        np.testing.assert_array_equal(result.trajectories, expected_traj)
        np.testing.assert_array_equal(result.fee_indices, expected_fee_indices)
        np.testing.assert_allclose(result.amounts, expected_amounts)
        np.testing.assert_allclose(
            np.bincount(result.trajectories, weights=result.amounts, minlength=5),
            np.array([curve_inputs[0], 0.0, curve_inputs[1], 0.0, 0.0]),
        )

        UniswapV3FeeAccounting().apply_fees(state, result, fee_multiplier)
        expected_fees = np.zeros_like(state[fee_key])
        expected_fees[expected_traj, expected_fee_indices] = fee_multiplier * expected_amounts
        np.testing.assert_allclose(state[fee_key], expected_fees)
        np.testing.assert_array_equal(state[other_fee_key], 0.0)

    def test_zero_realized_move_returns_none_and_leaves_state_unchanged(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        before = {key: value.copy() for key, value in state.items()}
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(0.0),
            depth_window=1,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert result is None
        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)

    def test_positive_buy_zero_realized_move_returns_fee_basis_at_first_interval(self):
        fee_tier = 0.003
        fee_multiplier = fee_tier / (1.0 - fee_tier)
        gross_input = 10.0
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        before_tick = state[POOL_CURRENT_TICK_KEY].copy()
        before_price = state[POOL_SQRT_PRICE_KEY].copy()
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(gross_input),
            depth_window=1,
            seed=7,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )
        UniswapV3FeeAccounting().apply_fees(state, result, fee_multiplier)

        np.testing.assert_array_equal(state[POOL_CURRENT_TICK_KEY], before_tick)
        np.testing.assert_array_equal(state[POOL_SQRT_PRICE_KEY], before_price)
        assert result.fee_key == FEES1_KEY
        np.testing.assert_array_equal(result.trajectories, np.array([0]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx]))
        np.testing.assert_allclose(result.amounts, np.array([gross_input / (1.0 + fee_multiplier)]))
        assert np.sum(state[FEES0_KEY]) == 0.0
        assert np.sum(state[FEES1_KEY]) == pytest.approx(fee_tier * gross_input)

    def test_positive_sell_zero_realized_move_returns_fee_basis_at_first_interval(self):
        fee_tier = 0.003
        fee_multiplier = fee_tier / (1.0 - fee_tier)
        gross_token1_notional = 40.0
        external_midprice = 100.0
        state, sqrt_grid = _make_state(
            tick_lower_global=100,
            current_tick=105,
            asset_price=external_midprice,
        )
        idx = 105 - 100
        before_tick = state[POOL_CURRENT_TICK_KEY].copy()
        before_price = state[POOL_SQRT_PRICE_KEY].copy()
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(gross_token1_notional),
            depth_window=1,
            seed=7,
            trade_size_unit="token1_notional",
        )

        result = model.process_swap(
            state,
            np.array([True]),
            -1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )
        UniswapV3FeeAccounting().apply_fees(state, result, fee_multiplier)

        gross_token0_input = gross_token1_notional / external_midprice
        np.testing.assert_array_equal(state[POOL_CURRENT_TICK_KEY], before_tick)
        np.testing.assert_array_equal(state[POOL_SQRT_PRICE_KEY], before_price)
        assert result.fee_key == FEES0_KEY
        np.testing.assert_array_equal(result.trajectories, np.array([0]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx - 1]))
        np.testing.assert_allclose(result.amounts, np.array([gross_token0_input / (1.0 + fee_multiplier)]))
        assert np.sum(state[FEES0_KEY]) == pytest.approx(fee_tier * gross_token0_input)
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_sampled_price_impact_uses_curve_input_after_fee_deduction(self):
        fee_multiplier = 1.0
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        depth = _buy_capacity(state, sqrt_grid, 0, idx)
        before_tick = state[POOL_CURRENT_TICK_KEY].copy()
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(0.75 * depth),
            depth_window=1,
            seed=7,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )

        np.testing.assert_array_equal(state[POOL_CURRENT_TICK_KEY], before_tick)
        np.testing.assert_array_equal(result.fee_indices, np.array([idx]))
        np.testing.assert_allclose(result.amounts, np.array([0.375 * depth]))

    def test_sampled_total_fee_is_gross_fee_for_zero_one_and_multi_tick_jumps(self):
        fee_tier = 0.003
        fee_multiplier = fee_tier / (1.0 - fee_tier)
        state, sqrt_grid = _make_state(
            num_trajectories=3,
            tick_lower_global=100,
            current_tick=105,
        )
        idx = 105 - 100
        depth = _buy_capacity(state, sqrt_grid, 0, idx)
        gross_inputs = np.array([0.1 * depth, 1.2 * depth, 2.2 * depth])
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(gross_inputs),
            depth_window=1,
            num_trajectories=3,
            seed=7,
        )

        result = model.process_swap(
            state,
            np.array([True, True, True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )
        UniswapV3FeeAccounting().apply_fees(state, result, fee_multiplier)

        np.testing.assert_array_equal(result.trajectories, np.array([0, 1, 2, 2]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx, idx, idx + 1]))
        np.testing.assert_allclose(
            state[FEES1_KEY].sum(axis=1),
            fee_tier * gross_inputs,
        )
        np.testing.assert_allclose(
            np.array([
                result.amounts[result.trajectories == i].sum()
                for i in range(3)
            ]),
            gross_inputs / (1.0 + fee_multiplier),
        )
        assert np.sum(state[FEES0_KEY]) == 0.0

    def test_concentrated_liquidity_does_not_inflate_sampled_fee_entries(self):
        fee_tier = 0.003
        fee_multiplier = fee_tier / (1.0 - fee_tier)
        gross_input = 500.0
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        state[POOL_LIQUIDITY_ARRAY_KEY][0, idx + 1] *= 100.0
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(gross_input),
            depth_window=2,
            seed=7,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )
        UniswapV3FeeAccounting().apply_fees(state, result, fee_multiplier)

        np.testing.assert_array_equal(result.trajectories, np.array([0]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx]))
        np.testing.assert_allclose(result.amounts, np.array([gross_input / (1.0 + fee_multiplier)]))
        assert np.sum(state[FEES1_KEY]) == pytest.approx(fee_tier * gross_input)

    def test_vectorized_mixed_sampled_batch_includes_only_positive_executable_arrivals(self):
        fee_multiplier = 0.25
        state, sqrt_grid = _make_state(
            num_trajectories=5,
            tick_lower_global=100,
            current_tick=105,
        )
        state[POOL_CURRENT_TICK_KEY][4] = 110
        state[POOL_SQRT_PRICE_KEY][4] = sqrt_grid[10]
        idx = 105 - 100
        depth = _buy_capacity(state, sqrt_grid, 0, idx)
        gross_inputs_for_active = np.array([0.0, 0.1 * depth, 1.2 * depth, 1e9])
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(gross_inputs_for_active),
            depth_window=1,
            num_trajectories=5,
            seed=7,
        )

        result = model.process_swap(
            state,
            np.array([True, True, True, False, True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )

        np.testing.assert_array_equal(state[POOL_CURRENT_TICK_KEY], np.array([105, 105, 106, 105, 110]))
        np.testing.assert_array_equal(result.trajectories, np.array([1, 2]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx]))
        np.testing.assert_allclose(
            result.amounts,
            gross_inputs_for_active[1:3] / (1.0 + fee_multiplier),
        )

    def test_lp_accrues_fee_from_zero_tick_sampled_arrival(self):
        fee_tier = 0.003
        fee_multiplier = fee_tier / (1.0 - fee_tier)
        gross_input = 10.0
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        state.update(
            {
                LP_LIQUIDITY_KEY: np.array([1e6], dtype=np.float64),
                LP_TICK_LOWER_KEY: np.array([105], dtype=np.float64),
                LP_TICK_UPPER_KEY: np.array([106], dtype=np.float64),
                LP_COLLECTED_FEES0_KEY: np.zeros(1, dtype=np.float64),
                LP_COLLECTED_FEES1_KEY: np.zeros(1, dtype=np.float64),
                LP_UNCLAIMED_FEES0_KEY: np.zeros(1, dtype=np.float64),
                LP_UNCLAIMED_FEES1_KEY: np.zeros(1, dtype=np.float64),
                LP_FEE_SNAPSHOT0_KEY: np.zeros(1, dtype=np.float64),
                LP_FEE_SNAPSHOT1_KEY: np.zeros(1, dtype=np.float64),
                LP_EVER_DEPLOYED_KEY: np.array([True]),
                INITIAL_WEALTH_KEY: np.array([0.0], dtype=np.float64),
            }
        )
        model = UniswapV3ModelDynamics(
            price_impact_model=LiquidityDepthUniswapV3PriceImpact(
                trade_size_sampler=_fixed_sampler(gross_input),
                depth_window=1,
                seed=7,
            ),
            num_trajectories=1,
            fee_tier=fee_tier,
            tau=5,
            num_ticks=10,
        )
        model.state = state
        model.tick_lower_global = 100
        model.sqrt_grid = sqrt_grid

        model._process_swap(np.array([True]), 1)
        model._accrue_lp_fees()

        assert state[POOL_CURRENT_TICK_KEY][0] == 105
        assert state[FEES1_KEY][0, idx] == pytest.approx(fee_tier * gross_input)
        assert state[LP_UNCLAIMED_FEES1_KEY][0] == pytest.approx(fee_tier * gross_input)
        assert state[LP_COLLECTED_FEES1_KEY][0] == pytest.approx(fee_tier * gross_input)

    def test_lp_active_only_in_later_crossed_interval_accrues_weighted_fee_share(self):
        fee_tier = 0.003
        fee_multiplier = fee_tier / (1.0 - fee_tier)
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        lp_liquidity = state[POOL_LIQUIDITY_ARRAY_KEY][0, idx + 1]
        capacities = np.array([
            _buy_capacity(state, sqrt_grid, 0, idx),
            _buy_capacity(state, sqrt_grid, 0, idx + 1),
        ])
        curve_input = capacities.sum()
        gross_input = curve_input * (1.0 + fee_multiplier)
        expected_later_fee = fee_multiplier * capacities[1]
        state.update(
            {
                LP_LIQUIDITY_KEY: np.array([lp_liquidity], dtype=np.float64),
                LP_TICK_LOWER_KEY: np.array([106], dtype=np.float64),
                LP_TICK_UPPER_KEY: np.array([107], dtype=np.float64),
                LP_COLLECTED_FEES0_KEY: np.zeros(1, dtype=np.float64),
                LP_COLLECTED_FEES1_KEY: np.zeros(1, dtype=np.float64),
                LP_UNCLAIMED_FEES0_KEY: np.zeros(1, dtype=np.float64),
                LP_UNCLAIMED_FEES1_KEY: np.zeros(1, dtype=np.float64),
                LP_FEE_SNAPSHOT0_KEY: np.zeros(1, dtype=np.float64),
                LP_FEE_SNAPSHOT1_KEY: np.zeros(1, dtype=np.float64),
                LP_EVER_DEPLOYED_KEY: np.array([True]),
                INITIAL_WEALTH_KEY: np.array([0.0], dtype=np.float64),
            }
        )
        model = UniswapV3ModelDynamics(
            price_impact_model=LiquidityDepthUniswapV3PriceImpact(
                trade_size_sampler=_fixed_sampler(gross_input),
                depth_window=2,
            ),
            num_trajectories=1,
            fee_tier=fee_tier,
            tau=5,
            num_ticks=10,
        )
        model.state = state
        model.tick_lower_global = 100
        model.sqrt_grid = sqrt_grid

        model._process_swap(np.array([True]), 1)
        model._accrue_lp_fees()

        assert state[POOL_CURRENT_TICK_KEY][0] == 107
        assert state[FEES1_KEY][0, idx] == pytest.approx(fee_multiplier * capacities[0])
        assert state[FEES1_KEY][0, idx + 1] == pytest.approx(expected_later_fee)
        assert state[LP_UNCLAIMED_FEES1_KEY][0] == pytest.approx(expected_later_fee)
        assert state[LP_COLLECTED_FEES1_KEY][0] == pytest.approx(expected_later_fee)

    def test_invalid_direction_raises(self):
        state, sqrt_grid = _make_state()
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(100.0),
        )

        with pytest.raises(ValueError, match="direction must be -1 for sell or 1 for buy"):
            model.process_swap(
                state,
                np.array([True]),
                0,
                tick_lower_global=100,
                sqrt_grid=sqrt_grid,
                num_ticks=10,
            )

    def test_negative_sampled_trade_size_raises(self):
        state, sqrt_grid = _make_state()
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(-1.0),
        )

        with pytest.raises(AssertionError, match="non-negative trade sizes"):
            model.process_swap(
                state,
                np.array([True]),
                1,
                tick_lower_global=100,
                sqrt_grid=sqrt_grid,
                num_ticks=10,
            )

    def test_invalid_trade_size_unit_raises(self):
        with pytest.raises(AssertionError, match="trade_size_unit"):
            LiquidityDepthUniswapV3PriceImpact(
                trade_size_sampler=_fixed_sampler(1.0),
                trade_size_unit="usd",
            )

    def test_token1_notional_sell_size_is_converted_to_token0_using_external_midprice(self):
        external_midprice = 150.0
        state, sqrt_grid = _make_state(
            tick_lower_global=100,
            current_tick=105,
            asset_price=external_midprice,
        )
        idx = 105 - 100
        average_token0_depth = np.mean([
            _sell_capacity(state, sqrt_grid, 0, idx - 1),
            _sell_capacity(state, sqrt_grid, 0, idx - 2),
        ])
        pool_price = state[POOL_SQRT_PRICE_KEY][0] ** 2
        assert not np.isclose(pool_price, external_midprice)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(2.0 * average_token0_depth * external_midprice),
            depth_window=2,
            trade_size_unit="token1_notional",
        )

        result = model.process_swap(
            state,
            np.array([True]),
            -1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 103
        capacities = np.array([
            _sell_capacity(state, sqrt_grid, 0, idx - 1),
            _sell_capacity(state, sqrt_grid, 0, idx - 2),
        ])
        np.testing.assert_array_equal(result.fee_indices, np.array([idx - 1, idx - 2]))
        np.testing.assert_allclose(
            result.amounts,
            _capacity_weighted_amounts(capacities, 2.0 * average_token0_depth),
        )

    def test_token1_notional_sell_requires_positive_external_midprice(self):
        state, sqrt_grid = _make_state(asset_price=0.0)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(1.0),
            depth_window=1,
            trade_size_unit="token1_notional",
        )

        with pytest.raises(AssertionError, match="external midprice must be positive"):
            model.process_swap(
                state,
                np.array([True]),
                -1,
                tick_lower_global=100,
                sqrt_grid=sqrt_grid,
                num_ticks=10,
            )

    def test_token1_notional_buy_size_uses_token1_directly(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        average_token1_depth = np.mean([
            _buy_capacity(state, sqrt_grid, 0, idx),
            _buy_capacity(state, sqrt_grid, 0, idx + 1),
        ])
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(2.0 * average_token1_depth),
            depth_window=2,
            trade_size_unit="token1_notional",
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 107
        capacities = np.array([
            _buy_capacity(state, sqrt_grid, 0, idx),
            _buy_capacity(state, sqrt_grid, 0, idx + 1),
        ])
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx + 1]))
        np.testing.assert_allclose(
            result.amounts,
            _capacity_weighted_amounts(capacities, 2.0 * average_token1_depth),
        )


class TestUniswapV3FeeAccounting:
    def test_none_swap_result_is_noop(self):
        state, _ = _make_state(num_trajectories=2)
        before = {key: value.copy() for key, value in state.items()}
        model = UniswapV3FeeAccounting()

        model.apply_fees(state, None, fee_multiplier=0.003)

        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)

    def test_sell_result_updates_fees0(self):
        state, _ = _make_state(num_trajectories=2)
        model = UniswapV3FeeAccounting()
        result = SwapResult(
            trajectories=np.array([0, 1]),
            fee_key=FEES0_KEY,
            fee_indices=np.array([3, 4]),
            amounts=np.array([10.0, 20.0]),
            direction=-1,
        )

        model.apply_fees(state, result, fee_multiplier=0.003)

        assert state[FEES0_KEY][0, 3] == pytest.approx(0.03)
        assert state[FEES0_KEY][1, 4] == pytest.approx(0.06)
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_buy_result_updates_fees1(self):
        state, _ = _make_state(num_trajectories=3)
        model = UniswapV3FeeAccounting()
        result = SwapResult(
            trajectories=np.array([1, 2]),
            fee_key=FEES1_KEY,
            fee_indices=np.array([5, 6]),
            amounts=np.array([30.0, 40.0]),
            direction=1,
        )

        model.apply_fees(state, result, fee_multiplier=0.003)

        assert np.sum(state[FEES0_KEY]) == 0.0
        assert state[FEES1_KEY][1, 5] == pytest.approx(0.09)
        assert state[FEES1_KEY][2, 6] == pytest.approx(0.12)

    def test_repeated_fee_indices_are_scatter_added(self):
        state, _ = _make_state(num_trajectories=1)
        model = UniswapV3FeeAccounting()
        result = SwapResult(
            trajectories=np.array([0, 0, 0]),
            fee_key=FEES0_KEY,
            fee_indices=np.array([3, 3, 4]),
            amounts=np.array([10.0, 20.0, 40.0]),
            direction=-1,
        )

        model.apply_fees(state, result, fee_multiplier=0.003)

        assert state[FEES0_KEY][0, 3] == pytest.approx(0.09)
        assert state[FEES0_KEY][0, 4] == pytest.approx(0.12)
        assert np.sum(state[FEES1_KEY]) == 0.0


class TestUniswapV3ModelDynamicsPriceImpactWiring:
    def test_default_models_are_configured(self):
        model = UniswapV3ModelDynamics(num_trajectories=2)

        assert isinstance(model.price_impact_model, OneTickUniswapV3PriceImpact)
        assert isinstance(model.fee_accounting_model, UniswapV3FeeAccounting)

    def test_process_swap_uses_injected_models(self):
        class PriceImpactSpy:
            def __init__(self):
                self.called_with = None

            def process_swap(
                self,
                state,
                active,
                direction,
                *,
                tick_lower_global,
                sqrt_grid,
                num_ticks,
                fee_multiplier=0.0,
            ):
                self.called_with = (
                    state,
                    active.copy(),
                    direction,
                    tick_lower_global,
                    sqrt_grid,
                    num_ticks,
                    fee_multiplier,
                )
                return SwapResult(
                    trajectories=np.array([1]),
                    fee_key=FEES0_KEY,
                    fee_indices=np.array([4]),
                    amounts=np.array([12.0]),
                    direction=direction,
                )

        class FeeAccountingSpy:
            def __init__(self):
                self.called_with = None

            def apply_fees(self, state, swap_result, fee_multiplier):
                self.called_with = (state, swap_result, fee_multiplier)

        price_impact = PriceImpactSpy()
        fee_accounting = FeeAccountingSpy()
        model = UniswapV3ModelDynamics(
            price_impact_model=price_impact,
            fee_accounting_model=fee_accounting,
            num_trajectories=2,
            fee_tier=0.003,
        )
        state, sqrt_grid = _make_state(num_trajectories=2)
        model.state = state
        model.tick_lower_global = 100
        model.sqrt_grid = sqrt_grid
        model.num_ticks = 10
        active = np.array([False, True])

        model._process_swap(active, -1)

        assert price_impact.called_with[0] is state
        np.testing.assert_array_equal(price_impact.called_with[1], active)
        assert price_impact.called_with[2] == -1
        assert price_impact.called_with[3] == 100
        assert price_impact.called_with[4] is sqrt_grid
        assert price_impact.called_with[5] == 10
        assert price_impact.called_with[6] == pytest.approx(model.fee_multiplier)
        assert fee_accounting.called_with[0] is state
        assert fee_accounting.called_with[1].amounts[0] == 12.0
        assert fee_accounting.called_with[2] == pytest.approx(model.fee_multiplier)

    def test_default_process_swap_moves_price_and_accounts_fees(self):
        model = UniswapV3ModelDynamics(num_trajectories=1, fee_tier=0.003)
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        model.state = state
        model.tick_lower_global = 100
        model.sqrt_grid = sqrt_grid
        model.num_ticks = 10

        model._process_swap(np.array([True]), -1)

        idx = 105 - 100
        fee_idx = idx - 1
        expected_amount = 1e6 * (1.0 / sqrt_grid[idx - 1] - 1.0 / sqrt_grid[idx])
        assert state[POOL_CURRENT_TICK_KEY][0] == 104
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[4]
        assert state[FEES0_KEY][0, fee_idx] == pytest.approx(model.fee_multiplier * expected_amount)
        assert np.sum(state[FEES1_KEY]) == 0.0
