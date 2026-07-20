import numpy as np
import pytest

from SAiFE_gym.gym.index_names import (
    FEES0_KEY,
    FEES1_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
)
from SAiFE_gym.stochastic_processes.price_impact_models import OneTickUniswapV3PriceImpact


EXPONENTIAL_VALUE = 1.0001


def _make_state(num_trajectories=1, num_ticks=10, tick_lower_global=100, current_tick=105):
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
    }, sqrt_grid


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
            fee_multiplier=0.003,
            num_ticks=10,
        )

        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)

    def test_sell_moves_down_one_tick_and_accrues_fees0(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        model = OneTickUniswapV3PriceImpact()
        fee_multiplier = 0.003

        model.process_swap(
            state,
            np.array([True]),
            -1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            fee_multiplier=fee_multiplier,
            num_ticks=10,
        )

        idx = 105 - 100
        fee_idx = idx - 1
        expected_amount = 1e6 * (1.0 / sqrt_grid[idx - 1] - 1.0 / sqrt_grid[idx])
        assert state[POOL_CURRENT_TICK_KEY][0] == 104
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[4]
        assert state[FEES0_KEY][0, fee_idx] == pytest.approx(fee_multiplier * expected_amount)
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_buy_moves_up_one_tick_and_accrues_fees1(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        model = OneTickUniswapV3PriceImpact()
        fee_multiplier = 0.003

        model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            fee_multiplier=fee_multiplier,
            num_ticks=10,
        )

        idx = 105 - 100
        expected_amount = 1e6 * (sqrt_grid[idx + 1] - sqrt_grid[idx])
        assert state[POOL_CURRENT_TICK_KEY][0] == 106
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[6]
        assert state[FEES1_KEY][0, idx] == pytest.approx(fee_multiplier * expected_amount)
        assert np.sum(state[FEES0_KEY]) == 0.0

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
                fee_multiplier=0.003,
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
                fee_multiplier=0.003,
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
                fee_multiplier=0.003,
                num_ticks=10,
            )
