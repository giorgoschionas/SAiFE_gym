import numpy as np
import pytest

from SAiFE_gym.gym.index_names import (
    FEES0_KEY,
    FEES1_KEY,
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


def _buy_capacity(state, sqrt_grid, trajectory, fee_idx):
    liquidity = state[POOL_LIQUIDITY_ARRAY_KEY][trajectory, fee_idx]
    return liquidity * (sqrt_grid[fee_idx + 1] - sqrt_grid[fee_idx])


def _sell_capacity(state, sqrt_grid, trajectory, fee_idx):
    liquidity = state[POOL_LIQUIDITY_ARRAY_KEY][trajectory, fee_idx]
    return liquidity * (1.0 / sqrt_grid[fee_idx] - 1.0 / sqrt_grid[fee_idx + 1])


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
        average_depth = np.mean([
            _buy_capacity(state, sqrt_grid, 0, idx),
            _buy_capacity(state, sqrt_grid, 0, idx + 1),
        ])
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

        expected_amounts = np.array([
            _buy_capacity(state, sqrt_grid, 0, idx),
            _buy_capacity(state, sqrt_grid, 0, idx + 1),
        ])
        assert state[POOL_CURRENT_TICK_KEY][0] == 107
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[7]
        assert result.fee_key == FEES1_KEY
        assert result.direction == 1
        np.testing.assert_array_equal(result.trajectories, np.array([0, 0]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx + 1]))
        np.testing.assert_allclose(result.amounts, expected_amounts)
        assert np.sum(state[FEES0_KEY]) == 0.0
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_sell_moves_exact_integer_ticks_and_returns_crossed_fee_entries(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        average_depth = np.mean([
            _sell_capacity(state, sqrt_grid, 0, idx - 1),
            _sell_capacity(state, sqrt_grid, 0, idx - 2),
        ])
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

        expected_amounts = np.array([
            _sell_capacity(state, sqrt_grid, 0, idx - 1),
            _sell_capacity(state, sqrt_grid, 0, idx - 2),
        ])
        assert state[POOL_CURRENT_TICK_KEY][0] == 103
        assert state[POOL_SQRT_PRICE_KEY][0] == sqrt_grid[3]
        assert result.fee_key == FEES0_KEY
        assert result.direction == -1
        np.testing.assert_array_equal(result.trajectories, np.array([0, 0]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx - 1, idx - 2]))
        np.testing.assert_allclose(result.amounts, expected_amounts)
        assert np.sum(state[FEES0_KEY]) == 0.0
        assert np.sum(state[FEES1_KEY]) == 0.0

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
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx + 1]))

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
        np.testing.assert_array_equal(result.trajectories, np.array([0, 2, 2]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx, idx + 1]))
        np.testing.assert_allclose(
            result.amounts,
            np.array([
                _buy_capacity(state, sqrt_grid, 0, idx),
                _buy_capacity(state, sqrt_grid, 2, idx),
                _buy_capacity(state, sqrt_grid, 2, idx + 1),
            ]),
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
        np.testing.assert_array_equal(result.fee_indices, np.array([8, 9]))

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

            def process_swap(self, state, active, direction, *, tick_lower_global, sqrt_grid, num_ticks):
                self.called_with = (state, active.copy(), direction, tick_lower_global, sqrt_grid, num_ticks)
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
