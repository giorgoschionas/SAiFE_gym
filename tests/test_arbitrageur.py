import numpy as np
import pytest

from SAiFE_gym.agents.BaselineAgents import ArbitrageurAgent, SpeedControlArbitrageurAgent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    FEES0_KEY,
    FEES1_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
)
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel


def _make_env(num_trajectories=1, num_ticks=100, initial_price=100.0):
    step_size = 0.1
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=0.0,
        initial_price=initial_price,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=7,
    )
    arrival_model = PoissonArrivalModel(
        intensity=np.array([0.0, 0.0]),
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=8,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=0.003,
        tau=5,
        num_ticks=num_ticks,
        seed=9,
    )
    env = AMMEnvironment(
        terminal_time=1.0,
        n_steps=10,
        model_dynamics=model_dynamics,
        initial_wealth=1e6,
        num_trajectories=num_trajectories,
        initial_pool_price=initial_price,
        seed=10,
    )
    state, _ = env.reset()
    return env, state


def _current_idx(env):
    md = env.model_dynamics
    return int(md.state[POOL_CURRENT_TICK_KEY][0] - md.tick_lower_global)


def _buy_capacity(env, trajectory=0):
    md = env.model_dynamics
    idx = _current_idx(env)
    liquidity = md.state[POOL_LIQUIDITY_ARRAY_KEY][trajectory, idx]
    return liquidity * (md.sqrt_grid[idx + 1] - md.sqrt_grid[idx])


def _sell_capacity(env, trajectory=0):
    md = env.model_dynamics
    idx = _current_idx(env)
    liquidity = md.state[POOL_LIQUIDITY_ARRAY_KEY][trajectory, idx - 1]
    return liquidity * (1.0 / md.sqrt_grid[idx - 1] - 1.0 / md.sqrt_grid[idx])


def _gross_input(env, curve_input):
    return curve_input * (1.0 + env.model_dynamics.fee_multiplier)


class TestArbitrageurAgent:
    def test_action_signs_follow_fee_adjusted_mispricing(self):
        env, state = _make_env(num_trajectories=3)
        pool_price = state[POOL_SQRT_PRICE_KEY] ** 2
        state[ASSET_PRICE_KEY] = pool_price * np.array([1.01, 0.99, 1.0])

        action = ArbitrageurAgent(env, max_ticks_per_trade=3).get_action(state)

        assert action.shape == (3, 1)
        assert action[0, 0] > 0.0
        assert action[1, 0] < 0.0
        assert action[2, 0] == 0.0

    def test_inside_no_arbitrage_band_returns_zero(self):
        env, state = _make_env()
        pool_price = state[POOL_SQRT_PRICE_KEY] ** 2
        state[ASSET_PRICE_KEY] = pool_price * 1.001

        action = ArbitrageurAgent(env).get_action(state)

        np.testing.assert_array_equal(action, np.zeros((1, 1)))


class TestSpeedControlArbitrageurAgent:
    def test_action_signs_follow_fee_adjusted_mispricing(self):
        env, state = _make_env(num_trajectories=3)
        pool_price = state[POOL_SQRT_PRICE_KEY] ** 2
        state[ASSET_PRICE_KEY] = pool_price * np.array([1.01, 0.99, 1.0])

        action = SpeedControlArbitrageurAgent(env, max_ticks_per_step=3).get_action(state)

        assert action.shape == (3, 1)
        assert action[0, 0] > 0.0
        assert action[1, 0] < 0.0
        assert action[2, 0] == 0.0

    def test_inside_no_arbitrage_band_returns_zero(self):
        env, state = _make_env()
        pool_price = state[POOL_SQRT_PRICE_KEY] ** 2
        state[ASSET_PRICE_KEY] = pool_price * 1.001

        action = SpeedControlArbitrageurAgent(env).get_action(state)

        np.testing.assert_array_equal(action, np.zeros((1, 1)))

    def test_speed_times_step_size_matches_size_control_target(self):
        env, state = _make_env()
        pool_price = state[POOL_SQRT_PRICE_KEY] ** 2
        state[ASSET_PRICE_KEY] = pool_price * 1.01

        size_action = ArbitrageurAgent(env, max_ticks_per_trade=1).get_action(state)
        speed_action = SpeedControlArbitrageurAgent(
            env,
            max_ticks_per_step=1,
        ).get_action(state)

        np.testing.assert_allclose(speed_action * env.step_size, size_action)

    def test_speed_cap_is_applied_in_speed_units(self):
        env, state = _make_env()
        pool_price = state[POOL_SQRT_PRICE_KEY] ** 2
        state[ASSET_PRICE_KEY] = pool_price * 1.01

        action = SpeedControlArbitrageurAgent(
            env,
            max_ticks_per_step=3,
            max_speed=25.0,
        ).get_action(state)

        assert action[0, 0] == pytest.approx(25.0)


class TestLiquidityTakerExecution:
    def test_vectorized_positive_negative_and_zero_orders(self):
        env, state = _make_env(num_trajectories=3)
        md = env.model_dynamics
        idx = md.state[POOL_CURRENT_TICK_KEY].astype(np.int64) - md.tick_lower_global
        traj = np.arange(md.num_trajectories)
        buy_curve_input = md.state[POOL_LIQUIDITY_ARRAY_KEY][traj, idx] * (
            md.sqrt_grid[idx + 1] - md.sqrt_grid[idx]
        )
        sell_curve_input = md.state[POOL_LIQUIDITY_ARRAY_KEY][traj, idx - 1] * (
            1.0 / md.sqrt_grid[idx - 1] - 1.0 / md.sqrt_grid[idx]
        )
        gross_multiplier = 1.0 + md.fee_multiplier
        start_tick = state[POOL_CURRENT_TICK_KEY].copy()

        diagnostics = md.execute_liquidity_taker_orders(
            np.array(
                [
                    [gross_multiplier * buy_curve_input[0]],
                    [-gross_multiplier * sell_curve_input[1]],
                    [0.0],
                ]
            )
        )

        np.testing.assert_array_equal(
            state[POOL_CURRENT_TICK_KEY],
            start_tick + np.array([1, -1, 0]),
        )
        np.testing.assert_array_equal(diagnostics["tick_movement"], np.array([1, -1, 0]))
        assert diagnostics["executed_input"][0] > 0.0
        assert diagnostics["executed_input"][1] < 0.0
        assert diagnostics["executed_input"][2] == 0.0
        assert diagnostics["curve_input"][0] == pytest.approx(buy_curve_input[0])
        assert diagnostics["curve_input"][1] == pytest.approx(-sell_curve_input[1])
        assert diagnostics["curve_input"][2] == 0.0
        assert np.sum(state[FEES1_KEY][0]) > 0.0
        assert np.sum(state[FEES0_KEY][1]) > 0.0
        assert np.sum(state[FEES0_KEY][2]) == 0.0
        assert np.sum(state[FEES1_KEY][2]) == 0.0

    def test_positive_order_moves_up_and_credits_token1_fees(self):
        env, state = _make_env()
        md = env.model_dynamics
        start_tick = int(state[POOL_CURRENT_TICK_KEY][0])
        idx = _current_idx(env)
        liquidity = md.state[POOL_LIQUIDITY_ARRAY_KEY][0, idx]
        curve_input = _buy_capacity(env)
        gross_input = _gross_input(env, curve_input)
        expected_token0_out = liquidity * (
            1.0 / md.sqrt_grid[idx] - 1.0 / md.sqrt_grid[idx + 1]
        )

        diagnostics = md.execute_liquidity_taker_orders(np.array([[gross_input]]))

        assert state[POOL_CURRENT_TICK_KEY][0] == start_tick + 1
        assert diagnostics["tick_movement"][0] == 1
        assert diagnostics["executed_input"][0] == pytest.approx(gross_input)
        assert diagnostics["unfilled_input"][0] == pytest.approx(0.0)
        assert diagnostics["curve_input"][0] == pytest.approx(curve_input)
        assert diagnostics["token0_delta"][0] == pytest.approx(expected_token0_out)
        assert diagnostics["token1_delta"][0] == pytest.approx(-gross_input)
        assert diagnostics["fee_input"][0] == pytest.approx(md.fee_multiplier * curve_input)
        assert np.sum(state[FEES0_KEY]) == 0.0
        assert np.sum(state[FEES1_KEY]) == pytest.approx(md.fee_multiplier * curve_input)

    def test_negative_order_moves_down_and_credits_token0_fees(self):
        env, state = _make_env()
        md = env.model_dynamics
        start_tick = int(state[POOL_CURRENT_TICK_KEY][0])
        idx = _current_idx(env)
        liquidity = md.state[POOL_LIQUIDITY_ARRAY_KEY][0, idx - 1]
        curve_input = _sell_capacity(env)
        gross_input = _gross_input(env, curve_input)
        expected_token1_out = liquidity * (
            md.sqrt_grid[idx] - md.sqrt_grid[idx - 1]
        )

        diagnostics = md.execute_liquidity_taker_orders(np.array([[-gross_input]]))

        assert state[POOL_CURRENT_TICK_KEY][0] == start_tick - 1
        assert diagnostics["tick_movement"][0] == -1
        assert diagnostics["executed_input"][0] == pytest.approx(-gross_input)
        assert diagnostics["unfilled_input"][0] == pytest.approx(0.0)
        assert diagnostics["curve_input"][0] == pytest.approx(-curve_input)
        assert diagnostics["token0_delta"][0] == pytest.approx(-gross_input)
        assert diagnostics["token1_delta"][0] == pytest.approx(expected_token1_out)
        assert diagnostics["fee_input"][0] == pytest.approx(md.fee_multiplier * curve_input)
        assert np.sum(state[FEES0_KEY]) == pytest.approx(md.fee_multiplier * curve_input)
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_zero_order_leaves_pool_state_unchanged(self):
        env, state = _make_env()
        before = {
            key: state[key].copy()
            for key in (POOL_CURRENT_TICK_KEY, POOL_SQRT_PRICE_KEY, FEES0_KEY, FEES1_KEY)
        }

        diagnostics = env.model_dynamics.execute_liquidity_taker_orders(np.array([[0.0]]))

        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)
        assert diagnostics["executed_input"][0] == 0.0
        assert diagnostics["unfilled_input"][0] == 0.0
        assert diagnostics["curve_input"][0] == 0.0
        assert diagnostics["token0_delta"][0] == 0.0
        assert diagnostics["token1_delta"][0] == 0.0
        assert diagnostics["fee_input"][0] == 0.0

    def test_oversized_order_stops_at_boundary_and_reports_unfilled_input(self):
        env, state = _make_env(num_ticks=20)
        md = env.model_dynamics
        upper_tick = md.tick_lower_global + md.num_ticks

        diagnostics = md.execute_liquidity_taker_orders(np.array([[1e12]]))

        assert state[POOL_CURRENT_TICK_KEY][0] == upper_tick
        assert diagnostics["tick_movement"][0] > 0
        assert diagnostics["unfilled_input"][0] > 0.0
        assert diagnostics["executed_input"][0] > 0.0


class TestLiquidityTakerSpeedExecution:
    def test_positive_speed_moves_up_and_credits_token1_fees(self):
        env, state = _make_env()
        md = env.model_dynamics
        start_tick = int(state[POOL_CURRENT_TICK_KEY][0])
        curve_input = _buy_capacity(env)
        gross_input = _gross_input(env, curve_input)
        speed = gross_input / env.step_size

        diagnostics = md.execute_liquidity_taker_speeds(np.array([[speed]]))

        assert state[POOL_CURRENT_TICK_KEY][0] == start_tick + 1
        assert diagnostics["tick_movement"][0] == 1
        assert diagnostics["trading_speed"][0] == pytest.approx(speed)
        assert diagnostics["step_size"] == pytest.approx(env.step_size)
        assert diagnostics["unfilled_input"][0] == pytest.approx(0.0)
        assert diagnostics["executed_input"][0] == pytest.approx(gross_input)
        assert diagnostics["curve_input"][0] == pytest.approx(curve_input)
        assert np.sum(state[FEES0_KEY]) == 0.0
        assert np.sum(state[FEES1_KEY]) == pytest.approx(md.fee_multiplier * curve_input)

    def test_negative_speed_moves_down_and_credits_token0_fees(self):
        env, state = _make_env()
        md = env.model_dynamics
        start_tick = int(state[POOL_CURRENT_TICK_KEY][0])
        curve_input = _sell_capacity(env)
        gross_input = _gross_input(env, curve_input)
        speed = -gross_input / env.step_size

        diagnostics = md.execute_liquidity_taker_speeds(np.array([[speed]]))

        assert state[POOL_CURRENT_TICK_KEY][0] == start_tick - 1
        assert diagnostics["tick_movement"][0] == -1
        assert diagnostics["trading_speed"][0] == pytest.approx(speed)
        assert diagnostics["unfilled_input"][0] == pytest.approx(0.0)
        assert diagnostics["executed_input"][0] == pytest.approx(-gross_input)
        assert diagnostics["curve_input"][0] == pytest.approx(-curve_input)
        assert np.sum(state[FEES0_KEY]) == pytest.approx(md.fee_multiplier * curve_input)
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_zero_speed_leaves_pool_state_unchanged(self):
        env, state = _make_env()
        before = {
            key: state[key].copy()
            for key in (POOL_CURRENT_TICK_KEY, POOL_SQRT_PRICE_KEY, FEES0_KEY, FEES1_KEY)
        }

        diagnostics = env.model_dynamics.execute_liquidity_taker_speeds(np.array([[0.0]]))

        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)
        assert diagnostics["executed_input"][0] == 0.0
        assert diagnostics["unfilled_input"][0] == 0.0
        assert diagnostics["curve_input"][0] == 0.0

    def test_speed_execution_matches_direct_order_execution(self):
        env_speed, state_speed = _make_env()
        env_order, state_order = _make_env()
        gross_input = _gross_input(env_speed, _buy_capacity(env_speed))
        speed = gross_input / env_speed.step_size

        speed_diagnostics = env_speed.model_dynamics.execute_liquidity_taker_speeds(
            np.array([[speed]])
        )
        order_diagnostics = env_order.model_dynamics.execute_liquidity_taker_orders(
            np.array([[gross_input]])
        )

        np.testing.assert_allclose(
            state_speed[POOL_CURRENT_TICK_KEY],
            state_order[POOL_CURRENT_TICK_KEY],
        )
        np.testing.assert_allclose(state_speed[FEES0_KEY], state_order[FEES0_KEY])
        np.testing.assert_allclose(state_speed[FEES1_KEY], state_order[FEES1_KEY])
        np.testing.assert_allclose(
            speed_diagnostics["executed_input"],
            order_diagnostics["executed_input"],
        )
        np.testing.assert_allclose(
            speed_diagnostics["curve_input"],
            order_diagnostics["curve_input"],
        )
