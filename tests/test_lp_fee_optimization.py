import numpy as np
import pytest

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics, _PositionFeeInputs
from SAiFE_gym.gym.index_names import (
    FEES0_KEY, FEES1_KEY, INITIAL_WEALTH_KEY, LP_FEE_SNAPSHOT0_KEY,
    LP_FEE_SNAPSHOT1_KEY, LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_UNCLAIMED_FEES0_KEY, LP_UNCLAIMED_FEES1_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    PORTFOLIO_VALUE_KEY, TIME_KEY,
)
from SAiFE_gym.rewards.RewardFunctions import (
    DenseExponentialUtility, ExponentialUtility, PnL, RunningInventoryPenalty,
)
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel


def dense_claimable(md):
    state = md.state
    pool = state[POOL_LIQUIDITY_ARRAY_KEY]
    ticks = md.tick_lower_global + np.arange(md.num_ticks)
    in_range = (ticks >= state[LP_TICK_LOWER_KEY][:, None]) & (ticks < state[LP_TICK_UPPER_KEY][:, None])
    shares = np.where(pool > 0, state[LP_LIQUIDITY_KEY][:, None] / np.where(pool > 0, pool, 1.0), 0.0)
    # The original out-of-place product also determines the summation layout.
    shares = shares * in_range
    return tuple(np.maximum((state[key] * shares).sum(axis=1) - state[snapshot], 0.0)
                 for key, snapshot in ((FEES0_KEY, LP_FEE_SNAPSHOT0_KEY),
                                       (FEES1_KEY, LP_FEE_SNAPSHOT1_KEY)))


@pytest.mark.parametrize("order,dtype", [("C", np.float64), ("F", np.float64), ("C", np.float32)])
def test_fee_range_and_cache_match_full_grid_after_external_edits(order, dtype):
    env = AMMEnvironment(num_trajectories=4)
    state, _ = env.reset()
    md = env.model_dynamics
    anchor = md.tick_lower_global
    state[LP_TICK_LOWER_KEY][:] = anchor + np.array([-5, 1017, 2000, 100])
    state[LP_TICK_UPPER_KEY][:] = anchor + np.array([43, 1159, 2005, 100])
    state[LP_LIQUIDITY_KEY][:] = [0.8, 0.6, 1.0, 1.0]
    rng = np.random.default_rng(13)
    for key in (FEES0_KEY, FEES1_KEY, POOL_LIQUIDITY_ARRAY_KEY):
        state[key] = np.array(np.exp(rng.uniform(-5, 10, (4, md.num_ticks))), dtype=dtype, order=order)
    state[POOL_LIQUIDITY_ARRAY_KEY][0, ::3] = 0.0

    def check():
        expected = dense_claimable(md)
        np.testing.assert_array_equal(md.compute_unclaimed_fees(), expected)
        md._accrue_lp_fees()
        np.testing.assert_array_equal((state[LP_UNCLAIMED_FEES0_KEY], state[LP_UNCLAIMED_FEES1_KEY]), expected)

    check()
    state[FEES0_KEY][1, 1020] += 300
    state[LP_FEE_SNAPSHOT0_KEY][0] += 0.12
    state[POOL_LIQUIDITY_ARRAY_KEY][1, 1034] *= 3
    check()
    state[LP_LIQUIDITY_KEY][0] *= 2
    state[LP_TICK_LOWER_KEY][1] += 11
    state[LP_TICK_UPPER_KEY][3] += 750
    check()
    env.reset()
    assert md._fee_cache is None


@pytest.mark.parametrize("shared_shape", [(-1,), (1, -1)])
def test_shared_liquidity_preserves_broadcast_fee_calculation(shared_shape):
    env = AMMEnvironment(num_trajectories=2)
    state, _ = env.reset()
    md = env.model_dynamics
    state[POOL_LIQUIDITY_ARRAY_KEY] = state[POOL_LIQUIDITY_ARRAY_KEY][0].reshape(shared_shape)
    state[LP_LIQUIDITY_KEY][:] = [1., 2.]
    first_tick = state[LP_TICK_LOWER_KEY][0] - md.tick_lower_global
    state[FEES0_KEY][:, first_tick] = [2., 3.]
    expected = dense_claimable(md)
    np.testing.assert_array_equal(md.compute_unclaimed_fees(), expected)
    md._accrue_lp_fees()
    np.testing.assert_array_equal(state[LP_UNCLAIMED_FEES0_KEY], expected[0])


def test_only_changed_paths_undergo_fee_reduction(monkeypatch):
    env = AMMEnvironment(num_trajectories=3)
    state, _ = env.reset()
    md = env.model_dynamics
    state[LP_LIQUIDITY_KEY][:] = 1.0
    first_tick = state[LP_TICK_LOWER_KEY][0] - md.tick_lower_global
    reductions = []
    original = _PositionFeeInputs.gross_fees

    def record(inputs, *args, **kwargs):
        reductions.append(len(inputs.start))
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(_PositionFeeInputs, "gross_fees", record)
    md._accrue_lp_fees()
    state[FEES1_KEY][:, first_tick - 1] += 1.0
    md._accrue_lp_fees()
    state[FEES1_KEY][1, first_tick] += 20.0
    md._accrue_lp_fees()
    state[LP_FEE_SNAPSHOT1_KEY][1] += 0.001
    md._accrue_lp_fees()
    assert reductions == [3, 1]
    np.testing.assert_array_equal(state[LP_UNCLAIMED_FEES1_KEY], dense_claimable(md)[1])


def test_uncached_hold_reuses_one_fee_calculation_for_valuation(monkeypatch):
    env = AMMEnvironment()
    env.reset(seed=21)
    env.model_dynamics.cache_lp_fees = False
    env.step([-1, 1, -1])
    calls = []
    original = _PositionFeeInputs.gross_fees

    def record(inputs, *args, **kwargs):
        calls.append(len(inputs.start))
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(_PositionFeeInputs, "gross_fees", record)
    state, *_ = env.step([-1, 1, 1])
    assert calls == [1]
    np.testing.assert_array_equal(state[PORTFOLIO_VALUE_KEY], env.model_dynamics.compute_portfolio_value())
    assert calls == [1, 1]


class CustomPnL(PnL):
    pass


@pytest.mark.parametrize("reward,keys", [
    (PnL(), {PORTFOLIO_VALUE_KEY}), (RunningInventoryPenalty(), {PORTFOLIO_VALUE_KEY, TIME_KEY}),
    (DenseExponentialUtility(0.0), {PORTFOLIO_VALUE_KEY, INITIAL_WEALTH_KEY}),
    (ExponentialUtility(), set()), (CustomPnL(), None),
])
def test_reward_receives_detached_previous_fields(monkeypatch, reward, keys):
    env = AMMEnvironment(reward_function=reward)
    state, _ = env.reset(seed=42)
    expected = {key: value.copy() for key, value in state.items() if keys is None or key in keys}
    calculate = reward.calculate

    def check(current, action, following, terminal):
        assert current.keys() == expected.keys()
        for key in current:
            np.testing.assert_array_equal(current[key], expected[key])
            assert not np.shares_memory(current[key], following[key])
        return calculate(current, action, following, terminal)

    monkeypatch.setattr(reward, "calculate", check)
    env.step([-1, 1, -1])


def test_custom_fee_formula_uses_post_step_time_for_observations():
    class TimeRebateDynamics(UniswapV3ModelDynamics):
        def _compute_gross_lp_fees(self, state=None):
            state = self._state_or_current(state)
            gross0, gross1, shares = super()._compute_gross_lp_fees(state)
            return gross0, gross1 + state[TIME_KEY], shares

    base = AMMEnvironment().model_dynamics
    md = TimeRebateDynamics(base.midprice_model, base.arrival_model)
    env = AMMEnvironment(model_dynamics=md)
    env.reset(seed=42)
    state, *_ = env.step([-1, 1, -1])
    assert state[LP_UNCLAIMED_FEES1_KEY][0] >= env.step_size
    np.testing.assert_array_equal(state[LP_UNCLAIMED_FEES1_KEY], md.compute_unclaimed_fees()[1])
    np.testing.assert_array_equal(state[PORTFOLIO_VALUE_KEY], md.compute_portfolio_value())


def test_market_process_can_change_fee_inputs_after_accrual():
    class MarketWithFees(BrownianMotionMidpriceModel):
        def update(self, arrivals, fills, actions, state):
            super().update(arrivals, fills, actions, state)
            state[FEES1_KEY] += 2.0

    env = AMMEnvironment()
    md = env.model_dynamics
    md.midprice_model = MarketWithFees(step_size=env.step_size)
    env.reset(seed=7)
    state, *_ = env.step([-1, 1, -1])
    expected = dense_claimable(md)[1]
    assert expected[0] > 0
    np.testing.assert_array_equal(state[LP_UNCLAIMED_FEES1_KEY], expected)
    np.testing.assert_array_equal(state[PORTFOLIO_VALUE_KEY], md.compute_portfolio_value())
