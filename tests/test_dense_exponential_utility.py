"""Dense CARA rewards preserve the terminal objective and LP accounting."""

import numpy as np
import pytest

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    INITIAL_WEALTH_KEY,
    LP_TOKEN0_AMOUNT_KEY,
    LP_UNCLAIMED_FEES0_KEY,
    LP_UNCLAIMED_FEES1_KEY,
    PORTFOLIO_VALUE_KEY,
)
from SAiFE_gym.rewards.RewardFunctions import DenseExponentialUtility, PnL
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel


def portfolio_state(wealth, initial_wealth):
    return {
        PORTFOLIO_VALUE_KEY: np.atleast_1d(wealth).astype(np.float64),
        INITIAL_WEALTH_KEY: np.atleast_1d(initial_wealth).astype(np.float64),
    }


@pytest.mark.parametrize("risk_aversion", [-0.01, 0.0, 0.01])
def test_batched_returns_telescope_independently_of_path(risk_aversion):
    initial_wealth = np.array([100.0, 1000.0, 1e6])
    reward = DenseExponentialUtility(risk_aversion=risk_aversion)
    pnls = np.array([[0, 0, 0], [10, -5, 20], [-5, 10, -20], [5, 5, 5]])
    total = np.zeros(3)
    current = portfolio_state(initial_wealth, initial_wealth)
    reward.reset(current)

    for step, pnl in enumerate(pnls[1:], start=1):
        following = portfolio_state(initial_wealth + pnl, initial_wealth)
        total += reward.calculate(
            current, None, following, is_terminal_step=step == len(pnls) - 1
        )
        current = following

    expected = (
        pnls[-1] if risk_aversion == 0.0
        else -np.expm1(-risk_aversion * pnls[-1]) / risk_aversion
    )
    np.testing.assert_allclose(total, expected, rtol=1e-12)
    np.testing.assert_allclose(total, total[0], rtol=1e-12)


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("risk_aversion", [-0.01, 0.0, 0.01])
def test_terminal_uses_same_increment_and_unchanged_wealth_is_zero(terminal, risk_aversion):
    reward = DenseExponentialUtility(risk_aversion=risk_aversion)
    current = portfolio_state([100, 100, 100], [100, 100, 100])
    following = portfolio_state([110, 90, 100], [100, 100, 100])
    actual = reward.calculate(current, None, following, is_terminal_step=terminal)
    pnl = np.array([10, -10, 0])
    expected = pnl if risk_aversion == 0.0 else -np.expm1(-risk_aversion * pnl) / risk_aversion
    np.testing.assert_allclose(actual, expected)
    assert actual[0] > 0
    assert actual[1] < 0
    assert actual[2] == 0
    np.testing.assert_array_equal(
        reward.calculate(following, None, following, is_terminal_step=terminal),
        np.zeros(3),
    )


def test_fair_gamble_is_worse_than_cash_and_more_aversion_increases_penalty():
    current = portfolio_state([100, 100], [100, 100])
    following = portfolio_state([110, 90], [100, 100])
    low = DenseExponentialUtility(risk_aversion=0.01).calculate(current, None, following)
    high = DenseExponentialUtility(risk_aversion=0.1).calculate(current, None, following)
    assert low.mean() < 0
    assert high.mean() < low.mean()
    assert abs(low[1]) > low[0]


def test_fair_gamble_distinguishes_all_three_risk_profiles():
    current = portfolio_state([100, 100], [100, 100])
    following = portfolio_state([110, 90], [100, 100])
    averse = DenseExponentialUtility(0.01).calculate(current, None, following)
    neutral = DenseExponentialUtility(0.0).calculate(current, None, following)
    seeking = DenseExponentialUtility(-0.01).calculate(current, None, following)
    more_seeking = DenseExponentialUtility(-0.1).calculate(current, None, following)
    assert averse.mean() < neutral.mean() == 0 < seeking.mean() < more_seeking.mean()
    assert seeking[0] > abs(seeking[1])


@pytest.mark.parametrize("risk_aversion", [0.001, 0.01, 0.1])
def test_positive_risk_preserves_previous_reward_up_to_scaling(risk_aversion):
    current = portfolio_state([90, 100, 110], [100, 100, 100])
    following = portfolio_state([95, 100, 105], [100, 100, 100])
    old_reward = np.exp(-risk_aversion * np.array([-10, 0, 10])) - np.exp(
        -risk_aversion * np.array([-5, 0, 5])
    )
    actual = DenseExponentialUtility(risk_aversion).calculate(current, None, following)
    np.testing.assert_allclose(actual, old_reward / risk_aversion, rtol=1e-12, atol=0)


def test_neutral_reward_is_exactly_pnl_even_away_from_initial_wealth():
    current = portfolio_state([80, 100, 110], [100, 100, 100])
    following = portfolio_state([100, 95, 111], [100, 100, 100])
    actual = DenseExponentialUtility(0.0).calculate(current, None, following)
    np.testing.assert_array_equal(actual, PnL().calculate(current, None, following))


@pytest.mark.parametrize("risk_aversion", [-0.01, 0.0, 0.01])
def test_increment_depends_on_cumulative_pnl_not_just_step_pnl(risk_aversion):
    reward = DenseExponentialUtility(risk_aversion=risk_aversion)
    current = portfolio_state([90, 100, 110], [100, 100, 100])
    following = portfolio_state([91, 101, 111], [100, 100, 100])
    actual = reward.calculate(current, None, following)
    assert np.all(actual > 0)
    np.testing.assert_allclose(actual[:-1] / actual[1:], np.exp(10 * risk_aversion))


@pytest.mark.parametrize("risk_aversion", [-1.0, 0.0, 1.0])
def test_tiny_pnl_changes_do_not_cancel_to_zero(risk_aversion):
    reward = DenseExponentialUtility(risk_aversion=risk_aversion)
    actual = reward.calculate(
        portfolio_state([0, 0], [0, 0]), None,
        portfolio_state([1e-16, -1e-16], [0, 0]),
    )
    np.testing.assert_allclose(actual, [1e-16, -1e-16], rtol=1e-14, atol=0)


@pytest.mark.parametrize("risk_aversion", [-1.0, 1.0])
def test_large_move_and_reversal_avoid_artificial_overflow(risk_aversion):
    reward = DenseExponentialUtility(risk_aversion=risk_aversion)
    actual = reward.calculate(
        portfolio_state(np.array([0, 1000]) * risk_aversion, [0, 0]), None,
        portfolio_state(np.array([1000, 0]) * risk_aversion, [0, 0]),
    )
    np.testing.assert_array_equal(actual, np.array([1, -1]) * risk_aversion)


@pytest.mark.parametrize("risk_aversion", [-1.0, 1.0])
def test_small_increment_is_finite_even_when_utility_level_overflows(risk_aversion):
    reward = DenseExponentialUtility(risk_aversion=risk_aversion)
    current = portfolio_state([0, 0], np.array([710, 710]) * risk_aversion)
    following = portfolio_state(np.array([1e-6, 0]) * risk_aversion, np.array([710, 710]) * risk_aversion)
    actual = reward.calculate(current, None, following)
    assert np.isfinite(actual).all()
    assert np.sign(actual[0]) == np.sign(risk_aversion)
    assert actual[1] == 0
    # exp(710) is outside float64, but this increment is approximately exp(710)*1e-6.
    np.testing.assert_allclose(np.log(abs(actual[0])), 710 + np.log(1e-6) - 0.5e-6)


@pytest.mark.parametrize("risk_aversion", [-1.0, 1.0])
def test_unrepresentable_reward_raises_instead_of_clipping(risk_aversion):
    reward = DenseExponentialUtility(risk_aversion=risk_aversion)
    with pytest.raises(FloatingPointError, match=r"reduce abs\(risk_aversion\)"):
        reward.calculate(portfolio_state(1000, 1000), None, portfolio_state(1000 - 1000 * risk_aversion, 1000))


@pytest.mark.parametrize("risk_aversion", [-1e-12, 1e-12, -1e-320, 1e-320])
def test_near_zero_risk_converges_to_pnl_without_underflow(risk_aversion):
    current = portfolio_state([0, 0, 1, -1], [0, 0, 0, 0])
    following = portfolio_state([1e-6, -1e-6, 2, -3], [0, 0, 0, 0])
    actual = DenseExponentialUtility(risk_aversion).calculate(current, None, following)
    np.testing.assert_allclose(actual, PnL().calculate(current, None, following), rtol=5e-12, atol=0)


@pytest.mark.parametrize("risk_aversion", [-0.01, 0.0, 0.01])
def test_scalar_wealth_is_supported(risk_aversion):
    current = {PORTFOLIO_VALUE_KEY: 100.0, INITIAL_WEALTH_KEY: 100.0}
    following = {PORTFOLIO_VALUE_KEY: 110.0}
    expected = 10.0 if risk_aversion == 0.0 else -np.expm1(-10.0 * risk_aversion) / risk_aversion
    assert DenseExponentialUtility(risk_aversion).calculate(current, None, following) == pytest.approx(expected)


@pytest.mark.parametrize("invalid_wealth", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("risk_aversion", [-0.01, 0.0, 0.01])
def test_nonfinite_wealth_is_rejected(invalid_wealth, risk_aversion):
    reward = DenseExponentialUtility(risk_aversion=risk_aversion)
    with pytest.raises(ValueError, match="finite wealth"):
        reward.calculate(portfolio_state(100, 100), None, portfolio_state(invalid_wealth, 100))


@pytest.mark.parametrize("risk_aversion", [np.nan, np.inf, -np.inf])
def test_invalid_risk_aversion(risk_aversion):
    with pytest.raises(ValueError, match="risk_aversion must be finite"):
        DenseExponentialUtility(risk_aversion=risk_aversion)


@pytest.mark.parametrize("num_trajectories", [1, 3])
@pytest.mark.parametrize("risk_aversion", [-0.001, 0.0, 0.001])
def test_environment_fees_price_moves_rebalancing_and_reset(monkeypatch, num_trajectories, risk_aversion):
    n_steps = 4
    midprice = BrownianMotionMidpriceModel(
        initial_price=100.0, volatility=0.0, step_size=1.0 / n_steps,
        num_trajectories=num_trajectories,
    )
    arrivals = PoissonArrivalModel(
        intensity=np.zeros(2), step_size=1.0 / n_steps,
        num_trajectories=num_trajectories,
    )
    model = UniswapV3ModelDynamics(
        midprice_model=midprice, arrival_model=arrivals,
        num_trajectories=num_trajectories, num_ticks=100, gas_cost=10.0, seed=42,
    )
    env = AMMEnvironment(
        model_dynamics=model, num_trajectories=num_trajectories, n_steps=n_steps,
        initial_wealth=1000.0,
        reward_function=DenseExponentialUtility(risk_aversion=risk_aversion),
    )
    initial, _ = env.reset(seed=42)
    initial_wealth = initial[INITIAL_WEALTH_KEY].copy()
    deploy = np.tile([-5, 5, -1], (num_trajectories, 1))
    hold = np.tile([-5, 5, 1], (num_trajectories, 1))

    # Initial deployment preserves wealth. It does not reset the cash benchmark.
    state, rewards, _, _, _ = env.step(deploy)
    np.testing.assert_allclose(rewards, 0, atol=1e-12)
    total = rewards.copy()

    # A matched sell/buy accrues fees while returning to the starting pool tick.
    monkeypatch.setattr(arrivals, "get_arrivals", lambda: np.ones((num_trajectories, 2), dtype=bool))
    state, rewards, _, _, _ = env.step(hold)
    assert np.all(state[LP_UNCLAIMED_FEES0_KEY] > 0)
    assert np.all(state[LP_UNCLAIMED_FEES1_KEY] > 0)
    assert np.all(rewards > 0)
    total += rewards

    # Without trades, existing token0 fees also gain value as the market rises.
    monkeypatch.setattr(arrivals, "get_arrivals", lambda: np.zeros((num_trajectories, 2), dtype=bool))
    before = {key: value.copy() for key, value in state.items()}
    midprice.drift = 4.0
    state, rewards, _, _, _ = env.step(hold)
    delta_price = state[ASSET_PRICE_KEY] - before[ASSET_PRICE_KEY]
    expected_pnl = (before[LP_TOKEN0_AMOUNT_KEY] + before[LP_UNCLAIMED_FEES0_KEY]) * delta_price
    np.testing.assert_allclose(state[PORTFOLIO_VALUE_KEY] - before[PORTFOLIO_VALUE_KEY], expected_pnl)
    assert np.all(rewards > 0)
    total += rewards

    # Reinvesting fees is not new income; with no market move, only gas is paid.
    midprice.drift = 0.0
    before_value = state[PORTFOLIO_VALUE_KEY].copy()
    state, rewards, terminated, _, _ = env.step(deploy)
    np.testing.assert_allclose(state[PORTFOLIO_VALUE_KEY] - before_value, -10.0)
    assert np.all(rewards < 0)
    assert terminated.all()
    total += rewards
    final_pnl = state[PORTFOLIO_VALUE_KEY] - initial_wealth
    expected = final_pnl if risk_aversion == 0.0 else -np.expm1(-risk_aversion * final_pnl) / risk_aversion
    np.testing.assert_allclose(
        total, expected,
        rtol=1e-11, atol=1e-14,
    )

    # A new episode starts in cash and earns no reward merely from price changes.
    env.reset(seed=42)
    midprice.drift = 4.0
    state, rewards, _, _, _ = env.step(hold)
    np.testing.assert_array_equal(state[PORTFOLIO_VALUE_KEY], initial_wealth)
    np.testing.assert_array_equal(rewards, np.zeros(num_trajectories))
