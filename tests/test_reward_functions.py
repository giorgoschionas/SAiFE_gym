"""
Tests for reward functions: PnL, ExponentialUtility, RunningInventoryPenalty.

Tests verify:
1. PnL: zero reward, positive reward, initial wealth, rebalancing cost, vectorized
2. ExponentialUtility: zero per-step, negative terminal, risk aversion scaling
3. RunningInventoryPenalty: zero aversion = PnL, penalty reduces reward, terminal penalty
4. _token0_amount: correct computation in-range, above, below
"""

import numpy as np
import pytest

from SAiFE_gym.rewards.RewardFunctions import (
    PnL, ExponentialUtility, RunningInventoryPenalty, CjCriterion,
    _token0_amount
)
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
    LP_FEE_SNAPSHOT0_KEY, LP_FEE_SNAPSHOT1_KEY
)
from SAiFE_gym.gym.helpers.AMM_utils import get_position_value_vec
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel


EXPONENTIAL_VALUE = 1.0001


def make_state(num_traj=1, price=100.0, lp_liquidity=1e6, lp_lower_tick=None,
               lp_upper_tick=None, num_ticks=100, time=0.0):
    """Build a minimal dict state for testing the reward function."""
    tick = int(np.floor(np.log(price) / np.log(EXPONENTIAL_VALUE)))
    if lp_lower_tick is None:
        lp_lower_tick = tick - 5
    if lp_upper_tick is None:
        lp_upper_tick = tick + 5

    return {
        POOL_SQRT_PRICE_KEY: np.full(num_traj, np.sqrt(price), dtype=np.float64),
        POOL_CURRENT_TICK_KEY: np.full(num_traj, tick, dtype=np.float64),
        POOL_LIQUIDITY_ARRAY_KEY: np.full((num_traj, num_ticks), 1e6, dtype=np.float64),
        FEES0_KEY: np.zeros((num_traj, num_ticks), dtype=np.float64),
        FEES1_KEY: np.zeros((num_traj, num_ticks), dtype=np.float64),
        LP_LIQUIDITY_KEY: np.full(num_traj, lp_liquidity, dtype=np.float64),
        LP_TICK_LOWER_KEY: np.full(num_traj, lp_lower_tick, dtype=np.float64),
        LP_TICK_UPPER_KEY: np.full(num_traj, lp_upper_tick, dtype=np.float64),
        LP_COLLECTED_FEES0_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_COLLECTED_FEES1_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_FEE_SNAPSHOT0_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_FEE_SNAPSHOT1_KEY: np.zeros(num_traj, dtype=np.float64),
        ASSET_PRICE_KEY: np.full(num_traj, price, dtype=np.float64),
        TIME_KEY: np.full(num_traj, time, dtype=np.float64),
    }


# =====================================================================
# PnL tests
# =====================================================================

class TestPnLZeroReward:
    def test_zero_reward_no_price_change(self):
        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE)
        state = make_state(price=100.0, lp_liquidity=1e6)
        reward = reward_fn.calculate(state, None, state)
        assert np.isclose(reward[0], 0.0, atol=1e-10)


class TestPnLPositiveReward:
    def test_positive_reward_price_increase_in_range(self):
        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE)
        state_before = make_state(price=100.0, lp_liquidity=1e6)
        state_after = make_state(price=101.0, lp_liquidity=1e6,
                                 lp_lower_tick=int(state_before[LP_TICK_LOWER_KEY][0]),
                                 lp_upper_tick=int(state_before[LP_TICK_UPPER_KEY][0]))
        reward = reward_fn.calculate(state_before, None, state_after)
        assert reward[0] > 0


class TestPnLInitialWealth:
    def test_initial_state_uses_initial_wealth(self):
        initial_wealth = 5e5
        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE, initial_wealth=initial_wealth)
        state_no_pos = make_state(price=100.0, lp_liquidity=0.0)
        value = reward_fn._portfolio_value(state_no_pos)
        assert np.isclose(value[0], initial_wealth)

    def test_reward_first_deployment(self):
        initial_wealth = 1e6
        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE, initial_wealth=initial_wealth)
        state_before = make_state(price=100.0, lp_liquidity=0.0)

        tick = int(state_before[POOL_CURRENT_TICK_KEY][0])
        lp_lower, lp_upper = tick - 5, tick + 5
        sqrt_p = np.sqrt(100.0)
        sqrt_p_lower = np.sqrt(EXPONENTIAL_VALUE ** lp_lower)
        sqrt_p_upper = np.sqrt(EXPONENTIAL_VALUE ** lp_upper)

        v_per_unit = get_position_value_vec(
            np.array([1.0]), np.array([100.0]), np.array([sqrt_p]),
            np.array([sqrt_p_lower]), np.array([sqrt_p_upper])
        )[0]
        lp_liq = initial_wealth / v_per_unit

        state_after = make_state(price=100.0, lp_liquidity=lp_liq,
                                 lp_lower_tick=lp_lower, lp_upper_tick=lp_upper)
        reward = reward_fn.calculate(state_before, None, state_after)
        assert np.isclose(reward[0], 0.0, atol=1e-6)


class TestPnLRebalancingCost:
    def test_reward_reflects_rebalancing_cost(self):
        initial_wealth = 1e6
        model_no_cost = _create_test_model(initial_wealth=initial_wealth,
                                            gas_cost=0.0)
        model_with_cost = _create_test_model(initial_wealth=initial_wealth,
                                              gas_cost=10.0)
        _initialize_model_state(model_no_cost)
        _initialize_model_state(model_with_cost)

        action = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)
        model_no_cost.update_state(arrivals, action)
        model_with_cost.update_state(arrivals, action)

        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE, initial_wealth=initial_wealth)
        state_before_no_cost = {k: v.copy() for k, v in model_no_cost.state.items()}
        state_before_with_cost = {k: v.copy() for k, v in model_with_cost.state.items()}

        model_no_cost.update_state(arrivals, action)
        model_with_cost.update_state(arrivals, action)

        reward_no_cost = reward_fn.calculate(state_before_no_cost, action, model_no_cost.state)
        reward_with_cost = reward_fn.calculate(state_before_with_cost, action, model_with_cost.state)
        assert reward_with_cost[0] < reward_no_cost[0]


class TestPnLVectorized:
    def test_reward_vectorized(self):
        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE)
        tick_100 = int(np.floor(np.log(100.0) / np.log(EXPONENTIAL_VALUE)))
        lp_lower, lp_upper = tick_100 - 10, tick_100 + 10

        state_before = make_state(num_traj=3, price=100.0, lp_liquidity=1e6,
                                  lp_lower_tick=lp_lower, lp_upper_tick=lp_upper)
        state_after = make_state(num_traj=3, price=100.0, lp_liquidity=1e6,
                                 lp_lower_tick=lp_lower, lp_upper_tick=lp_upper)
        state_after[ASSET_PRICE_KEY] = np.array([101.0, 100.0, 99.0])
        state_after[POOL_SQRT_PRICE_KEY] = np.sqrt(state_after[ASSET_PRICE_KEY])

        reward = reward_fn.calculate(state_before, None, state_after)
        assert reward.shape == (3,)
        assert reward[0] > 0
        assert np.isclose(reward[1], 0.0, atol=1e-10)
        assert reward[2] < 0


# =====================================================================
# _token0_amount tests
# =====================================================================

class TestToken0Amount:
    """Test the LP inventory (token0 holdings) computation."""

    def test_in_range_positive(self):
        """Price in range: LP holds some token0."""
        state = make_state(price=100.0, lp_liquidity=1e6)
        x = _token0_amount(state, EXPONENTIAL_VALUE)
        assert x[0] > 0

    def test_above_range_zero(self):
        """Price above range: LP holds no token0 (100% token1)."""
        tick = int(np.floor(np.log(100.0) / np.log(EXPONENTIAL_VALUE)))
        # Position range is [tick-5, tick+5], price at tick+10 → way above
        state = make_state(price=100.0, lp_liquidity=1e6,
                           lp_lower_tick=tick - 15, lp_upper_tick=tick - 10)
        x = _token0_amount(state, EXPONENTIAL_VALUE)
        assert np.isclose(x[0], 0.0)

    def test_below_range_max(self):
        """Price below range: LP holds maximum token0."""
        tick = int(np.floor(np.log(100.0) / np.log(EXPONENTIAL_VALUE)))
        # Position range is [tick+10, tick+15], price at tick → below
        state = make_state(price=100.0, lp_liquidity=1e6,
                           lp_lower_tick=tick + 10, lp_upper_tick=tick + 15)
        x = _token0_amount(state, EXPONENTIAL_VALUE)
        # Should equal L * (1/sqrt_p_lower - 1/sqrt_p_upper)
        sqrt_p_l = np.sqrt(EXPONENTIAL_VALUE ** (tick + 10))
        sqrt_p_u = np.sqrt(EXPONENTIAL_VALUE ** (tick + 15))
        expected = 1e6 * (1.0 / sqrt_p_l - 1.0 / sqrt_p_u)
        assert np.isclose(x[0], expected, rtol=1e-10)

    def test_zero_liquidity(self):
        """No position: token0 amount is zero."""
        state = make_state(price=100.0, lp_liquidity=0.0)
        x = _token0_amount(state, EXPONENTIAL_VALUE)
        assert np.isclose(x[0], 0.0)

    def test_vectorized_mixed(self):
        """Multiple trajectories in different regimes."""
        tick = int(np.floor(np.log(100.0) / np.log(EXPONENTIAL_VALUE)))
        state = make_state(num_traj=3, price=100.0, lp_liquidity=1e6)
        # Trajectory 0: in range (default)
        # Trajectory 1: above range (shrink upper bound below current price)
        state[LP_TICK_LOWER_KEY][1] = tick - 15
        state[LP_TICK_UPPER_KEY][1] = tick - 10
        # Trajectory 2: below range (push lower bound above current price)
        state[LP_TICK_LOWER_KEY][2] = tick + 10
        state[LP_TICK_UPPER_KEY][2] = tick + 15

        x = _token0_amount(state, EXPONENTIAL_VALUE)
        assert x[0] > 0         # in range
        assert np.isclose(x[1], 0.0)  # above range
        assert x[2] > x[0]      # below range → max exposure


# =====================================================================
# ExponentialUtility tests
# =====================================================================

class TestExponentialUtility:
    """Terminal CARA utility on portfolio value."""

    def test_zero_reward_non_terminal(self):
        """Non-terminal steps should return 0."""
        reward_fn = ExponentialUtility(risk_aversion=0.1,
                                       exponential_value=EXPONENTIAL_VALUE)
        state = make_state(price=100.0, lp_liquidity=1e6)
        reward = reward_fn.calculate(state, None, state, is_terminal_step=False)
        assert np.all(reward == 0.0)

    def test_negative_reward_terminal(self):
        """Terminal step should return -exp(-a * W_T) < 0."""
        reward_fn = ExponentialUtility(risk_aversion=0.1,
                                       exponential_value=EXPONENTIAL_VALUE)
        state = make_state(price=100.0, lp_liquidity=1e6)
        reward = reward_fn.calculate(state, None, state, is_terminal_step=True)
        assert reward[0] < 0

    def test_concavity(self):
        """Risk-averse utility is concave: gain from +delta < loss from -delta."""
        rf = ExponentialUtility(risk_aversion=1e-6,
                                exponential_value=EXPONENTIAL_VALUE)

        state_mid = make_state(price=100.0, lp_liquidity=1e6)
        state_high = make_state(price=101.0, lp_liquidity=1e6,
                                lp_lower_tick=int(state_mid[LP_TICK_LOWER_KEY][0]),
                                lp_upper_tick=int(state_mid[LP_TICK_UPPER_KEY][0]))
        state_low = make_state(price=99.0, lp_liquidity=1e6,
                               lp_lower_tick=int(state_mid[LP_TICK_LOWER_KEY][0]),
                               lp_upper_tick=int(state_mid[LP_TICK_UPPER_KEY][0]))

        r_mid = rf.calculate(state_mid, None, state_mid, is_terminal_step=True)[0]
        r_high = rf.calculate(state_high, None, state_high, is_terminal_step=True)[0]
        r_low = rf.calculate(state_low, None, state_low, is_terminal_step=True)[0]

        # Concavity: u(mid) > (u(high) + u(low)) / 2
        assert r_mid > (r_high + r_low) / 2

    def test_higher_wealth_less_negative(self):
        """Higher portfolio value → less negative terminal reward (closer to 0)."""
        rf = ExponentialUtility(risk_aversion=1e-6,
                                exponential_value=EXPONENTIAL_VALUE)

        state_low = make_state(price=100.0, lp_liquidity=1e5)
        state_high = make_state(price=100.0, lp_liquidity=1e6)

        r_low = rf.calculate(state_low, None, state_low, is_terminal_step=True)
        r_high = rf.calculate(state_high, None, state_high, is_terminal_step=True)

        # -exp(-a * W): higher W → less negative
        assert r_high[0] > r_low[0]

    def test_vectorized(self):
        """Multiple trajectories produce per-trajectory rewards."""
        rf = ExponentialUtility(risk_aversion=1e-6,
                                exponential_value=EXPONENTIAL_VALUE)
        state = make_state(num_traj=3, price=100.0, lp_liquidity=1e6)
        reward = rf.calculate(state, None, state, is_terminal_step=True)
        assert reward.shape == (3,)
        assert np.all(reward < 0)


# =====================================================================
# RunningInventoryPenalty tests
# =====================================================================

class TestRunningInventoryPenalty:
    """PnL with running inventory (token0 exposure) penalty."""

    def test_zero_aversion_equals_pnl(self):
        """With zero aversion, RunningInventoryPenalty == PnL."""
        pnl_fn = PnL(exponential_value=EXPONENTIAL_VALUE)
        rip_fn = RunningInventoryPenalty(
            per_step_inventory_aversion=0.0,
            terminal_inventory_aversion=0.0,
            exponential_value=EXPONENTIAL_VALUE
        )

        state_before = make_state(price=100.0, lp_liquidity=1e6, time=0.0)
        state_after = make_state(price=101.0, lp_liquidity=1e6, time=0.005,
                                 lp_lower_tick=int(state_before[LP_TICK_LOWER_KEY][0]),
                                 lp_upper_tick=int(state_before[LP_TICK_UPPER_KEY][0]))

        r_pnl = pnl_fn.calculate(state_before, None, state_after)
        r_rip = rip_fn.calculate(state_before, None, state_after)
        assert np.isclose(r_pnl[0], r_rip[0], rtol=1e-12)

    def test_penalty_reduces_reward(self):
        """Non-zero per-step aversion should reduce reward vs plain PnL."""
        tick = int(np.floor(np.log(100.0) / np.log(EXPONENTIAL_VALUE)))
        # Wide range so price 100.1 stays well within [tick-500, tick+500]
        lp_lower, lp_upper = tick - 500, tick + 500

        pnl_fn = PnL(exponential_value=EXPONENTIAL_VALUE)
        rip_fn = RunningInventoryPenalty(
            per_step_inventory_aversion=100.0,
            exponential_value=EXPONENTIAL_VALUE
        )

        state_before = make_state(price=100.0, lp_liquidity=1e6, time=0.0,
                                  lp_lower_tick=lp_lower, lp_upper_tick=lp_upper)
        state_after = make_state(price=100.1, lp_liquidity=1e6, time=0.005,
                                 lp_lower_tick=lp_lower, lp_upper_tick=lp_upper)

        r_pnl = pnl_fn.calculate(state_before, None, state_after)
        r_rip = rip_fn.calculate(state_before, None, state_after)
        assert r_rip[0] < r_pnl[0]

    def test_terminal_penalty_fires_only_at_terminal(self):
        """Terminal inventory penalty only applies when is_terminal_step=True."""
        rip_fn = RunningInventoryPenalty(
            per_step_inventory_aversion=0.0,
            terminal_inventory_aversion=1.0,
            exponential_value=EXPONENTIAL_VALUE
        )

        state_before = make_state(price=100.0, lp_liquidity=1e6, time=0.0)
        state_after = make_state(price=100.0, lp_liquidity=1e6, time=0.005,
                                 lp_lower_tick=int(state_before[LP_TICK_LOWER_KEY][0]),
                                 lp_upper_tick=int(state_before[LP_TICK_UPPER_KEY][0]))

        r_non_terminal = rip_fn.calculate(state_before, None, state_after,
                                           is_terminal_step=False)
        r_terminal = rip_fn.calculate(state_before, None, state_after,
                                       is_terminal_step=True)

        # Non-terminal: only PnL (which is ~0 since same price), no terminal penalty
        assert np.isclose(r_non_terminal[0], 0.0, atol=1e-10)
        # Terminal: PnL (~0) minus terminal penalty (negative)
        assert r_terminal[0] < r_non_terminal[0]

    def test_no_penalty_when_no_position(self):
        """With LP_LIQUIDITY=0 (no position), inventory is 0 → no penalty."""
        rip_fn = RunningInventoryPenalty(
            per_step_inventory_aversion=1.0,
            terminal_inventory_aversion=1.0,
            exponential_value=EXPONENTIAL_VALUE
        )

        state_before = make_state(price=100.0, lp_liquidity=0.0, time=0.0)
        state_after = make_state(price=100.0, lp_liquidity=0.0, time=0.005)

        r = rip_fn.calculate(state_before, None, state_after, is_terminal_step=True)
        # PnL is 0 (both states have initial_wealth), inventory is 0 → penalty is 0
        assert np.isclose(r[0], 0.0, atol=1e-10)

    def test_vectorized(self):
        """Multiple trajectories: higher liquidity → bigger penalty."""
        tick = int(np.floor(np.log(100.0) / np.log(EXPONENTIAL_VALUE)))
        lp_lower, lp_upper = tick - 500, tick + 500

        rip_fn = RunningInventoryPenalty(
            per_step_inventory_aversion=100.0,
            exponential_value=EXPONENTIAL_VALUE
        )

        # Both before & after have same liquidity per trajectory (no PnL from liq change)
        state_before = make_state(num_traj=2, price=100.0, lp_liquidity=1e6,
                                  lp_lower_tick=lp_lower, lp_upper_tick=lp_upper,
                                  time=0.0)
        state_after = make_state(num_traj=2, price=100.0, lp_liquidity=1e6,
                                 lp_lower_tick=lp_lower, lp_upper_tick=lp_upper,
                                 time=0.005)
        # Give trajectory 1 higher liquidity in BOTH states → same PnL but more penalty
        state_before[LP_LIQUIDITY_KEY][1] = 1e7
        state_after[LP_LIQUIDITY_KEY][1] = 1e7

        r = rip_fn.calculate(state_before, None, state_after)
        assert r.shape == (2,)
        # Trajectory 1 has more inventory → bigger penalty → lower reward
        assert r[1] < r[0]


class TestCjCriterionAlias:
    def test_alias(self):
        assert CjCriterion is RunningInventoryPenalty


# =====================================================================
# Helpers for integration tests using ModelDynamics
# =====================================================================

def _create_test_model(num_trajectories=1, initial_wealth=1e6, gas_cost=0.0):
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0, volatility=0.0, initial_price=100.0,
        terminal_time=1.0, step_size=0.005,
        num_trajectories=num_trajectories
    )
    arrival_model = PoissonArrivalModel(
        intensity=np.array([100.0, 100.0]),
        step_size=0.005, num_trajectories=num_trajectories, seed=42
    )
    return UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=0.003, tau=5, num_ticks=100,
        exponential_value=EXPONENTIAL_VALUE,
        initial_wealth=initial_wealth,
        gas_cost=gas_cost,
        seed=42
    )


def _initialize_model_state(model):
    num_traj = model.num_trajectories
    num_ticks = model.num_ticks
    initial_price = model.initial_price
    initial_tick = int(np.floor(np.log(initial_price) / np.log(model.exponential_value)))
    model.tick_lower_global = initial_tick - num_ticks // 2

    p_low = model.exponential_value ** initial_tick
    p_high = model.exponential_value ** (initial_tick + 1)
    initial_sqrt_price = np.sqrt((p_low + p_high) / 2)

    model.state = {
        POOL_SQRT_PRICE_KEY: np.full(num_traj, initial_sqrt_price, dtype=np.float64),
        POOL_CURRENT_TICK_KEY: np.full(num_traj, initial_tick, dtype=np.float64),
        POOL_LIQUIDITY_ARRAY_KEY: np.full((num_traj, num_ticks), 1e6, dtype=np.float64),
        FEES0_KEY: np.zeros((num_traj, num_ticks), dtype=np.float64),
        FEES1_KEY: np.zeros((num_traj, num_ticks), dtype=np.float64),
        LP_LIQUIDITY_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_TICK_LOWER_KEY: np.full(num_traj, initial_tick - model.tau, dtype=np.float64),
        LP_TICK_UPPER_KEY: np.full(num_traj, initial_tick + model.tau, dtype=np.float64),
        LP_COLLECTED_FEES0_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_COLLECTED_FEES1_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_FEE_SNAPSHOT0_KEY: np.zeros(num_traj, dtype=np.float64),
        LP_FEE_SNAPSHOT1_KEY: np.zeros(num_traj, dtype=np.float64),
        ASSET_PRICE_KEY: np.full(num_traj, initial_price, dtype=np.float64),
        TIME_KEY: np.zeros(num_traj, dtype=np.float64),
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
