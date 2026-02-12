"""
Tests for PnL reward function.

Tests verify:
1. Zero reward when position and price unchanged
2. Positive reward when price increases within LP range
3. Initial state (LP_LIQUIDITY=0) uses initial_wealth
4. Rebalancing cost reduces reward
5. Vectorized: multiple trajectories produce per-trajectory rewards
"""

import numpy as np
import pytest

from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY
)
from SAiFE_gym.gym.helpers.AMM_utils import get_position_value_vec
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel


EXPONENTIAL_VALUE = 1.0001


def make_state(num_traj=1, price=100.0, lp_liquidity=1e6, lp_lower_tick=None,
               lp_upper_tick=None, num_ticks=100):
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
        ASSET_PRICE_KEY: np.full(num_traj, price, dtype=np.float64),
        TIME_KEY: np.zeros(num_traj, dtype=np.float64),
    }


class TestZeroReward:
    """Reward should be zero when nothing changes."""

    def test_zero_reward_no_price_change(self):
        """Same position, same price -> reward ~ 0."""
        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE)
        state = make_state(price=100.0, lp_liquidity=1e6)
        reward = reward_fn.calculate(state, None, state)
        assert np.isclose(reward[0], 0.0, atol=1e-10)


class TestPositiveReward:
    """Reward should be positive when portfolio value increases."""

    def test_positive_reward_price_increase_in_range(self):
        """Price increases within LP range -> positive reward."""
        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE)

        price_before = 100.0
        price_after = 101.0  # ~1% increase, still within LP range

        state_before = make_state(price=price_before, lp_liquidity=1e6)
        state_after = make_state(price=price_after, lp_liquidity=1e6,
                                 lp_lower_tick=int(state_before[LP_TICK_LOWER_KEY][0]),
                                 lp_upper_tick=int(state_before[LP_TICK_UPPER_KEY][0]))

        reward = reward_fn.calculate(state_before, None, state_after)
        assert reward[0] > 0, f"Expected positive reward, got {reward[0]}"


class TestInitialWealth:
    """When LP_LIQUIDITY=0, portfolio value should equal initial_wealth."""

    def test_initial_state_uses_initial_wealth(self):
        """Portfolio value for zero-liquidity state == initial_wealth."""
        initial_wealth = 5e5
        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE, initial_wealth=initial_wealth)

        state_no_pos = make_state(price=100.0, lp_liquidity=0.0)
        value = reward_fn._portfolio_value(state_no_pos)
        assert np.isclose(value[0], initial_wealth)

    def test_reward_first_deployment(self):
        """Reward from no-position to deployed position should reflect value change."""
        initial_wealth = 1e6
        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE, initial_wealth=initial_wealth)

        state_before = make_state(price=100.0, lp_liquidity=0.0)

        # After deployment, position value equals initial_wealth (same price)
        # Build a state where position value matches initial_wealth
        tick = int(state_before[POOL_CURRENT_TICK_KEY][0])
        lp_lower = tick - 5
        lp_upper = tick + 5
        sqrt_p = np.sqrt(100.0)
        sqrt_p_lower = np.sqrt(EXPONENTIAL_VALUE ** lp_lower)
        sqrt_p_upper = np.sqrt(EXPONENTIAL_VALUE ** lp_upper)

        # Compute liquidity that yields position value == initial_wealth
        v_per_unit = get_position_value_vec(
            np.array([1.0]), np.array([100.0]), np.array([sqrt_p]),
            np.array([sqrt_p_lower]), np.array([sqrt_p_upper])
        )[0]
        lp_liq = initial_wealth / v_per_unit

        state_after = make_state(price=100.0, lp_liquidity=lp_liq,
                                 lp_lower_tick=lp_lower, lp_upper_tick=lp_upper)

        reward = reward_fn.calculate(state_before, None, state_after)
        # Position value == initial_wealth, so reward should be ~0
        assert np.isclose(reward[0], 0.0, atol=1e-6)


class TestRebalancingCost:
    """Rebalancing cost reduces position value, which should show as lower reward."""

    def test_reward_reflects_rebalancing_cost(self):
        """With rebalance_cost, position value is lower -> reward is reduced."""
        initial_wealth = 1e6
        cost_coeff = 0.01  # 1%

        model_no_cost = _create_test_model(initial_wealth=initial_wealth,
                                            rebalance_cost_coeff=0.0)
        model_with_cost = _create_test_model(initial_wealth=initial_wealth,
                                              rebalance_cost_coeff=cost_coeff)

        _initialize_model_state(model_no_cost)
        _initialize_model_state(model_with_cost)

        action = np.array([[-2, 2]], dtype=np.float64)
        arrivals = np.array([[0, 0]], dtype=np.int64)

        # First rebalance (no cost for either — initial deployment)
        model_no_cost.update_state(arrivals, action)
        model_with_cost.update_state(arrivals, action)

        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE, initial_wealth=initial_wealth)

        # Snapshot state after first rebalance
        state_before_no_cost = {k: v.copy() for k, v in model_no_cost.state.items()}
        state_before_with_cost = {k: v.copy() for k, v in model_with_cost.state.items()}

        # Second rebalance (cost applied for model_with_cost)
        model_no_cost.update_state(arrivals, action)
        model_with_cost.update_state(arrivals, action)

        reward_no_cost = reward_fn.calculate(
            state_before_no_cost, action, model_no_cost.state)
        reward_with_cost = reward_fn.calculate(
            state_before_with_cost, action, model_with_cost.state)

        # Cost model should have lower (more negative) reward
        assert reward_with_cost[0] < reward_no_cost[0]


class TestVectorized:
    """Reward function should produce per-trajectory rewards."""

    def test_reward_vectorized(self):
        """Multiple trajectories with different prices produce different rewards."""
        reward_fn = PnL(exponential_value=EXPONENTIAL_VALUE)

        # Common LP position parameters
        tick_100 = int(np.floor(np.log(100.0) / np.log(EXPONENTIAL_VALUE)))
        lp_lower = tick_100 - 10
        lp_upper = tick_100 + 10

        state_before = make_state(num_traj=3, price=100.0, lp_liquidity=1e6,
                                  lp_lower_tick=lp_lower, lp_upper_tick=lp_upper)

        # Different price movements per trajectory
        prices_after = np.array([101.0, 100.0, 99.0])
        state_after = make_state(num_traj=3, price=100.0, lp_liquidity=1e6,
                                 lp_lower_tick=lp_lower, lp_upper_tick=lp_upper)
        state_after[ASSET_PRICE_KEY] = prices_after
        state_after[POOL_SQRT_PRICE_KEY] = np.sqrt(prices_after)

        reward = reward_fn.calculate(state_before, None, state_after)

        assert reward.shape == (3,)
        assert reward[0] > 0     # price went up
        assert np.isclose(reward[1], 0.0, atol=1e-10)  # no change
        assert reward[2] < 0     # price went down


# ---- Helpers for integration-style tests using ModelDynamics ----

def _create_test_model(num_trajectories=1, initial_wealth=1e6,
                        rebalance_cost_coeff=0.0):
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0, volatility=0.0, initial_price=100.0,
        terminal_time=1.0, step_size=0.005,
        num_trajectories=num_trajectories
    )
    arrival_model = PoissonArrivalModel(
        intensity=np.array([100.0, 100.0]),
        step_size=0.005, num_trajectories=num_trajectories, seed=42
    )
    model = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=0.003, tau=5, num_ticks=100,
        exponential_value=EXPONENTIAL_VALUE,
        initial_wealth=initial_wealth, seed=42
    )
    model.rebalance_cost_coeff = rebalance_cost_coeff
    return model


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
        ASSET_PRICE_KEY: np.full(num_traj, initial_price, dtype=np.float64),
        TIME_KEY: np.zeros(num_traj, dtype=np.float64),
    }
    model._xi_stale = True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
