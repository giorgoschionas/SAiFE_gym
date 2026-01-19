"""
Tests for update_state implementation with unified_swap_single_tick.

Tests cover:
1. No arrivals -> no state change
2. Sell arrival -> price decreases
3. Buy arrival -> price increases
4. Fee accumulation
5. Time updates
6. Multiple trajectories
"""
import numpy as np
import pytest
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, TIME_KEY
)
from SAiFE_gym.gym.helpers.AMM_utils import price_to_tick


def create_test_dynamics(num_trajectories=2, initial_price=100.0):
    """Helper to create test model dynamics."""
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=2.0,
        initial_price=initial_price,
        terminal_time=1.0,
        step_size=0.005,
        num_trajectories=num_trajectories
    )

    arrival_model = PoissonArrivalModel(
        intensity=np.array([100.0, 100.0]),
        step_size=0.005,
        num_trajectories=num_trajectories
    )

    dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        tau=5,
        num_ticks=1000,
        fee_tier=0.003,
        exponential_value=1.0001
    )

    return dynamics


def initialize_test_state(dynamics, initial_price=100.0, initial_liquidity=100000.0):
    """Helper to initialize state for tests."""
    num_traj = dynamics.num_trajectories
    num_ticks = dynamics.num_ticks
    initial_tick = price_to_tick(initial_price)

    # Set tick_lower_global
    dynamics.tick_lower_global = initial_tick - num_ticks // 2

    dynamics.state = {
        POOL_SQRT_PRICE_KEY: np.full(num_traj, np.sqrt(initial_price), dtype=np.float64),
        POOL_CURRENT_TICK_KEY: np.full(num_traj, initial_tick, dtype=np.int64),
        POOL_LIQUIDITY_ARRAY_KEY: np.full((num_traj, num_ticks), initial_liquidity, dtype=np.float64),
        FEES0_KEY: np.zeros((num_traj, num_ticks), dtype=np.float64),
        FEES1_KEY: np.zeros((num_traj, num_ticks), dtype=np.float64),
        TIME_KEY: np.zeros(num_traj, dtype=np.float64),
    }


class TestUpdateStateNoArrivals:
    """Tests for update_state with no arrivals."""

    def test_no_arrivals_no_price_change(self):
        """When there are no arrivals, price should not change."""
        dynamics = create_test_dynamics(num_trajectories=2)
        initialize_test_state(dynamics)

        initial_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY].copy()

        # No arrivals
        arrivals = np.array([[False, False], [False, False]])
        action = np.ones((2, 11)) / 11.0

        dynamics.update_state(arrivals, action)

        final_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY]

        np.testing.assert_array_almost_equal(
            final_sqrt_price, initial_sqrt_price,
            err_msg="Price should not change with no arrivals"
        )

    def test_no_arrivals_no_fees(self):
        """When there are no arrivals, no fees should be collected."""
        dynamics = create_test_dynamics(num_trajectories=2)
        initialize_test_state(dynamics)

        arrivals = np.array([[False, False], [False, False]])
        action = np.ones((2, 11)) / 11.0

        dynamics.update_state(arrivals, action)

        total_fees0 = dynamics.state[FEES0_KEY].sum()
        total_fees1 = dynamics.state[FEES1_KEY].sum()

        assert total_fees0 == 0.0, "No fees should be collected in token0"
        assert total_fees1 == 0.0, "No fees should be collected in token1"


class TestUpdateStateSellArrivals:
    """Tests for update_state with sell arrivals (token0 sold to pool)."""

    def test_sell_decreases_price(self):
        """Sell arrival should decrease the price."""
        dynamics = create_test_dynamics(num_trajectories=1)
        initialize_test_state(dynamics)

        initial_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY].copy()

        # Sell arrival: token0 into pool
        arrivals = np.array([[True, False]])
        action = np.ones((1, 11)) / 11.0

        dynamics.update_state(arrivals, action)

        final_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY]

        assert final_sqrt_price[0] < initial_sqrt_price[0], \
            f"Sell should decrease price: {initial_sqrt_price[0]} -> {final_sqrt_price[0]}"

    def test_sell_collects_fees_in_token0(self):
        """Sell arrival should collect fees in token0."""
        dynamics = create_test_dynamics(num_trajectories=1)
        initialize_test_state(dynamics)

        arrivals = np.array([[True, False]])
        action = np.ones((1, 11)) / 11.0

        dynamics.update_state(arrivals, action)

        total_fees0 = dynamics.state[FEES0_KEY].sum()
        total_fees1 = dynamics.state[FEES1_KEY].sum()

        assert total_fees0 > 0.0, "Sell should collect fees in token0"
        assert total_fees1 == 0.0, "Sell should not collect fees in token1"


class TestUpdateStateBuyArrivals:
    """Tests for update_state with buy arrivals (token0 bought from pool)."""

    def test_buy_increases_price(self):
        """Buy arrival should increase the price."""
        dynamics = create_test_dynamics(num_trajectories=1)
        initialize_test_state(dynamics)

        initial_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY].copy()

        # Buy arrival: token0 out of pool
        arrivals = np.array([[False, True]])
        action = np.ones((1, 11)) / 11.0

        dynamics.update_state(arrivals, action)

        final_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY]

        assert final_sqrt_price[0] > initial_sqrt_price[0], \
            f"Buy should increase price: {initial_sqrt_price[0]} -> {final_sqrt_price[0]}"

    def test_buy_collects_fees_in_token1(self):
        """Buy arrival should collect fees in token1."""
        dynamics = create_test_dynamics(num_trajectories=1)
        initialize_test_state(dynamics)

        arrivals = np.array([[False, True]])
        action = np.ones((1, 11)) / 11.0

        dynamics.update_state(arrivals, action)

        total_fees0 = dynamics.state[FEES0_KEY].sum()
        total_fees1 = dynamics.state[FEES1_KEY].sum()

        assert total_fees0 == 0.0, "Buy should not collect fees in token0"
        assert total_fees1 > 0.0, "Buy should collect fees in token1"


class TestUpdateStateMultipleTrajectories:
    """Tests for update_state with multiple trajectories."""

    def test_independent_trajectories(self):
        """Different trajectories should update independently."""
        dynamics = create_test_dynamics(num_trajectories=2)
        initialize_test_state(dynamics)

        initial_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY].copy()

        # Traj 0: Sell (price decreases), Traj 1: Buy (price increases)
        arrivals = np.array([[True, False], [False, True]])
        action = np.ones((2, 11)) / 11.0

        dynamics.update_state(arrivals, action)

        final_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY]

        # Trajectory 0: sell -> price decreased
        assert final_sqrt_price[0] < initial_sqrt_price[0], \
            "Trajectory 0 (sell) should have decreased price"

        # Trajectory 1: buy -> price increased
        assert final_sqrt_price[1] > initial_sqrt_price[1], \
            "Trajectory 1 (buy) should have increased price"

    def test_mixed_arrivals(self):
        """Test with mixed arrival patterns across trajectories."""
        dynamics = create_test_dynamics(num_trajectories=3)
        initialize_test_state(dynamics)

        initial_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY].copy()

        # Traj 0: no arrival, Traj 1: sell, Traj 2: buy
        arrivals = np.array([
            [False, False],
            [True, False],
            [False, True]
        ])
        action = np.ones((3, 11)) / 11.0

        dynamics.update_state(arrivals, action)

        final_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY]

        # Trajectory 0: no change
        np.testing.assert_almost_equal(
            final_sqrt_price[0], initial_sqrt_price[0],
            err_msg="Trajectory 0 (no arrival) should have unchanged price"
        )

        # Trajectory 1: sell -> decreased
        assert final_sqrt_price[1] < initial_sqrt_price[1], \
            "Trajectory 1 (sell) should have decreased price"

        # Trajectory 2: buy -> increased
        assert final_sqrt_price[2] > initial_sqrt_price[2], \
            "Trajectory 2 (buy) should have increased price"


class TestUpdateStateTimeUpdate:
    """Tests for time update in update_state."""

    def test_time_increments(self):
        """Time should increment after each update_state call."""
        dynamics = create_test_dynamics(num_trajectories=1)
        initialize_test_state(dynamics)

        initial_time = dynamics.state[TIME_KEY].copy()
        step_size = dynamics.midprice_model.step_size

        arrivals = np.array([[False, False]])
        action = np.ones((1, 11)) / 11.0

        dynamics.update_state(arrivals, action)

        final_time = dynamics.state[TIME_KEY]
        expected_time = initial_time + step_size

        np.testing.assert_array_almost_equal(
            final_time, expected_time,
            err_msg=f"Time should increment by step_size ({step_size})"
        )


class TestUpdateStateTickUpdate:
    """Tests for tick update in update_state."""

    def test_tick_updates_with_price(self):
        """Current tick should update when price changes significantly."""
        dynamics = create_test_dynamics(num_trajectories=1)
        initialize_test_state(dynamics)

        initial_tick = dynamics.state[POOL_CURRENT_TICK_KEY].copy()

        # Large arrival to move price significantly
        arrivals = np.array([[True, False]])  # Sell -> price decreases
        action = np.ones((1, 11)) / 11.0

        dynamics.update_state(arrivals, action)

        final_tick = dynamics.state[POOL_CURRENT_TICK_KEY]

        # With single-tick swap hitting boundary, tick should decrease
        assert final_tick[0] <= initial_tick[0], \
            f"Tick should decrease with sell: {initial_tick[0]} -> {final_tick[0]}"


class TestUpdateStateEdgeCases:
    """Tests for edge cases in update_state."""

    def test_zero_liquidity_no_crash(self):
        """Update state should handle zero liquidity gracefully."""
        dynamics = create_test_dynamics(num_trajectories=1)
        initialize_test_state(dynamics, initial_liquidity=0.0)

        initial_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY].copy()

        arrivals = np.array([[True, False]])
        action = np.ones((1, 11)) / 11.0

        # Should not crash
        dynamics.update_state(arrivals, action)

        # Price should not change with zero liquidity
        final_sqrt_price = dynamics.state[POOL_SQRT_PRICE_KEY]
        np.testing.assert_array_almost_equal(
            final_sqrt_price, initial_sqrt_price,
            err_msg="Price should not change with zero liquidity"
        )

    def test_uninitialized_state_raises(self):
        """Update state should raise error if state is not initialized."""
        dynamics = create_test_dynamics(num_trajectories=1)
        # Don't initialize state

        arrivals = np.array([[True, False]])
        action = np.ones((1, 11)) / 11.0

        with pytest.raises(ValueError, match="State not initialized"):
            dynamics.update_state(arrivals, action)


class TestAMMEnvironmentIntegration:
    """Integration tests for AMMEnvironment with update_state."""

    def test_env_reset_initializes_state(self):
        """Environment reset should properly initialize state."""
        env = AMMEnvironment(
            terminal_time=1.0,
            n_steps=200,
            num_trajectories=1
        )

        obs = env.reset()

        assert obs is not None
        assert isinstance(obs, dict)
        assert POOL_SQRT_PRICE_KEY in obs
        assert POOL_LIQUIDITY_ARRAY_KEY in obs
        assert FEES0_KEY in obs
        assert FEES1_KEY in obs

    def test_env_step_updates_state(self):
        """Environment step should update state correctly."""
        from SAiFE_gym.rewards.RewardFunctions import RewardFunction

        # Create a dummy reward function that handles dict state
        class DummyReward(RewardFunction):
            def calculate(self, current_state, action, next_state, is_terminal_step=False):
                return np.zeros(1)
            def reset(self, initial_state):
                pass

        env = AMMEnvironment(
            terminal_time=1.0,
            n_steps=200,
            num_trajectories=1,
            reward_function=DummyReward()
        )

        obs = env.reset()
        initial_time = obs[TIME_KEY].copy()

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        # Time should have advanced
        assert obs[TIME_KEY][0] > initial_time[0], "Time should advance after step"

    def test_full_episode(self):
        """Test running a full episode."""
        from SAiFE_gym.rewards.RewardFunctions import RewardFunction

        # Create a dummy reward function that handles dict state
        class DummyReward(RewardFunction):
            def calculate(self, current_state, action, next_state, is_terminal_step=False):
                return np.zeros(1)
            def reset(self, initial_state):
                pass

        env = AMMEnvironment(
            terminal_time=0.1,  # Short episode for testing
            n_steps=20,
            num_trajectories=1,
            reward_function=DummyReward()
        )

        obs = env.reset()
        done = np.array([False])  # Initialize as array
        steps = 0

        while not done[0]:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1

            # Verify state is valid
            assert np.all(obs[POOL_SQRT_PRICE_KEY] > 0), "Price should remain positive"

        assert steps == 20, f"Should complete {env.n_steps} steps, got {steps}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
