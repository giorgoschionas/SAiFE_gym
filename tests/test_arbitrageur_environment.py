import numpy as np
import pytest

from SAiFE_gym.gym.ArbitrageurEnvironment import (
    DEFAULT_ARBITRAGEUR_OBS_KEYS,
    LP_POLICY_DEPLOY_ONCE,
    ArbitrageurEnvironment,
    build_arbitrageur_environment,
)
from SAiFE_gym.gym.index_names import (
    LP_EVER_DEPLOYED_KEY,
    LP_LIQUIDITY_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    TIME_KEY,
)


def _zero_alpha():
    return np.array([0.0, 0.0])


def _build_env(**kwargs) -> ArbitrageurEnvironment:
    params = {
        "num_trajectories": 1,
        "n_steps": 10,
        "seed": 11,
        "volatility": 0.0,
        "fee_tier": 0.0,
        "initial_pool_price": 99.5,
        "alpha0": _zero_alpha(),
        "alpha1": _zero_alpha(),
        "alpha2": _zero_alpha(),
        "alpha3": _zero_alpha(),
        "max_speed": 1e9,
    }
    params.update(kwargs)
    return build_arbitrageur_environment(**params)


def _buy_capacity(env: ArbitrageurEnvironment, trajectory: int = 0) -> float:
    md = env.model_dynamics
    state = env.state
    idx = int(state[POOL_CURRENT_TICK_KEY][trajectory] - md.tick_lower_global)
    return state[POOL_LIQUIDITY_ARRAY_KEY][trajectory, idx] * (
        md.sqrt_grid[idx + 1] - md.sqrt_grid[idx]
    )


class TestArbitrageurEnvironment:
    def test_reset_defaults_to_no_op_lp_without_advancing_time(self):
        env = _build_env()

        obs, info = env.reset()

        assert obs.shape == (1, len(DEFAULT_ARBITRAGEUR_OBS_KEYS))
        assert env.observation_space.shape == obs.shape
        assert env.observation_space.contains(obs)
        assert info == {}
        assert env.state[TIME_KEY][0] == pytest.approx(0.0)
        assert not bool(env.state[LP_EVER_DEPLOYED_KEY][0])
        assert env.state[LP_LIQUIDITY_KEY][0] == 0.0

    def test_deploy_once_lp_policy_can_be_enabled(self):
        env = _build_env(lp_policy=LP_POLICY_DEPLOY_ONCE)

        obs, info = env.reset()

        assert obs.shape == (1, len(DEFAULT_ARBITRAGEUR_OBS_KEYS))
        assert env.observation_space.contains(obs)
        assert info == {}
        assert env.state[TIME_KEY][0] == pytest.approx(0.0)
        assert bool(env.state[LP_EVER_DEPLOYED_KEY][0])
        assert env.state[LP_LIQUIDITY_KEY][0] > 0.0

    def test_spaces_match_vectorized_observations_and_actions(self):
        env = _build_env(num_trajectories=3)
        obs, _ = env.reset()
        action = np.array([[0.0], [1.0], [-1.0]], dtype=np.float32)

        assert env.observation_space.shape == (
            3,
            len(DEFAULT_ARBITRAGEUR_OBS_KEYS),
        )
        assert env.action_space.shape == (3, 1)
        assert env.observation_space.contains(obs)
        assert env.action_space.contains(action)

        obs, rewards, terminated, truncated, _ = env.step(action)

        assert env.observation_space.contains(obs)
        assert rewards.shape == (3,)
        assert terminated.shape == (3,)
        assert truncated.shape == (3,)

    def test_sampled_action_steps_environment(self):
        env = _build_env(num_trajectories=2)
        env.reset()

        action = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(action)

        assert action.shape == (2, 1)
        assert env.action_space.contains(action)
        assert env.observation_space.contains(obs)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert info["trading_speed"].shape == (2,)

    def test_step_accepts_vectorized_speed_actions(self):
        env = _build_env(num_trajectories=3, initial_pool_price=99.5)
        obs, _ = env.reset()
        assert obs.shape == (3, len(DEFAULT_ARBITRAGEUR_OBS_KEYS))
        amount = _buy_capacity(env, trajectory=0)
        speed = amount / env.step_size

        obs, rewards, terminated, truncated, info = env.step(
            np.array([[speed], [0.0], [-speed]])
        )

        assert obs.shape == (3, len(DEFAULT_ARBITRAGEUR_OBS_KEYS))
        assert env.observation_space.contains(obs)
        assert rewards.shape == (3,)
        assert terminated.shape == (3,)
        assert truncated.shape == (3,)
        assert info["trading_speed"].shape == (3,)
        assert info["token0_delta"].shape == (3,)
        assert info["token1_delta"].shape == (3,)
        assert info["cumulative_arb_pnl"].shape == (3,)

    def test_profitable_one_tick_buy_has_positive_hedged_pnl_with_zero_fee(self):
        env = _build_env(initial_pool_price=99.5, fee_tier=0.0)
        env.reset()
        amount = _buy_capacity(env)
        speed = amount / env.step_size

        _, rewards, _, _, info = env.step(np.array([[speed]]))

        assert info["token0_delta"][0] > 0.0
        assert info["token1_delta"][0] < 0.0
        assert rewards[0] > 0.0
        assert info["hedged_pnl"][0] == pytest.approx(rewards[0])

    def test_zero_speed_has_zero_arb_reward_without_speed_cost(self):
        env = _build_env(initial_pool_price=99.5, fee_tier=0.0)
        env.reset()

        _, rewards, _, _, info = env.step(np.array([[0.0]]))

        assert rewards[0] == pytest.approx(0.0)
        assert info["token0_delta"][0] == 0.0
        assert info["token1_delta"][0] == 0.0
        assert info["speed_cost"][0] == 0.0

    def test_speed_cost_reduces_reward_by_quadratic_term(self):
        speed_cost_coefficient = 0.25
        env_no_cost = _build_env(speed_cost_coefficient=0.0)
        env_with_cost = _build_env(speed_cost_coefficient=speed_cost_coefficient)
        env_no_cost.reset()
        env_with_cost.reset()
        amount = _buy_capacity(env_no_cost)
        speed = amount / env_no_cost.step_size

        _, reward_no_cost, _, _, _ = env_no_cost.step(np.array([[speed]]))
        _, reward_with_cost, _, _, info = env_with_cost.step(np.array([[speed]]))

        expected_cost = speed_cost_coefficient * speed**2 * env_no_cost.step_size
        assert info["speed_cost"][0] == pytest.approx(expected_cost)
        assert reward_with_cost[0] == pytest.approx(reward_no_cost[0] - expected_cost)
