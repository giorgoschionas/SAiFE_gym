"""Background liquidity survives resets independently of LP position capital."""

import numpy as np
import pytest

from experiments import helpers, train_robust_lp_agent as training
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ArbitrageurEnvironment import build_arbitrageur_environment
from SAiFE_gym.gym.index_names import (
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
)


@pytest.mark.parametrize("num_trajectories", [1, 3])
@pytest.mark.parametrize("background", [0.0, 30_000.0])
def test_lp_rebalances_add_to_background_and_reset_restores_it(num_trajectories, background):
    env = AMMEnvironment(num_trajectories=num_trajectories, n_steps=5,
                         initial_wealth=1000, background_liquidity=background, seed=7)
    obs, _ = env.reset(seed=7)
    assert obs[POOL_LIQUIDITY_ARRAY_KEY].dtype == np.float64
    np.testing.assert_array_equal(obs[POOL_LIQUIDITY_ARRAY_KEY], background)
    for offsets in ((-2, 2), (-1, 3)):
        obs, *_ = env.step(np.tile([*offsets, -1.0], (num_trajectories, 1)))
        ticks = env.model_dynamics.tick_lower_global + np.arange(env.model_dynamics.num_ticks)
        covered = ((ticks >= obs[LP_TICK_LOWER_KEY][:, None])
                   & (ticks < obs[LP_TICK_UPPER_KEY][:, None]))
        np.testing.assert_allclose(obs[POOL_LIQUIDITY_ARRAY_KEY],
                                   background + obs[LP_LIQUIDITY_KEY][:, None] * covered,
                                   rtol=1e-12, atol=1e-8)
    obs, _ = env.reset(seed=8)
    np.testing.assert_array_equal(obs[POOL_LIQUIDITY_ARRAY_KEY], background)
    np.testing.assert_array_equal(obs[LP_LIQUIDITY_KEY], 0.0)
    env.close()


@pytest.mark.parametrize("background", [0.0, 30_000.0])
def test_arbitrage_factory_and_reset_use_configured_background(background):
    env = build_arbitrageur_environment(num_trajectories=3, n_steps=5, num_ticks=64,
                                        background_liquidity=background, volatility=0)
    env.reset()
    np.testing.assert_array_equal(env.state[POOL_LIQUIDITY_ARRAY_KEY], background)
    env.state[POOL_LIQUIDITY_ARRAY_KEY][:, 0] += 50
    env.reset()
    np.testing.assert_array_equal(env.state[POOL_LIQUIDITY_ARRAY_KEY], background)
    env.close()


@pytest.mark.parametrize("factory", [AMMEnvironment, build_arbitrageur_environment])
def test_general_environment_defaults_retain_legacy_background(factory):
    env = factory(num_trajectories=2, n_steps=5)
    env.reset()
    np.testing.assert_array_equal(env.state[POOL_LIQUIDITY_ARRAY_KEY], 100_000.0)
    env.close()


@pytest.mark.parametrize("factory", [AMMEnvironment, build_arbitrageur_environment])
@pytest.mark.parametrize("background", [-1, np.nan, np.inf, -np.inf, [10_000, 20_000], None])
def test_invalid_background_rejected_before_simulation(factory, background):
    with pytest.raises(ValueError, match="background_liquidity"):
        factory(background_liquidity=background)


def test_common_helper_forwards_background_liquidity():
    env = helpers.get_amm_env(num_trajectories=2, n_steps=5, background_liquidity=12_345.0)
    state, _ = env.reset()
    np.testing.assert_array_equal(state[POOL_LIQUIDITY_ARRAY_KEY], 12_345.0)
    env.close()


def test_randomized_training_preserves_background_across_episode_domains():
    args = training.parse_args(["--background-liquidity", "42000", "--num-trajectories", "3",
                                "--n-steps", "5", "--train-domains-per-reset", "2"])
    env = training.make_domain_randomized_env(args)
    for seed in (7, 8):
        state, _ = env.reset(seed=seed)
        np.testing.assert_array_equal(state[POOL_LIQUIDITY_ARRAY_KEY], 42_000.0)
        env.step(np.tile([-2.0, 2.0, -1.0], (3, 1)))
    env.close()


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_invalid_training_background_is_rejected_during_configuration_validation(value):
    args = training.parse_args(["--background-liquidity", value])
    with pytest.raises(ValueError, match="background_liquidity"):
        training.validate_evaluation_configuration(args)
