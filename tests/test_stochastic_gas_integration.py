"""Gas timing, accounting, and lifecycle through simulator interfaces."""

from copy import deepcopy

import numpy as np
import pytest

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ArbitrageurEnvironment import ArbitrageurEnvironment
from SAiFE_gym.gym.GymnasiumAMMEnvironment import GymnasiumAMMEnvironment, GymnasiumAMMVectorEnv
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import (
    DEFAULT_OBS_KEYS,
    StableBaselinesAMMEnvironment,
)
from SAiFE_gym.gym.domain_randomization import (
    DomainRandomizedAMMEnvironment,
    UniformDomainRandomizationConfig,
)
from SAiFE_gym.gym.index_names import GAS_COST_KEY, LP_LIQUIDITY_KEY, PORTFOLIO_VALUE_KEY
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.stochastic_processes.gas_cost_models import OrnsteinUhlenbeckGasCostModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel


def make_gas(n=3, **kwargs):
    params = dict(theta=2.0, mu=2.0, sigma=1.0, initial_cost=5.0, step_size=0.25)
    params.update(kwargs)
    return OrnsteinUhlenbeckGasCostModel(num_trajectories=n, **params)


def make_env(gas_model=None, *, n=3, initial_wealth=100.0, gas_cost=99.0,
             environment_class=AMMEnvironment, volatility=0.0, arrival_rate=0.0):
    md = UniswapV3ModelDynamics(
        midprice_model=BrownianMotionMidpriceModel(
            volatility=volatility, num_trajectories=n, step_size=0.25,
        ),
        arrival_model=PoissonArrivalModel(
            intensity=np.full(2, arrival_rate), num_trajectories=n, step_size=0.25,
        ),
        gas_cost_model=gas_model, gas_cost=gas_cost, num_trajectories=n, num_ticks=128,
    )
    return environment_class(
        model_dynamics=md, num_trajectories=n, n_steps=4,
        initial_wealth=initial_wealth, seed=0,
    )


def actions(n=3, hold=False):
    return np.tile([-2.0, 2.0, 1.0 if hold else -1.0], (n, 1))


def test_rebalance_charges_observed_gas_and_hold_advances_process():
    gas = make_gas()
    env = make_env(gas)
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs[GAS_COST_KEY], np.full(3, 5.0))
    reference = deepcopy(gas)

    # First deployment is free, but gas advances for the next action.
    obs, reward, *_ = env.step(actions())
    reference.update(None, None, None)
    np.testing.assert_allclose(reward, 0.0, atol=1e-10)
    np.testing.assert_array_equal(obs[GAS_COST_KEY], reference.current_gas_cost)
    observed_cost = obs[GAS_COST_KEY].copy()

    mixed_action = actions()
    mixed_action[1, 2] = 1.0
    obs, reward, *_ = env.step(mixed_action)
    reference.update(None, None, None)
    expected_cost = observed_cost * np.array([1.0, 0.0, 1.0])
    np.testing.assert_allclose(reward, -expected_cost, atol=1e-10)
    np.testing.assert_allclose(obs[PORTFOLIO_VALUE_KEY], 100.0 - expected_cost)
    np.testing.assert_array_equal(obs[GAS_COST_KEY], reference.current_gas_cost)

    obs, reward, *_ = env.step(actions(hold=True))
    reference.update(None, None, None)
    np.testing.assert_allclose(reward, 0.0, atol=1e-10)
    np.testing.assert_array_equal(obs[GAS_COST_KEY], reference.current_gas_cost)
    assert env.observation_space.contains(obs)


def test_negative_gas_never_credits_portfolio():
    gas = make_gas(n=1, mu=0.0, initial_cost=0.0)
    env = make_env(gas, n=1)
    env.reset(seed=0)
    obs, *_ = env.step(actions(1))
    assert gas.current_state[0, 0] < 0.0  # First draw from gas stream seed 4.
    assert obs[GAS_COST_KEY][0] == 0.0
    obs, reward, *_ = env.step(actions(1))
    np.testing.assert_allclose(reward, 0.0, atol=1e-10)
    np.testing.assert_allclose(obs[PORTFOLIO_VALUE_KEY], 100.0)


def test_gas_can_exhaust_wealth_without_replenishing_it():
    env = make_env(make_gas(mu=20.0, initial_cost=20.0, sigma=0.0), initial_wealth=10.0)
    env.reset()
    env.step(actions())
    obs, reward, *_ = env.step(actions())
    np.testing.assert_allclose(reward, -10.0)
    np.testing.assert_array_equal(obs[PORTFOLIO_VALUE_KEY], np.zeros(3))
    np.testing.assert_array_equal(obs[LP_LIQUIDITY_KEY], np.zeros(3))
    obs, reward, *_ = env.step(actions())
    np.testing.assert_array_equal(obs[PORTFOLIO_VALUE_KEY], np.zeros(3))
    np.testing.assert_array_equal(reward, np.zeros(3))


def test_constant_ou_matches_fixed_gas_without_changing_market_rngs():
    fixed = make_env(gas_cost=7.0, volatility=0.1, arrival_rate=4.0)
    stochastic = make_env(
        make_gas(mu=7.0, initial_cost=7.0, sigma=0.0),
        volatility=0.1, arrival_rate=4.0,
    )
    fixed.reset(seed=42)
    stochastic.reset(seed=42)
    for hold in (False, False, True, False):
        expected = fixed.step(actions(hold=hold))
        actual = stochastic.step(actions(hold=hold))
        for key in expected[0]:
            np.testing.assert_array_equal(actual[0][key], expected[0][key])
        np.testing.assert_array_equal(actual[1], expected[1])
    for name in ("midprice_model", "arrival_model", "price_impact_model"):
        assert (getattr(fixed.model_dynamics, name).rng.bit_generator.state
                == getattr(stochastic.model_dynamics, name).rng.bit_generator.state)


@pytest.mark.parametrize("environment_class", [AMMEnvironment, ArbitrageurEnvironment])
def test_environment_seed_reset_and_no_action_lifecycle(environment_class):
    gas = make_gas()
    gas.update(None, None, None)
    env = make_env(gas, environment_class=environment_class)
    np.testing.assert_array_equal(gas.current_gas_cost, np.full(3, 5.0))
    assert gas.seed_ == 4
    action = None if environment_class is AMMEnvironment else np.zeros((3, 1))

    def path(seed=None):
        env.reset(seed=seed)
        np.testing.assert_array_equal(env.state[GAS_COST_KEY], np.full(3, 5.0))
        costs = []
        for _ in range(3):
            env.step(action)
            costs.append(env.state[GAS_COST_KEY].copy())
        return np.array(costs)

    first = path(0)
    np.testing.assert_array_equal(path(0), first)
    assert not np.array_equal(path(), first)
    assert not np.array_equal(path(1), first)
    np.testing.assert_array_equal(env._initial_state[GAS_COST_KEY], np.full(3, 5.0))


@pytest.mark.parametrize("environment_class", [AMMEnvironment, ArbitrageurEnvironment])
@pytest.mark.parametrize("kwargs,match", [
    ({"n": 2}, "num_trajectories"),
    ({"step_size": 0.1}, "step_size"),
])
def test_mismatched_gas_configuration_rejected(environment_class, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_env(make_gas(**kwargs), environment_class=environment_class)


@pytest.mark.parametrize("vector", [False, True])
def test_gymnasium_observations_and_resets(vector):
    n = 3 if vector else 1
    raw = make_env(make_gas(n=n), n=n)
    env = GymnasiumAMMVectorEnv(raw) if vector else GymnasiumAMMEnvironment(raw)
    action = actions(n, hold=True) if vector else actions(1, hold=True)[0]
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(4):
        obs, *_ = env.step(action)
        assert env.observation_space.contains(obs)
    if vector:
        obs, *_ = env.step(action)  # Next-step autoreset.
    else:
        obs, _ = env.reset()
    np.testing.assert_array_equal(obs[GAS_COST_KEY], np.full((n,) if vector else (), 5.0))


def test_sb3_explicit_gas_feature_and_autoreset():
    assert GAS_COST_KEY not in DEFAULT_OBS_KEYS
    env = StableBaselinesAMMEnvironment(
        make_env(make_gas()), obs_keys=[*DEFAULT_OBS_KEYS, GAS_COST_KEY],
    )
    obs = env.reset()
    np.testing.assert_array_equal(obs[:, -1], np.full(3, 5.0))
    for _ in range(4):
        obs, _, done, infos = env.step(actions(hold=True))
    assert done.all()
    np.testing.assert_array_equal(obs[:, -1], np.full(3, 5.0))
    assert not np.array_equal(
        [info["terminal_observation"][-1] for info in infos], obs[:, -1],
    )


def test_domain_randomization_leaves_gas_parameters_and_rng_independent():
    gas = make_gas()
    raw = make_env(gas)
    wrapped = DomainRandomizedAMMEnvironment(
        raw, UniformDomainRandomizationConfig(
            sigma_range=(0.0, 0.1), arrival_rate_range=(0.0, 2.0),
        ), num_domains=3,
    )
    reference = deepcopy(raw)
    for seed in (0, None):
        wrapped.reset(seed=seed)
        reference.reset(seed=seed)
        assert (gas.theta, gas.mu, gas.sigma) == (2.0, 2.0, 1.0)
        for _ in range(3):
            obs, *_ = wrapped.step(actions(hold=True))
            expected, *_ = reference.step(actions(hold=True))
            np.testing.assert_array_equal(obs[GAS_COST_KEY], expected[GAS_COST_KEY])
