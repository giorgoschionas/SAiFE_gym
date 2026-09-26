"""OU transition law, vectorization, and the gas-cost boundary."""

from copy import deepcopy

import numpy as np
import pytest

from SAiFE_gym.stochastic_processes.gas_cost_models import OrnsteinUhlenbeckGasCostModel


def make_model(**kwargs):
    params = dict(theta=2.0, mu=5.0, sigma=1.0, num_trajectories=3, seed=17)
    params.update(kwargs)
    return OrnsteinUhlenbeckGasCostModel(**params)


@pytest.mark.parametrize("n", [1, 4])
def test_initial_state_shapes_and_reset(n):
    model = make_model(num_trajectories=n)
    assert model.current_state.shape == (n, 1)
    assert model.current_gas_cost.shape == (n,)
    assert model.current_state.dtype == model.current_gas_cost.dtype == np.float64
    np.testing.assert_array_equal(model.current_state, np.full((n, 1), 5.0))
    model.update(None, None, None)
    rng_state = deepcopy(model.rng.bit_generator.state)
    model.reset()
    np.testing.assert_array_equal(model.current_state, np.full((n, 1), 5.0))
    assert model.rng.bit_generator.state == rng_state
    assert not np.shares_memory(model.initial_state, model.current_state)
    costs = model.current_gas_cost
    costs[:] = 0.0
    np.testing.assert_array_equal(model.current_state, np.full((n, 1), 5.0))


@pytest.mark.parametrize("step_size", [0.01, 2.0])
def test_zero_volatility_matches_continuous_time_mean_reversion(step_size):
    model = make_model(initial_cost=20.0, sigma=0.0, step_size=step_size)
    for step in range(1, 6):
        model.update(None, None, None)
        expected = 5.0 + 15.0 * np.exp(-2.0 * step * step_size)
        np.testing.assert_allclose(model.current_state, expected, rtol=1e-14)


def test_negative_latent_state_is_preserved_when_cost_is_floored():
    model = make_model(mu=1.0, sigma=0.0, step_size=0.1)
    model.current_state[:] = -10.0
    np.testing.assert_array_equal(model.current_gas_cost, np.zeros(3))
    model.update(None, None, None)
    np.testing.assert_allclose(model.current_state, 1.0 - 11.0 * np.exp(-0.2))
    np.testing.assert_array_equal(model.current_gas_cost, np.zeros(3))


def test_seeded_paths_replay_and_reset_without_seed_continues_rng():
    model = make_model()

    def path():
        states = []
        for _ in range(4):
            model.update(None, None, None)
            states.append(model.current_state.copy())
        return np.array(states)

    first = path()
    model.reset()
    assert not np.array_equal(path(), first)
    model.seed(17)
    model.reset()
    np.testing.assert_array_equal(path(), first)
    assert not np.array_equal(first[:, 0], first[:, 1])


def test_transition_moments_and_independent_trajectory_noise():
    n = 100_000
    model = make_model(num_trajectories=n, initial_cost=9.0, sigma=3.0, step_size=0.2)
    for _ in range(5):
        model.update(None, None, None)
    # Compare the full one-time-unit distribution with the analytical OU law.
    expected_mean = 5.0 + 4.0 * np.exp(-2.0)
    expected_variance = 9.0 / 4.0 * (1.0 - np.exp(-4.0))
    samples = model.current_state[:, 0]
    assert samples.mean() == pytest.approx(
        expected_mean, abs=6.0 * np.sqrt(expected_variance / n)
    )
    assert samples.var() == pytest.approx(expected_variance, rel=0.025)
    assert abs(np.corrcoef(samples[::2], samples[1::2])[0, 1]) < 0.025


def test_small_reversion_rate_retains_brownian_variance():
    model = make_model(theta=1e-20, mu=0.0, sigma=2.0, step_size=0.1)
    expected = 2.0 * np.sqrt(0.1) * np.random.default_rng(17).normal(size=(3, 1))
    model.update(None, None, None)
    np.testing.assert_allclose(model.current_state, expected, rtol=1e-14)


@pytest.mark.parametrize("name,value", [
    ("theta", 0.0), ("theta", -1.0), ("theta", np.inf),
    ("theta", "2"), ("theta", [2.0]),
    ("mu", -1.0), ("mu", np.nan),
    ("sigma", -1.0), ("sigma", np.inf),
    ("initial_cost", -1.0), ("initial_cost", np.nan),
    ("step_size", 0.0), ("step_size", -1.0), ("step_size", np.inf),
    ("terminal_time", 0.0), ("terminal_time", -1.0), ("terminal_time", np.nan),
    ("num_trajectories", 0), ("num_trajectories", -1),
    ("num_trajectories", 1.5), ("num_trajectories", True),
])
def test_invalid_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        make_model(**{name: value})
