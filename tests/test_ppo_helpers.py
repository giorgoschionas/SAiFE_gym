"""Regression tests for the nominal PPO helper's evaluation pipeline."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from experiments import helpers
from SAiFE_gym.gym.index_names import ASSET_PRICE_KEY
from SAiFE_gym.rewards.RewardFunctions import PnL


class StatefulPnL(PnL):
    """Make reset/step interference observable for custom reward functions."""

    def __init__(self):
        self.reset_count = 0
        self.step_count = 0

    def reset(self, initial_state):
        self.reset_count += 1
        self.step_count = 0

    def calculate(self, current_state, action, next_state, is_terminal_step=False):
        self.step_count += 1
        return super().calculate(current_state, action, next_state, is_terminal_step)


@pytest.fixture
def make_env(monkeypatch):
    # Retain the real simulator and stochastic models with smaller test arrays.
    monkeypatch.setattr(helpers, "NUM_TICKS", 1024)

    def factory(num_trajectories=2, n_steps=64, **kwargs):
        return helpers.get_amm_env(
            num_trajectories=num_trajectories, n_steps=n_steps, **kwargs
        )

    return factory


@pytest.fixture
def make_learner(tmp_path):
    pipelines = []

    def factory(env, **kwargs):
        model, callback = helpers.get_ppo_learner_and_callback(
            env, best_model_path=str(tmp_path / "best"), **kwargs
        )
        pipelines.extend([model.get_env(), callback.eval_env])
        return model, callback

    yield factory
    for pipeline in pipelines:
        pipeline.close()


def raw_env(vec):
    while hasattr(vec, "venv"):
        vec = vec.venv
    return vec.env


def assert_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_equal(a, b)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def components(env):
    md = env.model_dynamics
    return [env, md, md.midprice_model, md.arrival_model, md.price_impact_model]


def snapshot(env):
    md = env.model_dynamics
    return deepcopy({
        "state": env.state,
        "initial_state": env.initial_state,
        "rngs": [component.rng.bit_generator.state for component in components(env)],
        "midprice": md.midprice_model.current_state,
        "arrival": md.arrival_model.current_state,
        "reward": vars(env.reward_function),
    })


def normalization_snapshot(vec):
    return deepcopy({
        "obs_rms": vars(vec.obs_rms),
        "ret_rms": vars(vec.ret_rms),
        "returns": vec.returns,
    })


def test_automatic_copy_preserves_configuration_and_leaves_training_untouched(
    make_env, make_learner,
):
    env = make_env(
        tau=7, volatility=1.3, arrival_rate=55.0, alpha3=0.7,
        gas_cost=12.0, swap_fee_rate=0.002, reward_function=StatefulPnL(), seed=17,
    )
    env.reset()
    env.step(np.tile([-3, 4, -1], (env.num_trajectories, 1)))
    before = snapshot(env)
    model, callback = make_learner(env)
    evaluation = raw_env(callback.eval_env)

    assert raw_env(model.get_env()) is env
    assert_equal(snapshot(env), before)
    assert evaluation.num_trajectories == env.num_trajectories
    assert evaluation.n_steps == env.n_steps
    assert evaluation.initial_wealth == env.initial_wealth
    assert evaluation.reward_function is not env.reward_function
    assert isinstance(evaluation.reward_function, StatefulPnL)
    assert evaluation.reward_function.step_count == 0
    for name in ("tau", "fee_tier", "gas_cost", "swap_fee_rate", "num_ticks"):
        assert getattr(evaluation.model_dynamics, name) == getattr(env.model_dynamics, name)
    assert_equal(evaluation.model_dynamics.arrival_model.alpha, env.model_dynamics.arrival_model.alpha)
    assert_equal(evaluation.model_dynamics.midprice_model.volatility, env.model_dynamics.midprice_model.volatility)
    assert_equal(evaluation.initial_state, env.initial_state)
    for original, copied in zip(components(env), components(evaluation)):
        assert copied is not original
        assert copied.rng is not original.rng
        assert copied.rng.bit_generator.state != original.rng.bit_generator.state
    for key in env.state:
        assert not np.shares_memory(env.state[key], evaluation.state[key])
        assert not np.shares_memory(env._initial_state[key], evaluation._initial_state[key])


@pytest.mark.parametrize("normalise_obs", [False, True])
@pytest.mark.parametrize("num_trajectories", [1, 3])
def test_real_evaluation_does_not_change_training_or_next_transition(
    make_env, make_learner, tmp_path, normalise_obs, num_trajectories,
):
    env = make_env(num_trajectories=num_trajectories, reward_function=StatefulPnL())
    model, callback = make_learner(env, normalise_obs=normalise_obs)
    training_vec = model.get_env()
    control_vec = helpers.wrap_env(deepcopy(env), normalise_obs=normalise_obs)
    try:
        assert_equal(training_vec.reset(), control_vec.reset())
        action = np.tile([-2, 3, -1], (num_trajectories, 1))
        transition = training_vec.step(action)
        assert_equal(transition, control_vec.step(action))
        model._last_obs = transition[0].copy()
        retained_obs = model._last_obs.copy()
        before = snapshot(env)
        if normalise_obs:
            before_norm = normalization_snapshot(training_vec)

        model.set_logger(configure(str(tmp_path), format_strings=[]))
        callback.init_callback(model)
        callback.n_calls = callback.eval_freq - 1
        assert callback.on_step()

        assert_equal(snapshot(env), before)
        np.testing.assert_array_equal(model._last_obs, retained_obs)
        assert np.isfinite(callback.last_mean_reward)
        if normalise_obs:
            assert_equal(normalization_snapshot(training_vec), before_norm)
            evaluation_vec = callback.eval_env
            assert evaluation_vec.training is False
            assert evaluation_vec.norm_reward is False
            for name in ("obs_rms", "ret_rms"):
                train_stats = getattr(training_vec, name)
                eval_stats = getattr(evaluation_vec, name)
                assert_equal(vars(eval_stats), vars(train_stats))
                assert eval_stats is not train_stats
                assert not np.shares_memory(eval_stats.mean, train_stats.mean)
        else:
            assert not isinstance(callback.eval_env, VecNormalize)

        next_action = np.tile([-2, 3, 1], (num_trajectories, 1))
        assert_equal(training_vec.step(next_action), control_vec.step(next_action))
        assert_equal(snapshot(env), snapshot(raw_env(control_vec)))
    finally:
        control_vec.close()


def test_explicit_environment_can_have_different_batch_size(make_env, make_learner):
    env = make_env(num_trajectories=3)
    evaluation = make_env(num_trajectories=1, seed=999)
    model, callback = make_learner(env, eval_env=evaluation, eval_seed=0)
    assert raw_env(model.get_env()) is env
    assert raw_env(callback.eval_env) is evaluation
    assert callback.eval_env.num_envs == 1
    assert evaluation.model_dynamics.midprice_model.seed_ == 0
    assert evaluation.model_dynamics.arrival_model.seed_ == 1
    assert evaluation.model_dynamics.price_impact_model.seed_ == 2
    assert evaluation.model_dynamics.seed == 3


@pytest.mark.parametrize("shared_component", [
    "environment", "model_dynamics", "reward_function", "midprice_model",
    "arrival_model", "price_impact_model", "fee_accounting_model",
])
def test_shared_components_rejected_before_reset(make_env, make_learner, shared_component):
    env = make_env(reward_function=StatefulPnL())
    env.reset()
    env.step(np.tile([-1, 1, -1], (env.num_trajectories, 1)))
    evaluation = make_env()
    if shared_component == "environment":
        evaluation = env
    elif shared_component in ("model_dynamics", "reward_function"):
        setattr(evaluation, shared_component, getattr(env, shared_component))
    else:
        setattr(evaluation.model_dynamics, shared_component, getattr(env.model_dynamics, shared_component))
    before = snapshot(env)
    with pytest.raises(ValueError, match=f"must not share {shared_component}"):
        make_learner(env, eval_env=evaluation)
    assert_equal(snapshot(env), before)


def test_incompatible_actions_rejected_before_reset(make_env, make_learner):
    env = make_env(tau=5)
    evaluation = make_env(tau=7)
    before = snapshot(evaluation)
    with pytest.raises(ValueError, match="observation/action spaces must match"):
        make_learner(env, eval_env=evaluation)
    assert_equal(snapshot(evaluation), before)


def test_copy_failure_explains_explicit_environment_escape_hatch(make_env, make_learner):
    class UncopyablePnL(PnL):
        def __deepcopy__(self, memo):
            raise TypeError("Custom resource cannot be copied")

    env = make_env(reward_function=UncopyablePnL())
    before = snapshot(env)
    with pytest.raises(ValueError, match="supply a separately constructed eval_env") as error:
        make_learner(env)
    assert isinstance(error.value.__cause__, TypeError)
    assert_equal(snapshot(env), before)
    evaluation = make_env(reward_function=UncopyablePnL())
    _, callback = make_learner(env, eval_env=evaluation)
    assert raw_env(callback.eval_env) is evaluation


@pytest.mark.parametrize("explicit", [False, True])
def test_evaluation_seed_zero_is_reproducible_and_stream_advances(
    make_env, make_learner, explicit,
):
    callbacks = []
    for seed in (0, 0, 1):
        kwargs = {"eval_env": make_env(seed=999)} if explicit else {}
        _, callback = make_learner(make_env(), eval_seed=seed, **kwargs)
        callbacks.append(callback)
    paths = [[] for _ in callbacks]
    for _ in range(2):
        for callback, path in zip(callbacks, paths):
            vec = callback.eval_env
            vec.reset()
            for _ in range(3):
                vec.step(np.tile([-1, 1, 1], (vec.num_envs, 1)))
                path.append(raw_env(vec).state[ASSET_PRICE_KEY].copy())
    assert_equal(paths[0], paths[1])
    assert not np.array_equal(paths[0], paths[2])
    assert not np.array_equal(paths[0][:3], paths[0][3:])


@pytest.mark.parametrize("num_trajectories,n_steps", [(1, 64), (3, 64), (50, 200)])
def test_frequency_counts_batch_steps(make_env, make_learner, num_trajectories, n_steps):
    model, callback = make_learner(make_env(num_trajectories, n_steps))
    assert callback.eval_freq == 10 * model.n_steps
    assert callback.n_eval_episodes == 10
    assert callback.deterministic is True
    if num_trajectories == 50:
        assert callback.eval_freq == 2000
        assert callback.eval_freq * num_trajectories == 100_000
        assert (2_000_000 // num_trajectories) // callback.eval_freq == 20


@pytest.mark.parametrize("normalise_obs", [False, True])
def test_ppo_run_evaluates_twice_and_saves_best_model(
    make_env, make_learner, tmp_path, normalise_obs,
):
    env = make_env(num_trajectories=2, n_steps=32)
    model, callback = make_learner(
        env, normalise_obs=normalise_obs, eval_log_path=str(tmp_path / "eval"),
    )
    model.learn(total_timesteps=20 * env.n_steps * env.num_trajectories, callback=callback)

    assert model.num_timesteps == 1280
    assert callback.n_calls == 640
    assert callback.evaluations_timesteps == [640, 1280]
    assert np.isfinite(callback.best_mean_reward)
    best_model = Path(callback.best_model_save_path) / "best_model.zip"
    assert best_model.is_file()
    assert best_model.stat().st_size > 0
    with np.load(tmp_path / "eval" / "evaluations.npz") as results:
        np.testing.assert_array_equal(results["timesteps"], [640, 1280])
        np.testing.assert_array_equal(results["ep_lengths"], np.full((2, 10), 32))
        assert results["results"].shape == (2, 10)
        assert np.all(np.isfinite(results["results"]))
