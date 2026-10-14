"""Conditional LP decisions, unique categorical IDs, and legacy compatibility."""

from copy import deepcopy
from types import SimpleNamespace
import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector.utils import batch_space
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.vec_env import VecNormalize

from SAiFE_gym.gym.lp_actions import (
    CATEGORICAL_ENCODING, STRUCTURED_ENCODING, CategoricalLPActionCodec,
    LPActionSpace, decode_policy_actions, hold_action, rebalance_action,
)
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from SAiFE_gym.wrappers import StructuredMultiDiscreteVecEnv
from experiments.helpers import _plot_action_bounds
from experiments import train_robust_lp_agent as training
from experiments.evaluate_robust_lp_agents import load_config
from experiments.train_robust_lp_agent import DecisionStrideVecEnv, validate_policy_action_encoding
from test_stable_baselines_amm_environment import create_test_amm_env


def test_conditional_sampling_seeding_and_json_roundtrip():
    space = LPActionSpace(5)
    space.seed(123)
    samples = [space.sample() for _ in range(100)]
    assert {s[0] for s in samples} == {0, 1}
    assert all(space.contains(a) for a in samples)
    assert all(a[1] == () for a in samples if a[0] == 0)
    restored = space.from_jsonable(space.to_jsonable(samples))
    assert data_equivalence(samples, restored, exact=True)
    space.seed(123)
    assert data_equivalence(samples, [space.sample() for _ in samples], exact=True)
    batched = batch_space(space, 3)
    assert batched.contains((hold_action(), rebalance_action(-2, 3), hold_action()))


@pytest.mark.parametrize("action", [
    (0, (1,)), (1, [-6, 2]), (1, [-2, 6]), (1, [0, 0]),
    (1, [2, -2]), (1, [-2.5, 3]), (1, [-2.0, 3.0]), (2, ()), (True, [-2, 3]),
])
def test_invalid_conditional_action_does_not_change_state(action):
    env = create_test_amm_env()
    env.reset(seed=42)
    before = deepcopy(env.state)
    assert not env.action_space.contains(action)
    with pytest.raises(ValueError):
        env.step(action)
    assert data_equivalence(before, env.state, exact=True)


@pytest.mark.parametrize("n", [1, 3])
def test_conditional_and_raw_execution_are_identical(n):
    env = create_test_amm_env(num_trajectories=n)
    raw = deepcopy(env)
    env.reset(seed=123)
    raw.reset(seed=123)
    sequences = [
        [rebalance_action(-5, 5)] * n,
        [hold_action()] * n,
        [rebalance_action(-2, 3)] + [hold_action()] * (n - 1),
        [hold_action()] + [rebalance_action(1, 4)] * (n - 1),
    ]
    for decisions in sequences:
        conditional = decisions[0] if n == 1 else tuple(decisions)
        commands = env.normalize_action(conditional)
        actual = env.step(conditional)
        expected = raw.step_command(commands)
        assert data_equivalence(actual, expected, exact=True)


@pytest.mark.parametrize("tau,stride,count", [(5, 1, 56), (50, 5, 211), (5, 3, 11), (2, 9, 2)])
def test_categorical_table_is_unique_complete_and_stable(tau, stride, count):
    codec = CategoricalLPActionCodec(tau, stride)
    assert codec.action_space == gym.spaces.Discrete(count)
    table = codec.decode(np.arange(count))
    np.testing.assert_array_equal(table[0], [0, 1, 1])
    assert np.count_nonzero(table[:, 2] > 0) == 1
    assert len(np.unique(table[1:, :2], axis=0)) == count - 1
    endpoints = list(range(-tau, tau + 1, stride))
    if endpoints[-1] != tau:
        endpoints.append(tau)
    expected = [[lo, hi, -1] for i, lo in enumerate(endpoints) for hi in endpoints[i + 1:]]
    np.testing.assert_array_equal(table[1:], expected)
    assert all(LPActionSpace(tau).contains(rebalance_action(int(lo), int(hi))) for lo, hi, _ in table[1:])
    table[0] = 99
    np.testing.assert_array_equal(codec.decode(0), [0, 1, 1])


@pytest.mark.parametrize("ids", [-1, 56, 1.0, True, [0, -1], [0.0, 1.5]])
def test_invalid_categorical_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="action IDs"):
        CategoricalLPActionCodec(5).decode(ids)


@pytest.mark.parametrize("n", [1, 3])
def test_default_sb3_categorical_matches_raw_adapter(n):
    base = create_test_amm_env(num_trajectories=n)
    categorical = StableBaselinesAMMEnvironment(base)
    raw = StableBaselinesAMMEnvironment(deepcopy(base), action_encoding="raw")
    assert categorical.action_space == gym.spaces.Discrete(56)
    assert isinstance(raw.action_space, gym.spaces.Box)
    np.testing.assert_array_equal(categorical.reset(), raw.reset())
    for ids in (np.full(n, 10), np.zeros(n, dtype=int), np.arange(n) + 20):
        assert data_equivalence(categorical.step(ids), raw.step(categorical.decode_actions(ids)), exact=True)
    before = deepcopy(base.state)
    with pytest.raises(ValueError):
        categorical.step(np.full(n, -1))
    assert data_equivalence(before, base.state, exact=True)


def test_categorical_decision_stride_executes_one_action_then_holds():
    base = create_test_amm_env(num_trajectories=2, n_steps=4)
    reference = deepcopy(base)
    adapter = StableBaselinesAMMEnvironment(base)
    stride = DecisionStrideVecEnv(adapter, 2)
    stride.reset()
    reference.reset()
    ids = np.array([10, 0])
    first = reference.step(adapter.decode_actions(ids))
    second = reference.step(np.tile([0, 1, 1], (2, 1)))
    obs, reward, done, _ = stride.step(ids)
    np.testing.assert_allclose(reward, first[1] + second[1])
    np.testing.assert_array_equal(obs, adapter._flatten_obs(reference.state))
    assert not done.any()
    assert data_equivalence(base.state, reference.state, exact=True)


def test_standard_ppo_save_load_and_encoding_validation(tmp_path):
    env = StableBaselinesAMMEnvironment(create_test_amm_env(num_trajectories=2, tau=50), tick_stride=5)
    model = PPO("MlpPolicy", env, n_steps=5, batch_size=10, n_epochs=1, seed=7, verbose=0)
    assert model.action_space == gym.spaces.Discrete(211)
    assert isinstance(model.policy.action_dist, CategoricalDistribution)
    model.lp_action_config = env.action_codec.metadata
    model.learn(total_timesteps=10)
    path = tmp_path / "categorical"
    model.save(path)
    loaded = PPO.load(path)
    assert loaded.lp_action_config == model.lp_action_config
    obs = env.reset()
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
    args = SimpleNamespace(tau=50, tick_stride=5, action_encoding=CATEGORICAL_ENCODING, action_count=211)
    validate_policy_action_encoding(loaded, args)
    with pytest.raises(ValueError, match="encoding"):
        validate_policy_action_encoding(loaded, SimpleNamespace(**{**vars(args), "action_encoding": STRUCTURED_ENCODING}))
    with pytest.raises(ValueError, match="configured endpoint grid"):
        validate_policy_action_encoding(loaded, SimpleNamespace(**{**vars(args), "tick_stride": 1}))
    with pytest.raises(ValueError, match="metadata"):
        validate_policy_action_encoding(loaded, SimpleNamespace(**{**vars(args), "tau": 100, "tick_stride": 10}))
    with pytest.raises(ValueError, match="action count"):
        validate_policy_action_encoding(loaded, SimpleNamespace(**{**vars(args), "action_count": 56}))


def test_legacy_multidiscrete_decoding_and_validation():
    env = StructuredMultiDiscreteVecEnv(
        StableBaselinesAMMEnvironment(create_test_amm_env(), action_encoding="raw"), tau=5,
    )
    model = PPO("MlpPolicy", env, n_steps=5, batch_size=5, seed=7, verbose=0)
    args = SimpleNamespace(tau=5, tick_stride=1, action_encoding=STRUCTURED_ENCODING)
    validate_policy_action_encoding(model, args)
    actions = np.array([[5, 1, 0], [10, 4, 1]])
    np.testing.assert_array_equal(decode_policy_actions(actions, model.action_space, 5), [[-2, 2, -1], [0, 5, 1]])


def test_categorical_plots_mask_hold_bounds():
    model = SimpleNamespace(action_space=gym.spaces.Discrete(56))
    bounds = _plot_action_bounds(model, np.array([0, 1]), 5, 1)
    assert np.isnan(bounds[0]).all()
    np.testing.assert_array_equal(bounds[1], [-5, -4])


def test_legacy_tuple_commands_and_strict_serialized_actions():
    env = create_test_amm_env(num_trajectories=2)
    env.reset()
    env.step(((-2, 3), (-1, 1)))
    with pytest.raises(ValueError):
        env.action_space.from_jsonable([[0.5, []]])
    with pytest.raises(ValueError):
        env.action_space.from_jsonable([[1, [-2.0, 3.0]]])


def test_legacy_saved_policy_evaluates_with_saved_normalization(tmp_path):
    args = training.parse_args([
        "--smoke-test", "--decision-stride", "2", "--tau", "5",
        "--tick-stride", "1", "--periodic-width", "5",
    ])
    training.apply_smoke_overrides(args)
    # Historical runs have no explicit action-encoding metadata.
    (tmp_path / "config.json").write_text(json.dumps(vars(args)))
    args.action_encoding = load_config(tmp_path)["action_encoding"]
    assert args.action_encoding == STRUCTURED_ENCODING
    params = training.DomainParameters(sigma=args.nominal_sigma, arrival_rate=args.nominal_arrival_rate)
    raw = training.make_fixed_env(args, params, seed=7)
    normalizer = training.wrap_env(raw, action_encoding="raw")
    legacy = StructuredMultiDiscreteVecEnv(DecisionStrideVecEnv(normalizer, 2), tau=5)
    model = PPO("MlpPolicy", legacy, n_steps=2, batch_size=4, n_epochs=1, seed=7, verbose=0)
    model.learn(total_timesteps=4)
    model.save(tmp_path / "legacy")
    normalizer.save(tmp_path / "normalizer.pkl")
    normalizer.training = False
    expected = training.evaluate_ppo("nominal_ppo", model, normalizer, params, "in_distribution", args, 99)
    restored_env = training.wrap_env(
        training.make_fixed_env(args, params, seed=7), normalise_obs=False, action_encoding="raw",
    )
    restored_normalizer = VecNormalize.load(tmp_path / "normalizer.pkl", restored_env)
    restored_normalizer.training = False
    restored = PPO.load(tmp_path / "legacy")
    actual = training.evaluate_ppo("nominal_ppo", restored, restored_normalizer, params, "in_distribution", args, 99)
    assert actual == expected
