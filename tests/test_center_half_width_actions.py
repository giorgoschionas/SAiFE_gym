"""Explicit RL grids preserve exact ranges, policy meanings, and legacy behavior."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence
import numpy as np
import pytest
from stable_baselines3 import PPO

from SAiFE_gym.gym.index_names import LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY, POOL_CURRENT_TICK_KEY
from SAiFE_gym.gym.lp_actions import (
    CENTER_HALF_WIDTH_ENCODING, CATEGORICAL_ENCODING,
    CenterHalfWidthActionGrid, CenterHalfWidthLPActionCodec,
    action_grid_from_metadata, validate_policy_metadata,
)
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from experiments import evaluate_robust_lp_agents as evaluation
from experiments import train_robust_lp_agent as training
from experiments.helpers import _plot_action_bounds
from test_stable_baselines_amm_environment import create_test_amm_env


EXAMPLE = CenterHalfWidthActionGrid([50, 100, 150, 200], [50, 100, 150, 200, 250, 300])


def test_example_is_55_unique_exact_decisions_with_stable_order():
    codec = CenterHalfWidthLPActionCodec(500, EXAMPLE)
    assert codec.action_space == gym.spaces.Discrete(55)
    commands = codec.decode(np.arange(55))
    assert np.count_nonzero(commands[:, 2] > 0) == 1
    assert len(np.unique(commands[1:, :2], axis=0)) == 54
    np.testing.assert_array_equal(codec.decode([0, 1, 6, 25, 30, 55 - 1]), [
        [0, 1, 1], [-250, -150, -1], [-500, 100, -1],
        [-50, 50, -1], [-300, 300, -1], [-100, 500, -1],
    ])
    assert np.count_nonzero(np.sum(commands[1:, :2], axis=1) == 0) == 6
    dynamics = create_test_amm_env(tau=500).model_dynamics
    np.testing.assert_array_equal(dynamics.validate_action(commands[1:]), commands[1:])
    np.testing.assert_array_equal(codec.decode(1), commands[1])
    commands[1] = 0
    np.testing.assert_array_equal(codec.decode(1), [-250, -150, -1])
    assert action_grid_from_metadata(json.loads(json.dumps(codec.metadata))) == EXAMPLE


def test_grid_canonicalizes_input_and_is_immutable():
    grid = CenterHalfWidthActionGrid([2, 0, 1, 2], [3, 1, 3])
    assert grid.centers == (-2, -1, 0, 1, 2)
    assert grid.half_widths == (1, 3)
    with pytest.raises(FrozenInstanceError):
        grid.half_widths = (5,)
    assert CenterHalfWidthLPActionCodec(3, CenterHalfWidthActionGrid([0], [3])).action_space.n == 2


@pytest.mark.parametrize("centers,widths", [
    ([], [1]), ([1], []), ([-1], [1]), ([1.0], [1]), ([True, 1], [1]),
    ([1], [0]), ([1], [-1]), ([1], [1.0]), ([1], [True]),
    ([[1]], [1]), (None, [1]),
])
def test_invalid_grid_inputs_are_rejected(centers, widths):
    with pytest.raises(ValueError):
        CenterHalfWidthActionGrid(centers, widths)


@pytest.mark.parametrize("ids", [-1, 55, 0.0, True, [1, -1], [1.5, 2]])
def test_invalid_grid_action_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="action IDs"):
        CenterHalfWidthLPActionCodec(500, EXAMPLE).decode(ids)


def test_adapter_rejects_small_tau_or_raw_mode_without_changing_environment():
    base = create_test_amm_env(tau=499)
    before = deepcopy(base.state)
    with pytest.raises(ValueError, match="requires tau >= 500"):
        StableBaselinesAMMEnvironment(base, action_grid=EXAMPLE)
    assert base.model_dynamics.tau == 499
    assert data_equivalence(before, base.state, exact=True)
    with pytest.raises(ValueError, match="requires categorical"):
        StableBaselinesAMMEnvironment(base, action_encoding="raw", action_grid=EXAMPLE)


@pytest.mark.parametrize("magnitude,width,error", [
    (2**24, 1, "exactly representable"),
    (2**63 - 1, 1, "int64 tick range"),
])
def test_grid_rejects_numeric_conversion_that_would_change_endpoints(magnitude, width, error):
    grid = CenterHalfWidthActionGrid([magnitude], [width])
    with pytest.raises(ValueError, match=error):
        CenterHalfWidthLPActionCodec(grid.required_tau, grid)


@pytest.mark.parametrize("n", [1, 3])
def test_grid_adapter_matches_direct_endpoint_execution(n):
    base = create_test_amm_env(num_trajectories=n, n_steps=10)
    raw = StableBaselinesAMMEnvironment(deepcopy(base), action_encoding="raw")
    grid = CenterHalfWidthActionGrid([1, 2], [1, 2, 3])
    adapter = StableBaselinesAMMEnvironment(base, action_grid=grid, tick_stride=999)
    assert adapter.action_space == gym.spaces.Discrete(16)
    assert adapter.action_codec.metadata["action_encoding"] == CENTER_HALF_WIDTH_ENCODING
    np.testing.assert_array_equal(adapter.reset(), raw.reset())
    for ids in (np.arange(n) + 1, np.zeros(n, dtype=int), np.arange(n) + 12):
        assert data_equivalence(adapter.step(ids), raw.step(adapter.decode_actions(ids)), exact=True)
    before = deepcopy(base.state)
    with pytest.raises(ValueError):
        adapter.step(np.full(n, 16))
    assert data_equivalence(before, base.state, exact=True)


def test_hold_preserves_absolute_bounds_as_price_leaves_center_grid(monkeypatch):
    base = create_test_amm_env(n_steps=10)
    adapter = StableBaselinesAMMEnvironment(base, action_grid=CenterHalfWidthActionGrid([0], [1]))
    adapter.reset()
    monkeypatch.setattr(base.model_dynamics.arrival_model, "get_arrivals", lambda: np.array([[False, True]]))
    adapter.step(np.array([1]))
    lower, upper, tick = (base.state[k].copy() for k in (
        LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY, POOL_CURRENT_TICK_KEY,
    ))
    adapter.step(np.array([0]))
    np.testing.assert_array_equal(base.state[LP_TICK_LOWER_KEY], lower)
    np.testing.assert_array_equal(base.state[LP_TICK_UPPER_KEY], upper)
    np.testing.assert_array_equal(base.state[POOL_CURRENT_TICK_KEY], tick + 1)


def test_grid_decision_stride_matches_one_rebalance_then_hold():
    base = create_test_amm_env(num_trajectories=2, n_steps=10)
    reference = deepcopy(base)
    adapter = StableBaselinesAMMEnvironment(base, action_grid=CenterHalfWidthActionGrid([1], [1, 2]))
    stride = training.DecisionStrideVecEnv(adapter, 2)
    stride.reset()
    reference.reset()
    first = reference.step(adapter.decode_actions([1, 6]))
    second = reference.step(np.tile([0, 1, 1], (2, 1)))
    _, reward, _, _ = stride.step(np.array([1, 6]))
    np.testing.assert_allclose(reward, first[1] + second[1])
    assert data_equivalence(base.state, reference.state, exact=True)


def test_metadata_rejects_equal_sized_different_grids_and_missing_metadata():
    codec = CenterHalfWidthLPActionCodec(500, EXAMPLE)
    model = SimpleNamespace(action_space=codec.action_space, lp_action_config=codec.metadata)
    different = CenterHalfWidthActionGrid([25, 100, 150, 200], EXAMPLE.half_widths)
    assert CenterHalfWidthLPActionCodec(500, different).action_space == codec.action_space
    with pytest.raises(ValueError, match="metadata"):
        validate_policy_metadata(model, 500, 5, action_grid=different)
    with pytest.raises(ValueError, match="require saved action metadata"):
        validate_policy_metadata(SimpleNamespace(), 500, 5, action_grid=EXAMPLE)
    invalid = {**codec.metadata, "centers": list(reversed(EXAMPLE.centers))}
    with pytest.raises(ValueError, match="Invalid saved"):
        action_grid_from_metadata(invalid)
    bounds = _plot_action_bounds(model, np.array([0, 25, 54]), 500, None)
    assert np.isnan(bounds[0]).all()
    np.testing.assert_array_equal(bounds[1:], [[-50, 50], [-100, 500]])


@pytest.mark.parametrize("flags", [
    ["--center-magnitudes", "50"], ["--half-widths", "50"],
    ["--center-magnitudes", "200", "--half-widths", "300"],
])
def test_invalid_cli_grid_fails_before_run_directory_creation(tmp_path, monkeypatch, flags):
    args = training.parse_args(["--output-dir", str(tmp_path / "runs"), *flags])
    monkeypatch.setattr(training, "parse_args", lambda: args)
    with pytest.raises(ValueError):
        training.main()
    assert not (tmp_path / "runs").exists()


def test_default_training_encoding_remains_endpoint_pairs():
    args = training.parse_args([])
    training.configure_action_encoding(args)
    assert args.action_encoding == CATEGORICAL_ENCODING
    assert args.action_count == 211


def test_saved_run_grid_mismatch_rejected_even_with_same_action_count(tmp_path):
    args = training.parse_args([
        "--tau", "500", "--center-magnitudes", "50", "100", "150", "200",
        "--half-widths", "50", "100", "150", "200", "250", "300",
    ])
    training.configure_action_encoding(args)
    args.center_magnitudes = [25, 100, 150, 200]
    (tmp_path / "config.json").write_text(json.dumps(vars(args)))
    with pytest.raises(ValueError, match="metadata"):
        evaluation.load_config(tmp_path)


def test_grid_ppo_save_load_and_standalone_evaluation(tmp_path):
    args = training.parse_args([
        "--tau", "500", "--center-magnitudes", "50", "100", "150", "200",
        "--half-widths", "50", "100", "150", "200", "250", "300",
        "--n-steps", "4", "--num-trajectories", "2", "--decision-stride", "2",
        "--periodic-rebalance-every", "2", "--n-eval-episodes", "2",
    ])
    training.validate_and_derive_decision_timing(args)
    training.configure_action_encoding(args)
    params = training.DomainParameters(args.nominal_sigma, args.nominal_arrival_rate)
    model, vec, normalizer = training.build_ppo(training.make_fixed_env(args, params, 7), args)
    model.n_epochs = 1
    model.learn(total_timesteps=4)
    assert model.action_space == gym.spaces.Discrete(55)
    model.save(tmp_path / "nominal_ppo")
    normalizer.save(tmp_path / "nominal_vecnormalize.pkl")
    (tmp_path / "config.json").write_text(json.dumps(vars(args)))
    config = evaluation.load_config(tmp_path)
    assert config["lp_action_config"]["centers"] == list(EXAMPLE.centers)
    loaded = PPO.load(tmp_path / "nominal_ppo")
    training.validate_policy_action_encoding(loaded, SimpleNamespace(**config))
    observations = vec.reset()
    np.testing.assert_array_equal(model.predict(observations, deterministic=True)[0],
                                  loaded.predict(observations, deterministic=True)[0])
    normalizer.training = False
    expected = training.evaluate_ppo("nominal_ppo", model, normalizer, params, "in_distribution", args, 99)
    actual = training.evaluate_ppo("nominal_ppo", loaded, normalizer, params, "in_distribution", args, 99)
    assert actual == expected
    assert evaluation.main([
        "--run-dirs", str(tmp_path), "--output-dir", str(tmp_path / "evaluation"),
        "--policies", "nominal_ppo", "--sigma-values", ".03", "--arrival-rate-values", "300",
        "--bootstrap-resamples", "10", "--no-plots",
    ]) == 0
    vec.close()
