"""Exact factorized grids, economic entropy, and legacy-compatible adapters."""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence
import numpy as np
import pytest
import torch
from stable_baselines3.common.distributions import CategoricalDistribution, MultiCategoricalDistribution

from SAiFE_gym.gym.lp_actions import (
    CenterHalfWidthActionGrid, CenterHalfWidthLPActionCodec,
    MultiDiscreteCenterHalfWidthLPActionCodec, action_grid_from_metadata,
    decode_policy_actions, validate_policy_metadata,
)
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from experiments import train_robust_lp_agent as training
from experiments.helpers import _plot_action_bounds
from experiments.learning_diagnostics import economic_action_probabilities
from test_stable_baselines_amm_environment import create_test_amm_env


GRID = CenterHalfWidthActionGrid([50, 100, 150, 200], [50, 100, 150, 200, 250, 300])


def test_all_108_combinations_are_exactly_55_economic_actions():
    codec = MultiDiscreteCenterHalfWidthLPActionCodec(500, GRID)
    flat = CenterHalfWidthLPActionCodec(500, GRID)
    indices = np.array(list(product(range(9), range(6), range(2))))
    commands = codec.decode(indices)
    assert codec.action_space == gym.spaces.MultiDiscrete([9, 6, 2])
    assert len(np.unique(commands, axis=0)) == 55
    np.testing.assert_array_equal(commands[indices[:, 2] == 0], flat.action_table[1:])
    np.testing.assert_array_equal(commands[indices[:, 2] == 1], np.tile([0, 1, 1], (54, 1)))
    np.testing.assert_array_equal(codec.decode([8, 5, 0]), [-100, 500, -1])
    np.testing.assert_array_equal(decode_policy_actions(indices, codec.action_space, 500, action_grid=GRID), commands)
    assert codec.metadata['action_count'] == 108
    assert codec.metadata['economic_action_count'] == 55
    assert codec.metadata['component_order'] == ['center_index', 'half_width_index', 'hold']
    assert action_grid_from_metadata(codec.metadata) == GRID


@pytest.mark.parametrize('indices', [
    [9, 0, 0], [-1, 0, 1], [0, 6, 1], [0, 0, 2], [0., 0., 1.],
    [False, False, True], [0, 0], 0, [[0, 0, 0], [0, -1, 0]],
])
def test_invalid_indices_including_hold_aliases_raise(indices):
    with pytest.raises(ValueError, match='indices'):
        MultiDiscreteCenterHalfWidthLPActionCodec(500, GRID).decode(indices)


@pytest.mark.parametrize('key,value', [
    ('centers', [-200, -150, -100, -25, 0, 25, 100, 150, 200]),
    ('nvec', [6, 9, 2]), ('hold_value', 0), ('rebalance_value', 1),
    ('component_order', ['hold', 'center_index', 'half_width_index']),
    ('action_count', 55), ('economic_action_count', 108), ('tau', 499),
])
def test_metadata_mismatches_rejected(key, value):
    codec = MultiDiscreteCenterHalfWidthLPActionCodec(500, GRID)
    metadata = {**codec.metadata, key: value}
    model = SimpleNamespace(action_space=codec.action_space, lp_action_config=metadata)
    with pytest.raises(ValueError, match='metadata'):
        validate_policy_metadata(model, 500, 5, action_grid=GRID)
    # Different centers can form a valid new grid but must not match the configured one.
    if key != 'centers':
        with pytest.raises(ValueError, match='metadata'):
            action_grid_from_metadata(metadata)


def test_grid_bounds_precision_and_required_explicit_metadata():
    with pytest.raises(ValueError, match='requires tau >= 500'):
        MultiDiscreteCenterHalfWidthLPActionCodec(499, GRID)
    with pytest.raises(ValueError, match='exactly representable'):
        MultiDiscreteCenterHalfWidthLPActionCodec(2**24 + 1, CenterHalfWidthActionGrid([2**24], [1]))
    with pytest.raises(ValueError, match='explicit'):
        training.configure_action_encoding(training.parse_args(['--policy-action-space', 'multidiscrete']))
    codec = MultiDiscreteCenterHalfWidthLPActionCodec(500, GRID)
    model = SimpleNamespace(action_space=codec.action_space)
    with pytest.raises(ValueError, match='require saved'):
        validate_policy_metadata(model, 500, 5, action_grid=GRID)
    model.lp_action_config = codec.metadata
    with pytest.raises(ValueError, match='action space'):
        decode_policy_actions([0, 0, 0], gym.spaces.MultiDiscrete([6, 9, 2]), 500, action_grid=GRID)
    bounds = _plot_action_bounds(model, np.array([[8, 5, 1], [0, 0, 0], [8, 5, 0]]), 500, None)
    assert np.isnan(bounds[0]).all()
    np.testing.assert_array_equal(bounds[1:], [[-250, -150], [-100, 500]])


@pytest.mark.parametrize('n', [1, 3])
def test_adapter_and_stride_use_valid_integer_holds(n):
    base = create_test_amm_env(num_trajectories=n, n_steps=8)
    reference = deepcopy(base)
    # One center and width: an incorrect [0, 1, 1] hold cannot pass validation.
    adapter = StableBaselinesAMMEnvironment(base, action_encoding='multidiscrete',
                                           action_grid=CenterHalfWidthActionGrid([0], [1]))
    stride = training.DecisionStrideVecEnv(adapter, 3)
    stride.reset()
    reference.reset()
    assert stride._hold.dtype == np.int64
    assert all(adapter.action_space.contains(action) for action in stride._hold)
    actions = np.tile([0, 0, 0], (n, 1))
    reward = reference.step(adapter.decode_actions(actions))[1]
    for _ in range(2):
        reward += reference.step(np.tile([0, 1, 1], (n, 1)))[1]
    np.testing.assert_allclose(stride.step(actions)[1], reward)
    assert data_equivalence(base.state, reference.state, exact=True)
    with pytest.raises(ValueError, match='shape'):
        adapter.step(np.zeros(n, dtype=int))


def test_entropy_merges_hold_aliases_and_matches_equivalent_flat_distribution():
    space = gym.spaces.MultiDiscrete([9, 6, 2])
    distribution = MultiCategoricalDistribution([9, 6, 2]).proba_distribution(torch.zeros(1, 17))
    probabilities = economic_action_probabilities(distribution, space)
    assert probabilities.shape == (1, 55)
    assert probabilities[0, 0] == pytest.approx(.5)
    np.testing.assert_allclose(probabilities[0, 1:], 1 / 108)
    entropy = -torch.special.xlogy(probabilities, probabilities).sum()
    assert entropy.item() == pytest.approx(np.log(2) + .5 * np.log(54))
    assert distribution.entropy().item() == pytest.approx(np.log(108))
    flat = CategoricalDistribution(55).proba_distribution(probabilities.log())
    assert flat.entropy().item() == pytest.approx(entropy.item())
    # Nonuniform heads verify that aggregation is the joint range probability.
    distribution = MultiCategoricalDistribution([9, 6, 2]).proba_distribution(torch.arange(17.).reshape(1, -1))
    actual = economic_action_probabilities(distribution, space)
    heads = [d.probs.numpy()[0] for d in distribution.distribution]
    expected = [heads[2][1]] + [heads[0][i] * heads[1][j] * heads[2][0] for i in range(9) for j in range(6)]
    np.testing.assert_allclose(actual[0], expected)
