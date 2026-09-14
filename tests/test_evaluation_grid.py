from collections import Counter
from itertools import product

import pytest

from experiments.evaluation_grid import configured_evaluation_grid


def grid_config():
    return {
        "evaluation_grid_version": 2,
        "train_sigma_range": [.01, .05],
        "train_arrival_rate_range": [200., 400.],
        "eval_in_distribution_sigma_values": [.01, .05],
        "eval_in_distribution_arrival_rate_values": [200., 400.],
        "eval_stress_sigma_values": [.0, .08],
        "eval_stress_arrival_rate_values": [150., 450.],
    }


def test_support_boundaries_and_lower_stress_values_are_classified_correctly():
    rows = configured_evaluation_grid(grid_config())
    assert len(rows) == 16
    assert Counter(group for group, _, _ in rows) == {
        "in_distribution": 4, "stress": 4,
        "sigma_only_stress": 4, "arrival_only_stress": 4,
    }
    assert ("in_distribution", .05, 400.) in rows
    assert ("in_distribution", .01, 200.) in rows
    assert ("sigma_only_stress", .0, 400.) in rows
    assert ("arrival_only_stress", .05, 150.) in rows


@pytest.mark.parametrize("overlap_parameter", ["sigma", "arrival_rate"])
def test_overlapping_axes_are_deduplicated_and_classified_by_actual_support(overlap_parameter):
    config = grid_config()
    config[f"eval_stress_{overlap_parameter}_values"].insert(
        0, config[f"eval_in_distribution_{overlap_parameter}_values"][0],
    )
    rows = configured_evaluation_grid(config)
    coordinates = [(sigma, arrival) for _, sigma, arrival in rows]
    assert len(coordinates) == len(set(coordinates)) == 16
    if overlap_parameter == "sigma":
        assert rows[4] == ("arrival_only_stress", .01, 150.)
    else:
        assert rows[4] == ("sigma_only_stress", .0, 200.)


def test_additional_in_support_stress_axis_values_join_the_in_distribution_group():
    config = grid_config()
    config["eval_stress_sigma_values"] = [.03, .08]
    rows = configured_evaluation_grid(config)
    assert ("in_distribution", .03, 200.) in rows
    assert ("arrival_only_stress", .03, 150.) in rows
    assert len(rows) == len({(s, a) for _, s, a in rows}) == 16


def test_equal_training_bounds_are_valid():
    config = grid_config()
    config.update(train_sigma_range=[.03, .03], eval_in_distribution_sigma_values=[.03])
    rows = configured_evaluation_grid(config)
    assert ("in_distribution", .03, 200.) in rows
    assert len(rows) == 12


@pytest.mark.parametrize("version", [None, 0, 3, "2", True, 2.0])
def test_unknown_or_malformed_grid_versions_are_rejected(version):
    config = grid_config()
    config["evaluation_grid_version"] = version
    with pytest.raises(ValueError, match="evaluation_grid_version"):
        configured_evaluation_grid(config)


@pytest.mark.parametrize("explicit_version", [False, True])
def test_legacy_grid_does_not_require_training_bounds(explicit_version):
    config = grid_config()
    del config["train_sigma_range"]
    del config["train_arrival_rate_range"]
    if explicit_version:
        config["evaluation_grid_version"] = 1
    else:
        del config["evaluation_grid_version"]
    rows = configured_evaluation_grid(config)
    assert rows == [
        (label, sigma, arrival)
        for label in ["in_distribution", "stress"]
        for sigma, arrival in product(
            config[f"eval_{label}_sigma_values"], config[f"eval_{label}_arrival_rate_values"],
        )
    ]


@pytest.mark.parametrize("bounds", [[.05, .01], [.01], [False, .05], [0., float("nan")]])
def test_invalid_training_bounds_are_rejected(bounds):
    config = grid_config()
    config["train_sigma_range"] = bounds
    with pytest.raises(ValueError, match="train_sigma_range"):
        configured_evaluation_grid(config)
