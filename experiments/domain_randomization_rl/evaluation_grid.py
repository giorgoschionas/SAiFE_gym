"""Evaluation-grid layout shared by training and result aggregation.

This module uses only the standard library so aggregation does not need SB3.
Unversioned saved configs describe the original two Cartesian grids; version 2
also crosses their axes and classifies domains by actual training support.
"""

from collections.abc import Mapping
from itertools import chain, product
from math import isfinite
from numbers import Integral, Real
from typing import Literal


EVALUATION_GRID_VERSION = 2
EvaluationSet = Literal[
    "in_distribution", "sigma_only_stress", "arrival_only_stress", "stress",
]
EVALUATION_SETS: tuple[EvaluationSet, ...] = (
    "in_distribution", "sigma_only_stress", "arrival_only_stress", "stress",
)


def _numeric_values(config: Mapping, name: str) -> list[float]:
    if name not in config:
        raise ValueError(f"missing required {name}")
    values = config[name]
    if not isinstance(values, (list, tuple)) or not values or any(
        isinstance(value, bool) or not isinstance(value, Real) for value in values
    ):
        raise ValueError(f"{name} must be a nonempty one-dimensional list of numbers")
    try:
        result = [float(value) for value in values]
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not all(isfinite(value) and value >= 0.0 for value in result):
        raise ValueError(f"{name} must contain only finite, non-negative values")
    return result


def _axis(config: Mapping, name: str) -> list[float]:
    values = _numeric_values(config, name)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values")
    return values


def _training_range(config: Mapping, name: str) -> tuple[float, float]:
    values = _numeric_values(config, name)
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError(f"{name} must contain two bounds in ascending order")
    return values[0], values[1]


def configured_evaluation_grid(
    config: Mapping, *, version: int | None = None,
) -> list[tuple[EvaluationSet, float, float]]:
    """Return validated, ordered ``(evaluation_set, sigma, arrival)`` entries.

    Training explicitly requests the current version. Aggregation reads the
    saved version, defaulting to the legacy layout when the marker is absent.
    Original regimes come first to preserve their index-based evaluation seeds.
    """
    grid_version = (
        config.get("evaluation_grid_version", 1) if version is None else version
    )
    if (
        isinstance(grid_version, bool)
        or not isinstance(grid_version, Integral)
        or grid_version not in (1, EVALUATION_GRID_VERSION)
    ):
        raise ValueError(f"unsupported evaluation_grid_version: {grid_version!r}")

    in_sigma = _axis(config, "eval_in_distribution_sigma_values")
    in_arrival = _axis(config, "eval_in_distribution_arrival_rate_values")
    stress_sigma = _axis(config, "eval_stress_sigma_values")
    stress_arrival = _axis(config, "eval_stress_arrival_rate_values")
    original: list[tuple[EvaluationSet, float, float]] = [
        ("in_distribution", sigma, arrival)
        for sigma, arrival in product(in_sigma, in_arrival)
    ] + [
        ("stress", sigma, arrival)
        for sigma, arrival in product(stress_sigma, stress_arrival)
    ]
    if grid_version == 1:
        return original

    sigma_low, sigma_high = _training_range(config, "train_sigma_range")
    arrival_low, arrival_high = _training_range(config, "train_arrival_rate_range")
    for name, axis, low, high in (
        ("eval_in_distribution_sigma_values", in_sigma, sigma_low, sigma_high),
        ("eval_in_distribution_arrival_rate_values", in_arrival, arrival_low, arrival_high),
    ):
        outside = [value for value in axis if not low <= value <= high]
        if outside:
            raise ValueError(
                f"{name} must lie within training range [{low}, {high}], "
                f"got out-of-support values {outside}"
            )
    invalid_stress = [
        (sigma, arrival) for sigma, arrival in product(stress_sigma, stress_arrival)
        if sigma_low <= sigma <= sigma_high and arrival_low <= arrival <= arrival_high
    ]
    if invalid_stress:
        raise ValueError(
            "every stress evaluation regime must have at least one parameter "
            "outside the training support; fully in-support regimes: "
            f"{invalid_stress}"
        )

    groups: dict[tuple[bool, bool], EvaluationSet] = {
        (True, True): "in_distribution",
        (False, True): "sigma_only_stress",
        (True, False): "arrival_only_stress",
        (False, False): "stress",
    }
    candidates = chain(
        ((sigma, arrival) for _, sigma, arrival in original),
        product(in_sigma, stress_arrival),
        product(stress_sigma, in_arrival),
    )
    seen = set()
    result = []
    for sigma, arrival in candidates:
        if (sigma, arrival) in seen:
            continue
        seen.add((sigma, arrival))
        in_support = (
            sigma_low <= sigma <= sigma_high,
            arrival_low <= arrival <= arrival_high,
        )
        group = groups[in_support]
        result.append((group, sigma, arrival))
    return result
