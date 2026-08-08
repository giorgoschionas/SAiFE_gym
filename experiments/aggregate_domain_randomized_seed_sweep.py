"""Aggregate domain-randomized PPO results across independent training seeds."""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.policy_behavior_diagnostics import (  # noqa: E402
    BEHAVIOR_DIAGNOSTIC_COLUMNS,
    FRACTION_BEHAVIOR_DIAGNOSTIC_COLUMNS,
    NONNEGATIVE_BEHAVIOR_DIAGNOSTIC_COLUMNS,
)


RESULT_FILENAME = "evaluation_grid.csv"
CONFIG_FILENAME = "config.json"
EXPECTED_POLICIES = {
    "cash",
    "domain_randomized_ppo",
    "nominal_ppo",
    "periodic_rebalance",
}
REGIME_FIELDS = (
    "evaluation_set",
    "policy",
    "sigma",
    "arrival_rate",
)
DOMAIN_FIELDS = (
    "evaluation_set",
    "sigma",
    "arrival_rate",
)
REQUIRED_RESULT_COLUMNS = {
    *REGIME_FIELDS,
    "training_seed",
    "evaluation_seed",
    "mean_running_inventory_objective",
    "evaluation_path_std_running_inventory_objective",
    "n_evaluation_paths",
    *BEHAVIOR_DIAGNOSTIC_COLUMNS,
    "domain_randomized_vs_nominal_mean_running_inventory_objective_gap",
    (
        "domain_randomized_vs_periodic_rebalance_"
        "mean_running_inventory_objective_gap"
    ),
    "domain_randomized_vs_cash_mean_running_inventory_objective_gap",
}
LEGACY_RESULT_COLUMNS = {
    "std_running_inventory_objective",
    "n_samples",
}
GAP_COLUMNS = {
    "domain_randomized_vs_nominal": (
        "domain_randomized_vs_nominal_mean_running_inventory_objective_gap"
    ),
    "domain_randomized_vs_periodic_rebalance": (
        "domain_randomized_vs_periodic_rebalance_"
        "mean_running_inventory_objective_gap"
    ),
    "domain_randomized_vs_cash": (
        "domain_randomized_vs_cash_mean_running_inventory_objective_gap"
    ),
}
SET_METRICS = {
    "mean_of_regime_mean_running_inventory_objective": np.mean,
    "minimum_regime_mean_running_inventory_objective": np.min,
    "maximum_regime_mean_running_inventory_objective": np.max,
}


@dataclass(frozen=True)
class SeedRun:
    run_dir: Path
    config: dict
    training_seed: int
    evaluation_seed: int
    rows_by_key: dict[tuple, dict]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate evaluation results across independent PPO training seeds."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args(argv)


def validate_aggregation_args(args: argparse.Namespace) -> None:
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be at least 1")
    if args.bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing required run artifact: {path}")
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _parse_int(value: str, name: str, result_path: Path) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} in {result_path}: {value!r}") from exc
    return parsed


def _parse_float(value: str, name: str, result_path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} in {result_path}: {value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"non-finite {name} in {result_path}: {value!r}")
    return parsed


def _read_result_rows(result_path: Path) -> list[dict]:
    with result_path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        legacy = columns & LEGACY_RESULT_COLUMNS
        if legacy:
            raise ValueError(
                f"legacy result schema in {result_path}; unsupported columns: "
                f"{sorted(legacy)}"
            )
        missing = REQUIRED_RESULT_COLUMNS - columns
        if missing:
            raise ValueError(
                f"incomplete result schema in {result_path}; missing columns: "
                f"{sorted(missing)}"
            )

        rows = []
        for raw_row in reader:
            row = {
                "evaluation_set": raw_row["evaluation_set"],
                "training_seed": _parse_int(
                    raw_row["training_seed"], "training_seed", result_path
                ),
                "evaluation_seed": _parse_int(
                    raw_row["evaluation_seed"], "evaluation_seed", result_path
                ),
                "policy": raw_row["policy"],
                "sigma": _parse_float(raw_row["sigma"], "sigma", result_path),
                "arrival_rate": _parse_float(
                    raw_row["arrival_rate"], "arrival_rate", result_path
                ),
                "mean_running_inventory_objective": _parse_float(
                    raw_row["mean_running_inventory_objective"],
                    "mean_running_inventory_objective",
                    result_path,
                ),
                "evaluation_path_std_running_inventory_objective": _parse_float(
                    raw_row["evaluation_path_std_running_inventory_objective"],
                    "evaluation_path_std_running_inventory_objective",
                    result_path,
                ),
                "n_evaluation_paths": _parse_int(
                    raw_row["n_evaluation_paths"],
                    "n_evaluation_paths",
                    result_path,
                ),
            }
            for gap_column in GAP_COLUMNS.values():
                row[gap_column] = _parse_float(
                    raw_row[gap_column], gap_column, result_path
                )
            for diagnostic in BEHAVIOR_DIAGNOSTIC_COLUMNS:
                row[diagnostic] = _parse_float(
                    raw_row[diagnostic], diagnostic, result_path
                )
            if row["evaluation_path_std_running_inventory_objective"] < 0.0:
                raise ValueError(
                    "evaluation path standard deviation must be non-negative "
                    f"in {result_path}"
                )
            if row["n_evaluation_paths"] < 1:
                raise ValueError(
                    f"n_evaluation_paths must be positive in {result_path}"
                )
            for diagnostic in NONNEGATIVE_BEHAVIOR_DIAGNOSTIC_COLUMNS:
                if row[diagnostic] < -1e-10:
                    raise ValueError(
                        f"{diagnostic} must be non-negative in {result_path}"
                    )
            for diagnostic in FRACTION_BEHAVIOR_DIAGNOSTIC_COLUMNS:
                if not -1e-10 <= row[diagnostic] <= 1.0 + 1e-10:
                    raise ValueError(
                        f"{diagnostic} must lie in [0, 1] in {result_path}"
                    )
            decomposed_objective = (
                row["mean_pnl_per_path"]
                - row["mean_inventory_penalty_per_path"]
            )
            if not np.isclose(
                row["mean_running_inventory_objective"],
                decomposed_objective,
                rtol=1e-9,
                atol=1e-7,
            ):
                raise ValueError(
                    "objective does not equal PnL minus inventory penalty in "
                    f"{result_path}"
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"no result rows found in {result_path}")
    return rows


def _validate_run_rows(
    rows: list[dict],
    config: dict,
    result_path: Path,
) -> dict[tuple, dict]:
    if "seed" not in config or "evaluation_seed" not in config:
        raise ValueError(
            f"config in {result_path.parent} must contain seed and evaluation_seed"
        )
    training_seed = int(config["seed"])
    evaluation_seed = int(config["evaluation_seed"])
    rows_by_key = {}
    domain_rows: dict[tuple, list[dict]] = {}

    for row in rows:
        if row["training_seed"] != training_seed:
            raise ValueError(
                f"row training_seed does not match config in {result_path}"
            )
        if row["evaluation_seed"] != evaluation_seed:
            raise ValueError(
                f"row evaluation_seed does not match config in {result_path}"
            )
        key = tuple(row[field] for field in REGIME_FIELDS)
        if key in rows_by_key:
            raise ValueError(f"duplicate regime-policy row {key} in {result_path}")
        rows_by_key[key] = row
        domain_key = tuple(row[field] for field in DOMAIN_FIELDS)
        domain_rows.setdefault(domain_key, []).append(row)

    evaluation_sets = {row["evaluation_set"] for row in rows}
    if evaluation_sets != {"in_distribution", "stress"}:
        raise ValueError(
            f"evaluation sets in {result_path} are {sorted(evaluation_sets)}, "
            "expected ['in_distribution', 'stress']"
        )

    for domain_key, grouped_rows in domain_rows.items():
        policies = {row["policy"] for row in grouped_rows}
        if policies != EXPECTED_POLICIES:
            raise ValueError(
                f"domain {domain_key} in {result_path} has policies "
                f"{sorted(policies)}, expected {sorted(EXPECTED_POLICIES)}"
            )
        for gap_column in GAP_COLUMNS.values():
            gap_values = {row[gap_column] for row in grouped_rows}
            if len(gap_values) != 1:
                raise ValueError(
                    f"inconsistent {gap_column} for domain {domain_key} in "
                    f"{result_path}"
                )

    return rows_by_key


def _config_signature(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in {"seed", "output_dir"}
    }


def discover_seed_runs(input_dir: Path) -> list[SeedRun]:
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    result_paths = sorted(input_dir.rglob(RESULT_FILENAME))
    if not result_paths:
        raise ValueError(f"no {RESULT_FILENAME} files found under {input_dir}")

    runs = []
    seen_training_seeds = set()
    reference_signature = None
    reference_keys = None
    reference_path_counts = None

    for result_path in result_paths:
        config = _load_json(result_path.parent / CONFIG_FILENAME)
        rows = _read_result_rows(result_path)
        rows_by_key = _validate_run_rows(rows, config, result_path)
        training_seed = int(config["seed"])
        evaluation_seed = int(config["evaluation_seed"])

        if training_seed in seen_training_seeds:
            raise ValueError(f"duplicate training seed discovered: {training_seed}")
        seen_training_seeds.add(training_seed)

        signature = _config_signature(config)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            differing_keys = sorted({
                key
                for key in set(signature) | set(reference_signature)
                if signature.get(key) != reference_signature.get(key)
            })
            raise ValueError(
                f"incompatible run configuration for seed {training_seed}; "
                f"differing keys: {differing_keys}"
            )

        keys = set(rows_by_key)
        path_counts = {
            key: row["n_evaluation_paths"] for key, row in rows_by_key.items()
        }
        if reference_keys is None:
            reference_keys = keys
            reference_path_counts = path_counts
        elif keys != reference_keys:
            raise ValueError(
                f"incomplete or inconsistent regime coverage for seed {training_seed}"
            )
        elif path_counts != reference_path_counts:
            raise ValueError(
                f"mismatched evaluation path counts for seed {training_seed}"
            )

        runs.append(
            SeedRun(
                run_dir=result_path.parent,
                config=config,
                training_seed=training_seed,
                evaluation_seed=evaluation_seed,
                rows_by_key=rows_by_key,
            )
        )

    if len(runs) < 2:
        raise ValueError("at least two distinct training-seed runs are required")
    return sorted(runs, key=lambda run: run.training_seed)


def make_bootstrap_indices(
    num_training_seeds: int,
    num_resamples: int,
    bootstrap_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(bootstrap_seed)
    return rng.integers(
        0,
        num_training_seeds,
        size=(num_resamples, num_training_seeds),
    )


def summarize_seed_values(
    values: np.ndarray,
    bootstrap_indices: np.ndarray,
    confidence_level: float,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("seed-level values must be one-dimensional with size >= 2")
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    sample_std = float(np.std(values, ddof=1))
    return {
        "n_training_seeds": int(values.size),
        "mean_across_training_seeds": float(np.mean(values)),
        "training_seed_sample_std": sample_std,
        "training_seed_standard_error": float(sample_std / np.sqrt(values.size)),
        "confidence_level": float(confidence_level),
        "training_seed_confidence_interval_lower": float(lower),
        "training_seed_confidence_interval_upper": float(upper),
    }


def aggregate_regimes(
    runs: list[SeedRun],
    bootstrap_indices: np.ndarray,
    confidence_level: float,
) -> list[dict]:
    rows = []
    for key in sorted(runs[0].rows_by_key):
        seed_rows = [run.rows_by_key[key] for run in runs]
        values = np.array([
            row["mean_running_inventory_objective"] for row in seed_rows
        ])
        stats = summarize_seed_values(values, bootstrap_indices, confidence_level)
        rows.append({
            **{field: value for field, value in zip(REGIME_FIELDS, key)},
            "n_training_seeds": stats["n_training_seeds"],
            "n_evaluation_paths_per_training_seed": seed_rows[0][
                "n_evaluation_paths"
            ],
            "mean_running_inventory_objective_across_training_seeds": stats[
                "mean_across_training_seeds"
            ],
            "training_seed_std_running_inventory_objective": stats[
                "training_seed_sample_std"
            ],
            "training_seed_standard_error_running_inventory_objective": stats[
                "training_seed_standard_error"
            ],
            "confidence_level": stats["confidence_level"],
            "training_seed_confidence_interval_lower_running_inventory_objective": (
                stats["training_seed_confidence_interval_lower"]
            ),
            "training_seed_confidence_interval_upper_running_inventory_objective": (
                stats["training_seed_confidence_interval_upper"]
            ),
            "mean_evaluation_path_std_running_inventory_objective": float(
                np.mean([
                    row["evaluation_path_std_running_inventory_objective"]
                    for row in seed_rows
                ])
            ),
        })
    return rows


def aggregate_gaps(
    runs: list[SeedRun],
    bootstrap_indices: np.ndarray,
    confidence_level: float,
) -> list[dict]:
    domain_keys = sorted({
        tuple(row[field] for field in DOMAIN_FIELDS)
        for row in runs[0].rows_by_key.values()
    })
    rows = []
    for domain_key in domain_keys:
        domain_mapping = dict(zip(DOMAIN_FIELDS, domain_key))
        policy_key = (
            domain_mapping["evaluation_set"],
            "domain_randomized_ppo",
            domain_mapping["sigma"],
            domain_mapping["arrival_rate"],
        )
        for comparison, gap_column in GAP_COLUMNS.items():
            values = np.array([
                run.rows_by_key[policy_key][gap_column] for run in runs
            ])
            stats = summarize_seed_values(
                values,
                bootstrap_indices,
                confidence_level,
            )
            rows.append({
                **domain_mapping,
                "comparison": comparison,
                "n_training_seeds": stats["n_training_seeds"],
                "mean_gap_across_training_seeds": stats[
                    "mean_across_training_seeds"
                ],
                "training_seed_std_gap": stats["training_seed_sample_std"],
                "training_seed_standard_error_gap": stats[
                    "training_seed_standard_error"
                ],
                "confidence_level": stats["confidence_level"],
                "training_seed_confidence_interval_lower_gap": stats[
                    "training_seed_confidence_interval_lower"
                ],
                "training_seed_confidence_interval_upper_gap": stats[
                    "training_seed_confidence_interval_upper"
                ],
            })
    return rows


def aggregate_behavior_diagnostics(
    runs: list[SeedRun],
    bootstrap_indices: np.ndarray,
    confidence_level: float,
) -> list[dict]:
    """Return long-form training-seed statistics for every behavior metric."""
    rows = []
    for key in sorted(runs[0].rows_by_key):
        seed_rows = [run.rows_by_key[key] for run in runs]
        regime_mapping = dict(zip(REGIME_FIELDS, key))
        for diagnostic in BEHAVIOR_DIAGNOSTIC_COLUMNS:
            values = np.array([row[diagnostic] for row in seed_rows])
            stats = summarize_seed_values(
                values,
                bootstrap_indices,
                confidence_level,
            )
            rows.append({
                **regime_mapping,
                "diagnostic": diagnostic,
                "n_training_seeds": stats["n_training_seeds"],
                "mean_across_training_seeds": stats[
                    "mean_across_training_seeds"
                ],
                "training_seed_sample_std": stats[
                    "training_seed_sample_std"
                ],
                "training_seed_standard_error": stats[
                    "training_seed_standard_error"
                ],
                "confidence_level": stats["confidence_level"],
                "training_seed_confidence_interval_lower": stats[
                    "training_seed_confidence_interval_lower"
                ],
                "training_seed_confidence_interval_upper": stats[
                    "training_seed_confidence_interval_upper"
                ],
            })
    return rows


def aggregate_set_summaries(
    runs: list[SeedRun],
    bootstrap_indices: np.ndarray,
    confidence_level: float,
) -> dict:
    policies = sorted({key[1] for key in runs[0].rows_by_key})
    evaluation_sets = sorted({key[0] for key in runs[0].rows_by_key})
    summary = {}
    for policy in policies:
        summary[policy] = {}
        for evaluation_set in evaluation_sets:
            seed_regime_values = []
            for run in runs:
                seed_regime_values.append([
                    row["mean_running_inventory_objective"]
                    for key, row in run.rows_by_key.items()
                    if key[0] == evaluation_set and key[1] == policy
                ])
            summary[policy][evaluation_set] = {}
            for metric_name, reducer in SET_METRICS.items():
                values = np.array([
                    reducer(regime_values)
                    for regime_values in seed_regime_values
                ])
                summary[policy][evaluation_set][metric_name] = (
                    summarize_seed_values(
                        values,
                        bootstrap_indices,
                        confidence_level,
                    )
                )
            summary[policy][evaluation_set][
                "behavior_diagnostics_mean_across_regimes"
            ] = {
                diagnostic: summarize_seed_values(
                    np.array([
                        np.mean([
                            row[diagnostic]
                            for key, row in run.rows_by_key.items()
                            if key[0] == evaluation_set and key[1] == policy
                        ])
                        for run in runs
                    ]),
                    bootstrap_indices,
                    confidence_level,
                )
                for diagnostic in BEHAVIOR_DIAGNOSTIC_COLUMNS
            }
    return summary


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def aggregate_seed_sweep(args: argparse.Namespace) -> Path:
    validate_aggregation_args(args)
    input_dir = Path(args.input_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else input_dir / "aggregate"
    )
    runs = discover_seed_runs(input_dir)
    bootstrap_indices = make_bootstrap_indices(
        len(runs),
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    regime_rows = aggregate_regimes(
        runs,
        bootstrap_indices,
        args.confidence_level,
    )
    gap_rows = aggregate_gaps(
        runs,
        bootstrap_indices,
        args.confidence_level,
    )
    behavior_rows = aggregate_behavior_diagnostics(
        runs,
        bootstrap_indices,
        args.confidence_level,
    )
    summary = {
        "summary_scope": "training_seed_aggregate",
        "n_training_seeds": len(runs),
        "training_seeds": [run.training_seed for run in runs],
        "evaluation_seed": runs[0].evaluation_seed,
        "bootstrap": {
            "method": "percentile",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "confidence_level": args.confidence_level,
        },
        "source_runs": [
            {
                "training_seed": run.training_seed,
                "run_dir": str(run.run_dir.relative_to(input_dir)),
            }
            for run in runs
        ],
        "policies": aggregate_set_summaries(
            runs,
            bootstrap_indices,
            args.confidence_level,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "training_seed_regime_summary.csv", regime_rows)
    save_csv(output_dir / "training_seed_gap_summary.csv", gap_rows)
    save_csv(output_dir / "training_seed_behavior_summary.csv", behavior_rows)
    save_json(output_dir / "training_seed_summary.json", summary)
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = aggregate_seed_sweep(args)
    print(f"Saved training-seed aggregate artifacts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
