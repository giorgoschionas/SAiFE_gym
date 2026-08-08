import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.aggregate_domain_randomized_seed_sweep import (
    aggregate_seed_sweep,
    discover_seed_runs,
    parse_args,
)
from experiments.policy_behavior_diagnostics import (
    BEHAVIOR_DIAGNOSTIC_COLUMNS,
    cash_behavior_diagnostics,
    zero_behavior_diagnostics,
)


POLICIES = (
    "domain_randomized_ppo",
    "nominal_ppo",
    "periodic_rebalance",
    "cash",
)


def _policy_values(seed_index: int, evaluation_set: str) -> dict[str, float]:
    if evaluation_set == "in_distribution":
        return {
            "domain_randomized_ppo": 3.0 + 2.0 * seed_index,
            "nominal_ppo": 1.0 + seed_index,
            "periodic_rebalance": -1.0,
            "cash": 0.0,
        }
    return {
        "domain_randomized_ppo": -2.0 + 2.0 * seed_index,
        "nominal_ppo": -3.0 + 2.0 * seed_index,
        "periodic_rebalance": -4.0,
        "cash": 0.0,
    }


def _behavior_values(
    policy: str,
    objective: float,
    seed_index: int,
) -> dict[str, float]:
    if policy == "cash":
        return cash_behavior_diagnostics()

    diagnostics = zero_behavior_diagnostics()
    seed_variation = seed_index if policy in {
        "domain_randomized_ppo",
        "nominal_ppo",
    } else 0
    diagnostics.update({
        "never_deployed_fraction": 0.1 * seed_variation,
        "hold_action_fraction": 0.5 + 0.05 * seed_variation,
        "rebalance_action_fraction": 0.5 - 0.05 * seed_variation,
        "mean_rebalances_after_deployment_per_path": 2.0 + seed_variation,
        "mean_selected_range_width_ticks": 4.0,
        "mean_active_range_width_ticks": 4.0,
        "mean_gas_spend_per_path": 3.0,
        "mean_fee_income_token1_per_path": 8.0,
        "mean_inventory_penalty_per_path": 2.0,
        "mean_in_range_fraction_among_deployed_paths": 0.75,
    })
    diagnostics["mean_pnl_per_path"] = (
        objective + diagnostics["mean_inventory_penalty_per_path"]
    )
    return diagnostics


def _make_rows(
    training_seed: int,
    evaluation_seed: int,
    seed_index: int,
    n_evaluation_paths: int,
) -> list[dict]:
    rows = []
    for evaluation_set, params in [
        ("in_distribution", (0.055, 125.0)),
        ("stress", (0.10, 300.0)),
    ]:
        values = _policy_values(seed_index, evaluation_set)
        gaps = {
            "domain_randomized_vs_nominal_mean_running_inventory_objective_gap": (
                values["domain_randomized_ppo"] - values["nominal_ppo"]
            ),
            (
                "domain_randomized_vs_periodic_rebalance_"
                "mean_running_inventory_objective_gap"
            ): (
                values["domain_randomized_ppo"] - values["periodic_rebalance"]
            ),
            "domain_randomized_vs_cash_mean_running_inventory_objective_gap": (
                values["domain_randomized_ppo"] - values["cash"]
            ),
        }
        for policy in POLICIES:
            behavior = _behavior_values(policy, values[policy], seed_index)
            rows.append({
                "evaluation_set": evaluation_set,
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "policy": policy,
                "sigma": params[0],
                "arrival_rate": params[1],
                "mean_running_inventory_objective": values[policy],
                "evaluation_path_std_running_inventory_objective": (
                    0.0
                    if policy == "cash"
                    else 10.0 + seed_index
                ),
                "n_evaluation_paths": n_evaluation_paths,
                **behavior,
                **gaps,
            })
    return rows


def _write_seed_run(
    root: Path,
    directory_name: str,
    training_seed: int,
    seed_index: int,
    *,
    evaluation_seed: int = 100042,
    n_evaluation_paths: int = 6,
    drop_last_row: bool = False,
    legacy_schema: bool = False,
    missing_behavior_schema: bool = False,
) -> Path:
    run_dir = root / directory_name / "run_20260802_000000"
    run_dir.mkdir(parents=True)
    config = {
        "seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "output_dir": str(root / directory_name),
        "n_eval_episodes": 2,
        "num_trajectories": 3,
        "evaluation_design": "test_fixture",
    }
    (run_dir / "config.json").write_text(json.dumps(config))

    rows = _make_rows(
        training_seed,
        evaluation_seed,
        seed_index,
        n_evaluation_paths,
    )
    if drop_last_row:
        rows.pop()
    if legacy_schema:
        for row in rows:
            row["std_running_inventory_objective"] = row.pop(
                "evaluation_path_std_running_inventory_objective"
            )
            row["n_samples"] = row.pop("n_evaluation_paths")
    if missing_behavior_schema:
        for row in rows:
            row.pop("never_deployed_fraction")

    with (run_dir / "evaluation_grid.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_aggregator_parser_defaults():
    args = parse_args(["--input-dir", "/tmp/results"])

    assert args.output_dir is None
    assert args.bootstrap_resamples == 10_000
    assert args.bootstrap_seed == 42
    assert args.confidence_level == 0.95


def test_seed_sweep_aggregation_reports_training_and_path_variation(tmp_path):
    for seed_index, training_seed in enumerate([43, 44, 45]):
        _write_seed_run(
            tmp_path,
            f"seed_{training_seed}",
            training_seed,
            seed_index,
        )

    first_output = tmp_path / "aggregate_a"
    args = parse_args([
        "--input-dir",
        str(tmp_path),
        "--output-dir",
        str(first_output),
        "--bootstrap-resamples",
        "2000",
        "--bootstrap-seed",
        "9",
    ])

    assert aggregate_seed_sweep(args) == first_output.resolve()

    regime_rows = _read_csv(first_output / "training_seed_regime_summary.csv")
    in_distribution_dr = next(
        row
        for row in regime_rows
        if row["evaluation_set"] == "in_distribution"
        and row["policy"] == "domain_randomized_ppo"
    )
    assert int(in_distribution_dr["n_training_seeds"]) == 3
    assert int(in_distribution_dr["n_evaluation_paths_per_training_seed"]) == 6
    assert float(
        in_distribution_dr[
            "mean_running_inventory_objective_across_training_seeds"
        ]
    ) == 5.0
    assert float(
        in_distribution_dr["training_seed_std_running_inventory_objective"]
    ) == 2.0
    assert float(
        in_distribution_dr[
            "training_seed_standard_error_running_inventory_objective"
        ]
    ) == pytest.approx(2.0 / np.sqrt(3.0))
    assert float(
        in_distribution_dr[
            "mean_evaluation_path_std_running_inventory_objective"
        ]
    ) == 11.0

    behavior_rows = _read_csv(
        first_output / "training_seed_behavior_summary.csv"
    )
    never_deployed = next(
        row
        for row in behavior_rows
        if row["evaluation_set"] == "in_distribution"
        and row["policy"] == "domain_randomized_ppo"
        and row["diagnostic"] == "never_deployed_fraction"
    )
    assert float(never_deployed["mean_across_training_seeds"]) == pytest.approx(
        0.1
    )
    assert float(never_deployed["training_seed_sample_std"]) == pytest.approx(
        0.1
    )
    assert len(behavior_rows) == 2 * len(POLICIES) * len(
        BEHAVIOR_DIAGNOSTIC_COLUMNS
    )

    for baseline_policy in ["cash", "periodic_rebalance"]:
        baseline_row = next(
            row
            for row in regime_rows
            if row["evaluation_set"] == "in_distribution"
            and row["policy"] == baseline_policy
        )
        assert float(
            baseline_row["training_seed_std_running_inventory_objective"]
        ) == 0.0
        interval_lower = float(
            baseline_row[
                "training_seed_confidence_interval_lower_running_inventory_objective"
            ]
        )
        interval_upper = float(
            baseline_row[
                "training_seed_confidence_interval_upper_running_inventory_objective"
            ]
        )
        assert interval_lower == interval_upper

    gap_rows = _read_csv(first_output / "training_seed_gap_summary.csv")
    nominal_gap = next(
        row
        for row in gap_rows
        if row["evaluation_set"] == "in_distribution"
        and row["comparison"] == "domain_randomized_vs_nominal"
    )
    assert float(nominal_gap["mean_gap_across_training_seeds"]) == 3.0
    assert float(nominal_gap["training_seed_std_gap"]) == 1.0

    summary = json.loads(
        (first_output / "training_seed_summary.json").read_text()
    )
    assert summary["summary_scope"] == "training_seed_aggregate"
    assert summary["training_seeds"] == [43, 44, 45]
    assert summary["evaluation_seed"] == 100042
    assert summary["bootstrap"] == {
        "method": "percentile",
        "resamples": 2000,
        "seed": 9,
        "confidence_level": 0.95,
    }
    set_stats = summary["policies"]["domain_randomized_ppo"][
        "in_distribution"
    ]["mean_of_regime_mean_running_inventory_objective"]
    assert set_stats["mean_across_training_seeds"] == 5.0
    assert set_stats["training_seed_sample_std"] == 2.0
    behavior_stats = summary["policies"]["domain_randomized_ppo"][
        "in_distribution"
    ]["behavior_diagnostics_mean_across_regimes"][
        "never_deployed_fraction"
    ]
    assert behavior_stats["mean_across_training_seeds"] == pytest.approx(0.1)
    assert behavior_stats["training_seed_sample_std"] == pytest.approx(0.1)

    second_output = tmp_path / "aggregate_b"
    second_args = parse_args([
        "--input-dir",
        str(tmp_path),
        "--output-dir",
        str(second_output),
        "--bootstrap-resamples",
        "2000",
        "--bootstrap-seed",
        "9",
    ])
    aggregate_seed_sweep(second_args)
    assert (
        first_output / "training_seed_regime_summary.csv"
    ).read_text() == (
        second_output / "training_seed_regime_summary.csv"
    ).read_text()
    assert (
        first_output / "training_seed_behavior_summary.csv"
    ).read_text() == (
        second_output / "training_seed_behavior_summary.csv"
    ).read_text()


def test_discovery_rejects_duplicate_training_seeds(tmp_path):
    _write_seed_run(tmp_path, "first", 43, 0)
    _write_seed_run(tmp_path, "second", 43, 1)

    with pytest.raises(ValueError, match="duplicate training seed"):
        discover_seed_runs(tmp_path)


def test_discovery_rejects_inconsistent_evaluation_seed(tmp_path):
    _write_seed_run(tmp_path, "seed_43", 43, 0)
    _write_seed_run(
        tmp_path,
        "seed_44",
        44,
        1,
        evaluation_seed=100043,
    )

    with pytest.raises(ValueError, match="incompatible run configuration"):
        discover_seed_runs(tmp_path)


@pytest.mark.parametrize(
    ("keyword_args", "error"),
    [
        ({"drop_last_row": True}, "expected"),
        ({"n_evaluation_paths": 7}, "path counts"),
        ({"legacy_schema": True}, "legacy result schema"),
        ({"missing_behavior_schema": True}, "missing columns"),
    ],
)
def test_discovery_rejects_incomplete_or_incompatible_results(
    tmp_path,
    keyword_args,
    error,
):
    _write_seed_run(tmp_path, "seed_43", 43, 0)
    _write_seed_run(tmp_path, "seed_44", 44, 1, **keyword_args)

    with pytest.raises(ValueError, match=error):
        discover_seed_runs(tmp_path)


def test_discovery_requires_at_least_two_training_seeds(tmp_path):
    _write_seed_run(tmp_path, "seed_43", 43, 0)

    with pytest.raises(ValueError, match="at least two"):
        discover_seed_runs(tmp_path)
