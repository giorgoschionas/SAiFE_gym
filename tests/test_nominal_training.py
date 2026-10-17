"""Nominal-only runs train, save, reload and evaluate both requested action grids."""

import argparse
import csv
import json
from unittest.mock import Mock

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from experiments import evaluate_robust_lp_agents as evaluation
from experiments import train_robust_lp_agent as training
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment


@pytest.mark.parametrize("grid_args,action_count", [
    (["--tau", "50", "--tick-stride", "5"], 211),
    (["--tau", "500", "--center-magnitudes", "50", "100", "150", "200",
      "--half-widths", "50", "100", "150", "200", "250", "300"], 55),
])
def test_nominal_only_training_and_reload(tmp_path, monkeypatch, grid_args, action_count):
    args = training.parse_args([
        "--training-mode", "nominal", "--evaluation-scope", "nominal",
        "--output-dir", str(tmp_path), "--smoke-test", "--decision-stride", "2",
        "--convergence-eval-every-rollouts", "1", *grid_args,
    ])
    monkeypatch.setattr(training, "parse_args", lambda: args)
    randomized = Mock(side_effect=AssertionError("must not construct a randomized agent"))
    monkeypatch.setattr(training, "make_domain_randomized_env", randomized)
    train = Mock(wraps=training.train_and_save)
    monkeypatch.setattr(training, "train_and_save", train)

    assert training.main() == 0
    randomized.assert_not_called()
    assert train.call_count == 1
    assert train.call_args.args[0] == "nominal"
    assert train.call_args.kwargs["environment_seed"] == 20042
    run_dir, = tmp_path.iterdir()
    assert not list(run_dir.glob("domain_randomized*"))
    for filename in (
        "nominal_ppo.zip", "nominal_vecnormalize.pkl", "nominal_convergence.csv",
        "config.json", "evaluation_grid.csv", "summary.json",
    ):
        assert (run_dir / filename).is_file()

    config = evaluation.load_config(run_dir)
    assert config["training_mode"] == config["evaluation_scope"] == "nominal"
    assert config["seed"] == 42
    assert config["evaluation_seed"] == 100042
    assert config["action_count"] == action_count
    assert config["lp_action_config"]["action_count"] == action_count
    saved_args = argparse.Namespace(**config)
    codec = training.categorical_action_codec(
        saved_args.tau, saved_args.tick_stride,
        action_grid=training.configured_action_grid(saved_args),
    )
    assert len(np.unique(codec.action_table, axis=0)) == action_count
    assert evaluation.regime_seed(config, .03, 300., 100042) == 100042

    with (run_dir / "evaluation_grid.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["policy"] for row in rows} == {"nominal_ppo", "periodic_rebalance", "cash"}
    for row in rows:
        assert row["evaluation_set"] == "in_distribution"
        assert float(row["sigma"]) == .03
        assert float(row["arrival_rate"]) == 300.
        for metric in ("mean_running_inventory_objective", *training.BEHAVIOR_DIAGNOSTIC_COLUMNS):
            assert np.isfinite(float(row[metric]))
    policy_row = next(row for row in rows if row["policy"] == "nominal_ppo")
    assert sum(float(policy_row[key]) for key in (
        "agent_hold_decision_fraction", "agent_rebalance_decision_fraction",
    )) == pytest.approx(1.)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert set(summary["policies"]) == {row["policy"] for row in rows}
    assert all(set(value) == {"in_distribution"} for value in summary["policies"].values())

    with (run_dir / "nominal_convergence.csv").open() as handle:
        convergence = list(csv.DictReader(handle))
    assert [row["evaluation_set"] for row in convergence] == [
        "convergence_validation", "convergence_validation_final",
    ]
    assert int(convergence[-1]["num_timesteps"]) == config["total_timesteps"]

    model = PPO.load(run_dir / "nominal_ppo.zip")
    assert model.action_space.n == action_count
    assert model.num_timesteps == config["ppo_total_timesteps"]
    assert model.lp_action_config == config["lp_action_config"]
    training.validate_policy_action_encoding(model, saved_args)
    params = training.DomainParameters(saved_args.nominal_sigma, saved_args.nominal_arrival_rate)
    base = training.make_fixed_env(saved_args, params, saved_args.evaluation_seed)
    adapter = StableBaselinesAMMEnvironment(
        base, tick_stride=saved_args.tick_stride,
        action_grid=training.configured_action_grid(saved_args),
    )
    normalizer = VecNormalize.load(run_dir / "nominal_vecnormalize.pkl", adapter)
    normalizer.training = False
    assert np.isfinite(normalizer.reset()).all()
    result = training.evaluate_ppo(
        "nominal_ppo", model, normalizer, params, "in_distribution",
        saved_args, saved_args.evaluation_seed,
    )
    for metric in ("mean_running_inventory_objective", *training.BEHAVIOR_DIAGNOSTIC_COLUMNS):
        assert result[metric] == pytest.approx(float(policy_row[metric]))
    normalizer.close()


def test_default_modes_and_nominal_scope_are_independent():
    defaults = training.parse_args([])
    assert defaults.training_mode == "both"
    assert defaults.evaluation_scope == "grid"
    assert len(training.evaluation_regimes(defaults)) == 25
    args = training.parse_args([
        "--evaluation-scope", "nominal", "--nominal-sigma", ".06",
        "--nominal-arrival-rate", "500",
    ])
    assert args.training_mode == "both"
    training.validate_evaluation_configuration(args)
    regime, = training.evaluation_regimes(args)
    assert regime.parameters == training.DomainParameters(.06, 500.)
