"""Check the controlled sweep and width accounting without expensive learning."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from experiments import sweep_rl_background_liquidity as sweep


def test_defaults_preserve_training_configuration_and_pair_seeds():
    cli, base = sweep.parse_args([])
    defaults = sweep.training.parse_args([])
    assert cli.background_values == [10_000, 15_000, 20_000, 25_000, 30_000]
    assert cli.seeds == [defaults.seed]
    for key, value in vars(defaults).items():
        if key != "output_dir":
            assert getattr(base, key) == value
    for background in cli.background_values:
        args = sweep.cell_args(base, background, cli.seeds[0])
        assert args.background_liquidity == background
        assert {k: v for k, v in vars(args).items() if k != "background_liquidity"} == {
            k: v for k, v in vars(base).items() if k != "background_liquidity"
        }


@pytest.mark.parametrize("flags", [
    ["--background-values", "nan", "30000"],
    ["--background-values", "10000", "10000"],
    ["--background-values", "-1", "30000"],
    ["--background-liquidity", "20000"],
    ["--seeds", "42", "42"],
    ["--seeds", "-1"],
    ["--seed", "100042", "--evaluation-seed", "100042"],
    ["--num-trajectories", "0"],
])
def test_invalid_sweeps_rejected(flags):
    with pytest.raises(SystemExit):
        sweep.parse_args(flags)


def test_dry_run_writes_nothing(tmp_path, capsys):
    output = tmp_path / "runs"
    sweep.main(["--dry-run", "--output-dir", str(output), "--seeds", "43", "44"])
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["jobs"]) == 10
    assert not output.exists()


@pytest.mark.parametrize("deploy", [False, True])
def test_real_evaluation_counts_width_and_excludes_undeployed_steps(tmp_path, monkeypatch, deploy):
    _, base = sweep.parse_args([
        "--n-steps", "4", "--decision-stride", "2", "--num-trajectories", "2",
        "--n-eval-episodes", "1", "--action-mode", "symmetric_width",
    ])
    args = sweep.cell_args(base, 10_000., 42)
    model = Mock()
    # In symmetric mode, index 2 selects half-width 15 => full width 30.
    model.predict.return_value = (np.full(2, 2), None)
    train = Mock(return_value=(model, None))
    monkeypatch.setattr(sweep.training, "train_and_save", train)
    if not deploy:
        # Force holds through decoding to exercise the real evaluator's
        # no-position accounting without changing its accumulation logic.
        wrapper = Mock()
        wrapper.unscale.return_value = np.tile([0., 1., 1.], (2, 1))
        monkeypatch.setattr(sweep.training, "make_policy_action_wrapper", Mock(return_value=wrapper))
    row = sweep.run_cell(args, tmp_path / "cell")
    assert train.call_args.args[2] is args
    assert train.call_args.args[1].background_liquidity == 10_000
    assert row["deployed_width_ticks"] == (30 if deploy else None)
    assert row["selected_width_ticks"] == (30 if deploy else None)
    assert row["never_deployed_fraction"] == (0 if deploy else 1)
    assert model.predict.call_count == 2
    model.get_env().close.assert_called_once()
    assert json.loads((tmp_path / "cell" / "metrics.json").read_text()) == row
    summary = sweep.summarize([row])[0]
    assert summary["mean_deployed_width_ticks"] == row["deployed_width_ticks"]
    assert summary["seed_std_deployed_width_ticks"] is None


def test_summary_excludes_missing_width_and_uses_seed_standard_deviation():
    rows = [{
        "background_liquidity": 10_000, "deployed_width_ticks": width,
        "selected_width_ticks": width, "never_deployed_fraction": float(width is None),
        "agent_hold_decision_fraction": .5,
        "mean_agent_rebalances_after_deployment_per_path": 2.,
        "mean_pnl_per_path": 10., "mean_running_inventory_objective": 8.,
    } for width in (10., 20., None)]
    summary = sweep.summarize(rows)[0]
    assert summary["mean_deployed_width_ticks"] == 15
    assert summary["seed_std_deployed_width_ticks"] == pytest.approx(np.sqrt(50))
    assert summary["n_seeds_with_deployed_width_ticks"] == 2
    assert summary["never_deployed_fraction"] == pytest.approx(1 / 3)
