"""Check sweep orchestration and accounting without training an RL policy."""

import json
from unittest.mock import Mock

import numpy as np
import pytest
import gymnasium

from experiments import sweep_nominal_lp_volatility as sweep


def options(tmp_path, *extra):
    return sweep.parse_args(["train", "--output-dir", str(tmp_path / "runs"), *extra])


def metric_row(sigma, seed, evaluation_seed, width=8.0, pnl=10.0):
    return {
        "sigma": sigma, "training_seed": seed, "evaluation_seed": evaluation_seed,
        "mean_active_range_width_ticks": width or 0.0,
        "mean_deployed_width_ticks": width,
        "equivalent_half_width_ticks": width / 2 if width is not None else None,
        "mean_pnl_per_path": pnl, "mean_inventory_penalty_per_path": 0.0,
        "never_deployed_fraction": 1.0 if width is None else 0.0,
        "agent_hold_decision_fraction": .5,
        "mean_agent_rebalances_after_deployment_per_path": 3.0,
    }


def save_cell(cli, sigma, seed, width=8.0, pnl=10.0):
    config = sweep.run_config(cli, sigma, seed)
    directory = sweep.cell_directory(cli.output_dir, sigma, seed)
    directory.mkdir(parents=True)
    sweep.write_json(directory / "config.json", config)
    sweep.write_json(directory / "result.json", {
        "config": config, "metrics": metric_row(sigma, seed, cli.evaluation_seed, width, pnl),
    })
    return directory


def test_default_job_mapping_and_nominal_pnl_settings(tmp_path):
    cli = options(tmp_path)
    jobs = sweep.job_grid(cli)
    assert len(jobs) == 15
    assert jobs[:4] == [(.03, 43), (.03, 44), (.03, 45), (.035, 43)]
    assert jobs[-1] == (.05, 45)
    args = sweep.training_args(cli, .045, 44)
    assert (args.nominal_sigma, args.seed, args.inventory_phi) == (.045, 44, 0.0)
    assert args.evaluation_seed == cli.validation_seed != cli.evaluation_seed
    assert (args.n_steps, args.decision_stride, args.ppo_n_steps) == (10_000, 100, 100)
    assert args.ppo_total_timesteps == 1_000_000
    assert (args.background_liquidity, args.trade_size_notional, args.price_impact_depth_window) == (30_000, 250, 15)
    assert args.half_widths == [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    assert args.action_mode == "symmetric_width"


@pytest.mark.parametrize("extra", [
    ["--sigma-values", "nan"], ["--sigma-values", "0.03", "0.03"],
    ["--seeds", "43", "43"], ["--task-index", "15"], ["--task-index", "-1"],
    ["--background-liquidity", "-1"], ["--n-steps", "999"],
    ["--evaluation-seed", "100043"], ["--validation-seed", "43"],
    ["--num-trajectories", "0"], ["--initial-wealth", "inf"],
    ["--half-widths", "2", "2"], ["--half-widths", "5", "2"],
    ["--half-widths", "0", "5"], ["--half-widths", "2", "51"],
    ["--tau", "20"],
])
def test_invalid_plan_rejected_without_output(tmp_path, extra):
    cli = options(tmp_path, *extra)
    with pytest.raises(ValueError):
        sweep.job_grid(cli)
    assert not cli.output_dir.exists()


def test_dry_run_does_not_train_or_write(tmp_path, monkeypatch, capsys):
    run = Mock(side_effect=AssertionError("training was requested"))
    monkeypatch.setattr(sweep, "run_cell", run)
    sweep.main(["train", "--dry-run", "--output-dir", str(tmp_path / "runs")])
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["jobs"]) == 15
    assert plan["rollouts_per_policy"] == 100
    assert plan["n_actions"] == 11
    assert plan["half_widths"] == [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    run.assert_not_called()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("width", [12.0, None])
def test_one_cell_trains_only_nominal_and_evaluates_on_separate_paths(tmp_path, monkeypatch, width):
    cli = options(tmp_path, "--sigma-values", ".04", "--seeds", "44")
    model, normalization, env = Mock(), Mock(), Mock()
    factory = Mock(return_value=env)
    train = Mock(return_value=(model, normalization))
    evaluate = Mock(return_value=metric_row(.04, 44, cli.evaluation_seed, width))
    monkeypatch.setattr(sweep.training, "make_fixed_env", factory)
    monkeypatch.setattr(sweep.training, "make_domain_randomized_env", Mock(side_effect=AssertionError("randomized training")))
    monkeypatch.setattr(sweep.training, "train_and_save", train)
    monkeypatch.setattr(sweep.training, "evaluate_ppo", evaluate)
    result = sweep.run_cell(cli, .04, 44)
    train_args = train.call_args.args
    assert train_args[0] == "nominal" and train_args[1] is env
    assert train_args[2].inventory_phi == 0
    assert train_args[2].action_mode == "symmetric_width"
    assert train_args[2].half_widths == cli.half_widths
    assert train_args[2].nominal_sigma == .04
    assert train_args[2].evaluation_seed == cli.validation_seed
    eval_args = evaluate.call_args.args
    assert eval_args[1:3] == (model, normalization)
    assert eval_args[3].sigma == .04
    assert eval_args[5].evaluation_seed == cli.evaluation_seed
    assert eval_args[5].half_widths == cli.half_widths
    assert evaluate.call_args.kwargs["regime_seed"] == cli.evaluation_seed
    assert result["metrics"]["mean_deployed_width_ticks"] == width
    assert result["metrics"]["equivalent_half_width_ticks"] == (width / 2 if width else None)
    reference, saved = sweep.load_results(cli.output_dir)
    assert saved[(.04, 44)] == result
    assert reference["inventory_phi"] == 0
    model.get_env().close.assert_called_once()
    with pytest.raises(FileExistsError):
        sweep.run_cell(cli, .04, 44)
    assert train.call_count == 1


def test_training_failure_is_not_marked_complete(tmp_path, monkeypatch):
    cli = options(tmp_path, "--sigma-values", ".04", "--seeds", "44")
    env = Mock()
    monkeypatch.setattr(sweep.training, "make_fixed_env", Mock(return_value=env))
    monkeypatch.setattr(sweep.training, "train_and_save", Mock(side_effect=RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        sweep.run_cell(cli, .04, 44)
    env.close.assert_called_once()
    with pytest.raises(ValueError, match="Unfinished"):
        sweep.load_results(cli.output_dir)


def test_actual_width_and_pnl_accounting_with_scripted_policy(tmp_path):
    # This only steps a tiny environment; no model construction or learning.
    cli = options(tmp_path, "--n-steps", "4", "--decision-stride", "2",
                  "--num-trajectories", "2", "--n-eval-episodes", "1")
    args = sweep.training_args(cli, .04, 44)
    args.evaluation_seed = cli.evaluation_seed
    policy = Mock()
    # Coarse-grid action index three selects half-width fifteen, full width thirty.
    policy.predict.return_value = (np.full(2, 3), None)
    row = sweep.training.evaluate_ppo(
        "scripted", policy, None, sweep.training.DomainParameters(.04, 300),
        "in_distribution", args, regime_seed=cli.evaluation_seed,
    )
    assert row["mean_active_range_width_ticks"] == 30
    assert row["mean_agent_selected_range_width_ticks"] == 30
    assert row["mean_inventory_penalty_per_path"] == pytest.approx(0, abs=1e-7)
    assert row["mean_running_inventory_objective"] == pytest.approx(row["mean_pnl_per_path"])
    assert policy.predict.call_count == 2
    assert row["agent_hold_decision_fraction"] == 0
    assert row["mean_agent_rebalances_after_deployment_per_path"] == 1
    assert row["mean_gas_spend_per_path"] == 2


@pytest.mark.parametrize("mode", ["structured", "symmetric_width"])
def test_ppo_builder_and_evaluator_share_action_mapping_without_learning(tmp_path, monkeypatch, mode):
    cli = options(tmp_path, "--n-steps", "4", "--decision-stride", "2", "--num-trajectories", "2")
    args = sweep.training_args(cli, .04, 44)
    args.action_mode = mode
    if mode == "structured":
        args.half_widths = None
        args.tick_stride = 1
    raw = sweep.training.make_fixed_env(args, sweep.training.DomainParameters(.04, 300), 44)
    constructor = Mock()
    monkeypatch.setattr(sweep.training, "PPO", constructor)
    model, wrapped, _ = sweep.training.build_ppo(raw, args)
    assert constructor.call_args.args[1] is wrapped
    assert constructor.call_args.kwargs["gamma"] == 1
    evaluation_wrapper = sweep.training.make_policy_action_wrapper(wrapped, args)
    if mode == "symmetric_width":
        assert isinstance(wrapped.action_space, gymnasium.spaces.Discrete)
        assert wrapped.action_space.n == 11
        actions = np.array([0, 10])
        np.testing.assert_array_equal(wrapped.unscale(actions), [[-2, 2, -1], [-50, 50, -1]])
    else:
        assert isinstance(wrapped.action_space, gymnasium.spaces.MultiDiscrete)
        actions = np.array([[50, 0, 0], [50, 49, 1]])
    np.testing.assert_array_equal(wrapped.unscale(actions), evaluation_wrapper.unscale(actions))
    model.learn.assert_not_called()
    wrapped.close()


def test_summary_averages_training_seeds_and_marks_missing_width(tmp_path):
    cli = options(tmp_path, "--sigma-values", ".03", ".04")
    for sigma in cli.sigma_values:
        for seed, width, pnl in zip(cli.seeds, [2.0, 4.0, None], [10.0, 20.0, 30.0]):
            save_cell(cli, sigma, seed, width, pnl)
    rows = sweep.summarize_results(cli.output_dir)
    for row in rows:
        assert row["mean_deployed_width_ticks"] == 3
        assert row["equivalent_half_width_ticks"] == 1.5
        assert row["seed_std_width_ticks"] == pytest.approx(np.sqrt(2))
        assert row["n_width_seeds"] == 2
        assert row["n_training_seeds"] == 3
        assert row["mean_pnl"] == 20
        assert row["mean_never_deployed_fraction"] == pytest.approx(1 / 3)
    for name in ("width_by_volatility.png", "width_by_volatility.pdf", "width_by_volatility.csv", "per_seed.csv", "summary.json"):
        assert (cli.output_dir / "summary" / name).stat().st_size > 0
    assert sweep.mean_and_std([None, None]) == (None, None, 0)


@pytest.mark.parametrize("corruption", ["missing", "unfinished", "settings", "source", "widths", "metadata", "nan"])
def test_aggregation_rejects_incomplete_or_incompatible_runs(tmp_path, corruption):
    cli = options(tmp_path, "--sigma-values", ".04", "--seeds", "43", "44")
    save_cell(cli, .04, 43)
    if corruption != "missing":
        directory = save_cell(cli, .04, 44)
        if corruption == "unfinished":
            (directory / "result.json").unlink()
        elif corruption in ("settings", "source", "widths"):
            config = json.loads((directory / "config.json").read_text())
            if corruption == "source":
                config["code_sha256"] = "different"
            elif corruption == "widths":
                config["half_widths"] = [5, 10, 20, 50]
            else:
                config["settings"]["background_liquidity"] = 100_000
            sweep.write_json(directory / "config.json", config)
        else:
            result = json.loads((directory / "result.json").read_text())
            if corruption == "metadata":
                result["metrics"]["evaluation_seed"] += 1
            else:
                result["metrics"]["mean_pnl_per_path"] = float("nan")
            (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        sweep.summarize_results(cli.output_dir)
    assert not (cli.output_dir / "summary").exists()
