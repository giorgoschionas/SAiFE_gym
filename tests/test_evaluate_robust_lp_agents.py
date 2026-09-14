import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from experiments import evaluate_robust_lp_agents as evaluation
from experiments import train_robust_lp_agent as training
from experiments.fast_lp_evaluation import make_evaluation_env


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    args = training.parse_args([])
    args.seed = 43
    (path / "config.json").write_text(json.dumps(vars(args)))
    return path


def options(run_dir, tmp_path, *extra):
    return evaluation.parse_args([
        "--run-dirs", str(run_dir), "--output-dir", str(tmp_path / "output"), *extra,
    ])


def test_grid_preserves_old_seeds_and_is_stable_across_subsets(run_dir, tmp_path):
    cli = options(run_dir, tmp_path)
    config = evaluation.load_config(run_dir)
    _, jobs = evaluation.build_jobs(cli)
    assert len(jobs) == 20
    assert len({job["regime_seed"] for job in jobs}) == 20
    assert evaluation.regime_seed(config, .03, 300, 100042) == 104042
    assert evaluation.regime_seed(config, .065, 450, 100042) == 110042
    original = next(j for j in jobs if j["sigma"] == .08 and j["arrival"] == 1000)
    _, subset = evaluation.build_jobs(options(
        run_dir, tmp_path, "--sigma-values", ".08", "--arrival-rate-values", "1000",
        "--n-steps-values", "5000", "10000",
    ))
    assert all(j["regime_seed"] == original["regime_seed"] for j in subset)
    assert [j["args"]["decision_stride"] for j in subset] == [500, 1000]
    assert [j["args"]["periodic_rebalance_every"] for j in subset] == [500, 1000]
    assert all(j["args"]["max_agent_decisions_per_episode"] == 10 for j in subset)


@pytest.mark.parametrize("extra", [
    ["--sigma-values", "nan"], ["--arrival-rate-values", "-1"],
    ["--sigma-values", ".03", ".03"], ["--n-steps-values", "1001"],
    ["--n-steps-values", "1000", "1000"], ["--workers", "0"],
    ["--n-eval-episodes", "1"], ["--num-trajectories", "0"],
    ["--episode-offset", "995"], ["--evaluation-seed", "-1"],
])
def test_invalid_settings_rejected_before_creating_output(run_dir, tmp_path, extra):
    with pytest.raises(ValueError):
        evaluation.build_jobs(options(run_dir, tmp_path, *extra))
    assert not (tmp_path / "output").exists()


def test_periodic_is_not_duplicated_across_training_seeds(run_dir, tmp_path):
    other = tmp_path / "source_44"
    other.mkdir()
    config = json.loads((run_dir / "config.json").read_text())
    config["seed"] = 44
    (other / "config.json").write_text(json.dumps(config))
    cli = options(run_dir, tmp_path)
    cli.run_dirs.append(other)
    assert len(evaluation.build_jobs(cli)[1]) == 20


def test_saved_normalization_is_required_for_ppo(run_dir, tmp_path):
    (run_dir / "nominal_ppo.zip").touch()
    with pytest.raises(FileNotFoundError, match="nominal_vecnormalize"):
        evaluation.build_jobs(options(run_dir, tmp_path, "--policies", "nominal_ppo"))


def test_uncertainty_resamples_episodes_and_pools_within_episode_variance():
    rows = []
    for value in [1., 3., 5., 7.]:
        row = dict.fromkeys(evaluation.METRICS, value)
        row.update(dict.fromkeys([
            "policy", "source_training_seed", "source_run_dir", "evaluation_set",
            "sigma", "arrival_rate", "n_steps", "decision_stride", "periodic_rebalance_every",
            "regime_seed", "episode_offset", "num_trajectories",
        ], 1))
        row.update(n_evaluation_paths=100, evaluation_path_std_running_inventory_objective=2.)
        rows.append(row)
    summary = evaluation.summarize_episodes(rows, 20000, 42)
    assert summary[evaluation.REWARD] == 4.
    assert summary["n_independent_episodes"] == 4
    assert summary["n_evaluation_paths"] == 400
    assert summary["evaluation_path_std_running_inventory_objective"] == 3.
    assert summary[evaluation.REWARD + "_ci_lower"] < 4 < summary[evaluation.REWARD + "_ci_upper"]


def test_real_evaluation_only_smoke_preserves_source_and_observes_arrivals(run_dir, tmp_path, monkeypatch):
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    config.update(n_steps=10, decision_stride=2, periodic_rebalance_every=2,
                  num_trajectories=2, n_eval_episodes=2)
    config_path.write_text(json.dumps(config))
    original = config_path.read_bytes()
    monkeypatch.setattr(training, "train_and_save", Mock(side_effect=AssertionError("must not train")))
    assert evaluation.main([
        "--run-dirs", str(run_dir), "--output-dir", str(tmp_path / "output"),
        "--sigma-values", ".08", "--arrival-rate-values", "1000",
        "--bootstrap-resamples", "100", "--no-plots",
    ]) == 0
    assert config_path.read_bytes() == original
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_jobs"] == 1
    import csv
    with (tmp_path / "output/evaluation_summary.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["mean_rebalances_after_deployment_per_path"]) == 4.
    assert float(row["mean_gas_spend_per_path"]) == 8.
    assert float(row["mean_intensity_times_dt"]) >= 100.
    assert float(row["mean_sell_arrival_indicators_per_path"]) == 10.
    assert float(row[evaluation.REWARD]) == pytest.approx(
        float(row["mean_pnl_per_path"]) - float(row["mean_inventory_penalty_per_path"]), abs=1e-7,
    )
    with pytest.raises(FileExistsError):
        evaluation.main([
            "--run-dirs", str(run_dir), "--output-dir", str(tmp_path / "output"),
            "--sigma-values", ".08", "--arrival-rate-values", "1000", "--no-plots",
        ])


def test_observer_does_not_change_periodic_results():
    args = training.parse_args([])
    args.n_steps, args.num_trajectories, args.n_eval_episodes = 5, 2, 1
    args.periodic_rebalance_every = 2
    params = training.DomainParameters(.065, 1000.)
    reference = training.evaluate_periodic_rebalance(params, "stress", args, 42)
    observer = evaluation.ArrivalDiagnostics()
    actual = training.evaluate_periodic_rebalance(params, "stress", args, 42, step_observer=observer)
    assert reference == actual
    assert observer.side_steps == 5 * 2 * 2


@pytest.mark.parametrize("sigma,arrival", [(.03, 300.), (.065, 1000.), (.08, 1000.)])
def test_fast_evaluator_matches_original_state_rewards_and_hold_decisions(sigma, arrival):
    args = training.parse_args([])
    args.n_steps, args.num_trajectories = 60, 3
    params = training.DomainParameters(sigma, arrival)
    reference = training.make_fixed_env(args, params, 104042)
    fast = make_evaluation_env(args, params, 104042)
    reference.reset(seed=104042)
    fast.reset(seed=104042)
    total_reference = np.zeros(3)
    total_fast = np.zeros(3)
    rng = np.random.default_rng(51)
    for step in range(args.n_steps):
        widths = rng.integers(1, 51, size=3)
        action = np.column_stack([-widths, widths, np.ones(3)])
        if step % 6 == 0:
            action[:, 2] = [-1., 1., -1.] if step % 12 else -1.
        state, reward, terminated, truncated, _ = reference.step(action)
        fast_state, fast_reward, fast_terminated, fast_truncated, _ = fast.step(action)
        np.testing.assert_allclose(fast_reward, reward, rtol=1e-9, atol=1e-8)
        np.testing.assert_array_equal(fast_terminated, terminated)
        np.testing.assert_array_equal(fast_truncated, truncated)
        for key in state:
            np.testing.assert_allclose(fast_state[key], state[key], rtol=1e-9, atol=1e-8, err_msg=key)
        total_reference += reward
        total_fast += fast_reward
    np.testing.assert_allclose(total_fast, total_reference, rtol=1e-9, atol=1e-8)


def test_fast_fee_accounting_handles_clipped_empty_and_zero_liquidity_ranges():
    from SAiFE_gym.gym.index_names import (
        FEES0_KEY, FEES1_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
        LP_LIQUIDITY_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    )
    args = training.parse_args([])
    args.num_trajectories = 4
    env = make_evaluation_env(args, training.DomainParameters(.03, 300.), 42)
    env.reset(seed=42)
    dynamics = env.model_dynamics
    state = dynamics.state
    anchor = dynamics.tick_lower_global
    state[LP_TICK_LOWER_KEY] = np.array([anchor - 5, anchor + 6998, anchor, anchor + 5])
    state[LP_TICK_UPPER_KEY] = np.array([anchor + 10, anchor + 7010, anchor, anchor + 9])
    state[LP_LIQUIDITY_KEY] = np.ones(4)
    rng = np.random.default_rng(42)
    for key in (FEES0_KEY, FEES1_KEY):
        state[key] = rng.random((4, 7000))
    state[POOL_LIQUIDITY_ARRAY_KEY][3] = 0.
    expected = super(type(dynamics), dynamics)._claimable_lp_fees()
    actual = dynamics._claimable_lp_fees()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_resume_recovers_only_missing_episodes_and_rejects_changed_grid(run_dir, tmp_path, monkeypatch):
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    config.update(n_steps=10, decision_stride=2, periodic_rebalance_every=2,
                  num_trajectories=2, n_eval_episodes=2)
    config_path.write_text(json.dumps(config))
    argv = [
        "--run-dirs", str(run_dir), "--output-dir", str(tmp_path / "output"),
        "--sigma-values", ".08", "--arrival-rate-values", "1000",
        "--bootstrap-resamples", "100", "--no-plots",
    ]
    original = training.evaluate_periodic_rebalance
    calls = []

    def interrupt_second(*args, **kwargs):
        calls.append(args[3])
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "evaluate_periodic_rebalance", interrupt_second)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluation.main(argv)
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert len(evaluation.read_episode_checkpoint(manifest["jobs"][0])) == 1

    resumed_calls = []
    def observe_resume(*args, **kwargs):
        resumed_calls.append(args[3])
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "evaluate_periodic_rebalance", observe_resume)
    assert evaluation.main(argv + ["--resume"]) == 0
    assert resumed_calls == [calls[1]]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["resume_history"]) == 1
    resumed = evaluation.read_episode_checkpoint(manifest["jobs"][0])
    assert len(resumed) == 2
    assert evaluation.main(argv + ["--resume"]) == 0
    assert resumed_calls == [calls[1]]  # Completed jobs never simulate again.
    with pytest.raises(ValueError, match="same policies, grid"):
        evaluation.main(argv + ["--arrival-rate-values", "800", "--resume"])


def test_checkpoint_rejects_duplicate_episode_seeds(run_dir, tmp_path):
    cli = options(run_dir, tmp_path, "--sigma-values", ".08", "--arrival-rate-values", "1000")
    job = evaluation.build_jobs(cli)[1][0]
    path = Path(job["episode_path"])
    path.parent.mkdir(parents=True)
    evaluation.write_csv(path, [{"policy": "periodic_rebalance", "episode_index": 1}])
    with pytest.raises(ValueError, match="checkpoint parameters or seed sequence"):
        evaluation.read_episode_checkpoint(job)
