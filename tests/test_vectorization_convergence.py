"""Check experiment accounting, real PPO artifacts, and sweep integrity."""

import csv
import json
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from experiments import vectorization_convergence as sweep


@pytest.fixture(scope="module", autouse=True)
def small_torch_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def options(output_dir, count, *extra):
    return sweep.parse_args([
        "train", "--output-dir", str(output_dir),
        "--num-trajectories", str(count), *extra,
    ])


@pytest.mark.parametrize("count,simulator_budget,decision_budget", [
    (10, 1_000_000, 10_000),
    (100, 10_000_000, 100_000),
    (1000, 100_000_000, 1_000_000),
    (10000, 1_000_000_000, 10_000_000),
])
def test_budget_and_nominal_defaults(tmp_path, count, simulator_budget, decision_budget):
    args = sweep.training_configuration(options(tmp_path, count))
    assert args.total_timesteps == simulator_budget
    assert args.ppo_total_timesteps == decision_budget
    assert args.ppo_n_steps == 10
    assert args.rollout_updates == 100
    assert args.convergence_eval_every_rollouts == 0
    nominal = sweep.training.parse_args([])
    changed = {"num_trajectories", "total_timesteps", "output_dir", "convergence_eval_every_rollouts"}
    for key, value in vars(nominal).items():
        if key not in changed:
            assert getattr(args, key) == value


@pytest.mark.parametrize("flag,value", [
    ("--num-trajectories", "0"), ("--num-trajectories", "-1"),
    ("--rollout-updates", "0"), ("--seed", "-1"),
])
def test_invalid_training_options(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        options(tmp_path, 10, flag, value)


def callback_fixture(tmp_path, count):
    args = sweep.training_configuration(options(tmp_path, count))
    callback = sweep.RolloutRewardCallback(args, tmp_path / "rewards.csv")
    model = SimpleNamespace(num_timesteps=0, logger=Mock(), _n_updates=1000)
    callback.init_callback(model)
    callback.on_training_start({}, {})
    return callback, model


def deliver_episodes(callback, model, returns, lengths=None, done=True):
    if lengths is None:
        lengths = np.full(len(returns), callback.args.n_steps)
    callback.update_locals({
        "dones": np.full(len(returns), done),
        "infos": [
            {"episode": {"r": reward, "l": length}}
            for reward, length in zip(returns, lengths, strict=True)
        ],
    })
    model.num_timesteps += callback.args.ppo_n_steps * callback.args.num_trajectories
    assert callback.on_step()


def test_callback_averages_every_episode_and_resets_each_rollout(tmp_path):
    callback, model = callback_fixture(tmp_path, 10000)
    for returns in (np.arange(10000, dtype=float), np.full(10000, -17.0)):
        callback.on_rollout_start()
        # Non-terminal metadata must never contribute to episode statistics.
        callback.update_locals({
            "dones": np.zeros(10000, dtype=bool),
            "infos": [{"episode": {"r": 1e10, "l": 1000}}] * 10000,
        })
        callback.on_step()
        deliver_episodes(callback, model, returns)
        callback.on_rollout_end()
    with callback.output_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["rollout_update"]) for row in rows] == [1, 2]
    assert float(rows[0]["mean_episode_reward"]) == pytest.approx(4999.5)
    assert float(rows[0]["std_episode_reward"]) == pytest.approx(np.arange(10000).std())
    assert float(rows[1]["mean_episode_reward"]) == -17
    assert float(rows[1]["std_episode_reward"]) == 0
    assert [int(row["episode_count"]) for row in rows] == [10000, 10000]
    assert int(rows[1]["num_timesteps"]) == 20_000_000
    assert int(rows[1]["num_decision_timesteps"]) == 200_000
    assert callback.rollouts_completed == 2


@pytest.mark.parametrize("returns,lengths,error", [
    ([1], [1000], "one completed episode"),
    ([1, 2], [999, 1000], "Episode lengths"),
    ([1, np.nan], [1000, 1000], "Non-finite"),
])
def test_callback_rejects_incomplete_or_invalid_episodes(tmp_path, returns, lengths, error):
    callback, model = callback_fixture(tmp_path, 2)
    callback.on_rollout_start()
    deliver_episodes(callback, model, returns, lengths)
    with pytest.raises(ValueError, match=error):
        callback.on_rollout_end()


@pytest.fixture(scope="module")
def smoke_sweep(tmp_path_factory):
    directory = tmp_path_factory.mktemp("vectorization-smoke")
    original_build = sweep.training.build_ppo

    def inspect_seed(env, args, **kwargs):
        model, ppo_env, normalize = original_build(env, args, **kwargs)
        original_learn = model.learn

        def learn(*learn_args, **learn_kwargs):
            assert model.seed == 42
            assert env.model_dynamics.midprice_model.seed_ == 20042
            assert env.model_dynamics.midprice_model.rng.bit_generator.state == np.random.default_rng(20042).bit_generator.state
            del model.learn
            return original_learn(*learn_args, **learn_kwargs)

        model.learn = learn
        return model, ppo_env, normalize

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sweep.training, "build_ppo", inspect_seed)
        for count in (1, 2, 3, 4):
            sweep.train_setting(options(directory, count, "--smoke-test"))
    return directory


def test_real_ppo_smoke_writes_complete_reloadable_runs_and_plot(smoke_sweep):
    for count in (1, 2, 3, 4):
        directory = smoke_sweep / f"trajectories_{count}"
        config, rows = sweep.load_run(directory, count)
        assert config["smoke_test"] is True
        assert config["rollout_updates"] == len(rows) == 2
        assert config["n_steps"] == 4
        assert config["environment_seed"] == 20042
        assert config["ppo"]["device"] == "cpu"
        assert config["ppo"]["policy_kwargs"] == {"net_arch": {"pi": [256, 256], "vf": [256, 256]}}
        assert all(row["episode_count"] == count for row in rows)
        model = PPO.load(directory / "nominal_ppo.zip", device="cpu")
        assert model.num_timesteps == count * 4
        assert model.n_epochs == 10
        assert model._n_updates == 20  # Two rollout updates, ten PPO epochs each.
        assert not (directory / "nominal_convergence.csv").exists()
        assert not (directory / "evaluation_grid.csv").exists()

    assert sweep.main([
        "plot", "--input-dir", str(smoke_sweep),
        "--trajectory-counts", "1", "2", "3", "4",
    ]) == 0
    output = smoke_sweep / "plots"
    assert (output / "vectorization_convergence.png").read_bytes().startswith(b"\x89PNG")
    assert (output / "vectorization_convergence.pdf").read_bytes().startswith(b"%PDF")
    with (output / "combined_training_rewards.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {int(row["num_trajectories"]) for row in rows} == {1, 2, 3, 4}


def test_existing_run_is_preserved(smoke_sweep):
    path = smoke_sweep / "trajectories_1" / "training_rewards.csv"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        sweep.train_setting(options(smoke_sweep, 1, "--smoke-test"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("corruption,error", [
    ("missing_completion", "completion.json"),
    ("unfinished", "Incomplete training"),
    ("missing_model", "Missing training artifact"),
    ("seed", "Mismatched settings"),
    ("nominal_sigma", "Mismatched settings"),
    ("batch_size", "minibatch size"),
    ("missing_row", "rollout rows"),
    ("rollout_update", "Invalid rollout row"),
    ("episode_count", "Invalid rollout row"),
    ("num_timesteps", "Invalid rollout row"),
    ("mean_episode_reward", "Non-finite"),
])
def test_plot_rejects_incomplete_or_mismatched_results(smoke_sweep, tmp_path, corruption, error, capsys):
    directory = tmp_path / "sweep"
    shutil.copytree(smoke_sweep, directory, ignore=shutil.ignore_patterns("plots"))
    run = directory / "trajectories_2"
    if corruption == "missing_completion":
        (run / "completion.json").unlink()
    elif corruption == "missing_model":
        (run / "nominal_ppo.zip").unlink()
    elif corruption == "unfinished":
        payload = json.loads((run / "completion.json").read_text())
        payload["rollouts_completed"] = 1
        (run / "completion.json").write_text(json.dumps(payload))
    elif corruption in {"seed", "nominal_sigma", "batch_size"}:
        payload = json.loads((run / "config.json").read_text())
        if corruption == "batch_size":
            payload["ppo"]["batch_size"] += 1
        else:
            payload[corruption] += 1
        (run / "config.json").write_text(json.dumps(payload))
    else:
        path = run / "training_rewards.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        if corruption == "missing_row":
            rows.pop()
        else:
            rows[-1][corruption] = "nan" if corruption == "mean_episode_reward" else "0"
        sweep.training.save_csv(path, rows)
    assert sweep.main([
        "plot", "--input-dir", str(directory), "--trajectory-counts", "1", "2", "3", "4",
    ]) == 1
    assert error in capsys.readouterr().err
    assert not (directory / "plots").exists()


def test_plot_defaults_require_all_four_production_settings(smoke_sweep, capsys):
    assert sweep.parse_args(["plot", "--input-dir", str(smoke_sweep)]).trajectory_counts == (10, 100, 1000, 10000)
    assert sweep.main(["plot", "--input-dir", str(smoke_sweep)]) == 1
    assert "trajectories_10" in capsys.readouterr().err
