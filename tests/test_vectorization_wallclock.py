"""Check elapsed-time accounting, snapshot integrity and independent evaluation."""

import csv
import json
import pickle
import shutil
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from experiments import vectorization_wallclock as wallclock


@pytest.fixture(scope="module", autouse=True)
def torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def options(root, count=2, *extra):
    return wallclock.parse_args([
        "train", "--output-dir", str(root), "--num-trajectories", str(count), *extra,
    ])


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_defaults_preserve_nominal_configuration(tmp_path):
    opts = options(tmp_path)
    args = wallclock.training_configuration(opts)
    assert (opts.training_seconds, opts.checkpoint_seconds) == (3600, 300)
    assert (opts.evaluation_episodes, opts.evaluation_seed) == (1000, 100042)
    nominal = wallclock.training.parse_args([])
    for key, value in vars(nominal).items():
        if key not in {"num_trajectories", "total_timesteps", "convergence_eval_every_rollouts"}:
            assert getattr(args, key) == value
    assert args.ppo_total_timesteps == 20
    assert args.ppo_n_steps == 10
    assert wallclock.parse_args(["plot", "--input-dir", str(tmp_path)]).trajectory_counts == (10, 100, 1000)


@pytest.mark.parametrize("flag,value", [
    ("--training-seconds", "0"), ("--training-seconds", "nan"),
    ("--training-seconds", "inf"), ("--checkpoint-seconds", "-1"),
    ("--evaluation-episodes", "0"), ("--evaluation-seed", "-1"),
])
def test_invalid_options(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        options(tmp_path, 2, flag, value)


@pytest.mark.parametrize("durations,expected_rollouts,expected_times", [
    ([0.6, 0.5, 0.2, 0.8], [0, 2, 3], [0.0, 1.1, 1.3]),
    ([1.0, 1.0], [0, 1, 2], [0.0, 1.0, 2.0]),
    ([3.0], [0], [0.0]),
])
def test_deadline_keeps_last_eligible_policy_and_its_normalization(
    tmp_path, durations, expected_rollouts, expected_times,
):
    clock = Clock()
    normalize = SimpleNamespace(revision=0)
    args = wallclock.training_configuration(options(tmp_path))

    class Model:
        num_timesteps = 0
        revision = 0

        def save(self, stream):
            stream.write(json.dumps({"revision": self.revision}).encode())

        def learn(self, total_timesteps, reset_num_timesteps, callback):
            assert reset_num_timesteps is False
            clock.now += durations[self.revision]
            self.revision += 1
            self.num_timesteps += total_timesteps
            normalize.revision = self.revision
            callback.statistics = {"episode_count": 2, "mean_episode_reward": 1.0, "std_episode_reward": 2.0}

    model = Model()
    done = wallclock.train_cycles(model, normalize, args, tmp_path, 2.0, 1.0, clock=clock)
    manifest = json.loads((tmp_path / "checkpoints.json").read_text())
    assert [item["rollouts_completed"] for item in manifest] == expected_rollouts
    assert [item["elapsed_seconds"] for item in manifest] == pytest.approx(expected_times)
    assert done["rollouts_completed"] == len(durations)
    assert done["eligible_rollouts_completed"] == expected_rollouts[-1]
    assert done["overrun_seconds"] == pytest.approx(sum(durations) - 2)
    for item in manifest:
        expected = item["rollouts_completed"]
        assert json.loads((tmp_path / item["model_file"]).read_bytes())["revision"] == expected
        assert pickle.loads((tmp_path / item["normalization_file"]).read_bytes()).revision == expected


def test_checkpoint_overhead_counts_toward_training_budget(tmp_path):
    clock = Clock()
    args = wallclock.training_configuration(options(tmp_path))

    class Model:
        num_timesteps = 0

        def save(self, stream):
            clock.now += 0.8
            stream.write(b"snapshot")

        def learn(self, total_timesteps, reset_num_timesteps, callback):
            clock.now += 0.4
            self.num_timesteps += total_timesteps
            callback.statistics = {"episode_count": 2, "mean_episode_reward": 0.0, "std_episode_reward": 1.0}

    done = wallclock.train_cycles(Model(), None, args, tmp_path, 2.0, 1.0, clock=clock)
    assert done["rollouts_completed"] == 2
    assert done["elapsed_seconds"] == pytest.approx(2.4)
    assert done["eligible_rollouts_completed"] == 2
    manifest = json.loads((tmp_path / "checkpoints.json").read_text())
    assert manifest[-1]["elapsed_seconds"] == pytest.approx(1.6)


def test_training_reward_uses_every_episode(tmp_path):
    args = wallclock.training_configuration(options(tmp_path, 1000))
    callback = wallclock.EpisodeRewards(args)
    callback.locals = {
        "dones": np.ones(1000, dtype=bool),
        "infos": [{"episode": {"r": float(i), "l": 1000}} for i in range(1000)],
    }
    callback._on_rollout_start()
    assert callback._on_step()
    callback._on_rollout_end()
    assert callback.statistics["mean_episode_reward"] == 499.5
    assert callback.statistics["episode_count"] == 1000


@pytest.fixture(scope="module")
def smoke_sweep(tmp_path_factory):
    root = tmp_path_factory.mktemp("wallclock-smoke")
    original_learn = PPO.learn
    original_evaluate = wallclock.training.evaluate_ppo
    observed_evaluation_counts = []

    def checked_evaluation(*args, **kwargs):
        normalize = args[2]
        eval_args = args[5]
        assert (eval_args.num_trajectories, eval_args.n_eval_episodes) == (4, 1)
        assert normalize.training is False and normalize.norm_reward is False
        before = pickle.dumps((normalize.obs_rms, normalize.ret_rms))
        result = original_evaluate(*args, **kwargs)
        assert pickle.dumps((normalize.obs_rms, normalize.ret_rms)) == before
        assert normalize.training is False
        observed_evaluation_counts.append(result["n_evaluation_paths"])
        return result

    for count in (1, 2, 3):
        clock = Clock()

        def timed_learn(model, *args, **kwargs):
            result = original_learn(model, *args, **kwargs)
            clock.now += 0.6
            return result

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(PPO, "learn", timed_learn)
            patch.setattr(wallclock.training, "evaluate_ppo", checked_evaluation)
            run = wallclock.train_setting(options(root, count, "--smoke-test"), clock=clock)
            wallclock.evaluate_setting(SimpleNamespace(run_dir=run))
    assert observed_evaluation_counts == [4] * 9
    return root


def test_real_ppo_snapshots_evaluation_and_retrieval_bundle(smoke_sweep):
    for count in (1, 2, 3):
        run = smoke_sweep / f"trajectories_{count}"
        config, manifest = wallclock.load_training(run)
        _, rows = wallclock.load_evaluation(run)
        assert [item["rollouts_completed"] for item in manifest] == [0, 2, 3]
        assert [row["elapsed_seconds"] for row in rows] == pytest.approx([0, 1.2, 1.8])
        assert config["ppo"]["device"] == "cpu"
        for item in manifest:
            model = PPO.load(run / item["model_file"], device="cpu")
            assert model._n_updates == 10 * item["rollouts_completed"]
            assert model.num_timesteps == count * 2 * item["rollouts_completed"]
            if item["rollouts_completed"]:
                optimizer_steps = [state["step"].item() for state in model.policy.optimizer.state.values()]
                assert set(optimizer_steps) == {10 * item["rollouts_completed"]}
    output = wallclock.plot_sweep(wallclock.parse_args([
        "plot", "--input-dir", str(smoke_sweep), "--trajectory-counts", "1", "2", "3",
    ]))
    assert (output / "vectorization_wallclock.png").read_bytes().startswith(b"\x89PNG")
    assert (output / "vectorization_wallclock.pdf").read_bytes().startswith(b"%PDF")
    with tarfile.open(output / "vectorization_wallclock_review.tar.gz") as bundle:
        names = bundle.getnames()
    assert "combined_evaluation_rewards.csv" in names
    assert "trajectories_3/evaluation_completion.json" in names
    assert not any(name.endswith((".zip", ".pkl")) for name in names)


def test_consecutive_learn_calls_match_uninterrupted_nominal_training(smoke_sweep, tmp_path):
    """Catch accidental environment/RNG/normalization resets between cycles."""
    args = wallclock.training_configuration(options(tmp_path, 2, "--smoke-test"))
    env = wallclock.training.make_fixed_env(
        args, wallclock.training.DomainParameters(args.nominal_sigma, args.nominal_arrival_rate),
        seed=args.seed + 20_000,
    )
    model, ppo_env, normalize = wallclock.training.build_ppo(env, args, device="cpu")
    try:
        ppo_env.seed(args.seed + 20_000)
        model.learn(total_timesteps=3 * args.ppo_total_timesteps)
        run = smoke_sweep / "trajectories_2"
        _, manifest = wallclock.load_training(run)
        checkpoint = manifest[-1]
        saved = PPO.load(run / checkpoint["model_file"], device="cpu")
        for key, tensor in model.policy.state_dict().items():
            torch.testing.assert_close(tensor, saved.policy.state_dict()[key], rtol=0, atol=0)
        saved_normalize = pickle.loads((run / checkpoint["normalization_file"]).read_bytes())
        np.testing.assert_array_equal(normalize.obs_rms.mean, saved_normalize.obs_rms.mean)
        np.testing.assert_array_equal(normalize.obs_rms.var, saved_normalize.obs_rms.var)
        assert normalize.obs_rms.count == saved_normalize.obs_rms.count
    finally:
        ppo_env.close()


def test_evaluation_can_be_repeated_without_retraining(smoke_sweep):
    run = smoke_sweep / "trajectories_1"
    checkpoint_bytes = (run / "checkpoints.json").read_bytes()
    _, before = wallclock.load_evaluation(run)
    wallclock.evaluate_setting(SimpleNamespace(run_dir=run))
    _, after = wallclock.load_evaluation(run)
    assert (run / "checkpoints.json").read_bytes() == checkpoint_bytes
    for first, second in zip(before, after, strict=True):
        assert {k: v for k, v in first.items() if k != "evaluation_seconds"} == {
            k: v for k, v in second.items() if k != "evaluation_seconds"
        }


@pytest.mark.parametrize("corruption", [
    "missing_training_completion", "missing_evaluation_completion", "model_checksum",
    "late_checkpoint", "episode_count", "evaluation_seed", "nan_reward", "mismatched_seed",
    "missing_training_row",
])
def test_plot_rejects_invalid_or_incomparable_runs(smoke_sweep, tmp_path, corruption):
    root = tmp_path / "sweep"
    shutil.copytree(smoke_sweep, root, ignore=shutil.ignore_patterns("plots"))
    run = root / "trajectories_2"
    if corruption == "missing_training_row":
        path = run / "training_rewards.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        wallclock.atomic_csv(path, rows[:-1])
    elif corruption.startswith("missing_"):
        (run / (corruption.removeprefix("missing_") + ".json")).unlink()
    elif corruption == "model_checksum":
        (run / "checkpoints/update_000000/ppo.zip").write_bytes(b"bad model")
    elif corruption == "late_checkpoint":
        path = run / "checkpoints.json"
        items = json.loads(path.read_text())
        items[-1]["elapsed_seconds"] = 3
        wallclock.atomic_json(path, items)
    elif corruption == "mismatched_seed":
        path = run / "config.json"
        config = json.loads(path.read_text())
        config["training"]["seed"] += 1
        wallclock.atomic_json(path, config)
        path = run / "evaluation_completion.json"
        done = json.loads(path.read_text())
        done["config_sha256"] = wallclock.digest((run / "config.json").read_bytes())
        wallclock.atomic_json(path, done)
    else:
        path = run / "evaluation_rewards.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        key = "mean_episode_reward" if corruption == "nan_reward" else corruption
        rows[-1][key] = "nan" if corruption == "nan_reward" else "999"
        wallclock.atomic_csv(path, rows)
    assert wallclock.main([
        "plot", "--input-dir", str(root), "--trajectory-counts", "1", "2", "3",
    ]) == 1
    assert not (root / "plots").exists()


def test_refuses_to_overwrite_training(smoke_sweep):
    with pytest.raises(FileExistsError):
        wallclock.train_setting(options(smoke_sweep, 2, "--smoke-test"))
