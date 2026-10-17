"""Short real PPO experiments, checkpoints, reproducibility and learning records."""

import argparse
from copy import deepcopy
import csv
import json
import random

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from experiments import train_robust_lp_agent as training
from experiments import evaluate_robust_lp_agents as evaluation
from experiments.learning_diagnostics import economic_action_probabilities


def flags(encoding):
    return [
        '--training-mode', 'nominal', '--evaluation-scope', 'nominal',
        '--policy-action-space', encoding, '--tau', '500',
        '--center-magnitudes', '50', '100', '150', '200',
        '--half-widths', '50', '100', '150', '200', '250', '300',
        '--n-steps', '6', '--num-trajectories', '3', '--decision-stride', '2',
        '--total-timesteps', '36', '--n-eval-episodes', '2',
        '--periodic-rebalance-every', '2', '--evaluation-seed', '200042',
        '--convergence-evaluation-seed', '100042',
        '--convergence-eval-every-rollouts', '1',
    ]


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize('encoding', ['discrete', 'multidiscrete'])
def test_real_training_artifacts_replay_and_rng_isolation(tmp_path, monkeypatch, encoding):
    args = training.parse_args([
        *flags(encoding), '--output-dir', str(tmp_path), '--record-learning-diagnostics',
        '--evaluation-action-modes', 'deterministic', 'stochastic',
    ])
    monkeypatch.setattr(training, 'parse_args', lambda: args)
    assert training.main() == 0
    run_dir, = tmp_path.iterdir()
    config = evaluation.load_config(run_dir)
    args = argparse.Namespace(**config)
    assert config['ppo_total_timesteps'] == 18
    assert config['economic_action_count'] == 55
    model = PPO.load(run_dir / 'nominal_ppo.zip', device='cpu')
    training.validate_policy_action_encoding(model, args)
    params = training.DomainParameters(args.nominal_sigma, args.nominal_arrival_rate)
    adapter = training.wrap_env(training.make_fixed_env(args, params, 7), normalise_obs=False,
                                action_grid=training.configured_action_grid(args),
                                action_encoding=training.adapter_action_encoding(args))
    normalizer = VecNormalize.load(run_dir / 'nominal_vecnormalize.pkl', adapter)
    before_stats = deepcopy(normalizer.obs_rms.__dict__)
    for mode in args.evaluation_action_modes:
        suffix = training.evaluation_suffix(mode)
        rows = csv_rows(run_dir / f'nominal_convergence{suffix}.csv')
        assert [row['policy_phase'] for row in rows] == ['initial', 'before_update', 'before_update', 'after_update']
        assert [int(row['optimizer_updates']) for row in rows] == [0, 0, 10, 20]
        assert [int(row['num_decision_timesteps']) for row in rows] == [0, 9, 18, 18]
        assert all(int(row['evaluation_seed']) == 100042 for row in rows)
        assert all(int(row['n_evaluation_paths']) == 3 for row in rows)
        for row in rows:
            assert (run_dir / row['policy_checkpoint']).is_file()
            assert (run_dir / row['normalization_checkpoint']).is_file()
        assert {r['policy'] for r in csv_rows(run_dir / f'validation_baselines{suffix}.csv')} == {'cash', 'periodic_rebalance'}
        paths = csv_rows(run_dir / f'evaluation_paths{suffix}.csv')
        assert len(paths) == 18
        assert {int(row['episode_seed']) for row in paths} == {200042, 200043}
        for row in paths:
            assert float(row['objective']) == pytest.approx(float(row['pnl']) - float(row['inventory_penalty']), abs=1e-10)
            assert 0 <= float(row['time_in_range_fraction']) <= 1
            assert float(row['maximum_drawdown']) >= 0
        summaries = csv_rows(run_dir / f'evaluation_grid{suffix}.csv')
        for summary in summaries:
            selected = [row for row in paths if row['policy'] == summary['policy']]
            for summary_key, path_key in (
                ('mean_running_inventory_objective', 'objective'), ('mean_pnl_per_path', 'pnl'),
                ('mean_inventory_penalty_per_path', 'inventory_penalty'),
                ('mean_gas_spend_per_path', 'gas_spend'), ('mean_fee_income_token1_per_path', 'fee_income_token1'),
                ('mean_rebalances_after_deployment_per_path', 'rebalances_after_deployment'),
            ):
                assert float(summary[summary_key]) == pytest.approx(np.mean([float(row[path_key]) for row in selected]), abs=1e-10)
        replay_paths = []
        model.policy.set_training_mode(True)
        random.seed(351)
        np.random.seed(352)
        torch.manual_seed(353)
        expected_draws = (random.random(), np.random.random(), torch.rand(3))
        random.seed(351)
        np.random.seed(352)
        torch.manual_seed(353)
        result = training.evaluate_ppo('nominal_ppo', model, normalizer, params, 'in_distribution', args,
                                       args.evaluation_seed, action_mode=mode, path_records=replay_paths)
        assert model.policy.training
        assert random.random() == expected_draws[0]
        assert np.random.random() == expected_draws[1]
        assert torch.equal(torch.rand(3), expected_draws[2])
        second_paths = []
        assert result == training.evaluate_ppo('nominal_ppo', model, normalizer, params, 'in_distribution', args,
                                               args.evaluation_seed, action_mode=mode, path_records=second_paths)
        assert replay_paths == second_paths
        summary = next(row for row in summaries if row['policy'] == 'nominal_ppo')
        assert result['mean_running_inventory_objective'] == pytest.approx(float(summary['mean_running_inventory_objective']))
        # Reload a before-update checkpoint together with its own normalization.
        saved = rows[2]
        checkpoint = PPO.load(run_dir / saved['policy_checkpoint'], device='cpu')
        checkpoint_norm = VecNormalize.load(run_dir / saved['normalization_checkpoint'], adapter)
        eval_args = argparse.Namespace(**config)
        eval_args.n_eval_episodes = 1
        eval_args.evaluation_seed = 100042
        replay = training.evaluate_ppo('nominal_ppo', checkpoint, checkpoint_norm, params, 'convergence_validation',
                                       eval_args, 100042, action_mode=mode)
        assert replay['mean_running_inventory_objective'] == pytest.approx(float(saved['mean_running_inventory_objective']))
    for key, value in before_stats.items():
        np.testing.assert_array_equal(normalizer.obs_rms.__dict__[key], value)
    records = [json.loads(line) for line in (run_dir / 'nominal_learning_diagnostics.jsonl').read_text().splitlines()]
    rollouts = [r for r in records if r['record_type'] == 'rollout']
    optimizers = [r for r in records if r['record_type'] == 'optimizer']
    assert len(rollouts) == len(optimizers) == 2
    for row in rollouts:
        assert row['sampled_agent_decisions'] == row['forced_simulator_holds'] == 9
        assert sum(row['economic_action_counts']) == 9
        assert sum(row['encoded_action_counts']) == 9
        assert sum(row['economic_action_frequencies']) == pytest.approx(1)
        assert row['sampled_hold_fraction'] + row['sampled_rebalance_fraction'] == pytest.approx(1)
        assert 0 <= row['predicted_hold_probability'] <= 1
        assert 0 <= row['economic_action_entropy'] <= np.log(55)
        assert row['economic_action_entropy'] <= row['raw_action_entropy'] + 1e-5
    for row in optimizers:
        assert all(key in row and np.isfinite(row[key]) for key in (
            'train/entropy_loss', 'train/value_loss', 'train/policy_gradient_loss',
            'train/approx_kl', 'train/clip_fraction', 'train/explained_variance',
        ))
    normalizer.close()


def test_native_multidiscrete_greedy_behavior_is_preserved():
    args = training.parse_args(flags('multidiscrete'))
    training.validate_and_derive_decision_timing(args)
    params = training.DomainParameters(.03, 300.)
    model, vec, _ = training.build_ppo(training.make_fixed_env(args, params, 7), args)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(torch.tensor([0.] * 15 + [np.log(.6), np.log(.4)]))
    obs = vec.reset()
    actions = model.predict(obs, deterministic=True)[0]
    assert (actions[:, 2] == 0).all()  # independent greedy hold head selects rebalance
    with torch.no_grad():
        probabilities = economic_action_probabilities(model.policy.get_distribution(torch.as_tensor(obs)), model.action_space)
    assert (probabilities.argmax(dim=1) == 0).all()  # joint economic argmax would instead Hold
    vec.close()


def test_validation_seed_default_and_invalid_evaluation_modes():
    args = training.parse_args(['--evaluation-seed', '765'])
    assert args.convergence_evaluation_seed == 765
    assert args.evaluation_action_modes == ['deterministic']
    for flags_, error in [
        (['--convergence-evaluation-seed', '-1'], 'convergence_evaluation_seed'),
        (['--evaluation-action-modes', 'stochastic', 'stochastic'], 'evaluation_action_modes'),
    ]:
        with pytest.raises(ValueError, match=error):
            training.validate_evaluation_configuration(training.parse_args(flags_))


@pytest.mark.parametrize('encoding', ['discrete', 'multidiscrete'])
def test_evaluation_and_diagnostics_leave_training_unchanged(tmp_path, encoding):
    base_args = training.parse_args(flags(encoding))
    training.validate_and_derive_decision_timing(base_args)
    params = training.DomainParameters(.03, 300.)
    snapshots = []
    for record in (False, True):
        args = argparse.Namespace(**vars(base_args))
        args.record_learning_diagnostics = record
        args.evaluation_action_modes = ['deterministic', 'stochastic']
        args.convergence_eval_every_rollouts = 1 if record else 0
        run = tmp_path / str(record)
        run.mkdir()
        model, normalizer = training.train_and_save(
            'nominal', training.make_fixed_env(args, params, 7), args, run, environment_seed=20042,
        )
        snapshots.append((deepcopy(model.policy.state_dict()), deepcopy(normalizer.obs_rms.__dict__)))
        model.get_env().close()
    for first, second in zip(snapshots[0], snapshots[1]):
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


def test_evaluation_restores_rng_and_policy_mode_on_failure():
    args = training.parse_args(flags('multidiscrete'))
    params = training.DomainParameters(.03, 300.)
    model, vec, _ = training.build_ppo(training.make_fixed_env(args, params, 7), args)
    model.policy.set_training_mode(True)
    torch.manual_seed(35)
    torch_state = torch.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()

    def failing_env(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        model.policy.set_training_mode(False)
        raise RuntimeError('failed evaluation')

    with pytest.raises(RuntimeError, match='failed evaluation'):
        training.evaluate_ppo('nominal_ppo', model, None, params, 'in_distribution', args, 200042,
                              action_mode='stochastic', env_factory=failing_env)
    assert model.policy.training
    assert random.getstate() == python_state
    for actual, expected in zip(np.random.get_state(), numpy_state):
        np.testing.assert_array_equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), torch_state)
    vec.close()
