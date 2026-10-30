"""Validate reward replay, environment reuse, and uncertainty for width sweeps."""

import numpy as np
import pytest

from experiments import train_robust_lp_agent as trainer
from experiments.sweep_dense_exponential_width import (
    parse_args,
    replay_dense_rewards,
    run_episode,
    summarize,
)
from SAiFE_gym.rewards.RewardFunctions import DenseExponentialUtility, RunningInventoryPenalty


def utility(pnl, risk):
    return np.asarray(pnl) if risk == 0.0 else -np.expm1(-risk * np.asarray(pnl)) / risk


def test_replay_preserves_dense_objective_for_multiple_risks_and_paths():
    wealth = np.array([[100, 200], [110, 180], [95, 220], [103, 198]], dtype=float)
    risks = [-0.01, 0.0, 0.001, 0.01, 0.1]
    total, error = replay_dense_rewards(wealth, wealth[0], risks)
    expected = np.stack([utility([3, -2], risk) for risk in risks])
    np.testing.assert_allclose(total, expected, rtol=1e-12, atol=1e-14)
    assert np.max(error) < 1e-12


@pytest.mark.parametrize("risk", [-1e-320, 1e-320])
def test_replay_retains_pnl_limit_at_subnormal_risk(risk):
    wealth = np.array([[1000], [1001.2345], [1002.1234]])
    total, error = replay_dense_rewards(wealth, wealth[0], [risk])
    np.testing.assert_allclose(total[0], wealth[-1] - wealth[0], rtol=1e-12)
    assert error[0] < 1e-12


@pytest.mark.parametrize("risk", [-1e300, 1e300])
def test_replay_accepts_finite_reward_when_unscaled_exponential_overflows(risk):
    wealth = -np.sign(risk) * np.array([[0], [5e-298], [1e-297]])
    total, error = replay_dense_rewards(wealth, wealth[0], [risk])
    assert np.isfinite(total).all()
    assert np.sign(total[0, 0]) == -np.sign(risk)
    np.testing.assert_allclose(np.log(np.abs(total)), 1000 - np.log(abs(risk)), atol=1e-10, rtol=0)
    assert error[0] < 1e-12


def test_factory_reward_override_preserves_existing_default():
    args = trainer.parse_args(["--num-trajectories", "1"])
    params = trainer.DomainParameters(args.nominal_sigma, args.nominal_arrival_rate)
    default = trainer.make_fixed_env(args, params, 42)
    reward = DenseExponentialUtility(0.01)
    custom = trainer.make_fixed_env(args, params, 42, reward_function=reward)
    assert isinstance(default.reward_function, RunningInventoryPenalty)
    assert custom.reward_function is reward
    before, _ = default.reset(seed=42)
    after, _ = custom.reset(seed=42)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    default.close()
    custom.close()


@pytest.mark.parametrize("num_trajectories", [1, 3])
def test_real_environment_replay_and_rebalance_accounting(num_trajectories):
    args = trainer.parse_args([
        "--num-trajectories", str(num_trajectories), "--n-steps", "20",
        "--terminal-time", "0.02", "--periodic-rebalance-every", "5",
    ])
    risks = [-0.01, 0.0, 0.01]
    result = run_episode(args, 10, 0, risks)
    assert result["total_rewards"].shape == (3, num_trajectories)
    np.testing.assert_allclose(
        result["total_rewards"], np.stack([utility(result["pnl"], risk) for risk in risks]),
        rtol=1e-10, atol=1e-12,
    )
    assert result["diagnostics"]["mean_gas_spend_per_path"] == 6
    assert result["diagnostics"]["mean_selected_range_width_ticks"] == 20
    assert result["diagnostics"]["mean_rebalances_after_deployment_per_path"] == 3
    repeat = run_episode(args, 10, 0, risks)
    np.testing.assert_array_equal(result["pnl"], repeat["pnl"])


def test_optimum_uses_expected_utility_not_utility_of_mean_pnl():
    # Narrower is a fairer-return/risk tradeoff; width 10 is dominated by width 5.
    pnl = np.array([[[20, -10], [20, -10]], [[2, 2], [2, 2]], [[1, 1], [1, 1]]])
    risks = [0.01, 0.2]
    rewards = np.stack([utility(pnl, risk) for risk in risks], axis=1)
    rows, optima, summary = summarize(pnl, rewards, [1, 5, 10], risks, 100)
    assert summary["optimal_half_widths"] == [1, 5]
    assert summary["optimal_half_widths_nondecreasing"]
    assert optima[1]["certainty_equivalent_pnl"] == pytest.approx(2)
    assert rows[0]["mean_total_reward"] == pytest.approx(np.mean(utility([20, -10], 0.01)))
    assert not summary["curve_shape"][0]["sample_curve_is_concave"]
    assert summary["curve_shape"][1]["sample_curve_is_concave"]


def test_bootstrap_preserves_within_episode_dependence():
    pnl = np.array([[[1], [5], [-3]], [[2], [4], [-1]]], dtype=float)
    rewards = utility(pnl[:, None], 0.1)
    original, _, _ = summarize(pnl, rewards, [1, 5], [0.1], 100)
    repeated, _, _ = summarize(np.repeat(pnl, 20, axis=-1), np.repeat(rewards, 20, axis=-1), [1, 5], [0.1], 100)
    for a, b in zip(original, repeated):
        for key in ["mean_total_reward", "reward_ci_low", "reward_ci_high", "ce_ci_low", "ce_ci_high"]:
            assert a[key] == pytest.approx(b[key])


def test_certainty_equivalent_and_optimum_support_all_risk_profiles():
    # A fair gamble versus cash: seeking prefers the gamble, averse prefers cash.
    pnl = np.array([[[-10, 10], [-10, 10]], [[0, 0], [0, 0]]], dtype=float)
    risks = [-0.1, 0.0, 0.1]
    rewards = np.stack([utility(pnl, risk) for risk in risks], axis=1)
    rows, _, summary = summarize(pnl, rewards, [1, 5], risks, 100)
    assert summary["optimal_half_widths"] == [1, 1, 5]
    for row in rows:
        wi = [1, 5].index(row["half_width_ticks"])
        risk = row["risk_aversion"]
        expected_ce = pnl[wi].mean() if risk == 0.0 else -np.log(np.exp(-risk * pnl[wi]).mean()) / risk
        assert row["certainty_equivalent_pnl"] == pytest.approx(expected_ce)
        assert row["ce_ci_low"] <= row["certainty_equivalent_pnl"] <= row["ce_ci_high"]
        if risk == 0.0:
            assert row["utility_weight_ess"] == 4
            assert row["largest_utility_weight"] == 0.25
            assert row["ce_ci_low"] == row["reward_ci_low"]
            assert row["ce_ci_high"] == row["reward_ci_high"]


@pytest.mark.parametrize("risk", [-1e-12, 1e-12])
def test_certainty_equivalent_approaches_mean_pnl_near_zero_risk(risk):
    pnl = np.array([[[20, -10], [20, -10]], [[2, 2], [2, 2]]], dtype=float)
    rows, _, _ = summarize(pnl, utility(pnl[:, None], risk), [1, 5], [risk], 100)
    np.testing.assert_allclose([row["certainty_equivalent_pnl"] for row in rows], [5, 2], atol=2e-10, rtol=0)


@pytest.mark.parametrize("risk", [-1e-320, 1e-320])
def test_certainty_equivalent_retains_mean_pnl_limit_at_subnormal_risk(risk):
    pnl = np.array([[[20.1234, -10.2345], [1e-6, 2e-6]], [[1.2345, 2.3456], [3.4567, 4.5678]]])
    # At this risk magnitude, the utility correction is below float64 precision.
    rows, _, _ = summarize(pnl, pnl[:, None], [1, 5], [risk], 100)
    for row, paths in zip(rows, pnl):
        assert row["certainty_equivalent_pnl"] == pytest.approx(paths.mean(), abs=1e-12)
        assert row["ce_ci_low"] == pytest.approx(row["reward_ci_low"], abs=1e-12)
        assert row["ce_ci_high"] == pytest.approx(row["reward_ci_high"], abs=1e-12)


@pytest.mark.parametrize("risk", [-0.5, 0.5])
def test_ce_bootstrap_stays_finite_when_extreme_episode_is_not_resampled(risk):
    pnl = np.array([[[-1000], [1000]], [[-900], [900]]], dtype=float)
    rewards = utility(pnl[:, None], risk)
    assert np.isfinite(rewards).all()
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        rows, _, _ = summarize(pnl, rewards, [1, 5], [risk], 100)
    for row, bound in zip(rows, [1000, 900]):
        expected_ce = -np.log(np.cosh(risk * bound)) / risk
        assert row["certainty_equivalent_pnl"] == pytest.approx(expected_ce)
        # The bootstrap includes draws containing only either episode, whose
        # deterministic certainty equivalents equal that episode's PnL.
        assert row["ce_ci_low"] == pytest.approx(-bound)
        assert row["ce_ci_high"] == pytest.approx(bound)


def test_sweep_accepts_seeking_neutral_and_averse_parameters():
    sweep, _ = parse_args(["--risk-aversion-values", "-0.01", "0", "0.01"])
    assert sweep.risk_aversion_values == [-0.01, 0.0, 0.01]


def test_sweep_uses_trainer_defaults_and_accepts_environment_overrides():
    sweep, environment = parse_args(["--nominal-sigma", "0.04", "--half-widths", "1", "5", "50"])
    defaults = trainer.parse_args([])
    assert environment.nominal_sigma == 0.04
    for key in ["n_steps", "num_trajectories", "initial_wealth", "nominal_arrival_rate", "nominal_gas_cost", "periodic_rebalance_every", "trade_size_notional", "alpha3"]:
        assert getattr(environment, key) == getattr(defaults, key)
    assert sweep.half_widths == [1, 5, 50]


def test_invalid_width_cannot_be_silently_clipped():
    with pytest.raises(SystemExit):
        parse_args(["--half-widths", "1", "51"])
