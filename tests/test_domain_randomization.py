from argparse import Namespace

import numpy as np
from stable_baselines3 import PPO

from experiments.helpers import INITIAL_WEALTH
from experiments.train_robust_lp_agent import (
    add_gap_columns,
    make_fixed_env,
    parse_args,
    summarize_running_inventory_objective,
    summarize_rows,
)
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import (
    DEFAULT_OBS_KEYS,
    StableBaselinesAMMEnvironment,
)
from SAiFE_gym.gym.domain_randomization import (
    DomainParameters,
    DomainRandomizedAMMEnvironment,
    UniformDomainRandomizationConfig,
)
from SAiFE_gym.gym.index_names import GAS_COST_KEY
from SAiFE_gym.rewards.RewardFunctions import RunningInventoryPenalty
from SAiFE_gym.stochastic_processes.arrival_models import (
    LiquidityKernelArrivalModel,
    PoissonLinearArrivalModel,
)
from SAiFE_gym.stochastic_processes.midprice_models import (
    BrownianMotionMidpriceModel,
    GeometricBrownianMotionMidpriceModel,
)
from SAiFE_gym.stochastic_processes.price_impact_models import (
    LiquidityDepthUniswapV3PriceImpact,
)
from SAiFE_gym.wrappers import StructuredMultiDiscreteVecEnv


def create_test_amm_env(
    num_trajectories: int = 2,
    n_steps: int = 5,
    seed: int = 42,
) -> AMMEnvironment:
    step_size = 1.0 / n_steps
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=2.0,
        initial_price=100.0,
        terminal_time=1.0,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed,
    )
    arrival_model = PoissonLinearArrivalModel(
        alpha=np.array([
            [1.0, 1.0],
            [100.0, 100.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        tau=5,
        num_ticks=100,
        gas_cost=0.0,
        seed=seed + 2,
    )
    return AMMEnvironment(
        terminal_time=1.0,
        n_steps=n_steps,
        model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        seed=seed,
    )


def test_sampled_values_stay_within_ranges():
    config = UniformDomainRandomizationConfig(
        sigma_range=(1.0, 4.0),
        arrival_rate_range=(50.0, 200.0),
        gas_cost_range=(0.0, 20.0),
    )
    rng = np.random.default_rng(123)

    for _ in range(100):
        params = config.sample(rng)
        assert 1.0 <= params.sigma <= 4.0
        assert 50.0 <= params.arrival_rate <= 200.0
        assert 0.0 <= params.gas_cost <= 20.0


def test_seeded_sampler_is_reproducible():
    config = UniformDomainRandomizationConfig()
    env_a = DomainRandomizedAMMEnvironment(create_test_amm_env(), config, seed=7)
    env_b = DomainRandomizedAMMEnvironment(create_test_amm_env(), config, seed=7)

    _, info_a = env_a.reset()
    _, info_b = env_b.reset()

    assert info_a["domain_parameters"] == info_b["domain_parameters"]


def test_reset_applies_domain_parameters_to_models_and_state():
    config = UniformDomainRandomizationConfig(
        sigma_range=(3.0, 3.0),
        arrival_rate_range=(123.0, 123.0),
        gas_cost_range=(17.0, 17.0),
    )
    env = DomainRandomizedAMMEnvironment(create_test_amm_env(), config, seed=1)

    obs, info = env.reset()

    assert info["domain_parameters"] == {
        "sigma": 3.0,
        "arrival_rate": 123.0,
        "gas_cost": 17.0,
    }
    assert env.model_dynamics.midprice_model.volatility == 3.0
    np.testing.assert_allclose(env.model_dynamics.arrival_model.alpha[1], [123.0, 123.0])
    assert env.model_dynamics.gas_cost == 17.0
    np.testing.assert_allclose(obs[GAS_COST_KEY], np.full(env.num_trajectories, 17.0))
    np.testing.assert_allclose(
        env.initial_state[GAS_COST_KEY],
        np.full(env.num_trajectories, 17.0),
    )


def test_consecutive_resets_can_sample_different_regimes():
    config = UniformDomainRandomizationConfig(
        sigma_range=(1.0, 4.0),
        arrival_rate_range=(50.0, 200.0),
        gas_cost_range=(0.0, 20.0),
    )
    env = DomainRandomizedAMMEnvironment(create_test_amm_env(), config, seed=3)

    env.reset()
    first = env.last_domain_parameters
    env.reset()
    second = env.last_domain_parameters

    assert first != second


def test_default_sb3_observation_hides_gas_cost():
    assert GAS_COST_KEY not in DEFAULT_OBS_KEYS

    config = UniformDomainRandomizationConfig(gas_cost_range=(11.0, 11.0))
    env = DomainRandomizedAMMEnvironment(create_test_amm_env(), config, seed=4)
    sb3_env = StableBaselinesAMMEnvironment(env)

    obs = sb3_env.reset()

    assert obs.shape == (env.num_trajectories, len(DEFAULT_OBS_KEYS))
    assert obs.shape[1] == 4
    assert env.state[GAS_COST_KEY][0] == 11.0


def test_robust_script_factory_uses_requested_market_components():
    args = Namespace(
        num_trajectories=2,
        terminal_time=1.0,
        n_steps=5,
        tau=5,
        alpha3=7.0,
        initial_wealth=INITIAL_WEALTH,
        inventory_phi=20.0,
        arrival_alpha2=2.0,
        kernel_beta=0.25,
        kernel_window=3,
        liquidity_scale=1e5,
        trade_size_notional=12.0,
        price_impact_depth_window=4,
        price_impact_min_depth=1e-9,
    )
    params = DomainParameters(sigma=0.12, arrival_rate=80.0, gas_cost=6.0)

    env = make_fixed_env(args, params, seed=123)

    assert isinstance(env.model_dynamics.midprice_model, GeometricBrownianMotionMidpriceModel)
    assert env.model_dynamics.midprice_model.volatility == 0.12
    assert isinstance(env.model_dynamics.arrival_model, LiquidityKernelArrivalModel)
    assert env.model_dynamics.arrival_model.beta == 0.25
    assert env.model_dynamics.arrival_model.K == 3
    assert env.model_dynamics.arrival_model.liquidity_scale == 1e5
    np.testing.assert_allclose(env.model_dynamics.arrival_model.alpha[0], [10.0, 10.0])
    np.testing.assert_allclose(env.model_dynamics.arrival_model.alpha[1], [80.0, 80.0])
    np.testing.assert_allclose(env.model_dynamics.arrival_model.alpha[2], [2.0, 2.0])
    np.testing.assert_allclose(env.model_dynamics.arrival_model.alpha[3], [7.0, 7.0])
    assert isinstance(env.model_dynamics.price_impact_model, LiquidityDepthUniswapV3PriceImpact)
    assert env.model_dynamics.price_impact_model.depth_window == 4
    assert env.model_dynamics.price_impact_model.min_depth == 1e-9
    assert env.model_dynamics.price_impact_model.trade_size_unit == "token1_notional"
    np.testing.assert_allclose(
        env.model_dynamics.price_impact_model.trade_size_sampler(
            np.random.default_rng(0),
            3,
        ),
        np.full(3, 12.0),
    )
    assert isinstance(env.reward_function, RunningInventoryPenalty)
    assert env.reward_function.per_step_inventory_aversion == 20.0
    assert env.reward_function.terminal_inventory_aversion == 0.0
    assert env.reward_function.inventory_exponent == 2.0
    # Penalty is value-normalized, so phi is a fraction of wealth per unit time
    assert env.reward_function.reference_wealth == INITIAL_WEALTH
    assert env.model_dynamics.gas_cost == 6.0


def test_robust_script_parser_defaults_use_inventory_penalty_domain_ranges():
    args = parse_args([])

    assert args.inventory_phi == 0.02
    assert tuple(args.train_sigma_range) == (0.01, 0.10)
    assert tuple(args.train_gas_cost_range) == (1.0, 6.0)
    assert tuple(args.train_arrival_rate_range) == (50.0, 200.0)
    assert args.eval_sigma_values == [0.025, 0.10, 0.30]
    assert args.eval_gas_cost_values == [0.0, 10.0, 40.0]


def test_robust_script_reports_running_inventory_objective_columns():
    params = DomainParameters(sigma=0.01, arrival_rate=50.0, gas_cost=1.0)

    row = summarize_running_inventory_objective(
        "robust_ppo",
        params,
        np.array([-2.0, 4.0]),
    )

    assert row["mean_running_inventory_objective"] == 1.0
    assert row["std_running_inventory_objective"] == 3.0
    assert "mean_pnl" not in row
    assert "mean_final_wealth" not in row

    rows = [
        summarize_running_inventory_objective(
            "robust_ppo",
            params,
            np.array([3.0]),
        ),
        summarize_running_inventory_objective(
            "nominal_ppo",
            params,
            np.array([1.0]),
        ),
        summarize_running_inventory_objective(
            "uniform",
            params,
            np.array([-1.0]),
        ),
    ]
    add_gap_columns(rows)

    assert rows[0]["robust_vs_nominal_mean_running_inventory_objective_gap"] == 2.0
    assert rows[0]["robust_vs_uniform_mean_running_inventory_objective_gap"] == 4.0
    assert "robust_vs_nominal_mean_pnl_gap" not in rows[0]

    summary = summarize_rows(rows)
    assert summary["robust_ppo"]["mean_of_regime_mean_running_inventory_objective"] == 3.0
    assert summary["robust_ppo"]["worst_regime_mean_running_inventory_objective"] == 3.0
    assert summary["robust_ppo"]["best_regime_mean_running_inventory_objective"] == 3.0
    assert "mean_of_regime_mean_pnl" not in summary["robust_ppo"]


def test_ppo_smoke_learns_on_domain_randomized_env():
    config = UniformDomainRandomizationConfig(
        sigma_range=(1.0, 1.5),
        arrival_rate_range=(50.0, 60.0),
        gas_cost_range=(0.0, 1.0),
    )
    env = DomainRandomizedAMMEnvironment(create_test_amm_env(), config, seed=5)
    sb3_env = StableBaselinesAMMEnvironment(env)
    train_env = StructuredMultiDiscreteVecEnv(sb3_env, tau=5)
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=5,
        batch_size=10,
        n_epochs=1,
        gamma=1.0,
        verbose=0,
        seed=5,
    )

    np.testing.assert_array_equal(model.action_space.nvec, np.array([11, 5, 2]))
    model.learn(total_timesteps=10)
