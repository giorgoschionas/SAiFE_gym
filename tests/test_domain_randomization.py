from argparse import Namespace

import numpy as np
import pytest
from stable_baselines3 import PPO

from experiments.helpers import INITIAL_WEALTH
from experiments.policy_behavior_diagnostics import (
    BEHAVIOR_DIAGNOSTIC_COLUMNS,
    zero_behavior_diagnostics,
)
from experiments.train_robust_lp_agent import (
    DEFAULT_DOMAIN_RANDOMIZATION_SEED_OFFSET,
    add_gap_columns,
    apply_smoke_overrides,
    evaluate_cash,
    evaluation_regime_seed,
    evaluation_regimes,
    make_domain_randomized_env,
    make_fixed_env,
    make_robust_env,
    parse_args,
    resolve_train_domains_per_reset,
    summarize_running_inventory_objective,
    summarize_rows,
    summarize_single_training_seed,
    validate_evaluation_configuration,
    validate_and_derive_decision_timing,
)
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import (
    DEFAULT_OBS_KEYS,
    StableBaselinesAMMEnvironment,
)
from SAiFE_gym.gym.domain_randomization import (
    BatchedDomainParameters,
    DomainParameters,
    DomainRandomizedAMMEnvironment,
    UniformDomainRandomizationConfig,
)
from SAiFE_gym.gym.index_names import (
    GAS_COST_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    POOL_CURRENT_TICK_KEY,
    PORTFOLIO_VALUE_KEY,
)
from SAiFE_gym.rewards.RewardFunctions import RunningInventoryPenalty
from SAiFE_gym.stochastic_processes.arrival_models import (
    LiquidityKernelArrivalModel,
    PoissonArrivalModel,
    PoissonLinearArrivalModel,
    PoissonNonLinearArrivalModel,
)
from SAiFE_gym.stochastic_processes.midprice_models import (
    BrownianMotionMidpriceModel,
    GeometricBrownianMotionMidpriceModel,
    OrnsteinUhlenbeckMidpriceModel,
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
    )
    rng = np.random.default_rng(123)

    for _ in range(100):
        params = config.sample(rng)
        assert 1.0 <= params.sigma <= 4.0
        assert 50.0 <= params.arrival_rate <= 200.0


def test_seeded_sampler_is_reproducible():
    config = UniformDomainRandomizationConfig()
    env_a = DomainRandomizedAMMEnvironment(create_test_amm_env(), config, seed=7)
    env_b = DomainRandomizedAMMEnvironment(create_test_amm_env(), config, seed=7)

    _, info_a = env_a.reset()
    _, info_b = env_b.reset()

    assert info_a["domain_parameters"] == info_b["domain_parameters"]


def test_batched_sampler_is_balanced_bounded_and_reproducible():
    config = UniformDomainRandomizationConfig(
        sigma_range=(1.0, 4.0),
        arrival_rate_range=(50.0, 200.0),
    )

    params_a = config.sample_batch(np.random.default_rng(123), 11, 4)
    params_b = config.sample_batch(np.random.default_rng(123), 11, 4)

    assert params_a.sigma.shape == (11, 1)
    assert params_a.arrival_rate.shape == (11, 2)
    assert params_a.domain_id.shape == (11,)
    assert np.all((1.0 <= params_a.sigma) & (params_a.sigma <= 4.0))
    assert np.all(
        (50.0 <= params_a.arrival_rate)
        & (params_a.arrival_rate <= 200.0)
    )
    np.testing.assert_array_equal(
        np.sort(np.unique(params_a.domain_id)),
        np.arange(4),
    )
    counts = np.bincount(params_a.domain_id, minlength=4)
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(params_a.sigma, params_b.sigma)
    np.testing.assert_array_equal(params_a.arrival_rate, params_b.arrival_rate)
    np.testing.assert_array_equal(params_a.domain_id, params_b.domain_id)

    for domain_id in range(4):
        mask = params_a.domain_id == domain_id
        assert np.unique(params_a.sigma[mask]).size == 1
        assert np.unique(params_a.arrival_rate[mask], axis=0).shape[0] == 1


@pytest.mark.parametrize(
    ("num_trajectories", "num_domains"),
    [(0, 1), (3, 0), (3, 4), (3, 1.5)],
)
def test_batched_sampler_rejects_invalid_sizes(num_trajectories, num_domains):
    with pytest.raises(ValueError):
        UniformDomainRandomizationConfig().sample_batch(
            np.random.default_rng(1),
            num_trajectories,
            num_domains,
        )


def test_reset_applies_domain_parameters_to_models_and_state():
    config = UniformDomainRandomizationConfig(
        sigma_range=(3.0, 3.0),
        arrival_rate_range=(123.0, 123.0),
    )
    env = DomainRandomizedAMMEnvironment(create_test_amm_env(), config, seed=1)

    obs, info = env.reset()

    assert info["domain_parameters"] == {
        "sigma": 3.0,
        "arrival_rate": 123.0,
    }
    assert env.model_dynamics.midprice_model.volatility == 3.0
    np.testing.assert_allclose(env.model_dynamics.arrival_model.alpha[1], [123.0, 123.0])
    assert env.model_dynamics.gas_cost == 0.0
    np.testing.assert_allclose(obs[GAS_COST_KEY], np.zeros(env.num_trajectories))
    np.testing.assert_allclose(
        env.initial_state[GAS_COST_KEY],
        np.zeros(env.num_trajectories),
    )


def test_consecutive_resets_can_sample_different_regimes():
    config = UniformDomainRandomizationConfig(
        sigma_range=(1.0, 4.0),
        arrival_rate_range=(50.0, 200.0),
    )
    env = DomainRandomizedAMMEnvironment(create_test_amm_env(), config, seed=3)

    env.reset()
    first = env.last_domain_parameters
    env.reset()
    second = env.last_domain_parameters

    assert first != second


def test_batched_reset_applies_trajectory_parameters_for_whole_episode():
    config = UniformDomainRandomizationConfig(
        sigma_range=(1.0, 4.0),
        arrival_rate_range=(50.0, 200.0),
    )
    env = DomainRandomizedAMMEnvironment(
        create_test_amm_env(num_trajectories=6),
        config,
        seed=19,
        num_domains=3,
    )

    obs, info = env.reset()
    params = env.last_domain_parameters

    assert isinstance(params, BatchedDomainParameters)
    np.testing.assert_array_equal(
        env.model_dynamics.midprice_model.episode_volatility,
        params.sigma,
    )
    np.testing.assert_array_equal(
        env.model_dynamics.arrival_model.episode_baseline_intensity,
        params.arrival_rate,
    )
    np.testing.assert_array_equal(obs[GAS_COST_KEY], np.zeros(env.num_trajectories))
    np.testing.assert_array_equal(info["domain_parameters"]["sigma"], params.sigma)

    # Batched randomization does not turn scalar configuration attributes into
    # arrays or overwrite their nominal constructor values.
    assert env.model_dynamics.midprice_model.volatility == 2.0
    np.testing.assert_array_equal(
        env.model_dynamics.arrival_model.alpha[1],
        [100.0, 100.0],
    )
    assert env.model_dynamics.gas_cost == 0.0

    first_sigma = params.sigma.copy()
    first_arrival_rate = params.arrival_rate.copy()
    action = np.tile(np.array([-2.0, 2.0, 1.0]), (env.num_trajectories, 1))
    for _ in range(2):
        obs, _, _, _, _ = env.step(action)
        np.testing.assert_array_equal(
            env.model_dynamics.midprice_model.episode_volatility,
            first_sigma,
        )
        np.testing.assert_array_equal(
            env.model_dynamics.arrival_model.episode_baseline_intensity,
            first_arrival_rate,
        )
        np.testing.assert_array_equal(obs[GAS_COST_KEY], np.zeros(env.num_trajectories))
        assert env.model_dynamics.midprice_model.current_state.shape == (6, 1)
        assert env.model_dynamics.arrival_model.current_state.shape == (6, 2)

    env.reset()
    second = env.last_domain_parameters
    assert isinstance(second, BatchedDomainParameters)
    assert not np.array_equal(first_sigma, second.sigma)
    assert not np.array_equal(first_arrival_rate, second.arrival_rate)


def test_batched_reset_metadata_arrays_are_copied():
    env = DomainRandomizedAMMEnvironment(
        create_test_amm_env(num_trajectories=4),
        UniformDomainRandomizationConfig(),
        seed=22,
        num_domains=4,
    )
    _, info = env.reset()
    expected = env.last_domain_parameters.sigma.copy()

    info["domain_parameters"]["sigma"][:] = -1.0

    np.testing.assert_array_equal(env.last_domain_parameters.sigma, expected)
    np.testing.assert_array_equal(
        env.model_dynamics.midprice_model.episode_volatility,
        expected,
    )


def test_domain_parameters_do_not_modify_fixed_gas_cost_state():
    base_env = create_test_amm_env(num_trajectories=3)
    base_env.model_dynamics.gas_cost = 7.0
    env = DomainRandomizedAMMEnvironment(
        base_env,
        UniformDomainRandomizationConfig(),
        seed=1,
        num_domains=3,
    )
    params = BatchedDomainParameters(
        sigma=np.zeros((3, 1)),
        arrival_rate=np.zeros((3, 2)),
        domain_id=np.arange(3),
    )
    base_env._initial_state[GAS_COST_KEY] = np.full(3, 7.0)
    env.apply_domain_parameters(params)
    obs, _ = base_env.reset()
    np.testing.assert_array_equal(obs[GAS_COST_KEY], np.full(3, 7.0))


@pytest.mark.parametrize(
    "model_class",
    [
        BrownianMotionMidpriceModel,
        GeometricBrownianMotionMidpriceModel,
        OrnsteinUhlenbeckMidpriceModel,
    ],
)
def test_midprice_episode_volatility_preserves_batched_shape(model_class):
    model = model_class(
        volatility=0.2,
        initial_price=100.0,
        step_size=0.1,
        num_trajectories=4,
        seed=8,
    )
    volatility = np.array([[0.0], [0.1], [0.2], [0.3]])
    model.set_episode_volatility(volatility)

    model.update(None, None, None)

    assert isinstance(model.volatility, float)
    assert model.current_state.shape == (4, 1)
    np.testing.assert_array_equal(model.episode_volatility, volatility)


@pytest.mark.parametrize(
    "arrival_model",
    [
        PoissonArrivalModel(num_trajectories=3),
        PoissonLinearArrivalModel(num_trajectories=3),
        PoissonNonLinearArrivalModel(num_trajectories=3),
        LiquidityKernelArrivalModel(num_trajectories=3),
    ],
)
def test_all_arrival_models_reset_to_episode_baseline(arrival_model):
    baseline = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    arrival_model.set_episode_baseline_intensity(baseline)
    arrival_model.current_state[:] = -1.0

    arrival_model.reset()

    np.testing.assert_array_equal(arrival_model.current_state, baseline)
    assert arrival_model.current_state.shape == (3, 2)


@pytest.mark.parametrize(
    "model_class",
    [PoissonLinearArrivalModel, PoissonNonLinearArrivalModel],
)
def test_state_dependent_arrival_update_uses_trajectory_baseline(model_class):
    alpha = np.array([
        [0.0, 0.0],
        [100.0, 100.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    model = model_class(
        alpha=alpha,
        num_trajectories=3,
    )
    baseline = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    model.set_episode_baseline_intensity(baseline)
    state = {
        "active_liquidity": np.zeros(3),
        "amm_price": np.full(3, 100.0),
        "midprice": np.full(3, 100.0),
    }

    model.update(None, None, None, state)

    np.testing.assert_array_equal(model.current_state, baseline)


def test_liquidity_kernel_update_uses_trajectory_baseline():
    alpha = np.array([
        [0.0, 0.0],
        [100.0, 100.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    model = LiquidityKernelArrivalModel(
        alpha=alpha,
        K=2,
        num_trajectories=3,
    )
    baseline = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    model.set_episode_baseline_intensity(baseline)
    state = {
        "liquidity_array": np.zeros((3, 5)),
        "current_tick": np.full(3, 2),
        "tick_lower_global": 0,
        "amm_price": np.full(3, 100.0),
        "midprice": np.full(3, 100.0),
    }

    model.update(None, None, None, state)

    np.testing.assert_array_equal(model.current_state, baseline)


def test_episode_parameter_setters_validate_vectorized_shapes():
    midprice_model = BrownianMotionMidpriceModel(num_trajectories=3)
    arrival_model = PoissonArrivalModel(num_trajectories=3)

    with pytest.raises(ValueError, match="shape"):
        midprice_model.set_episode_volatility(np.ones(3))
    with pytest.raises(ValueError, match="shape"):
        arrival_model.set_episode_baseline_intensity(np.ones((3, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        midprice_model.set_episode_volatility(-np.ones((3, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        arrival_model.set_episode_baseline_intensity(-np.ones((3, 2)))


def test_nominal_environment_scalar_api_and_seeded_behavior_are_unchanged():
    env_a = create_test_amm_env(num_trajectories=3, seed=77)
    env_b = create_test_amm_env(num_trajectories=3, seed=77)

    assert isinstance(env_a.model_dynamics.midprice_model.volatility, float)
    assert env_a.model_dynamics.midprice_model.episode_volatility.shape == (3, 1)
    assert env_a.model_dynamics.arrival_model.alpha.shape == (4, 2)
    assert env_a.model_dynamics.arrival_model.episode_baseline_intensity.shape == (3, 2)
    assert np.isscalar(env_a.model_dynamics.gas_cost)

    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key], obs_b[key])

    action = np.tile(np.array([-2.0, 2.0, -1.0]), (3, 1))
    transition_a = env_a.step(action)
    transition_b = env_b.step(action)
    for key in transition_a[0]:
        np.testing.assert_array_equal(transition_a[0][key], transition_b[0][key])
    for value_a, value_b in zip(transition_a[1:4], transition_b[1:4]):
        np.testing.assert_array_equal(value_a, value_b)

    poisson_model = PoissonArrivalModel(
        intensity=np.array([12.0, 13.0]),
        num_trajectories=3,
    )
    assert poisson_model.intensity.shape == (2,)
    np.testing.assert_array_equal(poisson_model.intensity, [12.0, 13.0])


def test_amm_environment_seed_resets_model_dynamics_and_price_impact_rngs():
    env = create_test_amm_env(num_trajectories=3, seed=77)

    env.seed(123)
    first_model_draw = env.model_dynamics.rng.integers(0, 1_000_000, size=5)
    first_price_impact_draw = env.model_dynamics.price_impact_model.rng.uniform(
        size=5
    )

    env.seed(123)
    second_model_draw = env.model_dynamics.rng.integers(0, 1_000_000, size=5)
    second_price_impact_draw = env.model_dynamics.price_impact_model.rng.uniform(
        size=5
    )

    np.testing.assert_array_equal(first_model_draw, second_model_draw)
    np.testing.assert_array_equal(
        first_price_impact_draw,
        second_price_impact_draw,
    )


def test_script_domain_randomization_seed_offset_survives_external_seed_calls():
    args = parse_args([
        "--num-trajectories",
        "4",
        "--n-steps",
        "5",
        "--seed",
        "11",
        "--train-domains-per-reset",
        "2",
    ])
    env = make_domain_randomized_env(args)

    env.seed(args.seed)
    env.reset()

    expected = UniformDomainRandomizationConfig(
        sigma_range=tuple(args.train_sigma_range),
        arrival_rate_range=tuple(args.train_arrival_rate_range),
    ).sample_batch(
        np.random.default_rng(args.seed + DEFAULT_DOMAIN_RANDOMIZATION_SEED_OFFSET),
        args.num_trajectories,
        args.train_domains_per_reset,
    )

    assert env.seed_ == args.seed
    assert env.domain_seed_ == args.seed + DEFAULT_DOMAIN_RANDOMIZATION_SEED_OFFSET
    assert isinstance(env.last_domain_parameters, BatchedDomainParameters)
    np.testing.assert_array_equal(env.last_domain_parameters.sigma, expected.sigma)
    np.testing.assert_array_equal(
        env.last_domain_parameters.arrival_rate,
        expected.arrival_rate,
    )
    np.testing.assert_array_equal(
        env.last_domain_parameters.domain_id,
        expected.domain_id,
    )


def test_default_sb3_observation_hides_gas_cost():
    assert GAS_COST_KEY not in DEFAULT_OBS_KEYS

    env = DomainRandomizedAMMEnvironment(
        create_test_amm_env(),
        UniformDomainRandomizationConfig(),
        seed=4,
    )
    sb3_env = StableBaselinesAMMEnvironment(env)

    obs = sb3_env.reset()

    assert obs.shape == (env.num_trajectories, len(DEFAULT_OBS_KEYS))
    assert obs.shape[1] == 6
    assert env.state[GAS_COST_KEY][0] == 0.0


def test_domain_randomized_script_factory_uses_requested_market_components():
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
        nominal_gas_cost=6.0,
    )
    params = DomainParameters(sigma=0.12, arrival_rate=80.0)

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
    assert env.model_dynamics.gas_cost == 6.0


def test_domain_randomized_script_parser_defaults_use_expected_configuration():
    args = parse_args([])

    assert args.output_dir == "experiments/results/domain_randomized_ppo"
    assert args.n_steps == 1000
    assert args.decision_stride == 20
    assert args.tau == 500
    assert args.tick_stride == 5
    assert args.alpha3 == 4000.0
    assert args.initial_wealth == 1_000.0
    assert args.inventory_phi == 0.4
    assert args.evaluation_seed == 100042
    assert args.periodic_rebalance_every == 5
    assert args.periodic_width == 125
    assert args.nominal_sigma == 0.03
    assert args.nominal_arrival_rate == 250.0
    assert tuple(args.train_sigma_range) == (0.01, 0.05)
    assert tuple(args.train_arrival_rate_range) == (200.0, 400.0)
    assert args.nominal_gas_cost == 2.0
    assert args.eval_in_distribution_sigma_values == [0.015, 0.030, 0.045]
    assert args.eval_in_distribution_arrival_rate_values == [250.0, 300.0, 350.0]
    assert args.eval_stress_sigma_values == [0.065, 0.08]
    assert args.eval_stress_arrival_rate_values == [150.0, 450.0]


def test_decision_stride_defaults_derive_ppo_training_budget():
    args = parse_args([])

    validate_and_derive_decision_timing(args)

    assert args.max_agent_decisions_per_episode == 50
    assert args.ppo_n_steps == 50
    assert args.ppo_total_timesteps == 300_000


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (["--decision-stride", "0"], "positive"),
        (["--n-steps", "1000", "--decision-stride", "30"], "n_steps"),
        (["--total-timesteps", "1001", "--decision-stride", "20"], "total_timesteps"),
    ],
)
def test_decision_stride_validation_rejects_invalid_timing(argv, error):
    args = parse_args(argv)

    with pytest.raises(ValueError, match=error):
        validate_and_derive_decision_timing(args)


def test_domain_randomized_script_defaults_propagate_to_environment_and_action_space():
    args = parse_args([])
    args.num_trajectories = 2
    args.n_steps = 5
    params = DomainParameters(
        sigma=args.nominal_sigma,
        arrival_rate=args.nominal_arrival_rate,
    )

    env = make_fixed_env(args, params, seed=123)
    state, _ = env.reset(seed=123)
    ppo_env = StructuredMultiDiscreteVecEnv(
        StableBaselinesAMMEnvironment(env),
        args.tau,
        args.tick_stride,
    )

    assert env.initial_wealth == 1_000.0
    assert env.reward_function.per_step_inventory_aversion == args.inventory_phi
    assert env.model_dynamics.tau == 500
    assert args.tick_stride == 5
    np.testing.assert_array_equal(env.action_space.low, [-500.0, -499.0, -1.0])
    np.testing.assert_array_equal(env.action_space.high, [499.0, 500.0, 1.0])
    np.testing.assert_array_equal(ppo_env.action_space.nvec, [201, 100, 2])
    np.testing.assert_array_equal(state[PORTFOLIO_VALUE_KEY], [1_000.0, 1_000.0])
    np.testing.assert_array_equal(
        state[LP_TICK_LOWER_KEY],
        state[POOL_CURRENT_TICK_KEY] - args.tau,
    )
    np.testing.assert_array_equal(
        state[LP_TICK_UPPER_KEY],
        state[POOL_CURRENT_TICK_KEY] + args.tau,
    )
    assert resolve_train_domains_per_reset(args) == args.num_trajectories
    validate_evaluation_configuration(args)


def test_evaluation_seed_is_independent_of_training_seed():
    first = parse_args(["--seed", "43"])
    second = parse_args(["--seed", "52"])

    assert first.evaluation_seed == second.evaluation_seed == 100042
    assert evaluation_regime_seed(first, 3) == 103042
    assert evaluation_regime_seed(first, 3) == evaluation_regime_seed(second, 3)

    overridden = parse_args(["--seed", "52", "--evaluation-seed", "7"])
    assert evaluation_regime_seed(overridden, 3) == 3007

    invalid = parse_args(["--evaluation-seed", "-1"])
    with pytest.raises(ValueError, match="non-negative"):
        validate_evaluation_configuration(invalid)


def test_evaluation_regimes_separate_interpolation_and_stress_grids():
    args = parse_args([])

    regimes = evaluation_regimes(args)
    in_distribution = [
        regime for regime in regimes if regime.evaluation_set == "in_distribution"
    ]
    stress = [regime for regime in regimes if regime.evaluation_set == "stress"]

    assert len(in_distribution) == 9
    assert len(stress) == 4
    assert regimes[:9] == in_distribution
    assert regimes[9:] == stress
    assert len({regime.parameters for regime in regimes}) == 13
    assert in_distribution[0].parameters == DomainParameters(0.015, 250.0)
    assert in_distribution[-1].parameters == DomainParameters(0.045, 350.0)
    assert stress[0].parameters == DomainParameters(0.065, 150.0)
    assert stress[-1].parameters == DomainParameters(0.08, 450.0)
    assert DomainParameters(0.08, 450.0) in {
        regime.parameters for regime in stress
    }


def test_smoke_overrides_keep_one_regime_in_each_evaluation_set():
    args = parse_args(["--smoke-test"])

    apply_smoke_overrides(args)
    validate_evaluation_configuration(args)

    assert [
        (regime.evaluation_set, regime.parameters)
        for regime in evaluation_regimes(args)
    ] == [
        ("in_distribution", DomainParameters(0.030, 300.0)),
        ("stress", DomainParameters(0.08, 450.0)),
    ]


def test_removed_generic_evaluation_cli_flags_are_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--eval-gas-cost-values", "0.0"])
    with pytest.raises(SystemExit):
        parse_args(["--eval-in-distribution-gas-cost-values", "0.0"])


@pytest.mark.parametrize(
    ("attribute", "values", "error"),
    [
        ("eval_in_distribution_arrival_rate_values", [150.0], "training range"),
        ("eval_in_distribution_sigma_values", [0.025, 0.025], "duplicate"),
        ("eval_stress_arrival_rate_values", [-1.0], "non-negative"),
        ("eval_stress_sigma_values", [np.inf], "finite"),
    ],
)
def test_evaluation_grid_value_validation(attribute, values, error):
    args = parse_args([])
    setattr(args, attribute, values)

    with pytest.raises(ValueError, match=error):
        validate_evaluation_configuration(args)


def test_stress_grid_rejects_any_fully_in_support_cartesian_regime():
    args = parse_args([])
    args.eval_stress_sigma_values = [0.030, 0.30]
    args.eval_stress_arrival_rate_values = [300.0, 450.0]

    with pytest.raises(ValueError, match="every stress evaluation regime"):
        validate_evaluation_configuration(args)


def test_domain_randomized_factory_default_domains_are_capped_by_trajectory_count():
    args = parse_args(["--num-trajectories", "3", "--n-steps", "5"])

    assert resolve_train_domains_per_reset(args) == 3
    assert args.train_domains_per_reset == 3


def test_domain_randomized_factory_default_uses_ten_domains_for_standard_batch():
    args = parse_args(["--num-trajectories", "100", "--n-steps", "5"])

    assert resolve_train_domains_per_reset(args) == 10
    assert args.train_domains_per_reset == 10


def test_robust_factory_alias_preserves_existing_imports():
    canonical_args = parse_args(["--num-trajectories", "3", "--n-steps", "5"])
    compatibility_args = parse_args(["--num-trajectories", "3", "--n-steps", "5"])

    canonical_env = make_domain_randomized_env(canonical_args)
    compatibility_env = make_robust_env(compatibility_args)

    assert isinstance(canonical_env, DomainRandomizedAMMEnvironment)
    assert isinstance(compatibility_env, DomainRandomizedAMMEnvironment)
    assert compatibility_env.num_domains == canonical_env.num_domains == 3


@pytest.mark.parametrize("num_domains", [0, 4, 1.5])
def test_domain_randomized_domain_count_validation(num_domains):
    args = parse_args(["--num-trajectories", "3"])
    args.train_domains_per_reset = num_domains

    with pytest.raises(ValueError, match="integer|1 <= value <= num_trajectories"):
        resolve_train_domains_per_reset(args)


def test_domain_randomized_script_reports_unambiguous_objective_columns():
    params = DomainParameters(sigma=0.01, arrival_rate=50.0)

    row = summarize_running_inventory_objective(
        "domain_randomized_ppo",
        params,
        "in_distribution",
        np.array([-2.0, 4.0]),
        training_seed=43,
        evaluation_seed=100042,
    )

    assert row["evaluation_set"] == "in_distribution"
    assert row["training_seed"] == 43
    assert row["evaluation_seed"] == 100042
    assert row["sigma"] == 0.01
    assert row["arrival_rate"] == 50.0
    assert "gas_cost" not in row
    assert row["mean_running_inventory_objective"] == 1.0
    assert row["evaluation_path_std_running_inventory_objective"] == 3.0
    assert row["n_evaluation_paths"] == 2
    assert all(row[name] == 0.0 for name in BEHAVIOR_DIAGNOSTIC_COLUMNS)
    assert "std_running_inventory_objective" not in row
    assert "n_samples" not in row
    assert "mean_pnl" not in row
    assert "mean_final_wealth" not in row

    rows = [
        summarize_running_inventory_objective(
            "domain_randomized_ppo",
            params,
            "in_distribution",
            np.array([3.0]),
            training_seed=43,
            evaluation_seed=100042,
        ),
        summarize_running_inventory_objective(
            "nominal_ppo",
            params,
            "in_distribution",
            np.array([1.0]),
            training_seed=43,
            evaluation_seed=100042,
        ),
        summarize_running_inventory_objective(
            "periodic_rebalance",
            params,
            "in_distribution",
            np.array([-1.0]),
            training_seed=43,
            evaluation_seed=100042,
        ),
        summarize_running_inventory_objective(
            "cash",
            params,
            "in_distribution",
            np.array([0.0]),
            training_seed=43,
            evaluation_seed=100042,
        ),
        summarize_running_inventory_objective(
            "domain_randomized_ppo",
            params,
            "stress",
            np.array([-2.0]),
            training_seed=43,
            evaluation_seed=100042,
        ),
        summarize_running_inventory_objective(
            "nominal_ppo",
            params,
            "stress",
            np.array([-3.0]),
            training_seed=43,
            evaluation_seed=100042,
        ),
        summarize_running_inventory_objective(
            "periodic_rebalance",
            params,
            "stress",
            np.array([-4.0]),
            training_seed=43,
            evaluation_seed=100042,
        ),
        summarize_running_inventory_objective(
            "cash",
            params,
            "stress",
            np.array([0.0]),
            training_seed=43,
            evaluation_seed=100042,
        ),
    ]
    add_gap_columns(rows)

    assert (
        rows[0][
            "domain_randomized_vs_nominal_mean_running_inventory_objective_gap"
        ]
        == 2.0
    )
    assert (
        rows[0][
            "domain_randomized_vs_periodic_rebalance_"
            "mean_running_inventory_objective_gap"
        ]
        == 4.0
    )
    assert (
        rows[0]["domain_randomized_vs_cash_mean_running_inventory_objective_gap"]
        == 3.0
    )
    assert (
        rows[4][
            "domain_randomized_vs_nominal_mean_running_inventory_objective_gap"
        ]
        == 1.0
    )
    assert (
        rows[4][
            "domain_randomized_vs_periodic_rebalance_"
            "mean_running_inventory_objective_gap"
        ]
        == 2.0
    )
    assert not any(key.startswith("robust_") for key in rows[0])

    summary = summarize_rows(rows)
    domain_randomized_summary = summary["domain_randomized_ppo"]
    assert (
        domain_randomized_summary["in_distribution"][
            "mean_of_regime_mean_running_inventory_objective"
        ]
        == 3.0
    )
    assert (
        domain_randomized_summary["in_distribution"][
            "minimum_regime_mean_running_inventory_objective"
        ]
        == 3.0
    )
    assert (
        domain_randomized_summary["stress"][
            "maximum_regime_mean_running_inventory_objective"
        ]
        == -2.0
    )
    assert domain_randomized_summary["in_distribution"]["num_regimes"] == 1
    assert domain_randomized_summary["stress"]["num_regimes"] == 1
    assert "robust_ppo" not in summary
    assert not any("evaluation_grid" in key for key in domain_randomized_summary)


def test_summary_aggregates_each_evaluation_set_independently():
    params = DomainParameters(sigma=0.055, arrival_rate=125.0)
    rows = [
        summarize_running_inventory_objective(
            "domain_randomized_ppo",
            params,
            evaluation_set,
            np.array([value]),
            training_seed=43,
            evaluation_seed=100042,
        )
        for evaluation_set, values in [
            ("in_distribution", [1.0, 5.0]),
            ("stress", [-8.0, -2.0]),
        ]
        for value in values
    ]

    summary = summarize_rows(rows)["domain_randomized_ppo"]

    assert summary["in_distribution"]["num_regimes"] == 2
    assert (
        summary["in_distribution"][
            "mean_of_regime_mean_running_inventory_objective"
        ]
        == 3.0
    )
    assert (
        summary["in_distribution"][
            "minimum_regime_mean_running_inventory_objective"
        ]
        == 1.0
    )
    assert (
        summary["in_distribution"][
            "maximum_regime_mean_running_inventory_objective"
        ]
        == 5.0
    )
    assert summary["stress"]["num_regimes"] == 2
    assert (
        summary["stress"]["mean_of_regime_mean_running_inventory_objective"]
        == -5.0
    )
    assert (
        summary["stress"]["minimum_regime_mean_running_inventory_objective"]
        == -8.0
    )
    assert (
        summary["stress"]["maximum_regime_mean_running_inventory_objective"]
        == -2.0
    )
    for evaluation_set in ["in_distribution", "stress"]:
        assert summary[evaluation_set][
            "behavior_diagnostics_mean_across_regimes"
        ] == zero_behavior_diagnostics()

    wrapped = summarize_single_training_seed(
        rows,
        training_seed=43,
        evaluation_seed=100042,
    )
    assert wrapped["summary_scope"] == "single_training_seed"
    assert wrapped["training_seed"] == 43
    assert wrapped["evaluation_seed"] == 100042
    assert wrapped["policies"]["domain_randomized_ppo"] == summary


def test_cash_baseline_is_zero_with_expected_sample_count():
    args = parse_args([])
    args.n_eval_episodes = 3
    args.num_trajectories = 4
    params = DomainParameters(sigma=0.3, arrival_rate=300.0)

    row = evaluate_cash(params, "stress", args)

    assert row["evaluation_set"] == "stress"
    assert row["policy"] == "cash"
    assert row["mean_running_inventory_objective"] == 0.0
    assert row["evaluation_path_std_running_inventory_objective"] == 0.0
    assert row["n_evaluation_paths"] == 12
    assert row["never_deployed_fraction"] == 1.0
    assert row["hold_action_fraction"] == 1.0
    assert row["rebalance_action_fraction"] == 0.0
    assert row["bankruptcy_fraction"] == 0.0
    assert row["mean_pnl_per_path"] == 0.0
    assert row["mean_inventory_penalty_per_path"] == 0.0


def test_objective_summary_validates_behavior_reward_decomposition():
    params = DomainParameters(sigma=0.055, arrival_rate=125.0)
    diagnostics = zero_behavior_diagnostics()
    diagnostics["mean_pnl_per_path"] = 5.0
    diagnostics["mean_inventory_penalty_per_path"] = 2.0

    row = summarize_running_inventory_objective(
        "domain_randomized_ppo",
        params,
        "in_distribution",
        np.array([1.0, 5.0]),
        training_seed=43,
        evaluation_seed=100042,
        behavior_diagnostics=diagnostics,
    )
    assert row["mean_running_inventory_objective"] == 3.0

    diagnostics["mean_pnl_per_path"] = 6.0
    with pytest.raises(ValueError, match="PnL minus mean inventory penalty"):
        summarize_running_inventory_objective(
            "domain_randomized_ppo",
            params,
            "in_distribution",
            np.array([1.0, 5.0]),
            training_seed=43,
            evaluation_seed=100042,
            behavior_diagnostics=diagnostics,
        )


def test_ppo_smoke_learns_on_domain_randomized_env():
    config = UniformDomainRandomizationConfig(
        sigma_range=(1.0, 1.5),
        arrival_rate_range=(50.0, 60.0),
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
