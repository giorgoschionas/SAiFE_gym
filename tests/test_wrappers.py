import gymnasium
import numpy as np
import pytest

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import (
    DEFAULT_OBS_KEYS,
    StableBaselinesAMMEnvironment,
)
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.wrappers import (
    DiscreteActionVecEnv,
    DiscreteActionWrapper,
    StructuredMultiDiscreteVecEnv,
    build_discrete_action_table,
)


def create_test_amm_env(
    num_trajectories: int = 1,
    n_steps: int = 5,
    tau: int = 5,
) -> AMMEnvironment:
    step_size = 1.0 / n_steps
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=0.0,
        initial_price=100.0,
        terminal_time=1.0,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=42,
    )
    arrival_model = PoissonArrivalModel(
        intensity=np.array([0.0, 0.0]),
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=42,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        tau=tau,
        num_ticks=100,
        seed=42,
    )
    return AMMEnvironment(
        terminal_time=1.0,
        n_steps=n_steps,
        model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        seed=42,
    )


class TestBuildDiscreteActionTable:
    def test_full_tick_grid_action_count_and_hold_row(self):
        table = build_discrete_action_table(tau=5, tick_stride=1)

        assert table.shape == (56, 3)
        np.testing.assert_array_equal(table[0], np.array([0.0, 1.0, 1.0]))

    def test_rebalance_rows_are_valid(self):
        table = build_discrete_action_table(tau=5, tick_stride=1)
        rebalance = table[1:]

        assert np.all(rebalance[:, 2] == -1.0)
        assert np.all(rebalance[:, 0] < rebalance[:, 1])
        assert np.all(rebalance[:, 0] >= -5)
        assert np.all(rebalance[:, 1] <= 5)

    def test_strided_grid_includes_upper_tau(self):
        table = build_discrete_action_table(tau=5, tick_stride=3)
        offsets = np.unique(table[1:, :2])

        np.testing.assert_array_equal(offsets, np.array([-5.0, -2.0, 1.0, 4.0, 5.0]))
        assert table.shape == (11, 3)

    @pytest.mark.parametrize("tau,tick_stride", [(0, 1), (5, 0)])
    def test_invalid_arguments_raise(self, tau, tick_stride):
        with pytest.raises(ValueError):
            build_discrete_action_table(tau=tau, tick_stride=tick_stride)


class TestDiscreteActionWrapper:
    def test_single_trajectory_exposes_discrete_space(self):
        env = create_test_amm_env(num_trajectories=1, tau=5)
        wrapped = DiscreteActionWrapper(env, tau=5)

        assert isinstance(wrapped.action_space, gymnasium.spaces.Discrete)
        assert wrapped.action_space.n == 56
        mapped = wrapped.action(0)
        assert mapped.shape == (1, 3)
        np.testing.assert_array_equal(mapped[0], np.array([0.0, 1.0, 1.0]))

    def test_batched_env_exposes_multidiscrete_space(self):
        env = create_test_amm_env(num_trajectories=3, tau=5)
        wrapped = DiscreteActionWrapper(env, tau=5)

        assert isinstance(wrapped.action_space, gymnasium.spaces.MultiDiscrete)
        np.testing.assert_array_equal(wrapped.action_space.nvec, np.full(3, 56))

        mapped = wrapped.action(np.array([0, 1, 2]))
        assert mapped.shape == (3, 3)
        assert mapped[0, 2] == 1.0
        assert np.all(mapped[1:, 2] == -1.0)


class TestDiscreteActionVecEnv:
    def test_vec_env_exposes_discrete_space_and_maps_actions(self):
        amm_env = create_test_amm_env(num_trajectories=2, tau=5)
        sb_env = StableBaselinesAMMEnvironment(amm_env)
        wrapped = DiscreteActionVecEnv(sb_env, tau=5)

        assert isinstance(wrapped.action_space, gymnasium.spaces.Discrete)
        assert wrapped.action_space.n == 56

        wrapped.reset()
        wrapped.step_async(np.array([0, 1]))
        assert sb_env.actions.shape == (2, 3)
        np.testing.assert_array_equal(sb_env.actions, wrapped.action_table[[0, 1]])

    def test_vec_env_step_smoke(self):
        amm_env = create_test_amm_env(num_trajectories=2, tau=5)
        sb_env = StableBaselinesAMMEnvironment(amm_env)
        wrapped = DiscreteActionVecEnv(sb_env, tau=5)

        obs = wrapped.reset()
        assert obs.shape == (2, len(DEFAULT_OBS_KEYS))

        wrapped.step_async(np.array([0, 1]))
        obs, rewards, dones, infos = wrapped.step_wait()

        assert obs.shape == (2, len(DEFAULT_OBS_KEYS))
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2


class TestStructuredMultiDiscreteVecEnv:
    def test_exposes_center_width_hold_space(self):
        amm_env = create_test_amm_env(num_trajectories=2, tau=5)
        sb_env = StableBaselinesAMMEnvironment(amm_env)
        wrapped = StructuredMultiDiscreteVecEnv(sb_env, tau=5)

        assert isinstance(wrapped.action_space, gymnasium.spaces.MultiDiscrete)
        np.testing.assert_array_equal(wrapped.action_space.nvec, np.array([11, 5, 2]))

    def test_unscale_maps_center_width_hold_to_internal_action(self):
        amm_env = create_test_amm_env(num_trajectories=2, tau=5)
        sb_env = StableBaselinesAMMEnvironment(amm_env)
        wrapped = StructuredMultiDiscreteVecEnv(sb_env, tau=5)

        actions = np.array([
            [5, 1, 0],   # center=0, half_width=2, rebalance
            [10, 4, 1],  # center=5, half_width=5, hold
        ])
        mapped = wrapped.unscale(actions)

        np.testing.assert_array_equal(
            mapped,
            np.array([[-2.0, 2.0, -1.0], [0.0, 5.0, 1.0]], dtype=np.float32),
        )

    def test_step_async_maps_structured_actions(self):
        amm_env = create_test_amm_env(num_trajectories=2, tau=5)
        sb_env = StableBaselinesAMMEnvironment(amm_env)
        wrapped = StructuredMultiDiscreteVecEnv(sb_env, tau=5)

        wrapped.reset()
        actions = np.array([[5, 0, 0], [4, 1, 1]])
        wrapped.step_async(actions)

        np.testing.assert_array_equal(sb_env.actions, wrapped.unscale(actions))
