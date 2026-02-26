"""
Tests for StableBaselinesAMMEnvironment — the SB3 VecEnv wrapper around AMMEnvironment.
"""

import numpy as np
import pytest
import gymnasium

from stable_baselines3 import PPO

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import (
    DEFAULT_OBS_KEYS,
    StableBaselinesAMMEnvironment,
)
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    FEES0_KEY,
    POOL_SQRT_PRICE_KEY,
    TIME_KEY,
)
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_amm_env(num_trajectories: int = 1, n_steps: int = 5, tau: int = 5) -> AMMEnvironment:
    step_size = 1.0 / n_steps
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=0.0,
        initial_price=100.0,
        terminal_time=1.0,
        step_size=step_size,
        num_trajectories=num_trajectories,
    )
    arrival_model = PoissonArrivalModel(
        intensity=np.array([100.0, 100.0]),
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


def create_sb3_env(
    num_trajectories: int = 1,
    n_steps: int = 5,
    obs_keys=None,
    store_terminal_observation_info: bool = True,
) -> StableBaselinesAMMEnvironment:
    amm_env = create_test_amm_env(num_trajectories=num_trajectories, n_steps=n_steps)
    return StableBaselinesAMMEnvironment(
        amm_env,
        obs_keys=obs_keys,
        store_terminal_observation_info=store_terminal_observation_info,
    )


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_shape_single_trajectory(self):
        env = create_sb3_env(num_trajectories=1)
        obs = env.reset()
        assert obs.shape == (1, 9)

    def test_shape_multi_trajectory(self):
        env = create_sb3_env(num_trajectories=3)
        obs = env.reset()
        assert obs.shape == (3, 9)

    def test_dtype(self):
        env = create_sb3_env(num_trajectories=1)
        obs = env.reset()
        assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# TestStepWait
# ---------------------------------------------------------------------------

class TestStepWait:
    def _step(self, env: StableBaselinesAMMEnvironment):
        env.reset()
        action = env.action_space.sample()
        # action_space.sample() returns shape (2,); replicate for all trajectories
        actions = np.tile(action, (env.num_trajectories, 1))
        env.step_async(actions)
        return env.step_wait()

    def test_obs_shape(self):
        num_traj = 2
        env = create_sb3_env(num_trajectories=num_traj)
        obs, _, _, _ = self._step(env)
        assert obs.shape == (num_traj, 9)

    def test_rewards_shape(self):
        num_traj = 2
        env = create_sb3_env(num_trajectories=num_traj)
        _, rewards, _, _ = self._step(env)
        assert rewards.shape == (num_traj,)

    def test_dones_shape_and_dtype(self):
        num_traj = 2
        env = create_sb3_env(num_trajectories=num_traj)
        _, _, dones, _ = self._step(env)
        assert dones.shape == (num_traj,)
        assert dones.dtype == bool


# ---------------------------------------------------------------------------
# TestInfosList
# ---------------------------------------------------------------------------

class TestInfosList:
    def _get_infos(self, env: StableBaselinesAMMEnvironment):
        env.reset()
        actions = np.zeros((env.num_trajectories, 2), dtype=np.float32)
        env.step_async(actions)
        _, _, _, infos = env.step_wait()
        return infos

    def test_is_list(self):
        env = create_sb3_env(num_trajectories=1)
        infos = self._get_infos(env)
        assert isinstance(infos, list)

    def test_length(self):
        num_traj = 3
        env = create_sb3_env(num_trajectories=num_traj)
        infos = self._get_infos(env)
        assert len(infos) == num_traj

    def test_elements_are_dicts(self):
        env = create_sb3_env(num_trajectories=2)
        infos = self._get_infos(env)
        assert all(isinstance(info, dict) for info in infos)


# ---------------------------------------------------------------------------
# TestAutoReset
# ---------------------------------------------------------------------------

class TestAutoReset:
    def _run_to_done(self, env: StableBaselinesAMMEnvironment):
        """Step until dones.all() is True, return final (obs, dones)."""
        env.reset()
        actions = np.zeros((env.num_trajectories, 2), dtype=np.float32)
        for _ in range(env.n_steps):
            env.step_async(actions)
            obs, _, dones, _ = env.step_wait()
        return obs, dones

    def test_triggers_at_end(self):
        env = create_sb3_env(num_trajectories=1, n_steps=5)
        obs, dones = self._run_to_done(env)
        assert dones.all()
        assert obs is not None

    def test_shape_preserved_after_auto_reset(self):
        num_traj = 2
        env = create_sb3_env(num_trajectories=num_traj, n_steps=5)
        obs, _ = self._run_to_done(env)
        assert obs.shape == (num_traj, 9)


# ---------------------------------------------------------------------------
# TestTerminalObs
# ---------------------------------------------------------------------------

class TestTerminalObs:
    def _run_episode(self, env: StableBaselinesAMMEnvironment):
        env.reset()
        actions = np.zeros((env.num_trajectories, 2), dtype=np.float32)
        mid_infos = None
        final_infos = None
        for step in range(env.n_steps):
            env.step_async(actions)
            _, _, dones, infos = env.step_wait()
            if step == env.n_steps // 2:
                mid_infos = infos
            if dones.all():
                final_infos = infos
                break
        return mid_infos, final_infos

    def test_key_present_on_done(self):
        env = create_sb3_env(num_trajectories=1, n_steps=5)
        _, final_infos = self._run_episode(env)
        assert final_infos is not None
        assert "terminal_observation" in final_infos[0]

    def test_shape_is_1d(self):
        env = create_sb3_env(num_trajectories=1, n_steps=5)
        _, final_infos = self._run_episode(env)
        assert final_infos[0]["terminal_observation"].shape == (9,)

    def test_absent_before_done(self):
        env = create_sb3_env(num_trajectories=1, n_steps=5)
        mid_infos, _ = self._run_episode(env)
        if mid_infos is not None:
            assert "terminal_observation" not in mid_infos[0]

    def test_disabled_flag(self):
        env = create_sb3_env(
            num_trajectories=1, n_steps=5, store_terminal_observation_info=False
        )
        _, final_infos = self._run_episode(env)
        assert final_infos is not None
        assert "terminal_observation" not in final_infos[0]


# ---------------------------------------------------------------------------
# TestCustomObsKeys
# ---------------------------------------------------------------------------

class TestCustomObsKeys:
    def test_custom_keys_change_dim(self):
        custom_keys = [POOL_SQRT_PRICE_KEY, TIME_KEY]
        env = create_sb3_env(num_trajectories=1, obs_keys=custom_keys)
        obs = env.reset()
        assert obs.shape == (1, 2)

    def test_array_key_raises(self):
        with pytest.raises(ValueError, match="array key"):
            create_sb3_env(num_trajectories=1, obs_keys=[FEES0_KEY])


# ---------------------------------------------------------------------------
# TestMultiTrajectory
# ---------------------------------------------------------------------------

class TestMultiTrajectory:
    def test_four_trajectories_full_episode(self):
        num_traj = 4
        n_steps = 5
        env = create_sb3_env(num_trajectories=num_traj, n_steps=n_steps)
        obs = env.reset()
        assert obs.shape == (num_traj, 9)

        actions = np.zeros((num_traj, 2), dtype=np.float32)
        for _ in range(n_steps):
            env.step_async(actions)
            obs, rewards, dones, infos = env.step_wait()
            assert obs.shape == (num_traj, 9)
            assert rewards.shape == (num_traj,)
            assert dones.shape == (num_traj,)
            assert len(infos) == num_traj


# ---------------------------------------------------------------------------
# TestSB3Integration
# ---------------------------------------------------------------------------

class TestSB3Integration:
    def test_ppo_instantiation(self):
        env = create_sb3_env(num_trajectories=1, n_steps=5)
        model = PPO("MlpPolicy", env, n_steps=5, batch_size=5, verbose=0)
        assert isinstance(model.observation_space, gymnasium.spaces.Box)
        assert isinstance(model.action_space, gymnasium.spaces.Box)

    def test_ppo_learn(self):
        env = create_sb3_env(num_trajectories=1, n_steps=5)
        model = PPO("MlpPolicy", env, n_steps=5, batch_size=5, verbose=0)
        model.learn(total_timesteps=20)  # should not raise


# ---------------------------------------------------------------------------
# TestVecEnvInterface
# ---------------------------------------------------------------------------

class TestVecEnvInterface:
    def test_num_envs(self):
        num_traj = 3
        env = create_sb3_env(num_trajectories=num_traj)
        assert env.num_envs == num_traj

    def test_env_is_wrapped(self):
        num_traj = 2
        env = create_sb3_env(num_trajectories=num_traj)
        import gymnasium as gym_mod
        result = env.env_is_wrapped(gym_mod.Wrapper)
        assert result == [False, False]

    def test_obs_space_is_gymnasium_box(self):
        env = create_sb3_env(num_trajectories=1)
        assert isinstance(env.observation_space, gymnasium.spaces.Box)

    def test_action_space_is_gymnasium_box(self):
        env = create_sb3_env(num_trajectories=1)
        assert isinstance(env.action_space, gymnasium.spaces.Box)
