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
    BOUNDARY_PROXIMITY_KEY,
    FEES0_KEY,
    LP_LOWER_OFFSET_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_UPPER_OFFSET_KEY,
    MISPRICING_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_SQRT_PRICE_KEY,
    POSITION_WIDTH_KEY,
    RECENT_REALIZED_VOLATILITY_KEY,
    TIME_KEY,
)
from SAiFE_gym.gym.observation_features import (
    SB3_DERIVED_OBS_KEYS,
    compute_sb3_observation_features,
)
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_amm_env(
    num_trajectories: int = 1,
    n_steps: int = 5,
    tau: int = 5,
    realized_vol_window: int = 50,
) -> AMMEnvironment:
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
        realized_vol_window=realized_vol_window,
        seed=42,
    )


def create_sb3_env(
    num_trajectories: int = 1,
    n_steps: int = 5,
    obs_keys=None,
    store_terminal_observation_info: bool = True,
    realized_vol_window: int = 50,
) -> StableBaselinesAMMEnvironment:
    amm_env = create_test_amm_env(
        num_trajectories=num_trajectories,
        n_steps=n_steps,
        realized_vol_window=realized_vol_window,
    )
    return StableBaselinesAMMEnvironment(
        amm_env,
        obs_keys=obs_keys,
        store_terminal_observation_info=store_terminal_observation_info,
    )


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_default_obs_keys_include_directional_offsets(self):
        assert LP_LOWER_OFFSET_KEY in DEFAULT_OBS_KEYS
        assert LP_UPPER_OFFSET_KEY in DEFAULT_OBS_KEYS
        assert RECENT_REALIZED_VOLATILITY_KEY in DEFAULT_OBS_KEYS
        assert DEFAULT_OBS_KEYS.index(RECENT_REALIZED_VOLATILITY_KEY) < DEFAULT_OBS_KEYS.index(
            LP_LOWER_OFFSET_KEY
        )
        assert DEFAULT_OBS_KEYS.index(LP_LOWER_OFFSET_KEY) < DEFAULT_OBS_KEYS.index(
            BOUNDARY_PROXIMITY_KEY
        )
        assert DEFAULT_OBS_KEYS.index(LP_UPPER_OFFSET_KEY) < DEFAULT_OBS_KEYS.index(
            BOUNDARY_PROXIMITY_KEY
        )

    def test_shape_single_trajectory(self):
        env = create_sb3_env(num_trajectories=1)
        obs = env.reset()
        assert obs.shape == (1, len(DEFAULT_OBS_KEYS))

    def test_shape_multi_trajectory(self):
        env = create_sb3_env(num_trajectories=3)
        obs = env.reset()
        assert obs.shape == (3, len(DEFAULT_OBS_KEYS))

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
        # action_space.sample() returns shape (3,); replicate for all trajectories
        actions = np.tile(action, (env.num_trajectories, 1))
        env.step_async(actions)
        return env.step_wait()

    def test_obs_shape(self):
        num_traj = 2
        env = create_sb3_env(num_trajectories=num_traj)
        obs, _, _, _ = self._step(env)
        assert obs.shape == (num_traj, len(DEFAULT_OBS_KEYS))

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
# TestRawInfo
# ---------------------------------------------------------------------------

class TestRawInfo:
    INFO_KEYS = {"asset_price", "pool_price", "time", "action", "reward"}

    def test_batched_info_contains_post_step_values(self):
        env = create_test_amm_env(num_trajectories=2, n_steps=5)
        env.reset()
        action = np.array([[-2.0, 2.0, -1.0], [0.0, 1.0, 1.0]], dtype=np.float32)

        state, rewards, _, _, info = env.step(action)

        assert set(info) == self.INFO_KEYS
        assert info["asset_price"].shape == (2,)
        assert info["pool_price"].shape == (2,)
        assert info["time"].shape == (2,)
        assert info["action"].shape == (2, 3)
        assert info["reward"].shape == (2,)
        np.testing.assert_allclose(info["asset_price"], state[ASSET_PRICE_KEY])
        np.testing.assert_allclose(info["pool_price"], state[POOL_SQRT_PRICE_KEY] ** 2)
        np.testing.assert_allclose(info["time"], state[TIME_KEY])
        np.testing.assert_allclose(info["action"], action)
        np.testing.assert_allclose(info["reward"], rewards)

    def test_batched_info_copies_mutable_arrays(self):
        env = create_test_amm_env(num_trajectories=1, n_steps=5)
        env.reset()
        action = np.array([[-2.0, 2.0, -1.0]], dtype=np.float32)

        _, rewards, _, _, info = env.step(action)
        action[0, 0] = 99.0
        rewards[0] = 99.0

        assert info["action"][0, 0] == -2.0
        assert info["reward"][0] != 99.0


# ---------------------------------------------------------------------------
# TestInfosList
# ---------------------------------------------------------------------------

class TestInfosList:
    def _get_infos(self, env: StableBaselinesAMMEnvironment):
        env.reset()
        actions = np.zeros((env.num_trajectories, 3), dtype=np.float32)
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

    def test_contains_per_trajectory_step_info(self):
        env = create_sb3_env(num_trajectories=2)
        env.reset()
        actions = np.array([[-2.0, 2.0, -1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
        env.step_async(actions)

        _, rewards, _, infos = env.step_wait()

        for i, info in enumerate(infos):
            assert {"asset_price", "pool_price", "time", "action", "reward"} <= set(info)
            assert np.isscalar(info["asset_price"])
            assert np.isscalar(info["pool_price"])
            assert np.isscalar(info["time"])
            assert info["action"].shape == (3,)
            np.testing.assert_allclose(info["action"], actions[i])
            np.testing.assert_allclose(info["reward"], rewards[i])


# ---------------------------------------------------------------------------
# TestAutoReset
# ---------------------------------------------------------------------------

class TestAutoReset:
    def _run_to_done(self, env: StableBaselinesAMMEnvironment):
        """Step until dones.all() is True, return final (obs, dones)."""
        env.reset()
        actions = np.zeros((env.num_trajectories, 3), dtype=np.float32)
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
        assert obs.shape == (num_traj, len(DEFAULT_OBS_KEYS))


# ---------------------------------------------------------------------------
# TestTerminalObs
# ---------------------------------------------------------------------------

class TestTerminalObs:
    def _run_episode(self, env: StableBaselinesAMMEnvironment):
        env.reset()
        actions = np.zeros((env.num_trajectories, 3), dtype=np.float32)
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
        assert final_infos[0]["terminal_observation"].shape == (len(DEFAULT_OBS_KEYS),)

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

    def test_step_info_present_on_done(self):
        env = create_sb3_env(num_trajectories=1, n_steps=5)
        _, final_infos = self._run_episode(env)
        assert final_infos is not None
        assert {"asset_price", "pool_price", "time", "action", "reward"} <= set(final_infos[0])
        assert "terminal_observation" in final_infos[0]


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
# TestObservationFeatures
# ---------------------------------------------------------------------------

class TestObservationFeatures:
    def test_compute_sb3_observation_features(self):
        env = create_test_amm_env(num_trajectories=2)
        state, _ = env.reset()

        features = compute_sb3_observation_features(state)
        lower_offset = state[POOL_CURRENT_TICK_KEY] - state[LP_TICK_LOWER_KEY]
        upper_offset = state[LP_TICK_UPPER_KEY] - state[POOL_CURRENT_TICK_KEY]
        pool_price = state[POOL_SQRT_PRICE_KEY] ** 2

        assert set(features) == SB3_DERIVED_OBS_KEYS
        expected_mispricing = (
            (state[ASSET_PRICE_KEY] - pool_price)
            / np.maximum(state[ASSET_PRICE_KEY], 1e-12)
        )

        np.testing.assert_allclose(features[MISPRICING_KEY], expected_mispricing)
        np.testing.assert_allclose(features[POOL_SQRT_PRICE_KEY], pool_price)
        np.testing.assert_allclose(features[LP_LOWER_OFFSET_KEY], lower_offset)
        np.testing.assert_allclose(features[LP_UPPER_OFFSET_KEY], upper_offset)
        np.testing.assert_allclose(
            features[BOUNDARY_PROXIMITY_KEY],
            np.minimum(lower_offset, upper_offset),
        )
        np.testing.assert_allclose(features[POSITION_WIDTH_KEY], lower_offset + upper_offset)


# ---------------------------------------------------------------------------
# TestRecentRealizedVolatility
# ---------------------------------------------------------------------------

class TestRecentRealizedVolatility:
    def test_reset_and_first_return_are_zero(self):
        env = create_test_amm_env(
            num_trajectories=2,
            n_steps=4,
            realized_vol_window=2,
        )
        state, _ = env.reset()

        np.testing.assert_array_equal(
            state[RECENT_REALIZED_VOLATILITY_KEY],
            np.zeros(2),
        )

        env.state[ASSET_PRICE_KEY] = np.array([101.0, 102.0])
        env._update_realized_volatility()

        np.testing.assert_array_equal(
            env.state[RECENT_REALIZED_VOLATILITY_KEY],
            np.zeros(2),
        )

    def test_expanding_and_rolling_unit_horizon_volatility(self):
        env = create_test_amm_env(
            num_trajectories=2,
            n_steps=4,
            realized_vol_window=2,
        )
        env.reset()
        step_size = env.step_size

        prices = [
            np.array([101.0, 102.0]),
            np.array([103.0, 101.0]),
            np.array([104.0, 105.0]),
        ]
        returns = []
        previous = np.array([100.0, 100.0])

        for price in prices:
            env.state[ASSET_PRICE_KEY] = price
            env._update_realized_volatility()
            returns.append(np.log(price / previous))
            previous = price

        expected = (
            np.std(np.asarray(returns[-2:]), axis=0, ddof=1)
            / np.sqrt(step_size)
        )
        np.testing.assert_allclose(
            env.state[RECENT_REALIZED_VOLATILITY_KEY],
            expected,
        )

    def test_flatten_obs_includes_recent_realized_volatility(self):
        env = create_sb3_env(num_trajectories=2)
        env.reset()
        state = env.env.state
        state[RECENT_REALIZED_VOLATILITY_KEY] = np.array([0.03, 0.05])

        obs = env._flatten_obs(state)

        col = env.obs_keys.index(RECENT_REALIZED_VOLATILITY_KEY)
        np.testing.assert_allclose(obs[:, col], np.array([0.03, 0.05], dtype=np.float32))


# ---------------------------------------------------------------------------
# TestMultiTrajectory
# ---------------------------------------------------------------------------

class TestMultiTrajectory:
    def test_four_trajectories_full_episode(self):
        num_traj = 4
        n_steps = 5
        env = create_sb3_env(num_trajectories=num_traj, n_steps=n_steps)
        obs = env.reset()
        assert obs.shape == (num_traj, len(DEFAULT_OBS_KEYS))

        actions = np.zeros((num_traj, 3), dtype=np.float32)
        for _ in range(n_steps):
            env.step_async(actions)
            obs, rewards, dones, infos = env.step_wait()
            assert obs.shape == (num_traj, len(DEFAULT_OBS_KEYS))
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

    def test_get_attr_respects_indices(self):
        env = create_sb3_env(num_trajectories=3, n_steps=5)

        assert env.get_attr("n_steps", indices=1) == [5]
        assert env.get_attr("n_steps", indices=[2, 0]) == [5, 5]
        assert env.get_attr("render_mode", indices=[1]) == [None]

    def test_env_is_wrapped_respects_indices(self):
        env = create_sb3_env(num_trajectories=3)
        import gymnasium as gym_mod

        assert env.env_is_wrapped(gym_mod.Wrapper, indices=2) == [False]
        assert env.env_is_wrapped(gym_mod.Wrapper, indices=[2, 0]) == [False, False]

    def test_set_attr_allows_full_selection(self):
        env = create_sb3_env(num_trajectories=2)

        env.set_attr("_test_shared_attr", "value")
        assert env.env._test_shared_attr == "value"

        env.set_attr("_test_shared_attr", "updated", indices=[0, 1])
        assert env.env._test_shared_attr == "updated"

    def test_set_attr_rejects_partial_indices(self):
        env = create_sb3_env(num_trajectories=2)

        with pytest.raises(NotImplementedError, match="Partial set_attr"):
            env.set_attr("_test_shared_attr", "value", indices=0)
        assert not hasattr(env.env, "_test_shared_attr")

    def test_env_method_allows_full_selection(self):
        env = create_sb3_env(num_trajectories=2)

        result = env.env_method("seed", 123)

        assert result == [None, None]

    def test_env_method_rejects_partial_indices(self):
        env = create_sb3_env(num_trajectories=2)

        with pytest.raises(NotImplementedError, match="Partial env_method"):
            env.env_method("seed", 123, indices=0)

    def test_obs_space_is_gymnasium_box(self):
        env = create_sb3_env(num_trajectories=1)
        assert isinstance(env.observation_space, gymnasium.spaces.Box)

    def test_action_space_is_gymnasium_box(self):
        env = create_sb3_env(num_trajectories=1)
        assert isinstance(env.action_space, gymnasium.spaces.Box)


# ---------------------------------------------------------------------------
# TestRelativeObsKeys
# ---------------------------------------------------------------------------

class TestRelativeObsKeys:
    RAW_OFFSET_OBS_KEYS = [
        MISPRICING_KEY,
        LP_LOWER_OFFSET_KEY,
        LP_UPPER_OFFSET_KEY,
        TIME_KEY,
    ]

    def _get_state_and_obs(self, env: StableBaselinesAMMEnvironment):
        """Reset env and return (raw_state_dict, flat_obs)."""
        obs = env.reset()
        state = env.env.model_dynamics.state
        return state, obs

    def test_mispricing_value(self):
        """mispricing = (asset_price - sqrt_price²) / asset_price"""
        env = create_sb3_env(num_trajectories=1)
        state, obs = self._get_state_and_obs(env)

        pool_price = state[POOL_SQRT_PRICE_KEY] ** 2
        expected = (
            (state[ASSET_PRICE_KEY] - pool_price)
            / np.maximum(state[ASSET_PRICE_KEY], 1e-12)
        )
        obs_keys = env.obs_keys
        mispricing_col = obs_keys.index(MISPRICING_KEY)
        np.testing.assert_allclose(obs[:, mispricing_col], expected.astype(np.float32), rtol=1e-5)

    def test_lp_lower_offset_value(self):
        """lp_lower_offset = current_tick - lp_tick_lower"""
        env = create_sb3_env(num_trajectories=1, obs_keys=self.RAW_OFFSET_OBS_KEYS)
        state, obs = self._get_state_and_obs(env)

        expected = state[POOL_CURRENT_TICK_KEY] - state[LP_TICK_LOWER_KEY]
        obs_keys = env.obs_keys
        col = obs_keys.index(LP_LOWER_OFFSET_KEY)
        np.testing.assert_allclose(obs[:, col], expected.astype(np.float32), rtol=1e-5)

    def test_lp_upper_offset_value(self):
        """lp_upper_offset = lp_tick_upper - current_tick"""
        env = create_sb3_env(num_trajectories=1, obs_keys=self.RAW_OFFSET_OBS_KEYS)
        state, obs = self._get_state_and_obs(env)

        expected = state[LP_TICK_UPPER_KEY] - state[POOL_CURRENT_TICK_KEY]
        obs_keys = env.obs_keys
        col = obs_keys.index(LP_UPPER_OFFSET_KEY)
        np.testing.assert_allclose(obs[:, col], expected.astype(np.float32), rtol=1e-5)

    def test_offsets_non_negative_when_in_range(self):
        """After reset with default uniform LP position, both offsets should be ≥ 0."""
        env = create_sb3_env(num_trajectories=2, obs_keys=self.RAW_OFFSET_OBS_KEYS)
        state, obs = self._get_state_and_obs(env)

        lower_col = env.obs_keys.index(LP_LOWER_OFFSET_KEY)
        upper_col = env.obs_keys.index(LP_UPPER_OFFSET_KEY)
        assert (obs[:, lower_col] >= 0).all(), "lp_lower_offset should be ≥ 0 when in-range"
        assert (obs[:, upper_col] >= 0).all(), "lp_upper_offset should be ≥ 0 when in-range"

    def test_offsets_change_after_rebalance(self):
        """After a step that rebalances to a new tick range, offsets reflect new position."""
        env = create_sb3_env(num_trajectories=1, obs_keys=self.RAW_OFFSET_OBS_KEYS)
        env.reset()

        # Step with action [lower_offset=-3, upper_offset=3]
        actions = np.array([[-3.0, 3.0, -1.0]], dtype=np.float32)
        env.step_async(actions)
        obs, _, _, _ = env.step_wait()

        state = env.env.model_dynamics.state
        lower_col = env.obs_keys.index(LP_LOWER_OFFSET_KEY)
        upper_col = env.obs_keys.index(LP_UPPER_OFFSET_KEY)

        expected_lower = (state[POOL_CURRENT_TICK_KEY] - state[LP_TICK_LOWER_KEY]).astype(np.float32)
        expected_upper = (state[LP_TICK_UPPER_KEY] - state[POOL_CURRENT_TICK_KEY]).astype(np.float32)
        np.testing.assert_allclose(obs[:, lower_col], expected_lower, rtol=1e-5)
        np.testing.assert_allclose(obs[:, upper_col], expected_upper, rtol=1e-5)

    def test_derived_keys_individually(self):
        """Single-key obs_keys work for each derived key."""
        for key in [MISPRICING_KEY, LP_LOWER_OFFSET_KEY, LP_UPPER_OFFSET_KEY]:
            env = create_sb3_env(num_trajectories=1, obs_keys=[key])
            obs = env.reset()
            assert obs.shape == (1, 1), f"Expected (1,1) for key '{key}', got {obs.shape}"
            assert obs.dtype == np.float32
