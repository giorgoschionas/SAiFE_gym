"""Public Gymnasium contracts and compatibility with the native simulator."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env, data_equivalence
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.GymnasiumAMMEnvironment import (
    GymnasiumAMMEnvironment,
    GymnasiumAMMVectorEnv,
)
from SAiFE_gym.gym.domain_randomization import (
    DomainRandomizedAMMEnvironment,
    UniformDomainRandomizationConfig,
)
from SAiFE_gym.gym.index_names import (
    GAS_COST_KEY,
    INITIAL_WEALTH_KEY,
    LP_EVER_DEPLOYED_KEY,
    LP_FEE_SNAPSHOT0_KEY,
    LP_FEE_SNAPSHOT1_KEY,
    LP_LIQUIDITY_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    TIME_KEY,
)


def _make_raw(n=1, n_steps=5):
    return AMMEnvironment(num_trajectories=n, n_steps=n_steps, seed=7)


def _make_randomized(n=1, num_domains=1):
    return DomainRandomizedAMMEnvironment(
        _make_raw(n),
        UniformDomainRandomizationConfig(
            sigma_range=(1.0, 2.0), arrival_rate_range=(20.0, 40.0),
        ),
        num_domains=num_domains,
        seed=7,
    )


def _assert_observation(space, obs):
    assert space.contains(obs)
    assert set(space.spaces) == set(obs)
    assert len(obs) == 22
    assert {
        GAS_COST_KEY, INITIAL_WEALTH_KEY, LP_EVER_DEPLOYED_KEY,
        LP_FEE_SNAPSHOT0_KEY, LP_FEE_SNAPSHOT1_KEY,
    } <= obs.keys()
    tick_keys = {POOL_CURRENT_TICK_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY}
    for key, value in obs.items():
        assert isinstance(value, np.ndarray)
        assert value.shape == space[key].shape
        expected_dtype = (
            np.int64 if key in tick_keys
            else np.bool_ if key == LP_EVER_DEPLOYED_KEY
            else np.float64
        )
        assert value.dtype == space[key].dtype == np.dtype(expected_dtype)


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("n_steps", [5, 200])
def test_raw_observation_membership_through_terminal_step(n, n_steps):
    env = _make_raw(n, n_steps)
    obs, _ = env.reset(seed=0)
    _assert_observation(env.observation_space, obs)
    for step in range(n_steps):
        actions = np.tile([-2.0, 2.0, 1.0], (n, 1))
        actions[:, 2] = np.where((step + np.arange(n)) % 3 == 0, -1.0, 1.0)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        _assert_observation(env.observation_space, obs)
        assert rewards.shape == terminated.shape == truncated.shape == (n,)
        assert terminated.dtype == truncated.dtype == np.bool_
        np.testing.assert_array_equal(terminated, step == n_steps - 1)
        assert not truncated.any()
    if n_steps == 200:
        assert np.all(obs[TIME_KEY] > env.terminal_time)
    env.close()


@pytest.mark.parametrize("n", [1, 3])
def test_raw_sampled_actions_and_legacy_batched_actions(n):
    env = _make_raw(n)
    assert env.single_action_space is env.action_space
    assert env.action_space.shape == (3,)
    assert env.batched_action_space.shape == (n, 3)
    env.reset(seed=0)
    action = env.action_space.sample() if n == 1 else env.batched_action_space.sample()
    obs, rewards, _, _, info = env.step(action)
    _assert_observation(env.observation_space, obs)
    assert rewards.shape == (n,)
    assert info["action"].shape == (n, 3)
    obs, *_ = env.step(np.tile([-2.0, 2.0], (n, 1)))
    _assert_observation(env.observation_space, obs)
    obs, *_ = env.step(None)
    _assert_observation(env.observation_space, obs)


@pytest.mark.parametrize("shape", [(3,), (1, 3), (3, 1), (3, 4)])
def test_raw_rejects_wrong_batch_shape_before_changing_state(shape):
    env = _make_raw(3)
    obs, _ = env.reset(seed=0)
    before = deepcopy(obs)
    with pytest.raises(ValueError, match="batched action shape"):
        env.step(np.zeros(shape))
    assert data_equivalence(before, env.state, exact=True)


@pytest.mark.filterwarnings("ignore:.*Box.*:UserWarning")
@pytest.mark.parametrize("randomized", [False, True])
def test_single_adapter_passes_gymnasium_checker(randomized):
    env = GymnasiumAMMEnvironment(_make_randomized() if randomized else _make_raw())
    check_env(env, skip_render_check=True)
    assert env.render() is None
    env.close()
    env.close()


def test_single_adapter_spaces_and_scalar_returns_match_raw():
    raw = _make_raw()
    reference = _make_raw()
    env = GymnasiumAMMEnvironment(raw)
    obs, info = env.reset(seed=0)
    expected, _ = reference.reset(seed=0)
    _assert_observation(env.observation_space, obs)
    assert obs[TIME_KEY].shape == ()
    assert obs[POOL_LIQUIDITY_ARRAY_KEY].shape == (raw.model_dynamics.num_ticks,)
    assert info == {}
    for key in obs:
        np.testing.assert_array_equal(obs[key], expected[key][0])
    action = np.array([-2.0, 2.0, -1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    expected, rewards, terms, truncs, raw_info = reference.step(action[None, :])
    _assert_observation(env.observation_space, obs)
    assert type(reward) is float
    assert type(terminated) is type(truncated) is bool
    assert reward == rewards[0]
    assert terminated == terms[0] and truncated == truncs[0]
    for key in obs:
        np.testing.assert_array_equal(obs[key], expected[key][0])
    assert info["action"].shape == (3,)
    assert type(info["time"]) is float
    for key in raw_info:
        np.testing.assert_array_equal(info[key], raw_info[key][0])


def test_single_adapter_requires_one_trajectory_and_explicit_resets():
    with pytest.raises(ValueError, match="exactly one trajectory"):
        GymnasiumAMMEnvironment(_make_raw(3))
    env = GymnasiumAMMEnvironment(_make_raw(n_steps=1))
    action = np.array([0.0, 1.0, 1.0])
    with pytest.raises(gym.error.ResetNeeded):
        env.step(action)
    env.reset()
    assert env.step(action)[2] is True
    with pytest.raises(gym.error.ResetNeeded):
        env.step(action)
    env.reset()
    assert env.step(action)[2] is True


def test_single_gymnasium_statistics_flattening_and_time_limit():
    env = gym.wrappers.RecordEpisodeStatistics(
        gym.wrappers.FlattenObservation(GymnasiumAMMEnvironment(_make_raw()))
    )
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    total = 0.0
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        assert env.observation_space.contains(obs)
    assert terminated and not truncated
    assert info["episode"]["r"] == pytest.approx(total)
    assert info["episode"]["l"] == 5
    env.close()

    limited = gym.wrappers.TimeLimit(GymnasiumAMMEnvironment(_make_raw()), 2)
    limited.reset(seed=0)
    limited.step(limited.action_space.sample())
    _, _, terminated, truncated, _ = limited.step(limited.action_space.sample())
    assert not terminated and truncated
    limited.close()


@pytest.mark.parametrize("n", [1, 3])
def test_vector_adapter_matches_raw_and_autoresets_on_next_step(n):
    env = GymnasiumAMMVectorEnv(_make_raw(n))
    reference = _make_raw(n)
    assert env.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP
    assert env.num_envs == n
    assert env.single_action_space.shape == (3,)
    assert env.action_space.shape == (n, 3)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(env.action_space.sample())
    obs, _ = env.reset(seed=0)
    reference.reset(seed=0)
    _assert_observation(env.observation_space, obs)
    _assert_observation(
        env.single_observation_space,
        {key: np.asarray(value[0]) for key, value in obs.items()},
    )
    for _ in range(5):
        actions = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(actions)
        expected, r, term, trunc, raw_info = reference.step(actions)
        _assert_observation(env.observation_space, obs)
        assert data_equivalence(obs, expected, exact=True)
        np.testing.assert_array_equal(rewards, r)
        np.testing.assert_array_equal(terminated, term)
        np.testing.assert_array_equal(truncated, trunc)
        assert rewards.shape == terminated.shape == truncated.shape == (n,)
        for key, value in raw_info.items():
            np.testing.assert_array_equal(info[key], value)
            np.testing.assert_array_equal(info[f"_{key}"], np.ones(n, dtype=bool))
    terminal_obs = deepcopy(obs)
    assert terminated.all()
    # A rebalance would deploy liquidity if the autoreset call executed actions.
    obs, rewards, terms, truncs, info = env.step(np.tile([-2.0, 2.0, -1.0], (n, 1)))
    expected, _ = reference.reset()
    assert data_equivalence(obs, expected, exact=True)
    np.testing.assert_array_equal(rewards, 0.0)
    assert not terms.any() and not truncs.any()
    assert info == {}
    assert terminated.all()
    np.testing.assert_array_equal(obs[LP_LIQUIDITY_KEY], 0.0)
    np.testing.assert_array_equal(terminal_obs[TIME_KEY], 1.0)
    obs, *_ = env.step(np.tile([-2.0, 2.0, -1.0], (n, 1)))
    assert np.all(obs[LP_LIQUIDITY_KEY] > 0.0)
    assert env.render() is None
    env.close()
    env.close()
    assert env.closed


@pytest.mark.parametrize("num_domains", [1, 3])
def test_vector_wrappers_domain_info_and_episode_statistics(num_domains):
    raw = _make_randomized(3, num_domains)
    adapter = GymnasiumAMMVectorEnv(raw)
    env = gym.wrappers.vector.DictInfoToList(
        gym.wrappers.vector.RecordEpisodeStatistics(
            gym.wrappers.vector.FlattenObservation(adapter)
        )
    )
    obs, infos = env.reset(seed=0)
    params = raw.last_domain_parameters.to_dict()
    for i, info in enumerate(infos):
        for key, value in params.items():
            np.testing.assert_array_equal(
                info["domain_parameters"][key], value if num_domains == 1 else value[i]
            )
    assert env.observation_space.contains(obs)
    for episode in range(2):
        if episode:
            obs, rewards, terms, truncs, infos = env.step(env.action_space.sample())
            assert not terms.any() and not truncs.any()
            np.testing.assert_array_equal(rewards, 0.0)
            assert all("domain_parameters" in info for info in infos)
        totals = np.zeros(3)
        for _ in range(5):
            obs, rewards, terms, truncs, infos = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            totals += rewards
        assert terms.all() and not truncs.any()
        for i, info in enumerate(infos):
            assert info["episode"]["r"] == pytest.approx(totals[i])
            assert info["episode"]["l"] == 5
    env.close()


@pytest.mark.parametrize("vector", [False, True])
def test_adapters_copy_observations_and_diagnostics(vector):
    raw = _make_raw(3 if vector else 1)
    env = GymnasiumAMMVectorEnv(raw) if vector else GymnasiumAMMEnvironment(raw)
    obs, _ = env.reset(seed=0)
    obs[POOL_LIQUIDITY_ARRAY_KEY].fill(-1)
    assert np.all(raw.state[POOL_LIQUIDITY_ARRAY_KEY] > 0)
    obs, _ = env.reset(seed=0)
    reset_snapshot = deepcopy(obs)
    step = env.step(env.action_space.sample())
    step_snapshot = deepcopy(step)
    env.step(env.action_space.sample())
    env.reset()
    assert data_equivalence(obs, reset_snapshot, exact=True)
    assert data_equivalence(step, step_snapshot, exact=True)


@pytest.mark.parametrize("vector", [False, True])
def test_adapters_replay_seed_zero_across_randomized_episodes(vector):
    raw = _make_randomized(3, 3) if vector else _make_randomized()
    env = GymnasiumAMMVectorEnv(raw) if vector else GymnasiumAMMEnvironment(raw)
    action = np.tile([-2.0, 2.0, -1.0], (3, 1)) if vector else np.array([-2.0, 2.0, -1.0])
    recordings = []
    for _ in range(2):
        recording = [env.reset(seed=0)]
        assert env.np_random_seed == 0
        recording.append(env.step(action))
        recording.append(env.reset())
        recording.append(env.step(action))
        recordings.append(recording)
    assert data_equivalence(recordings[0], recordings[1], exact=True)
    _, different_info = env.reset(seed=9)
    assert not data_equivalence(recordings[0][0][1], different_info, exact=True)


def test_vector_reset_validation_preserves_state_and_options():
    env = GymnasiumAMMVectorEnv(_make_randomized(3, 3))
    env.reset(seed=0)
    env.step(env.action_space.sample())
    before = deepcopy(env.amm_env.state)
    with pytest.raises(NotImplementedError, match="Per-trajectory seeds"):
        env.reset(seed=[1, 2, 3])
    with pytest.raises(NotImplementedError, match="whole-batch resets"):
        env.reset(options={"reset_mask": np.array([True, False, True])})
    with pytest.raises(ValueError, match="boolean array"):
        env.reset(options={"reset_mask": np.ones(3)})
    assert data_equivalence(before, env.amm_env.state, exact=True)
    options = {"reset_mask": np.ones(3, dtype=bool)}
    obs, _ = env.reset(options=options)
    assert "reset_mask" in options
    np.testing.assert_array_equal(options["reset_mask"], True)
    np.testing.assert_array_equal(obs[TIME_KEY], 0.0)


@pytest.mark.parametrize("vector,shape", [(False, (1, 3)), (False, (2,)), (True, (3,)), (True, (3, 2))])
def test_adapter_rejects_wrong_action_shape(vector, shape):
    raw = _make_raw(3 if vector else 1)
    env = GymnasiumAMMVectorEnv(raw) if vector else GymnasiumAMMEnvironment(raw)
    obs, _ = env.reset()
    with pytest.raises(ValueError, match="expected action shape"):
        env.step(np.zeros(shape))
    np.testing.assert_array_equal(raw.state[TIME_KEY], 0.0)


def test_separate_single_adapters_support_independent_gymnasium_resets():
    env = SyncVectorEnv([lambda: GymnasiumAMMEnvironment(_make_raw()) for _ in range(2)])
    env.reset(seed=[1, 2])
    obs, *_ = env.step(env.action_space.sample())
    second_obs = {key: value[1].copy() for key, value in obs.items()}
    obs, _ = env.reset(options={"reset_mask": np.array([True, False])})
    assert obs[TIME_KEY][0] == 0.0
    for key, value in second_obs.items():
        np.testing.assert_array_equal(obs[key][1], value)
    env.close()
