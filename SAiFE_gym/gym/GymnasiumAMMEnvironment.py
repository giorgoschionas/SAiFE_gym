"""Gymnasium interfaces for the native batched LP simulator.

These adapters preserve the full state dictionary and simulation precision.
Stable-Baselines3 users should keep using StableBaselinesAMMEnvironment, whose
feature selection and autoreset contract are different.
"""

from copy import deepcopy

import gymnasium
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.domain_randomization import DomainRandomizedAMMEnvironment


def _single_info(info: dict) -> dict:
    """Copy diagnostics and remove the single trajectory dimension recursively."""
    result = {}
    for key, value in info.items():
        if isinstance(value, dict):
            result[key] = _single_info(value)
        elif isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] == 1:
                value = value[0]
            result[key] = value.item() if value.ndim == 0 else value.copy()
        else:
            result[key] = deepcopy(value)
    return result


def _vector_info(info: dict, num_envs: int) -> dict:
    """Copy native diagnostics into Gymnasium's batched values and masks."""
    result = {}
    for key, value in info.items():
        if isinstance(value, dict):
            result[key] = _vector_info(value, num_envs)
        else:
            array = np.asarray(deepcopy(value))
            if array.ndim == 0 or array.shape[0] != num_envs:
                array = np.broadcast_to(array, (num_envs, *array.shape))
            result[key] = array.copy()
        result[f"_{key}"] = np.ones(num_envs, dtype=bool)
    return result


def _action_array(action, expected_shape: tuple[int, ...]) -> np.ndarray:
    action = np.asarray(action)
    if action.shape != expected_shape:
        raise ValueError(f"expected action shape {expected_shape}, got {action.shape}")
    return action


class GymnasiumAMMEnvironment(gymnasium.Env):
    """Expose one AMM trajectory as a standard Gymnasium environment.

    Wrap an AMMEnvironment (optionally domain-randomized) with exactly one
    trajectory. Scalar observation fields are zero-dimensional arrays; per-tick
    fields have shape ``(num_ticks,)``. Actions have shape ``(3,)``. Reward and
    termination flags are Python scalars. Call reset after an episode ends.
    """

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, amm_env: AMMEnvironment | DomainRandomizedAMMEnvironment):
        super().__init__()
        if amm_env.num_trajectories != 1:
            raise ValueError("GymnasiumAMMEnvironment requires exactly one trajectory")
        self.amm_env = amm_env
        self.observation_space = deepcopy(amm_env.single_observation_space)
        self.action_space = deepcopy(amm_env.single_action_space)
        self._needs_reset = True

    @staticmethod
    def _observation(state: dict) -> dict:
        return {key: np.asarray(value[0]).copy() for key, value in state.items()}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        state, info = self.amm_env.reset(seed=seed, options=options)
        self._needs_reset = False
        return self._observation(state), _single_info(info)

    def step(self, action):
        if self._needs_reset:
            raise gymnasium.error.ResetNeeded("Call reset before starting a new episode")
        action = _action_array(action, self.action_space.shape)
        state, rewards, terminated, truncated, info = self.amm_env.step(action[None, :])
        terminated, truncated = bool(terminated[0]), bool(truncated[0])
        self._needs_reset = terminated or truncated
        return (
            self._observation(state), float(rewards[0]), terminated, truncated,
            _single_info(info),
        )

    def render(self):
        return None

    def close(self):
        self._needs_reset = True
        self.amm_env.close()


class GymnasiumAMMVectorEnv(VectorEnv):
    """Expose the native trajectory batch through Gymnasium's VectorEnv API.

    Uses next-step autoreset: terminal observations are returned on the terminal
    step, and the next step resets the batch without executing its actions.
    All trajectories share a horizon and RNG streams, so only whole-batch
    resets and a single integer seed (or None) are supported. For independent
    resets/seeds, vectorize separate GymnasiumAMMEnvironment instances instead.
    """

    metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.NEXT_STEP}
    render_mode = None

    def __init__(self, amm_env: AMMEnvironment | DomainRandomizedAMMEnvironment):
        self.closed = True
        self.amm_env = amm_env
        self.num_envs = amm_env.num_trajectories
        self.single_observation_space = deepcopy(amm_env.single_observation_space)
        self.single_action_space = deepcopy(amm_env.single_action_space)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self._needs_reset = True
        self._autoreset_pending = False
        self.closed = False

    @staticmethod
    def _observation(state: dict) -> dict:
        return {key: value.copy() for key, value in state.items()}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if isinstance(seed, (list, tuple, np.ndarray)):
            raise NotImplementedError(
                "Per-trajectory seeds are not supported; use one integer seed for the batch"
            )
        if options is not None:
            options = dict(options)
            if "reset_mask" in options:
                mask = options.pop("reset_mask")
                if (
                    not isinstance(mask, np.ndarray)
                    or mask.dtype != np.bool_
                    or mask.shape != (self.num_envs,)
                ):
                    raise ValueError(
                        f"reset_mask must be a boolean array of shape ({self.num_envs},)"
                    )
                if not mask.all():
                    raise NotImplementedError("Only whole-batch resets are supported")
        super().reset(seed=seed)
        state, info = self.amm_env.reset(seed=seed, options=options)
        self._needs_reset = False
        self._autoreset_pending = False
        return self._observation(state), _vector_info(info, self.num_envs)

    def step(self, actions):
        if self._needs_reset:
            raise gymnasium.error.ResetNeeded("Call reset before stepping the environment")
        actions = _action_array(actions, self.action_space.shape)
        if self._autoreset_pending:
            obs, info = self.reset()
            return (
                obs, np.zeros(self.num_envs, dtype=np.float64),
                np.zeros(self.num_envs, dtype=bool), np.zeros(self.num_envs, dtype=bool),
                info,
            )

        state, rewards, terminated, truncated, info = self.amm_env.step(actions)
        dones = terminated | truncated
        if dones.any() and not dones.all():
            self._needs_reset = True
            raise NotImplementedError("AMM trajectories must finish their episodes together")
        self._autoreset_pending = bool(dones.all())
        return (
            self._observation(state), rewards.copy(), terminated.copy(), truncated.copy(),
            _vector_info(info, self.num_envs),
        )

    def render(self):
        return None

    def close_extras(self, **kwargs):
        self._needs_reset = True
        self.amm_env.close()
