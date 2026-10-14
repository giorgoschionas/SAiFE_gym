from typing import Any, List, Optional, Sequence, Type

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn

from SAiFE_gym.gym.lp_actions import (
    CategoricalLPActionCodec,
    _validate_action_table_args, _offset_grid, _half_width_grid,
    build_discrete_action_table, decode_policy_actions,
)


class DiscreteActionWrapper(gymnasium.ActionWrapper):
    """Expose discrete hold/rebalance actions for a batched ``AMMEnvironment``.

    The wrapped environment still receives its native batched 3-column action:
    ``[lower_offset, upper_offset, hold_flag]``. Returns retain the raw batched
    API; this action converter is not a standard single-environment adapter.
    """

    def __init__(self, env: gymnasium.Env, tau: int, tick_stride: int = 1):
        super().__init__(env)
        self.action_codec = CategoricalLPActionCodec(tau, tick_stride)
        self.action_table = self.action_codec.action_table
        self.num_actions = len(self.action_table)
        self.num_trajectories = int(getattr(env, "num_trajectories", 1))

        if self.num_trajectories == 1:
            self.action_space = gymnasium.spaces.Discrete(self.num_actions)
        else:
            self.action_space = gymnasium.spaces.MultiDiscrete(
                np.full(self.num_trajectories, self.num_actions, dtype=np.int64)
            )

    def action(self, action: int | np.ndarray) -> np.ndarray:
        indices = np.asarray(action)
        if self.num_trajectories == 1:
            indices = indices.reshape(-1)
            if indices.size != 1:
                raise ValueError(f"expected one action id, got shape {np.asarray(action).shape}")
            return self.action_codec.decode(indices[0]).reshape(1, 3)

        if indices.shape != (self.num_trajectories,):
            raise ValueError(
                f"expected action shape ({self.num_trajectories},), got {indices.shape}"
            )
        return self.action_codec.decode(indices)


class DiscreteActionVecEnv(VecEnv):
    """Expose a discrete action space for SB3 VecEnv AMM training."""

    def __init__(self, vec_env: VecEnv, tau: int, tick_stride: int = 1):
        if not isinstance(vec_env.action_space, gymnasium.spaces.Box):
            raise ValueError("Legacy action wrappers require a raw Box adapter (action_encoding=\"raw\")")
        self._wrapped = vec_env
        self.action_codec = CategoricalLPActionCodec(tau, tick_stride)
        self.action_table = self.action_codec.action_table
        self.num_actions = len(self.action_table)
        action_space = gymnasium.spaces.Discrete(self.num_actions)
        super().__init__(vec_env.num_envs, vec_env.observation_space, action_space)

    def unscale(self, actions: int | np.ndarray) -> np.ndarray:
        indices = np.asarray(actions)
        if indices.ndim == 0:
            indices = indices.reshape(1)
        return self.action_codec.decode(indices)

    def reset(self) -> VecEnvObs:
        return self._wrapped.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self._wrapped.step_async(self.unscale(actions))

    def step_wait(self) -> VecEnvStepReturn:
        return self._wrapped.step_wait()

    def close(self) -> None:
        self._wrapped.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self._wrapped.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._wrapped.set_attr(attr_name, value, indices)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        return self._wrapped.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type, indices: VecEnvIndices = None
    ) -> List[bool]:
        return self._wrapped.env_is_wrapped(wrapper_class, indices)

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        return self._wrapped.seed(seed)

    def get_images(self) -> Sequence[np.ndarray]:
        return self._wrapped.get_images()


class StructuredMultiDiscreteVecEnv(VecEnv):
    """Expose decoupled ``center``, ``half_width``, and ``hold`` action heads.

    With the default ``tick_stride=1``, the SB3-facing action space is
    ``MultiDiscrete([2*tau + 1, tau, 2])``. Larger strides expose a coarser
    center/half-width grid while preserving the same internal
    ``[lower_offset, upper_offset, hold_flag]`` command format.
    """

    def __init__(self, vec_env: VecEnv, tau: int, tick_stride: int = 1):
        if not isinstance(vec_env.action_space, gymnasium.spaces.Box):
            raise ValueError("Legacy action wrappers require a raw Box adapter (action_encoding=\"raw\")")
        self._wrapped = vec_env
        self.tau, self.tick_stride = _validate_action_table_args(tau, tick_stride)
        self.center_offsets = _offset_grid(self.tau, self.tick_stride)
        self.half_widths = _half_width_grid(self.tau, self.tick_stride)
        action_space = gymnasium.spaces.MultiDiscrete([
            len(self.center_offsets),
            len(self.half_widths),
            2,
        ])
        super().__init__(vec_env.num_envs, vec_env.observation_space, action_space)

    def unscale(self, actions: np.ndarray) -> np.ndarray:
        """Map MultiDiscrete indices to ``[lower, upper, hold_flag]`` actions."""
        return decode_policy_actions(actions, self.action_space, self.tau, self.tick_stride)

    def reset(self) -> VecEnvObs:
        return self._wrapped.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self._wrapped.step_async(self.unscale(actions))

    def step_wait(self) -> VecEnvStepReturn:
        return self._wrapped.step_wait()

    def close(self) -> None:
        self._wrapped.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self._wrapped.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._wrapped.set_attr(attr_name, value, indices)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        return self._wrapped.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type, indices: VecEnvIndices = None
    ) -> List[bool]:
        return self._wrapped.env_is_wrapped(wrapper_class, indices)

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        return self._wrapped.seed(seed)

    def get_images(self) -> Sequence[np.ndarray]:
        return self._wrapped.get_images()
