from typing import Any, List, Optional, Sequence, Type

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn


def _validate_action_table_args(tau: int, tick_stride: int) -> tuple[int, int]:
    tau = int(tau)
    tick_stride = int(tick_stride)
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got {tau}")
    if tick_stride < 1:
        raise ValueError(f"tick_stride must be >= 1, got {tick_stride}")
    return tau, tick_stride


def _offset_grid(tau: int, tick_stride: int) -> np.ndarray:
    offsets = list(range(-tau, tau + 1, tick_stride))
    if offsets[-1] != tau:
        offsets.append(tau)
    return np.asarray(offsets, dtype=np.float32)


def build_discrete_action_table(tau: int, tick_stride: int = 1) -> np.ndarray:
    """Map discrete action ids to internal ``[lower, upper, hold_flag]`` actions.

    Action id 0 is the dedicated hold/no-op action. All remaining rows rebalance
    into valid ``lower_offset < upper_offset`` ranges sampled from the tick grid.
    """
    tau, tick_stride = _validate_action_table_args(tau, tick_stride)
    actions = [[0.0, 1.0, 1.0]]
    offsets = _offset_grid(tau, tick_stride)

    for i, lower in enumerate(offsets):
        for upper in offsets[i + 1:]:
            actions.append([lower, upper, -1.0])

    return np.asarray(actions, dtype=np.float32)


class DiscreteActionWrapper(gymnasium.ActionWrapper):
    """Expose discrete hold/rebalance actions for a batched ``AMMEnvironment``.

    The wrapped environment still receives its native batched 3-column action:
    ``[lower_offset, upper_offset, hold_flag]``.
    """

    def __init__(self, env: gymnasium.Env, tau: int, tick_stride: int = 1):
        super().__init__(env)
        self.action_table = build_discrete_action_table(tau, tick_stride)
        self.num_actions = len(self.action_table)
        self.num_trajectories = int(getattr(env, "num_trajectories", 1))

        if self.num_trajectories == 1:
            self.action_space = gymnasium.spaces.Discrete(self.num_actions)
        else:
            self.action_space = gymnasium.spaces.MultiDiscrete(
                np.full(self.num_trajectories, self.num_actions, dtype=np.int64)
            )

    def action(self, action: int | np.ndarray) -> np.ndarray:
        indices = np.asarray(action, dtype=np.int64)
        if self.num_trajectories == 1:
            indices = indices.reshape(-1)
            if indices.size != 1:
                raise ValueError(f"expected one action id, got shape {np.asarray(action).shape}")
            return self.action_table[indices[0]].reshape(1, 3).astype(np.float32)

        if indices.shape != (self.num_trajectories,):
            raise ValueError(
                f"expected action shape ({self.num_trajectories},), got {indices.shape}"
            )
        return self.action_table[indices].astype(np.float32)


class DiscreteActionVecEnv(VecEnv):
    """Expose a discrete action space for SB3 VecEnv AMM training."""

    def __init__(self, vec_env: VecEnv, tau: int, tick_stride: int = 1):
        self._wrapped = vec_env
        self.action_table = build_discrete_action_table(tau, tick_stride)
        self.num_actions = len(self.action_table)
        action_space = gymnasium.spaces.Discrete(self.num_actions)
        super().__init__(vec_env.num_envs, vec_env.observation_space, action_space)

    def unscale(self, actions: int | np.ndarray) -> np.ndarray:
        indices = np.asarray(actions, dtype=np.int64)
        if indices.ndim == 0:
            indices = indices.reshape(1)
        return self.action_table[indices].astype(np.float32)

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

    The SB3-facing action space is ``MultiDiscrete([2*tau + 1, tau, 2])``:
    center tick in ``[-tau, tau]``, half-width in ``[1, tau]``, and a binary
    rebalance/hold choice. Actions are mapped to the wrapped env's internal
    ``[lower_offset, upper_offset, hold_flag]`` command format.
    """

    def __init__(self, vec_env: VecEnv, tau: int):
        self._wrapped = vec_env
        self.tau = int(tau)
        if self.tau < 1:
            raise ValueError(f"tau must be >= 1, got {self.tau}")
        action_space = gymnasium.spaces.MultiDiscrete([2 * self.tau + 1, self.tau, 2])
        super().__init__(vec_env.num_envs, vec_env.observation_space, action_space)

    def unscale(self, actions: np.ndarray) -> np.ndarray:
        """Map MultiDiscrete indices to ``[lower, upper, hold_flag]`` actions."""
        actions = np.asarray(actions, dtype=np.int64)
        center = actions[..., 0] - self.tau
        half_width = actions[..., 1] + 1
        hold_flag = np.where(actions[..., 2] == 0, -1.0, 1.0).astype(np.float32)

        lower = np.clip(center - half_width, -self.tau, self.tau - 1).astype(np.float32)
        upper = np.clip(center + half_width, -self.tau + 1, self.tau).astype(np.float32)
        return np.stack([lower, upper, hold_flag], axis=-1)

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
