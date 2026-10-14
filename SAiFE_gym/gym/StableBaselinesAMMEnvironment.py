from typing import Any, List, Optional, Sequence, Type

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.lp_actions import CategoricalLPActionCodec, HOLD_COMMAND
from SAiFE_gym.gym.index_names import (
    FEES0_KEY,
    FEES1_KEY,
    HAS_POSITION_KEY,
    LP_LOWER_OFFSET_KEY,
    LP_UPPER_OFFSET_KEY,
    MISPRICING_KEY,
    PENDING_FEE_SHARE_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    PORTFOLIO_VALUE_RATIO_KEY,
    RELATIVE_GAS_COST_KEY,
    TIME_KEY,
)
from SAiFE_gym.gym.observation_features import (
    SB3_DERIVED_OBS_KEYS,
    compute_sb3_observation_features,
)

DEFAULT_OBS_KEYS = [
    MISPRICING_KEY,            # adverse selection signal
    LP_LOWER_OFFSET_KEY,       # distance from current tick to lower bound
    LP_UPPER_OFFSET_KEY,       # distance from current tick to upper bound
    # LP_LIQUIDITY_KEY,        # not directly actionable for hold/rebalance
    # LP_COLLECTED_FEES0_KEY,  # cumulative, not actionable
    # LP_COLLECTED_FEES1_KEY,  # cumulative, not actionable
    # ASSET_PRICE_KEY,         # nearly constant at low volatility; captured by mispricing
    TIME_KEY,                  # episode clock
    HAS_POSITION_KEY,          # disambiguates cash/undeployed from active LP ranges
    PORTFOLIO_VALUE_RATIO_KEY, # wealth scaled by initial capital
    PENDING_FEE_SHARE_KEY,     # pending fee value / total portfolio value
    RELATIVE_GAS_COST_KEY,     # total portfolio value / current rebalance gas cost
]  # obs_dim = 8

_ARRAY_KEYS = {POOL_LIQUIDITY_ARRAY_KEY, FEES0_KEY, FEES1_KEY}


class StableBaselinesAMMEnvironment(VecEnv):
    """
    Wraps AMMEnvironment to expose a stable-baselines3 VecEnv interface.

    Uses one categorical ID per LP decision by default (ID 0 holds).
    Set action_encoding="raw" for legacy Box commands.

    Responsibilities:
    - Flattens the Dict observation into a float32 array of shape (num_trajectories, obs_dim)
    - Implements auto-reset: when all trajectories are done, resets internally and
      stores the terminal observation in infos[i]["terminal_observation"]
    """

    def __init__(
        self,
        amm_env: AMMEnvironment,
        obs_keys: Optional[List[str]] = None,
        store_terminal_observation_info: bool = True,
        *,
        tick_stride: int = 1,
        action_encoding: str = "categorical",
    ):
        self.env = amm_env  # must be set before super().__init__() calls get_attr()
        self.obs_keys = obs_keys if obs_keys is not None else DEFAULT_OBS_KEYS
        self.obs_dim = len(self.obs_keys)
        self.store_terminal_observation_info = store_terminal_observation_info
        if action_encoding not in ("categorical", "raw"):
            raise ValueError("action_encoding must be 'categorical' or 'raw'")
        self.action_encoding = action_encoding
        self.tick_stride = tick_stride
        self.action_codec = (
            CategoricalLPActionCodec(amm_env.model_dynamics.tau, tick_stride)
            if action_encoding == "categorical" else None
        )
        self.actions = np.tile(HOLD_COMMAND, (amm_env.num_trajectories, 1))

        for k in self.obs_keys:
            if k in _ARRAY_KEYS:
                raise ValueError(
                    f"obs_keys may only contain scalar keys; '{k}' is an array key"
                )

        flat_obs_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        gymnasium_act_space = (
            self.action_codec.action_space if action_encoding == "categorical"
            else amm_env.command_space
        )
        super().__init__(amm_env.num_trajectories, flat_obs_space, gymnasium_act_space)

    # ------------------------------------------------------------------
    # Core VecEnv methods
    # ------------------------------------------------------------------

    def _flatten_obs(self, state_dict: dict) -> np.ndarray:
        """Return shape (num_trajectories, obs_dim) float32 array."""
        n = self.env.num_trajectories
        derived = compute_sb3_observation_features(state_dict)
        cols = [
            derived[k].reshape(n, 1) if k in SB3_DERIVED_OBS_KEYS
            else state_dict[k].reshape(n, 1)
            for k in self.obs_keys
        ]
        return np.concatenate(cols, axis=1).astype(np.float32)

    def _split_batched_info(self, batched_info: dict) -> List[dict]:
        """Convert AMMEnvironment's batched info dict to SB3's list-of-dicts."""
        n = self.env.num_trajectories
        infos = [{} for _ in range(n)]
        for key, value in batched_info.items():
            value_array = np.asarray(value)
            if value_array.ndim > 0 and value_array.shape[0] == n:
                for i, info in enumerate(infos):
                    item = value_array[i]
                    info[key] = item.copy() if isinstance(item, np.ndarray) else item.item()
            else:
                for info in infos:
                    info[key] = value
        return infos

    def reset(self) -> VecEnvObs:
        obs, _ = self.env.reset()
        return self._flatten_obs(obs)

    def decode_actions(self, actions):
        """Decode predictions to raw commands without advancing the simulator."""
        if self.action_encoding == "categorical":
            return self.action_codec.decode(actions)
        return np.asarray(actions)

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        if self.action_encoding == "categorical" and actions.shape != (self.num_envs,):
            raise ValueError(f"expected action IDs with shape ({self.num_envs},), got {actions.shape}")
        self.actions = self.decode_actions(actions)

    def step_wait(self) -> VecEnvStepReturn:
        state_dict, rewards, terminated, truncated, raw_info = self.env.step(self.actions)
        dones = terminated | truncated
        flat_obs = self._flatten_obs(state_dict)
        infos = self._split_batched_info(raw_info)
        if dones.all():
            if self.store_terminal_observation_info:
                for i, info in enumerate(infos):
                    info["terminal_observation"] = flat_obs[i]  # shape (obs_dim,)
            reset_obs, _ = self.env.reset()
            flat_obs = self._flatten_obs(reset_obs)
        return flat_obs, rewards, dones, infos

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # VecEnv stubs
    # ------------------------------------------------------------------

    def _selected_indices(self, indices: VecEnvIndices = None) -> List[int]:
        selected = list(self._get_indices(indices))
        n = self.env.num_trajectories
        for index in selected:
            if index < 0 or index >= n:
                raise IndexError(f"env index {index} out of range for {n} envs")
        return selected

    def _require_full_selection(self, indices: VecEnvIndices, method_name: str) -> List[int]:
        selected = self._selected_indices(indices)
        all_indices = list(range(self.env.num_trajectories))
        if sorted(selected) != all_indices:
            raise NotImplementedError(
                f"Partial {method_name} is not supported because "
                "StableBaselinesAMMEnvironment wraps one shared vectorized AMMEnvironment"
            )
        return selected

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        selected = self._selected_indices(indices)
        if attr_name == "render_mode":
            return [None] * len(selected)
        return [getattr(self.env, attr_name)] * len(selected)

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        self._require_full_selection(indices, "set_attr")
        setattr(self.env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        selected = self._require_full_selection(indices, "env_method")
        result = getattr(self.env, method_name)(*method_args, **method_kwargs)
        return [result] * len(selected)

    def env_is_wrapped(
        self, wrapper_class: Type, indices: VecEnvIndices = None
    ) -> List[bool]:
        selected = self._selected_indices(indices)
        return [False] * len(selected)

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        self.env.seed(seed)
        return [seed] * self.env.num_trajectories

    def get_images(self) -> Sequence[np.ndarray]:
        return []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_trajectories(self) -> int:
        return self.env.num_trajectories

    @property
    def n_steps(self) -> int:
        return self.env.n_steps
