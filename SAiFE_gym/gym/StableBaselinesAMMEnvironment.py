from typing import Any, List, Optional, Sequence, Type

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    BOUNDARY_PROXIMITY_KEY,
    FEES0_KEY,
    FEES1_KEY,
    GAS_COST_KEY,
    LP_ALPHA_KEY,
    PORTFOLIO_VALUE_KEY,
    LP_COLLECTED_FEES0_KEY,
    LP_COLLECTED_FEES1_KEY,
    LP_LIQUIDITY_KEY,
    LP_LOWER_OFFSET_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_UPPER_OFFSET_KEY,
    MISPRICING_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
    POSITION_WIDTH_KEY,
    TIME_KEY,
)

DEFAULT_OBS_KEYS = [
    MISPRICING_KEY,            # adverse selection signal
    BOUNDARY_PROXIMITY_KEY,    # min(lower_offset, upper_offset) — distance to nearest boundary
    POSITION_WIDTH_KEY,        # lower_offset + upper_offset — position concentration
    # LP_LOWER_OFFSET_KEY,     # replaced by boundary_proximity + position_width
    # LP_UPPER_OFFSET_KEY,     # replaced by boundary_proximity + position_width
    # LP_LIQUIDITY_KEY,        # not directly actionable for hold/rebalance
    # LP_COLLECTED_FEES0_KEY,  # cumulative, not actionable
    # LP_COLLECTED_FEES1_KEY,  # cumulative, not actionable
    # ASSET_PRICE_KEY,         # nearly constant at low volatility; captured by mispricing
    TIME_KEY,                  # remaining time to recoup gas cost
    GAS_COST_KEY,              # rebalancing cost
]  # obs_dim = 5

_ARRAY_KEYS = {POOL_LIQUIDITY_ARRAY_KEY, FEES0_KEY, FEES1_KEY}
_DERIVED_KEYS = {MISPRICING_KEY, LP_LOWER_OFFSET_KEY, LP_UPPER_OFFSET_KEY,
                 BOUNDARY_PROXIMITY_KEY, POSITION_WIDTH_KEY,
                 POOL_SQRT_PRICE_KEY}  # exposed as squared pool price in obs


class StableBaselinesAMMEnvironment(VecEnv):
    """
    Wraps AMMEnvironment to expose a stable-baselines3 VecEnv interface.

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
    ):
        self.env = amm_env  # must be set before super().__init__() calls get_attr()
        self.obs_keys = obs_keys if obs_keys is not None else DEFAULT_OBS_KEYS
        self.obs_dim = len(self.obs_keys)
        self.store_terminal_observation_info = store_terminal_observation_info
        self.actions = np.zeros((amm_env.num_trajectories, amm_env.action_space.shape[0]), dtype=np.float32)

        for k in self.obs_keys:
            if k in _ARRAY_KEYS:
                raise ValueError(
                    f"obs_keys may only contain scalar keys; '{k}' is an array key"
                )

        flat_obs_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        old_act = amm_env.action_space
        gymnasium_act_space = gymnasium.spaces.Box(
            low=old_act.low.astype(np.float32),
            high=old_act.high.astype(np.float32),
            shape=old_act.shape,
            dtype=np.float32,
        )
        super().__init__(amm_env.num_trajectories, flat_obs_space, gymnasium_act_space)

    # ------------------------------------------------------------------
    # Core VecEnv methods
    # ------------------------------------------------------------------

    def _compute_derived(self, state_dict: dict) -> dict:
        """Compute derived observation features from raw state."""
        lower_offset = (state_dict[POOL_CURRENT_TICK_KEY]
                        - state_dict[LP_TICK_LOWER_KEY])
        upper_offset = (state_dict[LP_TICK_UPPER_KEY]
                        - state_dict[POOL_CURRENT_TICK_KEY])
        return {
            MISPRICING_KEY:          state_dict[ASSET_PRICE_KEY]
                                     - state_dict[POOL_SQRT_PRICE_KEY] ** 2,
            POOL_SQRT_PRICE_KEY:     state_dict[POOL_SQRT_PRICE_KEY] ** 2,
            LP_LOWER_OFFSET_KEY:     lower_offset,
            LP_UPPER_OFFSET_KEY:     upper_offset,
            BOUNDARY_PROXIMITY_KEY:  np.minimum(lower_offset, upper_offset),
            POSITION_WIDTH_KEY:      lower_offset + upper_offset,
        }

    def _flatten_obs(self, state_dict: dict) -> np.ndarray:
        """Return shape (num_trajectories, obs_dim) float32 array."""
        n = self.env.num_trajectories
        derived = self._compute_derived(state_dict)
        cols = [
            derived[k].reshape(n, 1) if k in _DERIVED_KEYS
            else state_dict[k].reshape(n, 1)
            for k in self.obs_keys
        ]
        return np.concatenate(cols, axis=1).astype(np.float32)

    def reset(self) -> VecEnvObs:
        obs, _ = self.env.reset()
        return self._flatten_obs(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        state_dict, rewards, terminated, truncated, _ = self.env.step(self.actions)
        dones = terminated | truncated
        flat_obs = self._flatten_obs(state_dict)
        infos = [{} for _ in range(self.env.num_trajectories)]
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

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        if attr_name == "render_mode":
            return [None] * self.env.num_trajectories
        return [getattr(self.env, attr_name)] * self.env.num_trajectories

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        setattr(self.env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        result = getattr(self.env, method_name)(*method_args, **method_kwargs)
        return [result] * self.env.num_trajectories

    def env_is_wrapped(
        self, wrapper_class: Type, indices: VecEnvIndices = None
    ) -> List[bool]:
        return [False] * self.env.num_trajectories

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
