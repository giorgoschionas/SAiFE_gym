"""Equivalent accounting shortcuts for fixed-policy RunningInventoryPenalty evals.

Market evolution and rebalancing use the original implementations. Claimable
fees only need the LP's occupied ticks; the full share matrix remains available
through the original code when rebalancing needs to subtract fees from the pool.
"""

import numpy as np

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    FEES0_KEY, FEES1_KEY, LP_FEE_SNAPSHOT0_KEY, LP_FEE_SNAPSHOT1_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    POOL_LIQUIDITY_ARRAY_KEY, PORTFOLIO_VALUE_KEY,
)
from SAiFE_gym.rewards.RewardFunctions import RunningInventoryPenalty


class RangeFeeDynamics(UniswapV3ModelDynamics):
    def _claimable_lp_fees(self, state=None, include_gross=False):
        if include_gross:
            return super()._claimable_lp_fees(state, include_gross=True)
        state = self._state_or_current(state)
        lower = np.clip(state[LP_TICK_LOWER_KEY].astype(np.int64) - self.tick_lower_global, 0, self.num_ticks)
        upper = np.clip(state[LP_TICK_UPPER_KEY].astype(np.int64) - self.tick_lower_global, 0, self.num_ticks)
        width = np.maximum(upper - lower, 0)
        offsets = np.arange(int(width.max()))
        indices = np.minimum(lower[:, None] + offsets, self.num_ticks - 1)
        active = offsets < width[:, None]
        trajectories = np.arange(self.num_trajectories)[:, None]
        liquidity = state[POOL_LIQUIDITY_ARRAY_KEY][trajectories, indices]
        share = np.divide(
            state[LP_LIQUIDITY_KEY][:, None], liquidity,
            out=np.zeros_like(liquidity), where=(liquidity > 0) & active,
        )
        gross0 = np.sum(state[FEES0_KEY][trajectories, indices] * share, axis=1)
        gross1 = np.sum(state[FEES1_KEY][trajectories, indices] * share, axis=1)
        return (
            np.maximum(gross0 - state[LP_FEE_SNAPSHOT0_KEY], 0.0),
            np.maximum(gross1 - state[LP_FEE_SNAPSHOT1_KEY], 0.0),
        )


class InventoryEvaluationEnvironment(AMMEnvironment):
    def step(self, action):
        # This exact reward class reads only previous portfolio value. Other
        # rewards keep the original full-state snapshot behavior.
        if type(self.reward_function) is not RunningInventoryPenalty:
            return super().step(action)
        # dt uses both current and next TIME_KEY in RunningInventoryPenalty.
        from SAiFE_gym.gym.index_names import TIME_KEY
        current = {
            key: self.model_dynamics.state[key].copy()
            for key in (PORTFOLIO_VALUE_KEY, TIME_KEY)
        }
        following = self._update_state(action)
        terminated = self._get_terminated()
        truncated = np.zeros(self.num_trajectories, dtype=bool)
        rewards = self.reward_function.calculate(current, action, following, terminated[0])
        info = self._calculate_infos(following, action, rewards)
        return following, rewards, terminated, truncated, info


def make_evaluation_env(args, params, seed):
    from experiments.train_robust_lp_agent import make_fixed_env
    return make_fixed_env(
        args, params, seed,
        dynamics_class=RangeFeeDynamics,
        environment_class=InventoryEvaluationEnvironment,
    )
