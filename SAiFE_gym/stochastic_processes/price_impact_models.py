import abc

import numpy as np

from SAiFE_gym.gym.index_names import (
    FEES0_KEY,
    FEES1_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
)
from SAiFE_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


class PriceImpactModel(StochasticProcessModel):
    """Base class for AMM price-impact rules.

    SAiFE price impact is an AMM pool-state transition, not a scalar execution
    price adjustment. Stateless models can use the default empty process state.
    """

    def __init__(self, num_trajectories: int = 1, seed: int = None):
        empty_state = np.empty((1, 0))
        super().__init__(
            min_value=empty_state,
            max_value=empty_state,
            step_size=None,
            terminal_time=0.0,
            initial_state=empty_state,
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray, state: dict = None):
        return self.current_state

    @abc.abstractmethod
    def process_swap(
        self,
        state: dict,
        active: np.ndarray,
        direction: int,
        *,
        tick_lower_global: int,
        sqrt_grid: np.ndarray,
        fee_multiplier: float,
        num_ticks: int,
    ) -> None:
        pass


class OneTickUniswapV3PriceImpact(PriceImpactModel):
    """One-tick Uniswap V3 lattice impact model.

    Each active sell token0 arrival moves the pool down one tick. Each active
    buy token0 arrival moves the pool up one tick. Trade size and fees are
    computed from active liquidity at the crossed interval.
    """

    def process_swap(
        self,
        state: dict,
        active: np.ndarray,
        direction: int,
        *,
        tick_lower_global: int,
        sqrt_grid: np.ndarray,
        fee_multiplier: float,
        num_ticks: int,
    ) -> None:
        if not np.any(active):
            return
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 for sell or 1 for buy")

        current_tick = state[POOL_CURRENT_TICK_KEY].astype(np.int64)
        idx_all = current_tick - tick_lower_global
        traj = np.flatnonzero(active)
        idx = idx_all[traj]

        if direction == -1:
            label = "sell"
            valid_min, valid_max = 1, num_ticks
            valid_tick_min = tick_lower_global + 1
            valid_tick_max = tick_lower_global + num_ticks
            fee_key = FEES0_KEY
            fee_idx = idx - 1
        else:
            label = "buy"
            valid_min, valid_max = 0, num_ticks - 1
            valid_tick_min = tick_lower_global
            valid_tick_max = tick_lower_global + num_ticks - 1
            fee_key = FEES1_KEY
            fee_idx = idx

        assert idx.min() >= valid_min and idx.max() <= valid_max, (
            f"_process_swap({label}): tick out of array window. "
            f"current_tick range [{current_tick[traj].min()}, {current_tick[traj].max()}], "
            f"valid [{valid_tick_min}, {valid_tick_max}]. "
            f"Increase num_ticks."
        )

        liquidity = state[POOL_LIQUIDITY_ARRAY_KEY][traj, fee_idx]
        if direction == -1:
            amount = liquidity * (1.0 / sqrt_grid[idx - 1] - 1.0 / sqrt_grid[idx])
        else:
            amount = liquidity * (sqrt_grid[idx + 1] - sqrt_grid[idx])

        state[fee_key][traj, fee_idx] += fee_multiplier * amount

        new_tick = current_tick.copy()
        new_tick[traj] += direction
        state[POOL_CURRENT_TICK_KEY] = new_tick
        state[POOL_SQRT_PRICE_KEY] = sqrt_grid[new_tick - tick_lower_global]
