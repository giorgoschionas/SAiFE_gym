import abc
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from SAiFE_gym.gym.index_names import (
    FEES0_KEY,
    FEES1_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
)
from SAiFE_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


@dataclass
class SwapResult:
    """Vectorized metadata for one swap side."""

    trajectories: np.ndarray
    fee_key: str
    fee_indices: np.ndarray
    amounts: np.ndarray
    direction: int


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
        num_ticks: int,
    ) -> Optional[SwapResult]:
        pass


class OneTickUniswapV3PriceImpact(PriceImpactModel):
    """One-tick Uniswap V3 lattice impact model.

    Each active sell token0 arrival moves the pool down one tick. Each active
    buy token0 arrival moves the pool up one tick. Trade size and fees are
    computed from active liquidity at the crossed interval and returned for
    separate fee accounting.
    """

    def process_swap(
        self,
        state: dict,
        active: np.ndarray,
        direction: int,
        *,
        tick_lower_global: int,
        sqrt_grid: np.ndarray,
        num_ticks: int,
    ) -> Optional[SwapResult]:
        if not np.any(active):
            return None
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

        new_tick = current_tick.copy()
        new_tick[traj] += direction
        state[POOL_CURRENT_TICK_KEY] = new_tick
        state[POOL_SQRT_PRICE_KEY] = sqrt_grid[new_tick - tick_lower_global]

        return SwapResult(
            trajectories=traj,
            fee_key=fee_key,
            fee_indices=fee_idx,
            amounts=amount,
            direction=direction,
        )


class LiquidityDepthUniswapV3PriceImpact(PriceImpactModel):
    """Reduced-form liquidity-depth Uniswap V3 impact model.

    The model samples an active trade size, converts it to an expected tick
    impact using average directional one-tick capacity, then stochastically
    rounds the result to an integer tick move. Fees are returned for every
    realized crossed tick interval so LP fee attribution remains per-tick.
    """

    def __init__(
        self,
        trade_size_sampler: Callable[[np.random.Generator, int], np.ndarray],
        depth_window: int = 10,
        min_depth: float = 1e-12,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        assert callable(trade_size_sampler), "trade_size_sampler must be callable"
        assert depth_window >= 1, f"depth_window must be >= 1, got {depth_window}"
        assert min_depth > 0, f"min_depth must be positive, got {min_depth}"
        self.trade_size_sampler = trade_size_sampler
        self.depth_window = depth_window
        self.min_depth = min_depth
        super().__init__(num_trajectories=num_trajectories, seed=seed)

    def process_swap(
        self,
        state: dict,
        active: np.ndarray,
        direction: int,
        *,
        tick_lower_global: int,
        sqrt_grid: np.ndarray,
        num_ticks: int,
    ) -> Optional[SwapResult]:
        if not np.any(active):
            return None
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 for sell or 1 for buy")

        current_tick = state[POOL_CURRENT_TICK_KEY].astype(np.int64)
        idx_all = current_tick - tick_lower_global
        traj = np.flatnonzero(active)
        idx = idx_all[traj]
        liquidity_array = state[POOL_LIQUIDITY_ARRAY_KEY]

        offsets = np.arange(self.depth_window)
        traj_idx = traj[:, None]
        if direction == -1:
            interval_indices = idx[:, None] - 1 - offsets[None, :]
            fee_key = FEES0_KEY
            max_tick_move = np.maximum(idx, 0)
        else:
            interval_indices = idx[:, None] + offsets[None, :]
            fee_key = FEES1_KEY
            max_tick_move = np.maximum(num_ticks - idx, 0)

        valid = (interval_indices >= 0) & (interval_indices < num_ticks)
        interval_indices_safe = np.clip(interval_indices, 0, num_ticks - 1)
        liquidity = liquidity_array[traj_idx, interval_indices_safe] * valid
        capacities = self._interval_capacity(
            liquidity,
            interval_indices_safe,
            sqrt_grid,
            direction,
        ) * valid

        valid_counts = valid.sum(axis=1)
        safe_counts = np.where(valid_counts > 0, valid_counts, 1)
        average_depth = capacities.sum(axis=1) / safe_counts
        average_depth = np.maximum(average_depth, self.min_depth)

        trade_size = np.asarray(
            self.trade_size_sampler(self.rng, len(traj)),
            dtype=np.float64,
        ).reshape(-1)
        assert trade_size.shape == (len(traj),), (
            "trade_size_sampler must return one non-negative trade size per "
            f"active trajectory, got shape {trade_size.shape}"
        )
        assert np.all(trade_size >= 0.0), "trade_size_sampler must return non-negative trade sizes"

        eta = trade_size / average_depth
        whole_ticks = np.floor(eta).astype(np.int64)
        fractional_ticks = eta - whole_ticks
        rounded_ticks = whole_ticks + (
            self.rng.uniform(size=len(traj)) < fractional_ticks
        ).astype(np.int64)
        tick_move = np.minimum(rounded_ticks, max_tick_move).astype(np.int64)

        moving = tick_move > 0
        if not np.any(moving):
            return None

        new_tick = current_tick.copy()
        new_tick[traj[moving]] += direction * tick_move[moving]
        state[POOL_CURRENT_TICK_KEY] = new_tick
        state[POOL_SQRT_PRICE_KEY] = sqrt_grid[new_tick - tick_lower_global]

        max_crossed = int(tick_move[moving].max())
        crossed_offsets = np.arange(max_crossed)
        crossed_traj = traj[moving]
        crossed_idx = idx[moving]
        crossed_move = tick_move[moving]
        crossed_mask = crossed_offsets[None, :] < crossed_move[:, None]

        if direction == -1:
            fee_indices_matrix = crossed_idx[:, None] - 1 - crossed_offsets[None, :]
        else:
            fee_indices_matrix = crossed_idx[:, None] + crossed_offsets[None, :]

        fee_indices = fee_indices_matrix[crossed_mask]
        result_trajectories = np.broadcast_to(
            crossed_traj[:, None],
            fee_indices_matrix.shape,
        )[crossed_mask]
        result_liquidity = liquidity_array[result_trajectories, fee_indices]
        amounts = self._interval_capacity(
            result_liquidity,
            fee_indices,
            sqrt_grid,
            direction,
        )

        return SwapResult(
            trajectories=result_trajectories,
            fee_key=fee_key,
            fee_indices=fee_indices,
            amounts=amounts,
            direction=direction,
        )

    @staticmethod
    def _interval_capacity(
        liquidity: np.ndarray,
        interval_indices: np.ndarray,
        sqrt_grid: np.ndarray,
        direction: int,
    ) -> np.ndarray:
        if direction == -1:
            return liquidity * (
                1.0 / sqrt_grid[interval_indices]
                - 1.0 / sqrt_grid[interval_indices + 1]
            )
        return liquidity * (
            sqrt_grid[interval_indices + 1]
            - sqrt_grid[interval_indices]
        )
