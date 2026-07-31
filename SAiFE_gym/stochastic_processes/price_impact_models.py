import abc
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
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


@dataclass
class OrderExecutionResult:
    """Vectorized metadata for explicit liquidity-taker order execution."""

    order_input: np.ndarray
    executed_input: np.ndarray
    unfilled_input: np.ndarray
    curve_input: np.ndarray
    tick_movement: np.ndarray
    direction: np.ndarray
    token0_delta: np.ndarray
    token1_delta: np.ndarray
    fee_input: np.ndarray
    swap_results: tuple[SwapResult, ...]


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

    def process_order_sizes(
        self,
        state: dict,
        order_sizes: np.ndarray,
        *,
        tick_lower_global: int,
        sqrt_grid: np.ndarray,
        num_ticks: int,
        fee_multiplier: float = 0.0,
    ) -> OrderExecutionResult:
        """
        Execute signed explicit liquidity-taker sizes using full tick intervals.

        Positive orders are gross token1 input used to buy token0 from the
        pool. Negative orders are gross token0 input sold to the pool. Fees
        are paid from this gross input budget. Any amount that cannot cross a
        complete tick interval remains unfilled.
        """
        if sqrt_grid is None or tick_lower_global is None:
            raise ValueError("sqrt_grid is not initialized. Build/reset the environment first.")

        orders = np.asarray(order_sizes, dtype=np.float64).reshape(
            self.num_trajectories, -1
        )
        if orders.shape[1] != 1:
            raise ValueError(
                "order_sizes must have shape (num_trajectories, 1) or "
                "(num_trajectories,)"
            )
        orders = orders[:, 0]

        tick_before = state[POOL_CURRENT_TICK_KEY].astype(np.int64).copy()
        curve_abs = np.zeros(self.num_trajectories, dtype=np.float64)
        output_abs = np.zeros(self.num_trajectories, dtype=np.float64)
        tick_movement = np.zeros(self.num_trajectories, dtype=np.int64)
        direction = np.sign(orders).astype(np.int64)
        swap_results = []
        gross_multiplier = 1.0 + float(fee_multiplier)

        (
            buy_exec,
            buy_output,
            buy_move,
            buy_traj,
            buy_fee_idx,
            buy_amounts,
        ) = self._execute_full_tick_orders(
            state,
            np.abs(orders),
            orders > 0.0,
            direction=1,
            tick_lower_global=tick_lower_global,
            sqrt_grid=sqrt_grid,
            num_ticks=num_ticks,
            gross_multiplier=gross_multiplier,
        )
        (
            sell_exec,
            sell_output,
            sell_move,
            sell_traj,
            sell_fee_idx,
            sell_amounts,
        ) = self._execute_full_tick_orders(
            state,
            np.abs(orders),
            orders < 0.0,
            direction=-1,
            tick_lower_global=tick_lower_global,
            sqrt_grid=sqrt_grid,
            num_ticks=num_ticks,
            gross_multiplier=gross_multiplier,
        )

        curve_abs += buy_exec + sell_exec
        output_abs += buy_output + sell_output
        tick_movement += buy_move - sell_move

        if buy_amounts.size:
            swap_results.append(
                SwapResult(
                    trajectories=buy_traj,
                    fee_key=FEES1_KEY,
                    fee_indices=buy_fee_idx,
                    amounts=buy_amounts,
                    direction=1,
                )
            )
        if sell_amounts.size:
            swap_results.append(
                SwapResult(
                    trajectories=sell_traj,
                    fee_key=FEES0_KEY,
                    fee_indices=sell_fee_idx,
                    amounts=sell_amounts,
                    direction=-1,
                )
            )

        new_tick = tick_before + tick_movement
        state[POOL_CURRENT_TICK_KEY] = new_tick
        state[POOL_SQRT_PRICE_KEY] = sqrt_grid[new_tick - tick_lower_global]

        executed_abs = curve_abs * gross_multiplier
        executed_input = direction * executed_abs
        curve_input = direction * curve_abs
        fee_input = curve_abs * float(fee_multiplier)
        token0_delta = np.zeros(self.num_trajectories, dtype=np.float64)
        token1_delta = np.zeros(self.num_trajectories, dtype=np.float64)
        buy_mask = orders > 0.0
        sell_mask = orders < 0.0
        token0_delta[buy_mask] = output_abs[buy_mask]
        token1_delta[buy_mask] = -executed_abs[buy_mask]
        token0_delta[sell_mask] = -executed_abs[sell_mask]
        token1_delta[sell_mask] = output_abs[sell_mask]
        return OrderExecutionResult(
            order_input=orders.copy(),
            executed_input=executed_input,
            unfilled_input=orders - executed_input,
            curve_input=curve_input,
            tick_movement=tick_movement.copy(),
            direction=direction.copy(),
            token0_delta=token0_delta,
            token1_delta=token1_delta,
            fee_input=fee_input,
            swap_results=tuple(swap_results),
        )

    def _execute_full_tick_orders(
        self,
        state: dict,
        abs_orders: np.ndarray,
        active: np.ndarray,
        direction: int,
        *,
        tick_lower_global: int,
        sqrt_grid: np.ndarray,
        num_ticks: int,
        gross_multiplier: float,
    ):
        """Return executable full tick intervals for one explicit order side."""
        executed = np.zeros(self.num_trajectories, dtype=np.float64)
        output = np.zeros(self.num_trajectories, dtype=np.float64)
        tick_move = np.zeros(self.num_trajectories, dtype=np.int64)
        if not np.any(active):
            empty_int = np.array([], dtype=np.int64)
            empty_float = np.array([], dtype=np.float64)
            return executed, output, tick_move, empty_int, empty_int, empty_float

        current_tick = state[POOL_CURRENT_TICK_KEY].astype(np.int64)
        idx_all = current_tick - tick_lower_global
        traj = np.flatnonzero(active)
        idx = idx_all[traj]
        requested = abs_orders[traj]

        if direction == 1:
            max_tick_move = np.maximum(num_ticks - idx, 0)
        elif direction == -1:
            max_tick_move = np.maximum(idx, 0)
        else:
            raise ValueError("direction must be -1 for sell or 1 for buy")

        max_crossed = int(max_tick_move.max()) if max_tick_move.size else 0
        if max_crossed == 0:
            empty_int = np.array([], dtype=np.int64)
            empty_float = np.array([], dtype=np.float64)
            return executed, output, tick_move, empty_int, empty_int, empty_float

        offsets = np.arange(max_crossed, dtype=np.int64)
        if direction == 1:
            interval_indices = idx[:, None] + offsets[None, :]
        else:
            interval_indices = idx[:, None] - 1 - offsets[None, :]

        valid = offsets[None, :] < max_tick_move[:, None]
        interval_indices_safe = np.clip(interval_indices, 0, num_ticks - 1)
        liquidity = (
            state[POOL_LIQUIDITY_ARRAY_KEY][traj[:, None], interval_indices_safe]
            * valid
        )
        capacities = self._interval_input_capacity(
            liquidity,
            interval_indices_safe,
            sqrt_grid,
            direction,
        ) * valid
        outputs = self._interval_output_amount(
            liquidity,
            interval_indices_safe,
            sqrt_grid,
            direction,
        ) * valid

        gross_costs = capacities * gross_multiplier
        cumulative = np.cumsum(gross_costs, axis=1)
        crossed = valid & (cumulative <= requested[:, None] + 1e-12)
        tick_move[traj] = crossed.sum(axis=1)
        executed[traj] = np.sum(capacities * crossed, axis=1)
        output[traj] = np.sum(outputs * crossed, axis=1)

        if not np.any(crossed):
            empty_int = np.array([], dtype=np.int64)
            empty_float = np.array([], dtype=np.float64)
            return executed, output, tick_move, empty_int, empty_int, empty_float

        fee_indices = interval_indices_safe[crossed]
        fee_traj = np.broadcast_to(traj[:, None], interval_indices_safe.shape)[crossed]
        amounts = capacities[crossed]
        return executed, output, tick_move, fee_traj, fee_indices, amounts

    @staticmethod
    def _interval_input_capacity(
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

    @staticmethod
    def _interval_output_amount(
        liquidity: np.ndarray,
        interval_indices: np.ndarray,
        sqrt_grid: np.ndarray,
        direction: int,
    ) -> np.ndarray:
        if direction == -1:
            return liquidity * (
                sqrt_grid[interval_indices + 1]
                - sqrt_grid[interval_indices]
            )
        return liquidity * (
            1.0 / sqrt_grid[interval_indices]
            - 1.0 / sqrt_grid[interval_indices + 1]
        )


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

    The model samples an active trade size, converts it to the swap input
    token's units, maps that amount to an expected tick impact using average
    directional one-tick capacity, then stochastically rounds the result to an
    integer tick move. Fees are returned for every realized crossed tick
    interval so LP fee attribution remains per-tick.

    ``trade_size_unit="input_token"`` keeps the sampled size in token0 units
    for sells and token1 units for buys. ``trade_size_unit="token1_notional"``
    treats the sampled size as token1 value on both sides, converting sell sizes
    to token0 by dividing by the external midprice.
    """

    def __init__(
        self,
        trade_size_sampler: Callable[[np.random.Generator, int], np.ndarray],
        depth_window: int = 10,
        min_depth: float = 1e-12,
        trade_size_unit: str = "input_token",
        num_trajectories: int = 1,
        seed: int = None,
    ):
        assert callable(trade_size_sampler), "trade_size_sampler must be callable"
        assert depth_window >= 1, f"depth_window must be >= 1, got {depth_window}"
        assert min_depth > 0, f"min_depth must be positive, got {min_depth}"
        assert trade_size_unit in {"input_token", "token1_notional"}, (
            "trade_size_unit must be either 'input_token' or 'token1_notional', "
            f"got {trade_size_unit!r}"
        )
        self.trade_size_sampler = trade_size_sampler
        self.depth_window = depth_window
        self.min_depth = min_depth
        self.trade_size_unit = trade_size_unit
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

        sampled_trade_size = np.asarray(
            self.trade_size_sampler(self.rng, len(traj)),
            dtype=np.float64,
        ).reshape(-1)
        assert sampled_trade_size.shape == (len(traj),), (
            "trade_size_sampler must return one non-negative trade size per "
            f"active trajectory, got shape {sampled_trade_size.shape}"
        )
        assert np.all(sampled_trade_size >= 0.0), (
            "trade_size_sampler must return non-negative trade sizes"
        )

        trade_size = self._to_input_token_amount(
            sampled_trade_size,
            state,
            traj,
            direction,
        )

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

    def _to_input_token_amount(
        self,
        sampled_trade_size: np.ndarray,
        state: dict,
        trajectories: np.ndarray,
        direction: int,
    ) -> np.ndarray:
        """Convert sampled trade sizes to the swap input token's native units."""
        if self.trade_size_unit == "input_token":
            return sampled_trade_size

        if direction == 1:
            return sampled_trade_size

        external_midprice = state[ASSET_PRICE_KEY][trajectories]
        assert np.all(external_midprice > 0.0), "external midprice must be positive"
        return sampled_trade_size / external_midprice
