import gymnasium
import numpy as np
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment

from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    ASSET_PRICE_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_EVER_DEPLOYED_KEY,
    TIME_KEY,
)


class RandomAgent(Agent):
    """
    Randomly samples LP position bounds uniformly from [-tau, tau].

    Uses order statistics: samples two points, sorts them to ensure lower < upper.
    This guarantees valid actions where lower_offset < upper_offset.
    """
    def __init__(self, env: gymnasium.Env, seed: int = None):
        self.tau = env.model_dynamics.tau
        self.num_trajectories = env.num_trajectories
        self.rng = np.random.default_rng(seed)

    def get_action(self, state: dict) -> np.ndarray:
        # Sample two points uniformly from [-tau, tau] for each trajectory
        # Shape: (num_trajectories, 2)
        samples = self.rng.uniform(-self.tau, self.tau, size=(self.num_trajectories, 2))

        # Sort along axis=1 so that [:, 0] < [:, 1]
        actions = np.sort(samples, axis=1)

        # Clip to valid action space bounds: lower ∈ [-tau, tau-1], upper ∈ [-tau+1, tau]
        actions[:, 0] = np.clip(actions[:, 0], -self.tau, self.tau - 1)
        actions[:, 1] = np.clip(actions[:, 1], -self.tau + 1, self.tau)

        # Ensure minimum width of 1 tick (lower < upper)
        too_close = actions[:, 1] <= actions[:, 0]
        actions[too_close, 1] = actions[too_close, 0] + 1

        # Append hold_flag = -1.0 (always rebalance)
        hold_col = np.full((self.num_trajectories, 1), -1.0, dtype=np.float32)
        return np.concatenate([actions.astype(np.float32), hold_col], axis=1)


class UniformAllocationAgent(Agent):
    """
    Allocates capital across the full active tick range around the current price.

    Action format: [lower_offset, upper_offset, hold_flag] = [-tau, +tau, -1.0]
    This covers 2*tau+1 ticks centered on the current price. Always rebalances.
    """
    def __init__(self, env: AMMEnvironment):
        self.env = env
        self.tau = env.model_dynamics.tau


    def get_action(self, state: dict) -> np.ndarray:
        action = np.array([[-self.tau, self.tau, -1.0]], dtype=np.float32)
        return np.repeat(action, self.env.num_trajectories, axis=0)


class DeployOnceAgent(Agent):
    """
    Deploys liquidity once at a fixed tick range and holds for the entire episode.

    On the first step (LP not yet deployed), emits hold_flag = -1 to trigger
    deployment at [lower_offset, upper_offset]. On all subsequent steps, emits
    hold_flag = +1 to hold the existing position without rebalancing.

    Defaults to the symmetric full active range [-tau, +tau]. Pass
    `lower_offset` / `upper_offset` to quote asymmetrically, e.g. [-3, +7].
    """
    def __init__(self, env: AMMEnvironment,
                 lower_offset: int = None, upper_offset: int = None):
        self.env = env
        tau = env.model_dynamics.tau
        self.lower_offset = -tau if lower_offset is None else int(lower_offset)
        self.upper_offset = tau if upper_offset is None else int(upper_offset)
        assert -tau <= self.lower_offset < self.upper_offset <= tau, (
            f"need -tau <= lower_offset < upper_offset <= tau, got "
            f"[{self.lower_offset}, {self.upper_offset}] with tau={tau}"
        )

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        lower = np.full(n, self.lower_offset, dtype=np.float32)
        upper = np.full(n, self.upper_offset, dtype=np.float32)
        ever_deployed = state[LP_EVER_DEPLOYED_KEY]
        hold_flag = np.where(ever_deployed, 1.0, -1.0).astype(np.float32)
        return np.column_stack([lower, upper, hold_flag])


class PeriodicRebalanceAgent(Agent):
    """
    Quotes a fixed ±width tick range around the current price and rebalances
    every `rebalance_every` steps (holds in between).

    Step index is derived from TIME_KEY so the agent is stateless and resets
    automatically with env.reset(). Step 0 always deploys (0 % N == 0).
    """
    def __init__(self, env: AMMEnvironment, rebalance_every: int = 5, width: int = 2):
        assert rebalance_every >= 1, f"rebalance_every must be >= 1, got {rebalance_every}"
        assert 1 <= width <= env.model_dynamics.tau, (
            f"width must be in [1, tau={env.model_dynamics.tau}], got {width}"
        )
        self.env = env
        self.rebalance_every = rebalance_every
        self.width = width

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        lower = np.full(n, -self.width, dtype=np.float32)
        upper = np.full(n,  self.width, dtype=np.float32)

        step_idx = int(np.round(state[TIME_KEY][0] / self.env.step_size))
        rebalance = (step_idx % self.rebalance_every) == 0
        hold_flag = np.full(n, -1.0 if rebalance else 1.0, dtype=np.float32)

        return np.column_stack([lower, upper, hold_flag])


class ArrivalRebalanceAgent(Agent):
    """
    Quotes a fixed tick range around the current price; rebalances after
    every ``rebalance_every`` liquidity-taking arrivals (sell or buy).

    Range:
      - Symmetric (default): ``width=W`` → ``[-W, +W]``.
      - Asymmetric: pass ``lower_offset`` and ``upper_offset`` to quote
        e.g. ``[-3, +7]``. When both are provided they take precedence
        over ``width``.

    Reads the arrivals that produced the *current* state directly from
    ``env.model_dynamics.last_arrivals`` (cached after each get_arrivals
    call), so the count is exact — no fee-delta or |Δtick| approximation.
    """
    def __init__(self, env: AMMEnvironment, rebalance_every: int = 10,
                 width: int = 2,
                 lower_offset: int = None, upper_offset: int = None):
        assert rebalance_every >= 1, f"rebalance_every must be >= 1, got {rebalance_every}"
        tau = env.model_dynamics.tau

        if lower_offset is not None or upper_offset is not None:
            assert lower_offset is not None and upper_offset is not None, (
                "pass both lower_offset and upper_offset, or neither"
            )
            self.lower_offset = int(lower_offset)
            self.upper_offset = int(upper_offset)
        else:
            assert 1 <= width <= tau, f"width must be in [1, tau={tau}], got {width}"
            self.lower_offset = -int(width)
            self.upper_offset = int(width)

        assert -tau <= self.lower_offset < self.upper_offset <= tau, (
            f"need -tau <= lower_offset < upper_offset <= tau, got "
            f"[{self.lower_offset}, {self.upper_offset}] with tau={tau}"
        )

        self.env = env
        self.rebalance_every = rebalance_every
        self.arrival_count = np.zeros(env.num_trajectories, dtype=np.int64)

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        is_episode_start = state[TIME_KEY][0] < self.env.step_size / 2

        if is_episode_start:
            self.arrival_count = np.zeros(n, dtype=np.int64)
            rebalance = np.ones(n, dtype=bool)  # initial deploy
        else:
            new_arrivals = self.env.model_dynamics.last_arrivals.sum(axis=1).astype(np.int64)
            self.arrival_count += new_arrivals
            rebalance = self.arrival_count >= self.rebalance_every
            self.arrival_count = np.where(rebalance, 0, self.arrival_count)

        lower = np.full(n, self.lower_offset, dtype=np.float32)
        upper = np.full(n, self.upper_offset, dtype=np.float32)
        hold_flag = np.where(rebalance, -1.0, 1.0).astype(np.float32)
        return np.column_stack([lower, upper, hold_flag])


class ArbitrageurAgent(Agent):
    """
    Deterministic liquidity-taker arbitrage baseline for Uniswap-v3 pools.

    Action format: one signed gross input amount per trajectory, shape
    ``(num_trajectories, 1)``.

    - ``order_size > 0``: gross token1 input, buy token0 from the pool, move price up.
    - ``order_size < 0``: gross token0 input, sell token0 to the pool, move price down.
    - ``order_size == 0``: no trade.

    The default policy trades toward the external midprice until the AMM price
    is back inside the Uniswap-fee no-arbitrage band, using full tick intervals.
    """

    def __init__(
        self,
        env: AMMEnvironment,
        max_ticks_per_trade: int = None,
        max_order_size: float = np.inf,
        min_order_size: float = 0.0,
    ):
        assert isinstance(env.model_dynamics, UniswapV3ModelDynamics), (
            "ArbitrageurAgent requires UniswapV3ModelDynamics"
        )
        assert max_ticks_per_trade is None or max_ticks_per_trade >= 1, (
            "max_ticks_per_trade must be None or >= 1"
        )
        assert max_order_size > 0.0, "max_order_size must be positive"
        assert min_order_size >= 0.0, "min_order_size must be non-negative"
        self.env = env
        self.model_dynamics = env.model_dynamics
        self.num_trajectories = env.num_trajectories
        self.max_ticks_per_trade = max_ticks_per_trade
        self.max_order_size = float(max_order_size)
        self.min_order_size = float(min_order_size)

    def get_action(self, state: dict) -> np.ndarray:
        orders = self._target_order_sizes(
            state,
            max_ticks=self.max_ticks_per_trade,
            max_order_size=self.max_order_size,
            min_order_size=self.min_order_size,
        )
        return orders.reshape(self.num_trajectories, 1)

    def _target_order_sizes(
        self,
        state: dict,
        max_ticks: int = None,
        max_order_size: float = np.inf,
        min_order_size: float = 0.0,
    ) -> np.ndarray:
        md = self.model_dynamics
        current_tick = state[POOL_CURRENT_TICK_KEY].astype(np.int64)
        pool_price = md.get_pool_sqrt_price(state) ** 2
        external_midprice = state[ASSET_PRICE_KEY].astype(np.float64)

        fee_discount = 1.0 - md.fee_tier
        buy_token0 = external_midprice > pool_price / fee_discount
        sell_token0 = external_midprice < pool_price * fee_discount

        desired_up = np.zeros(self.num_trajectories, dtype=np.int64)
        desired_down = np.zeros(self.num_trajectories, dtype=np.int64)

        valid_midprice = external_midprice > 0.0
        safe_midprice = np.where(valid_midprice, external_midprice, 1.0)
        if np.any(buy_token0 & valid_midprice):
            target_price = safe_midprice * fee_discount
            target_tick = np.ceil(
                np.log(target_price) / np.log(md.exponential_value)
            ).astype(np.int64)
            desired_up = np.maximum(target_tick - current_tick, 0)

        if np.any(sell_token0 & valid_midprice):
            target_price = safe_midprice / fee_discount
            target_tick = np.floor(
                np.log(target_price) / np.log(md.exponential_value)
            ).astype(np.int64)
            desired_down = np.maximum(current_tick - target_tick, 0)

        desired_up = np.where(buy_token0 & valid_midprice, desired_up, 0)
        desired_down = np.where(sell_token0 & valid_midprice, desired_down, 0)

        idx = current_tick - md.tick_lower_global
        desired_up = np.minimum(desired_up, np.maximum(md.num_ticks - idx, 0))
        desired_down = np.minimum(desired_down, np.maximum(idx, 0))
        if max_ticks is not None:
            desired_up = np.minimum(desired_up, max_ticks)
            desired_down = np.minimum(desired_down, max_ticks)

        gross_multiplier = 1.0 + md.fee_multiplier
        orders = gross_multiplier * self._input_for_tick_moves(
            state,
            desired_up,
            direction=1,
        )
        orders -= gross_multiplier * self._input_for_tick_moves(
            state,
            desired_down,
            direction=-1,
        )

        if np.isfinite(max_order_size):
            orders = np.sign(orders) * np.minimum(np.abs(orders), max_order_size)
        orders = np.where(np.abs(orders) >= min_order_size, orders, 0.0)
        return orders

    def _input_for_tick_moves(
        self,
        state: dict,
        tick_moves: np.ndarray,
        direction: int,
    ) -> np.ndarray:
        md = self.model_dynamics
        amounts = np.zeros(self.num_trajectories, dtype=np.float64)
        active = tick_moves > 0
        if not np.any(active):
            return amounts

        traj = np.flatnonzero(active)
        current_tick = state[POOL_CURRENT_TICK_KEY].astype(np.int64)
        idx = current_tick[traj] - md.tick_lower_global
        max_crossed = int(tick_moves[traj].max())
        offsets = np.arange(max_crossed, dtype=np.int64)

        if direction == 1:
            interval_indices = idx[:, None] + offsets[None, :]
        elif direction == -1:
            interval_indices = idx[:, None] - 1 - offsets[None, :]
        else:
            raise ValueError("direction must be -1 for sell or 1 for buy")

        # NumPy gathers before masking, so padded intervals need safe indices.
        interval_indices_safe = np.clip(interval_indices, 0, md.num_ticks - 1)
        liquidity = state[POOL_LIQUIDITY_ARRAY_KEY][traj[:, None], interval_indices_safe]
        if direction == 1:
            capacities = liquidity * (
                md.sqrt_grid[interval_indices_safe + 1] - md.sqrt_grid[interval_indices_safe]
            )
        else:
            capacities = liquidity * (
                1.0 / md.sqrt_grid[interval_indices_safe]
                - 1.0 / md.sqrt_grid[interval_indices_safe + 1]
            )

        crossed = offsets[None, :] < tick_moves[traj, None]
        amounts[traj] = np.sum(capacities * crossed, axis=1)
        return amounts


class SpeedControlArbitrageurAgent(ArbitrageurAgent):
    """
    Deterministic liquidity-taker arbitrage baseline with speed controls.

    Action format: one signed trading speed per trajectory, shape
    ``(num_trajectories, 1)``. Execution size over a step is
    ``order_size = trading_speed * env.step_size``.

    Positive speeds are gross token1-per-unit-time inputs used to buy token0
    from the pool. Negative speeds are gross token0-per-unit-time inputs sold to
    the pool.
    """

    def __init__(
        self,
        env: AMMEnvironment,
        max_ticks_per_step: int = 1,
        max_speed: float = np.inf,
        min_speed: float = 0.0,
    ):
        assert max_ticks_per_step is None or max_ticks_per_step >= 1, (
            "max_ticks_per_step must be None or >= 1"
        )
        assert max_speed > 0.0, "max_speed must be positive"
        assert min_speed >= 0.0, "min_speed must be non-negative"
        super().__init__(
            env,
            max_ticks_per_trade=max_ticks_per_step,
            max_order_size=np.inf,
            min_order_size=0.0,
        )
        self.max_ticks_per_step = max_ticks_per_step
        self.max_speed = float(max_speed)
        self.min_speed = float(min_speed)

    def get_action(self, state: dict) -> np.ndarray:
        order_sizes = self._target_order_sizes(
            state,
            max_ticks=self.max_ticks_per_step,
            max_order_size=np.inf,
            min_order_size=0.0,
        )
        step_size = float(self.env.step_size)
        if step_size <= 0.0:
            raise ValueError("env.step_size must be positive for speed controls")

        speeds = order_sizes / step_size
        if np.isfinite(self.max_speed):
            speeds = np.sign(speeds) * np.minimum(np.abs(speeds), self.max_speed)
        speeds = np.where(np.abs(speeds) >= self.min_speed, speeds, 0.0)
        return speeds.reshape(self.num_trajectories, 1)
