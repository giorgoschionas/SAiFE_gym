from typing import Optional

import gymnasium
import numpy as np

from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    MISPRICING_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
    TIME_KEY,
)
from SAiFE_gym.gym.observation_features import compute_sb3_observation_features
from SAiFE_gym.gym.simulation_core import (
    advance_market_state,
    compute_derived_obs,
    create_uniswap_v3_initial_state,
    reset_model_state,
    reset_stochastic_processes,
    terminated_flags,
)
from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import OrnsteinUhlenbeckMidpriceModel
from SAiFE_gym.stochastic_processes.price_impact_models import PriceImpactModel


ACTIVE_LIQUIDITY_KEY = "active_liquidity"

DEFAULT_ARBITRAGEUR_OBS_KEYS = [
    MISPRICING_KEY,
    ASSET_PRICE_KEY,
    POOL_SQRT_PRICE_KEY,
    ACTIVE_LIQUIDITY_KEY,
    TIME_KEY,
]


class ArbitrageurEnvironment(gymnasium.Env):
    """
    Speed-control arbitrage environment over a SAiFE Uniswap-v3 AMM.

    This environment owns the same vectorized pool, arrival, and midprice state
    as the LP-facing AMMEnvironment, but exposes the liquidity taker's signed
    trading speed as the external action and rewards immediate hedged token1 PnL.
    No internal LP policy is applied.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 1000,
        initial_wealth: float = 1000.0,
        model_dynamics: UniswapV3ModelDynamics = None,
        num_trajectories: int = 1,
        initial_pool_price: float = None,
        seed: int = None,
        max_speed: float = np.inf,
        speed_cost_coefficient: float = 0.0,
        obs_keys: Optional[list[str]] = None,
    ):
        if not isinstance(model_dynamics, UniswapV3ModelDynamics):
            raise TypeError("ArbitrageurEnvironment requires UniswapV3ModelDynamics")
        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        if speed_cost_coefficient < 0.0:
            raise ValueError("speed_cost_coefficient must be non-negative")

        super().__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.initial_wealth = initial_wealth
        self.model_dynamics = model_dynamics
        self.num_trajectories = num_trajectories
        self.initial_pool_price = initial_pool_price
        self._step_size = self.terminal_time / self.n_steps
        self.max_speed = float(max_speed)
        self.speed_cost_coefficient = float(speed_cost_coefficient)
        self.obs_keys = obs_keys if obs_keys is not None else DEFAULT_ARBITRAGEUR_OBS_KEYS
        self.cumulative_arb_pnl = np.zeros(self.num_trajectories, dtype=np.float64)

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_trajectories, len(self.obs_keys)),
            dtype=np.float32,
        )
        low = np.full((self.num_trajectories, 1), -self.max_speed, dtype=np.float32)
        high = np.full((self.num_trajectories, 1), self.max_speed, dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

        self._initial_state = create_uniswap_v3_initial_state(
            self.model_dynamics,
            self.num_trajectories,
            self.initial_wealth,
            self.initial_pool_price,
        )
        reset_model_state(
            self.model_dynamics,
            self._initial_state,
            self.num_trajectories,
        )

        if seed:
            self.seed(seed)
        self.rng = np.random.default_rng(seed)

    @property
    def step_size(self):
        return self._step_size

    @property
    def state(self):
        return self.model_dynamics.state

    @property
    def initial_state(self):
        return {k: v.copy() for k, v in self._initial_state.items()}

    def seed(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        if self.model_dynamics.midprice_model:
            self.model_dynamics.midprice_model.seed(seed)
        if self.model_dynamics.arrival_model:
            self.model_dynamics.arrival_model.seed(seed + 1 if seed else None)

    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            self.seed(seed)

        reset_stochastic_processes(self.model_dynamics)
        reset_model_state(
            self.model_dynamics,
            self._initial_state,
            self.num_trajectories,
        )
        compute_derived_obs(self.model_dynamics.state, self.model_dynamics)
        self.cumulative_arb_pnl = np.zeros(self.num_trajectories, dtype=np.float64)
        return self._flatten_obs(self.model_dynamics.state), {}

    def step(self, action: np.ndarray):
        trading_speeds = self._prepare_action(action)
        observed_midprice = self.model_dynamics.state[ASSET_PRICE_KEY].copy()

        arb_info = self.model_dynamics.execute_liquidity_taker_speeds(
            trading_speeds,
            step_size=self.step_size,
        )
        compute_derived_obs(self.model_dynamics.state, self.model_dynamics)

        hedged_pnl = (
            arb_info["token1_delta"]
            + observed_midprice * arb_info["token0_delta"]
        )
        speed_cost = (
            self.speed_cost_coefficient
            * trading_speeds[:, 0] ** 2
            * self.step_size
        )
        rewards = hedged_pnl - speed_cost
        self.cumulative_arb_pnl += rewards

        arrivals = self.model_dynamics.get_arrivals()
        self.model_dynamics.update_state(arrivals, action=None)
        advance_market_state(self.model_dynamics, arrivals, action=None)
        self.model_dynamics.state[TIME_KEY] += self.step_size
        compute_derived_obs(self.model_dynamics.state, self.model_dynamics)

        terminated = terminated_flags(
            self.model_dynamics.state,
            self.terminal_time,
            self.step_size,
            self.num_trajectories,
        )
        truncated = np.zeros(self.num_trajectories, dtype=bool)
        obs = self._flatten_obs(self.model_dynamics.state)
        info = self._calculate_info(arb_info, rewards, hedged_pnl, speed_cost)
        return obs, rewards, terminated, truncated, info

    def _prepare_action(self, action: np.ndarray) -> np.ndarray:
        speeds = np.asarray(action, dtype=np.float64)
        if speeds.shape == (1,):
            speeds = np.repeat(speeds.reshape(1, 1), self.num_trajectories, axis=0)
        else:
            speeds = speeds.reshape(self.num_trajectories, -1)
            if speeds.shape[1] != 1:
                raise ValueError(
                    "arbitrage speed action must have shape "
                    "(num_trajectories, 1) or (1,)"
                )
        return np.clip(speeds, -self.max_speed, self.max_speed)

    def _flatten_obs(self, state: dict) -> np.ndarray:
        features = compute_sb3_observation_features(state)
        features[ACTIVE_LIQUIDITY_KEY] = self._active_liquidity(state)
        cols = []
        for key in self.obs_keys:
            value = features[key] if key in features else state[key]
            cols.append(np.asarray(value, dtype=np.float64).reshape(self.num_trajectories, 1))
        return np.concatenate(cols, axis=1).astype(np.float32)

    def _active_liquidity(self, state: dict) -> np.ndarray:
        idx = np.clip(
            (state[POOL_CURRENT_TICK_KEY] - self.model_dynamics.tick_lower_global).astype(np.int64),
            0,
            self.model_dynamics.num_ticks - 1,
        )
        return state[POOL_LIQUIDITY_ARRAY_KEY][np.arange(self.num_trajectories), idx]

    def _calculate_info(
        self,
        arb_info: dict,
        rewards: np.ndarray,
        hedged_pnl: np.ndarray,
        speed_cost: np.ndarray,
    ) -> dict:
        info = {key: np.asarray(value).copy() for key, value in arb_info.items()}
        info.update(
            {
                "hedged_pnl": hedged_pnl.copy(),
                "speed_cost": speed_cost.copy(),
                "arb_reward": rewards.copy(),
                "cumulative_arb_pnl": self.cumulative_arb_pnl.copy(),
                "asset_price": self.model_dynamics.state[ASSET_PRICE_KEY].copy(),
                "pool_price": (
                    self.model_dynamics.state[POOL_SQRT_PRICE_KEY] ** 2
                ).copy(),
                "time": self.model_dynamics.state[TIME_KEY].copy(),
            }
        )
        return info


def build_arbitrageur_environment(
    num_trajectories: int = 1,
    n_steps: int = 1000,
    seed: int = 6,
    terminal_time: float = 1.0,
    initial_wealth: float = 1000.0,
    initial_price: float = 100.0,
    initial_pool_price: float = None,
    volatility: float = 0.009,
    fee_tier: float = 0.003,
    tau: int = 20,
    num_ticks: int = 5000,
    exponential_value: float = 1.0001,
    liquidity_scale: float = 1e5,
    alpha0: np.ndarray = None,
    alpha1: np.ndarray = None,
    alpha2: np.ndarray = None,
    alpha3: np.ndarray = None,
    max_speed: float = np.inf,
    speed_cost_coefficient: float = 0.0,
    price_impact_model: PriceImpactModel = None,
) -> ArbitrageurEnvironment:
    """Build a SAiFE arbitrage speed-control environment."""
    step_size = terminal_time / n_steps
    alpha0 = np.array([1.0, 1.0]) if alpha0 is None else np.asarray(alpha0, dtype=np.float64)
    alpha1 = np.array([15.0, 15.0]) if alpha1 is None else np.asarray(alpha1, dtype=np.float64)
    alpha2 = np.array([0.0, 0.0]) if alpha2 is None else np.asarray(alpha2, dtype=np.float64)
    alpha3 = np.array([0.0, 0.0]) if alpha3 is None else np.asarray(alpha3, dtype=np.float64)
    alpha = np.array([alpha0, alpha1, alpha2, alpha3], dtype=np.float64)

    midprice_model = OrnsteinUhlenbeckMidpriceModel(
        mean_reversion=0.4,
        long_term_mean=initial_price,
        volatility=volatility,
        initial_price=initial_price,
        terminal_time=terminal_time,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed,
    )
    arrival_model = LiquidityKernelArrivalModel(
        alpha=alpha,
        beta=0.1,
        K=20,
        liquidity_scale=liquidity_scale,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1 if seed is not None else None,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        price_impact_model=price_impact_model,
        num_trajectories=num_trajectories,
        fee_tier=fee_tier,
        tau=tau,
        num_ticks=num_ticks,
        exponential_value=exponential_value,
        seed=seed + 2 if seed is not None else None,
    )
    return ArbitrageurEnvironment(
        terminal_time=terminal_time,
        n_steps=n_steps,
        initial_wealth=initial_wealth,
        model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        initial_pool_price=initial_pool_price,
        seed=seed,
        max_speed=max_speed,
        speed_cost_coefficient=speed_cost_coefficient,
    )
