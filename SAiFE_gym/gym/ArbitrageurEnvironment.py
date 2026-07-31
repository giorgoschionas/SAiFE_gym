from typing import Callable, Optional

import gymnasium
import numpy as np

from SAiFE_gym.agents.BaselineAgents import DeployOnceAgent
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
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
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import OrnsteinUhlenbeckMidpriceModel
from SAiFE_gym.stochastic_processes.price_impact_models import PriceImpactModel


ACTIVE_LIQUIDITY_KEY = "active_liquidity"
LP_POLICY_NONE = "none"
LP_POLICY_DEPLOY_ONCE = "deploy_once"

DEFAULT_ARBITRAGEUR_OBS_KEYS = [
    MISPRICING_KEY,
    ASSET_PRICE_KEY,
    POOL_SQRT_PRICE_KEY,
    ACTIVE_LIQUIDITY_KEY,
    TIME_KEY,
]


class NoOpLPAgent(Agent):
    """LP placeholder that never deploys or rebalances."""

    def __init__(self, env: AMMEnvironment):
        self.env = env

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        lower = np.zeros(n, dtype=np.float32)
        upper = np.ones(n, dtype=np.float32)
        hold_flag = np.ones(n, dtype=np.float32)
        return np.column_stack([lower, upper, hold_flag])


class ArbitrageurEnvironment(gymnasium.Env):
    """
    Speed-control arbitrage environment over a SAiFE Uniswap-v3 AMM.

    The wrapped AMM environment still owns LP positioning, arrivals, fees, and
    midprice dynamics. This environment exposes the liquidity taker's signed
    trading speed as the external action and rewards immediate hedged token1 PnL.
    It follows SAiFE's vectorized convention: observations and actions are
    batched across trajectories.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        amm_env: AMMEnvironment,
        lp_agent: Optional[Agent] = None,
        lp_agent_factory: Optional[Callable[[AMMEnvironment], Agent]] = None,
        max_speed: float = np.inf,
        speed_cost_coefficient: float = 0.0,
        obs_keys: Optional[list[str]] = None,
    ):
        if not isinstance(amm_env.model_dynamics, UniswapV3ModelDynamics):
            raise TypeError("ArbitrageurEnvironment requires UniswapV3ModelDynamics")
        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        if speed_cost_coefficient < 0.0:
            raise ValueError("speed_cost_coefficient must be non-negative")

        self.env = amm_env
        self.model_dynamics = amm_env.model_dynamics
        self.num_trajectories = amm_env.num_trajectories
        self.max_speed = float(max_speed)
        self.speed_cost_coefficient = float(speed_cost_coefficient)
        self.obs_keys = obs_keys if obs_keys is not None else DEFAULT_ARBITRAGEUR_OBS_KEYS
        self.lp_agent = lp_agent
        self.lp_agent_factory = lp_agent_factory or (lambda env: NoOpLPAgent(env))
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

    @property
    def step_size(self):
        return self.env.step_size

    @property
    def state(self):
        return self.env.state

    def reset(self, seed: int = None, options: dict = None):
        state, info = self.env.reset(seed=seed, options=options)
        self.cumulative_arb_pnl = np.zeros(self.num_trajectories, dtype=np.float64)
        if self.lp_agent is None:
            self.lp_agent = self.lp_agent_factory(self.env)

        lp_action = self.lp_agent.get_action(state)
        zero_arrivals = np.zeros((self.num_trajectories, 2), dtype=bool)
        self.model_dynamics.update_state(zero_arrivals, lp_action)
        self.env._compute_derived_obs()
        return self._flatten_obs(self.env.state), info

    def step(self, action: np.ndarray):
        trading_speeds = self._prepare_action(action)
        observed_midprice = self.env.state[ASSET_PRICE_KEY].copy()

        arb_info = self.model_dynamics.execute_liquidity_taker_speeds(
            trading_speeds,
            step_size=self.step_size,
        )
        self.env._compute_derived_obs()

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

        lp_action = self.lp_agent.get_action(self.env.state)
        _, _, terminated, truncated, _ = self.env.step(lp_action)
        obs = self._flatten_obs(self.env.state)
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
                "asset_price": self.env.state[ASSET_PRICE_KEY].copy(),
                "pool_price": (self.env.state[POOL_SQRT_PRICE_KEY] ** 2).copy(),
                "time": self.env.state[TIME_KEY].copy(),
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
    lp_lower_offset: int = -10,
    lp_upper_offset: int = 10,
    lp_policy: str = LP_POLICY_NONE,
    max_speed: float = np.inf,
    speed_cost_coefficient: float = 0.0,
    price_impact_model: PriceImpactModel = None,
) -> ArbitrageurEnvironment:
    """Build a SAiFE arbitrage speed-control environment."""
    if lp_policy not in {LP_POLICY_NONE, LP_POLICY_DEPLOY_ONCE}:
        raise ValueError(
            f"lp_policy must be '{LP_POLICY_NONE}' or '{LP_POLICY_DEPLOY_ONCE}', "
            f"got {lp_policy!r}"
        )

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
    amm_env = AMMEnvironment(
        terminal_time=terminal_time,
        n_steps=n_steps,
        initial_wealth=initial_wealth,
        reward_function=PnL(),
        model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        initial_pool_price=initial_pool_price,
        seed=seed,
    )
    if lp_policy == LP_POLICY_DEPLOY_ONCE:
        lp_agent_factory = lambda env: DeployOnceAgent(
            env,
            lower_offset=lp_lower_offset,
            upper_offset=lp_upper_offset,
        )
    else:
        lp_agent_factory = lambda env: NoOpLPAgent(env)

    return ArbitrageurEnvironment(
        amm_env,
        lp_agent_factory=lp_agent_factory,
        max_speed=max_speed,
        speed_cost_coefficient=speed_cost_coefficient,
    )
