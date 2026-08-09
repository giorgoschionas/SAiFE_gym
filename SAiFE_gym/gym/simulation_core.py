import numpy as np

from SAiFE_gym.gym.ModelDynamics import ModelDynamics
from SAiFE_gym.gym.helpers.AMM_utils import price_to_tick
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    FEES0_KEY,
    FEES1_KEY,
    GAS_COST_KEY,
    INITIAL_WEALTH_KEY,
    LP_ALPHA_KEY,
    LP_COLLECTED_FEES0_KEY,
    LP_COLLECTED_FEES1_KEY,
    LP_EVER_DEPLOYED_KEY,
    LP_FEE_SNAPSHOT0_KEY,
    LP_FEE_SNAPSHOT1_KEY,
    LP_LIQUIDITY_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_TOKEN0_AMOUNT_KEY,
    LP_UNCLAIMED_FEES0_KEY,
    LP_UNCLAIMED_FEES1_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
    PORTFOLIO_VALUE_KEY,
    RECENT_REALIZED_VOLATILITY_KEY,
    TIME_KEY,
)


def compute_derived_obs(state: dict, model_dynamics: ModelDynamics) -> None:
    """Compute derived LP observation keys in-place on ``state``."""
    state[POOL_SQRT_PRICE_KEY] = model_dynamics.get_pool_sqrt_price(state)
    unclaimed0, unclaimed1 = model_dynamics.compute_unclaimed_fees(state)
    state[LP_UNCLAIMED_FEES0_KEY] = unclaimed0
    state[LP_UNCLAIMED_FEES1_KEY] = unclaimed1
    state[PORTFOLIO_VALUE_KEY] = model_dynamics.compute_portfolio_value(state)
    state[LP_ALPHA_KEY] = model_dynamics.compute_lp_alpha(state)
    state[LP_TOKEN0_AMOUNT_KEY] = model_dynamics.compute_lp_token0_amount(state)


def create_uniswap_v3_initial_state(
    model_dynamics: ModelDynamics,
    num_trajectories: int,
    initial_wealth: float,
    initial_pool_price: float = None,
) -> dict:
    """Create the shared vectorized Uniswap-v3 initial state."""
    initial_price = model_dynamics.initial_price
    pool_price = initial_pool_price if initial_pool_price is not None else initial_price
    pool_tick = price_to_tick(pool_price)

    num_ticks = model_dynamics.num_ticks
    model_dynamics.tick_lower_global = pool_tick - num_ticks // 2
    model_dynamics._build_sqrt_grid()

    initial_sqrt_price = model_dynamics.sqrt_grid[
        pool_tick - model_dynamics.tick_lower_global
    ]
    initial_liquidity = 100000.0

    return {
        POOL_SQRT_PRICE_KEY: np.full(
            num_trajectories, initial_sqrt_price, dtype=np.float64
        ),
        POOL_CURRENT_TICK_KEY: np.full(
            num_trajectories, pool_tick, dtype=np.int64
        ),
        POOL_LIQUIDITY_ARRAY_KEY: np.full(
            (num_trajectories, num_ticks), initial_liquidity, dtype=np.float64
        ),
        FEES0_KEY: np.zeros((num_trajectories, num_ticks), dtype=np.float64),
        FEES1_KEY: np.zeros((num_trajectories, num_ticks), dtype=np.float64),
        LP_LIQUIDITY_KEY: np.zeros(num_trajectories, dtype=np.float64),
        LP_TICK_LOWER_KEY: np.full(
            num_trajectories, pool_tick - model_dynamics.tau, dtype=np.int64
        ),
        LP_TICK_UPPER_KEY: np.full(
            num_trajectories, pool_tick + model_dynamics.tau, dtype=np.int64
        ),
        LP_COLLECTED_FEES0_KEY: np.zeros(num_trajectories, dtype=np.float64),
        LP_COLLECTED_FEES1_KEY: np.zeros(num_trajectories, dtype=np.float64),
        LP_UNCLAIMED_FEES0_KEY: np.zeros(num_trajectories, dtype=np.float64),
        LP_UNCLAIMED_FEES1_KEY: np.zeros(num_trajectories, dtype=np.float64),
        LP_FEE_SNAPSHOT0_KEY: np.zeros(num_trajectories, dtype=np.float64),
        LP_FEE_SNAPSHOT1_KEY: np.zeros(num_trajectories, dtype=np.float64),
        LP_EVER_DEPLOYED_KEY: np.zeros(num_trajectories, dtype=bool),
        ASSET_PRICE_KEY: np.full(
            num_trajectories, initial_price, dtype=np.float64
        ),
        TIME_KEY: np.zeros(num_trajectories, dtype=np.float64),
        GAS_COST_KEY: np.full(
            num_trajectories, model_dynamics.gas_cost, dtype=np.float64
        ),
        INITIAL_WEALTH_KEY: np.full(
            num_trajectories, initial_wealth, dtype=np.float64
        ),
        PORTFOLIO_VALUE_KEY: np.full(
            num_trajectories, initial_wealth, dtype=np.float64
        ),
        LP_ALPHA_KEY: np.zeros(num_trajectories, dtype=np.float64),
        LP_TOKEN0_AMOUNT_KEY: np.zeros(num_trajectories, dtype=np.float64),
        RECENT_REALIZED_VOLATILITY_KEY: np.zeros(
            num_trajectories, dtype=np.float64
        ),
    }


def reset_stochastic_processes(model_dynamics: ModelDynamics) -> None:
    if model_dynamics.midprice_model:
        model_dynamics.midprice_model.reset()
    if model_dynamics.arrival_model:
        model_dynamics.arrival_model.reset()


def reset_model_state(
    model_dynamics: ModelDynamics,
    initial_state: dict,
    num_trajectories: int,
) -> None:
    model_dynamics.state = {k: v.copy() for k, v in initial_state.items()}
    model_dynamics.last_arrivals = np.zeros((num_trajectories, 2), dtype=bool)


def advance_market_state(
    model_dynamics: ModelDynamics,
    arrivals: np.ndarray,
    action: np.ndarray,
) -> None:
    """Advance midprice and arrival-process state after pool state mutation."""
    md = model_dynamics

    md.midprice_model.update(arrivals, None, action, md.state)
    md.state[ASSET_PRICE_KEY] = md.midprice_model.current_state[:, 0].copy()

    tick_idx = np.clip(
        (md.state[POOL_CURRENT_TICK_KEY] - md.tick_lower_global).astype(np.int64),
        0,
        md.num_ticks - 1,
    )
    active_liq = md.state[POOL_LIQUIDITY_ARRAY_KEY][
        np.arange(md.num_trajectories), tick_idx
    ]
    context = {
        "active_liquidity": active_liq,
        "amm_price": md.state[POOL_SQRT_PRICE_KEY] ** 2,
        "midprice": md.state[ASSET_PRICE_KEY],
        "liquidity_array": md.state[POOL_LIQUIDITY_ARRAY_KEY],
        "current_tick": md.state[POOL_CURRENT_TICK_KEY],
        "tick_lower_global": md.tick_lower_global,
    }
    md.arrival_model.update(arrivals, None, action, context)


def terminated_flags(
    state: dict,
    terminal_time: float,
    step_size: float,
    num_trajectories: int,
) -> np.ndarray:
    done = state[TIME_KEY][0] >= terminal_time - step_size / 2
    return np.full((num_trajectories,), done, dtype=bool)
