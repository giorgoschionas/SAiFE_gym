import abc
from typing import Union

import numpy as np
from SAiFE_gym.gym.index_names import (
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    POOL_SQRT_PRICE_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_EVER_DEPLOYED_KEY,
)
from SAiFE_gym.gym.helpers.AMM_utils import get_position_value_vec


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _tick_to_sqrt_price(tick: np.ndarray, exponential_value: float) -> np.ndarray:
    """Convert tick values to sqrt(price)."""
    return np.sqrt(exponential_value ** tick.astype(np.float64))


def _token0_amount(state: dict, exponential_value: float) -> np.ndarray:
    """Compute LP's token0 holdings (risky asset exposure).

    For a Uniswap V3 position with liquidity L in range [sqrt_p_lower, sqrt_p_upper]:
        - In range:    x = L * (1/sqrt_p - 1/sqrt_p_upper)
        - Below range: x = L * (1/sqrt_p_lower - 1/sqrt_p_upper)  (100% token0)
        - Above range: x = 0  (100% token1)

    This is the LP's analog of market maker inventory: higher token0 exposure means
    more directional risk from price movements (the LP accumulates the depreciating
    token as price moves against them).
    """
    lp_liq = state[LP_LIQUIDITY_KEY]
    has_position = lp_liq > 0

    sqrt_p = state[POOL_SQRT_PRICE_KEY]
    sqrt_p_lower = _tick_to_sqrt_price(state[LP_TICK_LOWER_KEY], exponential_value)
    sqrt_p_upper = _tick_to_sqrt_price(state[LP_TICK_UPPER_KEY], exponential_value)

    above = sqrt_p >= sqrt_p_upper
    below = sqrt_p <= sqrt_p_lower

    x_in = lp_liq * (1.0 / sqrt_p - 1.0 / sqrt_p_upper)
    x_below = lp_liq * (1.0 / sqrt_p_lower - 1.0 / sqrt_p_upper)

    x = np.where(above, 0.0, np.where(below, x_below, x_in))
    return np.where(has_position, x, 0.0)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self, current_state: Union[np.ndarray, dict], action: np.ndarray,
        next_state: Union[np.ndarray, dict], is_terminal_step: bool = False
    ) -> Union[float, np.ndarray]:
        pass

    @abc.abstractmethod
    def reset(self, initial_state: Union[np.ndarray, dict]):
        pass


# ---------------------------------------------------------------------------
# Concrete reward functions
# ---------------------------------------------------------------------------

class PnL(RewardFunction):
    """Mark-to-market PnL reward: change in LP portfolio value between steps.

    Portfolio value is the position's mark-to-market value via get_position_value_vec.
    Fees earned are already folded into LP_LIQUIDITY during rebalancing, so no
    separate fee term is needed. When LP_LIQUIDITY == 0 (before first deployment),
    portfolio value equals initial_wealth.
    """

    def __init__(self, exponential_value: float = 1.0001, initial_wealth: float = 1e6):
        self.exponential_value = exponential_value
        self.initial_wealth = initial_wealth

    def _portfolio_value(self, state: dict) -> np.ndarray:
        lp_liq = state[LP_LIQUIDITY_KEY]
        has_position = lp_liq > 0

        sqrt_p_lower = _tick_to_sqrt_price(state[LP_TICK_LOWER_KEY], self.exponential_value)
        sqrt_p_upper = _tick_to_sqrt_price(state[LP_TICK_UPPER_KEY], self.exponential_value)

        pos_value = get_position_value_vec(
            lp_liq, state[ASSET_PRICE_KEY], state[POOL_SQRT_PRICE_KEY],
            sqrt_p_lower, sqrt_p_upper
        )

        # Trajectories with lp_liq == 0 are either:
        #   - pre-deployment (ever_deployed=False): use initial_wealth as the cash baseline
        #   - bankrupt      (ever_deployed=True):  wealth is genuinely 0
        ever_deployed = state.get(LP_EVER_DEPLOYED_KEY, np.zeros_like(lp_liq, dtype=bool))
        no_position_value = np.where(ever_deployed, 0.0, self.initial_wealth)
        return np.where(has_position, pos_value, no_position_value)

    def calculate(
        self, current_state: dict, action: np.ndarray,
        next_state: dict, is_terminal_step: bool = False
    ) -> np.ndarray:
        return self._portfolio_value(next_state) - self._portfolio_value(current_state)

    def reset(self, initial_state: Union[np.ndarray, dict]):
        pass


class ExponentialUtility(RewardFunction):
    """Terminal CARA utility on LP portfolio value.

    Gives 0 per-step reward and -exp(-a * W_T) at the terminal step,
    where W_T is the LP's mark-to-market portfolio value.
    """

    def __init__(self, risk_aversion: float = 0.1,
                 exponential_value: float = 1.0001, initial_wealth: float = 1e6):
        self.risk_aversion = risk_aversion
        self._pnl = PnL(exponential_value, initial_wealth)

    def calculate(
        self, current_state: dict, action: np.ndarray,
        next_state: dict, is_terminal_step: bool = False
    ) -> np.ndarray:
        if is_terminal_step:
            return -np.exp(-self.risk_aversion * self._pnl._portfolio_value(next_state))
        return np.zeros_like(next_state[LP_LIQUIDITY_KEY])

    def reset(self, initial_state: Union[np.ndarray, dict]):
        pass


class RunningInventoryPenalty(RewardFunction):
    """PnL with running penalty on LP's risky asset exposure.

    reward = PnL_t - phi * dt * x_t^p  -  alpha * 1_{terminal} * x_t^p

    where x_t is the LP's token0 amount (risky asset "inventory"),
    phi is per_step_inventory_aversion, and alpha is terminal_inventory_aversion.

    The LP's token0 holdings are the direct analog of market maker inventory:
      - As price drops, the LP accumulates more token0 (buys the depreciating asset)
      - As price rises, the LP sheds token0 (sells the appreciating asset)
    Penalizing x_t^p encourages ranges that reduce directional exposure,
    e.g. wider ranges or ranges shifted above current price.

    """

    def __init__(
        self,
        per_step_inventory_aversion: float = 0.01,
        terminal_inventory_aversion: float = 0.0,
        inventory_exponent: float = 2.0,
        exponential_value: float = 1.0001,
        initial_wealth: float = 1e6,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.inventory_exponent = inventory_exponent
        self._pnl = PnL(exponential_value, initial_wealth)
        self.exponential_value = exponential_value

    def calculate(
        self, current_state: dict, action: np.ndarray,
        next_state: dict, is_terminal_step: bool = False
    ) -> np.ndarray:
        dt = next_state[TIME_KEY] - current_state[TIME_KEY]
        inventory = _token0_amount(next_state, self.exponential_value)
        inventory_penalty = inventory ** self.inventory_exponent

        pnl_reward = self._pnl.calculate(
            current_state, action, next_state, is_terminal_step
        )
        running_penalty = self.per_step_inventory_aversion * dt * inventory_penalty
        terminal_penalty = (
            self.terminal_inventory_aversion * inventory_penalty
            if is_terminal_step else 0.0
        )

        return pnl_reward - running_penalty - terminal_penalty

    def reset(self, initial_state: Union[np.ndarray, dict]):
        pass


# Cartea-Jaimungal criterion is the same as inventory-adjusted PnL
CjCriterion = RunningInventoryPenalty
