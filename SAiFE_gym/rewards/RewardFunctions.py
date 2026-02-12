import abc
from typing import Union

import numpy as np
from SAiFE_gym.gym.index_names import (
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    POOL_SQRT_PRICE_KEY, ASSET_PRICE_KEY
)
from SAiFE_gym.gym.helpers.AMM_utils import get_position_value_vec


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

        lp_lower = state[LP_TICK_LOWER_KEY].astype(np.float64)
        lp_upper = state[LP_TICK_UPPER_KEY].astype(np.float64)
        sqrt_p_lower = np.sqrt(self.exponential_value ** lp_lower)
        sqrt_p_upper = np.sqrt(self.exponential_value ** lp_upper)
        sqrt_p = state[POOL_SQRT_PRICE_KEY]
        external_price = state[ASSET_PRICE_KEY]

        pos_value = get_position_value_vec(
            lp_liq, external_price, sqrt_p, sqrt_p_lower, sqrt_p_upper
        )

        return np.where(has_position, pos_value, self.initial_wealth)

    def calculate(
        self, current_state: dict, action: np.ndarray,
        next_state: dict, is_terminal_step: bool = False
    ) -> np.ndarray:
        return self._portfolio_value(next_state) - self._portfolio_value(current_state)

    def reset(self, initial_state: Union[np.ndarray, dict]):
        pass
