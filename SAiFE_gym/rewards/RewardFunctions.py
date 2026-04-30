import abc
from typing import Union

import numpy as np
from SAiFE_gym.gym.index_names import (
    PORTFOLIO_VALUE_KEY, LP_TOKEN0_AMOUNT_KEY, TIME_KEY,
)


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

    Reads `PORTFOLIO_VALUE_KEY` from state, which the env computes in
    `AMMEnvironment._compute_derived_obs`.
    """

    def calculate(
        self, current_state: dict, action: np.ndarray,
        next_state: dict, is_terminal_step: bool = False
    ) -> np.ndarray:
        return next_state[PORTFOLIO_VALUE_KEY] - current_state[PORTFOLIO_VALUE_KEY]

    def reset(self, initial_state: Union[np.ndarray, dict]):
        pass


class ExponentialUtility(RewardFunction):
    """Terminal CARA utility on LP portfolio value.

    Gives 0 per-step reward and -exp(-a * W_T) at the terminal step,
    where W_T is the LP's mark-to-market portfolio value.
    """

    def __init__(self, risk_aversion: float = 0.1):
        self.risk_aversion = risk_aversion

    def calculate(
        self, current_state: dict, action: np.ndarray,
        next_state: dict, is_terminal_step: bool = False
    ) -> np.ndarray:
        if is_terminal_step:
            return -np.exp(-self.risk_aversion * next_state[PORTFOLIO_VALUE_KEY])
        return np.zeros_like(next_state[PORTFOLIO_VALUE_KEY])

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
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.inventory_exponent = inventory_exponent

    def calculate(
        self, current_state: dict, action: np.ndarray,
        next_state: dict, is_terminal_step: bool = False
    ) -> np.ndarray:
        dt = next_state[TIME_KEY] - current_state[TIME_KEY]
        inventory = next_state[LP_TOKEN0_AMOUNT_KEY]
        inventory_penalty = inventory ** self.inventory_exponent

        pnl_reward = next_state[PORTFOLIO_VALUE_KEY] - current_state[PORTFOLIO_VALUE_KEY]
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
