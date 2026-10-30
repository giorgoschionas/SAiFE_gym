import abc
from typing import Optional, Union

import numpy as np
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY, PORTFOLIO_VALUE_KEY, LP_TOKEN0_AMOUNT_KEY, TIME_KEY,
    INITIAL_WEALTH_KEY,
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


class DenseExponentialUtility(RewardFunction):
    """Per-step extended CARA utility on cumulative PnL from initial cash.

    For G_t = portfolio_value_t - initial_wealth and a = risk_aversion,
    U(G) = (1 - exp(-a * G)) / a for a != 0, and U(G) = G for a == 0.
    Positive a is risk averse, zero is risk neutral, and negative a is risk
    seeking. Return U(G_next) - U(G_current); with RL gamma=1 and G_0=0,
    episode rewards sum to U(G_T), since U(0)=0. At a=0 this is exactly PnL.
    The terminal step uses the same increment, without an extra terminal bonus.

    For a > 0, rewards equal the former -exp(-a * G) increments divided by a:
    strategy rankings are preserved, but reward magnitudes change.
    Portfolio value already includes fees and reflects paid rebalancing costs.
    ``risk_aversion`` is in inverse token1 units: choose it for the PnL scale
    (e.g. 1e-6 for initial wealth 1e6 gives an exponent of 1 at total loss).
    There is no automatic normalization, clipping, or extra inventory penalty.
    Unrepresentable rewards raise FloatingPointError rather than silently
    clipping the utility and changing the objective.
    """

    def __init__(self, risk_aversion: float):
        if not np.isfinite(risk_aversion):
            raise ValueError("risk_aversion must be finite")
        self.risk_aversion = float(risk_aversion)

    def calculate(
        self, current_state: dict, action: np.ndarray,
        next_state: dict, is_terminal_step: bool = False
    ) -> np.ndarray:
        current_value = np.asarray(current_state[PORTFOLIO_VALUE_KEY], dtype=np.float64)
        next_value = np.asarray(next_state[PORTFOLIO_VALUE_KEY], dtype=np.float64)
        initial_wealth = np.asarray(current_state[INITIAL_WEALTH_KEY], dtype=np.float64)
        if not (
            np.all(np.isfinite(current_value))
            and np.all(np.isfinite(next_value))
            and np.all(np.isfinite(initial_wealth))
        ):
            raise ValueError("DenseExponentialUtility requires finite wealth values")

        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                pnl_change = next_value - current_value
                if self.risk_aversion == 0.0:
                    return pnl_change

                # Factor out the larger exponential: it occurs at the smaller
                # wealth for a > 0 and the larger wealth for a < 0. The reward
                # has the sign of the PnL change for BOTH risk profiles.
                anchor = (
                    np.minimum(current_value, next_value)
                    if self.risk_aversion > 0.0
                    else np.maximum(current_value, next_value)
                )
                anchor = np.where(pnl_change == 0.0, initial_wealth, anchor)
                exponent = -self.risk_aversion * (anchor - initial_wealth)
                absolute_risk = abs(self.risk_aversion)
                # An infinite scaled gap has the well-defined limit expm1(-inf)=-1.
                with np.errstate(over="ignore"):
                    scaled_gap = absolute_risk * np.abs(pnl_change)
                difference = -np.expm1(-scaled_gap)
                small_gap = scaled_gap <= 1.0
                # For tiny a, divide by a analytically before taking logs. This
                # retains the PnL limit even when a * abs(PnL change) underflows.
                ratio = np.divide(
                    difference, scaled_gap, out=np.ones_like(scaled_gap),
                    where=small_gap & (scaled_gap != 0.0),
                )
                with np.errstate(divide="ignore"):
                    log_gap = np.where(
                        small_gap,
                        np.log(np.abs(pnl_change)) + np.log(ratio),
                        np.log(difference) - np.log(absolute_risk),
                    )
                    # Log space permits finite increments even if the utility
                    # level itself overflows. An unchanged portfolio returns 0.
                    magnitude = np.exp(exponent + log_gap)
                return np.sign(pnl_change) * magnitude
        except FloatingPointError as error:
            raise FloatingPointError(
                "DenseExponentialUtility cannot represent this reward; "
                "check that wealth is finite and reduce abs(risk_aversion) to "
                "match the token1 PnL scale."
            ) from error

    def reset(self, initial_state: Union[np.ndarray, dict]):
        # The fixed per-trajectory baseline is already carried in each state.
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
