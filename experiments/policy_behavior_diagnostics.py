"""Vectorized policy-behavior diagnostics for LP evaluation episodes."""

from dataclasses import dataclass

import numpy as np

from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    GAS_COST_KEY,
    LP_COLLECTED_FEES0_KEY,
    LP_COLLECTED_FEES1_KEY,
    LP_EVER_DEPLOYED_KEY,
    LP_LIQUIDITY_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    POOL_CURRENT_TICK_KEY,
    PORTFOLIO_VALUE_KEY,
)


BEHAVIOR_DIAGNOSTICS_SCHEMA_VERSION = 1

BEHAVIOR_DIAGNOSTIC_COLUMNS = (
    "never_deployed_fraction",
    "bankruptcy_fraction",
    "hold_action_fraction",
    "rebalance_action_fraction",
    "mean_rebalances_after_deployment_per_path",
    "mean_selected_range_width_ticks",
    "mean_active_range_width_ticks",
    "mean_gas_spend_per_path",
    "mean_fee_income_token1_per_path",
    "mean_pnl_per_path",
    "mean_inventory_penalty_per_path",
    "mean_in_range_fraction_among_deployed_paths",
)

FRACTION_BEHAVIOR_DIAGNOSTIC_COLUMNS = {
    "never_deployed_fraction",
    "bankruptcy_fraction",
    "hold_action_fraction",
    "rebalance_action_fraction",
    "mean_in_range_fraction_among_deployed_paths",
}

NONNEGATIVE_BEHAVIOR_DIAGNOSTIC_COLUMNS = (
    set(BEHAVIOR_DIAGNOSTIC_COLUMNS)
    - {"mean_pnl_per_path"}
)


@dataclass(frozen=True)
class BehaviorStepSnapshot:
    """Pre-transition values needed for exact post-transition accounting."""

    portfolio_value: np.ndarray
    collected_fees0: np.ndarray
    collected_fees1: np.ndarray


class PolicyBehaviorAccumulator:
    """Accumulate vectorized behavior statistics across evaluation episodes.

    Action fractions use every trajectory-step as a decision. A rebalance action
    includes an initial deployment attempt; ``rebalances_after_deployment`` only
    counts rebalance decisions made while a live position already exists.
    """

    def __init__(self) -> None:
        self.num_paths = 0
        self.action_decisions = 0
        self.hold_actions = 0
        self.rebalance_actions = 0
        self.rebalances_after_deployment = 0
        self.selected_range_width_total = 0.0
        self.selected_range_count = 0
        self.active_range_width_total = 0.0
        self.active_range_count = 0
        self.gas_spend_total = 0.0
        self.fee_income_token1_total = 0.0
        self.pnl_total = 0.0
        self.inventory_penalty_total = 0.0
        self.never_deployed_paths = 0
        self.bankrupt_paths = 0
        self.deployed_paths = 0
        self.in_range_fraction_sum = 0.0
        self._deployed_step_counts: np.ndarray | None = None
        self._in_range_step_counts: np.ndarray | None = None

    def begin_episode(self, num_trajectories: int) -> None:
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be positive")
        if self._deployed_step_counts is not None:
            raise RuntimeError("finish the current episode before beginning another")
        self._deployed_step_counts = np.zeros(num_trajectories, dtype=np.int64)
        self._in_range_step_counts = np.zeros(num_trajectories, dtype=np.int64)

    def before_step(self, state: dict, action: np.ndarray) -> BehaviorStepSnapshot:
        num_trajectories = self._require_active_episode()
        action = np.asarray(action)
        if action.shape != (num_trajectories, 3):
            raise ValueError(
                f"action must have shape ({num_trajectories}, 3), got {action.shape}"
            )

        portfolio_value = np.asarray(
            state[PORTFOLIO_VALUE_KEY], dtype=np.float64
        )
        has_position = np.asarray(state[LP_LIQUIDITY_KEY]) > 0.0
        rebalance = action[:, 2] <= 0.0
        hold = ~rebalance

        self.action_decisions += num_trajectories
        self.hold_actions += int(np.count_nonzero(hold))
        self.rebalance_actions += int(np.count_nonzero(rebalance))
        self.rebalances_after_deployment += int(
            np.count_nonzero(rebalance & has_position)
        )

        selected_widths = action[rebalance, 1] - action[rebalance, 0]
        if selected_widths.size:
            self.selected_range_width_total += float(np.sum(selected_widths))
            self.selected_range_count += int(selected_widths.size)

        gas_cost = np.broadcast_to(
            np.asarray(state[GAS_COST_KEY], dtype=np.float64),
            (num_trajectories,),
        )
        charged = rebalance & has_position
        self.gas_spend_total += float(
            np.sum(np.minimum(gas_cost[charged], portfolio_value[charged]))
        )

        return BehaviorStepSnapshot(
            portfolio_value=portfolio_value.copy(),
            collected_fees0=np.asarray(
                state[LP_COLLECTED_FEES0_KEY], dtype=np.float64
            ).copy(),
            collected_fees1=np.asarray(
                state[LP_COLLECTED_FEES1_KEY], dtype=np.float64
            ).copy(),
        )

    def after_step(
        self,
        snapshot: BehaviorStepSnapshot,
        next_state: dict,
        rewards: np.ndarray,
    ) -> None:
        num_trajectories = self._require_active_episode()
        rewards = np.asarray(rewards, dtype=np.float64)
        if rewards.shape != (num_trajectories,):
            raise ValueError(
                f"rewards must have shape ({num_trajectories},), got {rewards.shape}"
            )

        next_portfolio_value = np.asarray(
            next_state[PORTFOLIO_VALUE_KEY], dtype=np.float64
        )
        pnl = next_portfolio_value - snapshot.portfolio_value
        self.pnl_total += float(np.sum(pnl))
        self.inventory_penalty_total += float(np.sum(pnl - rewards))

        fee0_delta = (
            np.asarray(next_state[LP_COLLECTED_FEES0_KEY], dtype=np.float64)
            - snapshot.collected_fees0
        )
        fee1_delta = (
            np.asarray(next_state[LP_COLLECTED_FEES1_KEY], dtype=np.float64)
            - snapshot.collected_fees1
        )
        asset_price = np.asarray(next_state[ASSET_PRICE_KEY], dtype=np.float64)
        self.fee_income_token1_total += float(
            np.sum(fee0_delta * asset_price + fee1_delta)
        )

        deployed = np.asarray(next_state[LP_LIQUIDITY_KEY]) > 0.0
        widths = (
            np.asarray(next_state[LP_TICK_UPPER_KEY], dtype=np.float64)
            - np.asarray(next_state[LP_TICK_LOWER_KEY], dtype=np.float64)
        )
        self.active_range_width_total += float(np.sum(widths[deployed]))
        self.active_range_count += int(np.count_nonzero(deployed))

        current_tick = np.asarray(next_state[POOL_CURRENT_TICK_KEY])
        in_range = (
            deployed
            & (current_tick >= np.asarray(next_state[LP_TICK_LOWER_KEY]))
            & (current_tick < np.asarray(next_state[LP_TICK_UPPER_KEY]))
        )
        self._deployed_step_counts += deployed
        self._in_range_step_counts += in_range

    def finish_episode(self, final_state: dict) -> None:
        num_trajectories = self._require_active_episode()
        ever_deployed = np.asarray(final_state[LP_EVER_DEPLOYED_KEY], dtype=bool)
        portfolio_value = np.asarray(
            final_state[PORTFOLIO_VALUE_KEY], dtype=np.float64
        )
        live_position = np.asarray(final_state[LP_LIQUIDITY_KEY]) > 0.0

        self.num_paths += num_trajectories
        self.never_deployed_paths += int(np.count_nonzero(~ever_deployed))
        self.bankrupt_paths += int(
            np.count_nonzero(ever_deployed & ~live_position & (portfolio_value <= 0.0))
        )

        deployed_paths = self._deployed_step_counts > 0
        self.deployed_paths += int(np.count_nonzero(deployed_paths))
        if np.any(deployed_paths):
            self.in_range_fraction_sum += float(
                np.sum(
                    self._in_range_step_counts[deployed_paths]
                    / self._deployed_step_counts[deployed_paths]
                )
            )

        self._deployed_step_counts = None
        self._in_range_step_counts = None

    def summarize(self) -> dict[str, float]:
        if self._deployed_step_counts is not None:
            raise RuntimeError("finish the active episode before summarizing")
        if self.num_paths < 1:
            raise RuntimeError("at least one completed episode is required")

        diagnostics = {
            "never_deployed_fraction": self.never_deployed_paths / self.num_paths,
            "bankruptcy_fraction": self.bankrupt_paths / self.num_paths,
            "hold_action_fraction": self._safe_ratio(
                self.hold_actions, self.action_decisions
            ),
            "rebalance_action_fraction": self._safe_ratio(
                self.rebalance_actions, self.action_decisions
            ),
            "mean_rebalances_after_deployment_per_path": (
                self.rebalances_after_deployment / self.num_paths
            ),
            "mean_selected_range_width_ticks": self._safe_ratio(
                self.selected_range_width_total, self.selected_range_count
            ),
            "mean_active_range_width_ticks": self._safe_ratio(
                self.active_range_width_total, self.active_range_count
            ),
            "mean_gas_spend_per_path": self.gas_spend_total / self.num_paths,
            "mean_fee_income_token1_per_path": (
                self.fee_income_token1_total / self.num_paths
            ),
            "mean_pnl_per_path": self.pnl_total / self.num_paths,
            "mean_inventory_penalty_per_path": (
                self.inventory_penalty_total / self.num_paths
            ),
            "mean_in_range_fraction_among_deployed_paths": self._safe_ratio(
                self.in_range_fraction_sum, self.deployed_paths
            ),
        }
        _validate_behavior_diagnostics(diagnostics)
        return diagnostics

    def _require_active_episode(self) -> int:
        if self._deployed_step_counts is None:
            raise RuntimeError("begin_episode must be called first")
        return int(self._deployed_step_counts.shape[0])

    @staticmethod
    def _safe_ratio(numerator: float, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0


def zero_behavior_diagnostics() -> dict[str, float]:
    return {name: 0.0 for name in BEHAVIOR_DIAGNOSTIC_COLUMNS}


def cash_behavior_diagnostics() -> dict[str, float]:
    diagnostics = zero_behavior_diagnostics()
    diagnostics["never_deployed_fraction"] = 1.0
    diagnostics["hold_action_fraction"] = 1.0
    return diagnostics


def _validate_behavior_diagnostics(diagnostics: dict[str, float]) -> None:
    if set(diagnostics) != set(BEHAVIOR_DIAGNOSTIC_COLUMNS):
        raise ValueError("behavior diagnostics do not match the required schema")
    for name, value in diagnostics.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        if name in NONNEGATIVE_BEHAVIOR_DIAGNOSTIC_COLUMNS and value < -1e-10:
            raise ValueError(f"{name} must be non-negative, got {value}")
        if name in FRACTION_BEHAVIOR_DIAGNOSTIC_COLUMNS and not -1e-10 <= value <= 1.0 + 1e-10:
            raise ValueError(f"{name} must lie in [0, 1], got {value}")
