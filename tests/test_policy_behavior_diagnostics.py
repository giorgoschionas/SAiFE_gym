import numpy as np
import pytest

from experiments.policy_behavior_diagnostics import (
    BEHAVIOR_DIAGNOSTIC_COLUMNS,
    PolicyBehaviorAccumulator,
    cash_behavior_diagnostics,
)
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


def _state(
    *,
    portfolio_value,
    liquidity,
    ever_deployed,
    collected_fees0,
    collected_fees1,
    tick_lower,
    tick_upper,
    current_tick,
    gas_cost=(5.0, 5.0),
    asset_price=(100.0, 100.0),
):
    return {
        PORTFOLIO_VALUE_KEY: np.asarray(portfolio_value, dtype=np.float64),
        LP_LIQUIDITY_KEY: np.asarray(liquidity, dtype=np.float64),
        LP_EVER_DEPLOYED_KEY: np.asarray(ever_deployed, dtype=bool),
        LP_COLLECTED_FEES0_KEY: np.asarray(
            collected_fees0, dtype=np.float64
        ),
        LP_COLLECTED_FEES1_KEY: np.asarray(
            collected_fees1, dtype=np.float64
        ),
        LP_TICK_LOWER_KEY: np.asarray(tick_lower, dtype=np.int64),
        LP_TICK_UPPER_KEY: np.asarray(tick_upper, dtype=np.int64),
        POOL_CURRENT_TICK_KEY: np.asarray(current_tick, dtype=np.int64),
        GAS_COST_KEY: np.asarray(gas_cost, dtype=np.float64),
        ASSET_PRICE_KEY: np.asarray(asset_price, dtype=np.float64),
    }


def test_accumulator_reports_vectorized_policy_behavior_and_reward_decomposition():
    initial = _state(
        portfolio_value=(100.0, 100.0),
        liquidity=(0.0, 0.0),
        ever_deployed=(False, False),
        collected_fees0=(0.0, 0.0),
        collected_fees1=(0.0, 0.0),
        tick_lower=(-5, -5),
        tick_upper=(5, 5),
        current_tick=(0, 0),
    )
    after_deployment = _state(
        portfolio_value=(90.0, 100.0),
        liquidity=(10.0, 0.0),
        ever_deployed=(True, False),
        collected_fees0=(1.0, 0.0),
        collected_fees1=(2.0, 0.0),
        tick_lower=(-2, -5),
        tick_upper=(2, 5),
        current_tick=(0, 0),
    )
    final = _state(
        portfolio_value=(0.0, 100.0),
        liquidity=(0.0, 0.0),
        ever_deployed=(True, False),
        collected_fees0=(1.0, 0.0),
        collected_fees1=(2.0, 0.0),
        tick_lower=(-1, -5),
        tick_upper=(1, 5),
        current_tick=(0, 0),
    )

    accumulator = PolicyBehaviorAccumulator()
    accumulator.begin_episode(2)

    first_action = np.array([
        [-2.0, 2.0, -1.0],
        [-5.0, 5.0, 1.0],
    ])
    first_snapshot = accumulator.before_step(initial, first_action)
    accumulator.after_step(
        first_snapshot,
        after_deployment,
        rewards=np.array([-12.0, 0.0]),
    )

    second_action = np.array([
        [-1.0, 1.0, -1.0],
        [-5.0, 5.0, 1.0],
    ])
    second_snapshot = accumulator.before_step(after_deployment, second_action)
    accumulator.after_step(
        second_snapshot,
        final,
        rewards=np.array([-93.0, 0.0]),
    )
    accumulator.finish_episode(final)

    diagnostics = accumulator.summarize()

    assert set(diagnostics) == set(BEHAVIOR_DIAGNOSTIC_COLUMNS)
    assert diagnostics["never_deployed_fraction"] == 0.5
    assert diagnostics["bankruptcy_fraction"] == 0.5
    assert diagnostics["hold_action_fraction"] == 0.5
    assert diagnostics["rebalance_action_fraction"] == 0.5
    assert diagnostics["mean_rebalances_after_deployment_per_path"] == 0.5
    assert diagnostics["mean_selected_range_width_ticks"] == 3.0
    assert diagnostics["mean_active_range_width_ticks"] == 4.0
    assert diagnostics["mean_gas_spend_per_path"] == 2.5
    assert diagnostics["mean_fee_income_token1_per_path"] == 51.0
    assert diagnostics["mean_pnl_per_path"] == -50.0
    assert diagnostics["mean_inventory_penalty_per_path"] == 2.5
    assert diagnostics["mean_in_range_fraction_among_deployed_paths"] == 1.0
    assert (
        diagnostics["mean_pnl_per_path"]
        - diagnostics["mean_inventory_penalty_per_path"]
    ) == -52.5


def test_cash_diagnostics_identify_never_deployed_hold_policy():
    diagnostics = cash_behavior_diagnostics()

    assert set(diagnostics) == set(BEHAVIOR_DIAGNOSTIC_COLUMNS)
    assert diagnostics["never_deployed_fraction"] == 1.0
    assert diagnostics["hold_action_fraction"] == 1.0
    assert diagnostics["rebalance_action_fraction"] == 0.0
    assert diagnostics["mean_pnl_per_path"] == 0.0
    assert diagnostics["mean_inventory_penalty_per_path"] == 0.0


def test_accumulator_requires_completed_episode_before_summary():
    accumulator = PolicyBehaviorAccumulator()

    with pytest.raises(RuntimeError, match="completed episode"):
        accumulator.summarize()

    accumulator.begin_episode(2)
    with pytest.raises(RuntimeError, match="finish the active episode"):
        accumulator.summarize()
