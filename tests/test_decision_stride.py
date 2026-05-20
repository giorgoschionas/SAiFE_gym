"""Verification suite for DecisionStrideEnv / DecisionStrideVecEnv wrappers.

Covers:
  1. stride=1 is a bit-exact passthrough vs the bare env
  2. With a constant-action agent (DeployOnce), stride=5 cumulative PnL matches
     bare-env cumulative PnL
  3. Per-window reward at decision i equals the sum of bare per-step rewards
     for env steps [i*stride, (i+1)*stride)
  4. DecisionStrideVecEnv reward stream matches DecisionStrideEnv reward stream
  5. collect_single_trajectory under stride=5 logs once per env step (n_steps rows)
  6. cumulative_pnl[-1] telescopes to PV_final - PV_initial
  7. HOLD substeps never move LP_TICK_LOWER / LP_TICK_UPPER
  8. Gas is charged only on rebalance decision steps (zero-vol, zero-arrival env)

Run:
    /Users/xarisk/miniconda3/envs/SAiFE/bin/python tests/test_decision_stride.py
"""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, '.')

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.agents.BaselineAgents import DeployOnceAgent
from SAiFE_gym.gym.index_names import (
    PORTFOLIO_VALUE_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
)

_spec = importlib.util.spec_from_file_location('ac', 'notebooks/agent_comparison.py')
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)


def make_env(seed=42, n_steps=40, volatility=0.5, intensity=100.0, gas_cost=5.0, initial_wealth=1000.0):
    step_size = 1.0 / n_steps
    return AMMEnvironment(
        terminal_time=1.0, n_steps=n_steps, num_trajectories=1,
        model_dynamics=UniswapV3ModelDynamics(
            midprice_model=BrownianMotionMidpriceModel(
                step_size=step_size, num_trajectories=1, seed=seed,
                volatility=volatility, initial_price=10.0,
            ),
            arrival_model=PoissonArrivalModel(
                intensity=np.array([intensity, intensity]),
                step_size=step_size, num_trajectories=1, seed=seed + 1,
            ),
            num_trajectories=1, tau=5, num_ticks=200,
            seed=seed + 2, gas_cost=gas_cost,
        ),
        initial_wealth=initial_wealth, seed=seed,
    )


def run_episode(env, agent_fn):
    rewards = []
    state, _ = env.reset()
    while True:
        state, r, term, _, _ = env.step(agent_fn(state))
        rewards.append(r[0])
        if term[0]:
            break
    return np.array(rewards)


def test_1_stride1_passthrough():
    print("=== Test 1: stride=1 is bit-exact passthrough ===")
    bare = make_env(seed=123)
    wrapped = ac.DecisionStrideEnv(make_env(seed=123), stride=1)
    a_bare = DeployOnceAgent(bare, lower_offset=-2, upper_offset=2)
    a_wrap = DeployOnceAgent(wrapped, lower_offset=-2, upper_offset=2)
    r_bare = run_episode(bare, a_bare.get_action)
    r_wrap = run_episode(wrapped, a_wrap.get_action)
    print(f"  bare len={len(r_bare)}, wrapped len={len(r_wrap)}")
    print(f"  max abs diff: {np.max(np.abs(r_bare - r_wrap)):.2e}")
    assert len(r_bare) == len(r_wrap) == 40
    assert np.allclose(r_bare, r_wrap, atol=0.0)
    print("  PASS\n")


def test_2_cumulative_pnl_invariant():
    print("=== Test 2: cumulative PnL (stride=5) == cumulative PnL (bare) for DeployOnce ===")
    bare = make_env(seed=789)
    wrapped = ac.DecisionStrideEnv(make_env(seed=789), stride=5)
    a_bare = DeployOnceAgent(bare, lower_offset=-2, upper_offset=2)
    a_wrap = DeployOnceAgent(wrapped, lower_offset=-2, upper_offset=2)
    r_bare = run_episode(bare, a_bare.get_action)
    r_wrap = run_episode(wrapped, a_wrap.get_action)
    print(f"  bare cumulative: {r_bare.sum():.6f}")
    print(f"  stride=5 cumulative: {r_wrap.sum():.6f}")
    print(f"  bare len: {len(r_bare)}, wrapped len: {len(r_wrap)}")
    assert len(r_wrap) == 8  # 40 / 5
    assert np.isclose(r_bare.sum(), r_wrap.sum(), atol=1e-9)
    print("  PASS\n")


def test_3_per_window_sum():
    print("=== Test 3: per-window reward == sum of per-step rewards in that window ===")
    stride, n_steps, seed = 5, 40, 555
    bare = make_env(seed=seed, n_steps=n_steps)
    wrapped = ac.DecisionStrideEnv(make_env(seed=seed, n_steps=n_steps), stride=stride)
    a_bare = DeployOnceAgent(bare, lower_offset=-2, upper_offset=2)
    a_wrap = DeployOnceAgent(wrapped, lower_offset=-2, upper_offset=2)
    r_bare = run_episode(bare, a_bare.get_action)
    r_wrap = run_episode(wrapped, a_wrap.get_action)
    bare_per_window = r_bare.reshape(-1, stride).sum(axis=1)
    print(f"  bare per-window sums: {bare_per_window}")
    print(f"  wrapper rewards:     {r_wrap}")
    print(f"  max abs diff: {np.max(np.abs(bare_per_window - r_wrap)):.2e}")
    assert np.allclose(bare_per_window, r_wrap, atol=1e-9)
    print("  PASS\n")


def test_4_vec_vs_amm_consistency():
    print("=== Test 4: DecisionStrideVecEnv matches DecisionStrideEnv ===")
    sb = StableBaselinesAMMEnvironment(make_env(seed=222), obs_keys=ac.SB3_OBS_KEYS)
    vec_w = ac.DecisionStrideVecEnv(sb, stride=5)
    amm_w = ac.DecisionStrideEnv(make_env(seed=222), stride=5)

    action = np.array([[-2.0, 2.0, -1.0]], dtype=np.float32)
    hold = np.array([[-2.0, 2.0, +1.0]], dtype=np.float32)
    vec_w.reset(); amm_w.reset()
    vec_r, amm_r = [], []
    for i in range(8):
        act = action if i == 0 else hold
        vec_w.step_async(act)
        _, vr, vd, _ = vec_w.step_wait()
        _, ar, at, _, _ = amm_w.step(act)
        vec_r.append(vr[0]); amm_r.append(ar[0])
        if vd[0]:
            break
    vec_r, amm_r = np.array(vec_r), np.array(amm_r)
    print(f"  Vec rewards: {vec_r}")
    print(f"  AMM rewards: {amm_r}")
    assert np.allclose(vec_r, amm_r, atol=1e-5)
    print("  PASS\n")


def test_5_per_step_logging():
    print("=== Test 5: collect_single_trajectory logs every env step ===")
    env = make_env(seed=1234, n_steps=40)
    agent = DeployOnceAgent(env, lower_offset=-2, upper_offset=2)
    data = ac.collect_single_trajectory(env, agent.get_action, decision_stride=5)
    print(f"  rows: {len(data['time'])} (expect 40)")
    assert len(data['time']) == 40
    # hold_flag pattern: -1 at decision 0 (initial deploy), +1 at all subsequent
    # decisions (DeployOnce holds), +1 at HOLD substeps. So total: one -1.
    n_minus_one = int(np.sum(data['hold_flag'] == -1.0))
    print(f"  hold_flag == -1 count: {n_minus_one} (expect 1 — initial deploy only)")
    assert n_minus_one == 1
    # Position bounds: constant after deploy
    assert len(np.unique(data['position_lower_price'])) == 1
    assert len(np.unique(data['position_upper_price'])) == 1
    print("  PASS\n")


def test_6_cum_pnl_telescopes():
    print("=== Test 6: cumulative_pnl[-1] == PV_final - PV_initial ===")
    env = make_env(seed=5678, n_steps=40)
    state, _ = env.reset()
    pv0 = float(state[PORTFOLIO_VALUE_KEY][0])
    # Re-create agent and env from scratch (collect_single_trajectory will reset)
    env = make_env(seed=5678, n_steps=40)
    agent = DeployOnceAgent(env, lower_offset=-2, upper_offset=2)
    data = ac.collect_single_trajectory(env, agent.get_action, decision_stride=5)
    pv_final = float(env.state[PORTFOLIO_VALUE_KEY][0])
    cum = float(data['cumulative_pnl'][-1])
    diff = abs(cum - (pv_final - pv0))
    print(f"  PV_initial: {pv0:.4f}")
    print(f"  PV_final:   {pv_final:.4f}")
    print(f"  PV_final - PV_initial: {pv_final - pv0:.4f}")
    print(f"  cumulative_pnl[-1]:    {cum:.4f}")
    print(f"  diff: {diff:.2e}")
    assert diff < 1e-6
    print("  PASS\n")


def test_7_hold_preserves_bounds():
    print("=== Test 7: HOLD substeps never move LP tick bounds ===")
    env = ac.DecisionStrideEnv(make_env(seed=999, n_steps=40), stride=5)
    agent = DeployOnceAgent(env, lower_offset=-2, upper_offset=2)
    state, _ = env.reset()
    # Decision 0: deploy
    state, _, _, _, _ = env.step(agent.get_action(state))
    lo0, hi0 = float(state[LP_TICK_LOWER_KEY][0]), float(state[LP_TICK_UPPER_KEY][0])
    print(f"  bounds after deploy: [{lo0:.0f}, {hi0:.0f}]")
    # Subsequent decisions: agent emits hold>0, wrapper injects HOLD between → bounds frozen
    for i in range(5):
        action = agent.get_action(state)
        assert action[0, 2] > 0  # DeployOnce holds after first deploy
        state, _, _, _, _ = env.step(action)
        lo, hi = float(state[LP_TICK_LOWER_KEY][0]), float(state[LP_TICK_UPPER_KEY][0])
        assert lo == lo0 and hi == hi0, f"bounds drifted at decision {i+1}: [{lo}, {hi}]"
    print(f"  bounds after 5 HOLD windows: [{lo0:.0f}, {hi0:.0f}] (unchanged)")
    print("  PASS\n")


def test_8_gas_only_at_rebalance():
    print("=== Test 8: gas charged only on rebalance decision steps ===")
    gas = 10.0
    # Zero volatility + zero arrival rate → reward isolates gas charge.
    env = ac.DecisionStrideEnv(
        make_env(seed=42, n_steps=40, volatility=0.0, intensity=0.0, gas_cost=gas),
        stride=5,
    )
    state, _ = env.reset()
    deploy = np.array([[-2.0, 2.0, -1.0]], dtype=np.float32)
    hold = np.array([[-2.0, 2.0, +1.0]], dtype=np.float32)
    rebalance = np.array([[-3.0, 3.0, -1.0]], dtype=np.float32)

    # First deploy: env charges no gas (has_position=False at start of _rebalance).
    state, r0, *_ = env.step(deploy)
    # HOLD window: no gas, no fees, no price movement → 0 reward.
    state, r1, *_ = env.step(hold)
    # Re-rebalance: gas charged once (has_position=True now).
    state, r2, *_ = env.step(rebalance)
    # HOLD: 0 reward.
    state, r3, *_ = env.step(hold)

    print(f"  r0 (first deploy, gas-free per env):  {r0[0]:+.4f}  (expect ≈ 0)")
    print(f"  r1 (HOLD window):                     {r1[0]:+.4f}  (expect 0)")
    print(f"  r2 (rebalance, gas charged):          {r2[0]:+.4f}  (expect -{gas})")
    print(f"  r3 (HOLD window):                     {r3[0]:+.4f}  (expect 0)")
    assert abs(r0[0]) < 1e-6, f"first deploy is free per env logic; got {r0[0]}"
    assert abs(r1[0]) < 1e-6, f"HOLD window should be 0; got {r1[0]}"
    assert abs(r2[0] + gas) < 1e-6, f"rebalance should charge -{gas}; got {r2[0]}"
    assert abs(r3[0]) < 1e-6, f"HOLD window should be 0; got {r3[0]}"
    print("  PASS\n")


if __name__ == "__main__":
    test_1_stride1_passthrough()
    test_2_cumulative_pnl_invariant()
    test_3_per_window_sum()
    test_4_vec_vs_amm_consistency()
    test_5_per_step_logging()
    test_6_cum_pnl_telescopes()
    test_7_hold_preserves_bounds()
    test_8_gas_only_at_rebalance()
    print("=" * 60)
    print("ALL TESTS PASS")
    print("=" * 60)
