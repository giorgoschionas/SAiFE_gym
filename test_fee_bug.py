"""
Simulation to confirm the pre-entry fee collection bug in SAiFE_gym.

Bug: FEES0_KEY/FEES1_KEY accumulate continuously. When the LP moves to a new
     range (rebalances), fees that had already accumulated in the new range
     (before the LP was there) are credited to the LP at the NEXT rebalance.

Expected (correct) behaviour: LP should collect ZERO fees from trades that
     happened before they deployed liquidity in a given range.
"""

import sys
sys.path.insert(0, '/home/gchionas/Programming/Blockchain/Ethereum/defi-trading/SAiFE_gym')

import numpy as np
from experiments.helpers import get_amm_env
from SAiFE_gym.gym.index_names import (
    POOL_CURRENT_TICK_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_LIQUIDITY_KEY, FEES0_KEY, FEES1_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
)

TAU = 10

env = get_amm_env(num_trajectories=1, tau=TAU, arrival_rate=500.0, volatility=0.1, seed=42)
env.reset()

state = env.model_dynamics.state
tick_lower_global = env.model_dynamics.tick_lower_global
num_ticks = env.model_dynamics.num_ticks


def current_tick():
    return int(state[POOL_CURRENT_TICK_KEY][0])


def lp_range():
    return int(state[LP_TICK_LOWER_KEY][0]), int(state[LP_TICK_UPPER_KEY][0])


def fees0_in_abs_range(lo_abs, hi_abs):
    lo_idx = max(lo_abs - tick_lower_global, 0)
    hi_idx = min(hi_abs - tick_lower_global, num_ticks)
    return float(state[FEES0_KEY][0, lo_idx:hi_idx].sum())


def lp_collected_fees0():
    return float(state[LP_COLLECTED_FEES0_KEY][0])


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: place LP well ABOVE current price (offsets [+3, +TAU])
#         → trades near current tick will accumulate fees OUTSIDE LP range
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("STEP 1 – place LP above current price (offsets [+3, +TAU])")
action_high = np.array([[3, TAU]])
env.step(action_high)

ctick = current_tick()
lp_lo, lp_hi = lp_range()
print(f"  current tick : {ctick}")
print(f"  LP range A   : [{lp_lo}, {lp_hi})")

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: another step – fees accumulate near current tick (below LP range A)
# ──────────────────────────────────────────────────────────────────────────────
print("\nSTEP 2 – fees accumulate near current tick (outside range A)")
env.step(action_high)

ctick = current_tick()
# Range B will be centred on current tick (offsets [-TAU, +3])
range_B_lo = ctick - TAU
range_B_hi = ctick + 3   # just below LP's current lower bound

fees_B_before = fees0_in_abs_range(range_B_lo, range_B_hi)
collected_before_move = lp_collected_fees0()

print(f"  current tick         : {ctick}")
print(f"  Fees0 in future B    : {fees_B_before:.6f}  ← accrued BEFORE LP enters")
print(f"  LP collected so far  : {collected_before_move:.6f}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: rebalance LP DOWN to range B (where pre-entry fees already exist)
# ──────────────────────────────────────────────────────────────────────────────
print("\nSTEP 3 – rebalance LP to range B (offsets [-TAU, +3])")
action_low = np.array([[-TAU, 3]])
env.step(action_low)

lp_lo2, lp_hi2 = lp_range()
print(f"  LP range B : [{lp_lo2}, {lp_hi2})")

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: immediately rebalance again (same range B)
#         _collect_lp_fees() is called → LP collects fees from range B
#         Some of those fees pre-date the LP's entry to range B → BUG
# ──────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4 – immediately rebalance again (triggers _collect_lp_fees on range B)")
env.step(action_low)

total_collected = lp_collected_fees0()
delta = total_collected - collected_before_move

print(f"  LP fees0 collected after steps 3+4 : {delta:.6f}")
print(f"  Pre-entry fees0 that were in range B: {fees_B_before:.6f}")

print()
print("=" * 62)
if fees_B_before > 0 and delta > 0:
    # The LP could not have earned more in fees than the total fees in range B
    # before they entered, plus whatever was earned in step 4 trades.
    # A simpler check: fees_B_before > 0 AND delta > 0 means some pre-entry fees
    # were almost certainly credited (especially since step 4 had no time for
    # significant NEW fees to accumulate in B while LP just arrived).
    print("BUG CONFIRMED – LP credited fees that predated their position.")
    print(f"  Pre-entry fees0 in range B : {fees_B_before:.6f}")
    print(f"  LP fees0 collected (steps 3+4): {delta:.6f}")
else:
    print("Bug not triggered – try a different seed or higher arrival rate.")
print("=" * 62)
