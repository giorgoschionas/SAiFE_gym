"""
Sanity check: verify LP fee accounting for a deposit-and-hold strategy.

Per-arrival fee (matches ModelDynamics._process_buy / _process_sell exactly):

    fee_multiplier = fee_tier / (1 - fee_tier)
    r              = exponential_value (≈ 1.0001)

    buy  at pre-swap √P :  fee in token1 = fee_multiplier · L_active · √P · (√r − 1)
    sell at pre-swap √P :  fee in token0 = fee_multiplier · L_active · (√r − 1) / √P

With a single LP (no background liquidity) the LP captures 100% of pool fees
when in range, so L_active = LP_LIQUIDITY whenever the price is inside
[LP_tick_lower, LP_tick_upper).

Setup:
- Deposit at full ±tau range on step 0, hold for the rest of the episode.
- Range half-width tau ≫ √(2·intensity·T), so the price almost never exits.
- Many trajectories average out Monte Carlo noise.

Arrivals are captured directly by wrapping ``arrival_model.get_arrivals`` —
this avoids the ~p² bias of inferring buys/sells from per-step tick changes
(simultaneous buy+sell leaves Δtick=0 and would be undercounted).

For the deposit-and-hold setup, ``LP_UNCLAIMED_FEES{0,1}`` at terminal time
equals cumulative gross fees, so it's the right empirical quantity to compare.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments.helpers import get_amm_env, ALPHA0
from SAiFE_gym.gym.index_names import (
    LP_LIQUIDITY_KEY, LP_UNCLAIMED_FEES0_KEY, LP_UNCLAIMED_FEES1_KEY,
    LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY, POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY,
)


def run_hold_episode(env, tau):
    """Deposit at full ±tau range on step 0, hold thereafter.

    Records pre-step √P, pre-step tick, and the actual arrivals (sells, buys)
    by wrapping the arrival model's get_arrivals. Capturing arrivals directly
    avoids the same-step buy+sell undercount that Δtick-based inference suffers.
    """
    state = env.model_dynamics.state
    arrival_model = env.model_dynamics.arrival_model
    n_steps = env.n_steps
    num_traj = env.num_trajectories

    arrivals_log = []
    original_get_arrivals = arrival_model.get_arrivals

    def recording_get_arrivals():
        a = original_get_arrivals()
        arrivals_log.append(a.copy())
        return a

    arrival_model.get_arrivals = recording_get_arrivals
    try:
        env.reset()

        sqrt_p_pre = np.zeros((n_steps, num_traj))
        tick_pre = np.zeros((n_steps, num_traj), dtype=np.int64)

        # Action format: (center_offset, half_width, hold_flag).
        # Full-range deposit ⇒ center=0, half_width=tau.
        deposit_action = np.tile(np.array([0.0, tau, -1.0], dtype=np.float32), (num_traj, 1))
        hold_action = np.tile(np.array([0.0, tau, +1.0], dtype=np.float32), (num_traj, 1))

        for t in range(n_steps):
            sqrt_p_pre[t] = state[POOL_SQRT_PRICE_KEY]
            tick_pre[t] = state[POOL_CURRENT_TICK_KEY]
            env.step(deposit_action if t == 0 else hold_action)
    finally:
        arrival_model.get_arrivals = original_get_arrivals

    arrivals = np.stack(arrivals_log, axis=0)            # (n_steps, num_traj, 2)
    is_sell = arrivals[..., 0].astype(bool)
    is_buy = arrivals[..., 1].astype(bool)
    return sqrt_p_pre, tick_pre, is_buy, is_sell


def main():
    # Parameters tuned so price stays in range and same-step arrivals are rare.
    num_traj = 500
    tau = 30
    arrival_rate = 5.0       # alpha1; total per-side intensity = ALPHA0 + arrival_rate
    n_steps = 1000
    terminal_time = 1.0
    volatility = 0.0
    fee_tier = 0.003
    r = 1.0001

    sqrt_r = np.sqrt(r)
    fee_multiplier = fee_tier / (1.0 - fee_tier)
    # Effective per-side intensity is max(α₀, α₁ + α₂·L ± α₃·(S-Z)).
    # With α₂=α₃=0 (defaults) it reduces to max(α₀, α₁).
    intensity_per_side = max(float(ALPHA0[0]), arrival_rate)

    env = get_amm_env(
        num_trajectories=num_traj,
        terminal_time=terminal_time,
        n_steps=n_steps,
        tau=tau,
        volatility=volatility,
        arrival_rate=arrival_rate,
    )

    sqrt_p_pre, tick_pre, is_buy, is_sell = run_hold_episode(env, tau)
    state = env.model_dynamics.state
    L = state[LP_LIQUIDITY_KEY]                             # (num_traj,)
    final_fee0 = state[LP_UNCLAIMED_FEES0_KEY]              # token0
    final_fee1 = state[LP_UNCLAIMED_FEES1_KEY]              # token1
    lp_lo = state[LP_TICK_LOWER_KEY]
    lp_hi = state[LP_TICK_UPPER_KEY]

    # Each step's arrivals act at the pre-step (sqrt_p_pre, tick_pre). When
    # both arrive together, _process_sell runs first (with prob 0.5) and then
    # the partner sees a one-tick-shifted state. We approximate that effect
    # below with a half-tick correction; with p_per_side ≪ 1 it's a tiny term.
    #
    # Did the buy/sell act inside the LP range at the pre-step state?
    #   buy at tick i  → uses L_i,    in-range iff lp_lo ≤ i < lp_hi
    #   sell at tick i → uses L_{i-1}, in-range iff lp_lo < i ≤ lp_hi
    buy_in_range = is_buy & (tick_pre >= lp_lo[None, :]) & (tick_pre < lp_hi[None, :])
    sell_in_range = is_sell & (tick_pre > lp_lo[None, :]) & (tick_pre <= lp_hi[None, :])

    # Theoretical per-arrival fees, evaluated at the pre-step √P.
    # For same-step buy+sell, the second-executed leg sees a √P shifted by √r;
    # average over the random ordering: ½·(1 + 1/√r) ≈ 1 - (√r-1)/2.
    both = is_buy & is_sell
    avg_factor = np.where(both, 0.5 * (1.0 + 1.0 / sqrt_r), 1.0)

    fee1_per_buy = fee_multiplier * L[None, :] * sqrt_p_pre * (sqrt_r - 1.0) * avg_factor
    fee0_per_sell = fee_multiplier * L[None, :] * (sqrt_r - 1.0) / sqrt_p_pre * avg_factor

    theory_fee1 = (fee1_per_buy * buy_in_range).sum(axis=0)
    theory_fee0 = (fee0_per_sell * sell_in_range).sum(axis=0)

    eps = 1e-12
    ratio0_pop = final_fee0.sum() / max(theory_fee0.sum(), eps)
    ratio1_pop = final_fee1.sum() / max(theory_fee1.sum(), eps)

    n_buys = is_buy.sum(axis=0)
    n_sells = is_sell.sum(axis=0)
    n_oor_buy = (n_buys - buy_in_range.sum(axis=0)).mean()
    n_oor_sell = (n_sells - sell_in_range.sum(axis=0)).mean()
    n_both = both.sum(axis=0).mean()
    p_per_side = intensity_per_side * (terminal_time / n_steps)

    print("=" * 64)
    print("Fee-accounting sanity check  (deposit-and-hold, single LP)")
    print("=" * 64)
    print(f"num_trajectories : {num_traj}")
    print(f"n_steps / T      : {n_steps} / {terminal_time}")
    print(f"tau (range ±)    : {tau} ticks")
    print(f"intensity / side : {intensity_per_side}  (α0={ALPHA0[0]}, α1={arrival_rate}, α2=α3=0)")
    print(f"P(arrival/side)  : {p_per_side:.4f}    P(both same step) ≈ {p_per_side**2:.2e}")
    print(f"fee_tier         : {fee_tier}    fee_multiplier: {fee_multiplier:.6f}")
    print(f"L_LP (per traj)  : {L.mean():.3e}  (std {L.std():.1e})")
    print()
    print(f"buys  / traj : mean {n_buys.mean():6.2f}   out-of-range mean {n_oor_buy:.3f}")
    print(f"sells / traj : mean {n_sells.mean():6.2f}   out-of-range mean {n_oor_sell:.3f}")
    print(f"same-step buy+sell / traj : mean {n_both:.3f}")
    print()
    print(f"{'':<14}{'empirical':>16}{'theoretical':>16}{'ratio':>10}")
    print(f"{'token1 fees':<14}{final_fee1.mean():>16.4f}{theory_fee1.mean():>16.4f}{ratio1_pop:>10.4f}")
    print(f"{'token0 fees':<14}{final_fee0.mean():>16.6f}{theory_fee0.mean():>16.6f}{ratio0_pop:>10.4f}")
    print("=" * 64)
    print("Both ratios should be ≈ 1.000 to within Monte Carlo noise.")


if __name__ == "__main__":
    main()
