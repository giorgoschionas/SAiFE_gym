"""
Reproduce mbt_gym's "Figure 1: Speedup of vectorization using NumPy" for amm_sim.

Compares two ways of rolling out ``N`` independent AMM trajectories:

  1. NumPy vectorised   — ONE ``AMMEnvironment(num_trajectories=N)`` rolled out
                          once. The trajectory axis is a leading NumPy dimension,
                          so all N worlds advance together under vectorised ops.
  2. concurrent.futures — N separate ``AMMEnvironment(num_trajectories=1)`` envs,
                          each rolled out in its own worker process via
                          ``concurrent.futures.ProcessPoolExecutor``. This is the
                          classic "spin up many envs across CPUs" approach.

Both paths run the *identical* per-trajectory workload — a full ``N_STEPS``
episode driving the raw environment with a fixed rebalance-every-step action
(no policy network, no training) — so the plot times the environment's
``step()``, nothing else.

Fairness notes (defensible in the thesis):
  * The process pool is created and warmed up OUTSIDE the timed region, so the
    concurrent curve reflects per-trajectory dispatch + compute, not one-off
    process-spawn cost.
  * Each worker gets a distinct seed, so N genuinely distinct paths are produced
    — exactly what the vectorised env's RNG yields in a single call.
  * The pool tick moves at most one tick per step, so over ``N_STEPS`` steps the
    price can drift at most ``±N_STEPS`` ticks. With ``--num-ticks`` wide enough
    (± num_ticks/2 > N_STEPS) the lattice can never overflow, for any N.

``--num-ticks`` controls the per-trajectory element work (O(N·num_ticks) in the
vectorised path). Smaller values push the vectorised line's flat→rising crossover
to higher N and widen the small/mid-N speedup; the large-N ceiling stays ~cores
because both paths must do the same total element work.

Output: a log-log figure (matching the paper) + a CSV of the raw timings, both
written next to this script under ``figures/``, tagged with num_ticks.

Run:
    python vectorization_speedup.py                     # num_ticks=800, sweep 1..1000
    python vectorization_speedup.py --num-ticks 450     # lighter per-traj workload
    python vectorization_speedup.py --max-exp 4         # extend sweep to 10000
    python vectorization_speedup.py --workers 8
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from amm_sim.env.AMMEnvironment import AMMEnvironment
from amm_sim.env.ModelDynamics import UniswapV3ModelDynamics
from amm_sim.stochastic_processes.midprice_models import (
    GeometricBrownianMotionMidpriceModel,
)
from amm_sim.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from amm_sim.rewards.RewardFunctions import PnL
from amm_sim.env.index_names import PORTFOLIO_VALUE_KEY

# ============================================================================
# Fixed simulation config — identical for both the vectorised and the
# per-process rollout, so the ONLY thing that differs is how the N trajectories
# are parallelised. Kept at module scope so ProcessPoolExecutor's 'spawn'
# workers reconstruct the same constants on re-import. (num_ticks is threaded
# through the task args instead, so a --num-ticks override reaches the workers.)
# ============================================================================

TERMINAL_TIME = 1.0
# Temporarily 15 for a num_ticks=50 run (±25 window needs drift ≤ N_STEPS).
N_STEPS = 15                  # steps per episode (per-trajectory work)
DEFAULT_NUM_TICKS = 800       # lattice width; ±num_ticks/2 > N_STEPS ⇒ safe
TAU = 5
FEE_TIER = 0.003
EXP_VALUE = 1.0001
INITIAL_PRICE = 100.0
INITIAL_WEALTH = 1e6
VOLATILITY = 0.1
DRIFT = 0.0
LIQUIDITY_SCALE = 1e6
# LiquidityKernelArrivalModel alpha rows: [floor, baseline, liq-coef, arb-coef]
ALPHA = np.array([
    [10.0, 10.0],
    [100.0, 100.0],
    [50.0, 50.0],
    [5.0, 5.0],
])

BASE_SEED = 12345

# Fixed action applied every step: rebalance (hold_flag = -1) to a symmetric
# ±2-tick range around the current tick. Exercises the full _rebalance path
# (the heaviest per-step vectorised code) on every step.
_ACTION_ROW = np.array([-2.0, 2.0, -1.0], dtype=np.float32)


def make_env(num_trajectories: int, seed: int, num_ticks: int) -> AMMEnvironment:
    """Build a self-contained AMM env (GBM midprice + liquidity-kernel arrivals)."""
    step_size = TERMINAL_TIME / N_STEPS
    midprice_model = GeometricBrownianMotionMidpriceModel(
        drift=DRIFT, volatility=VOLATILITY, initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME, step_size=step_size,
        num_trajectories=num_trajectories, seed=seed,
    )
    arrival_model = LiquidityKernelArrivalModel(
        alpha=ALPHA, beta=0.001, K=100, liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size, num_trajectories=num_trajectories,
        seed=seed + 1,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model, arrival_model=arrival_model,
        num_trajectories=num_trajectories, fee_tier=FEE_TIER, tau=TAU,
        num_ticks=num_ticks, exponential_value=EXP_VALUE, seed=seed + 2,
    )
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS,
        initial_wealth=INITIAL_WEALTH, reward_function=PnL(),
        model_dynamics=model_dynamics, num_trajectories=num_trajectories,
        seed=seed,
    )


def rollout(num_trajectories: int, seed: int, num_ticks: int) -> float:
    """Run one full episode over ``num_trajectories`` worlds; return mean final PnL.

    The return value is consumed by the caller so the work can't be optimised
    away, and doubles as a cheap correctness signal.
    """
    env = make_env(num_trajectories, seed, num_ticks)
    state, _ = env.reset()
    start_pv = state[PORTFOLIO_VALUE_KEY].astype(np.float64).copy()
    action = np.tile(_ACTION_ROW, (num_trajectories, 1))
    for _ in range(env.n_steps):
        state, _, _, _, _ = env.step(action)
    final_pv = state[PORTFOLIO_VALUE_KEY].astype(np.float64)
    return float(np.mean(final_pv - start_pv))


def _rollout_worker(args) -> float:
    """Top-level (picklable) worker: one vectorised rollout of ``batch_size`` trajs.

    ``batch_size == 1`` is the pure-multiprocessing case (one env per task);
    ``batch_size > 1`` is the combined/ensemble case (a vectorised batch per task).
    """
    seed, num_ticks, batch_size = args
    return rollout(batch_size, seed, num_ticks)


# ============================================================================
# Timing drivers
# ============================================================================

def time_vectorized(n: int, repeats: int, num_ticks: int) -> float:
    """Median wall-clock of one vectorised rollout of ``n`` trajectories."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        rollout(n, BASE_SEED, num_ticks)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_concurrent(n: int, repeats: int, executor: ProcessPoolExecutor,
                    num_ticks: int) -> float:
    """Median wall-clock to produce ``n`` single-trajectory rollouts via the pool.

    The executor is created/warmed up by the caller, so process-spawn cost is
    excluded — this measures dispatch + compute for N independent envs.
    """
    times = []
    for _ in range(repeats):
        args = [(BASE_SEED + i, num_ticks, 1) for i in range(n)]
        t0 = time.perf_counter()
        # chunksize=1: one env per task, honest to the "many envs" model.
        list(executor.map(_rollout_worker, args, chunksize=1))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_combined(n: int, repeats: int, executor: ProcessPoolExecutor,
                  num_ticks: int, procs: int) -> float:
    """Median wall-clock of the ensemble approach: ``P`` worker processes, each
    running a *vectorised* env of ~n/P trajectories (the SubprocVecEnv pattern).

    Gets both wins at once: the numpy batch inside each process amortises the
    fixed per-step overhead, while the P processes spread the O(N*num_ticks)
    element work across cores. P is capped at n (can't have empty batches).
    """
    P = min(procs, n)
    base, rem = divmod(n, P)              # split n as evenly as possible
    batch_sizes = [base + 1 if p < rem else base for p in range(P)]
    times = []
    for _ in range(repeats):
        args = [(BASE_SEED + p * 10_000, num_ticks, bs)
                for p, bs in enumerate(batch_sizes)]
        t0 = time.perf_counter()
        list(executor.map(_rollout_worker, args, chunksize=1))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def repeats_for(n: int) -> int:
    """Fewer repeats as N grows — each large-N point already averages many worlds."""
    if n <= 100:
        return 3
    if n < 3000:
        return 2
    return 1


def main():
    import argparse
    import csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--max-exp', type=int, default=3,
        help='Sweep N over 10**0 .. 10**max-exp (log-spaced). Default 3 → up to '
             '1000, matching the mbt_gym Figure 1 x-axis. Use 4 to extend to 10000.',
    )
    parser.add_argument(
        '--num-ticks', type=int, default=DEFAULT_NUM_TICKS,
        help='Liquidity-array width. Must exceed 2*N_STEPS so the lattice cannot '
             'overflow. Smaller → lighter per-trajectory work → flatter vectorised '
             f'curve. Default {DEFAULT_NUM_TICKS}.',
    )
    parser.add_argument(
        '--workers', type=int, default=os.cpu_count(),
        help='Process pool size (default: os.cpu_count()).',
    )
    parser.add_argument(
        '--repeats', type=int, default=None,
        help='Fixed number of timing repeats per point (median taken). Higher = '
             'smoother, less wall-clock noise. Default: adaptive (3/2/1 by N).',
    )
    parser.add_argument(
        '--combined-procs', type=int, default=os.cpu_count(),
        help='Number of processes for the combined/ensemble line, each running a '
             'vectorised env of ~N/P trajectories (default: os.cpu_count()).',
    )
    parser.add_argument(
        '--no-combined', action='store_true',
        help='Skip the combined/ensemble line — plot only vectorised vs concurrent.',
    )
    parser.add_argument(
        '--no-title', action='store_true',
        help='Omit the figure title (e.g. for thesis figures with a caption).',
    )
    parser.add_argument(
        '--outdir', type=str,
        default=os.path.join(os.path.dirname(__file__), 'figures'),
        help='Directory for the figure + CSV.',
    )
    args = parser.parse_args()

    num_ticks = args.num_ticks
    # Safety: pool tick drifts at most ±N_STEPS; the array is centred, so we need
    # num_ticks/2 > N_STEPS or _process_sell/_process_buy can assert mid-run.
    min_safe = 2 * N_STEPS + 20
    if num_ticks < min_safe:
        parser.error(
            f"--num-ticks={num_ticks} too small; need >= {min_safe} "
            f"(2*N_STEPS + margin) so the lattice cannot overflow."
        )

    # Log-spaced trajectory grid: 1, 3, 10, 30, 100, ... up to 10**max-exp.
    grid = []
    for e in range(args.max_exp + 1):
        base = 10 ** e
        grid.append(base)
        if e < args.max_exp:
            grid.append(3 * base)
    trajectory_counts = sorted(set(grid))

    os.makedirs(args.outdir, exist_ok=True)

    combined_procs = args.combined_procs
    run_combined = not args.no_combined
    print(f"CPU count: {os.cpu_count()} | pool workers: {args.workers} "
          f"| combined procs: {combined_procs}")
    print(f"Per-trajectory workload: {N_STEPS} steps, num_ticks={num_ticks}")
    print(f"Trajectory sweep: {trajectory_counts}\n")

    # --- Warm-up: JIT numpy caches + spawn all pool workers, OUTSIDE timing ---
    print("Warming up (imports, numpy caches, worker processes)...")
    rollout(1, BASE_SEED, num_ticks)          # warm the vectorised path
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Force every worker to spawn + import before any timing starts.
        list(executor.map(
            _rollout_worker,
            [(BASE_SEED + i, num_ticks, 1) for i in range(args.workers)],
        ))

        if run_combined:
            header = (
                f"{'N':>7} | {'vec (s)':>10} | {'multiproc (s)':>13} | {'combined (s)':>12} "
                f"| {'mp/vec':>7} | {'vec/comb':>8}"
            )
        else:
            header = f"{'N':>7} | {'vec (s)':>10} | {'multiproc (s)':>13} | {'mp/vec':>7}"
        print("\n" + header)
        print("-" * len(header))

        results = []
        for n in trajectory_counts:
            reps = args.repeats if args.repeats is not None else repeats_for(n)
            t_vec = time_vectorized(n, reps, num_ticks)
            t_con = time_concurrent(n, reps, executor, num_ticks)
            sp_mp_vec = t_con / t_vec if t_vec > 0 else float('nan')
            if run_combined:
                t_comb = time_combined(n, reps, executor, num_ticks, combined_procs)
                sp_vec_comb = t_vec / t_comb if t_comb > 0 else float('nan')
                results.append((n, t_vec, t_con, t_comb, sp_mp_vec, sp_vec_comb))
                print(f"{n:>7} | {t_vec:>10.4f} | {t_con:>13.4f} | {t_comb:>12.4f} "
                      f"| {sp_mp_vec:>6.1f}x | {sp_vec_comb:>7.1f}x")
            else:
                results.append((n, t_vec, t_con, float('nan'), sp_mp_vec, float('nan')))
                print(f"{n:>7} | {t_vec:>10.4f} | {t_con:>13.4f} | {sp_mp_vec:>6.1f}x")

    # --- Persist raw timings (filename tagged with num_ticks) ---
    tag = f"ticks{num_ticks}"
    csv_path = os.path.join(args.outdir, f'vectorization_speedup_{tag}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if run_combined:
            writer.writerow(['num_trajectories', 'vectorized_s', 'concurrent_s',
                             'combined_s', 'mp_over_vec', 'vec_over_comb'])
            writer.writerows(results)
        else:
            writer.writerow(['num_trajectories', 'vectorized_s', 'concurrent_s', 'speedup'])
            writer.writerows([(r[0], r[1], r[2], r[4]) for r in results])
    print(f"\nTimings written to: {csv_path}")

    # --- Figure (monochrome, matching the mbt_gym paper) ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ns = [r[0] for r in results]
    vec = [r[1] for r in results]
    con = [r[2] for r in results]
    comb = [r[3] for r in results]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ns, con, color='black', linestyle='--', linewidth=2,
            label='concurrent.futures (1 traj / process)')
    ax.plot(ns, vec, color='black', linestyle='-', linewidth=2,
            label='NumPy vectorized (1 process)')
    if run_combined:
        ax.plot(ns, comb, color='black', linestyle='-.', linewidth=2,
                label=f'combined ({combined_procs} processes × vectorised)')

    label_fs = 20            # legend font size
    axis_fs = label_fs + 3   # axis names a touch larger
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of trajectories', fontsize=axis_fs)
    ax.set_ylabel('Rollout time (s)', fontsize=axis_fs)
    if not args.no_title:
        ax.set_title(f'NumPy vs concurrent.futures  (num_ticks={num_ticks})')
    ax.legend(loc='upper left', frameon=True, framealpha=1.0, edgecolor='0.7',
              fontsize=label_fs)
    ax.grid(True, which='major', linestyle=':', linewidth=0.6, color='0.8')
    fig.tight_layout()

    png_path = os.path.join(args.outdir, f'vectorization_speedup_{tag}.png')
    fig.savefig(png_path, dpi=200)
    fig.savefig(os.path.join(args.outdir, f'vectorization_speedup_{tag}.pdf'))
    print(f"Figure written to:  {png_path}")

    # Headline numbers for the caption.
    n_max = ns[-1]
    if run_combined:
        print(
            f"\nAt N={n_max}: multiprocessing {con[-1]:.1f}s | vectorised "
            f"{vec[-1]:.3f}s | combined {comb[-1]:.3f}s.\n"
            f"  vectorised vs multiprocessing : {con[-1] / vec[-1]:.0f}x\n"
            f"  combined  vs vectorised       : {vec[-1] / comb[-1]:.1f}x\n"
            f"  combined  vs multiprocessing  : {con[-1] / comb[-1]:.0f}x"
        )
    else:
        print(
            f"\nAt N={n_max}: multiprocessing {con[-1]:.1f}s | vectorised "
            f"{vec[-1]:.3f}s  →  {con[-1] / vec[-1]:.0f}x speedup."
        )


if __name__ == '__main__':
    main()
