"""Sweep ArrivalRebalance's rebalancing frequency across the diagonal cells.

``ArrivalRebalanceAgent`` re-quotes after every ``rebalance_every`` liquidity-
taking arrivals (an exact count read from ``model_dynamics.last_arrivals``, not
a fee-delta approximation). The main results use a single value; this script
scans a range so the benchmark is reported at its own best operating point
rather than at one arbitrary setting.

Runs on the same evaluation protocol as ``reevaluate_saved_models.py`` -- eval
seed 1005, 1000 shared trajectories, no decision stride (baselines act on every
environment step) -- so the numbers are directly comparable to the main table.

Only the three (gas, volatility) pairs matter: the agent is rule-based, so its
result depends on neither the training seed nor the inventory penalty phi. One
run directory per cell therefore supplies the whole configuration.

Usage
-----
    conda run -n your_env python arrival_rebalance_sweep.py
    conda run -n your_env python arrival_rebalance_sweep.py \
        --every 50,100,150 --dump-trajectories arrival_dumps
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reevaluate_saved_models import (
    discover_runs, make_env, summarise, dump_trajectories,
    ArrivalRebalanceAgent, evaluate_on_trajectories_with_attribution,
)

# (gas_cost, volatility) -- the diagonal used in the main results.
CELLS = [(2.0, 0.01), (4.0, 0.02), (6.0, 0.03)]

DEFAULT_EVERY = [50, 100, 150, 200, 250, 300, 350, 400]


def pick_run(runs: list[dict], gas: float, vol: float) -> dict:
    """One run directory per cell; any will do since baselines ignore seed/phi.

    Prefer phi=0 for determinism of the choice, falling back to whatever exists.
    """
    match = [r for r in runs
             if np.isclose(float(r["gas_cost"]), gas)
             and np.isclose(float(r["cfg"]["VOLATILITY"]), vol)]
    if not match:
        raise SystemExit(f"No run found for gas={gas:g} vol={vol:g}")
    zero_phi = [r for r in match if float(r["cfg"]["INVENTORY_PHI"]) == 0.0]
    pool = zero_phi or match
    return sorted(pool, key=lambda r: r["run_dir"].name)[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "results_with_seeds")
    ap.add_argument("--out-csv", type=Path,
                    default=Path(__file__).parent / "arrival_rebalance_sweep.csv")
    ap.add_argument("--every", default=",".join(map(str, DEFAULT_EVERY)),
                    help="Comma-separated rebalance_every values (arrivals).")
    ap.add_argument("--eval-seed", type=int, default=1005)
    ap.add_argument("--num-trajectories", type=int, default=None,
                    help="Default: whatever the run config used (1000).")
    ap.add_argument("--dump-trajectories", type=Path, default=None,
                    help="Optional directory for per-trajectory .npz arrays.")
    args = ap.parse_args()

    every_values = [int(x) for x in args.every.split(",") if x.strip()]

    print("=" * 78)
    print("ArrivalRebalance: rebalancing-frequency sweep")
    print("=" * 78)
    print(f"  cells      : {[(f'g={g:g}', f'sigma={v:g}') for g, v in CELLS]}")
    print(f"  every      : {every_values}  (liquidity-taking arrivals)")
    print(f"  eval seed  : {args.eval_seed}")
    print()

    runs = discover_runs(args.root)
    if not runs:
        raise SystemExit(f"No runs found under {args.root}")

    rows = []
    t_start = time.time()
    for gas, vol in CELLS:
        run = pick_run(runs, gas, vol)
        cfg = run["cfg"]
        n_eval = args.num_trajectories or cfg["NUM_TRAJECTORIES_EVAL"]
        lower, upper = cfg["ARRIVAL_REBALANCE_LOWER"], cfg["ARRIVAL_REBALANCE_UPPER"]
        print(f"[gas={gas:g} sigma={vol:g}] {run['run_dir'].name}  "
              f"range=[{lower:+d},{upper:+d}]  n_eval={n_eval}")

        for every in every_values:
            t0 = time.time()
            # Fresh env per setting so each sees the identical seeded paths.
            env = make_env(cfg, gas, n_eval, args.eval_seed)
            agent = ArrivalRebalanceAgent(
                env, rebalance_every=every,
                width=cfg["ARRIVAL_REBALANCE_WIDTH"],
                lower_offset=lower, upper_offset=upper,
            )
            # Baselines act on every environment step -- no DecisionStrideEnv.
            res = evaluate_on_trajectories_with_attribution(env, agent.get_action)
            metrics = summarise(res)

            if args.dump_trajectories is not None:
                dump_trajectories(
                    args.dump_trajectories, f"g{gas:g}_s{vol:g}",
                    f"ArrivalRebalance-every{every}", args.eval_seed, res,
                    {"gas_cost": gas, "train_seed": -1, "volatility": vol,
                     "inventory_phi": 0.0, "n_eval": n_eval,
                     "rebalance_every": every, "reused_from": ""})

            rows.append({
                "gas_cost": gas, "volatility": vol, "rebalance_every": every,
                "lower_offset": lower, "upper_offset": upper,
                "eval_seed": args.eval_seed, "n_eval": n_eval,
                "run_id": run["run_dir"].name, **metrics,
            })
            print(f"   every={every:<4} mean={metrics['mean_pnl']:+8.2f} "
                  f"cvar5={metrics['cvar5_pnl']:+9.2f} "
                  f"fees={metrics['attrib_fees']:7.2f} "
                  f"gas={metrics['attrib_gas']:7.2f} "
                  f"reb/ep={metrics['rebalances_per_ep']:6.2f} "
                  f"({time.time() - t0:.0f}s)")
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    # ----------------------------- report ---------------------------------- #
    for gas, vol in CELLS:
        sub = df[(df.gas_cost == gas) & (df.volatility == vol)]
        if sub.empty:
            continue
        best = sub.loc[sub.mean_pnl.idxmax()]
        print("=" * 96)
        print(f"gas={gas:g}  sigma={vol:g}      (best mean PnL at "
              f"every={int(best.rebalance_every)})")
        print("=" * 96)
        print(f"{'every':>7}{'PnL':>10}{'CVaR5':>10}{'P5':>9}{'P95':>9}"
              f"{'fees':>9}{'IL':>9}{'gas':>9}{'reb/ep':>9}{'profit%':>9}")
        print("-" * 96)
        for _, r in sub.sort_values("rebalance_every").iterrows():
            mark = "  <-- best" if r.rebalance_every == best.rebalance_every else ""
            print(f"{int(r.rebalance_every):>7}{r.mean_pnl:>10.2f}{r.cvar5_pnl:>10.2f}"
                  f"{r.p5_pnl:>9.2f}{r.p95_pnl:>9.2f}{r.attrib_fees:>9.2f}"
                  f"{r.attrib_il:>9.2f}{r.attrib_gas:>9.2f}"
                  f"{r.rebalances_per_ep:>9.2f}{100*r.profitable_frac:>9.1f}{mark}")
        print("-" * 96)
        print()

    print(f"Wrote {len(df)} rows to {args.out_csv}  "
          f"(total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
