"""CDM in its original, frictionless form: no deadband, no gas.

The main results run CDM with ``rebalance_tolerance=30``, a deadband added to
stop the strategy bleeding gas -- without it the agent re-quotes on essentially
every step, which at g=2..6 is ruinous (see the gas attribution at
sigma=.02, g=4: 107 paid in gas against 52 earned in fees).

``rebalance_tolerance=0`` recovers the policy as stated in Cartea-Drissi-Monga:
continuously track the optimal range. Evaluating it at ``gas_cost=0`` therefore
measures the strategy on its own terms, in the frictionless setting the closed
form was derived for, separating "the control law is wrong" from "the control
law is right but too expensive to execute".

Everything else matches the main protocol: evaluation seed 1005, the same 1000
trajectories, no decision stride (baselines act on every environment step), and
the per-volatility configuration taken from the same run directories used
elsewhere. Only the gas cost and the tolerance change.

Usage
-----
    conda run -n your_env python cdm_no_gas.py
    conda run -n your_env python cdm_no_gas.py --tolerances 0,30 --gas 0,2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reevaluate_saved_models import (
    discover_runs, make_env, summarise, dump_trajectories,
    CarteaPLAgent, evaluate_on_trajectories_with_attribution, DecisionStrideEnv,
)

# The three volatility scenarios. The gas cost each is paired with in the main
# results is recorded only for provenance -- it is overridden below.
VOLATILITIES = [(0.01, 2.0), (0.02, 4.0), (0.03, 6.0)]


def pick_run(runs: list[dict], vol: float, orig_gas: float) -> dict:
    """The run directory supplying the configuration for one volatility.

    Prefers the (gas, vol) pairing used in the main results so every parameter
    other than gas and tolerance is identical to the published cells.
    """
    match = [r for r in runs if np.isclose(float(r["cfg"]["VOLATILITY"]), vol)]
    if not match:
        raise SystemExit(f"No run found with VOLATILITY={vol:g}")
    same_gas = [r for r in match if np.isclose(float(r["gas_cost"]), orig_gas)]
    zero_phi = [r for r in (same_gas or match)
                if float(r["cfg"]["INVENTORY_PHI"]) == 0.0]
    pool = zero_phi or same_gas or match
    return sorted(pool, key=lambda r: r["run_dir"].name)[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "results_with_seeds")
    ap.add_argument("--out-csv", type=Path,
                    default=Path(__file__).parent / "cdm_no_gas.csv")
    ap.add_argument("--tolerances", default="0",
                    help="Comma-separated rebalance_tolerance values.")
    ap.add_argument("--gas", default="0",
                    help="Comma-separated gas costs, or 'orig' to use the gas "
                         "cost each volatility is paired with in the main "
                         "results (2, 4, 6).")
    ap.add_argument("--decision-stride", type=int, default=None,
                    help="Restrict CDM to one decision every STRIDE steps, the "
                         "same constraint the RL agents face (DECISION_STRIDE"
                         "=100 gives 10 decisions per episode). Gas is then "
                         "charged only on the re-quotes it actually makes at "
                         "those points, so decision freedom and execution cost "
                         "are both matched. Default: acts on every step.")
    ap.add_argument("--max-gas-events", type=int, default=None,
                    help="Cap gas charges at this many rebalances per "
                         "trajectory. The RL agents get N_STEPS/DECISION_STRIDE "
                         "= 10, so --max-gas-events 10 gives a continuously "
                         "re-quoting CDM the same execution budget. Default: "
                         "uncapped, i.e. every rebalance is charged.")
    ap.add_argument("--eval-seed", type=int, default=1005)
    ap.add_argument("--num-trajectories", type=int, default=None)
    ap.add_argument("--dump-trajectories", type=Path, default=None)
    args = ap.parse_args()

    tolerances = [int(x) for x in args.tolerances.split(",") if x.strip()]
    use_orig_gas = args.gas.strip().lower() == "orig"
    gases = ([] if use_orig_gas
             else [float(x) for x in args.gas.split(",") if x.strip()])

    print("=" * 78)
    print("CDM, original form: rebalance_tolerance=0, no gas")
    print("=" * 78)
    print(f"  volatilities : {[v for v, _ in VOLATILITIES]}")
    print(f"  tolerances   : {tolerances}")
    print(f"  gas costs    : {gases}")
    print(f"  eval seed    : {args.eval_seed}")
    print()

    runs = discover_runs(args.root)
    if not runs:
        raise SystemExit(f"No runs found under {args.root}")

    rows = []
    t_start = time.time()
    for vol, orig_gas in VOLATILITIES:
        run = pick_run(runs, vol, orig_gas)
        cfg = run["cfg"]
        n_eval = args.num_trajectories or cfg["NUM_TRAJECTORIES_EVAL"]
        print(f"[sigma={vol:g}] {run['run_dir'].name}  "
              f"gamma={cfg['GAMMA_CARTEA']:g}  n_eval={n_eval}")

        for gas in ([orig_gas] if use_orig_gas else gases):
            for tol in tolerances:
                t0 = time.time()
                env = make_env(cfg, gas, n_eval, args.eval_seed,
                               max_gas_events=args.max_gas_events)
                agent = CarteaPLAgent(env, gamma=cfg["GAMMA_CARTEA"],
                                      rebalance_tolerance=tol,
                                      seed=cfg["SEED"])
                # Agent is built on the raw env (it reads model_dynamics
                # directly); the stride wrapper only governs how often its
                # action is applied -- the same arrangement the RL agents use.
                eval_env = (env if args.decision_stride is None
                            else DecisionStrideEnv(env, stride=args.decision_stride))
                res = evaluate_on_trajectories_with_attribution(eval_env, agent.get_action)

                # The attribution reconstructs gas as `rebalances x gas_cost`
                # (agent_comparison.py:916), which is correct only when every
                # rebalance is charged. Under a cap it overstates gas, and since
                # IL is the residual `hodl + fees - gas - pnl` the error lands
                # there too. Wealth-derived quantities (PnL and everything
                # computed from it) are unaffected. Substitute the gas the
                # environment actually charged and re-derive IL.
                if args.max_gas_events is not None:
                    true_gas = env.model_dynamics.cum_gas_paid.copy()
                    res["il"] = res["il"] + res["gas"] - true_gas
                    res["gas"] = true_gas

                metrics = summarise(res)

                if args.dump_trajectories is not None:
                    dump_trajectories(
                        args.dump_trajectories, f"s{vol:g}_g{gas:g}",
                        f"CDM-tol{tol}", args.eval_seed, res,
                        {"gas_cost": gas, "train_seed": -1, "volatility": vol,
                         "inventory_phi": 0.0, "n_eval": n_eval,
                         "rebalance_tolerance": tol, "reused_from": ""})

                rows.append({
                    "volatility": vol, "gas_cost": gas,
                    "rebalance_tolerance": tol,
                    "max_gas_events": args.max_gas_events,
                    "decision_stride": args.decision_stride,
                    "original_gas_cost": orig_gas,
                    "gamma": cfg["GAMMA_CARTEA"],
                    "eval_seed": args.eval_seed, "n_eval": n_eval,
                    "run_id": run["run_dir"].name, **metrics,
                })
                print(f"   gas={gas:g} tol={tol:<3} "
                      f"mean={metrics['mean_pnl']:+9.2f} "
                      f"cvar5={metrics['cvar5_pnl']:+10.2f} "
                      f"fees={metrics['attrib_fees']:8.2f} "
                      f"IL={metrics['attrib_il']:8.2f} "
                      f"gas$={metrics['attrib_gas']:7.2f} "
                      f"reb/ep={metrics['rebalances_per_ep']:7.2f} "
                      f"({time.time() - t0:.0f}s)")
                pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("=" * 112)
    print("CDM (tol=0, no gas)  --  same metrics as the main tables")
    print("=" * 112)
    print(f"{'sigma':>7}{'gas':>5}{'tol':>5}{'PnL':>11}{'Std':>10}{'P5':>10}"
          f"{'P95':>10}{'CVaR5':>10}{'CVaR1':>10}{'fees':>9}{'IL':>9}"
          f"{'reb/ep':>9}{'profit%':>9}")
    print("-" * 112)
    for _, r in df.iterrows():
        print(f"{r.volatility:>7.2f}{r.gas_cost:>5.0f}{int(r.rebalance_tolerance):>5}"
              f"{r.mean_pnl:>11.2f}{r.std_pnl:>10.2f}{r.p5_pnl:>10.2f}"
              f"{r.p95_pnl:>10.2f}{r.cvar5_pnl:>10.2f}{r.cvar1_pnl:>10.2f}"
              f"{r.attrib_fees:>9.2f}{r.attrib_il:>9.2f}"
              f"{r.rebalances_per_ep:>9.2f}{100*r.profitable_frac:>9.1f}")
    print("-" * 112)
    print(f"\nWrote {len(df)} rows to {args.out_csv}  "
          f"(total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
