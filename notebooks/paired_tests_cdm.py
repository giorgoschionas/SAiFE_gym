"""PPO_narrow vs gas-capped CDM at sigma=.01, g=2.

The main results run CDM with a rebalance deadband (``tolerance=30``) and charge
gas on every re-quote, which is what destroys it: at tolerance=0 it tracks the
optimal range continuously, ~728 times per episode, and the gas bill exceeds the
entire starting capital.

``cdm_no_gas.py --max-gas-events 10`` instead lets CDM run its control law as
the theory states -- continuous re-quoting -- while charging it only the gas an
RL agent could have paid. The RL agents act under a decision stride
(``N_STEPS=1000``, ``DECISION_STRIDE=100``), giving them 10 rebalancing
opportunities and therefore at most 10 gas charges per episode. Capping CDM at
the same budget makes the execution costs comparable, isolating the control laws
themselves.

The cap is enforced inside the environment (``max_gas_events``), not by
subtracting a constant afterwards: gas leaves the LP's capital as it is
incurred, so the reduced position earns correspondingly fewer fees. That
feedback is worth ~1 unit of PnL here, and a post-hoc subtraction would also
have shifted CVaR rigidly instead of letting the loss tail re-form.

Tests are unchanged from ``paired_tests.py`` -- a crossed-variance paired
t-test for PnL and a paired bootstrap for CVaR, both propagating training-seed
and market-path uncertainty. CDM is rule-based, so it contributes no
training-seed axis and its result is identical across phi; only PPO_narrow
differs between the risk-neutral and risk-averse blocks.

Usage
-----
    conda run -n your_env python paired_tests_cdm.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paired_tests import pnl_tests, cvar_tests, load_dumps, select, stars

DEFAULT_GAS, DEFAULT_VOL = 2, 0.01
RL_AGENT = "PPO_narrow"
BENCHMARK = "CDM (tol=0, gas capped at 10)"

# (phi, metric, regime label)
COMPARISONS = [
    (0,  "pnl",   "risk-neutral"),
    (0,  "cvar5", "risk-neutral"),
    (50, "pnl",   "risk-averse"),
    (50, "cvar5", "risk-averse"),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", type=Path,
                    default=Path(__file__).parent / "traj_dumps")
    ap.add_argument("--cdm-dumps", type=Path,
                    default=Path(__file__).parent / "cdm_dumps")
    ap.add_argument("--out-csv", type=Path,
                    default=Path(__file__).parent / "paired_tests_cdm.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=20260731)
    ap.add_argument("--eval-seed", type=int, default=1005)
    ap.add_argument("--gas", type=float, default=DEFAULT_GAS,
                    help="Gas cost of the cell (selects the RL runs).")
    ap.add_argument("--vol", type=float, default=DEFAULT_VOL,
                    help="Volatility of the cell.")
    ap.add_argument("--phis", default="0,50",
                    help="Comma-separated inventory phi values to test.")
    ap.add_argument("--cdm-gas", type=float, default=None,
                    help="Gas cost the CDM run was evaluated at; selects which "
                         "dump file to read. Defaults to the cell's gas cost "
                         "(2). Pass 0 to compare against a gas-free CDM. The RL "
                         "side always pays its real gas.")
    args = ap.parse_args()

    GAS, VOL = args.gas, args.vol
    phis = [int(x) for x in args.phis.split(",") if x.strip()]
    cdm_gas = GAS if args.cdm_gas is None else args.cdm_gas
    cdm_path = (args.cdm_dumps /
                f"s{VOL:g}_g{cdm_gas:g}__CDM-tol0__eval{args.eval_seed}.npz")
    if not cdm_path.exists():
        raise SystemExit(
            f"missing {cdm_path}\nRun: cdm_no_gas.py --gas orig --tolerances 0 "
            f"--max-gas-events 10 --dump-trajectories cdm_dumps")
    cdm = np.load(cdm_path)
    bench = cdm["pnl"]

    print("=" * 78)
    print(f"sigma={VOL}, g={GAS}:  {RL_AGENT}  vs  {BENCHMARK}")
    print("=" * 78)
    print(f"  CDM dump       : {cdm_path.name}")
    print(f"  CDM gas paid   : {cdm['gas'].mean():.2f} over "
          f"{cdm['rebalance_count'].mean():.2f} rebalances/episode")
    print(f"  bootstrap      : {args.n_boot} replicates (seed {args.boot_seed})")
    print()

    df = load_dumps(args.dumps)
    rng = np.random.default_rng(args.boot_seed)
    rows = []

    for phi, metric, regime in [c for c in COMPARISONS if c[0] in phis]:
        sel = select(df, GAS, VOL, phi, RL_AGENT)
        if sel.empty:
            raise SystemExit(f"no {RL_AGENT} dumps for phi={phi}")
        rl = np.vstack(sel["pnl"].to_numpy())

        if metric == "pnl":
            res = pnl_tests(rl, bench)
            res["test"] = "crossed t (seed x path)"
        else:
            res = cvar_tests(rl, bench, 0.05, args.n_boot, rng)
            res["test"] = "paired bootstrap"

        rows.append({
            "regime": regime, "gas_cost": GAS, "volatility": VOL,
            "inventory_phi": phi, "metric": metric, "rl_agent": RL_AGENT,
            "benchmark": "CDM-tol0-capped10", **res,
        })

        winner = "RL" if res["diff"] > 0 else "CDM"
        print(f"{regime.upper()} (phi={phi}), {metric}")
        print(f"   {RL_AGENT} {res['rl_mean']:+8.3f}   CDM {res['bench_mean']:+8.3f}"
              f"   diff {res['diff']:+8.3f}   favours {winner}")
        print(f"   95% CI [{res['ci_lo']:+8.3f}, {res['ci_hi']:+8.3f}]"
              f"    p = {res['p']:.4g}  {stars(res['p'])}")
        if metric == "pnl":
            print(f"   SE_seed {res['se_seed']:.3f}   SE_path {res['se_traj']:.3f}"
                  f"   seed share {res['seed_share_pct']:.1f}%")
        else:
            print(f"   bootstrap SE {res['se_combined']:.3f}")
        print(f"   training seeds ahead of CDM: "
              f"{res['n_seeds_ahead']}/{rl.shape[0]}")
        print()

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} comparisons to {args.out_csv}")


if __name__ == "__main__":
    main()
