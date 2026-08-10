"""PPO_narrow vs PPO on mean PnL, per sweep cell.

Companion to ``paired_tests.py``, which compares the best RL agent against the
best rule-based benchmark. This one compares the two RL variants with each
other, to test whether restricting the action space actually helps.

Why the pairing differs from paired_tests.py
--------------------------------------------
There, the benchmark is rule-based and carries no training-seed variance, so
only the RL side contributed a seed axis. Here *both* sides are trained, and
crucially both are trained inside the same run directory -- so training seed s
denotes a matched pair sharing one environment RNG stream. Both agents are
also evaluated on the same 1000 paths. The comparison is therefore paired on
*both* axes, and the same crossed-variance estimator applies:

    SE^2 = var_s(dbar_s)/N + mean_s(var_i d[s,i])/M

Pairing on seeds is safe even if the two agents' seeds were effectively
independent: in that case var_s(dbar_s) = var_a1 + var_a2, and dividing by N
reproduces the unpaired two-sample variance exactly. Where they are positively
correlated it is strictly tighter. So it is never worse than not pairing.

Note that ``seed%`` runs much higher here than in the benchmark comparison
(both sides carry training noise, while the shared path noise largely cancels
between two similar agents), so training variance genuinely dominates this
table -- the opposite of the RL-vs-benchmark case.

Usage
-----
    conda run -n your_env python ppo_vs_narrow.py
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paired_tests import holm, stars

CELLS = [(2.0, 0.01, 0.0), (4.0, 0.02, 0.0), (6.0, 0.03, 0.0),
         (2.0, 0.01, 50.0), (4.0, 0.02, 50.0), (6.0, 0.03, 50.0)]


def load(dump_dir: Path, cell, agent, eval_seed: int):
    """Every training seed of one agent in one cell, sorted by (seed, run)."""
    rows = []
    for f in glob.glob(str(dump_dir / f"*__{agent}__eval{eval_seed}.npz")):
        d = np.load(f)
        if (float(d["meta_gas_cost"]), float(d["meta_volatility"]),
                float(d["meta_inventory_phi"])) == cell:
            run = Path(f).name.split("__")[0]
            rows.append((int(d["meta_train_seed"]), run, d["pnl"]))
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", type=Path,
                    default=Path(__file__).parent / "traj_dumps")
    ap.add_argument("--out-csv", type=Path,
                    default=Path(__file__).parent / "ppo_vs_narrow.csv")
    ap.add_argument("--eval-seed", type=int, default=1005)
    args = ap.parse_args()

    res = []
    for cell in CELLS:
        a = load(args.dumps, cell, "PPO_narrow", args.eval_seed)
        b = load(args.dumps, cell, "PPO", args.eval_seed)
        if not a or not b:
            print(f"  MISSING data for cell {cell} -- skipped")
            continue
        # The pairing is only meaningful if the rows really do correspond.
        # Verify rather than assume: same training seeds, same run directories.
        if [s for s, _, _ in a] != [s for s, _, _ in b]:
            raise SystemExit(f"cell {cell}: training seeds do not match")
        if [r for _, r, _ in a] != [r for _, r, _ in b]:
            raise SystemExit(f"cell {cell}: run dirs differ -- not matched pairs")

        X = np.vstack([p for _, _, p in a])       # PPO_narrow
        Y = np.vstack([p for _, _, p in b])       # PPO
        N, M = X.shape
        d = X - Y                                  # positive favours PPO_narrow
        dbar_s = d.mean(axis=1)
        dbar = float(dbar_s.mean())

        c1 = float(np.var(dbar_s, ddof=1)) / N               # training-seed
        c2 = float(np.mean(np.var(d, axis=1, ddof=1))) / M   # market path
        se = float(np.sqrt(c1 + c2))
        if se > 0:
            df = (c1 + c2) ** 2 / (c1 ** 2 / (N - 1) + c2 ** 2 / (M - 1))
            t = dbar / se
            p = float(2 * stats.t.sf(abs(t), df))
            crit = float(stats.t.ppf(0.975, df))
            lo, hi = dbar - crit * se, dbar + crit * se
        else:
            # Both variants identical on every path (both sitting in cash).
            df = t = float("nan")
            p, lo, hi = 1.0, dbar, dbar

        res.append({
            "gas_cost": cell[0], "volatility": cell[1], "inventory_phi": cell[2],
            "ppo_narrow_mean": float(X.mean()), "ppo_mean": float(Y.mean()),
            "diff": dbar, "se_combined": se,
            "se_seed": float(np.sqrt(c1)), "se_path": float(np.sqrt(c2)),
            "seed_share_pct": 100 * c1 / (c1 + c2) if (c1 + c2) > 0 else float("nan"),
            "t": t, "df": df, "p": p, "ci_lo": lo, "ci_hi": hi,
            "n_seeds_ahead": int((dbar_s > 0).sum()), "n_seeds": N, "n_traj": M,
        })

    if not res:
        raise SystemExit("No cells could be evaluated.")

    out = pd.DataFrame(res)
    out["p_holm"] = holm(out["p"].to_numpy())
    out["significant_05"] = out["p_holm"] < 0.05
    out.to_csv(args.out_csv, index=False)

    print("=" * 108)
    print("PPO_narrow  vs  PPO   --   mean PnL   (positive favours PPO_narrow)")
    print("=" * 108)
    print(f"{'gas':>4}{'vol':>7}{'phi':>5}  {'PPO_narrow':>11}{'PPO':>9}{'diff':>9}"
          f"{'95% CI':>22}{'p':>11}{'p_holm':>11}  {'ahead':>7}{'seed%':>7}")
    print("-" * 108)
    for _, r in out.iterrows():
        ci = f"[{r.ci_lo:+7.2f},{r.ci_hi:+7.2f}]"
        print(f"{r.gas_cost:>4.0f}{r.volatility:>7.2f}{r.inventory_phi:>5.0f}  "
              f"{r.ppo_narrow_mean:>11.2f}{r.ppo_mean:>9.2f}{r['diff']:>+9.2f}"
              f"{ci:>22}{r.p:>11.2e}{r.p_holm:>11.2e} {stars(r.p_holm):>4} "
              f"{r.n_seeds_ahead}/{r.n_seeds:<3}{r.seed_share_pct:>6.1f}%")
    print("-" * 108)
    print("  Paired on BOTH axes: training seed (same run dir -> same env RNG")
    print("  stream) and evaluation path (same trajectories). Holm-Bonferroni")
    print(f"  applied across the {len(out)} tests.")
    print()
    print(f"Wrote {len(out)} comparisons to {args.out_csv}")


if __name__ == "__main__":
    main()
