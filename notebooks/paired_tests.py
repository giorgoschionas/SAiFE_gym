"""Significance tests: best RL agent vs best benchmark, per sweep cell.

Consumes the per-trajectory ``.npz`` dumps written by
``reevaluate_saved_models.py --dump-trajectories`` and tests, for each
(cell, metric) pair, whether the RL agent's advantage over the benchmark is
distinguishable from noise.

The data layout
---------------
For one cell, arrange the per-trajectory differences
``d[s,i] = PnL_RL(seed s, path i) - PnL_bench(path i)`` as a grid whose rows
are the N training seeds and whose columns are the M evaluation paths::

                  path 1   path 2   ...   path M  |  row mean
     seed 1        +12.3    - 4.5   ...    +28.7  |   33.5
     ...             ...      ...            ...  |    ...
     seed N        +12.8    - 4.1   ...    +28.2  |   33.9
                                                    -------
                                       overall  =    33.54

Two independent noise sources
-----------------------------
1. **Training seed** -- spread *down* the row-mean column. The RL agent is N
   separately trained models; the benchmarks are rule-based and have no
   training seed at all (their across-seed spread is exactly 0).
2. **Market path** -- spread *across* a row. The reported figure is a mean
   over a finite sample of M paths, so it carries Monte Carlo error even
   though a different evaluation seed would draw from the identical process.

Because every agent sees the same M paths, trajectory ``i`` is directly
comparable across agents and the comparison is genuinely paired. Pairing is
not cosmetic: the agents' PnLs are positively correlated across paths, so
differencing cancels part of the common market movement and shrinks the error
bar materially.

Why the obvious test is not enough
----------------------------------
A one-sample t-test on the N row means (test B below) sees only source 1.
Every seed is scored on the *same* M paths, so the path draw enters each row
mean as an identical constant -- and a constant has zero variance across rows.
Averaging down the column therefore does not attenuate path error at all; it
survives at full strength in the answer while being invisible to the spread.
Test B supports only the narrow claim "averaged over training seeds, on these
particular paths". Empirically the path component is the larger of the two
here, so ignoring it understates the error bar several-fold.

Conversely a paired t-test over the M paths for one fixed model (test A) sees
only source 2, conditioning away the training variance.

The combined test (headline)
----------------------------
Treat the grid as a crossed random-effects layout
``d[s,i] = mu + a_s + b_i + e_si``, so

    Var(mean d) = var_a/N + var_b/M + var_e/(N*M)

Both components are identifiable from a single evaluation seed:

    var_s(mean_i d[s,i])  = var_a + var_e/M          (down the column)
    mean_s(var_i d[s,i])  = var_b + var_e            (across the rows)

giving the plug-in estimator

    SE^2 = var_s(dbar_s)/N + mean_s(var_i d[s,i])/M

which over-counts only by var_e/M, i.e. is mildly *conservative*. Degrees of
freedom follow Satterthwaite.

CVaR
----
CVaR is a functional of the whole return distribution, not a per-trajectory
quantity, so no per-trajectory difference exists and the t-tests above do not
apply. A paired bootstrap instead resamples path indices *once per replicate*
and applies them to both agents (preserving the pairing), while independently
resampling training seeds on the RL side -- carrying both noise sources into
one interval.

Reporting
---------
Alongside the combined test, the per-seed tally (how many of the N models beat
the benchmark individually) is reported as a *consistency* check: it shows the
result is not driven by one lucky seed. It is deliberately not the headline --
the N per-seed tests share one path draw, so they are strongly dependent, and
counting significances discards effect size. Holm-Bonferroni adjusted
p-values control the family-wise error rate over the pre-registered set.

Usage
-----
    conda run -n your_env python paired_tests.py
    conda run -n your_env python paired_tests.py --dumps traj_dumps \
        --out-csv paired_tests.csv --n-boot 10000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# The pre-registered comparison set                                           #
# --------------------------------------------------------------------------- #
# (gas, volatility, phi, rl_agent, benchmark, metric)
#
# Each row pits the best RL agent against the best *benchmark* in that cell by
# that metric, read off the across-seed means. The benchmark deliberately
# varies: ArrivalRebalance beats DeployNarrow on PnL at low gas and CDM has the
# strongest tail there, so pinning one benchmark everywhere would flatter the
# RL agent by comparing it against a weaker opponent than it actually faces.
COMPARISONS = [
    (2, 0.01,  0, "PPO_narrow", "ArrivalRebalance", "pnl"),
    (2, 0.01,  0, "PPO_narrow", "CDM",              "cvar5"),
    (4, 0.02,  0, "PPO_narrow", "DeployNarrow",     "pnl"),
    (4, 0.02,  0, "PPO_narrow", "DeployNarrow",     "cvar5"),
    (6, 0.03,  0, "PPO_narrow", "DeployNarrow",     "pnl"),
    (6, 0.03,  0, "PPO_narrow", "DeployNarrow",     "cvar5"),
    (2, 0.01, 50, "PPO_narrow", "ArrivalRebalance", "pnl"),
    (2, 0.01, 50, "PPO_narrow", "CDM",              "cvar5"),
    (4, 0.02, 50, "PPO_narrow", "DeployNarrow",     "pnl"),
    (4, 0.02, 50, "PPO_narrow", "DeployNarrow",     "cvar5"),
    (6, 0.03, 50, "PPO",        "DeployNarrow",     "pnl"),
    (6, 0.03, 50, "PPO_narrow", "DeployNarrow",     "cvar5"),
]

CVAR_ALPHA = {"cvar5": 0.05, "cvar1": 0.01}

# "<run_id>__<agent>__eval<seed>.npz". Both run ids and agent labels contain
# underscores (PPO_narrow), so anchor on the "__eval" suffix and split the
# remainder on its last "__" rather than trying to tokenise the whole name.
DUMP_RE = re.compile(r"^(?P<rest>.+)__eval(?P<seed>\d+)\.npz$")


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

def load_dumps(dump_dir: Path) -> pd.DataFrame:
    """Index every .npz dump by cell / agent / training seed."""
    rows = []
    for path in sorted(dump_dir.glob("*.npz")):
        m = DUMP_RE.match(path.name)
        if not m or "__" not in m.group("rest"):
            print(f"  skipping unparseable dump name: {path.name}")
            continue
        run_id, agent = m.group("rest").rsplit("__", 1)
        d = np.load(path, allow_pickle=False)
        rows.append({
            "run_id": run_id,
            "agent": agent,
            "eval_seed": int(m.group("seed")),
            "gas_cost": float(d["meta_gas_cost"]),
            "volatility": float(d["meta_volatility"]),
            "inventory_phi": float(d["meta_inventory_phi"]),
            "train_seed": int(d["meta_train_seed"]),
            "pnl": d["pnl"],
        })
    if not rows:
        raise SystemExit(f"No .npz dumps found in {dump_dir}")
    return pd.DataFrame(rows)


def select(df: pd.DataFrame, gas, vol, phi, agent) -> pd.DataFrame:
    """All training seeds of one agent in one cell, ordered by training seed."""
    sel = df[np.isclose(df.gas_cost, gas) & np.isclose(df.volatility, vol)
             & np.isclose(df.inventory_phi, phi) & (df.agent == agent)]
    return sel.sort_values("train_seed")


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

def cvar(pnl: np.ndarray, alpha: float, axis: int = -1) -> np.ndarray:
    """Exact expected shortfall: mean of the worst ``alpha`` fraction.

    Matches ``reevaluate_saved_models.cvar`` (floor, at least one observation)
    but vectorised over leading axes so the bootstrap stays fast.
    """
    n = pnl.shape[axis]
    k = max(1, int(np.floor(alpha * n)))
    part = np.partition(pnl, k - 1, axis=axis)
    sl = [slice(None)] * pnl.ndim
    sl[axis] = slice(0, k)
    return part[tuple(sl)].mean(axis=axis)


def safe_ttest_rel(a: np.ndarray, b: np.ndarray) -> float:
    """Paired t-test p-value, tolerating degenerate inputs.

    Near-cash cells produce PnL vectors that are all (or almost all) zero, for
    which the paired differences have no variance and the t statistic is
    undefined. Return NaN rather than fabricate a p-value.
    """
    d = np.asarray(a) - np.asarray(b)
    if d.size == 0 or not np.all(np.isfinite(d)) or np.allclose(d, d[0]):
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(stats.ttest_rel(a, b).pvalue)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #

def pnl_tests(rl: np.ndarray, bench: np.ndarray) -> dict:
    """Trajectory-paired PnL test propagating training-seed and path noise.

    ``rl`` is (n_seeds, n_traj); ``bench`` is (n_traj,) -- the benchmark is
    deterministic given the paths, so it carries no seed axis.
    """
    n_seeds, n_traj = rl.shape
    d = rl - bench[None, :]           # paired: column i is one shared path
    dbar_s = d.mean(axis=1)           # row means
    dbar = float(dbar_s.mean())

    # --- test A: paired over paths, separately per training seed -----------
    # Conditions on the trained model, so it ignores training variance. Feeds
    # the consistency tally only, never the headline.
    a_p = np.array([safe_ttest_rel(rl[s], bench) for s in range(n_seeds)])
    finite = a_p[np.isfinite(a_p)]

    # --- test B: across training seeds ------------------------------------
    # The benchmark contributes a single constant, so the paired t-test
    # degenerates to a one-sample t-test on the row means.
    if n_seeds > 1 and np.ptp(dbar_s) > 0:
        b = stats.ttest_1samp(dbar_s, 0.0)
        b_t, b_p = float(b.statistic), float(b.pvalue)
    else:
        b_t = b_p = float("nan")

    # --- test C: combined (headline) --------------------------------------
    var_seed = float(np.var(dbar_s, ddof=1)) if n_seeds > 1 else 0.0
    var_traj = float(np.mean(np.var(d, axis=1, ddof=1)))
    c1 = var_seed / n_seeds          # training-seed component
    c2 = var_traj / n_traj           # market-path component
    se = float(np.sqrt(c1 + c2))
    if se > 0:
        t_c = dbar / se
        denom = c2 ** 2 / (n_traj - 1)
        if n_seeds > 1:
            denom += c1 ** 2 / (n_seeds - 1)
        df = (c1 + c2) ** 2 / denom if denom > 0 else float("nan")
        p_c = float(2 * stats.t.sf(abs(t_c), df))
        crit = float(stats.t.ppf(0.975, df))
        ci_lo, ci_hi = dbar - crit * se, dbar + crit * se
    else:
        # Both agents identical on every path (e.g. both sitting in cash).
        t_c = df = float("nan")
        p_c = 1.0
        ci_lo = ci_hi = dbar

    return {
        "diff": dbar,
        "rl_mean": float(rl.mean()),
        "bench_mean": float(bench.mean()),
        "se_combined": se,
        "se_seed": float(np.sqrt(c1)),
        "se_traj": float(np.sqrt(c2)),
        "seed_share_pct": 100.0 * c1 / (c1 + c2) if (c1 + c2) > 0 else float("nan"),
        "t": t_c, "df": df, "p": p_c,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_seeds_ahead": int(np.sum(dbar_s > 0)),
        "n_seeds_testable": int(finite.size),
        "n_sig_05": int(np.sum(finite < 0.05)),
        "n_sig_001": int(np.sum(finite < 0.001)),
        "p_testA_median": float(np.median(finite)) if finite.size else float("nan"),
        "t_testB": b_t, "p_testB": b_p,
    }


def cvar_tests(rl: np.ndarray, bench: np.ndarray, alpha: float,
               n_boot: int, rng: np.random.Generator) -> dict:
    """Paired bootstrap for the CVaR difference.

    Each replicate resamples path indices once and applies them to both agents
    (preserving the pairing), and independently resamples training seeds on the
    RL side so training variance propagates too. The RL statistic is the mean
    CVaR over resampled seeds, matching the estimand "expected CVaR of a policy
    trained by this method".
    """
    n_seeds, n_traj = rl.shape
    obs_rl = float(cvar(rl, alpha, axis=1).mean())
    obs_bn = float(cvar(bench, alpha))

    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_traj, n_traj)      # shared -> preserves pairing
        sdx = rng.integers(0, n_seeds, n_seeds)    # training-seed resample
        boot[b] = (cvar(rl[np.ix_(sdx, idx)], alpha, axis=1).mean()
                   - cvar(bench[idx], alpha))

    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided bootstrap p: mass on the far side of the null, doubled, floored
    # at 1/n_boot -- the bootstrap cannot resolve a p-value finer than that.
    tail = min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    p = float(min(1.0, max(2 * tail, 1.0 / n_boot)))

    per_seed = cvar(rl, alpha, axis=1)
    return {
        "diff": obs_rl - obs_bn,
        "rl_mean": obs_rl,
        "bench_mean": obs_bn,
        "se_combined": float(boot.std(ddof=1)),
        "se_seed": float(np.std(per_seed, ddof=1) / np.sqrt(n_seeds)),
        "se_traj": float("nan"),
        "seed_share_pct": float("nan"),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "p": p,
        "n_seeds_ahead": int(np.sum(per_seed > obs_bn)),
        "n_seeds_testable": -1, "n_sig_05": -1, "n_sig_001": -1,
        "p_testA_median": float("nan"),
        "t": float("nan"), "df": float("nan"),
        "t_testB": float("nan"), "p_testB": float("nan"),
        "boot_n": n_boot,
    }


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def stars(p: float) -> str:
    if not np.isfinite(p):
        return "?"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", type=Path,
                    default=Path(__file__).parent / "traj_dumps",
                    help="Directory of per-trajectory .npz dumps.")
    ap.add_argument("--out-csv", type=Path,
                    default=Path(__file__).parent / "paired_tests.csv")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="Bootstrap replicates for the CVaR tests.")
    ap.add_argument("--boot-seed", type=int, default=20260731,
                    help="Bootstrap RNG seed, so the output is reproducible.")
    args = ap.parse_args()

    print("=" * 78)
    print("Best RL agent vs best benchmark -- significance tests")
    print("=" * 78)
    print(f"  dumps     : {args.dumps}")
    print(f"  bootstrap : {args.n_boot} replicates (seed {args.boot_seed})")
    print()

    df = load_dumps(args.dumps)
    eval_seeds = sorted(df.eval_seed.unique())
    print(f"Loaded {len(df)} dumps | eval seed(s) {eval_seeds} | "
          f"agents {sorted(df.agent.unique())}")
    if len(eval_seeds) > 1:
        raise SystemExit(
            f"Multiple evaluation seeds present {eval_seeds}. The variance "
            f"decomposition assumes one shared path set; point --dumps at a "
            f"directory holding exactly one evaluation seed.")
    print()

    rng = np.random.default_rng(args.boot_seed)
    results = []

    for gas, vol, phi, rl_agent, bench_agent, metric in COMPARISONS:
        rl_sel = select(df, gas, vol, phi, rl_agent)
        bn_sel = select(df, gas, vol, phi, bench_agent)
        if rl_sel.empty or bn_sel.empty:
            print(f"  MISSING gas={gas} vol={vol} phi={phi} "
                  f"{rl_agent} vs {bench_agent} -- skipped")
            continue

        rl = np.vstack(rl_sel["pnl"].to_numpy())
        bench_all = np.vstack(bn_sel["pnl"].to_numpy())

        # The benchmark is rule-based, so it must be identical across training
        # seeds. Verify rather than assume: a non-zero spread would mean the
        # baseline had somehow acquired a dependence on the trained model.
        spread = float(np.abs(bench_all - bench_all[0]).max())
        if spread > 0:
            print(f"  WARNING: {bench_agent} varies across training seeds "
                  f"(max |delta| = {spread:.3e}); averaging.")
            bench = bench_all.mean(axis=0)
        else:
            bench = bench_all[0]

        base = {"gas_cost": gas, "volatility": vol, "inventory_phi": phi,
                "rl_agent": rl_agent, "benchmark": bench_agent, "metric": metric,
                "n_seeds": rl.shape[0], "n_traj": rl.shape[1]}

        if metric == "pnl":
            res = pnl_tests(rl, bench)
            res["test"] = "crossed t (seed x path)"
        else:
            res = cvar_tests(rl, bench, CVAR_ALPHA[metric], args.n_boot, rng)
            res["test"] = "paired bootstrap"
        results.append({**base, **res})

    if not results:
        raise SystemExit("No comparisons could be evaluated.")

    out = pd.DataFrame(results)
    out["p_holm"] = holm(out["p"].to_numpy())
    out["significant_05"] = out["p_holm"] < 0.05
    out.to_csv(args.out_csv, index=False)

    # ----------------------------- report ---------------------------------- #
    for metric, title in (("pnl", "MEAN PnL"), ("cvar5", "CVaR (5%)")):
        sub = out[out.metric == metric]
        if sub.empty:
            continue
        print()
        print("=" * 118)
        print(f"{title}   --   RL minus benchmark; positive favours RL")
        print("=" * 118)
        print(f"{'gas':>4} {'vol':>6} {'phi':>4}  {'RL':<11} {'benchmark':<17}"
              f"{'RL':>9} {'bench':>9} {'diff':>9} {'95% CI':>24} "
              f"{'p':>10} {'p_holm':>10}")
        print("-" * 118)
        for _, r in sub.iterrows():
            ci = f"[{r.ci_lo:+9.3f},{r.ci_hi:+9.3f}]"
            print(f"{r.gas_cost:>4.0f} {r.volatility:>6.2f} {r.inventory_phi:>4.0f}  "
                  f"{r.rl_agent:<11} {r.benchmark:<17}"
                  f"{r.rl_mean:>9.3f} {r.bench_mean:>9.3f} {r['diff']:>+9.3f} "
                  f"{ci:>24} {r.p:>10.2e} {r.p_holm:>10.2e} {stars(r.p_holm)}")
        print("-" * 118)

    # Variance decomposition: explains why a large gap can still be n.s., and
    # documents which noise source the error bar is actually made of.
    pnl_rows = out[out.metric == "pnl"]
    if not pnl_rows.empty:
        print()
        print("=" * 104)
        print("PnL noise decomposition and per-seed consistency")
        print("=" * 104)
        print(f"{'gas':>4} {'vol':>6} {'phi':>4}  {'SE_seed':>8} {'SE_path':>8} "
              f"{'SE_tot':>8} {'seed%':>7}  {'ahead':>7} {'sig<.05':>8} "
              f"{'sig<.001':>9}  {'testB p':>10}")
        print("-" * 104)
        for _, r in pnl_rows.iterrows():
            ahead = f"{r.n_seeds_ahead}/{r.n_seeds}"
            s05 = f"{r.n_sig_05}/{r.n_seeds_testable}"
            s001 = f"{r.n_sig_001}/{r.n_seeds_testable}"
            print(f"{r.gas_cost:>4.0f} {r.volatility:>6.2f} {r.inventory_phi:>4.0f}  "
                  f"{r.se_seed:>8.3f} {r.se_traj:>8.3f} {r.se_combined:>8.3f} "
                  f"{r.seed_share_pct:>6.1f}%  {ahead:>7} {s05:>8} {s001:>9}  "
                  f"{r.p_testB:>10.2e}")
        print("-" * 104)
        print("  SE_seed : training-seed component,  sqrt(var_s(dbar_s)/N)")
        print("  SE_path : market-path component,    sqrt(mean_s var_i(d)/M)")
        print("  seed%   : share of total variance coming from training noise")
        print("  ahead   : training seeds whose mean beats the benchmark")
        print("  sig<..  : seeds individually significant -- shared paths make "
              "these dependent, so read as consistency, not as evidence strength")
        print("  testB p : t over the N seed means; blind to path noise, shown "
              "for contrast with the headline p")

    print()
    print(f"Wrote {len(out)} comparisons to {args.out_csv}")
    print(f"Holm-Bonferroni applied across all {len(out)} pre-registered tests.")


if __name__ == "__main__":
    main()
