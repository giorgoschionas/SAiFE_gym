"""Re-evaluate every saved model under a results tree on a FIXED trajectory seed.

Motivation
----------
``agent_comparison.py`` evaluates each run on ``EVAL_SEED = SEED + 999``, so a
sweep over training seeds changes the *market paths* at the same time as the
*trained policy*. The two effects are then confounded. This script reloads the
persisted PPO / PPO_narrow models and re-runs Phase 2 with the evaluation seed
pinned to ONE value for every run, so all remaining dispersion in the results is
attributable to training variance alone.

The default is 1005 — the value ``agent_comparison`` derived for the canonical
``SEED=6`` config (6 + 999). Two properties matter and both are deliberate:

  * it is *fixed*, not per-run: every policy meets the identical 1000
    trajectories, which is what makes the across-seed spread interpretable as
    training variance and what makes agent-vs-agent paired tests valid;
  * it sits outside the set of training seeds ({2, 4, 6, 8, 10}, and any
    extension of it), so no run is ever evaluated on the RNG stream its own
    training environment was seeded from.

If the training-seed set is ever extended to include 1005, change this.

What it does per run directory
------------------------------
  1. reads ``config.txt`` (the dump written by ``save_config_to_file``);
  2. reads the gas cost from the ``GAS_COST=<n>`` ancestor directory — it is
     NOT recorded in config.txt (see the note below);
  3. rebuilds the environment exactly as ``agent_comparison.create_environment``
     does, but with the fixed evaluation seed and the explicit gas cost;
  4. loads ``<agent>_model.zip`` + ``<agent>_vec_normalize.pkl`` and rebuilds the
     same action wrapper the agent was trained under;
  5. runs ``evaluate_on_trajectories_with_attribution`` — the very function the
     original Phase 2 used — and records the full metric set.

Baseline (rule-based) agents are evaluated too when requested: they carry no
training seed, so on a fixed evaluation seed they must return *identical*
numbers across every run sharing a (gas, volatility) cell. That makes them a
free correctness check on the harness — see ``--agents``.

IMPORTANT — gas cost
--------------------
``create_environment`` in agent_comparison.py never passes ``gas_cost`` to
``UniswapV3ModelDynamics``, so it silently takes the class default (5). The
HPC batches were run with the default patched per submission, and the only
surviving record of the value is the ``GAS_COST=<n>`` directory name. This
script therefore takes gas cost from the directory and passes it explicitly.
Since ``gas_cost`` is also one of the SB3 observation features, getting this
wrong would corrupt both the policy input and the gas attribution.

Usage
-----
    # all runs, PPO + PPO_narrow, eval seed 6
    python reevaluate_saved_models.py --root results_with_seeds

    # include the rule-based baselines as a control
    python reevaluate_saved_models.py --agents all

    # smoke test on two runs
    python reevaluate_saved_models.py --limit 2

Output
------
    <out-csv>  tidy one-row-per-(run, agent) table with every Phase-2 metric
    stdout     per-run tables plus an across-seed variability summary
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from amm_sim.env.AMMEnvironment import AMMEnvironment
from amm_sim.env.ModelDynamics import UniswapV3ModelDynamics
from amm_sim.env.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from amm_sim.stochastic_processes.midprice_models import (
    GeometricBrownianMotionMidpriceModel,
)
from amm_sim.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from amm_sim.agents.BaselineAgents import (
    DoNothingAgent, DeployOnceAgent, CarteaPLAgent, ArrivalRebalanceAgent,
)
from amm_sim.agents.SbAgent import SbAgent
from amm_sim.rewards.RewardFunctions import (
    PnL, RunningInventoryPenalty, ExponentialUtility,
)

# Reuse the *exact* evaluation and action-decoding code paths from the driver so
# the numbers are directly comparable to the original results.txt tables.
from agent_comparison import (  # noqa: E402
    evaluate_on_trajectories_with_attribution,
    DecisionStrideEnv,
    StructuredMultiDiscreteVecEnv,
    NarrowMultiDiscreteVecEnv,
)


GAS_DIR_RE = re.compile(r"GAS_COST=(\d+(?:\.\d+)?)")

# Hard-coded in agent_comparison.create_environment (not part of the sweep, so
# absent from config.txt). Mirrored here so the rebuilt env is byte-identical.
ARRIVAL_BETA = 0.001
ARRIVAL_K = 100
NUM_TICKS = 5000

RL_AGENTS = ["PPO", "PPO_narrow"]
BASELINE_AGENTS = ["DeployNarrow", "DeployWide", "ArrivalRebalance", "CDM"]

# model/vecnormalize filename stems, keyed by agent label
MODEL_STEM = {"PPO": "ppo", "PPO_narrow": "ppo_narrow"}

# Per-trajectory arrays worth persisting for downstream significance testing.
# Scalars in the CSV are collapsed over trajectories, which destroys the
# pairing: two agents evaluated on the same seed see the *same* market paths,
# so trajectory i is comparable across agents and a paired test is available --
# but only if the raw vectors survive. ~8 kB per array, so keeping them is free.
DUMP_KEYS = ["pnl", "fees", "il", "gas", "hodl_pnl", "utility",
             "deploy_count", "rebalance_count"]


# --------------------------------------------------------------------------- #
# config.txt parsing                                                          #
# --------------------------------------------------------------------------- #

def parse_config(path: Path) -> dict:
    """Parse a ``config.txt`` produced by ``save_config_to_file``.

    Values were written with ``repr()`` (arrays as ``np.array([...])``), so
    ``eval`` with numpy in scope is the exact inverse. The input is a file this
    repo generated itself, not untrusted data.
    """
    cfg = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        key, raw = key.strip(), raw.strip()
        try:
            cfg[key] = eval(raw, {"np": np, "nan": float("nan"), "inf": float("inf")})
        except Exception:
            cfg[key] = raw  # leave anything exotic as the literal string
    return cfg


def discover_runs(root: Path) -> list[dict]:
    """Find every run dir under ``root`` holding a config + at least one model.

    Layout-agnostic: recurses to any depth and picks up the gas cost from the
    nearest ``GAS_COST=<n>`` ancestor, so it handles both the flat
    ``results_with_seeds/GAS_COST=n/<job>`` tree and the older
    ``results_newest/GAS_COST=n/<reward>/<job>`` tree.
    """
    runs = []
    for cfg_path in sorted(root.rglob("config.txt")):
        run_dir = cfg_path.parent
        gas = None
        for parent in run_dir.parents:
            m = GAS_DIR_RE.fullmatch(parent.name)
            if m:
                gas = float(m.group(1))
                break
        if gas is None:
            print(f"  [skip] no GAS_COST=<n> ancestor for {run_dir}")
            continue
        models = [a for a in RL_AGENTS
                  if (run_dir / f"{MODEL_STEM[a]}_model.zip").exists()
                  and (run_dir / f"{MODEL_STEM[a]}_vec_normalize.pkl").exists()]
        if not models:
            print(f"  [skip] no saved models in {run_dir}")
            continue
        runs.append({
            "run_dir": run_dir,
            "gas_cost": gas,
            "cfg": parse_config(cfg_path),
            "available_models": models,
        })
    return runs


# --------------------------------------------------------------------------- #
# Environment reconstruction                                                  #
# --------------------------------------------------------------------------- #

def build_reward(cfg: dict):
    kind = cfg.get("REWARD_KIND", "pnl")
    if kind == "pnl":
        return PnL()
    if kind == "inventory":
        return RunningInventoryPenalty(
            per_step_inventory_aversion=cfg["INVENTORY_PHI"],
            terminal_inventory_aversion=cfg["INVENTORY_TERMINAL_AVERSION"],
            inventory_exponent=cfg["INVENTORY_EXPONENT"],
        )
    if kind == "exponential":
        return ExponentialUtility(risk_aversion=cfg["EXP_RISK_AVERSION"])
    raise ValueError(f"unknown REWARD_KIND={kind!r}")


def make_env(cfg: dict, gas_cost: float, num_trajectories: int,
             seed: int, max_gas_events: int = None) -> AMMEnvironment:
    """Rebuild the evaluation environment for one run.

    Mirrors ``agent_comparison.create_environment`` line for line, with two
    deliberate differences: ``seed`` is the fixed evaluation seed rather than
    ``SEED + 999``, and ``gas_cost`` is passed explicitly instead of falling
    through to the ``UniswapV3ModelDynamics`` default.
    """
    step_size = cfg["TERMINAL_TIME"] / cfg["N_STEPS"]
    alpha = np.array([cfg["ALPHA0"], cfg["ALPHA1"], cfg["ALPHA2"], cfg["ALPHA3"]])

    midprice_model = GeometricBrownianMotionMidpriceModel(
        drift=cfg["DRIFT"], volatility=cfg["VOLATILITY"],
        initial_price=cfg["INITIAL_PRICE"], terminal_time=cfg["TERMINAL_TIME"],
        step_size=step_size, num_trajectories=num_trajectories, seed=seed,
    )
    arrival_model = LiquidityKernelArrivalModel(
        alpha=alpha, beta=ARRIVAL_BETA, K=ARRIVAL_K,
        liquidity_scale=cfg["LIQUIDITY_SCALE"], step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1 if seed else None,   # `if seed` idiom kept from the source
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model, arrival_model=arrival_model,
        num_trajectories=num_trajectories, fee_tier=cfg["FEE_TIER"],
        tau=cfg["TAU"], num_ticks=NUM_TICKS,
        exponential_value=cfg["EXP_VALUE"],
        gas_cost=gas_cost,
        max_gas_events=max_gas_events,
        seed=seed + 2 if seed else None,
    )
    return AMMEnvironment(
        terminal_time=cfg["TERMINAL_TIME"], n_steps=cfg["N_STEPS"],
        initial_wealth=cfg["INITIAL_WEALTH"],
        reward_function=build_reward(cfg), model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        initial_pool_price=cfg["INITIAL_POOL_PRICE"], seed=seed,
    )


# --------------------------------------------------------------------------- #
# Action-function builders                                                    #
# --------------------------------------------------------------------------- #

# Config fields a rule-based baseline's behaviour can actually depend on.
# Everything else in config.txt (SEED, REWARD_KIND, INVENTORY_*, all the RL
# hyperparameters) leaves the baselines bit-for-bit identical, because they read
# only the market/pool state and their own fixed offsets — no policy, no reward.
# Two runs agreeing on gas plus these keys therefore produce the same numbers,
# which is what --baselines-once exploits.
BASELINE_CONFIG_KEYS = (
    "TERMINAL_TIME", "N_STEPS", "NUM_TRAJECTORIES_EVAL", "INITIAL_WEALTH",
    "TAU", "LIQUIDITY_SCALE", "INITIAL_PRICE", "INITIAL_POOL_PRICE", "DRIFT",
    "VOLATILITY", "FEE_TIER", "EXP_VALUE",
    "ALPHA0", "ALPHA1", "ALPHA2", "ALPHA3",
    "GAMMA_CARTEA", "REBALANCE_TOLERANCE_CARTEA",
    "ARRIVAL_REBALANCE_EVERY", "ARRIVAL_REBALANCE_WIDTH",
    "ARRIVAL_REBALANCE_LOWER", "ARRIVAL_REBALANCE_UPPER",
    "DEPLOYONCE_LOWER", "DEPLOYONCE_UPPER",
)


def baseline_config_key(cfg: dict, gas_cost: float) -> tuple:
    """Hashable identity of everything a baseline's result depends on."""
    parts = [("gas_cost", gas_cost)]
    for k in BASELINE_CONFIG_KEYS:
        v = cfg.get(k)
        if isinstance(v, np.ndarray):
            v = tuple(v.tolist())
        parts.append((k, v))
    return tuple(parts)


def make_baseline_action_fn(agent: str, env: AMMEnvironment, cfg: dict):
    if agent == "DeployNarrow":
        return DoNothingAgent(env).get_action
    if agent == "DeployWide":
        return DeployOnceAgent(
            env, lower_offset=cfg["DEPLOYONCE_LOWER"],
            upper_offset=cfg["DEPLOYONCE_UPPER"],
        ).get_action
    if agent == "ArrivalRebalance":
        return ArrivalRebalanceAgent(
            env, rebalance_every=cfg["ARRIVAL_REBALANCE_EVERY"],
            width=cfg["ARRIVAL_REBALANCE_WIDTH"],
            lower_offset=cfg["ARRIVAL_REBALANCE_LOWER"],
            upper_offset=cfg["ARRIVAL_REBALANCE_UPPER"],
        ).get_action
    if agent == "CDM":
        return CarteaPLAgent(
            env, gamma=cfg["GAMMA_CARTEA"],
            rebalance_tolerance=cfg["REBALANCE_TOLERANCE_CARTEA"],
            seed=cfg["SEED"],
        ).get_action
    raise ValueError(f"unknown baseline {agent!r}")


def make_rl_action_fn(agent: str, run_dir: Path, env: AMMEnvironment,
                      cfg: dict, num_trajectories: int):
    """Rebuild the trained policy's action function.

    Same composition as agent_comparison's Phase 2:
        state dict -> _flatten_obs -> VecNormalize.normalize_obs
                   -> model.predict -> wrapper.unscale -> [lower, upper, hold]
    """
    stem = MODEL_STEM[agent]
    obs_keys = cfg["SB3_OBS_KEYS"]
    sb_env = StableBaselinesAMMEnvironment(env, obs_keys=obs_keys)
    model = PPO.load(run_dir / f"{stem}_model.zip", device="cpu")

    # VecNormalize.load() wants a live venv to attach to; we only need the
    # frozen running stats, so unpickle directly (same trick as probe_policy).
    with open(run_dir / f"{stem}_vec_normalize.pkl", "rb") as f:
        vec_norm = pickle.load(f)
    vec_norm.training = False   # never update stats during evaluation

    wrapper_cls = (StructuredMultiDiscreteVecEnv if agent == "PPO"
                   else NarrowMultiDiscreteVecEnv)
    # Instantiated purely for its .unscale(); it is not stepped.
    wrapper = wrapper_cls(sb_env, cfg["TAU"])
    sb_agent = SbAgent(model, num_trajectories=num_trajectories)

    def action_fn(state):
        flat = sb_env._flatten_obs(state)
        return wrapper.unscale(sb_agent.get_action(vec_norm.normalize_obs(flat)))

    return action_fn


# --------------------------------------------------------------------------- #
# Metric assembly                                                             #
# --------------------------------------------------------------------------- #

def cvar(pnl: np.ndarray, alpha: float) -> float:
    """Exact conditional value-at-risk (expected shortfall) of the loss tail.

    ``CVaR_alpha = E[PnL | PnL <= VaR_alpha]`` — the mean of the worst
    ``alpha`` fraction of trajectories. Computed directly from the sample, so
    it is exact rather than a moment-based approximation: a Cornish-Fisher or
    Gaussian reconstruction from (mean, std, skew, kurt) misses by ~0.5 and
    ~1.7-9.6 respectively on these distributions, and degenerates entirely on
    near-cash cells where the higher moments are undefined.

    More negative = worse tail. For a cash policy (all-zero PnL) it is 0.
    """
    n = len(pnl)
    k = max(1, int(np.floor(alpha * n)))
    return float(np.sort(pnl)[:k].mean())


def dump_trajectories(out_dir: Path, run_id: str, agent: str, eval_seed: int,
                      res: dict, meta: dict) -> Path:
    """Persist the per-trajectory arrays for one (run, agent) evaluation.

    Write-only side channel: called after ``res`` is computed and never read
    back into the evaluation, so enabling it cannot perturb any reported metric.

    One ``.npz`` per evaluation, named so the (run, agent, eval seed) triple is
    recoverable from the filename alone. Config scalars ride along inside the
    archive so downstream analysis never has to re-read config.txt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}__{agent}__eval{eval_seed}.npz"
    arrays = {k: np.asarray(res[k]) for k in DUMP_KEYS if k in res}
    np.savez_compressed(path, **arrays,
                        **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    return path


def summarise(res: dict) -> dict:
    """Collapse the per-trajectory arrays into the reported scalar metrics.

    Column names and definitions match the Phase-2 tables in results.txt
    (including the P5 / P95 percentiles) plus the Phase-4 attribution row,
    with the exact tail risk measures (CVaR) added.
    """
    pnl = res["pnl"]
    ms, ts = res["mean_spread"], res["temporal_std_spread"]
    mc, tc = res["mean_center"], res["temporal_std_center"]
    mdc, tdc = res["mean_deploy_center"], res["temporal_std_deploy_center"]
    decisions = max(int(res["decision_count"]), 1)
    reb = res["rebalance_count"]

    def nanmean(a):
        return float(np.nanmean(a)) if np.any(~np.isnan(a)) else float("nan")

    def nanstd(a):
        return float(np.nanstd(a)) if np.any(~np.isnan(a)) else float("nan")

    try:
        p_w = float(stats.wilcoxon(pnl).pvalue)
    except ValueError:
        p_w = float("nan")

    return {
        "mean_pnl":    float(np.mean(pnl)),
        "std_pnl":     float(np.std(pnl)),
        "median_pnl":  float(np.median(pnl)),
        "p5_pnl":      float(np.percentile(pnl, 5)),
        "p95_pnl":     float(np.percentile(pnl, 95)),
        # Exact expected shortfall of the loss tail (mean of the worst 5% / 1%).
        # p5_pnl is the *threshold*; cvar5_pnl is the mean *beyond* it, so
        # cvar5_pnl <= p5_pnl always.
        "cvar5_pnl":   cvar(pnl, 0.05),
        "cvar1_pnl":   cvar(pnl, 0.01),
        "skew_pnl":    float(stats.skew(pnl)),
        "kurt_pnl":    float(stats.kurtosis(pnl)),
        "profitable_frac": float(np.mean(pnl > 0)),
        "p_ttest":     float(stats.ttest_1samp(pnl, 0.0).pvalue),
        "p_wilcoxon":  p_w,
        "mean_utility": float(np.mean(res["utility"])),
        "mean_spread":          nanmean(ms),
        "spread_std_within":    nanmean(ts),
        "spread_std_between":   nanstd(ms),
        "mean_center":          nanmean(mc),
        "center_std_within":    nanmean(tc),
        "center_std_between":   nanstd(mc),
        "deploy_mean_center":       nanmean(mdc),
        "deploy_center_std_within": nanmean(tdc),
        "deploy_center_std_between": nanstd(mdc),
        "deploys_per_ep":     float(res["deploy_count"].mean()),
        "rebalances_per_ep":  float(reb.mean()),
        "rebalance_rate_pct": 100.0 * float(reb.mean()) / decisions,
        "decisions_per_ep":   decisions,
        "attrib_fees": float(np.mean(res["fees"])),
        "attrib_il":   float(np.mean(res["il"])),
        "attrib_gas":  float(np.mean(res["gas"])),
        "attrib_hodl": float(np.mean(res["hodl_pnl"])),
    }


#            key                 label      width  format
PNL_COLS = [
    ("mean_pnl",          "Mean",     9, "+9.2f"),
    ("std_pnl",           "Std",      9,  "9.2f"),
    ("median_pnl",        "Median",   9, "+9.2f"),
    ("p5_pnl",            "P5",       9, "+9.2f"),
    ("p95_pnl",           "P95",      9, "+9.2f"),
    ("skew_pnl",          "Skew",     7, "+7.2f"),
    ("kurt_pnl",          "Kurt",     7, "+7.2f"),
    ("profitable_frac",   "Profit",   7,  "7.1%"),
    ("mean_spread",       "Spread",   8,  "8.3f"),
    ("rebalances_per_ep", "Reb/ep",   7,  "7.2f"),
]


def print_run_table(rows: dict):
    """Print a Phase-2-style table, including the new P5 / P95 columns."""
    hdr = "  " + f"{'Agent':<18}" + "".join(
        f" | {lbl:>{width}}" for _, lbl, width, _ in PNL_COLS)
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for agent, m in rows.items():
        cells = "".join(f" | {m[key]:>{fmt}}" for key, _, _, fmt in PNL_COLS)
        print(f"  {agent:<18}{cells}")
    print("  " + "-" * (len(hdr) - 2))


# --------------------------------------------------------------------------- #
# Across-seed variability summary                                             #
# --------------------------------------------------------------------------- #

def write_variability_summary(df: pd.DataFrame, out_csv: Path, eval_seed) -> pd.DataFrame:
    """Collapse the per-run table to one row per (gas, vol, phi, agent).

    Every run in a group shares one evaluation seed, so the dispersion of
    ``mean_pnl`` across the group is pure training variance.

    Columns, and what each is a mean *of*:
      mean / std / min / max / range / cv_pct
          statistics of the per-run ``mean_pnl`` values (one per training seed).
          ``mean`` is therefore a mean of means; ``std`` is the training-seed
          dispersion — the "+/-" for a results table.
      p5_mean / p95_mean
          the per-run 5th / 95th trajectory percentiles, averaged across seeds.
          This is the "typical policy's" tail, and is the right partner for
          ``mean`` in a caption reading "quantiles over trajectories, +/- over
          training seeds": pooling all seeds' trajectories instead would fold
          training variance into the bracket and double-count it against the +/-.
      p5_std / p95_std
          how much those tail estimates themselves move across seeds.
      std_pnl_mean
          mean of the per-run trajectory-level std (dispersion *within* a run),
          kept distinct from ``std`` above to avoid conflating the two.
    """
    keys = ["gas_cost", "volatility", "inventory_phi", "agent"]
    g = (df.groupby(keys)["mean_pnl"]
           .agg(n_seeds="count", mean="mean", std="std", min="min", max="max")
           .reset_index())
    g["range"] = g["max"] - g["min"]
    g["cv_pct"] = 100.0 * g["std"] / g["mean"].abs()

    agg = dict(p5_mean=("p5_pnl", "mean"), p5_std=("p5_pnl", "std"),
               p95_mean=("p95_pnl", "mean"), p95_std=("p95_pnl", "std"),
               std_pnl_mean=("std_pnl", "mean"))
    # CVaR columns are absent from CSVs written before they were added, so
    # aggregate them only when present (keeps --summary-only working on old files).
    has_cvar = "cvar5_pnl" in df.columns
    if has_cvar:
        agg.update(cvar5_mean=("cvar5_pnl", "mean"), cvar5_std=("cvar5_pnl", "std"),
                   cvar1_mean=("cvar1_pnl", "mean"), cvar1_std=("cvar1_pnl", "std"))
    extra = df.groupby(keys).agg(**agg).reset_index()
    g = g.merge(extra, on=keys)

    # Order columns so the table reads mean, [P5, P95], +/- left to right.
    cvar_cols = ["cvar5_mean", "cvar5_std", "cvar1_mean", "cvar1_std"] if has_cvar else []
    g = g[keys + ["n_seeds", "mean", "p5_mean", "p95_mean", "std",
                  "p5_std", "p95_std"] + cvar_cols +
          ["std_pnl_mean", "min", "max", "range", "cv_pct"]]

    print("\n" + "=" * 92)
    print("TRAINING VARIABILITY  (same market paths, different training seeds)")
    print("=" * 92)
    print("  Dispersion is pure training variance: every row shares one")
    print(f"  evaluation seed ({eval_seed}), so market paths are identical.")
    print("  mean / P5 / P95 are averaged across seeds; +/- is the std of the")
    print("  per-seed means. Table-ready form:  mean [P5, P95] +/- std\n")

    cv_hdr = f"{'CVaR5':>9} {'CVaR1':>9} " if has_cvar else ""
    hdr = (f"  {'gas':>4} {'vol':>6} {'phi':>5} {'agent':<12} {'n':>3} "
           f"{'mean':>9} {'P5':>9} {'P95':>9} {cv_hdr}{'+/-':>7} "
           f"{'p5 sd':>7} {'p95 sd':>7} {'cv%':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in g.sort_values(["agent", "gas_cost", "volatility", "inventory_phi"]).iterrows():
        cv = (f"{r['cvar5_mean']:>+9.2f} {r['cvar1_mean']:>+9.2f} ") if has_cvar else ""
        print(f"  {r['gas_cost']:>4g} {r['volatility']:>6g} {r['inventory_phi']:>5g} "
              f"{r['agent']:<12} {int(r['n_seeds']):>3} {r['mean']:>+9.2f} "
              f"{r['p5_mean']:>+9.2f} {r['p95_mean']:>+9.2f} {cv}{r['std']:>7.2f} "
              f"{r['p5_std']:>7.2f} {r['p95_std']:>7.2f} {r['cv_pct']:>7.1f}")
    print("  " + "-" * (len(hdr) - 2))

    summary_path = out_csv.with_name(out_csv.stem + "_variability.csv")
    g.to_csv(summary_path, index=False)
    print(f"\nWrote across-seed summary to {summary_path}")
    return g


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "results_with_seeds",
                    help="Results tree to walk (default: results_with_seeds).")
    ap.add_argument("--out-csv", type=Path,
                    default=Path(__file__).parent / "reevaluated_fixed_seed.csv")
    ap.add_argument("--eval-seed", type=int, default=1005,
                    help="Trajectory seed, identical for every run (default: 1005 "
                         "= 6+999, the EVAL_SEED agent_comparison.py derived for the "
                         "canonical SEED=6 config). Held fixed on purpose: a per-run "
                         "seed would change the market paths alongside the policy and "
                         "reintroduce the confound. Keep it clear of every training "
                         "seed so no run is evaluated on its own training stream.")
    ap.add_argument("--agents", default="rl", choices=["rl", "all", "baselines"],
                    help="'rl' = PPO + PPO_narrow (default); 'all' adds the "
                         "rule-based baselines as a control; 'baselines' only.")
    ap.add_argument("--num-trajectories", type=int, default=None,
                    help="Override NUM_TRAJECTORIES_EVAL from config.txt.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N runs (smoke testing).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip runs already present in --out-csv and append.")
    ap.add_argument("--baselines-once", action="store_true",
                    help="Evaluate each rule-based baseline once per distinct "
                         "(gas, market-config) group and reuse the result for every "
                         "run sharing it, instead of recomputing per training seed "
                         "and phi. Identical output, ~18x faster on the current tree.")
    ap.add_argument("--allow-seed-collision", action="store_true",
                    help="Permit --eval-seed to equal one of the training seeds. "
                         "Off by default; see the seed note in the module docstring.")
    ap.add_argument("--dump-trajectories", type=Path, default=None,
                    help="Directory to write per-trajectory .npz arrays into "
                         "(one per run/agent). Required for paired significance "
                         "tests, which need the raw vectors rather than the "
                         "collapsed scalars. Off by default; purely additive.")
    ap.add_argument("--cells", default=None,
                    help="Restrict to specific sweep cells, as a comma-separated "
                         "list of gas:vol:phi triples, e.g. "
                         "'2:0.01:0,4:0.02:50'. Default: every discovered run.")
    ap.add_argument("--include-agents", default=None,
                    help="Explicit comma-separated agent list, overriding "
                         "--agents. e.g. 'PPO_narrow,DeployNarrow,CDM'.")
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip evaluation entirely: recompute the variability "
                         "summary from an existing --out-csv. The summary is a "
                         "pure function of that file, so this is instant.")
    args = ap.parse_args()

    if args.summary_only:
        if not args.out_csv.exists():
            raise SystemExit(f"--summary-only needs an existing {args.out_csv}")
        df = pd.read_csv(args.out_csv)
        seeds = sorted(df["eval_seed"].unique())
        print(f"Recomputing summary from {args.out_csv} "
              f"({len(df)} rows, eval seed(s) {seeds}).")
        write_variability_summary(df, args.out_csv,
                                  seeds[0] if len(seeds) == 1 else seeds)
        return

    if args.include_agents:
        wanted = [a.strip() for a in args.include_agents.split(",") if a.strip()]
        unknown = [a for a in wanted if a not in RL_AGENTS + BASELINE_AGENTS]
        if unknown:
            raise SystemExit(
                f"--include-agents: unknown agent(s) {unknown}. "
                f"Known: {RL_AGENTS + BASELINE_AGENTS}")
    elif args.agents == "rl":
        wanted = list(RL_AGENTS)
    elif args.agents == "baselines":
        wanted = list(BASELINE_AGENTS)
    else:
        wanted = BASELINE_AGENTS + RL_AGENTS

    print("=" * 78)
    print("Re-evaluating saved models on a FIXED trajectory seed")
    print("=" * 78)
    print(f"  root       : {args.root}")
    print(f"  eval seed  : {args.eval_seed}  (fixed for every run)")
    print(f"  agents     : {', '.join(wanted)}")
    print(f"  out csv    : {args.out_csv}")
    print()

    runs = discover_runs(args.root)
    if not runs:
        raise SystemExit(f"No runs with saved models found under {args.root}")
    print(f"Found {len(runs)} run(s) with saved models.")

    # A run whose training seed equals the evaluation seed would be scored on
    # the RNG stream its own training env was seeded from. Refuse rather than
    # quietly produce one contaminated row among many.
    #
    # Checked against the FULL discovered set, before --limit is applied: a
    # truncated smoke run must not silently pass a seed choice that would
    # collide on the full sweep.
    train_seeds = sorted({r["cfg"]["SEED"] for r in runs})
    print(f"Training seeds present: {train_seeds}")
    if args.eval_seed in train_seeds and not args.allow_seed_collision:
        raise SystemExit(
            f"\nERROR: --eval-seed {args.eval_seed} is also a TRAINING seed.\n"
            f"  The run(s) trained at seed {args.eval_seed} would be evaluated on the\n"
            f"  same RNG stream they were trained from, while every other run is not —\n"
            f"  an asymmetry that undermines the comparison.\n"
            f"  Pick a seed outside {train_seeds} (default 1005), or pass\n"
            f"  --allow-seed-collision to override deliberately."
        )
    # Cell filter is applied *after* the seed-collision guard so that narrowing
    # the sweep can never smuggle in a seed choice the full grid would reject.
    if args.cells:
        want_cells = set()
        for spec in args.cells.split(","):
            spec = spec.strip()
            if not spec:
                continue
            try:
                g, v, p = spec.split(":")
            except ValueError:
                raise SystemExit(f"--cells: bad triple {spec!r}; want gas:vol:phi")
            want_cells.add((float(g), float(v), float(p)))
        runs = [r for r in runs
                if (float(r["gas_cost"]), float(r["cfg"]["VOLATILITY"]),
                    float(r["cfg"]["INVENTORY_PHI"])) in want_cells]
        if not runs:
            raise SystemExit(f"--cells matched no runs: {sorted(want_cells)}")
        matched = {(float(r["gas_cost"]), float(r["cfg"]["VOLATILITY"]),
                    float(r["cfg"]["INVENTORY_PHI"])) for r in runs}
        print(f"--cells: {len(runs)} run(s) in {len(matched)} cell(s).")
        for miss in sorted(want_cells - matched):
            print(f"   WARNING: no runs for requested cell gas={miss[0]:g} "
                  f"vol={miss[1]:g} phi={miss[2]:g}")

    if args.limit:
        runs = runs[:args.limit]
        print(f"--limit: evaluating the first {len(runs)} run(s).")
    print()

    done = set()
    existing = []
    if args.resume and args.out_csv.exists():
        prev = pd.read_csv(args.out_csv)
        existing = [prev]
        done = set(zip(prev["run_id"], prev["agent"]))
        print(f"Resuming: {len(done)} (run, agent) pair(s) already done.\n")

    rows = []
    baseline_cache: dict = {}
    t_start = time.time()

    if any(a in BASELINE_AGENTS for a in wanted):
        n_groups = len({baseline_config_key(r["cfg"], r["gas_cost"]) for r in runs})
        if args.baselines_once:
            print(f"--baselines-once: {n_groups} distinct baseline configuration(s) "
                  f"across {len(runs)} run(s); each baseline is evaluated "
                  f"{n_groups} time(s) and reused elsewhere.\n")
        else:
            print(f"NOTE: baselines do not depend on the training seed or phi, so the "
                  f"{len(runs)} run(s)\n      collapse to {n_groups} distinct baseline "
                  f"configuration(s). Without --baselines-once\n      each is recomputed "
                  f"{len(runs) // max(n_groups, 1)}x over. Pass --baselines-once to skip "
                  f"the redundancy.\n")
    for i, run in enumerate(runs, 1):
        run_dir, cfg, gas = run["run_dir"], run["cfg"], run["gas_cost"]
        run_id = run_dir.name
        n_eval = args.num_trajectories or cfg["NUM_TRAJECTORIES_EVAL"]
        stride = cfg["DECISION_STRIDE"]

        print(f"[{i}/{len(runs)}] {run_dir.relative_to(args.root)}   "
              f"seed={cfg['SEED']} vol={cfg['VOLATILITY']} gas={gas:g} "
              f"phi={cfg['INVENTORY_PHI']} n_eval={n_eval}")

        todo = [a for a in wanted
                if (a in BASELINE_AGENTS or a in run["available_models"])
                and (run_id, a) not in done]
        if not todo:
            print("   (nothing to do)\n")
            continue

        run_rows = {}
        for agent in todo:
            t0 = time.time()

            # --baselines-once: a baseline's result is fully determined by
            # (gas, market/pool config) — not by the training seed or phi — so
            # evaluate it for the first run of each distinct group and reuse the
            # numbers for the rest, tagging the source run for traceability.
            if agent in BASELINE_AGENTS and args.baselines_once:
                bkey = (agent, baseline_config_key(cfg, gas))
                if bkey in baseline_cache:
                    src, metrics, cached_res = baseline_cache[bkey]
                    run_rows[agent] = metrics
                    # Re-emit the (identical) arrays under this run_id so every
                    # (run, agent) pair has a dump and the analysis needs no
                    # special case for reused baselines.
                    if args.dump_trajectories is not None:
                        dump_trajectories(
                            args.dump_trajectories, run_id, agent,
                            args.eval_seed, cached_res,
                            {"gas_cost": gas, "train_seed": cfg["SEED"],
                             "volatility": cfg["VOLATILITY"],
                             "inventory_phi": cfg["INVENTORY_PHI"],
                             "n_eval": n_eval, "reused_from": src})
                    rows.append({
                        "run_id": run_id,
                        "run_dir": str(run_dir.relative_to(args.root)),
                        "gas_cost": gas, "train_seed": cfg["SEED"],
                        "volatility": cfg["VOLATILITY"],
                        "reward_kind": cfg["REWARD_KIND"],
                        "inventory_phi": cfg["INVENTORY_PHI"],
                        "tau": cfg["TAU"], "decision_stride": stride,
                        "eval_seed": args.eval_seed, "n_eval": n_eval,
                        "agent": agent, "baseline_source_run": src,
                        **metrics,
                    })
                    print(f"   {agent:<12} reused from {src}")
                    continue

            # Fresh env per agent so every one sees the identical seeded paths.
            env = make_env(cfg, gas, n_eval, args.eval_seed)
            if agent in BASELINE_AGENTS:
                # Baselines act on EVERY env step in agent_comparison's Phase 2 —
                # only the RL agents are wrapped in DecisionStrideEnv there. (The
                # write-up in docs/experiment_setup.md says all agents share the
                # stride, but the code does not; we match the code, since that is
                # what produced the published numbers.)
                action_fn = make_baseline_action_fn(agent, env, cfg)
                eval_env = env
            else:
                action_fn = make_rl_action_fn(agent, run_dir, env, cfg, n_eval)
                eval_env = DecisionStrideEnv(env, stride=stride)
            res = evaluate_on_trajectories_with_attribution(eval_env, action_fn)
            metrics = summarise(res)
            if args.dump_trajectories is not None:
                dump_trajectories(
                    args.dump_trajectories, run_id, agent, args.eval_seed, res,
                    {"gas_cost": gas, "train_seed": cfg["SEED"],
                     "volatility": cfg["VOLATILITY"],
                     "inventory_phi": cfg["INVENTORY_PHI"],
                     "n_eval": n_eval, "reused_from": ""})
            if agent in BASELINE_AGENTS and args.baselines_once:
                baseline_cache[(agent, baseline_config_key(cfg, gas))] = (
                    run_id, metrics,
                    {k: np.asarray(res[k]) for k in DUMP_KEYS if k in res})
            run_rows[agent] = metrics
            rows.append({
                "run_id": run_id,
                "run_dir": str(run_dir.relative_to(args.root)),
                "gas_cost": gas,
                "train_seed": cfg["SEED"],
                "volatility": cfg["VOLATILITY"],
                "reward_kind": cfg["REWARD_KIND"],
                "inventory_phi": cfg["INVENTORY_PHI"],
                "tau": cfg["TAU"],
                "decision_stride": stride,
                "eval_seed": args.eval_seed,
                "n_eval": n_eval,
                "agent": agent,
                **metrics,
            })
            print(f"   {agent:<12} mean={metrics['mean_pnl']:+8.2f} "
                  f"std={metrics['std_pnl']:7.2f} "
                  f"[P5 {metrics['p5_pnl']:+7.2f}, P95 {metrics['p95_pnl']:+7.2f}] "
                  f"({time.time() - t0:.1f}s)")

        if run_rows:
            print()
            print_run_table(run_rows)
            print()

        # Flush after every run so a long job is never lost.
        df_out = pd.concat(existing + [pd.DataFrame(rows)], ignore_index=True)
        df_out.to_csv(args.out_csv, index=False)

    if not rows:
        print("Nothing evaluated.")
        return

    df = pd.concat(existing + [pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {args.out_csv}  "
          f"(total {time.time() - t_start:.0f}s)")

    write_variability_summary(df, args.out_csv, args.eval_seed)


if __name__ == "__main__":
    main()
