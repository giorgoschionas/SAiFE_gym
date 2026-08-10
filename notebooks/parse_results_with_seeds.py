"""Parse all runs under results_with_seeds/ into a single summary txt file.

Same logic as ``parse_results.py``, adapted to the seeded sweep:

  * directory pattern is ``GAS_COST=<n>/SEED=<s>/<job_id>/`` (the older tree
    parsed by ``parse_results.py`` is ``GAS_COST=<n>/<reward_kind>/<job_id>/``);
  * the training ``SEED`` is captured and reported, since it is the dimension
    the sweep varies;
  * runs are emitted grouped by configuration with the seeds adjacent, so the
    training variability for one (gas, phi, volatility) cell reads top-to-bottom
    in one block rather than being scattered across the file.

For each leaf run directory it extracts:
  - gas_cost   (from the ``GAS_COST=<n>`` folder)
  - seed       (from the ``SEED=<s>`` folder, cross-checked against config.txt)
  - VOLATILITY, REWARD_KIND, INVENTORY_PHI (from config.txt)
  - the full Phase 2 and Phase 4 blocks from results.txt

Phase blocks are copied verbatim, so the newer ``P5`` / ``P95`` columns in the
PnL table pass straight through with no changes needed here.

Usage:
    python parse_results_with_seeds.py
    python parse_results_with_seeds.py --root results_with_seeds --out parsed_results_with_seeds.txt
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Block extraction is column-agnostic and already correct — reuse it rather
# than re-implementing the phase-banner scanning.
from parse_results import extract_phase2, extract_phase4  # noqa: E402


GAS_DIR_RE = re.compile(r"GAS_COST=(\d+(?:\.\d+)?)$")
SEED_DIR_RE = re.compile(r"SEED=(\d+)$")
CONFIG_KEYS = ("VOLATILITY", "REWARD_KIND", "INVENTORY_PHI", "SEED")


def parse_config(config_path: Path) -> dict:
    values = {}
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key in CONFIG_KEYS:
            values[key] = val.strip()
    missing = [k for k in CONFIG_KEYS if k not in values]
    if missing:
        raise ValueError(f"{config_path}: missing keys {missing}")
    return values


def find_runs(root: Path):
    """Yield (gas_cost, seed, run_dir) for every leaf run dir under the tree.

    Expects ``GAS_COST=<n>/SEED=<s>/<run>/`` and requires both config.txt and
    results.txt to be present.
    """
    for gas_dir in sorted(root.iterdir()):
        m_gas = GAS_DIR_RE.match(gas_dir.name)
        if not m_gas or not gas_dir.is_dir():
            continue
        gas_cost = float(m_gas.group(1))
        for seed_dir in sorted(gas_dir.iterdir()):
            m_seed = SEED_DIR_RE.match(seed_dir.name)
            if not m_seed or not seed_dir.is_dir():
                continue
            seed = int(m_seed.group(1))
            for run_dir in sorted(seed_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                if (run_dir / "config.txt").exists() and (run_dir / "results.txt").exists():
                    yield gas_cost, seed, run_dir


def risk_label(cfg: dict) -> str:
    """'pnl' when the reward carries no inventory penalty, else 'phi=<n>'."""
    kind = cfg["REWARD_KIND"].strip().strip("'\"")
    phi = float(cfg["INVENTORY_PHI"])
    if kind == "pnl" or phi == 0:
        return "pnl"
    return f"phi={int(phi)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).parent / "results_with_seeds")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "parsed_results_with_seeds.txt")
    args = parser.parse_args()

    runs = list(find_runs(args.root))
    if not runs:
        raise SystemExit(f"No runs found under {args.root}")

    # Collect first so we can group by configuration and order the seeds within
    # each group. Sorting key: gas, risk profile, volatility, seed.
    records = []
    mismatches = []
    for gas_cost, seed, run_dir in runs:
        cfg = parse_config(run_dir / "config.txt")
        cfg_seed = int(float(cfg["SEED"]))
        if cfg_seed != seed:
            mismatches.append((run_dir, seed, cfg_seed))
        records.append({
            "gas_cost": gas_cost,
            "seed": seed,
            "cfg_seed": cfg_seed,
            "volatility": float(cfg["VOLATILITY"]),
            "risk": risk_label(cfg),
            "reward_kind": cfg["REWARD_KIND"],
            "inventory_phi": cfg["INVENTORY_PHI"],
            "run_dir": run_dir,
        })

    # Some SLURM tasks were killed mid-flight (wall-clock), so their results.txt
    # stops inside Phase 1 or right after the Phase 2 banner. Extract what is
    # there and record the rest as incomplete instead of aborting the whole file.
    for r in records:
        res_path = r["run_dir"] / "results.txt"
        try:
            r["phase2"] = extract_phase2(res_path)
        except ValueError:
            r["phase2"] = None
        try:
            r["phase4"] = extract_phase4(res_path)
        except ValueError:
            r["phase4"] = None
        r["n_models"] = len(list(r["run_dir"].glob("*_model.zip")))

    incomplete = [r for r in records if r["phase2"] is None or r["phase4"] is None]

    records.sort(key=lambda r: (r["gas_cost"], r["risk"], r["volatility"], r["seed"]))

    groups = defaultdict(list)
    for r in records:
        groups[(r["gas_cost"], r["risk"], r["volatility"])].append(r)

    box = "#" * 80
    with args.out.open("w") as fh:
        # ---------------- index ----------------
        fh.write(f"{box}\n")
        fh.write("# Parsed results — seeded sweep\n")
        fh.write(f"# Root: {args.root}\n")
        fh.write(f"# Runs: {len(records)} across {len(groups)} configuration(s)\n")
        fh.write("#\n")
        fh.write("# Grouped by (gas_cost, risk profile, volatility); the training\n")
        fh.write("# seeds for one configuration appear consecutively, so the spread\n")
        fh.write("# across seeds is the training variability for that cell.\n")
        fh.write("#\n")
        fh.write("# NOTE: each run was evaluated on EVAL_SEED = SEED + 999, so the\n")
        fh.write("# market paths differ between seeds as well as the trained policy.\n")
        fh.write("# Use reevaluate_saved_models.py for a fixed-path comparison.\n")
        fh.write(f"{box}\n\n")

        if mismatches:
            fh.write("!! SEED folder / config.txt disagreement:\n")
            for run_dir, folder_seed, cfg_seed in mismatches:
                fh.write(f"   {run_dir}: folder SEED={folder_seed} "
                         f"but config.txt SEED={cfg_seed}\n")
            fh.write("\n")

        if incomplete:
            fh.write("!! INCOMPLETE RUNS "
                     f"({len(incomplete)} of {len(records)}) — killed before finishing.\n")
            fh.write("   Whatever phases exist are still included below; missing\n")
            fh.write("   phases are marked in place. 'models' counts saved *_model.zip,\n")
            fh.write("   so runs with models but no Phase 2 can still be recovered by\n")
            fh.write("   re-running evaluation (see reevaluate_saved_models.py).\n\n")
            fh.write(f"   {'run':<34} {'gas':>4} {'seed':>5} {'vol':>6} "
                     f"{'phase2':>7} {'phase4':>7} {'models':>7}\n")
            fh.write("   " + "-" * 74 + "\n")
            for r in sorted(incomplete, key=lambda x: (x["gas_cost"], x["seed"])):
                fh.write(f"   {str(r['run_dir'].relative_to(args.root)):<34} "
                         f"{r['gas_cost']:>4g} {r['seed']:>5} {r['volatility']:>6g} "
                         f"{'yes' if r['phase2'] else 'MISSING':>7} "
                         f"{'yes' if r['phase4'] else 'MISSING':>7} "
                         f"{r['n_models']:>7}\n")
            fh.write("\n")

        fh.write("Index of configurations:\n")
        fh.write(f"  {'gas':>5} {'risk':<8} {'vol':>7} {'seeds':<28} {'n':>3} {'complete':>9}\n")
        fh.write("  " + "-" * 66 + "\n")
        for (gas, risk, vol), rs in sorted(groups.items()):
            seeds = ",".join(str(r["seed"]) for r in rs)
            n_ok = sum(1 for r in rs if r["phase2"] is not None)
            flag = "" if n_ok == len(rs) else "  <-- gaps"
            fh.write(f"  {gas:>5g} {risk:<8} {vol:>7g} {seeds:<28} {len(rs):>3} "
                     f"{n_ok:>9}{flag}\n")
        fh.write("\n\n")

        # ---------------- per-run blocks ----------------
        for (gas, risk, vol), rs in sorted(groups.items()):
            fh.write("=" * 80 + "\n")
            fh.write(f"CONFIGURATION   gas_cost={gas:g}   risk={risk}   "
                     f"volatility={vol:g}   ({len(rs)} seed(s))\n")
            fh.write("=" * 80 + "\n\n")

            for r in rs:
                run_dir = r["run_dir"]
                fh.write(f"{box}\n")
                fh.write(f"# Run: {run_dir.relative_to(args.root)}\n")
                fh.write(f"# gas_cost      = {r['gas_cost']:g}\n")
                fh.write(f"# SEED          = {r['seed']}\n")
                fh.write(f"# VOLATILITY    = {r['volatility']:g}\n")
                fh.write(f"# REWARD_KIND   = {r['reward_kind']}\n")
                fh.write(f"# INVENTORY_PHI = {r['inventory_phi']}\n")
                if r["phase2"] is None or r["phase4"] is None:
                    fh.write(f"# INCOMPLETE    = run killed before finishing "
                             f"({r['n_models']} model(s) saved)\n")
                fh.write(f"{box}\n\n")
                if r["phase2"] is not None:
                    fh.write(r["phase2"])
                else:
                    fh.write("*** Phase 2 MISSING — run did not reach evaluation ***\n")
                fh.write("\n")
                if r["phase4"] is not None:
                    fh.write(r["phase4"])
                else:
                    fh.write("*** Phase 4 MISSING — run did not reach plotting ***\n")
                fh.write("\n")

    n_full = sum(1 for r in records if r["phase2"] is not None and r["phase4"] is not None)
    print(f"Wrote {len(records)} runs ({len(groups)} configurations) to {args.out}")
    print(f"  complete (Phase 2 + Phase 4): {n_full}")
    print(f"  with Phase 2 table          : {sum(1 for r in records if r['phase2'])}")
    print(f"  with Phase 4 attribution    : {sum(1 for r in records if r['phase4'])}")
    if incomplete:
        print(f"  INCOMPLETE                  : {len(incomplete)} "
              f"(listed at the top of the output file)")
    if mismatches:
        print(f"WARNING: {len(mismatches)} run(s) whose SEED folder disagrees with "
              f"config.txt — see the top of the output file.")


if __name__ == "__main__":
    main()
