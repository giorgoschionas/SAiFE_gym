"""Parse all runs under results_newest/ into a single summary txt file.

For each leaf run directory it extracts:
  - gas_cost     (from parent folder name `GAS_COST=<n>`)
  - VOLATILITY, REWARD_KIND, INVENTORY_PHI (from config.txt)
  - the full Phase 2 block from results.txt

Usage:
    python parse_results.py [--root results_newest] [--out parsed_results.txt]
"""

import argparse
import re
from pathlib import Path


GAS_DIR_RE = re.compile(r"GAS_COST=(\d+)")
CONFIG_KEYS = ("VOLATILITY", "REWARD_KIND", "INVENTORY_PHI")
PHASE2_HEADER = "Phase 2: Evaluating enabled agents on 1000 trajectories"
PHASE4_HEADER = "Phase 4: Generating plots"
PHASE_BANNER_RE = re.compile(r"^=+\s*$")


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


def _extract_phase_block(lines: list[str], header: str) -> str:
    """Return the block for the given phase header (banner + body),
    stopping at the next `===\\nPhase N: ...\\n===` group or EOF."""
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i - 1 if i > 0 and PHASE_BANNER_RE.match(lines[i - 1]) else i
            break
    if start is None:
        raise ValueError(f"Phase header not found: {header!r}")

    end = len(lines)
    i = start + 3  # skip our own 3 header lines
    while i < len(lines):
        if (
            PHASE_BANNER_RE.match(lines[i])
            and i + 1 < len(lines)
            and lines[i + 1].lstrip().startswith("Phase ")
        ):
            end = i
            break
        i += 1

    return "\n".join(lines[start:end]).rstrip() + "\n"


def extract_phase2(results_path: Path) -> str:
    return _extract_phase_block(results_path.read_text().splitlines(), PHASE2_HEADER)


def extract_phase4(results_path: Path) -> str:
    """Phase 4 'Mean attribution per agent' table (same 1000 trajectories as Phase 2)."""
    return _extract_phase_block(results_path.read_text().splitlines(), PHASE4_HEADER)


def find_runs(root: Path):
    """Yield (gas_cost, run_dir) for every leaf run dir containing config + results."""
    for gas_dir in sorted(root.iterdir()):
        m = GAS_DIR_RE.match(gas_dir.name)
        if not m or not gas_dir.is_dir():
            continue
        gas_cost = int(m.group(1))
        for reward_dir in sorted(gas_dir.iterdir()):
            if not reward_dir.is_dir():
                continue
            for run_dir in sorted(reward_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                if (run_dir / "config.txt").exists() and (run_dir / "results.txt").exists():
                    yield gas_cost, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).parent / "results_newest")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "parsed_results.txt")
    args = parser.parse_args()

    runs = list(find_runs(args.root))
    if not runs:
        raise SystemExit(f"No runs found under {args.root}")

    with args.out.open("w") as fh:
        for idx, (gas_cost, run_dir) in enumerate(runs):
            cfg = parse_config(run_dir / "config.txt")
            phase2 = extract_phase2(run_dir / "results.txt")
            phase4 = extract_phase4(run_dir / "results.txt")

            header_box = "#" * 80
            fh.write(f"{header_box}\n")
            fh.write(f"# Run: {run_dir.relative_to(args.root)}\n")
            fh.write(f"# gas_cost     = {gas_cost}\n")
            fh.write(f"# VOLATILITY   = {cfg['VOLATILITY']}\n")
            fh.write(f"# REWARD_KIND  = {cfg['REWARD_KIND']}\n")
            fh.write(f"# INVENTORY_PHI = {cfg['INVENTORY_PHI']}\n")
            fh.write(f"{header_box}\n\n")
            fh.write(phase2)
            fh.write("\n")
            fh.write(phase4)
            fh.write("\n")

    print(f"Wrote {len(runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
