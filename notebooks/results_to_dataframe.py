"""Load all Phase 2 results into a single tidy DataFrame.

One row per (gas_cost, volatility, reward_kind, inventory_phi, agent),
with PnL stats from the first Phase 2 table and width/rebalance stats
from the quoting-behavior table.

Usage as a library:
    from results_to_dataframe import load_results
    df = load_results()

Or as a script (writes csv next to the source):
    python results_to_dataframe.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent / "results_newest"
GAS_DIR_RE = re.compile(r"GAS_COST=(\d+)")

CONFIG_KEYS = {
    "VOLATILITY": ("volatility", float),
    "REWARD_KIND": ("reward_kind", lambda s: s.strip().strip("'\"")),
    "INVENTORY_PHI": ("inventory_phi", float),
}

AGENT_NAMES = {
    "DeployNarrow", "DeployWide", "ArrivalRebalance", "CDM", "PPO", "PPO_narrow",
}


def _parse_config(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key in CONFIG_KEYS:
            col, cast = CONFIG_KEYS[key]
            out[col] = cast(val.strip())
    return out


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.split("|")]


def _parse_pnl_table(lines: list[str], start: int) -> dict[str, dict]:
    """Parse the per-agent PnL stats table. Returns {agent: {mean, std, median, ...}}."""
    out = {}
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip().startswith(tuple(AGENT_NAMES)):
            if line.startswith("--") and out:
                break
            i += 1
            continue
        cells = _split_row(line)
        agent = cells[0]
        # Cells: [Agent, Mean, Std, Median, Skew, Kurt, Profitable, p(t) vs 0, p(W) vs 0]
        out[agent] = {
            "mean_pnl": float(cells[1]),
            "std_pnl": float(cells[2]),
            "median_pnl": float(cells[3]),
            "skew_pnl": float(cells[4]),
            "kurt_pnl": float(cells[5]),
            # cells[6] like "932/1000 (93%)"
            "profitable_frac": int(cells[6].split("/")[0]) / int(cells[6].split("/")[1].split()[0]),
        }
        i += 1
    return out


def _parse_quoting_table(lines: list[str], start: int) -> dict[str, dict]:
    """Parse the 'Quoting behavior' table."""
    out = {}
    i = start
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith(tuple(AGENT_NAMES)):
            if line.startswith("--") and out:
                break
            i += 1
            continue
        cells = _split_row(line)
        # [Agent, Mean spread, Std (within ep), Std (between traj),
        #  Mean center, Std (within ep), Std (between traj),
        #  Rebalances/ep, Rate (%)]
        out[cells[0]] = {
            "mean_spread": float(cells[1]),
            "spread_std_within": float(cells[2]),
            "spread_std_between": float(cells[3]),
            "mean_center": float(cells[4]),
            "center_std_within": float(cells[5]),
            "center_std_between": float(cells[6]),
            "rebalances_per_ep": float(cells[7]),
            "rebalance_rate_pct": float(cells[8]),
        }
        i += 1
    return out


def _parse_deploy_table(lines: list[str], start: int) -> dict[str, dict]:
    """Parse 'Deploy-time center asymmetry' table (sampled only on rebalance).

    Columns: Agent | Mean center | Std (within ep) | Std (between traj) | Deploys/ep
    These columns are bounded by tau because the center is recorded at deploy time
    as an offset from the current tick (unlike the every-step quoting table).
    """
    out = {}
    i = start
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith(tuple(AGENT_NAMES)):
            if line.startswith("--") and out:
                break
            i += 1
            continue
        cells = _split_row(line)
        out[cells[0]] = {
            "deploy_mean_center":         float(cells[1]),
            "deploy_center_std_within":   float(cells[2]),
            "deploy_center_std_between":  float(cells[3]),
            "deploys_per_ep":             float(cells[4]),
        }
        i += 1
    return out


def _parse_attribution_table(lines: list[str], start: int) -> dict[str, dict]:
    """Parse Phase 4 'Mean attribution per agent' table.

    Columns: Agent | PnL | HODL | Fees | IL | Gas
    PnL here matches Phase 2's Mean exactly (same 1000 trajectories), so we drop it.
    """
    out = {}
    i = start
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith(tuple(AGENT_NAMES)):
            if line.startswith("--") and out:
                break
            i += 1
            continue
        cells = _split_row(line)
        out[cells[0]] = {
            "attrib_hodl": float(cells[2]),
            "attrib_fees": float(cells[3]),
            "attrib_il":   float(cells[4]),
            "attrib_gas":  float(cells[5]),
        }
        i += 1
    return out


def _parse_results(path: Path) -> dict[str, dict]:
    lines = path.read_text().splitlines()
    pnl_stats: dict[str, dict] = {}
    quoting: dict[str, dict] = {}
    deploy: dict[str, dict] = {}
    attribution: dict[str, dict] = {}

    for i, line in enumerate(lines):
        ls = line.strip()
        if ls.startswith("Agent") and "Mean" in ls and "Profitable" in ls:
            pnl_stats = _parse_pnl_table(lines, i + 2)
        elif ls.startswith("Agent") and "Mean spread" in ls:
            quoting = _parse_quoting_table(lines, i + 2)
        elif ls.startswith("Agent") and "Mean center" in ls and "Deploys/ep" in ls:
            deploy = _parse_deploy_table(lines, i + 2)
        elif ls.startswith("Agent") and "HODL" in ls and "Fees" in ls:
            attribution = _parse_attribution_table(lines, i + 2)

    merged = {}
    all_agents = pnl_stats.keys() | quoting.keys() | deploy.keys() | attribution.keys()
    for agent in all_agents:
        merged[agent] = {
            **pnl_stats.get(agent, {}),
            **quoting.get(agent, {}),
            **deploy.get(agent, {}),
            **attribution.get(agent, {}),
        }
    return merged


def load_results(root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for gas_dir in sorted(root.iterdir()):
        m = GAS_DIR_RE.match(gas_dir.name)
        if not m or not gas_dir.is_dir():
            continue
        gas_cost = int(m.group(1))
        for reward_dir in sorted(gas_dir.iterdir()):
            if not reward_dir.is_dir():
                continue
            for run_dir in sorted(reward_dir.iterdir()):
                cfg_path = run_dir / "config.txt"
                res_path = run_dir / "results.txt"
                if not (cfg_path.exists() and res_path.exists()):
                    continue
                cfg = _parse_config(cfg_path)
                per_agent = _parse_results(res_path)
                for agent, metrics in per_agent.items():
                    rows.append({
                        "gas_cost": gas_cost,
                        **cfg,
                        "agent": agent,
                        **metrics,
                    })
    df = pd.DataFrame(rows)
    # Risk profile label: 'pnl' or 'phi=<n>'
    df["risk_profile"] = df.apply(
        lambda r: "pnl" if r["reward_kind"] == "pnl" else f"phi={int(r['inventory_phi'])}",
        axis=1,
    )
    return df


if __name__ == "__main__":
    df = load_results()
    out = Path(__file__).parent / "parsed_results.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows ({df['agent'].nunique()} agents × "
          f"{df.groupby(['gas_cost','volatility','risk_profile']).ngroups} configs) "
          f"to {out}")
    print("\nColumns:", list(df.columns))
    print("\nRisk profiles:", sorted(df['risk_profile'].unique()))
    print("Agents:", sorted(df['agent'].unique()))
