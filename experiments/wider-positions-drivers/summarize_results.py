"""Rebuild the comparison from saved sweeps without running simulations.

Run from any directory. Requires matplotlib only when generating plots.
The original artifacts are immutable; only overview.csv and overview.{png,pdf}
are regenerated. Widths in the overview use full spans in ticks.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_csv(relative_path):
    with (ROOT / relative_path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_snapshots():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    count = 0
    for run in manifest["runs"]:
        for artifact in run["files"]:
            path = ROOT / run["id"] / artifact["path"]
            content = path.read_bytes()
            if (len(content) != artifact["bytes"]
                    or hashlib.sha256(content).hexdigest() != artifact["sha256"]):
                raise ValueError(f"Snapshot differs from original: {path}")
            count += 1
    return count


def collect_rows():
    rows = []
    for driver, run, parameter in [
        ("background_liquidity", "background-liquidity/fixed-width", "background_liquidity"),
        ("volatility", "volatility/fixed-width-bg30000", "sigma"),
        ("volatility", "volatility/fixed-width-bg100000", "sigma"),
    ]:
        source = f"{run}/summary.csv"
        grid = read_csv(source)
        for value in sorted({float(row[parameter]) for row in grid}):
            group = [row for row in grid if float(row[parameter]) == value]
            # Break exact ties in favor of the smaller sampled width.
            best = max(group, key=lambda row: (float(row["mean_pnl"]), -int(row["half_width_ticks"])))
            rows.append({
                "driver": driver, "run": run, "parameter": parameter,
                "parameter_value": value, "width_statistic": "best_sampled_fixed_full_width",
                "full_width_ticks": float(best["full_width_ticks"]),
                "objective": "mean_net_pnl", "mean_pnl": float(best["mean_pnl"]),
                "certainty_equivalent_pnl": "", "bootstrap_best_fraction": "",
                "seed_std_full_width_ticks": "", "source_csv": source,
            })

    for driver, run, parameter, filename, pnl_key, std_key, objective in [
        ("background_liquidity", "background-liquidity/ppo", "background_liquidity",
         "width_by_background_liquidity.csv", "mean_pnl_per_path",
         "seed_std_deployed_width_ticks", "running_inventory_penalized_pnl"),
        ("volatility", "volatility/ppo", "sigma", "summary/width_by_volatility.csv",
         "mean_pnl", "seed_std_width_ticks", "mean_net_pnl"),
    ]:
        source = f"{run}/{filename}"
        for row in read_csv(source):
            rows.append({
                "driver": driver, "run": run, "parameter": parameter,
                "parameter_value": float(row[parameter]),
                "width_statistic": "ppo_mean_deployed_full_width",
                "full_width_ticks": float(row["mean_deployed_width_ticks"]),
                "objective": objective, "mean_pnl": float(row[pnl_key]),
                "certainty_equivalent_pnl": "", "bootstrap_best_fraction": "",
                "seed_std_full_width_ticks": row[std_key], "source_csv": source,
            })

    for run in ["risk-aversion/cara", "risk-aversion/legacy-cara"]:
        source = f"{run}/best_widths.csv"
        for row in read_csv(source):
            rows.append({
                "driver": "risk_aversion", "run": run, "parameter": "risk_aversion",
                "parameter_value": float(row["risk_aversion"]),
                "width_statistic": "best_sampled_fixed_full_width",
                "full_width_ticks": float(row["full_width_ticks"]),
                "objective": "expected_cara_utility", "mean_pnl": float(row["mean_pnl"]),
                "certainty_equivalent_pnl": float(row["certainty_equivalent_pnl"]),
                "bootstrap_best_fraction": float(row["bootstrap_best_fraction"]),
                "seed_std_full_width_ticks": "", "source_csv": source,
            })
    return sorted(rows, key=lambda row: (row["driver"], row["run"], row["parameter_value"]))


def plot_overview(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    panels = [
        ("background-liquidity/fixed-width", "Lower background liquidity", "Background liquidity per interval", "#0072B2"),
        ("volatility/fixed-width-bg30000", "Higher midprice volatility", "GBM volatility (sigma)", "#009E73"),
        ("risk-aversion/cara", "Higher CARA risk aversion", "Risk aversion (1 / token1)", "#AA4499"),
    ]
    for ax, (run, title, xlabel, color) in zip(axes, panels):
        selected = [row for row in rows if row["run"] == run]
        ax.plot([row["parameter_value"] for row in selected],
                [row["full_width_ticks"] for row in selected], "o-", color=color,
                label="Background 30,000" if run.startswith("volatility") else None)
        ax.set(title=title, xlabel=xlabel, ylabel="Best sampled full width (ticks)")
        ax.grid(alpha=0.2)
    axes[0].invert_xaxis()
    axes[0].ticklabel_format(axis="x", style="plain")
    control = [row for row in rows if row["run"] == "volatility/fixed-width-bg100000"]
    axes[1].plot([row["parameter_value"] for row in control],
                 [row["full_width_ticks"] for row in control], "s--", color="#666666",
                 label="Background 100,000")
    axes[1].legend(fontsize=8)
    axes[2].set_xscale("symlog", linthresh=0.003)
    axes[2].set_xticks([0, 0.003, 0.01, 0.03, 0.1], ["0", ".003", ".01", ".03", ".1"])
    fig.suptitle("Wider positions: evidence from completed fixed-width sweeps", fontsize=14)
    fig.text(0.5, 0.035,
             "Separate experiments: liquidity / volatility maximize mean net PnL; CARA maximizes expected utility.\n"
             "Full width = 2 × half-width. Sampled maxima are estimates; experiment settings differ (see README).",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    for suffix in ["png", "pdf"]:
        fig.savefig(ROOT / f"overview.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="Verify and rebuild CSV using only the standard library")
    args = parser.parse_args()
    count = verify_snapshots()
    rows = collect_rows()
    with (ROOT / "overview.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_plots:
        plot_overview(rows)
    print(f"Verified {count} source artifacts; wrote {len(rows)} comparison rows to {ROOT / 'overview.csv'}")


if __name__ == "__main__":
    main()
