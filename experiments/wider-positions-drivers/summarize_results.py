"""Rebuild the comparison from saved sweeps without running simulations.

Run from any directory. Requires matplotlib only when generating plots.
Rebuild overview.csv for detailed width lookups and overview.{png,pdf} for the
qualitative comparison of deployed PPO ranges and periodic LP preferences.
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

    run = "risk-aversion/cara"
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
    fig, (depth_ax, vol_ax, risk_ax) = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")

    seed_rows = read_csv("volatility/ppo/summary/per_seed.csv")
    for index, seed in enumerate(sorted({row["seed"] for row in seed_rows})):
        series = sorted((row for row in seed_rows if row["seed"] == seed), key=lambda r: float(r["sigma"]))
        vol_ax.plot([float(r["sigma"]) for r in series],
                    [float(r["mean_deployed_width_ticks"]) for r in series],
                    ".--", color="#81B5A6", linewidth=1,
                    label="Individual PPO seed" if index == 0 else None)

    for ax, run, color, label in [
        (depth_ax, "background-liquidity/ppo", "#176491", "One PPO seed"),
        (vol_ax, "volatility/ppo", "#137457", "Mean across three seeds"),
    ]:
        policies = [r for r in rows if r["run"] == run]
        ax.plot([r["parameter_value"] for r in policies], [r["full_width_ticks"] for r in policies],
                "o-", color=color, linewidth=2.4, label=label)
        ax.set_ylabel("Mean deployed full width (ticks)")
        ax.legend(frameon=False, fontsize=9)
    depth_ax.set(title="Background liquidity", xlabel="Background liquidity per tick interval")
    depth_ax.set_xticks([10_000, 15_000, 20_000, 25_000, 30_000], ["10k", "15k", "20k", "25k", "30k"])
    depth_ax.set_yticks([75, 80, 85, 90, 95])
    vol_ax.set(title="Midprice volatility", xlabel="Volatility (σ) at background 30,000")
    vol_ax.set_xticks([0.03, 0.035, 0.04, 0.045, 0.05])

    # Compare the same two payoff distributions at every risk preference.
    utility_grid = read_csv("risk-aversion/cara/reward_grid.csv")
    ce = {(float(r["risk_aversion"]), int(r["half_width_ticks"])): float(r["certainty_equivalent_pnl"])
          for r in utility_grid}
    risks = sorted({a for a, _ in ce})
    risk_ax.axhline(0, color="#797979", linestyle="--", linewidth=1)
    risk_ax.plot(risks, [ce[a, 100] - ce[a, 50] for a in risks], "o-", color="#943566",
                 linewidth=2.4, label="Periodic LP, ±100 versus ±50 ticks")
    risk_ax.set(title="Risk aversion", xlabel="CARA risk aversion (1/token1)",
                ylabel="Gain in certainty-equivalent PnL (token1)")
    risk_ax.legend(frameon=False, fontsize=9)

    for ax in (depth_ax, vol_ax, risk_ax):
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("What draws an LP toward a wider range?", fontsize=15)
    fig.savefig(ROOT / "overview.png", dpi=180)
    fig.savefig(ROOT / "overview.pdf", metadata={"CreationDate": None})
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
