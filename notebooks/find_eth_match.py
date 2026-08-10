"""Search the sweep dataframe for the (gas, volatility) configuration whose
4-PPO + 1-PPO_narrow liquidity profile best matches an Ethereum-style
triangular concentration around mid-market.

Method
------
For each (gas, volatility):
  1. Pull the four PPO mean spreads (one per risk profile) and the average
     PPO_narrow mean spread.
  2. Find the non-negative wealths for the 5 LPs that minimise the
     mean-squared error between the *shape* of the resulting per-tick
     liquidity profile and a target Ethereum-style triangle.
  3. Score the fit by that MSE.

We use shape matching (peak-normalised) instead of an absolute-percentage
match because the ETH chart's "Normalized liquidity (%)" y-axis uses a
normalisation (per-sample then median across time) that doesn't translate
directly to our single-snapshot bin-share definition. Matching shape and
then displaying our profile on the same "% of total at this bin" axis is
the apples-to-apples version we can actually compute.

Output: the top configurations printed to console and a side-by-side plot
of the best match vs the ETH target saved to ``figures_sweep/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(__file__))
from results_to_dataframe import load_results


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

FIGURES_DIR = Path(__file__).parent / "figures_sweep"

RISK_PROFILE_ORDER = ["pnl", "phi=20", "phi=50", "phi=80"]

WIDE_AGENT   = "PPO"
NARROW_AGENT = "PPO_narrow"

# Plot/search window in ticks. We map ETH's ±5 % → ±PLOT_HALF_TICK ticks
# (purely visual; our agents live on the 0.01 %-per-tick scale so the
# physical span is much smaller — see scale note in plot caption).
PLOT_HALF_TICK = 15

# ETH target triangle: peak at centre, floor at the edges.
ETH_PEAK_PCT = 40.0
ETH_EDGE_PCT = 5.0


def eth_target_profile(ticks: np.ndarray) -> np.ndarray:
    """Linear decay from ETH_PEAK_PCT at tick 0 to ETH_EDGE_PCT at ±PLOT_HALF_TICK,
    zero beyond. Returned values are in the ETH paper's display units (%).
    """
    target = ETH_PEAK_PCT - (ETH_PEAK_PCT - ETH_EDGE_PCT) * np.abs(ticks) / PLOT_HALF_TICK
    target = np.where(np.abs(ticks) <= PLOT_HALF_TICK, target, 0.0)
    target = np.maximum(target, ETH_EDGE_PCT * (np.abs(ticks) <= PLOT_HALF_TICK))
    return target


# --------------------------------------------------------------------------- #
# Per-config width extraction                                                 #
# --------------------------------------------------------------------------- #

def widths_for_config(df, gas, vol):
    """Return (ppo_widths, narrow_width) for the slice, or (None, None) if
    incomplete.
    """
    ppo_widths = []
    for profile in RISK_PROFILE_ORDER:
        sub = df[
            (df["agent"] == WIDE_AGENT)
            & (df["risk_profile"] == profile)
            & (df["gas_cost"] == gas)
            & (df["volatility"] == vol)
        ]
        if sub.empty:
            return None, None
        w = float(sub["mean_spread"].iloc[0])
        if not np.isfinite(w) or w <= 0:
            return None, None
        ppo_widths.append(w)

    sub_n = df[
        (df["agent"] == NARROW_AGENT)
        & (df["gas_cost"] == gas)
        & (df["volatility"] == vol)
    ]
    if sub_n.empty:
        narrow_width = 2.0  # narrow agent's fixed half-width = 1
    else:
        narrow_width = float(sub_n["mean_spread"].mean())
        if not np.isfinite(narrow_width) or narrow_width <= 0:
            narrow_width = 2.0

    return ppo_widths, narrow_width


# --------------------------------------------------------------------------- #
# Liquidity model + wealth optimisation                                       #
# --------------------------------------------------------------------------- #

def lp_indicator_matrix(widths, ticks):
    """``M[t, i] = (1/w_i)`` if integer tick ``ticks[t]`` is inside LP ``i``'s
    symmetric range ``[-w_i/2, +w_i/2]``, else 0.

    Then per-tick liquidity ``L(t) = M @ W`` for wealth vector ``W``.
    """
    M = np.zeros((len(ticks), len(widths)))
    for i, w in enumerate(widths):
        M[np.abs(ticks) <= w / 2.0, i] = 1.0 / w
    return M


def fit_wealths_to_target(widths, ticks, target_pct):
    """NNLS fit: find W >= 0 minimising ||M·W − target_pct||².

    We then rescale ``W`` so that ``Σ_t (M·W)[t]`` (≈ total wealth) matches
    the target's total mass, which makes the resulting "per-tick share of
    total" axis line up with the ETH plotting convention.
    """
    M = lp_indicator_matrix(widths, ticks)
    W, residual = nnls(M, target_pct)
    return W, residual


def normalise_profile_as_bin_share(L):
    """Return ``100 · L / Σ L`` — each bin's share of total liquidity in %."""
    total = L.sum()
    return 100.0 * L / total if total > 0 else np.zeros_like(L)


# --------------------------------------------------------------------------- #
# Search                                                                      #
# --------------------------------------------------------------------------- #

def search_all_configs(df):
    ticks  = np.arange(-PLOT_HALF_TICK, PLOT_HALF_TICK + 1)
    target = eth_target_profile(ticks)

    results = []
    gas_levels = sorted(df["gas_cost"].unique())
    for gas in gas_levels:
        vol_levels = sorted(
            df[df["gas_cost"] == gas]["volatility"].unique(),
        )
        for vol in vol_levels:
            ppo_widths, narrow_width = widths_for_config(df, gas, vol)
            if ppo_widths is None:
                continue

            all_widths = ppo_widths + [narrow_width]
            W, residual = fit_wealths_to_target(all_widths, ticks, target)
            if W.sum() == 0:
                continue

            M = lp_indicator_matrix(all_widths, ticks)
            L = M @ W
            bin_share = normalise_profile_as_bin_share(L)

            # Score against the target expressed on the same "bin-share of
            # total" axis (so the comparison is apples-to-apples).
            target_bin_share = 100.0 * target / target.sum()
            mse = float(np.mean((bin_share - target_bin_share) ** 2))

            results.append({
                "gas":         gas,
                "vol":         vol,
                "ppo_widths":  ppo_widths,
                "narrow_w":    narrow_width,
                "wealths":     W,
                "bin_share":   bin_share,
                "target_bs":   target_bin_share,
                "L":           L,
                "ticks":       ticks,
                "mse":         mse,
                "residual":    float(residual),
            })

    results.sort(key=lambda r: r["mse"])
    return results


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #

def print_top(results, n=10):
    print(f"\nTop {n} configurations by MSE to ETH-style target:\n")
    header = (
        f"  {'rank':>4} | {'gas':>4} | {'vol':>7} | {'MSE':>9} | "
        f"{'peak %':>6} | {'PPO widths':<28} | "
        f"{'narrow w':>9} | {'wealth split (PPO_pnl…PPO_φ80, narrow)'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, r in enumerate(results[:n], start=1):
        widths_s = " ".join(f"{w:5.1f}" for w in r["ppo_widths"])
        wealth_s = " ".join(f"{w:6.0f}" for w in r["wealths"])
        peak     = r["bin_share"].max()
        print(
            f"  {i:>4} | {r['gas']:>4} | {r['vol']:>7.4f} | {r['mse']:>9.4f} | "
            f"{peak:>6.2f} | {widths_s:<28} | {r['narrow_w']:>9.2f} | {wealth_s}"
        )


def plot_best_match(
    best,
    output_filename: str = "11_best_eth_match.png",
) -> plt.Figure:
    ticks      = best["ticks"]
    profile    = best["bin_share"]
    target_bs  = best["target_bs"]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(ticks, profile, step="mid",
                    alpha=0.30, color="#1f4e79", linewidth=0)
    ax.step(ticks, profile, where="mid",
            color="#1f4e79", linewidth=2.0,
            label=(
                f"Best fit  (gas={best['gas']}, vol={best['vol']:.4f}, "
                f"MSE={best['mse']:.3f})"
            ))
    ax.plot(ticks, target_bs,
            color="#d62728", linewidth=2.0, linestyle="--",
            label="ETH-style target")

    ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Tick offset from current price", fontsize=13)
    ax.set_ylabel("Normalized liquidity (% of total)", fontsize=13)
    ax.set_xlim(-PLOT_HALF_TICK, PLOT_HALF_TICK)
    ax.set_ylim(0, max(profile.max(), target_bs.max()) * 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / output_filename
    fig.savefig(path, dpi=200)
    print(f"\nSaved: {path}")
    return fig


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    df = load_results()
    results = search_all_configs(df)
    if not results:
        raise SystemExit("No complete (gas, vol) configurations found.")

    print_top(results, n=15)
    best = results[0]
    print(
        f"\n=== Best match ==="
        f"\n  gas      : {best['gas']}"
        f"\n  vol      : {best['vol']:.4f}"
        f"\n  MSE      : {best['mse']:.4f}"
        f"\n  PPO widths : "
        + ", ".join(f"{w:.2f}" for w in best['ppo_widths'])
        + f"\n  narrow w   : {best['narrow_w']:.2f}"
        + f"\n  optimal wealths (PPO_pnl, PPO_φ20, PPO_φ50, PPO_φ80, narrow):"
        + f"\n    " + ", ".join(f"{w:.1f}" for w in best['wealths'])
        + f"\n  total wealth : {best['wealths'].sum():.1f}"
        + f"\n  peak bin %   : {best['bin_share'].max():.2f}"
        + f"\n  edge bin %   : {best['bin_share'][0]:.2f}"
    )

    plot_best_match(best)
