"""Triangular liquidity concentration plot.

Conceptual recreation of the ETH-style mid-market liquidity concentration
shape using our own LPs:

  - 4 PPO LPs (one per risk profile, widths pulled from the sweep dataframe)
  - 1 PPO_narrow LP (width pulled from the dataframe — typically 2 by
    construction since the narrow agent has half-width = 1)

The wealth of each LP is **analytically solved** from a vector of target
plateau heights — one per LP — going outwards from the centre:

    PLATEAU_TARGETS = [P_0, P_1, P_2, P_3, P_4]

      P_0 = central plateau height          (all 5 LPs cover ⇒ ticks |t| ≤ 1)
      P_1 = next plateau out                (narrow drops ⇒ 2 ≤ |t| ≤ 7-ish)
      P_2 = plateau after narrowest PPO     (8 ≤ |t| ≤ 10-ish)
      P_3 = plateau after second narrowest  (|t| ≈ 11)
      P_4 = plateau on the widest PPO only  (|t| ≈ 12)

Because each LP appears in a contiguous set of plateaus starting from the
centre, the per-tick contribution of LP i (sorted narrow→wide) equals
``P_i − P_{i+1}`` (with ``P_N = 0``), and its wealth is that drop times
the number of integer ticks it covers. The y-axis is then plotted in the
*display* units of those plateau targets — peak = ``P_0``, decay defined
by the rest — directly comparable to the ETH chart's "Normalized
liquidity (%)" axis.

Output: ``figures_sweep/10_triangular_liquidity.png``.

Scale note for the paper caption: with ``exponential_value = 1.0001`` one
tick is a 0.01 % price move, so the figure spans only ±0.13 % around mid
(vs the ±5 % of the ETH chart) — the *qualitative* triangular shape is
the same; the x-axis scale differs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(__file__))
from results_to_dataframe import load_results


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

FIGURES_DIR = Path(__file__).parent / "figures_sweep"

RISK_PROFILE_ORDER = ["pnl", "phi=20", "phi=50", "phi=80"]
RISK_PROFILE_LABELS = {
    "pnl":    "PPO",
    "phi=20": r"$\varphi = 20$",
    "phi=50": r"$\varphi = 50$",
    "phi=80": r"$\varphi = 80$",
}
# Colour palette: Okabe-Ito (colour-blind safe for deuteranopia / protanopia
# / tritanopia). PPO_narrow → bluish green (user request), PPO_pnl /
# Risk-Neutral → vermillion red (user request), the rest spread across the
# remaining Okabe-Ito hues so adjacent risk profiles stay distinguishable.
RISK_PROFILE_COLORS = {
    "pnl":    "#D55E00",  # vermillion (≈ red)
    "phi=20": "#E69F00",  # orange
    "phi=50": "#0072B2",  # blue
    "phi=80": "#CC79A7",  # reddish purple
}
NARROW_COLOR = "#009E73"  # bluish green
BG_COLOR     = "#BBBBBB"  # neutral gray
# Envelope (smoothed approximation line over the stacked profile).
# Approximated from the user-supplied swatch — soft cobalt blue.
ENVELOPE_COLOR = "#5A6FE0"
BG_WIDTH     = 30  # ticks → covers |t| ≤ 15 (31 integer ticks)

WIDE_AGENT   = "PPO"
NARROW_AGENT = "PPO_narrow"
TARGET_GAS   = 2
TARGET_VOL   = 0.020

# Target plateau heights in "display %" units, ordered from centre outwards.
# Position i corresponds to the i-th narrowest LP being the *innermost* one
# of those still contributing at that radius.
#
#   index 0 → centre               (narrow + 4 PPO + bg cover)  → 40 %
#   index 1 → just outside narrow  (4 PPO + bg cover)            → 35 %
#   index 2 → outside narrowest PPO (3 widest PPO + bg cover)    → 22 %
#   index 3 → outside next PPO      (2 widest PPO + bg cover)    → 18 %
#   index 4 → only widest PPO + bg  cover                        → 10 %
#   index 5 → only bg covers (|t| in 13..15)                     →  3 %
#
# Must be strictly decreasing (each LP must have non-negative wealth ⇒
# adjacent drops ≥ 0). The script raises if you violate it.
PLATEAU_TARGETS = [46.8, 37.0, 21.0, 16.0, 11.0, 6.0]

# Total wealth across all 5 LPs (the relative shape is invariant under
# uniform scaling — this knob only changes the printed absolute numbers).
TOTAL_WEALTH = 6000.0

# Y-axis upper limit in display units (peak sits at PLATEAU_TARGETS[0]).
Y_MAX_PCT = 80

# X-axis half-window in ticks.
PLOT_HALF_RANGE = 20

# Smoothing kernel σ for the envelope line overlaid on the staircase.
# Units = integer ticks. Larger → softer / more Gaussian-looking outline.
SMOOTH_SIGMA = 1.5

TICK_PCT = 0.01  # 1 tick at exponential_value = 1.0001 → 0.01 % move


# --------------------------------------------------------------------------- #
# Width selection                                                             #
# --------------------------------------------------------------------------- #

def select_widths(
    df,
    agent: str,
    gas: int = TARGET_GAS,
    vol: float = TARGET_VOL,
) -> dict[str, float]:
    """Pull ``mean_spread`` for ``agent`` at each risk profile."""
    out: dict[str, float] = {}
    for profile in RISK_PROFILE_ORDER:
        sub = df[
            (df["agent"] == agent)
            & (df["risk_profile"] == profile)
            & (df["gas_cost"] == gas)
        ]
        if sub.empty:
            print(f"  warn: no rows for {agent}/{profile}/gas={gas}")
            continue
        idx = (sub["volatility"] - vol).abs().idxmin()
        out[profile] = float(sub.loc[idx, "mean_spread"])
    return out


def mean_narrow_width(df) -> float:
    widths = select_widths(df, NARROW_AGENT)
    if not widths:
        print("  warn: no PPO_narrow widths found — defaulting to 2.0")
        return 2.0
    return float(np.mean(list(widths.values())))


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def n_integer_ticks(width: float) -> int:
    """Number of integer ticks covered by an LP with the given continuous
    width centred on 0: ``{t : |t| ≤ width/2}``.

    Examples
    --------
        width = 2  → 3 ticks ({-1, 0, +1})
        width = 15 → 15 ticks ({-7, ..., +7})
        width = 20 → 21 ticks ({-10, ..., +10})
        width = 23 → 23 ticks ({-11, ..., +11})
    """
    return int(2 * np.floor(width / 2.0) + 1)


def solve_wealths_from_plateaus(
    widths_sorted_narrow_to_wide: list[float],
    plateau_targets: list[float],
) -> list[float]:
    """Given LP widths sorted narrow→wide and plateau heights from centre
    outwards, return the wealth each LP needs to realise those plateaus.

    Each LP ``i`` (sorted narrow→wide) covers plateaus ``0..i`` (it is
    inside its own and all narrower regions). The per-tick contribution
    of LP ``i`` therefore equals the drop ``P_i − P_{i+1}`` (with
    ``P_N = 0`` at the boundary), and its wealth is

        W_i = (P_i − P_{i+1}) · n_i,    n_i = # integer ticks covered.

    Requires the plateau sequence to be strictly non-increasing so that
    every wealth is non-negative.
    """
    n = len(widths_sorted_narrow_to_wide)
    if len(plateau_targets) != n:
        raise ValueError(
            f"need one plateau target per LP: got {len(plateau_targets)} "
            f"targets for {n} LPs"
        )

    extended = list(plateau_targets) + [0.0]
    drops = [extended[i] - extended[i + 1] for i in range(n)]
    if any(d < 0 for d in drops):
        bad = [(i, d) for i, d in enumerate(drops) if d < 0]
        raise ValueError(
            f"plateau targets must be non-increasing (with ≥0 to the edge): "
            f"got non-positive drops at indices {bad}"
        )

    wealths = []
    for drop, w in zip(drops, widths_sorted_narrow_to_wide):
        wealths.append(drop * n_integer_ticks(w))
    return wealths


def build_components(
    widths_by_name: dict[str, float],
    wealths_by_name: dict[str, float],
    half_range: int,
):
    """Per-LP per-integer-tick liquidity contribution.

    Returns ``(ticks, L_total, components)`` where ``components[name]``
    is the per-tick contribution of LP ``name``.
    """
    ticks = np.arange(-half_range, half_range + 1)
    comps: dict[str, np.ndarray] = {}
    for name, w in widths_by_name.items():
        n = n_integer_ticks(w)
        contribution = wealths_by_name[name] / n
        comps[name] = np.where(np.abs(ticks) <= w / 2.0, contribution, 0.0)
    L = np.sum(list(comps.values()), axis=0)
    return ticks, L, comps


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #

def plot_triangular(
    df,
    output_filename: str = "10_triangular_liquidity.png",
) -> plt.Figure:
    ppo_widths   = select_widths(df, WIDE_AGENT)
    narrow_width = mean_narrow_width(df)

    if not ppo_widths:
        raise RuntimeError(
            f"no PPO widths for gas={TARGET_GAS}, vol≈{TARGET_VOL}"
        )

    # Build the narrow→wide ordering used by the plateau solver.
    # PPO LPs sorted by width ascending, narrow LP first, background LP last.
    ppo_sorted = sorted(ppo_widths.items(), key=lambda kv: kv[1])
    ordered_names  = ["narrow"] + [profile for profile, _ in ppo_sorted] + ["bg"]
    ordered_widths = [narrow_width] + [w for _, w in ppo_sorted] + [BG_WIDTH]

    # Solve wealths in raw "drop × n_ticks" units. The resulting peak height
    # equals PLATEAU_TARGETS[0] by construction (display units, e.g. 40 %).
    raw_wealths = solve_wealths_from_plateaus(ordered_widths, PLATEAU_TARGETS)

    # Re-scale so total wealth equals the user's chosen TOTAL_WEALTH.
    raw_total = sum(raw_wealths)
    scale     = TOTAL_WEALTH / raw_total
    wealths   = [w * scale for w in raw_wealths]

    # `display_scale` rescales per-tick L back to the original plateau-target
    # units so the y-axis shows 40 / 30 / 23 / 12 / 8 directly.
    display_scale = 1.0 / scale

    widths_by_name = dict(zip(ordered_names, ordered_widths))
    wealths_by_name = dict(zip(ordered_names, wealths))

    half = max(PLOT_HALF_RANGE, int(np.ceil(max(ordered_widths) / 2.0)) + 2)
    ticks, L, comps = build_components(widths_by_name, wealths_by_name, half)
    L_display       = L * display_scale  # back to plateau-target units
    comps_display   = {k: v * display_scale for k, v in comps.items()}

    # ----- Diagnostics --------------------------------------------------- #
    print(f"\nWidths used (gas={TARGET_GAS}, vol≈{TARGET_VOL}):")
    for name, w in widths_by_name.items():
        n_t = n_integer_ticks(w)
        wealth = wealths_by_name[name]
        per_tick = wealth / n_t
        if name in RISK_PROFILE_LABELS:
            label = RISK_PROFILE_LABELS[name]
        elif name == "narrow":
            label = NARROW_AGENT
        elif name == "bg":
            label = f"Background (±{BG_WIDTH // 2})"
        else:
            label = name
        print(
            f"  {label:<22s}  w = {w:6.2f}  n_ticks = {n_t:3d}  "
            f"wealth = {wealth:8.2f}  per-tick = {per_tick:6.3f}"
        )

    print(
        f"\nTotals:"
        f"\n  Σ wealth        : {sum(wealths):.2f}  "
        f"(target {TOTAL_WEALTH:.0f})"
        f"\n  L(0) (display)  : {L_display[ticks == 0][0]:.2f}  "
        f"(target {PLATEAU_TARGETS[0]:.1f})"
        f"\n  Σ_t L(t) display: {L_display.sum():.2f}"
        f"\n  Realised plateau drops: "
        + ", ".join(f"{wealth/n_integer_ticks(w):.2f}"
                    for w, wealth in zip(ordered_widths, wealths))
        + " (per-tick contributions; should match the drop sequence)"
    )

    # ----- Plot ---------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(9, 5))

    # Stack widest at the bottom, narrow at the top so the spike sits on
    # top of the wide-LP shoulders.
    stack_order = list(reversed(ordered_names))

    series, colors, labels = [], [], []
    for key in stack_order:
        if key == "narrow":
            color  = NARROW_COLOR
            label  = f"{NARROW_AGENT} (w={narrow_width:.1f})"
        elif key == "bg":
            color  = BG_COLOR
            label  = f"Background (w={BG_WIDTH})"
        else:
            color  = RISK_PROFILE_COLORS[key]
            label  = f"{RISK_PROFILE_LABELS[key]} (w={widths_by_name[key]:.1f})"
        series.append(comps_display[key])
        colors.append(color)
        labels.append(label)

    ax.stackplot(
        ticks, *series,
        colors=colors, labels=labels,
        step="mid", edgecolor="white", linewidth=0.4, alpha=0.4,
    )
    # Smoothed envelope over the staircase — analogous to overlaying a
    # Gaussian curve on a histogram. Sampled on a denser grid so the curve
    # is visibly continuous even though the data lives on integer ticks.
    dense_ticks = np.linspace(ticks[0], ticks[-1], len(ticks) * 8)
    L_dense     = np.interp(dense_ticks, ticks, L_display)
    L_smooth    = gaussian_filter1d(L_dense, sigma=SMOOTH_SIGMA * 8,
                                    mode="constant", cval=0.0)
    ax.plot(dense_ticks, L_smooth,
            color=ENVELOPE_COLOR, linewidth=1.8, alpha=0.95)
    ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.6)

    ax.set_xlabel("Tick offset from current price", fontsize=16)
    ax.set_ylabel("Normalized liquidity (%)", fontsize=16)
    ax.set_xlim(-half, half)
    ax.set_ylim(0, Y_MAX_PCT)
    ax.set_yticks([0, 20, 40, 60, 80])

    # ETH-paper-style chrome: no box around the plot area; only
    # horizontal grid lines; tick marks suppressed but tick labels kept.
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    # Y-axis: hide tick stubs (labels read off the horizontal grid lines).
    ax.tick_params(axis="y", which="both", length=0,
                   colors="#444444", labelsize=12)
    # X-axis: short vertical stubs below the data area like the ETH figure.
    ax.tick_params(axis="x", which="major", length=6, width=1.0,
                   direction="out", color="#999999",
                   labelcolor="#444444", labelsize=12, pad=4)

    # Reverse the legend order so the entries read narrow→wide (centre
    # outwards), matching how the plateaus step down on the chart.
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lbls[::-1],
              loc="upper right", fontsize=15, frameon=False)

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / output_filename
    fig.savefig(path, dpi=200)
    print(f"\nSaved: {path}")
    return fig


if __name__ == "__main__":
    df = load_results()
    plot_triangular(df)
