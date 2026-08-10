"""Faceted line plots for the gas × volatility × risk-profile sweep.

Three figures (one per metric): rows = risk profile, cols = gas cost,
x = volatility, line = agent, shaded band = ±std (or skipped).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from results_to_dataframe import load_results


RISK_PROFILE_ORDER = ["pnl", "phi=20", "phi=50", "phi=80"]
# Display labels — rename freely; keys MUST stay the same as in the dataframe.
RISK_PROFILE_LABELS = {
    "pnl":    "Risk Neutral",
    "phi=20": "φ = 20",
    "phi=50": "φ = 50",
    "phi=80": "φ = 80",
}
AGENT_ORDER = ["PPO", "PPO_narrow"]
AGENT_COLORS = {
    "PPO":        "#fb4545",
    "PPO_narrow": "#62f848",
}
# Full agent roster (for the "all algorithms" overview plot only).
ALL_AGENTS_ORDER = [
    "DeployNarrow", "DeployWide", "ArrivalRebalance", "CDM",
    "PPO", "PPO_narrow",
]
ALL_AGENT_COLORS = {
    "DeployNarrow":     "#1f77b4",
    "DeployWide":       "#17becf",
    "ArrivalRebalance": "#ff7f0e",
    "CDM":              "#9467bd",
    "PPO":              "#fb4545",
    "PPO_narrow":       "#62f848",
}
CASH_MARKER_COLORS = {
    "PPO":        "#b22222",  # firebrick (a touch lighter than #8b0000)
    "PPO_narrow": "#1e6b1e",  # dark green
}
# y-position (axis fraction) for the cash markers, per agent
CASH_MARKER_Y = {
    "PPO_narrow": 0.07,  # green on the lower row
    "PPO":        0.12,  # red on the upper row
}
# Data-coord vertical offset around y=0 used when ``cash_marker_at_zero``
# is on, so the per-agent markers stack (one above 0, one below) instead of
# overlapping. Sign mirrors the axis-fraction ordering: PPO above, PPO_narrow
# below. Magnitude is small relative to typical PnL ranges (~tens) so the
# markers stay visually anchored at "≈0".
CASH_MARKER_Y_OFFSET = {
    "PPO":        +4.0,
    "PPO_narrow": -4.0,
}

FIGURES_DIR = Path(__file__).parent / "figures_sweep"


def _is_no_deploy(row) -> bool:
    """Trained agent never deployed during eval (cash policy)."""
    return (row["rebalances_per_ep"] == 0) and pd.isna(row["mean_spread"])


def _facet_grid(df, mean_col, std_col, ylabel, filename,
                logy=False, share_y=True,
                agents=None, agent_colors=None,
                cash_marker_at_zero=False):
    """Generic faceted line plot: rows=risk_profile, cols=gas_cost.

    Configs where the agent stayed in cash (no deploys, NaN spread) are marked
    with an `×` so the gap is annotated, not silent. By default the marker
    sits at the bottom of the panel (axis-fraction y). When
    ``cash_marker_at_zero=True`` it sits at data y=0 instead — useful for
    metrics like PnL where 0 is the meaningful "no deploy" reference and a
    shared negative y-range would otherwise stretch the marker far below 0.

    `agents` / `agent_colors` default to the PPO-only globals; pass the
    `ALL_AGENTS_*` constants to render every algorithm.
    """
    agents = agents if agents is not None else AGENT_ORDER
    agent_colors = agent_colors if agent_colors is not None else AGENT_COLORS

    gas_levels = sorted(df["gas_cost"].unique())
    nrows = len(RISK_PROFILE_ORDER)
    ncols = len(gas_levels)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows),
        sharex=True, sharey=share_y, squeeze=False,
        layout="constrained",
    )
    #fig.suptitle(title, fontsize=16, y=0.995)

    for r, profile in enumerate(RISK_PROFILE_ORDER):
        for c, gas in enumerate(gas_levels):
            ax = axes[r, c]
            sub = df[(df["risk_profile"] == profile) & (df["gas_cost"] == gas)]

            for agent_i, agent in enumerate(agents):
                a_df = sub[sub["agent"] == agent].sort_values("volatility")
                if a_df.empty:
                    continue
                color = agent_colors[agent]

                # Plot only configs where the agent actually deployed; NaNs
                # break the polyline naturally, but we drop them to keep the
                # connecting line continuous between deploy points.
                deployed = a_df[~a_df.apply(_is_no_deploy, axis=1)]
                if not deployed.empty:
                    x = deployed["volatility"].to_numpy()
                    y = deployed[mean_col].to_numpy()
                    ax.plot(x, y, marker="o", markersize=4, linewidth=1.4,
                            color=color, label=agent)
                    if std_col is not None:
                        s = deployed[std_col].to_numpy()
                        ax.fill_between(x, y - s, y + s, alpha=0.12,
                                        color=color, linewidth=0)

                # Annotate cash-policy configs with × — either anchored at
                # data y=0 (PnL plots) or at the panel bottom in axis-fraction
                # y (everything else, where 0 may not be the natural floor).
                cash = a_df[a_df.apply(_is_no_deploy, axis=1)]
                if not cash.empty:
                    if cash_marker_at_zero:
                        y_off = CASH_MARKER_Y_OFFSET.get(agent, 0.0)
                        ax.scatter(
                            cash["volatility"].to_numpy(),
                            np.full(len(cash), y_off),
                            marker="x", s=40, linewidths=1.8,
                            color=CASH_MARKER_COLORS[agent],
                            zorder=5,
                            label=f"{agent} (no deploy)",
                        )
                    else:
                        ax.scatter(
                            cash["volatility"].to_numpy(),
                            np.full(len(cash), CASH_MARKER_Y[agent]),
                            marker="x", s=40, linewidths=1.8,
                            color=CASH_MARKER_COLORS[agent],
                            transform=ax.get_xaxis_transform(),
                            clip_on=False, zorder=5,
                            label=f"{agent} (no deploy)",
                        )

            if logy:
                ax.set_yscale("symlog" if (df[mean_col] < 0).any() else "log")
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(f"Gas Cost = {gas}", fontsize=20)
            if c == 0:
                ax.set_ylabel(f"{RISK_PROFILE_LABELS[profile]}\n\n{ylabel}", fontsize=20)
            if r == nrows - 1:
                ax.set_xlabel("Volatility", fontsize=20)

    # Single legend at the bottom, deduplicated across panels
    seen = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="outside lower center",
               ncol=min(len(seen), 4), frameon=False, fontsize=20)

    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    return fig


def plot_mean_pnl_all_agents(df_full):
    """Same as plot_mean_pnl but with every baseline + RL agent overlaid."""
    return _facet_grid(
        df_full, mean_col="mean_pnl", std_col="std_pnl",
        ylabel="Mean PnL",
        filename="08_mean_pnl_all.png",
        share_y=True,
        agents=ALL_AGENTS_ORDER,
        agent_colors=ALL_AGENT_COLORS,
        cash_marker_at_zero=True,
    )


# Shared layout for the comparable PnL / attribution figures so panels and
# overall canvas size match exactly across plots. Values are in figure
# fraction; the left margin is wide enough to host either "Mean PnL" or the
# math ``$\Delta PnL_{...}$`` label without reflowing the panels.
_PNL_FIGSIZE_PER_COL = 4.2  # inches per column
_PNL_FIGSIZE_PER_ROW = 2.8  # inches per row
_PNL_SUBPLOT_ADJUST = dict(
    left=0.12, right=0.98,
    bottom=0.10, top=0.95,
    wspace=0.10, hspace=0.18,
)


def _cornish_fisher_quantile(mu, sigma, skew, kurt_excess, q):
    """Cornish-Fisher approximation of the q-quantile from the first four moments.

    Inputs are array-like; ``kurt_excess`` is Fisher's excess kurtosis (= 0
    for the normal distribution), matching ``scipy.stats.kurtosis`` defaults.

    The expansion is:
        z_CF = z
             + (z² − 1)·S/6
             + (z³ − 3z)·K/24
             − (2z³ − 5z)·S²/36
        VaR_q ≈ μ + σ · z_CF

    Negative skew thickens the *lower* tail, so VaR_low − μ grows in magnitude
    while VaR_high − μ shrinks → asymmetric band, wider on the left.
    """
    z = norm.ppf(q)
    z_cf = (
        z
        + (z**2 - 1) * skew / 6.0
        + (z**3 - 3 * z) * kurt_excess / 24.0
        - (2 * z**3 - 5 * z) * skew**2 / 36.0
    )
    return mu + sigma * z_cf


def plot_mean_pnl_cf(df, q_lower: float = 0.05, q_upper: float = 0.95,
                     filename: str = "09_mean_pnl_cf.png"):
    """Mean PnL with an asymmetric Cornish-Fisher band derived from the
    four moments (mean, std, skew, kurt) we already store per config.

    Same facet layout as ``plot_mean_pnl`` (rows = risk profile, cols = gas).
    The shaded ribbon spans p_lower..p_upper estimated by the Cornish-Fisher
    expansion. Under negative skew / fat lower tails (typical for the
    risk-neutral PnL row), the band visibly extends further below the mean
    than above it; risk-averse profiles (φ > 0) should compress that
    asymmetry as the left tail thins out.

    Caveat: CF is a moment-based approximation — accurate for moderate
    skew/kurt, less so for very heavy tails. Switch to true per-trajectory
    quantiles (option 2) when the bands look implausible.
    """
    agents = AGENT_ORDER
    agent_colors = AGENT_COLORS

    gas_levels = sorted(df["gas_cost"].unique())
    nrows = len(RISK_PROFILE_ORDER)
    ncols = len(gas_levels)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(_PNL_FIGSIZE_PER_COL * ncols, _PNL_FIGSIZE_PER_ROW * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.subplots_adjust(**_PNL_SUBPLOT_ADJUST)

    for r, profile in enumerate(RISK_PROFILE_ORDER):
        for c, gas in enumerate(gas_levels):
            ax = axes[r, c]
            sub = df[(df["risk_profile"] == profile) & (df["gas_cost"] == gas)]

            for agent in agents:
                a_df = sub[sub["agent"] == agent].sort_values("volatility")
                if a_df.empty:
                    continue
                color = agent_colors[agent]

                deployed = a_df[~a_df.apply(_is_no_deploy, axis=1)]
                if not deployed.empty:
                    x = deployed["volatility"].to_numpy()
                    mu = deployed["mean_pnl"].to_numpy()
                    sd = deployed["std_pnl"].to_numpy()
                    sk = deployed["skew_pnl"].to_numpy()
                    ku = deployed["kurt_pnl"].to_numpy()

                    lower = _cornish_fisher_quantile(mu, sd, sk, ku, q_lower)
                    upper = _cornish_fisher_quantile(mu, sd, sk, ku, q_upper)

                    ax.plot(x, mu, marker="o", markersize=4, linewidth=1.4,
                            color=color, label=agent)
                    ax.fill_between(x, lower, upper, alpha=0.18,
                                    color=color, linewidth=0)

                # Cash-policy × markers at y=0 with the same per-agent
                # stagger as plot_mean_pnl, so the two plots read the same.
                cash = a_df[a_df.apply(_is_no_deploy, axis=1)]
                if not cash.empty:
                    y_off = CASH_MARKER_Y_OFFSET.get(agent, 0.0)
                    ax.scatter(
                        cash["volatility"].to_numpy(),
                        np.full(len(cash), y_off),
                        marker="x", s=40, linewidths=1.8,
                        color=CASH_MARKER_COLORS[agent],
                        zorder=5,
                        label=f"{agent} (no deploy)",
                    )

            ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(f"Gas Cost = {gas}", fontsize=20)
            if c == 0:
                ax.set_ylabel(
                    f"{RISK_PROFILE_LABELS[profile]}\n\nMean PnL",
                    fontsize=20,
                )
            if r == nrows - 1:
                ax.set_xlabel("Volatility", fontsize=20)

    seen = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(),
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(len(seen), 4), frameon=False, fontsize=20)

    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    return fig


def plot_mean_width(df):
    return _facet_grid(
        df, mean_col="mean_spread", std_col="spread_std_between",
        ylabel="Mean Spread",
        filename="02_mean_width.png",
        share_y=True,
    )


def plot_rebalance_freq(df):
    return _facet_grid(
        df, mean_col="rebalance_rate_pct", std_col=None,
        ylabel="Rebalance Rate",
        filename="03_rebalances.png",
        share_y=True,
        logy=False,
    )


def plot_mean_center(df):
    return _facet_grid(
        df, mean_col="deploy_mean_center", std_col="deploy_center_std_between",
        ylabel="Deploy center (ticks vs current price)",
        filename="07_mean_center.png",
        share_y=True,
    )


ATTRIBUTION_COMPONENT_COLORS = {
    "Fees": "#1f9d55",   # green
    "IL":   "#7a3f9a",   # purple (subtraction)
    "Gas":  "#f0a020",   # orange (subtraction)
}


def plot_pnl_attribution(df, agent: str, filename: str):
    """Stacked-bar PnL attribution for one agent.

    PnL ≈ Fees − IL − Gas. (The exact identity in agent_comparison.py is
    PnL = HODL + Fees − IL − Gas, but HODL averages to ~0 under zero-drift
    Brownian motion, so we omit it for clarity. The diamond shows the actual
    Net PnL, which equals the sum of the three bars up to that small residual.)
    """
    sub_all = df[df["agent"] == agent]
    gas_levels = sorted(df["gas_cost"].unique())
    nrows = len(RISK_PROFILE_ORDER)
    ncols = len(gas_levels)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows),
        sharex=True, sharey=True, squeeze=False,
        layout="constrained",
    )
    #fig.suptitle(
    #    f"PnL attribution — {agent}   (PnL ≈ Fees − IL − Gas, token1 units)",
    #    fontsize=16, y=0.995,
    #)

    for r, profile in enumerate(RISK_PROFILE_ORDER):
        for c, gas in enumerate(gas_levels):
            ax = axes[r, c]
            sub = (sub_all[(sub_all["risk_profile"] == profile) &
                           (sub_all["gas_cost"] == gas)]
                   .sort_values("volatility"))
            if sub.empty:
                continue

            cash_mask = sub.apply(_is_no_deploy, axis=1).to_numpy()
            deployed_mask = ~cash_mask

            x = sub["volatility"].to_numpy()
            bar_w = 0.003
            fees = sub["attrib_fees"].to_numpy()
            il_neg  = -sub["attrib_il"].to_numpy()   # plotted as negative
            gas_neg = -sub["attrib_gas"].to_numpy()
            pnl = sub["mean_pnl"].to_numpy()

            # Positive stack
            ax.bar(x, fees,    bar_w, color=ATTRIBUTION_COMPONENT_COLORS["Fees"], label="Fees")

            # Negative stack
            ax.bar(x, il_neg,  bar_w, color=ATTRIBUTION_COMPONENT_COLORS["IL"],  label="−IL")
            ax.bar(x, gas_neg, bar_w, bottom=il_neg,
                   color=ATTRIBUTION_COMPONENT_COLORS["Gas"], label="−Gas")

            # Net PnL marker (skip cash configs — those get the × instead)
            if deployed_mask.any():
                ax.scatter(x[deployed_mask], pnl[deployed_mask],
                           marker="D", s=28, color="black",
                           zorder=5, label="Net PnL")

            # No-deploy × replaces the diamond at y=0 (cash PnL is 0)
            if cash_mask.any():
                ax.scatter(
                    x[cash_mask], np.zeros(cash_mask.sum()),
                    marker="x", s=50, linewidths=2.0,
                    color=CASH_MARKER_COLORS[agent],
                    zorder=6,
                    label=f"{agent} (no deploy)",
                )

            ax.axhline(0, color="black", linewidth=0.6)
            ax.grid(True, alpha=0.3, axis="y")
            if r == 0:
                ax.set_title(f"gas_cost = {gas}", fontsize=20)
            if c == 0:
                ax.set_ylabel(f"{RISK_PROFILE_LABELS[profile]}\n\nPnL contribution", fontsize=20)
            if r == nrows - 1:
                ax.set_xlabel("Volatility" , fontsize=20)

    # Deduped legend
    seen = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="outside lower center",
               ncol=min(len(seen), 6), frameon=False, fontsize=20)

    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    return fig


def plot_attribution_diff(df, agent_a: str = "PPO_narrow", agent_b: str = "PPO",
                          filename: str = "06_attribution_diff.png"):
    """Stacked-bar plot of attribution differences (agent_a - agent_b).

    Positive bar = agent_a advantage on that component:
      Δ Fees     = a.Fees − b.Fees           (positive ⇒ a earns more fees)
      −Δ IL      = −(a.IL − b.IL) = b.IL − a.IL    (positive ⇒ a suffers less IL)
      −Δ Gas     = b.Gas − a.Gas             (positive ⇒ a spends less gas)
    The three bars sum to Δ PnL = a.PnL − b.PnL (diamond).
    """
    gas_levels = sorted(df["gas_cost"].unique())
    nrows = len(RISK_PROFILE_ORDER)
    ncols = len(gas_levels)

    # Use the SAME figsize and SAME subplot margins as plot_mean_pnl_cf
    # (the 09 plot). Constrained-layout is disabled in both, so the panel
    # positions in figure-fraction coords are now identical → matching
    # panel sizes on screen. The wider math y-label has to fit inside the
    # same left-margin allocation as 09's "Mean PnL"; if it clips, shorten
    # the label rather than perturb this layout.
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(_PNL_FIGSIZE_PER_COL * ncols, _PNL_FIGSIZE_PER_ROW * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.subplots_adjust(**_PNL_SUBPLOT_ADJUST)
    #fig.suptitle(
    #    f"PnL-attribution difference: {agent_a} − {agent_b}   "
    #    "(positive ⇒ " + agent_a + " advantage)",
    #    fontsize=16, y=0.995,
    #)

    for r, profile in enumerate(RISK_PROFILE_ORDER):
        for c, gas in enumerate(gas_levels):
            ax = axes[r, c]
            a = (df[(df["agent"] == agent_a) & (df["risk_profile"] == profile)
                    & (df["gas_cost"] == gas)].sort_values("volatility")
                 .reset_index(drop=True))
            b = (df[(df["agent"] == agent_b) & (df["risk_profile"] == profile)
                    & (df["gas_cost"] == gas)].sort_values("volatility")
                 .reset_index(drop=True))
            if a.empty or b.empty:
                continue

            # Volatility points where *both* agents actually deployed
            a_cash = a.apply(_is_no_deploy, axis=1).to_numpy()
            b_cash = b.apply(_is_no_deploy, axis=1).to_numpy()
            both_deployed = ~(a_cash | b_cash)
            any_cash = a_cash | b_cash

            x_all = a["volatility"].to_numpy()
            d_fees = (a["attrib_fees"] - b["attrib_fees"]).to_numpy()
            d_il_adv  = (b["attrib_il"]  - a["attrib_il"]).to_numpy()    # −Δ IL
            d_gas_adv = (b["attrib_gas"] - a["attrib_gas"]).to_numpy()   # −Δ Gas
            d_pnl  = (a["mean_pnl"] - b["mean_pnl"]).to_numpy()

            x = x_all[both_deployed]
            if x.size:
                bar_w = 0.003
                fees = d_fees[both_deployed]
                il   = d_il_adv[both_deployed]
                gas_ = d_gas_adv[both_deployed]

                # Each component split into pos/neg halves so they stack on
                # the correct side of zero.
                for comp, color, label in [
                    (fees, ATTRIBUTION_COMPONENT_COLORS["Fees"], "Δ Fees"),
                    (il,   ATTRIBUTION_COMPONENT_COLORS["IL"],   "−Δ IL"),
                    (gas_, ATTRIBUTION_COMPONENT_COLORS["Gas"],  "−Δ Gas"),
                ]:
                    # Compute running offsets so stacks add cumulatively
                    pass  # placeholder — actual stacking below

                # Build cumulative stacks separately for positive and negative parts.
                pos_offset = np.zeros_like(x)
                neg_offset = np.zeros_like(x)
                for comp, color, label in [
                    (fees, ATTRIBUTION_COMPONENT_COLORS["Fees"], "Δ Fees"),
                    (il,   ATTRIBUTION_COMPONENT_COLORS["IL"],   "−Δ IL"),
                    (gas_, ATTRIBUTION_COMPONENT_COLORS["Gas"],  "−Δ Gas"),
                ]:
                    pos = np.where(comp > 0, comp, 0.0)
                    neg = np.where(comp < 0, comp, 0.0)
                    ax.bar(x, pos, bar_w, bottom=pos_offset, color=color, label=label)
                    ax.bar(x, neg, bar_w, bottom=neg_offset, color=color)
                    pos_offset += pos
                    neg_offset += neg

                # Net Δ PnL diamond
                ax.scatter(x, d_pnl[both_deployed],
                           marker="D", s=28, color="black", zorder=5,
                           label="Δ Net PnL")

            # Cash-policy × markers omitted on purpose — when one agent
            # sits in cash the panel range stretches and the red (PPO) ×
            # ends up outside the plot area. The bar gap at that volatility
            # already signals the cash policy.

            ax.axhline(0, color="black", linewidth=0.6)
            ax.grid(True, alpha=0.3, axis="y")
            if r == 0:
                ax.set_title(f"Gas Cost = {gas}", fontsize=20)
            if c == 0:
                # Build a LaTeX label "ΔPnL_{<agent_a> − <agent_b>}".
                # Each agent name is rendered upright (\mathrm) with any
                # underscore-suffix promoted to a nested subscript
                # (e.g. "PPO_narrow" → "PPO_{narrow}") so the math reads
                # cleanly instead of italic-ising the agent names.
                def _math_name(n):
                    parts = n.split('_', 1)
                    if len(parts) == 1:
                        return rf"\mathrm{{{parts[0]}}}"
                    return rf"\mathrm{{{parts[0]}}}_{{\mathrm{{{parts[1]}}}}}"
                diff_label = (
                    rf"$\Delta \mathrm{{PnL}}$"
                )
                ax.set_ylabel(
                    f"{RISK_PROFILE_LABELS[profile]}\n\n{diff_label}",
                    fontsize=20,
                )
            if r == nrows - 1:
                ax.set_xlabel("Volatility", fontsize=20)

    seen = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(),
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(len(seen), 6), frameon=False, fontsize=20)

    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    df_full = load_results()
    df = df_full[df_full["agent"].isin(AGENT_ORDER)]
    plot_mean_width(df)
    plot_rebalance_freq(df)
    plot_pnl_attribution(df, "PPO",        "04_attribution_ppo.png")
    plot_pnl_attribution(df, "PPO_narrow", "05_attribution_ppo_narrow.png")
    plot_attribution_diff(df)
    plot_mean_center(df)
    plot_mean_pnl_all_agents(df_full)
    plot_mean_pnl_cf(df)
