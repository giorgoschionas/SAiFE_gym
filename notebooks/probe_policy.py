"""Probe a trained PPO model on a synthetic state grid and plot the learned
policy as a function of (mispricing, lp_token0_amount).

Inputs:
    --run-dir   Directory containing `<agent>_model.zip` + `<agent>_vec_normalize.pkl`
                and `config.txt` (produced by agent_comparison.py).
    --agent     'ppo' (default) or 'ppo_narrow'.

Output figures (saved next to the model files):
    <agent>_probe_center_halfwidth.png   2×2 grid of (center, half-width) reactions
    <agent>_probe_hold_heatmap.png       hold-flag heatmap over (mispricing, inventory)

The probe reads the obs-key list from config.txt and reconstructs the exact
feature order the model was trained on. Non-swept features are held at
neutral defaults (mid-episode, just-deployed narrow position, half-wealth
inventory split).
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


# --------------------------------------------------------------------------- #
# Config parsing                                                              #
# --------------------------------------------------------------------------- #

CONFIG_RE = re.compile(r"^(\w+)\s*=\s*(.+?)\s*$")


def parse_config(config_path: Path) -> dict:
    """Pull the scalar fields we need from agent_comparison.py's config.txt."""
    out: dict = {}
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = CONFIG_RE.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        out[key] = raw
    return out


def _eval_obs_keys(raw: str) -> list[str]:
    """`SB3_OBS_KEYS = ['mispricing', ...]` → Python list."""
    # Safe because the values are plain string literals.
    return [s.strip().strip("'\"") for s in re.findall(r"'[^']+'|\"[^\"]+\"", raw)]


# --------------------------------------------------------------------------- #
# Synthetic-obs construction                                                  #
# --------------------------------------------------------------------------- #

NEUTRAL_DEFAULTS = {
    # Filled per-call from config; placeholders so the dict is complete.
    "mispricing":         0.0,
    "lp_lower_offset":    1.0,   # just-deployed narrow position (half_width=1)
    "lp_upper_offset":    1.0,
    "time":               0.5,   # mid-episode
    "gas_cost":           None,  # taken from config
    "lp_token0_amount":   0.5,   # half of initial holding (assuming ~50/50)
    "lp_token1_amount":   500.0, # INITIAL_WEALTH / 2 (set from config)
    # Features we don't usually train on but might appear in older configs:
    "midprice":           1000.0,
    "sqrt_price":         1.0e6,  # squared in the SB3 wrapper → ~price
    "lp_collected_fees_0": 0.0,
    "lp_collected_fees_1": 0.0,
}


def build_obs_grid(
    obs_keys: list[str],
    swept: dict[str, np.ndarray],
    fixed: dict[str, float],
) -> np.ndarray:
    """Build an (N1*N2, len(obs_keys)) obs array, sweeping `swept` features
    via meshgrid and holding the rest at `fixed` values."""
    swept_names = list(swept.keys())
    swept_arrays = list(swept.values())
    mesh = np.meshgrid(*swept_arrays, indexing="ij")
    n_total = mesh[0].size

    cols = []
    for key in obs_keys:
        if key in swept:
            cols.append(mesh[swept_names.index(key)].reshape(-1))
        elif key in fixed:
            cols.append(np.full(n_total, fixed[key], dtype=np.float32))
        else:
            raise KeyError(f"No value or default for obs key {key!r}")
    return np.column_stack(cols).astype(np.float32)


# --------------------------------------------------------------------------- #
# Action decoding (matches StructuredMultiDiscreteVecEnv / Narrow…)           #
# --------------------------------------------------------------------------- #

def decode_action(raw_action: np.ndarray, agent: str, tau: int) -> dict:
    """Return {'center', 'half_width', 'hold'} from the model's MultiDiscrete output.

    PPO (StructuredMultiDiscreteVecEnv):  [center_idx, half_width_idx, hold_idx]
        center     = center_idx - tau              ∈ {-tau, …, +tau}
        half_width = half_width_idx + 1            ∈ {1, …, tau}
        hold       = -1 if hold_idx == 0 else +1

    PPO_narrow (NarrowMultiDiscreteVecEnv):  [center_idx, hold_idx]
        center     = center_idx - tau
        half_width = 1  (fixed)
        hold       = -1 / +1
    """
    a = np.asarray(raw_action, dtype=np.int64)
    center = a[..., 0] - tau
    if agent == "ppo":
        half_width = a[..., 1] + 1
        hold_idx = a[..., 2]
    elif agent == "ppo_narrow":
        half_width = np.ones_like(center)
        hold_idx = a[..., 1]
    else:
        raise ValueError(f"agent must be 'ppo' or 'ppo_narrow', got {agent!r}")
    hold = np.where(hold_idx == 0, -1.0, 1.0)
    return {"center": center, "half_width": half_width, "hold": hold}


# --------------------------------------------------------------------------- #
# Normalisation                                                               #
# --------------------------------------------------------------------------- #

def normalise(obs: np.ndarray, vec_norm: VecNormalize) -> np.ndarray:
    """Apply VecNormalize's obs normalisation manually (no need to wrap an env)."""
    rms = vec_norm.obs_rms
    eps = vec_norm.epsilon
    clip = vec_norm.clip_obs
    return np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #

CMAP = plt.get_cmap("viridis")

# Discrete 2-colour map for the hold flag (binary: ±1). Order matches
# the BoundaryNorm edges below: index 0 = rebalance, index 1 = hold.
HOLD_CMAP = ListedColormap(["#ff7f0e", "#1f77b4"])  # orange = rebalance, blue = hold
HOLD_NORM = BoundaryNorm(boundaries=[-1.5, 0.0, 1.5], ncolors=HOLD_CMAP.N)


def _overlay_hold_hatch(
    ax, swept_a: np.ndarray, swept_b: np.ndarray,
    hold_flat: np.ndarray,
) -> None:
    """Overlay diagonal hatching where hold == +1.

    Marks regions where the policy's hold-head suppresses the rebalance, so
    the underlying centre / half-width values shown by the heatmap are NOT
    executed by the environment that step ("ghost actions"). Hatching is
    transparent so the underlying colour stays visible for diagnostic
    inspection.
    """
    hold_grid = (hold_flat.reshape(swept_a.size, swept_b.size) == 1).T
    if not np.any(hold_grid):
        return
    ax.contourf(
        swept_a, swept_b, hold_grid.astype(float),
        levels=[0.5, 1.5], colors="none", hatches=["///"],
    )


def _line_panel(ax, x, y_matrix, color_values, color_label, ylabel, title,
                hold_matrix=None):
    """Draw one line plot: one curve per row of `y_matrix`, coloured by `color_values`.

    If `hold_matrix` is provided (same shape as `y_matrix`, values ±1), segments
    where hold == +1 are drawn dashed + faded to mark "ghost" actions the env
    does not execute. Segments where hold == -1 (rebalance) are drawn solid.
    """
    norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
    for i, cv in enumerate(color_values):
        color = CMAP(norm(cv))
        y = y_matrix[i]
        if hold_matrix is None:
            ax.plot(x, y, "-", color=color, alpha=0.7, linewidth=1.4)
        else:
            held = hold_matrix[i] == 1
            y_exec = np.where(~held, y, np.nan)
            y_held = np.where( held, y, np.nan)
            ax.plot(x, y_exec, "-",  color=color, alpha=0.85, linewidth=1.4)
            ax.plot(x, y_held, "--", color=color, alpha=0.35, linewidth=1.0)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_label, fontsize=11)


def plot_center_halfwidth(
    swept_a: np.ndarray, swept_b: np.ndarray,
    actions: dict[str, np.ndarray], out_path: Path, agent: str,
    label_a: str = "Mispricing  (S − Z)",
    label_b: str = "LP token0 amount",
):
    """2×2 grid: rows = (center, half-width), cols = (x = swept_a, x = swept_b)."""
    n_a = swept_a.size
    n_b = swept_b.size
    # Reshape (N1*N2,) → (N1, N2). swept_order was [swept_a, swept_b] in __main__.
    center = actions["center"].reshape(n_a, n_b)
    half_w = actions["half_width"].reshape(n_a, n_b)
    hold   = actions["hold"].reshape(n_a, n_b)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(
        f"Policy probe — {agent}  (centre & half-width vs state; "
        f"solid = rebalance, dashed = hold)",
        fontsize=13, y=1.02,
    )

    # Row 0: center as function of swept_a (cols=0) and swept_b (cols=1)
    _line_panel(
        axes[0, 0], swept_a, center.T, swept_b,
        color_label=label_b, ylabel="Center (ticks)",
        title=f"Center vs {label_a}",
        hold_matrix=hold.T,
    )
    axes[0, 0].set_xlabel(label_a, fontsize=12)
    _line_panel(
        axes[0, 1], swept_b, center, swept_a,
        color_label=label_a, ylabel="Center (ticks)",
        title=f"Center vs {label_b}",
        hold_matrix=hold,
    )
    axes[0, 1].set_xlabel(label_b, fontsize=12)

    # Row 1: half-width
    _line_panel(
        axes[1, 0], swept_a, half_w.T, swept_b,
        color_label=label_b, ylabel="Half-width (ticks)",
        title=f"Half-width vs {label_a}",
        hold_matrix=hold.T,
    )
    axes[1, 0].set_xlabel(label_a, fontsize=12)
    _line_panel(
        axes[1, 1], swept_b, half_w, swept_a,
        color_label=label_a, ylabel="Half-width (ticks)",
        title=f"Half-width vs {label_b}",
        hold_matrix=hold,
    )
    axes[1, 1].set_xlabel(label_b, fontsize=12)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_center_heatmap(
    swept_a: np.ndarray, swept_b: np.ndarray,
    actions: dict[str, np.ndarray], out_path: Path, agent: str, tau: int,
    label_a: str = "Mispricing  (S − Z)",
    label_b: str = "LP token0 amount",
):
    """Standalone heatmap of the position center over (swept_a × swept_b)."""
    center = actions["center"].reshape(swept_a.size, swept_b.size)
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    im = ax.imshow(
        center.T, origin="lower", aspect="auto",
        extent=(swept_a.min(), swept_a.max(),
                swept_b.min(), swept_b.max()),
        cmap="RdBu_r", vmin=-tau, vmax=tau, interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Center (ticks)")
    _overlay_hold_hatch(ax, swept_a, swept_b, actions["hold"])
    ax.set_xlabel(label_a, fontsize=12)
    ax.set_ylabel(label_b, fontsize=12)
    ax.set_title(f"Position center — {agent}  (hatched = hold)", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_halfwidth_heatmap(
    swept_a: np.ndarray, swept_b: np.ndarray,
    actions: dict[str, np.ndarray], out_path: Path, agent: str, tau: int,
    label_a: str = "Mispricing  (S − Z)",
    label_b: str = "LP token0 amount",
):
    """Standalone heatmap of the position half-width over (swept_a × swept_b)."""
    half_w = actions["half_width"].reshape(swept_a.size, swept_b.size)
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    im = ax.imshow(
        half_w.T, origin="lower", aspect="auto",
        extent=(swept_a.min(), swept_a.max(),
                swept_b.min(), swept_b.max()),
        cmap="viridis", vmin=1, vmax=tau, interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Half-width (ticks)")
    _overlay_hold_hatch(ax, swept_a, swept_b, actions["hold"])
    ax.set_xlabel(label_a, fontsize=12)
    ax.set_ylabel(label_b, fontsize=12)
    ax.set_title(f"Position half-width — {agent}  (hatched = hold)", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_hold_heatmap(
    swept_a: np.ndarray, swept_b: np.ndarray,
    actions: dict[str, np.ndarray], out_path: Path, agent: str,
    label_a: str = "Mispricing  (S − Z)",
    label_b: str = "LP token0 amount",
):
    """Heatmap of the hold flag over (swept_a × swept_b)."""
    hold = actions["hold"].reshape(swept_a.size, swept_b.size)

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    im = ax.imshow(
        hold.T, origin="lower", aspect="auto",
        extent=(swept_a.min(), swept_a.max(),
                swept_b.min(), swept_b.max()),
        cmap=HOLD_CMAP, norm=HOLD_NORM, interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 1])
    cbar.ax.set_yticklabels(["rebalance", "hold"])
    ax.set_xlabel(label_a, fontsize=12)
    ax.set_ylabel(label_b, fontsize=12)
    ax.set_title(f"Hold-flag decision — {agent}", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Run directory containing the saved model + vec_normalize + config.txt")
    ap.add_argument("--agent", choices=["ppo", "ppo_narrow"], default="ppo")
    ap.add_argument("--n-mispricing", type=int, default=51)
    ap.add_argument("--n-inventory",  type=int, default=21)
    ap.add_argument("--mispricing-range", type=float, nargs=2, default=(-8.0, 8.0))
    ap.add_argument("--inventory-range",  type=float, nargs=2, default=(0.0, 1.0))
    ap.add_argument("--lp-lower-offset", type=float, default=1.0,
                    help="Value of LP_LOWER_OFFSET_KEY held fixed during the sweep.")
    ap.add_argument("--lp-upper-offset", type=float, default=1.0,
                    help="Value of LP_UPPER_OFFSET_KEY held fixed during the sweep.")
    ap.add_argument("--gas-cost", type=float, default=None,
                    help="Value of GAS_COST_KEY held fixed during the sweep. "
                         "Defaults to GAS_COST from config.txt.")

    # --- Second sweep: lp_lower_offset × lp_upper_offset ----------------- #
    ap.add_argument("--n-lower", type=int, default=21,
                    help="Number of lp_lower_offset values in the offset sweep.")
    ap.add_argument("--n-upper", type=int, default=21,
                    help="Number of lp_upper_offset values in the offset sweep.")
    ap.add_argument("--lower-range", type=float, nargs=2, default=(-10.0, 10.0),
                    help="(min, max) for lp_lower_offset in the offset sweep.")
    ap.add_argument("--upper-range", type=float, nargs=2, default=(-10.0, 10.0),
                    help="(min, max) for lp_upper_offset in the offset sweep.")
    ap.add_argument("--fixed-mispricing", type=float, default=0.0,
                    help="Value of MISPRICING held fixed during the offset sweep.")
    ap.add_argument("--fixed-inventory", type=float, default=0.5,
                    help="Value of LP_TOKEN0_AMOUNT held fixed during the offset sweep.")
    ap.add_argument("--skip-mispricing-inventory", action="store_true",
                    help="Skip the (mispricing × inventory) sweep.")
    ap.add_argument("--skip-offsets", action="store_true",
                    help="Skip the (lp_lower × lp_upper) sweep.")
    ap.add_argument("--skip-mispricing-upper", action="store_true",
                    help="Skip the (mispricing × lp_upper) sweep.")
    ap.add_argument("--skip-mispricing-lower", action="store_true",
                    help="Skip the (mispricing × lp_lower) sweep.")
    args = ap.parse_args()

    # --- Load config ---
    cfg = parse_config(args.run_dir / "config.txt")
    tau = int(cfg["TAU"])
    initial_wealth = float(cfg["INITIAL_WEALTH"])
    gas_cost = float(cfg.get("GAS_COST", 0.0))
    obs_keys = _eval_obs_keys(cfg["SB3_OBS_KEYS"])
    print(f"Loaded config: TAU={tau}, INITIAL_WEALTH={initial_wealth}, "
          f"GAS_COST={gas_cost}, obs_keys={obs_keys}")

    # --- Load model + VecNormalize ---
    model_path = args.run_dir / f"{args.agent}_model.zip"
    vec_path   = args.run_dir / f"{args.agent}_vec_normalize.pkl"
    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            f"Missing model artefacts in {args.run_dir}: expected "
            f"{model_path.name} and {vec_path.name}."
        )
    model = PPO.load(model_path, device="cpu")
    # `VecNormalize.load()` requires a real venv to attach to; we only need the
    # running stats (obs_rms, clip_obs, epsilon), so we unpickle directly to
    # avoid having to recreate the training env just for normalisation.
    with open(vec_path, "rb") as f:
        vec_norm = pickle.load(f)
    print(f"Loaded {args.agent} model + VecNormalize stats.")

    # CLI overrides take precedence over config / NEUTRAL_DEFAULTS so that the
    # produced images explicitly reflect the chosen LP_LOWER_OFFSET_KEY,
    # LP_UPPER_OFFSET_KEY and GAS_COST_KEY initialisations.
    probe_gas_cost = args.gas_cost if args.gas_cost is not None else gas_cost

    # ================================================================== #
    # Sweep 1 — Mispricing × LP_token0 inventory                         #
    # ================================================================== #
    if not args.skip_mispricing_inventory:
        mispricing = np.linspace(args.mispricing_range[0], args.mispricing_range[1],
                                 args.n_mispricing)
        inventory  = np.linspace(args.inventory_range[0], args.inventory_range[1],
                                 args.n_inventory)

        fixed = dict(NEUTRAL_DEFAULTS)
        fixed["gas_cost"] = probe_gas_cost
        fixed["lp_lower_offset"] = args.lp_lower_offset
        fixed["lp_upper_offset"] = args.lp_upper_offset
        fixed["lp_token1_amount"] = initial_wealth / 2
        print(f"\n[Sweep 1: mispricing × inventory] fixed: "
              f"lp_lower_offset={args.lp_lower_offset}, "
              f"lp_upper_offset={args.lp_upper_offset}, "
              f"gas_cost={probe_gas_cost}")

        obs = build_obs_grid(
            obs_keys=obs_keys,
            swept={"mispricing": mispricing, "lp_token0_amount": inventory},
            fixed=fixed,
        )
        norm_obs = normalise(obs, vec_norm)
        raw_actions, _ = model.predict(norm_obs, deterministic=True)
        actions = decode_action(raw_actions, agent=args.agent, tau=tau)

        init_tag = (
            f"__lo{args.lp_lower_offset:g}"
            f"__up{args.lp_upper_offset:g}"
            f"__gas{probe_gas_cost:g}"
        )
        plot_center_halfwidth(
            mispricing, inventory, actions,
            args.run_dir / f"{args.agent}_probe_center_halfwidth{init_tag}.png",
            args.agent,
            label_a="Mispricing  (S − Z)", label_b="LP token0 amount",
        )
        plot_center_heatmap(
            mispricing, inventory, actions,
            args.run_dir / f"{args.agent}_probe_center_heatmap{init_tag}.png",
            args.agent, tau,
            label_a="Mispricing  (S − Z)", label_b="LP token0 amount",
        )
        plot_halfwidth_heatmap(
            mispricing, inventory, actions,
            args.run_dir / f"{args.agent}_probe_halfwidth_heatmap{init_tag}.png",
            args.agent, tau,
            label_a="Mispricing  (S − Z)", label_b="LP token0 amount",
        )
        plot_hold_heatmap(
            mispricing, inventory, actions,
            args.run_dir / f"{args.agent}_probe_hold_heatmap{init_tag}.png",
            args.agent,
            label_a="Mispricing  (S − Z)", label_b="LP token0 amount",
        )

    # ================================================================== #
    # Sweep 2 — lp_lower_offset × lp_upper_offset                        #
    # ================================================================== #
    if not args.skip_offsets:
        lower_offs = np.linspace(args.lower_range[0], args.lower_range[1],
                                 args.n_lower)
        upper_offs = np.linspace(args.upper_range[0], args.upper_range[1],
                                 args.n_upper)

        fixed_off = dict(NEUTRAL_DEFAULTS)
        fixed_off["gas_cost"] = probe_gas_cost
        fixed_off["mispricing"] = args.fixed_mispricing
        fixed_off["lp_token0_amount"] = args.fixed_inventory
        fixed_off["lp_token1_amount"] = initial_wealth / 2
        # lp_lower_offset / lp_upper_offset are now the swept dims, so the
        # defaults for them are irrelevant.
        print(f"\n[Sweep 2: lp_lower × lp_upper] fixed: "
              f"mispricing={args.fixed_mispricing}, "
              f"inventory={args.fixed_inventory}, "
              f"gas_cost={probe_gas_cost}")

        obs_off = build_obs_grid(
            obs_keys=obs_keys,
            swept={
                "lp_lower_offset": lower_offs,
                "lp_upper_offset": upper_offs,
            },
            fixed=fixed_off,
        )
        norm_obs_off = normalise(obs_off, vec_norm)
        raw_off, _ = model.predict(norm_obs_off, deterministic=True)
        actions_off = decode_action(raw_off, agent=args.agent, tau=tau)

        off_tag = (
            f"__misp{args.fixed_mispricing:g}"
            f"__inv{args.fixed_inventory:g}"
            f"__gas{probe_gas_cost:g}"
        )
        plot_center_halfwidth(
            lower_offs, upper_offs, actions_off,
            args.run_dir
            / f"{args.agent}_probe_offset_center_halfwidth{off_tag}.png",
            args.agent,
            label_a="LP lower offset", label_b="LP upper offset",
        )
        plot_center_heatmap(
            lower_offs, upper_offs, actions_off,
            args.run_dir
            / f"{args.agent}_probe_offset_center_heatmap{off_tag}.png",
            args.agent, tau,
            label_a="LP lower offset", label_b="LP upper offset",
        )
        plot_halfwidth_heatmap(
            lower_offs, upper_offs, actions_off,
            args.run_dir
            / f"{args.agent}_probe_offset_halfwidth_heatmap{off_tag}.png",
            args.agent, tau,
            label_a="LP lower offset", label_b="LP upper offset",
        )
        plot_hold_heatmap(
            lower_offs, upper_offs, actions_off,
            args.run_dir
            / f"{args.agent}_probe_offset_hold_heatmap{off_tag}.png",
            args.agent,
            label_a="LP lower offset", label_b="LP upper offset",
        )

    # ================================================================== #
    # Sweep 3 — mispricing × lp_upper_offset                             #
    # ================================================================== #
    if not args.skip_mispricing_upper:
        mispricing = np.linspace(args.mispricing_range[0], args.mispricing_range[1],
                                 args.n_mispricing)
        upper_offs = np.linspace(args.upper_range[0], args.upper_range[1],
                                 args.n_upper)

        fixed_mu = dict(NEUTRAL_DEFAULTS)
        fixed_mu["gas_cost"] = probe_gas_cost
        fixed_mu["lp_lower_offset"] = args.lp_lower_offset
        fixed_mu["lp_token0_amount"] = args.fixed_inventory
        fixed_mu["lp_token1_amount"] = initial_wealth / 2
        print(f"\n[Sweep 3: mispricing × lp_upper] fixed: "
              f"lp_lower_offset={args.lp_lower_offset}, "
              f"inventory={args.fixed_inventory}, "
              f"gas_cost={probe_gas_cost}")

        obs_mu = build_obs_grid(
            obs_keys=obs_keys,
            swept={"mispricing": mispricing, "lp_upper_offset": upper_offs},
            fixed=fixed_mu,
        )
        norm_mu = normalise(obs_mu, vec_norm)
        raw_mu, _ = model.predict(norm_mu, deterministic=True)
        actions_mu = decode_action(raw_mu, agent=args.agent, tau=tau)

        mu_tag = (
            f"__lo{args.lp_lower_offset:g}"
            f"__inv{args.fixed_inventory:g}"
            f"__gas{probe_gas_cost:g}"
        )
        plot_center_halfwidth(
            mispricing, upper_offs, actions_mu,
            args.run_dir
            / f"{args.agent}_probe_misp_upper_center_halfwidth{mu_tag}.png",
            args.agent,
            label_a="Mispricing  (S − Z)", label_b="LP upper offset",
        )
        plot_center_heatmap(
            mispricing, upper_offs, actions_mu,
            args.run_dir
            / f"{args.agent}_probe_misp_upper_center_heatmap{mu_tag}.png",
            args.agent, tau,
            label_a="Mispricing  (S − Z)", label_b="LP upper offset",
        )
        plot_halfwidth_heatmap(
            mispricing, upper_offs, actions_mu,
            args.run_dir
            / f"{args.agent}_probe_misp_upper_halfwidth_heatmap{mu_tag}.png",
            args.agent, tau,
            label_a="Mispricing  (S − Z)", label_b="LP upper offset",
        )
        plot_hold_heatmap(
            mispricing, upper_offs, actions_mu,
            args.run_dir
            / f"{args.agent}_probe_misp_upper_hold_heatmap{mu_tag}.png",
            args.agent,
            label_a="Mispricing  (S − Z)", label_b="LP upper offset",
        )

    # ================================================================== #
    # Sweep 4 — mispricing × lp_lower_offset                             #
    # ================================================================== #
    if not args.skip_mispricing_lower:
        mispricing = np.linspace(args.mispricing_range[0], args.mispricing_range[1],
                                 args.n_mispricing)
        lower_offs = np.linspace(args.lower_range[0], args.lower_range[1],
                                 args.n_lower)

        fixed_ml = dict(NEUTRAL_DEFAULTS)
        fixed_ml["gas_cost"] = probe_gas_cost
        fixed_ml["lp_upper_offset"] = args.lp_upper_offset
        fixed_ml["lp_token0_amount"] = args.fixed_inventory
        fixed_ml["lp_token1_amount"] = initial_wealth / 2
        print(f"\n[Sweep 4: mispricing × lp_lower] fixed: "
              f"lp_upper_offset={args.lp_upper_offset}, "
              f"inventory={args.fixed_inventory}, "
              f"gas_cost={probe_gas_cost}")

        obs_ml = build_obs_grid(
            obs_keys=obs_keys,
            swept={"mispricing": mispricing, "lp_lower_offset": lower_offs},
            fixed=fixed_ml,
        )
        norm_ml = normalise(obs_ml, vec_norm)
        raw_ml, _ = model.predict(norm_ml, deterministic=True)
        actions_ml = decode_action(raw_ml, agent=args.agent, tau=tau)

        ml_tag = (
            f"__up{args.lp_upper_offset:g}"
            f"__inv{args.fixed_inventory:g}"
            f"__gas{probe_gas_cost:g}"
        )
        plot_center_halfwidth(
            mispricing, lower_offs, actions_ml,
            args.run_dir
            / f"{args.agent}_probe_misp_lower_center_halfwidth{ml_tag}.png",
            args.agent,
            label_a="Mispricing  (S − Z)", label_b="LP lower offset",
        )
        plot_center_heatmap(
            mispricing, lower_offs, actions_ml,
            args.run_dir
            / f"{args.agent}_probe_misp_lower_center_heatmap{ml_tag}.png",
            args.agent, tau,
            label_a="Mispricing  (S − Z)", label_b="LP lower offset",
        )
        plot_halfwidth_heatmap(
            mispricing, lower_offs, actions_ml,
            args.run_dir
            / f"{args.agent}_probe_misp_lower_halfwidth_heatmap{ml_tag}.png",
            args.agent, tau,
            label_a="Mispricing  (S − Z)", label_b="LP lower offset",
        )
        plot_hold_heatmap(
            mispricing, lower_offs, actions_ml,
            args.run_dir
            / f"{args.agent}_probe_misp_lower_hold_heatmap{ml_tag}.png",
            args.agent,
            label_a="Mispricing  (S − Z)", label_b="LP lower offset",
        )


if __name__ == "__main__":
    main()
