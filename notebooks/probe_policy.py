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


def _line_panel(ax, x, y_matrix, color_values, color_label, ylabel, title):
    """Draw one line plot: one curve per row of `y_matrix`, coloured by `color_values`."""
    norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
    for i, cv in enumerate(color_values):
        ax.plot(x, y_matrix[i], "-", color=CMAP(norm(cv)), alpha=0.7, linewidth=1.4)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_label, fontsize=11)


def plot_center_halfwidth(
    mispricing: np.ndarray, inventory: np.ndarray,
    actions: dict[str, np.ndarray], out_path: Path, agent: str,
):
    """2×2 grid: rows = (center, half-width), cols = (x = mispricing, x = inventory)."""
    n_misp = mispricing.size
    n_inv = inventory.size
    # Reshape (N1*N2,) → (N1, N2). swept_order was [mispricing, inventory] in __main__.
    center = actions["center"].reshape(n_misp, n_inv)
    half_w = actions["half_width"].reshape(n_misp, n_inv)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(f"Policy probe — {agent}  (center & half-width vs state)",
                 fontsize=13, y=1.02)

    # Row 0: center as function of mispricing (cols) and inventory (cols)
    _line_panel(
        axes[0, 0], mispricing, center.T, inventory,
        color_label="LP token0 (inventory)", ylabel="Center (ticks)",
        title="Center vs mispricing",
    )
    axes[0, 0].set_xlabel("Mispricing  (S − Z)", fontsize=12)
    _line_panel(
        axes[0, 1], inventory, center, mispricing,
        color_label="Mispricing (S − Z)", ylabel="Center (ticks)",
        title="Center vs LP inventory",
    )
    axes[0, 1].set_xlabel("LP token0 amount", fontsize=12)

    # Row 1: half-width
    _line_panel(
        axes[1, 0], mispricing, half_w.T, inventory,
        color_label="LP token0 (inventory)", ylabel="Half-width (ticks)",
        title="Half-width vs mispricing",
    )
    axes[1, 0].set_xlabel("Mispricing  (S − Z)", fontsize=12)
    _line_panel(
        axes[1, 1], inventory, half_w, mispricing,
        color_label="Mispricing (S − Z)", ylabel="Half-width (ticks)",
        title="Half-width vs LP inventory",
    )
    axes[1, 1].set_xlabel("LP token0 amount", fontsize=12)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_center_heatmap(
    mispricing: np.ndarray, inventory: np.ndarray,
    actions: dict[str, np.ndarray], out_path: Path, agent: str, tau: int,
):
    """Standalone heatmap of the position center over (mispricing × inventory)."""
    center = actions["center"].reshape(mispricing.size, inventory.size)
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    im = ax.imshow(
        center.T, origin="lower", aspect="auto",
        extent=(mispricing.min(), mispricing.max(),
                inventory.min(), inventory.max()),
        cmap="RdBu_r", vmin=-tau, vmax=tau, interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Center (ticks)")
    ax.set_xlabel("Mispricing  (S − Z)", fontsize=12)
    ax.set_ylabel("LP token0 amount", fontsize=12)
    ax.set_title(f"Position center — {agent}", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_halfwidth_heatmap(
    mispricing: np.ndarray, inventory: np.ndarray,
    actions: dict[str, np.ndarray], out_path: Path, agent: str, tau: int,
):
    """Standalone heatmap of the position half-width over (mispricing × inventory)."""
    half_w = actions["half_width"].reshape(mispricing.size, inventory.size)
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    im = ax.imshow(
        half_w.T, origin="lower", aspect="auto",
        extent=(mispricing.min(), mispricing.max(),
                inventory.min(), inventory.max()),
        cmap="viridis", vmin=1, vmax=tau, interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Half-width (ticks)")
    ax.set_xlabel("Mispricing  (S − Z)", fontsize=12)
    ax.set_ylabel("LP token0 amount", fontsize=12)
    ax.set_title(f"Position half-width — {agent}", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_hold_heatmap(
    mispricing: np.ndarray, inventory: np.ndarray,
    actions: dict[str, np.ndarray], out_path: Path, agent: str,
):
    """Heatmap of the hold flag over (mispricing × inventory)."""
    hold = actions["hold"].reshape(mispricing.size, inventory.size)

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    im = ax.imshow(
        hold.T, origin="lower", aspect="auto",
        extent=(mispricing.min(), mispricing.max(),
                inventory.min(), inventory.max()),
        cmap=HOLD_CMAP, norm=HOLD_NORM, interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 1])
    cbar.ax.set_yticklabels(["rebalance", "hold"])
    ax.set_xlabel("Mispricing  (S − Z)", fontsize=12)
    ax.set_ylabel("LP token0 amount", fontsize=12)
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

    # --- Build probe grid ---
    mispricing = np.linspace(args.mispricing_range[0], args.mispricing_range[1],
                             args.n_mispricing)
    inventory  = np.linspace(args.inventory_range[0], args.inventory_range[1],
                             args.n_inventory)

    # CLI overrides take precedence over config / NEUTRAL_DEFAULTS so that the
    # produced images explicitly reflect the chosen LP_LOWER_OFFSET_KEY,
    # LP_UPPER_OFFSET_KEY and GAS_COST_KEY initialisations.
    probe_gas_cost = args.gas_cost if args.gas_cost is not None else gas_cost
    fixed = dict(NEUTRAL_DEFAULTS)
    fixed["gas_cost"] = probe_gas_cost
    fixed["lp_lower_offset"] = args.lp_lower_offset
    fixed["lp_upper_offset"] = args.lp_upper_offset
    fixed["lp_token1_amount"] = initial_wealth / 2
    print(f"Probe initialisations: lp_lower_offset={args.lp_lower_offset}, "
          f"lp_upper_offset={args.lp_upper_offset}, gas_cost={probe_gas_cost}")

    obs = build_obs_grid(
        obs_keys=obs_keys,
        swept={"mispricing": mispricing, "lp_token0_amount": inventory},
        fixed=fixed,
    )
    print(f"Built probe grid: obs.shape = {obs.shape}")

    # --- Normalise + predict ---
    norm_obs = normalise(obs, vec_norm)
    raw_actions, _ = model.predict(norm_obs, deterministic=True)
    actions = decode_action(raw_actions, agent=args.agent, tau=tau)

    # --- Plot ---
    # Encode the initialisation values in the filename so successive runs with
    # different fixed-point states don't overwrite each other.
    init_tag = (
        f"__lo{args.lp_lower_offset:g}"
        f"__up{args.lp_upper_offset:g}"
        f"__gas{probe_gas_cost:g}"
    )
    out_lines     = args.run_dir / f"{args.agent}_probe_center_halfwidth{init_tag}.png"
    out_center    = args.run_dir / f"{args.agent}_probe_center_heatmap{init_tag}.png"
    out_halfwidth = args.run_dir / f"{args.agent}_probe_halfwidth_heatmap{init_tag}.png"
    out_hold      = args.run_dir / f"{args.agent}_probe_hold_heatmap{init_tag}.png"
    plot_center_halfwidth(mispricing, inventory, actions, out_lines, args.agent)
    plot_center_heatmap(mispricing, inventory, actions, out_center, args.agent, tau)
    plot_halfwidth_heatmap(mispricing, inventory, actions, out_halfwidth, args.agent, tau)
    plot_hold_heatmap(mispricing, inventory, actions, out_hold, args.agent)


if __name__ == "__main__":
    main()
