"""Mispricing-response figure for the trained PPO / PPO_narrow policy.

Builds a 3-panel line plot showing how the policy's (center, half_width,
hold) action components react to mispricing under several physically-
consistent LP-state slices. This is a synthetic-state probe (like
probe_policy.py) but specialised to the mispricing-isolation question:
instead of a 2-D heatmap that conflates state with mispricing, it sweeps
mispricing on a 1-D grid and overlays one line per fixed conditioning
state so the reader can judge how robust the response is.

Conditioning states are hard-coded near the top of the file; edit
``STATES_COMMON`` / ``STATES_PPO_ONLY`` to add or remove slices.

Usage:
    python mispricing_response.py --run-dir <run-dir> --agent ppo
    python mispricing_response.py --run-dir <run-dir> --agent ppo_narrow
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


# ────────────────────────────────────────────────────────────────────────
# Conditioning states
# ────────────────────────────────────────────────────────────────────────
# Each dict defines one line in the figure. The (lower, upper, token0)
# triple is chosen so the cell is physically consistent: out-of-range
# states pin token0 to 0 (price above range → 100% token1) or to ~1
# (price below range → 100% token0); in-range states use a plausible
# midway value. ``lp_token1_amount`` is held at INITIAL_WEALTH/2 for all
# slices (see ``fixed`` in main).

STATES_COMMON = [
    dict(name="A: centred narrow",      lower=1,  upper=1,  token0=0.5, color="#1f77b4"),
    dict(name="B: drifted out top",     lower=3,  upper=-1, token0=0.0, color="#ff7f0e"),
    dict(name="C: drifted out bottom",  lower=-1, upper=3,  token0=1.0, color="#2ca02c"),
    dict(name="E: deep out top",        lower=5,  upper=-3, token0=0.0, color="#d62728"),
    dict(name="F: deep out bottom",     lower=-3, upper=5,  token0=1.0, color="#9467bd"),
]

# Extra states meaningful only for the variable-width agent. PPO_narrow's
# action space forces half_width = 1, so positions wider than ±1 are
# unreachable by that agent and probing them would be off-distribution
# in a way that says nothing about the agent's learned behaviour.
STATES_PPO_ONLY = [
    dict(name="G: slightly wider in-range", lower=2, upper=2, token0=0.5, color="#e377c2"),
    dict(name="D: wide in-range",           lower=5, upper=5, token0=0.5, color="#8c564b"),
]


# ────────────────────────────────────────────────────────────────────────
# Helpers (kept self-contained — duplicated from probe_policy.py)
# ────────────────────────────────────────────────────────────────────────

CONFIG_RE = re.compile(r"^(\w+)\s*=\s*(.+?)\s*$")


def parse_config(path: Path) -> dict:
    out: dict = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = CONFIG_RE.match(line)
        if not m:
            continue
        out[m.group(1)] = m.group(2)
    return out


def _eval_obs_keys(raw: str) -> list[str]:
    return [s.strip().strip("'\"") for s in re.findall(r"'[^']+'|\"[^\"]+\"", raw)]


def decode_action(raw_action: np.ndarray, agent: str, tau: int) -> dict:
    """Mirror of probe_policy.decode_action."""
    a = np.asarray(raw_action, dtype=np.int64)
    center = a[..., 0] - tau
    if agent == "ppo":
        half_width = a[..., 1] + 1
        hold_idx   = a[..., 2]
    elif agent == "ppo_narrow":
        half_width = np.ones_like(center)
        hold_idx   = a[..., 1]
    else:
        raise ValueError(f"agent must be 'ppo' or 'ppo_narrow', got {agent!r}")
    hold = np.where(hold_idx == 0, -1.0, 1.0)
    return {"center": center, "half_width": half_width, "hold": hold}


def normalise(obs: np.ndarray, vec_norm) -> np.ndarray:
    rms = vec_norm.obs_rms
    return np.clip(
        (obs - rms.mean) / np.sqrt(rms.var + vec_norm.epsilon),
        -vec_norm.clip_obs, vec_norm.clip_obs,
    ).astype(np.float32)


def build_obs_row(
    obs_keys: list[str],
    mispricing: np.ndarray,
    state: dict,
    fixed: dict,
) -> np.ndarray:
    """Build (N_mp, len(obs_keys)) obs varying mispricing only."""
    n = mispricing.size
    cols = []
    for key in obs_keys:
        if key == "mispricing":
            cols.append(mispricing)
        elif key == "lp_lower_offset":
            cols.append(np.full(n, state["lower"],  dtype=np.float32))
        elif key == "lp_upper_offset":
            cols.append(np.full(n, state["upper"],  dtype=np.float32))
        elif key == "lp_token0_amount":
            cols.append(np.full(n, state["token0"], dtype=np.float32))
        elif key in fixed:
            cols.append(np.full(n, fixed[key],      dtype=np.float32))
        else:
            raise KeyError(f"no value or default for obs key {key!r}")
    return np.column_stack(cols).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Directory with <agent>_model.zip, <agent>_vec_normalize.pkl, config.txt.")
    ap.add_argument("--agent", choices=["ppo", "ppo_narrow"], default="ppo")
    ap.add_argument("--n-mispricing", type=int, default=101)
    ap.add_argument("--mispricing-range", type=float, nargs=2, default=(-3.0, 3.0),
                    help="Limit to the policy's realistic range (use the sanity-check 1%%–99%%).")
    ap.add_argument("--time", type=float, default=0.5,
                    help="Value of TIME_KEY held fixed across the sweep.")
    ap.add_argument("--gas-cost", type=float, default=None,
                    help="Defaults to GAS_COST from config.txt.")
    args = ap.parse_args()

    # ── Config ──
    cfg = parse_config(args.run_dir / "config.txt")
    tau = int(cfg["TAU"])
    initial_wealth = float(cfg["INITIAL_WEALTH"])
    cfg_gas = float(cfg.get("GAS_COST", 0.0))
    gas = args.gas_cost if args.gas_cost is not None else cfg_gas
    obs_keys = _eval_obs_keys(cfg["SB3_OBS_KEYS"])
    print(f"Config: TAU={tau}, INITIAL_WEALTH={initial_wealth}, GAS_COST={gas}, "
          f"obs_keys={obs_keys}")

    # ── Model + VecNormalize ──
    model_path = args.run_dir / f"{args.agent}_model.zip"
    vec_path   = args.run_dir / f"{args.agent}_vec_normalize.pkl"
    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            f"Missing artefacts in {args.run_dir}: expected "
            f"{model_path.name} and {vec_path.name}."
        )
    model = PPO.load(model_path, device="cpu")
    with open(vec_path, "rb") as f:
        vec_norm = pickle.load(f)
    print(f"Loaded {args.agent} model + VecNormalize stats.")

    # ── Fixed neutral defaults for the non-swept, non-state features ──
    fixed = {
        "time":               args.time,
        "gas_cost":           gas,
        "lp_token1_amount":   initial_wealth / 2.0,
        "midprice":           1000.0,
        "sqrt_price":         1000.0,
        "lp_collected_fees_0": 0.0,
        "lp_collected_fees_1": 0.0,
    }

    # ── Pick conditioning states for this agent ──
    states = list(STATES_COMMON)
    if args.agent == "ppo":
        states.extend(STATES_PPO_ONLY)
    print(f"Conditioning states ({len(states)}): {[s['name'] for s in states]}")

    # ── Predict per state ──
    mispricing = np.linspace(
        args.mispricing_range[0], args.mispricing_range[1], args.n_mispricing,
    ).astype(np.float32)
    actions_per_state: dict[str, dict] = {}
    for s in states:
        obs = build_obs_row(obs_keys, mispricing, s, fixed)
        norm_obs = normalise(obs, vec_norm)
        raw_actions, _ = model.predict(norm_obs, deterministic=True)
        actions_per_state[s["name"]] = decode_action(raw_actions, args.agent, tau)

    # ── Plot: 2-panel figure ──
    # The hold dimension is conveyed by the dashed segments on each line, so
    # a standalone hold panel would just restate the same information.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        f"Policy response to mispricing — {args.agent}  "
        f"(solid = rebalance, dashed = hold;  time={args.time}, gas={gas})",
        fontsize=13,
    )

    panels = [
        ("center",     "Center (ticks)",     (-tau - 1, tau + 1)),
        ("half_width", "Half-width (ticks)", (0,         tau + 1)),
    ]

    for ax, (key, ylabel, ylim) in zip(axes, panels):
        for s in states:
            y = actions_per_state[s["name"]][key]
            color = s["color"]
            # Mask cells where hold == +1 so dashed segments mark "ghost"
            # actions the env won't execute.
            hold = actions_per_state[s["name"]]["hold"]
            held = hold == 1
            y_exec = np.where(~held, y, np.nan)
            y_held = np.where( held, y, np.nan)
            ax.plot(mispricing, y_exec, "-",  color=color, linewidth=1.5,
                    alpha=0.85, label=s["name"])
            ax.plot(mispricing, y_held, "--", color=color, linewidth=1.0,
                    alpha=0.4)
        ax.set_xlabel("Mispricing  (S − Z)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(ylim)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="best", fontsize=8)

    out_path = (args.run_dir /
                f"{args.agent}_mispricing_response__time{args.time:g}__gas{gas:g}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
