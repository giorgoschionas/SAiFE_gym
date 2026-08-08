"""
Plot price evolution for the fixed-parameter AMM market model.

The environment is constructed through ``train_robust_lp_agent.make_fixed_env``
so this diagnostic uses the same GBM midprice, liquidity-kernel arrivals, and
liquidity-depth Uniswap v3 price impact as the domain-randomized PPO experiment.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/saife_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.helpers import INITIAL_WEALTH, SEED, TERMINAL_TIME  # noqa: E402
from experiments.train_robust_lp_agent import make_fixed_env  # noqa: E402
from SAiFE_gym.agents.BaselineAgents import DoNothingAgent  # noqa: E402
from SAiFE_gym.gym.domain_randomization import DomainParameters  # noqa: E402
from SAiFE_gym.gym.index_names import (  # noqa: E402
    ASSET_PRICE_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_SQRT_PRICE_KEY,
    TIME_KEY,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot fixed-parameter AMM pool price and external midprice."
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/figures/amm_price_evolution",
    )
    parser.add_argument("--num-sims", type=int, default=1)
    parser.add_argument("--terminal-time", type=float, default=TERMINAL_TIME)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.009)
    parser.add_argument("--arrival-rate", type=float, default=100.0)
    parser.add_argument("--gas-cost", type=float, default=0.0)
    parser.add_argument("--alpha3", type=float, default=15000.0)
    parser.add_argument("--initial-wealth", type=float, default=INITIAL_WEALTH)
    parser.add_argument("--inventory-phi", type=float, default=0.02)
    parser.add_argument("--arrival-alpha2", type=float, default=0.0)
    parser.add_argument("--kernel-beta", type=float, default=0.5)
    parser.add_argument("--kernel-window", type=int, default=10)
    parser.add_argument("--liquidity-scale", type=float, default=1e6)
    parser.add_argument("--trade-size-notional", type=float, default=40.0)
    parser.add_argument("--price-impact-depth-window", type=int, default=10)
    parser.add_argument("--price-impact-min-depth", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def collect_price_path(args: argparse.Namespace, seed: int) -> dict[str, np.ndarray]:
    env_args = argparse.Namespace(
        **vars(args),
        num_trajectories=1,
        nominal_gas_cost=args.gas_cost,
    )
    env = make_fixed_env(
        env_args,
        DomainParameters(
            sigma=args.sigma,
            arrival_rate=args.arrival_rate,
        ),
        seed=seed,
    )
    agent = DoNothingAgent(env, hold_cash=True)
    state, _ = env.reset(seed=seed)

    rows = {
        "time": [],
        "pool_price": [],
        "midprice": [],
        "current_tick": [],
        "sell_arrival": [],
        "buy_arrival": [],
        "cumulative_arrivals": [],
        "reward": [],
    }
    cumulative_arrivals = 0

    def append_row(reward: float = 0.0, sell_arrival: bool = False, buy_arrival: bool = False) -> None:
        rows["time"].append(float(state[TIME_KEY][0]))
        rows["pool_price"].append(float(state[POOL_SQRT_PRICE_KEY][0] ** 2))
        rows["midprice"].append(float(state[ASSET_PRICE_KEY][0]))
        rows["current_tick"].append(int(state[POOL_CURRENT_TICK_KEY][0]))
        rows["sell_arrival"].append(int(sell_arrival))
        rows["buy_arrival"].append(int(buy_arrival))
        rows["cumulative_arrivals"].append(int(cumulative_arrivals))
        rows["reward"].append(float(reward))

    append_row()
    for _ in range(env.n_steps):
        action = agent.get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        sell_arrival, buy_arrival = env.model_dynamics.last_arrivals[0]
        cumulative_arrivals += int(sell_arrival) + int(buy_arrival)
        append_row(
            reward=float(reward[0]),
            sell_arrival=bool(sell_arrival),
            buy_arrival=bool(buy_arrival),
        )
        if (terminated | truncated).all():
            break

    return {key: np.asarray(value) for key, value in rows.items()}


def save_rollout_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    fieldnames = list(data.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(data["time"])):
            writer.writerow({key: data[key][idx].item() for key in fieldnames})


def plot_price_paths(paths: list[dict[str, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    alpha = max(0.25, 0.9 / max(len(paths), 1))
    for idx, data in enumerate(paths):
        pool_label = "Pool price" if idx == 0 else None
        mid_label = "External midprice" if idx == 0 else None
        ax.plot(
            data["time"],
            data["pool_price"],
            color="black",
            linewidth=1.2,
            alpha=alpha,
            label=pool_label,
        )
        ax.plot(
            data["time"],
            data["midprice"],
            color="gray",
            linewidth=1.0,
            alpha=alpha,
            label=mid_label,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", vars(args))

    paths = []
    for sim_idx in range(args.num_sims):
        data = collect_price_path(args, seed=args.seed + sim_idx)
        paths.append(data)
        save_rollout_csv(output_dir / f"rollout_sim_{sim_idx + 1:03d}.csv", data)

    plot_price_paths(paths, output_dir / "price_evolution.png")
    print(f"Saved AMM price evolution artifacts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
