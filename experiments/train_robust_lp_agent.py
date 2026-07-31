"""
Train nominal and domain-randomized PPO LP agents.

This script is intended for HPC runs. Use --smoke-test for a tiny local
end-to-end check that exercises model construction, learning, saving, and
held-out evaluation without running a real experiment.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/saife_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.helpers import (  # noqa: E402
    INITIAL_WEALTH,
    SEED,
    TERMINAL_TIME,
    get_amm_env,
    wrap_env,
)
from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent  # noqa: E402
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import (  # noqa: E402
    StableBaselinesAMMEnvironment,
)
from SAiFE_gym.gym.domain_randomization import (  # noqa: E402
    DomainParameters,
    DomainRandomizedAMMEnvironment,
    UniformDomainRandomizationConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate robust LP PPO with domain randomization."
    )
    parser.add_argument("--output-dir", default="experiments/results/robust_rl")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--terminal-time", type=float, default=TERMINAL_TIME)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--alpha3", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--no-normalise-obs", dest="normalise_obs", action="store_false")
    parser.set_defaults(normalise_obs=True)

    parser.add_argument("--nominal-sigma", type=float, default=2.0)
    parser.add_argument("--nominal-arrival-rate", type=float, default=100.0)
    parser.add_argument("--nominal-gas-cost", type=float, default=0.0)

    parser.add_argument("--train-sigma-range", nargs=2, type=float, default=(1.0, 4.0))
    parser.add_argument(
        "--train-arrival-rate-range", nargs=2, type=float, default=(50.0, 200.0)
    )
    parser.add_argument("--train-gas-cost-range", nargs=2, type=float, default=(0.0, 20.0))

    parser.add_argument("--eval-sigma-values", nargs="+", type=float, default=[0.5, 2.0, 6.0])
    parser.add_argument(
        "--eval-arrival-rate-values",
        nargs="+",
        type=float,
        default=[25.0, 100.0, 300.0],
    )
    parser.add_argument("--eval-gas-cost-values", nargs="+", type=float, default=[0.0, 10.0, 40.0])
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny end-to-end check instead of an HPC-sized experiment.",
    )
    return parser.parse_args()


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return

    args.total_timesteps = 20
    args.num_trajectories = 2
    args.n_steps = 5
    args.n_eval_episodes = 1
    args.eval_sigma_values = [1.0, 2.0]
    args.eval_arrival_rate_values = [50.0, 100.0]
    args.eval_gas_cost_values = [0.0, 10.0]


def make_fixed_env(
    args: argparse.Namespace,
    params: DomainParameters,
    seed: int,
):
    return get_amm_env(
        num_trajectories=args.num_trajectories,
        terminal_time=args.terminal_time,
        n_steps=args.n_steps,
        tau=args.tau,
        volatility=params.sigma,
        arrival_rate=params.arrival_rate,
        alpha3=args.alpha3,
        gas_cost=params.gas_cost,
        seed=seed,
    )


def make_robust_env(args: argparse.Namespace):
    base_env = make_fixed_env(
        args,
        DomainParameters(
            sigma=args.nominal_sigma,
            arrival_rate=args.nominal_arrival_rate,
            gas_cost=args.nominal_gas_cost,
        ),
        seed=args.seed,
    )
    config = UniformDomainRandomizationConfig(
        sigma_range=tuple(args.train_sigma_range),
        arrival_rate_range=tuple(args.train_arrival_rate_range),
        gas_cost_range=tuple(args.train_gas_cost_range),
    )
    return DomainRandomizedAMMEnvironment(base_env, config, seed=args.seed + 10_000)


def build_ppo(env, args: argparse.Namespace) -> tuple[PPO, object]:
    vec_env = wrap_env(env, normalise_obs=args.normalise_obs)
    rollout_size = args.n_steps * args.num_trajectories
    batch_size = min(max(64, rollout_size // 16), rollout_size)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_epochs=10,
        batch_size=batch_size,
        normalize_advantage=True,
        n_steps=args.n_steps,
        gae_lambda=0.95,
        gamma=1.0,
        seed=args.seed,
    )
    return model, vec_env


def train_and_save(
    label: str,
    env,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[PPO, Optional[VecNormalize]]:
    model, vec_env = build_ppo(env, args)
    model.learn(total_timesteps=args.total_timesteps)

    model.save(str(run_dir / f"{label}_ppo"))
    vec_normalize = vec_env if isinstance(vec_env, VecNormalize) else None
    if vec_normalize is not None:
        vec_normalize.save(str(run_dir / f"{label}_vecnormalize.pkl"))
        vec_normalize.training = False

    return model, vec_normalize


def evaluate_ppo(
    policy_name: str,
    model: PPO,
    vec_normalize: Optional[VecNormalize],
    params: DomainParameters,
    args: argparse.Namespace,
    regime_seed: int,
) -> dict:
    wealths = []
    for episode_idx in range(args.n_eval_episodes):
        episode_seed = regime_seed + episode_idx
        env = make_fixed_env(args, params, seed=episode_seed)
        sb3_env = StableBaselinesAMMEnvironment(env)
        obs, _ = env.reset(seed=episode_seed)
        cumulative_reward = np.zeros(env.num_trajectories)

        for _ in range(env.n_steps):
            flat_obs = sb3_env._flatten_obs(obs)
            model_obs = normalize_obs(flat_obs, vec_normalize)
            action, _ = model.predict(model_obs, deterministic=True)
            obs, rewards, terminated, truncated, _ = env.step(action)
            cumulative_reward += rewards
            if (terminated | truncated).all():
                break

        wealths.append(INITIAL_WEALTH + cumulative_reward)

    return summarize_wealth(policy_name, params, np.concatenate(wealths))


def evaluate_uniform(
    params: DomainParameters,
    args: argparse.Namespace,
    regime_seed: int,
) -> dict:
    wealths = []
    for episode_idx in range(args.n_eval_episodes):
        episode_seed = regime_seed + episode_idx
        env = make_fixed_env(args, params, seed=episode_seed)
        agent = UniformAllocationAgent(env)
        obs, _ = env.reset(seed=episode_seed)
        cumulative_reward = np.zeros(env.num_trajectories)

        for _ in range(env.n_steps):
            action = agent.get_action(obs)
            obs, rewards, terminated, truncated, _ = env.step(action)
            cumulative_reward += rewards
            if (terminated | truncated).all():
                break

        wealths.append(INITIAL_WEALTH + cumulative_reward)

    return summarize_wealth("uniform", params, np.concatenate(wealths))


def normalize_obs(
    obs: np.ndarray,
    vec_normalize: Optional[VecNormalize],
) -> np.ndarray:
    if vec_normalize is None:
        return obs

    old_training = vec_normalize.training
    vec_normalize.training = False
    normalized = vec_normalize.normalize_obs(obs)
    vec_normalize.training = old_training
    return normalized


def summarize_wealth(
    policy_name: str,
    params: DomainParameters,
    final_wealths: np.ndarray,
) -> dict:
    pnl = final_wealths - INITIAL_WEALTH
    return {
        "policy": policy_name,
        "sigma": params.sigma,
        "arrival_rate": params.arrival_rate,
        "gas_cost": params.gas_cost,
        "mean_final_wealth": float(np.mean(final_wealths)),
        "std_final_wealth": float(np.std(final_wealths)),
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl)),
        "n_samples": int(final_wealths.shape[0]),
    }


def evaluation_grid(args: argparse.Namespace) -> list[DomainParameters]:
    return [
        DomainParameters(sigma=sigma, arrival_rate=arrival_rate, gas_cost=gas_cost)
        for sigma in args.eval_sigma_values
        for arrival_rate in args.eval_arrival_rate_values
        for gas_cost in args.eval_gas_cost_values
    ]


def add_gap_columns(rows: list[dict]) -> None:
    by_key = {
        (row["sigma"], row["arrival_rate"], row["gas_cost"], row["policy"]): row
        for row in rows
    }
    for row in rows:
        key = (row["sigma"], row["arrival_rate"], row["gas_cost"])
        robust = by_key.get((*key, "robust_ppo"))
        nominal = by_key.get((*key, "nominal_ppo"))
        uniform = by_key.get((*key, "uniform"))
        row["robust_vs_nominal_mean_pnl_gap"] = (
            robust["mean_pnl"] - nominal["mean_pnl"]
            if robust is not None and nominal is not None
            else ""
        )
        row["robust_vs_uniform_mean_pnl_gap"] = (
            robust["mean_pnl"] - uniform["mean_pnl"]
            if robust is not None and uniform is not None
            else ""
        )


def summarize_rows(rows: list[dict]) -> dict:
    policies = sorted({row["policy"] for row in rows})
    return {
        policy: {
            "mean_of_regime_mean_pnl": float(
                np.mean([row["mean_pnl"] for row in rows if row["policy"] == policy])
            ),
            "worst_regime_mean_pnl": float(
                np.min([row["mean_pnl"] for row in rows if row["policy"] == policy])
            ),
            "best_regime_mean_pnl": float(
                np.max([row["mean_pnl"] for row in rows if row["policy"] == policy])
            ),
        }
        for policy in policies
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def make_run_dir(output_dir: str, smoke_test: bool) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "smoke" if smoke_test else "run"
    run_dir = Path(output_dir) / f"{suffix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> int:
    args = parse_args()
    apply_smoke_overrides(args)
    run_dir = make_run_dir(args.output_dir, args.smoke_test)
    save_json(run_dir / "config.json", vars(args))

    robust_env = make_robust_env(args)
    robust_model, robust_vecnormalize = train_and_save(
        "robust", robust_env, args, run_dir
    )

    nominal_params = DomainParameters(
        sigma=args.nominal_sigma,
        arrival_rate=args.nominal_arrival_rate,
        gas_cost=args.nominal_gas_cost,
    )
    nominal_env = make_fixed_env(args, nominal_params, seed=args.seed + 20_000)
    nominal_model, nominal_vecnormalize = train_and_save(
        "nominal", nominal_env, args, run_dir
    )

    rows = []
    for regime_idx, params in enumerate(evaluation_grid(args)):
        regime_seed = args.seed + 100_000 + regime_idx * 1_000
        rows.append(
            evaluate_ppo(
                "robust_ppo",
                robust_model,
                robust_vecnormalize,
                params,
                args,
                regime_seed,
            )
        )
        rows.append(
            evaluate_ppo(
                "nominal_ppo",
                nominal_model,
                nominal_vecnormalize,
                params,
                args,
                regime_seed,
            )
        )
        rows.append(evaluate_uniform(params, args, regime_seed))

    add_gap_columns(rows)
    save_csv(run_dir / "evaluation_grid.csv", rows)
    save_json(run_dir / "summary.json", summarize_rows(rows))

    print(f"Saved robust RL run artifacts to: {run_dir}")
    print(json.dumps(summarize_rows(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
