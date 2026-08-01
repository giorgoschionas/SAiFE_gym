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
from typing import Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/saife_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.helpers import (  # noqa: E402
    FEE_TIER,
    INITIAL_PRICE,
    INITIAL_WEALTH,
    NUM_TICKS,
    SEED,
    TERMINAL_TIME,
    wrap_env,
)
from SAiFE_gym.agents.BaselineAgents import PeriodicRebalanceAgent  # noqa: E402
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment  # noqa: E402
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics  # noqa: E402
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import (  # noqa: E402
    StableBaselinesAMMEnvironment,
)
from SAiFE_gym.gym.domain_randomization import (  # noqa: E402
    DomainParameters,
    DomainRandomizedAMMEnvironment,
    UniformDomainRandomizationConfig,
)
from SAiFE_gym.rewards.RewardFunctions import RunningInventoryPenalty  # noqa: E402
from SAiFE_gym.stochastic_processes.arrival_models import (  # noqa: E402
    LiquidityKernelArrivalModel,
)
from SAiFE_gym.stochastic_processes.midprice_models import (  # noqa: E402
    GeometricBrownianMotionMidpriceModel,
)
from SAiFE_gym.stochastic_processes.price_impact_models import (  # noqa: E402
    LiquidityDepthUniswapV3PriceImpact,
)
from SAiFE_gym.wrappers import StructuredMultiDiscreteVecEnv  # noqa: E402


def constant_trade_size_sampler(trade_notional: float):
    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, trade_notional, dtype=np.float64)

    return sampler


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate robust LP PPO with domain randomization."
    )
    parser.add_argument("--output-dir", default="experiments/results/robust_rl")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--terminal-time", type=float, default=TERMINAL_TIME)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--alpha3", type=float, default=4000.0)
    # LP capital. Fee income scales with pool volume, not with this, so raising
    # it dilutes fees relative to the position's mark-to-market price noise. At
    # the 1e6 default, per-episode fees (~1e1) are dwarfed by PnL spread (~5e4)
    # and holding no position is the reward-maximizing policy.
    parser.add_argument("--initial-wealth", type=float, default=INITIAL_WEALTH)
    # Risk charge as a fraction of --initial-wealth per unit time at full token0
    # exposure; see RunningInventoryPenalty's value-normalized formulation.
    parser.add_argument("--inventory-phi", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--periodic-rebalance-every", type=int, default=5)
    parser.add_argument("--periodic-width", type=int, default=2)
    parser.add_argument("--no-normalise-obs", dest="normalise_obs", action="store_false")
    parser.set_defaults(normalise_obs=True)

    parser.add_argument("--nominal-sigma", type=float, default=0.10)
    parser.add_argument("--nominal-arrival-rate", type=float, default=100.0)
    parser.add_argument("--nominal-gas-cost", type=float, default=0.0)

    parser.add_argument("--train-sigma-range", nargs=2, type=float, default=(0.01, 0.10))
    parser.add_argument(
        "--train-arrival-rate-range", nargs=2, type=float, default=(50.0, 200.0)
    )
    parser.add_argument("--train-gas-cost-range", nargs=2, type=float, default=(1.0, 6.0))
    parser.add_argument("--arrival-alpha2", type=float, default=0.0)
    parser.add_argument("--kernel-beta", type=float, default=0.5)
    parser.add_argument("--kernel-window", type=int, default=10)
    parser.add_argument("--liquidity-scale", type=float, default=1e6)
    parser.add_argument("--trade-size-notional", type=float, default=40.0)
    parser.add_argument("--price-impact-depth-window", type=int, default=10)
    parser.add_argument("--price-impact-min-depth", type=float, default=1e-12)

    parser.add_argument("--eval-sigma-values", nargs="+", type=float, default=[0.025, 0.10, 0.30])
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
    return parser.parse_args(argv)


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return

    args.total_timesteps = 20
    args.num_trajectories = 2
    args.n_steps = 5
    args.n_eval_episodes = 1
    args.eval_sigma_values = [0.05, 0.10]
    args.eval_arrival_rate_values = [50.0, 100.0]
    args.eval_gas_cost_values = [0.0, 10.0]


def make_fixed_env(
    args: argparse.Namespace,
    params: DomainParameters,
    seed: int,
):
    step_size = args.terminal_time / args.n_steps
    alpha = np.array([
        [10.0, 10.0],
        [params.arrival_rate, params.arrival_rate],
        [args.arrival_alpha2, args.arrival_alpha2],
        [args.alpha3, args.alpha3],
    ])
    midprice_model = GeometricBrownianMotionMidpriceModel(
        drift=0.0,
        volatility=params.sigma,
        initial_price=INITIAL_PRICE,
        terminal_time=args.terminal_time,
        step_size=step_size,
        num_trajectories=args.num_trajectories,
        seed=seed,
    )
    arrival_model = LiquidityKernelArrivalModel(
        alpha=alpha,
        beta=args.kernel_beta,
        K=args.kernel_window,
        liquidity_scale=args.liquidity_scale,
        step_size=step_size,
        num_trajectories=args.num_trajectories,
        seed=seed + 1,
    )
    price_impact_model = LiquidityDepthUniswapV3PriceImpact(
        trade_size_sampler=constant_trade_size_sampler(args.trade_size_notional),
        depth_window=args.price_impact_depth_window,
        min_depth=args.price_impact_min_depth,
        trade_size_unit="token1_notional",
        num_trajectories=args.num_trajectories,
        seed=seed + 2,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        price_impact_model=price_impact_model,
        num_trajectories=args.num_trajectories,
        fee_tier=FEE_TIER,
        tau=args.tau,
        num_ticks=NUM_TICKS,
        exponential_value=1.0001,
        gas_cost=params.gas_cost,
        swap_fee_rate=0.0,
        seed=seed + 3,
    )
    return AMMEnvironment(
        terminal_time=args.terminal_time,
        n_steps=args.n_steps,
        model_dynamics=model_dynamics,
        reward_function=RunningInventoryPenalty(
            per_step_inventory_aversion=args.inventory_phi,
            reference_wealth=args.initial_wealth,
        ),
        initial_wealth=args.initial_wealth,
        num_trajectories=args.num_trajectories,
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


def build_ppo(env, args: argparse.Namespace) -> tuple[PPO, object, Optional[VecNormalize]]:
    vec_env = wrap_env(env, normalise_obs=args.normalise_obs)
    vec_normalize = vec_env if isinstance(vec_env, VecNormalize) else None
    ppo_env = StructuredMultiDiscreteVecEnv(vec_env, args.tau)
    rollout_size = args.n_steps * args.num_trajectories
    batch_size = min(max(64, rollout_size // 16), rollout_size)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO(
        "MlpPolicy",
        ppo_env,
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
    return model, ppo_env, vec_normalize


def train_and_save(
    label: str,
    env,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[PPO, Optional[VecNormalize]]:
    model, _, vec_normalize = build_ppo(env, args)
    model.learn(total_timesteps=args.total_timesteps)

    model.save(str(run_dir / f"{label}_ppo"))
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
    objective_values = []
    for episode_idx in range(args.n_eval_episodes):
        episode_seed = regime_seed + episode_idx
        env = make_fixed_env(args, params, seed=episode_seed)
        sb3_env = StableBaselinesAMMEnvironment(env)
        action_wrapper = StructuredMultiDiscreteVecEnv(sb3_env, args.tau)
        obs, _ = env.reset(seed=episode_seed)
        cumulative_reward = np.zeros(env.num_trajectories)

        for _ in range(env.n_steps):
            flat_obs = sb3_env._flatten_obs(obs)
            model_obs = normalize_obs(flat_obs, vec_normalize)
            structured_action, _ = model.predict(model_obs, deterministic=True)
            action = action_wrapper.unscale(structured_action)
            obs, rewards, terminated, truncated, _ = env.step(action)
            cumulative_reward += rewards
            if (terminated | truncated).all():
                break

        objective_values.append(cumulative_reward)

    return summarize_running_inventory_objective(
        policy_name,
        params,
        np.concatenate(objective_values),
    )


def evaluate_periodic_rebalance(
    params: DomainParameters,
    args: argparse.Namespace,
    regime_seed: int,
) -> dict:
    objective_values = []
    for episode_idx in range(args.n_eval_episodes):
        episode_seed = regime_seed + episode_idx
        env = make_fixed_env(args, params, seed=episode_seed)
        agent = PeriodicRebalanceAgent(
            env,
            rebalance_every=args.periodic_rebalance_every,
            width=args.periodic_width,
        )
        obs, _ = env.reset(seed=episode_seed)
        cumulative_reward = np.zeros(env.num_trajectories)

        for _ in range(env.n_steps):
            action = agent.get_action(obs)
            obs, rewards, terminated, truncated, _ = env.step(action)
            cumulative_reward += rewards
            if (terminated | truncated).all():
                break

        objective_values.append(cumulative_reward)

    return summarize_running_inventory_objective(
        "periodic_rebalance",
        params,
        np.concatenate(objective_values),
    )


def evaluate_cash(
    params: DomainParameters,
    args: argparse.Namespace,
) -> dict:
    """Report the undeployed token1 baseline without simulating market paths.

    Token1 is the numeraire. An agent that never deploys keeps portfolio value
    fixed at initial wealth and has no token0 inventory, so both PnL and the
    running inventory penalty are exactly zero under the current objective.
    """
    objective_values = np.zeros(args.n_eval_episodes * args.num_trajectories)
    return summarize_running_inventory_objective(
        "cash",
        params,
        objective_values,
    )


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


def summarize_running_inventory_objective(
    policy_name: str,
    params: DomainParameters,
    objective_values: np.ndarray,
) -> dict:
    return {
        "policy": policy_name,
        "sigma": params.sigma,
        "arrival_rate": params.arrival_rate,
        "gas_cost": params.gas_cost,
        "mean_running_inventory_objective": float(np.mean(objective_values)),
        "std_running_inventory_objective": float(np.std(objective_values)),
        "n_samples": int(objective_values.shape[0]),
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
        periodic_rebalance = by_key.get((*key, "periodic_rebalance"))
        cash = by_key.get((*key, "cash"))
        row["robust_vs_nominal_mean_running_inventory_objective_gap"] = (
            robust["mean_running_inventory_objective"]
            - nominal["mean_running_inventory_objective"]
            if robust is not None and nominal is not None
            else ""
        )
        row["robust_vs_periodic_rebalance_mean_running_inventory_objective_gap"] = (
            robust["mean_running_inventory_objective"]
            - periodic_rebalance["mean_running_inventory_objective"]
            if robust is not None and periodic_rebalance is not None
            else ""
        )
        row["robust_vs_cash_mean_running_inventory_objective_gap"] = (
            robust["mean_running_inventory_objective"]
            - cash["mean_running_inventory_objective"]
            if robust is not None and cash is not None
            else ""
        )


def summarize_rows(rows: list[dict]) -> dict:
    policies = sorted({row["policy"] for row in rows})
    return {
        policy: {
            "mean_of_regime_mean_running_inventory_objective": float(
                np.mean([
                    row["mean_running_inventory_objective"]
                    for row in rows
                    if row["policy"] == policy
                ])
            ),
            "worst_regime_mean_running_inventory_objective": float(
                np.min([
                    row["mean_running_inventory_objective"]
                    for row in rows
                    if row["policy"] == policy
                ])
            ),
            "best_regime_mean_running_inventory_objective": float(
                np.max([
                    row["mean_running_inventory_objective"]
                    for row in rows
                    if row["policy"] == policy
                ])
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
    """Create a fresh run directory, tolerating same-second sibling jobs.

    Slurm array tasks start within the same second and share an output dir, so
    the timestamp alone collides. mkdir(exist_ok=False) is atomic, which makes
    the retry loop safe against concurrent tasks racing for the same name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(output_dir) / f"{'smoke' if smoke_test else 'run'}_{timestamp}"

    run_dir = base
    attempt = 1
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            run_dir = base.with_name(f"{base.name}_{attempt}")
            attempt += 1


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
        rows.append(evaluate_periodic_rebalance(params, args, regime_seed))
        rows.append(evaluate_cash(params, args))

    add_gap_columns(rows)
    save_csv(run_dir / "evaluation_grid.csv", rows)
    save_json(run_dir / "summary.json", summarize_rows(rows))

    print(f"Saved robust RL run artifacts to: {run_dir}")
    print(json.dumps(summarize_rows(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
