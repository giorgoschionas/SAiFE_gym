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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/saife_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.helpers import (  # noqa: E402
    FEE_TIER,
    INITIAL_PRICE,
    NUM_TICKS,
    SEED,
    TERMINAL_TIME,
    wrap_env,
)
from experiments.policy_behavior_diagnostics import (  # noqa: E402
    BEHAVIOR_DIAGNOSTIC_COLUMNS,
    BEHAVIOR_DIAGNOSTICS_SCHEMA_VERSION,
    PolicyBehaviorAccumulator,
    cash_behavior_diagnostics,
    zero_behavior_diagnostics,
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
from SAiFE_gym.gym.index_names import LP_LIQUIDITY_KEY  # noqa: E402
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


DEFAULT_EVALUATION_SEED = SEED + 100_000
DEFAULT_TRAIN_DOMAINS_PER_RESET = 10
DEFAULT_DOMAIN_RANDOMIZATION_SEED_OFFSET = 10_000
FORCED_HOLD_ACTION = np.array([0.0, 1.0, 1.0], dtype=np.float32)
AGENT_DECISION_DIAGNOSTIC_COLUMNS = (
    "agent_decision_stride",
    "agent_decisions_per_episode",
    "agent_hold_decision_fraction",
    "agent_rebalance_decision_fraction",
    "mean_agent_rebalances_after_deployment_per_path",
    "mean_agent_selected_range_width_ticks",
)


@dataclass(frozen=True)
class EvaluationRegime:
    """A fixed evaluation domain and its diagnostic evaluation set."""

    evaluation_set: Literal["in_distribution", "stress"]
    parameters: DomainParameters


def constant_trade_size_sampler(trade_notional: float):
    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, trade_notional, dtype=np.float64)

    return sampler


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate domain-randomized LP PPO."
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/domain_randomized_ppo",
    )
    parser.add_argument("--total-timesteps", type=int, default=6_000_000)
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--terminal-time", type=float, default=TERMINAL_TIME)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument(
        "--decision-stride",
        type=int,
        default=20,
        help=(
            "Number of simulator steps per PPO decision. The agent acts on "
            "the first simulator step and forced hold is used for the "
            "remaining steps in the decision window."
        ),
    )
    parser.add_argument("--tau", type=int, default=500)
    parser.add_argument("--tick-stride", type=int, default=5)
    parser.add_argument("--alpha3", type=float, default=4000.0)
    # LP capital. Fee income scales with pool volume, not with this, so raising
    # it dilutes fees relative to the position's mark-to-market price noise.
    # The previous 1e6 default made fee income negligible relative to PnL;
    # 1e3 keeps the LP's capital scale close to the modeled trading volume.
    parser.add_argument("--initial-wealth", type=float, default=1_000.0)
    # Risk charge on raw token0 inventory per unit time.
    parser.add_argument("--inventory-phi", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=DEFAULT_EVALUATION_SEED,
        help=(
            "Fixed seed for evaluation paths, independent of the PPO training "
            "seed."
        ),
    )
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument(
        "--convergence-eval-every-rollouts",
        type=int,
        default=10,
        help=(
            "Evaluate the current deterministic policy every N PPO rollouts "
            "on a fixed nominal validation regime; set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--convergence-n-eval-episodes",
        type=int,
        default=1,
        help="Number of vectorized episodes per convergence-check evaluation.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--periodic-rebalance-every", type=int, default=5)
    parser.add_argument("--periodic-width", type=int, default=125)
    parser.add_argument("--no-normalise-obs", dest="normalise_obs", action="store_false")
    parser.set_defaults(normalise_obs=True)

    parser.add_argument("--nominal-sigma", type=float, default=0.03)
    parser.add_argument("--nominal-arrival-rate", type=float, default=250.0)
    parser.add_argument("--nominal-gas-cost", type=float, default=2.0)

    parser.add_argument("--train-sigma-range", nargs=2, type=float, default=(0.01, 0.05))
    parser.add_argument(
        "--train-arrival-rate-range", nargs=2, type=float, default=(200.0, 400.0)
    )
    parser.add_argument(
        "--train-domains-per-reset",
        type=int,
        default=None,
        help=(
            "Number of balanced episode domains assigned across training "
            "trajectories (default: 10, capped at num-trajectories)."
        ),
    )
    parser.add_argument("--arrival-alpha2", type=float, default=0.0)
    parser.add_argument("--kernel-beta", type=float, default=0.5)
    parser.add_argument("--kernel-window", type=int, default=10)
    parser.add_argument("--liquidity-scale", type=float, default=1e6)
    parser.add_argument("--trade-size-notional", type=float, default=250.0)
    parser.add_argument("--price-impact-depth-window", type=int, default=10)
    parser.add_argument("--price-impact-min-depth", type=float, default=1e-12)

    parser.add_argument(
        "--eval-in-distribution-sigma-values",
        nargs="+",
        type=float,
        default=[0.015, 0.030, 0.045],
        help="Sigma values for the in-distribution Cartesian evaluation grid.",
    )
    parser.add_argument(
        "--eval-in-distribution-arrival-rate-values",
        nargs="+",
        type=float,
        default=[250.0, 300.0, 350.0],
        help=(
            "Arrival-rate values for the in-distribution Cartesian "
            "evaluation grid."
        ),
    )
    parser.add_argument(
        "--eval-stress-sigma-values",
        nargs="+",
        type=float,
        default=[0.065, 0.08],
        help="Sigma values for the out-of-distribution stress grid.",
    )
    parser.add_argument(
        "--eval-stress-arrival-rate-values",
        nargs="+",
        type=float,
        default=[150.0, 450.0],
        help="Arrival-rate values for the out-of-distribution stress grid.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny end-to-end check instead of an HPC-sized experiment.",
    )
    return parser.parse_args(argv)


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return

    args.total_timesteps = 2 * args.decision_stride
    args.num_trajectories = 2
    args.n_steps = args.decision_stride
    args.n_eval_episodes = 1
    args.eval_in_distribution_sigma_values = [0.030]
    args.eval_in_distribution_arrival_rate_values = [300.0]
    args.eval_stress_sigma_values = [0.08]
    args.eval_stress_arrival_rate_values = [450.0]


def validate_and_derive_decision_timing(args: argparse.Namespace) -> None:
    """Validate decision-stride timing and store PPO-derived step counts."""
    if isinstance(args.decision_stride, bool) or not isinstance(
        args.decision_stride,
        (int, np.integer),
    ):
        raise ValueError("decision_stride must be an integer")
    if args.decision_stride < 1:
        raise ValueError("decision_stride must be positive")
    if args.n_steps < 1:
        raise ValueError("n_steps must be positive")
    if args.total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    if args.n_steps % args.decision_stride != 0:
        raise ValueError(
            "n_steps must be divisible by decision_stride, got "
            f"n_steps={args.n_steps}, decision_stride={args.decision_stride}"
        )
    if args.total_timesteps % args.decision_stride != 0:
        raise ValueError(
            "total_timesteps must be divisible by decision_stride so it can "
            "remain a simulator-step budget, got "
            f"total_timesteps={args.total_timesteps}, "
            f"decision_stride={args.decision_stride}"
        )

    args.max_agent_decisions_per_episode = int(args.n_steps // args.decision_stride)
    args.ppo_n_steps = int(args.max_agent_decisions_per_episode)
    args.ppo_total_timesteps = int(args.total_timesteps // args.decision_stride)


def max_agent_decisions_per_episode(args: argparse.Namespace) -> int:
    return int(
        getattr(
            args,
            "max_agent_decisions_per_episode",
            args.n_steps // args.decision_stride,
        )
    )


def ppo_rollout_steps(args: argparse.Namespace) -> int:
    return int(getattr(args, "ppo_n_steps", max_agent_decisions_per_episode(args)))


def ppo_total_timesteps(args: argparse.Namespace) -> int:
    return int(
        getattr(
            args,
            "ppo_total_timesteps",
            args.total_timesteps // args.decision_stride,
        )
    )


def resolve_train_domains_per_reset(args: argparse.Namespace) -> int:
    """Resolve and validate the randomized environment's domain batch size."""
    num_domains = getattr(args, "train_domains_per_reset", None)
    if num_domains is None:
        num_domains = min(DEFAULT_TRAIN_DOMAINS_PER_RESET, args.num_trajectories)
    if isinstance(num_domains, bool) or not isinstance(
        num_domains, (int, np.integer)
    ):
        raise ValueError("train_domains_per_reset must be an integer")
    if not 1 <= num_domains <= args.num_trajectories:
        raise ValueError(
            "train_domains_per_reset must satisfy 1 <= value <= "
            f"num_trajectories ({args.num_trajectories}), got {num_domains}"
        )
    args.train_domains_per_reset = int(num_domains)
    return args.train_domains_per_reset


def _validate_evaluation_values(name: str, values: Sequence[float]) -> None:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1 or values_array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(values_array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values_array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    if len(set(values_array.tolist())) != values_array.size:
        raise ValueError(f"{name} must not contain duplicate values")


def _parameters_within_training_support(
    params: DomainParameters,
    config: UniformDomainRandomizationConfig,
) -> bool:
    return all([
        config.sigma_range[0] <= params.sigma <= config.sigma_range[1],
        config.arrival_rate_range[0]
        <= params.arrival_rate
        <= config.arrival_rate_range[1],
    ])


def validate_evaluation_configuration(args: argparse.Namespace) -> None:
    """Validate that diagnostic grids match their declared semantics."""
    if isinstance(args.evaluation_seed, bool) or not isinstance(
        args.evaluation_seed,
        (int, np.integer),
    ):
        raise ValueError("evaluation_seed must be an integer")
    if args.evaluation_seed < 0:
        raise ValueError("evaluation_seed must be non-negative")

    config = UniformDomainRandomizationConfig(
        sigma_range=tuple(args.train_sigma_range),
        arrival_rate_range=tuple(args.train_arrival_rate_range),
    )
    dimensions = [
        (
            "sigma",
            config.sigma_range,
            args.eval_in_distribution_sigma_values,
            args.eval_stress_sigma_values,
        ),
        (
            "arrival_rate",
            config.arrival_rate_range,
            args.eval_in_distribution_arrival_rate_values,
            args.eval_stress_arrival_rate_values,
        ),
    ]

    for parameter_name, training_range, in_distribution, stress in dimensions:
        in_distribution_name = f"eval_in_distribution_{parameter_name}_values"
        stress_name = f"eval_stress_{parameter_name}_values"
        _validate_evaluation_values(in_distribution_name, in_distribution)
        _validate_evaluation_values(stress_name, stress)

        low, high = training_range
        outside = [value for value in in_distribution if not low <= value <= high]
        if outside:
            raise ValueError(
                f"{in_distribution_name} must lie within training range "
                f"[{low}, {high}], got out-of-support values {outside}"
            )

    in_support_stress_regimes = [
        regime.parameters
        for regime in evaluation_regimes(args)
        if regime.evaluation_set == "stress"
        and _parameters_within_training_support(regime.parameters, config)
    ]
    if in_support_stress_regimes:
        raise ValueError(
            "every stress evaluation regime must have at least one parameter "
            "outside the training support; fully in-support regimes: "
            f"{in_support_stress_regimes}"
        )


def evaluation_regime_seed(args: argparse.Namespace, regime_index: int) -> int:
    """Return a deterministic regime seed independent of PPO training."""
    return int(args.evaluation_seed + regime_index * 1_000)


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
        gas_cost=args.nominal_gas_cost,
        swap_fee_rate=0.0,
        seed=seed + 3,
    )
    return AMMEnvironment(
        terminal_time=args.terminal_time,
        n_steps=args.n_steps,
        model_dynamics=model_dynamics,
        reward_function=RunningInventoryPenalty(
            per_step_inventory_aversion=args.inventory_phi,
        ),
        initial_wealth=args.initial_wealth,
        num_trajectories=args.num_trajectories,
        seed=seed,
    )


def make_domain_randomized_env(args: argparse.Namespace):
    """Build the PPO training environment with episode-level randomization."""
    num_domains = resolve_train_domains_per_reset(args)
    base_env = make_fixed_env(
        args,
        DomainParameters(
            sigma=args.nominal_sigma,
            arrival_rate=args.nominal_arrival_rate,
        ),
        seed=args.seed,
    )
    config = UniformDomainRandomizationConfig(
        sigma_range=tuple(args.train_sigma_range),
        arrival_rate_range=tuple(args.train_arrival_rate_range),
    )
    return DomainRandomizedAMMEnvironment(
        base_env,
        config,
        seed=args.seed,
        num_domains=num_domains,
        domain_seed_offset=DEFAULT_DOMAIN_RANDOMIZATION_SEED_OFFSET,
    )


def make_robust_env(args: argparse.Namespace):
    """Compatibility alias for :func:`make_domain_randomized_env`."""
    return make_domain_randomized_env(args)


class DecisionStrideVecEnv(VecEnv):
    """Expose one PPO step per fixed simulator decision window."""

    def __init__(self, vec_env: VecEnv, stride: int):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self._wrapped = vec_env
        self.stride = int(stride)
        self._hold = np.tile(FORCED_HOLD_ACTION, (vec_env.num_envs, 1))
        self._pending_action: Optional[np.ndarray] = None
        super().__init__(
            vec_env.num_envs,
            vec_env.observation_space,
            vec_env.action_space,
        )

    def reset(self) -> VecEnvObs:
        return self._wrapped.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_action = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> VecEnvStepReturn:
        if self._pending_action is None:
            raise RuntimeError("step_async must be called before step_wait")

        action = self._pending_action
        self._pending_action = None
        if self.stride == 1:
            self._wrapped.step_async(action)
            return self._wrapped.step_wait()

        reward_sum = np.zeros(self.num_envs, dtype=np.float32)
        obs = None
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        for inner_step in range(self.stride):
            inner_action = action if inner_step == 0 else self._hold
            self._wrapped.step_async(inner_action)
            obs, rewards, dones, infos = self._wrapped.step_wait()
            reward_sum += rewards.astype(np.float32)
            if np.all(dones):
                break

        if obs is None:
            raise RuntimeError("decision stride did not execute any simulator steps")
        return obs, reward_sum, dones, infos

    def close(self) -> None:
        self._wrapped.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list:
        return self._wrapped.get_attr(attr_name, indices)

    def set_attr(
        self,
        attr_name: str,
        value,
        indices: VecEnvIndices = None,
    ) -> None:
        self._wrapped.set_attr(attr_name, value, indices)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> list:
        return self._wrapped.env_method(
            method_name,
            *method_args,
            indices=indices,
            **method_kwargs,
        )

    def env_is_wrapped(
        self,
        wrapper_class: type,
        indices: VecEnvIndices = None,
    ) -> list[bool]:
        return self._wrapped.env_is_wrapped(wrapper_class, indices)

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        return self._wrapped.seed(seed)

    def get_images(self) -> Sequence[np.ndarray]:
        return self._wrapped.get_images()


class AgentDecisionAccumulator:
    """Track PPO decision-level hold/rebalance diagnostics."""

    def __init__(self, stride: int, decisions_per_episode: int) -> None:
        self.stride = int(stride)
        self.decisions_per_episode = int(decisions_per_episode)
        self.num_paths = 0
        self.action_decisions = 0
        self.hold_actions = 0
        self.rebalance_actions = 0
        self.rebalances_after_deployment = 0
        self.selected_range_width_total = 0.0
        self.selected_range_count = 0
        self._active_num_trajectories: Optional[int] = None

    def begin_episode(self, num_trajectories: int) -> None:
        if self._active_num_trajectories is not None:
            raise RuntimeError("finish the current episode before beginning another")
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be positive")
        self._active_num_trajectories = int(num_trajectories)

    def record_decision(self, state: dict, action: np.ndarray) -> None:
        num_trajectories = self._require_active_episode()
        action = np.asarray(action)
        if action.shape != (num_trajectories, 3):
            raise ValueError(
                f"action must have shape ({num_trajectories}, 3), got {action.shape}"
            )

        has_position = np.asarray(state[LP_LIQUIDITY_KEY]) > 0.0
        rebalance = action[:, 2] <= 0.0
        hold = ~rebalance
        self.action_decisions += num_trajectories
        self.hold_actions += int(np.count_nonzero(hold))
        self.rebalance_actions += int(np.count_nonzero(rebalance))
        self.rebalances_after_deployment += int(
            np.count_nonzero(rebalance & has_position)
        )

        selected_widths = action[rebalance, 1] - action[rebalance, 0]
        if selected_widths.size:
            self.selected_range_width_total += float(np.sum(selected_widths))
            self.selected_range_count += int(selected_widths.size)

    def finish_episode(self) -> None:
        num_trajectories = self._require_active_episode()
        self.num_paths += num_trajectories
        self._active_num_trajectories = None

    def summarize(self) -> dict[str, float]:
        if self._active_num_trajectories is not None:
            raise RuntimeError("finish the active episode before summarizing")
        if self.num_paths < 1:
            raise RuntimeError("at least one completed episode is required")
        return {
            "agent_decision_stride": float(self.stride),
            "agent_decisions_per_episode": float(self.decisions_per_episode),
            "agent_hold_decision_fraction": self._safe_ratio(
                self.hold_actions,
                self.action_decisions,
            ),
            "agent_rebalance_decision_fraction": self._safe_ratio(
                self.rebalance_actions,
                self.action_decisions,
            ),
            "mean_agent_rebalances_after_deployment_per_path": (
                self.rebalances_after_deployment / self.num_paths
            ),
            "mean_agent_selected_range_width_ticks": self._safe_ratio(
                self.selected_range_width_total,
                self.selected_range_count,
            ),
        }

    def _require_active_episode(self) -> int:
        if self._active_num_trajectories is None:
            raise RuntimeError("begin_episode must be called first")
        return self._active_num_trajectories

    @staticmethod
    def _safe_ratio(numerator: float, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0


def zero_agent_decision_diagnostics(args: argparse.Namespace) -> dict[str, float | str]:
    return {
        "agent_decision_stride": float(args.decision_stride),
        "agent_decisions_per_episode": float(max_agent_decisions_per_episode(args)),
        "agent_hold_decision_fraction": "",
        "agent_rebalance_decision_fraction": "",
        "mean_agent_rebalances_after_deployment_per_path": "",
        "mean_agent_selected_range_width_ticks": "",
    }


def build_ppo(env, args: argparse.Namespace) -> tuple[PPO, object, Optional[VecNormalize]]:
    vec_env = wrap_env(env, normalise_obs=args.normalise_obs)
    vec_normalize = vec_env if isinstance(vec_env, VecNormalize) else None
    decision_env = DecisionStrideVecEnv(vec_env, args.decision_stride)
    ppo_env = StructuredMultiDiscreteVecEnv(decision_env, args.tau, args.tick_stride)
    rollout_size = ppo_rollout_steps(args) * args.num_trajectories
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
        n_steps=ppo_rollout_steps(args),
        gae_lambda=0.95,
        gamma=1.0,
        seed=args.seed,
    )
    return model, ppo_env, vec_normalize


class ConvergenceEvaluationCallback(BaseCallback):
    """Append fixed-regime policy evaluations during PPO training."""

    def __init__(
        self,
        policy_name: str,
        vec_normalize: Optional[VecNormalize],
        args: argparse.Namespace,
        output_path: Path,
    ):
        super().__init__(verbose=0)
        if args.convergence_eval_every_rollouts < 0:
            raise ValueError("convergence_eval_every_rollouts must be non-negative")
        if args.convergence_n_eval_episodes < 1:
            raise ValueError("convergence_n_eval_episodes must be positive")
        self.policy_name = policy_name
        self.vec_normalize = vec_normalize
        self.args = args
        self.output_path = output_path
        self.rollouts_completed = 0
        self._fieldnames: Optional[list[str]] = None

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        frequency = self.args.convergence_eval_every_rollouts
        if frequency == 0:
            return

        self.rollouts_completed += 1
        if self.rollouts_completed % frequency != 0:
            return

        eval_args = argparse.Namespace(**vars(self.args))
        eval_args.n_eval_episodes = self.args.convergence_n_eval_episodes
        params = DomainParameters(
            sigma=self.args.nominal_sigma,
            arrival_rate=self.args.nominal_arrival_rate,
        )
        row = evaluate_ppo(
            self.policy_name,
            self.model,
            self.vec_normalize,
            params,
            "convergence_validation",
            eval_args,
            regime_seed=self.args.evaluation_seed,
        )
        row = {
            "num_timesteps": int(self.num_timesteps * self.args.decision_stride),
            "num_decision_timesteps": int(self.num_timesteps),
            "rollouts_completed": int(self.rollouts_completed),
            **row,
        }
        self._append_row(row)

    def _append_row(self, row: dict) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            write_header = not self.output_path.exists()
        else:
            write_header = False

        with self.output_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def append_final_convergence_evaluation(
    callback: ConvergenceEvaluationCallback,
    model: PPO,
    vec_normalize: Optional[VecNormalize],
    args: argparse.Namespace,
) -> None:
    """Append a validation row for the final post-update policy."""
    if args.convergence_eval_every_rollouts == 0:
        return

    eval_args = argparse.Namespace(**vars(args))
    eval_args.n_eval_episodes = args.convergence_n_eval_episodes
    params = DomainParameters(
        sigma=args.nominal_sigma,
        arrival_rate=args.nominal_arrival_rate,
    )
    row = evaluate_ppo(
        callback.policy_name,
        model,
        vec_normalize,
        params,
        "convergence_validation_final",
        eval_args,
        regime_seed=args.evaluation_seed,
    )
    row = {
        "num_timesteps": int(model.num_timesteps * args.decision_stride),
        "num_decision_timesteps": int(model.num_timesteps),
        "rollouts_completed": int(callback.rollouts_completed),
        **row,
    }
    callback._append_row(row)


def train_and_save(
    label: str,
    env,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[PPO, Optional[VecNormalize]]:
    model, _, vec_normalize = build_ppo(env, args)
    callback = ConvergenceEvaluationCallback(
        f"{label}_ppo",
        vec_normalize,
        args,
        run_dir / f"{label}_convergence.csv",
    )
    model.learn(total_timesteps=ppo_total_timesteps(args), callback=callback)
    append_final_convergence_evaluation(callback, model, vec_normalize, args)

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
    evaluation_set: Literal["in_distribution", "stress"],
    args: argparse.Namespace,
    regime_seed: int,
) -> dict:
    objective_values = []
    behavior = PolicyBehaviorAccumulator()
    decision_behavior = AgentDecisionAccumulator(
        args.decision_stride,
        max_agent_decisions_per_episode(args),
    )
    for episode_idx in range(args.n_eval_episodes):
        episode_seed = regime_seed + episode_idx
        env = make_fixed_env(args, params, seed=episode_seed)
        sb3_env = StableBaselinesAMMEnvironment(env)
        action_wrapper = StructuredMultiDiscreteVecEnv(
            sb3_env,
            args.tau,
            args.tick_stride,
        )
        obs, _ = env.reset(seed=episode_seed)
        cumulative_reward = np.zeros(env.num_trajectories)
        behavior.begin_episode(env.num_trajectories)
        decision_behavior.begin_episode(env.num_trajectories)
        hold_action = np.tile(FORCED_HOLD_ACTION, (env.num_trajectories, 1))

        for step_idx in range(env.n_steps):
            if step_idx % args.decision_stride == 0:
                flat_obs = sb3_env._flatten_obs(obs)
                model_obs = normalize_obs(flat_obs, vec_normalize)
                structured_action, _ = model.predict(model_obs, deterministic=True)
                action = action_wrapper.unscale(structured_action)
                decision_behavior.record_decision(obs, action)
            else:
                action = hold_action
            behavior_snapshot = behavior.before_step(obs, action)
            obs, rewards, terminated, truncated, _ = env.step(action)
            behavior.after_step(behavior_snapshot, obs, rewards)
            cumulative_reward += rewards
            if (terminated | truncated).all():
                break

        behavior.finish_episode(obs)
        decision_behavior.finish_episode()
        objective_values.append(cumulative_reward)

    row = summarize_running_inventory_objective(
        policy_name,
        params,
        evaluation_set,
        np.concatenate(objective_values),
        training_seed=args.seed,
        evaluation_seed=args.evaluation_seed,
        behavior_diagnostics=behavior.summarize(),
    )
    row.update(decision_behavior.summarize())
    return row


def evaluate_periodic_rebalance(
    params: DomainParameters,
    evaluation_set: Literal["in_distribution", "stress"],
    args: argparse.Namespace,
    regime_seed: int,
) -> dict:
    objective_values = []
    behavior = PolicyBehaviorAccumulator()
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
        behavior.begin_episode(env.num_trajectories)

        for _ in range(env.n_steps):
            action = agent.get_action(obs)
            behavior_snapshot = behavior.before_step(obs, action)
            obs, rewards, terminated, truncated, _ = env.step(action)
            behavior.after_step(behavior_snapshot, obs, rewards)
            cumulative_reward += rewards
            if (terminated | truncated).all():
                break

        behavior.finish_episode(obs)
        objective_values.append(cumulative_reward)

    row = summarize_running_inventory_objective(
        "periodic_rebalance",
        params,
        evaluation_set,
        np.concatenate(objective_values),
        training_seed=args.seed,
        evaluation_seed=args.evaluation_seed,
        behavior_diagnostics=behavior.summarize(),
    )
    row.update(zero_agent_decision_diagnostics(args))
    return row


def evaluate_cash(
    params: DomainParameters,
    evaluation_set: Literal["in_distribution", "stress"],
    args: argparse.Namespace,
) -> dict:
    """Report the undeployed token1 baseline without simulating market paths.

    Token1 is the numeraire. An agent that never deploys keeps portfolio value
    fixed at initial wealth and has no token0 inventory, so both PnL and the
    running inventory penalty are exactly zero under the current objective.
    """
    objective_values = np.zeros(args.n_eval_episodes * args.num_trajectories)
    row = summarize_running_inventory_objective(
        "cash",
        params,
        evaluation_set,
        objective_values,
        training_seed=args.seed,
        evaluation_seed=args.evaluation_seed,
        behavior_diagnostics=cash_behavior_diagnostics(),
    )
    row.update(zero_agent_decision_diagnostics(args))
    return row


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
    evaluation_set: Literal["in_distribution", "stress"],
    objective_values: np.ndarray,
    training_seed: int,
    evaluation_seed: int,
    behavior_diagnostics: Optional[dict[str, float]] = None,
) -> dict:
    objective_values = np.asarray(objective_values, dtype=np.float64)
    if objective_values.ndim != 1 or objective_values.size == 0:
        raise ValueError("objective_values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(objective_values)):
        raise ValueError("objective_values must contain only finite values")

    diagnostics = (
        zero_behavior_diagnostics()
        if behavior_diagnostics is None
        else behavior_diagnostics
    )
    if set(diagnostics) != set(BEHAVIOR_DIAGNOSTIC_COLUMNS):
        raise ValueError("behavior diagnostics do not match the required schema")
    if not all(np.isfinite(value) for value in diagnostics.values()):
        raise ValueError("behavior diagnostics must contain only finite values")
    if behavior_diagnostics is not None:
        decomposed_objective = (
            diagnostics["mean_pnl_per_path"]
            - diagnostics["mean_inventory_penalty_per_path"]
        )
        if not np.isclose(
            np.mean(objective_values),
            decomposed_objective,
            rtol=1e-9,
            atol=1e-7,
        ):
            raise ValueError(
                "mean objective must equal mean PnL minus mean inventory penalty"
            )

    return {
        "evaluation_set": evaluation_set,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "policy": policy_name,
        "sigma": params.sigma,
        "arrival_rate": params.arrival_rate,
        "mean_running_inventory_objective": float(np.mean(objective_values)),
        "evaluation_path_std_running_inventory_objective": float(
            np.std(objective_values)
        ),
        "n_evaluation_paths": int(objective_values.shape[0]),
        **diagnostics,
    }


def evaluation_regimes(args: argparse.Namespace) -> list[EvaluationRegime]:
    grid_definitions = [
        (
            "in_distribution",
            args.eval_in_distribution_sigma_values,
            args.eval_in_distribution_arrival_rate_values,
        ),
        (
            "stress",
            args.eval_stress_sigma_values,
            args.eval_stress_arrival_rate_values,
        ),
    ]
    return [
        EvaluationRegime(
            evaluation_set=evaluation_set,
            parameters=DomainParameters(
                sigma=sigma,
                arrival_rate=arrival_rate,
            ),
        )
        for evaluation_set, sigma_values, arrival_rate_values
        in grid_definitions
        for sigma in sigma_values
        for arrival_rate in arrival_rate_values
    ]


def add_gap_columns(rows: list[dict]) -> None:
    by_key = {
        (
            row["evaluation_set"],
            row["sigma"],
            row["arrival_rate"],
            row["policy"],
        ): row
        for row in rows
    }
    for row in rows:
        key = (
            row["evaluation_set"],
            row["sigma"],
            row["arrival_rate"],
        )
        domain_randomized = by_key.get((*key, "domain_randomized_ppo"))
        nominal = by_key.get((*key, "nominal_ppo"))
        periodic_rebalance = by_key.get((*key, "periodic_rebalance"))
        cash = by_key.get((*key, "cash"))
        row["domain_randomized_vs_nominal_mean_running_inventory_objective_gap"] = (
            domain_randomized["mean_running_inventory_objective"]
            - nominal["mean_running_inventory_objective"]
            if domain_randomized is not None and nominal is not None
            else ""
        )
        row[
            "domain_randomized_vs_periodic_rebalance_"
            "mean_running_inventory_objective_gap"
        ] = (
            domain_randomized["mean_running_inventory_objective"]
            - periodic_rebalance["mean_running_inventory_objective"]
            if domain_randomized is not None and periodic_rebalance is not None
            else ""
        )
        row["domain_randomized_vs_cash_mean_running_inventory_objective_gap"] = (
            domain_randomized["mean_running_inventory_objective"]
            - cash["mean_running_inventory_objective"]
            if domain_randomized is not None and cash is not None
            else ""
        )


def summarize_rows(rows: list[dict]) -> dict:
    policies = sorted({row["policy"] for row in rows})
    evaluation_sets = ["in_distribution", "stress"]
    return {
        policy: {
            evaluation_set: _summarize_policy_evaluation_set(
                rows,
                policy,
                evaluation_set,
            )
            for evaluation_set in evaluation_sets
        }
        for policy in policies
    }


def summarize_single_training_seed(
    rows: list[dict],
    training_seed: int,
    evaluation_seed: int,
) -> dict:
    return {
        "summary_scope": "single_training_seed",
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "policies": summarize_rows(rows),
    }


def _summarize_policy_evaluation_set(
    rows: list[dict],
    policy: str,
    evaluation_set: str,
) -> dict:
    values = [
        row["mean_running_inventory_objective"]
        for row in rows
        if row["policy"] == policy and row["evaluation_set"] == evaluation_set
    ]
    if not values:
        raise ValueError(
            f"no rows found for policy={policy!r}, evaluation_set={evaluation_set!r}"
        )
    return {
        "num_regimes": len(values),
        "mean_of_regime_mean_running_inventory_objective": float(np.mean(values)),
        "minimum_regime_mean_running_inventory_objective": float(np.min(values)),
        "maximum_regime_mean_running_inventory_objective": float(np.max(values)),
        "behavior_diagnostics_mean_across_regimes": {
            diagnostic: float(np.mean([
                row[diagnostic]
                for row in rows
                if row["policy"] == policy
                and row["evaluation_set"] == evaluation_set
            ]))
            for diagnostic in BEHAVIOR_DIAGNOSTIC_COLUMNS
        },
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
    validate_and_derive_decision_timing(args)
    resolve_train_domains_per_reset(args)
    validate_evaluation_configuration(args)
    args.behavior_diagnostics_schema_version = (
        BEHAVIOR_DIAGNOSTICS_SCHEMA_VERSION
    )
    run_dir = make_run_dir(args.output_dir, args.smoke_test)
    save_json(run_dir / "config.json", vars(args))

    domain_randomized_env = make_domain_randomized_env(args)
    domain_randomized_model, domain_randomized_vecnormalize = train_and_save(
        "domain_randomized", domain_randomized_env, args, run_dir
    )

    nominal_params = DomainParameters(
        sigma=args.nominal_sigma,
        arrival_rate=args.nominal_arrival_rate,
    )
    nominal_env = make_fixed_env(args, nominal_params, seed=args.seed + 20_000)
    nominal_model, nominal_vecnormalize = train_and_save(
        "nominal", nominal_env, args, run_dir
    )

    rows = []
    for regime_idx, regime in enumerate(evaluation_regimes(args)):
        params = regime.parameters
        regime_seed = evaluation_regime_seed(args, regime_idx)
        rows.append(
            evaluate_ppo(
                "domain_randomized_ppo",
                domain_randomized_model,
                domain_randomized_vecnormalize,
                params,
                regime.evaluation_set,
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
                regime.evaluation_set,
                args,
                regime_seed,
            )
        )
        rows.append(
            evaluate_periodic_rebalance(
                params,
                regime.evaluation_set,
                args,
                regime_seed,
            )
        )
        rows.append(evaluate_cash(params, regime.evaluation_set, args))

    add_gap_columns(rows)
    summary = summarize_single_training_seed(
        rows,
        training_seed=args.seed,
        evaluation_seed=args.evaluation_seed,
    )
    save_csv(run_dir / "evaluation_grid.csv", rows)
    save_json(run_dir / "summary.json", summary)

    print(f"Saved domain-randomized PPO run artifacts to: {run_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
