"""
Experiment helpers for SAiFE_gym RL training and evaluation.

AMM liquidity-provision setting:
  - Dict-based observations → StableBaselinesAMMEnvironment flattens them
  - Baseline comparator is UniformAllocationAgent (full-range LP)
  - Reward functions: PnL, ExponentialUtility, RunningInventoryPenalty
"""

import os
import sys
from copy import deepcopy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent
from SAiFE_gym.agents.SbAgent import SbAgent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from SAiFE_gym.rewards.RewardFunctions import PnL, RewardFunction
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel


# ---------------------------------------------------------------------------
# Default experiment parameters
# ---------------------------------------------------------------------------

TERMINAL_TIME = 1.0
N_STEPS = 200
INITIAL_PRICE = 100.0
VOLATILITY = 2.0
FEE_TIER = 0.003
NUM_TICKS = 7000
LIQUIDITY_SCALE = 1e6
INITIAL_WEALTH = 1e6
SEED = 42

# Arrival model coefficients (α shape: (4, 2) for [sell, buy])
#   α₀ = minimum floor, α₁ = baseline, α₂ = liquidity coeff, α₃ = arbitrage (toxicity)
ALPHA0 = np.array([10.0,  10.0])
ALPHA1 = np.array([100.0, 100.0])
ALPHA2 = np.array([0.0,   0.0])   # no liquidity-dependent component by default


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def get_amm_env(
    num_trajectories: int = 1,
    terminal_time: float = TERMINAL_TIME,
    n_steps: int = N_STEPS,
    tau: int = 5,
    volatility: float = VOLATILITY,
    arrival_rate: float = 100.0,
    alpha3: float = 0.0,
    gas_cost: float = 0.0,
    swap_fee_rate: float = 0.0,
    reward_function: RewardFunction = None,
    seed: int = SEED,
) -> AMMEnvironment:
    """Build an AMMEnvironment for LP training / evaluation.

    Args:
        num_trajectories: Parallel trajectories (acts as vectorised batch size).
        terminal_time:    Episode length.
        n_steps:          Number of discrete steps per episode.
        tau:              LP position half-width in ticks (action range ±tau).
        volatility:       Brownian motion volatility of the external mid-price.
        arrival_rate:     Baseline Poisson order arrival rate (α₁ for both sides).
        alpha3:           Arbitrage / toxicity coefficient (α₃). Higher values
                          mean informed traders exploit mispricing more aggressively.
        gas_cost:         Fixed cost per rebalance in token1 units (e.g. 10.0).
                          Applied only when the LP already holds a position.
        swap_fee_rate:    Fee rate on the imbalanced swap amount when rebalancing
                          (e.g. 0.001 = 0.1%). Cost = rate * W * |α_new - α_current|.
        reward_function:  Defaults to PnL.
        seed:             Random seed.

    Returns:
        Configured AMMEnvironment ready for reset / step.
    """
    step_size = terminal_time / n_steps

    alpha = np.array([
        ALPHA0,
        np.array([arrival_rate, arrival_rate]),
        ALPHA2,
        np.array([alpha3, alpha3]),
    ])

    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=volatility,
        initial_price=INITIAL_PRICE,
        terminal_time=terminal_time,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed,
    )
    arrival_model = PoissonLinearArrivalModel(
        alpha=alpha,
        liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=FEE_TIER,
        tau=tau,
        num_ticks=NUM_TICKS,
        exponential_value=1.0001,
        gas_cost=gas_cost,
        swap_fee_rate=swap_fee_rate,
        seed=seed + 2,
    )
    return AMMEnvironment(
        terminal_time=terminal_time,
        n_steps=n_steps,
        model_dynamics=model_dynamics,
        reward_function=reward_function or PnL(),
        initial_wealth=INITIAL_WEALTH,
        num_trajectories=num_trajectories,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# SB3 wrapping
# ---------------------------------------------------------------------------

def wrap_env(env: AMMEnvironment, normalise_obs: bool = True) -> VecMonitor:
    """Wrap AMMEnvironment for SB3 training.

    Pipeline: AMMEnvironment → StableBaselinesAMMEnvironment (Dict→flat obs)
              → VecMonitor (episode stats) → VecNormalize (running mean/std).

    VecNormalize is strongly recommended when features span very different
    scales (e.g. time ∈ [0,1] vs lp_liquidity ∈ [0, 2×10⁸]).  It maintains
    a running mean and std for each feature and clips at ±10σ, preventing
    large activations from dominating the first layer's gradients.

    Args:
        normalise_obs: If True (default), wrap with VecNormalize.
                       Disable only when loading a pre-trained model that
                       already carries its own normalisation stats.
    """
    vec = VecMonitor(StableBaselinesAMMEnvironment(env))
    if normalise_obs:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec


# ---------------------------------------------------------------------------
# PPO setup
# ---------------------------------------------------------------------------

def _validate_separate_eval_env(env: AMMEnvironment, eval_env: AMMEnvironment):
    """Reject shared simulator components before evaluation can mutate them."""
    components = [
        ("environment", env, eval_env),
        ("model_dynamics", env.model_dynamics, eval_env.model_dynamics),
        ("reward_function", env.reward_function, eval_env.reward_function),
    ]
    for name in (
        "midprice_model", "arrival_model", "price_impact_model", "fee_accounting_model"
    ):
        components.append((
            name, getattr(env.model_dynamics, name),
            getattr(eval_env.model_dynamics, name),
        ))
    for name, training_component, evaluation_component in components:
        if training_component is not None and training_component is evaluation_component:
            raise ValueError(
                f"Training and evaluation must not share {name}; "
                "construct an independent evaluation environment."
            )


def get_ppo_learner_and_callback(
    env: AMMEnvironment,
    tensorboard_base_logdir: str = None,
    best_model_path: str = "./best_models",
    tau: int = None,
    alpha3: float = None,
    normalise_obs: bool = True,
    learning_rate: float = 3e-4,
    eval_log_path: str = None,
    *,
    eval_env: AMMEnvironment = None,
    eval_seed: int = SEED + 10_000,
):
    """Build a PPO model and EvalCallback for the given environment.

    Evaluation uses a deep copy of the raw training environment by default,
    preserving its configuration without sharing simulator state or components.
    Pass a separately constructed ``eval_env`` for components that cannot be
    deep-copied or to use a different evaluation trajectory count. Its SB3
    observation and action spaces must match those of the training environment.
    ``eval_seed`` (default 10042) seeds the selected evaluation simulator once,
    including when ``eval_env`` is supplied; subsequent resets advance its own
    random streams. Training state and random streams are left untouched.

    Evaluation runs every ten rollouts: ``10 * env.n_steps`` callback calls,
    or ``10 * env.n_steps * env.num_trajectories`` training transitions.
    With normalization enabled, EvalCallback copies training statistics before
    evaluation; evaluation neither updates those statistics nor scales rewards.

    Hyperparameter guidance
    -----------------------
    n_steps:      One full episode per rollout (env.n_steps).  With gamma=1
                  the agent needs complete trajectories to estimate returns.

    batch_size:   Rollout buffer has n_steps * num_trajectories transitions.
                  We split it into 16 mini-batches per epoch so each update
                  uses a representative slice of the collected data.
                  Rule of thumb: aim for 8–16 mini-batches per epoch.

    n_epochs:     10 reuses per rollout is standard.  Reduce to 4–6 if you
                  observe policy loss exploding (use clip_range 0.2 as guard).

    learning_rate: Default 3e-4 works well when obs are normalised (VecNormalize).
                   Without normalisation use 1e-4 to avoid overshooting with
                   large-magnitude raw features (current_tick ~46k, lp_liquidity ~2e8).

    total_timesteps (caller's choice):
                  Scales roughly as O(obs_dim * log(obs_dim)) with state
                  dimension.  Practical guideline for this environment:
                    obs_dim=2  (mbt reduced)  →   500k –   2M
                    obs_dim=9  (SAiFE)        →    2M  –   5M   (with VecNormalize)
                    obs_dim=9  (no normalise) →    5M  –  20M

    Returns:
        (model, callback) — call model.learn(..., callback=callback) to train.
    """
    if eval_env is None:
        try:
            eval_env = deepcopy(env)
        except Exception as exc:
            raise ValueError(
                "Could not copy the training environment for evaluation; "
                "supply a separately constructed eval_env."
            ) from exc
    _validate_separate_eval_env(env, eval_env)

    training_vec = wrap_env(env, normalise_obs=normalise_obs)
    eval_vec = wrap_env(eval_env, normalise_obs=normalise_obs)
    if (
        training_vec.observation_space != eval_vec.observation_space
        or training_vec.action_space != eval_vec.action_space
    ):
        raise ValueError("Training and evaluation SB3 observation/action spaces must match.")
    eval_env.reset(seed=eval_seed)
    if isinstance(eval_vec, VecNormalize):
        eval_vec.training = False  # EvalCallback syncs copies of training statistics
        eval_vec.norm_reward = False

    tau = tau if tau is not None else env.model_dynamics.tau
    alpha3 = alpha3 if alpha3 is not None else 0.0
    experiment_str = get_experiment_string(env, tau=tau, alpha3=alpha3)

    rollout_size = env.n_steps * env.num_trajectories
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    ppo_params = dict(
        policy="MlpPolicy",
        env=training_vec,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(tensorboard_base_logdir, experiment_str) if tensorboard_base_logdir else None,
        learning_rate=learning_rate,
        n_epochs=10,
        batch_size=max(64, rollout_size // 16),   # 16 mini-batches per epoch
        normalize_advantage=True,
        n_steps=env.n_steps,
        gae_lambda=0.95,
        gamma=1.0,
    )
    callback_params = dict(
        eval_env=eval_vec,
        n_eval_episodes=10,
        best_model_save_path=os.path.join(best_model_path, experiment_str),
        log_path=eval_log_path,
        deterministic=True,
        eval_freq=env.n_steps * 10,  # callbacks count batch steps, not transitions
    )
    model = PPO(**ppo_params)
    callback = EvalCallback(**callback_params)
    return model, callback


def get_experiment_string(
    env: AMMEnvironment,
    tau: int = None,
    alpha3: float = None,
    gas_cost: float = None,
    swap_fee_rate: float = None,
) -> str:
    tau = tau if tau is not None else env.model_dynamics.tau
    alpha3 = alpha3 if alpha3 is not None else 0.0
    reward_name = type(env.reward_function).__name__
    s = (
        f"n_traj_{env.num_trajectories}"
        f"__tau_{tau}"
        f"__alpha3_{alpha3}"
        f"__vol_{env.model_dynamics.midprice_model.volatility}"
        f"__reward_{reward_name}"
    )
    if gas_cost is not None and gas_cost > 0:
        s += f"__gas_{gas_cost}"
    if swap_fee_rate is not None and swap_fee_rate > 0:
        s += f"__swapfee_{swap_fee_rate}"
    return s


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------

def run_episode(
    env: AMMEnvironment,
    agent,
    obs_transform=None,
    initial_wealth: float = INITIAL_WEALTH,
) -> np.ndarray:
    """Run one full episode and return final wealth per trajectory.

    Args:
        env:            Raw AMMEnvironment (not wrapped).
        agent:          Agent with a get_action(obs) method.
        obs_transform:  Optional callable applied to dict obs before get_action.
                        Pass StableBaselinesAMMEnvironment._flatten_obs for SbAgent.
        initial_wealth: Starting wealth used to compute final wealth from cumulative PnL.

    Returns:
        Final wealth per trajectory, shape (num_trajectories,).
    """
    obs, _ = env.reset()
    cumulative_reward = np.zeros(env.num_trajectories)

    for _ in range(env.n_steps):
        agent_obs = obs_transform(obs) if obs_transform else obs
        action = agent.get_action(agent_obs)
        obs, rewards, terminated, truncated, _ = env.step(action)
        cumulative_reward += rewards

    return initial_wealth + cumulative_reward


def compare_rl_vs_uniform(
    model: PPO,
    env: AMMEnvironment,
    sb3_env: StableBaselinesAMMEnvironment,
    n_eval_episodes: int = 10,
    initial_wealth: float = INITIAL_WEALTH,
    vec_normalize: VecNormalize = None,
) -> dict:
    """Evaluate a trained PPO model against UniformAllocationAgent.

    Both agents are evaluated on the same raw AMMEnvironment over multiple
    independent episodes so results are directly comparable.

    Args:
        model:          Trained PPO model.
        env:            Raw AMMEnvironment used for evaluation.
        sb3_env:        The StableBaselinesAMMEnvironment wrapping the same env
                        (used only to access the obs flattening function).
        n_eval_episodes: Number of episodes to average over.
        initial_wealth: Starting wealth for final wealth calculation.
        vec_normalize:  Optional VecNormalize wrapper from training. When provided,
                        its running statistics (mean/std) are applied to flatten
                        observations before feeding them to the RL agent, matching
                        the normalization the model was trained with.

    Returns:
        dict with keys 'rl' and 'uniform', each containing:
            'final_wealths': array of shape (n_eval_episodes * num_trajectories,)
            'mean': scalar mean final wealth
            'std':  scalar std of final wealth
    """
    rl_agent = SbAgent(model, num_trajectories=env.num_trajectories)
    uniform_agent = UniformAllocationAgent(env)
    flatten = sb3_env._flatten_obs

    # Build RL obs transform: flatten → normalize (if VecNormalize stats available)
    if vec_normalize is not None:
        vec_normalize.training = False  # don't update running stats during eval
        rl_obs_transform = lambda obs: vec_normalize.normalize_obs(flatten(obs))
    else:
        rl_obs_transform = flatten

    results = {}
    for name, agent, transform in [
        ("rl",      rl_agent,      rl_obs_transform),
        ("uniform", uniform_agent, None),
    ]:
        all_wealths = []
        for _ in range(n_eval_episodes):
            final_w = run_episode(env, agent, obs_transform=transform,
                                  initial_wealth=initial_wealth)
            all_wealths.append(final_w)

        all_wealths = np.concatenate(all_wealths)
        results[name] = {
            "final_wealths": all_wealths,
            "mean": float(np.mean(all_wealths)),
            "std":  float(np.std(all_wealths)),
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

AGENT_COLORS = {"rl": "#2ca02c", "uniform": "#1f77b4"}
AGENT_LABELS = {"rl": "PPO (RL)", "uniform": "Uniform"}


def plot_wealth_distributions(
    results: dict,
    title: str = "Final wealth: RL vs Uniform",
    save_figure: bool = False,
    figures_dir: str = "./figures",
    filename: str = "rl_vs_uniform_wealth.png",
):
    """Box + strip chart comparing final-wealth distributions of both agents."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13)

    # Left: box plot
    ax = axes[0]
    data = [results[k]["final_wealths"] - INITIAL_WEALTH for k in ("rl", "uniform")]
    labels = [AGENT_LABELS[k] for k in ("rl", "uniform")]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, key in zip(bp["boxes"], ("rl", "uniform")):
        patch.set_facecolor(AGENT_COLORS[key])
        patch.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Break-even")
    ax.set_ylabel("ΔWealth (final − initial)")
    ax.set_title("Distribution of ΔWealth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Right: mean ± std bar chart
    ax = axes[1]
    keys = ("rl", "uniform")
    x = np.arange(len(keys))
    means = [results[k]["mean"] - INITIAL_WEALTH for k in keys]
    stds  = [results[k]["std"]  for k in keys]
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=[AGENT_COLORS[k] for k in keys],
                  alpha=0.7, error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[k] for k in keys])
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Mean ΔWealth ± std")
    ax.set_title("Mean ± Std of ΔWealth")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_figure:
        os.makedirs(figures_dir, exist_ok=True)
        path = os.path.join(figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()

    return fig


def plot_training_curve_vs_uniform(
    reward_history: list,
    uniform_mean_wealth: float,
    title: str = "PPO training curve vs Uniform baseline",
    save_figure: bool = False,
    figures_dir: str = "./figures",
    filename: str = "training_curve.png",
):
    """Plot mean episode reward over training against the Uniform agent benchmark.

    Args:
        reward_history:      List of mean episodic rewards recorded during training
                             (e.g. from a custom callback or Monitor logs).
        uniform_mean_wealth: Mean final wealth of UniformAllocationAgent (used as
                             a horizontal reference line).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(reward_history, color=AGENT_COLORS["rl"], linewidth=1.5, label="PPO (RL)")
    ax.axhline(
        uniform_mean_wealth - INITIAL_WEALTH,
        color=AGENT_COLORS["uniform"], linestyle="--", linewidth=1.5,
        label=f"Uniform baseline (mean ΔW = {uniform_mean_wealth - INITIAL_WEALTH:+.0f})",
    )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean ΔWealth")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_figure:
        os.makedirs(figures_dir, exist_ok=True)
        path = os.path.join(figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()

    return fig


def plot_convergence_curve(
    log_path: str,
    uniform_baseline_reward: float = None,
    title: str = "PPO convergence: mean eval reward vs timesteps",
    save_figure: bool = False,
    figures_dir: str = "./figures",
    filename: str = "convergence_curve.png",
) -> plt.Figure:
    """Load EvalCallback's evaluations.npz and plot convergence.

    Args:
        log_path:                Directory containing evaluations.npz
                                 (same value passed as eval_log_path in
                                 get_ppo_learner_and_callback).
        uniform_baseline_reward: Mean episode reward of UniformAllocationAgent
                                 (ΔWealth, not absolute wealth). If provided,
                                 drawn as a dashed reference line.
    """
    data = np.load(os.path.join(log_path, "evaluations.npz"))
    timesteps = data["timesteps"]               # (n_checkpoints,)
    results   = data["results"]                 # (n_checkpoints, n_eval_episodes)
    mean_rewards = results.mean(axis=1)
    std_rewards  = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(timesteps, mean_rewards,
            color=AGENT_COLORS["rl"], linewidth=1.5, label=AGENT_LABELS["rl"])
    ax.fill_between(timesteps,
                    mean_rewards - std_rewards,
                    mean_rewards + std_rewards,
                    alpha=0.2, color=AGENT_COLORS["rl"])
    if uniform_baseline_reward is not None:
        ax.axhline(uniform_baseline_reward,
                   color=AGENT_COLORS["uniform"], linestyle="--", linewidth=1.5,
                   label=f"{AGENT_LABELS['uniform']} ({uniform_baseline_reward:+.0f})")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_figure:
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(os.path.join(figures_dir, filename), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


def create_policy_behavior_plot(
    model: PPO,
    vec_normalize: VecNormalize,
    tau: int,
    n_points: int = 101,
    mispricing_range: tuple = (-0.05, 0.05),
    fixed_lp_liquidity: float = 2e8,
    fixed_time: float = 0.5,
    gas_cost: float = 0.0,
    title: str = "Policy behavior vs mispricing",
    save_figure: bool = False,
    figures_dir: str = "./figures",
    filename: str = "policy_behavior.png",
) -> plt.Figure:
    """Plot tick offsets chosen by RL vs Uniform agent as mispricing varies.

    Analogous to mbt's create_inventory_plot(): sweeps mispricing on the
    x-axis (the LP's primary adverse-selection signal) while holding all
    other observations at neutral values.

    A converged policy should show a structured response to mispricing
    (e.g. narrowing the range or shifting it defensively when mispricing
    is large), unlike the flat Uniform baseline.

    Obs column order (DEFAULT_OBS_KEYS):
        0: MISPRICING_KEY       ← swept
        1: LP_LOWER_OFFSET_KEY
        2: LP_UPPER_OFFSET_KEY
        3: BOUNDARY_PROXIMITY_KEY
        4: POSITION_WIDTH_KEY
        5: TIME_KEY
        6: HAS_POSITION_KEY
        7: PORTFOLIO_VALUE_RATIO_KEY
        8: UNCLAIMED_FEE_VALUE_RATIO_KEY
    """
    mispricings = np.linspace(mispricing_range[0], mispricing_range[1], n_points)

    raw_obs = np.column_stack([
        mispricings,
        np.full(n_points, float(tau)),
        np.full(n_points, float(tau)),
        np.full(n_points, float(tau)),
        np.full(n_points, float(2 * tau)),
        np.full(n_points, fixed_time),
        np.ones(n_points),
        np.ones(n_points),
        np.zeros(n_points),
    ]).astype(np.float32)

    old_training = vec_normalize.training
    vec_normalize.training = False
    norm_obs = vec_normalize.normalize_obs(raw_obs)
    vec_normalize.training = old_training

    rl_actions, _ = model.predict(norm_obs, deterministic=True)  # (n_points, 2)
    rl_lower, rl_upper = rl_actions[:, 0], rl_actions[:, 1]

    fig, (ax_lo, ax_hi) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    for ax, rl_vals, uniform_val, ylabel, sub_title in [
        (ax_lo, rl_lower, -tau, "lower_offset", "Lower bound action"),
        (ax_hi, rl_upper,  tau, "upper_offset", "Upper bound action"),
    ]:
        ax.plot(mispricings, rl_vals,
                color=AGENT_COLORS["rl"], linewidth=1.5, label=AGENT_LABELS["rl"])
        ax.axhline(uniform_val,
                   color=AGENT_COLORS["uniform"], linestyle="--", linewidth=1.5,
                   label=AGENT_LABELS["uniform"])
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Relative mispricing ((asset_price - AMM_price) / asset_price)")
        ax.set_ylabel(ylabel)
        ax.set_title(sub_title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_figure:
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(os.path.join(figures_dir, filename), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


def create_time_behavior_plot(
    model: PPO,
    vec_normalize: VecNormalize,
    tau: int,
    mispricing_levels: list = None,
    n_time_points: int = 101,
    fixed_lp_liquidity: float = 2e8,
    gas_cost: float = 0.0,
    title: str = "Policy behavior vs time",
    save_figure: bool = False,
    figures_dir: str = "./figures",
    filename: str = "policy_time_behavior.png",
) -> plt.Figure:
    """Plot how tick offsets evolve over the episode at fixed mispricing levels.

    Analogous to mbt's create_time_plot(). Multiple lines (one per mispricing
    level) show whether the policy adapts its range width over time.
    """
    if mispricing_levels is None:
        mispricing_levels = [-0.05, 0.0, 0.05]
    time_vals  = np.linspace(0.0, TERMINAL_TIME, n_time_points)
    n_levels   = len(mispricing_levels)
    n_total    = n_levels * n_time_points

    mispricing_rep = np.repeat(mispricing_levels, n_time_points).astype(np.float32)
    time_tiled     = np.tile(time_vals, n_levels).astype(np.float32)
    raw_obs = np.column_stack([
        mispricing_rep,
        np.full(n_total, float(tau)),
        np.full(n_total, float(tau)),
        np.full(n_total, float(tau)),
        np.full(n_total, float(2 * tau)),
        time_tiled,
        np.ones(n_total),
        np.ones(n_total),
        np.zeros(n_total),
    ]).astype(np.float32)

    old_training = vec_normalize.training
    vec_normalize.training = False
    norm_obs = vec_normalize.normalize_obs(raw_obs)
    vec_normalize.training = old_training

    rl_actions, _ = model.predict(norm_obs, deterministic=True)       # (n_total, 2)
    rl_actions = rl_actions.reshape(n_levels, n_time_points, 2)       # (levels, time, 2)

    level_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_levels))
    fig, (ax_lo, ax_hi) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title)

    for i, (mp, color) in enumerate(zip(mispricing_levels, level_colors)):
        label = f"mispricing={mp:+.1f}"
        ax_lo.plot(time_vals, rl_actions[i, :, 0], color=color, linewidth=1.5, label=label)
        ax_hi.plot(time_vals, rl_actions[i, :, 1], color=color, linewidth=1.5, label=label)

    for ax, uniform_val, ylabel, sub_title in [
        (ax_lo, -tau, "lower_offset", "Lower bound action vs time"),
        (ax_hi,  tau, "upper_offset", "Upper bound action vs time"),
    ]:
        ax.axhline(uniform_val,
                   color=AGENT_COLORS["uniform"], linestyle="--", linewidth=1.2,
                   label=AGENT_LABELS["uniform"])
        ax.set_xlabel("Time t / T")
        ax.set_ylabel(ylabel)
        ax.set_title(sub_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_figure:
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(os.path.join(figures_dir, filename), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


def print_comparison_table(results: dict, tau: int, alpha3: float):
    """Print a summary table of RL vs Uniform results."""
    header = f"{'Agent':<12} | {'Mean ΔWealth':>15} | {'Std':>12} | {'Mean Wealth':>15}"
    sep = "-" * len(header)
    print()
    print(f"tau = {tau},  alpha3 = {alpha3}")
    print(sep)
    print(header)
    print(sep)
    for key in ("rl", "uniform"):
        r = results[key]
        delta = r["mean"] - INITIAL_WEALTH
        print(f"{AGENT_LABELS[key]:<12} | {delta:>+15,.1f} | {r['std']:>12,.0f} | {r['mean']:>15,.1f}")
    print(sep)
