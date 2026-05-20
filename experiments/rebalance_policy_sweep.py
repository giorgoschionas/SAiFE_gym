"""
Rebalance Policy Sweep — find the cost regime where PPO learns to hold.

Two studies:
  1. gas_cost sweep  (swap_fee_rate=0)
  2. swap_fee_rate sweep (gas_cost=0.7, the default)

For each config: train PPO, evaluate on 1000 trajectories recording per-step
hold decisions, and compare against Uniform baseline.

Outputs (experiments/figures/):
  - hold_frequency_vs_cost.png
  - pnl_comparison.png
  - hold_frequency_over_time.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent
from SAiFE_gym.agents.SbAgent import SbAgent
from SAiFE_gym.rewards.RewardFunctions import PnL

# ============================================================================
# Configuration — matches agent_comparison.py where always-rebalance was observed
# ============================================================================

SEED = 123
TERMINAL_TIME = 1.0
N_STEPS = 200
NUM_TRAJECTORIES_TRAIN = 200
NUM_TRAJECTORIES_EVAL = 1000
INITIAL_WEALTH = 1000
TAU = 100
LIQUIDITY_SCALE = 1e4

INITIAL_PRICE = 100.0
DRIFT = 0
VOLATILITY = 0.01
FEE_TIER = 0.003
EXP_VALUE = 1.0001

ALPHA0 = np.array([10.0, 10.0])
ALPHA1 = np.array([150.0, 150.0])
ALPHA2 = np.array([0.0, 0.0])
ALPHA3 = np.array([5000.0, 5000.0])

# Training budget per config (keep low for fast sweeps; increase for publication)
PPO_TOTAL_TIMESTEPS = 1 * NUM_TRAJECTORIES_TRAIN * N_STEPS  # 1 epoch

# Sweep dimensions
GAS_COST_SWEEP = [0, 1, 5, 10, 25, 50, 100]
SWAP_FEE_RATE_SWEEP = [0, 0.001, 0.003, 0.005, 0.01]

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')


# ============================================================================
# Environment factory
# ============================================================================

def create_environment(
    num_trajectories: int,
    gas_cost: float = 0.7,
    swap_fee_rate: float = 0.0,
    seed: int = None,
) -> AMMEnvironment:
    step_size = TERMINAL_TIME / N_STEPS
    alpha = np.array([ALPHA0, ALPHA1, ALPHA2, ALPHA3])

    midprice_model = BrownianMotionMidpriceModel(
        drift=DRIFT, volatility=VOLATILITY, initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME, step_size=step_size,
        num_trajectories=num_trajectories, seed=seed,
    )
    arrival_model = PoissonLinearArrivalModel(
        alpha=alpha, liquidity_scale=LIQUIDITY_SCALE, step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1 if seed else None,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model, arrival_model=arrival_model,
        num_trajectories=num_trajectories, fee_tier=FEE_TIER, tau=TAU,
        num_ticks=3000, exponential_value=EXP_VALUE,
        initial_wealth=INITIAL_WEALTH,
        gas_cost=gas_cost,
        swap_fee_rate=swap_fee_rate,
        seed=seed + 2 if seed else None,
    )
    reward_function = PnL(exponential_value=EXP_VALUE, initial_wealth=INITIAL_WEALTH)
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS,
        reward_function=reward_function, model_dynamics=model_dynamics,
        num_trajectories=num_trajectories, seed=seed,
    )


# ============================================================================
# Callback — episode reward tracking (from agent_comparison.py)
# ============================================================================

class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.epoch_rewards = []
        self._episode_reward_sum = 0.0
        self._episode_steps = 0

    def _on_step(self):
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is not None:
            self._episode_reward_sum += float(np.mean(rewards))
            self._episode_steps += 1
        if dones is not None and np.all(dones):
            mean_reward = self._episode_reward_sum / max(self._episode_steps, 1)
            self.epoch_rewards.append(mean_reward)
            self._episode_reward_sum = 0.0
            self._episode_steps = 0
        return True


# ============================================================================
# Evaluation with hold tracking
# ============================================================================

def evaluate_with_hold_tracking(env, get_action_fn):
    """Run one full episode, return PnL and per-step hold decisions.

    Returns:
        pnl:        (num_trajectories,) cumulative PnL per trajectory
        hold_flags: (n_steps, num_trajectories) hold_flag value at each step
    """
    state, _ = env.reset()
    rewards_list = []
    hold_flags_list = []
    terminated = np.zeros(env.num_trajectories, dtype=bool)

    while not np.any(terminated):
        action = get_action_fn(state)
        # Record hold_flag (3rd action dimension): > 0 means hold
        if action.shape[1] >= 3:
            hold_flags_list.append(action[:, 2].copy())
        else:
            hold_flags_list.append(np.full(env.num_trajectories, -1.0))
        state, reward, terminated, _, _ = env.step(action)
        rewards_list.append(reward)

    pnl = np.sum(np.array(rewards_list), axis=0)
    hold_flags = np.array(hold_flags_list)  # (n_steps, num_trajectories)
    return pnl, hold_flags


# ============================================================================
# Per-config pipeline
# ============================================================================

def run_config(gas_cost: float, swap_fee_rate: float, label: str) -> dict:
    """Train PPO and evaluate for a single (gas_cost, swap_fee_rate) config."""
    print(f"\n{'='*60}")
    print(f"Config: {label}  (gas_cost={gas_cost}, swap_fee_rate={swap_fee_rate})")
    print(f"{'='*60}")

    # --- Train ---
    print("  Training PPO...")
    train_env = create_environment(NUM_TRAJECTORIES_TRAIN, gas_cost, swap_fee_rate, seed=SEED)
    sb_train = StableBaselinesAMMEnvironment(train_env)
    ppo_model = PPO(
        "MlpPolicy", sb_train,
        learning_rate=3e-4, n_steps=N_STEPS, batch_size=64,
        n_epochs=10, gamma=1.0, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=SEED,
    )
    cb = EpisodeRewardCallback()
    ppo_model.learn(total_timesteps=PPO_TOTAL_TIMESTEPS, callback=cb)
    print(f"  Training done ({len(cb.epoch_rewards)} episodes)")

    # --- Evaluate PPO ---
    print("  Evaluating PPO...")
    eval_seed = SEED + 9999
    eval_env = create_environment(NUM_TRAJECTORIES_EVAL, gas_cost, swap_fee_rate, seed=eval_seed)
    sb_eval = StableBaselinesAMMEnvironment(eval_env)
    ppo_agent = SbAgent(ppo_model, num_trajectories=NUM_TRAJECTORIES_EVAL)

    ppo_pnl, ppo_hold_flags = evaluate_with_hold_tracking(
        eval_env,
        lambda s, _sb=sb_eval, _a=ppo_agent: _a.get_action(_sb._flatten_obs(s)),
    )

    # --- Evaluate Uniform baseline ---
    print("  Evaluating Uniform baseline...")
    eval_env_uni = create_environment(NUM_TRAJECTORIES_EVAL, gas_cost, swap_fee_rate, seed=eval_seed)
    uniform_agent = UniformAllocationAgent(eval_env_uni)
    uni_pnl, _ = evaluate_with_hold_tracking(
        eval_env_uni,
        uniform_agent.get_action,
    )

    # --- Compute metrics ---
    # hold_flag > 0 means "hold" (don't rebalance)
    hold_pct = 100.0 * np.mean(ppo_hold_flags > 0)
    # Per-step hold rate: fraction of trajectories holding at each step
    per_step_hold_rate = np.mean(ppo_hold_flags > 0, axis=1)  # (n_steps,)

    result = {
        'gas_cost': gas_cost,
        'swap_fee_rate': swap_fee_rate,
        'label': label,
        'hold_pct': hold_pct,
        'ppo_mean_pnl': float(np.mean(ppo_pnl)),
        'ppo_std_pnl': float(np.std(ppo_pnl)),
        'uniform_mean_pnl': float(np.mean(uni_pnl)),
        'uniform_std_pnl': float(np.std(uni_pnl)),
        'ppo_win_pct': 100.0 * np.mean(ppo_pnl > 0),
        'per_step_hold_rate': per_step_hold_rate,
        'training_rewards': cb.epoch_rewards,
    }

    print(f"  Hold%: {hold_pct:.1f}%  |  PPO PnL: {result['ppo_mean_pnl']:+.1f} ± {result['ppo_std_pnl']:.1f}"
          f"  |  Uniform PnL: {result['uniform_mean_pnl']:+.1f} ± {result['uniform_std_pnl']:.1f}")

    return result


# ============================================================================
# Plotting
# ============================================================================

def plot_hold_frequency_vs_cost(results_list, sweep_key, xlabel, filename):
    """Plot hold frequency vs cost parameter — the key transition plot."""
    fig, ax = plt.subplots(figsize=(9, 5))

    x_vals = [r[sweep_key] for r in results_list]
    hold_pcts = [r['hold_pct'] for r in results_list]

    ax.plot(x_vals, hold_pcts, 'o-', color='#d62728', linewidth=2, markersize=8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Hold Frequency (%)', fontsize=12)
    ax.set_title('PPO Hold Frequency vs Rebalancing Cost', fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Annotate each point
    for x, y in zip(x_vals, hold_pcts):
        ax.annotate(f'{y:.0f}%', (x, y), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pnl_comparison(results_list, sweep_key, xlabel, filename):
    """PPO mean PnL vs Uniform mean PnL with error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x_vals = np.array([r[sweep_key] for r in results_list])
    ppo_means = np.array([r['ppo_mean_pnl'] for r in results_list])
    ppo_stds = np.array([r['ppo_std_pnl'] for r in results_list])
    uni_means = np.array([r['uniform_mean_pnl'] for r in results_list])
    uni_stds = np.array([r['uniform_std_pnl'] for r in results_list])

    width = (x_vals[-1] - x_vals[0]) / (len(x_vals) * 4) if len(x_vals) > 1 else 0.5
    x_idx = np.arange(len(x_vals))

    ax.bar(x_idx - width/2, ppo_means, width, yerr=ppo_stds, capsize=4,
           color='#d62728', alpha=0.7, label='PPO')
    ax.bar(x_idx + width/2, uni_means, width, yerr=uni_stds, capsize=4,
           color='#1f77b4', alpha=0.7, label='Uniform')

    ax.set_xticks(x_idx)
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Mean PnL', fontsize=12)
    ax.set_title('PnL: PPO vs Uniform at Different Cost Levels', fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_hold_frequency_over_time(results_list, sweep_key, filename):
    """For selected configs, show how hold frequency varies across timesteps."""
    # Pick a few representative configs (first, middle, last)
    n = len(results_list)
    if n <= 3:
        indices = list(range(n))
    else:
        indices = [0, n // 2, n - 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(indices)))

    for color, idx in zip(colors, indices):
        r = results_list[idx]
        hold_rate = r['per_step_hold_rate']
        # Smooth with a rolling window
        window = max(1, len(hold_rate) // 20)
        smoothed = np.convolve(hold_rate, np.ones(window)/window, mode='valid')
        time_axis = np.linspace(0, TERMINAL_TIME, len(smoothed))

        ax.plot(time_axis, 100 * smoothed, color=color, linewidth=2,
                label=f"{sweep_key}={r[sweep_key]}")

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Hold Frequency (%)', fontsize=12)
    ax.set_title('Hold Frequency Over Episode (Selected Configs)', fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_table(results_list, study_name):
    """Print a summary table of results."""
    header = (f"{'Config':<20} | {'Hold%':>6} | {'PPO PnL':>12} | {'Uni PnL':>12} "
              f"| {'PPO Win%':>8}")
    sep = '-' * len(header)
    print(f"\n{study_name}")
    print(sep)
    print(header)
    print(sep)
    for r in results_list:
        print(f"{r['label']:<20} | {r['hold_pct']:>5.1f}% "
              f"| {r['ppo_mean_pnl']:>+8.1f}±{r['ppo_std_pnl']:<4.0f}"
              f"| {r['uniform_mean_pnl']:>+8.1f}±{r['uniform_std_pnl']:<4.0f}"
              f"| {r['ppo_win_pct']:>7.1f}%")
    print(sep)


# ============================================================================
# Main
# ============================================================================

def main():
    np.random.seed(SEED)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Study 1: gas_cost sweep (swap_fee_rate = 0)
    # ------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# Study 1: gas_cost sweep (swap_fee_rate = 0)")
    print("#" * 60)

    gas_results = []
    for gc in GAS_COST_SWEEP:
        result = run_config(gas_cost=gc, swap_fee_rate=0.0, label=f"gas={gc}")
        gas_results.append(result)

    print_summary_table(gas_results, "Study 1: Gas Cost Sweep")

    print("\nGenerating Study 1 plots...")
    plot_hold_frequency_vs_cost(
        gas_results, 'gas_cost', 'Gas Cost (token1 units)',
        'hold_frequency_vs_gas_cost.png',
    )
    plot_pnl_comparison(
        gas_results, 'gas_cost', 'Gas Cost',
        'pnl_comparison_gas_cost.png',
    )
    plot_hold_frequency_over_time(
        gas_results, 'gas_cost', 'hold_over_time_gas_cost.png',
    )

    # ------------------------------------------------------------------
    # Study 2: swap_fee_rate sweep (gas_cost = 0.7)
    # ------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# Study 2: swap_fee_rate sweep (gas_cost = 0.7)")
    print("#" * 60)

    swap_results = []
    for sfr in SWAP_FEE_RATE_SWEEP:
        result = run_config(gas_cost=0.7, swap_fee_rate=sfr, label=f"swap={sfr}")
        swap_results.append(result)

    print_summary_table(swap_results, "Study 2: Swap Fee Rate Sweep")

    print("\nGenerating Study 2 plots...")
    plot_hold_frequency_vs_cost(
        swap_results, 'swap_fee_rate', 'Swap Fee Rate',
        'hold_frequency_vs_swap_fee_rate.png',
    )
    plot_pnl_comparison(
        swap_results, 'swap_fee_rate', 'Swap Fee Rate',
        'pnl_comparison_swap_fee_rate.png',
    )
    plot_hold_frequency_over_time(
        swap_results, 'swap_fee_rate', 'hold_over_time_swap_fee_rate.png',
    )

    print("\nDone! All figures saved to:", FIGURES_DIR)


if __name__ == '__main__':
    main()
