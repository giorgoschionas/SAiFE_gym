"""
Holistic comparison of LP agents:
  Uniform, DeployOnce, Cartea, REINFORCE (PolicyGradient), PPO, SAC, DQN.

Phases:
  1. Train enabled RL agents
  2. Evaluate enabled agents on 1000 trajectories → PnL distribution plot
  3. Run 1 trajectory per agent → price evolution, PnL evolution, position offsets

Each plot is saved individually to figures/.
Toggle agents on/off via the ENABLE_AGENTS dict.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent, DeployOnceAgent, CarteaPLAgent
from SAiFE_gym.agents.PolicyGradientAgent import PolicyGradientAgent
from SAiFE_gym.agents.SbAgent import SbAgent
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
)

# ============================================================================
# Configuration
# ============================================================================

SEED = 123
TERMINAL_TIME = 1.0
N_STEPS = 200
NUM_TRAJECTORIES_TRAIN = 200
NUM_TRAJECTORIES_EVAL = 1000
INITIAL_WEALTH = 1000
TAU = 20
LIQUIDITY_SCALE = 1e4

INITIAL_PRICE = 100.0
DRIFT = 0
VOLATILITY = 0.1
FEE_TIER = 0.003
EXP_VALUE = 1.0001

ALPHA0 = np.array([10.0, 10.0])
ALPHA1 = np.array([150.0, 150.0])
ALPHA2 = np.array([0.0, 0.0])
ALPHA3 = np.array([5000.0, 5000.0])

GAMMA_CARTEA = 0.000005

REINFORCE_EPOCHS = 250
REINFORCE_LR = 2e-4
ACTION_STD_INIT = 1.7

SB3_TOTAL_TIMESTEPS = REINFORCE_EPOCHS * NUM_TRAJECTORIES_TRAIN * N_STEPS

DQN_TICK_STRIDE = 10  # Discretization stride for tick offsets
NUM_SINGLE_SIMS = 10   # Number of single-trajectory simulations to plot

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# ── Agent toggles (set to False to skip training / evaluation) ──────
ENABLE_AGENTS = {
    'Uniform':    True,
    'DeployOnce': True,
    'Cartea':     True,
    'REINFORCE':  False,
    'PPO':        True,
    'SAC':        False,
    'DQN':        False,
}

AGENT_NAMES = [name for name, on in ENABLE_AGENTS.items() if on]
AGENT_COLORS = {
    'Uniform':    '#1f77b4',
    'DeployOnce': '#9467bd',
    'Cartea':     '#ff7f0e',
    'REINFORCE':  '#2ca02c',
    'PPO':        '#d62728',
    'SAC':        '#17becf',
    'DQN':        '#8c564b',
}

# ============================================================================
# Environment Factory
# ============================================================================

def create_environment(num_trajectories: int, seed: int = None):
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
        seed=seed + 2 if seed else None,
    )
    reward_function = PnL(exponential_value=EXP_VALUE, initial_wealth=INITIAL_WEALTH)
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS,
        reward_function=reward_function, model_dynamics=model_dynamics,
        num_trajectories=num_trajectories, seed=seed,
    )

# ============================================================================
# DQN Discrete-Action Wrapper
# ============================================================================

def build_discrete_action_table(tau: int, tick_stride: int) -> np.ndarray:
    """Build lookup table mapping discrete index → [lower, upper, hold_flag].

    Index 0 is the HOLD action (hold_flag=+1, offsets ignored).
    Indices 1..N are REBALANCE actions at every valid (lower, upper) pair
    sampled at the given tick stride.
    """
    # Action 0: HOLD (offsets are arbitrary — ignored when hold_flag > 0)
    actions = [[0.0, 1.0, 1.0]]
    # Remaining actions: rebalance to each valid (lower, upper) pair
    offsets = np.arange(-tau, tau + 1, tick_stride, dtype=np.float32)
    for i, lo in enumerate(offsets):
        for hi in offsets[i + 1:]:
            actions.append([lo, hi, -1.0])
    return np.array(actions, dtype=np.float32)


class DiscreteActionVecEnv(VecEnv):
    """Wraps StableBaselinesAMMEnvironment with a Discrete action space for DQN.

    Discretizes the continuous (lower_offset, upper_offset) tick offsets into a
    finite set of valid pairs at a configurable stride, plus a dedicated HOLD
    action (index 0).
    """

    def __init__(self, sb_env: StableBaselinesAMMEnvironment, tau: int,
                 tick_stride: int = DQN_TICK_STRIDE):
        self._wrapped = sb_env
        self.action_table = build_discrete_action_table(tau, tick_stride)
        act_space = gymnasium.spaces.Discrete(len(self.action_table))
        super().__init__(sb_env.num_envs, sb_env.observation_space, act_space)

    def reset(self):
        return self._wrapped.reset()

    def step_async(self, actions):
        continuous = self.action_table[actions]
        self._wrapped.step_async(continuous)

    def step_wait(self):
        return self._wrapped.step_wait()

    def close(self):
        self._wrapped.close()

    def get_attr(self, attr_name, indices=None):
        return self._wrapped.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        self._wrapped.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return self._wrapped.env_method(method_name, *args, indices=indices, **kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self._wrapped.env_is_wrapped(wrapper_class, indices)

    def seed(self, seed=None):
        return self._wrapped.seed(seed)

    def get_images(self):
        return self._wrapped.get_images()

# ============================================================================
# REINFORCE Policy Network
# ============================================================================

class SimpleStraddlePolicy(nn.Module):
    """Squashed Gaussian policy for REINFORCE.

    forward() returns raw unbounded means (3 dims).  Noise is added in this
    raw space by the agent, then transform() squashes samples into valid
    [lower_offset, upper_offset, hold_flag] actions via center/half_width
    parameterization.  log_prob_correction() provides the Jacobian term so
    the policy gradient accounts for the squashing.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, tau: int = TAU):
        super().__init__()
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x):
        return self.net(x)  # raw unbounded means

    def transform(self, raw):
        """Map raw unbounded samples → valid [lower, upper, hold_flag]."""
        center = self.tau * torch.tanh(raw[:, 0])
        half_width = 1.0 + (self.tau - 1.0) * torch.sigmoid(raw[:, 1])
        lower = torch.clamp(center - half_width, min=-self.tau)
        upper = torch.clamp(center + half_width, max=self.tau)
        hold_flag = torch.tanh(raw[:, 2])
        return torch.stack([lower, upper, hold_flag], dim=1)

    def log_prob_correction(self, raw):
        """Log |det Jacobian| of the squashing (up to additive constants)."""
        # tanh corrections for center (dim 0) and hold_flag (dim 2)
        log_jac_0 = torch.log(1 - torch.tanh(raw[:, 0]) ** 2 + 1e-6)
        log_jac_2 = torch.log(1 - torch.tanh(raw[:, 2]) ** 2 + 1e-6)
        # sigmoid correction for half_width (dim 1)
        s = torch.sigmoid(raw[:, 1])
        log_jac_1 = torch.log(s * (1 - s) + 1e-6)
        return log_jac_0 + log_jac_1 + log_jac_2

# ============================================================================
# SB3 Reward Loggers
# ============================================================================

class EpisodeRewardCallback(BaseCallback):
    """Logs mean per-step reward at each episode boundary.

    Works for both on-policy (PPO) and off-policy (SAC) algorithms.
    Accumulates mean-across-trajectories reward at each step, then logs the
    average per-step reward when the episode ends (all dones True).
    This matches REINFORCE's ``np.mean(rewards)`` metric for fair comparison.
    """
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
# Evaluation Helpers
# ============================================================================

def evaluate_on_trajectories(env, get_action_fn):
    """Run one full episode; return per-trajectory cumulative PnL."""
    state, _ = env.reset()
    rewards_list = []
    terminated = np.zeros(env.num_trajectories, dtype=bool)
    while not np.any(terminated):
        action = get_action_fn(state)
        state, reward, terminated, _, _ = env.step(action)
        rewards_list.append(reward)
    return np.sum(np.array(rewards_list), axis=0)


def collect_single_trajectory(env, get_action_fn):
    """Run one episode (num_trajectories=1) and return per-step data dict."""
    assert env.num_trajectories == 1
    data = {k: [] for k in [
        'time', 'pool_price', 'midprice',
        'position_lower_price', 'position_upper_price',
        'action_lower', 'action_upper',
        'actual_lower_offset', 'actual_upper_offset',
        'hold_flag',
        'reward', 'cumulative_pnl',
    ]}
    state, _ = env.reset()
    cum_pnl = 0.0

    for _ in range(env.n_steps):
        data['time'].append(state[TIME_KEY][0])
        data['pool_price'].append(state[POOL_SQRT_PRICE_KEY][0] ** 2)
        data['midprice'].append(state[ASSET_PRICE_KEY][0])

        action = get_action_fn(state)
        data['action_lower'].append(float(action[0, 0]))
        data['action_upper'].append(float(action[0, 1]))
        data['hold_flag'].append(float(action[0, 2]) if action.shape[1] >= 3 else -1.0)

        state, reward, terminated, _, _ = env.step(action)

        # Use actual LP position from post-step state (respects hold flag)
        data['position_lower_price'].append(EXP_VALUE ** state[LP_TICK_LOWER_KEY][0])
        data['position_upper_price'].append(EXP_VALUE ** state[LP_TICK_UPPER_KEY][0])

        # Actual position offsets relative to current tick
        current_tick = state[POOL_CURRENT_TICK_KEY][0]
        data['actual_lower_offset'].append(state[LP_TICK_LOWER_KEY][0] - current_tick)
        data['actual_upper_offset'].append(state[LP_TICK_UPPER_KEY][0] - current_tick)

        cum_pnl += reward[0]
        data['reward'].append(reward[0])
        data['cumulative_pnl'].append(cum_pnl)
        if terminated[0]:
            break

    return {k: np.array(v) for k, v in data.items()}

# ============================================================================
# Plotting
# ============================================================================

def plot_pnl_distribution(pnl_results):
    """Box-plot + histogram of cumulative PnL across evaluated trajectories."""
    agents = [n for n in AGENT_NAMES if n in pnl_results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'PnL Distribution — {NUM_TRAJECTORIES_EVAL} Trajectories',
        fontsize=14, fontweight='bold',
    )

    # Box plot
    box_data = [pnl_results[n] for n in agents]
    bp = ax1.boxplot(box_data, labels=agents, patch_artist=True, notch=False)
    for patch, name in zip(bp['boxes'], agents):
        patch.set_facecolor(AGENT_COLORS[name])
        patch.set_alpha(0.6)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_ylabel('Cumulative PnL')
    ax1.set_title('PnL Distribution')
    ax1.grid(True, alpha=0.3)

    # Histogram
    all_pnl = np.concatenate(list(pnl_results.values()))
    lo, hi = np.percentile(all_pnl, [1, 99])
    bins = np.linspace(lo, hi, 50)
    for name in agents:
        ax2.hist(
            pnl_results[name], bins=bins, alpha=0.35,
            color=AGENT_COLORS[name], density=True,
            label=f"{name} (mean={np.mean(pnl_results[name]):+.1f})",
        )
        ax2.axvline(
            np.mean(pnl_results[name]),
            color=AGENT_COLORS[name], linestyle='--', linewidth=1.5,
        )
    ax2.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Cumulative PnL')
    ax2.set_ylabel('Density')
    ax2.set_title('PnL Histogram')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'pnl_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_price_evolution(single_data):
    """Panels: pool price + midprice + LP range for each agent.

    single_data values are lists of per-sim data dicts.
    """
    agents = [n for n in AGENT_NAMES if n in single_data]
    n_agents = len(agents)
    ncols = 3
    nrows = (n_agents + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    if n_agents == 1:
        axes = np.array([axes])
    n_sims = max(len(v) for v in single_data.values())
    title_suffix = f'{n_sims} Simulation{"s" if n_sims > 1 else ""}'
    fig.suptitle(
        f'Price Evolution with LP Position Ranges — {title_suffix}',
        fontsize=14, fontweight='bold',
    )
    # Hide unused subplot(s)
    for ax in axes.flat[n_agents:]:
        ax.set_visible(False)

    for ax, name in zip(axes.flat, agents):
        color = AGENT_COLORS[name]
        sim_alpha = max(0.15, 0.8 / len(single_data[name]))
        for si, d in enumerate(single_data[name]):
            lbl_price = 'Pool Price' if si == 0 else None
            lbl_mid = 'Midprice' if si == 0 else None
            lbl_range = f'{name} Range' if si == 0 else None
            ax.plot(d['time'], d['pool_price'], 'k-', linewidth=1.0,
                    alpha=sim_alpha, label=lbl_price)
            ax.plot(d['time'], d['midprice'], color='gray', linewidth=0.8,
                    alpha=sim_alpha * 0.7, label=lbl_mid)
            ax.fill_between(
                d['time'], d['position_lower_price'], d['position_upper_price'],
                alpha=sim_alpha * 0.3, color=color, label=lbl_range,
            )
            ax.plot(d['time'], d['position_lower_price'], '--', color=color,
                    alpha=sim_alpha * 0.6, linewidth=0.8)
            ax.plot(d['time'], d['position_upper_price'], '--', color=color,
                    alpha=sim_alpha * 0.6, linewidth=0.8)

            # Show rebalance events as thin vertical lines (where hold_flag <= 0).
            has_hold_flag = not np.all(d['hold_flag'] == -1.0)
            if has_hold_flag:
                rebalance_mask = d['hold_flag'] <= 0
                rebalance_times = d['time'][rebalance_mask]
                if len(rebalance_times) > 0:
                    for rt in rebalance_times:
                        ax.axvline(rt, color='red', alpha=0.1, linewidth=0.5)
                    if si == 0:
                        ax.axvline(rebalance_times[0], color='red', alpha=0.3,
                                   linewidth=0.5, label='Rebalance')

        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title(name)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'price_evolution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save individual per-simulation plots
    if n_sims > 1:
        for si in range(n_sims):
            fig_i, axes_i = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
            if n_agents == 1:
                axes_i = np.array([axes_i])
            fig_i.suptitle(
                f'Price Evolution with LP Position Ranges — Simulation {si + 1}',
                fontsize=14, fontweight='bold',
            )
            for ax in axes_i.flat[n_agents:]:
                ax.set_visible(False)

            for ax, name in zip(axes_i.flat, agents):
                if si >= len(single_data[name]):
                    continue
                d = single_data[name][si]
                color = AGENT_COLORS[name]
                ax.plot(d['time'], d['pool_price'], 'k-', linewidth=1.5, label='Pool Price')
                ax.plot(d['time'], d['midprice'], color='gray', linewidth=1,
                        alpha=0.7, label='Midprice')
                ax.fill_between(
                    d['time'], d['position_lower_price'], d['position_upper_price'],
                    alpha=0.25, color=color, label=f'{name} Range',
                )
                ax.plot(d['time'], d['position_lower_price'], '--', color=color,
                        alpha=0.5, linewidth=0.8)
                ax.plot(d['time'], d['position_upper_price'], '--', color=color,
                        alpha=0.5, linewidth=0.8)

                has_hold_flag = not np.all(d['hold_flag'] == -1.0)
                if has_hold_flag:
                    rebalance_mask = d['hold_flag'] <= 0
                    rebalance_times = d['time'][rebalance_mask]
                    if len(rebalance_times) > 0:
                        for rt in rebalance_times:
                            ax.axvline(rt, color='red', alpha=0.15, linewidth=0.5)
                        ax.axvline(rebalance_times[0], color='red', alpha=0.3,
                                   linewidth=0.5, label='Rebalance')

                ax.set_xlabel('Time')
                ax.set_ylabel('Price')
                ax.set_title(name)
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path_i = os.path.join(FIGURES_DIR, f'price_evolution_sim{si + 1}.png')
            fig_i.savefig(path_i, dpi=150, bbox_inches='tight')
            plt.close(fig_i)
            print(f"  Saved: {path_i}")


def plot_pnl_evolution(single_data):
    """All enabled agents' cumulative PnL on a single plot.

    single_data values are lists of per-sim data dicts.
    """
    agents = [n for n in AGENT_NAMES if n in single_data]
    n_sims = max(len(v) for v in single_data.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    sim_alpha = max(0.2, 0.9 / n_sims)
    for name in agents:
        color = AGENT_COLORS[name]
        finals = [d['cumulative_pnl'][-1] for d in single_data[name]]
        mean_final = np.mean(finals)
        for si, d in enumerate(single_data[name]):
            label = (f"{name} (mean final={mean_final:+.1f})" if si == 0 else None)
            ax.plot(
                d['time'], d['cumulative_pnl'],
                color=color, linewidth=1.5, alpha=sim_alpha, label=label,
            )
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative PnL')
    title_suffix = f'{n_sims} Simulation{"s" if n_sims > 1 else ""}'
    ax.set_title(f'PnL Evolution — {title_suffix}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'pnl_evolution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_rewards(rl_rewards: dict):
    """Smoothed mean training reward per epoch for all trained RL agents.

    Args:
        rl_rewards: {agent_name: list_of_epoch_rewards}
    """
    if not rl_rewards:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    min_len = min(len(v) for v in rl_rewards.values())
    window = max(1, min(20, int(min_len // 4)))

    for name, rewards in rl_rewards.items():
        color = AGENT_COLORS[name]
        smoothed = [
            np.mean(rewards[max(0, i - window):i + 1])
            for i in range(len(rewards))
        ]
        ax.plot(smoothed, color=color, linewidth=2, label=name)
        ax.plot(rewards, color=color, alpha=0.15, linewidth=0.8)

    ax.set_xlabel('Epoch / Rollout')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('RL Training Rewards', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'training_rewards.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_position_offsets(single_data):
    """Panels: tick offsets (lower / upper) from current price.

    single_data values are lists of per-sim data dicts.
    """
    agents = [n for n in AGENT_NAMES if n in single_data]
    n_agents = len(agents)
    ncols = 3
    nrows = (n_agents + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    if n_agents == 1:
        axes = np.array([axes])
    n_sims = max(len(v) for v in single_data.values())
    title_suffix = f'{n_sims} Simulation{"s" if n_sims > 1 else ""}'
    fig.suptitle(
        f'Position Boundaries Relative to Current Price — {title_suffix}',
        fontsize=14, fontweight='bold',
    )
    for ax in axes.flat[n_agents:]:
        ax.set_visible(False)

    for ax, name in zip(axes.flat, agents):
        color = AGENT_COLORS[name]
        sim_alpha = max(0.2, 0.9 / len(single_data[name]))
        for si, d in enumerate(single_data[name]):
            lbl_lo = 'Lower Offset' if si == 0 else None
            lbl_hi = 'Upper Offset' if si == 0 else None
            ax.plot(d['time'], d['actual_lower_offset'], '--', color=color,
                    linewidth=1.5, alpha=sim_alpha, label=lbl_lo)
            ax.plot(d['time'], d['actual_upper_offset'], '-', color=color,
                    linewidth=1.5, alpha=sim_alpha, label=lbl_hi)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Tick Offset from Current Price')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'position_offsets.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# ============================================================================
# Main
# ============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Containers for trained RL models / agents
    reinforce_agent = None
    reinforce_rewards = []
    ppo_model = None
    ppo_reward_cb = EpisodeRewardCallback()
    sac_model = None
    sac_reward_cb = EpisodeRewardCallback()
    dqn_model = None
    dqn_reward_cb = EpisodeRewardCallback()
    dqn_action_table = None
    ppo_vec_normalize = None
    sac_vec_normalize = None

    # ==================================================================
    # Phase 1: Train RL agents
    # ==================================================================

    if ENABLE_AGENTS.get('REINFORCE'):
        print("=" * 60)
        print("Phase 1a: Training REINFORCE agent")
        print("=" * 60)

        reinforce_env = create_environment(NUM_TRAJECTORIES_TRAIN, SEED)

        # Determine input size via dummy forward pass
        dummy_policy = nn.Linear(7, 2)
        temp_agent = PolicyGradientAgent(dummy_policy, reinforce_env)
        input_size = temp_agent.input_size

        reinforce_policy = SimpleStraddlePolicy(input_size)
        action_std_decay = lambda t: max(0.3, ACTION_STD_INIT * (0.998 ** (t * REINFORCE_EPOCHS)))
        optimizer = torch.optim.Adam(reinforce_policy.parameters(), lr=REINFORCE_LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

        reinforce_agent = PolicyGradientAgent(
            policy=reinforce_policy, env=reinforce_env,
            action_std=action_std_decay, optimizer=optimizer,
            lr_scheduler=scheduler, max_grad_norm=1.0,
        )
        _, reinforce_rewards = reinforce_agent.train(
            num_epochs=REINFORCE_EPOCHS, reporting_freq=50,
        )

    if ENABLE_AGENTS.get('PPO'):
        print("\n" + "=" * 60)
        print("Phase 1b: Training PPO agent")
        print("=" * 60)

        ppo_env = create_environment(NUM_TRAJECTORIES_TRAIN, SEED)
        sb_train_env = StableBaselinesAMMEnvironment(ppo_env)
        ppo_vec_normalize = VecNormalize(
            VecMonitor(sb_train_env),
            norm_obs=True, norm_reward=False, clip_obs=10.0,
        )

        ppo_model = PPO(
            "MlpPolicy", ppo_vec_normalize,
            learning_rate=3e-4, n_steps=N_STEPS, batch_size=64,
            n_epochs=10, gamma=1.0, gae_lambda=0.95, clip_range=0.2,
            verbose=1, seed=SEED,
        )
        ppo_reward_cb = EpisodeRewardCallback()
        ppo_model.learn(total_timesteps=SB3_TOTAL_TIMESTEPS, callback=ppo_reward_cb)

    if ENABLE_AGENTS.get('SAC'):
        print("\n" + "=" * 60)
        print("Phase 1c: Training SAC agent")
        print("=" * 60)

        sac_env = create_environment(NUM_TRAJECTORIES_TRAIN, SEED)
        sb_sac_env = StableBaselinesAMMEnvironment(sac_env)
        sac_vec_normalize = VecNormalize(
            VecMonitor(sb_sac_env),
            norm_obs=True, norm_reward=False, clip_obs=10.0,
        )

        sac_learning_starts = 10 * N_STEPS * NUM_TRAJECTORIES_TRAIN
        sac_model = SAC(
            "MlpPolicy", sac_vec_normalize,
            learning_rate=3e-4, batch_size=256,
            gamma=0.99, tau=0.005, ent_coef='auto',
            learning_starts=sac_learning_starts,
            verbose=1, seed=SEED,
        )
        sac_reward_cb = EpisodeRewardCallback()
        sac_model.learn(total_timesteps=SB3_TOTAL_TIMESTEPS, callback=sac_reward_cb)

    if ENABLE_AGENTS.get('DQN'):
        print("\n" + "=" * 60)
        print("Phase 1d: Training DQN agent")
        print("=" * 60)

        dqn_env = create_environment(NUM_TRAJECTORIES_TRAIN, SEED)
        sb_dqn_env = StableBaselinesAMMEnvironment(dqn_env)
        dqn_wrapper = DiscreteActionVecEnv(sb_dqn_env, TAU, DQN_TICK_STRIDE)
        dqn_action_table = dqn_wrapper.action_table

        print(f"  Discrete action space: {dqn_wrapper.action_space.n} actions "
              f"(stride={DQN_TICK_STRIDE}, 1 hold + "
              f"{len(dqn_action_table) - 1} rebalance)")

        dqn_learning_starts = 5 * N_STEPS * NUM_TRAJECTORIES_TRAIN
        dqn_model = DQN(
            "MlpPolicy", dqn_wrapper,
            learning_rate=1e-4, batch_size=128,
            gamma=1.0,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            learning_starts=dqn_learning_starts,
            verbose=1, seed=SEED,
        )
        dqn_reward_cb = EpisodeRewardCallback()
        dqn_model.learn(total_timesteps=SB3_TOTAL_TIMESTEPS, callback=dqn_reward_cb)

    # ==================================================================
    # Phase 2: Evaluate enabled agents on eval trajectories
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Phase 2: Evaluating enabled agents on {NUM_TRAJECTORIES_EVAL} trajectories")
    print("=" * 60)

    EVAL_SEED = SEED + 999
    pnl_results = {}

    if ENABLE_AGENTS.get('Uniform'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        pnl_results['Uniform'] = evaluate_on_trajectories(
            env, UniformAllocationAgent(env).get_action,
        )

    if ENABLE_AGENTS.get('DeployOnce'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        pnl_results['DeployOnce'] = evaluate_on_trajectories(
            env, DeployOnceAgent(env).get_action,
        )

    if ENABLE_AGENTS.get('Cartea'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        pnl_results['Cartea'] = evaluate_on_trajectories(
            env, CarteaPLAgent(env, gamma=GAMMA_CARTEA, seed=SEED).get_action,
        )

    if ENABLE_AGENTS.get('REINFORCE') and reinforce_agent is not None:
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        pnl_results['REINFORCE'] = evaluate_on_trajectories(
            env, lambda s: reinforce_agent.get_action(s, deterministic=True),
        )

    if ENABLE_AGENTS.get('PPO') and ppo_model is not None:
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        sb_eval = StableBaselinesAMMEnvironment(env)
        ppo_eval = SbAgent(ppo_model, num_trajectories=NUM_TRAJECTORIES_EVAL)
        ppo_vec_normalize.training = False
        pnl_results['PPO'] = evaluate_on_trajectories(
            env, lambda s: ppo_eval.get_action(ppo_vec_normalize.normalize_obs(sb_eval._flatten_obs(s))),
        )

    if ENABLE_AGENTS.get('SAC') and sac_model is not None:
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        sb_eval_sac = StableBaselinesAMMEnvironment(env)
        sac_eval = SbAgent(sac_model, num_trajectories=NUM_TRAJECTORIES_EVAL)
        sac_vec_normalize.training = False
        pnl_results['SAC'] = evaluate_on_trajectories(
            env, lambda s: sac_eval.get_action(sac_vec_normalize.normalize_obs(sb_eval_sac._flatten_obs(s))),
        )

    if ENABLE_AGENTS.get('DQN') and dqn_model is not None:
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        sb_eval_dqn = StableBaselinesAMMEnvironment(env)
        _at = dqn_action_table  # local ref for closure
        def _dqn_eval_action(state, _sb=sb_eval_dqn, _m=dqn_model, _t=_at):
            flat = _sb._flatten_obs(state)
            disc, _ = _m.predict(flat, deterministic=True)
            return _t[disc]
        pnl_results['DQN'] = evaluate_on_trajectories(env, _dqn_eval_action)

    # Print summary table
    active = [n for n in AGENT_NAMES if n in pnl_results]
    header = f"{'Agent':<12} | {'Mean PnL':>10} | {'Std':>10} | {'Median':>10} | {'Profitable':>12}"
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for name in active:
        pnl = pnl_results[name]
        pct = 100 * np.mean(pnl > 0)
        print(
            f"{name:<12} | {np.mean(pnl):>+10.1f} | {np.std(pnl):>10.1f} "
            f"| {np.median(pnl):>+10.1f} | {np.sum(pnl > 0):>4d}/{len(pnl)} ({pct:.0f}%)"
        )
    print("-" * len(header))

    # ==================================================================
    # Phase 3: Collect single-trajectory data for each agent
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Phase 3: Collecting single-trajectory data ({NUM_SINGLE_SIMS} sim(s) per agent)")
    print("=" * 60)

    SINGLE_SEED_BASE = SEED + 7777
    single_data = {}  # {agent_name: [data_dict_sim0, data_dict_sim1, ...]}

    for sim_idx in range(NUM_SINGLE_SIMS):
        sim_seed = SINGLE_SEED_BASE + sim_idx

        if ENABLE_AGENTS.get('Uniform'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('Uniform', []).append(
                collect_single_trajectory(env, UniformAllocationAgent(env).get_action))

        if ENABLE_AGENTS.get('DeployOnce'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('DeployOnce', []).append(
                collect_single_trajectory(env, DeployOnceAgent(env).get_action))

        if ENABLE_AGENTS.get('Cartea'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('Cartea', []).append(
                collect_single_trajectory(env, CarteaPLAgent(env, gamma=GAMMA_CARTEA, seed=SEED).get_action))

        if ENABLE_AGENTS.get('REINFORCE') and reinforce_agent is not None:
            env = create_environment(1, sim_seed)
            single_data.setdefault('REINFORCE', []).append(
                collect_single_trajectory(env, lambda s: reinforce_agent.get_action(s, deterministic=True)))

        if ENABLE_AGENTS.get('PPO') and ppo_model is not None:
            env = create_environment(1, sim_seed)
            sb_single = StableBaselinesAMMEnvironment(env)
            ppo_single = SbAgent(ppo_model, num_trajectories=1)
            single_data.setdefault('PPO', []).append(
                collect_single_trajectory(env, lambda s, _a=ppo_single, _sb=sb_single: _a.get_action(ppo_vec_normalize.normalize_obs(_sb._flatten_obs(s)))))

        if ENABLE_AGENTS.get('SAC') and sac_model is not None:
            env = create_environment(1, sim_seed)
            sb_single_sac = StableBaselinesAMMEnvironment(env)
            sac_single = SbAgent(sac_model, num_trajectories=1)
            single_data.setdefault('SAC', []).append(
                collect_single_trajectory(env, lambda s, _a=sac_single, _sb=sb_single_sac: _a.get_action(sac_vec_normalize.normalize_obs(_sb._flatten_obs(s)))))

        if ENABLE_AGENTS.get('DQN') and dqn_model is not None:
            env = create_environment(1, sim_seed)
            sb_single_dqn = StableBaselinesAMMEnvironment(env)
            _at = dqn_action_table
            def _dqn_single_action(state, _sb=sb_single_dqn, _m=dqn_model, _t=_at):
                flat = _sb._flatten_obs(state)
                disc, _ = _m.predict(flat, deterministic=True)
                return _t[disc]
            single_data.setdefault('DQN', []).append(
                collect_single_trajectory(env, _dqn_single_action))

    for name in AGENT_NAMES:
        if name in single_data:
            finals = [d['cumulative_pnl'][-1] for d in single_data[name]]
            if len(finals) == 1:
                print(f"  {name:12s}  final PnL = {finals[0]:+.1f}")
            else:
                print(f"  {name:12s}  final PnL: mean={np.mean(finals):+.1f}, "
                      f"std={np.std(finals):.1f}, range=[{min(finals):+.1f}, {max(finals):+.1f}]")

    # ==================================================================
    # Phase 4: Generate plots (each saved individually)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 4: Generating plots")
    print("=" * 60)

    # Gather training reward series for enabled RL agents
    rl_training_rewards = {}
    if reinforce_rewards:
        rl_training_rewards['REINFORCE'] = reinforce_rewards
    if ppo_reward_cb.epoch_rewards:
        rl_training_rewards['PPO'] = ppo_reward_cb.epoch_rewards
    if sac_reward_cb.epoch_rewards:
        rl_training_rewards['SAC'] = sac_reward_cb.epoch_rewards
    if dqn_reward_cb.epoch_rewards:
        rl_training_rewards['DQN'] = dqn_reward_cb.epoch_rewards

    if rl_training_rewards:
        plot_training_rewards(rl_training_rewards)
    if pnl_results:
        plot_pnl_distribution(pnl_results)
    if single_data:
        plot_price_evolution(single_data)
        plot_pnl_evolution(single_data)
        plot_position_offsets(single_data)

    print("\nDone. All figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
