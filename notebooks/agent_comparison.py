"""
Holistic comparison of LP agents:
  Uniform, DeployOnce, CarteaDrissiMonga, REINFORCE (PolicyGradient), PPO, SAC, DQN.

Phases:
  1. Train enabled RL agents
  2. Evaluate enabled agents on 1000 trajectories → PnL distribution plot
  3. Run 1 trajectory per agent → price evolution, PnL evolution, position offsets

Each plot is saved individually to figures/.
Toggle agents on/off via the ENABLE_AGENTS dict.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
job_id = os.environ.get("SLURM_JOB_ID", "local")
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
from SAiFE_gym.wrappers import DiscreteActionVecEnv, StructuredMultiDiscreteVecEnv
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel, OrnsteinUhlenbeckMidpriceModel, GeometricBrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from SAiFE_gym.agents.BaselineAgents import (
    UniformAllocationAgent, DeployOnceAgent, CarteaPLAgent, ArrivalRebalanceAgent,
    DoNothingAgent,
)
from SAiFE_gym.agents.PolicyGradientAgent import PolicyGradientAgent
from SAiFE_gym.agents.SbAgent import SbAgent
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    MISPRICING_KEY, LP_LOWER_OFFSET_KEY, LP_UPPER_OFFSET_KEY, GAS_COST_KEY,
)

# ============================================================================
# Configuration
# ============================================================================

SEED = 6#2#123
TERMINAL_TIME = 1.0
N_STEPS = 1000
NUM_TRAJECTORIES_TRAIN = 200
NUM_TRAJECTORIES_EVAL = 100
INITIAL_WEALTH = 1000
TAU = 20
LIQUIDITY_SCALE = 1e5

INITIAL_PRICE = 100#0000
INITIAL_POOL_PRICE = None#99.98  # None → pool starts at INITIAL_PRICE; set a float to seed a mispricing
DRIFT = 0
VOLATILITY = 0.009
FEE_TIER = 0.003#5#0.003
EXP_VALUE = 1.0001

ALPHA0 = np.array([1.0, 1.0])
ALPHA1 = np.array([15.0, 15.0])
ALPHA2 = np.array([0.0, 0.0])
ALPHA3 = np.array([20000, 20000])

GAMMA_CARTEA = 0.000005

ARRIVAL_REBALANCE_EVERY = 15000000
ARRIVAL_REBALANCE_WIDTH = 1
# Asymmetric range for ArrivalRebalance (None → fall back to ±WIDTH).
# Pass both to quote e.g. [-3, +7]; the agent will use these instead of WIDTH.
ARRIVAL_REBALANCE_LOWER = -1
ARRIVAL_REBALANCE_UPPER = 0

# DeployOnce quote bounds (None → defaults to symmetric [-TAU, +TAU])
DEPLOYONCE_LOWER = -10#-5#200#140
DEPLOYONCE_UPPER = 10#200#155#TAU

REINFORCE_EPOCHS = 600#600
REINFORCE_LR = 2e-4
ACTION_STD_INIT = 1.7



DQN_TICK_STRIDE = 10  # Discretization stride for tick offsets
NUM_SINGLE_SIMS = 10   # Number of single-trajectory simulations to plot
MAX_TRADES_DEBUG = None  # int → cut each single-trajectory rollout off after this many arrivals; None = run full episode

# PPO action wrapper:
#   'multidiscrete' — MultiDiscrete([2·TAU+1, TAU, 2]) for (center, half_width,
#                     hold). True Categorical heads, valid by construction,
#                     uniform initial exploration. Recommended.
#   'rescaled'      — Box([-1, 1]^3) for raw (lower, upper, hold) with sign-
#                     thresholded auto-correction in validate_action. Gaussian
#                     policy, biased toward narrow centred actions.
PPO_ACTION_WRAPPER = 'multidiscrete'

# Decision stride: RL agent emits an action every DECISION_STRIDE underlying env
# steps; intermediate steps run with a forced HOLD. Stride=1 is a no-op (every
# env step is a decision step — identical to the original behaviour). Must
# divide N_STEPS evenly. Only applied to RL agents (REINFORCE/PPO/SAC/DQN);
# baselines are unaffected.
# DECISION_STRIDE = N_STEPS/#decisions per episode
DECISION_STRIDE = 1000

#600*200*1000/1000 = 120000 actions to learn 
SB3_TOTAL_TIMESTEPS = REINFORCE_EPOCHS * NUM_TRAJECTORIES_TRAIN * N_STEPS // DECISION_STRIDE


# Observation features for SB3 agents (PPO/SAC/DQN). Uses raw asymmetric position
# offsets instead of the default symmetric (boundary, width) encoding so the
# policy can express asymmetric quotes without inverting a non-linear compression.
SB3_OBS_KEYS = [
    MISPRICING_KEY,
    ASSET_PRICE_KEY,         # midprice
    POOL_SQRT_PRICE_KEY,     # exposed as squared (= pool price) by the SB3 wrapper
    LP_LOWER_OFFSET_KEY,
    LP_UPPER_OFFSET_KEY,
    TIME_KEY,
    GAS_COST_KEY,
]

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# ── Agent toggles (set to False to skip training / evaluation) ──────
ENABLE_AGENTS = {
    'DoNothing':  True,
    'Uniform':    False,
    'DeployOnce': True,
    'ArrivalRebalance': True,
    'CarteaDrissiMonga': False,
    'REINFORCE':  False,
    'PPO':        True,
    'SAC':        False,
    'DQN':        False,
}

AGENT_NAMES = [name for name, on in ENABLE_AGENTS.items() if on]
AGENT_COLORS = {
    'DoNothing':  '#7f7f7f',
    'Uniform':    '#1f77b4',
    'DeployOnce': '#9467bd',
    'ArrivalRebalance': '#e377c2',
    'CarteaDrissiMonga': '#ff7f0e',
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

    #midprice_model = BrownianMotionMidpriceModel(
    #    drift=DRIFT, volatility=VOLATILITY, initial_price=INITIAL_PRICE,
    #    terminal_time=TERMINAL_TIME, step_size=step_size,
    #    num_trajectories=num_trajectories, seed=seed,
    #)

    #midprice_model = GeometricBrownianMotionMidpriceModel(
    #    drift=DRIFT, volatility=VOLATILITY, initial_price=INITIAL_PRICE,
    #    terminal_time=TERMINAL_TIME, step_size=step_size,
    #    num_trajectories=num_trajectories, seed=seed,
    #)


    midprice_model = OrnsteinUhlenbeckMidpriceModel(                                                     
        mean_reversion=0.4,           # κ — pull strength toward θ                                       
        long_term_mean=INITIAL_PRICE, # θ — defaults to INITIAL_PRICE if omitted                                                                                               
        volatility=VOLATILITY,
        num_trajectories=num_trajectories, seed=seed,
        initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME, step_size=step_size
    )

    
    arrival_model = LiquidityKernelArrivalModel(
        alpha=alpha, beta=0.1, K=20, liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size, num_trajectories=num_trajectories,
        seed=seed + 1 if seed else None,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model, arrival_model=arrival_model,
        num_trajectories=num_trajectories, fee_tier=FEE_TIER, tau=TAU,
        num_ticks=5000, exponential_value=EXP_VALUE,
        seed=seed + 2 if seed else None,
    )
    reward_function = PnL()
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS,
        initial_wealth=INITIAL_WEALTH,
        reward_function=reward_function, model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        initial_pool_price=INITIAL_POOL_PRICE,
        seed=seed,
    )

class DecisionStrideEnv:
    """Decision-stride wrapper for ``AMMEnvironment`` (REINFORCE / eval path).

    Each outer ``step`` runs ``stride`` underlying env steps. The agent's action
    is applied on the first; the remaining ``stride-1`` steps run with a forced
    HOLD = ``[0, 1, +1]`` (offsets are ignored by ``update_state`` when
    ``hold_flag > 0``). Per-step rewards are summed into a single window reward.

    ``stride=1`` is a transparent passthrough.
    """

    def __init__(self, env, stride: int = 1):
        assert stride >= 1, f"stride must be >= 1, got {stride}"
        assert env.n_steps % stride == 0, (
            f"n_steps={env.n_steps} not divisible by stride={stride}"
        )
        self.env = env
        self.stride = int(stride)
        self._hold = np.tile(
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
            (env.num_trajectories, 1),
        )

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.stride == 1:
            return self.env.step(action)

        reward_sum = np.zeros(self.env.num_trajectories, dtype=np.float64)
        obs = None
        terminated = np.zeros(self.env.num_trajectories, dtype=bool)
        truncated = np.zeros(self.env.num_trajectories, dtype=bool)
        info = {}
        for k in range(self.stride):
            act = action if k == 0 else self._hold
            obs, reward, terminated, truncated, info = self.env.step(act)
            reward_sum += reward
            if np.any(terminated):
                break
        return obs, reward_sum, terminated, truncated, info

    def __getattr__(self, name):
        # Delegate any unknown attribute (model_dynamics, num_trajectories,
        # terminal_time, n_steps, step_size, action_space, ...) to the inner env.
        return getattr(self.env, name)


class DecisionStrideVecEnv(VecEnv):
    """Decision-stride wrapper for ``StableBaselinesAMMEnvironment``.

    SB3 sees one outer step per agent decision; internally ``stride`` underlying
    env steps run with the agent's action on the first and HOLD = ``[0, 1, +1]``
    on the rest. Returns the obs at the end of the window, the summed window
    reward, and forwards the inner ``dones`` / ``infos`` (auto-reset and
    ``terminal_observation`` plumbing live in the inner env).

    ``stride=1`` is a transparent passthrough.
    """

    def __init__(self, vec_env: VecEnv, stride: int = 1):
        assert stride >= 1, f"stride must be >= 1, got {stride}"
        n_steps = getattr(vec_env, 'n_steps', None)
        if n_steps is not None:
            assert n_steps % stride == 0, (
                f"n_steps={n_steps} not divisible by stride={stride}"
            )
        self._wrapped = vec_env
        self.stride = int(stride)
        self._hold = np.tile(
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
            (vec_env.num_envs, 1),
        )
        self._pending_action = None
        super().__init__(vec_env.num_envs, vec_env.observation_space, vec_env.action_space)

    def reset(self):
        return self._wrapped.reset()

    def step_async(self, actions):
        self._pending_action = actions

    def step_wait(self):
        action = self._pending_action
        if self.stride == 1:
            self._wrapped.step_async(action)
            return self._wrapped.step_wait()

        reward_sum = np.zeros(self.num_envs, dtype=np.float32)
        obs = dones = infos = None
        for k in range(self.stride):
            act = action if k == 0 else self._hold
            self._wrapped.step_async(act)
            obs, reward, dones, infos = self._wrapped.step_wait()
            reward_sum += reward.astype(np.float32)
            if np.all(dones):
                break
        return obs, reward_sum, dones, infos

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


class RescaledActionVecEnv(VecEnv):
    """Re-exposes a VecEnv with a unit-cube ``[-1, 1]^d`` action space.

    Without this, PPO's Gaussian noise (std≈1 from ``log_std_init=0``) explores
    only a ±2-tick slice of the ±tau offset range, so narrow vs wide vs
    asymmetric quotes are unreachable at standard hyperparameters. Rescaling
    every dim to ``[-1, 1]`` also fixes the scale mismatch between the offset
    dims (±tau) and the hold_flag (±1) so neither one drowns the other in the
    policy gradient.
    """

    def __init__(self, vec_env: VecEnv):
        self._wrapped = vec_env
        self._low = np.asarray(vec_env.action_space.low, dtype=np.float32)
        self._high = np.asarray(vec_env.action_space.high, dtype=np.float32)
        norm_act_space = gymnasium.spaces.Box(
            low=-np.ones_like(self._low), high=np.ones_like(self._high),
            dtype=np.float32,
        )
        super().__init__(vec_env.num_envs, vec_env.observation_space, norm_act_space)

    def unscale(self, action: np.ndarray) -> np.ndarray:
        """Map a ``[-1, 1]``-space action back to the wrapped env's action space."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return self._low + 0.5 * (a + 1.0) * (self._high - self._low)

    def reset(self):
        return self._wrapped.reset()

    def step_async(self, actions):
        self._wrapped.step_async(self.unscale(actions))

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


def collect_single_trajectory(env, get_action_fn, max_trades: int = None,
                              decision_stride: int = 1):
    """Run one episode (num_trajectories=1) and return per-step data dict.

    ``decision_stride`` controls how often the agent's action is sampled:
    once every ``stride`` env steps. Intermediate steps run with a forced
    HOLD = ``[0, 1, +1]``. Data is still recorded at every env step so plots
    keep full per-step resolution. Stride=1 reproduces the original behaviour.

    If ``max_trades`` is set, the rollout cuts off as soon as that many
    arrivals (sell + buy) have been seen — useful for debugging strategy
    behaviour around a controlled number of swap events.
    """
    assert env.num_trajectories == 1
    assert env.n_steps % decision_stride == 0, (
        f"n_steps={env.n_steps} not divisible by decision_stride={decision_stride}"
    )
    data = {k: [] for k in [
        'time', 'pool_price', 'midprice',
        'position_lower_price', 'position_upper_price',
        'action_lower', 'action_upper',
        'actual_lower_offset', 'actual_upper_offset',
        'hold_flag',
        'reward', 'cumulative_pnl',
        'trade_count',
    ]}
    state, _ = env.reset()
    cum_pnl = 0.0
    trade_count = 0
    hold_action = np.array([[0.0, 1.0, 1.0]], dtype=np.float32)

    for step_idx in range(env.n_steps):
        data['time'].append(state[TIME_KEY][0])
        data['pool_price'].append(state[POOL_SQRT_PRICE_KEY][0] ** 2)
        data['midprice'].append(state[ASSET_PRICE_KEY][0])

        is_decision = (step_idx % decision_stride) == 0
        action = get_action_fn(state) if is_decision else hold_action
        data['action_lower'].append(float(action[0, 0]))
        data['action_upper'].append(float(action[0, 1]))
        data['hold_flag'].append(float(action[0, 2]) if action.shape[1] >= 3 else -1.0)

        state, reward, terminated, _, _ = env.step(action)
        # last_arrivals is cached on model_dynamics by env.step's get_arrivals call.
        trade_count += int(env.model_dynamics.last_arrivals.sum())

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
        data['trade_count'].append(trade_count)

        if terminated[0]:
            break
        if max_trades is not None and trade_count >= max_trades:
            break

    return {k: np.array(v) for k, v in data.items()}

# ============================================================================
# Plotting
# ============================================================================

def plot_pnl_distribution(pnl_results):
    """Box-plot and histogram of cumulative PnL, saved as separate figures."""
    agents = [n for n in AGENT_NAMES if n in pnl_results]

    # Box plot
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    box_data = [pnl_results[n] for n in agents]
    bp = ax1.boxplot(box_data, labels=agents, patch_artist=True, notch=False)
    for patch, name in zip(bp['boxes'], agents):
        patch.set_facecolor(AGENT_COLORS[name])
        patch.set_alpha(0.6)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_ylabel('Cumulative PnL', fontsize=16)
    ax1.tick_params(axis='both', labelsize=14)
    #ax1.set_title('PnL Distribution')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, f'pnl_boxplot_{job_id}.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # Histogram
    fig2, ax2 = plt.subplots(figsize=(8, 5))
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
    ax2.set_xlabel('Cumulative PnL', fontsize=16)
    ax2.set_ylabel('Density', fontsize=16)
    ax2.tick_params(axis='both', labelsize=14)
    #ax2.set_title('PnL Histogram')
    ax2.legend(fontsize=16, loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, f'pnl_histogram_{job_id}.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {path2}")


def plot_price_evolution(single_data):
    """Panels: pool price + midprice + LP range for each agent.

    single_data values are lists of per-sim data dicts.
    """
    agents = [n for n in AGENT_NAMES if n in single_data]
    n_agents = len(agents)
    ncols = 2
    nrows = (n_agents + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)
    n_sims = max(len(v) for v in single_data.values())
    title_suffix = f'{n_sims} Simulation{"s" if n_sims > 1 else ""}'
    #fig.suptitle(
    #    f'Price Evolution with LP Position Ranges — {title_suffix}',
    #    fontsize=14, fontweight='bold',
    #)
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

        ax.set_xlabel('Time', fontsize=16)
        ax.set_ylabel('Price', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.ticklabel_format(axis='y', useOffset=False, style='plain')
        ax.set_title(name)
        ax.legend(fontsize=14, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'price_evolution_{job_id}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save standalone figure per trained RL agent (all sims overlaid)
    rl_agents = [n for n in agents if n in ('REINFORCE', 'PPO', 'SAC', 'DQN')]
    for name in rl_agents:
        fig_rl, ax_rl = plt.subplots(figsize=(8, 5))
        color = AGENT_COLORS[name]
        sim_alpha = max(0.15, 0.8 / len(single_data[name]))
        for si, d in enumerate(single_data[name]):
            lbl_price = 'Pool Price' if si == 0 else None
            lbl_mid = 'Midprice' if si == 0 else None
            lbl_range = f'{name} Range' if si == 0 else None
            ax_rl.plot(d['time'], d['pool_price'], 'k-', linewidth=1.0,
                       alpha=sim_alpha, label=lbl_price)
            ax_rl.plot(d['time'], d['midprice'], color='gray', linewidth=0.8,
                       alpha=sim_alpha * 0.7, label=lbl_mid)
            ax_rl.fill_between(
                d['time'], d['position_lower_price'], d['position_upper_price'],
                alpha=sim_alpha * 0.3, color=color, label=lbl_range,
            )
            ax_rl.plot(d['time'], d['position_lower_price'], '--', color=color,
                       alpha=sim_alpha * 0.6, linewidth=0.8)
            ax_rl.plot(d['time'], d['position_upper_price'], '--', color=color,
                       alpha=sim_alpha * 0.6, linewidth=0.8)

            has_hold_flag = not np.all(d['hold_flag'] == -1.0)
            if has_hold_flag:
                rebalance_mask = d['hold_flag'] <= 0
                rebalance_times = d['time'][rebalance_mask]
                if len(rebalance_times) > 0:
                    for rt in rebalance_times:
                        ax_rl.axvline(rt, color='red', alpha=0.1, linewidth=0.5)
                    if si == 0:
                        ax_rl.axvline(rebalance_times[0], color='red', alpha=0.3,
                                      linewidth=0.5, label='Rebalance')

        ax_rl.set_xlabel('Time', fontsize=16)
        ax_rl.set_ylabel('Price', fontsize=16)
        ax_rl.tick_params(axis='both', labelsize=14)
        ax_rl.ticklabel_format(axis='y', useOffset=False, style='plain')
        ax_rl.set_title(name)
        ax_rl.legend(fontsize=16, loc='lower right')
        ax_rl.grid(True, alpha=0.3)
        plt.tight_layout()
        path_rl = os.path.join(FIGURES_DIR, f'price_evolution_{name.lower()}_{job_id}.png')
        fig_rl.savefig(path_rl, dpi=200, bbox_inches='tight')
        plt.close(fig_rl)
        print(f"  Saved: {path_rl}")

    # Save individual per-simulation plots
    if n_sims > 1:
        for si in range(n_sims):
            fig_i, axes_i = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
            axes_i = np.atleast_2d(axes_i)
            #fig_i.suptitle(
            #    f'Price Evolution with LP Position Ranges — Simulation {si + 1}',
            #    fontsize=14, fontweight='bold',
            #)
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

                ax.set_xlabel('Time', fontsize=16)
                ax.set_ylabel('Price', fontsize=16)
                ax.tick_params(axis='both', labelsize=14)
                ax.ticklabel_format(axis='y', useOffset=False, style='plain')
                ax.set_title(name)
                ax.legend(fontsize=14, loc='lower right')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path_i = os.path.join(FIGURES_DIR, f'price_evolution_sim{si + 1}_{job_id}.png')
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
    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel('Cumulative PnL', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    title_suffix = f'{n_sims} Simulation{"s" if n_sims > 1 else ""}'
    #ax.set_title(f'PnL Evolution — {title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=16, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'pnl_evolution_{job_id}.png')
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

    ax.set_xlabel('Epoch / Rollout', fontsize=16)
    ax.set_ylabel('Mean Episode Reward', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_title('RL Training Rewards', fontsize=14, fontweight='bold')
    ax.legend(fontsize=16, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'training_rewards_{job_id}.png')
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
        ax.set_xlabel('Time', fontsize=16)
        ax.set_ylabel('Tick Offset from Current Price', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'position_offsets_{job_id}.png')
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
    ppo_action_wrapper = None
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

        reinforce_env = DecisionStrideEnv(
            create_environment(NUM_TRAJECTORIES_TRAIN, SEED), stride=DECISION_STRIDE,
        )

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
        t0 = time.time()
        _, reinforce_rewards = reinforce_agent.train(
            num_epochs=REINFORCE_EPOCHS, reporting_freq=50,
        )
        print(f"  REINFORCE training time: {time.time() - t0:.1f}s")

    if ENABLE_AGENTS.get('PPO'):
        print("\n" + "=" * 60)
        print("Phase 1b: Training PPO agent")
        print("=" * 60)

        ppo_env = create_environment(NUM_TRAJECTORIES_TRAIN, SEED)
        sb_train_env = DecisionStrideVecEnv(
            StableBaselinesAMMEnvironment(ppo_env, obs_keys=SB3_OBS_KEYS),
            stride=DECISION_STRIDE,
        )
        ppo_vec_normalize = VecNormalize(
            VecMonitor(sb_train_env),
            norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
        )
        # Wrap the action space so PPO sees a sensible exploration target
        # (see PPO_ACTION_WRAPPER docs). Both wrappers expose .unscale() and
        # produce env-facing [lower, upper, hold_flag] actions, so the eval
        # and single-trajectory closures below are wrapper-agnostic.
        if PPO_ACTION_WRAPPER == 'multidiscrete':
            ppo_action_wrapper = StructuredMultiDiscreteVecEnv(ppo_vec_normalize, TAU)
        elif PPO_ACTION_WRAPPER == 'rescaled':
            ppo_action_wrapper = RescaledActionVecEnv(ppo_vec_normalize)
        else:
            raise ValueError(
                f"PPO_ACTION_WRAPPER must be 'multidiscrete' or 'rescaled', "
                f"got {PPO_ACTION_WRAPPER!r}"
            )

        ppo_model = PPO(
            "MlpPolicy", ppo_action_wrapper,
            learning_rate=3e-4, n_steps=N_STEPS // DECISION_STRIDE, batch_size=64,
            n_epochs=5, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.05,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            verbose=1, seed=SEED,
        )
        ppo_reward_cb = EpisodeRewardCallback()
        t0 = time.time()
        ppo_model.learn(total_timesteps=SB3_TOTAL_TIMESTEPS, callback=ppo_reward_cb)
        print(f"  PPO training time: {time.time() - t0:.1f}s")

    if ENABLE_AGENTS.get('SAC'):
        print("\n" + "=" * 60)
        print("Phase 1c: Training SAC agent")
        print("=" * 60)

        sac_env = create_environment(NUM_TRAJECTORIES_TRAIN, SEED)
        sb_sac_env = DecisionStrideVecEnv(
            StableBaselinesAMMEnvironment(sac_env, obs_keys=SB3_OBS_KEYS),
            stride=DECISION_STRIDE,
        )
        sac_vec_normalize = VecNormalize(
            VecMonitor(sb_sac_env),
            norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
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
        t0 = time.time()
        sac_model.learn(total_timesteps=SB3_TOTAL_TIMESTEPS, callback=sac_reward_cb)
        print(f"  SAC training time: {time.time() - t0:.1f}s")

    if ENABLE_AGENTS.get('DQN'):
        print("\n" + "=" * 60)
        print("Phase 1d: Training DQN agent")
        print("=" * 60)

        dqn_env = create_environment(NUM_TRAJECTORIES_TRAIN, SEED)
        sb_dqn_env = DecisionStrideVecEnv(
            StableBaselinesAMMEnvironment(dqn_env, obs_keys=SB3_OBS_KEYS),
            stride=DECISION_STRIDE,
        )
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
        t0 = time.time()
        dqn_model.learn(total_timesteps=SB3_TOTAL_TIMESTEPS, callback=dqn_reward_cb)
        print(f"  DQN training time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Phase 2: Evaluate enabled agents on eval trajectories
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Phase 2: Evaluating enabled agents on {NUM_TRAJECTORIES_EVAL} trajectories")
    print("=" * 60)

    EVAL_SEED = SEED + 999
    pnl_results = {}

    if ENABLE_AGENTS.get('DoNothing'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        pnl_results['DoNothing'] = evaluate_on_trajectories(
            env, DoNothingAgent(env).get_action,
        )

    if ENABLE_AGENTS.get('Uniform'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        pnl_results['Uniform'] = evaluate_on_trajectories(
            env, UniformAllocationAgent(env).get_action,
        )

    if ENABLE_AGENTS.get('DeployOnce'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        pnl_results['DeployOnce'] = evaluate_on_trajectories(
            env, DeployOnceAgent(
                env, lower_offset=DEPLOYONCE_LOWER, upper_offset=DEPLOYONCE_UPPER,
            ).get_action,
        )

    if ENABLE_AGENTS.get('ArrivalRebalance'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        pnl_results['ArrivalRebalance'] = evaluate_on_trajectories(
            env, ArrivalRebalanceAgent(
                env, rebalance_every=ARRIVAL_REBALANCE_EVERY,
                width=ARRIVAL_REBALANCE_WIDTH,
                lower_offset=ARRIVAL_REBALANCE_LOWER,
                upper_offset=ARRIVAL_REBALANCE_UPPER,
            ).get_action,
        )

    if ENABLE_AGENTS.get('CarteaDrissiMonga'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        pnl_results['CarteaDrissiMonga'] = evaluate_on_trajectories(
            env, CarteaPLAgent(env, gamma=GAMMA_CARTEA, seed=SEED).get_action,
        )

    if ENABLE_AGENTS.get('REINFORCE') and reinforce_agent is not None:
        env = DecisionStrideEnv(
            create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED), stride=DECISION_STRIDE,
        )
        pnl_results['REINFORCE'] = evaluate_on_trajectories(
            env, lambda s: reinforce_agent.get_action(s, deterministic=True),
        )

    if ENABLE_AGENTS.get('PPO') and ppo_model is not None:
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        sb_eval = StableBaselinesAMMEnvironment(env, obs_keys=SB3_OBS_KEYS)
        ppo_eval = SbAgent(ppo_model, num_trajectories=NUM_TRAJECTORIES_EVAL)
        ppo_vec_normalize.training = False
        eval_env = DecisionStrideEnv(env, stride=DECISION_STRIDE)
        pnl_results['PPO'] = evaluate_on_trajectories(
            eval_env, lambda s: ppo_action_wrapper.unscale(
                ppo_eval.get_action(ppo_vec_normalize.normalize_obs(sb_eval._flatten_obs(s)))
            ),
        )

    if ENABLE_AGENTS.get('SAC') and sac_model is not None:
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        sb_eval_sac = StableBaselinesAMMEnvironment(env, obs_keys=SB3_OBS_KEYS)
        sac_eval = SbAgent(sac_model, num_trajectories=NUM_TRAJECTORIES_EVAL)
        sac_vec_normalize.training = False
        eval_env = DecisionStrideEnv(env, stride=DECISION_STRIDE)
        pnl_results['SAC'] = evaluate_on_trajectories(
            eval_env, lambda s: sac_eval.get_action(sac_vec_normalize.normalize_obs(sb_eval_sac._flatten_obs(s))),
        )

    if ENABLE_AGENTS.get('DQN') and dqn_model is not None:
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        sb_eval_dqn = StableBaselinesAMMEnvironment(env, obs_keys=SB3_OBS_KEYS)
        _at = dqn_action_table  # local ref for closure
        def _dqn_eval_action(state, _sb=sb_eval_dqn, _m=dqn_model, _t=_at):
            flat = _sb._flatten_obs(state)
            disc, _ = _m.predict(flat, deterministic=True)
            return _t[disc]
        eval_env = DecisionStrideEnv(env, stride=DECISION_STRIDE)
        pnl_results['DQN'] = evaluate_on_trajectories(eval_env, _dqn_eval_action)

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

        if ENABLE_AGENTS.get('DoNothing'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('DoNothing', []).append(
                collect_single_trajectory(env, DoNothingAgent(env).get_action, max_trades=MAX_TRADES_DEBUG))

        if ENABLE_AGENTS.get('Uniform'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('Uniform', []).append(
                collect_single_trajectory(env, UniformAllocationAgent(env).get_action, max_trades=MAX_TRADES_DEBUG))

        if ENABLE_AGENTS.get('DeployOnce'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('DeployOnce', []).append(
                collect_single_trajectory(env, DeployOnceAgent(
                    env, lower_offset=DEPLOYONCE_LOWER, upper_offset=DEPLOYONCE_UPPER,
                ).get_action, max_trades=MAX_TRADES_DEBUG))

        if ENABLE_AGENTS.get('ArrivalRebalance'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('ArrivalRebalance', []).append(
                collect_single_trajectory(env, ArrivalRebalanceAgent(
                    env, rebalance_every=ARRIVAL_REBALANCE_EVERY,
                    width=ARRIVAL_REBALANCE_WIDTH,
                    lower_offset=ARRIVAL_REBALANCE_LOWER,
                    upper_offset=ARRIVAL_REBALANCE_UPPER, 
                ).get_action, max_trades=MAX_TRADES_DEBUG))

        if ENABLE_AGENTS.get('CarteaDrissiMonga'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('CarteaDrissiMonga', []).append(
                collect_single_trajectory(env, CarteaPLAgent(env, gamma=GAMMA_CARTEA, seed=SEED).get_action, max_trades=MAX_TRADES_DEBUG))

        if ENABLE_AGENTS.get('REINFORCE') and reinforce_agent is not None:
            env = create_environment(1, sim_seed)
            single_data.setdefault('REINFORCE', []).append(
                collect_single_trajectory(env, lambda s: reinforce_agent.get_action(s, deterministic=True), max_trades=MAX_TRADES_DEBUG, decision_stride=DECISION_STRIDE))

        if ENABLE_AGENTS.get('PPO') and ppo_model is not None:
            env = create_environment(1, sim_seed)
            sb_single = StableBaselinesAMMEnvironment(env, obs_keys=SB3_OBS_KEYS)
            ppo_single = SbAgent(ppo_model, num_trajectories=1)
            single_data.setdefault('PPO', []).append(
                collect_single_trajectory(env, lambda s, _a=ppo_single, _sb=sb_single, _w=ppo_action_wrapper: _w.unscale(_a.get_action(ppo_vec_normalize.normalize_obs(_sb._flatten_obs(s)))), max_trades=MAX_TRADES_DEBUG, decision_stride=DECISION_STRIDE))

        if ENABLE_AGENTS.get('SAC') and sac_model is not None:
            env = create_environment(1, sim_seed)
            sb_single_sac = StableBaselinesAMMEnvironment(env, obs_keys=SB3_OBS_KEYS)
            sac_single = SbAgent(sac_model, num_trajectories=1)
            single_data.setdefault('SAC', []).append(
                collect_single_trajectory(env, lambda s, _a=sac_single, _sb=sb_single_sac: _a.get_action(sac_vec_normalize.normalize_obs(_sb._flatten_obs(s))), max_trades=MAX_TRADES_DEBUG, decision_stride=DECISION_STRIDE))

        if ENABLE_AGENTS.get('DQN') and dqn_model is not None:
            env = create_environment(1, sim_seed)
            sb_single_dqn = StableBaselinesAMMEnvironment(env, obs_keys=SB3_OBS_KEYS)
            _at = dqn_action_table
            def _dqn_single_action(state, _sb=sb_single_dqn, _m=dqn_model, _t=_at):
                flat = _sb._flatten_obs(state)
                disc, _ = _m.predict(flat, deterministic=True)
                return _t[disc]
            single_data.setdefault('DQN', []).append(
                collect_single_trajectory(env, _dqn_single_action, max_trades=MAX_TRADES_DEBUG, decision_stride=DECISION_STRIDE))

    for name in AGENT_NAMES:
        if name in single_data:
            finals = [d['cumulative_pnl'][-1] for d in single_data[name]]
            if len(finals) == 1:
                print(f"  {name:12s}  final PnL = {finals[0]:+.1f}")
            else:
                print(f"  {name:12s}  final PnL: mean={np.mean(finals):+.1f}, "
                      f"std={np.std(finals):.1f}, range=[{min(finals):+.1f}, {max(finals):+.1f}]")

    # When MAX_TRADES_DEBUG is set, Phase 2's full-episode eval is no longer
    # comparable to what the single-trajectory plots show. Print a
    # Phase-2-style summary built from the single-trajectory final PnLs so we
    # have a numerical comparison at the same trade cap (only meaningful with
    # NUM_SINGLE_SIMS large enough to be informative).
    if MAX_TRADES_DEBUG is not None and single_data:
        active_single = [n for n in AGENT_NAMES if n in single_data]
        header = f"{'Agent':<18} | {'Mean PnL':>10} | {'Std':>10} | {'Median':>10} | {'Profitable':>12}"
        print(f"\n  Per-trajectory PnL (cut off at {MAX_TRADES_DEBUG} trades, "
              f"{NUM_SINGLE_SIMS} sim(s) per agent):")
        print("  " + "-" * len(header))
        print("  " + header)
        print("  " + "-" * len(header))
        for name in active_single:
            finals = np.array([d['cumulative_pnl'][-1] for d in single_data[name]])
            pct = 100 * np.mean(finals > 0)
            print(
                f"  {name:<18} | {np.mean(finals):>+10.1f} | {np.std(finals):>10.1f} "
                f"| {np.median(finals):>+10.1f} | {np.sum(finals > 0):>4d}/{len(finals)} ({pct:.0f}%)"
            )
        print("  " + "-" * len(header))

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
