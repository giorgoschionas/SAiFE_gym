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
import argparse
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import gymnasium
import numpy as np
from scipy import stats
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
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel, OrnsteinUhlenbeckMidpriceModel, GeometricBrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import LiquidityKernelArrivalModel
from SAiFE_gym.agents.BaselineAgents import (
    UniformAllocationAgent, DeployOnceAgent, CarteaPLAgent, ArrivalRebalanceAgent,
    DoNothingAgent,
)
from SAiFE_gym.agents.PolicyGradientAgent import PolicyGradientAgent
from SAiFE_gym.agents.SbAgent import SbAgent
from SAiFE_gym.rewards.RewardFunctions import PnL, RunningInventoryPenalty, ExponentialUtility
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    MISPRICING_KEY, LP_LOWER_OFFSET_KEY, LP_UPPER_OFFSET_KEY, GAS_COST_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
    LP_LIQUIDITY_KEY, LP_EVER_DEPLOYED_KEY,
    LP_TOKEN0_AMOUNT_KEY, LP_TOKEN1_AMOUNT_KEY,
    PORTFOLIO_VALUE_KEY, INITIAL_WEALTH_KEY,
)

# ============================================================================
# Configuration — loaded from experiment_config.py via --config-idx
# ============================================================================

from experiment_config import (
    get_combination, count_combinations, SWEEP_VARS,
)

_parser = argparse.ArgumentParser()
_parser.add_argument(
    '--config-idx', type=int, default=0,
    help='Index of the combination from experiment_config.py to run.',
)
_args, _ = _parser.parse_known_args()
CONFIG_IDX = _args.config_idx
_combo = get_combination(CONFIG_IDX)
# Inject every sweep variable into module globals (SEED, TAU, ENABLE_AGENTS, …).
globals().update(_combo)

# Derived from the loaded combo.
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
    LP_COLLECTED_FEES0_KEY,  # cumulative LP fees in token0 units
    LP_COLLECTED_FEES1_KEY,  # cumulative LP fees in token1 units (numéraire)
    LP_TOKEN0_AMOUNT_KEY,    # LP inventory in the risky asset
    LP_TOKEN1_AMOUNT_KEY,    # LP inventory in the numéraire
]

# Output directory: one folder per (array) job under notebooks/results/.
_array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
_array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
if _array_job and _array_task:
    job_id = f"{_array_job}_{_array_task}"
else:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'results', job_id)

AGENT_NAMES = [name for name, on in ENABLE_AGENTS.items() if on]
AGENT_COLORS = {
    'DeployNarrow':  "#76e1ff",
    'Uniform':    '#1f77b4',
    'DeployWide': '#9467bd',
    'ArrivalRebalance': "#ff7dd8",
    'CDM': '#ff7f0e',
    'REINFORCE':  "#000000",
    'PPO':        "#fb4545",
    'PPO_narrow': "#62f848",
    'SAC':        '#17becf',
    'DQN':        '#8c564b',
}


def save_config_to_file(path: str):
    """Dump the active combination as a human-readable text file."""
    def fmt(v):
        if isinstance(v, np.ndarray):
            return f"np.array({v.tolist()!r})"
        return repr(v)
    total = count_combinations()
    with open(path, 'w') as f:
        f.write(f"# Experiment config (config_idx={CONFIG_IDX} of {total} combination(s))\n")
        f.write(f"# Job ID: {job_id}\n\n")
        f.write("# ── Sweep variables ──\n")
        for name in SWEEP_VARS:
            f.write(f"{name} = {fmt(_combo[name])}\n")
        f.write("\n# ── Derived ──\n")
        f.write(f"SB3_TOTAL_TIMESTEPS = {SB3_TOTAL_TIMESTEPS}\n")
        f.write(f"SB3_OBS_KEYS = {list(SB3_OBS_KEYS)!r}\n")


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

    midprice_model = GeometricBrownianMotionMidpriceModel(
        drift=DRIFT, volatility=VOLATILITY, initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME, step_size=step_size,
        num_trajectories=num_trajectories, seed=seed,
    )


    #midprice_model = OrnsteinUhlenbeckMidpriceModel(                                                     
    #    mean_reversion=0.0000000000000000001,           # κ — pull strength toward θ                                       
    #    long_term_mean=INITIAL_PRICE, # θ — defaults to INITIAL_PRICE if omitted                                                                                               
    #    volatility=VOLATILITY,
    #    num_trajectories=num_trajectories, seed=seed,
    #    initial_price=INITIAL_PRICE,
    #    terminal_time=TERMINAL_TIME, step_size=step_size
    #)

    
    arrival_model = LiquidityKernelArrivalModel(
        alpha=alpha, beta=0.001, K=100, liquidity_scale=LIQUIDITY_SCALE,
        step_size=step_size, num_trajectories=num_trajectories,
        seed=seed + 1 if seed else None,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model, arrival_model=arrival_model,
        num_trajectories=num_trajectories, fee_tier=FEE_TIER, tau=TAU,
        num_ticks=5000, exponential_value=EXP_VALUE,
        seed=seed + 2 if seed else None,
    )
    if REWARD_KIND == 'pnl':
        reward_function = PnL()
    elif REWARD_KIND == 'inventory':
        reward_function = RunningInventoryPenalty(
            per_step_inventory_aversion=INVENTORY_PHI,
            terminal_inventory_aversion=INVENTORY_TERMINAL_AVERSION,
            inventory_exponent=INVENTORY_EXPONENT,
        )
    elif REWARD_KIND == 'exponential':
        reward_function = ExponentialUtility(risk_aversion=EXP_RISK_AVERSION)
    else:
        raise ValueError(f"unknown REWARD_KIND={REWARD_KIND!r}")
    return AMMEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS,
        initial_wealth=INITIAL_WEALTH,
        reward_function=reward_function, model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        initial_pool_price=INITIAL_POOL_PRICE,
        seed=seed,
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


class StructuredMultiDiscreteVecEnv(VecEnv):
    """Re-exposes a VecEnv with a ``MultiDiscrete([2·tau+1, tau, 2])`` action space:

      dim 0: center_idx     ∈ {0, …, 2·tau}    → center tick ∈ {-tau, …, +tau}
      dim 1: half_width_idx ∈ {0, …, tau-1}    → half_width ∈ {1, …, tau}
      dim 2: hold_idx       ∈ {0, 1}           → hold flag ∈ {-1, +1}

    Output to the wrapped env is ``[center − half_width, center + half_width,
    ±1]``, so:
      * ``lower < upper`` is guaranteed by construction (no
        ``validate_action`` auto-correction needed),
      * the hold flag is a true ``Categorical(2)`` from the policy's
        perspective, with a clean Bernoulli-equivalent log-prob (no
        sign-thresholding waste),
      * the discretisation matches the env's actual action granularity:
        ``validate_action`` rounds to integer ticks anyway, so a continuous
        head over the same range carries no extra information.

    SB3 PPO supports ``MultiDiscrete`` natively via three independent
    ``Categorical`` distributions — no custom policy needed. Initial
    exploration is uniform over all ``(2·tau+1)·tau·2`` valid actions.
    """

    def __init__(self, vec_env: VecEnv, tau: int):
        self._wrapped = vec_env
        self.tau = int(tau)
        act_space = gymnasium.spaces.MultiDiscrete([2 * self.tau + 1, self.tau, 2])
        super().__init__(vec_env.num_envs, vec_env.observation_space, act_space)

    def unscale(self, action: np.ndarray) -> np.ndarray:
        """Map MultiDiscrete indices to the env's ``[lower, upper, hold_flag]``."""
        a = np.asarray(action, dtype=np.int64)
        center = a[..., 0] - self.tau              # {-tau, …, +tau}
        half_width = a[..., 1] + 1                  # {1, …, tau}
        hold = np.where(a[..., 2] == 0, -1.0, 1.0).astype(np.float32)
        lower = np.clip(center - half_width, -self.tau, self.tau - 1).astype(np.float32)
        upper = np.clip(center + half_width, -self.tau + 1, self.tau).astype(np.float32)
        return np.stack([lower, upper, hold], axis=-1)

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


class NarrowMultiDiscreteVecEnv(VecEnv):
    """``MultiDiscrete([2·tau+1, 2])`` wrapper for the PPO_narrow agent.

    Same idea as ``StructuredMultiDiscreteVecEnv`` but the half-width is fixed
    to the minimum allowed (= 1), so the policy only chooses (center, hold).
    The emitted env action is ``[center − 1, center + 1, ±1]`` — always a
    2-tick range centred on the chosen tick.

      dim 0: center_idx ∈ {0, …, 2·tau}  →  center tick ∈ {-tau, …, +tau}
      dim 1: hold_idx   ∈ {0, 1}         →  hold flag ∈ {-1, +1}
    """

    def __init__(self, vec_env: VecEnv, tau: int):
        self._wrapped = vec_env
        self.tau = int(tau)
        act_space = gymnasium.spaces.MultiDiscrete([2 * self.tau + 1, 2])
        super().__init__(vec_env.num_envs, vec_env.observation_space, act_space)

    def unscale(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.int64)
        center = a[..., 0] - self.tau              # {-tau, …, +tau}
        hold = np.where(a[..., 1] == 0, -1.0, 1.0).astype(np.float32)
        # Fixed half-width of 1 — smallest non-degenerate range.
        lower = np.clip(center - 1, -self.tau, self.tau - 1).astype(np.float32)
        upper = np.clip(center + 1, -self.tau + 1, self.tau).astype(np.float32)
        return np.stack([lower, upper, hold], axis=-1)

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
    """Run one full episode; return per-trajectory cumulative PnL.

    PnL is tracked via PORTFOLIO_VALUE_KEY deltas so it stays correct under
    any reward function (PnL, RunningInventoryPenalty, ExponentialUtility, ...).
    The agent's actual reward signal is ignored here — we want the apples-to-
    apples PnL axis for comparison plots regardless of training objective.
    """
    state, _ = env.reset()
    n = env.num_trajectories
    cum_pnl = np.zeros(n, dtype=np.float64)
    terminated = np.zeros(n, dtype=bool)
    while not np.any(terminated):
        prev_pv = state[PORTFOLIO_VALUE_KEY].astype(np.float64).copy()
        action = get_action_fn(state)
        state, _, terminated, _, _ = env.step(action)
        cum_pnl += state[PORTFOLIO_VALUE_KEY].astype(np.float64) - prev_pv
    return cum_pnl


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
    # ``cum_fees`` / ``cum_gas`` / ``cum_hodl_pnl`` follow Option B for IL
    # attribution: HODL reference (``hodl_t0``, ``hodl_t1``) resets at every
    # rebalance to the LP's post-rebalance token composition; between rebalances
    # the reference stays fixed and HODL PnL contributions accumulate as
    # ``hodl_t0 · ΔP`` per step (token1 is the numéraire so ``hodl_t1`` is
    # invariant in value). IL is backed out from the identity
    # ``Total_PnL = HODL_PnL − IL + Fees − Gas``.
    data = {k: [] for k in [
        'time', 'pool_price', 'midprice',
        'position_lower_price', 'position_upper_price',
        'action_lower', 'action_upper',
        'actual_lower_offset', 'actual_upper_offset',
        'hold_flag',
        'reward', 'cumulative_pnl', 'cumulative_utility',
        'trade_count',
        'cum_fees', 'cum_gas', 'cum_il', 'cum_hodl_pnl',
    ]}
    state, _ = env.reset()
    cum_pnl = 0.0       # true PnL via ΔPORTFOLIO_VALUE (reward-function-agnostic)
    cum_utility = 0.0   # agent's actual reward signal (= cum_pnl under PnL())
    trade_count = 0
    hold_action = np.array([[0.0, 1.0, 1.0]], dtype=np.float32)

    gas_cost = float(env.model_dynamics.gas_cost)
    cum_fees = 0.0
    cum_gas = 0.0
    cum_hodl_pnl = 0.0
    prev_c0 = float(state[LP_COLLECTED_FEES0_KEY][0])
    prev_c1 = float(state[LP_COLLECTED_FEES1_KEY][0])
    hodl_t0 = None  # None until first deployment
    hodl_t1 = None

    for step_idx in range(env.n_steps):
        data['time'].append(state[TIME_KEY][0])
        data['pool_price'].append(state[POOL_SQRT_PRICE_KEY][0] ** 2)
        data['midprice'].append(state[ASSET_PRICE_KEY][0])

        is_decision = (step_idx % decision_stride) == 0
        action = get_action_fn(state) if is_decision else hold_action
        data['action_lower'].append(float(action[0, 0]))
        data['action_upper'].append(float(action[0, 1]))
        data['hold_flag'].append(float(action[0, 2]) if action.shape[1] >= 3 else -1.0)

        # Snapshot pre-step quantities needed for attribution.
        pre_price = float(state[ASSET_PRICE_KEY][0])
        prev_pv = float(state[PORTFOLIO_VALUE_KEY][0])
        had_position = bool(state[LP_LIQUIDITY_KEY][0] > 0)
        hold_flag = float(action[0, 2]) if action.shape[1] >= 3 else -1.0

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

        # True PnL is ΔPORTFOLIO_VALUE (reward-function-agnostic); reward may
        # carry inventory/utility penalties under non-PnL reward functions.
        cum_pnl += float(state[PORTFOLIO_VALUE_KEY][0]) - prev_pv
        cum_utility += reward[0]
        data['reward'].append(reward[0])
        data['cumulative_pnl'].append(cum_pnl)
        data['cumulative_utility'].append(cum_utility)
        data['trade_count'].append(trade_count)

        # --- Attribution ---
        new_price = float(state[ASSET_PRICE_KEY][0])

        # Fees this step: value the LP_COLLECTED delta at end-of-step price.
        new_c0 = float(state[LP_COLLECTED_FEES0_KEY][0])
        new_c1 = float(state[LP_COLLECTED_FEES1_KEY][0])
        cum_fees += (new_c0 - prev_c0) * new_price + (new_c1 - prev_c1)
        prev_c0, prev_c1 = new_c0, new_c1

        # Gas this step: only paid on rebalance when a position already existed
        # (matches _rebalance's has_position gate; first deployment is free).
        rebalanced = (hold_flag <= 0) and had_position
        if rebalanced:
            cum_gas += gas_cost

        # HODL PnL contribution for this step uses the *pre-step* reference;
        # the reference is refreshed afterwards if a rebalance happened (or this
        # is the first deployment), so it stays valid for the next period.
        if hodl_t0 is not None:
            cum_hodl_pnl += hodl_t0 * (new_price - pre_price)

        first_deploy = (hodl_t0 is None) and bool(state[LP_EVER_DEPLOYED_KEY][0])
        if rebalanced or first_deploy:
            new_t0 = float(state[LP_TOKEN0_AMOUNT_KEY][0])
            new_pv = float(state[PORTFOLIO_VALUE_KEY][0])
            hodl_t0 = new_t0
            hodl_t1 = new_pv - new_t0 * new_price

        # IL backed out from the accounting identity.
        cum_il = cum_hodl_pnl + cum_fees - cum_gas - cum_pnl

        data['cum_fees'].append(cum_fees)
        data['cum_gas'].append(cum_gas)
        data['cum_il'].append(cum_il)
        data['cum_hodl_pnl'].append(cum_hodl_pnl)

        if terminated[0]:
            break
        if max_trades is not None and trade_count >= max_trades:
            break

    return {k: np.array(v) for k, v in data.items()}


def evaluate_on_trajectories_with_attribution(env, get_action_fn):
    """Vectorized version of ``evaluate_on_trajectories`` that also returns the
    per-trajectory attribution (Fees / Gas / IL / HODL PnL) at episode end.

    Uses Option B: HODL reference is the LP's post-rebalance token composition,
    refreshed at every rebalance. IL backed out from
    ``Total_PnL = HODL_PnL − IL + Fees − Gas``.

    Returns a dict with arrays of shape (num_trajectories,):
        'pnl'       — cumulative reward (= ΔPortfolioValue)
        'fees'      — cumulative fees in token1 (numéraire) units
        'gas'       — cumulative gas paid
        'il'        — cumulative impermanent loss (positive = cost)
        'hodl_pnl'  — cumulative HODL PnL of the (resetting) reference portfolio
    """
    state, _ = env.reset()
    n = env.num_trajectories
    gas_cost = float(env.model_dynamics.gas_cost)

    cum_pnl = np.zeros(n)      # true PnL via ΔPORTFOLIO_VALUE
    cum_utility = np.zeros(n)  # agent's reward signal (= cum_pnl under PnL())
    cum_fees = np.zeros(n)
    cum_gas = np.zeros(n)
    cum_hodl_pnl = np.zeros(n)
    prev_c0 = state[LP_COLLECTED_FEES0_KEY].astype(np.float64).copy()
    prev_c1 = state[LP_COLLECTED_FEES1_KEY].astype(np.float64).copy()
    hodl_t0 = np.zeros(n)
    hodl_t1 = np.zeros(n)
    ref_set = np.zeros(n, dtype=bool)
    # Per-trajectory spread and center-asymmetry accumulators (post-step
    # state). Online mean/var via Σx, Σx² so we only carry two arrays per
    # metric instead of a (T, n) buffer. ``center`` is the position midpoint
    # relative to the current tick — i.e. ``(upper_offset + lower_offset) / 2``
    # — so 0 is symmetric, positive means the position is skewed above the
    # current price, negative below.
    spread_sum = np.zeros(n, dtype=np.float64)
    spread_sumsq = np.zeros(n, dtype=np.float64)
    center_sum = np.zeros(n, dtype=np.float64)
    center_sumsq = np.zeros(n, dtype=np.float64)
    spread_count = 0
    # Deploy-time center accumulators: sampled only at refresh events
    # (rebalance or first deploy). Captures what the agent chose at
    # deployment, independent of how price drift bends the offset during
    # holds. ``deploy_count`` is per-trajectory because rebalance schedules
    # differ across trajectories.
    deploy_center_sum = np.zeros(n, dtype=np.float64)
    deploy_center_sumsq = np.zeros(n, dtype=np.float64)
    deploy_count = np.zeros(n, dtype=np.int64)
    # Per-trajectory rebalance counter. Uses the same mask as the gas charge
    # (first deploy excluded). spread_count doubles as decisions/episode since
    # the outer loop iterates once per agent decision — DecisionStride
    # collapses `stride` env steps into one outer step.
    rebalance_count = np.zeros(n, dtype=np.int64)

    terminated = np.zeros(n, dtype=bool)
    while not np.any(terminated):
        pre_price = state[ASSET_PRICE_KEY].astype(np.float64).copy()
        prev_pv = state[PORTFOLIO_VALUE_KEY].astype(np.float64).copy()
        had_position = state[LP_LIQUIDITY_KEY] > 0
        # Snapshot current_tick *before* env.step — this is the tick the agent
        # sees at decision time. Used as the reference for deploy-time center
        # so the metric isn't biased by trades that move current_tick during
        # the (possibly multi-step) DecisionStride window.
        pre_current_tick = state[POOL_CURRENT_TICK_KEY].astype(np.float64).copy()

        action = get_action_fn(state)
        hold_flags = action[:, 2] if action.shape[1] >= 3 else -np.ones(n)

        state, reward, terminated, _, _ = env.step(action)
        # PnL tracked from portfolio value (reward-function-agnostic); utility
        # captures whatever risk/penalty terms the reward function added.
        cum_pnl += state[PORTFOLIO_VALUE_KEY].astype(np.float64) - prev_pv
        cum_utility += reward

        # Post-step spread (upper − lower) and center asymmetry
        # ((upper + lower)/2 − current_tick) reflect the agent's actually
        # deployed position after any rebalance fired in this step.
        upper = state[LP_TICK_UPPER_KEY].astype(np.float64)
        lower = state[LP_TICK_LOWER_KEY].astype(np.float64)
        current = state[POOL_CURRENT_TICK_KEY].astype(np.float64)
        spread_step = upper - lower
        center_step = 0.5 * (upper + lower) - current
        spread_sum += spread_step
        spread_sumsq += spread_step * spread_step
        center_sum += center_step
        center_sumsq += center_step * center_step
        spread_count += 1

        new_price = state[ASSET_PRICE_KEY].astype(np.float64)

        # Fees
        new_c0 = state[LP_COLLECTED_FEES0_KEY].astype(np.float64)
        new_c1 = state[LP_COLLECTED_FEES1_KEY].astype(np.float64)
        cum_fees += (new_c0 - prev_c0) * new_price + (new_c1 - prev_c1)
        prev_c0[:] = new_c0
        prev_c1[:] = new_c1

        # Gas
        rebalanced = (hold_flags <= 0) & had_position
        cum_gas += rebalanced.astype(np.float64) * gas_cost
        rebalance_count += rebalanced.astype(np.int64)

        # HODL PnL contribution (only where reference is set)
        cum_hodl_pnl += np.where(ref_set, hodl_t0 * (new_price - pre_price), 0.0)

        # Refresh reference at every rebalance and on first deployment.
        first_deploy = state[LP_EVER_DEPLOYED_KEY] & (~ref_set)
        refresh = rebalanced | first_deploy
        if np.any(refresh):
            new_t0 = state[LP_TOKEN0_AMOUNT_KEY].astype(np.float64)
            new_pv = state[PORTFOLIO_VALUE_KEY].astype(np.float64)
            new_t1 = new_pv - new_t0 * new_price
            hodl_t0 = np.where(refresh, new_t0, hodl_t0)
            hodl_t1 = np.where(refresh, new_t1, hodl_t1)
            ref_set = ref_set | first_deploy  # latches True once set

            # Sample the just-deployed position's center against the *pre-step*
            # current_tick (the tick the agent saw at decision time). LP bounds
            # in state are absolute and reflect the new range; subtracting the
            # pre-step tick recovers the offset the agent quoted, independent
            # of any same-step trades that moved current_tick afterwards.
            deploy_center_step = 0.5 * (upper + lower) - pre_current_tick
            deploy_center_sum += np.where(refresh, deploy_center_step, 0.0)
            deploy_center_sumsq += np.where(
                refresh, deploy_center_step * deploy_center_step, 0.0,
            )
            deploy_count += refresh.astype(np.int64)

    cum_il = cum_hodl_pnl + cum_fees - cum_gas - cum_pnl

    # Per-trajectory spread / center stats: time-averaged value and temporal
    # std within each trajectory. Var clamped at 0 to absorb numerical noise.
    denom = max(spread_count, 1)
    mean_spread = spread_sum / denom
    var_spread = spread_sumsq / denom - mean_spread * mean_spread
    temporal_std_spread = np.sqrt(np.maximum(var_spread, 0.0))
    mean_center = center_sum / denom
    var_center = center_sumsq / denom - mean_center * mean_center
    temporal_std_center = np.sqrt(np.maximum(var_center, 0.0))

    # Deploy-time center stats: per-trajectory mean and within-traj std,
    # computed only over refresh events. Trajectories that never deployed
    # (possible only for synthetic stances like DoNothing(hold_cash=True))
    # get NaN so they don't pollute the averages.
    has_deploy = deploy_count > 0
    safe_deploy_count = np.where(has_deploy, deploy_count, 1).astype(np.float64)
    mean_deploy_center = np.where(
        has_deploy, deploy_center_sum / safe_deploy_count, np.nan,
    )
    var_deploy_center = np.where(
        has_deploy,
        deploy_center_sumsq / safe_deploy_count - mean_deploy_center * mean_deploy_center,
        np.nan,
    )
    temporal_std_deploy_center = np.sqrt(np.maximum(var_deploy_center, 0.0))

    return {
        'pnl': cum_pnl,
        'utility': cum_utility,
        'fees': cum_fees,
        'gas': cum_gas,
        'il': cum_il,
        'hodl_pnl': cum_hodl_pnl,
        'mean_spread': mean_spread,
        'temporal_std_spread': temporal_std_spread,
        'mean_center': mean_center,
        'temporal_std_center': temporal_std_center,
        'mean_deploy_center': mean_deploy_center,
        'temporal_std_deploy_center': temporal_std_deploy_center,
        'deploy_count': deploy_count,                 # per-traj, shape (n,)
        'rebalance_count': rebalance_count,           # per-traj, shape (n,)
        'decision_count': int(spread_count),          # scalar: decisions/episode
    }

# ============================================================================
# Stdout logging — capture every print to a text file as well
# ============================================================================

class _StdoutTee:
    """File-like object that writes to multiple underlying streams.

    Used to mirror everything printed to the terminal into a results file,
    without touching any existing print() call. Install via:
        sys.stdout = _StdoutTee(sys.stdout, open(path, 'w'))
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        # Some libraries (tqdm) check this to decide rich vs plain output.
        # Mirror the first stream's behavior so tqdm keeps using carriage
        # returns in the terminal — the file ends up with the final state
        # only, but that's the conventional log format.
        return getattr(self.streams[0], 'isatty', lambda: False)()


# ============================================================================
# Console tables
# ============================================================================

def print_spread_table(attribution_results):
    """Print the agent quoting-behavior tables.

    Two tables are printed: one sampled at every env step (time-averaged
    behaviour as the LP carries its position through the episode), one
    sampled only at refresh events (rebalance + first deploy — captures
    what the agent chose at deployment, before price drift moves the
    offset).

    Step-averaged table (sampled every env step):
      Spread (width = upper − lower, in ticks):
        - Mean spread:          E_{traj, step}[upper − lower]
        - Std (within episode): E_{traj}[ Std_step(spread) ]
        - Std (between traj):   Std_{traj}[ E_step(spread) ]
      Center asymmetry (center = (upper + lower)/2 − current_tick, in ticks;
      0 = symmetric quote, +/− = position skewed above/below current price):
        - Mean center:          E_{traj, step}[center]
        - Std (within episode): E_{traj}[ Std_step(center) ]
        - Std (between traj):   Std_{traj}[ E_step(center) ]
      Rebalance columns:
        - Rebalances/ep: mean across trajs of rebalance count per episode
                         (first deploy not counted — matches gas convention)
        - Rate (%):      Rebalances/ep / decisions_per_ep × 100
                         (per-decision-opportunity rate; comparable across
                         strides)

    Deploy-time table (sampled only on rebalance + first deploy):
        - Mean center:          E_{traj}[ E_deploy(center) ] over refreshes
        - Std (within episode): E_{traj}[ Std_deploy(center) ]
                                (trajectories with a single deploy event
                                contribute 0)
        - Std (between traj):   Std_{traj}[ E_deploy(center) ]
        - Deploys/ep:           mean deploy events per episode
                                (= rebalance_count + 1 when the LP ever
                                deploys; first deploy is included here)
    """
    active = [n for n in AGENT_NAMES if n in attribution_results]
    if not active:
        return

    # ----- Step-averaged table -----
    header = (
        f"  {'Agent':<18} | {'Mean spread':>12} | "
        f"{'Std (within ep)':>16} | {'Std (between traj)':>19} | "
        f"{'Mean center':>12} | {'Std (within ep)':>16} | "
        f"{'Std (between traj)':>19} | "
        f"{'Rebalances/ep':>14} | {'Rate (%)':>9}"
    )
    print("\nQuoting behavior across eval trajectories (sampled every step):")
    print("  " + "-" * (len(header) - 2))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in active:
        r = attribution_results[name]
        mean_s = r['mean_spread']
        temp_std = r['temporal_std_spread']
        mean_c = r['mean_center']
        temp_std_c = r['temporal_std_center']
        reb_count = r['rebalance_count']
        decisions = max(int(r['decision_count']), 1)

        col_mean = float(mean_s.mean())
        col_std_within = float(temp_std.mean())
        col_std_between = float(mean_s.std())
        col_mean_c = float(mean_c.mean())
        col_std_within_c = float(temp_std_c.mean())
        col_std_between_c = float(mean_c.std())
        col_reb_per_ep = float(reb_count.mean())
        col_rate = 100.0 * col_reb_per_ep / decisions
        print(
            f"  {name:<18} | {col_mean:>12.3f} | "
            f"{col_std_within:>16.3f} | {col_std_between:>19.3f} | "
            f"{col_mean_c:>+12.3f} | {col_std_within_c:>16.3f} | "
            f"{col_std_between_c:>19.3f} | "
            f"{col_reb_per_ep:>14.2f} | {col_rate:>9.2f}"
        )
    print("  " + "-" * (len(header) - 2))

    # ----- Deploy-time table -----
    header2 = (
        f"  {'Agent':<18} | {'Mean center':>12} | "
        f"{'Std (within ep)':>16} | {'Std (between traj)':>19} | "
        f"{'Deploys/ep':>11}"
    )
    print("\nDeploy-time center asymmetry (sampled only on rebalance + first deploy):")
    print("  " + "-" * (len(header2) - 2))
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for name in active:
        r = attribution_results[name]
        mean_dc = r['mean_deploy_center']
        temp_std_dc = r['temporal_std_deploy_center']
        dep_count = r['deploy_count']

        # nan-safe aggregation: trajs that never deployed are dropped from
        # the mean/std rather than coerced to 0 (which would bias toward 0).
        col_mean_dc = float(np.nanmean(mean_dc)) if np.any(~np.isnan(mean_dc)) else float('nan')
        col_std_within_dc = float(np.nanmean(temp_std_dc)) if np.any(~np.isnan(temp_std_dc)) else float('nan')
        col_std_between_dc = float(np.nanstd(mean_dc)) if np.any(~np.isnan(mean_dc)) else float('nan')
        col_deploys = float(dep_count.mean())
        print(
            f"  {name:<18} | {col_mean_dc:>+12.3f} | "
            f"{col_std_within_dc:>16.3f} | {col_std_between_dc:>19.3f} | "
            f"{col_deploys:>11.2f}"
        )
    print("  " + "-" * (len(header2) - 2))


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
    # Match the inclination used in the pnl_attribution boxplot / means
    # figures so agent labels read consistently across plots.
    plt.setp(ax1.get_xticklabels(), rotation=24)
    #ax1.set_title('PnL Distribution')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, f'pnl_boxplot.png')
    fig1.savefig(path1, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    #print(f"  Saved: {path1}")

    # Histogram. Two-pass render per agent:
    #   pass 1 — `stepfilled` with very low alpha for a translucent area cue;
    #   pass 2 — `step` with full opacity for a thin sharp outline.
    # Keeping the edge in its own call means the outline stays fully
    # saturated regardless of how transparent the fill is (matplotlib's
    # `alpha` kwarg on `stepfilled` dims both edge and face together).
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    all_pnl = np.concatenate(list(pnl_results.values()))
    lo, hi = np.percentile(all_pnl, [1, 99])
    bins = np.linspace(lo, hi, 50)
    for name in agents:
        # Translucent fill (no edge — second pass draws it).
        ax2.hist(
            pnl_results[name], bins=bins,
            histtype='stepfilled',
            facecolor=AGENT_COLORS[name], alpha=0.12,
            edgecolor='none', density=True,
        )
        # Crisp opaque outline; this is what carries the legend entry.
        ax2.hist(
            pnl_results[name], bins=bins,
            histtype='step',
            edgecolor=AGENT_COLORS[name], linewidth=1.1,
            alpha=1.0, density=True,
            label=f"{name} (mean={np.mean(pnl_results[name]):+.1f})",
        )
        ax2.axvline(
            np.mean(pnl_results[name]),
            color=AGENT_COLORS[name], linestyle='--', linewidth=1.2,
        )
    ax2.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Cumulative PnL', fontsize=16)
    ax2.set_ylabel('Density', fontsize=16)
    ax2.tick_params(axis='both', labelsize=14)
    #ax2.set_title('PnL Histogram')
    ax2.legend(fontsize=16, loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, f'pnl_histogram.png')
    fig2.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    #print(f"  Saved: {path2}")


def plot_utility_distribution(utility_results):
    """Box-plot and histogram of cumulative agent reward (utility), saved as separate figures.

    Mirrors plot_pnl_distribution but on the agent's reward signal — i.e. PnL
    minus the inventory/risk penalty when REWARD_KIND != 'pnl'.
    """
    agents = [n for n in AGENT_NAMES if n in utility_results]

    # Box plot
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    box_data = [utility_results[n] for n in agents]
    bp = ax1.boxplot(box_data, labels=agents, patch_artist=True, notch=False)
    for patch, name in zip(bp['boxes'], agents):
        patch.set_facecolor(AGENT_COLORS[name])
        patch.set_alpha(0.6)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_ylabel(f'Cumulative Utility ({REWARD_KIND})', fontsize=16)
    ax1.tick_params(axis='both', labelsize=14)
    # Match the inclination used in the pnl_attribution figures.
    plt.setp(ax1.get_xticklabels(), rotation=24)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, f'utility_boxplot.png')
    fig1.savefig(path1, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    #print(f"  Saved: {path1}")

    # Histogram. Two-pass render — translucent fill plus crisp outline,
    # matches plot_pnl_distribution's style.
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    all_u = np.concatenate(list(utility_results.values()))
    lo, hi = np.percentile(all_u, [1, 99])
    bins = np.linspace(lo, hi, 50) if hi > lo else 50
    for name in agents:
        ax2.hist(
            utility_results[name], bins=bins,
            histtype='stepfilled',
            facecolor=AGENT_COLORS[name], alpha=0.12,
            edgecolor='none', density=True,
        )
        ax2.hist(
            utility_results[name], bins=bins,
            histtype='step',
            edgecolor=AGENT_COLORS[name], linewidth=1.3,
            alpha=1.0, density=True,
            label=f"{name} (mean={np.mean(utility_results[name]):+.3g})",
        )
        ax2.axvline(
            np.mean(utility_results[name]),
            color=AGENT_COLORS[name], linestyle='--', linewidth=1.2,
        )
    ax2.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel(f'Cumulative Utility ({REWARD_KIND})', fontsize=16)
    ax2.set_ylabel('Density', fontsize=16)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.legend(fontsize=16, loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, f'utility_histogram.png')
    fig2.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    #print(f"  Saved: {path2}")


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
        #ax.set_title(name)
        ax.legend(fontsize=16, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'price_evolution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    #print(f"  Saved: {path}")

    # Save standalone figure per trained RL agent (all sims overlaid)
    rl_agents = [n for n in agents if n in ('REINFORCE', 'PPO', 'PPO_narrow', 'SAC', 'DQN')]
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
        #ax_rl.set_title(name)
        ax_rl.legend(fontsize=16, loc='lower right')
        ax_rl.grid(True, alpha=0.3)
        plt.tight_layout()
        path_rl = os.path.join(FIGURES_DIR, f'price_evolution_{name.lower()}.png')
        fig_rl.savefig(path_rl, dpi=200, bbox_inches='tight')
        plt.close(fig_rl)
        #print(f"  Saved: {path_rl}")

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
                #ax.set_title(name)
                ax.legend(fontsize=16, loc='lower right')
                ax.grid(True, alpha=0.3)

                # Final PnL for this sim/agent, top-right of the panel.
                final_pnl = d['cumulative_pnl'][-1]
                ax.text(
                    0.97, 0.97, f'Final PnL: {final_pnl:+.2f}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=14, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color, alpha=0.85),
                )

            plt.tight_layout()
            path_i = os.path.join(FIGURES_DIR, f'price_evolution_sim{si + 1}.png')
            fig_i.savefig(path_i, dpi=200, bbox_inches='tight')
            plt.close(fig_i)
            #print(f"  Saved: {path_i}")


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
    path = os.path.join(FIGURES_DIR, f'pnl_evolution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    #print(f"  Saved: {path}")


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
    path = os.path.join(FIGURES_DIR, f'training_rewards.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    #print(f"  Saved: {path}")


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
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'position_offsets.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    #print(f"  Saved: {path}")


def plot_pnl_attribution_individual(single_data):
    """Per-agent time series of the PnL decomposition for each individual sim.

    Each subplot shows four cumulative quantities over the episode:
      Total PnL (black)         — what env reports as cumulative reward
      Cumulative Fees (green)   — fees earned, valued in token1
      −Cumulative IL (red)      — impermanent loss (plotted negative as a cost)
      −Cumulative Gas (orange)  — rebalance gas (plotted negative as a cost)

    The HODL PnL line (dotted gray) is added for reference — it's the
    "would-have-happened-anyway" market drift of the Option B HODL portfolio.
    With Option B, the identity
        Total_PnL = HODL_PnL − IL + Fees − Gas
    holds exactly to numerical precision.
    """
    agents = [n for n in AGENT_NAMES if n in single_data]
    if not agents:
        return
    n_agents = len(agents)
    ncols = 2
    nrows = (n_agents + ncols - 1) // ncols
    n_sims = max(len(v) for v in single_data.values())

    # Combined overview: per-agent panel with all sims overlaid (light alpha).
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat[n_agents:]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, agents):
        sim_alpha = max(0.15, 0.8 / len(single_data[name]))
        for si, d in enumerate(single_data[name]):
            lbl = (lambda s: s if si == 0 else None)
            ax.plot(d['time'], d['cumulative_pnl'], 'k-', linewidth=1.4,
                    alpha=sim_alpha, label=lbl('Total PnL'))
            ax.plot(d['time'], d['cum_fees'], color='#2ca02c', linewidth=1.2,
                    alpha=sim_alpha, label=lbl('Fees'))
            ax.plot(d['time'], -d['cum_il'], color='#d62728', linewidth=1.2,
                    alpha=sim_alpha, label=lbl('−IL'))
            ax.plot(d['time'], -d['cum_gas'], color='#ff7f0e', linewidth=1.2,
                    alpha=sim_alpha, label=lbl('−Gas'))
            ax.plot(d['time'], d['cum_hodl_pnl'], color='gray',
                    linestyle=':', linewidth=1.0, alpha=sim_alpha,
                    label=lbl('HODL PnL'))
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.6, alpha=0.4)
        ax.set_xlabel('Time', fontsize=16)
        ax.set_ylabel('Cumulative value (token1)', fontsize=16)
        ax.set_title(name)
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(fontsize=16, loc='upper left')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'pnl_attribution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    #print(f"  Saved: {path}")

    # Per-sim breakdown (one figure per sim, panels per agent).
    if n_sims > 1:
        for si in range(n_sims):
            fig_i, axes_i = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
            axes_i = np.atleast_2d(axes_i)
            for ax in axes_i.flat[n_agents:]:
                ax.set_visible(False)
            for ax, name in zip(axes_i.flat, agents):
                if si >= len(single_data[name]):
                    continue
                d = single_data[name][si]
                ax.plot(d['time'], d['cumulative_pnl'], 'k-', linewidth=1.8,
                        label='Total PnL')
                ax.plot(d['time'], d['cum_fees'], color='#2ca02c', linewidth=1.5,
                        label='Fees')
                ax.plot(d['time'], -d['cum_il'], color='#d62728', linewidth=1.5,
                        label='−IL')
                ax.plot(d['time'], -d['cum_gas'], color='#ff7f0e', linewidth=1.5,
                        label='−Gas')
                ax.plot(d['time'], d['cum_hodl_pnl'], color='gray',
                        linestyle=':', linewidth=1.2, label='HODL PnL')
                ax.axhline(0, color='gray', linestyle='-', linewidth=0.6, alpha=0.4)
                # Final values pinned to the upper-left corner.
                ax.text(
                    0.03, 0.97,
                    f"PnL: {d['cumulative_pnl'][-1]:+.2f}\n"
                    f"Fees: {d['cum_fees'][-1]:+.2f}\n"
                    f"IL: {d['cum_il'][-1]:+.2f}\n"
                    f"Gas: {d['cum_gas'][-1]:+.2f}",
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.85),
                )
                ax.set_xlabel('Time', fontsize=16)
                ax.set_ylabel('Cumulative value (token1)', fontsize=16)
                ax.set_title(name)
                ax.tick_params(axis='both', labelsize=12)
                ax.legend(fontsize=16, loc='lower left')
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path_i = os.path.join(
                FIGURES_DIR, f'pnl_attribution_sim{si + 1}.png'
            )
            fig_i.savefig(path_i, dpi=200, bbox_inches='tight')
            plt.close(fig_i)
            #print(f"  Saved: {path_i}")


def plot_pnl_attribution_aggregate(attribution_results):
    """Distribution + mean decomposition of final attribution across eval trajectories.

    ``attribution_results`` is the dict returned by
    ``evaluate_on_trajectories_with_attribution``, keyed by agent name.

    Two figures:
      1. Three side-by-side box plots: distribution of final Fees / IL / Gas
         per agent across the eval trajectories.
      2. A stacked bar of *means*: Fees − IL − Gas attribution per agent, with
         the Total PnL marker overlaid for verification.
    """
    agents = [n for n in AGENT_NAMES if n in attribution_results]
    if not agents:
        return

    # ---- (1) Box plots of final fees / IL / gas per agent ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, title in zip(
        axes,
        ('fees', 'il', 'gas'),
        ('Cumulative Fees', 'Impermanent Loss', 'Gas paid'),
    ):
        box_data = [attribution_results[n][key] for n in agents]
        bp = ax.boxplot(box_data, labels=agents, patch_artist=True, notch=False)
        for patch, name in zip(bp['boxes'], agents):
            patch.set_facecolor(AGENT_COLORS[name])
            patch.set_alpha(0.6)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_ylabel(f'{title} (token1)', fontsize=16)
        ax.set_title(title)
        ax.tick_params(axis='both', labelsize=12)
        # Match the inclination used in pnl_attribution_means below.
        plt.setp(ax.get_xticklabels(), rotation=40)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, f'pnl_attribution_boxplots.png')
    fig.savefig(path1, dpi=200, bbox_inches='tight')
    plt.close(fig)
    #print(f"  Saved: {path1}")

    # ---- (2) Mean-decomposition bar chart ----
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(agents) + 4), 5))
    x = np.arange(len(agents))
    width = 0.6
    mean_fees = np.array([np.mean(attribution_results[n]['fees']) for n in agents])
    mean_il = np.array([np.mean(attribution_results[n]['il']) for n in agents])
    mean_gas = np.array([np.mean(attribution_results[n]['gas']) for n in agents])
    mean_pnl = np.array([np.mean(attribution_results[n]['pnl']) for n in agents])
    mean_hodl = np.array([np.mean(attribution_results[n]['hodl_pnl']) for n in agents])

    # Stacked: positive contributions (Fees, HODL_PnL) above 0; negatives (−IL, −Gas) below.
    ax.bar(x, mean_fees, width, color='#2ca02c', label='Fees', alpha=0.85)
    ax.bar(x, mean_hodl, width, bottom=mean_fees, color='gray',
           label='HODL PnL', alpha=0.65)
    ax.bar(x, -mean_il, width, color='#d62728', label='−IL', alpha=0.85)
    ax.bar(x, -mean_gas, width, bottom=-mean_il, color='#ff7f0e',
           label='−Gas', alpha=0.85)

    # Total PnL marker (should equal HODL_PnL − IL + Fees − Gas exactly).
    ax.scatter(x, mean_pnl, marker='D', s=80, color='black', zorder=5,
               label='Total PnL (mean)')

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=12)
    ax.set_ylabel('Mean across eval trajectories (token1)', fontsize=16)
    ax.set_title('PnL attribution (mean over eval trajectories)')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=16, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, f'pnl_attribution_means.png')
    fig.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    #print(f"  Saved: {path2}")

    # Console summary so the numbers are visible without opening the figure.
    header = (
        f"  {'Agent':<18} | {'PnL':>8} | {'HODL':>8} | {'Fees':>8} | "
        f"{'IL':>8} | {'Gas':>8}"
    )
    print("\n  Mean attribution per agent (token1 units):")
    print("  " + "-" * (len(header) - 2))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(agents):
        print(
            f"  {name:<18} | {mean_pnl[i]:>+8.2f} | {mean_hodl[i]:>+8.2f} "
            f"| {mean_fees[i]:>+8.2f} | {mean_il[i]:>+8.2f} | {mean_gas[i]:>+8.2f}"
        )
    print("  " + "-" * (len(header) - 2))


# ============================================================================
# Main
# ============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    _config_path = os.path.join(FIGURES_DIR, 'config.txt')
    save_config_to_file(_config_path)

    # Mirror every print() into a results .txt next to the figures. Wrapped in
    # try/finally further down so we always restore sys.stdout even on error.
    results_path = os.path.join(FIGURES_DIR, 'results.txt')
    _results_file = open(results_path, 'w', buffering=1)  # line-buffered
    _original_stdout = sys.stdout
    sys.stdout = _StdoutTee(_original_stdout, _results_file)
    print(f"Experiment {CONFIG_IDX} → {FIGURES_DIR}")
    print(f"  Config written to: {_config_path}")
    print(f"  Stdout logged to:  {results_path}")

    # Containers for trained RL models / agents
    reinforce_agent = None
    reinforce_rewards = []
    ppo_model = None
    ppo_action_wrapper = None
    ppo_reward_cb = EpisodeRewardCallback()
    ppo_narrow_model = None
    ppo_narrow_action_wrapper = None
    ppo_narrow_reward_cb = EpisodeRewardCallback()
    ppo_narrow_vec_normalize = None
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
            learning_rate=PPO_LEARNING_RATE,
            n_steps=N_STEPS // DECISION_STRIDE,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA, gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE, ent_coef=PPO_ENT_COEF,
            policy_kwargs=dict(net_arch=dict(pi=PPO_NET_ARCH, vf=PPO_NET_ARCH)),
            verbose=1, seed=SEED,
        )
        ppo_reward_cb = EpisodeRewardCallback()
        t0 = time.time()
        ppo_model.learn(total_timesteps=SB3_TOTAL_TIMESTEPS, callback=ppo_reward_cb)
        print(f"  PPO training time: {time.time() - t0:.1f}s")

    if ENABLE_AGENTS.get('PPO_narrow'):
        print("\n" + "=" * 60)
        print("Phase 1b': Training PPO_narrow agent (half_width fixed to 1)")
        print("=" * 60)

        ppo_narrow_env = create_environment(NUM_TRAJECTORIES_TRAIN, SEED)
        sb_train_env_narrow = DecisionStrideVecEnv(
            StableBaselinesAMMEnvironment(ppo_narrow_env, obs_keys=SB3_OBS_KEYS),
            stride=DECISION_STRIDE,
        )
        ppo_narrow_vec_normalize = VecNormalize(
            VecMonitor(sb_train_env_narrow),
            norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
        )
        # Half-width is fixed to 1, so only (center, hold) are learned. We always
        # use the narrow MultiDiscrete wrapper here regardless of the global
        # PPO_ACTION_WRAPPER setting — the fixed-width restriction is the whole
        # point of this agent.
        ppo_narrow_action_wrapper = NarrowMultiDiscreteVecEnv(
            ppo_narrow_vec_normalize, TAU,
        )

        ppo_narrow_model = PPO(
            "MlpPolicy", ppo_narrow_action_wrapper,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=N_STEPS // DECISION_STRIDE,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA, gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE, ent_coef=PPO_ENT_COEF,
            policy_kwargs=dict(net_arch=dict(pi=PPO_NET_ARCH, vf=PPO_NET_ARCH)),
            verbose=1, seed=SEED,
        )
        ppo_narrow_reward_cb = EpisodeRewardCallback()
        t0 = time.time()
        ppo_narrow_model.learn(
            total_timesteps=SB3_TOTAL_TIMESTEPS, callback=ppo_narrow_reward_cb,
        )
        print(f"  PPO_narrow training time: {time.time() - t0:.1f}s")

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
    # ``attribution_results[name]`` is the full {'pnl', 'fees', 'gas', 'il',
    # 'hodl_pnl'} dict from evaluate_on_trajectories_with_attribution.
    # ``pnl_results`` is derived from it for the existing significance tests
    # and PnL distribution plots.
    attribution_results = {}

    if ENABLE_AGENTS.get('DeployNarrow'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        attribution_results['DeployNarrow'] = evaluate_on_trajectories_with_attribution(
            env, DoNothingAgent(env).get_action,
        )

    if ENABLE_AGENTS.get('Uniform'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        attribution_results['Uniform'] = evaluate_on_trajectories_with_attribution(
            env, UniformAllocationAgent(env).get_action,
        )

    if ENABLE_AGENTS.get('DeployWide'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        attribution_results['DeployWide'] = evaluate_on_trajectories_with_attribution(
            env, DeployOnceAgent(
                env, lower_offset=DEPLOYONCE_LOWER, upper_offset=DEPLOYONCE_UPPER,
            ).get_action,
        )

    if ENABLE_AGENTS.get('ArrivalRebalance'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        attribution_results['ArrivalRebalance'] = evaluate_on_trajectories_with_attribution(
            env, ArrivalRebalanceAgent(
                env, rebalance_every=ARRIVAL_REBALANCE_EVERY,
                width=ARRIVAL_REBALANCE_WIDTH,
                lower_offset=ARRIVAL_REBALANCE_LOWER,
                upper_offset=ARRIVAL_REBALANCE_UPPER,
            ).get_action,
        )

    if ENABLE_AGENTS.get('CDM'):
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        attribution_results['CDM'] = evaluate_on_trajectories_with_attribution(
            env, CarteaPLAgent(
                env, gamma=GAMMA_CARTEA,
                rebalance_tolerance=REBALANCE_TOLERANCE_CARTEA,
                seed=SEED,
            ).get_action,
        )

    if ENABLE_AGENTS.get('REINFORCE') and reinforce_agent is not None:
        env = DecisionStrideEnv(
            create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED), stride=DECISION_STRIDE,
        )
        attribution_results['REINFORCE'] = evaluate_on_trajectories_with_attribution(
            env, lambda s: reinforce_agent.get_action(s, deterministic=True),
        )

    if ENABLE_AGENTS.get('PPO') and ppo_model is not None:
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        sb_eval = StableBaselinesAMMEnvironment(env, obs_keys=SB3_OBS_KEYS)
        ppo_eval = SbAgent(ppo_model, num_trajectories=NUM_TRAJECTORIES_EVAL)
        ppo_vec_normalize.training = False
        eval_env = DecisionStrideEnv(env, stride=DECISION_STRIDE)
        attribution_results['PPO'] = evaluate_on_trajectories_with_attribution(
            eval_env, lambda s: ppo_action_wrapper.unscale(
                ppo_eval.get_action(ppo_vec_normalize.normalize_obs(sb_eval._flatten_obs(s)))
            ),
        )

    if ENABLE_AGENTS.get('PPO_narrow') and ppo_narrow_model is not None:
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        sb_eval_n = StableBaselinesAMMEnvironment(env, obs_keys=SB3_OBS_KEYS)
        ppo_narrow_eval = SbAgent(ppo_narrow_model, num_trajectories=NUM_TRAJECTORIES_EVAL)
        ppo_narrow_vec_normalize.training = False
        eval_env_n = DecisionStrideEnv(env, stride=DECISION_STRIDE)
        attribution_results['PPO_narrow'] = evaluate_on_trajectories_with_attribution(
            eval_env_n, lambda s: ppo_narrow_action_wrapper.unscale(
                ppo_narrow_eval.get_action(
                    ppo_narrow_vec_normalize.normalize_obs(sb_eval_n._flatten_obs(s))
                )
            ),
        )

    if ENABLE_AGENTS.get('SAC') and sac_model is not None:
        env = create_environment(NUM_TRAJECTORIES_EVAL, EVAL_SEED)
        sb_eval_sac = StableBaselinesAMMEnvironment(env, obs_keys=SB3_OBS_KEYS)
        sac_eval = SbAgent(sac_model, num_trajectories=NUM_TRAJECTORIES_EVAL)
        sac_vec_normalize.training = False
        eval_env = DecisionStrideEnv(env, stride=DECISION_STRIDE)
        attribution_results['SAC'] = evaluate_on_trajectories_with_attribution(
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
        attribution_results['DQN'] = evaluate_on_trajectories_with_attribution(
            eval_env, _dqn_eval_action,
        )

    # Project out PnL for the existing significance tests / distribution plots.
    pnl_results = {n: r['pnl'] for n, r in attribution_results.items()}

    # Per-agent summary. Two significance tests printed side by side:
    #   p(t) — one-sample t-test on mean PnL vs 0 (parametric, CLT-justified at
    #          N=eval_trajectories but pulled around by heavy tails)
    #   p(W) — Wilcoxon signed-rank vs 0 (non-parametric, tests symmetry of the
    #          distribution around 0, rank-based so robust to long tails)
    # Skew/Kurt are Fisher's definitions: skew=0 → symmetric, kurt=0 → normal
    # tails. Strongly negative skew or kurt > 1 means the t-test is testing a
    # mean that's not representative of the typical trajectory.
    active = [n for n in AGENT_NAMES if n in pnl_results]
    header = (
        f"{'Agent':<18} | {'Mean':>8} | {'Std':>8} | {'Median':>8} "
        f"| {'Skew':>6} | {'Kurt':>6} | {'Profitable':>12} "
        f"| {'p(t) vs 0':>10} | {'p(W) vs 0':>10}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for name in active:
        pnl = pnl_results[name]
        pct = 100 * np.mean(pnl > 0)
        skew = stats.skew(pnl)
        kurt = stats.kurtosis(pnl)  # excess kurtosis (Fisher)
        p_t = stats.ttest_1samp(pnl, 0.0).pvalue
        # Wilcoxon errors on all-zero diffs; guard for the degenerate hold-cash case
        try:
            p_w = stats.wilcoxon(pnl).pvalue
        except ValueError:
            p_w = float('nan')
        print(
            f"{name:<18} | {np.mean(pnl):>+8.2f} | {np.std(pnl):>8.2f} "
            f"| {np.median(pnl):>+8.2f} | {skew:>+6.2f} | {kurt:>+6.2f} "
            f"| {np.sum(pnl > 0):>4d}/{len(pnl)} ({pct:.0f}%) "
            f"| {p_t:>10.3g} | {p_w:>10.3g}"
        )
    print("-" * len(header))

    # Pairwise comparison on per-trajectory PnL differences. Trajectories are
    # paired across agents because each evaluate_on_trajectories call uses the
    # same EVAL_SEED → identical OU midprice / arrival paths, so PnL[a] - PnL[b]
    # cancels the shared market noise. Skew of the differences indicates
    # whether the t-test's symmetry assumption holds on the *pair* (it can hold
    # for differences even when the marginals are skewed). Holm correction
    # controls family-wise error across the C(n, 2) comparisons; applied
    # independently to the t-test and Wilcoxon p-values.
    def _holm(raw_p):
        raw_p = np.asarray(raw_p, dtype=float)
        order = np.argsort(raw_p)
        adj = np.empty_like(raw_p)
        for rank, idx in enumerate(order):
            adj[idx] = min(1.0, raw_p[idx] * (len(raw_p) - rank))
        for k in range(1, len(order)):
            adj[order[k]] = max(adj[order[k]], adj[order[k - 1]])
        return adj

    if len(active) >= 2:
        pairs = list(combinations(active, 2))
        diffs = [pnl_results[a] - pnl_results[b] for a, b in pairs]
        mean_diffs = np.array([float(np.mean(d)) for d in diffs])
        med_diffs = np.array([float(np.median(d)) for d in diffs])
        skew_diffs = np.array([float(stats.skew(d)) for d in diffs])
        t_raw = np.array([stats.ttest_rel(pnl_results[a], pnl_results[b]).pvalue
                          for a, b in pairs])
        w_raw = np.array([
            stats.wilcoxon(pnl_results[a], pnl_results[b]).pvalue
            if np.any(pnl_results[a] != pnl_results[b]) else float('nan')
            for a, b in pairs
        ])
        t_adj = _holm(t_raw)
        w_adj = _holm(np.where(np.isnan(w_raw), 1.0, w_raw))

        pair_header = (
            f"{'Pair (A vs B)':<40} | {'mean(A-B)':>9} | {'med(A-B)':>8} "
            f"| {'skew':>6} | {'p(t) raw':>9} | {'p(t) Holm':>9} "
            f"| {'p(W) raw':>9} | {'p(W) Holm':>9}"
        )
        print("\nPairwise tests on PnL differences (paired t-test + Wilcoxon, Holm-corrected):")
        print("-" * len(pair_header))
        print(pair_header)
        print("-" * len(pair_header))
        for (a, b), md, mdn, sk, pt_r, pt_a, pw_r, pw_a in zip(
            pairs, mean_diffs, med_diffs, skew_diffs, t_raw, t_adj, w_raw, w_adj
        ):
            print(
                f"{a + ' vs ' + b:<40} | {md:>+9.2f} | {mdn:>+8.2f} "
                f"| {sk:>+6.2f} | {pt_r:>9.3g} | {pt_a:>9.3g} "
                f"| {pw_r:>9.3g} | {pw_a:>9.3g}"
            )
        print("-" * len(pair_header))

    # Quoting-spread summary across eval trajectories.
    print_spread_table(attribution_results)

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

        if ENABLE_AGENTS.get('DeployNarrow'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('DeployNarrow', []).append(
                collect_single_trajectory(env, DoNothingAgent(env).get_action, max_trades=MAX_TRADES_DEBUG))

        if ENABLE_AGENTS.get('Uniform'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('Uniform', []).append(
                collect_single_trajectory(env, UniformAllocationAgent(env).get_action, max_trades=MAX_TRADES_DEBUG))

        if ENABLE_AGENTS.get('DeployWide'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('DeployWide', []).append(
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

        if ENABLE_AGENTS.get('CDM'):
            env = create_environment(1, sim_seed)
            single_data.setdefault('CDM', []).append(
                collect_single_trajectory(
                    env,
                    CarteaPLAgent(
                        env, gamma=GAMMA_CARTEA,
                        rebalance_tolerance=REBALANCE_TOLERANCE_CARTEA,
                        seed=SEED,
                    ).get_action,
                    max_trades=MAX_TRADES_DEBUG,
                ))

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

        if ENABLE_AGENTS.get('PPO_narrow') and ppo_narrow_model is not None:
            env = create_environment(1, sim_seed)
            sb_single_n = StableBaselinesAMMEnvironment(env, obs_keys=SB3_OBS_KEYS)
            ppo_narrow_single = SbAgent(ppo_narrow_model, num_trajectories=1)
            single_data.setdefault('PPO_narrow', []).append(
                collect_single_trajectory(
                    env,
                    lambda s, _a=ppo_narrow_single, _sb=sb_single_n, _w=ppo_narrow_action_wrapper:
                        _w.unscale(_a.get_action(
                            ppo_narrow_vec_normalize.normalize_obs(_sb._flatten_obs(s))
                        )),
                    max_trades=MAX_TRADES_DEBUG, decision_stride=DECISION_STRIDE,
                ))

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
    if ppo_narrow_reward_cb.epoch_rewards:
        rl_training_rewards['PPO_narrow'] = ppo_narrow_reward_cb.epoch_rewards
    if sac_reward_cb.epoch_rewards:
        rl_training_rewards['SAC'] = sac_reward_cb.epoch_rewards
    if dqn_reward_cb.epoch_rewards:
        rl_training_rewards['DQN'] = dqn_reward_cb.epoch_rewards

    if rl_training_rewards:
        plot_training_rewards(rl_training_rewards)
    if pnl_results:
        plot_pnl_distribution(pnl_results)
    if attribution_results and REWARD_KIND != 'pnl':
        utility_results = {n: r['utility'] for n, r in attribution_results.items()}
        plot_utility_distribution(utility_results)
    if attribution_results:
        plot_pnl_attribution_aggregate(attribution_results)
    if single_data:
        plot_price_evolution(single_data)
        plot_pnl_evolution(single_data)
        plot_position_offsets(single_data)
        plot_pnl_attribution_individual(single_data)

    print("\nDone. All figures saved to", FIGURES_DIR)
    print(f"Results log:        {results_path}")


def _run_main_with_logging():
    """Wrap ``main()`` so sys.stdout is restored and the log file is closed
    even if main() raises (KeyboardInterrupt, RuntimeError, …)."""
    # We have to set the tee up before main() can call print(), so the actual
    # install happens inside main(); here we just ensure cleanup. main() leaves
    # `_results_file` and `_original_stdout` as module-level attrs would be
    # heavier than necessary — instead, rely on the fact that sys.stdout is
    # restored at main()'s last line on success, and restore here on failure.
    try:
        main()
    finally:
        # If main() crashed mid-run, sys.stdout is still the tee; restore it
        # and close the underlying file so the partial log isn't lost.
        if isinstance(sys.stdout, _StdoutTee):
            for s in sys.stdout.streams:
                if s is not sys.__stdout__:
                    try:
                        s.close()
                    except Exception:
                        pass
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    _run_main_with_logging()
