"""
PPO vs REINFORCE comparison for LP strategy learning.

Trains both algorithms on the same AMM environment, evaluates on identical
test trajectories, and produces side-by-side comparison plots.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.agents.PolicyGradientAgent import PolicyGradientAgent, generate_trajectory
from SAiFE_gym.agents.SbAgent import SbAgent
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, ASSET_PRICE_KEY, TIME_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY
)

# ============================================================================
# Shared Configuration
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
DRIFT = 1
VOLATILITY = 0.01
FEE_TIER = 0.003

ALPHA0 = np.array([10.0, 10.0])
ALPHA1 = np.array([150.0, 150.0])
ALPHA2 = np.array([0.0, 0.0])
ALPHA3 = np.array([5000.0, 5000.0])

# Training budget: match total environment steps
REINFORCE_EPOCHS = 200
# PPO sees NUM_TRAJECTORIES_TRAIN * N_STEPS steps per rollout
PPO_TOTAL_TIMESTEPS = REINFORCE_EPOCHS * NUM_TRAJECTORIES_TRAIN * N_STEPS

# ============================================================================
# Environment Factory
# ============================================================================

def create_environment(num_trajectories: int, seed: int = None):
    midprice_model = BrownianMotionMidpriceModel(
        drift=DRIFT, volatility=VOLATILITY, initial_price=INITIAL_PRICE,
        terminal_time=TERMINAL_TIME, step_size=TERMINAL_TIME / N_STEPS,
        num_trajectories=num_trajectories, seed=seed,
    )
    arrival_model = PoissonLinearArrivalModel(
        alpha=np.column_stack([ALPHA0, ALPHA1, ALPHA2, ALPHA3]).T,
        liquidity_scale=LIQUIDITY_SCALE,
        step_size=TERMINAL_TIME / N_STEPS,
        num_trajectories=num_trajectories,
        seed=seed + 1 if seed else None,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model, arrival_model=arrival_model,
        num_trajectories=num_trajectories, tau=TAU, num_ticks=10000,
        fee_tier=FEE_TIER, initial_wealth=INITIAL_WEALTH, seed=seed,
    )
    reward_function = PnL(exponential_value=1.0001, initial_wealth=INITIAL_WEALTH)
    env = AMMEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS,
        reward_function=reward_function, model_dynamics=model_dynamics,
        num_trajectories=num_trajectories, seed=seed,
    )
    return env

# ============================================================================
# Network architectures
# ============================================================================

def create_policy_network(input_size: int, hidden_size: int = 64, action_size: int = 2):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, action_size),
    )


class SimpleStraddlePolicy(nn.Module):
    def __init__(self, base_network, tau: int = 100):
        super().__init__()
        self.base_network = base_network
        self.tau = tau

    def forward(self, x):
        raw = self.base_network(x)
        center = self.tau * torch.tanh(raw[:, 0])
        half_width = 1.0 + (self.tau - 1.0) * torch.sigmoid(raw[:, 1])
        lower = torch.clamp(center - half_width, min=-self.tau)
        upper = torch.clamp(center + half_width, max=self.tau)
        return torch.stack([lower, upper], dim=1)

# ============================================================================
# PPO reward-logging callback
# ============================================================================

class RewardLoggerCallback(BaseCallback):
    """Logs mean episode reward each rollout for plotting."""
    def __init__(self):
        super().__init__()
        self.epoch_rewards = []

    def _on_rollout_end(self) -> None:
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            self.epoch_rewards.append(float(np.mean(rewards)))

    def _on_step(self) -> bool:
        return True

# ============================================================================
# Evaluation helper
# ============================================================================

def evaluate_agent(agent, env, is_sb_agent=False):
    """Run one full episode and return per-trajectory cumulative PnL."""
    if is_sb_agent:
        # SbAgent expects flat obs from StableBaselinesAMMEnvironment
        sb_env = StableBaselinesAMMEnvironment(env)
        obs = sb_env.reset()
        total_rewards = []
        dones = np.zeros(env.num_trajectories, dtype=bool)
        while not dones.all():
            action = agent.get_action(obs)
            obs, reward, dones, _ = sb_env.step_wait()
            # step_wait auto-resets; we only want the first episode
            total_rewards.append(reward)
            if dones.all():
                break
        return np.sum(np.array(total_rewards), axis=0)
    else:
        state, _ = env.reset()
        total_rewards = []
        terminated = np.zeros(env.num_trajectories, dtype=bool)
        while not np.any(terminated):
            action = agent.get_action(state, deterministic=True)
            state, reward, terminated, _, _ = env.step(action)
            total_rewards.append(reward)
        return np.sum(np.array(total_rewards), axis=0)

# ============================================================================
# Main
# ============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs("figures", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Train REINFORCE
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Training REINFORCE agent...")
    print("=" * 60)

    reinforce_env = create_environment(NUM_TRAJECTORIES_TRAIN, SEED)

    # Determine input size
    dummy_policy = nn.Linear(7, 2)
    temp_agent = PolicyGradientAgent(dummy_policy, reinforce_env)
    input_size = temp_agent.input_size

    base_net = create_policy_network(input_size, hidden_size=64, action_size=2)
    reinforce_policy = SimpleStraddlePolicy(base_net, tau=TAU)

    action_std_decay = lambda t: max(0.3, 1.7 * (0.998 ** (t * REINFORCE_EPOCHS)))

    optimizer = torch.optim.Adam(reinforce_policy.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    reinforce_agent = PolicyGradientAgent(
        policy=reinforce_policy, env=reinforce_env,
        action_std=action_std_decay, optimizer=optimizer,
        lr_scheduler=scheduler, max_grad_norm=1.0,
    )

    reinforce_losses, reinforce_rewards = reinforce_agent.train(
        num_epochs=REINFORCE_EPOCHS, reporting_freq=50,
    )

    # ------------------------------------------------------------------
    # 2. Train PPO
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training PPO agent...")
    print("=" * 60)

    ppo_env = create_environment(NUM_TRAJECTORIES_TRAIN, SEED)
    sb_env = StableBaselinesAMMEnvironment(ppo_env)

    reward_cb = RewardLoggerCallback()

    ppo_model = PPO(
        "MlpPolicy", sb_env,
        learning_rate=3e-4,
        n_steps=N_STEPS,         # rollout length = episode length
        batch_size=64,
        n_epochs=10,
        gamma=1.0,               # no discounting (finite horizon)
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=SEED,
    )
    ppo_model.learn(total_timesteps=PPO_TOTAL_TIMESTEPS, callback=reward_cb)

    ppo_agent = SbAgent(ppo_model, num_trajectories=NUM_TRAJECTORIES_EVAL)

    # ------------------------------------------------------------------
    # 3. Evaluate both on identical environment
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Evaluating on {NUM_TRAJECTORIES_EVAL} trajectories...")
    print("=" * 60)

    eval_env_reinforce = create_environment(NUM_TRAJECTORIES_EVAL, SEED + 999)
    eval_env_ppo = create_environment(NUM_TRAJECTORIES_EVAL, SEED + 999)

    reinforce_pnl = evaluate_agent(reinforce_agent, eval_env_reinforce, is_sb_agent=False)
    ppo_pnl = evaluate_agent(ppo_agent, eval_env_ppo, is_sb_agent=True)

    # Print summary statistics
    for name, pnl in [("REINFORCE", reinforce_pnl), ("PPO", ppo_pnl)]:
        print(f"\n--- {name} ---")
        print(f"  Mean PnL:   {np.mean(pnl):.2f}")
        print(f"  Std PnL:    {np.std(pnl):.2f}")
        print(f"  Median PnL: {np.median(pnl):.2f}")
        print(f"  Min / Max:  {np.min(pnl):.2f} / {np.max(pnl):.2f}")
        print(f"  Profitable: {np.sum(pnl > 0)}/{len(pnl)} ({100*np.mean(pnl > 0):.1f}%)")

    # ------------------------------------------------------------------
    # 4. Comparison plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO vs REINFORCE — LP Strategy Comparison", fontsize=14)

    # (a) Training reward curves
    ax = axes[0, 0]
    window = 20
    smooth_reinforce = [np.mean(reinforce_rewards[max(0, i-window):i+1]) for i in range(len(reinforce_rewards))]
    ax.plot(smooth_reinforce, label="REINFORCE", linewidth=2)
    if reward_cb.epoch_rewards:
        smooth_ppo = [np.mean(reward_cb.epoch_rewards[max(0, i-window):i+1]) for i in range(len(reward_cb.epoch_rewards))]
        ax.plot(smooth_ppo, label="PPO", linewidth=2)
    ax.set_xlabel("Epoch / Rollout")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) PnL distribution comparison
    ax = axes[0, 1]
    bins = np.linspace(
        min(np.min(reinforce_pnl), np.min(ppo_pnl)),
        max(np.max(reinforce_pnl), np.max(ppo_pnl)),
        50,
    )
    ax.hist(reinforce_pnl, bins=bins, alpha=0.5, label="REINFORCE", density=True)
    ax.hist(ppo_pnl, bins=bins, alpha=0.5, label="PPO", density=True)
    ax.axvline(np.mean(reinforce_pnl), color="C0", linestyle="--", label=f"REINFORCE mean={np.mean(reinforce_pnl):.1f}")
    ax.axvline(np.mean(ppo_pnl), color="C1", linestyle="--", label=f"PPO mean={np.mean(ppo_pnl):.1f}")
    ax.set_xlabel("Cumulative PnL")
    ax.set_ylabel("Density")
    ax.set_title("PnL Distribution (Eval)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Single-trajectory price & LP bounds comparison
    ax = axes[1, 0]
    single_env = create_environment(1, SEED + 1234)
    state, _ = single_env.reset()
    terminated = np.zeros(1, dtype=bool)
    times_r, lower_r, upper_r, prices_r = [], [], [], []
    exponential_value = 1.0001
    while not np.any(terminated):
        times_r.append(state[TIME_KEY][0])
        prices_r.append(state[POOL_SQRT_PRICE_KEY][0] ** 2)
        action = reinforce_agent.get_action(state, deterministic=True)
        # Record tick-based bounds after action
        lower_r.append(exponential_value ** (state[POOL_CURRENT_TICK_KEY][0] + action[0, 0]))
        upper_r.append(exponential_value ** (state[POOL_CURRENT_TICK_KEY][0] + action[0, 1]))
        state, _, terminated, _, _ = single_env.step(action)

    ax.plot(times_r, prices_r, "k-", label="Pool Price", linewidth=1.5)
    ax.fill_between(times_r, lower_r, upper_r, alpha=0.25, color="C0", label="REINFORCE range")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title("Single Trajectory — REINFORCE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) PPO single trajectory
    ax = axes[1, 1]
    single_env2 = create_environment(1, SEED + 1234)
    sb_single = StableBaselinesAMMEnvironment(single_env2)
    obs = sb_single.reset()
    raw_state, _ = single_env2.reset()  # re-sync to get dict state
    terminated2 = np.zeros(1, dtype=bool)
    times_p, lower_p, upper_p, prices_p = [], [], [], []
    # We need raw state for plotting — step the underlying env manually
    single_env3 = create_environment(1, SEED + 1234)
    state3, _ = single_env3.reset()
    sb_single3 = StableBaselinesAMMEnvironment(single_env3)
    obs3 = sb_single3.reset()
    terminated3 = np.zeros(1, dtype=bool)
    while not terminated3.all():
        times_p.append(state3[TIME_KEY][0])
        prices_p.append(state3[POOL_SQRT_PRICE_KEY][0] ** 2)
        action = ppo_agent.get_action(obs3)
        lower_p.append(exponential_value ** (state3[POOL_CURRENT_TICK_KEY][0] + action[0, 0]))
        upper_p.append(exponential_value ** (state3[POOL_CURRENT_TICK_KEY][0] + action[0, 1]))
        state3, _, term, _, _ = single_env3.step(action)
        obs3 = sb_single3._flatten_obs(state3)
        terminated3 = term

    ax.plot(times_p, prices_p, "k-", label="Pool Price", linewidth=1.5)
    ax.fill_between(times_p, lower_p, upper_p, alpha=0.25, color="C1", label="PPO range")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title("Single Trajectory — PPO")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/ppo_vs_reinforce_comparison.png", dpi=150, bbox_inches="tight")
    print("\nComparison plot saved to figures/ppo_vs_reinforce_comparison.png")


if __name__ == "__main__":
    main()
