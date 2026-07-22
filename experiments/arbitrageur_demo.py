#!/usr/bin/env python3
"""Run a vectorized SAiFE_gym episode with an LP and liquidity-taker arbitrage."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from SAiFE_gym.agents.BaselineAgents import ArbitrageurAgent, DeployOnceAgent
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    FEES0_KEY,
    FEES1_KEY,
    POOL_SQRT_PRICE_KEY,
)
from SAiFE_gym.rewards.RewardFunctions import PnL
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel


def build_env(
    num_trajectories: int = 8,
    n_steps: int = 50,
    seed: int = 42,
) -> AMMEnvironment:
    step_size = 1.0 / n_steps
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=2.0,
        initial_price=100.0,
        terminal_time=1.0,
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed,
    )
    arrival_model = PoissonArrivalModel(
        intensity=np.array([25.0, 25.0]),
        step_size=step_size,
        num_trajectories=num_trajectories,
        seed=seed + 1,
    )
    model_dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        num_trajectories=num_trajectories,
        fee_tier=0.003,
        tau=10,
        num_ticks=1000,
        gas_cost=0.0,
        seed=seed + 2,
    )
    return AMMEnvironment(
        terminal_time=1.0,
        n_steps=n_steps,
        reward_function=PnL(),
        model_dynamics=model_dynamics,
        initial_wealth=1e6,
        num_trajectories=num_trajectories,
        initial_pool_price=100.0,
        seed=seed,
    )


def run_demo(num_trajectories: int = 8, n_steps: int = 50, seed: int = 42) -> dict:
    env = build_env(num_trajectories=num_trajectories, n_steps=n_steps, seed=seed)
    lp_agent = DeployOnceAgent(env, lower_offset=-10, upper_offset=10)
    arbitrageur = ArbitrageurAgent(env, max_ticks_per_trade=25)

    obs, _ = env.reset()
    records = []
    for step in range(n_steps):
        lp_action = lp_agent.get_action(obs)
        obs, _, terminated, _, _ = env.step(lp_action)

        arb_action = arbitrageur.get_action(obs)
        arb_info = env.model_dynamics.execute_liquidity_taker_orders(arb_action)
        env._compute_derived_obs()
        obs = env.state

        records.append(
            {
                "step": step,
                "pool_price": (obs[POOL_SQRT_PRICE_KEY] ** 2).copy(),
                "external_price": obs[ASSET_PRICE_KEY].copy(),
                "action": arb_action[:, 0].copy(),
                "tick_movement": arb_info["tick_movement"].copy(),
                "fees0": np.sum(obs[FEES0_KEY], axis=1),
                "fees1": np.sum(obs[FEES1_KEY], axis=1),
                "unfilled_input": arb_info["unfilled_input"].copy(),
            }
        )
        if bool(terminated[0]):
            break

    return {"env": env, "records": records}


if __name__ == "__main__":
    result = run_demo(num_trajectories=8, n_steps=50)
    records = result["records"]
    final = records[-1]
    total_abs_action = sum(np.abs(record["action"]).sum() for record in records)
    total_abs_tick_moves = sum(np.abs(record["tick_movement"]).sum() for record in records)
    print("SAiFE_gym arbitrageur demo")
    print(f"steps: {len(records)}")
    print(f"trajectories: {len(final['pool_price'])}")
    print(f"total_abs_action: {total_abs_action:.6f}")
    print(f"total_abs_tick_moves: {int(total_abs_tick_moves)}")
    print(f"final_mean_pool_price: {final['pool_price'].mean():.6f}")
    print(f"final_mean_external_price: {final['external_price'].mean():.6f}")
    print(f"final_mean_fees0: {final['fees0'].mean():.6f}")
    print(f"final_mean_fees1: {final['fees1'].mean():.6f}")
