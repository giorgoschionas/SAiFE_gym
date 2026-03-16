# SAiFE Gym

A Gymnasium-compatible Reinforcement Learning environment for training liquidity provision (LP) agents in Uniswap V3 Automated Market Makers (AMMs) with concentrated liquidity.

## Overview

SAiFE_gym simulates a Uniswap V3 pool with stochastic order flow and price dynamics. The environment is **fully vectorized** — it runs multiple parallel trajectories in a single step call, making it efficient for RL training with algorithms like PPO.

The LP agent controls its position range at each step by choosing tick offsets `[lower_offset, upper_offset]` relative to the current price. The reward is mark-to-market PnL of the LP portfolio, with optional risk adjustments.

## Features

- **Uniswap V3 concentrated liquidity** — tick-based liquidity array, fee accumulation, and geometric midpoint price snapping on tick crossings
- **Vectorized simulation** — all operations batched over `num_trajectories` for fast parallel rollouts
- **Stochastic processes** — Brownian Motion midprice model; Poisson and linear arrival models for order flow (with toxicity/arbitrage coefficient α₃)
- **Reward functions** — `PnL`, `ExponentialUtility` (CARA), `RunningInventoryPenalty` (Cartea–Jaimungal-style)
- **Stable-Baselines3 integration** — `StableBaselinesAMMEnvironment` flattens the dict observation for SB3; `VecNormalize` wrapper included
- **Baseline agents** — `UniformAllocationAgent` (full-range LP) and `RandomAgent`

## Installation

```bash
git clone <repository-url>
cd SAiFE_gym
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent

# Build environment using the experiment helper
from experiments.helpers import get_amm_env

env = get_amm_env(num_trajectories=10, tau=5, volatility=2.0, arrival_rate=100.0)

# Run one episode with the uniform baseline agent
agent = UniformAllocationAgent(env)
obs, _ = env.reset()

for _ in range(env.n_steps):
    action = agent.get_action(obs)
    obs, rewards, terminated, truncated, info = env.step(action)

print("Final rewards:", rewards)  # shape: (num_trajectories,)
```

## Training with PPO (Stable-Baselines3)

```python
from experiments.helpers import get_amm_env, get_ppo_learner_and_callback

env = get_amm_env(num_trajectories=50, tau=5, volatility=2.0, alpha3=0.5)
model, callback = get_ppo_learner_and_callback(env, normalise_obs=True)
model.learn(total_timesteps=2_000_000, callback=callback)
```

## Architecture

```
SAiFE_gym/
├── SAiFE_gym/
│   ├── agents/
│   │   ├── Agent.py                    # Abstract base class
│   │   ├── BaselineAgents.py           # RandomAgent, UniformAllocationAgent
│   │   └── SbAgent.py                  # Wrapper for SB3 models
│   ├── gym/
│   │   ├── AMMEnvironment.py           # Main Gymnasium environment
│   │   ├── ModelDynamics.py            # UniswapV3ModelDynamics (core AMM logic)
│   │   ├── StableBaselinesAMMEnvironment.py  # SB3-compatible flat obs wrapper
│   │   ├── index_names.py              # State dictionary key constants
│   │   └── helpers/
│   │       └── AMM_utils.py            # Tick/price conversion, position value
│   ├── rewards/
│   │   └── RewardFunctions.py          # PnL, ExponentialUtility, RunningInventoryPenalty
│   └── stochastic_processes/
│       ├── StochasticProcessModel.py   # Abstract base classes
│       ├── midprice_models.py          # BrownianMotionMidpriceModel
│       └── arrival_models.py           # PoissonArrivalModel, PoissonLinearArrivalModel
├── experiments/
│   ├── helpers.py                      # Environment factory, PPO setup, plotting utilities
│   └── *.py                            # Experiment scripts
├── tests/                              # pytest test suite
├── notebooks/                          # Exploration and validation notebooks
└── docs/                               # Extended documentation
```

### State Dictionary

| Key | Shape | Description |
|-----|-------|-------------|
| `pool_sqrt_price` | `(num_trajectories,)` | Current pool √price |
| `pool_current_tick` | `(num_trajectories,)` | Current tick index |
| `pool_liquidity_array` | `(num_trajectories, num_ticks)` | Liquidity per tick |
| `fees_0` / `fees_1` | `(num_trajectories, num_ticks)` | Accumulated fees per tick |
| `lp_liquidity` | `(num_trajectories,)` | LP position liquidity |
| `lp_tick_lower` / `lp_tick_upper` | `(num_trajectories,)` | LP position bounds |
| `asset_price` | `(num_trajectories,)` | External market midprice |
| `time` | `(num_trajectories,)` | Current simulation time |

### Action Space

`Box(low=[-tau, -tau+1], high=[tau-1, tau], shape=(2,))` — tick offsets relative to current price.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 5 | LP position half-width in ticks |
| `num_trajectories` | 1 | Parallel simulation paths |
| `n_steps` | 200 | Steps per episode |
| `terminal_time` | 1.0 | Episode length |
| `volatility` | 2.0 | Brownian motion volatility |
| `arrival_rate` | 100.0 | Baseline Poisson order rate |
| `alpha3` | 0.0 | Arbitrage/toxicity coefficient |
| `fee_tier` | 0.003 | Uniswap V3 pool fee (0.3%) |
| `num_ticks` | 5000 | Size of liquidity array |

## Dependencies

- `gymnasium` — RL environment framework
- `numpy` — vectorized numerical computations
- `matplotlib` — visualization
- `stable-baselines3` — PPO and other RL algorithms
- `torch` — neural network backend (via SB3)
- `pytest` — test suite

See `requirements.txt` for the full pinned dependency list.

## License

[Add your license information here]
