# SAiFE Gym

A vectorized reinforcement learning simulator with Gymnasium and
Stable-Baselines3 adapters for training liquidity provision (LP) agents in
Uniswap v3 automated market makers with concentrated liquidity.

## Overview

SAiFE_gym simulates Uniswap v3 LP decisions under stochastic external prices,
liquidity-taking order flow, fee accrual, rebalancing costs, and mark-to-market
portfolio accounting. The simulator is vectorized end to end: one
`AMMEnvironment.step()` advances every trajectory in the batch.

The raw simulator action is `[lower_offset, upper_offset, hold_flag]`, where
the offsets are tick distances from the current pool tick. Rebalance actions
deploy the LP's available wealth into the requested range; hold actions keep
the existing position and avoid the step's gas cost.

## Features

- **Uniswap v3 concentrated liquidity** - tick-indexed liquidity arrays, pool
  fee accounting, LP fee snapshots, and mark-to-market LP portfolio value.
- **Vectorized rollouts** - state, actions, arrivals, swaps, rewards, and
  diagnostics are batched across `num_trajectories`.
- **Injectable market components** - Brownian and geometric Brownian midprices,
  Poisson/linear/kernel arrivals, one-tick or liquidity-depth price impact, and
  separate fee-accounting models.
- **Domain-randomized PPO workflow** - episode-level randomization of volatility
  (`sigma`) and baseline arrival rate for robust LP training.
- **Stable-Baselines3 integration** - flat SB3 observations, `VecNormalize`,
  decision-stride training, and discrete/structured action adapters.
- **Baselines and diagnostics** - random, uniform, deploy-once,
  periodic-rebalance, cash, and arbitrageur agents plus policy behavior
  diagnostics for evaluation runs.

## Installation

```bash
git clone <repository-url>
cd SAiFE_gym
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The project is developed against Python 3.12. `requirements.txt` is the
authoritative pinned dependency list.

## Quick Start

```python
import numpy as np

from experiments.helpers import get_amm_env
from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent

env = get_amm_env(num_trajectories=10, tau=5, volatility=2.0, arrival_rate=100.0)
agent = UniformAllocationAgent(env)

obs, _ = env.reset()
cumulative_rewards = np.zeros(env.num_trajectories)

for _ in range(env.n_steps):
    action = agent.get_action(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    cumulative_rewards += rewards
    if (terminated | truncated).all():
        break

print("Episode rewards:", cumulative_rewards)
```

## Training

For the basic nominal PPO helper:

```python
from experiments.helpers import get_amm_env, get_ppo_learner_and_callback

env = get_amm_env(num_trajectories=50, tau=5, volatility=2.0, alpha3=0.5)
model, callback = get_ppo_learner_and_callback(env, normalise_obs=True)
model.learn(total_timesteps=2_000_000, callback=callback)
```

For the current domain-randomized robust LP workflow, run the dedicated script:

```bash
python experiments/train_robust_lp_agent.py \
  --smoke-test \
  --output-dir experiments/results/domain_randomized_ppo/local_smoke
```

The smoke test trains tiny nominal and domain-randomized PPO models, saves the
models and `VecNormalize` statistics, and evaluates them on one in-distribution
and one stress regime.

## Domain-Randomized PPO

`experiments/train_robust_lp_agent.py` trains two PPO policies with the same
architecture and evaluation grid:

- `nominal_ppo` trains on the fixed nominal domain.
- `domain_randomized_ppo` wraps the fixed AMM environment with
  `DomainRandomizedAMMEnvironment`, sampling episode domains from
  `UniformDomainRandomizationConfig`.

The robust training environment uses:

- `GeometricBrownianMotionMidpriceModel`
- `LiquidityKernelArrivalModel`
- `LiquidityDepthUniswapV3PriceImpact` with token1-notional trade sizes
- `RunningInventoryPenalty`
- `DecisionStrideVecEnv`
- `StructuredMultiDiscreteVecEnv`
- SB3 `VecNormalize` unless `--no-normalise-obs` is passed

Important script defaults (also used by the HPC training launchers):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--total-timesteps` | `10_000_000` | Simulator-equivalent training budget |
| `--num-trajectories` | `100` | Vectorized paths per episode |
| `--n-steps` | `1000` | Simulator steps per episode |
| `--decision-stride` | `100` | PPO acts once every 100 simulator steps |
| `--tau` | `50` | Maximum center offset and half-width in ticks |
| `--tick-stride` | `5` | Structured action grid spacing |
| `--alpha3` | `4000.0` | Mispricing/arbitrage intensity coefficient |
| `--initial-wealth` | `1000.0` | LP starting capital in token1 units |
| `--inventory-phi` | `0.4` | Running token0 inventory penalty coefficient |
| `--nominal-sigma` | `0.03` | Fixed nominal GBM volatility |
| `--nominal-arrival-rate` | `300.0` | Fixed nominal baseline order rate |
| `--nominal-gas-cost` | `2.0` | Fixed rebalance gas cost |
| `--train-sigma-range` | `0.01 0.05` | Randomized training support for sigma |
| `--train-arrival-rate-range` | `200.0 400.0` | Randomized training support for arrival rate |
| `--evaluation-seed` | `100042` | Evaluation paths, independent of training seed |

Nominal training and convergence validation default to sigma `0.03` and arrival
rate `300.0`, the midpoints of the randomized training ranges. The final
in-distribution evaluation grid includes this nominal condition.

With the default decision stride, the default simulator budget corresponds to
`100_000` PPO decision timesteps and each episode exposes at most 10 PPO
decisions. Forced hold actions are used between PPO decisions.

By default, the Python script uses `min(10, num_trajectories)` balanced sampled
domains per reset. The Barkla2 Slurm launchers pass `--train-domains-per-reset`
explicitly and default to one sampled domain per trajectory.

Evaluation writes one row per policy/regime to `evaluation_grid.csv` and
summaries to `summary.json`. The policies are:

- `domain_randomized_ppo`
- `nominal_ppo`
- `periodic_rebalance`
- `cash`

The default in-distribution grid is the Cartesian product of sigma
`[0.015, 0.030, 0.045]` and arrival rate `[250.0, 300.0, 350.0]`. The default
stress grid is sigma `[0.065, 0.08]` by arrival rate `[150.0, 450.0]`.
In-distribution values must lie inside the training support; each stress tuple
must have at least one value outside training support.

For Barkla2/Slurm runs, see `hpc/README_BARKLA2.md`.

## Architecture

```text
SAiFE_gym/
├── SAiFE_gym/
│   ├── agents/
│   │   ├── Agent.py
│   │   ├── BaselineAgents.py
│   │   ├── PolicyGradientAgent.py
│   │   └── SbAgent.py
│   ├── gym/
│   │   ├── AMMEnvironment.py
│   │   ├── GymnasiumAMMEnvironment.py
│   │   ├── ModelDynamics.py
│   │   ├── StableBaselinesAMMEnvironment.py
│   │   ├── domain_randomization.py
│   │   ├── index_names.py
│   │   ├── observation_features.py
│   │   ├── simulation_core.py
│   │   └── helpers/AMM_utils.py
│   ├── rewards/RewardFunctions.py
│   ├── stochastic_processes/
│   │   ├── StochasticProcessModel.py
│   │   ├── arrival_models.py
│   │   ├── fee_accounting_models.py
│   │   ├── midprice_models.py
│   │   └── price_impact_models.py
│   └── wrappers.py
├── experiments/
│   ├── helpers.py
│   ├── train_robust_lp_agent.py
│   ├── aggregate_domain_randomized_seed_sweep.py
│   └── policy_behavior_diagnostics.py
├── hpc/
│   ├── README_BARKLA2.md
│   └── sbatch_*.sh
├── notebooks/
├── tests/
└── requirements.txt
```

## Environment Interfaces

Choose the interface that matches your rollout or training library:

| Interface | Observations | Step returns | Episode reset |
|-----------|--------------|--------------|---------------|
| `AMMEnvironment` | Full batched state dictionary | Batched rewards and termination/truncation arrays | Explicit whole-batch reset |
| `GymnasiumAMMEnvironment` | Full dictionary for one trajectory | Scalar reward and boolean termination/truncation flags | Explicit reset |
| `GymnasiumAMMVectorEnv` | Full batched state dictionary | Gymnasium vector arrays and masked info dictionary | Next-step whole-batch autoreset |
| `StableBaselinesAMMEnvironment` | Flat float32 features, nine by default | SB3 rewards, combined dones, and list of infos | Same-step whole-batch autoreset |

The raw `AMMEnvironment` retains its batched API even with one trajectory.
Its `gymnasium.Env` inheritance is retained for existing code; third-party
Gymnasium wrappers should use the new adapters instead of wrapping it directly.
`experiments/train_robust_lp_agent.py` continues to use the SB3 adapter with
its existing feature order, structured actions, and normalization.

For single-environment Gymnasium wrappers:

```python
import gymnasium as gym
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.GymnasiumAMMEnvironment import GymnasiumAMMEnvironment

env = gym.wrappers.RecordEpisodeStatistics(
    GymnasiumAMMEnvironment(AMMEnvironment(num_trajectories=1))
)
obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
if terminated or truncated:
    obs, info = env.reset()
env.close()
```

The single adapter requires exactly one trajectory. Scalar state fields have
shape `()`, while liquidity and fee arrays have shape `(num_ticks,)`. Actions
have shape `(3,)`. Standard `FlattenObservation` and `TimeLimit` wrappers can
also wrap this adapter.

For Gymnasium vector wrappers over the native batch:

```python
import gymnasium as gym
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.GymnasiumAMMEnvironment import GymnasiumAMMVectorEnv

envs = gym.wrappers.vector.RecordEpisodeStatistics(
    GymnasiumAMMVectorEnv(AMMEnvironment(num_trajectories=8))
)
obs, info = envs.reset(seed=7)
obs, rewards, terminated, truncated, info = envs.step(envs.action_space.sample())
envs.close()
```

This vector adapter exposes `single_observation_space`, `single_action_space`,
and batched `observation_space` and `action_space`. Its actions have shape
`(num_envs, 3)`. Terminal observations are returned on the terminal step. The
following `step()` resets the batch, ignores its actions, and returns zero
rewards and false termination/truncation flags. This is Gymnasium's
`AutoresetMode.NEXT_STEP`; the SB3 adapter keeps its existing same-step behavior.

Both Gymnasium adapters accept an existing `DomainRandomizedAMMEnvironment`
and copy observations and diagnostics so later steps cannot mutate previous
results. The vector adapter includes Gymnasium presence masks for info fields,
including nested domain parameters. Use `gym.wrappers.vector.DictInfoToList`
if a consumer needs one info dictionary per trajectory.

The native batch has shared RNG streams and a synchronized episode horizon.
The vector adapter accepts one integer seed or `None`, and only whole-batch
resets; per-trajectory seed lists and partial `reset_mask` values are rejected.
For independent seeds or resets, construct separate single adapters using
`gym.vector.SyncVectorEnv` or `gym.vector.AsyncVectorEnv`. Rendering is not
implemented by these adapters.

## State Dictionary

The raw environment observation is a dictionary of vectorized arrays. Core keys
are defined in `SAiFE_gym/gym/index_names.py`.

| Key | Shape | Description |
|-----|-------|-------------|
| `sqrt_price` | `(num_trajectories,)` | Current pool sqrt price |
| `current_tick` | `(num_trajectories,)` | Current absolute pool tick |
| `liquidity_array` | `(num_trajectories, num_ticks)` | Pool liquidity per tracked tick interval |
| `fees_0` / `fees_1` | `(num_trajectories, num_ticks)` | Pool fees accumulated per tick |
| `lp_liquidity` | `(num_trajectories,)` | Active LP position liquidity |
| `lp_tick_lower` / `lp_tick_upper` | `(num_trajectories,)` | Active LP tick bounds |
| `lp_collected_fees_0` / `lp_collected_fees_1` | `(num_trajectories,)` | Lifetime LP fees earned |
| `lp_unclaimed_fees_0` / `lp_unclaimed_fees_1` | `(num_trajectories,)` | Fees accrued since current position entry |
| `lp_fee_snapshot_0` / `lp_fee_snapshot_1` | `(num_trajectories,)` | Fee entitlement at position entry, used to exclude earlier fees |
| `lp_ever_deployed` | `(num_trajectories,)` | Whether the trajectory has previously deployed LP liquidity |
| `midprice` | `(num_trajectories,)` | External market price |
| `time` | `(num_trajectories,)` | Current simulation time |
| `gas_cost` | `(num_trajectories,)` | Fixed rebalance cost for the episode |
| `initial_wealth` | `(num_trajectories,)` | Starting LP capital |
| `portfolio_value` | `(num_trajectories,)` | Current mark-to-market LP value |
| `lp_alpha` | `(num_trajectories,)` | Token0 value fraction of the LP position |
| `lp_token0_amount` | `(num_trajectories,)` | Absolute LP token0 inventory |

`sqrt_price` stores sqrt(P), following the Uniswap v3 convention. Convert with
`price = sqrt_price ** 2`.

The raw observation space declares all 22 keys. Tick indices use int64,
`lp_ever_deployed` uses bool, and other fields use float64. The elapsed-time
space is nonnegative and unbounded to accommodate floating-point accumulation;
episodes still terminate at the configured trading horizon. Raw observations
reference simulator state, so copy them when retaining snapshots.

## SB3 Observation Adapter

`StableBaselinesAMMEnvironment` converts the raw dictionary into a flat
float32 observation with the default 9 features:

- `mispricing`
- `lp_lower_offset`
- `lp_upper_offset`
- `boundary_proximity`
- `position_width`
- `time`
- `has_position`
- `portfolio_value_ratio`
- `unclaimed_fee_value_ratio`

Array-valued raw state keys such as liquidity and fee arrays are deliberately
excluded from the default SB3 view.

## Action Spaces

`AMMEnvironment.action_space` (also `single_action_space`) is the low-level
command space for one trajectory:

```text
Box(low=[-tau, -tau+1, -1.0], high=[tau-1, tau, 1.0], shape=(3,), dtype=float32)
```

Raw `step()` takes a `(num_trajectories, 3)` batch. Sample that batch using
`env.batched_action_space.sample()`. For one trajectory, a `(3,)` action from
`env.action_space.sample()` is also accepted, while returns remain batched.

| Idx | Name | Range | Description |
|-----|------|-------|-------------|
| `[0]` | `lower_offset` | `[-tau, tau-1]` | Lower tick bound offset from current pool tick |
| `[1]` | `upper_offset` | `[-tau+1, tau]` | Upper tick bound offset from current pool tick |
| `[2]` | `hold_flag` | `[-1.0, 1.0]` | `<= 0` rebalances; `> 0` holds the existing position |

`validate_action` rounds offsets to integer ticks, clips them to the action
bounds, and enforces `lower_offset < upper_offset`.

Use wrapper action spaces when training agents:

- `DiscreteActionWrapper` exposes one hold action plus finite rebalance ranges
  for the native batched simulator. It converts actions and retains raw batched
  returns; it does not provide a standard single-environment Gymnasium API.
- `DiscreteActionVecEnv` exposes the same table for SB3 `VecEnv` pipelines.
- `StructuredMultiDiscreteVecEnv` exposes decoupled
  `[center_offset_id, half_width_id, hold_id]` action heads. With
  the default `tau=50` and `tick_stride=5`, the SB3 action space is
  `MultiDiscrete([21, 10, 2])`.

## Testing

Run the full test suite with:

```bash
python -m pytest
```

Targeted checks for the domain-randomized PPO workflow:

```bash
python -m pytest \
  tests/test_domain_randomization.py \
  tests/test_wrappers.py \
  tests/test_aggregate_domain_randomized_seed_sweep.py
```

## Dependencies

Main runtime and workflow dependencies include:

- `gymnasium`
- `numpy`
- `matplotlib`
- `pandas`
- `stable-baselines3`
- `torch`
- `cloudpickle`
- `tqdm`
- `pytest`
- `ipykernel` and `ipython` for notebooks

See `requirements.txt` for exact versions.
