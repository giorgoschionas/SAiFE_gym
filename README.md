# SAiFE Gym

`SAiFE_gym` is a module which provides a collection of Gymnasium environments for training reinforcement learning (RL) agents for dynamic liquidity provision (LP) in automated market makers (AMMs) with concentrated liquidity (CL), like Uniswap v3. We decompose the microstructure of AMMs with CL in interactive components that
allow researchers and practitioners to combine them and capture various economic settings. The module is vectorized end-to-end which allows faster training of RL agents. 

## Features

- **Mechanics of AMMs with CL** - tick-indexed liquidity arrays, pool price and 
  fee accounting, LP fee snapshots, and mark-to-market LP portfolio value.
- **Market components as stochastic processes** - External midprice models, orderflow models (that captures noise traders, liquidity-attracted traders and informed traders), price impact models, and separate fee-accounting models.
- **Vectorized rollouts** - state, actions, arrivals, swaps, rewards and
  diagnostics are batched across `num_trajectories`.
- **Domain-randomized PPO workflow** - episode-level randomization of volatility
  (`sigma`) and baseline arrival rate for robust LP training.
- **Stable-Baselines3 integration** - flat SB3 observations, `VecNormalize`,
  decision-stride training, and discrete/structured action adapters.
- **Baselines and diagnostics** - random, uniform, deploy-once,
  periodic-rebalance, and arbitrageur agents, and policy behavior diagnostics for evaluation runs.

## Installation

```bash
git clone https://github.com/giorgoschionas/SAiFE_gym.git
cd SAiFE_gym
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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

The helper automatically deep-copies the raw simulator for evaluation, preserving
its configuration while keeping state, reward objects, and random generators
independent. Evaluation uses seed `10042` by default, configurable with
`eval_seed`, and advances its own random streams between evaluations. When
normalization is enabled, evaluation uses copies of the training statistics
without updating them, and rewards remain unnormalized.

Evaluation runs every ten rollouts. SB3 callbacks count batch steps, so the
interval is `10 * env.n_steps` calls. With the settings above, that is 2,000
calls / 100,000 training transitions, giving 20 scheduled evaluations over the
run. Each evaluation scores ten episodes; a new best score saves
`best_model.zip` beneath `best_model_path` (default `./best_models`). Set
`eval_log_path` to also save `evaluations.npz`.

For custom components that cannot be deep-copied, or a smaller evaluation batch,
pass a separately constructed raw environment with compatible observation and
action spaces:

```python
eval_env = get_amm_env(num_trajectories=10, tau=5, volatility=2.0, alpha3=0.5)
model, callback = get_ppo_learner_and_callback(
    env, eval_env=eval_env, eval_seed=12345, normalise_obs=True,
)
```

The helper seeds the supplied evaluation environment with `eval_seed` as well.
Training and evaluation must have separate simulator components, including
reward objects.

For the current domain-randomized robust LP workflow, run the dedicated script:

```bash
python experiments/train_robust_lp_agent.py \
  --smoke-test \
  --output-dir experiments/results/domain_randomized_ppo/local_smoke
```

The smoke test trains tiny nominal and domain-randomized PPO models, saves the
models and `VecNormalize` statistics, and evaluates all four policies on the
full 2 × 2 grid: sigma `[0.030, 0.08]` crossed with arrival rate `[300.0, 450.0]`.
With the default training support, this gives one regime in each of the four
evaluation sets below, producing 16 rows in `evaluation_grid.csv`.

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
domains per reset. The hpc launchers pass `--train-domains-per-reset`
explicitly and default to one sampled domain per trajectory.

Evaluation writes one row per policy/regime to `evaluation_grid.csv` and
summaries to `summary.json`. The policies are:

- `domain_randomized_ppo`
- `nominal_ppo`
- `periodic_rebalance`
- `cash`

Final evaluation uses the full Cartesian product of the combined axis lists:
sigma `[0.015, 0.030, 0.045, 0.065, 0.08]` and arrival rate
`[150.0, 250.0, 300.0, 350.0, 450.0]`. The default 25 regimes produce 100 rows
across the four policies. This includes combinations where only one parameter
is outside the randomized training support.

Rows and summaries use four `evaluation_set` labels. Membership in the training
support includes both endpoints of each configured training range:

| Evaluation set | Sigma | Arrival rate | Default regimes |
|----------------|-------|--------------|----------------:|
| `in_distribution` | Inside | Inside | 9 |
| `sigma_only_stress` | Outside | Inside | 6 |
| `arrival_only_stress` | Inside | Outside | 6 |
| `stress` | Outside | Outside | 4 |

The in-distribution axis values must lie inside their training ranges. Each
tuple from the explicitly configured stress axes must have at least one value
outside support. The generator also crosses in-distribution sigma with stress
arrival rates, and stress sigma with in-distribution arrival rates; duplicate
combinations are evaluated once. Labels depend on actual training support.
The original 13 regimes retain their ordering and evaluation seeds, followed
by the mixed combinations.

For Slurm runs and standalone evaluation of saved runs, see the
[HPC guide](hpc/README.md).

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
│   │   ├── ArbitrageurEnvironment.py
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
│   ├── aggregate_domain_randomized_seed_sweep.py
│   ├── evaluate_robust_lp_agents.py
│   ├── evaluation_grid.py
│   ├── fast_lp_evaluation.py
│   ├── helpers.py
│   ├── policy_behavior_diagnostics.py
│   └── train_robust_lp_agent.py
├── hpc/
│   ├── .env.example
│   ├── README.md
│   ├── sbatch_aggregate_domain_randomized_seed_sweep.sh
│   ├── sbatch_robust_lp_seed_sweep_cpu.sh
│   ├── sbatch_robust_lp_smoke.sh
│   ├── sbatch_train_robust_lp_agent_cpu.sh
│   └── setup_env.sh
├── skills/
├── tests/
└── requirements.txt
```
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
