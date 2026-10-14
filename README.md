# SAiFE Gym

`SAiFE_gym` is a module which provides a collection of Gymnasium environments for training reinforcement learning (RL) agents for dynamic liquidity provision (LP) in automated market makers (AMMs) with concentrated liquidity (CL), like Uniswap v3. The microstructure of AMMs with CL is decomposed in interactive components that
allow researchers and practitioners to combine them and capture various economic settings. The module is vectorized end-to-end which allows faster training of RL agents.

The associated paper can be found at [https://arxiv.org/abs/2609.17788](https://arxiv.org/abs/2609.17788)

## Features

- **Mechanics of AMMs with CL** - tick-indexed liquidity array, pool price and 
  fee accounting, LP fee snapshots.
- **Market components as stochastic processes** - External midprice models, orderflow models (that capture baseline traders, liquidity-attracted traders and informed traders), price impact models, and separate fee-accounting models.
- **Vectorized rollouts** - state, actions, arrivals, swaps, rewards and
  diagnostics are batched across `num_trajectories`.
- **Domain-randomized PPO workflow** - episode-level randomization of volatility
  (`sigma`) and baseline arrival rate for robust LP training.


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

## LP action interfaces

The default LP action space is conditional: **Hold** has no parameters;
**Rebalance(lower, upper)** selects two integer tick offsets relative to the
current pool tick, with `-tau <= lower < upper <= tau`. Rebalancing deploys
all available wealth. There is no position-size action.

```python
from SAiFE_gym.gym.lp_actions import hold_action, rebalance_action

# One trajectory: a single tagged decision.
env = get_amm_env(num_trajectories=1, tau=5)
env.reset()
env.step(rebalance_action(-2, 3))  # (1, array([-2, 3]))
env.step(hold_action())           # (0, ())

# Multiple trajectories: one decision per trajectory.
env = get_amm_env(num_trajectories=2, tau=5)
env.reset()
env.step((hold_action(), rebalance_action(-5, 5)))
```

`action_space` / `single_action_space` describes one conditional decision;
`batched_action_space` describes the whole trajectory batch. The single and
vector Gymnasium adapters expose the corresponding conditional spaces.

**Stable-Baselines3 uses one categorical action ID**, with standard policies:

```python
from stable_baselines3 import PPO
from SAiFE_gym.gym.StableBaselinesAMMEnvironment import StableBaselinesAMMEnvironment

raw_env = get_amm_env(num_trajectories=10, tau=50)
training_env = StableBaselinesAMMEnvironment(raw_env, tick_stride=5)
assert training_env.action_space.n == 211
model = PPO("MlpPolicy", training_env)
```

ID `0` holds. Remaining IDs enumerate all valid lower/upper endpoint pairs,
ordered by lower offset and then upper offset. With `m` endpoints, the count
is `1 + m*(m-1)/2`. `tau=5, tick_stride=1` gives 56 actions;
`tau=50, tick_stride=5` gives 211. The grid includes both `-tau` and `+tau`,
appending `+tau` if the stride does not land on it. Each ID has a unique meaning.

For compatibility, `command_space` and `batched_command_space` expose the old
`[lower_offset, upper_offset, hold_flag]` Box. `step_command(commands)` and
legacy array inputs to `step` retain integer rounding/clipping and the flag
convention (`<= 0`: rebalance, `> 0`: hold). The SB3 adapter accepts
`action_encoding="raw"` for existing continuous policies or legacy action
wrappers. Conditional inputs instead reject invalid/non-integer ranges.

New training helpers and robust runs record the categorical encoding and grid
parameters in saved policy metadata; robust run configurations record them too.
Evaluation retains support for historical MultiDiscrete policies using their
original decoder. A new categorical policy must be trained from scratch.
The new training grid offers 210 ranges, compared with the 120 distinct ranges
reachable through the previous center/half-width mapping. Policy range plots
leave gaps on hold decisions because no range is selected; for externally
created saved policies without grid metadata, pass `tick_stride` explicitly.

## Stochastic Gas Costs

Gas is a cost per LP rebalance in token1 units. By default it is the fixed
`gas_cost` supplied to `UniswapV3ModelDynamics` or `get_amm_env`. To make it
vary over time, inject an `OrnsteinUhlenbeckGasCostModel`:

```python
from experiments.helpers import get_amm_env
from SAiFE_gym.stochastic_processes.gas_cost_models import OrnsteinUhlenbeckGasCostModel

gas_model = OrnsteinUhlenbeckGasCostModel(
    theta=5.0,          # Mean-reversion speed, in inverse simulation time
    mu=2.0,             # Long-term latent mean, in token1
    sigma=0.5,          # Volatility, in token1 / sqrt(simulation time)
    initial_cost=2.0,   # Defaults to mu
    step_size=1.0 / 200,
    terminal_time=1.0,
    num_trajectories=10,
)
env = get_amm_env(
    num_trajectories=10, terminal_time=1.0, n_steps=200,
    gas_cost_model=gas_model, seed=42,
)
obs, _ = env.reset()
print(obs["gas_cost"])  # Cost available to the next rebalance action
```

The process follows `dX = theta * (mu - X) dt + sigma dW`, using the exact
Gaussian transition and independent noise per trajectory. `theta` must be
positive; `mu`, `sigma`, and `initial_cost` must be nonnegative. The process's
trajectory count and timestep must match the environment. The injected model
takes precedence over the scalar `gas_cost` setting.

Charged gas is `max(X, 0)`. The latent OU state remains unclipped, so `mu` is
the latent mean and the average charged cost can exceed it. Each action uses
the gas cost in its observation, then gas advances once for the next simulator
step, including hold steps. First deployment and holds are free; subsequent
rebalances deduct gas plus any configured swap cost, with wealth floored at
zero. PnL rewards already include these deductions.

Reset restores the initial gas cost. `reset(seed=...)` also replays the RNG
stream; resets without a seed continue that stream. Gas uses environment seed
plus 4, independently of the existing market streams.

Raw and Gymnasium observations include `gas_cost` in token1 units; it also
remains available through explicit SB3 `obs_keys`. The default SB3 observation
vector has eight features, in order: `mispricing`,
`lp_lower_offset`, `lp_upper_offset`, `time`, `has_position`,
`portfolio_value_ratio`, `pending_fee_share`, and `relative_gas_cost`.
The last two features are dimensionless:

- `pending_fee_share` (index 6) is pending fee value divided by total portfolio
  value, or zero when portfolio value is zero. Pending token0 fees are valued
  at the external midprice and added to pending token1 fees. Total portfolio
  value already includes these fees.
- `relative_gas_cost` (index 7) is total portfolio value divided by
  `max(gas_cost, 1e-12)`. Larger values mean the portfolio can cover more gas
  charges: a portfolio worth 1,000 with gas cost 2 produces 500. Zero gas gives
  a large finite value for positive wealth; zero wealth gives zero.

The existing `VecNormalize` wrapper normalizes these features when enabled.
Stochastic gas is configured through Python; the training CLI continues to
use fixed gas. Policy plotting helpers accept keyword-only `initial_wealth`
to set the neutral portfolio value used for the relative gas feature; it
defaults to `experiments.helpers.INITIAL_WEALTH`, with pending fees set to zero.

The derived `boundary_proximity` and `position_width` observations have been
removed completely. Saved policies and normalization statistics that include
either feature require their original code version or retraining with fresh
normalization statistics. Older nine- and ten-feature defaults are incompatible
with the current eight-feature layout.

The former eight-feature layout using `unclaimed_fee_value_ratio` and raw
`gas_cost` is also incompatible, despite having the same dimension. Those
checkpoints and normalization statistics require their original code version
or retraining with fresh statistics. The old `unclaimed_fee_value_ratio`
derived feature has been removed.

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
- Categorical `StableBaselinesAMMEnvironment` (`Discrete(211)` by default)
- SB3 `VecNormalize` unless `--no-normalise-obs` is passed

Important script defaults (also used by the HPC training launchers):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--total-timesteps` | `10_000_000` | Simulator-equivalent training budget |
| `--num-trajectories` | `100` | Vectorized paths per episode |
| `--n-steps` | `1000` | Simulator steps per episode |
| `--decision-stride` | `100` | PPO acts once every 100 simulator steps |
| `--tau` | `50` | Maximum absolute tick offset of either range endpoint |
| `--tick-stride` | `5` | Categorical endpoint grid spacing |
| `--alpha3` | `4000.0` | arbitrage arrival rate (per-unit of mispricing) |
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

## License

Original contributions to SAiFE_gym are licensed under the [MIT License](LICENSE).
Reused code from `mbt_gym` retains its BSD-3-Clause terms. See
[third-party notices](THIRD_PARTY_NOTICES.md) for attribution and the full upstream
license.
