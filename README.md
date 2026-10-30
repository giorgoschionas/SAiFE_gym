# SAiFE Gym

A Gymnasium-compatible reinforcement learning environment for training
liquidity provision (LP) agents in Uniswap v3 automated market makers with
concentrated liquidity.

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

## Dense extended CARA reward

`DenseExponentialUtility` gives a reward at every simulator step using
any finite `risk_aversion`. The LP starts in cash, so its cumulative PnL is
`G_t = portfolio_value_t - initial_wealth`. For `a = risk_aversion`,

```text
U(G) = (1 - exp(-a * G)) / a    if a != 0
U(G) = G                       if a == 0
reward_t = U(G_next) - U(G_current)
sum(rewards) = U(G_terminal)    because U(0) = 0
```

| `risk_aversion` | Profile | Utility in PnL |
| --- | --- | --- |
| Positive | Risk averse | Increasing and concave |
| Zero | Risk neutral | Linear; the reward equals per-step PnL |
| Negative | Risk seeking | Increasing and convex |

This generalizes the previous `-exp(-a * G)` dense reward. For positive `a`,
each reward is now the previous reward divided by `a`. Expected-utility
rankings and optimal widths are preserved for a fixed positive parameter;
reward magnitudes and learning behavior can change. Previously saved sweep
outputs retain the old scale, so compare them with new runs after rescaling.

Use an RL discount factor of `gamma=1.0` to preserve this terminal-utility
objective. The last transition receives the same increment as every other
step; no extra terminal utility is added. Applying utility separately to each
step's PnL would define a different objective. The existing `ExponentialUtility`
class remains available for terminal-only utility on total wealth.

```python
from experiments.helpers import get_amm_env, get_ppo_learner_and_callback
from SAiFE_gym.rewards.RewardFunctions import DenseExponentialUtility

env = get_amm_env(
    num_trajectories=50,
    # This helper uses initial_wealth=1e6.
    reward_function=DenseExponentialUtility(risk_aversion=1e-6),
)
model, callback = get_ppo_learner_and_callback(env)  # gamma=1.0, norm_reward=False

# The same dense reward supports the other profiles:
neutral_reward = DenseExponentialUtility(risk_aversion=0.0)
seeking_reward = DenseExponentialUtility(risk_aversion=-1e-6)
```

`risk_aversion` is measured in inverse token1 units. Choose it for the monetary
PnL scale: the example gives `a * G = -1` at a total loss of the initial capital.
The reward does not automatically normalize PnL or clip utility. It uses a stable
exponential difference for small changes and large reversals, and raises
`FloatingPointError` if a reward cannot be represented in float64. Reduce
`abs(risk_aversion)` if the exponent becomes too large. Positive parameters
amplify losses; negative parameters amplify gains. Reward clipping or a changing
reward normalization would break the exact telescoping identity.

### Periodic baseline width sweep

Run the periodic baseline with the nominal environment and defaults from
`experiments/train_robust_lp_agent.py`, replacing its inventory-penalty reward:

```bash
venv/bin/python experiments/sweep_dense_exponential_width.py
```

This evaluates 12 symmetric half-widths from 1 to 50 ticks and
`risk_aversion` values `0.0001, 0.001, 0.003, 0.01, 0.03, 0.1`. A half-width
`w` means offsets `[-w, +w]`, spanning `2*w` ticks. Defaults are 10 episodes
of 100 trajectories per width, 1,000 simulator steps, rebalancing every
100 steps, initial cash 1,000, nominal volatility 0.03, baseline arrival
rate 300 per side, and gas cost 2. Domain randomization is disabled.

Override the sweep or the trainer's environment arguments directly:

```bash
venv/bin/python experiments/sweep_dense_exponential_width.py \
  --half-widths 5 10 20 30 40 50 \
  --risk-aversion-values 0.001 0.01 0.1 \
  --n-eval-episodes 20 --workers 4
```

To compare all three profiles, use `--risk-aversion-values -0.01 0 0.01`.
Negative and zero parameters are supported in the summaries and plots.

For a smaller exploratory run with 200 paths per width, add
`--num-trajectories 10 --n-eval-episodes 20`. This changes evaluation batching
and sample count only; the market and policy parameters stay fixed. Narrow
confidence intervals should not be assumed at this sample size, especially
for large risk aversion, where a few losses can dominate expected utility.

Results go into a new timestamped directory under
`experiments/results/dense_exponential_width_sweep/`: configuration, per-path
PnL and total rewards, accounting diagnostics, CSV summaries, and PNG/PDF
plots. `--smoke-test` uses two widths and four total paths per width while
preserving the full episode horizon. `--workers 1` runs sequentially.

The deterministic baseline ignores rewards. Each market path is therefore
simulated once per width, and the production reward is applied to **every
recorded transition** for all risk parameters. The first risk parameter also
runs inside the environment; every path is checked against the terminal
identity. Widths use matched episode seeds, preserving external price shocks,
although their endogenous swap paths can differ. All reward arithmetic remains
float64, with discount one and no reward normalization or clipping.

The score is the **mean of total rewards across paths**, not utility applied
to mean PnL. The certainty equivalent is also reported in token1 units:
`CE = -log(mean(exp(-risk_aversion * final_PnL))) / risk_aversion` for nonzero
parameters, and `CE = mean(final_PnL)` at zero.
For a fixed risk parameter it gives exactly the same ordering as expected
utility. A cash baseline has both total reward and CE equal to zero.

Uncertainty uses a paired bootstrap of entire episode batches because swap
execution order is shared among trajectories within a batch. Intervals are
pointwise, and a selected grid maximum is an estimate, not proof of the
population optimum. The summary also records bootstrap selection frequencies
and utility-weight concentration to expose tail-sensitive estimates. Concavity
is checked using changes in secant slopes on the sampled width grid; no curve
is forced to be concave. Concavity of utility **in PnL** alone guarantees
neither concavity **in width** nor increasing optimal width with risk aversion.

### Relation to the continuous-time dynamics

For `G = W - W_0` and the extended CARA utility above, Ito's formula uses
`U'(G) = exp(-a * G)` and `U''(G) = -a * exp(-a * G)`. If
`dG = b dt + v dW + sum(kappa_j dN_j)`, then

```text
dU = exp(-a * G_before) * [
    (b - 0.5 * a * v^2) dt
    + v dW
    + sum(U(kappa_j) dN_j)
]
```

The jump term uses the same piecewise utility, so at `a=0` this reduces to
`dU=dG`. The diffusion correction penalizes variance for positive `a` and
rewards it for negative `a`.

The implementation evaluates exact utility differences between simulated
states. It needs no intensity estimates, extra variance penalty, or additional
`dt` factor. Several accounting assumptions matter when applying a continuous
AMM derivation to this simulator:

- **Cash benchmark:** holding the reset inventory has constant value `W_0`.
  Thus `G` is total PnL from cash. Calling its fee-exclusive counterpart
  "impermanent loss" differs from the usual benchmark of holding the tokens
  deposited into an LP position.
- **Continuous valuation:** pure-jump reserves do not imply pure-jump PnL or
  impermanent loss. Existing risky holdings change value whenever the external
  price changes, even without a swap.
- **Fees in both tokens:** unclaimed token0 fees retain price exposure.
  A scalar fee value is not pure-jump unless risky-token fees are immediately
  converted to cash. The diffusion exposure includes those fee holdings.
  Portfolio value already includes claimable fees; collected fees are
  reinvested on rebalance, so adding lifetime fee counters would double-count.
- **LP control events:** initial deployment and subsequent rebalances change
  holdings independently of swap arrivals. Rebalancing gas and conversion costs
  must enter the wealth change. One fixed reserve map based only on net swap
  count is insufficient for a controlled LP whose range and liquidity change.
  The simulator currently charges these costs only for an existing position,
  making the first deployment free of those explicit costs.
- **Model assumptions:** a positive initial price alone does not guarantee a
  general diffusion stays positive. The simulator discretizes prices and
  arrivals, and can execute both swap directions within one step. Utility
  differences capture its realized transitions; they do not remove these
  market-model approximations. Expected-drift arguments also require the
  exponential integrability conditions for the transformed martingales.

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
| `midprice` | `(num_trajectories,)` | External market price |
| `time` | `(num_trajectories,)` | Current simulation time |
| `gas_cost` | `(num_trajectories,)` | Fixed rebalance cost for the episode |
| `initial_wealth` | `(num_trajectories,)` | Starting LP capital |
| `portfolio_value` | `(num_trajectories,)` | Current mark-to-market LP value |
| `lp_alpha` | `(num_trajectories,)` | Token0 value fraction of the LP position |
| `lp_token0_amount` | `(num_trajectories,)` | Absolute LP token0 inventory |

`sqrt_price` stores sqrt(P), following the Uniswap v3 convention. Convert with
`price = sqrt_price ** 2`.

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

`AMMEnvironment.action_space` is the low-level simulator command space:

```text
Box(low=[-tau, -tau+1, -1.0], high=[tau-1, tau, 1.0], shape=(3,), dtype=float32)
```

| Idx | Name | Range | Description |
|-----|------|-------|-------------|
| `[0]` | `lower_offset` | `[-tau, tau-1]` | Lower tick bound offset from current pool tick |
| `[1]` | `upper_offset` | `[-tau+1, tau]` | Upper tick bound offset from current pool tick |
| `[2]` | `hold_flag` | `[-1.0, 1.0]` | `<= 0` rebalances; `> 0` holds the existing position |

`validate_action` rounds offsets to integer ticks, clips them to the action
bounds, and enforces `lower_offset < upper_offset`.

Use wrapper action spaces when training agents:

- `DiscreteActionWrapper` exposes one hold action plus finite rebalance ranges
  for raw Gymnasium use.
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
