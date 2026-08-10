# Concentrated Liquidity Provision: a Reinforcement Learning Perspective

This is the anonymised code repository that accompanies the ICAIF submission
*"Concentrated Liquidity Provision: a Reinforcement Learning Perspective"*.

It contains `amm_sim`, a simulator of a Uniswapv3 style concentrated-liquidity pool, together with the reinforcement-learning agents, the closed-form and heuristic benchmarks, and the full analysis pipeline used to produce the results reported in the paper.

The liquidity provider chooses, at each decision point, a tick range
`[lower_offset, upper_offset]` relative to the current pool price, or holds its
existing position.

## Requirements

Python 3.11 or newer. Install the pinned dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies are `gymnasium`, `stable-baselines3`, `torch`, `numpy`,
`pandas`, `scipy` and `matplotlib`.

## Replicating the experimental results

The pipeline runs in four stages: configure, train, evaluate, analyse. Each
stage writes artefacts that the next stage consumes, so the stages can be run
independently and re-entered without repeating earlier work.

### 1. Configuration

Every parameter of an experiment is declared in `experiment_config.py`. Each
variable there is a *list* of values, and the Cartesian product of all lists
defines the experiment grid: market parameters (volatility, drift, initial
price, fee tier, tick width `tau`), order-flow coefficients, the reward
function, the decision stride, and the PPO hyperparameters. Freezing a variable
means wrapping a single value in a one-element list.

Which agents take part in a run is controlled by the `ENABLE_AGENTS` dictionary
in the same file. The configuration used for the reported results enables two
learned agents — `PPO`, and `PPO_narrow`, a restricted variant whose position
half-width is fixed — against four benchmarks: `DeployNarrow`, `DeployWide`,
`ArrivalRebalance`, and `CDM`, the closed-form Cartea–Drissi–Monga policy.

### 2. Training

The main training and in-run evaluation of the RL algorithms happens in
`agent_comparison.py`. It reads one grid point from `experiment_config.py`,
trains every enabled learned agent on it, evaluates all agents, and writes the
resulting figures and logs:

```bash
python agent_comparison.py --config-idx 0
```

Each run produces a self-contained output directory holding the trained model
weights, the observation-normalisation statistics, a verbatim dump of the exact
configuration used, and a log of all printed tables. Because the configuration
is written alongside the weights, every run is reproducible from its own output
directory without reference to the state of `experiment_config.py` at the time.

For a full sweep, `submit.sh` expands the configuration grid and submits one
array task per combination on a SLURM cluster. On a single machine, loop over
`--config-idx` from `0` to `count - 1` instead.

### 3. Evaluation on common seeds

Training runs each evaluate on a seed derived from their own training seed,
which confounds *policy* differences with *market-path* differences. To separate
the two, all saved models are re-evaluated on the **same** evaluation seeds in
`reevaluate_saved_models.py`. It reloads every persisted policy together with
its normalisation statistics, rebuilds the exact environment each was trained
against, and replays all of them over one identical set of trajectories:

```bash
python reevaluate_saved_models.py --root <results-root> \
    --out-csv reevaluated.csv --dump-trajectories traj_dumps
```

The evaluation seed is fixed (`--eval-seed`, default `1005`) and deliberately
disjoint from the set of training seeds, so no policy is ever scored on the
random stream its own training environment used. With every policy facing
identical market paths, the remaining spread across training seeds is
attributable to training variance alone, and agent-versus-agent comparisons
become properly paired.

`--dump-trajectories` additionally writes the per-trajectory profit-and-loss and
its attribution into fees, impermanent loss, gas and the buy-and-hold reference,
which is what the statistical tests consume.

### 4. Statistical tests, tables and figures

`paired_tests.py` consumes the per-trajectory dumps and tests, for each cell of
the grid, whether the RL agent's advantage over the strongest benchmark is
distinguishable from noise. It reports paired tests on mean profit-and-loss and
on the conditional value at risk of the tail, bootstrapped and corrected for
multiple comparisons across cells:

```bash
python paired_tests.py --dumps traj_dumps --out-csv paired_tests.csv
```

`parse_results.py` and `parse_results_with_seeds.py` scrape the logged
evaluation and attribution tables out of completed runs; `results_to_dataframe.py`
collects them into a single tidy frame, which `plot_results.py` turns into the
faceted comparison figures.

`probe_policy.py` evaluates a trained policy on a synthetic grid of states and
plots the learned quoting rule as a function of mispricing and inventory,
producing the policy-behaviour figures. `mispricing_response.py` plots the
policy's reaction to the mispricing signal alone.

```bash
python probe_policy.py --run-dir <run-directory> --agent ppo
```

## Determinism

Every stochastic component — the price process, the arrival process, the pool
dynamics and the policy — is seeded explicitly from the configuration, so a
given `--config-idx` reproduces bit-identical results on the same software
stack. Reported comparisons use the common evaluation seeds described in
stage 3 rather than each run's own training seed.

