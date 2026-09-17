# Slurm jobs for domain-randomized LP PPO

These templates run the domain-randomized PPO experiment on a Slurm cluster.
They use Python 3.12 and the checkout's `requirements.txt`. Adapt paths,
optional Python module loading, partitions, time limits, and resource requests
to your allocation. Submit from the checkout root, with `logs/` already created
so Slurm can open its output files before the job starts.

## Private cluster settings

For a new checkout:

```bash
git clone https://github.com/giorgoschionas/SAiFE_gym.git
cd SAiFE_gym
```

Create a private configuration once, unless you already have one:

```bash
cp hpc/.env.example hpc/.env
```

Edit `hpc/.env` for your cluster. It is a shell file containing exported
variables and is ignored by Git; only the generic `.env.example` is published.
If moving between checkouts or machines, copy your private configuration
separately. Pulling the repository does not copy it. Use absolute paths for
an existing checkout and virtual environment on shared storage accessible
from compute nodes.

| Setting | Default and purpose |
|---|---|
| `SAIFE_PROJECT_DIR` | Jobs use `SLURM_SUBMIT_DIR`, then the current directory. Setup uses its own checkout. The example config uses `$PWD` when sourced from the checkout root. |
| `SAIFE_VENV_DIR` | `venv/` inside the project. Override to reuse an existing environment; relative values resolve inside the project. |
| `SAIFE_PYTHON_MODULE` | Empty: leave loaded modules unchanged. If set, the scripts require the `module` command, purge modules, and load this exact module before activating the environment. |
| `SBATCH_PARTITION` | Omitted: use the cluster's default partition. Set this in your private config to select a training partition. |
| `SBATCH_ACCOUNT` | Optional allocation/account, handled by Slurm. |

At the start of each submission session, from the checkout root:

```bash
source hpc/.env
cd "$SAIFE_PROJECT_DIR"
mkdir -p logs
```

Source this file **before** calling `sbatch`: the jobs do not source it
automatically. Slurm selects partitions at submission time, and shell variables
are not expanded in `#SBATCH` directives. `SBATCH_PARTITION` supplies a default;
`sbatch --partition=YOUR_PARTITION ...` overrides it for a particular job.
See the [Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html).
The launchers keep `--export=ALL` so your exported settings reach the jobs.
Avoid `--export=NONE` when relying on this configuration.

The jobs preserve their single-node resource requests: 16 tasks for training
and each sweep replicate, two for the smoke job, and one for aggregation.
Their thread limits follow `SLURM_NTASKS`. Adjust resources and time limits
with `sbatch` options as required by your allocation.

New outputs default to these directories inside the checkout:

| Job | Default results directory |
|---|---|
| Smoke | `experiments/results/domain_randomized_ppo/smoke` |
| Full training | `experiments/results/domain_randomized_ppo/full` |
| Seed sweep | `experiments/results/domain_randomized_ppo/seed_sweep/seed_<seed>` |
| Aggregation | `experiments/results/domain_randomized_ppo/seed_sweep/aggregate` |

Set `OUTPUT_DIR` to override a job's destination. For a sweep it is the parent
directory, with `seed_<seed>` appended by each array task. Set aggregation's
`INPUT_DIR` to that same parent; its output defaults to `$INPUT_DIR/aggregate`.
Existing saved results are not moved or renamed. Supply their original paths
through `INPUT_DIR`, `OUTPUT_DIR`, or the evaluator's run-directory arguments
to continue using them. Relative output paths resolve inside the checkout.

Jobs preserve an exported `MPLCONFIGDIR`; otherwise the training launchers
create a unique Matplotlib cache beneath `${TMPDIR:-/tmp}`.

## One-time setup

Use an interactive or compute session permitted for package installation by
your cluster. After configuring and sourcing `hpc/.env`, run:

```bash
bash hpc/setup_env.sh
```

This creates the virtual environment and output directories, installs the
pinned dependencies, and prints Python, NumPy, SB3, and Torch versions.
If you already have a compatible environment, set `SAIFE_VENV_DIR` to it and
skip installation. Each job loads the selected module and activates the
environment itself.

Setup runs in a child shell. To run Python commands interactively afterward,
load the same module in your current shell when configured, then activate:

```bash
if [ -n "${SAIFE_PYTHON_MODULE:-}" ]; then
  module purge
  module load "$SAIFE_PYTHON_MODULE"
fi
source "$SAIFE_VENV_DIR/bin/activate"
```

## Smoke test

Submit a tiny end-to-end job first:

```bash
cd "$SAIFE_PROJECT_DIR"
sbatch hpc/sbatch_robust_lp_smoke.sh
```

If your cluster has a separate short-job partition, select it explicitly:

```bash
sbatch --partition=YOUR_SHORT_PARTITION hpc/sbatch_robust_lp_smoke.sh
```

Check that the job finishes successfully and saves both PPO models and an
`evaluation_grid.csv` containing 16 rows. This verifies module loading,
environment activation, and training on your actual compute nodes.
All submission examples assume `logs/` exists before calling `sbatch`.

## Full training

Submit the default production run:

```bash
cd "$SAIFE_PROJECT_DIR"
sbatch hpc/sbatch_train_robust_lp_agent_cpu.sh
```

Both the single-training and seed-sweep launchers default to
`TOTAL_TIMESTEPS=10000000`, `DECISION_STRIDE=100`, `INITIAL_WEALTH=1000.0`,
and `TAU=50`.
Here, `TAU` controls the policy's maximum center offset and half-width in ticks;
it does not force every position to have width 200. These values can still be
overridden through exported environment variables.

Both launchers default to `NOMINAL_SIGMA=0.03` and
`NOMINAL_ARRIVAL_RATE=300.0` for nominal training and convergence validation.
These are the midpoints of the randomized training ranges, sigma `[0.01, 0.05]`
and arrival rate `[200.0, 400.0]`. The final in-distribution evaluation grid
includes the nominal condition. Override either nominal parameter through its
exported environment variable when needed.

`TOTAL_TIMESTEPS` is a simulator-equivalent budget. With the default
`DECISION_STRIDE=100`, PPO acts once every 100 simulator steps, so the default
run trains on 100,000 PPO decision timesteps and each 1,000-step episode exposes
at most 10 agent decisions. The simulator uses forced hold actions between
agent decisions.

With the default `TAU=50` and `tick_stride=5`, the PPO action space is
`MultiDiscrete([21, 10, 2])`.

Override parameters at submission time when needed:

```bash
sbatch --export=ALL,TOTAL_TIMESTEPS=10000000,NUM_TRAJECTORIES=128,N_EVAL_EPISODES=20 \
  hpc/sbatch_train_robust_lp_agent_cpu.sh
```

With `--export=ALL,NAME=value`, an already exported `NAME` takes precedence
over the value in the option. To replace an existing exported value for one
submission, use a shell assignment instead, such as
`NUM_TRAJECTORIES=128 sbatch hpc/sbatch_train_robust_lp_agent_cpu.sh`.

The training job writes convergence traces during learning:
`domain_randomized_convergence.csv` and `nominal_convergence.csv`. By default
these evaluate the current deterministic policy every 10 PPO rollouts on the
fixed nominal validation regime. Override with
`CONVERGENCE_EVAL_EVERY_ROLLOUTS` and `CONVERGENCE_N_EVAL_EPISODES`, or set
`CONVERGENCE_EVAL_EVERY_ROLLOUTS=0` to disable.

Domain-randomized training uses one sampled domain per trajectory by default. Set
`TRAIN_DOMAINS_PER_RESET=1` to restore a single shared domain, or choose any
value from `1` through `NUM_TRAJECTORIES` for balanced grouped assignments.

Final evaluation uses the full Cartesian product of the combined volatility
and arrival-rate lists. The defaults cover all 25 combinations of sigma
`[0.015, 0.030, 0.045, 0.065, 0.08]` and arrival rate
`[150.0, 250.0, 300.0, 350.0, 450.0]`. Each combination is evaluated once for
nominal PPO, domain-randomized PPO, periodic rebalancing, and cash, producing
100 rows per training seed in `evaluation_grid.csv`.

Results and summaries distinguish four `evaluation_set` values, classified
against the randomized training ranges (including their endpoints):

| Evaluation set | Volatility | Arrival rate | Default regimes |
|---|---|---|---:|
| `in_distribution` | Inside | Inside | 9 |
| `sigma_only_stress` | Outside | Inside | 6 |
| `arrival_only_stress` | Inside | Outside | 6 |
| `stress` | Outside | Outside | 4 |

The original 13 regimes retain their order and evaluation seeds. The generator
then appends in-distribution volatility crossed with stress arrivals, followed
by stress volatility crossed with in-distribution arrivals, skipping duplicate
combinations. With the default evaluation seed, the original regime seeds are
100042–112042 in steps of 1000; the additions use 113042–124042. Configured axis
order is significant for these seed assignments. Overlapping axis lists are
deduplicated, and group labels depend on actual support, not the list names.

Override the axis lists independently with `EVAL_IN_DISTRIBUTION_SIGMA_VALUES`,
`EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES`,
`EVAL_STRESS_SIGMA_VALUES`, and `EVAL_STRESS_ARRIVAL_RATE_VALUES`. Each
variable is a space-separated list. Gas cost is fixed for the run via
`NOMINAL_GAS_COST`; the training script no longer supports gas-cost evaluation
grids. For example:

```bash
sbatch --export=ALL,EVAL_STRESS_SIGMA_VALUES="0.065 0.08",NOMINAL_GAS_COST=2.0 \
  hpc/sbatch_train_robust_lp_agent_cpu.sh
```

Every in-distribution value must lie within its corresponding training range,
and every tuple from the explicitly supplied stress axes must have at least
one value outside training support. These existing constraints still apply
before the cross combinations are added.

The usual submission commands enable the full grid without additional flags.
The training budget and nominal convergence checks are unchanged. Final
evaluation covers 25 rather than 13 regimes, approximately 1.92 times as much
final-evaluation work; this is not a multiplier for the entire job runtime.
Smoke mode uses a full 2 × 2 grid with one regime in each group and 16 result
rows.

## Moderate-arrival seed sweep

This experiment uses submission-time overrides for
`hpc/sbatch_robust_lp_seed_sweep_cpu.sh`; the Python and launcher defaults remain
as documented above. It trains on the following market parameters:

| Parameter | Domain-randomized PPO | Nominal PPO |
|---|---|---|
| Volatility sigma | Uniform `[0.015, 0.045]` | `0.030` |
| Baseline arrival alpha1 | Uniform `[300, 600]` | `450` |

The randomized agent samples the two parameters independently. The nominal
condition is the midpoint of both ranges and is also used for convergence
validation. Each reset assigns one sampled domain per trajectory across the
100 training trajectories.

These ranges follow the periodic baseline's original-resolution results: mean
total reward was `+32.13` at `(sigma=0.030, alpha1=450)`, `-10.80` at
`(0.045, 300)`, and `+12.64` at `(0.045, 600)`. The independent confirmation at
`(0.045, 450)` gave `+1.76`. These measurements motivate a training region that
includes profitable and moderately negative conditions; they do not establish
profitability at every point in the continuous region or guarantee PPO's
performance. Reward is PnL minus the accumulated inventory penalty, with gas
already included in PnL.

Final evaluation crosses sigma `[0.015, 0.030, 0.045, 0.050]` with arrival
`[250, 300, 450, 600, 800]`, giving a **4 x 5 heatmap**. Only sigma `0.050` is
outside the volatility training range; arrivals `250` and `800` are outside
the arrival training range. Arrival `800` tests extrapolation toward a more
favorable market. The sigma `0.050` row was not measured in the preceding
periodic baseline sweep.

| Evaluation set | Conditions | Rows per seed (four policies) |
|---|---:|---:|
| `in_distribution` | 9 | 36 |
| `sigma_only_stress` | 3 | 12 |
| `arrival_only_stress` | 6 | 24 |
| `stress` | 2 | 8 |
| Total | 20 | 80 |

From the HPC project root, with the full Cartesian evaluation update present,
paste this single line to submit seeds 43 through 52:

```bash
mkdir -p logs && sbatch --export=ALL,TOTAL_TIMESTEPS=10000000,N_STEPS=1000,DECISION_STRIDE=100,NUM_TRAJECTORIES=100,TRAIN_DOMAINS_PER_RESET=100,N_EVAL_EPISODES=10,EVALUATION_SEED=100042,TAU=50,ALPHA3=4000,INITIAL_WEALTH=1000,INVENTORY_PHI=0.4,NOMINAL_GAS_COST=2,PERIODIC_REBALANCE_EVERY=100,PERIODIC_WIDTH=50,NOMINAL_SIGMA=0.030,NOMINAL_ARRIVAL_RATE=450,TRAIN_SIGMA_MIN=0.015,TRAIN_SIGMA_MAX=0.045,TRAIN_ARRIVAL_RATE_MIN=300,TRAIN_ARRIVAL_RATE_MAX=600,EVAL_IN_DISTRIBUTION_SIGMA_VALUES="0.015 0.030 0.045",EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES="300 450 600",EVAL_STRESS_SIGMA_VALUES=0.050,EVAL_STRESS_ARRIVAL_RATE_VALUES="250 800",OUTPUT_DIR=experiments/results/domain_randomized_ppo/seed_sweep/moderate_arrivals_10m hpc/sbatch_robust_lp_seed_sweep_cpu.sh
```

Each policy retains a budget of 10 million simulator-equivalent steps: 100,000
PPO decision timesteps with ten decisions per 1,000-step episode. The periodic
baseline rebalances every 100 simulator steps. Initial wealth, gas, inventory
penalty, action ranges, and alpha3 match the preceding experiment. Conclusions
remain specific to this resolution; the earlier high-arrival diagnostics found
material changes in reward when the simulator timestep was refined.

The launcher saves runs under
`experiments/results/domain_randomized_ppo/seed_sweep/moderate_arrivals_10m/seed_<seed>/run_<timestamp>/`.
Use a different `OUTPUT_DIR` for a repeat sweep so aggregation does not encounter
duplicate training seeds. After all ten array tasks finish successfully, run:

```bash
sbatch --export=ALL,INPUT_DIR=experiments/results/domain_randomized_ppo/seed_sweep/moderate_arrivals_10m,OUTPUT_DIR=experiments/results/domain_randomized_ppo/seed_sweep/moderate_arrivals_10m/aggregate hpc/sbatch_aggregate_domain_randomized_seed_sweep.sh
```

Check that each saved `config.json` contains the selected bounds, nominal
parameters, and `evaluation_grid_version: 2`. Expect ten seed runs, each with
80 rows in `evaluation_grid.csv` and 1,000 evaluation paths per row. Aggregation
validates full coverage and computes paired randomized-minus-nominal reward
differences across training seeds. Compare both PPO agents with periodic
rebalancing and cash, and report reward and PnL separately.

## Evaluate higher arrival rates without training

`experiments/evaluate_robust_lp_agents.py` loads a saved run's configuration and
evaluates the periodic baseline by default. It does not call PPO training or
change the saved models/results. Use a completed run from the smoke or full
training instructions above. Alternatively, generate a small example run
from the project root with the project environment activated:

```bash
python experiments/train_robust_lp_agent.py \
  --smoke-test \
  --output-dir experiments/results/domain_randomized_ppo/evaluator_example
```

This creates `smoke_<timestamp>/` containing `config.json`, both PPO model
ZIPs, their `VecNormalize` files, and the training evaluation results. Smoke
models demonstrate the workflow; use a full training run for policy comparisons.

Replace `<timestamp>` below with the generated directory's actual timestamp,
or set `SAIFE_RUN_DIR` to your completed full run. On a compute node (or locally
for the small example), run:

```bash
SAIFE_RUN_DIR="experiments/results/domain_randomized_ppo/evaluator_example/smoke_<timestamp>"
python experiments/evaluate_robust_lp_agents.py \
  --run-dirs "$SAIFE_RUN_DIR" \
  --output-dir experiments/results/periodic_high_arrival/main_grid \
  --n-eval-episodes 2 \
  --workers 4
```

The default diagnostic grid is sigma `[0.03, 0.045, 0.065, 0.08]` crossed with
arrival `[300, 450, 600, 800, 1000]`. Episode count and trajectory batch size
default to the saved configuration unless overridden. The standalone evaluator
requires at least two evaluation episodes, so the command overrides the one
episode saved by smoke mode. Increase this count for substantive comparisons;
the smoke run uses only two trajectories. Choose a new `--output-dir` for each
independent evaluation.

Set `--policies periodic_rebalance nominal_ppo domain_randomized_ppo` to evaluate
saved PPO models as well; provide multiple `--run-dirs` for a training-seed
comparison. The fixed periodic baseline is evaluated once, not repeated for
each training seed. PPO requires each model's matching normalization file when
the saved run used it.

The evaluation runner uses occupied-range fee summation and copies only the
previous-state fields needed by `RunningInventoryPenalty`. These accounting
shortcuts preserve the market and action rules and are checked against the
original evaluator. Use `--reference-evaluator` to disable them. Production
training and its final evaluations retain their original implementations.

For a resolution check, add `--sigma-values .065 .08 --arrival-rate-values 1000
--n-steps-values 1000 5000 10000` and choose a fresh output directory. Both the
PPO decision stride and the periodic rebalance interval scale automatically to
preserve their physical timing; incompatible resolutions are rejected. These
are evaluations at different simulator resolutions, not identical coupled
market paths. Arrivals remain boolean indicators, at most one per side per
step, with probability `1 - exp(-intensity * dt)`. The periodic diagnostics
record indicator counts and actual `intensity * dt` to expose this truncation.

Outputs include per-episode CSVs, `evaluation_summary.csv`, a provenance
`manifest.json`, and reward-versus-arrival PNG/PDF plots. The marginal 95%
bootstrap intervals resample vectorized episodes, since trajectories within
each episode share swap-order randomness. They measure evaluation uncertainty,
not training-seed uncertainty. Existing output directories are rejected to
protect earlier results unless `--resume` is supplied. To recover an interrupted
sweep, repeat the original command with `--resume`: completed episode checkpoints
are validated and reused, and only missing episodes are simulated. The grid,
seeds, evaluation settings, and simulation source must agree with the saved
manifest. A directory lock prevents two active sweep processes from writing the
same outputs, and checkpoint CSVs are replaced atomically after each episode.
For independent follow-up episodes, use a fresh
directory and `--episode-offset 100`; seed assignments remain stable when
subsetting or reordering the diagnostic grid. Existing cells retain the source
run's regime seeds. Keep these diagnostic outputs separate from the production
seed-sweep aggregator, which expects its configured grid and all four policies.

To plot matched periodic-LP and nominal-PPO paths from a saved run, supply the
run explicitly; the plotting tool has no default checkpoint location:

```bash
python experiments/plot_nominal_lp_strategy_comparison.py --run-dir "$SAIFE_RUN_DIR"
```

## Seed sweep

The default array trains ten independent PPO replicates (seeds 43 through 52)
while evaluating every learned network on the same paths using
`EVALUATION_SEED=100042`. Submit the sweep and its aggregation job with a Slurm
dependency:

```bash
cd "$SAIFE_PROJECT_DIR"
SWEEP_JOB_ID=$(sbatch --parsable hpc/sbatch_robust_lp_seed_sweep_cpu.sh)
sbatch --dependency=afterok:$SWEEP_JOB_ID \
  hpc/sbatch_aggregate_domain_randomized_seed_sweep.sh
```

Add `--partition=YOUR_SHORT_PARTITION` to the aggregation submission if it
should use a different partition from training. Keep the `afterok` dependency.

Edit both `SEEDS=(...)` and the Slurm array bounds in the sweep script if you
want a different replicate count. The aggregation job runs only after every
array task succeeds and rejects duplicate seeds, incomplete grids, legacy
schemas, or incompatible configurations.

Each run must include its four evaluation-grid axes, `n_eval_episodes`, and
`num_trajectories` in `config.json`. New runs also record
`evaluation_grid_version: 2` and the randomized training bounds; aggregation
checks exact full-grid coverage and group labels for all four policies.
Existing configurations without the version marker (or with version 1) retain
the original two-grid interpretation and remain aggregatable. Unsupported
versions are rejected. Keep old and new grid versions in separate sweep
directories; aggregation rejects mixing them, and source results are not
migrated or edited.

Every row's evaluation path count must equal
`n_eval_episodes * num_trajectories`. Missing or unexpected
regimes are rejected even if every training seed has the same discrepancy.
Saved policy gaps must agree with the differences between policy means
(`rtol=1e-9`, `atol=1e-7`); inconsistent files are rejected, not repaired.
Aggregate gap statistics are calculated directly from the policy means.
Validation completes before aggregate outputs are created or overwritten, and
source result files are never modified. Valid existing runs need no retraining.

Each individual `evaluation_grid.csv` reports
`evaluation_path_std_running_inventory_objective`, which measures variation
over evaluation paths for one learned network. The aggregate artifacts report
`training_seed_std_running_inventory_objective`, standard errors, and
percentile-bootstrap confidence intervals across independently trained
networks. These are distinct uncertainty sources and should not be interpreted
interchangeably.

The aggregation job writes `training_seed_regime_summary.csv`,
`training_seed_gap_summary.csv`, `training_seed_behavior_summary.csv`, and
`training_seed_summary.json` under the seed-sweep aggregate directory.
`training_seed_behavior_summary.csv` is long-form: each row reports one
policy, regime, and behavior diagnostic with its training-seed mean, sample
standard deviation, standard error, and bootstrap interval. Periodic-rebalance
and cash results use the same fixed evaluation paths in every replicate, so
their across-training-seed standard deviations should be zero.

Every per-seed `evaluation_grid.csv` also reports policy behavior diagnostics:

- `never_deployed_fraction` and `bankruptcy_fraction` expose cash collapse and
  terminal loss of all deployed wealth.
- `hold_action_fraction`, `rebalance_action_fraction`, and
  `mean_rebalances_after_deployment_per_path` describe activity. Initial
  deployment is a rebalance action but is excluded from the latter count.
- `mean_selected_range_width_ticks` averages requested rebalance widths, while
  `mean_active_range_width_ticks` averages live position widths over deployed
  trajectory-steps. Both are zero when there are no applicable observations.
- `mean_gas_spend_per_path` reports modeled gas actually deducted, capped by
  remaining wealth.
- `mean_fee_income_token1_per_path` values each step's newly accrued token0
  fees at that step's external price and adds token1 fees.
- `mean_pnl_per_path` and `mean_inventory_penalty_per_path` provide an exact
  reward decomposition: mean objective equals mean PnL minus mean inventory
  penalty.
- `mean_in_range_fraction_among_deployed_paths` averages each deployed path's
  fraction of deployed steps spent in range; never-deployed paths are excluded.
- `agent_decision_stride`, `agent_decisions_per_episode`,
  `agent_hold_decision_fraction`, `agent_rebalance_decision_fraction`,
  `mean_agent_rebalances_after_deployment_per_path`, and
  `mean_agent_selected_range_width_ticks` report PPO decision-level activity
  separately from the simulator-step diagnostics. These are populated for PPO
  rows; baseline rows leave the decision fractions blank.

## Monitoring

Useful Slurm commands:

```bash
squeue -u $USER
squeue -j <job_id>
scancel <job_id>
```

Run artifacts are written under `experiments/results/domain_randomized_ppo/...` inside the project directory.
Slurm output/error files are written under `logs/`; the setup script creates that directory.
