# Barkla2 example Slurm scripts for domain-randomized LP PPO

These example scripts are written for the University of Liverpool Barkla2 Slurm setup described
in `docs/Barkla2_User_Guide (2).pdf`. They are templates for reproducing the domain-randomized PPO
experiment on that cluster; adjust paths, partitions, time limits, and resources for other HPC
systems.

Recommended layout on Barkla2:

```bash
/mnt/scratch/users/$USER/rl_experiments/SAiFE_gym        # project checkout / working directory
/mnt/fastscratch/users/$USER/venvs/rl_venv       # Python virtual environment
```

The guide recommends using `scratch` as a work directory and `fastscratch` for Python
environments. Avoid installing Python packages in `/users/$USER`.

## One-time setup

Run this from `barklaviz1` or `barklaviz2`, not the login node, because package installation can
be resource intensive:

```bash
cd /mnt/scratch/users/$USER/rl_experiments/SAiFE_gym
bash hpc/setup_barkla2_env.sh
```

## Smoke test

Submit a tiny end-to-end job first:

```bash
cd /mnt/scratch/users/$USER/rl_experiments/SAiFE_gym
sbatch hpc/sbatch_robust_lp_smoke.sh
```

## Full training

Submit the default production run:

```bash
cd /mnt/scratch/users/$USER/rl_experiments/SAiFE_gym
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
mkdir -p logs && sbatch --export=ALL,TOTAL_TIMESTEPS=10000000,N_STEPS=1000,DECISION_STRIDE=100,NUM_TRAJECTORIES=100,TRAIN_DOMAINS_PER_RESET=100,N_EVAL_EPISODES=10,EVALUATION_SEED=100042,TAU=50,ALPHA3=4000,INITIAL_WEALTH=1000,INVENTORY_PHI=0.4,NOMINAL_GAS_COST=2,PERIODIC_REBALANCE_EVERY=100,PERIODIC_WIDTH=50,NOMINAL_SIGMA=0.030,NOMINAL_ARRIVAL_RATE=450,TRAIN_SIGMA_MIN=0.015,TRAIN_SIGMA_MAX=0.045,TRAIN_ARRIVAL_RATE_MIN=300,TRAIN_ARRIVAL_RATE_MAX=600,EVAL_IN_DISTRIBUTION_SIGMA_VALUES="0.015 0.030 0.045",EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES="300 450 600",EVAL_STRESS_SIGMA_VALUES=0.050,EVAL_STRESS_ARRIVAL_RATE_VALUES="250 800",OUTPUT_DIR=experiments/results/barkla2_seed_sweep/moderate_arrivals_10m hpc/sbatch_robust_lp_seed_sweep_cpu.sh
```

Each policy retains a budget of 10 million simulator-equivalent steps: 100,000
PPO decision timesteps with ten decisions per 1,000-step episode. The periodic
baseline rebalances every 100 simulator steps. Initial wealth, gas, inventory
penalty, action ranges, and alpha3 match the preceding experiment. Conclusions
remain specific to this resolution; the earlier high-arrival diagnostics found
material changes in reward when the simulator timestep was refined.

The launcher saves runs under
`experiments/results/barkla2_seed_sweep/moderate_arrivals_10m/seed_<seed>/run_<timestamp>/`.
Use a different `OUTPUT_DIR` for a repeat sweep so aggregation does not encounter
duplicate training seeds. After all ten array tasks finish successfully, run:

```bash
sbatch --export=ALL,INPUT_DIR=experiments/results/barkla2_seed_sweep/moderate_arrivals_10m,OUTPUT_DIR=experiments/results/barkla2_seed_sweep/moderate_arrivals_10m/aggregate hpc/sbatch_aggregate_domain_randomized_seed_sweep.sh
```

Check that each saved `config.json` contains the selected bounds, nominal
parameters, and `evaluation_grid_version: 2`. Expect ten seed runs, each with
80 rows in `evaluation_grid.csv` and 1,000 evaluation paths per row. Aggregation
validates full coverage and computes paired randomized-minus-nominal reward
differences across training seeds. Compare both PPO agents with periodic
rebalancing and cash, and report reward and PnL separately.

## Seed sweep

The default array trains ten independent PPO replicates (seeds 43 through 52)
while evaluating every learned network on the same paths using
`EVALUATION_SEED=100042`. Submit the sweep and its aggregation job with a Slurm
dependency:

```bash
cd /mnt/scratch/users/$USER/rl_experiments/SAiFE_gym
SWEEP_JOB_ID=$(sbatch --parsable hpc/sbatch_robust_lp_seed_sweep_cpu.sh)
sbatch --dependency=afterok:$SWEEP_JOB_ID \
  hpc/sbatch_aggregate_domain_randomized_seed_sweep.sh
```

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
