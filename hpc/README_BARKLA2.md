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
`TOTAL_TIMESTEPS=6000000`, `DECISION_STRIDE=20`, `INITIAL_WEALTH=1000.0`,
and `TAU=250`.
Here, `TAU` controls the policy's maximum center offset and half-width in ticks;
it does not force every position to have width 200. These values can still be
overridden through exported environment variables.

`TOTAL_TIMESTEPS` is a simulator-equivalent budget. With the default
`DECISION_STRIDE=20`, PPO acts once every 20 simulator steps, so the default
run trains on 300,000 PPO decision timesteps and each 1,000-step episode exposes
at most 50 agent decisions. The simulator uses forced hold actions between
agent decisions.

With the default `TAU=250` and `tick_stride=5`, the PPO action space is
`MultiDiscrete([101, 50, 2])`.

Override parameters at submission time when needed:

```bash
sbatch --export=ALL,TOTAL_TIMESTEPS=6000000,NUM_TRAJECTORIES=128,N_EVAL_EPISODES=20 \
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

Evaluation is reported separately on an in-distribution interpolation grid and
an out-of-distribution stress grid. Override their Cartesian axes independently
with `EVAL_IN_DISTRIBUTION_SIGMA_VALUES`,
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
and every stress-grid tuple must have at least one value outside training
support.

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
