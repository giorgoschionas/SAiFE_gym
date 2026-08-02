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

Override parameters at submission time when needed:

```bash
sbatch --export=ALL,TOTAL_TIMESTEPS=2000000,NUM_TRAJECTORIES=128,N_EVAL_EPISODES=20 \
  hpc/sbatch_train_robust_lp_agent_cpu.sh
```

Domain-randomized training uses one sampled domain per trajectory by default. Set
`TRAIN_DOMAINS_PER_RESET=1` to restore a single shared domain, or choose any
value from `1` through `NUM_TRAJECTORIES` for balanced grouped assignments.

Evaluation is reported separately on an in-distribution interpolation grid and
an out-of-distribution stress grid. Override their Cartesian axes independently
with `EVAL_IN_DISTRIBUTION_SIGMA_VALUES`,
`EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES`,
`EVAL_IN_DISTRIBUTION_GAS_COST_VALUES`, `EVAL_STRESS_SIGMA_VALUES`,
`EVAL_STRESS_ARRIVAL_RATE_VALUES`, and `EVAL_STRESS_GAS_COST_VALUES`. Each
variable is a space-separated list, for example:

```bash
sbatch --export=ALL,EVAL_IN_DISTRIBUTION_GAS_COST_VALUES="2.0 3.5 5.0" \
  hpc/sbatch_train_robust_lp_agent_cpu.sh
```

Every in-distribution value must lie within its corresponding training range,
and every stress-grid tuple must have at least one value outside training
support.

## Seed sweep

Submit independent replicate runs:

```bash
cd /mnt/scratch/users/$USER/rl_experiments/SAiFE_gym
sbatch hpc/sbatch_robust_lp_seed_sweep_cpu.sh
```

Edit the `SEEDS=(...)` array inside the script if you want more or fewer replicates.

## Monitoring

Useful Slurm commands:

```bash
squeue -u $USER
squeue -j <job_id>
scancel <job_id>
```

Run artifacts are written under `experiments/results/domain_randomized_ppo/...` inside the project directory.
Slurm output/error files are written under `logs/`; the setup script creates that directory.
