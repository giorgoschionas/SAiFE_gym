# Barkla2 example Slurm scripts for robust LP PPO

These example scripts are written for the University of Liverpool Barkla2 Slurm setup described
in `docs/Barkla2_User_Guide (2).pdf`. They are templates for reproducing the robust LP PPO
experiment on that cluster; adjust paths, partitions, time limits, and resources for other HPC
systems.

Recommended layout on Barkla2:

```bash
/mnt/scratch/users/$USER/rl_experiments/SAiFE_gym        # project checkout / working directory
/mnt/fastscratch/users/$USER/venvs/saife_gym_py312       # Python virtual environment
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

Run artifacts are written under `experiments/results/robust_rl/...` inside the project directory.
Slurm output/error files are written under `logs/`; the setup script creates that directory.
