# VecNormalize Evaluation Bug

## Summary

The RL agent (PPO) was being evaluated with **raw unnormalized observations**, while it was trained on **VecNormalize-processed observations**. This caused the policy network to saturate and output degenerate (fixed) actions, making RL results unreliable.

## Root Cause

### Training pipeline (correct)
```
obs → StableBaselinesAMMEnvironment._flatten_obs
    → VecMonitor
    → VecNormalize (running mean/std → obs ~ N(0,1))
    → PPO policy network
```

### Evaluation pipeline (broken)
```
obs → sb3_env._flatten_obs (flatten only, NO normalization)
    → SbAgent.get_action → model.predict(RAW obs)
```

The `compare_rl_vs_uniform()` function in `experiments/helpers.py` used `sb3_env._flatten_obs` as the observation transform for the RL agent. This flattens the dict observation to a numpy array but does **not** apply VecNormalize's running statistics (mean/std).

## Why It Matters

The observation features span wildly different scales:

| Feature | Raw scale | After VecNormalize |
|---------|-----------|-------------------|
| MISPRICING | ~0 to ±5 | ~±1 |
| LP_LOWER_OFFSET | ~20 | ~±1 |
| LP_UPPER_OFFSET | ~20 | ~±1 |
| LP_LIQUIDITY | ~2,000,000,000 | ~±1 |
| LP_COLLECTED_FEES0 | ~0-200 | ~±1 |
| LP_COLLECTED_FEES1 | ~0-25,000 | ~±1 |
| ASSET_PRICE | ~100 | ~±1 |
| TIME | 0-1 | ~±1 |

When LP_LIQUIDITY=2e9 hits the first layer of a `[256,256]` network, it **saturates every neuron**. The tanh activations all go to ±1, and the network outputs a **fixed degenerate action** regardless of the actual market state.

## Evidence

### 1. Identical RL results across different tau values

In the comprehensive sweep (Sweep 4 heatmap), independently trained models at different tau values produced **exactly identical** RL ΔW:

| alpha3 | tau=20 | tau=50 | tau=200 | tau=400 |
|--------|--------|--------|---------|---------|
| 2000 | +65,147 | +65,147 | +65,147 | +65,147 |
| 5000 | +154,814 | +154,814 | +154,814 | +154,814 |

Each was independently trained (different training times: 824s, 796s, 1034s, 881s for alpha3=2000). Uniform ΔW varied correctly across tau. The RL agent's identical results meant it was outputting the same degenerate action regardless of tau.

### 2. Direct action comparison

With a trained model, comparing actions on the same observation:

```
Unnormalized obs: [0.113, -8.0, 9.0, 2034002944, 90.9, 10552.5, 99.7, 0.25]
Normalized obs:   [0.174, -0.460, 0.426, 0.444, -0.592, -0.568, -0.219, -0.780]

Action WITH normalization:    [[0.097, 0.030], [0.335, 0.032]]  ← different per trajectory
Action WITHOUT normalization: [[0.126, 0.019], [0.126, 0.019]]  ← identical (saturated)
```

Without normalization, the network can't distinguish between trajectories — it outputs the same fixed action for all.

## The Fix

### Changes to `experiments/helpers.py`

Added optional `vec_normalize` parameter to `compare_rl_vs_uniform()`:

```python
def compare_rl_vs_uniform(
    model, env, sb3_env, n_eval_episodes=10,
    initial_wealth=INITIAL_WEALTH,
    vec_normalize=None,           # NEW: pass VecNormalize from training
):
    flatten = sb3_env._flatten_obs

    if vec_normalize is not None:
        vec_normalize.training = False  # don't update stats during eval
        rl_obs_transform = lambda obs: vec_normalize.normalize_obs(flatten(obs))
    else:
        rl_obs_transform = flatten
    ...
```

### Changes to experiment scripts

`train_ppo()` now returns the VecNormalize wrapper alongside the model:

```python
def train_ppo(tau, alpha3, vol):
    env = get_amm_env(...)
    vec_env = wrap_env(env)  # includes VecNormalize
    model = PPO(env=vec_env, ...)
    model.learn(...)
    return model, model.get_env()  # return VecNormalize wrapper
```

`evaluate()` passes it through to `compare_rl_vs_uniform`:

```python
def evaluate(model, vec_normalize, tau, alpha3, vol):
    ...
    return compare_rl_vs_uniform(
        model, eval_env, sb3_env,
        vec_normalize=vec_normalize,  # apply training normalization stats
    )
```

## How mbt_gym Avoids This

mbt_gym does **not** use SB3's `VecNormalize` wrapper. Instead, it uses explicit normalization calls:

```python
# mbt's evaluation (in create_inventory_plot):
if model_uses_normalisation:
    reduced_state = normalised_env.normalise_observation(reduced_state)
action = ppo_agent.get_action(reduced_state)
if model_uses_normalisation:
    action = normalised_env.normalise_action(action, inverse=True)
```

This gives full control and ensures consistency between training and evaluation, because normalization is an explicit method call rather than buried inside a VecEnv wrapper.

## Affected Results

All RL results from `notebooks/rl_vs_uniform_comprehensive.py` run on 2025-03-05 are invalid. Uniform agent results are unaffected (Uniform doesn't use the model).

The first run's results are saved in `notebooks/results/comprehensive_results.json` for reference but should not be used for analysis.
