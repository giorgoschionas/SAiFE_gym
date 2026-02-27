# PPO Calibration Guide for SAiFE_gym

Practical rules for tuning PPO when training an RL LP agent in
`SAiFE_gym`, with direct comparison to the `mbt_gym` setup.

---

## 1. Observation normalisation — the most important step

SAiFE's 9 observation features span ~8 orders of magnitude at runtime:

| Feature | Typical value |
|---|---|
| `sqrt_price` | ~10 |
| `current_tick` | ~46,000 |
| `lp_liquidity` | 0 → ~2×10⁸ |
| `lp_tick_lower / upper` | ~46,000 |
| `lp_collected_fees_0/1` | 0 → large |
| `midprice` | ~100 |
| `time` | 0 → 1 |

Without normalisation the first weight layer of the policy network sees
numbers differing by factors of millions.  A weight calibrated for
`time ∈ [0, 1]` will be 46,000× too large when processing `current_tick`,
and the gradient signal from `time` will be negligible compared to
`lp_liquidity`.

**Solution — use `VecNormalize`:**

```python
from stable_baselines3.common.vec_env import VecNormalize

vec = VecNormalize(
    VecMonitor(StableBaselinesAMMEnvironment(env)),
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,   # cap at ±10σ to handle lp_liquidity jump on first step
)
```

`VecNormalize` maintains a running mean and std for each feature and
updates them online during training.  `clip_obs=10.0` prevents the
sudden jump of `lp_liquidity` from 0 → 2×10⁸ on the first step from
dominating the running stats.

**mbt comparison:**
mbt uses `ReduceStateSizeWrapper` to keep only `[inventory, time]`
(dim=2) and maps them to `[-1, 1]` using known analytical bounds.
SAiFE's features have no tight analytical bounds, so online normalisation
via `VecNormalize` is the right substitute.

---

## 2. Feature representation — prefer relative over absolute

Several SAiFE features are absolute values that happen to be large and
correlated.  Expressing them as relative offsets gives the network
smaller, more informative numbers:

| Raw feature | Problem | Better |
|---|---|---|
| `current_tick` | ~46,000, already encoded in `sqrt_price` | Drop |
| `lp_tick_lower` | absolute ~46,000, drifts with price | `lp_tick_lower − current_tick` |
| `lp_tick_upper` | same | `lp_tick_upper − current_tick` |

The relative offsets tell the agent whether its range is centred, above,
or below the current price — which is what matters for positioning.
The absolute tick adds nothing beyond what `sqrt_price` already provides.

Custom `obs_keys` can be passed to `StableBaselinesAMMEnvironment` to
swap these features out once the corresponding derived keys are available.

---

## 3. `n_steps` — collect complete episodes per rollout

```python
n_steps = env.n_steps   # one full episode per trajectory per rollout
```

With `gamma=1.0` (no discounting) the agent needs to see the full episode
return to correctly estimate the advantage.  Setting `n_steps` equal to
the episode length ensures every rollout buffer entry is paired with a
complete return.  Shorter rollouts with `gamma<1` would introduce
unnecessary bias via bootstrapped value estimates.

---

## 4. `batch_size` — aim for 8–16 mini-batches per epoch

The rollout buffer contains `n_steps × num_trajectories` transitions.
Split it into enough mini-batches that each gradient update sees a
representative cross-section of the data:

```python
rollout_size = n_steps * num_trajectories
batch_size   = max(64, rollout_size // 16)   # → 16 mini-batches per epoch
```

The original SAiFE setup used `rollout_size // 4` (4 mini-batches), which
means each PPO gradient step uses 25% of the buffer — large batches that
are computationally wasteful and offer fewer opportunities to improve the
policy per rollout.

---

## 5. `total_timesteps` — scales with observation dimension

The number of gradient updates needed to learn a well-performing policy
scales roughly as `O(obs_dim × log(obs_dim))` in empirical practice.

| Setting | obs_dim | Recommended `total_timesteps` |
|---|---|---|
| mbt (2 normalised features) | 2 | 500k – 2M |
| SAiFE (9 features, with `VecNormalize`) | 9 | 2M – 5M |
| SAiFE (9 features, no normalisation) | 9 | 5M – 20M |

The SAiFE experiments in `notebooks/rl_agent_evaluation.py` used 600k
steps (60 PPO rollouts of 50×200 transitions).  Results were already
positive at α₃=0 (+265k RL vs +63k Uniform) but convergence was
incomplete.  Re-running with 2M–5M steps and `VecNormalize` should
improve stability and widen the RL advantage further.

---

## 6. `n_epochs` — reuse each rollout 10 times (reduce if KL diverges)

```python
n_epochs = 10
```

Standard for PPO.  Monitor `approx_kl` in the training logs.  If it
consistently exceeds 0.02 per update, reduce `n_epochs` to 4–6 or
lower `learning_rate`.  The `clip_range=0.2` default acts as a guard
but does not eliminate the risk of policy collapse with too many epochs
on a stale rollout.

---

## 7. `learning_rate`

| Condition | Recommended `learning_rate` |
|---|---|
| With `VecNormalize` (normalised obs) | `3e-4` (SB3 default) |
| Without normalisation (raw obs) | `1e-4` |

Without normalisation, large-magnitude features produce large activations
which amplify gradient updates and cause overshooting.  A smaller
learning rate compensates, but fixing the root cause (normalisation) is
strongly preferred.

---

## 8. Network architecture

Both mbt and SAiFE use `[256, 256]` for both actor and critic.  For 9
input features this is more than sufficient in capacity.  Increasing
network width does not help when the bottleneck is training stability
(feature scale) rather than model expressivity.

```python
policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
```

---

## Quick-reference checklist

- [ ] Wrap with `VecNormalize(norm_obs=True, clip_obs=10.0)`
- [ ] Set `n_steps = env.n_steps` (complete episodes)
- [ ] Set `batch_size = max(64, rollout_size // 16)`
- [ ] Set `gamma = 1.0` (undiscounted PnL)
- [ ] Set `gae_lambda = 0.95`
- [ ] Set `n_epochs = 10`, reduce to 4–6 if `approx_kl > 0.02`
- [ ] Use `total_timesteps ≥ 2M` for 9-feature SAiFE obs
- [ ] Use `learning_rate = 3e-4` with normalisation, `1e-4` without
- [ ] Monitor `value_loss` — should decrease steadily; if it grows, reduce `learning_rate`
- [ ] Monitor `clip_fraction` — should stay below 0.1; if higher, reduce `learning_rate` or `n_epochs`
