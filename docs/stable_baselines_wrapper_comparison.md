# StableBaselines Wrapper Comparison: mbt_gym vs SAiFE_gym

Comparison of `mbt_gym/gym/StableBaselinesTradingEnvironment.py` and
`SAiFE_gym/gym/StableBaselinesAMMEnvironment.py`.

Both wrap a vectorised Gym environment to expose a `stable_baselines3` `VecEnv`
interface, but the AMM environment's richer state representation forces several
additional responsibilities onto the SAiFE wrapper.

---

## Shared Pattern

Both wrappers:

- Subclass `stable_baselines3.common.vec_env.VecEnv`
- Store the wrapped environment as `self.env`
- Implement `reset`, `step_async`, `step_wait`, `close`, `env_is_wrapped`
- Auto-reset when all trajectories finish simultaneously (`dones.all()`)
- Optionally record `terminal_observation` in `infos` before the auto-reset
- Expose `num_trajectories` and `n_steps` as properties

---

## Key Differences

### 1. Observation flattening

**mbt** — `TradingEnvironment.reset()` / `step()` already return a flat
`np.ndarray` of shape `(num_trajectories, obs_dim)`, so the wrapper passes it
straight through:

```python
def reset(self):
    return self.env.reset()
```

**SAiFE** — `AMMEnvironment` returns a `dict` of named arrays (sqrt price,
current tick, LP position bounds, collected fees, market price, time, …).
`MlpPolicy` needs a plain float32 matrix, so the wrapper must project the dict
down to a flat array:

```python
def _flatten_obs(self, state_dict):
    cols = [state_dict[k].reshape(n, 1) for k in self.obs_keys]
    return np.concatenate(cols, axis=1).astype(np.float32)
```

This `_flatten_obs` call is inserted into both `reset` and `step_wait`.

---

### 2. Observation key selection

**mbt** — no concept of key selection; the flat array is the full observation.

**SAiFE** — introduces `DEFAULT_OBS_KEYS`, a curated list of 9 scalar state
keys that are meaningful to the policy:

```python
DEFAULT_OBS_KEYS = [
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY,
    LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
    ASSET_PRICE_KEY, TIME_KEY,
]
```

Callers can override this via the `obs_keys` constructor argument. The wrapper
also guards against accidentally including **array-valued keys**
(`liquidity_array`, `fees_0`, `fees_1`) whose shape `(num_trajectories, num_ticks)`
would break the flat layout:

```python
for k in self.obs_keys:
    if k in _ARRAY_KEYS:
        raise ValueError(f"obs_keys may only contain scalar keys; '{k}' is an array key")
```

---

### 3. Gymnasium vs legacy gym spaces

**mbt** — passes the environment's existing spaces directly to `super().__init__()`.
These are legacy `gym` spaces, which was compatible with the SB3 version mbt
targets.

**SAiFE** — `AMMEnvironment` uses legacy `gym`, but SB3 2.7+ expects
`gymnasium` spaces. The wrapper constructs new `gymnasium.spaces.Box` objects
for both observation and action spaces before calling `super().__init__()`:

```python
flat_obs_space = gymnasium.spaces.Box(
    low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
)
gymnasium_act_space = gymnasium.spaces.Box(
    low=old_act.low.astype(np.float32),
    high=old_act.high.astype(np.float32),
    shape=old_act.shape, dtype=np.float32,
)
super().__init__(num_trajectories, flat_obs_space, gymnasium_act_space)
```

---

### 4. `get_attr` stub

**mbt** — the method body is `pass` (returns `None` implicitly). This worked
with older SB3 versions that did not call `get_attr` during construction.

**SAiFE** — SB3 2.7's `VecEnv.__init__` calls `self.get_attr('render_mode')`
immediately and expects a list back. The stub therefore returns a proper list
and handles the `render_mode` case explicitly:

```python
def get_attr(self, attr_name, indices=None):
    if attr_name == "render_mode":
        return [None] * self.env.num_trajectories
    return [getattr(self.env, attr_name)] * self.env.num_trajectories
```

`set_attr` and `env_method` are also implemented fully rather than left as
`pass`, so attribute writes and method delegation actually work.

---

### 5. Done check style

**mbt**: `if dones.min()` — works for boolean arrays (`min` of bools is logical AND).

**SAiFE**: `if dones.all()` — semantically identical but more explicit.

---

## Summary Table

| Aspect | mbt `StableBaselinesTradingEnvironment` | SAiFE `StableBaselinesAMMEnvironment` |
|---|---|---|
| Obs format in | flat `ndarray` | `dict` of named arrays |
| Obs format out | passthrough | projected flat float32 via `_flatten_obs` |
| Obs key selection | N/A | `DEFAULT_OBS_KEYS`, configurable |
| Array key guard | N/A | raises `ValueError` |
| Spaces type | legacy `gym` (passthrough) | new `gymnasium.spaces.Box` |
| `get_attr` | `pass` (returns `None`) | proper list, handles `render_mode` |
| `set_attr` / `env_method` | `pass` | fully implemented |
| Done check | `dones.min()` | `dones.all()` |
| SB3 version targeted | < 2.0 | 2.7+ |

---

## Why the extra complexity in SAiFE?

The root cause is the observation type. mbt's `TradingEnvironment` was designed
with SB3 in mind from the start and produces a flat array directly. SAiFE's
`AMMEnvironment` was designed for readability and debuggability, returning a
named dict so that every state component is self-documenting. The SAiFE wrapper
is the bridge that absorbs this impedance mismatch without changing either the
environment or the SB3 algorithm.
