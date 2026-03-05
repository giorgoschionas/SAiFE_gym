# SAiFE Logic Flow & Architecture

## Three-Layer Architecture

SAiFE follows a standard RL environment pattern with three nested layers:

```
┌─────────────────────────────────────────────────┐
│  StableBaselinesAMMEnvironment  (SB3 VecEnv)    │  ← Training interface
│  ┌───────────────────────────────────────────┐   │
│  │  AMMEnvironment  (gymnasium.Env)          │   │  ← RL orchestrator
│  │  ┌─────────────────────────────────────┐  │   │
│  │  │  ModelDynamics  (domain logic)      │  │   │  ← The "physics engine"
│  │  │  ├── StochasticProcesses            │  │   │
│  │  │  └── Uniswap V3 protocol math       │  │   │
│  │  └─────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

1. **ModelDynamics** — The domain-specific "rules of the world". Knows nothing about RL. Answers: *"given these arrivals and this action, what happens next?"*
2. **AMMEnvironment** — The standard Gymnasium `step()`/`reset()` RL loop. Handles the *when* (timing, episode boundaries), while ModelDynamics handles the *what* (protocol mechanics).
3. **StableBaselinesAMMEnvironment** — Thin wrapper that adapts the environment to SB3's `VecEnv` interface for training.

---

## Layer 1: ModelDynamics — The Physics Engine

**File:** `SAiFE_gym/gym/ModelDynamics.py`

### Class Hierarchy

```
ModelDynamics (Abstract)
├── midprice_model: MidpriceModel
├── arrival_model: ArrivalModel
├── state: dict
└── Methods:
    ├── update_state(arrivals, action)
    ├── get_arrivals() → (N, 2)
    └── get_action_space()

    └── UniswapV3ModelDynamics (Concrete)
            ├── fee_tier, tau, num_ticks, exponential_value
            ├── initial_wealth, gas_cost, swap_fee_rate
            ├── tick_lower_global: int (anchors liquidity array)
            ├── max_arrivals_per_step: int (safety cap)
            └── Methods:
                ├── get_action_space() → Box(low=[-tau, -tau+1], high=[tau-1, tau])
                ├── validate_action(action) → clipped & rounded action
                ├── update_state(arrivals, action) → rebalance + swaps
                ├── _rebalance(action)
                ├── _get_current_tick_liquidity() → (idx, L_current)
                ├── _compute_local_xi() → (xi_sell, xi_buy)
                ├── _process_sell_single(active, xi_sell)
                ├── _process_buy_single(active, xi_buy)
                ├── _collect_lp_fees() → (fee0, fee1)
                ├── _compute_token0_fraction_vec()
                ├── _process_arrivals(counts, xi_index, process_fn)
                ├── _process_arrivals_alternating(sell_counts, buy_counts)
                └── get_arrivals() → (N, 2)
```

### What It Tracks

- Pool sqrt_price, current tick, liquidity array (per tick)
- LP position (liquidity, lower/upper tick bounds)
- Accumulated fees (token0 and token1, per tick)

### How Trades Work

Each trade moves price by exactly one tick. Trade size (xi) is computed from the current tick's liquidity only (`_compute_local_xi()`). Arrival counts are Poisson-distributed integers. The engine loops over arrival counts, processing one trade at a time.

### Rebalancing Mechanism (`_rebalance`)

Three-phase process at the start of each `update_state()`:

1. **Compute wealth** from existing position (position value + collected fees)
2. **Apply rebalancing costs** (gas cost + swap fee proportional to portfolio rebalance)
3. **Deploy new position** at the action-specified tick range

---

## Layer 2: AMMEnvironment — The RL Orchestrator

**File:** `SAiFE_gym/gym/AMMEnvironment.py`

### Class Structure

```
gymnasium.Env (Abstract)
    └── AMMEnvironment
            ├── state: dict
            ├── model_dynamics: ModelDynamics
            ├── reward_function: RewardFunction
            ├── observation_space: gymnasium.spaces.Dict
            ├── action_space: Box(2,)
            └── Methods:
                ├── reset() → (obs_dict, info)
                ├── step(action) → (obs_dict, rewards, terminated, truncated, info)
                ├── _update_state(action)
                ├── _update_market_state(arrivals, action)
                └── _get_terminated()
```

### Step Execution Flow

```
AMMEnvironment.step(action)
│
├─→ [1] Save current_state copy
│
├─→ [2] _update_state(action)
│       │
│       ├─→ [2.1] model_dynamics.get_arrivals()
│       │         └─→ arrival_model.get_arrivals()
│       │             Poisson sample from intensity state
│       │             Returns: (N, 2) = [sell_counts, buy_counts]
│       │
│       ├─→ [2.2] model_dynamics.update_state(arrivals, action)
│       │         │
│       │         ├─→ _rebalance(action)
│       │         │   ├─→ Phase 1: Compute wealth (existing position)
│       │         │   ├─→ Phase 2: Apply rebalancing costs
│       │         │   └─→ Phase 3: Deploy new position
│       │         │
│       │         └─→ _process_arrivals_alternating(sell_counts, buy_counts)
│       │             For each round i:
│       │             ├─→ If i < sell_count: _process_sell_single()
│       │             │   (Update sqrt_price, current_tick, fees_0)
│       │             └─→ If i < buy_count: _process_buy_single()
│       │                 (Update sqrt_price, current_tick, fees_1)
│       │
│       ├─→ [2.3] _update_market_state(arrivals, action)
│       │         ├─→ midprice_model.update()  → BM/GBM step
│       │         └─→ arrival_model.update()   → recalculate intensity
│       │
│       └─→ [2.4] Advance time: state[TIME_KEY] += step_size
│
├─→ [3] _get_terminated() → check if time >= terminal_time
│
├─→ [4] reward_function.calculate(current_state, action, next_state, terminated)
│
└─→ [5] Return (next_state, rewards, terminated, truncated, info)
```

### State Representation (Dict-based, vectorized)

All state arrays have shape `(num_trajectories, ...)`.

| Key | Shape | Description |
|-----|-------|-------------|
| `'sqrt_price'` | `(N,)` | Current pool sqrt(P) (NOT P) |
| `'current_tick'` | `(N,)` | Current absolute tick |
| `'liquidity_array'` | `(N, num_ticks)` | Pool liquidity per tick |
| `'fees_0'` | `(N, num_ticks)` | Pool fees in token0 per tick |
| `'fees_1'` | `(N, num_ticks)` | Pool fees in token1 per tick |
| `'lp_liquidity'` | `(N,)` | LP's position liquidity |
| `'lp_tick_lower'` | `(N,)` | LP's position lower bound (absolute) |
| `'lp_tick_upper'` | `(N,)` | LP's position upper bound (absolute) |
| `'lp_collected_fees_0'` | `(N,)` | Cumulative fees in token0 |
| `'lp_collected_fees_1'` | `(N,)` | Cumulative fees in token1 |
| `'midprice'` | `(N,)` | External market midprice |
| `'time'` | `(N,)` | Current simulation time |

**Derived (computed, not stored):** `mispricing`, `lp_lower_offset`, `lp_upper_offset`

### Action Space

Format: `[lower_offset, upper_offset]` — tick offsets relative to current tick.

```python
Box(low=[-tau, -tau+1], high=[tau-1, tau], shape=(2,), dtype=np.float32)
```

Action `[-3, 4]` with `tau=5` means: deploy LP liquidity from `current_tick - 3` to `current_tick + 4` (8 ticks, asymmetric around current price).

---

## Layer 3: StableBaselinesAMMEnvironment — The Training Adapter

**File:** `SAiFE_gym/gym/StableBaselinesAMMEnvironment.py`

### Class Structure

```
stable_baselines3.common.vec_env.VecEnv (Abstract)
    └── StableBaselinesAMMEnvironment
            ├── env: AMMEnvironment
            ├── obs_keys: List[str]
            ├── observation_space: Box(obs_dim,)
            ├── action_space: Box(2,)
            └── Methods:
                ├── reset() → flat_obs
                ├── step_async(actions)
                ├── step_wait() → (flat_obs, rewards, dones, infos)
                ├── _flatten_obs(state_dict) → (N, obs_dim) array
                └── _compute_derived(state_dict) → {mispricing, offsets}
```

### What It Does

1. **Flattens observations** — SB3 wants `(num_envs, obs_dim)` float arrays, not dicts. The wrapper extracts selected keys via `obs_keys` and concatenates them.
2. **Computes derived features** — `mispricing = midprice - sqrt_price²`, tick offsets for the LP position.
3. **Handles auto-reset** — When all trajectories finish, resets and stores terminal obs in `info["terminal_observation"]`.
4. **Exposes `num_trajectories` as `num_envs`** — SB3 thinks it's running N separate envs, but it's one vectorized env.

### Default Observation Keys (8 features)

```python
['mispricing', 'lp_lower_offset', 'lp_upper_offset', 'lp_liquidity',
 'lp_collected_fees_0', 'lp_collected_fees_1', 'midprice', 'time']
```

### Training Flow

```
SB3 PPO/SAC
    │
    ▼
StableBaselinesAMMEnvironment.step_async(action)  → stores action
StableBaselinesAMMEnvironment.step_wait()          → calls env.step(), handles resets
    │
    ▼
AMMEnvironment.step(action)  → the standard RL loop
    │
    ▼
UniswapV3ModelDynamics.update_state(arrivals, action)  → protocol mechanics
```

---

## Supporting Components

### Stochastic Processes (`stochastic_processes/`)

```
StochasticProcessModel (Abstract)
├── MidpriceModel
│   ├── BrownianMotionMidpriceModel (drift + volatility)
│   └── GeometricBrownianMotionMidpriceModel
└── ArrivalModel
    ├── PoissonArrivalModel (constant intensity)
    └── PoissonLinearArrivalModel (state-dependent intensity)
```

All processes support `(num_trajectories, ...)` shapes and implement `update()`, `reset()`, `seed()`.

### Reward Functions (`rewards/RewardFunctions.py`)

```
RewardFunction (Abstract)
├── PnL                        — Mark-to-market portfolio value change
├── ExponentialUtility          — Terminal CARA utility: -exp(-a * W_T)
└── RunningInventoryPenalty     — PnL minus running + terminal inventory cost
```

### Agents (`agents/`)

```
Agent (Abstract)
├── RandomAgent                 — Uniform samples from action space
├── UniformAllocationAgent      — Full range [-tau, +tau]
└── SbAgent                     — Wraps trained SB3 model
```

---

## Key Parameters & Defaults

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `tau` | 5 | Ticks on each side of current price |
| `num_ticks` | 1000 | Total ticks in liquidity array |
| `fee_tier` | 0.003 | Pool fee (0.3%) |
| `exponential_value` | 1.0001 | Tick spacing base |
| `initial_wealth` | 1e6 | LP's starting capital |
| `gas_cost` | 0.0 | Fixed rebalance cost (token1) |
| `swap_fee_rate` | 0.0 | Proportional swap cost |
| `max_arrivals_per_step` | 50 | Safety cap on Poisson samples |
| `terminal_time` | 1.0 | Trading horizon |
| `n_steps` | 200 | Timesteps per episode |

---

## Vectorization Design

All operations are fully vectorized across `num_trajectories`:

- State arrays: `(num_trajectories, ...)`
- Trade processing loops over tick counts, not trajectories
- Per-trajectory masking with boolean arrays (e.g., `active_sell = counts > i`)
- No Python loops over trajectories — only NumPy array ops (`np.where`, `np.clip`, etc.)
