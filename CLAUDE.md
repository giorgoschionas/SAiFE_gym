## Project Overview

SAiFE_gym is a Reinforcement Learning environment for simulating Automated Market Maker (AMM) with Concentrated Liquidity trading in DeFi protocols, like Uniswap v3. It is Gymnasium-compatible for training RL agents to act as liquidity providers.

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# The project uses Python 3.12 with a virtual environment (venv/)
source venv/bin/activate   
```

## Architecture Philosophy
- **Readability**: Make code easy to understand
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions

### Vectorized Environment Design

**IMPORTANT** SAiFE_gym is being built with **full vectorization** for high-performance parallel simulation of multiple trajectories.

**Vectorization Strategy:**
- All core components operate on batched inputs: `(num_trajectories, ...)`
- State arrays, actions, rewards vectorized end-to-end
- Swap execution handles multiple price paths in parallel

**Array-Based Design:**
- Liquidity represented as NumPy arrays indexed by tick: `liquidity_array[tick_idx]`
- Price movements tracked per trajectory: `sqrt_price_current.shape = (num_trajectories,)`
- Active trajectory masking eliminates branching in hot loops

### Liquidity Array Indexing

**`tick_lower` - The Array Anchor:**

`tick_lower` is a fixed integer that anchors the `liquidity_array` to absolute Uniswap V3 tick space. It represents the lowest absolute tick tracked by the array.

**Indexing Convention:**
```
liquidity_array[i] = liquidity for absolute tick (tick_lower + i)
                   = liquidity in price range [1.0001^(tick_lower+i), 1.0001^(tick_lower+i+1)]
```

**Conversion between array index and absolute tick:**
```python
# Array index → Absolute tick
absolute_tick = tick_lower + array_index

# Absolute tick → Array index
array_index = absolute_tick - tick_lower
```

**Example:** For prices ~100-1000 with `tick_lower = 40000` and `num_ticks = 30000`:
- `liquidity_array[0]` → tick 40,000 → price range [1.0001^40000, 1.0001^40001) ≈ [54.6, 54.6]
- `liquidity_array[6052]` → tick 46,052 → price range ≈ [100.0, 100.01]
- `liquidity_array[29078]` → tick 69,078 → price range ≈ [1000.0, 1000.1]

**Key Properties:**
- `tick_lower` is set once at environment initialization and remains **fixed** throughout simulation
- Array size `num_ticks` determines the supported price range
- Shape: `(num_ticks,)` for shared liquidity, `(num_trajectories, num_ticks)` for per-trajectory

### Core Data Flow

The environment follows a standard RL cycle with AMM-specific components:

1. **AMMEnvironment** (`gym/AMMEnvironment.py`) - Main Gymnasium environment
   - Manages episodes, state transitions via `step()` and `reset()`
   - Delegates AMM logic to **ModelDynamics**
   - Handles observation/action/reward normalization

2. **ModelDynamics** (`gym/ModelDynamics.py`) - AMM protocol implementations
   - `UniswapV3ModelDynamics`: Concentrated liquidity logic
   - Processes order flow and updates LP state via `update_state()` method

3. **StochasticProcesses** (`stochastic_processes/`) - Market simulation
   - `MidpriceModel`: Price dynamics (Brownian Motion, Geometric Brownian Motion)
   - `ArrivalModel`: Order flow (Bernoulli arrivals)
   - Each process has `update()` method called per timestep
   - All processes support multiple trajectories for batch simulation

4. **Agents** (`agents/`) - Trading strategies
   - All inherit from `Agent` base class with `get_action(state) -> action` interface
   - **Baseline agents**:
     - `RandomAgent`: Samples from action space
     - `UniformAllocationAgent`: Full range allocation `[-tau, +tau]`

5. **RewardFunctions** (`rewards/RewardFunctions.py`) - Performance metrics
   - Abstract base class with `calculate()` and `reset()` methods

### State Representation

**Uniswap V3 State** (defined in `gym/index_names.py`):

The state is a **dictionary** with the following keys:

**Pool-level state:**
| Key | Shape | Description |
|-----|-------|-------------|
| `POOL_SQRT_PRICE_KEY` | `(num_trajectories,)` | Current pool √price (not P) |
| `POOL_CURRENT_TICK_KEY` | `(num_trajectories,)` | Current tick index |
| `POOL_LIQUIDITY_ARRAY_KEY` | `(num_trajectories, num_ticks)` | Liquidity per tick (see indexing below) |

**Fees (accumulated totals):**
| Key | Shape | Description |
|-----|-------|-------------|
| `FEES0_KEY` | `(num_trajectories, num_ticks)` | Total fees collected in token0 per tick |
| `FEES1_KEY` | `(num_trajectories, num_ticks)` | Total fees collected in token1 per tick |

**LP position state:**
| Key | Shape | Description |
|-----|-------|-------------|
| `LP_LIQUIDITY_KEY` | `(num_trajectories,)` | LP's position liquidity |
| `LP_TICK_LOWER_KEY` | `(num_trajectories,)` | LP's position lower tick bound |
| `LP_TICK_UPPER_KEY` | `(num_trajectories,)` | LP's position upper tick bound |

**Market state:**
| Key | Shape | Description |
|-----|-------|-------------|
| `MARKET_MIDPRICE_KEY` | `(num_trajectories,)` | External market midprice |
| `TIME_KEY` | `(num_trajectories,)` | Current simulation time |

**Per-tick array indexing** (applies to `liquidity_array`):
- Entry `[traj, i]` corresponds to tick `[tick_lower + i, tick_lower + i + 1)`
- See "Liquidity Array Indexing" section above for conversion formulas


### Action Space - 3D Offset Format

**Action format**: `[lower_offset, upper_offset, hold_flag]` - tick offsets relative to current tick plus a hold/rebalance flag.

**Action space**: `Box(low=[-tau, -tau+1, -1.0], high=[tau-1, tau, 1.0], shape=(3,))`

| Index | Name | Bounds | Description |
|-------|------|--------|-------------|
| `[0]` | `lower_offset` | `[-tau, tau-1]` | Tick offset of position lower bound from current tick |
| `[1]` | `upper_offset` | `[-tau+1, tau]` | Tick offset of position upper bound from current tick |
| `[2]` | `hold_flag` | `[-1.0, 1.0]` | `<= 0` → rebalance; `> 0` → hold current position |

**Key Concept:**
- **Tau (τ)**: Hyperparameter defining the maximum tick window on each side of current price
- Constraint: `lower_offset < upper_offset` (enforced by `validate_action`)
- `hold_flag` allows the agent to stay idle (no rebalance, no gas cost) for a step

**`validate_action` behaviour** (applied to `action[:, :2]` only):
- Rounds offsets to integers (continuous actions from PPO snap to tick grid)
- Clips to box bounds
- Enforces `lower_offset < upper_offset`: if violated, sets `upper = lower + 1` then re-clips


### State Update Mechanism

The `update_state()` method in `UniswapV3ModelDynamics` advances the state by one step size. Each step size `step_size = terminal_time/n_{steps}` is the finite discretization of the continuous-time infinitesimal $dt$ and so, for small enough $\lambda \cdot \Delta t$, we approximate Poisson counts with a Bernoulli trial that has at most one sell and one buy arrival (boolean arrays). Each trade moves the price by exactly one tick.

**Core Principle**: Each trade = one tick of price movement.
- The square root of the AMM price is stored in grid `AMM[i]`. Each trade moves the price by one entry in the grid. 
- Arrivals are Bernoulli trials (boolean arrays): `P(arrival) = intensity * step_size`
- When both sell and buy arrive simultaneously, execution order is randomized via coin flip
- On crossing, sqrt_price snaps to the **geometric midpoint** of the new tick to prevent drift


**Key Parameters**:
- `fee_tier` (default: 0.003): Pool fee rate (0.3%)
- `exponential_value` (default: 1.0001): Tick spacing base

## Important Implementation Details

### Vectorization Best Practices

When writing code for SAiFE_gym, **always prioritize vectorization**:

**DO:**
- ✅ Use NumPy array operations: `np.where()`, `np.clip()`, `np.maximum()`, etc.
- ✅ Design functions to accept both scalars and arrays via `np.atleast_1d()`
- ✅ Use active trajectory masking to avoid branching: `active = condition; result[active] = ...`
- ✅ Represent liquidity as NumPy arrays indexed by tick offset
- ✅ Batch operations across trajectories: `(num_trajectories, ...)`
- ✅ Test with both single and multiple trajectories

**DON'T:**
- ❌ Use Python loops over trajectories (kills performance)
- ❌ Use dict-based liquidity lookups (replaced by array indexing)
- ❌ Branch with if/else on array conditions (use `np.where()` instead)
- ❌ Iterate tick-by-tick when vectorization is possible
- ❌ Mix scalar and array logic without `np.atleast_1d()` conversion

**Example - Bad (Sequential):**
```python
# DON'T DO THIS
for i in range(num_trajectories):
    if amount_remaining[i] > 0:
        price[i] = update_price(price[i], amount[i])
```

**Example - Good (Vectorized):**
```python
# DO THIS INSTEAD
active = amount_remaining > 0
price = np.where(active, update_price(price, amount), price)
```

### Environment Initialization

When creating `UniswapV3ModelDynamics`:
- **Must provide `tau`** parameter (number of ticks on each side of current price)
- **Optional**: Specify `exponential_value` (default 1.0001 for Uniswap V3 tick spacing)
- The action space is automatically set to `Box(shape=(3,))` with bounds described above
- Example: `tau=5` allows LP positions spanning up to 11 ticks (current tick ± 5)

When creating `AMMEnvironment`:
- **`initial_wealth`** (default `1e6`): LP's starting capital before first deployment
- **`gas_cost`** is a parameter of `UniswapV3ModelDynamics`, stored in state as `GAS_COST_KEY`






