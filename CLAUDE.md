## Project Overview

SAiFE_gym is a Reinforcement Learning environment for simulating Automated Market Maker (AMM) with Concentrated Liquidity trading in DeFi protocols, like Uniswap v3. It provides an OpenAI Gym-compatible environment for training RL agents to act as liquidity providers.

**Current Status**: Active development. Core components implemented but integration is ongoing.

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# The project uses Python 3.12 with a virtual environment (venv/)
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
- Out-of-bounds ticks treated as zero liquidity (no dict lookups)

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
- Out-of-bounds ticks (index < 0 or ≥ num_ticks) are treated as zero liquidity
- Shape: `(num_ticks,)` for shared liquidity, `(num_trajectories, num_ticks)` for per-trajectory

### Core Data Flow

The environment follows a standard RL cycle with AMM-specific components:

1. **AMMEnvironment** (`gym/AMMEnvironment.py`) - Main Gym environment
   - Manages episodes, state transitions via `step()` and `reset()`
   - Delegates AMM logic to **ModelDynamics**
   - Handles observation/action/reward normalization

2. **ModelDynamics** (`gym/ModelDynamics.py`) - AMM protocol implementations
   - `UniswapV3ModelDynamics`: Concentrated liquidity logic
   - **Dynamic action space**: Only ticks within `2*tau +1` of current price are "active"
   - Processes order flow and updates LP state via `update_state()` method

3. **StochasticProcesses** (`stochastic_processes/`) - Market simulation
   - `MidpriceModel`: Price dynamics (Brownian Motion, Geometric Brownian Motion)
   - `ArrivalModel`: Order flow (e.g. Poisson arrivals)
   - Each process has `update()` method called per timestep
   - All processes support multiple trajectories for batch simulation

4. **Agents** (`agents/`) - Trading strategies
   - All inherit from `Agent` base class with `get_action(state) -> action` interface
   - **Baseline agents**:
     - `RandomAgent`: Samples from action space
     - `UniformAllocationAgent`: Equal allocation across 2τ+1 active tick

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
| `FEES0_KEY` | `(num_trajectories,)` | Total fees collected in token0 |
| `FEES1_KEY` | `(num_trajectories,)` | Total fees collected in token1 |

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

**Critical Note**: `POOL_SQRT_PRICE_KEY` stores √P (not P), following Uniswap V3 convention. Convert with `price = sqrt_price ** 2`.

### Action Space - Dynamic Active Ticks


**Key Concept:**
- **Tau (τ)**: Hyperparameter defining the active tick window
- **Active ticks**: 2τ+1 consecutive price ticks centered on the current price
  - τ ticks below current price
  - 1 tick containing current price (center tick at index τ)
  - τ ticks above current price
- **Action space shape**: `Box(low=0, high=1, shape=(2*tau+1,))`

**Why dynamic?**
- Focuses liquidity provision around the current trading range
- Reduces action space dimensionality
- Moves the active window as price evolves and LPs gets out of range

**tick Structure:**
- Each tick: `{'p_low': lower_price, 'p_high': upper_price}`
- tick endpoints use exponential spacing: `base^tick_id` (default base=1.0001, matching Uniswap V3)
- ticks are created on-demand based on current price using helper functions


### State Update Mechanism

The `update_state()` method in `UniswapV3ModelDynamics` implements **local xi + Poisson arrivals**. Each trade moves the price by exactly one tick, and multiple trades per step are generated by Poisson sampling.

**Core Principle**: Each trade = one tick of price movement.
- Trade size (xi) is computed from the **current tick's liquidity only** (`_compute_local_xi()`)
- `xi_sell = L * tick_factor / sqrt_p_low`, `xi_buy = L * tick_factor * sqrt_p_high`
- Arrival counts are Poisson-distributed integers (can exceed 1 per step)
- `update_state()` loops over arrival counts, processing one trade at a time
- On crossing, sqrt_price snaps to the **geometric midpoint** of the new tick to prevent drift

**Crossing Behavior**:
- With mid-tick price and local xi (full tick capacity), xi > x_to_boundary → crossing occurs
- Zero-liquidity ticks are always crossed (no resistance)
- On crossing, `sqrt_price = sqrt_p_boundary / exp_quarter` (sell) or `sqrt_p_boundary * exp_quarter` (buy)
- Fees are charged only at the current tick for the x_to_boundary / y_to_boundary portion

**Key Parameters**:
- `fee_tier` (default: 0.003): Pool fee rate (0.3%)
- `exponential_value` (default: 1.0001): Tick spacing base
- `tick_factor`: Precomputed `sqrt(exponential_value) - 1`
- `exp_quarter`: Precomputed `exponential_value ** 0.25` (for midpoint snap)
- `max_arrivals_per_step` (default: 50): Safety cap on Poisson counts

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
- The action space is automatically set to `Box(shape=(2*tau+1,))`
- Example: `tau=5` creates action space over 11 active ticks (5 left + 1 center + 5 right)







