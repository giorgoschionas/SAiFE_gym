# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAiFE_gym is a Reinforcement Learning environment for simulating Automated Market Maker (AMM) with Concentrated Liquidity trading in DeFi protocols, particularly Uniswap v3. It provides an OpenAI Gym-compatible environment for training RL agents to act as liquidity providers.

**Current Status**: Active development. Core components implemented but integration is ongoing.

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# The project uses Python 3.12 with a virtual environment (venv/)
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Architecture Overview

### Vectorized Environment Design

**KEY FEATURE**: SAiFE_gym is being built with **full vectorization** for high-performance parallel simulation of multiple trajectories.

**Performance Benefits:**
- Batch processing of thousands of trajectories simultaneously
- Efficient memory usage with array-based state representation
- GPU-compatible operations (via NumPy → JAX/CuPy conversion)

**Vectorization Strategy:**
- All core components operate on batched inputs: `(num_trajectories, ...)`
- State arrays, actions, rewards vectorized end-to-end
- Swap execution handles multiple price paths in parallel
- Stochastic processes generate correlated trajectory batches

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
                   = liquidity in price range [1.0001^(tick_lower+i), 1.0001^(tick_lower+i+1))
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
   - Currently under development: `_update_state()` integration

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
   - **Action format**: Probability distribution over **active ticks only** (shape: `num_trajectories × (2*tau+1)`)
   - **Baseline agents**:
     - `RandomAgent`: Samples from action space
     - `UniformAllocationAgent`: Equal allocation across 2τ+1 active tick

5. **RewardFunctions** (`rewards/RewardFunctions.py`) - Performance metrics
   - Abstract base class with `calculate()` and `reset()` methods
   - Implementations planned: `ImpermanentLoss`, `LVR`
   - 🚧 Currently stubs - needs implementation

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

**IMPORTANT**: The action space is DYNAMIC - agents allocate only to "active ticks" around the current price.

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

**Helper functions** (`gym/helpers/AMM_utils.py`):
- `find_tick_id(price)`: Maps price → tick ID (tick index) via `floor(log(price))`
- `get_ticks_given_center_tick_id(center_id, tau)`: Creates 2τ+1 ticks centered around a tick
- `price_to_tick(price)` / `tick_to_price(tick)`: Standard Uniswap V3 conversions


### State Update Mechanism

The `update_state()` method in `UniswapV3ModelDynamics` implements **liquidity-dependent price impact** where trade size and crossing behavior depend on pool liquidity depth.

**Core Principle**: Price impact is inversely proportional to liquidity.
- Trade size (xi) is computed as the minimum trade that crosses at most one tick
- Trades may or may not cross tick boundaries depending on liquidity depth
- Higher liquidity → smaller price change within tick
- Lower liquidity → price crosses tick boundary

**Trade Size (xi) Computation**:
```python
# xi is computed lazily and cached (recompute when liquidity changes)
# For SELL (token0 into pool): xi_sell = min_i { L_i * tick_factor / sqrt(p_i) }
# For BUY (token1 into pool):  xi_buy  = min_i { L_i * tick_factor * sqrt(p_i) }

def _compute_xi(self):
    # Compute sqrt prices for all ticks
    sqrt_prices = np.sqrt(1.0001 ** (tick_lower_global + np.arange(num_ticks)))

    # Compute capacity per tick
    x_capacity = liquidity_array * tick_factor / sqrt_prices  # sell capacity
    y_capacity = liquidity_array * tick_factor * sqrt_prices  # buy capacity

    # Take global minimum (ignore zero-liquidity ticks)
    self.xi_sell = np.min(np.where(L > 0, x_capacity, np.inf), axis=1)
    self.xi_buy = np.min(np.where(L > 0, y_capacity, np.inf), axis=1)
```

**Price Update Logic**:
```python
# SELL (price decreases):
#   No crossing: 1/sqrt_p_new = 1/sqrt_p_c + xi_sell/L_current
#   Crossing:    1/sqrt_p_new = 1/sqrt_p_low + xi_remaining/L_prev, tick -= 1

# BUY (price increases):
#   No crossing: sqrt_p_new = sqrt_p_c + xi_buy/L_current
#   Crossing:    sqrt_p_new = sqrt_p_high + yi_remaining/L_next, tick += 1
```

**Implementation** (11 steps):
```python
def update_state(self, arrivals: np.ndarray, action: np.ndarray):
    # Step 0: Compute xi if stale
    if self._xi_stale:
        self._compute_xi()

    # Step 1: Extract current state
    current_tick = self.state[POOL_CURRENT_TICK_KEY].copy()
    sqrt_p_c = self.state[POOL_SQRT_PRICE_KEY].copy()

    # Step 2: Compute tick boundaries
    sqrt_p_low = np.sqrt(1.0001 ** current_tick)
    sqrt_p_high = np.sqrt(1.0001 ** (current_tick + 1))

    # Step 3: Get liquidity at current, previous, and next ticks
    L_current = liquidity_array[current_tick_idx]
    L_prev = liquidity_array[prev_tick_idx]
    L_next = liquidity_array[next_tick_idx]

    # Step 4: Compute capacity to boundary
    x_to_boundary = L_current * (1/sqrt_p_low - 1/sqrt_p_c)  # sell capacity
    y_to_boundary = L_current * (sqrt_p_high - sqrt_p_c)     # buy capacity

    # Step 5: Determine arrival types
    is_sell = arrivals[:, 0].astype(bool)
    is_buy = arrivals[:, 1].astype(bool)

    # Step 6: Determine tick crossings
    sell_crosses = is_sell & (xi_sell > x_to_boundary)
    buy_crosses = is_buy & (xi_buy > y_to_boundary)

    # Step 7: Compute new sqrt_price for all 4 cases (vectorized)
    # Step 8: Combine results using np.where (no branching)
    # Step 9: Update tick based on crossing
    # Step 10: Calculate fees proportional to xi
    # Step 11: Update time
```

**Key Behavior with Uniform vs Non-Uniform Liquidity**:

| Liquidity Distribution | Crossing Behavior | Use Case |
|----------------------|-------------------|----------|
| Uniform high liquidity | Always crosses (xi ~2x boundary capacity) | Constant depth model |
| Non-uniform (low at edges) | No crossing (small xi from edge ticks) | Concentrated liquidity |
| Zero at current tick | Always crosses (zero capacity) | Out-of-range positions |

**Important**: With **uniform liquidity**, xi is determined by the tick with highest/lowest price (depending on direction), which has the lowest capacity per unit liquidity. This means crossing occurs when starting mid-tick. To get no-crossing behavior, use non-uniform liquidity where the xi-determining tick has low liquidity.

**Key Parameters**:
- `fee_tier` (default: 0.003): Pool fee rate (0.3%)
- `exponential_value` (default: 1.0001): Tick spacing base
- `tick_factor`: Precomputed `sqrt(exponential_value) - 1`
- `xi_sell`, `xi_buy`: Cached trade sizes (mark stale via `mark_xi_stale()` after liquidity changes)

**Staleness Management**:
```python
# After modifying liquidity_array, call:
model.mark_xi_stale()  # Next update_state() will recompute xi
```

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

### Agent Implementation Pattern

```python
from SAiFE_gym.gym.index_names import POOL_SQRT_PRICE_KEY, POOL_LIQUIDITY_ARRAY_KEY

class MyAgent(Agent):
    def __init__(self, env: AMMEnvironment):
        self.num_active_ticks = env.model_dynamics.num_active_ticks  # 2*tau+1
        self.tau = env.model_dynamics.tau
        self.num_trajectories = getattr(env, 'num_trajectories', 1)

    def get_action(self, state: dict) -> np.ndarray:
        # state is a dict with keys from index_names.py
        # Access current price: state[POOL_SQRT_PRICE_KEY] -> shape: (num_trajectories,)
        # return shape: (num_trajectories, 2*tau+1)
        # action must be valid probability distribution (sum to 1)
        # Index tau is the center tick (contains current price)
        sqrt_price = state[POOL_SQRT_PRICE_KEY]
        action = self._compute_strategy(sqrt_price)
        return np.repeat(action.reshape(1, -1), self.num_trajectories, axis=0)
```

**Important Notes:**
- Agents return distributions over **active ticks only**, not all possible ticks
- The center tick (index `tau`) always contains the current price
- No need to track tick boundaries - handled by ModelDynamics

### State Access Patterns

Always use the key constants from `gym/index_names.py`:
```python
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY,
    FEES1_KEY,
    LP_LIQUIDITY_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    ASSET_PRICE_KEY,
    TIME_KEY
)

# Get current sqrt price (remember: stores √P, not P)
sqrt_price = state[POOL_SQRT_PRICE_KEY]  # shape: (num_trajectories,)
actual_price = sqrt_price ** 2  # Convert to regular price

# Get liquidity array (per-tick liquidity)
liquidity_array = state[POOL_LIQUIDITY_ARRAY_KEY]  # shape: (num_trajectories, num_ticks)

# Get accumulated fees (total per trajectory)
fees_token0 = state[FEES0_KEY]  # shape: (num_trajectories,)
fees_token1 = state[FEES1_KEY]  # shape: (num_trajectories,)

# Get LP position bounds
lp_tick_lower = state[LP_TICK_LOWER_KEY]  # shape: (num_trajectories,)
lp_tick_upper = state[LP_TICK_UPPER_KEY]  # shape: (num_trajectories,)
```

## Work in Progress

Areas under active development:
- Reward function implementations (ImpermanentLoss, LVR) - 🚧 Currently stubs
- Full integration of state updates in `AMMEnvironment.step()`
- Comprehensive testing and validation framework

## Recently Completed

- ✅ **Liquidity-dependent price impact in `update_state()`** (LATEST)
  - Trade size (xi) computed as minimum capacity across all ticks
  - Price impact inversely proportional to liquidity depth
  - Crossing behavior depends on xi vs. boundary capacity
  - Fees proportional to xi (actual trade size)
  - Staleness mechanism for xi caching (`mark_xi_stale()`)
  - Comprehensive test suite (19 tests covering xi computation, crossing, vectorization)
- ✅ **Simplified `update_state()` following mbt_gym pattern** (PREVIOUS)
  - Each arrival moves price by exactly 1 tick (like mbt_gym inventory changes by 1)
  - Uses `tick_factor = sqrt(1.0001) - 1` constant for simplified fee calculation
  - No complex swap functions needed - direct tick arithmetic
  - Dict-based state with scalar fee accumulation
  - Fully vectorized across trajectories
- ✅ **Unified single-tick swap with NO branching**
  - `unified_swap_single_tick()`: Single function for complex swaps (multi-tick support)
  - Two-column arrivals: `[sell_token0, buy_token0]` with NET amount processing
  - Uses `np.where()` and array indexing for direction-dependent logic
  - `get_tick_boundaries()`: Helper to compute tick boundaries from current price
- ✅ **Fully vectorized swap implementation** with 10-100x performance improvement
  - `swap_step_within_tick_vec()`: Single-tick vectorized swap with `np.where()` conditionals
  - `execute_swap_vec_array()`: Multi-tick vectorized swap with active trajectory masking
  - Array-based liquidity representation (replaced dict-based approach)
- ✅ Fee calculation functions (`delta_x_vec`, `delta_y_vec`)
- ✅ Dict-based state representation with keys from `index_names.py`
- ✅ Dynamic active tick system with tau parameter
- ✅ Baseline agent implementations

## Common Patterns

### Creating a Custom Environment

```python
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel

# Create stochastic processes
midprice_model = BrownianMotionMidpriceModel(
    drift=0.0,
    volatility=2.0,
    initial_price=100.0,
    terminal_time=1.0,
    step_size=0.005,
    num_trajectories=1
)

arrival_model = PoissonArrivalModel(
    intensity=np.array([100, 100]),
    step_size=0.005,
    num_trajectories=1
)

# Create model dynamics with tau (active tick window size)
model = UniswapV3ModelDynamics(
    midprice_model=midprice_model,
    arrival_model=arrival_model,
    tau=5,  # 5 ticks on each side of current price (11 total active ticks)
    initial_capital=10000.0,  # Total LP capital
    fee_tier=0.003,  # 0.3% fee
    exponential_value=1.0001,  # Uniswap V3 tick spacing
    non_arb_lambda=0.00005  # Price impact per noisy trader order
)

env = AMMEnvironment(
    model_dynamics=model,
    terminal_time=1.0,
    n_steps=200
)

# Action space will be Box(shape=(11,)) for probability distribution over 11 active ticks
print(f"Action space: {env.action_space}")  # Box(0.0, 1.0, (11,), float32)
```

### Testing Baseline Agents

```python
from SAiFE_gym.agents.BaselineAgents import UniformAllocationAgent

agent = UniformAllocationAgent(env)
obs = env.reset()

for _ in range(100):
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

### Using Vectorized Swap Functions

**IMPORTANT**: Always use the **array-based** implementation for production code.

#### Single-Tick Swap (vectorized across trajectories)

```python
import numpy as np
from SAiFE_gym.gym.helpers.AMM_utils import swap_step_within_tick_vec

# Setup: 1000 trajectories with different initial prices
num_trajectories = 1000
sqrt_price_current = np.full(num_trajectories, 100.0)  # All start at √P = 100
sqrt_price_target = np.full(num_trajectories, 101.0)   # Target √P = 101
liquidity = np.full(num_trajectories, 1e6)             # 1M liquidity per trajectory
amount_remaining = np.random.uniform(100, 1000, num_trajectories)  # Random swap sizes

# Execute vectorized swap
sqrt_price_next, amount_in, amount_out, fee = swap_step_within_tick_vec(
    sqrt_price_current=sqrt_price_current,
    sqrt_price_target=sqrt_price_target,
    liquidity=liquidity,
    amount_remaining=amount_remaining,
    fee_rate=0.003,
    zero_for_one=True  # Selling token0 for token1
)

# Results shape: (1000,) - one value per trajectory
print(f"Final prices: {sqrt_price_next.shape}")  # (1000,)
print(f"Fees collected: {fee.sum():.2f}")        # Total across all trajectories
```

#### Multi-Tick Swap with Array-Based Liquidity

```python
import numpy as np
from SAiFE_gym.gym.helpers.AMM_utils import (
    execute_swap_vec_array,
    price_to_tick,
    tick_to_price
)

# Create liquidity array centered around current price
current_price = 100.0
center_tick = price_to_tick(current_price)
num_ticks = 200  # Track liquidity for 200 ticks
tick_lower = center_tick - 100  # Start 100 ticks below

# Create liquidity distribution (e.g., concentrated around current price)
tick_indices = np.arange(num_ticks)
tick_distance = np.abs(tick_indices - 100)  # Distance from center
liquidity_array = 1e6 * np.exp(-tick_distance / 20.0)  # Gaussian-like distribution

# Setup trajectories
num_trajectories = 1000
sqrt_price_current = np.full(num_trajectories, np.sqrt(current_price))
amount_in = np.random.uniform(1000, 10000, num_trajectories)

# Execute multi-tick swap across all trajectories in parallel
sqrt_price_final, consumed, amount_out, total_fee = execute_swap_vec_array(
    sqrt_price_current=sqrt_price_current,
    liquidity_array=liquidity_array,  # Shape: (200,)
    tick_lower=tick_lower,
    amount_in=amount_in,              # Shape: (1000,)
    zero_for_one=True,
    fee_rate=0.003,
    exponential_value=1.0001
)

# Analyze results
final_prices = sqrt_price_final ** 2
price_impact = (final_prices - current_price) / current_price * 100
print(f"Average price impact: {price_impact.mean():.2f}%")
print(f"Total fees collected: {total_fee.sum():.2f}")
```

**Key Points:**
- `liquidity_array` can be 1D (shared across trajectories) or 2D (per-trajectory liquidity)
- Array indices represent tick offsets from `tick_lower`
- Out-of-bounds ticks automatically treated as zero liquidity
- No explicit tick boundaries needed - handled internally
- Performance: processes 1000 trajectories faster than 1 sequential trajectory with dicts

#### Unified Single-Tick Swap (No Branching) - NEW

```python
import numpy as np
from SAiFE_gym.gym.helpers.AMM_utils import (
    unified_swap_single_tick,
    get_tick_boundaries
)

# Setup: 1000 trajectories
num_trajectories = 1000
sqrt_price_current = np.full(num_trajectories, 10.0)  # sqrt(100) = 10
liquidity = np.full(num_trajectories, 100000.0)

# Two-column arrivals: [sell_token0, buy_token0]
# Some trajectories sell, some buy, some do both
rng = np.random.RandomState(42)
amount_in = rng.uniform(0, 100, size=(num_trajectories, 2))

# Get tick boundaries
tick_lower, tick_upper, current_tick = get_tick_boundaries(sqrt_price_current)

# Execute unified swap (no if/else branching internally)
sqrt_price_next, token0_net, token1_net, fee0, fee1, hit_boundary = unified_swap_single_tick(
    sqrt_price_current=sqrt_price_current,
    liquidity=liquidity,
    amount_in=amount_in,
    tick_lower_boundary=tick_lower,
    tick_upper_boundary=tick_upper,
    fee_rate=0.003
)

# Results shape: (1000,) for all outputs
print(f"Trajectories with boundary crossing: {hit_boundary.sum()}")
print(f"Total token0 fees: {fee0.sum():.2f}")
print(f"Total token1 fees: {fee1.sum():.2f}")
```

**Key Points:**
- Input `amount_in` has shape `(num_trajectories, 2)`: column 0 = sell token0, column 1 = buy token0
- Net direction computed internally: `net = amount_in[:, 0] - amount_in[:, 1]`
- No `if zero_for_one` branching - uses multiplier/indexing approach
- Single-tick assumption: suitable for small swaps or high-frequency state updates
- Returns both token flows with sign convention: positive = into pool

## Git Workflow

Current branch: `feature/trading-env`

Recent focus areas (from commit history):
- **Vectorized swap implementation** (array-based, 10-100x speedup)
  - `swap_step_within_tick_vec()`: Fully vectorized single-tick swaps
  - `execute_swap_vec_array()`: Multi-tick vectorized execution with active masking
  - Removed all dict-based legacy code for cleaner codebase
  - Comprehensive test suite (11 tests, all passing)
- Complete `update_state()` implementation with arbitrage, noisy trades, and fee collection
- Fee calculation functions for Uniswap V3 concentrated liquidity (`delta_x_vec`, `delta_y_vec`)
- Dynamic active tick implementation (tau-based action space)
- tick creation and price discretization utilities
- UniswapV3ModelDynamics refactoring to use (√P, L) state representation
- Baseline agent implementations for dynamic action space
