## Project Overview

SAiFE_gym is a vectorized Reinforcement Learning simulator for Automated Market Maker (AMM) trading with Concentrated Liquidity in DeFi protocols, like Uniswap v3. Dedicated Gymnasium and Stable-Baselines3 adapters support training RL liquidity providers.

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
- Directional liquidity lookups use safe array indices and validity masks;
  swap behavior at the tracked boundaries depends on the price-impact model

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
- Valid liquidity interval indices are `0 <= index < num_ticks`; the price
  lattice has `num_ticks + 1` boundaries. Kernel-arrival and depth lookups mask
  unavailable intervals, so they contribute zero to those calculations.
- Swap execution stays within the tracked window: one-tick impact asserts on
  an outward boundary crossing, while liquidity-depth impact caps movement.
- Shape: `(num_ticks,)` for shared liquidity, `(num_trajectories, num_ticks)` for per-trajectory

### Core Data Flow

The environment follows a standard RL cycle with AMM-specific components:

1. **AMMEnvironment** (`gym/AMMEnvironment.py`) - Native batched simulator
   - Manages episodes, state transitions via `step()` and `reset()`
   - Delegates AMM logic to **ModelDynamics**
   - Returns full-precision batched state, rewards, and termination flags
   - Delegates library-specific interfaces to the Gymnasium and SB3 adapters

2. **ModelDynamics** (`gym/ModelDynamics.py`) - AMM protocol implementations
   - `UniswapV3ModelDynamics`: Concentrated liquidity logic
   - Processes order flow and updates LP state via `update_state()` method
   - Delegates swap price impact to a `PriceImpactModel`
   - Delegates swap fee writes to a `FeeAccountingModel`

3. **StochasticProcesses** (`stochastic_processes/`) - Market simulation and swap components
   - `MidpriceModel`: Price dynamics (Brownian Motion, Geometric Brownian Motion)
   - `ArrivalModel`: Order flow (Bernoulli arrivals)
   - `PriceImpactModel`: AMM pool-state transition for swaps
   - `FeeAccountingModel`: swap fee accounting based on price-impact output
   - Stochastic process models have an `update()` method called per timestep
   - Core simulation components support multiple trajectories for batch simulation

4. **Agents** (`agents/`) - Trading strategies
   - All inherit from `Agent` base class with `get_action(state) -> action` interface
   - **Baseline agents**:
     - `RandomAgent`: Samples from action space
     - `UniformAllocationAgent`: Full range allocation `[-tau, +tau]`

5. **RewardFunctions** (`rewards/RewardFunctions.py`) - Performance metrics
   - Abstract base class with `calculate()` and `reset()` methods

### Environment Interfaces

- Preserve `AMMEnvironment`'s batched returns, including for one trajectory.
  Its `action_space`/`single_action_space` describes one `(3,)` command;
  `batched_action_space` describes `(num_trajectories, 3)` commands. Raw `step`
  accepts `(3,)` only for one trajectory, plus existing batched commands.
- `GymnasiumAMMEnvironment` in `gym/GymnasiumAMMEnvironment.py` adapts one
  trajectory to `gymnasium.Env`: unbatched Dict observations, scalar reward,
  boolean flags, and explicit reset. Apply standard Gymnasium wrappers here.
- `GymnasiumAMMVectorEnv` in the same module adapts the native batch to
  `gymnasium.vector.VectorEnv`. Use Gymnasium vector wrappers here. It exposes
  single and batched spaces, masked info dictionaries, and next-step autoreset.
  All trajectories reset together; a scalar seed seeds the shared RNG streams.
  Seed lists and partial reset masks are unsupported. Use separate single
  adapters with Gymnasium SyncVectorEnv/AsyncVectorEnv for independent resets.
- Both Gymnasium adapters preserve the full Dict observation and return copies.
  They accept raw or domain-randomized AMM environments and support no rendering.
- `StableBaselinesAMMEnvironment` remains the SB3 adapter with the existing
  float32 feature vector and same-step autoreset. Preserve this interface for
  `experiments/train_robust_lp_agent.py`, including observation ordering,
  normalization, and structured action mappings.
- `DiscreteActionWrapper` is a native batched action converter; it does not
  convert the raw simulator's returns into a single Gymnasium Env API.

### State Representation

**Uniswap V3 State** (defined in `gym/index_names.py`):

The state is a **dictionary** with 22 keys, all declared in the observation
space. Tick indices remain int64 through reset and rebalance; `lp_ever_deployed`
is boolean; the remaining fields use float64. The tables below group core keys.
Raw scalar fields have shape `(num_trajectories,)`; the single Gymnasium adapter
removes that dimension. Elapsed time has a nonnegative, unbounded observation
space to accommodate roundoff at the configured terminal horizon.

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
| `LP_EVER_DEPLOYED_KEY` | `(num_trajectories,)` | Whether liquidity was previously deployed |
| `LP_FEE_SNAPSHOT0_KEY` / `LP_FEE_SNAPSHOT1_KEY` | `(num_trajectories,)` | Fee entitlement at position entry |
| `GAS_COST_KEY` / `INITIAL_WEALTH_KEY` | `(num_trajectories,)` | Episode rebalance cost and starting capital |

**Market state:**
| Key | Shape | Description |
|-----|-------|-------------|
| `ASSET_PRICE_KEY` | `(num_trajectories,)` | External market midprice |
| `TIME_KEY` | `(num_trajectories,)` | Current simulation time |

**Per-tick array indexing** (applies to `liquidity_array`):
- Entry `[traj, i]` corresponds to tick `[tick_lower + i, tick_lower + i + 1)`
- See "Liquidity Array Indexing" section above for conversion formulas

**Critical Note**: `POOL_SQRT_PRICE_KEY` stores √P (not P), following Uniswap V3 convention. Convert with `price = sqrt_price ** 2`.

### Action Space - 3D Offset Format

`AMMEnvironment.action_space` is the low-level simulator command space, not the
canonical discrete LP decision space. The raw 3D Box is useful for direct model
execution and continuous-control agents, but it contains redundant hold
commands because offsets are ignored whenever `hold_flag > 0`. Use
`SAiFE_gym.wrappers.DiscreteActionWrapper` or `DiscreteActionVecEnv` when an
agent should see one unique hold action plus finite rebalance ranges.

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


### Swap Price Impact and Fee Accounting

Swap handling is intentionally split into two independent responsibilities:

1. **Price impact** (`stochastic_processes/price_impact_models.py`)
   - `PriceImpactModel` defines the interface for AMM price-impact rules.
   - `OneTickUniswapV3PriceImpact` is the default model for `UniswapV3ModelDynamics`.
   - `OneTickUniswapV3PriceImpact` moves each active swap exactly one tick in the swap direction. LP liquidity affects the implied swap amount and fees, but not the number of ticks crossed.
   - `LiquidityDepthUniswapV3PriceImpact` samples gross trade sizes, deducts fees before price impact, and maps the resulting curve input to tick movement using local directional liquidity depth. Deeper liquidity reduces impact for the same sampled trade size; thinner liquidity increases it.
   - `LiquidityDepthUniswapV3PriceImpact` supports `trade_size_unit="input_token"` and `trade_size_unit="token1_notional"`. In token1-notional mode, sampled sizes are gross token1 notionals; sell-side notionals are converted to gross token0 using the external midprice (`ASSET_PRICE_KEY`) before fee deduction.
   - Price-impact models mutate only pool price state: `POOL_CURRENT_TICK_KEY` and `POOL_SQRT_PRICE_KEY`.
   - Price-impact models return a vectorized `SwapResult` containing affected trajectories, fee array key, fee indices, curve/net input amounts used as the fee basis, and direction.
   - Price-impact models do not write to `FEES0_KEY` or `FEES1_KEY`.

2. **Fee accounting** (`stochastic_processes/fee_accounting_models.py`)
   - `FeeAccountingModel` defines the interface for applying fees.
   - `UniswapV3FeeAccounting` is the default model.
   - It consumes `SwapResult` and applies:
     ```python
     state[swap_result.fee_key][swap_result.trajectories, swap_result.fee_indices] += (
         fee_multiplier * swap_result.amounts
     )
     ```
   - `swap_result is None` is a no-op, which covers steps with no active swap on that side.

`UniswapV3ModelDynamics._process_swap()` orchestrates these two operations:

```python
swap_result = self.price_impact_model.process_swap(...)
self.fee_accounting_model.apply_fees(self.state, swap_result, self.fee_multiplier)
```

This preserves the previous one-tick Uniswap V3 behavior by default while
making both components injectable and testable independently.

### State Update Mechanism

The `update_state()` method in `UniswapV3ModelDynamics` advances the state by
`dt = terminal_time / n_steps`. Arrival models draw boolean indicators for
each side using `P(arrival) = 1 - exp(-max(intensity, 0) * dt)`, the probability
of at least one Poisson arrival over the step. Each side can generate at most
one arrival indicator per step; multiple arrivals on that side are not
simulated. For small `intensity * dt`, this probability is approximately
`intensity * dt`; the implementation uses the exponential formula. The
injected `PriceImpactModel` determines how far an active trade moves the pool.

**Default One-Tick Principle**: With `OneTickUniswapV3PriceImpact`, each trade = one tick of price movement.
- Trade size is computed from the **crossed tick interval's liquidity only** in `OneTickUniswapV3PriceImpact`
- LP liquidity affects fees and implied swap amount, but not the one-tick price move
- Sell token0 arrival (`direction=-1`): fees are accounted in `FEES0_KEY` at `idx - 1`
- Buy token0 arrival (`direction=1`): fees are accounted in `FEES1_KEY` at `idx`
- Sell amount: `L * (1 / sqrt_grid[idx - 1] - 1 / sqrt_grid[idx])`
- Buy amount: `L * (sqrt_grid[idx + 1] - sqrt_grid[idx])`
- Arrivals use the Bernoulli probability described above, independently of
  the choice of price-impact model
- When both sell and buy arrive simultaneously, execution order is randomized via coin flip
- On crossing, `sqrt_price` is set from the tick lattice: `sqrt_grid[new_tick - tick_lower_global]`

**Liquidity-Depth Principle**: With `LiquidityDepthUniswapV3PriceImpact`, each trade can cross zero, one, or many ticks.
- Sampled trade size is gross input; fee-adjusted curve input is divided by average local directional one-tick capacity over `depth_window`
- Fractional expected tick movement is stochastically rounded to an integer move
- Tick movement is capped by the available tick window
- Zero-tick sampled arrivals allocate all fees to the first executable interval
- Multi-tick sampled arrivals allocate the full curve input across realized crossed intervals using directional capacity weights
- This model remains vectorized across trajectories; it uses array masks and scatter-style fee accounting
- Mixed trajectory batches gather padded fee intervals through clipped indices
  before applying validity masks, keeping boundary lookups within the arrays

**Crossing Behavior**:
- Here `idx = current_tick - tick_lower_global` is the current price boundary.
- The one-tick model requires `1 <= idx <= num_ticks` for sells and
  `0 <= idx < num_ticks` for buys. An active outward swap at the lower/upper
  boundary raises an assertion; increase `num_ticks` to accommodate longer paths.
- Within that window, the one-tick model advances one tick even through an
  interval with zero liquidity; its returned swap amount is zero.
- Liquidity-depth impact caps movement to the available boundaries. An outward
  arrival already at a boundary has no executable interval, so it leaves the
  pool price unchanged and contributes no fees.

**Key Parameters**:
- `fee_tier` (default: 0.003): Pool fee rate (0.3%)
- `fee_multiplier = fee_tier / (1.0 - fee_tier)`: multiplier applied by fee accounting to swap amounts
- `exponential_value` (default: 1.0001): Tick spacing base
- `sqrt_grid`: Precomputed tick lattice with shape `(num_ticks + 1,)`

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
- **Optional**: Specify `tau` (default `5`), the maximum raw action-bound offset
  on each side of the current tick
- **Optional**: Specify `exponential_value` (default 1.0001 for Uniswap V3 tick spacing)
- **Optional**: Inject `price_impact_model`; defaults to `OneTickUniswapV3PriceImpact`
- **Optional**: Inject `fee_accounting_model`; defaults to `UniswapV3FeeAccounting`
- The action space is automatically set to `Box(shape=(3,))` with bounds described above
- Example: with `tau=5`, the full raw range `[-5, +5)` covers 10 tick intervals
  between 11 boundaries. In general, `[-tau, +tau)` covers `2 * tau` intervals.

When creating `AMMEnvironment`:
- **`initial_wealth`** (default `1e6`): LP's starting capital before first deployment
- **`gas_cost`** is a parameter of `UniswapV3ModelDynamics`, stored in state as `GAS_COST_KEY`
