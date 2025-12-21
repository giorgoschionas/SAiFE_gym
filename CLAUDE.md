# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAiFE_gym is a Reinforcement Learning environment for simulating Automated Market Maker (AMM) trading in DeFi protocols, particularly Uniswap v3. It provides an OpenAI Gym-compatible environment for training RL agents to act as liquidity providers.

**Current Status**: Active development. Core components implemented but integration is ongoing.

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# The project uses Python 3.12 with a virtual environment (venv/)
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Architecture Overview

### Core Data Flow

The environment follows a standard RL cycle with AMM-specific components:

1. **AMMEnvironment** (`gym/AMMEnvironment.py`) - Main Gym environment
   - Manages episodes, state transitions via `step()` and `reset()`
   - Delegates AMM logic to **ModelDynamics**
   - Handles observation/action/reward normalization
   - Currently under development: `_update_state()` integration

2. **ModelDynamics** (`gym/ModelDynamics.py`) - AMM protocol implementations
   - `UniswapV3ModelDynamics`: Concentrated liquidity logic
   - **Dynamic action space**: Only buckets within `2*tau +1` of current price are "active"
   - Processes order flow and updates LP state via `update_state()` method
   - Uses **(P, L) parameterization**: stores AMM sqrt price and LP's liquidity; computes token amounts on-demand
   - **3-Phase State Update**: Arbitrage correction → Noisy trader orders → Fee collection

3. **StochasticProcesses** (`stochastic_processes/`) - Market simulation
   - `MidpriceModel`: Price dynamics (Brownian Motion, Geometric Brownian Motion)
   - `ArrivalModel`: Order flow (Poisson arrivals)
   - Each process has `update()` method called per timestep
   - All processes support multiple trajectories for batch simulation

4. **Agents** (`agents/`) - Trading strategies
   - All inherit from `Agent` base class with `get_action(state) -> action` interface
   - **Action format**: Probability distribution over **active buckets only** (shape: `num_trajectories × (2*tau+1)`)
   - **Baseline agents**:
     - `RandomAgent`: Samples from action space
     - `UniformAllocationAgent`: Equal allocation across 2τ+1 active buckets
     - `SingleBucketAgent`: Concentrates on one active bucket (e.g., center, leftmost, rightmost)
     - `CurrentPriceBucketAgent`: Always allocates to center bucket (contains current price)

5. **RewardFunctions** (`rewards/RewardFunctions.py`) - Performance metrics
   - Abstract base class with `calculate()` and `reset()` methods
   - Implementations planned: `ImpermanentLoss`, `LVR`
   - 🚧 Currently stubs - needs implementation

### State Representation

**Uniswap V3 State** (defined in `gym/index_names.py`):
- Uses (P, L) parameterization - amounts computed via `calculate_position_amounts()`
- State shape: `(num_trajectories, 5)` for batch processing
- State indices:
  - `LIQUIDITY_INDEX = 0`: Liquidity amount L
  - `AMM_PRICE_INDEX = 1`: Current AMM **sqrt(price)** - IMPORTANT: stores √P, not P
  - `FEES_TOKEN_A_INDEX = 2`: Accumulated fees in Token A
  - `FEES_TOKEN_B_INDEX = 3`: Accumulated fees in Token B
  - `TIME_INDEX = 4`: Current simulation time

**Critical Note**: `AMM_PRICE_INDEX` stores the square root of price (√P), following Uniswap V3 convention. When calling helper functions that expect regular price, convert using `price = sqrt_price ** 2`.

### Action Space - Dynamic Active Buckets

**IMPORTANT**: The action space is DYNAMIC - agents allocate only to "active buckets" around the current price.

**Key Concept:**
- **Tau (τ)**: Hyperparameter defining the active bucket window
- **Active buckets**: 2τ+1 consecutive price buckets centered on the current price
  - τ buckets below current price
  - 1 bucket containing current price (center bucket at index τ)
  - τ buckets above current price
- **Action space shape**: `Box(low=0, high=1, shape=(2*tau+1,))`

**Why dynamic?**
- Focuses liquidity provision around the current trading range
- Reduces action space dimensionality
- Moves the active window as price evolves and LPs gets out of range

**Bucket Structure:**
- Each bucket: `{'p_low': lower_price, 'p_high': upper_price}`
- Bucket endpoints use exponential spacing: `base^tick_id` (default base=1.0001, matching Uniswap V3)
- Buckets are created on-demand based on current price using helper functions

**Helper functions** (`gym/helpers/AMM_utils.py`):
- `find_bucket_id(price)`: Maps price → bucket ID (tick index) via `floor(log(price))`
- `get_buckets_given_center_bucket_id(center_id, tau)`: Creates 2τ+1 buckets centered around a tick
- `create_buckets(endpoints)`: Converts tick endpoints to bucket dictionaries
- `price_to_tick(price)` / `tick_to_price(tick)`: Standard Uniswap V3 conversions

**Action interpretation:**
- If `use_mixed_strategy=True`: Action is probability distribution over 2τ+1 active buckets (must sum to 1.0)
- Otherwise: Single discrete bucket index (0 to 2τ)

### State Update Mechanism

**3-Phase Update Algorithm** (in `ModelDynamics.update_state()`):

The `update_state` method processes state transitions through three sequential phases:

#### Phase 1: Arbitrage Correction
- Checks if AMM price is outside no-arbitrage bounds
- Bounds (in sqrt space): `[√((1-fee)*midprice), √(midprice/(1-fee))]`
- Snaps price back to bounds if profitable arbitrage exists
- Prevents arbitrageurs from extracting value from the pool

#### Phase 2: Noisy Trader Orders
- Processes arrivals array: `(num_trajectories, 2)` where columns are `[SELL, BUY]`
- **SELL orders**: `sqrt_price *= (1 - non_arb_lambda)` (price decreases)
- **BUY orders**: `sqrt_price /= (1 - non_arb_lambda)` (price increases)
- Applies multiplicative price impact based on `non_arb_lambda` parameter
- If both BUY and SELL occur, both effects apply sequentially

#### Phase 3: Fee Collection
- Determines active buckets based on **initial** price (before movements)
- For each bucket with non-zero action probability:
  - Calculates liquidity: `L_bucket = action[bucket_idx] * initial_capital`
  - Tracks price sequence: [initial → after arbitrage → after noisy trades]
  - Calls `transaction_fee_one_step(bucket_low, bucket_high, p1, p2, fee_rate)`
  - Scales fees by liquidity: `fee_actual = fee_per_unit * L_bucket`
  - Accumulates in `FEES_TOKEN_A_INDEX` and `FEES_TOKEN_B_INDEX`
- Fees collected from **both** arbitrage and noisy trader price movements

**Key Parameters**:
- `non_arb_lambda` (default: 0.00005): Price impact per noisy trader order
- `initial_capital` (default: 10000.0): Total capital for liquidity provision
- `fee_tier` (default: 0.003): Pool fee rate (0.3%)

### Key Utility Functions

**AMM Utilities** (`gym/helpers/AMM_utils.py`):
- `CPMM_Spot_Price(X, Y)`: Constant product market maker price
- `calculate_liquidity_amounts(sqrt_price_current, sqrt_price_lower, sqrt_price_upper, amount0, amount1)`: Compute liquidity for given token amounts and price range
- `get_position_value(L, sqrt_price_current, sqrt_price_lower, sqrt_price_upper)`: Calculate mark-to-market value of LP position
- Price/tick conversions for Uniswap V3:
  - `price_to_tick(price)`: Convert price to tick index
  - `tick_to_price(tick)`: Convert tick index to price

**Fee Calculation Functions** (`gym/helpers/AMM_utils.py`):
- `delta_x(p1, p2)`: Calculate change in Token X reserves per unit liquidity
- `delta_y(p1, p2)`: Calculate change in Token Y reserves per unit liquidity
- `transaction_fee_one_step(a, b, p1, p2, fee_rate)`: Calculate fees for price movement from p1 to p2 within range [a, b]
  - Returns `(fee_token_a, fee_token_b)` **per unit of liquidity**
  - Must multiply by actual liquidity to get total fees
  - Used internally by `update_state()` for fee accumulation

## Important Implementation Details

### Environment Initialization

When creating `UniswapV3ModelDynamics`:
- **Must provide `tau`** parameter (number of buckets on each side of current price)
- **Optional**: Specify `exponential_value` (default 1.0001 for Uniswap V3 tick spacing)
- The action space is automatically set to `Box(shape=(2*tau+1,))`
- Example: `tau=5` creates action space over 11 active buckets (5 left + 1 center + 5 right)

### Agent Implementation Pattern

```python
class MyAgent(Agent):
    def __init__(self, env: AMMEnvironment):
        self.num_active_buckets = env.model_dynamics.num_active_buckets  # 2*tau+1
        self.tau = env.model_dynamics.tau
        self.num_trajectories = getattr(env, 'num_trajectories', 1)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        # state shape: (num_trajectories, state_dim)
        # return shape: (num_trajectories, 2*tau+1)
        # action must be valid probability distribution (sum to 1)
        # Index tau is the center bucket (contains current price)
        action = self._compute_strategy(state)
        return np.repeat(action.reshape(1, -1), self.num_trajectories, axis=0)
```

**Important Notes:**
- Agents return distributions over **active buckets only**, not all possible buckets
- The center bucket (index `tau`) always contains the current price
- No need to track bucket boundaries - handled by ModelDynamics

### State Access Patterns

Always use the index constants from `gym/index_names.py`:
```python
from SAiFE_gym.gym.index_names import (
    AMM_PRICE_INDEX,
    LIQUIDITY_INDEX,
    FEES_TOKEN_A_INDEX,
    FEES_TOKEN_B_INDEX
)

# Get sqrt price (remember: AMM_PRICE_INDEX stores √P, not P)
sqrt_price = state[0, AMM_PRICE_INDEX]
actual_price = sqrt_price ** 2  # Convert to regular price

# Get liquidity across all trajectories
liquidity = state[:, LIQUIDITY_INDEX]

# Get accumulated fees
fees_token_a = state[:, FEES_TOKEN_A_INDEX]
fees_token_b = state[:, FEES_TOKEN_B_INDEX]
```

## Work in Progress

Areas under active development:
- Reward function implementations (ImpermanentLoss, LVR) - 🚧 Currently stubs
- Full integration of state updates in `AMMEnvironment.step()`
- Comprehensive testing and validation framework

## Recently Completed

- ✅ `update_state()` method with 3-phase algorithm (arbitrage, noisy trades, fee collection)
- ✅ Fee calculation functions (`transaction_fee_one_step`, `delta_x`, `delta_y`)
- ✅ State representation with fee tracking (`FEES_TOKEN_A_INDEX`, `FEES_TOKEN_B_INDEX`)
- ✅ Dynamic active bucket system with tau parameter
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

# Create model dynamics with tau (active bucket window size)
model = UniswapV3ModelDynamics(
    midprice_model=midprice_model,
    arrival_model=arrival_model,
    tau=5,  # 5 buckets on each side of current price (11 total active buckets)
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

# Action space will be Box(shape=(11,)) for probability distribution over 11 active buckets
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

## Git Workflow

Current branch: `feature/trading-env`

Recent focus areas (from commit history):
- Complete `update_state()` implementation with arbitrage, noisy trades, and fee collection
- Fee calculation functions for Uniswap V3 concentrated liquidity
- Dynamic active bucket implementation (tau-based action space)
- Bucket creation and price discretization utilities
- UniswapV3ModelDynamics refactoring to use (√P, L) state representation
- Baseline agent implementations for dynamic action space
