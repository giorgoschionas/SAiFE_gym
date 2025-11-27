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
   - Processes order flow and updates LP state
   - Uses **(P, L) parameterization**: stores AMM price and LP's liquidity; computes token amounts on-demand

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
- State indices:
  - `LIQUIDITY_INDEX`: Liquidity amount L
  - `AMM_PRICE_INDEX`: Current LP price
  - `ASSET_PRICE_INDEX`: Current real price tick
  - `TIME_INDEX`: Current simulation time

**Important**: State shape is `(num_trajectories, state_dim)` for batch processing.

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

### Key Utility Functions

**AMM Utilities** (`gym/helpers/AMM_utils.py`):
- `CPMM_Spot_Price(X, Y)`: Constant product market maker price
- `calculate_liquidity_amounts()`: Compute liquidity for given token amounts and price range
- Price/tick conversions for Uniswap V3

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
from SAiFE_gym.gym.index_names import AMM_PRICE_INDEX, LIQUIDITY_INDEX

current_price = state[0, AMM_PRICE_INDEX]
liquidity = state[:, LIQUIDITY_INDEX]  # All trajectories
```

## Work in Progress

Areas under active development:
- State update mechanism integration in `AMMEnvironment.step()`
- Reward function implementations (ImpermanentLoss, LVR)
- Linking stochastic processes with model dynamics
- Testing and validation framework

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
    initial_capital=10000.0,
    fee_tier=0.003,
    exponential_value=1.0001  # Uniswap V3 tick spacing
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
- Dynamic active bucket implementation (tau-based action space)
- Bucket creation and price discretization utilities
- UniswapV3ModelDynamics refactoring to use (P, L) state
- Baseline agent implementations for dynamic action space
