# SAiFE vs Neural: Uniswap v3 Simulation Environment Comparison

This document outlines the key architectural and implementation differences between the two Uniswap v3 simulation environments.

## Overview

| Aspect | **saife** | **neural** |
|--------|-----------|------------|
| **Architecture** | OpenAI Gym environment | Custom simulation loop (not Gym) |
| **Vectorization** | Fully vectorized (1000+ trajectories in parallel) | Sequential (1 trajectory at a time) |
| **Performance** | 10-100x faster via NumPy batch ops | Slow Python loops |
| **Purpose** | High-fidelity RL training | Strategy research & prototyping |

---

## Swap Execution Model

| | **saife** | **neural** |
|--|-----------|------------|
| **Mechanism** | Each arrival moves price by 1 tick | Fixed batch of 10 trades/step, each moves price by `λ%` |
| **Liquidity** | Array-based per-tick liquidity tracking | No explicit liquidity tracking |
| **Fees** | Real-time accumulation during swaps | Post-hoc calculation from price path |
| **Swap Math** | Explicit `delta_x`, `delta_y` with `sqrt_price` | Simplified percentage price jumps |

### saife Implementation
```python
# Each arrival moves price by exactly 1 tick
tick_change = arrivals[:, 1] - arrivals[:, 0]  # buy - sell
new_tick = current_tick + tick_change
new_sqrt_price = sqrt(1.0001 ** new_tick)

# Fees calculated using tick_factor
tick_factor = sqrt(1.0001) - 1  # ≈ 0.00005
fee_token0 = fee_multiplier * L * tick_factor / sqrt_price
fee_token1 = fee_multiplier * L * sqrt_price * tick_factor
```

### neural Implementation
```python
# Fixed 10 trades per timestep, each moves price by non_arb_lambda
for j in range(num_non_arb):  # num_non_arb = 10
    if np.random.random() < 0.5:
        current_pool_price *= (1 - non_arb_lambda)  # Sell
    else:
        current_pool_price /= (1 - non_arb_lambda)  # Buy

# Fees calculated post-hoc from price movements
fees = transaction_fee_for_sequence(bucket_low, bucket_high, price_seq, fee_rate)
```

---

## State Representation

| | **saife** | **neural** |
|--|-----------|------------|
| **Format** | Dict with named keys | Scalar prices + bucket lists |
| **Price** | `sqrt_price` (follows Uniswap convention) | Regular price |
| **Liquidity** | `liquidity_array[traj, tick_idx]` | Implicit (1 unit per bucket) |
| **Tracking** | Per-trajectory arrays `(N, ...)` | Single values |

### saife State Dictionary
```python
state = {
    'sqrt_price': np.ndarray,      # (num_trajectories,)
    'current_tick': np.ndarray,    # (num_trajectories,)
    'liquidity_array': np.ndarray, # (num_trajectories, num_ticks)
    'fees_0': np.ndarray,          # (num_trajectories,)
    'fees_1': np.ndarray,          # (num_trajectories,)
    'lp_liquidity': np.ndarray,    # (num_trajectories,)
    'lp_tick_lower': np.ndarray,   # (num_trajectories,)
    'lp_tick_upper': np.ndarray,   # (num_trajectories,)
    'midprice': np.ndarray,        # (num_trajectories,)
    'time': np.ndarray,            # (num_trajectories,)
}
```

### neural State (Implicit)
```python
# No explicit state dict - variables tracked separately
pool_price_seq = [p0, p1, p2, ...]  # List of prices
external_price_seq = [e0, e1, ...]  # External market prices
center_bucket = int                  # Current bucket ID
wealth = float                       # Current portfolio value
```

---

## Action Space

| | **saife** | **neural** |
|--|-----------|------------|
| **Format** | 2D offset `[lower_offset, upper_offset]` | Softmax probabilities over `2τ+2` buckets |
| **Shape** | `(num_trajectories, 2)` | `(2*tau + 2,)` |
| **Meaning** | Tick bounds relative to current price | Allocation weights including "hold cash" option |
| **Capital** | Always fully deployed | Can hold cash (not deploy) |
| **Constraint** | `lower_offset < upper_offset` | Probabilities sum to 1 |

### saife Action Example (τ=5)
```python
# Action: [lower_offset, upper_offset]
action = [-3, 2]  # Position spans 6 ticks: current_tick-3 to current_tick+2

# Bounds
action_space = Box(low=[-5, -4], high=[4, 5], shape=(2,))
```

### neural Action Example (τ=5)
```python
# Action: probability distribution over 12 options
# [bucket_-5, bucket_-4, ..., bucket_0, ..., bucket_+5, cash]
action = softmax(nn_output)  # Shape: (12,)

# Allocation
for i, weight in enumerate(action[:-1]):
    allocate(bucket=center_bucket - tau + i, weight=weight)
cash_weight = action[-1]  # Hold as cash
```

---

## Order Flow (Arrivals)

| | **saife** | **neural** |
|--|-----------|------------|
| **Model** | Stochastic Poisson arrivals | Deterministic 10 trades/timestep |
| **State-dependent** | Yes (`PoissonLinearArrivalModel`) | No (only time-varying via `tanh`) |
| **Output** | Binary `[sell, buy]` per trajectory | Price impact magnitude |
| **Arbitrage** | Implicit via arrival intensity | Explicit price correction |

### saife Arrival Model
```python
class PoissonLinearArrivalModel:
    def update(self, state):
        # Intensity depends on mispricing
        mispricing = external_price - amm_price
        intensity_sell = max(α₀, α₁ + α₂*L - α₃*mispricing)
        intensity_buy = max(α₀, α₁ + α₂*L + α₃*mispricing)
        self.current_state = np.stack([intensity_sell, intensity_buy], axis=1)

    def get_arrivals(self):
        # Stochastic Poisson sampling
        return np.random.uniform(size=(N, 2)) < self.current_state * dt
```

### neural Arrival Model
```python
class TanhNonArbModel:
    def get_non_arb_params(self, t):
        # Time-varying only, no state dependence
        lambda_t = mean_lambda + amplitude * np.tanh(t / t_horizon * multiplier)
        return num_non_arb, lambda_t  # (10, ~0.00005)

# Arbitrage is explicit price correction
if pool_price < (1 - fee) * external_price:
    pool_price = (1 - fee) * external_price  # Instant correction
```

---

## Reward / Objective Function

| | **saife** | **neural** |
|--|-----------|------------|
| **Function** | Modular `RewardFunction` class | CARA utility on final wealth |
| **Risk** | Configurable per reward function | Fixed `risk_averse_a = 10` |
| **Timing** | Per-step rewards | Terminal wealth only |
| **Components** | PnL, Impermanent Loss, LVR (planned) | Fees + IL - rebalancing costs |

### saife Reward
```python
class RewardFunction(ABC):
    @abstractmethod
    def calculate(self, current_state, action, next_state, done) -> np.ndarray:
        pass

class PnL(RewardFunction):
    def calculate(self, current_state, action, next_state, done):
        return next_wealth - current_wealth  # Per-step PnL
```

### neural Reward
```python
def raw_risk_averse_utility(a, x):
    # CARA utility applied to final wealth only
    if a != 0:
        return (1 - torch.exp(-a * x)) / a
    return x

# Called only at episode end
final_utility = raw_risk_averse_utility(risk_averse_a=10, wealth=final_wealth)
```

---

## Vectorization Comparison

### saife (Vectorized)
```python
# Process 1000 trajectories in parallel
num_trajectories = 1000
sqrt_price = np.full(num_trajectories, 10.0)
arrivals = arrival_model.get_arrivals()  # Shape: (1000, 2)

# Vectorized tick update
tick_change = arrivals[:, 1] - arrivals[:, 0]  # Shape: (1000,)
new_tick = current_tick + tick_change          # Shape: (1000,)

# Vectorized fee calculation
active_liquidity = liquidity_array[np.arange(N), tick_idx]  # Shape: (1000,)
fees = fee_multiplier * active_liquidity * tick_factor      # Shape: (1000,)
```

### neural (Sequential)
```python
# Process 1 trajectory at a time
for sample_idx in range(num_samples):  # 1000 iterations
    price_seq = price_model.get_price_sequence_sample(p0, t_horizon)

    for t in range(t_horizon):  # 1000 timesteps
        # Sequential price updates
        for j in range(num_non_arb):  # 10 trades
            if np.random.random() < 0.5:
                current_price *= (1 - lambda_t)
            else:
                current_price /= (1 - lambda_t)
```

---

## Summary

### saife Strengths
- High-fidelity tick-by-tick AMM simulation
- Fully vectorized for GPU-compatible training
- State-dependent order flow (arbitrageurs react to mispricing)
- OpenAI Gym interface for standard RL algorithms
- Modular reward functions

### neural Strengths
- Clean, simple codebase for strategy research
- Built-in baseline comparisons (UPRA, ULRA, OIRA, ODRA)
- Risk-adjusted optimization (CARA utility)
- Explicit arbitrage modeling
- PyTorch integration for neural network policies

### When to Use Which

| Use Case | Recommended |
|----------|-------------|
| Training RL agents at scale | **saife** |
| Quick strategy prototyping | **neural** |
| Realistic AMM dynamics | **saife** |
| Risk-adjusted optimization | **neural** |
| Multi-trajectory Monte Carlo | **saife** |
| Single-trajectory analysis | **neural** |
