---
name: stochastic-control-approximation
description: Approximate optimal control policies for liquidity provision in Constant Function Markets with Concentrated Liquidity using stochastic processes. Use when analyzing optimal strategies, deriving control policies, or when the user asks "what is the optimal strategy for providing liquidity in this scenario?"
---

1. **Discretization**: All the stochastic processes (midprice, order arrivals, etc.) are discretized for numerical computation and simulation purposes. They share a common time `step_size` which is defined in the environment `AMMEnvironment.py`, it is defined as `step_size = terminal_time / n_steps` and it the finite discretization of the continuous-time infinitesimal dt in the stochastic processes. This allows for consistent updates across all processes at each timestep.

2. **Arrival Process**: The order arrival process is mainly modeled as a non-homogeneous Poisson process, where the arrival rate can vary over time. For small enough `intensity * step_size`, e.g., $\le 0.01$, at each step, we can approximate the Poisson counts with a Bernoulli random variable, with as success probability `p = intensity * step_size`. This means that at each step, there can be at most one buy and one sell arrival, which simplifies the simulation while still capturing the stochastic nature of order arrivals. 



