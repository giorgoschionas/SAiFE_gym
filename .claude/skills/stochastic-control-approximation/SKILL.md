---
name: stochastic-control-approximation
description: Approximate control policies for liquidity provision in Constant Function Markets with Concentrated Liquidity using stochastic processes. Use when analyzing optimal strategies, deriving control policies, or when the user asks "what is the optimal strategy for providing liquidity in this scenario?"
---

## Overview 
All the stochastic processes (midprice, order arrivals, etc.) are discretized for numerical computation and simulation purposes. They share a common time `step_size` which is defined in the environment `AMMEnvironment.py`, it is defined as `step_size = terminal_time / n_steps` and it the finite discretization of the continuous-time infinitesimal dt in the stochastic processes. This allows for consistent updates across all processes at each timestep.

## Core Stochastic Processes

1. **Midprice Process**: The midprice process is modeled as a geometric Brownian motion (GBM) with drift and volatility parameters. The volatility is proportional to the square root of the step size, i.e., `volatility * sqrt(step_size)`, which ensures that the price changes are appropriately scaled for the discrete time steps. 

2. **Arrival Process**: The order arrival process is mainly modeled as a non-homogeneous Poisson process, where the arrival rate can vary over time based on state variables. For small enough `intensity * step_size`, e.g., $\le 0.01$, at each step, we can approximate the Poisson counts with a Bernoulli random variable, with as success probability `p = intensity * step_size`. This means that at each step, there can be at most one buy and one sell arrival, which simplifies the simulation while still capturing the stochastic nature of order arrivals. 

3. **AMM Price Process**: The price impact is fixed and deterministic. Each trade moves the price by one tick, and so the trade size is determined by the liquidity at the current tick. So, the AMM price process is continuous-time jump process. 
