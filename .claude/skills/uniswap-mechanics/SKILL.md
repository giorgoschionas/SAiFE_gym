---
name: uniswap-mechanics
description: Explain this skill, when working with Uniswap v3 mechanics, including how the price changes when a new order arrives. 
---

*Price Impact*: When a new order arrives in Uniswap v3, it can cause a change in the price of the assets being traded. This is because Uniswap v3 uses a constant product formula to determine the price of assets in the pool. The formula is:

1. **Swap within a single tick**: 
***Sell order***
If the new order is within the current tick, the price will change according to the constant product formula. Consider a tick range of `[i, i+1]`, where `L[i]` is the liquidity at that tick range. Consider a sell order of `dx` that moves the price from tick `i+1` to tick `i`. The size of the order `dx` can be calculated using the formula:

```
dx = L[i] * (1/sqrt(P[i]) - 1/sqrt(P[i+1]))
```

Where `P[i]` is the price at tick `i` and `P[i+1]` is the price at tick `i+1`. This formula is derived from the constant product formula and the way liquidity is distributed across ticks in Uniswap v3. 

***Buy order***
If the new order is a buy order of `dy` that moves the price from tick `i` to tick `i+1`, the size of the order `dy` can be calculated using the formula: 

```
dy = L[i] * (sqrt(P[i+1]) - sqrt(P[i]))
```

2. **Swap across two geometric midpoints**:
Consider that currently the price sits at the geometric midpoint of tick `i` and tick `i+1`, which means that `sqrt(P) = (sqrt(P[i]) * sqrt(P[i+1]))`. Suppose that a new order arrives and moves the price to the neighbouring geometric midpoint:
***Sell order***
