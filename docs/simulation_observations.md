# Simulation Observations — Baseline Agent Benchmarks

All simulations use:
- `BrownianMotionMidpriceModel` (drift=0, volatility=2.0, initial price=100)
- `PoissonLinearArrivalModel` (α₀=10, α₁=100, α₂=0, α₃=variable)
- `UniswapV3ModelDynamics` (fee_tier=0.003, exponential_value=1.0001)
- `PnL` reward (mark-to-market, initial wealth = 1,000,000)
- N=200 steps, T=1.0, 200 trajectories per configuration

Scripts: `notebooks/baseline_agent_simulations.py`,
`notebooks/uniform_tau_comparison.py`

---

## Experiment 1 — Baseline Agent Comparison (RandomAgent vs UniformAllocationAgent)

**Sweep:** τ ∈ {2, 5, 10, 20}, α₃ ∈ {0, 500, 2000, 5000, 10000}

### Observations

**O1.1 — Non-monotonic relationship between α₃ and LP returns (unexpected).**
The hypothesis that LP returns decrease monotonically with orderflow toxicity (α₃) is
not supported. Returns follow a U-shape:

| α₃ | Uniform ΔW (τ=2) |
|----|-----------------|
| 0 | +152k |
| 500 | −81k |
| 2000 | +32k |
| 5000 | +115k |
| 10000 | +170k |

At high α₃, the enormous trading volume generates fee income that **outweighs** the
impermanent loss (IL) from adverse selection.

**O1.2 — Two competing effects explain the U-shape.**
- **Volume effect** (positive for LP): total arrival intensity ≈ α₁ + α₃·|S−Z|.
  At high α₃, even small mispricing produces hundreds of arrivals per step → large fee income.
- **Adverse selection / IL** (negative for LP): bounded by price variance σ²dt (fixed by
  volatility, independent of α₃).

At moderate α₃ (~500), IL slightly dominates; at high α₃ (~5000+), volume dominates overwhelmingly.

**O1.3 — Per-step rebalancing neutralises long-run adverse selection.**
The LP resets its position every step, preventing IL from accumulating across steps.
True adverse selection requires sustained directional drift that the LP cannot hedge;
here, each step neutralises that effect.

**O1.4 — Fee-per-trade greatly exceeds IL-per-trade.**
With `fee_tier = 0.3%` and tick spacing `≈ 0.01%`, the fee earned per individual trade is
~30× the impermanent loss from a one-tick price move. This means more trades (even
adversarial ones) are net-positive for the LP.
> Implication: to observe monotonically decreasing LP returns with α₃, you would need
> a fixed (non-rebalancing) LP position, a much smaller fee tier, or significantly larger
> per-trade price moves.

**O1.5 — UniformAllocationAgent consistently earns more than RandomAgent.**
Uniform covers 2τ+1 ticks with full capital concentration; Random covers a random sub-range
on average narrower. The LP's fee share at each tick is proportional to their liquidity,
so wider systematic coverage captures more total fee income.

**O1.6 — One-step fee lag (structural accounting issue).**
Fees from step t's swaps are stored in `FEES0/1_KEY` and only collected at step t+1's
rebalancing. The terminal step's swap fees are therefore never collected. Impact:
~1/N ≈ 0.5% understatement of total fee income. Small but systematic.

---

## Experiment 2 — Uniform LP: Effect of τ (zero rebalancing cost)

**Setup:** UniformAllocationAgent, τ ∈ {2, 5, 10, 20, 50, 100, 200, 500, 1000},
α₃ ∈ {0, 500, 5000}, `rebalance_cost_coeff = 0`, `num_ticks = 2000`.
τ = 1000 = `num_ticks // 2` covers the full liquidity array → **"Uniswap V2 equivalent"**.

### Observations

**O2.1 — Mean PnL decreases monotonically with τ, approximately as 1/τ.**
Example (α₃=0):

| τ | Mean ΔW |
|---|---------|
| 2 | +152k |
| 10 | +31k |
| 50 | +6.6k |
| 200 | +2.2k |
| 1000 (V2) | +1.0k |

The LP deploys fixed wealth W over 2τ+1 ticks, giving liquidity L ∝ W/τ. Their fee share
at each tick scales with L, so fee income ∝ 1/τ.

**O2.2 — Standard deviation also decreases with τ and converges to a floor.**
For τ ≥ 500, std ≈ 9,500 regardless of α₃. This is the **irreducible price risk**: the
LP's mark-to-market position value fluctuates with the external price regardless of how
wide the range is. The extra variance at small τ comes from the LP frequently going in
and out of range within a step.

**O2.3 — Return/Risk (Sharpe-like ratio) is highest at small τ.**
With α₃=5000, ΔW/Std = 3.68 at τ=2, falling to 0.13 at τ=1000. Concentrated liquidity
is superior on a risk-adjusted basis when there is sufficient volume.

**O2.4 — At the V2 equivalent (τ=1000), toxicity becomes irrelevant.**
Mean ΔW at τ=1000 is approximately +1k regardless of α₃ (range: +307 to +1223). In a
full-range pool, the fee-per-tick is so small that both toxicity and concentration effects
wash out. The LP earns a small near-constant carry regardless of market conditions.

**O2.5 — The α₃=500 break-even τ: sign flip from negative to positive.**
For α₃=500 (moderate toxicity), LP returns are negative at narrow τ (adverse selection >
concentrated fee income) but cross zero somewhere between τ=500 and τ=1000. This crossover
τ is where adverse selection and fee income balance.

**O2.6 — τ=50 marks the intra-step coverage threshold.**
`max_arrivals_per_step = 50` caps any single step at 50 tick moves. For τ ≥ 50, the LP
is guaranteed to stay in range during any individual step; for τ < 50, the LP can go
out of range mid-step and miss fee income.

---

## Experiment 3 — Impact of Rebalancing Costs

**Setup:** same as Experiment 2, plus `rebalance_cost_coeff` ∈ {0, 0.0001, 0.001}.
First-order expected total drag: `N · r · W`:

| r | Total drag | % of wealth |
|---|-----------|-------------|
| 0.0001 | ~20,000 | 2% |
| 0.001 | ~200,000 | 20% |

### Observations

**O3.1 — Rebalancing cost drag is approximately τ-independent.**
The observed drag (PnL with cost − PnL with cost=0) is nearly constant across all τ values
for each cost level:

| r | Theoretical −N·r·W | Observed drag range |
|---|-------------------|-------------------|
| 0.0001 | −20,000 | −18k to −23k |
| 0.001 | −200,000 | −165k to −208k |

The cost is proportional to wealth (~1M throughout), which is approximately τ-independent.
The fee income, however, falls as 1/τ. This asymmetry creates a hard break-even threshold.

**O3.2 — Break-even τ shifts dramatically with cost rate.**

| r | Break-even τ (α₃=0) | Break-even τ (α₃=5000) |
|---|---------------------|------------------------|
| 0.0000 | ∞ (all τ profitable) | ∞ |
| 0.0001 | ~15 | ~22 |
| 0.0010 | <2 (nothing survives) | <2 |

At r=0.001, even the most concentrated strategy (τ=2, zero-cost ΔW=+152k) becomes
ΔW=−56k after the ~200k drag. Only sub-tick (τ<2) positions could survive, which the
model does not support.

**O3.3 — V2-equivalent LP is immediately unprofitable under any realistic cost.**
At τ=1000: zero-cost ΔW ≈ +1k, but r=0.0001 gives ΔW ≈ −19k. The LP earns ~5 units
of fee per step but pays ~100 units of cost (at r=0.0001). V2-style positions should
not rebalance every step; rebalancing should be triggered only when the position is
significantly out of range.

**O3.4 — Cost increases ΔW/Std negatively without reducing risk proportionally.**
Costs are deterministic (applied to wealth each step), so they shift the mean strongly
but reduce std only slightly (by clipping the high-return tail). This makes the
risk-adjusted return worsen rapidly once costs exceed fee income.

**O3.5 — α₃ does not materially affect the break-even τ.**
Across α₃ ∈ {0, 500, 5000}, the break-even τ under r=0.0001 stays in the range 15–22.
The volume boost from high α₃ raises fee income at small τ, which shifts break-even
slightly upward, but the effect is small relative to the ~1/τ fee scaling.

---

## Summary of Identified Model Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Non-monotonic α₃ effect | High | Fee income scales with volume (∝ α₃·mispricing) while IL is bounded by σ²dt; at high α₃, volume dominates → LP earns more, not less |
| Terminal fee lag | Low | Fees from the final step's swaps are never collected; ~0.5% understatement of total fee income |
| Fee/tick ratio | Medium | fee_tier (0.3%) ≫ tick_size (0.01%) makes every individual trade net-positive for the LP; adverse selection only manifests at sustained directional moves, which per-step rebalancing eliminates |

---

## Implications for RL Agent Design

1. **Rebalancing frequency matters as much as range choice.** With realistic costs (r ≥ 0.0001),
   only τ ≤ 15–20 remains profitable. An RL agent must jointly optimise range width and
   rebalancing triggers.

2. **Concentrated liquidity dominates on a risk-adjusted basis** (higher Sharpe-like ratio)
   at all α₃ values tested, but at the price of higher absolute variance.

3. **The α₃=500 regime is the hardest environment.** Neither narrow nor wide ranges are
   clearly dominant here (negative returns at small τ, near-zero at large τ). An RL agent
   that adapts range width to market conditions may find value in this regime.

4. **Toxicity (α₃) affects the optimal τ only weakly under rebalancing costs.** An RL agent
   targeting a specific cost regime should primarily adapt to τ, not to α₃.
