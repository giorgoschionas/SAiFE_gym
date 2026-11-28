# What drives LPs toward wider positions?



| Change within a sweep | What happens in these simulations? |
| --- | --- |
| [Less background liquidity](background-liquidity/README.md) | For a fixed initial capital, policies trained in thinner pools deploy wider ranges on average. For a periodic LP, spreading its capital across more ticks gives up less fee income as the pool gets thinner. |
| [Higher midprice volatility](volatility/README.md)| Policies trained in the more volatile markets deploy wider ranges on average. The wider of two periodic ranges spends more time in range and overtakes the narrower one in mean net PnL as volatility rises. |
| [Greater LP risk aversion through exponential utility](risk-aversion/README.md) | A wider periodic range gives up some mean profit for a less variable payoff. An LP that puts more weight on poor outcomes can prefer that trade. |

![PPO policies deploy wider ranges at lower liquidity and higher volatility. The value of widening a fixed range changes from negative to positive as risk aversion increases.](overview.png)

The first two panels show the mean **full width** over the steps on which the LP
has a position, including time spent out of range. A separate policy was trained
for each market level and seed using proximal policy optimization (PPO).
The liquidity result uses one training seed. The thin volatility lines follow
the three training seeds, and the thick line is their mean. The last panel compares
two periodic policies. Above zero, the wider range has a higher certainty-equivalent
PnL, the sure profit the LP would value as highly as the simulated payoff.

## Try it with the simulator

Give two periodic LPs 1,000 token1 each. One deploys 15 ticks on either side of
the pool price (±15, a full width of 30 ticks), the other ±25. A tick here is about
0.01% in price. Both agents recenter every 0.01 time units. Hold background
liquidity at 30,000, move volatility from 0.03 to 0.05 and compare mean net
PnL over the same external price shocks.

| Volatility | Mean PnL at ±15 (token1) | Mean PnL at ±25 (token1) | Paths where widening paid |
| ---: | ---: | ---: | ---: |
| 0.03 | 121.33 | 117.03 | 34 of 100 |
| 0.05 | 30.32 | 36.60 | 72 of 100 |

Both policies earn less in the more volatile market. Within that market, widening
buys more time in range (59% to 72%) and increases mean net PnL by about 6 token1.
[Per-path PnLs](volatility/fixed-width-bg30000/pnl_by_path.npz) ·
[Policy diagnostics](volatility/fixed-width-bg30000/summary.csv)

With the [repository dependencies](../../README.md#installation) installed, run either
comparison from the repository root.

```bash
python experiments/wider-positions-drivers/compare_ranges.py volatility
python experiments/wider-positions-drivers/compare_ranges.py background-liquidity
```

Each command starts from one saved market configuration, changes the named input
and prints time in range, fees, gas and net PnL for the same two ranges. The
example calls the existing rollout, which couples a geometric Brownian midprice
to a Uniswap v3 pool with liquidity-dependent price impact. It steps a periodic
LP through a batch of 100 trajectories at the same evaluation seed. See the
[example](compare_ranges.py), [rollout](../sweep_dense_exponential_width.py)
and [periodic agent](../../SAiFE_gym/agents/BaselineAgents.py).

## What stays fixed?

The sweeps share a horizon of 1, starting wealth of 1,000 token1, gross trade
notional of 250 token1, a baseline arrival-rate parameter of 300 per side, and a gas
cost of 2 per paid rebalance. Compare levels **within a row**. Matching the external
shocks lets us compare LP rules on the same market noise while swaps and pool
prices respond to each position's liquidity.

| Sweep | Input varied | Fixed market setting | LP decisions per horizon | Evaluation and objective |
| --- | --- | --- | --- | --- |
| Liquidity, periodic | Background liquidity | Volatility 0.07 | 100 scheduled deployments | 100 paths per width and level, mean net PnL |
| Liquidity, PPO | Background liquidity | Volatility 0.03 | 10 opportunities to hold or choose both range bounds | 1 training seed, 1,000 evaluation paths per policy, inventory penalty with `phi=0.4` |
| Volatility, periodic | Midprice volatility | Background 30,000 | 100 scheduled deployments | 100 paths per width and level, mean net PnL |
| Volatility, PPO | Midprice volatility | Background 30,000 | 100 scheduled symmetric deployments | 3 training seeds, 1,000 evaluation paths per policy, net PnL |
| Risk preference, periodic | CARA risk aversion | Volatility 0.03, background 30,000 | 10 scheduled deployments | 5,000 paths per width, same payoffs scored at each risk aversion |

All rows use a 100-step decision interval. The 100-deployment rows have 10,000
simulator steps per horizon and the 10-decision rows have 1,000. The two market
grids each recorded one batch of 100 paths per cell. CARA uses 50 batches of 100
and resamples paired batches to assess uncertainty.

## Saved results and rebuilding

Each driver page links its configurations, full width grids and evaluation data.
The [overview CSV](overview.csv) retains the best sampled fixed widths and the
mean deployed PPO widths for quantitative lookup. The [manifest](manifest.json)
records the six source directories, artifact checksums and omitted files
(including model checkpoints). Use the settings saved with each run.

```bash
python experiments/wider-positions-drivers/summarize_results.py
```

This verifies the snapshots and rebuilds the CSV and the PNG/PDF figure. Pass
`--no-plots` to verify and rebuild the CSV using only the standard library.
