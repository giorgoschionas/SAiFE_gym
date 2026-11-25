# Background liquidity

How does the LP quote when other providers supply less liquidity per tick?
Two sweeps change background liquidity while holding their respective volatility,
LP capital, trade-size rule and decision cadence fixed.

## What the trained LPs do

With less background liquidity, the PPO policies generally spread their capital
more widely. The mean deployed position spans about 78 ticks at background
30,000 and 94 ticks at 10,000. There is a small reversal between 30,000 and
25,000. Each point comes from a separately trained policy using seed 42 and
1,000 evaluation paths. The single seed limits what we can infer about variation
between training runs. [Policy widths](ppo/width_by_background_liquidity.csv) ·
[Evaluation details](ppo/per_seed.csv)

These agents face volatility 0.03 and decide every 0.1 time units. They can hold
or redeploy, choose the two ends of the range, and incur a running inventory
penalty with `phi=0.4`. A periodic LP matched to each policy's average width
provides a comparison of fees, gas and rebalance counts.
[Training settings](ppo/sweep_config.json) · [Matched-width comparison](periodic-comparison/comparison.csv)

## What makes extra coverage worthwhile?

Fix the rule as well. Deploy the same capital at ±15 or ±25 ticks, recenter both
every 0.01 time units, and compare them over 100 matched external price paths
at volatility 0.07. Here a positive change means the wider policy earned more
per path. Gas spend is 198 token1 for either range.

| Background liquidity per tick | Mean ticks crossed per trade at ±15 | Change in fee income from widening (token1) | Change in net PnL from widening (token1) |
| ---: | ---: | ---: | ---: |
| 100,000 | 4.0 | −32.20 | −11.37 |
| 75,000 | 5.1 | −29.78 | −9.78 |
| 50,000 | 7.1 | −22.75 | −3.33 |
| 30,000 | 10.7 | −10.20 | +8.68 |
| 20,000 | 14.9 | −0.88 | +17.45 |

A trade of the same gross size moves farther through a shallower pool. The
background also affects the LP's share of the fees in each interval it covers.
At 20,000 the wider position gives up less than one token1 of fee income on
average, and the net PnL comparison favors extra coverage. Both effects change
together in this sweep. At that level, the ±15 and ±25 policies still have
negative mean PnL. [Full policy diagnostics](fixed-width/summary.csv) ·
[Individual path outcomes](fixed-width/pnl_by_path.npz)

![Mean net PnL across the sampled ranges at five background liquidity levels](fixed-width/pnl_width_background.png)

The pattern extends across the full grid of 11 half-widths from 2 to 50 ticks.
At background 100,000, every move to the next wider range reduces mean PnL.
At 20,000, every such move increases it, right up to the widest range tested.
[Configurations and all sampled widths](fixed-width/summary.json)
