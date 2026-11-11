# Midprice volatility

Higher volatility shifts the best sampled width upward at background liquidity
30,000. The separate 100,000-background sweep shows that this effect depends on
the liquidity regime.

## Fixed-width PnL evidence at background 30,000

Source: [summary.csv](fixed-width-bg30000/summary.csv),
[full results](fixed-width-bg30000/summary.json), and
[per-path PnL](fixed-width-bg30000/pnl_by_path.npz). Each cell has 100 paths,
10,000 steps, and rebalancing every 100 steps. Widths share external paths at a
given sigma; the underlying random shocks are matched across sigmas.

| Sigma | Best sampled half-width (ticks) | Mean net PnL (token1) |
| ---: | ---: | ---: |
| 0.030 | 15 | 121.332 |
| 0.035 | 20 | 100.106 |
| 0.040 | 20 | 79.143 |
| 0.045 | 25 | 57.527 |
| 0.050 | 25 | 36.596 |

The maximizing width increases while the maximum mean PnL falls. Higher
volatility favors wider positions relative to narrower ones in this regime;
it does not increase their absolute profitability. All sampled half-widths
`2, 5, 10, ..., 50` have positive mean PnL in these five scenarios.

![PnL versus width and volatility at background 30,000](fixed-width-bg30000/pnl_width_volatility.png)

## Higher-background comparison

At background liquidity 100,000, the best sampled half-width is **2 ticks at
every sigma** in `0.03, 0.04, 0.05, 0.06, 0.07`. Its mean PnLs are respectively
`35.317, -16.344, -65.458, -112.562, -157.168` token1. This sweep does not show
the widening pattern. See [the grid](fixed-width-bg100000/summary.csv) and
[original plot](fixed-width-bg100000/pnl_width_volatility.png).

Older configuration files omit `background_liquidity`; this run used the
historical implicit value 100,000. The snapshots preserve that omission.

## Learned-policy evidence

Source: [width summary](ppo/summary/width_by_volatility.csv),
[per-seed results](ppo/summary/per_seed.csv), and
[configuration](ppo/summary/summary.json). PPO chooses symmetric half-widths
and recenters every decision; it cannot hold. Background liquidity is 30,000,
`inventory_phi=0`, and there are three training seeds (43, 44, 45), each
evaluated on 1,000 shared evaluation paths.

| Sigma | Mean deployed full width (ticks) | Width SD across training seeds | Mean PnL (token1) |
| ---: | ---: | ---: | ---: |
| 0.030 | 41.639 | 7.632 | 114.577 |
| 0.035 | 44.518 | 5.313 | 95.376 |
| 0.040 | 45.099 | 3.052 | 75.200 |
| 0.045 | 49.003 | 4.846 | 53.994 |
| 0.050 | 52.970 | 4.603 | 34.481 |

The [periodic comparison](ppo/periodic_comparison/comparison.csv) evaluates
the selected fixed half-widths `15, 20, 20, 25, 25` on the PPO evaluation seeds.
Those baselines outperform PPO's mean PnL in all five scenarios, reinforcing
the distinction between a learned average width and a profit-maximizing width.

Greater price dispersion between rebalances is a plausible reason for wider
coverage to help. The comparison across background regimes shows that the
width trade-off also depends on liquidity and endogenous trading behavior.
