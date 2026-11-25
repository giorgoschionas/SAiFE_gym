# CARA risk aversion

In the 2026-11-10 sweep, increasing `risk_aversion` shifts the expected-utility
maximizer from half-width 50 to 100 ticks. This is a change in risk-adjusted
preference: the corresponding mean monetary PnL decreases.

## Main sweep

Source: [best widths](cara/best_widths.csv), [complete width/risk grid](cara/reward_grid.csv),
[configuration](cara/config.json), and [per-path arrays](cara/path_results.npz).
Background liquidity is 30,000, sigma 0.03, horizon 1, and starting wealth
1,000 token1. The policy rebalances every 100 of 1,000 steps. Each width has
50 episode batches of 100 trajectories (5,000 paths).

The tested half-widths are
`25, 40, 50, 60, 70, 80, 90, 100, 110, 125, 150, 200, 250`.

| Risk aversion (1/token1) | Best sampled half-width | Mean PnL (token1) | CE PnL (token1) | Bootstrap selection fraction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 50 | 115.854 | 115.854 | 1.0000 |
| 0.003 | 50 | 115.854 | 114.212 | 1.0000 |
| 0.010 | 50 | 115.854 | 110.269 | 0.8620 |
| 0.020 | 60 | 115.089 | 104.920 | 1.0000 |
| 0.030 | 60 | 115.089 | 99.306 | 1.0000 |
| 0.040 | 70 | 112.782 | 93.362 | 0.5325 |
| 0.050 | 70 | 112.782 | 87.609 | 0.8220 |
| 0.075 | 80 | 109.435 | 73.776 | 0.7435 |
| 0.100 | 100 | 101.149 | 62.152 | 0.7845 |

![CARA width selection and PnL distributions](cara/width_selection.png)

## What risk aversion changes

The deterministic policy ignores rewards. Each width's wealth paths are
simulated once, then the production reward is replayed for every risk parameter:

```text
G = terminal wealth - initial wealth
U_a(G) = (1 - exp(-a*G)) / a   for a != 0
U_0(G) = G
score = mean(U_a(G))
CE_a = -log(mean(exp(-a*G))) / a
CE_0 = mean(G)
```

For a fixed `a`, expected utility and CE rank widths identically. Greater
positive `a` emphasizes unfavorable PnL outcomes. In this run, moving from
half-width 50 to 100 reduces PnL standard deviation from 32.948 to 23.298
token1, alongside lower mean PnL. Compare widths at the same `a`; raw utility
levels across different risk preferences are not a common monetary score.

The 2,000-resample bootstrap resamples paired episode batches. Selection
fractions describe how often a width wins these resamples, not the probability
that it is the population optimum. At `a=0.04`, half-width 70 wins only 53.25%
of resamples. At `a=0.1`, the chosen width's utility-weight effective sample
size is about 15.8 out of 5,000, and one path carries about 17.6% of the weight.
Tail sensitivity remains substantial. See [summary.json](cara/summary.json)
and the reward grid for pointwise intervals and concentration diagnostics.
