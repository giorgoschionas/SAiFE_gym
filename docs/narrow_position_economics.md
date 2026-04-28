# Narrow LP Positions: Profitability vs Gas Cost and Arrival Model

This note works through the accounting of a *narrow-band rebalance* strategy in
SAiFE_gym — the LP that, every step, redeploys all its capital into the
narrowest viable range around the current tick. We quantify expected fees per
step, derive the break-even gas cost, then ask how the answer changes when the
arrival process itself depends on liquidity.

All numbers below use the experiment defaults from `experiments/helpers.py`
unless stated otherwise. **Notation:** $S$ denotes the *external* market
midprice (the mark-to-market reference); $P$ denotes the *AMM* price
(`POOL_SQRT_PRICE_KEY ** 2`). At episode start they coincide ($S_0 = P_0 = 100$).

| Parameter | Symbol | Value |
|---|---|---|
| Initial wealth | $W$ | $10^6$ token-1 |
| Initial external midprice | $S_0$ | $100$ |
| Initial AMM price | $P_0$ | $100$ |
| Initial $\sqrt P$ | $\sqrt P_0$ | $10$ |
| Fee tier | $f$ | $0.003$ |
| Tick base | $r$ | $1.0001$ |
| Tick factor | $\delta = \sqrt r - 1$ | $\approx 5\times 10^{-5}$ |
| Step size | $\Delta t$ | $1/200 = 0.005$ |
| Episode length | `N_STEPS` | $200$ |
| Per-side arrival rate | $\lambda$ | $100$ ($\lambda\Delta t = 0.5$) |

---

## 1. Initial wealth and corresponding liquidity

The LP starts each episode with $W = 10^6$ token-1 (the quote asset),
threaded from `experiments/helpers.py` into `AMMEnvironment` and stored
per-trajectory under `INITIAL_WEALTH_KEY`. It is the seed capital used the
first time the LP deploys (`gym/ModelDynamics.py:413-414`).

Inside `update_state()` (`gym/ModelDynamics.py:467-472`):

```python
value_per_L = get_position_value_vec(1, S, sqrt_P, sqrt_P_low, sqrt_P_up)
new_L       = wealth / value_per_L
```

(In the source the second argument is `external_p_current`, i.e. the external
midprice $S$ — not the AMM price.)

For an in-range position (`get_position_value_vec`, `gym/helpers/AMM_utils.py:68`):

$$
v_{\text{per L}} \;=\; \frac{S}{\sqrt P} + \sqrt P - \sqrt P_{\text{low}} - \frac{S}{\sqrt P_{\text{up}}}.
$$

For action `[-d, +d]` symmetric of width $w = 2d$, evaluated at deployment
when $S=P$:

$$
v_{\text{per L}} \;\approx\; w\cdot \delta\cdot \sqrt P \;=\; w\cdot 5\times 10^{-4} \;\;\text{(at } S=P=100\text{).}
$$

A few canonical widths:

| Action | Width $w$ | $v_{\text{per L}}$ | $L = W/v_{\text{per L}}$ |
|---|---:|---:|---:|
| `[0, +1]` (one-sided) | 1 | $5\times 10^{-4}$ | $2\times 10^9$ |
| `[-1, +1]` | 2 | $1\times 10^{-3}$ | $1\times 10^9$ |
| `[-2, +2]` | 4 | $2\times 10^{-3}$ | $5\times 10^8$ |
| `[-5, +5]` (uniform `tau=5`) | 10 | $5\times 10^{-3}$ | $2\times 10^8$ |

The `tau=5` figure is consistent with the `fixed_lp_liquidity = 2e8` baked
into the policy-behavior plotting helpers (`experiments/helpers.py:522, 607`).

**Scaling rule.** Liquidity per unit wealth scales as $1/w$: doubling the
range halves $L$.

---

## 2. Fees per step under the narrow-band strategy

Each arrival moves the AMM price by exactly one tick, so the LP captures the
full trade fee provided the next tick is in range. Using
$\xi_{\text{buy}} = L\cdot\delta\cdot\sqrt P$ from `_process_buy` and
$\xi_{\text{sell}} = L\cdot\delta/\sqrt P$ from `_process_sell`:

$$
\text{fee per arrival} \;\approx\; f\cdot L\cdot \delta\cdot \sqrt P \quad\text{(token-1 valued).}
$$

Sells produce the same value via $f\cdot L\cdot \delta/\sqrt P$ token-0 marked
at $S$. With $L=10^9$ for the centered width-2 position:

$$
\text{fee per arrival} \;=\; 0.003\cdot 10^9\cdot 5\times 10^{-5}\cdot 10 \;=\; \mathbf{1{,}500\;\text{token-1}.}
$$

Bernoulli buy and sell each fire with probability $\lambda\Delta t = 0.5$, so

$$
\mathbb E[\text{fee per step}] \;=\; (\lambda_b+\lambda_s)\Delta t \cdot 1500 \;=\; \mathbf{1{,}500\;\text{token-1}}.
$$

A useful invariant absorbs $L$ into wealth:

$$
\frac{\mathbb E[\text{fee per step}]}{W}
\;=\; \frac{f\cdot \Lambda\Delta t}{w}
\;=\; \frac{0.003\cdot 1}{w}.
$$

So narrowing $w$ scales fee revenue **linearly** with concentration.

**Per-episode totals (200 steps):**

| Strategy | $L$ | $\mathbb E[\text{fee/step}]$ | $\mathbb E[\text{fees/episode}]$ | $\%\,W$ |
|---|---:|---:|---:|---:|
| Width 2 narrow rebalance | $10^9$ | $1{,}500$ | $300{,}000$ | $30\%$ |
| Width 4 | $5\times 10^8$ | $750$ | $150{,}000$ | $15\%$ |
| Width 10 (`tau=5` uniform) | $2\times 10^8$ | $300$ | $60{,}000$ | $6\%$ |

---

## 3. Break-even gas cost

Each rebalance costs `gas_cost`, applied 200 times per episode. Profitability
ignoring impermanent loss:

$$
200\cdot \text{gas\_cost} \;\lesssim\; \mathbb E[\text{fees per episode}]
\;\Longleftrightarrow\;
\text{gas\_cost} \;\lesssim\; \frac{300{,}000}{200} \;=\; 1{,}500\;\text{token-1.}
$$

Plotting against the sweep used in `experiments/gas_cost_impact.py`:

| `gas_cost` | gas drag (200 steps) | net E[ΔW] (width 2) | verdict |
|---:|---:|---:|:---|
| $0$ | $0$ | $+300\text{k}$ | best case |
| $10$ | $2\text{k}$ | $+298\text{k}$ | trivially profitable |
| $100$ | $20\text{k}$ | $+280\text{k}$ | very profitable |
| $500$ | $100\text{k}$ | $+200\text{k}$ | still very profitable |
| $1{,}000$ | $200\text{k}$ | $+100\text{k}$ | $\approx 10\%$ return |
| **$1{,}500$** | **$300\text{k}$** | **$0$** | **break-even** |
| $2{,}000$ | $400\text{k}$ | $-100\text{k}$ | money-losing |
| $2.16$ (class default) | $432$ | $+299.6\text{k}$ | gas is irrelevant |

The class-level default `gas_cost = 1.0815*2 ≈ 2.16` in
`UniswapV3ModelDynamics.__init__` is roughly **three orders of magnitude**
below the threshold that would discipline this behaviour. The
helpers default of `0.0` means standard experiments don't model gas at all;
only the `gas_cost_impact.py` sweep (`[0, 10, 50, 100, 500, 1000]`) actually
exerts pressure, and even `1000` only halves PnL — it doesn't kill the
strategy.

**Comparison vs the uniform `tau=5` baseline.** The uniform agent has
$L\approx 2\times 10^8$, so its per-episode fee ceiling is $\sim 60\text{k}$.
That's why `gas_cost_impact.py` already shows the uniform agent under water at
gas $\approx 100$, while a width-2 narrow rebalancer breaks even only at
$\sim 1{,}500$.

**Caveats glossed over:**
- *Impermanent loss / inventory drift.* Brownian midprice with `vol=2.0` over
  $T=1$ has $\sigma_S \approx 2$ ($\sim 200$ ticks). The narrow LP tracks this
  by rebalancing every step, but each rebalance after a 1-tick move swaps
  $\sim 50\%$ of one token through the external price reference — non-zero
  `swap_fee_rate` would eat into the 1,500/step budget linearly.
- *Toxicity.* With `α₃ > 0`, arbitrageurs preferentially trade *into*
  mispricing, so the LP earns fees while taking the wrong side. The
  $30\%$/episode figure is the `α₃ = 0` benign-flow ceiling.
- *Bernoulli vs Poisson.* With $\lambda\Delta t = 0.5$, the Bernoulli cap of
  one buy and one sell per step is binding; the true Poisson rate would be
  slightly higher.

---

## 4. How concentration incentives change with the arrival model

Two distinct mechanisms in `SAiFE_gym/stochastic_processes/arrival_models.py`
encode "liquidity attracts flow." Their implications for narrow positions are
sharply different.

### 4.1 `PoissonLinearArrivalModel` — symmetric, point-wise

```python
L = state['active_liquidity'] / liquidity_scale
linear_part = α₁ + α₂·L  ±  α₃·(S - P)     # source uses (S - Z); Z ≡ P here
intensity   = max(α₀, linear_part)
```

Reads only the **scalar liquidity at the active tick**, and `α₂` is the same
for buy and sell (`alpha[2] = [50, 50]` by default). So:

- A tighter LP position with the same wealth puts more $L$ on the active
  tick → mechanically attracts more flow on **both** sides.
- The same $L_{\text{active}}$ that drives factor (A) — fee per trade — also
  drives factor (B) — arrival rate. **The two factors are the same lever.**
- Narrowing is rewarded twice over.

Joint expected fee revenue:

$$
\mathbb E[\text{fee/step}] \;\propto\; \big(\alpha_1 + \alpha_2 L/\text{scale}\big)\cdot L
\;=\; \alpha_1 L + \alpha_2 L^2/\text{scale}.
$$

Monotonically decreasing in width $w$ via $L\propto 1/w$. Optimum: **as
narrow as the action space allows**.

Note: the standard experiments set `ALPHA2 = [0, 0]` (`helpers.py:49`), so the
liquidity-feedback channel is **off by default**; only the constant baseline
$\alpha_1$ contributes. The discussion above describes the model with
`α₂ > 0` switched on.

### 4.2 `LiquidityKernelArrivalModel` — directional and spatial

```python
sell_indices = current_tick_idx − [1, 2, …, K]   # ticks LEFT of current
buy_indices  = current_tick_idx + [1, 2, …, K]   # ticks RIGHT of current

weighted_liq_sell = Σ_{d=1..K} exp(−β·d) · L(current − d) / liquidity_scale
weighted_liq_buy  = Σ_{d=1..K} exp(−β·d) · L(current + d) / liquidity_scale
```

Properties that change everything:

- **Direction-conditional.** Buy intensity reads only ticks *above* the
  current one; sell intensity reads only ticks *below*. A trade is attracted
  to depth in the direction it pushes the price.
- **Liquidity at the active tick contributes zero.** The kernel sums over
  $d=1,\dots,K$, so factor (B) is decoupled from $L_{\text{active}}$ — which
  still drives factor (A).
- **Localised by $\beta$.** With $\beta=0.5$:

  | $d$ | 1 | 2 | 3 | 5 | 10 | $\infty$ |
  |---|---:|---:|---:|---:|---:|---:|
  | $w(d) = e^{-\beta d}$ | 0.61 | 0.37 | 0.22 | 0.082 | 0.0067 | 0 |
  | cumulative $\Sigma(d) = \sum_{k=1}^{d} w(k)$ | 0.61 | 0.98 | 1.20 | 1.41 | 1.55 | $\approx 1.54$ |

  ~75% of kernel mass sits in the first 3 ticks; depth past $d\approx 6$ is
  ignored.

### 4.3 The two factors, side by side

| Factor | Linear model | Kernel model |
|---|---|---|
| **(A) Fee per trade** | $f\cdot L_{\text{active}}\cdot\delta\sqrt P$, narrower $\Rightarrow$ bigger | **Identical.** A trade is still local to one tick. |
| **(B) Arrival rate** | $\alpha_1 + \alpha_2 L_{\text{active}}/\text{scale}$ — same scalar as (A); narrower $\Rightarrow$ bigger on both sides | $\lambda_{\text{buy}}$ reads ticks above, $\lambda_{\text{sell}}$ reads ticks below; **liquidity at `cur` contributes zero**; boost requires the position to **straddle `cur` on the trade side**. |

In the linear model, A and B are the same lever; in the kernel model they
**compete**.

### 4.4 Spatial reach as a function of action width

A position `[lower_off, upper_off]` covers tick indices
$\{cur+\text{lower\_off}, \dots, cur+\text{upper\_off}-1\}$. The kernel reaches
$d=1,\dots,K$ ticks past `cur` on each side:

| Action | Reach left | Reach right | Sell boost $\sum w(d)$ | Buy boost $\sum w(d)$ |
|---|:---:|:---:|:---:|:---:|
| `[0, +1]` (width 1) | 0 | 0 | 0 | 0 |
| `[-1, +1]` (width 2, centred) | 1 | 0 | $0.61$ | **0** |
| `[-1, +2]` (width 3) | 1 | 1 | $0.61$ | $0.61$ |
| `[-2, +3]` (width 5) | 2 | 2 | $0.98$ | $0.98$ |
| `[-5, +6]` (width 11) | 5 | 5 | $1.41$ | $1.41$ |

Two consequences:

1. **Minimum width to symmetrically activate factor (B): width 3** (e.g.
   `[-1, +2]`). The "narrowest possible centred" width-2 action only fires
   the sell side and leaves buy intensity at the floor $\alpha_1$.
2. **Returns saturate fast.** Going from width 3 to width 11 multiplies the
   kernel boost by $\sim 2.3$, but $L_{\text{active}}$ is divided by
   $11/3\approx 3.7$. Pushing wider for stronger flow loses fees-per-trade
   roughly twice as fast as it gains kernel mass.

### 4.5 Joint expected fees under the kernel

For symmetric reach $d$, action `[-d, +d+1]`, width $w = 2d+1$, with the
cumulative kernel weight $\Sigma(d) = \sum_{k=1}^{d} e^{-\beta k}$ from §4.2:

$$
\mathbb E[\text{fee/step}]
\;\approx\;
2\Delta t\,\Big(\alpha_1 + \alpha_2\cdot \tfrac{L\,\Sigma(d)}{\text{scale}}\Big)\cdot f\,L\,\delta\sqrt P.
$$

Two terms:

- A *baseline-flow* term $\propto L \propto 1/w$, maximised at the narrowest $w$.
- A *feedback* term $\propto L^2 \Sigma(d) \propto \Sigma(d)/w^2$, maximised at
  the $d$ that maximises $\Sigma(d)/(2d+1)^2$ — for $\beta=0.5$ that is $d=1$
  (width 3).

With the in-file defaults `α₁=100, α₂=50, β=0.5, K=10` and $W=10^6$, the
feedback term blows past $\alpha_1$ for any $d\geq 1$ and the Bernoulli
probability saturates at one trade per side per step. So in practice:

- **Width 1 single-side**: kernel boost is zero, only $\alpha_1$ drives flow,
  and only one direction yields fees.
- **Width 2 centred**: only one side is boosted; the other is at $\alpha_1$.
- **Width 3** (e.g. `[-1, +2]`): both sides Bernoulli-saturated at one trade
  per step; $L_{\text{active}}$ is at its highest value compatible with
  symmetric saturation.
- **Width $> 3$**: Bernoulli still saturated, but $L$ falls — strictly worse.

### 4.6 Bottom line

| | Linear $\Rightarrow$ optimum | Kernel $\Rightarrow$ optimum |
|---|---|---|
| Width preferred | **as narrow as possible** (width 2 centred, or width 1 one-sided) | **width 3 with one tick reach on each side** (e.g. `[-1, +2]`) |
| Why | factors A and B both push toward narrow | factors A and B compete: A wants narrow, B needs spatial reach with $\beta$-decay |
| Asymmetry | none — boost is symmetric in buy/sell | LP can deliberately attract more buys (or sells) by skewing its range to one side |
| Saturation | $L_{\text{active}}$ drives both factors monotonically | factor B saturates after $d\approx 1/\beta$ ticks; further widening hurts factor A without buying much B |

So switching from `PoissonLinearArrivalModel` to `LiquidityKernelArrivalModel`
turns "narrowest is always best" into a **two-step decision**: (i) reach far
enough to wake up the kernel on both sides — width 3 with $\beta=0.5$; (ii)
within that constraint, go as narrow as possible. The kernel also gives the
LP a directional-skew lever the linear model cannot represent.

---

## 5. Implications for the gas-cost question

Tying everything back to gas:

- **Linear model with `α₂ > 0`.** The liquidity feedback only widens the
  break-even gas threshold further: narrower $\Rightarrow$ more $L$
  $\Rightarrow$ more *and* faster trades, all of them at higher per-trade
  fees. The width-2 break-even of $\sim 1{,}500$ token-1 derived in §3 is a
  lower bound — the true number under positive $\alpha_2$ is higher.
- **Linear model with `α₂ = 0`** (current `helpers.py` default). Arrivals are
  exogenous; §3's calculation is essentially exact. Break-even stays at
  $\sim 1{,}500$.
- **Kernel model.** The break-even threshold *for the actually optimal
  width-3 strategy* is comparable in magnitude to width 2 in the linear
  model — fees are roughly $L\cdot$flow, and width-3 has $\tfrac{2}{3}$ of
  the per-trade fee but Bernoulli-saturated flow on both sides, so total
  expected fees are similar. The qualitative picture is unchanged: gas would
  need to reach the $10^3$ token-1-per-rebalance scale to deter narrow
  rebalancing.

In short, with the current default of `gas_cost = 0` (or even $\sim 100$),
**no arrival model in this file deters a narrow-rebalance LP**. To reproduce
the empirical regime where LPs prefer wide passive ranges, either gas must
rise to the order of $10^3$ token-1 per rebalance, or toxicity (`α₃`) must
be turned up to make the *fee* side of the equation work against the LP
rather than for it.
