import gymnasium
import numpy as np
from amm_sim.agents.Agent import Agent
from amm_sim.env.AMMEnvironment import AMMEnvironment

from amm_sim.env.ModelDynamics import UniswapV3ModelDynamics
from amm_sim.env.index_names import (
    POOL_CURRENT_TICK_KEY,
    POOL_SQRT_PRICE_KEY,
    ASSET_PRICE_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_EVER_DEPLOYED_KEY,
    TIME_KEY,
    FEES0_KEY,
    FEES1_KEY
)

class DoNothingAgent(Agent):
    """
    HODL-token0 baseline: holds the initial wealth as the risky asset
    (token0) for the whole episode, so portfolio value marks to market with
    the external price:

        wealth_t  ≈  initial_wealth · (price_t / price_0)

    Mechanism: on the first step, deploys a one-tick range at the top of
    the action space (``[tau-1, tau]``) — i.e. just above current price —
    then emits ``hold_flag = +1`` forever. As long as pool price stays
    below the position's lower bound, the LP sits at 100% token0 with no
    swaps and no fees, so ``get_position_value_vec`` evaluates to
    ``W · external_price_t / external_price_0`` and the env's `PnL` reward
    produces the HODL-token0 path automatically.

    Caveat: if pool price ever crosses into the deployed range, the LP
    starts earning fees and converting to token1, so MTM diverges from
    pure HODL. This stays safe when expected price moves are smaller than
    ``1.0001^(tau-1) - 1`` (≈ 0.5% with default TAU=50).

    Set ``hold_cash=True`` for the alternative cash baseline: never deploy,
    portfolio value stays at ``initial_wealth`` (token1 is the numeraire
    so its value doesn't move), cumulative PnL ≡ 0.
    """
    def __init__(self, env: AMMEnvironment, hold_cash: bool = False):
        self.env = env
        self.hold_cash = hold_cash
        tau = env.model_dynamics.tau
        # Tightest "above current price" range available — minimises the
        # chance of price crossing into the range during the episode.
        self.lower_offset = -1#tau - 1
        self.upper_offset = 1#tau

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        if self.hold_cash:
            # Never deploy → portfolio_value = initial_wealth (constant).
            lower = np.zeros(n, dtype=np.float32)
            upper = np.ones(n, dtype=np.float32)
            hold_flag = np.ones(n, dtype=np.float32)
        else:
            # Deploy once far above current price → 100% token0 → MTM with price.
            lower = np.full(n, self.lower_offset, dtype=np.float32)
            upper = np.full(n, self.upper_offset, dtype=np.float32)
            ever_deployed = state[LP_EVER_DEPLOYED_KEY]
            hold_flag = np.where(ever_deployed, 1.0, -1.0).astype(np.float32)
        return np.column_stack([lower, upper, hold_flag])


class UniformAllocationAgent(Agent):
    """
    Allocates capital across the full active tick range around the current price.

    Action format: [lower_offset, upper_offset, hold_flag] = [-tau, +tau, -1.0]
    This covers 2*tau+1 ticks centered on the current price. Always rebalances.
    """
    def __init__(self, env: AMMEnvironment):
        self.env = env
        self.tau = env.model_dynamics.tau


    def get_action(self, state: dict) -> np.ndarray:

        #action = np.array([[-self.tau, self.tau]])
        action = np.array([[-0.4, 0.4]])
        return np.repeat(action, self.env.num_trajectories, axis=0)

class DeployOnceAgent(Agent):
    """
    Deploys liquidity once at a fixed tick range and holds for the entire episode.

    On the first step (LP not yet deployed), emits hold_flag = -1 to trigger
    deployment at [lower_offset, upper_offset]. On all subsequent steps, emits
    hold_flag = +1 to hold the existing position without rebalancing.

    Defaults to the symmetric full active range [-tau, +tau]. Pass
    `lower_offset` / `upper_offset` to quote asymmetrically, e.g. [-3, +7].
    """
    def __init__(self, env: AMMEnvironment,
                 lower_offset: int = None, upper_offset: int = None):
        self.env = env
        tau = env.model_dynamics.tau
        self.lower_offset = -tau if lower_offset is None else int(lower_offset)
        self.upper_offset = tau if upper_offset is None else int(upper_offset)
        assert -tau <= self.lower_offset < self.upper_offset <= tau, (
            f"need -tau <= lower_offset < upper_offset <= tau, got "
            f"[{self.lower_offset}, {self.upper_offset}] with tau={tau}"
        )

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        lower = np.full(n, self.lower_offset, dtype=np.float32)
        upper = np.full(n, self.upper_offset, dtype=np.float32)
        ever_deployed = state[LP_EVER_DEPLOYED_KEY]
        hold_flag = np.where(ever_deployed, 1.0, -1.0).astype(np.float32)
        return np.column_stack([lower, upper, hold_flag])


class ArrivalRebalanceAgent(Agent):
    """
    Quotes a fixed tick range around the current price; rebalances after
    every ``rebalance_every`` liquidity-taking arrivals (sell or buy).

    Range:
      - Symmetric (default): ``width=W`` → ``[-W, +W]``.
      - Asymmetric: pass ``lower_offset`` and ``upper_offset`` to quote
        e.g. ``[-3, +7]``. When both are provided they take precedence
        over ``width``.

    Reads the arrivals that produced the *current* state directly from
    ``env.model_dynamics.last_arrivals`` (cached after each get_arrivals
    call), so the count is exact — no fee-delta or |Δtick| approximation.
    """
    def __init__(self, env: AMMEnvironment, rebalance_every: int = 10,
                 width: int = 2,
                 lower_offset: int = None, upper_offset: int = None):
        assert rebalance_every >= 1, f"rebalance_every must be >= 1, got {rebalance_every}"
        tau = env.model_dynamics.tau

        if lower_offset is not None or upper_offset is not None:
            assert lower_offset is not None and upper_offset is not None, (
                "pass both lower_offset and upper_offset, or neither"
            )
            self.lower_offset = int(lower_offset)
            self.upper_offset = int(upper_offset)
        else:
            assert 1 <= width <= tau, f"width must be in [1, tau={tau}], got {width}"
            self.lower_offset = -int(width)
            self.upper_offset = int(width)

        assert -tau <= self.lower_offset < self.upper_offset <= tau, (
            f"need -tau <= lower_offset < upper_offset <= tau, got "
            f"[{self.lower_offset}, {self.upper_offset}] with tau={tau}"
        )

        self.env = env
        self.rebalance_every = rebalance_every
        self.arrival_count = np.zeros(env.num_trajectories, dtype=np.int64)

    def get_action(self, state: dict) -> np.ndarray:
        n = self.env.num_trajectories
        is_episode_start = state[TIME_KEY][0] < self.env.step_size / 2

        if is_episode_start:
            self.arrival_count = np.zeros(n, dtype=np.int64)
            rebalance = np.ones(n, dtype=bool)  # initial deploy
        else:
            new_arrivals = self.env.model_dynamics.last_arrivals.sum(axis=1).astype(np.int64)
            self.arrival_count += new_arrivals
            rebalance = self.arrival_count >= self.rebalance_every
            self.arrival_count = np.where(rebalance, 0, self.arrival_count)

        lower = np.full(n, self.lower_offset, dtype=np.float32)
        upper = np.full(n, self.upper_offset, dtype=np.float32)
        hold_flag = np.where(rebalance, -1.0, 1.0).astype(np.float32)
        return np.column_stack([lower, upper, hold_flag])


class CarteaPLAgent(Agent):
    """
    Cartea-Drissi-Monga Optimal Liquidity Provision Agent.

    Implements the optimal position strategy from Cartea et al. research,
    using dynamic fee rates and stochastic drift/volatility parameters
    to compute optimal position boundaries.

    Key formulas:
    - δₜˡ* = (2γ + μₜ²σ²)/(8πₜ - σ² + 2μₜ(μₜ - σ²/2)) - μₜ
    - δₜᵘ* = (2γ + μₜ²σ²)/(8πₜ - σ² + 2μₜ(μₜ - σ²/2)) + μₜ
    """

    def __init__(self, env: AMMEnvironment, gamma: float = 0.005,
                 rebalance_tolerance: int = 1, seed: int = None):
        """
        Initialize Cartea agent.

        Args:
            env: amm_sim AMM environment
            gamma: Risk aversion parameter (default 0.005)
            rebalance_tolerance: Tick-deadband for the deploy-mode rebalance
                gate. The agent only re-quotes when the optimal absolute range
                drifts more than `rebalance_tolerance` ticks from the currently
                deployed range. `tol=0` recovers the academic "always rebalance"
                policy (gas drain). `tol=1` (default) is a 1-tick deadband.
            seed: Random seed for reproducibility
        """
        self.env = env
        self.gamma = gamma
        self.rebalance_tolerance = int(rebalance_tolerance)
        assert self.rebalance_tolerance >= 0, "rebalance_tolerance must be >= 0"
        self.tau = env.model_dynamics.tau
        self.num_trajectories = env.num_trajectories
        self.rng = np.random.default_rng(seed)

        # Cache model dynamics for parameter access
        self.model_dynamics = env.model_dynamics

        # Cache midprice model parameters
        self.drift = self.model_dynamics.midprice_model.drift
        self.volatility = self.model_dynamics.midprice_model.volatility
        self.sigma_squared = self.volatility ** 2

        # Per-trajectory state: True if the agent is currently holding the
        # synthetic-withdrawal stance (deployed at the bottom of the action
        # space → price above range → 100% token1). Used to decide whether to
        # re-pay gas on consecutive negative-denominator steps. Reset on
        # episode start (detected in get_action via TIME_KEY).
        self.in_withdrawal = np.zeros(self.num_trajectories, dtype=bool)

    def calculate_dynamic_fee_rate(self, state: dict) -> np.ndarray:
        """
        Calculate dynamic pool fee rate π_t as cumulative fees / pool size.

        π_t = (total_fee_income) / (2 × κ × √price)

        Returns:
            π_t values for each trajectory, shape (num_trajectories,)
        """
        # Get active liquidity (κ) and current price
        _, active_liquidity = self.model_dynamics._get_current_tick_liquidity()
        sqrt_price = state[POOL_SQRT_PRICE_KEY]

        # Pool size: 2 × κ × √price
        pool_size = 2 * active_liquidity * sqrt_price + 1e-8  # Small epsilon to avoid division by zero

        # Total cumulative fees across all ticks
        total_fees_0 = np.sum(state[FEES0_KEY], axis=1)  # Shape: (num_trajectories,)
        total_fees_1 = np.sum(state[FEES1_KEY], axis=1)  # Shape: (num_trajectories,)

        # Convert both fee streams to token1 (numéraire) units so they share
        # units with pool_size = 2·κ·√P (which is also in token1). Without this
        # π_t is not dimensionless and the closed-form δ formula gets the wrong
        # scale. fee0 is in token0 → multiply by price to convert to token1.
        price = sqrt_price ** 2
        total_fee_income = total_fees_0 * price + total_fees_1

        # Convert the cumulative dimensionless yield to a per-unit-time rate by
        # dividing by elapsed simulation time. This matches the Cartea-Drissi-
        # Monga paper, where π_t is a windowed yield-per-time quantity (1/day in
        # the paper's ETH/USDC numerical example) comparable to σ² (variance per
        # time) in the closed-form δ formula. Without the divide, π_t grows
        # monotonically with episode time and the regime threshold drifts.
        # Lifetime average converges to the windowed paper quantity in a
        # stationary fee regime, which our LiquidityKernelArrivalModel produces.
        elapsed = np.maximum(state[TIME_KEY], self.env.step_size)
        pi_t = total_fee_income / pool_size / elapsed

        return pi_t

    def compute_optimal_deltas(self, state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute optimal boundary controls δ^ℓ* and δ^u* using Cartea formulas.

        Returns:
            (delta_lower, delta_upper, raw_delta_lower, raw_delta_upper, raw_denominator):
                - delta_lower, delta_upper: clipped to the paper's valid ranges
                  (0.0001, 2.0] and [0.0, 1.9999); used for the actual action.
                - raw_delta_lower, raw_delta_upper: un-clipped values; caller
                  uses these for the paper's profitability check
                  (δ^ℓ ∈ (0, 2], δ^u ∈ [0, 2)).
                - raw_denominator: the un-guarded denominator (sign-strict).
        """
        # Get dynamic fee rate
        pi_t = self.calculate_dynamic_fee_rate(state)

        # Extract parameters
        mu = self.drift
        sigma_sq = self.sigma_squared
        gamma = self.gamma

        # Compute common terms
        numerator = 2 * gamma + mu**2 * sigma_sq
        raw_denominator = 8 * pi_t - sigma_sq + 2 * mu * (mu - sigma_sq/2)

        # Handle edge case: very small or negative denominator
        # This can happen when fees are very low or drift/volatility parameters are extreme.
        # Note: this guard intentionally flips sign for small negatives — only safe to use
        # for the division below; sign-aware callers must use raw_denominator.
        denominator = np.where(np.abs(raw_denominator) < 1e-8, 1e-8, raw_denominator)

        # Compute base spread term
        base_term = numerator / denominator

        # Un-clipped deltas — used downstream for the profitability check.
        raw_delta_lower = base_term - mu
        raw_delta_upper = base_term + mu

        # Clipped deltas — what we actually deploy when profitable.
        delta_lower = np.clip(raw_delta_lower, 0.0001, 2.0)
        delta_upper = np.clip(raw_delta_upper, 0.0, 1.9999)

        return delta_lower, delta_upper, raw_delta_lower, raw_delta_upper, raw_denominator

    def deltas_to_tick_offsets(self, delta_lower: np.ndarray, delta_upper: np.ndarray,
                               sqrt_price: np.ndarray, state: dict) -> np.ndarray:
        """
        Convert continuous δ controls to discrete tick offsets.

        Academic formulas give δ in continuous price space, but amm_sim
        requires integer tick offsets relative to current tick.

        Args:
            delta_lower: δₗ* values, shape (num_trajectories,)
            delta_upper: δᵤ* values, shape (num_trajectories,)
            sqrt_price: Current pool √price, shape (num_trajectories,)

        Returns:
            actions: [lower_offset, upper_offset], shape (num_trajectories, 2)
        """
        # Convert δ to actual price boundaries using Cartea formulas (equation 6)
        # (Zₜˡ)^(1/2) = √price * (1 - δₗ/2)
        # (Zₜᵘ)^(1/2) = √price / (1 - δᵤ/2)

        sqrt_price_lower = sqrt_price * (1 - delta_lower/2)
        sqrt_price_upper = sqrt_price / (1 - delta_upper/2)

        # Convert √price to actual prices
        price = sqrt_price ** 2
        price_lower = sqrt_price_lower ** 2
        price_upper = sqrt_price_upper ** 2

        # Convert prices to ticks using amm_sim's exponential spacing
        exponential_value = self.model_dynamics.exponential_value
        current_tick = state[POOL_CURRENT_TICK_KEY]

        tick_lower = np.log(price_lower) / np.log(exponential_value)
        tick_upper = np.log(price_upper) / np.log(exponential_value)

        # Compute offsets relative to current tick
        lower_offset = tick_lower - current_tick
        upper_offset = tick_upper - current_tick

        # Round to integers (tick offsets must be whole numbers)
        lower_offset = np.round(lower_offset)
        upper_offset = np.round(upper_offset)

        # Ensure offsets are within tau bounds and satisfy lower < upper
        lower_offset = np.clip(lower_offset, -self.tau, self.tau - 1)
        upper_offset = np.clip(upper_offset, -self.tau + 1, self.tau)

        # Ensure minimum width of 1 tick
        invalid = lower_offset >= upper_offset
        upper_offset = np.where(invalid, lower_offset + 1, upper_offset)
        upper_offset = np.clip(upper_offset, -self.tau + 1, self.tau)

        # Stack into action format
        actions = np.column_stack([lower_offset, upper_offset])
        return actions.astype(np.float32)

    def get_action(self, state: dict) -> np.ndarray:
        """
        Compute optimal liquidity provision action using Cartea strategy.

        Decision rule (per trajectory):
          - profitable, ¬in_withdrawal, optimal range close to current  → hold
              (tick-deadband: small drifts absorbed without gas)
          - profitable, ¬in_withdrawal, optimal range drifted  → rebalance to Cartea
          - profitable, in_withdrawal  → rebalance to Cartea (exit withdrawal)
          - ¬profitable, ¬in_withdrawal → rebalance to withdrawal stance
                (deploy at [-τ, -τ+1] → above-range → 100% token1 numéraire)
          - ¬profitable, in_withdrawal  → hold (already withdrawn; no gas)

        "Profitable" uses the paper's full criterion (Section 3, eq. 25-26):
            δ^ℓ ∈ (0, 2], δ^u ∈ [0, 2), and δ^ℓ·δ^u/2 < δ^ℓ + δ^u
        computed from the UN-clipped deltas. This naturally subsumes the
        denominator-sign check and, for μ=0, the π > σ²/8 + γ/8 inequality.

        Returns:
            action: [lower_offset, upper_offset, hold_flag] for each trajectory,
                    shape (num_trajectories, 3)
        """
        n = self.num_trajectories

        # Reset internal state on the first step of a new episode.
        is_episode_start = state[TIME_KEY][0] < self.env.step_size / 2
        if is_episode_start:
            self.in_withdrawal = np.zeros(n, dtype=bool)

        # Cartea deltas: both clipped (for deployment) and un-clipped (for the
        # profitability test).
        (delta_lower, delta_upper,
         raw_dl, raw_du, _raw_denominator) = self.compute_optimal_deltas(state)

        # Paper's full profitability criterion. The geometric constraint
        # (δ^ℓ·δ^u/2 < δ^ℓ + δ^u) is implied by the range constraints when both
        # δ are in their valid ranges, but we include it for paper completeness.
        profitable = (
            (raw_dl > 0) & (raw_dl <= 2) &
            (raw_du >= 0) & (raw_du < 2) &
            (raw_dl * raw_du / 2 < raw_dl + raw_du)
        )  # shape (n,)

        # Cartea optimal range in tick-offset form.
        sqrt_price = state[POOL_SQRT_PRICE_KEY]
        cartea_offsets = self.deltas_to_tick_offsets(
            delta_lower, delta_upper, sqrt_price, state,
        )  # shape (n, 2)

        # Tick-deadband: compare the optimal ABSOLUTE range to the currently
        # deployed absolute range. Only rebalance in the profitable branch when
        # the drift exceeds `rebalance_tolerance` on either bound. tol=0
        # reproduces the academic "rebalance every step" policy.
        current_tick = state[POOL_CURRENT_TICK_KEY].astype(np.int64)
        cartea_abs_lower = current_tick + cartea_offsets[:, 0].astype(np.int64)
        cartea_abs_upper = current_tick + cartea_offsets[:, 1].astype(np.int64)
        lp_abs_lower = state[LP_TICK_LOWER_KEY].astype(np.int64)
        lp_abs_upper = state[LP_TICK_UPPER_KEY].astype(np.int64)
        drift = np.maximum(
            np.abs(cartea_abs_lower - lp_abs_lower),
            np.abs(cartea_abs_upper - lp_abs_upper),
        )
        needs_recenter = drift > self.rebalance_tolerance

        # Synthetic-withdrawal offsets: deploy as far below the current tick as
        # the action space allows → "price above range" → LP holds 100% token1.
        withdrawal_lower = np.full(n, -self.tau, dtype=np.float32)
        withdrawal_upper = np.full(n, -self.tau + 1, dtype=np.float32)

        # Per-trajectory branching (see docstring above for the full table).
        # Rebalance iff: (profitable AND (just left withdrawal OR drifted out of
        # deadband)) OR (not profitable AND not yet withdrawn).
        rebalancing = (
            (profitable & (self.in_withdrawal | needs_recenter)) |
            (~profitable & ~self.in_withdrawal)
        )

        lower = np.where(profitable, cartea_offsets[:, 0], withdrawal_lower)
        upper = np.where(profitable, cartea_offsets[:, 1], withdrawal_upper)
        hold_flag = np.where(rebalancing, -1.0, 1.0).astype(np.float32)

        # Persist state for the next call: a trajectory is "in withdrawal" iff
        # the closed-form profitability test currently fails.
        self.in_withdrawal = ~profitable

        return np.column_stack([lower, upper, hold_flag]).astype(np.float32)


