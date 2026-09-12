import abc
from typing import Optional

import numpy as np

from SAiFE_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


def _poisson_at_least_one_probability(intensity: np.ndarray, step_size: float) -> np.ndarray:
    """Return ``P(N >= 1)`` for a Poisson process over one time step.

    The dynamics still consume a boolean "at most one arrival" indicator, but
    using ``1 - exp(-lambda * dt)`` keeps the Bernoulli probability bounded by
    one even when ``lambda * dt`` is not small.
    """
    non_negative_rate = np.maximum(intensity, 0.0)
    return 1.0 - np.exp(-non_negative_rate * step_size)


class ArrivalModel(StochasticProcessModel):
    """ArrivalModel models the arrival of orders to the AMM. The first entry of arrivals represents an arrival
    of an exogenous SELL order (selling the risky asset) and the second entry represents an arrival of an
    exogenous BUY order (buying the risky asset).

    Following the mbt_gym pattern, arrival models OWN their internal state (intensity) which is updated
    via the update() method based on external AMM state.
    """

    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        super().__init__(min_value, max_value, step_size, terminal_time, initial_state, num_trajectories, seed)
        self.set_episode_baseline_intensity(self.initial_vector_state)

    def set_episode_baseline_intensity(self, intensity: np.ndarray) -> None:
        """Set the trajectory-aligned baseline used for the current episode."""
        episode_intensity = np.asarray(intensity, dtype=np.float64)
        expected_shape = (self.num_trajectories, 2)
        if episode_intensity.shape != expected_shape:
            raise ValueError(
                "episode baseline intensity must have shape "
                f"{expected_shape}, got {episode_intensity.shape}"
            )
        if not np.all(np.isfinite(episode_intensity)):
            raise ValueError(
                "episode baseline intensity must contain only finite values"
            )
        if np.any(episode_intensity < 0.0):
            raise ValueError("episode baseline intensity must be non-negative")
        self.episode_baseline_intensity = episode_intensity.copy()

    @abc.abstractmethod
    def get_arrivals(self) -> np.ndarray:
        """Generate boolean arrival indicators using internal state (no arguments).

        Each step returns at most one sell and one buy arrival.

        Returns:
            np.ndarray: Boolean arrival indicators of shape (num_trajectories, 2) for [SELL, BUY]
        """
        pass


class PoissonArrivalModel(ArrivalModel):
    """Poisson arrival model with constant intensity.

    This model owns its internal state (intensity) but it's constant and not affected by update().
    """

    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.asarray(intensity, dtype=float).reshape(-1)
        assert self.intensity.shape == (2,), f"intensity must contain two entries, got {self.intensity.shape}"
        assert np.all(self.intensity >= 0), "intensity must be non-negative"

        # Internal state is just the constant intensity (for interface consistency)
        self.current_state = np.ones((num_trajectories, 2)) * self.intensity

        super().__init__(
            min_value=np.array([[0, 0]]),
            max_value=np.array([[1, 1]]) * self.intensity * 10,
            step_size=step_size,
            terminal_time=0.0,
            initial_state=self.intensity.reshape(1, 2),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray,
               state: dict = None) -> np.ndarray:
        """Update is a no-op for constant intensity model.

        Returns:
            np.ndarray: Current intensity state (unchanged)
        """
        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        """Generate boolean arrival indicators via Poisson ``P(N >= 1)``.

        Each step has at most one sell and one buy arrival.

        Returns:
            np.ndarray: Boolean arrival indicators of shape (num_trajectories, 2) for [SELL, BUY]
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < _poisson_at_least_one_probability(self.current_state, self.step_size)

    def reset(self):
        """Reset internal state to constant intensity."""
        self.current_state = self.episode_baseline_intensity.copy()


class PoissonLinearArrivalModel(ArrivalModel):
    """
    State-dependent Poisson arrival model with linear intensity.

    This model OWNS its internal intensity state which is
    updated via update() based on external AMM state.

    Formula:
        intensity_sell = max(α₀, α₁ + α₂*L - α₃*(S-Z))
        intensity_buy  = max(α₀, α₁ + α₂*L + α₃*(S-Z))

    Where:
        - L = active liquidity at current tick (normalized by liquidity_scale)
        - Z = AMM price (sqrt_price ** 2)
        - S = external market midprice
        - (S - Z) = mispricing term

    Sign convention:
        - When S > Z (AMM underpriced): (S-Z) > 0 → higher BUY intensity, lower SELL intensity
        - When S < Z (AMM overpriced): (S-Z) < 0 → higher SELL intensity, lower BUY intensity

    Parameters:
        alpha: Array of shape (4, 2) for [sell, buy]:
            alpha[0] = α₀ = minimum intensity floor
            alpha[1] = α₁ = baseline intensity
            alpha[2] = α₂ = liquidity coefficient
            alpha[3] = α₃ = arbitrage coefficient
        liquidity_scale: Normalization factor for liquidity (default 1e6)
    """

    def __init__(
            self,
            alpha: np.ndarray = None,
            liquidity_scale: float = 1e6,
            step_size: float = 0.001,
            num_trajectories: int = 1,
            seed: Optional[int] = None,
    ):
        if alpha is None:
            alpha = np.array([
                [10.0, 10.0],    # α₀: minimum intensity floor
                [100.0, 100.0], # α₁: baseline intensity
                [50.0, 50.0],   # α₂: liquidity coefficient
                [5.0, 5.0],     # α₃: arbitrage coefficient
            ])

        self.alpha = np.atleast_2d(alpha)
        self.liquidity_scale = liquidity_scale

        assert self.alpha.shape == (4, 2), f"alpha must have shape (4, 2), got {self.alpha.shape}"
        assert np.all(self.alpha[0] >= 0), "α₀ (floor) must be non-negative"
        assert liquidity_scale > 0, f"liquidity_scale must be positive, got {liquidity_scale}"

        # INTERNAL STATE: Initialize intensity to baseline (α₁)
        self.current_state = np.ones((num_trajectories, 2)) * self.alpha[1]

        super().__init__(
            min_value=np.array([[0, 0]]),
            max_value=np.array([[1, 1]]) * self._get_max_intensity(),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=self.alpha[1].reshape(1, 2),  # baseline as initial
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def _get_max_intensity(self):
        """Compute maximum possible intensity for bounds (similar to HawkesArrivalModel)."""
        return self.alpha[1] * 10

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray,
               state: dict = None) -> np.ndarray:
        """Update internal intensity state based on AMM state.

        Formula:
            intensity_sell = max(α₀, α₁ + α₂*L - α₃*(S-Z))
            intensity_buy  = max(α₀, α₁ + α₂*L + α₃*(S-Z))

        Args:
            arrivals: Not used (for interface compatibility)
            fills: Not used (for interface compatibility)
            actions: Not used (for interface compatibility)
            state: Dict with keys:
                - 'active_liquidity': Liquidity at current tick, shape (num_trajectories,)
                - 'amm_price': AMM price (sqrt_price**2), shape (num_trajectories,)
                - 'midprice': External market midprice, shape (num_trajectories,)

        Returns:
            np.ndarray: Updated internal intensity state
        """
        if state is None:
            return self.current_state

        # Extract and normalize liquidity
        L = np.atleast_1d(state['active_liquidity']).astype(float) / self.liquidity_scale  # (N,)
        Z = np.atleast_1d(state['amm_price']).astype(float)                                 # (N,) - AMM price
        S = np.atleast_1d(state['midprice']).astype(float)                                  # (N,) - external midprice

        # Compute mispricing term (S - Z)
        # When S > Z (AMM underpriced): positive → increases BUY, decreases SELL
        # When S < Z (AMM overpriced): negative → increases SELL, decreases BUY
        mispricing = (S - Z)[:, None]  # (N, 1)
        L_expanded = L[:, None]         # (N, 1)

        # Sign multiplier: [-1, +1] for [sell, buy]
        # SELL: -a3*(S-Z) -> when S>Z, reduces sell intensity
        # BUY:  +a3*(S-Z) -> when S>Z, increases buy intensity
        sign_multiplier = np.array([-1.0, 1.0])

        # Compute linear part: a1 + a2*L +/- a3*(S-Z)
        linear_part = (self.episode_baseline_intensity + self.alpha[2] * L_expanded
                       + self.alpha[3] * sign_multiplier * mispricing)  # (N, 2)

        # Apply floor at minimum intensity
        self.current_state = np.maximum(self.alpha[0], linear_part)  # (N, 2)

        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        """Generate boolean arrival indicators via Poisson ``P(N >= 1)``.

        Each step has at most one sell and one buy arrival.

        Returns:
            np.ndarray: Boolean arrival indicators of shape (num_trajectories, 2) for [SELL, BUY]
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < _poisson_at_least_one_probability(self.current_state, self.step_size)

    def reset(self):
        """Reset internal state to baseline intensity (α₁)."""
        self.current_state = self.episode_baseline_intensity.copy()


class LiquidityKernelArrivalModel(ArrivalModel):
    """
    Arrival model where intensity depends on nearby directional liquidity
    with exponential decay.

    Buy intensity depends on liquidity in K executable intervals at/above the
    current tick. Sell intensity depends on liquidity in K executable intervals
    below the current tick. Closer intervals are weighted more heavily via the
    exponential kernel: w(d) = exp(-beta * d), where d=0 is the immediately
    executable interval.

    More nearby liquidity in the trade direction -> higher arrival intensity
    (thick markets attract volume / less slippage).

    Formula:
        weighted_liq_sell = sum_{d=0}^{K-1} exp(-beta*d) * L(current_tick - 1 - d)
        weighted_liq_buy  = sum_{d=0}^{K-1} exp(-beta*d) * L(current_tick + d)

        intensity_sell = max(alpha_0, alpha_1 + alpha_2 * weighted_liq_sell / liq_scale + alpha_3 * max(Z-S, 0))
        intensity_buy  = max(alpha_0, alpha_1 + alpha_2 * weighted_liq_buy  / liq_scale + alpha_3 * max(S-Z, 0))

    The alpha_3 (arbitrage) term is one-sided: arbs only fire on the side that
    profits from the gap. When S > Z (AMM underpriced) only buy intensity is
    boosted; when S < Z (AMM overpriced) only sell intensity is boosted. The
    disadvantaged side keeps its noise baseline (alpha_1 + alpha_2 * L) — noise
    traders don't disappear because of mispricing, they just aren't amplified
    by it.

    Parameters:
        alpha: Array of shape (4, 2) for [sell, buy]:
            alpha[0] = minimum intensity floor
            alpha[1] = baseline intensity
            alpha[2] = directional liquidity kernel coefficient
            alpha[3] = mispricing (arbitrage) coefficient
        beta: Exponential decay rate for the kernel (default 0.5)
        K: Number of executable intervals in each kernel window (default 10)
        liquidity_scale: Normalization factor for weighted liquidity (default 1e6)
    """

    def __init__(
        self,
        alpha: np.ndarray = None,
        beta: float = 0.5,
        K: int = 10,
        liquidity_scale: float = 1e6,
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        if alpha is None:
            alpha = np.array([
                [10.0, 10.0],    # alpha_0: minimum intensity floor
                [100.0, 100.0],  # alpha_1: baseline intensity
                [50.0, 50.0],    # alpha_2: directional liquidity kernel coefficient
                [5.0, 5.0],      # alpha_3: mispricing coefficient
            ])

        self.alpha = np.atleast_2d(alpha)
        self.liquidity_scale = liquidity_scale
        self.beta = beta
        self.K = K

        assert self.alpha.shape == (4, 2), f"alpha must have shape (4, 2), got {self.alpha.shape}"
        assert np.all(self.alpha[0] >= 0), "alpha_0 (floor) must be non-negative"
        assert beta > 0, f"beta must be positive, got {beta}"
        assert K >= 1, f"K must be >= 1, got {K}"
        assert liquidity_scale > 0, f"liquidity_scale must be positive, got {liquidity_scale}"

        # Pre-compute kernel weights for the K executable intervals in each
        # direction. Distance 0 is the immediately executable interval:
        # sells use L[i-1], buys use L[i].
        self.kernel_weights = np.exp(-beta * np.arange(K))  # shape (K,)

        # INTERNAL STATE: Initialize intensity to baseline (alpha_1)
        self.current_state = np.ones((num_trajectories, 2)) * self.alpha[1]

        super().__init__(
            min_value=np.array([[0, 0]]),
            max_value=np.array([[1, 1]]) * self.alpha[1] * 10,
            step_size=step_size,
            terminal_time=0.0,
            initial_state=self.alpha[1].reshape(1, 2),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray,
               state: dict = None) -> np.ndarray:
        """Update internal intensity state based on directional kernel-weighted liquidity.

        Args:
            arrivals: Not used (for interface compatibility)
            fills: Not used (for interface compatibility)
            actions: Not used (for interface compatibility)
            state: Dict with keys:
                - 'liquidity_array': Full liquidity per tick, shape (num_trajectories, num_ticks)
                - 'current_tick': Absolute current tick, shape (num_trajectories,)
                - 'tick_lower_global': Scalar int, converts array index to absolute tick
                - 'amm_price': AMM price (sqrt_price**2), shape (num_trajectories,)
                - 'midprice': External market midprice, shape (num_trajectories,)

        Returns:
            np.ndarray: Updated internal intensity state, shape (num_trajectories, 2)
        """
        if state is None:
            return self.current_state

        liquidity_array = np.asarray(state['liquidity_array'], dtype=float)  # (N, num_ticks)
        if liquidity_array.ndim == 1:
            liquidity_array = liquidity_array.reshape(1, -1)
        current_tick = np.atleast_1d(state['current_tick'])  # (N,)
        tick_lower_global = state['tick_lower_global']   # scalar
        Z = np.atleast_1d(state['amm_price']).astype(float)  # (N,)
        S = np.atleast_1d(state['midprice']).astype(float)   # (N,)

        num_ticks = liquidity_array.shape[1]
        current_tick_idx = (current_tick - tick_lower_global).astype(np.int64)  # (N,)

        # Build index arrays for K executable intervals in each direction.
        # At tick i, sells use L[i-1] first and buys use L[i] first.
        offsets = np.arange(self.K)  # (K,)
        sell_indices = current_tick_idx[:, None] - 1 - offsets[None, :]  # (N, K) -- left
        buy_indices = current_tick_idx[:, None] + offsets[None, :]       # (N, K) -- right

        # Validity masks (in-bounds check)
        sell_valid = (sell_indices >= 0) & (sell_indices < num_ticks)
        buy_valid = (buy_indices >= 0) & (buy_indices < num_ticks)

        # Clip for safe indexing, then zero out invalid positions
        sell_indices_safe = np.clip(sell_indices, 0, num_ticks - 1)
        buy_indices_safe = np.clip(buy_indices, 0, num_ticks - 1)

        traj_idx = np.arange(self.num_trajectories)[:, None]  # (N, 1)
        sell_liq = liquidity_array[traj_idx, sell_indices_safe] * sell_valid  # (N, K)
        buy_liq = liquidity_array[traj_idx, buy_indices_safe] * buy_valid    # (N, K)

        # Kernel-weighted sum: (N, K) @ (K,) -> (N,)
        weighted_liq_sell = sell_liq @ self.kernel_weights / self.liquidity_scale
        weighted_liq_buy = buy_liq @ self.kernel_weights / self.liquidity_scale

        # Stack directional liquidity: (N, 2)
        weighted_liq = np.stack([weighted_liq_sell, weighted_liq_buy], axis=1)

        # One-sided arbitrage term: arbs only fire on the side that profits
        # from the gap. S > Z (AMM underpriced) -> arb buys, no arb sells.
        # S < Z (AMM overpriced) -> arb sells, no arb buys. The disadvantaged
        # side keeps its noise baseline rather than being suppressed below it.
        gap = S - Z                                            # (N,)
        arb_buy = self.alpha[3, 1] * np.maximum(gap, 0.0)      # (N,)
        arb_sell = self.alpha[3, 0] * np.maximum(-gap, 0.0)    # (N,)
        arb = np.stack([arb_sell, arb_buy], axis=1)            # (N, 2)

        # Linear intensity: alpha_1 + alpha_2 * weighted_liq + arb
        linear_part = (
            self.episode_baseline_intensity
            + self.alpha[2] * weighted_liq
            + arb
        )  # (N, 2)

        # Apply floor
        self.current_state = np.maximum(self.alpha[0], linear_part)  # (N, 2)

        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        """Generate boolean arrival indicators via the exact Poisson P(>=1 arrival).

        Uses 1 - exp(-lambda * dt) instead of the linear Bernoulli approximation
        so probabilities remain valid (and bounded by 1) even when lambda*dt is
        not small. Still truncates to at most one arrival per step per side.

        Returns:
            np.ndarray: Boolean arrival indicators of shape (num_trajectories, 2) for [SELL, BUY]
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < _poisson_at_least_one_probability(self.current_state, self.step_size)

    def reset(self):
        """Reset internal state to baseline intensity (alpha_1)."""
        self.current_state = self.episode_baseline_intensity.copy()


class PoissonNonLinearArrivalModel(ArrivalModel):

    def __init__(
            self,
            alpha: np.ndarray = None,
            liquidity_scale: float = 1e6,
            step_size: float = 0.001,
            num_trajectories: int = 1,
            seed: Optional[int] = None,
    ):
        if alpha is None:
            alpha = np.array([
                [10.0, 10.0],    # α₀: minimum intensity floor
                [100.0, 100.0], # α₁: baseline intensity
                [50.0, 50.0],   # α₂: liquidity coefficient
                [5.0, 5.0],     # α₃: arbitrage coefficient
            ])

        self.alpha = np.atleast_2d(alpha)
        self.liquidity_scale = liquidity_scale

        assert self.alpha.shape == (4, 2), f"alpha must have shape (4, 2), got {self.alpha.shape}"
        assert np.all(self.alpha[0] >= 0), "α₀ (floor) must be non-negative"
        assert liquidity_scale > 0, f"liquidity_scale must be positive, got {liquidity_scale}"

        # INTERNAL STATE: Initialize intensity to baseline (α₁)
        self.current_state = np.ones((num_trajectories, 2)) * self.alpha[1]

        super().__init__(
            min_value=np.array([[0, 0]]),
            max_value=np.array([[1, 1]]) * self._get_max_intensity(),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=self.alpha[1].reshape(1, 2),  # baseline as initial
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def _get_max_intensity(self):
        """Compute maximum possible intensity for bounds (similar to HawkesArrivalModel)."""
        return self.alpha[1] * 10

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray,
               state: dict = None) -> np.ndarray:
        """Update internal intensity state based on AMM state.

        Formula:
            intensity_sell = max(α₀, α₁ + α₂*L - α₃*(S-Z))
            intensity_buy  = max(α₀, α₁ + α₂*L + α₃*(S-Z))

        Args:
            arrivals: Not used (for interface compatibility)
            fills: Not used (for interface compatibility)
            actions: Not used (for interface compatibility)
            state: Dict with keys:
                - 'active_liquidity': Liquidity at current tick, shape (num_trajectories,)
                - 'amm_price': AMM price (sqrt_price**2), shape (num_trajectories,)
                - 'midprice': External market midprice, shape (num_trajectories,)

        Returns:
            np.ndarray: Updated internal intensity state
        """
        if state is None:
            return self.current_state

        # Extract and normalize liquidity
        L = np.atleast_1d(state['active_liquidity']).astype(float) / self.liquidity_scale  # (N,)
        Z = np.atleast_1d(state['amm_price']).astype(float)                                 # (N,) - AMM price
        S = np.atleast_1d(state['midprice']).astype(float)                                  # (N,) - external midprice

        # Compute mispricing term (S - Z)
        # When S > Z (AMM underpriced): positive → increases BUY, decreases SELL
        # When S < Z (AMM overpriced): negative → increases SELL, decreases BUY
        mispricing = (S - Z)[:, None]  # (N, 1)
        L_expanded = L[:, None]         # (N, 1)

        # Sign multiplier: [-1, +1] for [sell, buy]
        # SELL: -a3*(S-Z) -> when S>Z, reduces sell intensity
        # BUY:  +a3*(S-Z) -> when S>Z, increases buy intensity
        sign_multiplier = np.array([-1.0, 1.0])

        # Compute linear part: a1 + a2*L +/- a3*(S-Z)
        linear_part = (self.episode_baseline_intensity + self.alpha[2] * L_expanded
                       + self.alpha[3] * sign_multiplier * mispricing)  # (N, 2)

        # Apply floor at minimum intensity
        self.current_state = np.maximum(self.alpha[0], linear_part)  # (N, 2)

        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        """Generate boolean arrival indicators via Poisson ``P(N >= 1)``.

        Each step has at most one sell and one buy arrival.

        Returns:
            np.ndarray: Boolean arrival indicators of shape (num_trajectories, 2) for [SELL, BUY]
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < _poisson_at_least_one_probability(self.current_state, self.step_size)

    def reset(self):
        """Reset internal state to baseline intensity (α₁)."""
        self.current_state = self.episode_baseline_intensity.copy()
