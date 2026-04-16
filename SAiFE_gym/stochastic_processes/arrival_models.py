import abc
from typing import Optional

import numpy as np

from SAiFE_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


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

    @abc.abstractmethod
    def get_arrivals(self) -> np.ndarray:
        """Generate boolean arrival indicators using internal state (no arguments).

        Each step is a Bernoulli trial: at most one sell and one buy per step.

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
        self.intensity = np.array(intensity)
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
        """Generate boolean arrival indicators via Bernoulli trials (uses internal state).

        Each step has at most one sell and one buy arrival.

        Returns:
            np.ndarray: Boolean arrival indicators of shape (num_trajectories, 2) for [SELL, BUY]
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.intensity * self.step_size

    def reset(self):
        """Reset internal state to constant intensity."""
        self.current_state = np.ones((self.num_trajectories, 2)) * self.intensity
    

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
        L = state['active_liquidity'] / self.liquidity_scale  # (N,)
        Z = state['amm_price']                                 # (N,) - AMM price
        S = state['midprice']                                  # (N,) - external midprice

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
        linear_part = (self.alpha[1] + self.alpha[2] * L_expanded
                       + self.alpha[3] * sign_multiplier * mispricing)  # (N, 2)

        # Apply floor at minimum intensity
        self.current_state = np.maximum(self.alpha[0], linear_part)  # (N, 2)

        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        """Generate boolean arrival indicators via Bernoulli trials (uses internal state).

        Each step has at most one sell and one buy arrival.

        Returns:
            np.ndarray: Boolean arrival indicators of shape (num_trajectories, 2) for [SELL, BUY]
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < np.maximum(self.current_state * self.step_size, 0.0)

    def reset(self):
        """Reset internal state to baseline intensity (α₁)."""
        self.current_state = np.ones((self.num_trajectories, 2)) * self.alpha[1]


class LiquidityKernelArrivalModel(ArrivalModel):
    """
    Arrival model where intensity depends on nearby directional liquidity
    with exponential decay.

    Buy intensity depends on liquidity in K ticks to the RIGHT (above current price).
    Sell intensity depends on liquidity in K ticks to the LEFT (below current price).
    Closer ticks are weighted more heavily via exponential kernel: w(d) = exp(-beta * d).

    More nearby liquidity in the trade direction -> higher arrival intensity
    (thick markets attract volume / less slippage).

    Formula:
        weighted_liq_sell = sum_{d=1}^{K} exp(-beta*d) * L(current_tick - d)
        weighted_liq_buy  = sum_{d=1}^{K} exp(-beta*d) * L(current_tick + d)

        intensity_sell = max(alpha_0, alpha_1 + alpha_2 * weighted_liq_sell / liq_scale - alpha_3 * (S-Z))
        intensity_buy  = max(alpha_0, alpha_1 + alpha_2 * weighted_liq_buy  / liq_scale + alpha_3 * (S-Z))

    Parameters:
        alpha: Array of shape (4, 2) for [sell, buy]:
            alpha[0] = minimum intensity floor
            alpha[1] = baseline intensity
            alpha[2] = directional liquidity kernel coefficient
            alpha[3] = mispricing (arbitrage) coefficient
        beta: Exponential decay rate for the kernel (default 0.5)
        K: Number of neighboring ticks in the kernel window (default 10)
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

        # Pre-compute kernel weights: w[d] = exp(-beta * d) for d = 1, ..., K
        self.kernel_weights = np.exp(-beta * np.arange(1, K + 1))  # shape (K,)

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

        liquidity_array = state['liquidity_array']      # (N, num_ticks)
        current_tick = state['current_tick']             # (N,)
        tick_lower_global = state['tick_lower_global']   # scalar
        Z = state['amm_price']                           # (N,)
        S = state['midprice']                            # (N,)

        num_ticks = liquidity_array.shape[1]
        current_tick_idx = (current_tick - tick_lower_global).astype(np.int64)  # (N,)

        # Build index arrays for K neighbors in each direction: (N, K)
        offsets = np.arange(1, self.K + 1)  # (K,)
        sell_indices = current_tick_idx[:, None] - offsets[None, :]  # (N, K) -- left
        buy_indices = current_tick_idx[:, None] + offsets[None, :]   # (N, K) -- right

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

        # Mispricing term
        mispricing = (S - Z)[:, None]       # (N, 1)
        sign_multiplier = np.array([-1.0, 1.0])

        # Linear intensity: alpha_1 + alpha_2 * weighted_liq +/- alpha_3 * (S - Z)
        linear_part = (self.alpha[1]
                       + self.alpha[2] * weighted_liq
                       + self.alpha[3] * sign_multiplier * mispricing)  # (N, 2)

        # Apply floor
        self.current_state = np.maximum(self.alpha[0], linear_part)  # (N, 2)

        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        """Generate boolean arrival indicators via Bernoulli trials (uses internal state).

        Returns:
            np.ndarray: Boolean arrival indicators of shape (num_trajectories, 2) for [SELL, BUY]
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < np.maximum(self.current_state * self.step_size, 0.0)

    def reset(self):
        """Reset internal state to baseline intensity (alpha_1)."""
        self.current_state = np.ones((self.num_trajectories, 2)) * self.alpha[1]


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
        L = state['active_liquidity'] / self.liquidity_scale  # (N,)
        Z = state['amm_price']                                 # (N,) - AMM price
        S = state['midprice']                                  # (N,) - external midprice

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
        linear_part = (self.alpha[1] + self.alpha[2] * L_expanded
                       + self.alpha[3] * sign_multiplier * mispricing)  # (N, 2)

        # Apply floor at minimum intensity
        self.current_state = np.maximum(self.alpha[0], linear_part)  # (N, 2)

        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        """Generate boolean arrival indicators via Bernoulli trials (uses internal state).

        Each step has at most one sell and one buy arrival.

        Returns:
            np.ndarray: Boolean arrival indicators of shape (num_trajectories, 2) for [SELL, BUY]
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < 1 - np.exp(-np.maximum(self.current_state * self.step_size, 0.0))

    def reset(self):
        """Reset internal state to baseline intensity (α₁)."""
        self.current_state = np.ones((self.num_trajectories, 2)) * self.alpha[1]



