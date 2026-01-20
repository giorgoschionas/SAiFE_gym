import abc
from typing import Optional

import numpy as np

from SAiFE_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


class ArrivalModel(StochasticProcessModel):
    """ArrivalModel models the arrival of orders to the AMM. The first entry of arrivals represents an arrival
    of an exogenous SELL order (selling the risky asset) and the second entry represents an arrival of an
    exogenous BUY order (buying the risky asset).
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
    def get_arrivals(self, context: dict = None) -> np.ndarray:
        """Generate arrival events.

        Args:
            context: Optional context dict with pre-computed values for state-dependent models.
                     Keys may include:
                     - 'active_liquidity': Liquidity at current tick, shape (num_trajectories,)
                     - 'amm_price': AMM price (sqrt_price**2), shape (num_trajectories,)
                     - 'midprice': External market midprice, shape (num_trajectories,)
                     Required for state-dependent models like PoissonLinearArrivalModel.

        Returns:
            np.ndarray: Arrival indicators of shape (num_trajectories, 2) for [SELL, BUY]
        """
        pass


class PoissonArrivalModel(ArrivalModel):
    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.array(intensity)
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_arrivals(self, context: dict = None) -> np.ndarray:
        """Generate arrivals using Poisson process (state-independent).

        Args:
            context: Optional context dict (unused, for signature compatibility)

        Returns:
            np.ndarray: Boolean arrivals of shape (num_trajectories, 2) for [SELL, BUY]
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.intensity * self.step_size
    

class PoissonLinearArrivalModel(ArrivalModel):
    """
    State-dependent Poisson arrival model with linear intensity.

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

        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_arrivals(self, context: dict) -> np.ndarray:
        """Generate arrivals with state-dependent linear intensity.

        Formula:
            intensity_sell = max(α₀, α₁ + α₂*L - α₃*(S-Z))
            intensity_buy  = max(α₀, α₁ + α₂*L + α₃*(S-Z))

        Args:
            context: Dict with keys:
                - 'active_liquidity': Liquidity at current tick, shape (num_trajectories,)
                - 'amm_price': AMM price (sqrt_price**2), shape (num_trajectories,)
                - 'midprice': External market midprice, shape (num_trajectories,)

        Returns:
            np.ndarray: Boolean arrivals of shape (num_trajectories, 2) for [SELL, BUY]
        """
        if context is None:
            raise ValueError("PoissonLinearArrivalModel requires context dict with "
                           "'active_liquidity', 'amm_price', and 'midprice' keys")

        # Extract and normalize liquidity
        L = context['active_liquidity'] / self.liquidity_scale  # (N,)
        Z = context['amm_price']                                 # (N,) - AMM price
        S = context['midprice']                                  # (N,) - external midprice

        # Compute mispricing term (S - Z)
        # When S > Z (AMM underpriced): positive → increases BUY, decreases SELL
        # When S < Z (AMM overpriced): negative → increases SELL, decreases BUY
        mispricing = (S - Z)[:, None]  # (N, 1)
        L_expanded = L[:, None]         # (N, 1)

        # Extract coefficients
        a0 = self.alpha[0]  # (2,) - minimum intensity floor
        a1 = self.alpha[1]  # (2,) - baseline intensity
        a2 = self.alpha[2]  # (2,) - liquidity coefficient
        a3 = self.alpha[3]  # (2,) - arbitrage coefficient

        # Sign multiplier: [-1, +1] for [sell, buy]
        # SELL: -a3*(S-Z) -> when S>Z, reduces sell intensity
        # BUY:  +a3*(S-Z) -> when S>Z, increases buy intensity
        sign_multiplier = np.array([-1.0, 1.0])

        # Compute linear part: a1 + a2*L +/- a3*(S-Z)
        linear_part = a1 + a2 * L_expanded + a3 * sign_multiplier * mispricing  # (N, 2)

        # Apply floor at minimum intensity
        intensity = np.maximum(a0, linear_part)  # (N, 2)

        # Sample Bernoulli trials for arrivals
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        arrivals = unif < intensity * self.step_size  # (N, 2)

        return arrivals


