import abc
from typing import Optional

import numpy as np

from SAiFE_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel
from SAiFE_gym.gym.index_names import (POOL_SQRT_PRICE_KEY, ASSET_PRICE_KEY, POOL_LIQUIDITY_ARRAY_KEY)


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
    def get_arrivals(self, state: np.ndarray = None) -> np.ndarray:
        """Generate arrival events.

        Args:
            state: Optional environment state of shape (num_trajectories, state_dim).
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

    def get_arrivals(self, state: np.ndarray = None) -> np.ndarray:
        """Generate arrivals using Poisson process (state-independent).

        Args:
            state: Optional state (unused, for signature compatibility)

        Returns:
            np.ndarray: Boolean arrivals of shape (num_trajectories, 2) for [SELL, BUY]
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.intensity * self.step_size
    

class PoissonLinearArrivalModel(ArrivalModel):
    def __init__(
            self,
            # applying the linear model with intensities a0, a1, a2, a3 - For now, I put arbitrary values
            intensity: np.ndarray = np.array([[140.0, 140.0], [130, 130], [120, 120], [110, 110]]),
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

    def get_arrivals(self, state: np.ndarray) -> np.ndarray:
        """Generate arrivals with state-dependent linear intensity.

        Formula: a = max(a_0, a_1 + a_2*L + a_3*(Z-S))
        Applied separately for SELL and BUY sides.

        Args:
            state: Environment state of shape (num_trajectories, state_dim).
                   Must contain LIQUIDITY_INDEX, AMM_PRICE_INDEX, ASSET_PRICE_INDEX.

        Returns:
            np.ndarray: Boolean arrivals of shape (num_trajectories, 2) for [SELL, BUY]
        """

        L = state[:, POOL_LIQUIDITY_ARRAY_KEY]           # (N,)
        Z_sqrt = state[:, POOL_SQRT_PRICE_KEY]      # (N,) - SQRT price!
        Z = Z_sqrt ** 2                          # Convert sqrt to regular price
        S = state[:, ASSET_PRICE_KEY]         # (N,)

        # Extract intensity parameters: shape (4, 2) for [SELL, BUY]
        a_0 = self.intensity[0, :]  # (2,) - minimum intensity
        a_1 = self.intensity[1, :]  # (2,) - base intensity
        a_2 = self.intensity[2, :]  # (2,) - liquidity coefficient
        a_3 = self.intensity[3, :]  # (2,) - price discrepancy coefficient

        # Compute intensity: broadcast (N,) × (2,) -> (N, 2)
        price_discrepancy = (Z - S)[:, None]  # (N, 1)
        linear_part = a_1 + a_2 * L[:, None] + a_3 * price_discrepancy  # (N, 2)

        # Apply floor at minimum intensity
        intensity_computed = np.maximum(a_0, linear_part)  # (N, 2)

        # Sample Bernoulli trials for arrivals
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        arrivals = unif < intensity_computed * self.step_size  # (N, 2)

        return arrivals 

    
