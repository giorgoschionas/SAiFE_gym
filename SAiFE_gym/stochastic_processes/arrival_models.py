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

        # Validate intensity shape
        assert self.intensity.shape == (4, 2), \
            f"intensity must have shape (4, 2) for [a0, a1, a2, a3] x [SELL, BUY], got {self.intensity.shape}"

        # Validate a_0 >= 0 (minimum intensity must be non-negative)
        assert np.all(self.intensity[0, :] >= 0), \
            f"a_0 (minimum intensity) must be non-negative, got {self.intensity[0, :]}"

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

        Raises:
            ValueError: If state is None or has wrong batch size
        """
        # Import indices (avoid circular import)
        from SAiFE_gym.gym.index_names import (
            LIQUIDITY_INDEX, AMM_PRICE_INDEX, ASSET_PRICE_INDEX
        )

        # Validate state
        if state is None:
            raise ValueError("PoissonLinearArrivalModel requires state parameter")
        if state.shape[0] != self.num_trajectories:
            raise ValueError(
                f"State batch size {state.shape[0]} != num_trajectories {self.num_trajectories}"
            )

        # Extract state variables (vectorized across trajectories)
        L = state[:, LIQUIDITY_INDEX]           # (N,)
        Z_sqrt = state[:, AMM_PRICE_INDEX]      # (N,) - SQRT price!
        Z = Z_sqrt ** 2                          # Convert sqrt to regular price
        S = state[:, ASSET_PRICE_INDEX]         # (N,)

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

    

class UnidirectionalPoissonArrivalModel(ArrivalModel):
      """Unidirectional uninformed arrivals for path-dependent models.
      
      Each step has at most ONE order (either buy OR sell, not both).
      Direction is random (50/50) since traders are uninformed.
      """

      def __init__(
          self,
          intensity: float = 140.0,  
          step_size: float = 0.001,
          num_trajectories: int = 1,
          seed: Optional[int] = None,
      ):
          self.intensity = intensity
          super().__init__(
              min_value=np.array([[]]),
              max_value=np.array([[]]),
              step_size=step_size,
              terminal_time=0.0,
              initial_state=np.array([[]]),
              num_trajectories=num_trajectories,
              seed=seed,
          )

      def get_arrivals(self, state: np.ndarray = None) -> np.ndarray:
          """Generate unidirectional arrivals (only one side per step).

          Args:
              state: Optional state (unused, for signature compatibility)

          Returns:
              np.ndarray: Boolean arrivals of shape (num_trajectories, 2) for [SELL, BUY]
          """
          # Step 1: Sample if order arrives
          unif_arrival = self.rng.uniform(size=(self.num_trajectories,))
          order_arrives = unif_arrival < self.intensity * self.step_size

          # Step 2: Randomly choose direction (50/50 for uninformed)
          unif_direction = self.rng.uniform(size=(self.num_trajectories,))
          is_buy = unif_direction < 0.5  # 50% chance buy
          is_sell = ~is_buy              # 50% chance sell

          # Step 3: Combine arrival AND direction (only one side active)
          arrivals = np.zeros((self.num_trajectories, 2))
          arrivals[:, 0] = order_arrives & is_sell  # Sell side
          arrivals[:, 1] = order_arrives & is_buy   # Buy side

          return arrivals

      def update(self, arrivals, fills, actions, state=None):
          pass  
    
class InformedArrivalModel(ArrivalModel):
    """Informed orderflow with directional bias based on the discrepancy between 
    midprice and AMM price.
    Only one side active per arrival (unidirectional).
    """

    def __init__(
        self,
        intensity: float = 14.0,  # 10x less than uninformed by default
        midprice_model: StochasticProcessModel = None,
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = intensity
        self.midprice_model = midprice_model
        self.previous_midprice = None

        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def get_arrivals(self, state: np.ndarray = None) -> np.ndarray:
        """Generate directional arrivals based on midprice movement.

        Args:
            state: Optional state (unused, uses self.midprice_model instead)

        Returns:
            np.ndarray: Boolean arrivals of shape (num_trajectories, 2) for [SELL, BUY]
        """
        # Get current midprice from midprice_model
        if self.midprice_model is None:
            return np.zeros((self.num_trajectories, 2))

        current_midprice = self.midprice_model.current_state[:, 0:1]  # (num_traj, 1)

        # Initialize on first call
        if self.previous_midprice is None:
            self.previous_midprice = current_midprice.copy()
            return np.zeros((self.num_trajectories, 2))

        # Calculate price change
        price_change = current_midprice - self.previous_midprice  # (num_traj, 1)

        # Sample if order arrives (Bernoulli)
        unif = self.rng.uniform(size=(self.num_trajectories,))
        order_arrives = unif < self.intensity * self.step_size  # (num_traj,)

        # Determine direction (contrarian):
        # Buy if price dropped (price_change < 0)
        # Sell if price rose (price_change >= 0)
        buy_signal = (price_change < 0).flatten()  # (num_traj,)
        sell_signal = (price_change >= 0).flatten()  # (num_traj,)

        # Combine: order arrives AND direction (only one side active)
        arrivals = np.zeros((self.num_trajectories, 2))
        arrivals[:, 0] = order_arrives & sell_signal  # Sell side
        arrivals[:, 1] = order_arrives & buy_signal   # Buy side

        # Update for next step
        self.previous_midprice = current_midprice.copy()

        return arrivals

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def reset(self):
        """Reset midprice tracking."""
        super().reset()
        self.previous_midprice = None

# NEED TO BE CHECKED
class MultiOrderflowArrivalModel(ArrivalModel):
    """Combines multiple orderflows with different sampling frequencies.

    Returns 3D array: (num_trajectories, num_orderflows, 2_sides)
    """

    def __init__(
        self,
        uninformed_model: ArrivalModel,
        informed_model: ArrivalModel,
        informed_frequency: int = 10,  # Sample informed every N steps
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.uninformed_model = uninformed_model
        self.informed_model = informed_model
        self.informed_frequency = informed_frequency
        self.step_counter = 0

        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=uninformed_model.step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def get_arrivals(self, state: np.ndarray = None) -> np.ndarray:
        """Get arrivals from both orderflows.

        Args:
            state: Optional environment state to pass to sub-models

        Returns:
            np.ndarray: Shape (num_trajectories, 2, 2)
                        [uninformed, informed] x [sell, buy]
        """
        # Always sample uninformed (pass state to sub-model)
        uninformed = self.uninformed_model.get_arrivals(state)  # (num_traj, 2)

        # Sample informed only every N steps
        if self.step_counter % self.informed_frequency == 0:
            informed = self.informed_model.get_arrivals(state)  # (num_traj, 2)
        else:
            informed = np.zeros((self.num_trajectories, 2))

        self.step_counter += 1

        # Stack into 3D: (num_traj, 2_orderflows, 2_sides)
        return np.stack([uninformed, informed], axis=1)

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        """Update both underlying models."""
        # Extract orderflows from 3D arrivals
        if arrivals.ndim == 3:
            uninformed_arrivals = arrivals[:, 0, :]  # (num_traj, 2)
            informed_arrivals = arrivals[:, 1, :]    # (num_traj, 2)
        else:
            # Fallback for 2D (shouldn't happen, but for safety)
            uninformed_arrivals = arrivals
            informed_arrivals = np.zeros_like(arrivals)

        self.uninformed_model.update(uninformed_arrivals, fills, actions, state)
        self.informed_model.update(informed_arrivals, fills, actions, state)

    def reset(self):
        """Reset both models and step counter."""
        super().reset()
        self.step_counter = 0
        self.uninformed_model.reset()
        self.informed_model.reset()