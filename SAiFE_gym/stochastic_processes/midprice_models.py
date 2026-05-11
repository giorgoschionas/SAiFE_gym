from typing import Optional
from math import sqrt

from SAiFE_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel

import numpy as np


MidpriceModel = StochasticProcessModel

class BrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 2.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.drift * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * np.sqrt(terminal_time)


class GeometricBrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 0.1,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.drift * self.current_state * self.step_size
            + self.volatility
            * self.current_state
            * sqrt(self.step_size)
            * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        stdev = sqrt(
            initial_price**2
            * np.exp(2 * self.drift * terminal_time)
            * (np.exp(self.volatility**2 * terminal_time) - 1)
        )
        return initial_price * np.exp(self.drift * terminal_time) + 4 * stdev


class OrnsteinUhlenbeckMidpriceModel(MidpriceModel):
    """Mean-reverting midprice process: dS = κ(θ - S) dt + σ dW.

    Euler-Maruyama discretization:
        S_{t+Δt} = S_t + κ(θ - S_t) Δt + σ √Δt · Z,   Z ~ N(0, 1)

    The stationary distribution is N(θ, σ² / (2κ)), so the long-run std is
    σ / √(2κ). Bounds use 4 stationary stds around θ.

    Latest testing: Put σ*S_t in the volatility to see behaviour.
    """

    def __init__(
        self,
        mean_reversion: float = 5.0,
        long_term_mean: Optional[float] = None,
        volatility: float = 2.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        assert mean_reversion > 0, f"mean_reversion (κ) must be positive, got {mean_reversion}"
        self.mean_reversion = mean_reversion
        self.long_term_mean = initial_price if long_term_mean is None else long_term_mean
        self.volatility = volatility
        self.terminal_time = terminal_time
        max_val = self._get_max_value(initial_price, terminal_time)
        super().__init__(
            min_value=np.array([[initial_price - (max_val - initial_price)]]),
            max_value=np.array([[max_val]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.mean_reversion
            * (self.long_term_mean - self.current_state)
            * self.step_size
            + self.volatility * self.current_state * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        stationary_std = self.volatility / sqrt(2 * self.mean_reversion)
        return self.long_term_mean + 4 * stationary_std

