import abc
from typing import Optional

import numpy as np

from SAiFE_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


class PriceImpactModel(StochasticProcessModel):
    """PriceImpactModel models the price impact of orders in the order book."""

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
    def get_impact(self, action: np.ndarray) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def max_speed(self) -> float:
        pass