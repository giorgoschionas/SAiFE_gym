
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

class ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model = None,
        arrival_model = None,
        fill_probability_model = None,
        price_impact_model = None,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model
        self.num_trajectories = num_trajectories
        self.rng = default_rng(seed)
        self.seed_ = seed
        self.fill_multiplier = self._get_fill_multiplier()
        self.round_initial_inventory = False
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.state = None

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass