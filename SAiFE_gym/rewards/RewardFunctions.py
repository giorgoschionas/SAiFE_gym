import abc
from typing import Union

import numpy as np


class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> Union[float, np.ndarray]:
        pass

    @abc.abstractmethod
    def reset(self, initial_state: np.ndarray):
        pass

class ImpermanentLoss(RewardFunction):
    def calculate(self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False):
        pass
    def reset(self, initial_state: np.ndarray):
        pass

class LVR(RewardFunction):
    def calculate(self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False):
        pass
    def reset(self, initial_state: np.ndarray):
        pass