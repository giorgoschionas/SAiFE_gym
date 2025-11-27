import abc
from typing import Union

import numpy as np
from SAiFE_gym.gym.index_names import LIQUIDITY_INDEX, ASSET_PRICE_INDEX    


class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> Union[float, np.ndarray]:
        pass

    @abc.abstractmethod
    def reset(self, initial_state: np.ndarray):
        pass

class ImpermanentLossV2(RewardFunction):
    
    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        assert len(current_state.shape) > 1, "Reward functions must be calculated on state matrices."
        
        current_price = current_state[:, ASSET_PRICE_INDEX]
        next_price = next_state[:, ASSET_PRICE_INDEX]

        impermanent_loss = 2 * np.sqrt((next_price / current_price)) / (1 + (next_price / current_price)) - 1
        return impermanent_loss

    def reset(self, initial_state: np.ndarray):
        pass

class PnL(RewardFunction):
    """A simple profit and loss reward function of the 'mark-to-market' value of the agent's portfolio."""

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        assert len(current_state.shape) > 1, "Reward functions must be calculated on state matrices."
        current_market_value = (
            2*current_state[:, LIQUIDITY_INDEX] * np.sqrt(current_state[:, ASSET_PRICE_INDEX])
        )
        next_market_value = (
            2*next_state[:, LIQUIDITY_INDEX] * np.sqrt(next_state[:, ASSET_PRICE_INDEX])
        )
        return next_market_value - current_market_value

    def reset(self, initial_state: np.ndarray):
        pass

# Impermanent Loss in Uniswap v3
class ImpermanentLossV3(RewardFunction):
    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        pass

    def reset(self, initial_state: np.ndarray):
        pass

