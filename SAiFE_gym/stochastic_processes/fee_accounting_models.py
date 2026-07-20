import abc

import numpy as np

from SAiFE_gym.stochastic_processes.price_impact_models import SwapResult


class FeeAccountingModel(metaclass=abc.ABCMeta):
    """Base class for swap fee accounting rules."""

    @abc.abstractmethod
    def apply_fees(self, state: dict, swap_result: SwapResult, fee_multiplier: float) -> None:
        pass


class UniswapV3FeeAccounting(FeeAccountingModel):
    """Apply Uniswap V3 swap fees to the pool fee arrays."""

    def apply_fees(self, state: dict, swap_result: SwapResult, fee_multiplier: float) -> None:
        if swap_result is None:
            return

        state[swap_result.fee_key][
            swap_result.trajectories,
            swap_result.fee_indices,
        ] += fee_multiplier * swap_result.amounts
