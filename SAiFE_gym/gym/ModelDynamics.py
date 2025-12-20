
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import (
    LIQUIDITY_INDEX, AMM_PRICE_INDEX, FEES_TOKEN_A_INDEX, FEES_TOKEN_B_INDEX, TIME_INDEX
)

from SAiFE_gym.gym.helpers.AMM_utils import (
    get_buckets_given_center_bucket_id, find_bucket_id, is_out_of_range, transaction_fee_one_step, transaction_fee_for_sequence
)


from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import MidpriceModel
from SAiFE_gym.stochastic_processes.price_impact_models import PriceImpactModel

class ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: PriceImpactModel = None,
        price_impact_model: PriceImpactModel = None,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model
        self.rng = default_rng(seed)
        self.seed_ = seed

        self.state = None 

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass

    def get_fills(self, action: np.ndarray):
        pass
    
    def get_arrivals_and_fills(self, action: np.ndarray):
        return None, None 


    def get_action_space(self) -> gym.spaces.Space:
        pass

    def get_required_stochastic_processes(self):
        pass

    @property
    def midprice(self):
        return self.midprice_model.current_state[:, 0].reshape(-1, 1)



class UniswapV3ModelDynamics(ModelDynamics):
    """
    Uniswap V3 Model Dynamics with concentrated liquidity.

    The agent (LP) can choose to allocate liquidity in different price ranges (buckets).
    The action space is DYNAMIC - only buckets within tau of the current price are "active".
    Agents return probability distributions over 2*tau+1 active buckets.
    """

    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: Optional[PriceImpactModel] = None,
        price_impact_model: Optional[PriceImpactModel] = None,
        initial_capital: float = 10000.0,  # Total initial capital to provide as liquidity
        fee_tier: float = 0.003,           # 0.3% fee tier
        tau: int = 5,                      # Number of buckets on each side of current price
        exponential_value: float = 1.0001, # Base for exponential bucket spacing (Uniswap V3 tick spacing)
        non_arb_lambda: float = 0.00005,
        seed: int = None,
    ):
        super().__init__(midprice_model, arrival_model, fill_probability_model, price_impact_model, seed)

        self.initial_capital = initial_capital
        self.initial_price = midprice_model.initial_state[0, 0] if midprice_model else 100.0
        self.fee_tier = fee_tier
        self.tau = tau  # Hyperparameter for active bucket window
        self.exponential_value = exponential_value
        self.non_arb_lambda = non_arb_lambda
        self.use_mixed_strategy = True

        # Dynamic action space: 2*tau + 1 active buckets around current price
        self.num_active_buckets = 2 * tau + 1

    def get_action_space(self):
        """
        Return the action space for the agent.
        Action is a probability distribution over 2*tau+1 active buckets around current price.
        """
        if self.use_mixed_strategy:
            # Output probability distribution over 2*tau+1 active buckets
            return gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_active_buckets,), dtype=np.float32)
        else:
            # Output single bucket index (from 0 to 2*tau)
            return gym.spaces.Discrete(self.num_active_buckets)



    def update_state(self, arrivals: np.ndarray, action: np.ndarray):
        """
        Update state based on arbitrage, noisy traders, and fee collection.

        Args:
            arrivals: Array of shape (num_trajectories, 2) - boolean array [SELL, BUY]
            action: Array of shape (num_trajectories, 2*tau+1) - probability distribution over active buckets

        The function implements a 3-phase update:
        1. Arbitrage trades: Snap AMM price to no-arbitrage bounds
        2. Noisy trader orders: Apply multiplicative price impacts from BUY/SELL arrivals
        3. Fee collection: Calculate and accumulate fees from all price movements
        """
        num_trajectories = self.state.shape[0]

        # ====================================================================
        # PHASE 0: Initialize price tracking and determine active buckets
        # ====================================================================
        # Store price sequence for fee calculation (in sqrt space)
        price_sequence_sqrt = []
        price_sequence_sqrt.append(self.state[:, AMM_PRICE_INDEX].copy())

        # Determine active buckets BEFORE any price changes
        # This ensures action probabilities map to correct absolute price ranges
        buckets_per_trajectory = []
        for traj_idx in range(num_trajectories):
            current_price = self.state[traj_idx, AMM_PRICE_INDEX] ** 2
            center_bucket_id = find_bucket_id(current_price, self.exponential_value)
            buckets = get_buckets_given_center_bucket_id(
                center_bucket_id, self.tau, self.exponential_value
            )
            buckets_per_trajectory.append(buckets)

        # ====================================================================
        # PHASE 1: Arbitrage Trades
        # ====================================================================
        # Calculate no-arbitrage bounds (in sqrt space)
        midprice = self.midprice.flatten()  # shape: (num_trajectories,)
        lower_bound = np.sqrt((1.0 - self.fee_tier) * midprice)
        upper_bound = np.sqrt(midprice / (1.0 - self.fee_tier))

        # Snap AMM price to bounds (vectorized)
        self.state[:, AMM_PRICE_INDEX] = np.clip(
            self.state[:, AMM_PRICE_INDEX],
            lower_bound,
            upper_bound
        )

        price_sequence_sqrt.append(self.state[:, AMM_PRICE_INDEX].copy())

        # ====================================================================
        # PHASE 2: Noisy Trader Orders
        # ====================================================================
        # Extract arrival masks
        sell_mask = arrivals[:, 0].astype(bool)  # Column 0 = SELL
        buy_mask = arrivals[:, 1].astype(bool)   # Column 1 = BUY

        # Apply multiplicative price impacts
        # SELL orders decrease price
        self.state[sell_mask, AMM_PRICE_INDEX] *= (1.0 - self.non_arb_lambda)

        # BUY orders increase price
        self.state[buy_mask, AMM_PRICE_INDEX] *= (1.0 + self.non_arb_lambda)

        # Safety: ensure prices stay positive
        self.state[:, AMM_PRICE_INDEX] = np.maximum(self.state[:, AMM_PRICE_INDEX], 1e-8)

        price_sequence_sqrt.append(self.state[:, AMM_PRICE_INDEX].copy())

        # ====================================================================
        # PHASE 3: Fee Collection
        # ====================================================================
        # For each trajectory, calculate fees across active buckets
        for traj_idx in range(num_trajectories):
            buckets = buckets_per_trajectory[traj_idx]

            # Convert sqrt prices to regular prices for this trajectory
            price_seq = [p[traj_idx] ** 2 for p in price_sequence_sqrt]

            # Optimize: remove duplicate consecutive prices
            price_seq_compact = [price_seq[0]]
            for p in price_seq[1:]:
                if not np.isclose(p, price_seq_compact[-1]):
                    price_seq_compact.append(p)

            # Skip if no price movement
            if len(price_seq_compact) == 1:
                continue

            # For each active bucket
            for bucket_idx, bucket in enumerate(buckets):
                action_prob = action[traj_idx, bucket_idx]

                if action_prob > 1e-10:  # Skip if essentially zero
                    # Calculate liquidity allocated to this bucket
                    L_bucket = action_prob * self.initial_capital

                    # Calculate fees through price sequence
                    for step_idx in range(len(price_seq_compact) - 1):
                        p1 = price_seq_compact[step_idx]
                        p2 = price_seq_compact[step_idx + 1]

                        # Get fees per unit liquidity
                        fee_a, fee_b = transaction_fee_one_step(
                            bucket['p_low'],
                            bucket['p_high'],
                            p1,
                            p2,
                            self.fee_tier
                        )

                        # Scale by actual liquidity and accumulate
                        self.state[traj_idx, FEES_TOKEN_A_INDEX] += fee_a * L_bucket
                        self.state[traj_idx, FEES_TOKEN_B_INDEX] += fee_b * L_bucket


    def get_arrivals_and_fills(self, action: np.ndarray):
        """
        Generate arrivals from the arrival model.
        In V3, fills = arrivals (all orders are filled by the AMM).
        """
        if self.arrival_model is not None:
            arrivals = self.arrival_model.get_next_state()
            # For AMMs, all arrivals are filled (no order book)
            fills = arrivals.copy()
            return arrivals[0], fills[0]  # Return first trajectory
        else:
            return np.zeros(2), np.zeros(2)
