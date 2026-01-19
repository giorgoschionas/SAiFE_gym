
import abc
import gym
from copy import copy
from typing import Optional

import numpy as np
from numpy.random import default_rng

from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    ASSET_PRICE_KEY, TIME_KEY
)
from SAiFE_gym.gym.helpers.AMM_utils import (
    unified_swap_single_tick, get_tick_boundaries, price_to_tick
)


from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import MidpriceModel
from SAiFE_gym.stochastic_processes.price_impact_models import PriceImpactModel

class ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        price_impact_model: PriceImpactModel = None,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.price_impact_model = price_impact_model
        self.num_trajectories = num_trajectories
        self.rng = default_rng(seed)
        self.seed_ = seed

        self.state = None 

    def update_state(self, arrivals: np.ndarray, action: np.ndarray):
        pass

    
    def get_arrivals(self, action: np.ndarray):
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
    Uniswap V3 Model Dynamics with Concentrated Liquidity.

    The agent (LP) can choose to allocate liquidity in specific price ranges.
    The action space is DISCRETE - agents specify position bounds and liquidity fraction.

    """

    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        num_trajectories: int = 1,
        fee_tier: float = 0.003,           # 0.3% fee tier
        tau: int = 5,                      # Number of ticks around current tick
        num_ticks: int = 1000,             # Total ticks to track in liquidity array
        exponential_value: float = 1.0001, # Base for exponential tick spacing
        seed: int = None,
    ):
        super().__init__(midprice_model = midprice_model,
                         arrival_model = arrival_model,
                         num_trajectories = num_trajectories,
                         seed = seed)

        self.initial_price = midprice_model.initial_state[0, 0] if midprice_model else 100.0
        self.fee_tier = fee_tier
        self.tau = tau  # Hyperparameter for active tick window
        self.num_ticks = num_ticks  # Total number of ticks to track
        self.exponential_value = exponential_value

        # Track the center of the liquidity array (set during state initialization)
        self.tick_lower_global = None


    def get_action_space(self):
        """
        Return the action space for the agent.

        Action format: [lower_offset, upper_offset, liquidity_fraction]
        - lower_offset: Tick offset from current tick (range: -tau to tau-1)
        - upper_offset: Tick offset from current tick (range: -tau+1 to tau)
        - liquidity_fraction: Fraction of available capital (range: 0.0 to 1.0)

        Constraint: lower_offset < upper_offset (enforced by validate_action)
        """

        return gym.spaces.Box(
            low=np.array([-self.tau, -self.tau + 1, 0.0], dtype=np.float32),
            high=np.array([self.tau - 1, self.tau, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Validate and clip action to ensure constraints.

        Args:
            action: (num_trajectories, 3) array of actions

        Returns:
            Validated action with same shape

        Ensures:
        - All values are within box bounds
        - lower_offset < upper_offset (minimum width of 1 tick)
        """
        action = action.copy()

        # Clip to box bounds
        action[:, 0] = np.clip(action[:, 0], -self.tau, self.tau - 1)
        action[:, 1] = np.clip(action[:, 1], -self.tau + 1, self.tau)
        action[:, 2] = np.clip(action[:, 2], 0.0, 1.0)

        # Ensure lower < upper (add minimum width of 1 tick if violated)
        invalid = action[:, 0] >= action[:, 1]
        action[invalid, 1] = action[invalid, 0] + 1

        # Re-clip upper after adjustment
        action[:, 1] = np.clip(action[:, 1], -self.tau + 1, self.tau)

        return action

    def update_state(self, arrivals: np.ndarray, action: np.ndarray):
        """
        Update pool state using unified single-tick swap.

        Implements the core AMM dynamics:
        1. Convert boolean arrivals to numeric amounts
        2. Extract current state variables
        3. Get active liquidity at current tick
        4. Get tick boundaries
        5. Execute swap (clamped to boundary automatically)
        6. Update state (sqrt_price, current_tick)
        7. Accumulate fees at the tick where swap occurred

        Args:
            arrivals: Boolean arrivals array, shape (num_trajectories, 2)
                      Column 0: sell_token0 arrivals
                      Column 1: buy_token0 arrivals
            action: Agent action array (for LP positioning, not used in swap)
        """
        if self.state is None:
            raise ValueError("State not initialized. Call reset() first.")

        num_traj = self.num_trajectories

        # Step 1: Convert boolean arrivals to numeric amounts
        # Pass large amount when arrival=True - unified_swap_single_tick will clamp to arrivals_max
        LARGE_AMOUNT = 1e18
        numeric_arrivals = arrivals.astype(np.float64) * LARGE_AMOUNT

        # Step 2: Extract current state
        sqrt_price_current = self.state[POOL_SQRT_PRICE_KEY].copy()
        current_tick = self.state[POOL_CURRENT_TICK_KEY].copy()

        # Step 3: Get active liquidity at current tick
        tick_array_idx = (current_tick - self.tick_lower_global).astype(np.int64)
        tick_array_idx = np.clip(tick_array_idx, 0, self.num_ticks - 1)
        active_liquidity = self.state[POOL_LIQUIDITY_ARRAY_KEY][
            np.arange(num_traj), tick_array_idx
        ]

        # Step 4: Get tick boundaries
        tick_lower_sqrt, tick_upper_sqrt, _ = get_tick_boundaries(
            sqrt_price_current, self.exponential_value
        )

        # Step 5: Execute swap (will clamp to boundary automatically)
        (sqrt_price_next, token0_net, token1_net,
         fee_token0, fee_token1, hit_boundary) = unified_swap_single_tick(
            sqrt_price_current=sqrt_price_current,
            liquidity=active_liquidity,
            arrivals=numeric_arrivals,
            tick_lower_boundary=tick_lower_sqrt,
            tick_upper_boundary=tick_upper_sqrt,
            fee_rate=self.fee_tier
        )

        # Step 6: Update state (sqrt_price, current_tick)
        self.state[POOL_SQRT_PRICE_KEY] = sqrt_price_next

        # Compute new tick from new sqrt price
        price_next = sqrt_price_next ** 2
        price_next = np.maximum(price_next, 1e-300)  # Numerical safety
        new_tick = np.floor(
            np.log(price_next) / np.log(self.exponential_value)
        ).astype(np.int64)
        self.state[POOL_CURRENT_TICK_KEY] = new_tick

        # Step 7: Accumulate fees at the tick where swap occurred
        np.add.at(
            self.state[FEES0_KEY],
            (np.arange(num_traj), tick_array_idx),
            fee_token0
        )
        np.add.at(
            self.state[FEES1_KEY],
            (np.arange(num_traj), tick_array_idx),
            fee_token1
        )

        # Update time
        step_size = self.midprice_model.step_size if self.midprice_model else 0.005
        self.state[TIME_KEY] = self.state[TIME_KEY] + step_size

    def get_arrivals(self) -> np.ndarray:
        """
        Get arrivals from arrival model.

        Returns:
            np.ndarray: Boolean arrivals array, shape (num_trajectories, 2)
                        Column 0: sell_token0 arrivals
                        Column 1: buy_token0 arrivals
        """
        if self.arrival_model is None:
            # Return no arrivals if no arrival model
            return np.zeros((self.num_trajectories, 2), dtype=bool)

        arrivals = self.arrival_model.get_arrivals()
        return arrivals
