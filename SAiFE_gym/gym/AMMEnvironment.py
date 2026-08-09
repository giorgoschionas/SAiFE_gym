import gymnasium
import numpy as np
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.gym.ModelDynamics import ModelDynamics, UniswapV3ModelDynamics
from SAiFE_gym.rewards.RewardFunctions import RewardFunction, PnL
from SAiFE_gym.gym.index_names import (
    POOL_SQRT_PRICE_KEY, POOL_CURRENT_TICK_KEY, POOL_LIQUIDITY_ARRAY_KEY,
    FEES0_KEY, FEES1_KEY, LP_LIQUIDITY_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY,
    LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
    LP_UNCLAIMED_FEES0_KEY, LP_UNCLAIMED_FEES1_KEY,
    LP_FEE_SNAPSHOT0_KEY, LP_FEE_SNAPSHOT1_KEY,
    LP_EVER_DEPLOYED_KEY,
    ASSET_PRICE_KEY, TIME_KEY, GAS_COST_KEY, INITIAL_WEALTH_KEY,
    PORTFOLIO_VALUE_KEY, LP_ALPHA_KEY, LP_TOKEN0_AMOUNT_KEY,
    RECENT_REALIZED_VOLATILITY_KEY,
)
from SAiFE_gym.gym.simulation_core import (
    advance_market_state,
    compute_derived_obs,
    create_uniswap_v3_initial_state,
    reset_model_state,
    reset_stochastic_processes,
    terminated_flags,
)


class AMMEnvironment(gymnasium.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 200,
        reward_function: RewardFunction = None,
        model_dynamics: ModelDynamics = None,
        initial_wealth: float = 1e6,
        num_trajectories: int = 1,
        initial_pool_price: float = None,
        realized_vol_window: int = 50,
        seed: int = None):
        super(AMMEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.initial_wealth = initial_wealth
        self.num_trajectories = num_trajectories
        self.initial_pool_price = initial_pool_price
        if isinstance(realized_vol_window, bool) or int(realized_vol_window) < 2:
            raise ValueError("realized_vol_window must be an integer >= 2")
        self.realized_vol_window = int(realized_vol_window)
        self._step_size = self.terminal_time / self.n_steps
        self._realized_vol_eps = 1e-12
        self._log_return_buffer = np.zeros(
            (self.realized_vol_window, self.num_trajectories),
            dtype=np.float64,
        )
        self._log_return_cursor = 0
        self._log_return_count = 0
        self._previous_midprice = np.zeros(self.num_trajectories, dtype=np.float64)

        # Create model dynamics if not provided
        self.model_dynamics = model_dynamics or UniswapV3ModelDynamics(
            midprice_model=BrownianMotionMidpriceModel(
                step_size=self._step_size, num_trajectories=num_trajectories, seed=seed
            ),
            arrival_model=PoissonArrivalModel(
                intensity=np.array([100, 100]), step_size=self._step_size, num_trajectories=num_trajectories, seed=seed
            ),
            num_trajectories=num_trajectories,
            seed=seed
        )

        # Create reward function (default to PnL which works with both array and dict states)
        self.reward_function = reward_function if reward_function else PnL()

        # Define observation and low-level command action spaces.
        # Use SAiFE_gym.wrappers for canonical discrete LP decision spaces.
        self.observation_space = self._create_observation_space()
        self.action_space = self.model_dynamics.get_action_space()

        # Initialize state based on model dynamics type
        self._initial_state = self._initial_v3_state()
        self.model_dynamics.state = {k: v.copy() for k, v in self._initial_state.items()}
        self._reset_realized_volatility()

        # Initialize random number generator
        if seed:
            self.seed(seed)
        self.rng = np.random.default_rng(seed)


    def _create_observation_space(self) -> gymnasium.spaces.Space:
        """
        Create observation space based on model dynamics type.

        Returns:
            gym.spaces.Dict for Uniswap V3, Box for others
        """
        if not isinstance(self.model_dynamics, UniswapV3ModelDynamics):
            # Fallback for other dynamics types (legacy flat array)
            return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        num_ticks = self.model_dynamics.num_ticks

        return gymnasium.spaces.Dict({
            # Pool state (global liquidity)
            POOL_SQRT_PRICE_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            POOL_CURRENT_TICK_KEY: gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.int32
            ),
            POOL_LIQUIDITY_ARRAY_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories, num_ticks),
                dtype=np.float32
            ),

            # Fee arrays (per-tick)
            FEES0_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories, num_ticks),
                dtype=np.float32
            ),
            FEES1_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories, num_ticks),
                dtype=np.float32
            ),

            # LP state (agent's position)
            LP_LIQUIDITY_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            LP_TICK_LOWER_KEY: gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.int32
            ),
            LP_TICK_UPPER_KEY: gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.int32
            ),

            # LP cumulative fee tracking
            LP_COLLECTED_FEES0_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            LP_COLLECTED_FEES1_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),

            # LP unclaimed fee bucket (accrued since last rebalance)
            LP_UNCLAIMED_FEES0_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            LP_UNCLAIMED_FEES1_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),

            # Market state
            ASSET_PRICE_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            TIME_KEY: gymnasium.spaces.Box(
                low=0.0, high=self.terminal_time,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),

            # Derived observation features
            PORTFOLIO_VALUE_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            LP_ALPHA_KEY: gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            LP_TOKEN0_AMOUNT_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            RECENT_REALIZED_VOLATILITY_KEY: gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
        })

    def _initial_v3_state(self) -> dict:
        """
        Initialize Uniswap v3 pool state.

        Sets up the initial state dictionary with:
        - Pool state: sqrt_price, current_tick, liquidity_array
        - Fee arrays: fees_0, fees_1 (per-tick)
        - LP state: lp_liquidity, lp_tick_lower, lp_tick_upper
        - Market state: midprice, time

        Returns:
            dict: Initial state dictionary with all required keys
        """
        return create_uniswap_v3_initial_state(
            self.model_dynamics,
            self.num_trajectories,
            self.initial_wealth,
            self.initial_pool_price,
        )

    def seed(self, seed: int = None):
        """Set random seed for the environment."""
        self.rng = np.random.default_rng(seed)
        # Seed stochastic processes via model dynamics
        if self.model_dynamics:
            if self.model_dynamics.midprice_model:
                self.model_dynamics.midprice_model.seed(seed)
            if self.model_dynamics.arrival_model:
                self.model_dynamics.arrival_model.seed(seed + 1 if seed else None)

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment to initial state.

        Returns:
            (obs, info) per Gymnasium API.
        """
        if seed is not None:
            self.seed(seed)

        reset_stochastic_processes(self.model_dynamics)

        reset_model_state(
            self.model_dynamics,
            self._initial_state,
            self.num_trajectories,
        )
        self._reset_realized_volatility()

        # Reset reward function
        self.reward_function.reset(self.model_dynamics.state)

        # Recompute derived obs from fresh state
        self._compute_derived_obs()

        return self.model_dynamics.state, {}


    def step(self, action: np.ndarray):
        """Execute one environment step.

        Returns:
            (obs, rewards, terminated, truncated, info) per Gymnasium API.
            terminated: episode reached its natural end (trading horizon elapsed).
            truncated: always False (no external time-limit truncation).
        """
        current_state = {k: v.copy() for k, v in self.model_dynamics.state.items()}

        # Update state
        next_state = self._update_state(action)

        # Calculate termination
        terminated = self._get_terminated()
        truncated = np.zeros(self.num_trajectories, dtype=bool)

        # Calculate rewards
        rewards = self.reward_function.calculate(current_state, action, next_state, terminated[0])
        # Calculate post-step info dict
        info = self._calculate_infos(next_state, action, rewards)
        return next_state, rewards, terminated, truncated, info

    def _compute_derived_obs(self):
        """Compute portfolio_value, lp_alpha, and lp_token0_amount, store in state dict."""
        compute_derived_obs(self.model_dynamics.state, self.model_dynamics)

    def _update_state(self, action: np.ndarray):
        # Step 1: Get arrivals from current model state (intensity at t)
        arrivals = self.model_dynamics.get_arrivals()

        # Step 2: Update pool state (rebalance + swaps + time advance)
        self.model_dynamics.update_state(arrivals, action)

        # Step 3: Advance all stochastic processes
        self._update_market_state(arrivals, action)
        self._update_realized_volatility()
        self.model_dynamics.state[TIME_KEY] += self.step_size

        # Step 4: Update derived observation features
        self._compute_derived_obs()

        return self.model_dynamics.state

    def _update_market_state(self, arrivals: np.ndarray, action: np.ndarray):
        """
        Update all stochastic processes after pool state has been updated.

        Processes are updated with ACTUAL arrivals
        after they are generated, not before.
        """
        advance_market_state(self.model_dynamics, arrivals, action)

    def _reset_realized_volatility(self) -> None:
        """Reset vectorized rolling log-return state for realized volatility."""
        self._log_return_buffer.fill(0.0)
        self._log_return_cursor = 0
        self._log_return_count = 0
        self._previous_midprice = np.asarray(
            self.model_dynamics.state[ASSET_PRICE_KEY],
            dtype=np.float64,
        ).copy()
        self.model_dynamics.state[RECENT_REALIZED_VOLATILITY_KEY] = np.zeros(
            self.num_trajectories,
            dtype=np.float64,
        )

    def _update_realized_volatility(self) -> None:
        """Update rolling unit-horizon volatility from vectorized log returns."""
        current_midprice = np.asarray(
            self.model_dynamics.state[ASSET_PRICE_KEY],
            dtype=np.float64,
        )
        safe_current = np.maximum(current_midprice, self._realized_vol_eps)
        safe_previous = np.maximum(self._previous_midprice, self._realized_vol_eps)
        log_return = np.log(safe_current / safe_previous)

        self._log_return_buffer[self._log_return_cursor] = log_return
        self._log_return_cursor = (
            self._log_return_cursor + 1
        ) % self.realized_vol_window
        self._log_return_count = min(
            self._log_return_count + 1,
            self.realized_vol_window,
        )
        self._previous_midprice = current_midprice.copy()

        if self._log_return_count < 2:
            realized_volatility = np.zeros(self.num_trajectories, dtype=np.float64)
        else:
            valid_returns = self._log_return_buffer[: self._log_return_count]
            realized_volatility = (
                np.std(valid_returns, axis=0, ddof=1) / np.sqrt(self.step_size)
            )

        self.model_dynamics.state[RECENT_REALIZED_VOLATILITY_KEY] = (
            realized_volatility
        )

    def _get_terminated(self):
        """Return terminated flags: True when the trading horizon has elapsed."""
        return terminated_flags(
            self.model_dynamics.state,
            self.terminal_time,
            self._step_size,
            self.num_trajectories,
        )

    def _calculate_infos(self, state, action, rewards):
        """Return lightweight batched diagnostics for the completed step."""
        return {
            "asset_price": state[ASSET_PRICE_KEY].copy(),
            "pool_price": (state[POOL_SQRT_PRICE_KEY] ** 2).copy(),
            "time": state[TIME_KEY].copy(),
            "action": np.asarray(action).copy(),
            "reward": rewards.copy(),
        }
    
    @property
    def initial_state(self):
        return {k: v.copy() for k, v in self._initial_state.items()}
    
    @property
    def state(self):
        return self.model_dynamics.state
    
    @property
    def step_size(self):
        return self._step_size
