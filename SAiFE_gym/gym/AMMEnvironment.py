import gymnasium
import numpy as np
from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel, PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.gym.ModelDynamics import ModelDynamics, UniswapV3ModelDynamics
from SAiFE_gym.agents.Agent import Agent
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
)
from SAiFE_gym.gym.helpers.AMM_utils import price_to_tick


def compute_derived_obs(state: dict, model_dynamics: 'ModelDynamics') -> None:
    """Compute portfolio_value, lp_alpha, and lp_token0_amount in-place on `state`.

    Centralized helper so the env, tests, and any other code that bypasses
    AMMEnvironment.step (e.g. driving model_dynamics directly) can keep the
    derived observation keys consistent with the rest of the state.
    """
    state[POOL_SQRT_PRICE_KEY] = model_dynamics.get_pool_sqrt_price(state)
    unclaimed0, unclaimed1 = model_dynamics.compute_unclaimed_fees(state)
    state[LP_UNCLAIMED_FEES0_KEY] = unclaimed0
    state[LP_UNCLAIMED_FEES1_KEY] = unclaimed1
    state[PORTFOLIO_VALUE_KEY] = model_dynamics.compute_portfolio_value(state)
    state[LP_ALPHA_KEY] = model_dynamics.compute_lp_alpha(state)
    state[LP_TOKEN0_AMOUNT_KEY] = model_dynamics.compute_lp_token0_amount(state)


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
        seed: int = None):
        super(AMMEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.initial_wealth = initial_wealth
        self.num_trajectories = num_trajectories
        self.initial_pool_price = initial_pool_price
        self._step_size = self.terminal_time / self.n_steps

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
        # External midprice comes from the midprice model; pool price defaults to
        # the same value but can be overridden via `initial_pool_price` to start
        # the simulation with a deliberate mispricing between the AMM and the market.
        initial_price = self.model_dynamics.initial_price
        pool_price = self.initial_pool_price if self.initial_pool_price is not None else initial_price
        pool_tick = price_to_tick(pool_price)

        # Center the lattice on the POOL tick (that's where the AMM lives) and
        # build the AMM lattice so POOL_SQRT_PRICE_KEY can be read straight off the grid.
        num_ticks = self.model_dynamics.num_ticks
        self.model_dynamics.tick_lower_global = pool_tick - num_ticks // 2
        self.model_dynamics._build_sqrt_grid()

        # Snap the pool sqrt_price to the lattice point AMM[pool_tick]. The external
        # midprice (ASSET_PRICE_KEY) is unchanged — only the on-chain pool price lives
        # on the lattice.
        initial_sqrt_price = self.model_dynamics.sqrt_grid[pool_tick - self.model_dynamics.tick_lower_global]

        # Initial liquidity (uniform distribution across all ticks)
        # This can be customized based on specific requirements
        initial_liquidity = 100000.0  # Base liquidity per tick

        return {
            # Pool state
            POOL_SQRT_PRICE_KEY: np.full(
                self.num_trajectories, initial_sqrt_price, dtype=np.float64
            ),
            POOL_CURRENT_TICK_KEY: np.full(
                self.num_trajectories, pool_tick, dtype=np.int64
            ),
            POOL_LIQUIDITY_ARRAY_KEY: np.full(
                (self.num_trajectories, num_ticks), initial_liquidity, dtype=np.float64
            ),

            # Fee arrays (per-tick, start at zero)
            FEES0_KEY: np.zeros(
                (self.num_trajectories, num_ticks), dtype=np.float64
            ),
            FEES1_KEY: np.zeros(
                (self.num_trajectories, num_ticks), dtype=np.float64
            ),

            # LP state (no position initially)
            LP_LIQUIDITY_KEY: np.zeros(self.num_trajectories, dtype=np.float64),
            LP_TICK_LOWER_KEY: np.full(
                self.num_trajectories, pool_tick - self.model_dynamics.tau, dtype=np.int64
            ),
            LP_TICK_UPPER_KEY: np.full(
                self.num_trajectories, pool_tick + self.model_dynamics.tau, dtype=np.int64
            ),

            # LP cumulative fee tracking (lifetime earnings, updated every step)
            LP_COLLECTED_FEES0_KEY: np.zeros(self.num_trajectories, dtype=np.float64),
            LP_COLLECTED_FEES1_KEY: np.zeros(self.num_trajectories, dtype=np.float64),

            # LP unclaimed fee bucket (accrued-but-not-yet-absorbed; reset at rebalance)
            LP_UNCLAIMED_FEES0_KEY: np.zeros(self.num_trajectories, dtype=np.float64),
            LP_UNCLAIMED_FEES1_KEY: np.zeros(self.num_trajectories, dtype=np.float64),

            # LP fee snapshots (for excluding pre-entry fees)
            LP_FEE_SNAPSHOT0_KEY: np.zeros(self.num_trajectories, dtype=np.float64),
            LP_FEE_SNAPSHOT1_KEY: np.zeros(self.num_trajectories, dtype=np.float64),

            # Deployment flag: False until first position is deployed (never resets to False)
            LP_EVER_DEPLOYED_KEY: np.zeros(self.num_trajectories, dtype=bool),

            # Market state
            ASSET_PRICE_KEY: np.full(
                self.num_trajectories, initial_price, dtype=np.float64
            ),
            TIME_KEY: np.zeros(self.num_trajectories, dtype=np.float64),

            # Environment parameters (constant per episode)
            GAS_COST_KEY: np.full(
                self.num_trajectories, self.model_dynamics.gas_cost, dtype=np.float64
            ),
            INITIAL_WEALTH_KEY: np.full(
                self.num_trajectories, self.initial_wealth, dtype=np.float64
            ),

            # Derived observation features (updated each step)
            PORTFOLIO_VALUE_KEY: np.full(
                self.num_trajectories, self.initial_wealth, dtype=np.float64
            ),
            LP_ALPHA_KEY: np.zeros(self.num_trajectories, dtype=np.float64),
            LP_TOKEN0_AMOUNT_KEY: np.zeros(self.num_trajectories, dtype=np.float64),
        }

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

        # Reset stochastic processes
        if self.model_dynamics:
            if self.model_dynamics.midprice_model:
                self.model_dynamics.midprice_model.reset()
            if self.model_dynamics.arrival_model:
                self.model_dynamics.arrival_model.reset()

        # Reset state
        self.model_dynamics.state = {k: v.copy() for k, v in self._initial_state.items()}
        self.model_dynamics.last_arrivals = np.zeros((self.num_trajectories, 2), dtype=bool)

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
        md = self.model_dynamics

        md.midprice_model.update(arrivals, None, action, md.state)
        md.state[ASSET_PRICE_KEY] = md.midprice_model.current_state[:, 0].copy()

        _, active_liq = md._get_current_tick_liquidity()
        context = {
            'active_liquidity': active_liq,
            'amm_price': md.state[POOL_SQRT_PRICE_KEY] ** 2,
            'midprice': md.state[ASSET_PRICE_KEY],
            'liquidity_array': md.state[POOL_LIQUIDITY_ARRAY_KEY],
            'current_tick': md.state[POOL_CURRENT_TICK_KEY],
            'tick_lower_global': md.tick_lower_global,
        }
        md.arrival_model.update(arrivals, None, action, context)

    def _get_terminated(self):
        """Return terminated flags: True when the trading horizon has elapsed."""
        done = self.model_dynamics.state[TIME_KEY][0] >= self.terminal_time - self._step_size / 2
        return np.full((self.num_trajectories,), done, dtype=bool)

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
