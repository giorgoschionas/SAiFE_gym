import gymnasium
import numpy as np
from gymnasium.vector.utils import batch_space
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
    """Native batched simulator, including when ``num_trajectories == 1``.

    Use GymnasiumAMMEnvironment or GymnasiumAMMVectorEnv with third-party
    Gymnasium wrappers, and StableBaselinesAMMEnvironment for SB3 training.
    Raw observations reference simulator state; the Gymnasium adapters copy it.
    """

    metadata = {"render_modes": []}
    render_mode = None

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
        self.single_observation_space = self._create_single_observation_space()
        self.observation_space = batch_space(
            self.single_observation_space, self.num_trajectories
        )
        self.action_space = self.model_dynamics.get_action_space()
        self.single_action_space = self.action_space
        self.batched_action_space = batch_space(self.action_space, self.num_trajectories)

        # Initialize state based on model dynamics type
        self._initial_state = self._initial_v3_state()
        self.model_dynamics.state = {k: v.copy() for k, v in self._initial_state.items()}

        # Initialize random number generator
        if seed is not None:
            self.seed(seed)
        self.rng = np.random.default_rng(seed)


    def _create_single_observation_space(self) -> gymnasium.spaces.Space:
        """Describe one trajectory without reducing simulator precision."""
        if not isinstance(self.model_dynamics, UniswapV3ModelDynamics):
            return gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
            )

        # Elapsed time is unbounded here: repeated additions can exceed the
        # nominal horizon by roundoff. Termination still uses the trading horizon.
        scalar_keys = (
            POOL_SQRT_PRICE_KEY, LP_LIQUIDITY_KEY,
            LP_COLLECTED_FEES0_KEY, LP_COLLECTED_FEES1_KEY,
            LP_UNCLAIMED_FEES0_KEY, LP_UNCLAIMED_FEES1_KEY,
            LP_FEE_SNAPSHOT0_KEY, LP_FEE_SNAPSHOT1_KEY,
            ASSET_PRICE_KEY, TIME_KEY, GAS_COST_KEY, INITIAL_WEALTH_KEY,
            PORTFOLIO_VALUE_KEY, LP_TOKEN0_AMOUNT_KEY,
        )
        spaces = {
            key: gymnasium.spaces.Box(0.0, np.inf, shape=(), dtype=np.float64)
            for key in scalar_keys
        }
        for key in (POOL_CURRENT_TICK_KEY, LP_TICK_LOWER_KEY, LP_TICK_UPPER_KEY):
            spaces[key] = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(), dtype=np.int64
            )
        for key in (POOL_LIQUIDITY_ARRAY_KEY, FEES0_KEY, FEES1_KEY):
            spaces[key] = gymnasium.spaces.Box(
                0.0, np.inf, shape=(self.model_dynamics.num_ticks,), dtype=np.float64
            )
        spaces[LP_ALPHA_KEY] = gymnasium.spaces.Box(
            0.0, 1.0, shape=(), dtype=np.float64
        )
        spaces[LP_EVER_DEPLOYED_KEY] = gymnasium.spaces.Box(
            0, 1, shape=(), dtype=np.bool_
        )
        return gymnasium.spaces.Dict(spaces)

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
            self.model_dynamics.rng = np.random.default_rng(
                seed + 3 if seed is not None else None
            )
            self.model_dynamics.seed = seed + 3 if seed is not None else None
            if self.model_dynamics.midprice_model:
                self.model_dynamics.midprice_model.seed(seed)
            if self.model_dynamics.arrival_model:
                self.model_dynamics.arrival_model.seed(
                    seed + 1 if seed is not None else None
                )
            if self.model_dynamics.price_impact_model:
                self.model_dynamics.price_impact_model.seed(
                    seed + 2 if seed is not None else None
                )

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment to initial state.

        Returns:
            (batched_obs, info); arrays keep their trajectory dimension.
        """
        if seed is not None:
            self.seed(seed)

        reset_stochastic_processes(self.model_dynamics)

        reset_model_state(
            self.model_dynamics,
            self._initial_state,
            self.num_trajectories,
        )

        # Reset reward function
        self.reward_function.reset(self.model_dynamics.state)

        # Recompute derived obs from fresh state
        self._compute_derived_obs()

        return self.model_dynamics.state, {}


    def step(self, action: np.ndarray):
        """Execute one step across the trajectory batch.

        Actions have shape ``(num_trajectories, 3)``. A ``(3,)`` command is
        also accepted for a single trajectory. Legacy two-column batched
        rebalance actions and ``None`` remain supported.

        Returns:
            (obs, rewards, terminated, truncated, info), with batched arrays.
            terminated: episode reached its natural end (trading horizon elapsed).
            truncated: always False (no external time-limit truncation).
        """
        if action is not None:
            action = np.asarray(action)
            if self.num_trajectories == 1 and action.shape == (3,):
                action = action.reshape(1, 3)
            if (
                action.ndim != 2
                or action.shape[0] != self.num_trajectories
                or action.shape[1] not in (2, 3)
            ):
                raise ValueError(
                    f"expected batched action shape ({self.num_trajectories}, 3) "
                    f"(or legacy ({self.num_trajectories}, 2)), got {action.shape}"
                )
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
        advance_market_state(self.model_dynamics, arrivals, action)

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
