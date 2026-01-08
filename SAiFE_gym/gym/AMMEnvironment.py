import gym
import numpy as np

from gym.spaces import Box
from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel, PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.gym.ModelDynamics import ModelDynamics, UniswapV3ModelDynamics
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.rewards.RewardFunctions import RewardFunction
from SAiFE_gym.gym.index_names import TIME_INDEX

from SAiFE_gym.gym.index_names import (
    LIQUIDITY_INDEX, AMM_PRICE_INDEX, ASSET_PRICE_INDEX, TIME_INDEX
)
from SAiFE_gym.gym.helpers.AMM_utils import price_to_tick



class AMMEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 200,
        initial_cash: float = 0.0,
        reward_function: RewardFunction = None,
        model_dynamics: ModelDynamics = None,
        num_trajectories: int = 1,  # FIXED: Added missing parameter
        seed: int = None):
        super(AMMEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.num_trajectories = num_trajectories
        self._step_size = self.terminal_time / self.n_steps

        # Initialize normalization flags
        self.normalise_observation_space_ = False
        self.normalise_action_space_ = False
        self.normalise_rewards_ = False
        self.reward_scaling = 1.0
        self._intercept_obs_norm = 0.0
        self._gradient_obs_norm = 1.0
        self._intercept_action_norm = 0.0
        self._gradient_action_norm = 1.0

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

        # Create reward function
        self.reward_function = reward_function if reward_function else RewardFunction()

        # Define observation and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self.model_dynamics.get_action_space()

        # Initialize state based on model dynamics type
        if isinstance(self.model_dynamics, UniswapV3ModelDynamics):
            self._initial_state = self._initialize_v3_state()
        else:
            # For other model types, initialize with placeholder or call their own init
            self._initial_state = None

        # Initialize random number generator
        if seed:
            self.seed(seed)
        self.rng = np.random.default_rng(seed)


    def _create_observation_space(self) -> gym.spaces.Space:
        """
        Create observation space based on model dynamics type.

        Returns:
            gym.spaces.Dict for Uniswap V3, Box for others
        """
        if not isinstance(self.model_dynamics, UniswapV3ModelDynamics):
            # Fallback for other dynamics types (legacy flat array)
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        num_ticks = self.model_dynamics.num_ticks

        return gym.spaces.Dict({
            # Pool state (global liquidity)
            'sqrt_price': gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            'current_tick': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.int32
            ),
            'liquidity_array': gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories, num_ticks),
                dtype=np.float32
            ),

            # LP state (agent's position)
            'lp_liquidity': gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            'lp_tick_lower': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.int32
            ),
            'lp_tick_upper': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.int32
            ),
            'fees_token_a': gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            'fees_token_b': gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),

            # Market state
            'midprice': gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
            'time': gym.spaces.Box(
                low=0.0, high=self.terminal_time,
                shape=(self.num_trajectories,),
                dtype=np.float32
            ),
        })

    def _initialize_v3_state(self) -> np.ndarray:
        """Initialize Uniswap v3 pool state (to be implemented in Phase 5)."""
        pass

    def seed(self, seed: int = None):
        """Set random seed for the environment."""
        self.rng = np.random.default_rng(seed)
        # Seed stochastic processes via model dynamics
        if self.model_dynamics:
            if self.model_dynamics.midprice_model:
                self.model_dynamics.midprice_model.seed(seed)
            if self.model_dynamics.arrival_model:
                self.model_dynamics.arrival_model.seed(seed + 1 if seed else None)

    # clears internal state & returns initial observation (what the agent sees at the start)
    def reset(self):
        """Reset the environment to initial state."""
        # Reset stochastic processes
        if self.model_dynamics:
            if self.model_dynamics.midprice_model:
                self.model_dynamics.midprice_model.reset()
            if self.model_dynamics.arrival_model:
                self.model_dynamics.arrival_model.reset()

        # Reset state
        if isinstance(self._initial_state, dict):
            self.model_dynamics.state = {k: v.copy() for k, v in self._initial_state.items()}
        else:
            self.model_dynamics.state = self._initial_state.copy() if self._initial_state is not None else None

        # Reset reward function
        self.reward_function.reset(self.model_dynamics.state)

        return self.normalise_observation(self.model_dynamics.state)



    # The step function is a core component of reinforcement learning environments, simulating one discrete time step 
    # within the trading environment. It takes an action from the agent, updates the environment's state based on that action, 
    # calculates the resulting reward, and determines if the episode has concluded.
    # It is standard structure for environments in OpenAI's Gym.
    def step(self, action: np.ndarray):
        """Execute one environment step."""
        action = self.normalise_action(action, inverse=True)

        # Copy current state (handle both Dict and array)
        if isinstance(self.model_dynamics.state, dict):
            current_state = {k: v.copy() for k, v in self.model_dynamics.state.items()}
        else:
            current_state = self.model_dynamics.state.copy()

        # Update state
        next_state = self._update_state(action)

        # Calculate termination
        dones = self._get_dones()

        # Calculate rewards
        rewards = self.reward_function.calculate(current_state, action, next_state, dones[0])

        # Calculate info dict
        infos = self._calculate_infos(current_state, action, rewards)

        return self.normalise_observation(next_state), self.normalise_rewards(rewards), dones, infos

    def normalise_observation(self, obs, inverse: bool = False):
        """Normalize observation (handles both Dict and array observations)."""
        # For Dict observations, return as-is for now
        # TODO: Implement per-key normalization if needed
        if isinstance(obs, dict):
            return obs

        # For array observations, use existing normalization
        if self.normalise_observation_space_ and not inverse:
            return (obs - self._intercept_obs_norm) / self._gradient_obs_norm - 1
        elif self.normalise_observation_space_ and inverse:
            return (obs + 1) * self._gradient_obs_norm + self._intercept_obs_norm
        else:
            return obs

    def normalise_action(self, action: np.ndarray, inverse: bool = False):
        if self.normalise_action_space_ and not inverse:
            return (action - self._intercept_action_norm) / self._gradient_action_norm - 1
        elif self.normalise_action_space_ and inverse:
            return (action + 1) * self._gradient_action_norm + self._intercept_action_norm
        else:
            return action
    
    def normalise_rewards(self, rewards: np.ndarray):
        return self.reward_scaling * rewards if self.normalise_rewards_ else rewards


    def _update_state(self, action: np.ndarray):
        """
        Update environment state by one timestep.

        Args:
            action: (num_trajectories, 3) action array

        Returns:
            Updated state (Dict or array depending on model dynamics)
        """
        # Update stochastic processes (midprice, arrivals)
        if self.model_dynamics.midprice_model:
            self.model_dynamics.midprice_model.update(None, None, None)

        # Get arrivals from arrival model
        arrivals = self.model_dynamics.get_arrivals()

        # Update pool state through model dynamics
        self.model_dynamics.update_state(arrivals, action)

        return self.model_dynamics.state

    def _update_market_state(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        """DEPRECATED: Use _update_state() instead."""
        pass

    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        """DEPRECATED: Use _update_state() instead."""
        pass

    def _get_dones(self):
        """Check if episode is complete."""
        # Handle both Dict and array state
        if isinstance(self.model_dynamics.state, dict):
            done = self.model_dynamics.state['time'][0] >= self.terminal_time - self._step_size / 2
        else:
            done = self.model_dynamics.state[0, TIME_INDEX] >= self.terminal_time - self._step_size / 2

        return np.full((self.num_trajectories,), done, dtype=bool)

    def _calculate_infos(self, current_state, action, rewards):
        """Calculate info dict for step return."""
        # Placeholder - can be extended with additional metrics
        return {}
    
    @property
    def initial_state(self):
        return self._initial_state.copy()
    
    @property
    def state(self):
        return self.model_dynamics.state
    
    @property
    def step_size(self):
        return self._step_size