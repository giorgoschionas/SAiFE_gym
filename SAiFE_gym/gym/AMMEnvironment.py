import gym
import numpy as np

from gym.spaces import Box
from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel, PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.gym.ModelDynamics import ModelDynamics
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.rewards.RewardFunctions import RewardFunction
from SAiFE_gym.gym.index_names import TIME_INDEX



class AMMEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(
        self, 
        terminal_time: float = 1.0,
        n_steps: int = 20 * 10,
        initial_cash: float = 0.0,

        reward_function: RewardFunction = None,
        model_dynamics: ModelDynamics = None,
        arrival_model: ArrivalModel = None,
        seed: int = None):
        super(AMMEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self._step_size = self.terminal_time / self.n_steps
        self.reward_function = reward_function if reward_function else RewardFunction()
        self.model_dynamics = model_dynamics or UniswapV2ModelDynamics(
            midprice_model=BrownianMotionMidpriceModel(
                step_size=self._step_size, num_trajectories=num_trajectories, seed=seed
            ),
            arrival_model=PoissonArrivalModel(
                intensity=np.array([100, 100]), step_size=self._step_size, num_trajectories=num_trajectories, seed=seed
            ))



        
        if seed:
            self.seed(seed)
            self.rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed)


    def seed(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        for i, process in enumerate(self.stochastic_processes.values()):
            process.seed(seed + i + 1)

    # clears internal state & returns initial observation (what the agent sees at the start)
    def reset(self):
        for process in self.stochastic_processes.values():
            process.reset()
        self.model_dynamics.state = self._initial_state
        self.reward_function.reset(self.model_dynamics.state.copy())
        return self.normalise_observation(self.model_dynamics.state.copy())



    # The step function is a core component of reinforcement learning environments, simulating one discrete time step 
    # within the trading environment. It takes an action from the agent, updates the environment's state based on that action, 
    # calculates the resulting reward, and determines if the episode has concluded.
    # It is standard structure for environments in OpenAI's Gym.
    def step(self, action: np.ndarray):
        action = self.normalise_action(action, inverse=True)
        current_state = self.model_dynamics.state.copy()
        next_state = self._update_state(action)
        dones = self._get_dones()
        rewards = self.reward_function.calculate(current_state, action, next_state, dones[0])
        infos = self._calculate_infos(current_state, action, rewards)
        return self.normalise_observation(next_state.copy()), self.normalise_rewards(rewards), dones, infos

    def normalise_observation(self, obs: np.ndarray, inverse: bool = False):
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


    # should be something like state[0]=reserves of asset 0, state[1]=reserves of asset 1, state[2]=spot price of AMM
    # and then remaining states depend on
    # the dimensionality of the arrival process, the midprice process
    def _udpate_state(self, actions: np.ndarray):
        pass
        

    def _update_market_state(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        pass
    
    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        pass

    
    def _get_dones(self):
        done = self.model_dynamics.state[0, TIME_INDEX] >= self.terminal_time - self.step_size / 2
        return np.full((self.num_trajectories,), done, dtype=bool)
