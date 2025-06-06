import gym
import numpy as np

from gym.spaces import Box
from SAiFE_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel
from SAiFE_gym.stochastic_processes.arrival_models import ArrivalModel, PoissonArrivalModel
from SAiFE_gym.stochastic_processes.midprice_models import MidPriceModel, BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.price_impact_models import PriceImpactModel
from SAiFE_gym.gym.ModelDynamics import ModelDynamics
from SAiFE_gym.agents.Agent import Agent
from SAiFE_gym.rewards import RewardFunction



class AMMEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(
        self, 
        terminal_time: float = 1.0,
        n_steps: int = 20 * 10,
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

    def step(self, action: np.ndarray):
        pass

    def _update_market_state(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        pass
    
    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        pass

    

