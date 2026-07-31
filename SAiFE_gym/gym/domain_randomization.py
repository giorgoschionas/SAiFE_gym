from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.gym.index_names import GAS_COST_KEY


@dataclass(frozen=True)
class DomainParameters:
    """Episode-level market parameters used for domain randomization."""

    sigma: float
    arrival_rate: float
    gas_cost: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class UniformDomainRandomizationConfig:
    """Uniform sampling ranges for robust LP training."""

    sigma_range: tuple[float, float] = (1.0, 4.0)
    arrival_rate_range: tuple[float, float] = (50.0, 200.0)
    gas_cost_range: tuple[float, float] = (0.0, 20.0)

    def __post_init__(self) -> None:
        for name, value_range in [
            ("sigma_range", self.sigma_range),
            ("arrival_rate_range", self.arrival_rate_range),
            ("gas_cost_range", self.gas_cost_range),
        ]:
            _validate_nonnegative_range(name, value_range)

    def sample(self, rng: np.random.Generator) -> DomainParameters:
        return DomainParameters(
            sigma=_sample_uniform(rng, self.sigma_range),
            arrival_rate=_sample_uniform(rng, self.arrival_rate_range),
            gas_cost=_sample_uniform(rng, self.gas_cost_range),
        )


class DomainRandomizedAMMEnvironment:
    """Apply episode-level AMM parameter randomization on top of AMMEnvironment.

    The wrapped environment keeps the same observation and action spaces. On each
    reset, one parameter tuple is sampled and shared across all vectorized
    trajectories in that episode.
    """

    def __init__(
        self,
        env: AMMEnvironment,
        config: UniformDomainRandomizationConfig,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.seed_ = seed
        self.last_domain_parameters: Optional[DomainParameters] = None

    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            self.seed(seed)

        params = self.config.sample(self.rng)
        self.apply_domain_parameters(params)
        obs, info = self.env.reset(seed=seed, options=options)
        self._set_state_gas_cost(params.gas_cost)

        self.last_domain_parameters = params
        info = dict(info)
        info["domain_parameters"] = params.to_dict()
        return obs, info

    def step(self, action: np.ndarray):
        return self.env.step(action)

    def seed(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.seed_ = seed
        return self.env.seed(seed)

    def apply_domain_parameters(self, params: DomainParameters) -> None:
        """Apply sampled parameters before an episode reset."""
        md = self.env.model_dynamics

        if not hasattr(md.midprice_model, "volatility"):
            raise AttributeError("midprice_model must expose a volatility attribute")
        md.midprice_model.volatility = float(params.sigma)

        self._set_arrival_rate(float(params.arrival_rate))

        md.gas_cost = float(params.gas_cost)
        self._set_initial_state_gas_cost(float(params.gas_cost))
        self._set_state_gas_cost(float(params.gas_cost))

    def _set_arrival_rate(self, arrival_rate: float) -> None:
        arrival_model = self.env.model_dynamics.arrival_model
        rate = np.array([arrival_rate, arrival_rate], dtype=float)

        if hasattr(arrival_model, "alpha"):
            arrival_model.alpha = np.array(arrival_model.alpha, dtype=float, copy=True)
            arrival_model.alpha[1] = rate
            arrival_model.initial_state = rate.reshape(1, 2)
            arrival_model.current_state = np.ones(
                (self.env.num_trajectories, 2), dtype=float
            ) * rate
            return

        if hasattr(arrival_model, "intensity"):
            arrival_model.intensity = rate
            arrival_model.initial_state = rate.reshape(1, 2)
            arrival_model.current_state = np.ones(
                (self.env.num_trajectories, 2), dtype=float
            ) * rate
            return

        raise AttributeError("arrival_model must expose alpha or intensity")

    def _set_initial_state_gas_cost(self, gas_cost: float) -> None:
        if hasattr(self.env, "_initial_state") and GAS_COST_KEY in self.env._initial_state:
            self.env._initial_state[GAS_COST_KEY] = np.full(
                self.env.num_trajectories, gas_cost, dtype=np.float64
            )

    def _set_state_gas_cost(self, gas_cost: float) -> None:
        state = getattr(self.env.model_dynamics, "state", None)
        if state is not None and GAS_COST_KEY in state:
            state[GAS_COST_KEY] = np.full(
                self.env.num_trajectories, gas_cost, dtype=np.float64
            )

    def __getattr__(self, name):
        return getattr(self.env, name)


def _sample_uniform(
    rng: np.random.Generator,
    value_range: tuple[float, float],
) -> float:
    low, high = value_range
    if low == high:
        return float(low)
    return float(rng.uniform(low, high))


def _validate_nonnegative_range(
    name: str,
    value_range: tuple[float, float],
) -> None:
    if len(value_range) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low, high = value_range
    if low < 0 or high < 0:
        raise ValueError(f"{name} must be non-negative")
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound")
