from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
@dataclass(frozen=True)
class DomainParameters:
    """Episode-level market parameters used for domain randomization."""

    sigma: float
    arrival_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BatchedDomainParameters:
    """Episode parameters aligned with vectorized environment trajectories."""

    sigma: np.ndarray
    arrival_rate: np.ndarray
    domain_id: np.ndarray

    def __post_init__(self) -> None:
        sigma = np.asarray(self.sigma, dtype=np.float64)
        arrival_rate = np.asarray(self.arrival_rate, dtype=np.float64)
        domain_id = np.asarray(self.domain_id)

        if sigma.ndim != 2 or sigma.shape[1] != 1 or sigma.shape[0] == 0:
            raise ValueError(
                "sigma must have shape (num_trajectories, 1), "
                f"got {sigma.shape}"
            )
        num_trajectories = sigma.shape[0]
        expected_shapes = {
            "arrival_rate": (num_trajectories, 2),
            "domain_id": (num_trajectories,),
        }
        for name, value in [
            ("arrival_rate", arrival_rate),
            ("domain_id", domain_id),
        ]:
            if value.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} must have shape {expected_shapes[name]}, "
                    f"got {value.shape}"
                )

        for name, value in [
            ("sigma", sigma),
            ("arrival_rate", arrival_rate),
        ]:
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            if np.any(value < 0.0):
                raise ValueError(f"{name} must be non-negative")

        if not np.issubdtype(domain_id.dtype, np.integer):
            raise ValueError("domain_id must contain integers")
        if np.any(domain_id < 0):
            raise ValueError("domain_id must be non-negative")

        object.__setattr__(self, "sigma", sigma.copy())
        object.__setattr__(self, "arrival_rate", arrival_rate.copy())
        object.__setattr__(self, "domain_id", domain_id.astype(np.int64, copy=True))

    @property
    def num_trajectories(self) -> int:
        return self.sigma.shape[0]

    def copy(self) -> "BatchedDomainParameters":
        return BatchedDomainParameters(**self.to_dict())

    def to_dict(self) -> dict:
        return {
            "sigma": self.sigma.copy(),
            "arrival_rate": self.arrival_rate.copy(),
            "domain_id": self.domain_id.copy(),
        }


@dataclass(frozen=True)
class UniformDomainRandomizationConfig:
    """Uniform sampling ranges for domain-randomized LP training."""

    sigma_range: tuple[float, float] = (1.0, 4.0)
    arrival_rate_range: tuple[float, float] = (50.0, 200.0)

    def __post_init__(self) -> None:
        for name, value_range in [
            ("sigma_range", self.sigma_range),
            ("arrival_rate_range", self.arrival_rate_range),
        ]:
            _validate_nonnegative_range(name, value_range)

    def sample(self, rng: np.random.Generator) -> DomainParameters:
        return DomainParameters(
            sigma=_sample_uniform(rng, self.sigma_range),
            arrival_rate=_sample_uniform(rng, self.arrival_rate_range),
        )

    def sample_batch(
        self,
        rng: np.random.Generator,
        num_trajectories: int,
        num_domains: int,
    ) -> BatchedDomainParameters:
        """Sample domains and assign them evenly across shuffled trajectories."""
        _validate_batch_sizes(num_trajectories, num_domains)

        domain_id = np.resize(
            np.arange(num_domains, dtype=np.int64),
            num_trajectories,
        )
        rng.shuffle(domain_id)

        domain_sigma = _sample_uniform_array(
            rng, self.sigma_range, num_domains
        )
        domain_arrival_rate = _sample_uniform_array(
            rng, self.arrival_rate_range, num_domains
        )

        assigned_arrival_rate = domain_arrival_rate[domain_id]
        return BatchedDomainParameters(
            sigma=domain_sigma[domain_id, None],
            arrival_rate=np.repeat(assigned_arrival_rate[:, None], 2, axis=1),
            domain_id=domain_id,
        )


class DomainRandomizedAMMEnvironment:
    """Apply episode-level AMM parameter randomization on top of AMMEnvironment.

    The wrapped environment keeps the same observation and action spaces. On each
    reset, episode parameters are sampled and held fixed. The default keeps one
    parameter tuple shared by every trajectory; ``num_domains > 1`` enables
    balanced per-trajectory domain assignments.
    """

    def __init__(
        self,
        env: AMMEnvironment,
        config: UniformDomainRandomizationConfig,
        seed: Optional[int] = None,
        num_domains: int = 1,
        domain_seed_offset: int = 0,
    ):
        _validate_batch_sizes(env.num_trajectories, num_domains)
        self.env = env
        self.config = config
        self.domain_seed_offset = _validate_domain_seed_offset(domain_seed_offset)
        domain_seed = self._domain_seed(seed)
        self.rng = np.random.default_rng(domain_seed)
        self.seed_ = seed
        self.domain_seed_ = domain_seed
        self.num_domains = int(num_domains)
        self.last_domain_parameters: Optional[
            DomainParameters | BatchedDomainParameters
        ] = None

    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            self.seed(seed)

        if self.num_domains == 1:
            params = self.config.sample(self.rng)
        else:
            params = self.config.sample_batch(
                self.rng,
                self.env.num_trajectories,
                self.num_domains,
            )
        self.apply_domain_parameters(params)
        obs, info = self.env.reset(seed=seed, options=options)

        self.last_domain_parameters = (
            params if isinstance(params, DomainParameters) else params.copy()
        )
        info = dict(info)
        info["domain_parameters"] = params.to_dict()
        return obs, info

    def step(self, action: np.ndarray):
        return self.env.step(action)

    def seed(self, seed: int = None):
        domain_seed = self._domain_seed(seed)
        self.rng = np.random.default_rng(domain_seed)
        self.seed_ = seed
        self.domain_seed_ = domain_seed
        return self.env.seed(seed)

    def _domain_seed(self, seed: Optional[int]) -> Optional[int]:
        if seed is None:
            return None
        return int(seed) + self.domain_seed_offset

    def apply_domain_parameters(
        self,
        params: DomainParameters | BatchedDomainParameters,
    ) -> None:
        """Apply sampled parameters before an episode reset."""
        md = self.env.model_dynamics

        if isinstance(params, DomainParameters):
            if not hasattr(md.midprice_model, "volatility"):
                raise AttributeError(
                    "midprice_model must expose a volatility attribute"
                )
            md.midprice_model.volatility = float(params.sigma)
            episode_volatility = np.full(
                (self.env.num_trajectories, 1),
                params.sigma,
                dtype=np.float64,
            )
            self._set_arrival_rate(float(params.arrival_rate))
        else:
            if not hasattr(md.midprice_model, "set_episode_volatility"):
                raise AttributeError(
                    "midprice_model must support episode volatility"
                )
            if params.num_trajectories != self.env.num_trajectories:
                raise ValueError(
                    "batched domain parameters must match environment "
                    f"num_trajectories={self.env.num_trajectories}, got "
                    f"{params.num_trajectories}"
                )
            episode_volatility = params.sigma
            self._set_batched_arrival_rate(params.arrival_rate)

        if hasattr(md.midprice_model, "set_episode_volatility"):
            md.midprice_model.set_episode_volatility(episode_volatility)

    def _set_arrival_rate(self, arrival_rate: float) -> None:
        arrival_model = self.env.model_dynamics.arrival_model
        rate = np.array([arrival_rate, arrival_rate], dtype=float)
        episode_rate = np.broadcast_to(
            rate,
            (self.env.num_trajectories, 2),
        ).copy()

        if hasattr(arrival_model, "alpha"):
            arrival_model.alpha = np.array(arrival_model.alpha, dtype=float, copy=True)
            arrival_model.alpha[1] = rate
            arrival_model.initial_state = rate.reshape(1, 2)
        elif hasattr(arrival_model, "intensity"):
            arrival_model.intensity = rate
            arrival_model.initial_state = rate.reshape(1, 2)
        else:
            raise AttributeError("arrival_model must expose alpha or intensity")

        if hasattr(arrival_model, "set_episode_baseline_intensity"):
            self._set_batched_arrival_rate(episode_rate)
        else:
            # Compatibility for pre-episode-array custom arrival models in the
            # shared-domain mode.
            arrival_model.current_state = episode_rate

    def _set_batched_arrival_rate(self, arrival_rate: np.ndarray) -> None:
        arrival_model = self.env.model_dynamics.arrival_model
        if not hasattr(arrival_model, "set_episode_baseline_intensity"):
            raise AttributeError(
                "arrival_model must support episode baseline intensity"
            )
        arrival_model.set_episode_baseline_intensity(arrival_rate)
        arrival_model.current_state = (
            arrival_model.episode_baseline_intensity.copy()
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


def _sample_uniform_array(
    rng: np.random.Generator,
    value_range: tuple[float, float],
    size: int,
) -> np.ndarray:
    low, high = value_range
    if low == high:
        return np.full(size, low, dtype=np.float64)
    return rng.uniform(low, high, size=size)


def _validate_batch_sizes(num_trajectories: int, num_domains: int) -> None:
    for name, value in [
        ("num_trajectories", num_trajectories),
        ("num_domains", num_domains),
    ]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if num_trajectories < 1:
        raise ValueError("num_trajectories must be at least 1")
    if not 1 <= num_domains <= num_trajectories:
        raise ValueError(
            "num_domains must satisfy 1 <= num_domains <= num_trajectories"
        )


def _validate_domain_seed_offset(domain_seed_offset: int) -> int:
    if isinstance(domain_seed_offset, bool) or not isinstance(
        domain_seed_offset,
        (int, np.integer),
    ):
        raise ValueError("domain_seed_offset must be an integer")
    return int(domain_seed_offset)


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
