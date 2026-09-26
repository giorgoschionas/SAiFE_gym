"""Vectorized gas costs, expressed in token1 units per LP rebalance."""

from numbers import Real
from typing import Optional

import numpy as np

from SAiFE_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


class GasCostModel(StochasticProcessModel):
    """Base process with nonnegative costs derived from its latent state."""

    @property
    def current_gas_cost(self) -> np.ndarray:
        """Return one cost per trajectory without clipping the latent state."""
        return np.maximum(self.current_state[:, 0], 0.0)


class OrnsteinUhlenbeckGasCostModel(GasCostModel):
    """Exact transitions of dX = theta * (mu - X) dt + sigma dW.

    ``theta`` is the positive mean-reversion rate (inverse simulation time),
    ``mu`` is the latent long-term mean in token1, and ``sigma`` is volatility
    in token1 per square root of simulation time. Each trajectory has its own
    Gaussian noise. The latent state can be negative; charged gas is max(X, 0).
    Consequently, the mean charged cost can exceed ``mu``.

    Reset restores ``initial_cost`` without rewinding the RNG. Call ``seed``
    as well to replay a path. Arrivals, fills, actions, and pool state do not
    influence this exogenous process.
    """

    def __init__(
        self,
        theta: float,
        mu: float,
        sigma: float,
        initial_cost: Optional[float] = None,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        initial_cost = mu if initial_cost is None else initial_cost
        parameters = {
            "theta": theta,
            "mu": mu,
            "sigma": sigma,
            "initial_cost": initial_cost,
            "terminal_time": terminal_time,
            "step_size": step_size,
        }
        for name, value in parameters.items():
            if not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
            if name in ("theta", "terminal_time", "step_size"):
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
            elif value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            isinstance(num_trajectories, (bool, np.bool_))
            or not isinstance(num_trajectories, (int, np.integer))
            or num_trajectories < 1
        ):
            raise ValueError("num_trajectories must be a positive integer")

        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        super().__init__(
            min_value=np.array([[-np.inf]]),
            max_value=np.array([[np.inf]]),
            step_size=float(step_size),
            terminal_time=float(terminal_time),
            initial_state=np.array([[initial_cost]], dtype=np.float64),
            num_trajectories=int(num_trajectories),
            seed=seed,
        )

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        action: np.ndarray,
        state: Optional[dict] = None,
    ) -> None:
        decay = np.exp(-self.theta * self.step_size)
        variance_factor = -np.expm1(-2.0 * self.theta * self.step_size) / (2.0 * self.theta)
        self.current_state = (
            self.mu
            + (self.current_state - self.mu) * decay
            + self.sigma * np.sqrt(variance_factor)
            * self.rng.normal(size=(self.num_trajectories, 1))
        )
