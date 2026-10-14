"""LP decisions and their stable categorical/raw execution encodings.

The public decision is Hold | Rebalance(lower, upper). Raw three-column
commands are an execution/compatibility format, not three independent choices.
"""

import gymnasium as gym
import numpy as np


CATEGORICAL_ENCODING = "categorical_endpoint_pairs_v1"
STRUCTURED_ENCODING = "structured_center_half_width_v1"
HOLD_COMMAND = np.array([0.0, 1.0, 1.0], dtype=np.float32)


def _validate_action_table_args(tau: int, tick_stride: int) -> tuple[int, int]:
    for name, value in (("tau", tau), ("tick_stride", tick_stride)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(tau), int(tick_stride)


def _offset_grid(tau: int, tick_stride: int) -> np.ndarray:
    offsets = np.arange(-tau, tau + 1, tick_stride, dtype=np.int64)
    if offsets[-1] != tau:
        offsets = np.append(offsets, tau)
    return offsets


def _half_width_grid(tau: int, tick_stride: int) -> np.ndarray:
    widths = np.arange(tick_stride, tau + 1, tick_stride, dtype=np.int64)
    if not widths.size or widths[-1] != tau:
        widths = np.append(widths, tau)
    return widths


def hold_action() -> tuple[int, tuple]:
    """The unique parameter-free hold decision."""
    return (0, ())


def rebalance_action(lower: int, upper: int) -> tuple[int, np.ndarray]:
    """Construct ordered integer offsets; the environment checks its tau bounds."""
    bounds = np.asarray([lower, upper])
    if bounds.dtype.kind not in "iu" or bounds.shape != (2,) or lower >= upper:
        raise ValueError("rebalance requires two integer offsets with lower < upper")
    return (1, bounds.astype(np.int64))


class RebalanceRangeSpace(gym.spaces.Space):
    """Ordered integer tick pairs, sampled uniformly over all valid ranges."""

    def __init__(self, tau: int, seed=None):
        self.tau, _ = _validate_action_table_args(tau, 1)
        super().__init__(shape=(2,), dtype=np.int64, seed=seed)

    @property
    def is_np_flattenable(self):
        return False

    def sample(self, mask=None, probability=None):
        if mask is not None or probability is not None:
            raise NotImplementedError("Range sampling does not support masks or probabilities")
        return np.sort(self.np_random.choice(2 * self.tau + 1, size=2, replace=False)) - self.tau

    def contains(self, x):
        try:
            bounds = np.asarray(x)
            return bool(
                bounds.shape == (2,) and bounds.dtype.kind in "iu"
                and -self.tau <= bounds[0] < bounds[1] <= self.tau
            )
        except (TypeError, ValueError):
            return False

    def to_jsonable(self, sample_n):
        return [np.asarray(sample).tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        return [np.asarray(sample, dtype=np.int64) for sample in sample_n]

    def __eq__(self, other):
        return isinstance(other, RebalanceRangeSpace) and self.tau == other.tau

    def __repr__(self):
        return f"RebalanceRange(-{self.tau} <= lower < upper <= {self.tau})"


class LPActionSpace(gym.spaces.OneOf):
    """Hold ``(0, ())`` or Rebalance ``(1, [lower, upper])``."""

    def __init__(self, tau: int, seed=None):
        super().__init__((gym.spaces.Tuple(()), RebalanceRangeSpace(tau)), seed=seed)

    def __repr__(self):
        return f"LPActionSpace(Hold, {self.spaces[1]!r})"

    def contains(self, action):
        return bool(
            isinstance(action, tuple) and len(action) == 2
            and isinstance(action[0], (int, np.integer))
            and not isinstance(action[0], (bool, np.bool_))
            and 0 <= action[0] < 2
            and self.spaces[action[0]].contains(action[1])
        )

    def to_jsonable(self, sample_n):
        # Gymnasium's generic OneOf serializer cannot serialize an empty Tuple.
        if not all(self.contains(action) for action in sample_n):
            raise ValueError("Expected valid Hold/Rebalance decisions")
        return [[int(mode), [] if mode == 0 else np.asarray(bounds).tolist()]
                for mode, bounds in sample_n]

    def from_jsonable(self, sample_n):
        actions = [(mode, () if mode == 0 and bounds == [] else np.asarray(bounds))
                   for mode, bounds in sample_n]
        if not all(self.contains(action) for action in actions):
            raise ValueError("Expected valid serialized Hold/Rebalance decisions")
        return [(np.int64(mode), bounds) for mode, bounds in actions]


def normalize_lp_actions(action, num_trajectories: int, space: LPActionSpace):
    """Normalize tagged decisions or legacy raw inputs to batched commands.

    Tagged Python objects are unpacked only at this API boundary. Categorical
    decoding and all subsequent simulation work use vectorized NumPy arrays.
    """
    if action is None:
        return None
    is_single_tag = (
        isinstance(action, tuple) and len(action) == 2
        and isinstance(action[0], (int, np.integer))
        and not np.isscalar(action[1])
    )
    is_tagged_batch = (
        isinstance(action, (tuple, list)) and len(action) > 0
        and isinstance(action[0], tuple) and len(action[0]) == 2
        and not np.isscalar(action[0][1])
    )
    if is_single_tag or is_tagged_batch:
        decisions = (action,) if is_single_tag else action
        if len(decisions) != num_trajectories or not all(space.contains(a) for a in decisions):
            raise ValueError(f"expected {num_trajectories} valid Hold/Rebalance decisions")
        modes, payloads = zip(*decisions)
        rebalance = np.asarray(modes) == 1
        bounds = np.asarray([p if m == 1 else (0, 1) for m, p in zip(modes, payloads)])
        return np.column_stack((bounds, np.where(rebalance, -1.0, 1.0))).astype(np.float32)

    action = np.asarray(action)
    if num_trajectories == 1 and action.shape == (3,):
        action = action.reshape(1, 3)
    if action.ndim != 2 or action.shape[0] != num_trajectories or action.shape[1] not in (2, 3):
        raise ValueError(
            f"expected batched action shape ({num_trajectories}, 3) "
            f"(or legacy ({num_trajectories}, 2)), got {action.shape}"
        )
    return action


def build_discrete_action_table(tau: int, tick_stride: int = 1) -> np.ndarray:
    """Hold first, then endpoint pairs ordered by lower and then upper offset."""
    tau, tick_stride = _validate_action_table_args(tau, tick_stride)
    offsets = _offset_grid(tau, tick_stride)
    lower, upper = np.triu_indices(len(offsets), k=1)
    ranges = np.column_stack((offsets[lower], offsets[upper], -np.ones(len(lower))))
    return np.vstack((HOLD_COMMAND, ranges)).astype(np.float32)


class CategoricalLPActionCodec:
    """One categorical ID per distinct LP decision, with vectorized decoding."""

    encoding = CATEGORICAL_ENCODING

    def __init__(self, tau: int, tick_stride: int = 1):
        self.tau, self.tick_stride = _validate_action_table_args(tau, tick_stride)
        self.action_table = build_discrete_action_table(self.tau, self.tick_stride)
        self.action_table.setflags(write=False)
        self.action_space = gym.spaces.Discrete(len(self.action_table))

    @property
    def metadata(self):
        return dict(action_encoding=self.encoding, tau=self.tau,
                    tick_stride=self.tick_stride, action_count=int(self.action_space.n))

    def decode(self, actions):
        indices = np.asarray(actions)
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= self.action_space.n):
            raise ValueError(f"action IDs must be integers in [0, {self.action_space.n - 1}]")
        return self.action_table[indices].copy()


def validate_policy_metadata(model, tau: int, tick_stride: int):
    """Check saved grid parameters even when two different grids have equal counts."""
    saved = getattr(model, "lp_action_config", None)
    if saved is not None:
        expected = CategoricalLPActionCodec(tau, tick_stride).metadata
        if saved != expected:
            raise ValueError("Saved policy action metadata does not match the configured endpoint grid")


def decode_policy_actions(actions, action_space, tau: int, tick_stride: int = 1):
    """Decode current categorical or explicitly recognized historical policies."""
    tau, tick_stride = _validate_action_table_args(tau, tick_stride)
    if isinstance(action_space, gym.spaces.Discrete):
        codec = CategoricalLPActionCodec(tau, tick_stride)
        if action_space != codec.action_space:
            raise ValueError("Policy action space does not match the configured endpoint grid")
        return codec.decode(actions)
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        centers = _offset_grid(tau, tick_stride)
        widths = _half_width_grid(tau, tick_stride)
        if not np.array_equal(action_space.nvec, [len(centers), len(widths), 2]):
            raise ValueError("Legacy policy action space does not match the configured structured grid")
        indices = np.asarray(actions)
        if indices.dtype.kind not in "iu" or indices.shape[-1:] != (3,) or np.any(indices < 0) or np.any(indices >= action_space.nvec):
            raise ValueError("Invalid structured action indices")
        center, width = centers[indices[..., 0]], widths[indices[..., 1]]
        return np.stack((
            np.clip(center - width, -tau, tau - 1),
            np.clip(center + width, -tau + 1, tau),
            np.where(indices[..., 2] == 0, -1, 1),
        ), axis=-1).astype(np.float32)
    if isinstance(action_space, gym.spaces.Box):
        return np.asarray(actions)
    raise ValueError(f"Unsupported policy action space: {action_space}")
