"""Sampling isolation and policy statistics for the action-encoding experiment."""

from contextlib import contextmanager
import random

import numpy as np
import torch
from gymnasium import spaces

from SAiFE_gym.gym.lp_actions import policy_action_codec, action_grid_from_metadata


@contextmanager
def evaluation_random_state(model):
    """Restore Python, NumPy, Torch (including CUDA), and policy training mode."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    training = model.policy.training
    # fork_rng saves the CPU generator and all initialized CUDA generators.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            model.policy.set_training_mode(training)


def action_sampling_seed(episode_seed):
    """A separate reproducible action stream, independent of the training seed."""
    return int(np.random.SeedSequence([episode_seed, 0xAC710]).generate_state(1)[0])


def economic_action_probabilities(distribution, action_space):
    """Hold first, followed by center-major ranges; merge every Hold alias."""
    if isinstance(action_space, spaces.Discrete):
        return distribution.distribution.probs
    center, width, hold = [head.probs for head in distribution.distribution]
    rebalance = center[:, :, None] * width[:, None, :] * hold[:, 0, None, None]
    return torch.cat((hold[:, 1, None], rebalance.flatten(start_dim=1)), dim=1)


def rollout_action_diagnostics(model, args):
    """Use the collected decision observations/actions before the PPO update.

    Forced simulator holds never enter SB3's rollout buffer. Its observations
    are already normalized using the statistics in force at collection time.
    """
    buffer = model.rollout_buffer
    observations = buffer.observations.reshape((-1, *model.observation_space.shape))
    actions = buffer.actions.astype(np.int64)
    codec = policy_action_codec(args.tau, args.tick_stride,
                                action_grid=action_grid_from_metadata(model.lp_action_config),
                                policy_action_space=args.policy_action_space)
    if isinstance(model.action_space, spaces.MultiDiscrete):
        actions = actions.reshape(-1, 3)
        economic_ids = codec.economic_action_ids(actions)
        encoded_ids = np.ravel_multi_index(actions.T, model.action_space.nvec)
    else:
        economic_ids = encoded_ids = actions.ravel()
    count = len(economic_ids)
    economic_count = codec.metadata.get("economic_action_count", codec.metadata["action_count"])
    frequencies = np.bincount(economic_ids, minlength=economic_count)
    probability_sum = np.zeros(economic_count)
    raw_entropy = economic_entropy = 0.0
    with torch.no_grad():
        for start in range(0, count, 4096):
            obs = torch.as_tensor(observations[start:start + 4096], device=model.device)
            distribution = model.policy.get_distribution(obs)
            probabilities = economic_action_probabilities(distribution, model.action_space)
            probability_sum += probabilities.sum(dim=0).cpu().numpy()
            economic_entropy += float((-torch.special.xlogy(probabilities, probabilities)).sum().cpu())
            raw_entropy += float(distribution.entropy().sum().cpu())
    return dict(
        sampled_agent_decisions=count,
        forced_simulator_holds=count * (args.decision_stride - 1),
        sampled_hold_fraction=float(frequencies[0] / count),
        sampled_rebalance_fraction=float(1 - frequencies[0] / count),
        predicted_hold_probability=float(probability_sum[0] / count),
        economic_action_entropy=economic_entropy / count,
        raw_action_entropy=raw_entropy / count,
        economic_action_counts=frequencies.tolist(),
        economic_action_frequencies=(frequencies / count).tolist(),
        predicted_economic_action_probabilities=(probability_sum / count).tolist(),
        encoded_action_counts=np.bincount(encoded_ids, minlength=codec.metadata["action_count"]).tolist(),
    )
