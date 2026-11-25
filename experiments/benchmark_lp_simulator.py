"""Time a native LP episode, or hash every transition for exact parity checks."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


def update_digest(digest, fields):
    for name, value in sorted(fields.items()):
        array = np.ascontiguousarray(value)
        digest.update(f"{name} {array.dtype} {array.shape}\n".encode())
        digest.update(memoryview(array).cast("B"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="SAiFE checkout to import, including its trainer factory")
    parser.add_argument("--policy", choices=("hold-heavy", "frequent"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth-window", type=int, help="Swap depth window in tick intervals")
    parser.add_argument("--no-fee-cache", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Hash reset and every step")
    options = parser.parse_args()
    sys.path.insert(0, str(options.root.resolve()))
    from experiments.domain_randomization_rl import train_domain_randomization_lp_agent as trainer

    trainer_options = []
    if options.depth_window is not None:
        if options.depth_window < 1:
            parser.error("--depth-window must be at least 1")
        trainer_options = ["--price-impact-depth-window", str(options.depth_window)]
    args = trainer.parse_args(trainer_options)
    domain = trainer.DomainParameters(args.nominal_sigma, args.nominal_arrival_rate)
    env = trainer.make_fixed_env(args, domain, options.seed)
    if options.no_fee_cache and hasattr(env.model_dynamics, "cache_lp_fees"):
        env.model_dynamics.cache_lp_fees = False

    paths = np.arange(env.num_trajectories)
    actions = np.empty((args.n_steps, env.num_trajectories, 3), dtype=np.float64)
    for step, action in enumerate(actions):
        action[:, 0] = -(1 + (paths + step) % args.tau)
        action[:, 1] = 1 + (paths * 3 + step) % args.tau
        if options.policy == "hold-heavy":
            rebalance = step % args.decision_stride == 0
        else:
            rebalance = (step == 0) | ((paths + step) % 5 == 0)
        action[:, 2] = np.where(rebalance, -1, 1)

    env.reset(seed=options.seed)
    for action in actions[:10]:
        env.step(action)
    state, _ = env.reset(seed=options.seed)
    digest = hashlib.sha256()
    if options.trace:
        update_digest(digest, state)
    total_reward = np.zeros(env.num_trajectories)
    start = perf_counter()
    for action in actions:
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if options.trace:
            update_digest(digest, state)
            update_digest(digest, info)
            update_digest(digest, dict(arrivals=env.model_dynamics.last_arrivals,
                                       terminated=terminated, truncated=truncated, reward=reward))
    elapsed = perf_counter() - start
    assert terminated.all() and not truncated.any()
    update_digest(digest, state)
    update_digest(digest, dict(total_reward=total_reward))
    print(json.dumps(dict(policy=options.policy, seed=options.seed, trace=options.trace,
                          steps=args.n_steps, paths=env.num_trajectories,
                          depth_window=env.model_dynamics.price_impact_model.depth_window,
                          ticks=env.model_dynamics.num_ticks, seconds=elapsed,
                          batch_steps_per_second=args.n_steps / elapsed,
                          sha256=digest.hexdigest())))


if __name__ == "__main__":
    main()
