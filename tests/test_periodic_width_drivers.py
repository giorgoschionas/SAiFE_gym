"""Check which inputs the periodic sweeps vary and which payoffs they rank."""

import csv
from importlib import import_module
from itertools import product

import numpy as np
import pytest

from experiments import sweep_dense_exponential_width as rollout

sweep = import_module("experiments.wider-positions-drivers.periodic_width_sweep")


@pytest.mark.parametrize("driver", ["volatility", "background-liquidity", "risk-aversion"])
@pytest.mark.parametrize("historical_seeds", [False, True])
def test_paired_markets_and_risk_replay(monkeypatch, tmp_path, driver, historical_seeds):
    config = dict(nominal_sigma=0.03, background_liquidity=30000, evaluation_seed=123,
                  num_trajectories=2, n_eval_episodes=2)
    if historical_seeds:
        config["evaluation_seed_stride"] = 1
    levels = {"volatility": [0.03, 0.05], "background-liquidity": [20000, 30000],
              "risk-aversion": [0, 0.4]}[driver]
    widths = [2, 5, 10]
    calls = []

    def run_episode(settings, width, episode, risks):
        seed = settings.evaluation_seed + episode
        calls.append((settings.nominal_sigma, settings.background_liquidity, width, seed))
        pnl = np.array({2: [8, -3], 5: [2, 2], 10: [0, 0]}[width], dtype=float) + episode
        rewards = np.stack([pnl if a == 0 else -np.expm1(-a * pnl) / a for a in risks])
        return {"pnl": pnl, "total_rewards": rewards, "seed": seed}

    monkeypatch.setattr(rollout, "run_episode", run_episode)
    rows = sweep.simulate(driver, config, levels, widths, 1, tmp_path)
    sigmas = levels if driver == "volatility" else [0.03]
    backgrounds = levels if driver == "background-liquidity" else [30000]
    expected_seeds = [123, 124] if historical_seeds else [123, 133]
    assert sorted(calls) == sorted(product(sigmas, backgrounds, widths, expected_seeds))
    with (tmp_path / "path_pnl.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == len(calls) * 2
    assert len(rows) == len(levels) * len(widths)
    best = {r["parameter_value"]: r for r in rows if r["peak"]}
    assert best[levels[0]]["half_width_ticks"] == 2
    assert best[levels[0]]["peak"] == "lower_edge"
    if driver == "risk-aversion":
        assert best[0]["certainty_equivalent_pnl"] == pytest.approx(best[0]["mean_pnl"])
        assert best[0.4]["half_width_ticks"] == 5
        assert best[0.4]["peak"] == "interior"
        for width in widths:
            assert len({r["mean_pnl"] for r in rows if r["half_width_ticks"] == width}) == 1
    else:
        assert best[levels[1]]["half_width_ticks"] == 2
