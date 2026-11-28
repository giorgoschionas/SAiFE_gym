"""Check the matched bootstrap and the checkpoint-to-plot workflow."""

import csv
from importlib import import_module
import json

import numpy as np
import pytest

sweep = import_module("experiments.wider-positions-drivers.fine_volatility_sweep")


def test_peak_frequency_resamples_one_batch_selection_for_every_cell():
    # Choosing batch A twice gives peaks [10, 11, 12], B twice [11, 12, 12].
    # A+B ties the first two widths in market 1 and the last two in market 2.
    # Narrower wins both exact ties, so A+B also has strict peaks [10, 11, 12].
    means = np.array([[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                      [[0, 2, 0], [0, 0, 2], [0, 0, 2]]])
    pooled, observed, bootstrap = sweep.peak_statistics(means, [10, 11, 12], 500, 43)
    np.testing.assert_array_equal(pooled, [10, 11, 12])
    np.testing.assert_array_equal(observed, [[10, 11, 12], [11, 12, 12]])
    np.testing.assert_array_equal(sweep.strictly_increasing(observed), [True, False])
    counts = np.random.default_rng(43).multinomial(2, [0.5, 0.5], size=500)
    np.testing.assert_array_equal(sweep.strictly_increasing(bootstrap), counts[:, 0] > 0)
    np.testing.assert_array_equal(bootstrap, np.where(counts[:, :1] > 0, observed[0], observed[1]))


def test_fine_sweep_design():
    assert sweep.SIGMAS == (0.03, 0.035, 0.04, 0.045, 0.05)
    assert sweep.WIDTHS == tuple(range(10, 36))
    assert len(sweep.BATCH_SEEDS) == 20 and min(np.diff(sweep.BATCH_SEEDS)) > 4
    assert sweep.PATHS_PER_BATCH == 100 and sweep.BOOTSTRAP_DRAWS == 10000


def test_simulate_analyze_and_resume_real_batches(tmp_path, monkeypatch):
    config, *_ = sweep.periodic.sweep_defaults("volatility")
    config.update(n_steps=10, terminal_time=0.01, periodic_rebalance_every=5)
    monkeypatch.setattr(sweep.periodic, "sweep_defaults", lambda driver: (config, None, None, None))
    monkeypatch.setattr(sweep, "SIGMAS", (0.03, 0.05))
    monkeypatch.setattr(sweep, "WIDTHS", (10, 11, 12))
    monkeypatch.setattr(sweep, "BATCH_SEEDS", (200042, 200052))
    monkeypatch.setattr(sweep, "PATHS_PER_BATCH", 3)
    monkeypatch.setattr(sweep, "BOOTSTRAP_DRAWS", 100)
    sweep.run(tmp_path, 1)

    settings = json.loads((tmp_path / "config.json").read_text())
    assert settings["base_environment"]["background_liquidity"] == 30000
    summary = json.loads((tmp_path / "peak_order.json").read_text())
    with (tmp_path / "width_grid.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 and {r["n_paths"] for r in rows} == {"6"}
    assert summary["pooled_peak_half_widths"] == [int(r["half_width_ticks"]) for r in rows if r["peak"]]
    with (tmp_path / "batch_peaks.csv").open() as stream:
        peaks = list(csv.DictReader(stream))
    assert [int(r["episode_seed"]) for r in peaks] == list(sweep.BATCH_SEEDS)
    assert sum(r["strictly_increasing"] == "True" for r in peaks) == summary["strict_batches"]
    assert summary["strict_bootstrap_fraction"] == summary["strict_bootstrap_draws"] / 100
    assert (tmp_path / "width_curves.png").read_bytes().startswith(b"\x89PNG")
    monkeypatch.setattr(sweep.periodic, "simulate", lambda *args: pytest.fail("recomputed a saved batch"))
    sweep.run(tmp_path, 1)
    assert json.loads((tmp_path / "peak_order.json").read_text()) == summary
    path = tmp_path / "batches/seed_200052.csv"
    with pytest.raises(ValueError, match="Invalid observation"):
        sweep.read_batch(path, seed=200042)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="Incomplete matched batch"):
        sweep.run(tmp_path, 1)
