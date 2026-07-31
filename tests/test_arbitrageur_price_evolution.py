import importlib.util
from pathlib import Path

import numpy as np


def _load_experiment_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "arbitrageur_price_evolution.py"
    spec = importlib.util.spec_from_file_location("arbitrageur_price_evolution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arbitrageur_price_evolution_smoke_run_without_plots():
    module = _load_experiment_module()

    output = module.run_experiment(n_steps=8, seed=11, save_plots=False)

    expected = {
        "HighAlpha3Arrivals",
        "NoiseOnly",
        "NoisePlusSizeArb",
        "NoisePlusSpeedArb",
    }
    assert set(output["results"]) == expected
    assert set(output["summaries"]) == expected

    for data in output["results"].values():
        assert data["time"].shape[0] > 0
        assert data["pool_price"].shape == data["midprice"].shape
        assert data["pool_price"].shape == data["time"].shape
        assert data["arbitrage_action"].shape == data["time"].shape
        assert data["tick_movement"].shape == data["time"].shape

    speed_data = output["results"]["NoisePlusSpeedArb"]
    step_size = module.TERMINAL_TIME / 8
    np.testing.assert_allclose(
        speed_data["arbitrage_order"],
        speed_data["arbitrage_action"] * step_size,
    )
