import csv

from experiments.plot_robust_price_evolution import main


def test_plot_robust_price_evolution_smoke(tmp_path):
    rc = main([
        "--output-dir",
        str(tmp_path),
        "--n-steps",
        "5",
        "--seed",
        "123",
    ])

    assert rc == 0
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "price_evolution.png").exists()

    rollout_files = sorted(tmp_path.glob("rollout_sim_*.csv"))
    assert [path.name for path in rollout_files] == ["rollout_sim_001.csv"]
    rollout_path = rollout_files[0]
    with rollout_path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 6
    assert {
        "time",
        "pool_price",
        "midprice",
        "current_tick",
        "sell_arrival",
        "buy_arrival",
        "cumulative_arrivals",
        "reward",
    } <= set(rows[0])
