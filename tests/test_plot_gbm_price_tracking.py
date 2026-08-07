import numpy as np

from experiments.plot_gbm_price_tracking import (
    main,
    parse_args,
    run_experiment,
    run_sweep,
)


def test_gbm_price_tracking_defaults_are_explicit():
    args = parse_args([])

    assert args.n_steps == 1000
    assert args.seed == 42
    assert args.sigmas == [0.01, 0.02, 0.04, 0.08, 0.10]
    assert args.trade_size_notional == 350.0
    assert args.lp_policies == ["hold", "periodic_uniform"]
    assert not args.no_plot
    assert not args.no_csv


def test_gbm_price_tracking_short_run_without_outputs():
    output = run_experiment(
        n_steps=8,
        seed=11,
        volatility=0.02,
        trade_size_notional=350.0,
        save_plot=False,
        save_csv_output=False,
    )
    data = output["data"]
    summary = output["summary"]

    assert output["plot_path"] is None
    assert output["csv_path"] is None
    assert data["time"].shape == (9,)
    assert data["pool_price"].shape == data["midprice"].shape
    assert data["price_gap"].shape == data["time"].shape
    assert np.all(np.isfinite(data["pool_price"]))
    assert np.all(np.isfinite(data["midprice"]))
    assert np.all(data["pool_price"] > 0.0)
    assert np.all(data["midprice"] > 0.0)
    for key in (
        "mean_abs_gap",
        "max_abs_gap",
        "correlation",
        "total_sell_arrivals",
        "total_buy_arrivals",
        "total_arrivals",
        "final_pool_price",
        "final_midprice",
        "final_gap",
    ):
        assert key in summary
    assert summary["total_arrivals"] == (
        summary["total_sell_arrivals"] + summary["total_buy_arrivals"]
    )


def test_gbm_price_tracking_periodic_uniform_short_run_without_outputs():
    output = run_experiment(
        n_steps=8,
        seed=11,
        volatility=0.02,
        trade_size_notional=350.0,
        lp_policy="periodic_uniform",
        save_plot=False,
        save_csv_output=False,
    )

    data = output["data"]
    summary = output["summary"]

    assert output["plot_path"] is None
    assert output["csv_path"] is None
    assert data["time"].shape == (9,)
    assert data["pool_price"].shape == data["midprice"].shape
    assert np.all(np.isfinite(data["pool_price"]))
    assert summary["total_arrivals"] >= 0


def test_gbm_price_tracking_sweep_short_run_without_outputs():
    output = run_sweep(
        sigmas=[0.01, 0.02],
        lp_policies=["hold", "periodic_uniform"],
        n_steps=8,
        seed=11,
        trade_size_notional=350.0,
        save_plot=False,
        save_csv_output=False,
    )

    assert output["plot_path"] is None
    assert output["csv_path"] is None
    assert output["summary_csv_path"] is None
    assert set(output["results"]) == {"hold", "periodic_uniform"}
    for policy_results in output["results"].values():
        assert set(policy_results) == {0.01, 0.02}
        for result in policy_results.values():
            assert result["data"]["time"].shape == (9,)
            assert result["summary"]["total_arrivals"] >= 0


def test_gbm_price_tracking_main_smoke(tmp_path):
    rc = main(
        [
            "--n-steps",
            "8",
            "--seed",
            "12",
            "--sigmas",
            "0.01",
            "0.02",
            "--trade-size-notional",
            "350",
            "--figures-dir",
            str(tmp_path / "figures"),
            "--results-dir",
            str(tmp_path / "results"),
        ]
    )

    assert rc == 0
    assert (
        tmp_path
        / "figures"
        / "gbm_price_tracking_policy_sweep_trade350_seed12.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "results"
        / "gbm_price_tracking_policy_sweep_trade350_seed12.csv"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "results"
        / "gbm_price_tracking_policy_sweep_summary_trade350_seed12.csv"
    ).stat().st_size > 0
