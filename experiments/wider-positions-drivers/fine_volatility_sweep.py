"""Run and resume a matched-batch periodic LP sweep over a single market driver."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

# Support direct script execution and imports from the experiment package.
if __package__:
    from . import periodic_width_sweep as periodic
else:
    import periodic_width_sweep as periodic


SIGMAS = (0.030, 0.035, 0.040, 0.045, 0.050)
WIDTHS = tuple(range(10, 36))
BATCH_SEEDS = tuple(200042 + periodic.BATCH_SEED_STRIDE * i for i in range(20))
PATHS_PER_BATCH = 100
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 98765


def market_grid(driver):
    """List (sigma, liquidity) pairs in the order in which peaks should widen."""
    if driver == "volatility":
        return [(s, 30_000.0) for s in SIGMAS], WIDTHS
    if driver == "background-liquidity":
        return [(0.07, level) for level in periodic.LIQUIDITY_LEVELS], periodic.LIQUIDITY_WIDTHS
    raise ValueError(f"Unknown market driver: {driver}")


def read_batch(path, seed, driver="volatility"):
    """Check matching path identities before reducing one batch to cell means."""
    markets, widths = market_grid(driver)
    pnl = np.full((len(markets), len(widths), PATHS_PER_BATCH), np.nan)
    market_index = {market: i for i, market in enumerate(markets)}
    width_index = {width: i for i, width in enumerate(widths)}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            market = float(row["sigma"]), float(row["background_liquidity"])
            width = int(row["half_width_ticks"])
            trajectory, value = int(row["path"]), float(row["pnl"])
            if (int(row["episode_seed"]) != seed or market not in market_index or width not in width_index
                    or not 0 <= trajectory < PATHS_PER_BATCH or not np.isfinite(value)):
                raise ValueError(f"Invalid observation in {path}: {row}")
            index = (market_index[market], width_index[width], trajectory)
            if not np.isnan(pnl[index]):
                raise ValueError(f"Duplicate path {trajectory} at market {market}, width {width} in {path}")
            pnl[index] = value
    if np.isnan(pnl).any():
        raise ValueError(f"Incomplete matched batch in {path}")
    return pnl.mean(axis=-1)


def peak_statistics(batch_means, widths, draws, bootstrap_seed):
    """Axes are batch, ordered market level, increasing half-width."""
    widths = np.asarray(widths)
    n_batches, n_levels, n_widths = batch_means.shape
    # One draw resamples whole batches jointly across all markets and widths.
    rng = np.random.default_rng(bootstrap_seed)
    counts = rng.multinomial(n_batches, np.full(n_batches, 1 / n_batches), size=draws)
    draw_means = (counts @ batch_means.reshape(n_batches, -1)) / n_batches
    draw_peaks = widths[draw_means.reshape(draws, n_levels, n_widths).argmax(axis=-1)]
    pooled_peaks = widths[batch_means.mean(axis=0).argmax(axis=-1)]
    batch_peaks = widths[batch_means.argmax(axis=-1)]
    return pooled_peaks, batch_peaks, draw_peaks


def strictly_increasing(peaks):
    return np.all(np.diff(peaks, axis=-1) > 0, axis=-1)


def write_analysis(directory, batch_means, driver, levels, widths):
    parameter = periodic.PARAMETERS[driver]
    pooled, by_batch, bootstrap = peak_statistics(batch_means, widths, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    batch_strict, bootstrap_strict = strictly_increasing(by_batch), strictly_increasing(bootstrap)
    score = batch_means.mean(axis=0)
    rows = [periodic.grid_row(driver, level, width, len(BATCH_SEEDS) * PATHS_PER_BATCH,
                              score[i, j], bootstrap_fraction=float(np.mean(bootstrap[:, i] == width)))
            for i, level in enumerate(levels) for j, width in enumerate(widths)]
    grid = periodic.ranked_grid(rows)
    periodic.write_grid(directory / "width_grid.csv", grid)
    periodic.plot_widths({driver: grid}, directory / "width_curves.png")
    with (directory / "batch_peaks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        precision = ".3f" if driver == "volatility" else "g"
        columns = [f"half_width_{parameter}_{level:{precision}}" for level in levels]
        writer.writerow(["episode_seed", *columns, "strictly_increasing"])
        for seed, peaks, strict in zip(BATCH_SEEDS, by_batch, batch_strict):
            writer.writerow([seed, *peaks, bool(strict)])
    summary = {
        parameter: list(levels), "pooled_peak_half_widths": pooled.tolist(),
        "pooled_strictly_increasing": bool(strictly_increasing(pooled)),
        "batches": len(by_batch), "strict_batches": int(batch_strict.sum()),
        "strict_batch_fraction": float(batch_strict.mean()),
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": len(bootstrap),
        "strict_bootstrap_draws": int(bootstrap_strict.sum()),
        "strict_bootstrap_fraction": float(bootstrap_strict.mean()),
        "bootstrap_fraction_increasing_by_adjacent_pair": (np.diff(bootstrap, axis=1) > 0).mean(axis=0).tolist(),
        "tie_rule": "Choose the smallest half-width attaining the maximum mean net PnL.",
    }
    (directory / "peak_order.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Pooled peak half-widths {pooled.tolist()} in {parameter} order {list(levels)}.\n"
          f"Strict ordering in {batch_strict.sum()}/{len(by_batch)} observed batches and "
          f"{bootstrap_strict.sum()}/{len(bootstrap)} paired bootstrap resamples "
          f"({bootstrap_strict.mean():.1%}).", flush=True)


def run(directory, workers, driver="volatility"):
    markets, widths = market_grid(driver)
    levels = [s if driver == "volatility" else liquidity for s, liquidity in markets]
    config, _, _, _ = periodic.sweep_defaults(driver)
    config.update(nominal_sigma=markets[0][0], background_liquidity=markets[0][1],
                  tau=max(config["tau"], max(widths)), num_trajectories=PATHS_PER_BATCH,
                  n_eval_episodes=1, evaluation_seed=BATCH_SEEDS[0])
    settings = {"base_environment": config, periodic.PARAMETERS[driver]: levels, "half_width_ticks": list(widths),
                "batch_seeds": list(BATCH_SEEDS), "policy": "PeriodicRebalanceAgent", "objective": "mean_net_pnl"}
    batches = directory / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as scratch:
        work = Path(scratch)
        config_path = directory / "config.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != settings:
                raise ValueError(f"Sweep configuration differs from {config_path}")
        else:
            if any(batches.glob("*.csv")):
                raise ValueError(f"Missing {config_path} for saved batches")
            (work / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
            (work / "config.json").replace(config_path)

        means = []
        for number, seed in enumerate(BATCH_SEEDS, start=1):
            path = batches / f"seed_{seed}.csv"
            if path.exists():
                print(f"Loading batch {number}/{len(BATCH_SEEDS)} from {path}", flush=True)
                means.append(read_batch(path, seed, driver))
            else:
                print(f"Simulating batch {number}/{len(BATCH_SEEDS)} with seed {seed}", flush=True)
                periodic.simulate(driver, dict(config, evaluation_seed=seed), levels, widths, workers, work)
                raw = work / "path_pnl.csv"
                means.append(read_batch(raw, seed, driver))
                raw.replace(path)
        write_analysis(work, np.stack(means), driver, levels, widths)
        for artifact in ("width_grid.csv", "width_curves.png", "batch_peaks.csv", "peak_order.json"):
            (work / artifact).replace(directory / artifact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", choices=("volatility", "background-liquidity"), default="volatility")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args.output_dir, args.workers, args.driver)


if __name__ == "__main__":
    main()
