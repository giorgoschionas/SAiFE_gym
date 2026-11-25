"""Run two periodic LP ranges under two levels of one market parameter."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.sweep_dense_exponential_width import run_episode


# Start each sweep from one recorded configuration and change one market setting.
SWEEPS = {
    "volatility": (
        "volatility/fixed-width-bg30000/sigma_0.03/configuration.json",
        "nominal_sigma", (0.03, 0.05),
    ),
    "background-liquidity": (
        "background-liquidity/fixed-width/background_30000/configuration.json",
        "background_liquidity", (100_000, 20_000),
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", choices=SWEEPS, nargs="?", default="volatility")
    options = parser.parse_args()
    config_file, parameter, levels = SWEEPS[options.driver]
    config = json.loads((HERE / config_file).read_text())

    print(f"{config['num_trajectories']} paths per policy. Fees, gas and net PnL in token1.")
    for level in levels:
        settings = argparse.Namespace(**(config | {parameter: level}))
        print(f"\nVolatility {settings.nominal_sigma:.3f}, background liquidity {settings.background_liquidity:,.0f}")
        print("range (ticks)  time in range     fees     gas  net PnL", flush=True)
        mean_pnls = []
        for half_width in (15, 25):
            # Episode 0 reuses the evaluation seed. Risk aversion 0 scores net PnL.
            rollout = run_episode(settings, width=half_width, episode=0, risk_aversions=[0.0])
            result = rollout["diagnostics"]
            mean_pnl = result["mean_pnl_per_path"]
            mean_pnls.append(mean_pnl)
            print(
                f"          ±{half_width}  {result['mean_in_range_fraction_among_deployed_paths']:12.1%}"
                f"  {result['mean_fee_income_token1_per_path']:7.2f}"
                f"  {result['mean_gas_spend_per_path']:6.2f}  {mean_pnl:7.2f}",
                flush=True,
            )
        print(f"Widening changes mean net PnL by {mean_pnls[1] - mean_pnls[0]:+.2f} token1.")


if __name__ == "__main__":
    main()
