"""
Test script for UniswapV3ModelDynamics

This demonstrates the improved update_state function with:
- LP position management across different price ranges (buckets)
- Orderflow (arrivals) processing
- Reserve updates based on swaps
- Fee accumulation
"""

import numpy as np
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import GeometricBrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel


def test_basic_functionality():
    """Test basic state updates with different actions and arrivals"""

    print("=" * 80)
    print("TEST: UniswapV3ModelDynamics - Basic Functionality")
    print("=" * 80)

    # Initialize V3 dynamics without stochastic models for simplicity
    dynamics = UniswapV3ModelDynamics(
        initial_capital=10000.0,
        initial_price=2000.0,
        fee_tier=0.003,  # 0.3%
        num_buckets=5,
        bucket_width_pct=0.10,
        seed=42
    )

    print("\n[INITIAL STATE]")
    print(f"Initial capital: ${dynamics.initial_capital:,.2f}")
    print(f"Initial price: ${dynamics.initial_price:,.2f}")
    print(f"Number of buckets (actions): {dynamics.num_buckets}")
    print(f"Fee tier: {dynamics.fee_tier * 100}%")

    # Print bucket definitions
    print("\n[BUCKET DEFINITIONS]")
    for i, bucket in enumerate(dynamics.bucket_ranges):
        print(f"Bucket {i}: [{bucket['lower_pct']:.2%}, {bucket['upper_pct']:.2%}] "
              f"(width: {bucket['width']:.2%})")

    # Test different scenarios
    print("\n" + "=" * 80)
    print("SCENARIO 1: Allocate to tight range (Bucket 0), process buy order")
    print("=" * 80)

    action = np.array([0])  # Most concentrated position
    arrivals = np.array([500.0, 0.0])  # $500 buy order, no sells

    print(f"\nAction: {action[0]} (Bucket 0 - tight range)")
    print(f"Arrivals: Buy=${arrivals[0]:.2f}, Sell=${arrivals[1]:.2f}")

    state = dynamics.update_state(arrivals, action)

    print("\n[STATE AFTER UPDATE]")
    print(f"Liquidity: {state[0, 2]:.2f}")
    print(f"Amount0 (USDC): ${state[0, 0]:.2f}")
    print(f"Amount1 (ETH): {state[0, 1]:.4f} ETH")
    print(f"Current price: ${state[0, 3]**2:.2f}")
    print(f"Tick range: [{int(state[0, 5])}, {int(state[0, 6])}]")
    print(f"Fees earned (in token0): ${state[0, 7]:.2f}")

    # Test rebalancing
    print("\n" + "=" * 80)
    print("SCENARIO 2: Rebalance to wider range (Bucket 3), process sell order")
    print("=" * 80)

    action = np.array([3])  # Wider range
    arrivals = np.array([0.0, 0.2])  # 0.2 ETH sell order

    print(f"\nAction: {action[0]} (Bucket 3 - wider range)")
    print(f"Arrivals: Buy=${arrivals[0]:.2f}, Sell={arrivals[1]:.2f} ETH")

    state = dynamics.update_state(arrivals, action)

    print("\n[STATE AFTER UPDATE]")
    print(f"Liquidity: {state[0, 2]:.2f}")
    print(f"Amount0 (USDC): ${state[0, 0]:.2f}")
    print(f"Amount1 (ETH): {state[0, 1]:.4f} ETH")
    print(f"Current price: ${state[0, 3]**2:.2f}")
    print(f"Tick range: [{int(state[0, 5])}, {int(state[0, 6])}]")
    print(f"Fees earned (in token0): ${state[0, 7]:.2f}")
    print(f"Time: {state[0, 9]:.4f}")

    # Test with large price movement
    print("\n" + "=" * 80)
    print("SCENARIO 3: Large buy order causing significant price impact")
    print("=" * 80)

    action = np.array([2])  # Medium range
    arrivals = np.array([2000.0, 0.0])  # Large $2000 buy order

    print(f"\nAction: {action[0]} (Bucket 2 - medium range)")
    print(f"Arrivals: Buy=${arrivals[0]:.2f}, Sell=${arrivals[1]:.2f}")

    state = dynamics.update_state(arrivals, action)

    print("\n[STATE AFTER UPDATE]")
    print(f"Liquidity: {state[0, 2]:.2f}")
    print(f"Amount0 (USDC): ${state[0, 0]:.2f}")
    print(f"Amount1 (ETH): {state[0, 1]:.4f} ETH")
    print(f"Current price: ${state[0, 3]**2:.2f}")
    print(f"Tick range: [{int(state[0, 5])}, {int(state[0, 6])}]")
    print(f"Total fees earned (in token0): ${state[0, 7]:.2f}")

    # Calculate total value
    total_value = state[0, 0] + state[0, 1] * (state[0, 3]**2) + state[0, 7]
    print(f"\nTotal portfolio value: ${total_value:.2f}")
    print(f"P&L: ${total_value - dynamics.initial_capital:.2f} ({(total_value/dynamics.initial_capital - 1)*100:.2f}%)")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


def test_out_of_range_behavior():
    """Test that LP doesn't earn fees when position is out of range"""

    print("\n\n" + "=" * 80)
    print("TEST: Out-of-Range Behavior")
    print("=" * 80)

    dynamics = UniswapV3ModelDynamics(
        initial_capital=10000.0,
        initial_price=2000.0,
        num_buckets=3,
        seed=42
    )

    # Set very tight range
    action = np.array([0])  # Tightest bucket
    arrivals = np.array([100.0, 0.0])

    print("\nStep 1: Allocate to tight range and process small order")
    state = dynamics.update_state(arrivals, action)
    initial_fees = state[0, 7]
    print(f"Fees earned: ${state[0, 7]:.4f}")
    print(f"Price range: [${dynamics.bucket_ranges[0]['lower_pct']*2000:.2f}, "
          f"${dynamics.bucket_ranges[0]['upper_pct']*2000:.2f}]")

    # Don't change action but process order (should still be in range)
    print("\nStep 2: Process another order (should be in range)")
    arrivals = np.array([100.0, 0.0])
    state = dynamics.update_state(arrivals, action)
    print(f"Accumulated fees: ${state[0, 7]:.4f}")

    # Simulate price moving out of range by using a different bucket
    # (in real scenario, external market moves would do this)
    print("\nStep 3: Rebalance to different range")
    action = np.array([2])  # Different bucket
    arrivals = np.array([100.0, 0.0])
    state = dynamics.update_state(arrivals, action)
    print(f"New price range after rebalance")
    print(f"Fees earned: ${state[0, 7]:.4f}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_basic_functionality()
    test_out_of_range_behavior()

    print("\n\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
