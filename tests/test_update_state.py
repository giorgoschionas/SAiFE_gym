"""
Quick test to verify update_state implementation
"""
import numpy as np
from SAiFE_gym.gym.ModelDynamics import UniswapV3ModelDynamics
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from SAiFE_gym.gym.index_names import AMM_PRICE_INDEX, FEES_TOKEN_A_INDEX, FEES_TOKEN_B_INDEX

def test_basic_update_state():
    """Test that update_state runs without errors"""
    print("Testing update_state implementation...")

    num_trajectories = 2

    # Create stochastic processes
    midprice_model = BrownianMotionMidpriceModel(
        drift=0.0,
        volatility=2.0,
        initial_price=100.0,
        terminal_time=1.0,
        step_size=0.005,
        num_trajectories=num_trajectories
    )

    arrival_model = PoissonArrivalModel(
        intensity=np.array([100.0, 100.0]),
        step_size=0.005,
        num_trajectories=num_trajectories
    )

    # Create model dynamics
    dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        tau=5,
        initial_capital=10000.0,
        fee_tier=0.003,
        non_arb_lambda=0.00005
    )

    # Initialize state
    dynamics.state = np.zeros((num_trajectories, 5))
    dynamics.state[:, AMM_PRICE_INDEX] = np.sqrt(100.0)  # sqrt price

    # Create test inputs
    arrivals = np.array([[False, True], [True, False]])  # BUY in traj 0, SELL in traj 1
    action = np.ones((num_trajectories, 11)) / 11.0  # Uniform allocation over 11 buckets

    # Store initial state
    initial_price = dynamics.state[:, AMM_PRICE_INDEX].copy()
    initial_fees_a = dynamics.state[:, FEES_TOKEN_A_INDEX].copy()
    initial_fees_b = dynamics.state[:, FEES_TOKEN_B_INDEX].copy()

    print(f"Initial sqrt price: {initial_price}")
    print(f"Initial fees A: {initial_fees_a}")
    print(f"Initial fees B: {initial_fees_b}")

    # Call update_state
    dynamics.update_state(arrivals, action)

    # Check results
    final_price = dynamics.state[:, AMM_PRICE_INDEX]
    final_fees_a = dynamics.state[:, FEES_TOKEN_A_INDEX]
    final_fees_b = dynamics.state[:, FEES_TOKEN_B_INDEX]

    print(f"\nFinal sqrt price: {final_price}")
    print(f"Final fees A: {final_fees_a}")
    print(f"Final fees B: {final_fees_b}")

    # Verify BUY increased price in trajectory 0
    assert final_price[0] > initial_price[0], "BUY should increase price"
    print(f"✓ BUY increased price: {initial_price[0]:.6f} -> {final_price[0]:.6f}")

    # Verify SELL decreased price in trajectory 1
    assert final_price[1] < initial_price[1], "SELL should decrease price"
    print(f"✓ SELL decreased price: {initial_price[1]:.6f} -> {final_price[1]:.6f}")

    # Verify fees were collected
    total_fees = final_fees_a.sum() + final_fees_b.sum()
    print(f"✓ Total fees collected: {total_fees:.6f}")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_basic_update_state()
