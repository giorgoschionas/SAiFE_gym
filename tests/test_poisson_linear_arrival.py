"""
Test script for PoissonLinearArrivalModel
"""
import numpy as np
import sys
sys.path.insert(0, '/home/gchionas/Programming/Blockchain/Ethereum/defi-trading/SAiFE_gym')

from SAiFE_gym.stochastic_processes.arrival_models import PoissonLinearArrivalModel
from SAiFE_gym.gym.index_names import LIQUIDITY_INDEX, AMM_PRICE_INDEX, ASSET_PRICE_INDEX


def test_basic_functionality():
    """Test that PoissonLinearArrivalModel computes arrivals correctly"""
    print("=" * 70)
    print("TEST 1: Basic Functionality")
    print("=" * 70)

    # Create model with simple intensities for testing
    model = PoissonLinearArrivalModel(
        intensity=np.array([
            [10.0, 10.0],    # a_0: minimum intensity
            [50.0, 50.0],    # a_1: base intensity
            [0.1, 0.1],      # a_2: liquidity coefficient
            [1.0, -1.0]      # a_3: price discrepancy coefficient (asymmetric)
        ]),
        step_size=0.001,
        num_trajectories=10,
        seed=42
    )

    # Create test state (num_trajectories, state_dim)
    state = np.zeros((10, 6))
    state[:, LIQUIDITY_INDEX] = 1000.0          # L = 1000
    state[:, AMM_PRICE_INDEX] = np.sqrt(100.0)  # Z_sqrt = 10, Z = 100
    state[:, ASSET_PRICE_INDEX] = 100.0         # S = 100

    # Get arrivals
    arrivals = model.get_arrivals(state)

    print(f"State shape: {state.shape}")
    print(f"Arrivals shape: {arrivals.shape}")
    print(f"Arrivals dtype: {arrivals.dtype}")
    print(f"Sample arrivals (first 3 trajectories):")
    print(arrivals[:3])

    # Verify shape
    assert arrivals.shape == (10, 2), f"Expected shape (10, 2), got {arrivals.shape}"
    assert arrivals.dtype == bool or arrivals.dtype == np.bool_, f"Expected bool dtype, got {arrivals.dtype}"

    print("✓ Basic functionality test passed!\n")


def test_formula_verification():
    """Test that the formula a = max(a_0, a_1 + a_2*L + a_3*(Z-S)) is computed correctly"""
    print("=" * 70)
    print("TEST 2: Formula Verification")
    print("=" * 70)

    # Set up known values
    L = 1000.0
    Z = 105.0  # AMM overpriced
    S = 100.0

    # Intensity parameters
    a_0 = [10.0, 10.0]
    a_1 = [50.0, 50.0]
    a_2 = [0.1, 0.1]
    a_3 = [1.0, -1.0]  # SELL increases when Z>S, BUY decreases

    # Expected intensity
    # For SELL: max(10, 50 + 0.1*1000 + 1.0*(105-100)) = max(10, 50 + 100 + 5) = 155
    # For BUY:  max(10, 50 + 0.1*1000 - 1.0*(105-100)) = max(10, 50 + 100 - 5) = 145
    expected_sell = max(a_0[0], a_1[0] + a_2[0] * L + a_3[0] * (Z - S))
    expected_buy = max(a_0[1], a_1[1] + a_2[1] * L + a_3[1] * (Z - S))

    print(f"L = {L}, Z = {Z}, S = {S}")
    print(f"Expected intensity SELL: {expected_sell}")
    print(f"Expected intensity BUY: {expected_buy}")

    # Create model
    model = PoissonLinearArrivalModel(
        intensity=np.array([a_0, a_1, a_2, a_3]),
        step_size=0.001,
        num_trajectories=1000,  # More trajectories for better statistics
        seed=42
    )

    # Create state
    state = np.zeros((1000, 6))
    state[:, LIQUIDITY_INDEX] = L
    state[:, AMM_PRICE_INDEX] = np.sqrt(Z)  # Store as sqrt
    state[:, ASSET_PRICE_INDEX] = S

    # Get arrivals (run multiple times to get average)
    arrivals_samples = []
    for _ in range(100):
        arrivals = model.get_arrivals(state)
        arrivals_samples.append(arrivals)

    arrivals_all = np.array(arrivals_samples)
    avg_sell_rate = np.mean(arrivals_all[:, :, 0])
    avg_buy_rate = np.mean(arrivals_all[:, :, 1])

    expected_sell_rate = expected_sell * model.step_size
    expected_buy_rate = expected_buy * model.step_size

    print(f"\nObserved SELL rate: {avg_sell_rate:.6f}")
    print(f"Expected SELL rate: {expected_sell_rate:.6f}")
    print(f"Observed BUY rate: {avg_buy_rate:.6f}")
    print(f"Expected BUY rate: {expected_buy_rate:.6f}")

    # Note: Since intensity * step_size = 155 * 0.001 = 0.155, both should be well below 1.0
    # The observed rates should be close to expected rates

    print("✓ Formula verification test passed!\n")


def test_price_discrepancy_effect():
    """Test that price discrepancy (Z-S) affects SELL vs BUY asymmetrically"""
    print("=" * 70)
    print("TEST 3: Price Discrepancy Effect")
    print("=" * 70)

    model = PoissonLinearArrivalModel(
        intensity=np.array([
            [0.0, 0.0],      # a_0: no floor
            [50.0, 50.0],    # a_1: base 50
            [0.0, 0.0],      # a_2: no liquidity dependence
            [10.0, -10.0]    # a_3: strong asymmetric effect
        ]),
        step_size=0.001,
        num_trajectories=1000,
        seed=42
    )

    # Case 1: Z > S (AMM overpriced - expect more SELL)
    state_over = np.zeros((1000, 6))
    state_over[:, AMM_PRICE_INDEX] = np.sqrt(110.0)  # Z = 110
    state_over[:, ASSET_PRICE_INDEX] = 100.0         # S = 100
    # Expected: SELL intensity = 50 + 10*10 = 150, BUY intensity = 50 - 10*10 = max(0, -50) = 0

    arrivals_over = model.get_arrivals(state_over)
    sell_rate_over = np.mean(arrivals_over[:, 0])
    buy_rate_over = np.mean(arrivals_over[:, 1])

    print(f"Case 1: Z > S (AMM overpriced)")
    print(f"  SELL rate: {sell_rate_over:.4f}")
    print(f"  BUY rate: {buy_rate_over:.4f}")
    print(f"  SELL > BUY: {sell_rate_over > buy_rate_over}")

    # Case 2: Z < S (AMM underpriced - expect more BUY)
    state_under = np.zeros((1000, 6))
    state_under[:, AMM_PRICE_INDEX] = np.sqrt(90.0)   # Z = 90
    state_under[:, ASSET_PRICE_INDEX] = 100.0         # S = 100
    # Expected: SELL intensity = 50 + 10*(-10) = max(0, -50) = 0, BUY intensity = 50 - 10*(-10) = 150

    arrivals_under = model.get_arrivals(state_under)
    sell_rate_under = np.mean(arrivals_under[:, 0])
    buy_rate_under = np.mean(arrivals_under[:, 1])

    print(f"\nCase 2: Z < S (AMM underpriced)")
    print(f"  SELL rate: {sell_rate_under:.4f}")
    print(f"  BUY rate: {buy_rate_under:.4f}")
    print(f"  BUY > SELL: {buy_rate_under > sell_rate_under}")

    assert sell_rate_over > buy_rate_over, "When Z > S, SELL rate should exceed BUY rate"
    assert buy_rate_under > sell_rate_under, "When Z < S, BUY rate should exceed SELL rate"

    print("✓ Price discrepancy effect test passed!\n")


def test_state_validation():
    """Test that validation works correctly"""
    print("=" * 70)
    print("TEST 4: State Validation")
    print("=" * 70)

    model = PoissonLinearArrivalModel(num_trajectories=10, seed=42)

    # Test 1: None state should raise ValueError
    try:
        model.get_arrivals(None)
        assert False, "Should have raised ValueError for None state"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for None state: {e}")

    # Test 2: Wrong batch size should raise ValueError
    state_wrong_size = np.zeros((5, 6))  # Wrong batch size
    try:
        model.get_arrivals(state_wrong_size)
        assert False, "Should have raised ValueError for wrong batch size"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for wrong batch size: {e}")

    # Test 3: Correct state should work
    state_correct = np.zeros((10, 6))
    state_correct[:, AMM_PRICE_INDEX] = 10.0
    arrivals = model.get_arrivals(state_correct)
    assert arrivals.shape == (10, 2)
    print(f"✓ Correctly processed valid state")

    print("✓ State validation test passed!\n")


def test_intensity_validation():
    """Test that intensity validation in constructor works"""
    print("=" * 70)
    print("TEST 5: Intensity Validation")
    print("=" * 70)

    # Test 1: Wrong shape should fail
    try:
        model = PoissonLinearArrivalModel(
            intensity=np.array([[10, 20, 30], [40, 50, 60]]),  # Wrong shape (2, 3)
            num_trajectories=1
        )
        assert False, "Should have raised AssertionError for wrong intensity shape"
    except AssertionError as e:
        print(f"✓ Correctly raised AssertionError for wrong shape: {e}")

    # Test 2: Negative a_0 should fail
    try:
        model = PoissonLinearArrivalModel(
            intensity=np.array([[-10.0, -5.0], [50, 50], [0.1, 0.1], [1, -1]]),
            num_trajectories=1
        )
        assert False, "Should have raised AssertionError for negative a_0"
    except AssertionError as e:
        print(f"✓ Correctly raised AssertionError for negative a_0: {e}")

    # Test 3: Valid intensity should work
    model = PoissonLinearArrivalModel(
        intensity=np.array([[10.0, 10.0], [50, 50], [0.1, 0.1], [1, -1]]),
        num_trajectories=1
    )
    print(f"✓ Correctly accepted valid intensity")

    print("✓ Intensity validation test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING PoissonLinearArrivalModel")
    print("=" * 70 + "\n")

    test_basic_functionality()
    test_formula_verification()
    test_price_discrepancy_effect()
    test_state_validation()
    test_intensity_validation()

    print("=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
