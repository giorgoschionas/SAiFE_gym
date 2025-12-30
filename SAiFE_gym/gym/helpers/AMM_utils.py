import numpy as np
import math

# ============================================================================
# Uniswap V3 Concentrated Liquidity Functions
# ============================================================================

def price_to_tick(price: float) -> int:
    """Convert price to tick index"""
    return int(np.floor(np.log(price) / np.log(1.0001)))

def tick_to_price(tick: int) -> float:
    """Convert tick index to price"""
    return 1.0001 ** tick

def get_sqrt_ratio_at_tick(tick: int) -> int:
    price = 1.0001 ** tick
    return np.sqrt(price)

def calculate_liquidity_amounts(
    sqrt_price_current: float,
    sqrt_price_lower: float, 
    sqrt_price_upper: float,
    amount0: float,
    amount1: float
) -> float:
    """Calculate liquidity for given token amounts and price range"""
    if sqrt_price_current <= sqrt_price_lower:
        liquidity = amount0 / (1/sqrt_price_lower - 1/sqrt_price_upper)
    elif sqrt_price_current >= sqrt_price_upper:
        liquidity = amount1 / (sqrt_price_upper - sqrt_price_lower)
    else:
        liquidity0 = amount0 / (1/sqrt_price_current - 1/sqrt_price_upper)
        liquidity1 = amount1 / (sqrt_price_current - sqrt_price_lower)
        liquidity = min(liquidity0, liquidity1)
    
    return liquidity

def get_position_value(L, sqrt_price_current, sqrt_price_lower, sqrt_price_upper):
    """
    Calculates the Mark-to-Market value of a Uniswap v3 position
    in terms of the Quote Asset (Asset Y).

    Args:
        L (float): Liquidity amount
        sqrt_price_current (float): Current sqrt(Price)
        sqrt_price_lower (float): Lower bound sqrt(Price) of the position
        sqrt_price_upper (float): Upper bound sqrt(Price) of the position

    Returns:
        float: Total value in terms of Asset Y
    """
    P = sqrt_price_current ** 2

    # Case 1: Current price is ABOVE the range (Position is 100% Asset Y)
    if sqrt_price_current >= sqrt_price_upper:
        return L * (sqrt_price_upper - sqrt_price_lower)

    # Case 2: Current price is BELOW the range (Position is 100% Asset X)
    elif sqrt_price_current <= sqrt_price_lower:
        # We hold max X, valued at current price P
        # x_max = L * (upper - lower) / (lower * upper)
        return P * L * (sqrt_price_upper - sqrt_price_lower) / (sqrt_price_lower * sqrt_price_upper)

    # Case 3: Current price is IN RANGE (Mix of X and Y)
    else:
        # Derived from V = y + x*P
        return L * (2 * sqrt_price_current - sqrt_price_lower - (P / sqrt_price_upper))



def bucket_bounds_from_center_ids(
    center_ids: np.ndarray,
    tau: int = 5,
    exponential_value: float = 1.0001
) -> tuple[np.ndarray, np.ndarray]:
    """
    center_ids: shape (N,) int
    returns:
      p_low:  shape (N, 2*tau+1)
      p_high: shape (N, 2*tau+1)
    """
    center_ids = np.asarray(center_ids, dtype=np.int64)
    N = center_ids.shape[0]

    # endpoints count is B+1 = (2*tau+1) + 1 = 2*tau+2
    # offsets for endpoints: [-tau, ..., +tau+1]
    endpoint_offsets = np.arange(-tau, tau + 2, dtype=np.int64)  # length 2*tau+2

    # exponents: shape (N, B+1)
    exponents = center_ids[:, None] + endpoint_offsets[None, :]

    # compute endpoints: ev**exponent, vectorized
    log_ev = np.log(exponential_value)
    endpoints = np.exp(exponents.astype(np.float64) * log_ev)  # (N, B+1)

    p_low = endpoints[:, :-1]   # (N, B)
    p_high = endpoints[:, 1:]   # (N, B)
    return p_low, p_high

def find_bucket_id_vec(prices: np.ndarray, exponential_value: float = 1.0001) -> np.ndarray:
    """
    prices: shape (N,) > 0
    returns: shape (N,) int64
    """
    prices = np.asarray(prices, dtype=np.float64)
    prices = np.maximum(prices, 1e-300)  # avoid log(0)
    log_ev = np.log(exponential_value)
    return np.floor(np.log(prices) / log_ev).astype(np.int64)


# Functions for collecting fees
# delta change of amount of Token A
# def delta_x(p1, p2):
#     return 1 / math.sqrt(p2) - 1 / math.sqrt(p1)


# # delta change of amount of Token B
# def delta_y(p1, p2):
#     return math.sqrt(p2) - math.sqrt(p1)

def delta_x_vec(p_high, p_low):
    p_high = np.asarray(p_high, dtype=np.float64)
    p_low = np.asarray(p_low, dtype=np.float64)
    p_high = np.maximum(p_high, 1e-300)
    p_low = np.maximum(p_low, 1e-300)
    return (1.0 / np.sqrt(p_low)) - (1.0 / np.sqrt(p_high))

def delta_y_vec(p_high, p_low):
    """
    Calculate change in Token Y (Token 1) per unit liquidity.

    Formula: Δy/L = sqrt(p_high) - sqrt(p_low)

    Args:
        p_high: Upper prices (regular price, not sqrt), array-like
        p_low: Lower prices (regular price, not sqrt), array-like

    Returns:
        np.ndarray: Delta Y per unit liquidity, same shape as inputs
    """
    p_high = np.asarray(p_high, dtype=np.float64)
    p_low = np.asarray(p_low, dtype=np.float64)
    p_high = np.maximum(p_high, 1e-300)
    p_low = np.maximum(p_low, 1e-300)
    return np.sqrt(p_high) - np.sqrt(p_low)

def swap_step_within_tick_vec(
    sqrt_price_current: np.ndarray | float,
    sqrt_price_target: np.ndarray | float,
    liquidity: np.ndarray | float,
    amount_remaining: np.ndarray | float,
    fee_rate: float,
    zero_for_one: bool
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """
    Compute a single swap step within one tick (L constant, sqrt_price changes).

    This function handles the Uniswap v3 constant product math for swaps that
    occur within a single tick range, where liquidity L remains constant.

    Supports both scalar and array inputs for vectorized processing across trajectories.

    Args:
        sqrt_price_current: Current sqrt(price) - scalar or array
        sqrt_price_target: Target sqrt(price) at tick boundary - scalar or array
        liquidity: Liquidity L in the current tick - scalar or array
        amount_remaining: Remaining input amount (after fees already applied) - scalar or array
        fee_rate: Trading fee rate (e.g., 0.003 for 0.3%)
        zero_for_one: True = selling Token 0 (price decreases),
                      False = buying Token 0 (price increases)

    Returns:
        tuple: (sqrt_price_next, amount_in_consumed, amount_out_generated, fee_consumed)
            - sqrt_price_next: New sqrt(price) after this step
            - amount_in_consumed: Input amount consumed (excluding fee)
            - amount_out_generated: Output amount generated
            - fee_consumed: Fee amount paid on this step
    """
    # Convert inputs to arrays for vectorized operations
    sqrt_price_current = np.atleast_1d(sqrt_price_current)
    sqrt_price_target = np.atleast_1d(sqrt_price_target)
    liquidity = np.atleast_1d(liquidity)
    amount_remaining = np.atleast_1d(amount_remaining)

    # Track which trajectories have liquidity
    has_liquidity = liquidity > 0

    # Guard against numerical issues
    sqrt_price_current = np.maximum(sqrt_price_current, 1e-150)
    sqrt_price_target = np.maximum(sqrt_price_target, 1e-150)

    if zero_for_one:
        # Selling Token 0 -> Price decreases (sqrt_price decreases)
        # Formula: Δx = L * (1/sqrt(P_new) - 1/sqrt(P_current))

        price_current = sqrt_price_current ** 2
        price_target = sqrt_price_target ** 2

        # Calculate maximum input to reach target price
        amount_in_max = delta_x_vec(price_current, price_target) * liquidity

        # Determine if we hit the tick boundary (vectorized)
        hit_boundary = amount_remaining >= amount_in_max

        # Compute sqrt_price_next for both cases
        sqrt_price_boundary = sqrt_price_target
        sqrt_price_no_boundary = np.where(
            has_liquidity,
            (liquidity * sqrt_price_current) / (liquidity + amount_remaining * sqrt_price_current),
            sqrt_price_current
        )

        sqrt_price_next = np.where(hit_boundary, sqrt_price_boundary, sqrt_price_no_boundary)
        amount_in_consumed = np.where(hit_boundary, amount_in_max, amount_remaining)

        # Calculate output amount: Δy = L * (sqrt(P_current) - sqrt(P_new))
        amount_out = liquidity * (sqrt_price_current - sqrt_price_next)

    else:
        # Buying Token 0 -> Price increases (sqrt_price increases)
        # Formula: Δy = L * (sqrt(P_new) - sqrt(P_current))

        price_current = sqrt_price_current ** 2
        price_target = sqrt_price_target ** 2

        # Calculate maximum input to reach target price
        amount_in_max = delta_y_vec(price_target, price_current) * liquidity

        # Determine if we hit the tick boundary (vectorized)
        hit_boundary = amount_remaining >= amount_in_max

        # Compute sqrt_price_next for both cases
        sqrt_price_boundary = sqrt_price_target
        sqrt_price_no_boundary = np.where(
            has_liquidity,
            sqrt_price_current + amount_remaining / liquidity,
            sqrt_price_current
        )

        sqrt_price_next = np.where(hit_boundary, sqrt_price_boundary, sqrt_price_no_boundary)
        amount_in_consumed = np.where(hit_boundary, amount_in_max, amount_remaining)

        # Calculate output amount: Δx = L * (1/sqrt(P_current) - 1/sqrt(P_new))
        # When price increases, Δx from pool perspective is negative (selling X)
        # But output to user is positive, so we use (current - next) order
        price_next = sqrt_price_next ** 2
        amount_out = delta_x_vec(price_next, price_current) * liquidity

    # Apply zero liquidity mask to all outputs
    sqrt_price_next = np.where(has_liquidity, sqrt_price_next, sqrt_price_current)
    amount_in_consumed = np.where(has_liquidity, amount_in_consumed, 0.0)
    amount_out = np.where(has_liquidity, amount_out, 0.0)

    # Calculate fee on the consumed amount
    # fee = amount_in_consumed * fee_rate / (1 - fee_rate)
    # This reverses the initial fee application: amount_after_fee = amount_before_fee * (1 - fee_rate)
    fee = amount_in_consumed * fee_rate / (1.0 - fee_rate) if fee_rate < 1.0 else 0.0

    return sqrt_price_next, amount_in_consumed, amount_out, fee



def execute_swap_vec_array(
    sqrt_price_current: np.ndarray,
    liquidity_array: np.ndarray,
    tick_lower: int,
    amount_in: np.ndarray,
    zero_for_one: bool,
    fee_rate: float = 0.003,
    exponential_value: float = 1.0001,
    sqrt_price_limit: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fully vectorized Uniswap v3 swap processing all trajectories in parallel.

    Args:
        sqrt_price_current: Current sqrt(price) for each trajectory, shape (num_trajectories,)
        liquidity_array: Liquidity per tick, either:
            - shape (num_ticks,): shared liquidity across all trajectories
            - shape (num_trajectories, num_ticks): per-trajectory liquidity
        tick_lower: Lower bound of tick range (offset for array indexing)
        amount_in: Input amount for each trajectory, shape (num_trajectories,)
        zero_for_one: True = selling Token 0, False = buying Token 0
        fee_rate: Trading fee rate (default: 0.003 for 0.3%)
        exponential_value: Tick spacing base (default: 1.0001 for Uniswap v3)
        sqrt_price_limit: Optional price limit, shape (num_trajectories,)

    Returns:
        tuple:
            - sqrt_price_final: Final sqrt(price) for each trajectory, shape (num_trajectories,)
            - amount_out: Total output amount for each trajectory, shape (num_trajectories,)
            - fee_amount: Total fee paid for each trajectory, shape (num_trajectories,)
            - ticks_crossed: Number of ticks crossed for each trajectory, shape (num_trajectories,)
    """

    # Ensure inputs are arrays
    sqrt_price_current = np.atleast_1d(sqrt_price_current)
    amount_in = np.atleast_1d(amount_in)
    num_trajectories = len(sqrt_price_current)

    # Apply fees upfront
    amount_after_fee = amount_in * (1.0 - fee_rate)
    amount_remaining = amount_after_fee.copy()

    # Initialize state arrays
    sqrt_p = sqrt_price_current.copy()
    current_ticks = np.floor(np.log(sqrt_p ** 2) / np.log(exponential_value)).astype(np.int64)

    # Initialize output accumulators
    amount_out_total = np.zeros(num_trajectories, dtype=np.float64)
    fee_total = np.zeros(num_trajectories, dtype=np.float64)
    ticks_crossed = np.zeros(num_trajectories, dtype=np.int64)

    # Active trajectory mask
    active = amount_remaining > 1e-12

    # Pre-compute tick prices for fast lookup
    num_ticks = liquidity_array.shape[-1]
    tick_range = np.arange(tick_lower, tick_lower + num_ticks)
    tick_prices_sqrt = np.sqrt(exponential_value ** tick_range)

    # Determine if liquidity is shared or per-trajectory
    shared_liquidity = liquidity_array.ndim == 1

    # Main iteration loop with active masking
    MAX_ITER = 1000
    for iteration in range(MAX_ITER):
        # Early exit if all trajectories done
        if not active.any():
            break

        # Vectorized liquidity lookup
        tick_indices = current_ticks - tick_lower

        # Bounds checking - treat out-of-bounds as zero  
        out_of_bounds = (tick_indices < 0) | (tick_indices >= num_ticks)

        # Get liquidity for active trajectories
        if shared_liquidity:
            # Shared liquidity across trajectories
            # Use safe indexing - out of bounds gets zero liquidity
            L = np.where(out_of_bounds, 0.0, liquidity_array[np.clip(tick_indices, 0, num_ticks - 1)])
        else:
            # Per-trajectory liquidity
            L = np.where(
                out_of_bounds,
                0.0,
                liquidity_array[np.arange(num_trajectories), np.clip(tick_indices, 0, num_ticks - 1)]
            )

        # Handle zero liquidity (including out-of-bounds): skip to next tick
        zero_liquidity = (L <= 0) & active
        if zero_liquidity.any():
            current_ticks = np.where(
                zero_liquidity,
                current_ticks + (-1 if zero_for_one else 1),
                current_ticks
            )
            ticks_crossed += zero_liquidity.astype(np.int64)
            continue

        # Compute target sqrt prices (vectorized)
        if zero_for_one:
            # Target is lower boundary of current tick
            sqrt_price_target = tick_prices_sqrt[tick_indices]
        else:
            # Target is upper boundary of current tick (next tick's lower boundary)
            next_indices = np.clip(tick_indices + 1, 0, num_ticks - 1)
            sqrt_price_target = tick_prices_sqrt[next_indices]

        # Execute vectorized swap step
        sqrt_p_next, amt_in, amt_out, fee = swap_step_within_tick_vec(
            sqrt_p,
            sqrt_price_target,
            L,
            amount_remaining,
            fee_rate,
            zero_for_one
        )

        # Masked updates (only modify active trajectories)
        amount_remaining = np.where(active, amount_remaining - amt_in, amount_remaining)
        amount_out_total += np.where(active, amt_out, 0.0)
        fee_total += np.where(active, fee, 0.0)
        sqrt_p = np.where(active, sqrt_p_next, sqrt_p)

        # Check tick crossing (did we hit the boundary?)
        tick_crossed = (np.abs(sqrt_p_next - sqrt_price_target) < 1e-10) & active

        # Update ticks and counts
        current_ticks = np.where(
            tick_crossed,
            current_ticks + (-1 if zero_for_one else 1),
            current_ticks
        )
        ticks_crossed += tick_crossed.astype(np.int64)

        # Update active mask
        active = (amount_remaining > 1e-12) & active

        # Check price limit if specified
        if sqrt_price_limit is not None:
            if zero_for_one:
                limit_hit = sqrt_p <= sqrt_price_limit
            else:
                limit_hit = sqrt_p >= sqrt_price_limit
            active = active & ~limit_hit

        # Safety check for runaway swaps
        if iteration == MAX_ITER - 1:
            import warnings
            n_incomplete = active.sum()
            if n_incomplete > 0:
                warnings.warn(
                    f"{n_incomplete}/{num_trajectories} trajectories exceeded "
                    f"{MAX_ITER} iterations. Consider increasing liquidity or "
                    f"reducing swap amounts."
                )

    return sqrt_p, amount_out_total, fee_total, ticks_crossed



# ============================================================================
# Uniswap V4 Hook Functions
# ============================================================================
