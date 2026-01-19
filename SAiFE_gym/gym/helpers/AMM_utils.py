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



def get_tick_boundaries(
    sqrt_price_current: np.ndarray,
    exponential_value: float = 1.0001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get lower and upper sqrt price boundaries for the current tick.

    Args:
        sqrt_price_current: Current sqrt(price) per trajectory, shape (num_trajectories,)
        exponential_value: Tick spacing base (default: 1.0001 for Uniswap v3)

    Returns:
        tuple:
            - tick_lower_sqrt: sqrt(price) at lower tick boundary
            - tick_upper_sqrt: sqrt(price) at upper tick boundary
            - current_tick: Current tick index
    """
    sqrt_price_current = np.atleast_1d(sqrt_price_current)
    price_current = sqrt_price_current ** 2

    # Avoid log of zero/negative
    price_current = np.maximum(price_current, 1e-300)

    current_tick = np.floor(np.log(price_current) / np.log(exponential_value)).astype(np.int64)

    # Lower boundary: exponential_value^tick
    tick_lower_sqrt = np.sqrt(exponential_value ** current_tick)

    # Upper boundary: exponential_value^(tick+1)
    tick_upper_sqrt = np.sqrt(exponential_value ** (current_tick + 1))

    return tick_lower_sqrt, tick_upper_sqrt, current_tick


def unified_swap_single_tick(
    sqrt_price_current: np.ndarray,
    liquidity: np.ndarray,
    arrivals: np.ndarray,
    tick_lower_boundary: np.ndarray,
    tick_upper_boundary: np.ndarray,
    fee_rate: float = 0.003,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified single-tick swap with two-column arrivals

    This function processes swaps that cross at most one tick, using a multiplier/indexing
    approach instead of conditional branching on swap direction.

    Mathematical basis:
    - Selling token0 (zero_for_one): price decreases, target = lower boundary
      - delta_x = L * (1/sqrt(P_new) - 1/sqrt(P_current))  [input]
      - delta_y = L * (sqrt(P_current) - sqrt(P_new))  [output]
      - sqrt_p_new = (L * sqrt_p) / (L + amount * sqrt_p)

    - Buying token0: price increases, target = upper boundary
      - delta_y = L * (sqrt(P_new) - sqrt(P_current))  [input]
      - delta_x = L * (1/sqrt(P_current) - 1/sqrt(P_new))  [output]
      - sqrt_p_new = sqrt_p + amount / L

    Args:
        sqrt_price_current: Current sqrt(price) per trajectory, shape (num_trajectories,)
        liquidity: Active liquidity per trajectory, shape (num_trajectories,)
        arrivals: Two-column arrivals [sell_token0, buy_token0], shape (num_trajectories, 2)
            - Column 0: Amount of token0 being sold (traders selling token0 to pool)
            - Column 1: Amount of token0 being bought (traders buying token0 from pool)
        tick_lower_boundary: sqrt(price) at lower tick boundary, shape (num_trajectories,)
        tick_upper_boundary: sqrt(price) at upper tick boundary, shape (num_trajectories,)
        fee_rate: Trading fee rate (default: 0.003 for 0.3%)

    Returns:
        tuple:
            - sqrt_price_next: New sqrt(price) after swap, shape (num_trajectories,)
            - amount_token0_net: Net token0 flow (positive = into pool), shape (num_trajectories,)
            - amount_token1_net: Net token1 flow (positive = into pool), shape (num_trajectories,)
            - fee_token0: Fee collected in token0, shape (num_trajectories,)
            - fee_token1: Fee collected in token1, shape (num_trajectories,)
            - hit_boundary: Whether tick boundary was crossed, shape (num_trajectories,)
    """
    # Ensure inputs are arrays
    sqrt_price_current = np.atleast_1d(sqrt_price_current)
    liquidity = np.atleast_1d(liquidity)
    arrivals = np.atleast_2d(arrivals)
    tick_lower_boundary = np.atleast_1d(tick_lower_boundary)
    tick_upper_boundary = np.atleast_1d(tick_upper_boundary)

    num_trajectories = len(sqrt_price_current)

    # ==================== Step 1: Direction from Net Amount ====================
    # Column 0 = sell_token0 (token0 into pool, price decreases)
    # Column 1 = buy_token0 (token0 out of pool, price increases)
    # net > 0: net selling of token0 (price decreases, zero_for_one)
    # net < 0: net buying of token0 (price increases)
    net_amount = arrivals[:, 0] - arrivals[:, 1]
    direction = np.sign(net_amount)  # +1 sell, -1 buy, 0 no trade
    abs_net_amount = np.abs(net_amount) * (1.0 - fee_rate)
    is_sell = direction > 0

    # ==================== Step 2: Boundary Selection via Indexing ====================
    # direction > 0 (sell token0): target lower boundary (index 0)
    # direction < 0 (buy token0): target upper boundary (index 1)
    boundaries = np.stack([tick_lower_boundary, tick_upper_boundary], axis=1)
    # Map direction to index: +1 -> 0, -1 -> 1, 0 -> 0 (doesn't matter for no-trade)
    boundary_idx = np.clip(((1 - direction) / 2).astype(np.int64), 0, 1)
    sqrt_price_target = boundaries[np.arange(num_trajectories), boundary_idx]

    # ==================== Step 3: Max Amount via Indexing ====================
    price_current = sqrt_price_current ** 2
    price_target = sqrt_price_target ** 2
    price_high = np.maximum(price_current, price_target)
    price_low = np.minimum(price_current, price_target)

    # Numerical safety
    price_low = np.maximum(price_low, 1e-300)
    price_high = np.maximum(price_high, 1e-300)

    # Compute both deltas unconditionally
    delta_x_abs = (1.0 / np.sqrt(price_low)) - (1.0 / np.sqrt(price_high))
    delta_y_abs = np.sqrt(price_high) - np.sqrt(price_low)

    # Stack and select: sell -> delta_x (idx 0), buy -> delta_y (idx 1)
    deltas = np.stack([delta_x_abs, delta_y_abs], axis=1)
    delta_selected = deltas[np.arange(num_trajectories), boundary_idx]
    arrivals_max = delta_selected * liquidity

    # ==================== Step 4: New Price via np.where ====================
    hit_boundary = abs_net_amount >= arrivals_max
    has_liquidity = liquidity > 0
    L_safe = np.where(has_liquidity, liquidity, 1.0)  # Avoid division by zero

    # Sell formula: sqrt_p_new = (L * sqrt_p) / (L + amt * sqrt_p)
    sqrt_p_new_sell = (L_safe * sqrt_price_current) / (L_safe + abs_net_amount * sqrt_price_current)

    # Buy formula: sqrt_p_new = sqrt_p + amt / L
    sqrt_p_new_buy = sqrt_price_current + abs_net_amount / L_safe

    # Select based on direction
    sqrt_p_partial = np.where(is_sell, sqrt_p_new_sell, sqrt_p_new_buy)
    sqrt_price_next = np.where(hit_boundary, sqrt_price_target, sqrt_p_partial)

    # Handle edge cases
    sqrt_price_next = np.where(direction == 0, sqrt_price_current, sqrt_price_next)
    sqrt_price_next = np.where(has_liquidity, sqrt_price_next, sqrt_price_current)

    # ==================== Step 5: Consumed Amount ====================
    amount_consumed = np.where(hit_boundary, arrivals_max, abs_net_amount)
    amount_consumed = np.where(has_liquidity & (direction != 0), amount_consumed, 0.0)

    # ==================== Step 6: Output Amounts ====================
    sqrt_p_high = np.maximum(sqrt_price_current, sqrt_price_next)
    sqrt_p_low = np.minimum(sqrt_price_current, sqrt_price_next)
    sqrt_p_low = np.maximum(sqrt_p_low, 1e-150)  # Numerical safety

    # Token1 output (sell case): Δy = L * (sqrt_P_current - sqrt_P_next)
    amount_out_1 = liquidity * (sqrt_p_high - sqrt_p_low)

    # Token0 output (buy case): Δx = L * (1/sqrt_P_current - 1/sqrt_P_next)
    amount_out_0 = liquidity * (1.0 / sqrt_p_low - 1.0 / sqrt_p_high)

    # ==================== Step 7: Net Flows (Pool Perspective) ====================
    # Positive = token flows INTO the pool
    # Sell token0: token0 in (+), token1 out (-)
    # Buy token0: token1 in (+), token0 out (-)
    amount_token0_net = np.where(is_sell, amount_consumed, -amount_out_0)
    amount_token0_net = np.where(direction == 0, 0.0, amount_token0_net)

    amount_token1_net = np.where(is_sell, -amount_out_1, amount_consumed)
    amount_token1_net = np.where(direction == 0, 0.0, amount_token1_net)

    # ==================== Step 8: Fees ====================
    # Fee is on the input token: fee = consumed / (1 - fee_rate) * fee_rate
    fee_amount = amount_consumed * fee_rate / (1.0 - fee_rate) if fee_rate < 1.0 else np.zeros(num_trajectories)

    # Assign fee to correct token based on direction
    # Sell token0: fee in token0
    # Buy token0: fee in token1
    fee_token0 = np.where(is_sell & (direction != 0), fee_amount, 0.0)
    fee_token1 = np.where((~is_sell) & (direction != 0), fee_amount, 0.0)

    return sqrt_price_next, amount_token0_net, amount_token1_net, fee_token0, fee_token1, hit_boundary

