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

def get_position_value_vec(L, external_p_current, sqrt_p_current, sqrt_p_lower, sqrt_p_upper):
    """
    Vectorized mark-to-market value of a Uniswap v3 position in terms of quote asset (token1).

    Handles arrays where each element may be above, below, or in range independently.

    Args:
        L: Liquidity amount, array-like shape (num_trajectories,)
        external_p_current: Current price of the asset (not sqrt), array-like shape (num_trajectories,)
        sqrt_p_current: Current sqrt(price), array-like shape (num_trajectories,)
        sqrt_p_lower: Lower bound sqrt(price), array-like shape (num_trajectories,)
        sqrt_p_upper: Upper bound sqrt(price), array-like shape (num_trajectories,)

    Returns:
        np.ndarray: Total value in terms of token1, shape (num_trajectories,)
    """
    L = np.atleast_1d(np.asarray(L, dtype=np.float64))
    sqrt_p_current = np.atleast_1d(np.asarray(sqrt_p_current, dtype=np.float64))
    sqrt_p_lower = np.atleast_1d(np.asarray(sqrt_p_lower, dtype=np.float64))
    sqrt_p_upper = np.atleast_1d(np.asarray(sqrt_p_upper, dtype=np.float64))


    above = sqrt_p_current >= sqrt_p_upper
    below = sqrt_p_current <= sqrt_p_lower

    # Case 1: price above range → 100% token1
    v_above = L * (sqrt_p_upper - sqrt_p_lower)
    # Case 2: price below range → 100% token0, valued at current price
    v_below = external_p_current * L * (sqrt_p_upper - sqrt_p_lower) / (sqrt_p_lower * sqrt_p_upper)
    # Case 3: price in range → mix: V = x * P_ext + y
    v_in = L * (external_p_current / sqrt_p_current + sqrt_p_current - sqrt_p_lower - external_p_current / sqrt_p_upper)

    return np.where(above, v_above, np.where(below, v_below, v_in))


def get_position_value(L, external_price_current, sqrt_price_current, sqrt_price_lower, sqrt_price_upper):
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

    # Case 1: Current price is ABOVE the range (Position is 100% Asset Y)
    if sqrt_price_current >= sqrt_price_upper:
        return L * (sqrt_price_upper - sqrt_price_lower)

    # Case 2: Current price is BELOW the range (Position is 100% Asset X)
    elif sqrt_price_current <= sqrt_price_lower:
        # We hold max X, valued at current price P
        # x_max = L * (upper - lower) / (lower * upper)
        return external_price_current * L * (sqrt_price_upper - sqrt_price_lower) / (sqrt_price_lower * sqrt_price_upper)

    # Case 3: Current price is IN RANGE (Mix of X and Y)
    else:
        # Derived from V = y + x*P_ext
        return L * (external_price_current / sqrt_price_current + sqrt_price_current - sqrt_price_lower - external_price_current / sqrt_price_upper)


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



