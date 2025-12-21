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

# transaction fee collected for a single price change by 1 unit of liquidity over [a, b]
def transaction_fee_one_step_vec(a, b, p1, p2, fee_rate):
    a, b, p1, p2 = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
        np.asarray(p2, dtype=np.float64),
    )

    fee_a = np.zeros_like(p1, dtype=np.float64)
    fee_b = np.zeros_like(p1, dtype=np.float64)

    no_overlap_or_no_move = (
        ((p1 < a) & (p2 < a)) |
        ((p1 > b) & (p2 > b)) |
        (p1 == p2)
    )
    valid = ~no_overlap_or_no_move

    up = valid & (p1 < p2)
    dn = valid & (p1 > p2)

    # Price increases => Token B fees on overlap
    if np.any(up):
        lo = np.maximum(p1, a)
        hi = np.minimum(p2, b)
        ok = up & (hi > lo)
        fee_b[ok] = fee_rate * (np.sqrt(hi[ok]) - np.sqrt(lo[ok]))

    # Price decreases => Token A fees on overlap
    if np.any(dn):
        hi = np.minimum(b, p1)
        lo = np.maximum(p2, a)
        ok = dn & (hi > lo)
        fee_a[ok] = fee_rate * delta_x_vec(hi[ok], lo[ok])

    if fee_a.ndim == 0:
        return float(fee_a), float(fee_b)
    return fee_a, fee_b



# ============================================================================
# Uniswap V4 Hook Functions
# ============================================================================
