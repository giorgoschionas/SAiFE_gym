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



def create_buckets(bucket_endpoints):
    buckets = []
    for i in range(1, len(bucket_endpoints)):
        newBucket = {'p_low': bucket_endpoints[i - 1],
                     'p_high': bucket_endpoints[i]}
        buckets.append(newBucket)
    return buckets

# 
def get_buckets_given_center_bucket_id(center_bucket_id, tau=5, exponential_value=1.0001):
    bucket_endpoints = []
    for i in range(center_bucket_id - tau, center_bucket_id + tau + 2):
        bucket_endpoints.append(exponential_value ** i)
    return create_buckets(bucket_endpoints)

def find_bucket_id(price, exponential_value=1.0001):
    return math.floor(math.log(price, exponential_value))

def is_out_of_range(price, center_bucket, tau=5):
    center_bucket_id = find_bucket_id(price)
    return center_bucket_id < center_bucket - tau or center_bucket_id > center_bucket + tau


# Functions for collecting fees
# delta change of amount of Token A
def delta_x(p1, p2):
    return 1 / math.sqrt(p2) - 1 / math.sqrt(p1)


# delta change of amount of Token B
def delta_y(p1, p2):
    return math.sqrt(p2) - math.sqrt(p1)

# transaction fee collected for a single price change by 1 unit of liquidity over [a, b]
def transaction_fee_one_step(a, b, p1, p2, fee_rate):
    # return token A, token B amounts
    if (p1 < a and p2 < a) or (p1 > b and p2 > b) or p1 == p2:
        return 0., 0.
    if p1 < p2:
        return 0., fee_rate * delta_y(max(p1, a), min(p2, b))
    else:
        return fee_rate * delta_x(min(b, p1), max(p2, a)), 0.


# calculate transaction fee for a price sequence
def transaction_fee_for_sequence(buckets, pool_price_seq, fee_rate):
    earned_token_a_each_bucket = []
    earned_token_b_each_bucket = []
    for _ in buckets:
        earned_token_a_each_bucket.append(0.)
        earned_token_b_each_bucket.append(0.)

    for i in range(len(pool_price_seq) - 1):
        for j, bucket in enumerate(buckets):
            earned_tokens = transaction_fee_one_step(bucket['p_low'], bucket['p_high'], pool_price_seq[i],
                                                     pool_price_seq[i + 1], fee_rate)
            earned_token_a_each_bucket[j] += earned_tokens[0]
            earned_token_b_each_bucket[j] += earned_tokens[1]
    return earned_token_a_each_bucket, earned_token_b_each_bucket




# ============================================================================
# Uniswap V4 Hook Functions
# ============================================================================
