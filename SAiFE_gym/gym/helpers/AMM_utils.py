import numpy as np
import math

# E.g for and ETH-USDC pair, Y is the reserves of ETH, and X is the reserves of USDC. P_{ETH in USDC} = X/Y
def CPMM_Spot_Price (X, Y):
    return X / Y

def update_CPMM(X, Y, dX, dY):
    X += dX
    Y += dY
    return X, Y

def Sell_CPMM(sell, X, Y, eta):
    # eta is the fee tier - e.g. 0.003 for 0.3% fee in Uniswap v2

    k = X * Y
    output = X - k / (Y + sell * (1 - eta))
    return output

def Buy_CPMM(buy, X, Y, eta):
    k = X * Y
    output = Y - k / (X + buy * (1 - eta))
    return output

def Arb_Trade_CPMM(S, X, Y, eta):
    sell_output = Sell_CPMM(S, X, Y, eta)
    buy_output = Buy_CPMM(S, X, Y, eta)
    return sell_output, buy_output


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




def create_buckets(bucket_endpoints):
    buckets = []
    for i in range(1, len(bucket_endpoints)):
        newBucket = {'p_low': bucket_endpoints[i - 1],
                     'p_high': bucket_endpoints[i]}
        buckets.append(newBucket)
    return buckets

# 
def get_buckets_given_center_bucket_id(center_bucket_id, tau, exponential_value=1.0001):
    bucket_endpoints = []
    for i in range(center_bucket_id - tau, center_bucket_id + tau + 2):
        bucket_endpoints.append(exponential_value ** i)
    return create_buckets(bucket_endpoints)

def find_bucket_id(price, exponential_value=1.0001):
    return math.floor(math.log(price, exponential_value))




# ============================================================================
# Uniswap V4 Hook Functions
# ============================================================================
