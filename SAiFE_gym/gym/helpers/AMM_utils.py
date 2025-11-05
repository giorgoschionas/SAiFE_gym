import numpy as np

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

def price_to_tick(price: float, tick_spacing: int = 60) -> int:
    """Convert price to tick index"""
    tick = np.log(price) / np.log(1.0001)
    return int(np.floor(tick / tick_spacing) * tick_spacing)

def tick_to_price(tick: int) -> float:
    """Convert tick index to price"""
    return 1.0001 ** tick

def sqrt_price_x96_to_price(sqrt_price_x96: int) -> float:
    """Convert sqrt price X96 to actual price"""
    sqrt_price = sqrt_price_x96 / (2**96)
    return sqrt_price ** 2

def price_to_sqrt_price_x96(price: float) -> int:
    """Convert price to sqrt price X96 format"""
    sqrt_price = np.sqrt(price)
    return int(sqrt_price * (2**96))

def get_sqrt_ratio_at_tick(tick: int) -> int:
    """Get sqrt price ratio at given tick (returns Q64.96 fixed point)"""
    # Simplified version for simulation - less precise but much more readable
    price = 1.0001 ** tick
    sqrt_price = np.sqrt(price)
    return int(sqrt_price * (2**96))

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

def calculate_position_amounts(
    liquidity: float,
    sqrt_price_current: float,
    sqrt_price_lower: float,
    sqrt_price_upper: float
) -> tuple:
    """Calculate token amounts for a given liquidity position"""
    if sqrt_price_current <= sqrt_price_lower:
        amount0 = liquidity * (1/sqrt_price_lower - 1/sqrt_price_upper)
        amount1 = 0
    elif sqrt_price_current >= sqrt_price_upper:
        amount0 = 0
        amount1 = liquidity * (sqrt_price_upper - sqrt_price_lower)
    else:
        amount0 = liquidity * (1/sqrt_price_current - 1/sqrt_price_upper)
        amount1 = liquidity * (sqrt_price_current - sqrt_price_lower)
    
    return amount0, amount1

def add_liquidity_v3(
    current_price: float,
    price_lower: float,
    price_upper: float,
    amount0_desired: float,
    amount1_desired: float,
    fee_tier: float = 0.003
) -> dict:
    """Add concentrated liquidity to Uniswap V3 position"""
    sqrt_price_current = np.sqrt(current_price)
    sqrt_price_lower = np.sqrt(price_lower)
    sqrt_price_upper = np.sqrt(price_upper)
    
    liquidity = calculate_liquidity_amounts(
        sqrt_price_current, sqrt_price_lower, sqrt_price_upper,
        amount0_desired, amount1_desired
    )
    
    amount0_actual, amount1_actual = calculate_position_amounts(
        liquidity, sqrt_price_current, sqrt_price_lower, sqrt_price_upper
    )
    
    return {
        'liquidity': liquidity,
        'amount0': amount0_actual,
        'amount1': amount1_actual,
        'price_lower': price_lower,
        'price_upper': price_upper,
        'fee_tier': fee_tier,
        'tick_lower': price_to_tick(price_lower),
        'tick_upper': price_to_tick(price_upper)
    }

def remove_liquidity_v3(position: dict, liquidity_to_remove: float) -> tuple:
    """Remove liquidity from Uniswap V3 position"""
    if liquidity_to_remove > position['liquidity']:
        raise ValueError("Cannot remove more liquidity than available")
    
    proportion = liquidity_to_remove / position['liquidity']
    amount0_removed = position['amount0'] * proportion
    amount1_removed = position['amount1'] * proportion
    
    position['liquidity'] -= liquidity_to_remove
    position['amount0'] -= amount0_removed
    position['amount1'] -= amount1_removed
    
    return amount0_removed, amount1_removed

def calculate_fees_earned(
    position: dict,
    volume0: float,
    volume1: float,
    total_liquidity: float
) -> tuple:
    """Calculate fees earned by a liquidity position"""
    if total_liquidity == 0:
        return 0, 0
    
    liquidity_share = position['liquidity'] / total_liquidity
    fee0_earned = volume0 * position['fee_tier'] * liquidity_share
    fee1_earned = volume1 * position['fee_tier'] * liquidity_share
    
    return fee0_earned, fee1_earned

def swap_v3_single_tick(
    amount_in: float,
    sqrt_price_current: float,
    liquidity: float,
    fee_tier: float,
    zero_for_one: bool
) -> dict:
    """Execute swap within a single tick range"""
    fee_amount = amount_in * fee_tier
    amount_in_after_fee = amount_in - fee_amount
    
    if zero_for_one:  # Swapping token0 for token1
        sqrt_price_next = liquidity / (liquidity / sqrt_price_current + amount_in_after_fee)
        amount_out = liquidity * (sqrt_price_current - sqrt_price_next)
    else:  # Swapping token1 for token0
        sqrt_price_next = sqrt_price_current + amount_in_after_fee / liquidity
        amount_out = liquidity * (1/sqrt_price_current - 1/sqrt_price_next)
    
    return {
        'amount_out': amount_out,
        'sqrt_price_next': sqrt_price_next,
        'fee_amount': fee_amount
    }

def calculate_impermanent_loss_v3(
    position: dict,
    price_current: float,
    price_at_deposit: float
) -> dict:
    """Calculate impermanent loss for V3 concentrated liquidity position"""
    sqrt_price_current = np.sqrt(price_current)
    sqrt_price_lower = np.sqrt(position['price_lower'])
    sqrt_price_upper = np.sqrt(position['price_upper'])
    
    # Current token amounts in position
    amount0_current, amount1_current = calculate_position_amounts(
        position['liquidity'], sqrt_price_current, sqrt_price_lower, sqrt_price_upper
    )
    
    # Initial token amounts when position was created
    sqrt_price_deposit = np.sqrt(price_at_deposit)
    amount0_initial, amount1_initial = calculate_position_amounts(
        position['liquidity'], sqrt_price_deposit, sqrt_price_lower, sqrt_price_upper
    )
    
    # Current value of position
    current_value = amount0_current + amount1_current * price_current
    
    # Value if tokens were held without providing liquidity
    hold_value = amount0_initial + amount1_initial * price_current
    
    impermanent_loss = (current_value - hold_value) / hold_value
    
    return {
        'impermanent_loss_pct': impermanent_loss * 100,
        'current_value': current_value,
        'hold_value': hold_value,
        'amount0_current': amount0_current,
        'amount1_current': amount1_current
    }

def is_position_in_range(position: dict, current_price: float) -> bool:
    """Check if current price is within position's range"""
    return position['price_lower'] <= current_price <= position['price_upper']

def calculate_capital_efficiency(
    position_v3: dict,
    position_v2_liquidity: float,
    price_range_factor: float
) -> float:
    """Calculate capital efficiency of V3 position vs V2"""
    price_range = position_v3['price_upper'] - position_v3['price_lower']
    full_range = position_v3['price_upper'] * 2  # Approximate full range
    concentration_factor = full_range / price_range
    
    return concentration_factor


# ============================================================================
# Uniswap V4 Hook Functions
# ============================================================================
