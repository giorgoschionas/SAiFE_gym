# Uniswap V3 State Indices
# State uses (P, L) parameterization - amounts (x, y) computed on-demand via calculate_position_amounts()
V3_LIQUIDITY_INDEX = 0        # Liquidity amount L
V3_SQRT_PRICE_INDEX = 1       # Current sqrt price of the pool
V3_TICK_INDEX = 2             # Current tick
V3_TICK_LOWER_INDEX = 3       # Lower tick of LP's range
V3_TICK_UPPER_INDEX = 4       # Upper tick of LP's range
V3_FEES_INDEX = 5             # Accumulated trading fees (in token0 equivalent)
V3_TIME_INDEX = 6           # Current time