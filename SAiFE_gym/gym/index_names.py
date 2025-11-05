X = 0
Y = 1

CASH_INDEX = 0
INVENTORY_INDEX = 1
TIME_INDEX = 2
ASSET_PRICE_INDEX = 3

# Uniswap V3 State Indices
V3_AMOUNT0_INDEX = 0          # Token0 amount (e.g., USDC) in LP position
V3_AMOUNT1_INDEX = 1          # Token1 amount (e.g., ETH) in LP position
V3_LIQUIDITY_INDEX = 2        # Liquidity amount L
V3_SQRT_PRICE_INDEX = 3       # Current sqrt price of the pool
V3_TICK_INDEX = 4             # Current tick
V3_TICK_LOWER_INDEX = 5       # Lower tick of LP's range
V3_TICK_UPPER_INDEX = 6       # Upper tick of LP's range
V3_FEES_INDEX = 7             # Accumulated trading fees (in token0 equivalent)
V3_MIDPRICE_INDEX = 8         # Reference midprice from stochastic process
V3_TIME_INDEX = 9             # Current time