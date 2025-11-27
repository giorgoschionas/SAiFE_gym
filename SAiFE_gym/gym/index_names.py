# Uniswap V3 State Indices
# State uses (P, L) parameterization - amounts (x, y) computed on-demand via calculate_position_amounts()
LIQUIDITY_INDEX = 0        # Liquidity amount L
AMM_PRICE_INDEX = 1       # Current sqrt price of the pool
ASSET_PRICE_INDEX = 2    # Current midprice of the asset
TIME_INDEX = 3           # Current time