# ============================================================================
# Dict-based State Keys for Full Pool Representation
# ============================================================================
# Use these keys to access state components in the new Dict-based state structure

POOL_SQRT_PRICE_KEY = 'sqrt_price'          # Current pool sqrt(price) - shape: (num_trajectories,)
POOL_CURRENT_TICK_KEY = 'current_tick'      # Current tick index - shape: (num_trajectories,)
POOL_LIQUIDITY_ARRAY_KEY = 'liquidity_array'  # Liquidity per tick - shape: (num_trajectories, num_ticks)

FEES_A_KEY = 'fees_a' # Fees in token A collected per tick - shape: (num_trajectories, num_ticks)
FEES_B_KEY = 'fees_b' # Fees in token B collected per tick - shape: (num_trajectories, num_ticks)


# LP-specific state (agent's position)
LP_LIQUIDITY_KEY = 'lp_liquidity'           # LP's position liquidity - shape: (num_trajectories,)
LP_TICK_LOWER_KEY = 'lp_tick_lower'         # LP's position lower bound - shape: (num_trajectories,)
LP_TICK_UPPER_KEY = 'lp_tick_upper'         # LP's position upper bound - shape: (num_trajectories,)


# Market state (external)
MARKET_MIDPRICE_KEY = 'midprice'            # External market price - shape: (num_trajectories,)
TIME_KEY = 'time'                           # Current simulation time - shape: (num_trajectories,)

