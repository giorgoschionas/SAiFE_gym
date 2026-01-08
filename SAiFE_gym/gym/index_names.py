# ============================================================================
# NEW: Dict-based State Keys for Full Pool Representation
# ============================================================================
# Use these keys to access state components in the new Dict-based state structure

# Pool-level state (global liquidity distribution)
POOL_SQRT_PRICE_KEY = 'sqrt_price'          # Current pool sqrt(price) - shape: (num_trajectories,)
POOL_CURRENT_TICK_KEY = 'current_tick'      # Current tick index - shape: (num_trajectories,)
POOL_LIQUIDITY_ARRAY_KEY = 'liquidity_array'  # Liquidity per tick - shape: (num_trajectories, num_ticks)

# Tick indexed state
GLOBAL_FEES_A_KEY = 'global_fees_a'
GLOBAL_FEES_B_KEY = 'global_fees_b'
FEE_OUTSIDE_A_KEY = 'fee_outside_a'
FEE_OUTSIDE_B_KEY = 'fee_outside_b'



# LP-specific state (agent's position)
LP_LIQUIDITY_KEY = 'lp_liquidity'           # LP's position liquidity - shape: (num_trajectories,)
LP_TICK_LOWER_KEY = 'lp_tick_lower'         # LP's position lower bound - shape: (num_trajectories,)
LP_TICK_UPPER_KEY = 'lp_tick_upper'         # LP's position upper bound - shape: (num_trajectories,)
LP_FEES_TOKEN_A_KEY = 'fees_token_a'        # Accumulated fees token A - shape: (num_trajectories,)
LP_FEES_TOKEN_B_KEY = 'fees_token_b'        # Accumulated fees token B - shape: (num_trajectories,)

# Market state (external)
MARKET_MIDPRICE_KEY = 'midprice'            # External market price - shape: (num_trajectories,)
TIME_KEY = 'time'                           # Current simulation time - shape: (num_trajectories,)

