# ============================================================================
# Dict-based State Keys for Full Pool Representation
# ============================================================================
# Use these keys to access state components in the new Dict-based state structure

POOL_SQRT_PRICE_KEY = 'sqrt_price'          # Current pool sqrt(price) - shape: (num_trajectories,)
POOL_CURRENT_TICK_KEY = 'current_tick'      # Current tick index - shape: (num_trajectories,)
POOL_LIQUIDITY_ARRAY_KEY = 'liquidity_array'  # Liquidity per tick - shape: (num_trajectories, num_ticks)

FEES0_KEY = 'fees_0'  # Total fees collected in token 0 - shape: (num_trajectories,)
FEES1_KEY = 'fees_1'  # Total fees collected in token 1 - shape: (num_trajectories,)


# LP-specific state (agent's position)
LP_LIQUIDITY_KEY = 'lp_liquidity'           # LP's position liquidity - shape: (num_trajectories,)
LP_TICK_LOWER_KEY = 'lp_tick_lower'         # LP's position lower bound - shape: (num_trajectories,)
LP_TICK_UPPER_KEY = 'lp_tick_upper'         # LP's position upper bound - shape: (num_trajectories,)


# Market state (external)
ASSET_PRICE_KEY = 'midprice'            # External market price - shape: (num_trajectories,)
TIME_KEY = 'time'                           # Current simulation time - shape: (num_trajectories,)


# ============================================================================
# Legacy Array-based Indices (for backwards compatibility)
# ============================================================================
# These are used with legacy flat-array state representations
# TODO: Remove these once all code is migrated to dict-based state

LIQUIDITY_INDEX = 0
AMM_PRICE_INDEX = 1
ASSET_PRICE_INDEX = 2
FEES_TOKEN_A_INDEX = 3
FEES_TOKEN_B_INDEX = 4
TIME_INDEX = 5

