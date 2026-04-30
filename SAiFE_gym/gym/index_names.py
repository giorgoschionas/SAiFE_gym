# ============================================================================
# Dict-based State Keys for Full Pool Representation
# ============================================================================
# Use these keys to access state components in the new Dict-based state structure
PORTFOLIO_VALUE_KEY = 'portfolio_value'
LP_ALPHA_KEY = 'lp_alpha'
LP_TOKEN0_AMOUNT_KEY = 'lp_token0_amount'  # LP's absolute token0 holdings (risky-asset inventory) - shape: (num_trajectories,)
POOL_SQRT_PRICE_KEY = 'sqrt_price'          # Current pool sqrt(price) - shape: (num_trajectories,)
POOL_CURRENT_TICK_KEY = 'current_tick'      # Current tick index - shape: (num_trajectories,)
POOL_LIQUIDITY_ARRAY_KEY = 'liquidity_array'  # Liquidity per tick - shape: (num_trajectories, num_ticks)

FEES0_KEY = 'fees_0'  # Pool fees collected in token 0 per tick - shape: (num_trajectories, num_ticks)
FEES1_KEY = 'fees_1'  # Pool fees collected in token 1 per tick - shape: (num_trajectories, num_ticks)


# LP-specific state (agent's position)
LP_LIQUIDITY_KEY = 'lp_liquidity'           # LP's position liquidity - shape: (num_trajectories,)
LP_TICK_LOWER_KEY = 'lp_tick_lower'         # LP's position lower bound - shape: (num_trajectories,)
LP_TICK_UPPER_KEY = 'lp_tick_upper'         # LP's position upper bound - shape: (num_trajectories,)

# LP cumulative fee tracking (lifetime earnings, refreshed every step)
LP_COLLECTED_FEES0_KEY = 'lp_collected_fees_0'  # Cumulative fees earned by LP in token 0 - shape: (num_trajectories,)
LP_COLLECTED_FEES1_KEY = 'lp_collected_fees_1'  # Cumulative fees earned by LP in token 1 - shape: (num_trajectories,)

# LP currently-unclaimed fees (accrued since last rebalance; reset at rebalance)
LP_UNCLAIMED_FEES0_KEY = 'lp_unclaimed_fees_0'  # Unclaimed LP fees in token 0 - shape: (num_trajectories,)
LP_UNCLAIMED_FEES1_KEY = 'lp_unclaimed_fees_1'  # Unclaimed LP fees in token 1 - shape: (num_trajectories,)

# LP fee snapshots (recorded at position entry to exclude pre-entry fees)
LP_FEE_SNAPSHOT0_KEY = 'lp_fee_snapshot_0'  # Gross LP fee0 at time of entry - shape: (num_trajectories,)
LP_FEE_SNAPSHOT1_KEY = 'lp_fee_snapshot_1'  # Gross LP fee1 at time of entry - shape: (num_trajectories,)

# Deployment flag: True once the LP has deployed at least once (never reset to False).
# Distinguishes "never deployed" (use initial_wealth) from "bankrupt" (use 0).
LP_EVER_DEPLOYED_KEY = 'lp_ever_deployed'   # shape: (num_trajectories,), dtype bool


# Market state (external)
ASSET_PRICE_KEY = 'midprice'            # External market price - shape: (num_trajectories,)
TIME_KEY = 'time'                           # Current simulation time - shape: (num_trajectories,)

# Environment parameters (constant per episode, exposed as observations)
GAS_COST_KEY = 'gas_cost'               # Fixed rebalancing cost in token1 units - shape: (num_trajectories,)
INITIAL_WEALTH_KEY = 'initial_wealth'   # LP's starting wealth before first deployment - shape: (num_trajectories,)

# Derived observation features (computed from state, not stored in state dict)
MISPRICING_KEY      = 'mispricing'       # asset_price - amm_price (= ASSET_PRICE - sqrt_price²)
LP_LOWER_OFFSET_KEY = 'lp_lower_offset'  # current_tick - lp_tick_lower  (≥ 0 when in-range)
LP_UPPER_OFFSET_KEY = 'lp_upper_offset'  # lp_tick_upper - current_tick   (≥ 0 when in-range)

BOUNDARY_PROXIMITY_KEY = 'boundary_proximity'  # min(lower_offset, upper_offset) — distance to nearest boundary
POSITION_WIDTH_KEY = 'position_width'          # lower_offset + upper_offset — position concentration



