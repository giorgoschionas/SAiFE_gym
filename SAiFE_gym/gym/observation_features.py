import numpy as np

from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    BOUNDARY_PROXIMITY_KEY,
    HAS_POSITION_KEY,
    INITIAL_WEALTH_KEY,
    LP_LIQUIDITY_KEY,
    LP_LOWER_OFFSET_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_UNCLAIMED_FEES0_KEY,
    LP_UNCLAIMED_FEES1_KEY,
    LP_UPPER_OFFSET_KEY,
    MISPRICING_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_SQRT_PRICE_KEY,
    PORTFOLIO_VALUE_KEY,
    PORTFOLIO_VALUE_RATIO_KEY,
    POSITION_WIDTH_KEY,
    UNCLAIMED_FEE_VALUE_RATIO_KEY,
)


SB3_DERIVED_OBS_KEYS = {
    MISPRICING_KEY,
    LP_LOWER_OFFSET_KEY,
    LP_UPPER_OFFSET_KEY,
    BOUNDARY_PROXIMITY_KEY,
    POSITION_WIDTH_KEY,
    POOL_SQRT_PRICE_KEY,
    HAS_POSITION_KEY,
    PORTFOLIO_VALUE_RATIO_KEY,
    UNCLAIMED_FEE_VALUE_RATIO_KEY,
}


def compute_sb3_observation_features(state_dict: dict) -> dict:
    """Compute derived features used by the SB3 flat observation adapter."""
    lower_offset = state_dict[POOL_CURRENT_TICK_KEY] - state_dict[LP_TICK_LOWER_KEY]
    upper_offset = state_dict[LP_TICK_UPPER_KEY] - state_dict[POOL_CURRENT_TICK_KEY]
    pool_price = state_dict[POOL_SQRT_PRICE_KEY] ** 2
    midprice = state_dict[ASSET_PRICE_KEY]
    relative_mispricing = (midprice - pool_price) / np.maximum(midprice, 1e-12)
    initial_wealth = np.maximum(state_dict[INITIAL_WEALTH_KEY], 1e-12)
    unclaimed_fee_value = (
        state_dict[LP_UNCLAIMED_FEES0_KEY] * midprice
        + state_dict[LP_UNCLAIMED_FEES1_KEY]
    )

    return {
        MISPRICING_KEY: relative_mispricing,
        POOL_SQRT_PRICE_KEY: pool_price,
        LP_LOWER_OFFSET_KEY: lower_offset,
        LP_UPPER_OFFSET_KEY: upper_offset,
        BOUNDARY_PROXIMITY_KEY: np.minimum(lower_offset, upper_offset),
        POSITION_WIDTH_KEY: lower_offset + upper_offset,
        HAS_POSITION_KEY: (state_dict[LP_LIQUIDITY_KEY] > 0.0).astype(np.float64),
        PORTFOLIO_VALUE_RATIO_KEY: state_dict[PORTFOLIO_VALUE_KEY] / initial_wealth,
        UNCLAIMED_FEE_VALUE_RATIO_KEY: unclaimed_fee_value / initial_wealth,
    }
