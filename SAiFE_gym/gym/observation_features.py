import numpy as np

from SAiFE_gym.gym.index_names import (
    ASSET_PRICE_KEY,
    BOUNDARY_PROXIMITY_KEY,
    LP_LOWER_OFFSET_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_UPPER_OFFSET_KEY,
    MISPRICING_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_SQRT_PRICE_KEY,
    POSITION_WIDTH_KEY,
)


SB3_DERIVED_OBS_KEYS = {
    MISPRICING_KEY,
    LP_LOWER_OFFSET_KEY,
    LP_UPPER_OFFSET_KEY,
    BOUNDARY_PROXIMITY_KEY,
    POSITION_WIDTH_KEY,
    POOL_SQRT_PRICE_KEY,
}


def compute_sb3_observation_features(state_dict: dict) -> dict:
    """Compute derived features used by the SB3 flat observation adapter."""
    lower_offset = state_dict[POOL_CURRENT_TICK_KEY] - state_dict[LP_TICK_LOWER_KEY]
    upper_offset = state_dict[LP_TICK_UPPER_KEY] - state_dict[POOL_CURRENT_TICK_KEY]
    pool_price = state_dict[POOL_SQRT_PRICE_KEY] ** 2

    return {
        MISPRICING_KEY: state_dict[ASSET_PRICE_KEY] - pool_price,
        POOL_SQRT_PRICE_KEY: pool_price,
        LP_LOWER_OFFSET_KEY: lower_offset,
        LP_UPPER_OFFSET_KEY: upper_offset,
        BOUNDARY_PROXIMITY_KEY: np.minimum(lower_offset, upper_offset),
        POSITION_WIDTH_KEY: lower_offset + upper_offset,
    }
