"""
Experiment configuration for ``agent_comparison.py``.

Each top-level variable is a **list of values to sweep over**. The Cartesian
product of all lists defines the full experiment set. Each combination is one
SLURM array task (or one ``--config-idx`` value when run locally).

To freeze a variable, wrap a single value in a one-element list, e.g. ``SEED = [6]``.
The order in ``SWEEP_VARS`` defines how combinations are enumerated (last var
cycles fastest, matching ``itertools.product``).
"""

import numpy as np


# ── Sweep variables ─────────────────────────────────────────────────
SEED = [6]
TERMINAL_TIME = [1.0]
N_STEPS = [1000]
NUM_TRAJECTORIES_TRAIN = [200]
NUM_TRAJECTORIES_EVAL = [500]#[1000]
INITIAL_WEALTH = [1000]
TAU = [20]
LIQUIDITY_SCALE = [1e5]

INITIAL_PRICE = [1000]
INITIAL_POOL_PRICE = [None]  # None → pool starts at INITIAL_PRICE
DRIFT = [0]
VOLATILITY = [0.01]
FEE_TIER = [0.003]
EXP_VALUE = [1.0001]

ALPHA0 = [np.array([1.0, 1.0])]
ALPHA1 = [np.array([15.0, 15.0])]
ALPHA2 = [np.array([0.0, 0.0])]
ALPHA3 = [np.array([4000, 4000])]

GAMMA_CARTEA = [0.000005]
# Tick-deadband for CarteaPLAgent's deploy-mode rebalance gate. tol=0 recovers
# the academic "rebalance on every meaningful pool move" policy (high gas);
# tol=1 (default) absorbs 1-tick drifts; larger values rebalance more sparsely.
REBALANCE_TOLERANCE_CARTEA = [20]

# ── Reward function selector ────────────────────────────────────────
#   'pnl'         — plain ΔPortfolioValue (risk-neutral; PnL == utility)
#   'inventory'   — RunningInventoryPenalty: PnL − φ·dt·x_t^p (Cartea–Jaimungal)
#   'exponential' — ExponentialUtility: sparse terminal CARA −exp(−a·W_T)
# With INITIAL_WEALTH=1000, x_t∈[0,1] in token0 units, dt=1e-3. Sensible scales:
#   INVENTORY_PHI ∈ [1, 100] — per-step penalty ≈ phi·1e-3·x² in token1 units.
#   EXP_RISK_AVERSION ≈ 1e-3 to keep a·W ~ O(1) (default 0.1 underflows).
REWARD_KIND = ['pnl']           # 'pnl' | 'inventory' | 'exponential'
INVENTORY_PHI = [0]
INVENTORY_TERMINAL_AVERSION = [0.0]
INVENTORY_EXPONENT = [2.0]
EXP_RISK_AVERSION = [1e-3]

ARRIVAL_REBALANCE_EVERY = [150]
ARRIVAL_REBALANCE_WIDTH = [2]
ARRIVAL_REBALANCE_LOWER = [-1]
ARRIVAL_REBALANCE_UPPER = [1]

DEPLOYONCE_LOWER = [-15]
DEPLOYONCE_UPPER = [15]

REINFORCE_EPOCHS = [0]
REINFORCE_LR = [2e-4]
ACTION_STD_INIT = [1.7]

DQN_TICK_STRIDE = [10]
NUM_SINGLE_SIMS = [15]
MAX_TRADES_DEBUG = [None]

PPO_ACTION_WRAPPER = ['multidiscrete']  # 'multidiscrete' or 'rescaled'
DECISION_STRIDE = [200]

ENABLE_AGENTS = [
    {
        'DeployNarrow':     True,
        'Uniform':          False,
        'DeployWide':       True,
        'ArrivalRebalance': True,
        'CDM': True,
        'REINFORCE':        False,
        'PPO':              True,
        'PPO_narrow':       True,
        'SAC':              False,
        'DQN':              False,
    },
]

# ── PPO hyperparameters ─────────────────────────────────────────────
PPO_LEARNING_RATE = [3e-4]
PPO_BATCH_SIZE = [200]
PPO_N_EPOCHS = [7]
PPO_GAMMA = [0.99]
PPO_GAE_LAMBDA = [0.95]
PPO_CLIP_RANGE = [0.15]
PPO_ENT_COEF = [0.03]
PPO_NET_ARCH = [[256, 256]]


# Canonical sweep order; combinations are enumerated with the last variable
# cycling fastest (standard itertools.product order).
SWEEP_VARS = [
    'SEED', 'TERMINAL_TIME', 'N_STEPS',
    'NUM_TRAJECTORIES_TRAIN', 'NUM_TRAJECTORIES_EVAL',
    'INITIAL_WEALTH', 'TAU', 'LIQUIDITY_SCALE',
    'INITIAL_PRICE', 'INITIAL_POOL_PRICE', 'DRIFT', 'VOLATILITY',
    'FEE_TIER', 'EXP_VALUE',
    'ALPHA0', 'ALPHA1', 'ALPHA2', 'ALPHA3',
    'GAMMA_CARTEA', 'REBALANCE_TOLERANCE_CARTEA',
    'REWARD_KIND', 'INVENTORY_PHI', 'INVENTORY_TERMINAL_AVERSION',
    'INVENTORY_EXPONENT', 'EXP_RISK_AVERSION',
    'ARRIVAL_REBALANCE_EVERY', 'ARRIVAL_REBALANCE_WIDTH',
    'ARRIVAL_REBALANCE_LOWER', 'ARRIVAL_REBALANCE_UPPER',
    'DEPLOYONCE_LOWER', 'DEPLOYONCE_UPPER',
    'REINFORCE_EPOCHS', 'REINFORCE_LR', 'ACTION_STD_INIT',
    'DQN_TICK_STRIDE', 'NUM_SINGLE_SIMS', 'MAX_TRADES_DEBUG',
    'PPO_ACTION_WRAPPER', 'DECISION_STRIDE',
    'ENABLE_AGENTS',
    'PPO_LEARNING_RATE', 'PPO_BATCH_SIZE', 'PPO_N_EPOCHS',
    'PPO_GAMMA', 'PPO_GAE_LAMBDA', 'PPO_CLIP_RANGE',
    'PPO_ENT_COEF', 'PPO_NET_ARCH',
]


def _sizes():
    g = globals()
    return [len(g[name]) for name in SWEEP_VARS]


def count_combinations() -> int:
    """Total number of (var₁, var₂, …) combinations across all sweep lists."""
    n = 1
    for s in _sizes():
        n *= s
    return n


def get_combination(idx: int) -> dict:
    """Return the ``idx``-th combination as a ``{var_name: value}`` dict."""
    total = count_combinations()
    if idx < 0 or idx >= total:
        raise IndexError(
            f"config-idx {idx} out of range; have {total} combination(s)."
        )
    g = globals()
    sizes = _sizes()
    rem = idx
    indices = [0] * len(SWEEP_VARS)
    for i in range(len(SWEEP_VARS) - 1, -1, -1):
        indices[i] = rem % sizes[i]
        rem //= sizes[i]
    return {name: g[name][indices[i]] for i, name in enumerate(SWEEP_VARS)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "count":
        # Machine-readable: just the number, for use in shell scripts.
        print(count_combinations())
    else:
        total = count_combinations()
        print(f"Total combinations: {total}")
        print(f"\nSweep variables ({len(SWEEP_VARS)}):")
        for name in SWEEP_VARS:
            vals = globals()[name]
            marker = ' *' if len(vals) > 1 else ''
            print(f"  {name:<28} {len(vals)} value(s){marker}")
