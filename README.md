# SAiFE Gym

A Reinforcement Learning environment for simulating Automated Market Maker (AMM) trading in DeFi protocols.

## Overview

SAiFE_gym provides an OpenAI Gym-compatible environment for training RL agents to act as liquidity providers in AMM protocols like Uniswap V2/V3. The framework simulates market dynamics, order flows, and reward mechanisms for DeFi trading strategies.

## Features

- **AMM Environment**: OpenAI Gym-compatible RL environment for AMM simulation
- **Multiple AMM Models**: Support for Uniswap V2 (constant product) and V3 (concentrated liquidity)
- **Stochastic Processes**: Configurable models for price dynamics and order arrivals
- **Reward Functions**: Framework for impermanent loss and LVR calculations
- **Agent Interface**: Abstract base class for implementing trading strategies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SAiFE_gym
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from SAiFE_gym.gym.AMMEnvironment import AMMEnvironment
from SAiFE_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from SAiFE_gym.stochastic_processes.arrival_models import PoissonArrivalModel

# Create environment (Note: Currently under development)
env = AMMEnvironment(
    terminal_time=1.0,
    n_steps=200,
    initial_cash=1000.0
)

# Basic usage
obs = env.reset()
action = env.action_space.sample()  # Random action
next_obs, reward, done, info = env.step(action)
```

## Architecture

### Core Components

- **AMMEnvironment**: Main RL environment managing episodes and state transitions
- **ModelDynamics**: AMM-specific logic for Uniswap V2/V3 protocols
- **StochasticProcesses**: Models for price movements and order arrivals
- **RewardFunctions**: Metrics for evaluating trading performance
- **Agents**: Interface for implementing trading strategies

### File Structure

```
SAiFE_gym/
├── agents/           # Agent implementations
├── gym/             # Core environment and dynamics
│   ├── helpers.py/  # AMM utility functions
│   └── ...
├── rewards/         # Reward function implementations
├── stochastic_processes/  # Market models
└── notebooks/       # Example usage and experiments
```

## Development Status

⚠️ **Work in Progress**: This project is under active development. Key components are implemented but not fully integrated:

- ✅ Basic environment structure
- ✅ AMM utility functions
- ✅ Stochastic process models
- 🚧 State update mechanisms
- 🚧 Reward function implementations
- 🚧 Action space definitions

## Dependencies

- `gym`: RL environment framework
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Visualization
- `web3`: Ethereum blockchain interaction
- See `requirements.txt` for complete list

## Contributing

This is an active research project. Contributions are welcome for:

- Completing reward function implementations
- Integrating stochastic processes with environment dynamics
- Adding more AMM protocols
- Improving documentation and examples

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information when available]
```