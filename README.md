# SAiFE Gym

`SAiFE_gym` is a module which provides a collection of Gymnasium environments for training reinforcement learning (RL) agents for dynamic liquidity provision (LP) in automated market makers (AMMs) with concentrated liquidity (CL), like Uniswap v3. The microstructure of AMMs with CL is decomposed in interactive components that
allow researchers and practitioners to combine them and capture various economic settings. The module is vectorized end-to-end which allows faster training of RL agents.

The associated paper can be found at [https://arxiv.org/abs/2609.17788](https://arxiv.org/abs/2609.17788)

## Features

- **Mechanics of AMMs with CL** - tick-indexed liquidity array, pool price and 
  fee accounting, LP fee snapshots.
- **Market components as stochastic processes** - External midprice models, orderflow models (that capture baseline traders, liquidity-attracted traders and informed traders), price impact models, and separate fee-accounting models, gas cost models.
- **Vectorized rollouts** - state, actions, arrivals, swaps, rewards and
  diagnostics are batched across `num_trajectories`.

## Installation

```bash
git clone https://github.com/giorgoschionas/SAiFE_gym.git
cd SAiFE_gym
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Architecture

```text
SAiFE_gym/
├── SAiFE_gym/
│   ├── agents/
│   │   ├── Agent.py
│   │   ├── BaselineAgents.py
│   │   ├── PolicyGradientAgent.py
│   │   └── SbAgent.py
│   ├── gym/
│   │   ├── AMMEnvironment.py
│   │   ├── ArbitrageurEnvironment.py
│   │   ├── GymnasiumAMMEnvironment.py
│   │   ├── ModelDynamics.py
│   │   ├── StableBaselinesAMMEnvironment.py
│   │   ├── domain_randomization.py
│   │   ├── index_names.py
│   │   ├── observation_features.py
│   │   ├── simulation_core.py
│   │   └── helpers/AMM_utils.py
│   ├── rewards/RewardFunctions.py
│   ├── stochastic_processes/
│   │   ├── StochasticProcessModel.py
│   │   ├── arrival_models.py
│   │   ├── fee_accounting_models.py
│   │   ├── midprice_models.py
│   │   └── price_impact_models.py
│   └── wrappers.py
├── experiments/
│   ├── helpers.py
│   ├── domain_randomization_rl/
│   │   ├── aggregate_domain_randomized_seed_sweep.py
│   │   ├── evaluate_domain_randomization_lp_agents.py
│   │   ├── evaluation_grid.py
│   │   ├── fast_lp_evaluation.py
│   │   ├── policy_behavior_diagnostics.py
│   │   └── train_domain_randomization_lp_agent.py
│   └── sweep_dense_exponential_width.py
├── hpc/
│   ├── .env.example
│   ├── README.md
│   ├── sbatch_aggregate_domain_randomized_seed_sweep.sh
│   ├── sbatch_domain_randomization_lp_seed_sweep_cpu.sh
│   ├── sbatch_domain_randomization_lp_smoke.sh
│   ├── sbatch_train_domain_randomization_lp_agent_cpu.sh
│   └── setup_env.sh
├── skills/
├── tests/
└── requirements.txt
```

## Citation

If you find the repository useful, please cite the study:

```bibtex
@inproceedings{chionas2026saife,
  title={SAiFE-gym: Model-based Environments for Automated Market Making with Concentrated Liquidity},
  author={Chionas, Georgios and Kleitsikas, Charalampos and Leonardos, Stefanos and S{\'a}nchez-Betancourt, Leandro and Ventre, Carmine},
  booktitle={Proceedings of the 7th ACM International Conference on AI in Finance},
  pages={604--612},
  year={2026}
}
```

## License

Original contributions to SAiFE_gym are licensed under the [MIT License](LICENSE).
Reused code from `mbt_gym` retains its BSD-3-Clause terms. See
[third-party notices](THIRD_PARTY_NOTICES.md) for attribution and the full upstream
license.
