---
name: stable-baselines3
description: Reinforcement learning algorithms (PPO, SAC, DQN, TD3, DDPG, A2C) with a scikit-learn-like API. Use for SB3 training, evaluation, callbacks, and Gymnasium integration. For SAiFE_gym batched simulation, use the project's native SB3 adapter and shipped experiment workflows.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Stable Baselines3

## Overview

Stable Baselines3 (SB3) is a PyTorch-based library providing reliable implementations of reinforcement learning algorithms. This skill provides comprehensive guidance for training RL agents, creating custom environments, implementing callbacks, and optimizing training workflows using SB3's unified API.

The repository pins SB3 2.7.1 in [requirements.txt](../../requirements.txt).
The library examples below use standard Gymnasium environments. For SAiFE_gym,
use [StableBaselinesAMMEnvironment](../../SAiFE_gym/gym/StableBaselinesAMMEnvironment.py)
for native batched SB3 training. Use the
[Gymnasium adapters](../../SAiFE_gym/gym/GymnasiumAMMEnvironment.py) for Gymnasium
wrappers and environment checks; the raw `AMMEnvironment` returns batched data.
See the [interface guide](../../README.md#environment-interfaces).

## Core Capabilities

### 1. Training RL Agents

**Basic Training Pattern:**

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("CartPole-v1")

# Initialize agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_cartpole")

# Load the model (without prior instantiation)
model = PPO.load("ppo_cartpole", env=env)
```

**Important Notes:**
- `total_timesteps` is a lower bound; actual training may exceed this due to batch collection
- Load with the algorithm class, for example `PPO.load(...)`; loading creates a new model
- The replay buffer is NOT saved with the model to save space

**Algorithm Selection:**
Use the [SB3 algorithm table][sb3-algorithms] for supported action spaces. Quick reference:
- **PPO/A2C**: Supports Box, Discrete, MultiDiscrete, and MultiBinary actions
- **SAC/TD3**: Continuous control, off-policy, sample-efficient
- **DQN**: Discrete actions, off-policy
- **HER**: Goal-conditioned tasks

For repository training examples, see the nominal PPO helper in
[experiments/helpers.py](../../experiments/helpers.py) and the complete
[robust LP training script](../../experiments/train_robust_lp_agent.py).
The [README training section](../../README.md#training) provides runnable commands;
the [SB3 examples][sb3-examples] cover generic environments.

### 2. Custom Environments

**Requirements:**
Custom environments must inherit from `gymnasium.Env` and implement:
- `__init__()`: Define action_space and observation_space
- `reset(seed, options)`: Return initial observation and info dict
- `step(action)`: Return observation, reward, terminated, truncated, info
- `render()`: Visualization (optional)
- `close()`: Cleanup resources

**Key Constraints:**
- Image observations must be `np.uint8` in range [0, 255]
- Use channel-first format when possible (channels, height, width)
- SB3 normalizes images automatically by dividing by 255
- Set `normalize_images=False` in policy_kwargs if pre-normalized
- SB3 does NOT support `Discrete` or `MultiDiscrete` spaces with `start!=0`

**Validation:**
```python
from stable_baselines3.common.env_checker import check_env

check_env(env, warn=True)
```

See the [official custom-environment example][sb3-custom-env] for a template.
For this simulator, construct a single-trajectory `GymnasiumAMMEnvironment`
before using `check_env`; use the native SB3 adapter for batched PPO training.

### 3. Vectorized Environments

**Purpose:**
Vectorized environments run multiple environment instances in parallel, accelerating training and enabling certain wrappers (frame-stacking, normalization).

**Types:**
- **DummyVecEnv**: Sequential execution on current process (for lightweight environments)
- **SubprocVecEnv**: Parallel execution across processes (for compute-heavy environments)

**Quick Setup:**
```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    # Create 4 parallel environments; run this example as a Python script.
    env = make_vec_env("CartPole-v1", n_envs=4, vec_env_cls=SubprocVecEnv)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    env.close()
```

**Off-Policy Optimization:**
When using multiple environments with off-policy algorithms (SAC, TD3, DQN), set `gradient_steps=-1` to perform one gradient update per environment step, balancing wall-clock time and sample efficiency.

**API Differences:**
- `reset()` returns only observations (info available in `vec_env.reset_infos`)
- `step()` returns 4-tuple: `(obs, rewards, dones, infos)` not 5-tuple
- Environments auto-reset after episodes
- Terminal observations available via `infos[env_idx]["terminal_observation"]`

See the [SB3 vector-environment guide][sb3-vec-envs] for wrapper and API details.
SAiFE_gym already batches trajectories inside one simulator. Its
[SB3 adapter](../../SAiFE_gym/gym/StableBaselinesAMMEnvironment.py) and
[action wrappers](../../SAiFE_gym/wrappers.py) support this native batch directly.

### 4. Callbacks for Monitoring and Control

**Purpose:**
Callbacks enable monitoring metrics, saving checkpoints, implementing early stopping, and custom training logic without modifying core algorithms.

**Common Callbacks:**
- **EvalCallback**: Evaluate periodically and save best model
- **CheckpointCallback**: Save model checkpoints at intervals
- **StopTrainingOnRewardThreshold**: Stop when target reward reached
- **ProgressBarCallback**: Display training progress with timing

**Custom Callback Structure:**
```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def _on_training_start(self):
        # Called before first rollout
        pass

    def _on_step(self):
        # Called after each environment step
        # Return False to stop training
        return True

    def _on_rollout_end(self):
        # Called at end of rollout
        pass
```

**Available Attributes:**
- `self.model`: The RL algorithm instance
- `self.num_timesteps`: Total environment steps
- `self.training_env`: The training environment

**Chaining Callbacks:**
```python
from stable_baselines3.common.callbacks import CallbackList

callback = CallbackList([eval_callback, checkpoint_callback, custom_callback])
model.learn(total_timesteps=10000, callback=callback)
```

See the [SB3 callback guide][sb3-callbacks]. Evaluation needs a separate
environment, and `eval_freq` counts callback calls: one per vector batch step.
The nominal helper creates an independent evaluation simulator and evaluates
every ten rollouts. The robust training script implements its own convergence
callback; follow those shipped examples when changing evaluation behavior.

### 5. Model Persistence and Inspection

**Saving and Loading:**
For a normalized policy, construct a fresh evaluation `VecEnv` with the same
observation and action wrappers as training, then restore its statistics before
attaching the loaded model. Here `vec_env` is the training `VecNormalize`, and
`eval_vec_env` is that fresh evaluation environment:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Save model
model.save("model_name")

# Save normalization statistics (if using VecNormalize)
vec_env.save("vec_normalize.pkl")

# Restore frozen evaluation normalization, then load the model against it
eval_vec_env = VecNormalize.load("vec_normalize.pkl", eval_vec_env)
eval_vec_env.training = False
eval_vec_env.norm_reward = False
model = PPO.load("model_name", env=eval_vec_env)
```

**Parameter Access:**
```python
# Get parameters
params = model.get_parameters()

# Set parameters
model.set_parameters(params)

# Access PyTorch state dict
state_dict = model.policy.state_dict()
```

### 6. Evaluation and Recording

**Evaluation:**
```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True
)
```

**Video Recording:**
```python
from stable_baselines3.common.vec_env import VecVideoRecorder

# Wrap environment with video recorder
env = VecVideoRecorder(
    env,
    "videos/",
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200
)
```

For SAiFE_gym, use the
[standalone evaluator](../../experiments/evaluate_robust_lp_agents.py) and its
[saved-run walkthrough](../../hpc/README_BARKLA2.md#evaluate-higher-arrival-rates-without-training).
It restores the saved run's configuration and matching normalization for PPO.
Generic evaluation is documented in the [SB3 evaluation helper][sb3-evaluation].
Video recording requires an environment supporting `rgb_array` rendering;
SAiFE_gym's adapters currently provide no rendering.

### 7. Advanced Features

**Learning Rate Schedules:**
```python
def linear_schedule(initial_value):
    def func(progress_remaining):
        # progress_remaining goes from 1 to 0
        return progress_remaining * initial_value
    return func

model = PPO("MlpPolicy", env, learning_rate=linear_schedule(0.001))
```

**Multi-Input Policies (Dict Observations):**
```python
model = PPO("MultiInputPolicy", env, verbose=1)
```
Use when observations are dictionaries (e.g., combining images with sensor data).

**Hindsight Experience Replay:**
```python
from stable_baselines3 import SAC, HerReplayBuffer

model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
)
```

**TensorBoard Integration:**
```python
model = PPO("MlpPolicy", env, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=10000)
```

## Workflow Guidance

**Starting a New RL Project:**

1. **Define the problem**: Identify observation space, action space, and reward structure
2. **Choose algorithm**: Check the [supported action spaces][sb3-algorithms]
3. **Create/adapt environment**: Follow the [custom-environment guide][sb3-custom-env] or use the shipped SAiFE_gym adapters
4. **Validate environment**: Run `check_env()` on a single Gymnasium environment; use the repository tests for the native SB3 batch adapter
5. **Set up training**: Follow the [README examples](../../README.md#training) and shipped experiment scripts
6. **Add monitoring**: Implement callbacks for evaluation and checkpointing
7. **Optimize performance**: Consider vectorized environments for speed
8. **Evaluate and iterate**: Follow the [saved-run walkthrough](../../hpc/README_BARKLA2.md#evaluate-higher-arrival-rates-without-training) for SAiFE_gym, or the [SB3 evaluation helper][sb3-evaluation] for generic environments

**Common Issues:**

- **Memory errors**: Reduce `buffer_size` for off-policy algorithms or use fewer parallel environments
- **Slow training**: Consider SubprocVecEnv for parallel environments
- **Unstable training**: Try different algorithms, tune hyperparameters, or check reward scaling
- **Import errors**: Ensure `stable_baselines3` is installed: `uv pip install stable-baselines3[extra]`

## Resources

### Shipped repository examples

- [Nominal PPO helper](../../experiments/helpers.py): training and independent evaluation setup
- [Robust LP training](../../experiments/train_robust_lp_agent.py): domain randomization, convergence callbacks, model and normalization persistence
- [Standalone evaluation](../../experiments/evaluate_robust_lp_agents.py): evaluation of saved runs
- [Gymnasium adapters](../../SAiFE_gym/gym/GymnasiumAMMEnvironment.py): single and vector Gymnasium interfaces
- [Native SB3 adapter](../../SAiFE_gym/gym/StableBaselinesAMMEnvironment.py): batched simulation through SB3's VecEnv API

### Official SB3 2.7.1 documentation

- [Algorithm capabilities][sb3-algorithms]
- [Custom environments][sb3-custom-env]
- [Vector environments and normalization][sb3-vec-envs]
- [Callbacks][sb3-callbacks]
- [Training and persistence examples][sb3-examples]
- [Evaluation helper][sb3-evaluation]

[sb3-algorithms]: https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/algos.html
[sb3-custom-env]: https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/custom_env.html
[sb3-vec-envs]: https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/vec_envs.html
[sb3-callbacks]: https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/callbacks.html
[sb3-examples]: https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/examples.html
[sb3-evaluation]: https://stable-baselines3.readthedocs.io/en/v2.7.1/common/evaluation.html

## Installation

```bash
# Basic installation
uv pip install stable-baselines3

# With extra dependencies (Tensorboard, etc.)
uv pip install stable-baselines3[extra]
```
