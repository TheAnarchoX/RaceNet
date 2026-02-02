# ML Training Example

This example demonstrates how to use RaceNet's Gymnasium-compatible environment for reinforcement learning.

## What You'll Learn

- How to create and configure the racing environment
- Understanding the observation and action spaces
- Running training episodes
- Tracking and analyzing rewards

## Running the Example

```bash
python train_agent.py
```

## Environment Details

### Observation Space

The observation is a normalized vector containing:
- Speed (normalized to max ~300 km/h)
- Lateral and longitudinal G-forces
- Yaw rate
- Engine RPM and throttle position
- Current gear
- Tire temperatures and wear
- TC/ABS activation status
- Track lookahead (curvatures ahead)

### Action Space (Continuous)

3-dimensional continuous action:
- **Steering**: -1.0 (full left) to 1.0 (full right)
- **Throttle**: 0.0 (none) to 1.0 (full)
- **Brake**: 0.0 (none) to 1.0 (full)

## Basic Usage

```python
from racenet.ml import RaceEnv

# Create environment
env = RaceEnv()

# Reset and get initial observation
obs, info = env.reset()

# Training loop
for step in range(10000):
    action = agent.predict(obs)  # Your RL agent
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Using Stable-Baselines3

For real training, use a proper RL library:

```bash
pip install stable-baselines3
```

```python
from stable_baselines3 import PPO
from racenet.ml import RaceEnv

env = RaceEnv()

# Create and train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save trained model
model.save("racing_agent")

# Test trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Reward Structure

The reward function considers:
- **Speed reward**: Encourages going fast
- **Progress reward**: Encourages making progress along the track
- **Style reward**: Rewards smooth inputs
- **Penalties**: Off-track penalties reduce reward

## Configuration Options

```python
from racenet.ml import RaceEnv, RaceEnvConfig

config = RaceEnvConfig(
    max_episode_time=120.0,     # Episode length in seconds
    dt=0.01,                    # Physics timestep
    generate_new_track=True,    # New track each reset
    terminate_on_off_track=False,  # Continue even when off track
)

env = RaceEnv(config)
```

## Next Steps

- Experiment with different RL algorithms (SAC, TD3, A2C)
- Use the [multi_car](../multi_car/) example for population-based training
- Analyze agent performance with [telemetry_analysis](../telemetry_analysis/)
