#!/usr/bin/env python3
"""
ML Training Example

This example demonstrates how to:
1. Use the Gymnasium-compatible environment
2. Implement a simple training loop
3. Access observations and rewards
4. Track training progress

Note: This example uses random actions. For real training, replace
with an actual RL algorithm (e.g., from Stable-Baselines3).

Run with: python train_agent.py
"""

import numpy as np
from racenet.ml import RaceEnv, RaceEnvConfig


def simple_policy(observation: np.ndarray) -> np.ndarray:
    """A simple heuristic policy for demonstration.
    
    This is NOT a trained policy - just a demonstration of the interface.
    In practice, you would use an RL algorithm like PPO or SAC.
    """
    # Extract relevant observation values
    speed = observation[0]  # Normalized speed
    
    # Simple logic: accelerate when slow, maintain when fast
    if speed < 0.3:
        throttle = 1.0
        brake = 0.0
    elif speed > 0.5:
        throttle = 0.5
        brake = 0.0
    else:
        throttle = 0.7
        brake = 0.0
    
    # Small random steering for variety
    steering = np.random.uniform(-0.1, 0.1)
    
    return np.array([steering, throttle, brake], dtype=np.float32)


def main():
    print("=" * 60)
    print("RaceNet ML Training Example")
    print("=" * 60)
    
    # Step 1: Create environment
    print("\n1. Creating environment...")
    config = RaceEnvConfig(
        max_episode_time=30.0,  # 30 second episodes
        dt=0.01,                # 100 Hz simulation
    )
    env = RaceEnv(config)
    
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Step 2: Run training episodes
    print("\n2. Running training episodes...")
    
    num_episodes = 5
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        steps = 0
        
        done = False
        while not done:
            # Get action from policy
            action = simple_policy(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"   Episode {episode + 1}: "
              f"Reward = {total_reward:.2f}, "
              f"Steps = {steps}, "
              f"Speed = {info['speed_kph']:.1f} km/h")
    
    env.close()
    
    # Step 3: Display training statistics
    print("\n3. Training Statistics:")
    print(f"   Episodes: {num_episodes}")
    print(f"   Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"   Std Reward: {np.std(episode_rewards):.2f}")
    print(f"   Mean Length: {np.mean(episode_lengths):.0f} steps")
    
    # Step 4: Show environment info structure
    print("\n4. Environment Info Structure:")
    obs, info = env.reset()
    print(f"   Available info keys: {list(info.keys())}")
    
    if 'scoring' in info:
        scoring = info['scoring']
        print(f"   Scoring metrics available: {list(scoring.keys())[:5]}...")
    
    print("\n" + "=" * 60)
    print("Training example complete!")
    print("=" * 60)
    print("\nFor real training, use an RL library like Stable-Baselines3:")
    print("  pip install stable-baselines3")
    print("  from stable_baselines3 import PPO")
    print("  model = PPO('MlpPolicy', env, verbose=1)")
    print("  model.learn(total_timesteps=100000)")


if __name__ == "__main__":
    main()
