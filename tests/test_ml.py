"""Basic tests for the RaceNet ML module."""

import pytest
import numpy as np

from racenet.ml.spaces import ObservationSpace, ActionSpace, ObservationConfig, ActionConfig
from racenet.ml.multi_car import MultiCarManager, MultiCarConfig
from racenet.ml.environment import RaceEnv, RaceEnvConfig


class TestObservationSpace:
    """Test observation space."""
    
    def test_observation_dimension(self):
        """Test observation space has correct dimension."""
        obs_space = ObservationSpace()
        
        assert obs_space.dimension > 0
        
    def test_observation_bounds(self):
        """Test observation bounds are correct."""
        obs_space = ObservationSpace()
        
        low = obs_space.get_low()
        high = obs_space.get_high()
        
        assert len(low) == obs_space.dimension
        assert len(high) == obs_space.dimension
        assert all(low <= high)
        
    def test_observation_extraction(self):
        """Test observation extraction from telemetry."""
        obs_space = ObservationSpace()
        
        # Mock telemetry
        telemetry = {
            "state": {"speed_mps": 50.0, "lateral_g": 0.5, "longitudinal_g": 0.2, "yaw_rate": 0.1},
            "engine": {"rpm": 6000, "throttle": 0.8},
            "transmission": {"gear": 4},
            "electronics": {"tc_active": False, "abs_active": False},
            "tires": {"avg_temp_c": 80, "avg_wear_pct": 10},
        }
        
        obs = obs_space.extract(telemetry)
        
        assert len(obs) == obs_space.dimension
        assert obs.dtype == np.float32


class TestActionSpace:
    """Test action space."""
    
    def test_continuous_action_space(self):
        """Test continuous action space."""
        config = ActionConfig(continuous=True)
        action_space = ActionSpace(config)
        
        assert action_space.is_continuous
        assert action_space.dimension >= 3  # steering, throttle, brake
        
    def test_discrete_action_space(self):
        """Test discrete action space."""
        config = ActionConfig(continuous=False)
        action_space = ActionSpace(config)
        
        assert not action_space.is_continuous
        assert action_space.dimension > 0
        
    def test_action_decode_continuous(self):
        """Test continuous action decoding."""
        action_space = ActionSpace(ActionConfig(continuous=True))
        
        action = np.array([0.5, 0.8, 0.0, 0.0])
        inputs = action_space.decode(action)
        
        assert "steering" in inputs
        assert "throttle" in inputs
        assert "brake" in inputs
        assert inputs["throttle"] == 0.8
        
    def test_action_sample(self):
        """Test action sampling."""
        action_space = ActionSpace()
        
        action = action_space.sample()
        
        assert action is not None


class TestMultiCarManager:
    """Test multi-car manager."""
    
    def test_manager_creation(self):
        """Test manager creation."""
        manager = MultiCarManager()
        
        assert manager.population_size == 0
        assert manager.current_generation == 0
        
    def test_population_initialization(self):
        """Test population initialization."""
        config = MultiCarConfig(population_size=10)
        manager = MultiCarManager(config)
        
        cars = manager.initialize_population()
        
        assert len(cars) == 10
        assert manager.population_size == 10
        
    def test_fitness_tracking(self):
        """Test fitness tracking."""
        config = MultiCarConfig(population_size=5)
        manager = MultiCarManager(config)
        manager.initialize_population()
        
        for i, car in enumerate(manager.population):
            manager.set_fitness(car.car_id, float(i))
        
        best = manager.get_best_car()
        assert manager.get_fitness(best.car_id) == 4.0
        
    def test_generation_advance(self):
        """Test generation advancement."""
        manager = MultiCarManager()
        manager.initialize_population()
        
        assert manager.current_generation == 0
        
        manager.next_generation()
        
        assert manager.current_generation == 1


class TestRaceEnv:
    """Test racing environment."""
    
    def test_env_creation(self):
        """Test environment creation."""
        env = RaceEnv()
        
        assert env.observation_space is not None
        assert env.action_space is not None
        
    def test_env_reset(self):
        """Test environment reset."""
        env = RaceEnv()
        
        obs, info = env.reset()
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert info is not None
        
    def test_env_step(self):
        """Test environment step."""
        env = RaceEnv()
        env.reset()
        
        # Random action
        action = np.array([0.0, 0.5, 0.0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
    def test_env_episode(self):
        """Test running a short episode."""
        config = RaceEnvConfig(max_episode_time=1.0)
        env = RaceEnv(config)
        
        obs, info = env.reset()
        
        steps = 0
        while True:
            action = np.array([0.0, 0.5, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if terminated or truncated:
                break
            
            if steps > 200:  # Safety limit
                break
        
        assert steps > 0
        env.close()
