"""
Racing environment - Gymnasium-compatible environment.

Provides:
- Gym-style interface for RL
- Observation and action spaces
- Step and reset methods
- Reward calculation
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Try to import gymnasium, but don't fail if not installed
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None

from racenet.car.car import Car, CarConfig, CarInputs
from racenet.track.track import Track
from racenet.track.generator import TrackGenerator, GeneratorConfig
from racenet.simulation.simulator import Simulator
from racenet.scoring.scoring_system import ScoringSystem
from racenet.ml.spaces import ObservationSpace, ActionSpace, ObservationConfig, ActionConfig


@dataclass
class RaceEnvConfig:
    """Racing environment configuration."""
    # Time
    max_episode_time: float = 120.0
    dt: float = 0.01
    
    # Track
    generate_new_track: bool = True
    track_seed: int | None = None
    
    # Termination
    terminate_on_off_track: bool = False
    max_off_track_time: float = 5.0
    
    # Observation/Action
    observation_config: ObservationConfig | None = None
    action_config: ActionConfig | None = None


class RaceEnv:
    """Gymnasium-compatible racing environment.
    
    Single-agent racing environment for RL training.
    Supports both continuous and discrete action spaces.
    
    Usage:
        env = RaceEnv()
        obs = env.reset()
        
        while True:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
    
    Note: If gymnasium is installed, this class can be wrapped
    to fully implement the Gym interface.
    """
    
    def __init__(self, config: RaceEnvConfig | None = None):
        """Initialize environment.
        
        Args:
            config: Environment configuration
        """
        self.config = config or RaceEnvConfig()
        
        # Spaces
        self._obs_space = ObservationSpace(self.config.observation_config)
        self._action_space = ActionSpace(self.config.action_config)
        
        # Components
        self._car: Optional[Car] = None
        self._track: Optional[Track] = None
        self._track_generator = TrackGenerator()
        self._scoring = ScoringSystem()
        
        # State
        self._time: float = 0.0
        self._steps: int = 0
        self._off_track_time: float = 0.0
        self._last_distance: float = 0.0
        
        # Initialize
        self._initialized = False
    
    @property
    def observation_space(self):
        """Get observation space.
        
        Returns:
            Gymnasium space if available, else dimensions
        """
        if HAS_GYMNASIUM:
            return spaces.Box(
                low=self._obs_space.get_low(),
                high=self._obs_space.get_high(),
                shape=self._obs_space.shape,
                dtype=np.float32,
            )
        return {
            "shape": self._obs_space.shape,
            "low": self._obs_space.get_low(),
            "high": self._obs_space.get_high(),
        }
    
    @property
    def action_space(self):
        """Get action space.
        
        Returns:
            Gymnasium space if available, else dimensions
        """
        if HAS_GYMNASIUM:
            if self._action_space.is_continuous:
                return spaces.Box(
                    low=self._action_space.get_low(),
                    high=self._action_space.get_high(),
                    dtype=np.float32,
                )
            else:
                return spaces.Discrete(self._action_space.dimension)
        return {
            "continuous": self._action_space.is_continuous,
            "dimension": self._action_space.dimension,
            "low": self._action_space.get_low(),
            "high": self._action_space.get_high(),
        }
    
    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode.
        
        Args:
            seed: Random seed for track generation
            options: Additional reset options
            
        Returns:
            Tuple of (initial_observation, info)
        """
        # Generate or reuse track
        if self.config.generate_new_track or self._track is None:
            track_seed = seed or self.config.track_seed
            self._track = self._track_generator.generate_with_seed(
                track_seed or np.random.randint(0, 10000),
            )
        
        # Create car
        self._car = Car()
        
        # Position at start
        start_pos = self._track.get_position_at_distance(0)
        start_heading = self._track.get_heading_at_distance(0)
        self._car.reset(x=start_pos[0], y=start_pos[1], heading=start_heading)
        
        # Reset scoring
        self._scoring.reset()
        self._scoring.start_lap(0.0)
        
        # Reset state
        self._time = 0.0
        self._steps = 0
        self._off_track_time = 0.0
        self._last_distance = 0.0
        self._obs_space.reset()
        
        self._initialized = True
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray | int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take environment step.
        
        Args:
            action: Agent action
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Decode action
        action_dict = self._action_space.decode(action)
        
        # Create car inputs
        inputs = CarInputs(
            throttle=action_dict["throttle"],
            brake=action_dict["brake"],
            steering=action_dict["steering"],
            shift_up=action_dict["shift_up"],
            shift_down=action_dict["shift_down"],
        )
        
        # Step car
        self._car.step(inputs, self.config.dt)
        self._time += self.config.dt
        self._steps += 1
        
        # Get track info
        track_info = self._track.get_track_info_at_position(
            self._car.state.x, self._car.state.y
        )
        
        distance = track_info["distance_m"]
        lateral_offset = track_info["lateral_offset_m"]
        is_off_track = abs(lateral_offset) > track_info["width_m"] / 2
        
        # Update off-track time
        if is_off_track:
            self._off_track_time += self.config.dt
        else:
            self._off_track_time = 0.0
        
        # Check for lap completion
        # Use half track length as threshold to detect wraparound at start/finish
        LAP_DETECTION_THRESHOLD = self._track.length / 2
        if distance < self._last_distance - LAP_DETECTION_THRESHOLD:
            # Crossed start/finish line
            self._scoring.complete_lap(self._time)
        
        self._last_distance = distance
        
        # Calculate reward
        reward = self._scoring.step(
            speed=self._car.speed,
            distance=distance,
            throttle=inputs.throttle,
            steering=inputs.steering,
            brake=inputs.brake,
            lateral_offset=lateral_offset,
            is_off_track=is_off_track,
            dt=self.config.dt,
        )
        
        # Check termination
        terminated = False
        truncated = False
        
        if self.config.terminate_on_off_track:
            if self._off_track_time > self.config.max_off_track_time:
                terminated = True
        
        if self._time >= self.config.max_episode_time:
            truncated = True
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["track_info"] = track_info
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Observation array
        """
        telemetry = self._car.get_telemetry()
        
        # Get track lookahead info
        track_info = self._track.get_track_info_at_position(
            self._car.state.x, self._car.state.y
        )
        
        # Add lookahead curvatures
        lookahead_curvatures = []
        lookahead_distances = []
        
        current_dist = track_info["distance_m"]
        for i in range(self._obs_space.config.lookahead_points):
            lookahead_dist = (i + 1) * (
                self._obs_space.config.lookahead_distance_m / 
                self._obs_space.config.lookahead_points
            )
            point_dist = (current_dist + lookahead_dist) % self._track.length
            
            curv = self._track.get_curvature_at_distance(point_dist)
            lookahead_curvatures.append(curv)
            lookahead_distances.append(lookahead_dist)
        
        track_info["lookahead_curvatures"] = lookahead_curvatures
        track_info["lookahead_distances"] = lookahead_distances
        
        return self._obs_space.extract(telemetry, track_info)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary.
        
        Returns:
            Info dictionary
        """
        return {
            "time": self._time,
            "steps": self._steps,
            "speed_kph": self._car.speed_kph,
            "off_track_time": self._off_track_time,
            "scoring": self._scoring.get_metrics(),
        }
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment.
        
        Currently a no-op. Future versions may support visualization.
        
        Args:
            mode: Render mode
            
        Returns:
            None or RGB array
        """
        return None
    
    def close(self) -> None:
        """Close environment."""
        self._initialized = False
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current car telemetry.
        
        Returns:
            Full telemetry dictionary
        """
        if self._car:
            return self._car.get_telemetry()
        return {}
