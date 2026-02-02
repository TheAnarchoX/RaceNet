"""
Observation and action spaces for ML integration.

Provides:
- Observation space definitions
- Action space definitions
- State normalization
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import numpy as np


@dataclass
class ObservationConfig:
    """Configuration for observation space."""
    # What to include
    include_speed: bool = True
    include_acceleration: bool = True
    include_yaw_rate: bool = True
    include_engine: bool = True
    include_gear: bool = True
    include_tire_info: bool = True
    include_electronics: bool = True
    include_track_info: bool = True
    
    # Track lookahead
    lookahead_points: int = 10
    lookahead_distance_m: float = 200.0
    
    # History
    history_length: int = 0  # Number of previous observations to include


class ObservationSpace:
    """Defines the observation space for ML agents.
    
    Provides a standardized way to extract normalized
    observations from car and track state.
    """
    
    def __init__(self, config: ObservationConfig | None = None):
        """Initialize observation space.
        
        Args:
            config: Observation configuration
        """
        self.config = config or ObservationConfig()
        
        # Calculate observation dimension
        self._dimension = self._calculate_dimension()
        
        # History buffer
        self._history: List[np.ndarray] = []
    
    def _calculate_dimension(self) -> int:
        """Calculate total observation dimension.
        
        Returns:
            Observation vector size
        """
        dim = 0
        
        if self.config.include_speed:
            dim += 1  # speed
        
        if self.config.include_acceleration:
            dim += 2  # lateral_g, longitudinal_g
        
        if self.config.include_yaw_rate:
            dim += 1  # yaw_rate
        
        if self.config.include_engine:
            dim += 2  # rpm, throttle
        
        if self.config.include_gear:
            dim += 1  # gear
        
        if self.config.include_tire_info:
            dim += 2  # avg_temp, avg_wear
        
        if self.config.include_electronics:
            dim += 2  # tc_active, abs_active
        
        if self.config.include_track_info:
            # Lookahead curvatures and distances
            dim += self.config.lookahead_points * 2
        
        # History
        base_dim = dim
        dim += base_dim * self.config.history_length
        
        return dim
    
    @property
    def dimension(self) -> int:
        """Observation vector dimension."""
        return self._dimension
    
    @property
    def shape(self) -> Tuple[int]:
        """Observation shape."""
        return (self._dimension,)
    
    def get_low(self) -> np.ndarray:
        """Get lower bounds for observations.
        
        Returns:
            Array of minimum values
        """
        return np.full(self._dimension, -1.0, dtype=np.float32)
    
    def get_high(self) -> np.ndarray:
        """Get upper bounds for observations.
        
        Returns:
            Array of maximum values
        """
        return np.full(self._dimension, 1.0, dtype=np.float32)
    
    def extract(
        self,
        car_state: Dict[str, Any],
        track_info: Dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Extract observation from state.
        
        Args:
            car_state: Car telemetry/state dictionary
            track_info: Track information at car position
            
        Returns:
            Normalized observation vector
        """
        obs = []
        
        state = car_state.get("state", {})
        engine = car_state.get("engine", {})
        transmission = car_state.get("transmission", {})
        electronics = car_state.get("electronics", {})
        tires = car_state.get("tires", {})
        
        if self.config.include_speed:
            # Normalize speed to ~300 kph
            speed = state.get("speed_mps", 0) / 83.33
            obs.append(np.clip(speed, 0, 1))
        
        if self.config.include_acceleration:
            # G-forces normalized to ~3g
            lat_g = state.get("lateral_g", 0) / 3.0
            long_g = state.get("longitudinal_g", 0) / 3.0
            obs.append(np.clip(lat_g, -1, 1))
            obs.append(np.clip(long_g, -1, 1))
        
        if self.config.include_yaw_rate:
            yaw = state.get("yaw_rate", 0) / 2.0
            obs.append(np.clip(yaw, -1, 1))
        
        if self.config.include_engine:
            rpm = engine.get("rpm", 0) / 9000.0
            throttle = engine.get("throttle", 0)
            obs.append(np.clip(rpm, 0, 1))
            obs.append(np.clip(throttle, 0, 1))
        
        if self.config.include_gear:
            gear = transmission.get("gear", 0) / 6.0
            obs.append(np.clip(gear, 0, 1))
        
        if self.config.include_tire_info:
            temp = tires.get("avg_temp_c", 50) / 100.0
            wear = tires.get("avg_wear_pct", 0) / 100.0
            obs.append(np.clip(temp, 0, 1.5))
            obs.append(np.clip(wear, 0, 1))
        
        if self.config.include_electronics:
            tc = float(electronics.get("tc_active", False))
            abs_active = float(electronics.get("abs_active", False))
            obs.append(tc)
            obs.append(abs_active)
        
        if self.config.include_track_info and track_info:
            # Add lookahead curvatures
            curvatures = track_info.get("lookahead_curvatures", [])
            distances = track_info.get("lookahead_distances", [])
            
            for i in range(self.config.lookahead_points):
                if i < len(curvatures):
                    # Normalize curvature (1/30m radius = max tight turn)
                    curv = curvatures[i] / 0.033
                    dist = distances[i] / self.config.lookahead_distance_m
                else:
                    curv = 0.0
                    dist = 1.0
                
                obs.append(np.clip(curv, -1, 1))
                obs.append(np.clip(dist, 0, 1))
        elif self.config.include_track_info:
            # Pad with zeros if no track info
            for _ in range(self.config.lookahead_points * 2):
                obs.append(0.0)
        
        # Create current observation
        current_obs = np.array(obs, dtype=np.float32)
        
        # Add history if configured
        if self.config.history_length > 0:
            self._history.append(current_obs.copy())
            
            # Limit history
            while len(self._history) > self.config.history_length:
                self._history.pop(0)
            
            # Build full observation with history
            full_obs = [current_obs]
            for i in range(self.config.history_length):
                if i < len(self._history):
                    full_obs.append(self._history[-(i+1)])
                else:
                    full_obs.append(np.zeros_like(current_obs))
            
            return np.concatenate(full_obs)
        
        return current_obs
    
    def reset(self) -> None:
        """Reset observation history."""
        self._history = []


@dataclass
class ActionConfig:
    """Configuration for action space."""
    # Control types
    continuous: bool = True  # Continuous vs discrete actions
    
    # Discrete action settings
    steering_bins: int = 5
    throttle_bins: int = 3
    brake_bins: int = 3
    
    # Action ranges (for continuous)
    steering_range: Tuple[float, float] = (-1.0, 1.0)
    throttle_range: Tuple[float, float] = (0.0, 1.0)
    brake_range: Tuple[float, float] = (0.0, 1.0)
    
    # Include gear control
    include_gear: bool = True


class ActionSpace:
    """Defines the action space for ML agents.
    
    Supports both continuous and discrete action spaces.
    """
    
    def __init__(self, config: ActionConfig | None = None):
        """Initialize action space.
        
        Args:
            config: Action configuration
        """
        self.config = config or ActionConfig()
        
        if self.config.continuous:
            self._dimension = 3  # steering, throttle, brake
            if self.config.include_gear:
                self._dimension += 1  # gear change
        else:
            self._dimension = (
                self.config.steering_bins *
                self.config.throttle_bins *
                self.config.brake_bins
            )
            if self.config.include_gear:
                self._dimension *= 3  # up, neutral, down
    
    @property
    def dimension(self) -> int:
        """Action dimension (continuous) or count (discrete)."""
        return self._dimension
    
    @property
    def is_continuous(self) -> bool:
        """Whether action space is continuous."""
        return self.config.continuous
    
    def get_low(self) -> np.ndarray:
        """Get action space lower bounds.
        
        Returns:
            Array of minimum values
        """
        if self.config.continuous:
            low = [
                self.config.steering_range[0],
                self.config.throttle_range[0],
                self.config.brake_range[0],
            ]
            if self.config.include_gear:
                low.append(-1.0)  # Down
            return np.array(low, dtype=np.float32)
        else:
            return np.array([0], dtype=np.int32)
    
    def get_high(self) -> np.ndarray:
        """Get action space upper bounds.
        
        Returns:
            Array of maximum values
        """
        if self.config.continuous:
            high = [
                self.config.steering_range[1],
                self.config.throttle_range[1],
                self.config.brake_range[1],
            ]
            if self.config.include_gear:
                high.append(1.0)  # Up
            return np.array(high, dtype=np.float32)
        else:
            return np.array([self._dimension - 1], dtype=np.int32)
    
    def decode(self, action: np.ndarray | int) -> Dict[str, float]:
        """Decode action to car inputs.
        
        Args:
            action: Action from agent
            
        Returns:
            Dictionary of car inputs
        """
        if self.config.continuous:
            return self._decode_continuous(action)
        else:
            return self._decode_discrete(int(action))
    
    def _decode_continuous(self, action: np.ndarray) -> Dict[str, float]:
        """Decode continuous action.
        
        Args:
            action: Continuous action array
            
        Returns:
            Car inputs
        """
        inputs = {
            "steering": np.clip(float(action[0]), -1.0, 1.0),
            "throttle": np.clip(float(action[1]), 0.0, 1.0),
            "brake": np.clip(float(action[2]), 0.0, 1.0),
            "shift_up": False,
            "shift_down": False,
        }
        
        if self.config.include_gear and len(action) > 3:
            if action[3] > 0.5:
                inputs["shift_up"] = True
            elif action[3] < -0.5:
                inputs["shift_down"] = True
        
        return inputs
    
    def _decode_discrete(self, action: int) -> Dict[str, float]:
        """Decode discrete action.
        
        Args:
            action: Discrete action index
            
        Returns:
            Car inputs
        """
        # Decode from single index to multiple discrete values
        remaining = action
        
        steering_idx = remaining % self.config.steering_bins
        remaining //= self.config.steering_bins
        
        throttle_idx = remaining % self.config.throttle_bins
        remaining //= self.config.throttle_bins
        
        brake_idx = remaining % self.config.brake_bins
        remaining //= self.config.brake_bins
        
        gear_idx = 1  # Neutral default
        if self.config.include_gear:
            gear_idx = remaining % 3
        
        # Convert to continuous values
        steering = -1.0 + 2.0 * steering_idx / (self.config.steering_bins - 1)
        throttle = throttle_idx / (self.config.throttle_bins - 1)
        brake = brake_idx / (self.config.brake_bins - 1)
        
        return {
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
            "shift_up": gear_idx == 2,
            "shift_down": gear_idx == 0,
        }
    
    def sample(self) -> np.ndarray:
        """Sample random action.
        
        Returns:
            Random action
        """
        if self.config.continuous:
            return np.random.uniform(self.get_low(), self.get_high())
        else:
            return np.array([np.random.randint(0, self._dimension)])
