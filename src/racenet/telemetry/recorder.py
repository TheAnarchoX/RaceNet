"""
Telemetry recorder - Records car telemetry over time.

Provides:
- Multi-channel recording
- Car state capture
- Lap-based organization
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

from racenet.telemetry.channel import TelemetryChannel, ChannelConfig
from racenet.car.car import Car


# Standard channel definitions for GT3 cars
STANDARD_CHANNELS = {
    # Motion
    "speed_kph": ChannelConfig("speed_kph", "km/h", 0, 400, 1),
    "lateral_g": ChannelConfig("lateral_g", "g", -5, 5, 2),
    "longitudinal_g": ChannelConfig("longitudinal_g", "g", -5, 5, 2),
    "yaw_rate": ChannelConfig("yaw_rate", "rad/s", -5, 5, 3),
    
    # Engine
    "rpm": ChannelConfig("rpm", "rpm", 0, 12000, 0),
    "throttle": ChannelConfig("throttle", "%", 0, 100, 1),
    "engine_temp": ChannelConfig("engine_temp", "°C", 0, 150, 1),
    
    # Brakes
    "brake": ChannelConfig("brake", "%", 0, 100, 1),
    "brake_bias": ChannelConfig("brake_bias", "%", 40, 70, 1),
    
    # Transmission
    "gear": ChannelConfig("gear", "", 0, 7, 0),
    
    # Steering
    "steering": ChannelConfig("steering", "deg", -180, 180, 1),
    
    # Electronics
    "tc_active": ChannelConfig("tc_active", "", 0, 1, 0),
    "abs_active": ChannelConfig("abs_active", "", 0, 1, 0),
    
    # Tires
    "tire_temp_fl": ChannelConfig("tire_temp_fl", "°C", 0, 150, 1),
    "tire_temp_fr": ChannelConfig("tire_temp_fr", "°C", 0, 150, 1),
    "tire_temp_rl": ChannelConfig("tire_temp_rl", "°C", 0, 150, 1),
    "tire_temp_rr": ChannelConfig("tire_temp_rr", "°C", 0, 150, 1),
    
    # Position
    "distance": ChannelConfig("distance", "m", 0, float('inf'), 1),
    "lateral_offset": ChannelConfig("lateral_offset", "m", -20, 20, 2),
}


@dataclass
class RecorderConfig:
    """Recorder configuration."""
    sample_rate_hz: float = 100.0  # Recording frequency
    channels: List[str] | None = None  # Channels to record (None = all)
    buffer_size: int = 100000  # Per-channel buffer size


class TelemetryRecorder:
    """Records car telemetry data over time.
    
    Captures telemetry from a car at regular intervals,
    organizing data into channels for analysis.
    
    Features:
    - Standard GT3 telemetry channels
    - Custom channel support
    - Lap-based organization
    - Real-time access to data
    """
    
    def __init__(
        self, 
        config: RecorderConfig | None = None,
        car: Car | None = None,
    ):
        """Initialize recorder.
        
        Args:
            config: Recorder configuration
            car: Car to record (can be set later)
        """
        self.config = config or RecorderConfig()
        self._car = car
        
        # Initialize channels
        self._channels: Dict[str, TelemetryChannel] = {}
        self._setup_channels()
        
        # Timing
        self._last_sample_time: float = 0.0
        self._sample_interval: float = 1.0 / self.config.sample_rate_hz
        
        # Lap tracking
        self._current_lap: int = 0
        self._lap_start_times: List[float] = [0.0]
    
    def _setup_channels(self) -> None:
        """Set up telemetry channels."""
        channel_names = self.config.channels or list(STANDARD_CHANNELS.keys())
        
        for name in channel_names:
            if name in STANDARD_CHANNELS:
                cfg = STANDARD_CHANNELS[name]
                cfg.buffer_size = self.config.buffer_size
            else:
                cfg = ChannelConfig(name=name, buffer_size=self.config.buffer_size)
            
            self._channels[name] = TelemetryChannel(cfg)
    
    def set_car(self, car: Car) -> None:
        """Set the car to record.
        
        Args:
            car: Car to record
        """
        self._car = car
    
    @property
    def channels(self) -> Dict[str, TelemetryChannel]:
        """Get all channels."""
        return self._channels
    
    @property
    def current_lap(self) -> int:
        """Current lap number."""
        return self._current_lap
    
    def get_channel(self, name: str) -> Optional[TelemetryChannel]:
        """Get channel by name.
        
        Args:
            name: Channel name
            
        Returns:
            Channel if found
        """
        return self._channels.get(name)
    
    def record(self, time: float, telemetry: Dict[str, Any] | None = None) -> bool:
        """Record telemetry at current time.
        
        Args:
            time: Current simulation time
            telemetry: Telemetry dict (fetches from car if None)
            
        Returns:
            True if sample was recorded
        """
        # Check sample rate
        if time - self._last_sample_time < self._sample_interval:
            return False
        
        self._last_sample_time = time
        
        # Get telemetry
        if telemetry is None and self._car is not None:
            telemetry = self._car.get_telemetry()
        
        if telemetry is None:
            return False
        
        # Extract and record values
        self._record_from_telemetry(time, telemetry)
        
        return True
    
    def _record_from_telemetry(self, time: float, telemetry: Dict[str, Any]) -> None:
        """Extract values from telemetry and record to channels.
        
        Args:
            time: Timestamp
            telemetry: Full telemetry dictionary
        """
        state = telemetry.get("state", {})
        engine = telemetry.get("engine", {})
        transmission = telemetry.get("transmission", {})
        electronics = telemetry.get("electronics", {})
        tires = telemetry.get("tires", {})
        
        # Map telemetry to channels
        channel_values = {
            "speed_kph": state.get("speed_kph", 0),
            "lateral_g": state.get("lateral_g", 0),
            "longitudinal_g": state.get("longitudinal_g", 0),
            "yaw_rate": state.get("yaw_rate", 0),
            "rpm": engine.get("rpm", 0),
            "throttle": engine.get("throttle", 0) * 100,
            "engine_temp": engine.get("temperature_c", 0),
            "gear": transmission.get("gear", 0),
            "brake_bias": electronics.get("brake_bias", 0.5) * 100,
            "tc_active": float(electronics.get("tc_active", False)),
            "abs_active": float(electronics.get("abs_active", False)),
        }
        
        # Tire temperatures
        if tires:
            channel_values["tire_temp_fl"] = tires.get("fl", {}).get("temperature_c", 0)
            channel_values["tire_temp_fr"] = tires.get("fr", {}).get("temperature_c", 0)
            channel_values["tire_temp_rl"] = tires.get("rl", {}).get("temperature_c", 0)
            channel_values["tire_temp_rr"] = tires.get("rr", {}).get("temperature_c", 0)
        
        # Record to channels
        for name, value in channel_values.items():
            if name in self._channels:
                self._channels[name].record(time, value)
    
    def new_lap(self, time: float) -> int:
        """Mark start of new lap.
        
        Args:
            time: Lap start time
            
        Returns:
            New lap number
        """
        self._current_lap += 1
        self._lap_start_times.append(time)
        return self._current_lap
    
    def get_lap_data(self, lap: int, channel: str) -> tuple[np.ndarray, np.ndarray]:
        """Get data for a specific lap.
        
        Args:
            lap: Lap number
            channel: Channel name
            
        Returns:
            Tuple of (times, values) for that lap
        """
        if channel not in self._channels:
            return np.array([]), np.array([])
        
        if lap < 0 or lap >= len(self._lap_start_times):
            return np.array([]), np.array([])
        
        start = self._lap_start_times[lap]
        end = self._lap_start_times[lap + 1] if lap + 1 < len(self._lap_start_times) else float('inf')
        
        return self._channels[channel].get_range(start, end)
    
    def get_current_values(self) -> Dict[str, float]:
        """Get most recent value from each channel.
        
        Returns:
            Dictionary of channel names to current values
        """
        return {name: ch.last_value for name, ch in self._channels.items()}
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all channels.
        
        Returns:
            Dictionary of channel statistics
        """
        return {name: ch.get_state() for name, ch in self._channels.items()}
    
    def clear(self) -> None:
        """Clear all recorded data."""
        for channel in self._channels.values():
            channel.clear()
        
        self._last_sample_time = 0.0
        self._current_lap = 0
        self._lap_start_times = [0.0]
    
    def get_state(self) -> dict:
        """Get recorder state.
        
        Returns:
            Dictionary containing recorder state
        """
        return {
            "sample_rate_hz": self.config.sample_rate_hz,
            "current_lap": self._current_lap,
            "total_samples": sum(ch.count for ch in self._channels.values()),
            "channels": self.get_statistics(),
        }
