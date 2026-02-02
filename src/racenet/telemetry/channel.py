"""
Telemetry channel - Individual data channel for recording.

Provides:
- Buffered data storage
- Statistics calculation
- Time-series management
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional
import numpy as np


@dataclass
class ChannelConfig:
    """Configuration for a telemetry channel."""
    name: str = "unnamed"
    unit: str = ""
    min_value: float = float('-inf')
    max_value: float = float('inf')
    precision: int = 3
    buffer_size: int = 10000


class TelemetryChannel:
    """Single telemetry data channel.
    
    Stores time-series data for a single measurement,
    with statistics and data management.
    """
    
    def __init__(self, config: ChannelConfig | None = None, name: str = "channel"):
        """Initialize channel.
        
        Args:
            config: Channel configuration
            name: Channel name (used if config not provided)
        """
        if config is None:
            config = ChannelConfig(name=name)
        self.config = config
        
        self._times: List[float] = []
        self._values: List[float] = []
        
        # Running statistics
        self._min: float = float('inf')
        self._max: float = float('-inf')
        self._sum: float = 0.0
        self._count: int = 0
    
    @property
    def name(self) -> str:
        """Channel name."""
        return self.config.name
    
    @property
    def count(self) -> int:
        """Number of recorded samples."""
        return self._count
    
    @property
    def min_value(self) -> float:
        """Minimum recorded value."""
        return self._min if self._count > 0 else 0.0
    
    @property
    def max_value(self) -> float:
        """Maximum recorded value."""
        return self._max if self._count > 0 else 0.0
    
    @property
    def mean(self) -> float:
        """Mean of recorded values."""
        return self._sum / self._count if self._count > 0 else 0.0
    
    @property
    def last_value(self) -> float:
        """Most recent value."""
        return self._values[-1] if self._values else 0.0
    
    def record(self, time: float, value: float) -> None:
        """Record a new value.
        
        Args:
            time: Timestamp
            value: Value to record
        """
        # Clamp to valid range
        value = np.clip(value, self.config.min_value, self.config.max_value)
        
        self._times.append(time)
        self._values.append(value)
        
        # Update statistics
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        self._sum += value
        self._count += 1
        
        # Limit buffer size
        if len(self._values) > self.config.buffer_size:
            removed = self._values.pop(0)
            self._times.pop(0)
            self._sum -= removed
    
    def get_values(self) -> np.ndarray:
        """Get all recorded values.
        
        Returns:
            Numpy array of values
        """
        return np.array(self._values)
    
    def get_times(self) -> np.ndarray:
        """Get all timestamps.
        
        Returns:
            Numpy array of times
        """
        return np.array(self._times)
    
    def get_last_n(self, n: int) -> np.ndarray:
        """Get last N values.
        
        Args:
            n: Number of values
            
        Returns:
            Numpy array of values
        """
        return np.array(self._values[-n:])
    
    def get_range(self, start_time: float, end_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Get values in time range.
        
        Args:
            start_time: Start of range
            end_time: End of range
            
        Returns:
            Tuple of (times, values) arrays
        """
        times = np.array(self._times)
        values = np.array(self._values)
        
        mask = (times >= start_time) & (times <= end_time)
        return times[mask], values[mask]
    
    def clear(self) -> None:
        """Clear all recorded data."""
        self._times.clear()
        self._values.clear()
        self._min = float('inf')
        self._max = float('-inf')
        self._sum = 0.0
        self._count = 0
    
    def get_state(self) -> dict:
        """Get channel state.
        
        Returns:
            Dictionary with channel data
        """
        return {
            "name": self.config.name,
            "unit": self.config.unit,
            "count": self._count,
            "min": round(self._min, self.config.precision) if self._count > 0 else None,
            "max": round(self._max, self.config.precision) if self._count > 0 else None,
            "mean": round(self.mean, self.config.precision) if self._count > 0 else None,
            "last": round(self.last_value, self.config.precision) if self._count > 0 else None,
        }
