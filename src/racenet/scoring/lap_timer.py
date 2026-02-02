"""
Lap timer - Lap and sector time tracking.

Provides:
- Lap time recording
- Sector split times
- Best lap/sector tracking
- Lap validity checking
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class LapTime:
    """Record of a single lap time."""
    lap_number: int
    lap_time_s: float
    sector_times: List[float]
    is_valid: bool = True
    track_limits_violations: int = 0
    timestamp: float = 0.0


@dataclass
class LapTimerConfig:
    """Lap timer configuration."""
    num_sectors: int = 3
    invalidate_on_track_limits: bool = True
    max_violations_per_lap: int = 3


class LapTimer:
    """Lap and sector timing management.
    
    Tracks lap times, sector splits, and best times
    for a single car.
    """
    
    def __init__(self, config: LapTimerConfig | None = None):
        """Initialize lap timer.
        
        Args:
            config: Timer configuration
        """
        self.config = config or LapTimerConfig()
        
        # Lap tracking
        self._current_lap: int = 0
        self._current_sector: int = 0
        self._lap_start_time: float = 0.0
        self._sector_start_time: float = 0.0
        
        # Times
        self._current_sector_times: List[float] = []
        self._lap_history: List[LapTime] = []
        
        # Best times
        self._best_lap_time: float = float('inf')
        self._best_sector_times: List[float] = [float('inf')] * self.config.num_sectors
        
        # Track limits
        self._current_lap_violations: int = 0
        self._total_violations: int = 0
    
    @property
    def current_lap(self) -> int:
        """Current lap number."""
        return self._current_lap
    
    @property
    def current_sector(self) -> int:
        """Current sector (0-indexed)."""
        return self._current_sector
    
    @property
    def best_lap_time(self) -> float:
        """Best lap time in seconds."""
        return self._best_lap_time if self._best_lap_time < float('inf') else 0.0
    
    @property
    def best_sector_times(self) -> List[float]:
        """Best sector times."""
        return [t if t < float('inf') else 0.0 for t in self._best_sector_times]
    
    @property
    def theoretical_best(self) -> float:
        """Theoretical best lap (sum of best sectors)."""
        best_sectors = self.best_sector_times
        if all(t > 0 for t in best_sectors):
            return sum(best_sectors)
        return 0.0
    
    @property
    def lap_count(self) -> int:
        """Number of completed laps."""
        return len(self._lap_history)
    
    @property
    def valid_lap_count(self) -> int:
        """Number of valid laps."""
        return sum(1 for lap in self._lap_history if lap.is_valid)
    
    def start_lap(self, time: float) -> None:
        """Start a new lap.
        
        Args:
            time: Current simulation time
        """
        self._lap_start_time = time
        self._sector_start_time = time
        self._current_sector = 0
        self._current_sector_times = []
        self._current_lap_violations = 0
    
    def cross_sector(self, time: float) -> Optional[float]:
        """Cross a sector line.
        
        Args:
            time: Current simulation time
            
        Returns:
            Sector time if completed, None if at final sector
        """
        sector_time = time - self._sector_start_time
        self._current_sector_times.append(sector_time)
        
        # Check for best sector
        if sector_time < self._best_sector_times[self._current_sector]:
            self._best_sector_times[self._current_sector] = sector_time
        
        self._current_sector += 1
        self._sector_start_time = time
        
        return sector_time
    
    def complete_lap(self, time: float) -> LapTime:
        """Complete current lap and start new one.
        
        Args:
            time: Current simulation time
            
        Returns:
            Completed lap time record
        """
        # Complete final sector
        final_sector_time = time - self._sector_start_time
        self._current_sector_times.append(final_sector_time)
        
        # Check best sector
        if final_sector_time < self._best_sector_times[-1]:
            self._best_sector_times[-1] = final_sector_time
        
        # Calculate total lap time
        lap_time = time - self._lap_start_time
        
        # Check validity
        is_valid = True
        if self.config.invalidate_on_track_limits:
            if self._current_lap_violations > self.config.max_violations_per_lap:
                is_valid = False
        
        # Record lap
        lap_record = LapTime(
            lap_number=self._current_lap,
            lap_time_s=lap_time,
            sector_times=self._current_sector_times.copy(),
            is_valid=is_valid,
            track_limits_violations=self._current_lap_violations,
            timestamp=time,
        )
        self._lap_history.append(lap_record)
        
        # Check best lap
        if is_valid and lap_time < self._best_lap_time:
            self._best_lap_time = lap_time
        
        # Start new lap
        self._current_lap += 1
        self.start_lap(time)
        
        return lap_record
    
    def record_violation(self) -> None:
        """Record a track limits violation."""
        self._current_lap_violations += 1
        self._total_violations += 1
    
    def get_current_lap_time(self, current_time: float) -> float:
        """Get current lap time.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Elapsed lap time
        """
        return current_time - self._lap_start_time
    
    def get_current_sector_time(self, current_time: float) -> float:
        """Get current sector time.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Elapsed sector time
        """
        return current_time - self._sector_start_time
    
    def get_delta_to_best(self, current_time: float) -> float:
        """Get delta to best lap.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Delta in seconds (positive = slower)
        """
        if self._best_lap_time == float('inf'):
            return 0.0
        
        current_lap_time = self.get_current_lap_time(current_time)
        
        # Estimate where we should be on best lap
        # This is simplified - proper implementation would use distance
        progress = current_lap_time / self._best_lap_time
        best_time_at_progress = self._best_lap_time * progress
        
        return current_lap_time - best_time_at_progress
    
    def get_lap(self, lap_number: int) -> Optional[LapTime]:
        """Get specific lap record.
        
        Args:
            lap_number: Lap number to get
            
        Returns:
            Lap time record if found
        """
        for lap in self._lap_history:
            if lap.lap_number == lap_number:
                return lap
        return None
    
    def get_last_lap(self) -> Optional[LapTime]:
        """Get most recent completed lap.
        
        Returns:
            Last lap record
        """
        return self._lap_history[-1] if self._lap_history else None
    
    def get_best_lap(self) -> Optional[LapTime]:
        """Get best lap record.
        
        Returns:
            Best valid lap
        """
        valid_laps = [lap for lap in self._lap_history if lap.is_valid]
        if not valid_laps:
            return None
        return min(valid_laps, key=lambda x: x.lap_time_s)
    
    def reset(self) -> None:
        """Reset all timing data."""
        self._current_lap = 0
        self._current_sector = 0
        self._lap_start_time = 0.0
        self._sector_start_time = 0.0
        self._current_sector_times = []
        self._lap_history = []
        self._best_lap_time = float('inf')
        self._best_sector_times = [float('inf')] * self.config.num_sectors
        self._current_lap_violations = 0
        self._total_violations = 0
    
    def get_state(self) -> dict:
        """Get timer state.
        
        Returns:
            Dictionary with timer state
        """
        return {
            "current_lap": self._current_lap,
            "current_sector": self._current_sector,
            "completed_laps": len(self._lap_history),
            "valid_laps": self.valid_lap_count,
            "best_lap_time": self.best_lap_time,
            "best_sector_times": self.best_sector_times,
            "theoretical_best": self.theoretical_best,
            "total_violations": self._total_violations,
            "last_lap": self.get_last_lap().__dict__ if self.get_last_lap() else None,
        }
