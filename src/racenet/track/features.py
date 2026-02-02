"""
Track features - Kerbs, track limits, and other features.

Defines:
- Kerb geometry and properties
- Track limits detection
- Run-off areas
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import numpy as np


class KerbType(Enum):
    """Types of kerbs."""
    FLAT = "flat"          # Low, flat kerb
    SAUSAGE = "sausage"    # Raised sausage kerb
    RUMBLE = "rumble"      # Rumble strips
    ASTRO = "astro"        # Astroturf-style


@dataclass
class Kerb:
    """Kerb feature on track edge.
    
    Kerbs are raised/textured areas at track edges that provide
    feedback and grip changes when driven over.
    """
    # Position along segment (start, end in meters from segment start)
    segment_id: int = 0
    start_distance_m: float = 0.0
    end_distance_m: float = 100.0
    
    # Which side
    side: str = "left"  # "left" or "right"
    
    # Kerb properties
    kerb_type: KerbType = KerbType.FLAT
    width_m: float = 1.0
    height_m: float = 0.05
    
    # Colors (for visualization)
    color_primary: str = "#FF0000"  # Red
    color_secondary: str = "#FFFFFF"  # White
    stripe_width_m: float = 0.5
    
    @property
    def length(self) -> float:
        """Get kerb length in meters."""
        return self.end_distance_m - self.start_distance_m
    
    def get_grip_modifier(self) -> float:
        """Get grip modifier when on this kerb.
        
        Returns:
            Grip multiplier (0-1)
        """
        modifiers = {
            KerbType.FLAT: 0.95,
            KerbType.SAUSAGE: 0.80,
            KerbType.RUMBLE: 0.85,
            KerbType.ASTRO: 0.75,
        }
        return modifiers.get(self.kerb_type, 0.9)
    
    def get_instability(self) -> float:
        """Get car instability when on this kerb.
        
        Returns:
            Instability factor (0-1)
        """
        instability = {
            KerbType.FLAT: 0.1,
            KerbType.SAUSAGE: 0.4,
            KerbType.RUMBLE: 0.3,
            KerbType.ASTRO: 0.2,
        }
        return instability.get(self.kerb_type, 0.2)
    
    def get_state(self) -> dict:
        """Get kerb state for serialization."""
        return {
            "segment_id": self.segment_id,
            "side": self.side,
            "start_m": self.start_distance_m,
            "end_m": self.end_distance_m,
            "type": self.kerb_type.value,
            "width_m": self.width_m,
            "height_m": self.height_m,
        }


@dataclass
class RunoffArea:
    """Run-off area beyond track limits.
    
    Defines what happens when a car goes off track.
    """
    segment_id: int = 0
    side: str = "left"
    
    # Run-off composition (from track edge outward)
    layers: List[Tuple[float, str]] = None  # (width, surface_type)
    
    def __post_init__(self):
        if self.layers is None:
            # Default: asphalt -> gravel -> barrier
            self.layers = [
                (3.0, "asphalt"),   # 3m of asphalt runoff
                (5.0, "gravel"),    # 5m of gravel trap
                (0.5, "barrier"),   # Barrier
            ]
    
    def get_surface_at_offset(self, offset: float) -> str:
        """Get surface type at given offset from track edge.
        
        Args:
            offset: Distance from track edge in meters
            
        Returns:
            Surface type string
        """
        cumulative = 0.0
        for width, surface in self.layers:
            cumulative += width
            if offset <= cumulative:
                return surface
        return "barrier"
    
    def get_grip_at_offset(self, offset: float) -> float:
        """Get grip level at given offset.
        
        Args:
            offset: Distance from track edge
            
        Returns:
            Grip multiplier (0-1)
        """
        surface = self.get_surface_at_offset(offset)
        grip_levels = {
            "asphalt": 0.9,
            "gravel": 0.3,
            "grass": 0.4,
            "sand": 0.2,
            "barrier": 0.0,
        }
        return grip_levels.get(surface, 0.5)


class TrackLimits:
    """Track limits detection and management.
    
    Monitors when cars exceed track limits and applies
    appropriate penalties or restrictions.
    """
    
    def __init__(
        self,
        track_width: float = 12.0,
        kerb_tolerance: float = 0.5,  # Can use this much of kerb
        off_track_threshold: float = 1.0,  # Beyond this = off track
    ):
        """Initialize track limits.
        
        Args:
            track_width: Default track width
            kerb_tolerance: How much kerb use is allowed
            off_track_threshold: Distance beyond edge for off-track
        """
        self.track_width = track_width
        self.kerb_tolerance = kerb_tolerance
        self.off_track_threshold = off_track_threshold
        
        # Violation tracking
        self._violation_count: int = 0
        self._time_off_track: float = 0.0
        self._current_off_track: bool = False
    
    @property
    def violation_count(self) -> int:
        """Number of track limit violations."""
        return self._violation_count
    
    @property
    def is_off_track(self) -> bool:
        """Check if currently off track."""
        return self._current_off_track
    
    def check_position(
        self,
        lateral_offset: float,
        track_width: float | None = None,
        kerb_width: float = 0.0,
    ) -> Tuple[str, float]:
        """Check position relative to track limits.
        
        Args:
            lateral_offset: Distance from track center (positive = right)
            track_width: Track width at this point (uses default if None)
            kerb_width: Width of kerb at this edge
            
        Returns:
            Tuple of (status, grip_modifier)
            Status is one of: "on_track", "on_kerb", "off_track"
        """
        width = track_width or self.track_width
        half_width = width / 2
        
        abs_offset = abs(lateral_offset)
        
        if abs_offset <= half_width:
            # On track proper
            return ("on_track", 1.0)
        
        beyond_edge = abs_offset - half_width
        
        if beyond_edge <= kerb_width + self.kerb_tolerance:
            # On kerb
            return ("on_kerb", 0.9)
        
        if beyond_edge <= self.off_track_threshold:
            # Just beyond limits
            return ("exceeding_limits", 0.7)
        
        # Off track
        return ("off_track", 0.4)
    
    def update(
        self,
        lateral_offset: float,
        dt: float,
        track_width: float | None = None,
    ) -> dict:
        """Update track limits state.
        
        Args:
            lateral_offset: Distance from track center
            dt: Time step
            track_width: Track width at this point
            
        Returns:
            Dictionary with limit status
        """
        status, grip = self.check_position(lateral_offset, track_width)
        
        was_off = self._current_off_track
        self._current_off_track = status == "off_track"
        
        if self._current_off_track:
            self._time_off_track += dt
            if not was_off:
                # Just went off track
                self._violation_count += 1
        
        return {
            "status": status,
            "grip_modifier": grip,
            "off_track": self._current_off_track,
            "violation_count": self._violation_count,
            "time_off_track": self._time_off_track,
        }
    
    def reset(self) -> None:
        """Reset track limits state."""
        self._violation_count = 0
        self._time_off_track = 0.0
        self._current_off_track = False
    
    def get_state(self) -> dict:
        """Get current state for telemetry."""
        return {
            "violation_count": self._violation_count,
            "time_off_track": self._time_off_track,
            "is_off_track": self._current_off_track,
        }


@dataclass
class Sector:
    """Track sector for timing purposes.
    
    Tracks are typically divided into 3 sectors for split times.
    """
    sector_number: int = 1
    start_distance_m: float = 0.0
    end_distance_m: float = 0.0
    
    # Segment range
    start_segment_id: int = 0
    end_segment_id: int = 0
    
    # Reference times (for delta calculation)
    best_time_s: float | None = None
    
    @property
    def length(self) -> float:
        """Get sector length in meters."""
        return self.end_distance_m - self.start_distance_m
    
    def get_state(self) -> dict:
        """Get sector state."""
        return {
            "sector": self.sector_number,
            "start_m": self.start_distance_m,
            "end_m": self.end_distance_m,
            "length_m": self.length,
            "best_time": self.best_time_s,
        }
