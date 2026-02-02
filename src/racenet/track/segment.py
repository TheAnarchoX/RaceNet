"""
Track segment - Individual track section with geometry and features.

Defines:
- Segment geometry (curvature, width, length)
- Elevation and banking
- Surface properties
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
import numpy as np


class SegmentType(Enum):
    """Types of track segments."""
    STRAIGHT = "straight"
    LEFT_TURN = "left_turn"
    RIGHT_TURN = "right_turn"
    CHICANE_LEFT = "chicane_left"   # Left-right chicane
    CHICANE_RIGHT = "chicane_right"  # Right-left chicane


class SurfaceType(Enum):
    """Track surface types."""
    ASPHALT = "asphalt"
    KERB = "kerb"
    RUMBLE_STRIP = "rumble_strip"
    GRAVEL = "gravel"
    GRASS = "grass"


@dataclass
class SurfaceProperties:
    """Surface grip and behavior properties."""
    surface_type: SurfaceType = SurfaceType.ASPHALT
    grip_multiplier: float = 1.0      # 1.0 = full grip
    roughness: float = 0.0            # Vibration/instability factor
    speed_reduction: float = 0.0       # Speed penalty (friction)
    
    @classmethod
    def asphalt(cls) -> "SurfaceProperties":
        return cls(SurfaceType.ASPHALT, 1.0, 0.0, 0.0)
    
    @classmethod
    def kerb(cls) -> "SurfaceProperties":
        return cls(SurfaceType.KERB, 0.9, 0.3, 0.02)
    
    @classmethod
    def rumble_strip(cls) -> "SurfaceProperties":
        return cls(SurfaceType.RUMBLE_STRIP, 0.85, 0.5, 0.05)
    
    @classmethod
    def gravel(cls) -> "SurfaceProperties":
        return cls(SurfaceType.GRAVEL, 0.4, 0.8, 0.3)
    
    @classmethod
    def grass(cls) -> "SurfaceProperties":
        return cls(SurfaceType.GRASS, 0.5, 0.6, 0.2)


@dataclass
class TrackSegment:
    """A single segment of the race track.
    
    Segments are the building blocks of a track, each defining:
    - Geometry (length, curvature, width)
    - Vertical profile (elevation change, banking)
    - Surface properties
    """
    # Identification
    segment_id: int = 0
    segment_type: SegmentType = SegmentType.STRAIGHT
    
    # Geometry
    length_m: float = 100.0           # Segment length in meters
    width_m: float = 12.0             # Track width
    curvature: float = 0.0            # 1/radius (positive = right turn)
    
    # Vertical profile
    elevation_start_m: float = 0.0    # Elevation at segment start
    elevation_end_m: float = 0.0      # Elevation at segment end
    banking_deg: float = 0.0          # Track banking (positive = banked right)
    
    # Start position and direction (calculated during track generation)
    start_x: float = 0.0
    start_y: float = 0.0
    start_heading: float = 0.0        # Radians, 0 = +X direction
    
    # End position and direction (calculated)
    end_x: float = 0.0
    end_y: float = 0.0
    end_heading: float = 0.0
    
    # Surface
    surface: SurfaceProperties = field(default_factory=SurfaceProperties.asphalt)
    
    # Kerb information
    left_kerb_width_m: float = 0.0    # Width of left kerb
    right_kerb_width_m: float = 0.0   # Width of right kerb
    
    @property
    def radius(self) -> float:
        """Get turn radius (infinity for straights).
        
        Returns:
            Turn radius in meters
        """
        if abs(self.curvature) < 1e-6:
            return float('inf')
        return 1.0 / abs(self.curvature)
    
    @property
    def turn_angle(self) -> float:
        """Get total turn angle in radians.
        
        Returns:
            Turn angle (positive = right turn)
        """
        return self.curvature * self.length_m
    
    @property
    def elevation_change(self) -> float:
        """Get elevation change across segment.
        
        Returns:
            Elevation change in meters (positive = uphill)
        """
        return self.elevation_end_m - self.elevation_start_m
    
    @property
    def gradient(self) -> float:
        """Get average gradient.
        
        Returns:
            Gradient as a fraction (e.g., 0.05 = 5%)
        """
        if self.length_m < 1e-6:
            return 0.0
        return self.elevation_change / self.length_m
    
    def get_position_at(self, distance: float) -> Tuple[float, float, float]:
        """Get world position at given distance along segment.
        
        Args:
            distance: Distance from segment start in meters
            
        Returns:
            Tuple of (x, y, z) world coordinates
        """
        # Clamp distance
        distance = np.clip(distance, 0, self.length_m)
        t = distance / self.length_m if self.length_m > 0 else 0
        
        if abs(self.curvature) < 1e-6:
            # Straight segment
            x = self.start_x + distance * np.cos(self.start_heading)
            y = self.start_y + distance * np.sin(self.start_heading)
        else:
            # Curved segment - arc calculation
            radius = 1.0 / self.curvature
            angle = distance * self.curvature
            
            # Center of the arc
            perp_angle = self.start_heading + np.pi / 2  # Perpendicular
            if self.curvature > 0:  # Right turn
                center_x = self.start_x + radius * np.cos(perp_angle)
                center_y = self.start_y + radius * np.sin(perp_angle)
            else:  # Left turn
                center_x = self.start_x - radius * np.cos(perp_angle)
                center_y = self.start_y - radius * np.sin(perp_angle)
            
            # Position on arc
            arc_angle = self.start_heading + angle - np.pi / 2
            x = center_x + abs(radius) * np.cos(arc_angle)
            y = center_y + abs(radius) * np.sin(arc_angle)
        
        # Interpolate elevation
        z = self.elevation_start_m + t * self.elevation_change
        
        return (x, y, z)
    
    def get_heading_at(self, distance: float) -> float:
        """Get heading at given distance along segment.
        
        Args:
            distance: Distance from segment start in meters
            
        Returns:
            Heading in radians
        """
        angle_change = distance * self.curvature
        return self.start_heading + angle_change
    
    def get_width_at(self, distance: float) -> Tuple[float, float, float]:
        """Get track boundaries at given distance.
        
        Args:
            distance: Distance from segment start
            
        Returns:
            Tuple of (left_edge, center, right_edge) in track coordinates
        """
        half_width = self.width_m / 2
        return (-half_width, 0.0, half_width)
    
    def calculate_end_point(self) -> None:
        """Calculate end position and heading from geometry."""
        end_pos = self.get_position_at(self.length_m)
        self.end_x = end_pos[0]
        self.end_y = end_pos[1]
        self.end_heading = self.get_heading_at(self.length_m)
        
        # Normalize heading to [-pi, pi]
        while self.end_heading > np.pi:
            self.end_heading -= 2 * np.pi
        while self.end_heading < -np.pi:
            self.end_heading += 2 * np.pi
    
    def is_point_on_track(
        self, 
        x: float, 
        y: float, 
        margin: float = 0.0
    ) -> Tuple[bool, float, float]:
        """Check if a point is on this segment.
        
        Args:
            x: World X coordinate
            y: World Y coordinate
            margin: Additional margin beyond track edges
            
        Returns:
            Tuple of (is_on_track, distance_along, lateral_offset)
        """
        # This is a simplified check - more accurate version would be needed
        # for production use
        
        if abs(self.curvature) < 1e-6:
            # Straight - project point onto segment line
            dx = x - self.start_x
            dy = y - self.start_y
            
            # Distance along segment
            along = dx * np.cos(self.start_heading) + dy * np.sin(self.start_heading)
            
            # Lateral offset
            lateral = -dx * np.sin(self.start_heading) + dy * np.cos(self.start_heading)
        else:
            # Curved - more complex projection needed
            # Simplified: find closest point on arc
            radius = 1.0 / self.curvature
            perp_angle = self.start_heading + np.pi / 2
            
            if self.curvature > 0:
                cx = self.start_x + radius * np.cos(perp_angle)
                cy = self.start_y + radius * np.sin(perp_angle)
            else:
                cx = self.start_x - abs(radius) * np.cos(perp_angle)
                cy = self.start_y - abs(radius) * np.sin(perp_angle)
            
            # Angle from center to point
            to_point = np.arctan2(y - cy, x - cx)
            
            # Distance from center
            dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Lateral offset
            lateral = dist_from_center - abs(radius)
            
            # Distance along (approximate)
            arc_start = self.start_heading - np.pi / 2
            arc_angle = to_point - arc_start
            while arc_angle < -np.pi:
                arc_angle += 2 * np.pi
            while arc_angle > np.pi:
                arc_angle -= 2 * np.pi
            
            along = abs(radius) * arc_angle
        
        # Check if within segment
        on_segment = 0 <= along <= self.length_m
        half_width = self.width_m / 2 + margin
        on_track = abs(lateral) <= half_width
        
        return (on_segment and on_track, along, lateral)
    
    def get_state(self) -> dict:
        """Get segment state for serialization.
        
        Returns:
            Dictionary containing segment data
        """
        return {
            "segment_id": self.segment_id,
            "type": self.segment_type.value,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "curvature": self.curvature,
            "radius": self.radius,
            "turn_angle_deg": np.degrees(self.turn_angle),
            "elevation_start": self.elevation_start_m,
            "elevation_end": self.elevation_end_m,
            "banking_deg": self.banking_deg,
            "start_pos": (self.start_x, self.start_y),
            "end_pos": (self.end_x, self.end_y),
            "start_heading_deg": np.degrees(self.start_heading),
            "end_heading_deg": np.degrees(self.end_heading),
        }
