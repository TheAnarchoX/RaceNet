"""
Track - Complete race track representation.

Contains:
- Collection of track segments
- Track metadata
- Start/finish line
- Pit lane (optional)
- Sector definitions
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from racenet.track.segment import TrackSegment, SegmentType
from racenet.track.features import Kerb, TrackLimits, Sector


@dataclass
class TrackConfig:
    """Track configuration and metadata."""
    name: str = "Generated Track"
    location: str = "Unknown"
    
    # Track dimensions
    target_length_m: float = 4000.0    # Target track length
    min_width_m: float = 10.0          # Minimum track width
    max_width_m: float = 15.0          # Maximum track width
    default_width_m: float = 12.0      # Default track width
    
    # Direction
    clockwise: bool = True             # Track direction
    
    # Number of sectors
    num_sectors: int = 3


class Track:
    """Complete race track.
    
    A track consists of connected segments forming a closed loop.
    Includes kerbs, sectors, and track limits management.
    
    Features:
    - Procedurally generated or manually defined
    - Closed loop verification
    - Distance-based position lookup
    - Sector timing support
    
    Usage:
        track = Track()
        track.add_segment(TrackSegment(...))
        track.finalize()
        
        pos = track.get_position_at_distance(500.0)
    """
    
    def __init__(self, config: TrackConfig | None = None):
        """Initialize track with optional configuration.
        
        Args:
            config: Track configuration. Uses defaults if None.
        """
        self.config = config or TrackConfig()
        
        # Track components
        self._segments: List[TrackSegment] = []
        self._kerbs: List[Kerb] = []
        self._sectors: List[Sector] = []
        
        # Track limits manager
        self.track_limits = TrackLimits(self.config.default_width_m)
        
        # Cached values
        self._total_length: float = 0.0
        self._segment_distances: List[float] = []  # Cumulative distance to segment start
        self._is_finalized: bool = False
        
        # Start/finish
        self._start_x: float = 0.0
        self._start_y: float = 0.0
        self._start_heading: float = 0.0
    
    @property
    def segments(self) -> List[TrackSegment]:
        """List of track segments."""
        return self._segments
    
    @property
    def length(self) -> float:
        """Total track length in meters."""
        return self._total_length
    
    @property
    def num_segments(self) -> int:
        """Number of segments."""
        return len(self._segments)
    
    @property
    def is_closed(self) -> bool:
        """Check if track forms a closed loop."""
        if len(self._segments) < 2:
            return False
        
        first = self._segments[0]
        last = self._segments[-1]
        
        # Check if end meets start
        dist = np.sqrt(
            (last.end_x - first.start_x)**2 + 
            (last.end_y - first.start_y)**2
        )
        
        # Allow small gap
        return dist < 5.0
    
    @property
    def start_position(self) -> Tuple[float, float]:
        """Get start/finish line position."""
        return (self._start_x, self._start_y)
    
    @property
    def start_heading(self) -> float:
        """Get heading at start/finish line."""
        return self._start_heading
    
    def add_segment(self, segment: TrackSegment) -> int:
        """Add a segment to the track.
        
        Args:
            segment: Track segment to add
            
        Returns:
            Segment index
        """
        if self._is_finalized:
            raise RuntimeError("Cannot add segments to finalized track")
        
        # Set segment ID
        segment.segment_id = len(self._segments)
        
        # Connect to previous segment
        if self._segments:
            prev = self._segments[-1]
            segment.start_x = prev.end_x
            segment.start_y = prev.end_y
            segment.start_heading = prev.end_heading
            segment.elevation_start_m = prev.elevation_end_m
        else:
            # First segment
            segment.start_x = self._start_x
            segment.start_y = self._start_y
            segment.start_heading = self._start_heading
        
        # Calculate end point
        segment.calculate_end_point()
        
        self._segments.append(segment)
        return segment.segment_id
    
    def add_kerb(self, kerb: Kerb) -> None:
        """Add a kerb to the track.
        
        Args:
            kerb: Kerb to add
        """
        self._kerbs.append(kerb)
    
    def finalize(self) -> bool:
        """Finalize track construction.
        
        Calculates cumulative distances, sectors, and validates track.
        
        Returns:
            True if track is valid
        """
        if not self._segments:
            return False
        
        # Calculate cumulative distances
        self._segment_distances = [0.0]
        cumulative = 0.0
        for segment in self._segments:
            cumulative += segment.length_m
            self._segment_distances.append(cumulative)
        
        self._total_length = cumulative
        
        # Create sectors
        self._create_sectors()
        
        # Store start position
        first = self._segments[0]
        self._start_x = first.start_x
        self._start_y = first.start_y
        self._start_heading = first.start_heading
        
        self._is_finalized = True
        return True
    
    def _create_sectors(self) -> None:
        """Create sector definitions."""
        self._sectors = []
        
        if self._total_length == 0:
            return
        
        sector_length = self._total_length / self.config.num_sectors
        
        for i in range(self.config.num_sectors):
            start_dist = i * sector_length
            end_dist = (i + 1) * sector_length
            
            # Find segment IDs
            start_seg = self._get_segment_at_distance(start_dist)[0]
            end_seg = self._get_segment_at_distance(end_dist)[0]
            
            sector = Sector(
                sector_number=i + 1,
                start_distance_m=start_dist,
                end_distance_m=end_dist,
                start_segment_id=start_seg,
                end_segment_id=end_seg,
            )
            self._sectors.append(sector)
    
    def _get_segment_at_distance(self, distance: float) -> Tuple[int, float]:
        """Get segment index and local distance for a track distance.
        
        Args:
            distance: Distance from start in meters
            
        Returns:
            Tuple of (segment_index, distance_in_segment)
        """
        # Wrap distance for closed tracks
        distance = distance % self._total_length if self._total_length > 0 else 0
        
        for i, seg_start in enumerate(self._segment_distances[:-1]):
            seg_end = self._segment_distances[i + 1]
            if seg_start <= distance < seg_end:
                return (i, distance - seg_start)
        
        # Return last segment if at end
        return (len(self._segments) - 1, self._segments[-1].length_m)
    
    def get_position_at_distance(self, distance: float) -> Tuple[float, float, float]:
        """Get world position at given distance from start.
        
        Args:
            distance: Distance from start line in meters
            
        Returns:
            Tuple of (x, y, z) world coordinates
        """
        if not self._is_finalized or not self._segments:
            return (0.0, 0.0, 0.0)
        
        seg_idx, local_dist = self._get_segment_at_distance(distance)
        segment = self._segments[seg_idx]
        
        return segment.get_position_at(local_dist)
    
    def get_heading_at_distance(self, distance: float) -> float:
        """Get track heading at given distance.
        
        Args:
            distance: Distance from start line
            
        Returns:
            Heading in radians
        """
        if not self._is_finalized or not self._segments:
            return 0.0
        
        seg_idx, local_dist = self._get_segment_at_distance(distance)
        segment = self._segments[seg_idx]
        
        return segment.get_heading_at(local_dist)
    
    def get_width_at_distance(self, distance: float) -> float:
        """Get track width at given distance.
        
        Args:
            distance: Distance from start line
            
        Returns:
            Track width in meters
        """
        if not self._is_finalized or not self._segments:
            return self.config.default_width_m
        
        seg_idx, _ = self._get_segment_at_distance(distance)
        return self._segments[seg_idx].width_m
    
    def get_curvature_at_distance(self, distance: float) -> float:
        """Get track curvature at given distance.
        
        Args:
            distance: Distance from start line
            
        Returns:
            Curvature (1/radius, positive = right turn)
        """
        if not self._is_finalized or not self._segments:
            return 0.0
        
        seg_idx, _ = self._get_segment_at_distance(distance)
        return self._segments[seg_idx].curvature
    
    def get_banking_at_distance(self, distance: float) -> float:
        """Get track banking at given distance.
        
        Args:
            distance: Distance from start line
            
        Returns:
            Banking angle in degrees
        """
        if not self._is_finalized or not self._segments:
            return 0.0
        
        seg_idx, _ = self._get_segment_at_distance(distance)
        return self._segments[seg_idx].banking_deg
    
    def get_elevation_at_distance(self, distance: float) -> float:
        """Get elevation at given distance.
        
        Args:
            distance: Distance from start line
            
        Returns:
            Elevation in meters
        """
        pos = self.get_position_at_distance(distance)
        return pos[2]
    
    def get_sector_at_distance(self, distance: float) -> int:
        """Get sector number at given distance.
        
        Args:
            distance: Distance from start line
            
        Returns:
            Sector number (1-indexed)
        """
        distance = distance % self._total_length if self._total_length > 0 else 0
        
        for sector in self._sectors:
            if sector.start_distance_m <= distance < sector.end_distance_m:
                return sector.sector_number
        
        return self.config.num_sectors
    
    def world_to_track_coords(
        self, 
        x: float, 
        y: float
    ) -> Tuple[float, float]:
        """Convert world coordinates to track coordinates.
        
        Args:
            x: World X coordinate
            y: World Y coordinate
            
        Returns:
            Tuple of (distance_along_track, lateral_offset)
        """
        # Find closest segment
        best_distance = float('inf')
        best_along = 0.0
        best_lateral = 0.0
        
        cumulative_dist = 0.0
        for segment in self._segments:
            on_seg, along, lateral = segment.is_point_on_track(x, y, margin=50.0)
            
            # Calculate distance to segment
            pos = segment.get_position_at(along)
            dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            
            if dist < best_distance:
                best_distance = dist
                best_along = cumulative_dist + along
                best_lateral = lateral
            
            cumulative_dist += segment.length_m
        
        return (best_along, best_lateral)
    
    def get_track_info_at_position(
        self, 
        x: float, 
        y: float
    ) -> dict:
        """Get comprehensive track info at world position.
        
        Args:
            x: World X coordinate
            y: World Y coordinate
            
        Returns:
            Dictionary with track information
        """
        along, lateral = self.world_to_track_coords(x, y)
        
        return {
            "distance_m": along,
            "lateral_offset_m": lateral,
            "sector": self.get_sector_at_distance(along),
            "curvature": self.get_curvature_at_distance(along),
            "width_m": self.get_width_at_distance(along),
            "banking_deg": self.get_banking_at_distance(along),
            "elevation_m": self.get_elevation_at_distance(along),
            "heading_rad": self.get_heading_at_distance(along),
        }
    
    def get_state(self) -> dict:
        """Get complete track state for serialization.
        
        Returns:
            Dictionary containing all track data
        """
        return {
            "name": self.config.name,
            "location": self.config.location,
            "length_m": self._total_length,
            "num_segments": len(self._segments),
            "is_closed": self.is_closed,
            "clockwise": self.config.clockwise,
            "segments": [s.get_state() for s in self._segments],
            "sectors": [s.get_state() for s in self._sectors],
            "kerbs": [k.get_state() for k in self._kerbs],
            "start_position": self.start_position,
            "start_heading_deg": np.degrees(self._start_heading),
        }
    
    def reset(self) -> None:
        """Reset track state (not geometry)."""
        self.track_limits.reset()
        for sector in self._sectors:
            sector.best_time_s = None
