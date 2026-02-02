"""
Track generator - Procedural track generation with realistic constraints.

Generates:
- Realistic racing circuits
- Configurable difficulty/characteristics
- Proper kerb and feature placement
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from racenet.track.track import Track, TrackConfig
from racenet.track.segment import TrackSegment, SegmentType
from racenet.track.features import Kerb, KerbType


@dataclass
class GeneratorConfig:
    """Configuration for procedural track generation."""
    # Track characteristics
    target_length_m: float = 4000.0
    min_length_m: float = 3000.0
    max_length_m: float = 6000.0
    
    # Track width
    min_width_m: float = 10.0
    max_width_m: float = 15.0
    default_width_m: float = 12.0
    
    # Segment constraints
    min_segment_length_m: float = 50.0
    max_segment_length_m: float = 800.0
    
    # Straight constraints
    min_straight_length_m: float = 100.0
    max_straight_length_m: float = 800.0
    
    # Turn constraints
    min_turn_radius_m: float = 30.0   # Tight hairpin
    max_turn_radius_m: float = 500.0  # Fast sweeper
    min_turn_angle_deg: float = 15.0
    max_turn_angle_deg: float = 180.0  # Hairpin
    
    # Elevation
    max_elevation_change_m: float = 50.0
    max_gradient: float = 0.12  # 12% max gradient
    
    # Banking
    min_banking_deg: float = 0.0
    max_banking_deg: float = 15.0
    
    # Track complexity
    complexity: float = 0.5  # 0.0 = simple, 1.0 = complex
    
    # Kerb generation
    generate_kerbs: bool = True
    kerb_probability: float = 0.8  # Probability of kerb on turn
    
    # Random seed (None for random)
    seed: int | None = None


class TrackGenerator:
    """Procedural race track generator.
    
    Generates realistic racing circuits with proper constraints
    on turn radii, segment lengths, and track flow.
    
    Features:
    - Configurable track characteristics
    - Automatic kerb placement
    - Elevation and banking variation
    - Closed loop guarantee
    
    Usage:
        generator = TrackGenerator()
        track = generator.generate()
    """
    
    def __init__(self, config: GeneratorConfig | None = None):
        """Initialize generator with optional configuration.
        
        Args:
            config: Generator configuration. Uses defaults if None.
        """
        self.config = config or GeneratorConfig()
        
        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        self._rng = np.random.default_rng(self.config.seed)
    
    def generate(self, name: str | None = None) -> Track:
        """Generate a new random track.
        
        Args:
            name: Optional track name
            
        Returns:
            Generated Track object
        """
        # Create track
        track_config = TrackConfig(
            name=name or f"Track_{self._rng.integers(1000, 9999)}",
            target_length_m=self.config.target_length_m,
            min_width_m=self.config.min_width_m,
            max_width_m=self.config.max_width_m,
            default_width_m=self.config.default_width_m,
            clockwise=bool(self._rng.choice([True, False])),
        )
        
        track = Track(track_config)
        
        # Generate segments
        segments = self._generate_segments()
        
        for segment in segments:
            track.add_segment(segment)
        
        # Add closing segment to complete loop
        self._close_track(track)
        
        # Add kerbs
        if self.config.generate_kerbs:
            self._add_kerbs(track)
        
        # Finalize
        track.finalize()
        
        return track
    
    def _generate_segments(self) -> List[TrackSegment]:
        """Generate the main track segments.
        
        Returns:
            List of track segments
        """
        segments = []
        total_length = 0.0
        total_turn = 0.0
        
        # Target: complete roughly 2*pi radians (full circle back to start)
        target_turn = 2 * np.pi
        
        while total_length < self.config.target_length_m * 0.8:
            # Decide: straight or turn
            # More turns needed as we haven't completed the loop
            remaining_turn = target_turn - abs(total_turn)
            
            if remaining_turn > np.radians(30) or self._rng.random() < 0.4:
                # Add a turn
                segment = self._generate_turn(remaining_turn > 0)
                total_turn += segment.turn_angle
            else:
                # Add a straight
                remaining_length = self.config.target_length_m - total_length
                segment = self._generate_straight(remaining_length)
            
            segments.append(segment)
            total_length += segment.length_m
            
            # Safety limit
            if len(segments) > 50:
                break
        
        return segments
    
    def _generate_straight(self, max_length: float | None = None) -> TrackSegment:
        """Generate a straight segment.
        
        Args:
            max_length: Maximum allowed length
            
        Returns:
            Straight track segment
        """
        max_len = min(
            max_length or float('inf'),
            self.config.max_straight_length_m,
        )
        
        length = self._rng.uniform(
            self.config.min_straight_length_m,
            max_len,
        )
        
        # Add some elevation variation
        elevation_change = self._rng.uniform(
            -self.config.max_elevation_change_m * 0.2,
            self.config.max_elevation_change_m * 0.2,
        )
        
        # Constrain by gradient
        max_change = length * self.config.max_gradient
        elevation_change = np.clip(elevation_change, -max_change, max_change)
        
        return TrackSegment(
            segment_type=SegmentType.STRAIGHT,
            length_m=length,
            width_m=self.config.default_width_m,
            curvature=0.0,
            elevation_end_m=elevation_change,
        )
    
    def _generate_turn(self, turn_right: bool = True) -> TrackSegment:
        """Generate a turn segment.
        
        Args:
            turn_right: If True, generate right turn
            
        Returns:
            Turn track segment
        """
        # Random radius
        radius = self._rng.uniform(
            self.config.min_turn_radius_m,
            self.config.max_turn_radius_m,
        )
        
        # Curvature (positive = right turn)
        curvature = 1.0 / radius
        if not turn_right:
            curvature = -curvature
        
        # Turn angle
        angle_deg = self._rng.uniform(
            self.config.min_turn_angle_deg,
            self.config.max_turn_angle_deg * (1 - self.config.complexity * 0.5),
        )
        
        # Length from angle and radius
        length = radius * np.radians(angle_deg)
        
        # Banking (bank into the turn)
        banking = self._rng.uniform(
            self.config.min_banking_deg,
            self.config.max_banking_deg,
        )
        if not turn_right:
            banking = -banking
        
        # Elevation change
        elevation_change = self._rng.uniform(
            -self.config.max_elevation_change_m * 0.1,
            self.config.max_elevation_change_m * 0.1,
        )
        
        return TrackSegment(
            segment_type=SegmentType.RIGHT_TURN if turn_right else SegmentType.LEFT_TURN,
            length_m=length,
            width_m=self.config.default_width_m,
            curvature=curvature,
            banking_deg=banking,
            elevation_end_m=elevation_change,
        )
    
    def _close_track(self, track: Track) -> None:
        """Add segments to close the track loop.
        
        Args:
            track: Track to close
        """
        if len(track.segments) < 2:
            return
        
        # Get current end and target start
        last = track.segments[-1]
        first = track.segments[0]
        
        # Calculate vector to start
        dx = first.start_x - last.end_x
        dy = first.start_y - last.end_y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 10.0:
            # Already close enough
            return
        
        # Heading to target
        target_heading = np.arctan2(dy, dx)
        
        # Current heading
        current_heading = last.end_heading
        
        # Need to turn towards target and then approach
        heading_diff = target_heading - current_heading
        while heading_diff > np.pi:
            heading_diff -= 2 * np.pi
        while heading_diff < -np.pi:
            heading_diff += 2 * np.pi
        
        # Add turn if needed
        if abs(heading_diff) > np.radians(10):
            turn_radius = min(distance * 0.3, self.config.max_turn_radius_m)
            turn_radius = max(turn_radius, self.config.min_turn_radius_m)
            
            turn_curvature = 1.0 / turn_radius
            if heading_diff < 0:
                turn_curvature = -turn_curvature
            
            turn_length = turn_radius * abs(heading_diff)
            
            turn_segment = TrackSegment(
                segment_type=SegmentType.RIGHT_TURN if heading_diff > 0 else SegmentType.LEFT_TURN,
                length_m=turn_length,
                width_m=self.config.default_width_m,
                curvature=turn_curvature,
            )
            track.add_segment(turn_segment)
        
        # Add final straight to close
        last = track.segments[-1]
        dx = first.start_x - last.end_x
        dy = first.start_y - last.end_y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > 10.0:
            close_segment = TrackSegment(
                segment_type=SegmentType.STRAIGHT,
                length_m=distance,
                width_m=self.config.default_width_m,
                curvature=0.0,
            )
            track.add_segment(close_segment)
    
    def _add_kerbs(self, track: Track) -> None:
        """Add kerbs to turns.
        
        Args:
            track: Track to add kerbs to
        """
        for segment in track.segments:
            if segment.segment_type == SegmentType.STRAIGHT:
                continue
            
            # Probability check
            if self._rng.random() > self.config.kerb_probability:
                continue
            
            # Determine kerb type based on turn sharpness
            if segment.radius < 50:
                kerb_type = KerbType.SAUSAGE  # Tight turns
            elif segment.radius < 150:
                kerb_type = KerbType.FLAT
            else:
                kerb_type = KerbType.RUMBLE  # Fast turns
            
            # Add kerbs on both sides
            for side in ["left", "right"]:
                kerb = Kerb(
                    segment_id=segment.segment_id,
                    start_distance_m=0.0,
                    end_distance_m=segment.length_m,
                    side=side,
                    kerb_type=kerb_type,
                    width_m=1.0,
                )
                track.add_kerb(kerb)
    
    def generate_with_seed(self, seed: int, name: str | None = None) -> Track:
        """Generate track with specific seed.
        
        Args:
            seed: Random seed
            name: Optional track name
            
        Returns:
            Generated track
        """
        self._rng = np.random.default_rng(seed)
        return self.generate(name)
