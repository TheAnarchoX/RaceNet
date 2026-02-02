"""Basic tests for the RaceNet track module."""

import pytest
import numpy as np

from racenet.track.track import Track, TrackConfig
from racenet.track.segment import TrackSegment, SegmentType
from racenet.track.generator import TrackGenerator, GeneratorConfig
from racenet.track.features import Kerb, KerbType, TrackLimits


class TestTrackSegment:
    """Test track segment."""
    
    def test_straight_segment(self):
        """Test straight segment geometry."""
        segment = TrackSegment(
            segment_type=SegmentType.STRAIGHT,
            length_m=100.0,
            width_m=12.0,
            curvature=0.0,
        )
        
        assert segment.radius == float('inf')
        assert segment.turn_angle == 0.0
        
    def test_turn_segment(self):
        """Test turn segment geometry."""
        segment = TrackSegment(
            segment_type=SegmentType.RIGHT_TURN,
            length_m=100.0,
            curvature=0.01,  # 100m radius
        )
        
        assert abs(segment.radius - 100.0) < 0.01
        assert segment.turn_angle > 0
        
    def test_position_at_distance(self):
        """Test position calculation along segment."""
        segment = TrackSegment(
            segment_type=SegmentType.STRAIGHT,
            length_m=100.0,
            start_x=0.0,
            start_y=0.0,
            start_heading=0.0,
        )
        
        pos = segment.get_position_at(50.0)
        assert abs(pos[0] - 50.0) < 0.01
        assert abs(pos[1]) < 0.01


class TestTrack:
    """Test track class."""
    
    def test_track_creation(self):
        """Test track can be created."""
        track = Track()
        assert track.length == 0.0
        assert track.num_segments == 0
        
    def test_track_add_segments(self):
        """Test adding segments to track."""
        track = Track()
        
        straight = TrackSegment(
            segment_type=SegmentType.STRAIGHT,
            length_m=100.0,
        )
        track.add_segment(straight)
        
        assert track.num_segments == 1
        
    def test_track_finalize(self):
        """Test track finalization."""
        track = Track()
        
        # Add a simple oval
        track.add_segment(TrackSegment(
            segment_type=SegmentType.STRAIGHT,
            length_m=200.0,
        ))
        track.add_segment(TrackSegment(
            segment_type=SegmentType.RIGHT_TURN,
            length_m=157.0,  # Half circle
            curvature=0.02,
        ))
        track.add_segment(TrackSegment(
            segment_type=SegmentType.STRAIGHT,
            length_m=200.0,
        ))
        track.add_segment(TrackSegment(
            segment_type=SegmentType.RIGHT_TURN,
            length_m=157.0,
            curvature=0.02,
        ))
        
        result = track.finalize()
        assert result
        assert track.length > 0
        
    def test_track_position_query(self):
        """Test position query on track."""
        track = Track()
        
        track.add_segment(TrackSegment(
            segment_type=SegmentType.STRAIGHT,
            length_m=100.0,
        ))
        track.finalize()
        
        pos = track.get_position_at_distance(50.0)
        assert pos is not None
        assert len(pos) == 3


class TestTrackGenerator:
    """Test track generator."""
    
    def test_generator_creates_track(self):
        """Test generator produces a track."""
        generator = TrackGenerator()
        track = generator.generate("Test Track")
        
        assert track is not None
        assert track.num_segments > 0
        assert track.length > 0
        
    def test_generator_with_seed(self):
        """Test generator produces reproducible tracks."""
        generator = TrackGenerator()
        
        track1 = generator.generate_with_seed(42, "Track A")
        track2 = generator.generate_with_seed(42, "Track B")
        
        assert track1.num_segments == track2.num_segments
        
    def test_generator_config(self):
        """Test generator respects configuration."""
        config = GeneratorConfig(
            target_length_m=3000.0,
            min_turn_radius_m=50.0,
        )
        generator = TrackGenerator(config)
        track = generator.generate()
        
        # Track should be roughly the target length
        assert track.length > config.target_length_m * 0.5
        assert track.length < config.target_length_m * 2.0


class TestTrackFeatures:
    """Test track features."""
    
    def test_kerb_creation(self):
        """Test kerb creation."""
        kerb = Kerb(
            segment_id=0,
            start_distance_m=0.0,
            end_distance_m=100.0,
            side="left",
            kerb_type=KerbType.FLAT,
        )
        
        assert kerb.length == 100.0
        assert kerb.get_grip_modifier() < 1.0
        
    def test_track_limits(self):
        """Test track limits detection."""
        limits = TrackLimits(track_width=12.0)
        
        # On track
        status, grip = limits.check_position(0.0)
        assert status == "on_track"
        assert grip == 1.0
        
        # Off track
        status, grip = limits.check_position(10.0)
        assert status != "on_track"
        assert grip < 1.0
