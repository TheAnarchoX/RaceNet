#!/usr/bin/env python3
"""
Track Generation Example

This example demonstrates how to:
1. Generate tracks with different configurations
2. Use seeds for reproducible tracks
3. Inspect track properties and segments
4. Create tracks with specific characteristics

Run with: python generate_tracks.py
"""

import math

from racenet.track import TrackGenerator, Track
from racenet.track.generator import GeneratorConfig
from racenet.track.segment import SegmentType


def generate_default_track():
    """Generate a track with default settings."""
    print("=" * 60)
    print("1. Default Track Generation")
    print("=" * 60)
    
    generator = TrackGenerator()
    track = generator.generate("Default Track")
    
    print(f"\nTrack: {track.config.name}")
    print(f"Length: {track.length:.0f} m")
    print(f"Segments: {track.num_segments}")
    print(f"Is closed: {track.is_closed}")
    
    return track


def generate_seeded_tracks():
    """Generate reproducible tracks using seeds."""
    print("\n" + "=" * 60)
    print("2. Seeded Track Generation (Reproducible)")
    print("=" * 60)
    
    generator = TrackGenerator()
    
    # Generate same track twice with same seed
    track1 = generator.generate_with_seed(12345, "Track Seed 12345 - A")
    track2 = generator.generate_with_seed(12345, "Track Seed 12345 - B")
    
    print(f"\nTrack A segments: {track1.num_segments}")
    print(f"Track B segments: {track2.num_segments}")
    print(f"Same layout: {track1.num_segments == track2.num_segments}")
    
    # Different seeds produce different tracks
    track3 = generator.generate_with_seed(99999, "Track Seed 99999")
    print(f"\nDifferent seed track segments: {track3.num_segments}")


def generate_short_track():
    """Generate a short sprint track."""
    print("\n" + "=" * 60)
    print("3. Short Sprint Track")
    print("=" * 60)
    
    config = GeneratorConfig(
        target_length_m=2000.0,
        min_turn_radius_m=50.0,
        max_turn_radius_m=200.0,
        complexity=0.3,  # Less complex
    )
    
    generator = TrackGenerator(config)
    track = generator.generate("Sprint Circuit")
    
    print(f"\nTrack: {track.config.name}")
    print(f"Length: {track.length:.0f} m (target: 2000 m)")
    print(f"Segments: {track.num_segments}")


def generate_technical_track():
    """Generate a technical track with tight corners."""
    print("\n" + "=" * 60)
    print("4. Technical Track (Tight Corners)")
    print("=" * 60)
    
    config = GeneratorConfig(
        target_length_m=4000.0,
        min_turn_radius_m=30.0,   # Very tight hairpins
        max_turn_radius_m=150.0,  # Even fast corners are tighter
        max_straight_length_m=400.0,  # Shorter straights
        complexity=0.8,  # More complex
    )
    
    generator = TrackGenerator(config)
    track = generator.generate("Technical Circuit")
    
    print(f"\nTrack: {track.config.name}")
    print(f"Length: {track.length:.0f} m")
    print(f"Segments: {track.num_segments}")


def generate_high_speed_track():
    """Generate a high-speed track with sweeping corners."""
    print("\n" + "=" * 60)
    print("5. High-Speed Track (Fast Sweepers)")
    print("=" * 60)
    
    config = GeneratorConfig(
        target_length_m=5000.0,
        min_turn_radius_m=100.0,   # No tight corners
        max_turn_radius_m=500.0,   # Very fast sweepers
        min_straight_length_m=200.0,  # Long straights
        max_straight_length_m=800.0,
        complexity=0.4,
    )
    
    generator = TrackGenerator(config)
    track = generator.generate("High Speed Ring")
    
    print(f"\nTrack: {track.config.name}")
    print(f"Length: {track.length:.0f} m")
    print(f"Segments: {track.num_segments}")


def inspect_track_segments(track: Track):
    """Inspect individual track segments."""
    print("\n" + "=" * 60)
    print("6. Track Segment Inspection")
    print("=" * 60)
    
    print(f"\nAnalyzing: {track.config.name}")
    print("-" * 60)
    
    straights = 0
    left_turns = 0
    right_turns = 0
    total_turn_angle = 0.0
    
    for segment in track.segments:
        if segment.segment_type == SegmentType.STRAIGHT:
            straights += 1
        elif segment.segment_type == SegmentType.LEFT_TURN:
            left_turns += 1
            total_turn_angle += abs(segment.turn_angle)
        elif segment.segment_type == SegmentType.RIGHT_TURN:
            right_turns += 1
            total_turn_angle += abs(segment.turn_angle)
    
    print(f"Straights: {straights}")
    print(f"Left turns: {left_turns}")
    print(f"Right turns: {right_turns}")
    print(f"Total turn angle: {total_turn_angle * 180 / math.pi:.0f}°")
    
    # Find longest straight
    longest_straight = max(
        (s for s in track.segments if s.segment_type == SegmentType.STRAIGHT),
        key=lambda s: s.length_m,
        default=None
    )
    
    if longest_straight:
        print(f"Longest straight: {longest_straight.length_m:.0f} m")
    
    # Find tightest corner
    turns = [s for s in track.segments if s.segment_type != SegmentType.STRAIGHT]
    if turns:
        tightest = min(turns, key=lambda s: s.radius)
        print(f"Tightest corner radius: {tightest.radius:.0f} m")


def main():
    # Generate different track types
    default_track = generate_default_track()
    generate_seeded_tracks()
    generate_short_track()
    generate_technical_track()
    generate_high_speed_track()
    
    # Inspect the default track
    inspect_track_segments(default_track)
    
    print("\n" + "=" * 60)
    print("Track generation examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
