"""
Track module - Procedural track generation and track data structures.

This module contains:
- Track: Main track class with segments and features
- TrackGenerator: Procedural track generation with realistic constraints
- TrackSegment: Individual track segment with width, banking, elevation
- TrackFeatures: Kerbs, run-off areas, track limits
"""

from racenet.track.track import Track
from racenet.track.generator import TrackGenerator
from racenet.track.segment import TrackSegment
from racenet.track.features import Kerb, TrackLimits

__all__ = [
    "Track",
    "TrackGenerator",
    "TrackSegment",
    "Kerb",
    "TrackLimits",
]
