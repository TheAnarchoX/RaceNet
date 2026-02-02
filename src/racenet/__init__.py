"""
RaceNet - A GT3-style racing simulation framework for machine learning.

This package provides a detailed physics-based racing simulation with:
- Realistic GT3-style car models with engine, aero, suspension, chassis, electronics
- Procedurally generated tracks with kerbs, elevation, and banking
- Telemetry system for real-time car data
- Scoring system for lap times and driving style metrics
- Multi-car/multi-generation spawning for ML training
"""

__version__ = "0.1.0"

from racenet.simulation.simulator import Simulator
from racenet.car.car import Car
from racenet.track.track import Track

__all__ = ["Simulator", "Car", "Track", "__version__"]
