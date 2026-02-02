"""
ML module - Machine learning integration interfaces.

This module contains:
- RaceEnv: Gymnasium-compatible environment
- ObservationSpace: Car state observation definitions
- ActionSpace: Car control action definitions
- MultiCarManager: Multi-car/multi-generation spawning
"""

from racenet.ml.environment import RaceEnv
from racenet.ml.spaces import ObservationSpace, ActionSpace
from racenet.ml.multi_car import MultiCarManager

__all__ = [
    "RaceEnv",
    "ObservationSpace",
    "ActionSpace",
    "MultiCarManager",
]
