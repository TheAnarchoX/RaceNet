"""
Simulation module - Core physics simulation loop.

This module contains:
- Simulator: Main simulation loop managing cars and tracks
- Physics: Core physics calculations
- World: World state management
"""

from racenet.simulation.simulator import Simulator
from racenet.simulation.physics import PhysicsEngine
from racenet.simulation.world import World

__all__ = [
    "Simulator",
    "PhysicsEngine",
    "World",
]
