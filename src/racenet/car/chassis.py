"""
Chassis component - Vehicle mass properties and center of gravity.

Defines:
- Total vehicle mass
- Center of gravity position
- Mass distribution
- Inertia properties
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ChassisConfig:
    """Configuration for GT3-style chassis.
    
    Default values based on typical GT3 car specifications.
    Minimum weight ~1300kg including driver.
    """
    # Mass properties
    total_mass_kg: float = 1350.0  # Including driver
    
    # Center of gravity position (relative to front axle and ground)
    cog_height_m: float = 0.45      # Height above ground
    cog_from_front_m: float = 1.175  # Distance from front axle (47% front)
    
    # Wheelbase and track (for reference, also in suspension)
    wheelbase_m: float = 2.50
    front_track_m: float = 1.65
    rear_track_m: float = 1.60
    
    # Vehicle dimensions
    length_m: float = 4.55
    width_m: float = 2.00
    height_m: float = 1.25
    
    # Moment of inertia approximations (kg*m^2)
    # Simplified as fractions of total mass * characteristic length^2
    yaw_inertia_factor: float = 0.25    # For rotation about vertical axis
    pitch_inertia_factor: float = 0.20  # For rotation about lateral axis
    roll_inertia_factor: float = 0.15   # For rotation about longitudinal axis


class Chassis:
    """GT3-style chassis simulation.
    
    Manages:
    - Mass properties
    - Center of gravity calculations
    - Moment of inertia
    - Weight distribution
    """
    
    def __init__(self, config: ChassisConfig | None = None):
        """Initialize chassis with optional custom configuration.
        
        Args:
            config: Chassis configuration. Uses GT3 defaults if None.
        """
        self.config = config or ChassisConfig()
        
        # Fuel mass (can be reduced during race)
        self._fuel_mass_kg: float = 0.0
        
        # Calculate derived properties
        self._update_derived_properties()
    
    def _update_derived_properties(self) -> None:
        """Recalculate derived properties after mass changes."""
        total = self.total_mass
        
        # Calculate moments of inertia
        self._yaw_inertia = (
            total * self.config.yaw_inertia_factor * 
            self.config.wheelbase_m ** 2
        )
        self._pitch_inertia = (
            total * self.config.pitch_inertia_factor * 
            self.config.wheelbase_m ** 2
        )
        self._roll_inertia = (
            total * self.config.roll_inertia_factor * 
            self.config.width_m ** 2
        )
    
    @property
    def total_mass(self) -> float:
        """Total vehicle mass including fuel in kg."""
        return self.config.total_mass_kg + self._fuel_mass_kg
    
    @property
    def cog_height(self) -> float:
        """Center of gravity height in meters."""
        # Simplified - could account for fuel level
        return self.config.cog_height_m
    
    @property
    def cog_from_front(self) -> float:
        """Center of gravity distance from front axle in meters."""
        return self.config.cog_from_front_m
    
    @property
    def front_weight_fraction(self) -> float:
        """Fraction of weight on front axle (static)."""
        return 1.0 - (self.config.cog_from_front_m / self.config.wheelbase_m)
    
    @property
    def rear_weight_fraction(self) -> float:
        """Fraction of weight on rear axle (static)."""
        return self.config.cog_from_front_m / self.config.wheelbase_m
    
    @property
    def yaw_inertia(self) -> float:
        """Yaw moment of inertia in kg*m^2."""
        return self._yaw_inertia
    
    @property
    def pitch_inertia(self) -> float:
        """Pitch moment of inertia in kg*m^2."""
        return self._pitch_inertia
    
    @property
    def roll_inertia(self) -> float:
        """Roll moment of inertia in kg*m^2."""
        return self._roll_inertia
    
    def set_fuel_mass(self, fuel_kg: float) -> None:
        """Set current fuel mass.
        
        Args:
            fuel_kg: Fuel mass in kg
        """
        self._fuel_mass_kg = max(0.0, fuel_kg)
        self._update_derived_properties()
    
    def consume_fuel(self, fuel_kg: float) -> None:
        """Reduce fuel mass by given amount.
        
        Args:
            fuel_kg: Amount of fuel consumed in kg
        """
        self._fuel_mass_kg = max(0.0, self._fuel_mass_kg - fuel_kg)
        self._update_derived_properties()
    
    def get_weight_distribution(self) -> tuple[float, float]:
        """Get static weight distribution.
        
        Returns:
            Tuple of (front_fraction, rear_fraction)
        """
        return self.front_weight_fraction, self.rear_weight_fraction
    
    def reset(self, fuel_kg: float = 0.0) -> None:
        """Reset chassis state.
        
        Args:
            fuel_kg: Initial fuel load in kg
        """
        self._fuel_mass_kg = fuel_kg
        self._update_derived_properties()
    
    def get_state(self) -> dict:
        """Get current chassis state for telemetry.
        
        Returns:
            Dictionary containing chassis state values
        """
        return {
            "total_mass_kg": self.total_mass,
            "fuel_mass_kg": self._fuel_mass_kg,
            "cog_height_m": self.cog_height,
            "front_weight_pct": self.front_weight_fraction * 100,
            "rear_weight_pct": self.rear_weight_fraction * 100,
            "yaw_inertia": self._yaw_inertia,
        }
