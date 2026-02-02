"""
Transmission component - Gearbox and drivetrain simulation.

Simulates:
- Sequential gearbox with multiple gear ratios
- Final drive ratio
- Shift timing and delays
- Clutch engagement (simplified)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List
import numpy as np


class GearState(IntEnum):
    """Gear positions."""
    NEUTRAL = 0
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4
    FIFTH = 5
    SIXTH = 6


@dataclass
class TransmissionConfig:
    """Configuration for a GT3-style sequential gearbox.
    
    Default values based on typical GT3 racing gearbox.
    """
    # Gear ratios (including final drive in calculations)
    gear_ratios: List[float] = field(default_factory=lambda: [
        0.0,    # Neutral
        3.91,   # 1st
        2.32,   # 2nd  
        1.65,   # 3rd
        1.28,   # 4th
        1.04,   # 5th
        0.88,   # 6th
    ])
    
    # Final drive ratio
    final_drive: float = 4.11
    
    # Shift characteristics
    shift_time_up_s: float = 0.050    # Upshift time (50ms - sequential box)
    shift_time_down_s: float = 0.080  # Downshift time (80ms with rev match)
    
    # Rev match parameters
    auto_rev_match: bool = True  # Automatic blip on downshift
    rev_match_intensity: float = 0.3  # How aggressive the blip is
    
    # Clutch (simplified for sequential)
    clutch_engage_time_s: float = 0.030
    
    # Wheel radius for speed calculations (meters)
    wheel_radius_m: float = 0.33


class Transmission:
    """GT3-style sequential gearbox simulation.
    
    Provides realistic shifting behavior including:
    - Sequential shift pattern
    - Shift delays
    - Automatic rev matching on downshifts
    - Drive ratio calculations
    """
    
    def __init__(self, config: TransmissionConfig | None = None):
        """Initialize transmission with optional custom configuration.
        
        Args:
            config: Transmission configuration. Uses GT3 defaults if None.
        """
        self.config = config or TransmissionConfig()
        
        # Current state
        self._current_gear: int = int(GearState.NEUTRAL)
        self._target_gear: int = int(GearState.NEUTRAL)
        self._shift_progress: float = 1.0  # 1.0 = shift complete
        self._shift_time_remaining: float = 0.0
        self._clutch_engaged: bool = True
        
        # Shift request tracking
        self._shift_requested: bool = False
        
    @property
    def current_gear(self) -> int:
        """Current gear number (0 = neutral)."""
        return self._current_gear
    
    @property
    def is_shifting(self) -> bool:
        """Check if a shift is in progress."""
        return self._shift_progress < 1.0
    
    @property
    def clutch_engaged(self) -> bool:
        """Check if clutch is fully engaged."""
        return self._clutch_engaged and not self.is_shifting
    
    @property
    def max_gear(self) -> int:
        """Maximum gear number."""
        return len(self.config.gear_ratios) - 1
    
    def get_gear_ratio(self, gear: int | None = None) -> float:
        """Get gear ratio for specified or current gear.
        
        Args:
            gear: Gear number (uses current if None)
            
        Returns:
            Gear ratio (0.0 for neutral)
        """
        gear = gear if gear is not None else self._current_gear
        if gear < 0 or gear >= len(self.config.gear_ratios):
            return 0.0
        return self.config.gear_ratios[gear]
    
    def get_total_ratio(self, gear: int | None = None) -> float:
        """Get total drive ratio (gear ratio * final drive).
        
        Args:
            gear: Gear number (uses current if None)
            
        Returns:
            Total drive ratio
        """
        return self.get_gear_ratio(gear) * self.config.final_drive
    
    def get_wheel_rpm_from_engine(self, engine_rpm: float, gear: int | None = None) -> float:
        """Calculate wheel RPM from engine RPM.
        
        Args:
            engine_rpm: Current engine RPM
            gear: Gear number (uses current if None)
            
        Returns:
            Wheel RPM
        """
        total_ratio = self.get_total_ratio(gear)
        if total_ratio == 0:
            return 0.0
        return engine_rpm / total_ratio
    
    def get_engine_rpm_from_wheel(self, wheel_rpm: float, gear: int | None = None) -> float:
        """Calculate engine RPM from wheel RPM.
        
        Args:
            wheel_rpm: Current wheel RPM
            gear: Gear number (uses current if None)
            
        Returns:
            Engine RPM
        """
        return wheel_rpm * self.get_total_ratio(gear)
    
    def get_speed_from_rpm(self, engine_rpm: float, gear: int | None = None) -> float:
        """Calculate vehicle speed (m/s) from engine RPM.
        
        Args:
            engine_rpm: Current engine RPM
            gear: Gear number (uses current if None)
            
        Returns:
            Vehicle speed in m/s
        """
        wheel_rpm = self.get_wheel_rpm_from_engine(engine_rpm, gear)
        # Speed = wheel circumference * RPM / 60
        wheel_circumference = 2 * np.pi * self.config.wheel_radius_m
        return wheel_rpm * wheel_circumference / 60.0
    
    def shift_up(self) -> bool:
        """Request upshift.
        
        Returns:
            True if shift request accepted
        """
        if self.is_shifting:
            return False
        if self._current_gear >= self.max_gear:
            return False
        
        self._target_gear = self._current_gear + 1
        self._shift_time_remaining = self.config.shift_time_up_s
        self._shift_progress = 0.0
        self._clutch_engaged = False
        return True
    
    def shift_down(self) -> bool:
        """Request downshift.
        
        Returns:
            True if shift request accepted
        """
        if self.is_shifting:
            return False
        if self._current_gear <= int(GearState.FIRST):
            return False
        
        self._target_gear = self._current_gear - 1
        self._shift_time_remaining = self.config.shift_time_down_s
        self._shift_progress = 0.0
        self._clutch_engaged = False
        return True
    
    def set_gear(self, gear: int) -> bool:
        """Directly set gear (for initialization or special cases).
        
        Args:
            gear: Target gear number
            
        Returns:
            True if gear set successfully
        """
        if gear < 0 or gear > self.max_gear:
            return False
        
        self._current_gear = gear
        self._target_gear = gear
        self._shift_progress = 1.0
        self._shift_time_remaining = 0.0
        self._clutch_engaged = True
        return True
    
    def update(self, dt: float) -> dict:
        """Update transmission state for one time step.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Dictionary with shift state info
        """
        shift_info = {
            "shifting": False,
            "gear_changed": False,
            "rev_match_request": 0.0,
        }
        
        if self.is_shifting:
            shift_info["shifting"] = True
            self._shift_time_remaining -= dt
            
            # Calculate shift progress
            if self._target_gear > self._current_gear:
                total_time = self.config.shift_time_up_s
            else:
                total_time = self.config.shift_time_down_s
                # Rev match during downshift
                if self.config.auto_rev_match:
                    shift_info["rev_match_request"] = self.config.rev_match_intensity
            
            self._shift_progress = 1.0 - (self._shift_time_remaining / total_time)
            self._shift_progress = np.clip(self._shift_progress, 0.0, 1.0)
            
            # Complete shift
            if self._shift_time_remaining <= 0:
                self._current_gear = self._target_gear
                self._shift_progress = 1.0
                self._clutch_engaged = True
                shift_info["gear_changed"] = True
        
        return shift_info
    
    def get_drive_torque(self, engine_torque: float) -> float:
        """Calculate torque at the wheels.
        
        Args:
            engine_torque: Torque from engine in Nm
            
        Returns:
            Wheel torque in Nm
        """
        if not self.clutch_engaged:
            return 0.0
        
        return engine_torque * self.get_total_ratio()
    
    def reset(self) -> None:
        """Reset transmission to initial state."""
        self._current_gear = int(GearState.NEUTRAL)
        self._target_gear = int(GearState.NEUTRAL)
        self._shift_progress = 1.0
        self._shift_time_remaining = 0.0
        self._clutch_engaged = True
    
    def get_state(self) -> dict:
        """Get current transmission state for telemetry.
        
        Returns:
            Dictionary containing transmission state values
        """
        return {
            "gear": self._current_gear,
            "target_gear": self._target_gear,
            "is_shifting": self.is_shifting,
            "shift_progress": self._shift_progress,
            "clutch_engaged": self._clutch_engaged,
            "gear_ratio": self.get_gear_ratio(),
            "total_ratio": self.get_total_ratio(),
        }
