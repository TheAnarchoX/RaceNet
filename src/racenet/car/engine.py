"""
Engine component - Power delivery simulation for GT3-style racing cars.

Simulates:
- RPM-based torque curves
- Power output calculations
- Rev limiter
- Engine temperature (thermal model)
- Fuel consumption
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class EngineConfig:
    """Configuration for a GT3-style engine.
    
    Default values are based on a typical GT3 4.0L flat-six engine
    producing approximately 500-520 HP.
    """
    # RPM limits
    idle_rpm: float = 1000.0
    max_rpm: float = 9000.0
    rev_limiter_rpm: float = 8500.0
    
    # Power characteristics
    max_torque_nm: float = 470.0  # Peak torque in Nm
    max_power_kw: float = 390.0   # Peak power in kW (~520 HP)
    
    # RPM points for torque curve (simplified 5-point curve)
    # Format: List of (rpm, torque_fraction) tuples
    torque_curve: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1000, 0.45),   # Low RPM
        (3000, 0.75),   # Building torque
        (5000, 0.95),   # Near peak torque
        (6500, 1.00),   # Peak torque
        (8500, 0.85),   # High RPM falloff
    ])
    
    # Thermal characteristics
    operating_temp_c: float = 90.0
    overheat_temp_c: float = 120.0
    
    # Fuel consumption (liters per kWh at full load)
    fuel_consumption_rate: float = 0.25
    
    # Engine inertia (affects throttle response)
    inertia_kg_m2: float = 0.15


class Engine:
    """GT3-style racing engine simulation.
    
    Provides realistic engine behavior including:
    - Torque/power delivery based on RPM
    - Rev limiter with fuel cut
    - Throttle response with inertia
    - Basic thermal model
    """
    
    def __init__(self, config: EngineConfig | None = None):
        """Initialize engine with optional custom configuration.
        
        Args:
            config: Engine configuration. Uses GT3 defaults if None.
        """
        self.config = config or EngineConfig()
        
        # Current state
        self._rpm: float = self.config.idle_rpm
        self._throttle: float = 0.0  # 0.0 to 1.0
        self._temperature: float = self.config.operating_temp_c
        self._fuel_used: float = 0.0
        
        # Rev limiter state
        self._rev_limiter_active: bool = False
        
        # Pre-compute interpolated torque curve
        self._torque_curve_rpms = np.array([p[0] for p in self.config.torque_curve])
        self._torque_curve_fractions = np.array([p[1] for p in self.config.torque_curve])
    
    @property
    def rpm(self) -> float:
        """Current engine RPM."""
        return self._rpm
    
    @rpm.setter
    def rpm(self, value: float) -> None:
        """Set engine RPM, clamped to valid range."""
        self._rpm = np.clip(value, self.config.idle_rpm, self.config.max_rpm)
    
    @property
    def throttle(self) -> float:
        """Current throttle position (0.0 to 1.0)."""
        return self._throttle
    
    @throttle.setter
    def throttle(self, value: float) -> None:
        """Set throttle position, clamped to 0-1."""
        self._throttle = np.clip(value, 0.0, 1.0)
    
    @property
    def temperature(self) -> float:
        """Current engine temperature in Celsius."""
        return self._temperature
    
    @property
    def fuel_used(self) -> float:
        """Total fuel consumed in liters."""
        return self._fuel_used
    
    @property
    def is_overheating(self) -> bool:
        """Check if engine is overheating."""
        return self._temperature > self.config.overheat_temp_c
    
    @property
    def rev_limiter_active(self) -> bool:
        """Check if rev limiter is cutting fuel."""
        return self._rev_limiter_active
    
    def get_torque_fraction(self, rpm: float) -> float:
        """Get torque fraction (0-1) at given RPM from torque curve.
        
        Args:
            rpm: Engine RPM to query
            
        Returns:
            Fraction of max torque available at this RPM
        """
        return float(np.interp(rpm, self._torque_curve_rpms, self._torque_curve_fractions))
    
    def get_torque(self) -> float:
        """Get current output torque in Nm.
        
        Takes into account:
        - Throttle position
        - Torque curve
        - Rev limiter
        
        Returns:
            Current torque output in Nm
        """
        # Rev limiter cuts fuel
        if self._rpm >= self.config.rev_limiter_rpm:
            self._rev_limiter_active = True
            return 0.0
        else:
            self._rev_limiter_active = False
        
        # Base torque from curve
        torque_fraction = self.get_torque_fraction(self._rpm)
        base_torque = self.config.max_torque_nm * torque_fraction
        
        # Apply throttle
        return base_torque * self._throttle
    
    def get_power(self) -> float:
        """Get current power output in kW.
        
        Returns:
            Current power output in kW
        """
        torque = self.get_torque()
        # Power (kW) = Torque (Nm) * RPM / 9549
        return torque * self._rpm / 9549.0
    
    def update(self, dt: float, wheel_rpm: float, gear_ratio: float) -> float:
        """Update engine state for one time step.
        
        Args:
            dt: Time step in seconds
            wheel_rpm: RPM at the driven wheels
            gear_ratio: Current total gear ratio (gearbox * final drive)
            
        Returns:
            Torque output in Nm for this time step
        """
        # Update RPM from wheel speed (through transmission)
        target_rpm = wheel_rpm * gear_ratio
        target_rpm = np.clip(target_rpm, self.config.idle_rpm, self.config.max_rpm)
        
        # Apply engine inertia for smooth RPM changes
        rpm_rate = 1.0 / (self.config.inertia_kg_m2 * 10)  # Simplified
        self._rpm += (target_rpm - self._rpm) * min(rpm_rate * dt, 1.0)
        
        # Get output torque
        torque = self.get_torque()
        
        # Update fuel consumption
        power_kw = self.get_power()
        fuel_rate = power_kw * self.config.fuel_consumption_rate / 3600.0  # L/s
        self._fuel_used += fuel_rate * dt
        
        # Simple thermal model
        heat_generation = power_kw * 0.01  # Heat from power
        heat_dissipation = (self._temperature - 30.0) * 0.02  # Cooling
        self._temperature += (heat_generation - heat_dissipation) * dt
        
        return torque
    
    def reset(self) -> None:
        """Reset engine to initial state."""
        self._rpm = self.config.idle_rpm
        self._throttle = 0.0
        self._temperature = self.config.operating_temp_c
        self._fuel_used = 0.0
        self._rev_limiter_active = False
    
    def get_state(self) -> dict:
        """Get current engine state for telemetry.
        
        Returns:
            Dictionary containing engine state values
        """
        return {
            "rpm": self._rpm,
            "throttle": self._throttle,
            "torque_nm": self.get_torque(),
            "power_kw": self.get_power(),
            "temperature_c": self._temperature,
            "fuel_used_l": self._fuel_used,
            "rev_limiter_active": self._rev_limiter_active,
        }
