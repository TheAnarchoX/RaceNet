"""
Tire component - Grip and tire dynamics simulation.

Simulates:
- Tire grip based on load and slip
- Temperature effects
- Wear modeling
- Combined slip (longitudinal + lateral)
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class TireConfig:
    """Configuration for GT3-style racing tires.
    
    Default values based on typical GT3 slick tires.
    """
    # Tire dimensions
    width_mm: float = 305.0        # Front/rear may differ
    aspect_ratio: float = 0.30     # Low profile racing tire
    rim_diameter_inch: float = 18.0
    
    # Grip coefficients
    peak_mu_x: float = 1.6         # Longitudinal grip coefficient
    peak_mu_y: float = 1.5         # Lateral grip coefficient
    
    # Slip angles/ratios at peak grip
    optimal_slip_ratio: float = 0.10    # 10% slip for peak longitudinal
    optimal_slip_angle_deg: float = 8.0  # 8 degrees for peak lateral
    
    # Grip falloff after peak
    slip_ratio_falloff: float = 0.7     # Grip at high slip vs peak
    slip_angle_falloff: float = 0.75    # Grip at high slip angle vs peak
    
    # Temperature effects
    optimal_temp_c: float = 90.0
    temp_range_c: float = 30.0      # +/- for good grip
    cold_grip_factor: float = 0.7   # Grip when cold
    hot_grip_factor: float = 0.85   # Grip when overheated
    
    # Wear effects
    wear_rate: float = 0.00001      # Wear per meter of slip distance
    worn_grip_factor: float = 0.85  # Grip at 100% wear
    
    # Load sensitivity (grip drops at higher loads)
    load_sensitivity: float = 0.0001  # Grip reduction per N of load


class Tire:
    """Single tire simulation.
    
    Calculates grip forces based on:
    - Vertical load
    - Slip ratio (longitudinal)
    - Slip angle (lateral)
    - Temperature
    - Wear
    """
    
    def __init__(self, config: TireConfig | None = None, position: str = "FL"):
        """Initialize tire with optional configuration.
        
        Args:
            config: Tire configuration. Uses defaults if None.
            position: Tire position identifier (FL, FR, RL, RR)
        """
        self.config = config or TireConfig()
        self.position = position
        
        # Calculate radius
        sidewall_height = self.config.width_mm * self.config.aspect_ratio
        rim_radius = (self.config.rim_diameter_inch * 25.4) / 2
        self._radius_m = (rim_radius + sidewall_height) / 1000.0
        
        # State
        self._temperature_c: float = 25.0  # Start cold
        self._wear: float = 0.0  # 0 = new, 1 = worn out
        self._slip_ratio: float = 0.0
        self._slip_angle_deg: float = 0.0
        self._load_n: float = 0.0
        
        # Output forces
        self._force_x: float = 0.0  # Longitudinal
        self._force_y: float = 0.0  # Lateral
    
    @property
    def radius(self) -> float:
        """Tire radius in meters."""
        return self._radius_m
    
    @property
    def temperature(self) -> float:
        """Current tire temperature in Celsius."""
        return self._temperature_c
    
    @property
    def wear(self) -> float:
        """Current tire wear (0 = new, 1 = worn)."""
        return self._wear
    
    @property
    def slip_ratio(self) -> float:
        """Current longitudinal slip ratio."""
        return self._slip_ratio
    
    @property
    def slip_angle(self) -> float:
        """Current slip angle in degrees."""
        return self._slip_angle_deg
    
    @property
    def force_x(self) -> float:
        """Current longitudinal force in N."""
        return self._force_x
    
    @property
    def force_y(self) -> float:
        """Current lateral force in N."""
        return self._force_y
    
    def _get_temperature_factor(self) -> float:
        """Calculate grip factor based on temperature.
        
        Returns:
            Grip multiplier (0-1)
        """
        temp_diff = abs(self._temperature_c - self.config.optimal_temp_c)
        
        if temp_diff <= self.config.temp_range_c:
            # In optimal range - linear interpolation to 1.0
            return 1.0 - (temp_diff / self.config.temp_range_c) * 0.1
        
        # Outside optimal range
        if self._temperature_c < self.config.optimal_temp_c:
            return self.config.cold_grip_factor
        else:
            return self.config.hot_grip_factor
    
    def _get_wear_factor(self) -> float:
        """Calculate grip factor based on wear.
        
        Returns:
            Grip multiplier (0-1)
        """
        # Linear interpolation from new (1.0) to worn (worn_grip_factor)
        return 1.0 - self._wear * (1.0 - self.config.worn_grip_factor)
    
    def _get_load_factor(self, load_n: float) -> float:
        """Calculate grip reduction from load sensitivity.
        
        Args:
            load_n: Vertical load in Newtons
            
        Returns:
            Grip multiplier (0-1)
        """
        # Higher load = slightly lower grip coefficient
        reduction = load_n * self.config.load_sensitivity
        return max(0.7, 1.0 - reduction)
    
    def _calculate_longitudinal_force(
        self,
        load_n: float,
        slip_ratio: float,
    ) -> float:
        """Calculate longitudinal (traction/braking) force.
        
        Uses simplified Pacejka-like curve.
        
        Args:
            load_n: Vertical load
            slip_ratio: Longitudinal slip ratio
            
        Returns:
            Longitudinal force in N
        """
        # Normalized slip
        sr = abs(slip_ratio)
        optimal = self.config.optimal_slip_ratio
        
        if sr < optimal:
            # Building up to peak - roughly linear
            normalized_force = sr / optimal
        else:
            # Past peak - gradual falloff
            excess = (sr - optimal) / optimal
            falloff = 1.0 - (1.0 - self.config.slip_ratio_falloff) * min(excess, 1.0)
            normalized_force = falloff
        
        # Apply modifiers
        temp_factor = self._get_temperature_factor()
        wear_factor = self._get_wear_factor()
        load_factor = self._get_load_factor(load_n)
        
        # Calculate force
        mu = self.config.peak_mu_x * temp_factor * wear_factor * load_factor
        force = mu * load_n * normalized_force
        
        # Apply sign
        return force if slip_ratio >= 0 else -force
    
    def _calculate_lateral_force(
        self,
        load_n: float,
        slip_angle_deg: float,
    ) -> float:
        """Calculate lateral (cornering) force.
        
        Uses simplified Pacejka-like curve.
        
        Args:
            load_n: Vertical load
            slip_angle_deg: Slip angle in degrees
            
        Returns:
            Lateral force in N
        """
        # Normalized slip angle
        sa = abs(slip_angle_deg)
        optimal = self.config.optimal_slip_angle_deg
        
        if sa < optimal:
            # Building up to peak
            normalized_force = sa / optimal
        else:
            # Past peak - gradual falloff
            excess = (sa - optimal) / optimal
            falloff = 1.0 - (1.0 - self.config.slip_angle_falloff) * min(excess, 1.0)
            normalized_force = falloff
        
        # Apply modifiers
        temp_factor = self._get_temperature_factor()
        wear_factor = self._get_wear_factor()
        load_factor = self._get_load_factor(load_n)
        
        # Calculate force
        mu = self.config.peak_mu_y * temp_factor * wear_factor * load_factor
        force = mu * load_n * normalized_force
        
        # Apply sign (positive slip angle = force to left)
        return -force if slip_angle_deg >= 0 else force
    
    def _apply_combined_slip(self, fx: float, fy: float) -> tuple[float, float]:
        """Apply combined slip friction circle limitation.
        
        Args:
            fx: Raw longitudinal force
            fy: Raw lateral force
            
        Returns:
            Tuple of (adjusted_fx, adjusted_fy)
        """
        # Calculate total force magnitude
        total = np.sqrt(fx**2 + fy**2)
        
        if total < 1e-6:
            return fx, fy
        
        # Maximum available force (simplified friction circle)
        max_force = max(abs(fx), abs(fy)) * 1.1  # Slight overlap allowed
        
        if total > max_force:
            # Scale down proportionally
            scale = max_force / total
            return fx * scale, fy * scale
        
        return fx, fy
    
    def update(
        self,
        load_n: float,
        slip_ratio: float,
        slip_angle_deg: float,
        ground_speed_mps: float,
        dt: float,
    ) -> tuple[float, float]:
        """Update tire state and calculate forces.
        
        Args:
            load_n: Vertical load in N
            slip_ratio: Longitudinal slip ratio
            slip_angle_deg: Slip angle in degrees
            ground_speed_mps: Ground speed for thermal calcs
            dt: Time step in seconds
            
        Returns:
            Tuple of (longitudinal_force, lateral_force) in N
        """
        self._load_n = load_n
        self._slip_ratio = slip_ratio
        self._slip_angle_deg = slip_angle_deg
        
        # Calculate raw forces
        fx = self._calculate_longitudinal_force(load_n, slip_ratio)
        fy = self._calculate_lateral_force(load_n, slip_angle_deg)
        
        # Apply combined slip limitation
        self._force_x, self._force_y = self._apply_combined_slip(fx, fy)
        
        # Update temperature
        slip_energy = abs(self._force_x * slip_ratio) + abs(self._force_y * np.radians(slip_angle_deg))
        heat_generation = slip_energy * 0.00001  # Simplified
        heat_dissipation = (self._temperature_c - 25.0) * ground_speed_mps * 0.0002
        self._temperature_c += (heat_generation - heat_dissipation) * dt
        self._temperature_c = np.clip(self._temperature_c, 20.0, 150.0)
        
        # Update wear
        slip_distance = abs(slip_ratio) * ground_speed_mps * dt
        self._wear += slip_distance * self.config.wear_rate
        self._wear = min(self._wear, 1.0)
        
        return self._force_x, self._force_y
    
    def reset(self) -> None:
        """Reset tire to initial state."""
        self._temperature_c = 25.0
        self._wear = 0.0
        self._slip_ratio = 0.0
        self._slip_angle_deg = 0.0
        self._force_x = 0.0
        self._force_y = 0.0
    
    def get_state(self) -> dict:
        """Get current tire state for telemetry.
        
        Returns:
            Dictionary containing tire state values
        """
        return {
            "position": self.position,
            "temperature_c": self._temperature_c,
            "wear_pct": self._wear * 100,
            "slip_ratio": self._slip_ratio,
            "slip_angle_deg": self._slip_angle_deg,
            "load_n": self._load_n,
            "force_x_n": self._force_x,
            "force_y_n": self._force_y,
            "grip_temp_factor": self._get_temperature_factor(),
            "grip_wear_factor": self._get_wear_factor(),
        }


@dataclass
class TireSetConfig:
    """Configuration for a complete set of 4 tires."""
    front_config: TireConfig | None = None
    rear_config: TireConfig | None = None


class TireSet:
    """Complete set of 4 tires.
    
    Manages all four tires with potentially different front/rear specs.
    """
    
    def __init__(self, config: TireSetConfig | None = None):
        """Initialize tire set with optional configuration.
        
        Args:
            config: Tire set configuration. Uses defaults if None.
        """
        self.config = config or TireSetConfig()
        
        # Create individual tires
        front_cfg = self.config.front_config or TireConfig(width_mm=275)
        rear_cfg = self.config.rear_config or TireConfig(width_mm=305)
        
        self.fl = Tire(front_cfg, "FL")
        self.fr = Tire(front_cfg, "FR")
        self.rl = Tire(rear_cfg, "RL")
        self.rr = Tire(rear_cfg, "RR")
        
        self._tires = [self.fl, self.fr, self.rl, self.rr]
    
    @property
    def tires(self) -> List[Tire]:
        """List of all 4 tires [FL, FR, RL, RR]."""
        return self._tires
    
    def get_tire(self, position: str) -> Tire:
        """Get tire by position.
        
        Args:
            position: Position string (FL, FR, RL, RR)
            
        Returns:
            Tire at specified position
        """
        positions = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
        return self._tires[positions.get(position.upper(), 0)]
    
    def get_average_temperature(self) -> float:
        """Get average tire temperature.
        
        Returns:
            Average temperature in Celsius
        """
        return np.mean([t.temperature for t in self._tires])
    
    def get_average_wear(self) -> float:
        """Get average tire wear.
        
        Returns:
            Average wear (0-1)
        """
        return np.mean([t.wear for t in self._tires])
    
    def get_max_slip_ratio(self) -> float:
        """Get maximum slip ratio across all tires.
        
        Returns:
            Maximum absolute slip ratio
        """
        return max(abs(t.slip_ratio) for t in self._tires)
    
    def get_rear_slip_ratio(self) -> float:
        """Get average rear slip ratio for TC.
        
        Returns:
            Average rear slip ratio
        """
        return (abs(self.rl.slip_ratio) + abs(self.rr.slip_ratio)) / 2
    
    def reset(self) -> None:
        """Reset all tires to initial state."""
        for tire in self._tires:
            tire.reset()
    
    def get_state(self) -> dict:
        """Get current tire set state for telemetry.
        
        Returns:
            Dictionary containing all tire states
        """
        return {
            "fl": self.fl.get_state(),
            "fr": self.fr.get_state(),
            "rl": self.rl.get_state(),
            "rr": self.rr.get_state(),
            "avg_temp_c": self.get_average_temperature(),
            "avg_wear_pct": self.get_average_wear() * 100,
        }
