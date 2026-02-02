"""
Aerodynamics component - Downforce and drag simulation.

Simulates:
- Speed-dependent downforce
- Drag coefficient calculations
- Front/rear aero balance
- Drag reduction (DRS-style, optional)
"""

from dataclasses import dataclass
import numpy as np


@dataclass 
class AeroConfig:
    """Configuration for GT3-style aerodynamics.
    
    Default values based on typical GT3 car at medium downforce setting.
    """
    # Air properties
    air_density_kg_m3: float = 1.225  # Standard sea level
    
    # Frontal area
    frontal_area_m2: float = 2.0
    
    # Coefficients
    drag_coefficient: float = 0.38      # Cd
    lift_coefficient: float = -1.8      # Negative = downforce (Cl)
    
    # Aero balance (0.0 = all front, 1.0 = all rear)
    aero_balance: float = 0.55  # 55% rear typical GT3
    
    # Center of pressure height (meters above ground)
    cop_height_m: float = 0.35
    
    # Reference area for downforce calculations
    wing_area_m2: float = 1.5
    
    # Optional DRS-style drag reduction
    drs_drag_reduction: float = 0.15    # Reduction in drag when active
    drs_downforce_reduction: float = 0.30  # Reduction in downforce when active


class Aero:
    """GT3-style aerodynamics simulation.
    
    Calculates speed-dependent forces:
    - Drag opposing motion
    - Downforce increasing tire grip
    - Aero balance affecting handling
    """
    
    def __init__(self, config: AeroConfig | None = None):
        """Initialize aerodynamics with optional custom configuration.
        
        Args:
            config: Aero configuration. Uses GT3 defaults if None.
        """
        self.config = config or AeroConfig()
        
        # State
        self._drs_active: bool = False
        self._current_downforce: float = 0.0
        self._current_drag: float = 0.0
        self._front_downforce: float = 0.0
        self._rear_downforce: float = 0.0
    
    @property
    def drs_active(self) -> bool:
        """Check if DRS is active."""
        return self._drs_active
    
    @drs_active.setter
    def drs_active(self, value: bool) -> None:
        """Activate or deactivate DRS."""
        self._drs_active = value
    
    @property
    def current_downforce(self) -> float:
        """Total downforce in Newtons."""
        return self._current_downforce
    
    @property
    def current_drag(self) -> float:
        """Current drag force in Newtons."""
        return self._current_drag
    
    @property
    def front_downforce(self) -> float:
        """Front axle downforce in Newtons."""
        return self._front_downforce
    
    @property
    def rear_downforce(self) -> float:
        """Rear axle downforce in Newtons."""
        return self._rear_downforce
    
    def _get_dynamic_pressure(self, speed: float) -> float:
        """Calculate dynamic pressure.
        
        Args:
            speed: Vehicle speed in m/s
            
        Returns:
            Dynamic pressure in Pa
        """
        return 0.5 * self.config.air_density_kg_m3 * speed * speed
    
    def calculate_drag(self, speed: float) -> float:
        """Calculate drag force at given speed.
        
        Args:
            speed: Vehicle speed in m/s (absolute value used)
            
        Returns:
            Drag force in Newtons (always positive)
        """
        speed = abs(speed)
        q = self._get_dynamic_pressure(speed)
        
        cd = self.config.drag_coefficient
        if self._drs_active:
            cd *= (1.0 - self.config.drs_drag_reduction)
        
        drag = q * cd * self.config.frontal_area_m2
        return drag
    
    def calculate_downforce(self, speed: float) -> float:
        """Calculate total downforce at given speed.
        
        Args:
            speed: Vehicle speed in m/s (absolute value used)
            
        Returns:
            Downforce in Newtons (positive = pushing down)
        """
        speed = abs(speed)
        q = self._get_dynamic_pressure(speed)
        
        # Lift coefficient is negative for downforce, we negate to get positive value
        cl = -self.config.lift_coefficient
        if self._drs_active:
            cl *= (1.0 - self.config.drs_downforce_reduction)
        
        downforce = q * cl * self.config.wing_area_m2
        return downforce
    
    def update(self, speed: float) -> tuple[float, float]:
        """Update aerodynamic forces based on current speed.
        
        Args:
            speed: Vehicle speed in m/s
            
        Returns:
            Tuple of (drag_force, downforce) in Newtons
        """
        self._current_drag = self.calculate_drag(speed)
        self._current_downforce = self.calculate_downforce(speed)
        
        # Calculate front/rear distribution
        self._rear_downforce = self._current_downforce * self.config.aero_balance
        self._front_downforce = self._current_downforce * (1.0 - self.config.aero_balance)
        
        return self._current_drag, self._current_downforce
    
    def get_aero_balance_percent(self) -> float:
        """Get current aero balance as rear percentage.
        
        Returns:
            Rear downforce percentage (0-100)
        """
        return self.config.aero_balance * 100.0
    
    def reset(self) -> None:
        """Reset aerodynamics to initial state."""
        self._drs_active = False
        self._current_downforce = 0.0
        self._current_drag = 0.0
        self._front_downforce = 0.0
        self._rear_downforce = 0.0
    
    def get_state(self) -> dict:
        """Get current aerodynamic state for telemetry.
        
        Returns:
            Dictionary containing aero state values
        """
        return {
            "drag_n": self._current_drag,
            "downforce_n": self._current_downforce,
            "front_downforce_n": self._front_downforce,
            "rear_downforce_n": self._rear_downforce,
            "drs_active": self._drs_active,
            "aero_balance_pct": self.get_aero_balance_percent(),
        }
