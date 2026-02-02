"""
Electronics component - Driver aids and control systems.

Simulates:
- Traction Control (TC) system
- Anti-lock Braking System (ABS)
- Engine maps
- Brake bias adjustment
"""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class TCLevel(IntEnum):
    """Traction control intervention levels."""
    OFF = 0
    TC1 = 1  # Minimal intervention
    TC2 = 2
    TC3 = 3
    TC4 = 4
    TC5 = 5  # Maximum intervention


class ABSLevel(IntEnum):
    """ABS intervention levels."""
    OFF = 0
    ABS1 = 1  # Minimal intervention
    ABS2 = 2
    ABS3 = 3
    ABS4 = 4
    ABS5 = 5  # Maximum intervention


@dataclass
class TractionControlConfig:
    """Configuration for traction control system."""
    # Slip threshold where TC starts to intervene
    slip_threshold_base: float = 0.08  # 8% slip at TC1
    slip_threshold_step: float = 0.02  # Each level reduces by 2%
    
    # Intervention response rate
    response_rate: float = 50.0  # How quickly TC reacts (Hz)
    
    # Maximum throttle reduction
    max_throttle_cut: float = 0.8  # Can cut up to 80% throttle


class TractionControl:
    """Traction Control system simulation.
    
    Monitors wheel spin and reduces throttle to maintain grip.
    """
    
    def __init__(self, config: TractionControlConfig | None = None):
        """Initialize TC with optional custom configuration.
        
        Args:
            config: TC configuration. Uses defaults if None.
        """
        self.config = config or TractionControlConfig()
        
        self._level: int = int(TCLevel.TC3)  # Default to medium TC
        self._active: bool = False
        self._throttle_cut: float = 0.0
    
    @property
    def level(self) -> int:
        """Current TC level."""
        return self._level
    
    @level.setter
    def level(self, value: int) -> None:
        """Set TC level."""
        self._level = np.clip(value, int(TCLevel.OFF), int(TCLevel.TC5))
    
    @property
    def is_active(self) -> bool:
        """Check if TC is currently intervening."""
        return self._active
    
    @property
    def throttle_cut(self) -> float:
        """Current throttle cut amount (0-1)."""
        return self._throttle_cut
    
    def get_slip_threshold(self) -> float:
        """Get slip threshold for current TC level.
        
        Returns:
            Slip ratio threshold
        """
        if self._level == int(TCLevel.OFF):
            return float('inf')  # Never intervene
        
        return self.config.slip_threshold_base - (
            (self._level - 1) * self.config.slip_threshold_step
        )
    
    def process(
        self,
        throttle_input: float,
        wheel_slip_rear: float,
        dt: float,
    ) -> float:
        """Process throttle through TC system.
        
        Args:
            throttle_input: Driver throttle input (0-1)
            wheel_slip_rear: Rear wheel slip ratio
            dt: Time step in seconds
            
        Returns:
            Modified throttle output (0-1)
        """
        if self._level == int(TCLevel.OFF):
            self._active = False
            self._throttle_cut = 0.0
            return throttle_input
        
        threshold = self.get_slip_threshold()
        
        if wheel_slip_rear > threshold:
            self._active = True
            
            # Calculate required throttle cut
            excess_slip = wheel_slip_rear - threshold
            target_cut = min(
                excess_slip * 10.0,  # Proportional response
                self.config.max_throttle_cut
            )
            
            # Smooth response
            rate = self.config.response_rate * dt
            self._throttle_cut += (target_cut - self._throttle_cut) * min(rate, 1.0)
        else:
            # Gradually release TC
            self._throttle_cut *= (1.0 - self.config.response_rate * dt * 0.5)
            if self._throttle_cut < 0.01:
                self._active = False
                self._throttle_cut = 0.0
        
        return throttle_input * (1.0 - self._throttle_cut)
    
    def reset(self) -> None:
        """Reset TC state."""
        self._active = False
        self._throttle_cut = 0.0


@dataclass
class ABSConfig:
    """Configuration for ABS system."""
    # Slip threshold where ABS starts to intervene
    slip_threshold_base: float = 0.12  # 12% slip at ABS1
    slip_threshold_step: float = 0.02  # Each level reduces by 2%
    
    # Modulation frequency
    modulation_freq_hz: float = 15.0  # ABS pump frequency
    
    # Maximum brake reduction
    max_brake_cut: float = 0.6  # Can reduce braking by 60%


class ABS:
    """Anti-lock Braking System simulation.
    
    Monitors wheel lock-up and modulates brake pressure.
    """
    
    def __init__(self, config: ABSConfig | None = None):
        """Initialize ABS with optional custom configuration.
        
        Args:
            config: ABS configuration. Uses defaults if None.
        """
        self.config = config or ABSConfig()
        
        self._level: int = int(ABSLevel.ABS3)  # Default to medium ABS
        self._active: bool = False
        self._brake_modulation: float = 1.0  # Multiplier for brake force
        self._modulation_phase: float = 0.0
    
    @property
    def level(self) -> int:
        """Current ABS level."""
        return self._level
    
    @level.setter
    def level(self, value: int) -> None:
        """Set ABS level."""
        self._level = np.clip(value, int(ABSLevel.OFF), int(ABSLevel.ABS5))
    
    @property
    def is_active(self) -> bool:
        """Check if ABS is currently intervening."""
        return self._active
    
    @property
    def brake_modulation(self) -> float:
        """Current brake modulation factor (0-1)."""
        return self._brake_modulation
    
    def get_slip_threshold(self) -> float:
        """Get slip threshold for current ABS level.
        
        Returns:
            Slip ratio threshold
        """
        if self._level == int(ABSLevel.OFF):
            return float('inf')
        
        return self.config.slip_threshold_base - (
            (self._level - 1) * self.config.slip_threshold_step
        )
    
    def process(
        self,
        brake_input: float,
        wheel_slip: float,  # Maximum slip across all wheels
        dt: float,
    ) -> float:
        """Process brake input through ABS system.
        
        Args:
            brake_input: Driver brake input (0-1)
            wheel_slip: Maximum wheel slip ratio (negative = locked)
            dt: Time step in seconds
            
        Returns:
            Modified brake output (0-1)
        """
        if self._level == int(ABSLevel.OFF):
            self._active = False
            self._brake_modulation = 1.0
            return brake_input
        
        threshold = self.get_slip_threshold()
        
        # For braking, we look at negative slip (wheel slower than ground)
        effective_slip = abs(wheel_slip)
        
        if effective_slip > threshold:
            self._active = True
            
            # Update modulation phase
            self._modulation_phase += 2 * np.pi * self.config.modulation_freq_hz * dt
            if self._modulation_phase > 2 * np.pi:
                self._modulation_phase -= 2 * np.pi
            
            # Calculate modulation based on slip excess
            excess_slip = effective_slip - threshold
            base_reduction = min(excess_slip * 5.0, self.config.max_brake_cut)
            
            # Add oscillation for pump effect
            oscillation = 0.5 + 0.5 * np.sin(self._modulation_phase)
            
            self._brake_modulation = 1.0 - base_reduction * oscillation
        else:
            self._active = False
            # Smoothly return to full braking
            self._brake_modulation += (1.0 - self._brake_modulation) * 5.0 * dt
        
        return brake_input * np.clip(self._brake_modulation, 0.2, 1.0)
    
    def reset(self) -> None:
        """Reset ABS state."""
        self._active = False
        self._brake_modulation = 1.0
        self._modulation_phase = 0.0


@dataclass
class ElectronicsConfig:
    """Configuration for all electronic systems."""
    tc_config: TractionControlConfig | None = None
    abs_config: ABSConfig | None = None
    
    # Brake bias (0.0 = all front, 1.0 = all rear)
    default_brake_bias: float = 0.55  # 55% front typical GT3
    
    # Engine map (1-10, affects throttle response)
    default_engine_map: int = 5


class Electronics:
    """Combined electronics systems for GT3 car.
    
    Manages:
    - Traction Control
    - ABS
    - Brake bias
    - Engine maps
    """
    
    def __init__(self, config: ElectronicsConfig | None = None):
        """Initialize electronics with optional configuration.
        
        Args:
            config: Electronics configuration. Uses defaults if None.
        """
        self.config = config or ElectronicsConfig()
        
        self.tc = TractionControl(self.config.tc_config)
        self.abs = ABS(self.config.abs_config)
        
        self._brake_bias: float = self.config.default_brake_bias
        self._engine_map: int = self.config.default_engine_map
    
    @property
    def brake_bias(self) -> float:
        """Current brake bias (front fraction)."""
        return self._brake_bias
    
    @brake_bias.setter
    def brake_bias(self, value: float) -> None:
        """Set brake bias."""
        self._brake_bias = np.clip(value, 0.4, 0.7)  # Realistic range
    
    @property
    def engine_map(self) -> int:
        """Current engine map setting."""
        return self._engine_map
    
    @engine_map.setter
    def engine_map(self, value: int) -> None:
        """Set engine map."""
        self._engine_map = np.clip(value, 1, 10)
    
    def adjust_brake_bias(self, delta: float) -> None:
        """Adjust brake bias by given amount.
        
        Args:
            delta: Amount to change brake bias (-0.1 to 0.1 typical)
        """
        self.brake_bias = self._brake_bias + delta
    
    def get_brake_distribution(self, total_brake: float) -> tuple[float, float]:
        """Get front and rear brake force distribution.
        
        Args:
            total_brake: Total brake input (0-1)
            
        Returns:
            Tuple of (front_brake, rear_brake) each 0-1
        """
        front = total_brake * self._brake_bias
        rear = total_brake * (1.0 - self._brake_bias)
        return front, rear
    
    def get_throttle_map_factor(self) -> float:
        """Get throttle response factor from engine map.
        
        Returns:
            Throttle curve factor (higher = more aggressive)
        """
        # Map 1 = very smooth, Map 10 = very aggressive
        return 0.5 + (self._engine_map / 10.0) * 0.5
    
    def apply_throttle_map(self, throttle_input: float) -> float:
        """Apply engine map to throttle input.
        
        Args:
            throttle_input: Raw throttle input (0-1)
            
        Returns:
            Mapped throttle output (0-1)
        """
        factor = self.get_throttle_map_factor()
        # Power curve for throttle response
        return np.power(throttle_input, 2.0 - factor)
    
    def process_inputs(
        self,
        throttle: float,
        brake: float,
        rear_wheel_slip: float,
        max_wheel_slip: float,
        dt: float,
    ) -> tuple[float, float, float]:
        """Process all driver inputs through electronics.
        
        Args:
            throttle: Raw throttle input (0-1)
            brake: Raw brake input (0-1)
            rear_wheel_slip: Rear wheel slip for TC
            max_wheel_slip: Max wheel slip for ABS
            dt: Time step in seconds
            
        Returns:
            Tuple of (processed_throttle, front_brake, rear_brake)
        """
        # Apply engine map
        mapped_throttle = self.apply_throttle_map(throttle)
        
        # Apply TC
        tc_throttle = self.tc.process(mapped_throttle, rear_wheel_slip, dt)
        
        # Apply ABS
        abs_brake = self.abs.process(brake, max_wheel_slip, dt)
        
        # Apply brake bias
        front_brake, rear_brake = self.get_brake_distribution(abs_brake)
        
        return tc_throttle, front_brake, rear_brake
    
    def reset(self) -> None:
        """Reset all electronics to initial state."""
        self.tc.reset()
        self.abs.reset()
        self._brake_bias = self.config.default_brake_bias
        self._engine_map = self.config.default_engine_map
    
    def get_state(self) -> dict:
        """Get current electronics state for telemetry.
        
        Returns:
            Dictionary containing electronics state values
        """
        return {
            "tc_level": self.tc.level,
            "tc_active": self.tc.is_active,
            "tc_throttle_cut": self.tc.throttle_cut,
            "abs_level": self.abs.level,
            "abs_active": self.abs.is_active,
            "abs_modulation": self.abs.brake_modulation,
            "brake_bias": self._brake_bias,
            "engine_map": self._engine_map,
        }
