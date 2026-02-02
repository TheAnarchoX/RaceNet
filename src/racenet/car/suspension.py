"""
Suspension component - Weight transfer and chassis dynamics.

Simulates:
- Weight transfer during acceleration, braking, cornering
- Spring and damper characteristics (simplified)
- Roll and pitch dynamics
- Ride height effects
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SuspensionConfig:
    """Configuration for GT3-style suspension.
    
    Default values based on typical GT3 racing suspension setup.
    """
    # Spring rates (N/m) - stiff for racing
    front_spring_rate: float = 180000.0  # 180 N/mm
    rear_spring_rate: float = 200000.0   # 200 N/mm
    
    # Damping ratios (0-1, critical damping = 1.0)
    front_damping_ratio: float = 0.7
    rear_damping_ratio: float = 0.7
    
    # Anti-roll bar rates (Nm/deg)
    front_arb_rate: float = 1500.0
    rear_arb_rate: float = 1200.0
    
    # Suspension geometry
    front_roll_center_height_m: float = 0.05
    rear_roll_center_height_m: float = 0.10
    
    # Track width (m)
    front_track_m: float = 1.65
    rear_track_m: float = 1.60
    
    # Wheelbase (m)
    wheelbase_m: float = 2.50
    
    # Static ride heights (m)
    front_ride_height_m: float = 0.060
    rear_ride_height_m: float = 0.080
    
    # Maximum suspension travel (m)
    max_compression_m: float = 0.050
    max_extension_m: float = 0.050


class Suspension:
    """GT3-style suspension simulation.
    
    Calculates:
    - Weight transfer to each wheel
    - Body roll and pitch angles
    - Vertical tire loads
    """
    
    def __init__(self, config: SuspensionConfig | None = None):
        """Initialize suspension with optional custom configuration.
        
        Args:
            config: Suspension configuration. Uses GT3 defaults if None.
        """
        self.config = config or SuspensionConfig()
        
        # State - wheel loads as fraction of total weight (0.25 each at rest)
        self._wheel_loads = np.array([0.25, 0.25, 0.25, 0.25])  # FL, FR, RL, RR
        
        # Body motion
        self._roll_angle_deg: float = 0.0
        self._pitch_angle_deg: float = 0.0
        
        # Ride heights at each corner
        self._ride_heights = np.array([
            self.config.front_ride_height_m,
            self.config.front_ride_height_m,
            self.config.rear_ride_height_m,
            self.config.rear_ride_height_m,
        ])
    
    @property
    def wheel_loads(self) -> np.ndarray:
        """Wheel load fractions [FL, FR, RL, RR]."""
        return self._wheel_loads.copy()
    
    @property
    def roll_angle(self) -> float:
        """Body roll angle in degrees (positive = right)."""
        return self._roll_angle_deg
    
    @property
    def pitch_angle(self) -> float:
        """Body pitch angle in degrees (positive = nose up)."""
        return self._pitch_angle_deg
    
    def calculate_weight_transfer(
        self,
        longitudinal_g: float,
        lateral_g: float,
        total_mass_kg: float,
        cog_height_m: float,
        aero_front_load_n: float = 0.0,
        aero_rear_load_n: float = 0.0,
    ) -> np.ndarray:
        """Calculate weight distribution to each wheel.
        
        Args:
            longitudinal_g: Longitudinal acceleration in g's (positive = accel)
            lateral_g: Lateral acceleration in g's (positive = right turn)
            total_mass_kg: Total vehicle mass in kg
            cog_height_m: Center of gravity height in meters
            aero_front_load_n: Additional front axle load from aero
            aero_rear_load_n: Additional rear axle load from aero
            
        Returns:
            Array of wheel load fractions [FL, FR, RL, RR]
        """
        # Base static weight distribution (assuming 47% front for GT3)
        static_front = 0.47
        static_rear = 0.53
        
        # Calculate longitudinal weight transfer
        # Forward accel transfers weight to rear, braking transfers to front
        long_transfer = (longitudinal_g * total_mass_kg * cog_height_m) / self.config.wheelbase_m
        long_transfer_fraction = long_transfer / (total_mass_kg * 9.81)
        
        # Calculate lateral weight transfer
        front_lat_transfer = (lateral_g * total_mass_kg * cog_height_m * static_front) / self.config.front_track_m
        rear_lat_transfer = (lateral_g * total_mass_kg * cog_height_m * static_rear) / self.config.rear_track_m
        
        front_lat_fraction = front_lat_transfer / (total_mass_kg * 9.81)
        rear_lat_fraction = rear_lat_transfer / (total_mass_kg * 9.81)
        
        # Calculate new wheel loads
        front_total = static_front - long_transfer_fraction
        rear_total = static_rear + long_transfer_fraction
        
        # Add aero loads
        total_weight = total_mass_kg * 9.81
        aero_front_fraction = aero_front_load_n / total_weight if total_weight > 0 else 0
        aero_rear_fraction = aero_rear_load_n / total_weight if total_weight > 0 else 0
        
        front_total += aero_front_fraction
        rear_total += aero_rear_fraction
        
        # Distribute to individual wheels
        fl = (front_total / 2) - front_lat_fraction
        fr = (front_total / 2) + front_lat_fraction
        rl = (rear_total / 2) - rear_lat_fraction
        rr = (rear_total / 2) + rear_lat_fraction
        
        # Clamp to prevent negative loads (wheel lift)
        wheel_loads = np.clip([fl, fr, rl, rr], 0.0, 1.0)
        
        # Normalize to maintain total load
        total_load = np.sum(wheel_loads)
        if total_load > 0:
            wheel_loads /= total_load
        
        return wheel_loads
    
    def calculate_body_roll(
        self,
        lateral_g: float,
        total_mass_kg: float,
        cog_height_m: float,
    ) -> float:
        """Calculate body roll angle.
        
        Args:
            lateral_g: Lateral acceleration in g's
            total_mass_kg: Total vehicle mass
            cog_height_m: Center of gravity height
            
        Returns:
            Roll angle in degrees
        """
        # Simplified roll calculation based on ARB stiffness
        roll_moment = lateral_g * total_mass_kg * 9.81 * cog_height_m
        
        # Combined roll resistance
        avg_arb = (self.config.front_arb_rate + self.config.rear_arb_rate) / 2
        roll_stiffness = avg_arb * 2  # Simplified
        
        if roll_stiffness > 0:
            roll_angle = np.degrees(roll_moment / roll_stiffness)
        else:
            roll_angle = 0.0
        
        # Limit to realistic values
        return np.clip(roll_angle, -5.0, 5.0)
    
    def calculate_body_pitch(
        self,
        longitudinal_g: float,
        total_mass_kg: float,
        cog_height_m: float,
    ) -> float:
        """Calculate body pitch angle.
        
        Args:
            longitudinal_g: Longitudinal acceleration in g's
            total_mass_kg: Total vehicle mass
            cog_height_m: Center of gravity height
            
        Returns:
            Pitch angle in degrees (positive = nose up)
        """
        # Calculate pitch moment
        pitch_moment = longitudinal_g * total_mass_kg * 9.81 * cog_height_m
        
        # Combined pitch resistance from springs
        avg_spring = (self.config.front_spring_rate + self.config.rear_spring_rate) / 2
        pitch_stiffness = avg_spring * self.config.wheelbase_m  # Simplified
        
        if pitch_stiffness > 0:
            pitch_angle = np.degrees(pitch_moment / pitch_stiffness)
        else:
            pitch_angle = 0.0
        
        # Limit to realistic values
        return np.clip(pitch_angle, -3.0, 3.0)
    
    def update(
        self,
        longitudinal_g: float,
        lateral_g: float,
        total_mass_kg: float,
        cog_height_m: float,
        aero_front_load_n: float = 0.0,
        aero_rear_load_n: float = 0.0,
    ) -> None:
        """Update suspension state.
        
        Args:
            longitudinal_g: Longitudinal acceleration in g's
            lateral_g: Lateral acceleration in g's
            total_mass_kg: Total vehicle mass
            cog_height_m: Center of gravity height
            aero_front_load_n: Front aero downforce
            aero_rear_load_n: Rear aero downforce
        """
        # Update wheel loads
        self._wheel_loads = self.calculate_weight_transfer(
            longitudinal_g,
            lateral_g,
            total_mass_kg,
            cog_height_m,
            aero_front_load_n,
            aero_rear_load_n,
        )
        
        # Update body angles
        self._roll_angle_deg = self.calculate_body_roll(
            lateral_g, total_mass_kg, cog_height_m
        )
        self._pitch_angle_deg = self.calculate_body_pitch(
            longitudinal_g, total_mass_kg, cog_height_m
        )
    
    def get_wheel_load_newtons(self, total_mass_kg: float) -> np.ndarray:
        """Get wheel loads in Newtons.
        
        Args:
            total_mass_kg: Total vehicle mass
            
        Returns:
            Array of wheel loads in N [FL, FR, RL, RR]
        """
        total_weight = total_mass_kg * 9.81
        return self._wheel_loads * total_weight
    
    def reset(self) -> None:
        """Reset suspension to initial state."""
        self._wheel_loads = np.array([0.25, 0.25, 0.25, 0.25])
        self._roll_angle_deg = 0.0
        self._pitch_angle_deg = 0.0
        self._ride_heights = np.array([
            self.config.front_ride_height_m,
            self.config.front_ride_height_m,
            self.config.rear_ride_height_m,
            self.config.rear_ride_height_m,
        ])
    
    def get_state(self) -> dict:
        """Get current suspension state for telemetry.
        
        Returns:
            Dictionary containing suspension state values
        """
        return {
            "wheel_load_fl": self._wheel_loads[0],
            "wheel_load_fr": self._wheel_loads[1],
            "wheel_load_rl": self._wheel_loads[2],
            "wheel_load_rr": self._wheel_loads[3],
            "roll_angle_deg": self._roll_angle_deg,
            "pitch_angle_deg": self._pitch_angle_deg,
        }
