"""
Car - Complete GT3-style racing car simulation.

Integrates all car components:
- Engine
- Transmission
- Aerodynamics
- Suspension
- Chassis
- Electronics (TC, ABS)
- Tires
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from racenet.car.engine import Engine, EngineConfig
from racenet.car.transmission import Transmission, TransmissionConfig, GearState
from racenet.car.aero import Aero, AeroConfig
from racenet.car.suspension import Suspension, SuspensionConfig
from racenet.car.chassis import Chassis, ChassisConfig
from racenet.car.electronics import Electronics, ElectronicsConfig
from racenet.car.tires import TireSet, TireSetConfig


@dataclass
class CarConfig:
    """Complete car configuration.
    
    Default values create a typical GT3 992-style car.
    """
    engine: EngineConfig | None = None
    transmission: TransmissionConfig | None = None
    aero: AeroConfig | None = None
    suspension: SuspensionConfig | None = None
    chassis: ChassisConfig | None = None
    electronics: ElectronicsConfig | None = None
    tires: TireSetConfig | None = None
    
    # Initial fuel load (kg)
    initial_fuel_kg: float = 100.0


@dataclass
class CarInputs:
    """Driver control inputs for the car."""
    throttle: float = 0.0      # 0.0 to 1.0
    brake: float = 0.0         # 0.0 to 1.0
    steering: float = 0.0      # -1.0 (left) to 1.0 (right)
    shift_up: bool = False     # Request upshift
    shift_down: bool = False   # Request downshift


@dataclass
class CarState:
    """Current car state for physics integration."""
    # Position (world coordinates, meters)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0  # Height/elevation
    
    # Orientation (radians)
    heading: float = 0.0       # Yaw angle (0 = +X direction)
    pitch: float = 0.0
    roll: float = 0.0
    
    # Velocities (m/s)
    velocity_x: float = 0.0    # World frame
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    
    # Angular velocities (rad/s)
    yaw_rate: float = 0.0
    
    # Derived values (updated during simulation)
    speed: float = 0.0         # Scalar speed
    lateral_g: float = 0.0     # Lateral acceleration in g
    longitudinal_g: float = 0.0  # Longitudinal acceleration in g


class Car:
    """Complete GT3-style racing car simulation.
    
    Integrates all subsystems to provide a realistic car simulation.
    The car processes driver inputs and updates its physics state.
    
    Features:
    - Realistic engine and transmission
    - Aerodynamic forces
    - Weight transfer through suspension
    - Tire grip model
    - TC and ABS driver aids
    - Full telemetry output
    
    Usage:
        car = Car()
        inputs = CarInputs(throttle=0.5, steering=0.1)
        car.step(inputs, dt=0.01)
        telemetry = car.get_telemetry()
    """
    
    def __init__(
        self,
        config: CarConfig | None = None,
        car_id: int = 0,
    ):
        """Initialize car with optional configuration.
        
        Args:
            config: Car configuration. Uses GT3 defaults if None.
            car_id: Unique identifier for this car instance
        """
        self.config = config or CarConfig()
        self.car_id = car_id
        
        # Initialize subsystems
        self.engine = Engine(self.config.engine)
        self.transmission = Transmission(self.config.transmission)
        self.aero = Aero(self.config.aero)
        self.suspension = Suspension(self.config.suspension)
        self.chassis = Chassis(self.config.chassis)
        self.electronics = Electronics(self.config.electronics)
        self.tires = TireSet(self.config.tires)
        
        # Car state
        self.state = CarState()
        
        # Initialize car
        self.reset()
    
    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading: float = 0.0,
    ) -> None:
        """Reset car to initial state at given position.
        
        Args:
            x: Starting X position
            y: Starting Y position
            heading: Starting heading in radians
        """
        # Reset all subsystems
        self.engine.reset()
        self.transmission.reset()
        self.aero.reset()
        self.suspension.reset()
        self.chassis.reset(self.config.initial_fuel_kg)
        self.electronics.reset()
        self.tires.reset()
        
        # Put car in first gear
        self.transmission.set_gear(int(GearState.FIRST))
        
        # Reset state
        self.state = CarState(x=x, y=y, heading=heading)
    
    @property
    def speed(self) -> float:
        """Current speed in m/s."""
        return self.state.speed
    
    @property
    def speed_kph(self) -> float:
        """Current speed in km/h."""
        return self.state.speed * 3.6
    
    @property
    def position(self) -> tuple[float, float]:
        """Current (x, y) position."""
        return (self.state.x, self.state.y)
    
    @property
    def heading(self) -> float:
        """Current heading in radians."""
        return self.state.heading
    
    def _calculate_wheel_speeds(self) -> tuple[float, float, float, float]:
        """Calculate wheel rotational speeds.
        
        Returns:
            Tuple of wheel speeds (rad/s) [FL, FR, RL, RR]
        """
        # Get speed along car's direction
        speed = self.state.speed
        
        # Calculate yaw rate effect on wheel speeds
        yaw_rate = self.state.yaw_rate
        track_f = self.suspension.config.front_track_m
        track_r = self.suspension.config.rear_track_m
        wheelbase = self.suspension.config.wheelbase_m
        
        # Speed difference due to turning
        # Front wheels have different path lengths
        fl_speed = speed - yaw_rate * track_f / 2
        fr_speed = speed + yaw_rate * track_f / 2
        rl_speed = speed - yaw_rate * track_r / 2
        rr_speed = speed + yaw_rate * track_r / 2
        
        # Convert to angular velocity (rad/s)
        radius = self.tires.fl.radius
        return (
            fl_speed / radius,
            fr_speed / radius,
            rl_speed / radius,
            rr_speed / radius,
        )
    
    def _calculate_slip_ratios(
        self,
        wheel_speeds: tuple[float, float, float, float],
        drive_torque: float,
        brake_front: float,
        brake_rear: float,
    ) -> tuple[float, float, float, float]:
        """Calculate tire slip ratios.
        
        Args:
            wheel_speeds: Wheel angular speeds (rad/s)
            drive_torque: Torque at driven wheels
            brake_front: Front brake input (0-1)
            brake_rear: Rear brake input (0-1)
            
        Returns:
            Tuple of slip ratios [FL, FR, RL, RR]
        """
        ground_speed = max(self.state.speed, 0.5)  # Avoid division by zero
        radius = self.tires.fl.radius
        
        # Simplified slip calculation - placeholder pending proper tire physics
        # See TASKS.md Task 1.1 for full Pacejka implementation plan
        SIMPLIFIED_BRAKE_SLIP_FACTOR = 0.3  # Approximate slip from braking
        
        # For rear-wheel drive GT3
        # Rear wheels get drive torque, all wheels get braking
        fl_slip = -brake_front * SIMPLIFIED_BRAKE_SLIP_FACTOR
        fr_slip = -brake_front * SIMPLIFIED_BRAKE_SLIP_FACTOR
        
        # Rear has drive + brake
        rl_driven_slip = drive_torque / (self.chassis.total_mass * 9.81 * 0.5) * 0.5
        rr_driven_slip = drive_torque / (self.chassis.total_mass * 9.81 * 0.5) * 0.5
        
        rl_slip = rl_driven_slip - brake_rear * SIMPLIFIED_BRAKE_SLIP_FACTOR
        rr_slip = rr_driven_slip - brake_rear * SIMPLIFIED_BRAKE_SLIP_FACTOR
        
        return (
            np.clip(fl_slip, -0.5, 0.5),
            np.clip(fr_slip, -0.5, 0.5),
            np.clip(rl_slip, -0.5, 0.5),
            np.clip(rr_slip, -0.5, 0.5),
        )
    
    def _calculate_slip_angles(self, steering: float) -> tuple[float, float, float, float]:
        """Calculate tire slip angles.
        
        Args:
            steering: Steering input (-1 to 1)
            
        Returns:
            Tuple of slip angles in degrees [FL, FR, RL, RR]
        """
        # Max steering angle (degrees) 
        max_steer = 25.0
        steer_angle = steering * max_steer
        
        # Calculate slip angles from steering and vehicle motion
        speed = max(self.state.speed, 0.5)
        yaw_rate = self.state.yaw_rate
        wheelbase = self.suspension.config.wheelbase_m
        
        # Simplified bicycle model slip angles
        # Front: steering angle - atan(yaw_rate * front_dist / speed)
        # Rear: -atan(yaw_rate * rear_dist / speed)
        
        front_dist = wheelbase * (1 - self.chassis.front_weight_fraction)
        rear_dist = wheelbase * self.chassis.front_weight_fraction
        
        alpha_f = steer_angle - np.degrees(np.arctan(yaw_rate * front_dist / speed))
        alpha_r = -np.degrees(np.arctan(yaw_rate * rear_dist / speed))
        
        # Small difference between left/right due to yaw
        yaw_effect = np.degrees(yaw_rate * 0.1)
        
        return (
            np.clip(alpha_f - yaw_effect, -20, 20),
            np.clip(alpha_f + yaw_effect, -20, 20),
            np.clip(alpha_r - yaw_effect, -20, 20),
            np.clip(alpha_r + yaw_effect, -20, 20),
        )
    
    def step(self, inputs: CarInputs, dt: float) -> CarState:
        """Advance car simulation by one time step.
        
        Args:
            inputs: Driver control inputs
            dt: Time step in seconds
            
        Returns:
            Updated car state
        """
        # Clamp inputs
        throttle = np.clip(inputs.throttle, 0.0, 1.0)
        brake = np.clip(inputs.brake, 0.0, 1.0)
        steering = np.clip(inputs.steering, -1.0, 1.0)
        
        # Process shifting
        if inputs.shift_up:
            self.transmission.shift_up()
        if inputs.shift_down:
            self.transmission.shift_down()
        
        # Update transmission
        shift_info = self.transmission.update(dt)
        
        # Process inputs through electronics (TC, ABS)
        rear_slip = self.tires.get_rear_slip_ratio()
        max_slip = self.tires.get_max_slip_ratio()
        
        processed_throttle, brake_front, brake_rear = self.electronics.process_inputs(
            throttle, brake, rear_slip, max_slip, dt
        )
        
        # Set engine throttle
        self.engine.throttle = processed_throttle
        
        # Handle rev matching during downshift
        if shift_info["rev_match_request"] > 0:
            self.engine.throttle = max(self.engine.throttle, shift_info["rev_match_request"])
        
        # Calculate wheel RPM for engine update
        wheel_speed_rad = self.state.speed / self.tires.rl.radius
        wheel_rpm = wheel_speed_rad * 60 / (2 * np.pi)
        
        # Update engine
        gear_ratio = self.transmission.get_total_ratio()
        engine_torque = self.engine.update(dt, wheel_rpm, gear_ratio)
        
        # Get drive torque at wheels
        drive_torque = self.transmission.get_drive_torque(engine_torque)
        
        # Update aerodynamics
        drag, downforce = self.aero.update(self.state.speed)
        
        # Update suspension with weight transfer
        self.suspension.update(
            self.state.longitudinal_g,
            self.state.lateral_g,
            self.chassis.total_mass,
            self.chassis.cog_height,
            self.aero.front_downforce,
            self.aero.rear_downforce,
        )
        
        # Get wheel loads
        wheel_loads = self.suspension.get_wheel_load_newtons(self.chassis.total_mass)
        
        # Calculate slip conditions
        wheel_speeds = self._calculate_wheel_speeds()
        slip_ratios = self._calculate_slip_ratios(
            wheel_speeds, drive_torque, brake_front, brake_rear
        )
        slip_angles = self._calculate_slip_angles(steering)
        
        # Update tires and get forces
        tire_forces_x = []
        tire_forces_y = []
        
        for i, tire in enumerate(self.tires.tires):
            fx, fy = tire.update(
                wheel_loads[i],
                slip_ratios[i],
                slip_angles[i],
                self.state.speed,
                dt,
            )
            tire_forces_x.append(fx)
            tire_forces_y.append(fy)
        
        # Calculate total forces in car frame
        total_fx = sum(tire_forces_x) - drag  # Subtract drag
        total_fy = sum(tire_forces_y)
        
        # Calculate accelerations
        mass = self.chassis.total_mass
        accel_x = total_fx / mass  # Forward acceleration
        accel_y = total_fy / mass  # Lateral acceleration
        
        # Store g-forces for next frame
        self.state.longitudinal_g = accel_x / 9.81
        self.state.lateral_g = accel_y / 9.81
        
        # Calculate yaw moment from tire forces
        wheelbase = self.suspension.config.wheelbase_m
        front_track = self.suspension.config.front_track_m
        rear_track = self.suspension.config.rear_track_m
        
        # Yaw moment from front and rear lateral forces
        front_moment = (tire_forces_y[0] + tire_forces_y[1]) * self.chassis.cog_from_front
        rear_moment = -(tire_forces_y[2] + tire_forces_y[3]) * (wheelbase - self.chassis.cog_from_front)
        
        # Additional moment from left/right force difference
        front_diff_moment = (tire_forces_y[1] - tire_forces_y[0]) * front_track / 2
        rear_diff_moment = (tire_forces_y[3] - tire_forces_y[2]) * rear_track / 2
        
        total_yaw_moment = front_moment + rear_moment + front_diff_moment + rear_diff_moment
        
        # Angular acceleration
        yaw_accel = total_yaw_moment / self.chassis.yaw_inertia
        
        # Update yaw rate with damping
        self.state.yaw_rate += yaw_accel * dt
        self.state.yaw_rate *= 0.98  # Damping
        
        # Update heading
        self.state.heading += self.state.yaw_rate * dt
        
        # Keep heading in [-pi, pi]
        while self.state.heading > np.pi:
            self.state.heading -= 2 * np.pi
        while self.state.heading < -np.pi:
            self.state.heading += 2 * np.pi
        
        # Convert car-frame acceleration to world frame
        cos_h = np.cos(self.state.heading)
        sin_h = np.sin(self.state.heading)
        
        world_accel_x = accel_x * cos_h - accel_y * sin_h
        world_accel_y = accel_x * sin_h + accel_y * cos_h
        
        # Update velocities
        self.state.velocity_x += world_accel_x * dt
        self.state.velocity_y += world_accel_y * dt
        
        # Update speed
        self.state.speed = np.sqrt(
            self.state.velocity_x**2 + self.state.velocity_y**2
        )
        
        # Update position
        self.state.x += self.state.velocity_x * dt
        self.state.y += self.state.velocity_y * dt
        
        return self.state
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get complete car telemetry.
        
        Returns:
            Dictionary containing all car telemetry data
        """
        return {
            "car_id": self.car_id,
            "state": {
                "x": self.state.x,
                "y": self.state.y,
                "z": self.state.z,
                "heading_rad": self.state.heading,
                "heading_deg": np.degrees(self.state.heading),
                "speed_mps": self.state.speed,
                "speed_kph": self.speed_kph,
                "velocity_x": self.state.velocity_x,
                "velocity_y": self.state.velocity_y,
                "yaw_rate": self.state.yaw_rate,
                "lateral_g": self.state.lateral_g,
                "longitudinal_g": self.state.longitudinal_g,
            },
            "engine": self.engine.get_state(),
            "transmission": self.transmission.get_state(),
            "aero": self.aero.get_state(),
            "suspension": self.suspension.get_state(),
            "chassis": self.chassis.get_state(),
            "electronics": self.electronics.get_state(),
            "tires": self.tires.get_state(),
        }
    
    def get_observation(self) -> np.ndarray:
        """Get car state as observation vector for ML.
        
        Returns:
            Numpy array of normalized state values
        """
        # Normalized observation suitable for neural networks
        return np.array([
            # Speed (normalized to ~300 kph max)
            self.state.speed / 83.33,
            
            # G-forces (normalized to ~3g max)
            self.state.lateral_g / 3.0,
            self.state.longitudinal_g / 3.0,
            
            # Yaw rate (normalized)
            self.state.yaw_rate / 2.0,
            
            # Engine
            self.engine.rpm / self.engine.config.max_rpm,
            self.engine.throttle,
            
            # Gear (normalized)
            self.transmission.current_gear / 6.0,
            
            # Tire temps (normalized to optimal range)
            self.tires.get_average_temperature() / 100.0,
            
            # Tire wear
            self.tires.get_average_wear(),
            
            # Electronics active
            float(self.electronics.tc.is_active),
            float(self.electronics.abs.is_active),
        ], dtype=np.float32)
