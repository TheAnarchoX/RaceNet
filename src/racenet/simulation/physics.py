"""
Physics engine - Core physics calculations for the simulation.

Provides:
- Force and acceleration calculations
- Collision detection (track boundaries)
- Coordinate transformations
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PhysicsConfig:
    """Physics simulation configuration."""
    # Time step
    default_dt: float = 0.01  # 100 Hz simulation
    
    # Gravity
    gravity: float = 9.81
    
    # Air properties
    air_density: float = 1.225
    
    # Track friction
    default_friction: float = 1.0
    
    # Damping factors
    linear_damping: float = 0.001
    angular_damping: float = 0.02


class PhysicsEngine:
    """Physics engine for vehicle simulation.
    
    Handles core physics calculations separate from vehicle model.
    This allows for potential multi-threading and consistent physics.
    """
    
    def __init__(self, config: PhysicsConfig | None = None):
        """Initialize physics engine.
        
        Args:
            config: Physics configuration. Uses defaults if None.
        """
        self.config = config or PhysicsConfig()
    
    def calculate_acceleration(
        self,
        force: np.ndarray,
        mass: float,
    ) -> np.ndarray:
        """Calculate acceleration from force.
        
        Args:
            force: Force vector [Fx, Fy, Fz] in Newtons
            mass: Mass in kg
            
        Returns:
            Acceleration vector [ax, ay, az] in m/s^2
        """
        return force / mass
    
    def calculate_angular_acceleration(
        self,
        moment: float,
        inertia: float,
    ) -> float:
        """Calculate angular acceleration from moment.
        
        Args:
            moment: Moment in Nm
            inertia: Moment of inertia in kg*m^2
            
        Returns:
            Angular acceleration in rad/s^2
        """
        return moment / inertia
    
    def integrate_position(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Integrate position from velocity.
        
        Args:
            position: Current position [x, y, z]
            velocity: Velocity [vx, vy, vz]
            dt: Time step
            
        Returns:
            New position
        """
        return position + velocity * dt
    
    def integrate_velocity(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Integrate velocity from acceleration.
        
        Args:
            velocity: Current velocity
            acceleration: Acceleration
            dt: Time step
            
        Returns:
            New velocity
        """
        new_vel = velocity + acceleration * dt
        # Apply linear damping scaled by timestep for stability across dt values.
        damping = np.exp(-self.config.linear_damping * dt)
        new_vel *= damping
        return new_vel

    def integrate_rigid_body_rk4(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        heading: float,
        yaw_rate: float,
        body_accel: np.ndarray,
        yaw_accel: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Integrate planar rigid body state using RK4.

        Args:
            position: Current position [x, y] in world frame
            velocity: Current velocity [vx, vy] in world frame
            heading: Current heading in radians
            yaw_rate: Current yaw rate in rad/s
            body_accel: Acceleration in body frame [ax, ay]
            yaw_accel: Yaw acceleration in rad/s^2
            dt: Time step in seconds

        Returns:
            Tuple of (position, velocity, heading, yaw_rate)
        """
        if dt <= 1e-9:
            return position, velocity, heading, yaw_rate

        state = np.array([
            position[0],
            position[1],
            velocity[0],
            velocity[1],
            heading,
            yaw_rate,
        ], dtype=float)

        def derivatives(state_vec: np.ndarray) -> np.ndarray:
            vx, vy = state_vec[2], state_vec[3]
            heading_local = state_vec[4]
            yaw_rate_local = state_vec[5]
            accel_world = self.local_to_world(body_accel, heading_local)
            return np.array([
                vx,
                vy,
                accel_world[0],
                accel_world[1],
                yaw_rate_local,
                yaw_accel,
            ], dtype=float)

        k1 = derivatives(state)
        k2 = derivatives(state + 0.5 * dt * k1)
        k3 = derivatives(state + 0.5 * dt * k2)
        k4 = derivatives(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return (
            np.array([new_state[0], new_state[1]]),
            np.array([new_state[2], new_state[3]]),
            float(new_state[4]),
            float(new_state[5]),
        )
    
    def rotate_vector_2d(
        self,
        vector: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """Rotate 2D vector by angle.
        
        Args:
            vector: 2D vector [x, y]
            angle: Rotation angle in radians
            
        Returns:
            Rotated vector
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([
            vector[0] * cos_a - vector[1] * sin_a,
            vector[0] * sin_a + vector[1] * cos_a,
        ])
    
    def world_to_local(
        self,
        world_vec: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        """Convert world vector to local (car) frame.
        
        Args:
            world_vec: Vector in world coordinates
            heading: Car heading in radians
            
        Returns:
            Vector in local coordinates
        """
        return self.rotate_vector_2d(world_vec, -heading)
    
    def local_to_world(
        self,
        local_vec: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        """Convert local (car) vector to world frame.
        
        Args:
            local_vec: Vector in car coordinates
            heading: Car heading in radians
            
        Returns:
            Vector in world coordinates
        """
        return self.rotate_vector_2d(local_vec, heading)
    
    def calculate_centripetal_force(
        self,
        mass: float,
        speed: float,
        radius: float,
    ) -> float:
        """Calculate centripetal force for circular motion.
        
        Args:
            mass: Mass in kg
            speed: Speed in m/s
            radius: Turn radius in m
            
        Returns:
            Required centripetal force in N
        """
        if radius < 1e-6:
            return 0.0
        return mass * speed**2 / radius
    
    def calculate_braking_distance(
        self,
        speed: float,
        deceleration: float,
    ) -> float:
        """Calculate braking distance.
        
        Args:
            speed: Initial speed in m/s
            deceleration: Deceleration in m/s^2
            
        Returns:
            Braking distance in m
        """
        if deceleration < 1e-6:
            return float('inf')
        return speed**2 / (2 * deceleration)
    
    def calculate_maximum_speed_for_corner(
        self,
        radius: float,
        friction: float,
        banking_rad: float = 0.0,
    ) -> float:
        """Calculate maximum cornering speed.
        
        Args:
            radius: Turn radius in m
            friction: Friction coefficient
            banking_rad: Banking angle in radians
            
        Returns:
            Maximum speed in m/s
        """
        g = self.config.gravity
        
        if banking_rad != 0:
            # Banked turn
            tan_bank = np.tan(banking_rad)
            numerator = g * radius * (friction + tan_bank)
            denominator = 1 - friction * tan_bank
            if denominator <= 0:
                return float('inf')  # Banked enough to corner at any speed
            return np.sqrt(numerator / denominator)
        else:
            # Flat turn
            return np.sqrt(friction * g * radius)
