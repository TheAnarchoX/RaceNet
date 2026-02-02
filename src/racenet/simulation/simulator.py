"""
Simulator - Main simulation loop and controller.

Provides:
- High-level simulation control
- Time stepping
- Multi-car management
- Telemetry collection
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import time
import numpy as np

from racenet.car.car import Car, CarInputs
from racenet.track.track import Track
from racenet.simulation.world import World, EnvironmentConditions
from racenet.simulation.physics import PhysicsEngine, PhysicsConfig


@dataclass
class SimulatorConfig:
    """Simulator configuration."""
    # Time stepping
    fixed_dt: float = 0.01           # Physics time step (100 Hz)
    max_dt: float = 0.1              # Maximum frame time
    real_time: bool = False          # Run in real-time or as fast as possible
    
    # Simulation limits
    max_time: float = 600.0          # Maximum simulation time (10 min default)
    max_laps: int = 0                # Max laps per car (0 = unlimited)
    
    # Multi-car
    allow_car_collision: bool = False  # Cars don't collide with each other
    
    # Callbacks
    enable_telemetry: bool = True


class Simulator:
    """Main racing simulator.
    
    Coordinates the simulation of multiple cars on a track,
    handling time stepping, physics updates, and telemetry.
    
    Features:
    - Fixed time-step physics
    - Multi-car simulation (no collisions between cars)
    - Real-time or fast-forward modes
    - Telemetry collection
    - Callback system for ML training
    
    Usage:
        sim = Simulator()
        sim.set_track(track)
        sim.spawn_cars(10)
        
        while sim.is_running:
            observations = sim.step(actions)
    """
    
    def __init__(self, config: SimulatorConfig | None = None):
        """Initialize simulator.
        
        Args:
            config: Simulator configuration. Uses defaults if None.
        """
        self.config = config or SimulatorConfig()
        
        # Core components
        self.physics = PhysicsEngine()
        self.world = World()
        
        # State
        self._running: bool = False
        self._paused: bool = False
        
        # Telemetry collection
        self._telemetry_buffer: List[Dict[str, Any]] = []
        
        # Step callbacks
        self._pre_step_callbacks: List[Callable] = []
        self._post_step_callbacks: List[Callable] = []
        
        # Real-time tracking
        self._last_real_time: float = 0.0
    
    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._running
    
    @property
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self._paused
    
    @property
    def time(self) -> float:
        """Current simulation time."""
        return self.world.time
    
    @property
    def track(self) -> Optional[Track]:
        """Current track."""
        return self.world.track
    
    @property
    def cars(self) -> List[Car]:
        """List of all cars."""
        return self.world.cars
    
    def set_track(self, track: Track) -> None:
        """Set the race track.
        
        Args:
            track: Track to use for simulation
        """
        self.world.set_track(track)
    
    def spawn_cars(
        self,
        count: int,
        generation: int | None = None,
        spacing_m: float = 10.0,
    ) -> List[int]:
        """Spawn cars on the track.
        
        Args:
            count: Number of cars to spawn
            generation: Generation number
            spacing_m: Spacing between cars on grid
            
        Returns:
            List of car IDs
        """
        return self.world.spawn_cars(count, generation, spacing_m)
    
    def add_car(self, car: Car) -> int:
        """Add a specific car to simulation.
        
        Args:
            car: Car to add
            
        Returns:
            Car ID
        """
        return self.world.add_car(car)
    
    def get_car(self, car_id: int) -> Optional[Car]:
        """Get car by ID.
        
        Args:
            car_id: Car ID
            
        Returns:
            Car if found
        """
        return self.world.get_car(car_id)
    
    def add_pre_step_callback(self, callback: Callable) -> None:
        """Add callback called before each step.
        
        Args:
            callback: Function taking (simulator, dt) arguments
        """
        self._pre_step_callbacks.append(callback)
    
    def add_post_step_callback(self, callback: Callable) -> None:
        """Add callback called after each step.
        
        Args:
            callback: Function taking (simulator, dt) arguments
        """
        self._post_step_callbacks.append(callback)
    
    def start(self) -> None:
        """Start the simulation."""
        if self.world.track is None:
            raise RuntimeError("No track set")
        
        self._running = True
        self._paused = False
        self._last_real_time = time.time()
    
    def stop(self) -> None:
        """Stop the simulation."""
        self._running = False
    
    def pause(self) -> None:
        """Pause the simulation."""
        self._paused = True
    
    def resume(self) -> None:
        """Resume the simulation."""
        self._paused = False
        self._last_real_time = time.time()
    
    def step(
        self,
        actions: Dict[int, CarInputs] | None = None,
        dt: float | None = None,
    ) -> Dict[int, np.ndarray]:
        """Advance simulation by one step.
        
        Args:
            actions: Dictionary mapping car_id to CarInputs
            dt: Time step (uses fixed_dt if None)
            
        Returns:
            Dictionary mapping car_id to observation arrays
        """
        if not self._running or self._paused:
            return {}
        
        dt = dt or self.config.fixed_dt
        
        # Handle real-time mode
        if self.config.real_time:
            current_time = time.time()
            elapsed = current_time - self._last_real_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
            self._last_real_time = time.time()
        
        # Pre-step callbacks
        for callback in self._pre_step_callbacks:
            callback(self, dt)
        
        # Default actions if not provided
        if actions is None:
            actions = {}
        
        # Update each car
        observations = {}
        for car in self.world.cars:
            # Get action for this car
            car_inputs = actions.get(car.car_id, CarInputs())
            
            # Step car physics
            car.step(car_inputs, dt)
            
            # Update track position
            if self.world.track:
                track_info = self.world.track.get_track_info_at_position(
                    car.state.x, car.state.y
                )
                # Could apply track limits, kerb effects here
            
            # Get observation
            observations[car.car_id] = car.get_observation()
        
        # Advance world time
        self.world.advance_time(dt)
        
        # Collect telemetry
        if self.config.enable_telemetry:
            self._collect_telemetry()
        
        # Check termination conditions
        if self.world.time >= self.config.max_time:
            self.stop()
        
        # Post-step callbacks
        for callback in self._post_step_callbacks:
            callback(self, dt)
        
        return observations
    
    def step_until(
        self,
        condition: Callable[["Simulator"], bool],
        action_provider: Callable[[Dict[int, np.ndarray]], Dict[int, CarInputs]] | None = None,
        max_steps: int = 100000,
    ) -> int:
        """Step simulation until condition is met.
        
        Args:
            condition: Function returning True when should stop
            action_provider: Function providing actions from observations
            max_steps: Maximum steps to take
            
        Returns:
            Number of steps taken
        """
        steps = 0
        observations = {car.car_id: car.get_observation() for car in self.cars}
        
        while self._running and steps < max_steps:
            if condition(self):
                break
            
            # Get actions
            if action_provider:
                actions = action_provider(observations)
            else:
                actions = {}
            
            # Step
            observations = self.step(actions)
            steps += 1
        
        return steps
    
    def _collect_telemetry(self) -> None:
        """Collect telemetry from all cars."""
        frame_telemetry = {
            "time": self.world.time,
            "frame": self.world.frame,
            "cars": {},
        }
        
        for car in self.world.cars:
            frame_telemetry["cars"][car.car_id] = car.get_telemetry()
        
        self._telemetry_buffer.append(frame_telemetry)
        
        # Limit buffer size
        if len(self._telemetry_buffer) > 10000:
            self._telemetry_buffer = self._telemetry_buffer[-5000:]
    
    def get_telemetry(self) -> List[Dict[str, Any]]:
        """Get collected telemetry data.
        
        Returns:
            List of telemetry frames
        """
        return self._telemetry_buffer.copy()
    
    def clear_telemetry(self) -> None:
        """Clear telemetry buffer."""
        self._telemetry_buffer.clear()
    
    def get_all_observations(self) -> Dict[int, np.ndarray]:
        """Get current observations for all cars.
        
        Returns:
            Dictionary mapping car_id to observation arrays
        """
        return {car.car_id: car.get_observation() for car in self.cars}
    
    def get_all_telemetry(self) -> Dict[int, Dict[str, Any]]:
        """Get current telemetry for all cars.
        
        Returns:
            Dictionary mapping car_id to telemetry dicts
        """
        return {car.car_id: car.get_telemetry() for car in self.cars}
    
    def reset(
        self,
        keep_track: bool = True,
        keep_cars: bool = False,
    ) -> None:
        """Reset simulation.
        
        Args:
            keep_track: Keep current track
            keep_cars: Keep current cars (reset to start positions)
        """
        if keep_cars:
            # Reset cars to starting positions
            for i, car in enumerate(self.cars):
                if self.world.track:
                    start_distance = i * 10.0
                    pos = self.world.track.get_position_at_distance(start_distance)
                    heading = self.world.track.get_heading_at_distance(start_distance)
                    car.reset(x=pos[0], y=pos[1], heading=heading)
                else:
                    car.reset()
        else:
            track = self.world.track if keep_track else None
            self.world.reset()
            if track:
                self.world.set_track(track)
        
        self._telemetry_buffer.clear()
        self._running = False
        self._paused = False
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete simulation state.
        
        Returns:
            Dictionary containing simulation state
        """
        return {
            "config": {
                "fixed_dt": self.config.fixed_dt,
                "real_time": self.config.real_time,
                "max_time": self.config.max_time,
            },
            "running": self._running,
            "paused": self._paused,
            "world": self.world.get_state(),
            "track": self.world.track.get_state() if self.world.track else None,
        }
