"""
World - World state management for the simulation.

Manages:
- Multiple cars in the simulation
- Track reference
- Global time
- Environment conditions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from racenet.car.car import Car
from racenet.track.track import Track


@dataclass
class EnvironmentConditions:
    """Environmental conditions affecting the simulation."""
    # Temperature
    ambient_temp_c: float = 25.0
    track_temp_c: float = 35.0
    
    # Air
    air_density: float = 1.225
    wind_speed_mps: float = 0.0
    wind_direction_rad: float = 0.0
    
    # Track condition
    track_grip_multiplier: float = 1.0  # 1.0 = dry, < 1.0 = damp/wet
    
    # Time of day (for future lighting/temp effects)
    time_of_day_hours: float = 14.0  # 2 PM default


class World:
    """World state container for simulation.
    
    Manages all entities in the simulation world:
    - Cars (multiple, organized by generation)
    - Track
    - Environment
    - Global timing
    """
    
    def __init__(
        self,
        track: Track | None = None,
        environment: EnvironmentConditions | None = None,
    ):
        """Initialize world.
        
        Args:
            track: Race track (can be set later)
            environment: Environment conditions
        """
        self.track = track
        self.environment = environment or EnvironmentConditions()
        
        # Car management
        self._cars: Dict[int, Car] = {}
        self._car_generations: Dict[int, List[int]] = {}  # generation -> car_ids
        self._next_car_id: int = 0
        self._current_generation: int = 0
        
        # Timing
        self._time: float = 0.0
        self._frame: int = 0
    
    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self._time
    
    @property
    def frame(self) -> int:
        """Current frame number."""
        return self._frame
    
    @property
    def cars(self) -> List[Car]:
        """List of all cars."""
        return list(self._cars.values())
    
    @property
    def car_count(self) -> int:
        """Number of cars in simulation."""
        return len(self._cars)
    
    @property
    def current_generation(self) -> int:
        """Current generation number."""
        return self._current_generation
    
    def set_track(self, track: Track) -> None:
        """Set the race track.
        
        Args:
            track: Track to use
        """
        self.track = track
    
    def add_car(self, car: Car, generation: int | None = None) -> int:
        """Add a car to the world.
        
        Args:
            car: Car to add
            generation: Generation number (uses current if None)
            
        Returns:
            Car ID
        """
        car_id = self._next_car_id
        self._next_car_id += 1
        
        car.car_id = car_id
        self._cars[car_id] = car
        
        gen = generation if generation is not None else self._current_generation
        if gen not in self._car_generations:
            self._car_generations[gen] = []
        self._car_generations[gen].append(car_id)
        
        return car_id
    
    def remove_car(self, car_id: int) -> bool:
        """Remove a car from the world.
        
        Args:
            car_id: ID of car to remove
            
        Returns:
            True if car was removed
        """
        if car_id not in self._cars:
            return False
        
        del self._cars[car_id]
        
        # Remove from generation tracking
        for gen, car_ids in self._car_generations.items():
            if car_id in car_ids:
                car_ids.remove(car_id)
                break
        
        return True
    
    def get_car(self, car_id: int) -> Optional[Car]:
        """Get car by ID.
        
        Args:
            car_id: Car ID
            
        Returns:
            Car if found, None otherwise
        """
        return self._cars.get(car_id)
    
    def get_cars_in_generation(self, generation: int) -> List[Car]:
        """Get all cars in a specific generation.
        
        Args:
            generation: Generation number
            
        Returns:
            List of cars
        """
        if generation not in self._car_generations:
            return []
        
        return [self._cars[cid] for cid in self._car_generations[generation] 
                if cid in self._cars]
    
    def spawn_cars(
        self,
        count: int,
        generation: int | None = None,
        spacing_m: float = 10.0,
    ) -> List[int]:
        """Spawn multiple cars on the track.
        
        Args:
            count: Number of cars to spawn
            generation: Generation number (uses current if None)
            spacing_m: Spacing between cars
            
        Returns:
            List of car IDs
        """
        if self.track is None:
            raise RuntimeError("No track set for spawning cars")
        
        gen = generation if generation is not None else self._current_generation
        car_ids = []
        
        for i in range(count):
            car = Car()
            
            # Calculate starting position with spacing
            start_distance = i * spacing_m
            pos = self.track.get_position_at_distance(start_distance)
            heading = self.track.get_heading_at_distance(start_distance)
            
            car.reset(x=pos[0], y=pos[1], heading=heading)
            car_id = self.add_car(car, gen)
            car_ids.append(car_id)
        
        return car_ids
    
    def new_generation(self) -> int:
        """Start a new generation.
        
        Returns:
            New generation number
        """
        self._current_generation += 1
        return self._current_generation
    
    def clear_generation(self, generation: int) -> int:
        """Remove all cars from a generation.
        
        Args:
            generation: Generation to clear
            
        Returns:
            Number of cars removed
        """
        if generation not in self._car_generations:
            return 0
        
        car_ids = list(self._car_generations[generation])
        for car_id in car_ids:
            self.remove_car(car_id)
        
        return len(car_ids)
    
    def advance_time(self, dt: float) -> None:
        """Advance simulation time.
        
        Args:
            dt: Time step in seconds
        """
        self._time += dt
        self._frame += 1
    
    def reset(self) -> None:
        """Reset world state."""
        self._cars.clear()
        self._car_generations.clear()
        self._next_car_id = 0
        self._current_generation = 0
        self._time = 0.0
        self._frame = 0
        
        if self.track:
            self.track.reset()
    
    def get_state(self) -> dict:
        """Get world state for serialization.
        
        Returns:
            Dictionary containing world state
        """
        return {
            "time": self._time,
            "frame": self._frame,
            "car_count": self.car_count,
            "current_generation": self._current_generation,
            "generations": {
                gen: len(ids) for gen, ids in self._car_generations.items()
            },
            "environment": {
                "ambient_temp": self.environment.ambient_temp_c,
                "track_temp": self.environment.track_temp_c,
                "track_grip": self.environment.track_grip_multiplier,
            },
        }
