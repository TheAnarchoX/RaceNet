"""
Multi-car manager - Manage multiple cars across generations.

Provides:
- Population management
- Generation tracking
- Parallel evaluation support
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any
import numpy as np

from racenet.car.car import Car, CarConfig, CarInputs


@dataclass
class MultiCarConfig:
    """Configuration for multi-car management."""
    # Population
    population_size: int = 20
    
    # Generations
    max_generations: int = 100
    
    # Evaluation
    max_steps_per_car: int = 10000
    
    # Selection
    elite_count: int = 2  # Top performers kept unchanged


class MultiCarManager:
    """Manages multiple cars for ML training.
    
    Supports population-based training approaches where
    multiple agents are evaluated in parallel.
    
    Features:
    - Population management
    - Generation tracking
    - Fitness-based selection
    - Parallel evaluation support
    """
    
    def __init__(self, config: MultiCarConfig | None = None):
        """Initialize manager.
        
        Args:
            config: Manager configuration
        """
        self.config = config or MultiCarConfig()
        
        # Population
        self._population: List[Car] = []
        self._fitness: Dict[int, float] = {}
        
        # Generation
        self._current_generation: int = 0
        self._generation_history: List[Dict[str, Any]] = []
        
        # Callbacks
        self._on_generation_complete: List[Callable] = []
    
    @property
    def population_size(self) -> int:
        """Current population size."""
        return len(self._population)
    
    @property
    def current_generation(self) -> int:
        """Current generation number."""
        return self._current_generation
    
    @property
    def population(self) -> List[Car]:
        """Current population of cars."""
        return self._population
    
    def initialize_population(
        self,
        car_config: CarConfig | None = None,
        start_positions: List[tuple] | None = None,
    ) -> List[Car]:
        """Initialize population of cars.
        
        Args:
            car_config: Configuration for all cars
            start_positions: Optional (x, y, heading) for each car
            
        Returns:
            List of created cars
        """
        self._population = []
        self._fitness = {}
        
        for i in range(self.config.population_size):
            car = Car(config=car_config, car_id=i)
            
            if start_positions and i < len(start_positions):
                x, y, heading = start_positions[i]
                car.reset(x=x, y=y, heading=heading)
            else:
                car.reset()
            
            self._population.append(car)
            self._fitness[i] = 0.0
        
        return self._population
    
    def get_car(self, index: int) -> Optional[Car]:
        """Get car by index.
        
        Args:
            index: Car index
            
        Returns:
            Car if found
        """
        if 0 <= index < len(self._population):
            return self._population[index]
        return None
    
    def set_fitness(self, car_id: int, fitness: float) -> None:
        """Set fitness for a car.
        
        Args:
            car_id: Car ID
            fitness: Fitness value
        """
        self._fitness[car_id] = fitness
    
    def get_fitness(self, car_id: int) -> float:
        """Get fitness for a car.
        
        Args:
            car_id: Car ID
            
        Returns:
            Fitness value
        """
        return self._fitness.get(car_id, 0.0)
    
    def get_all_fitness(self) -> Dict[int, float]:
        """Get fitness for all cars.
        
        Returns:
            Dictionary of car_id to fitness
        """
        return self._fitness.copy()
    
    def get_ranked_population(self) -> List[tuple]:
        """Get population ranked by fitness.
        
        Returns:
            List of (car, fitness) tuples, highest first
        """
        ranked = []
        for car in self._population:
            fitness = self._fitness.get(car.car_id, 0.0)
            ranked.append((car, fitness))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def get_best_car(self) -> Optional[Car]:
        """Get highest fitness car.
        
        Returns:
            Best performing car
        """
        ranked = self.get_ranked_population()
        return ranked[0][0] if ranked else None
    
    def get_generation_stats(self) -> Dict[str, float]:
        """Get statistics for current generation.
        
        Returns:
            Dictionary of statistics
        """
        fitness_values = list(self._fitness.values())
        
        if not fitness_values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        
        return {
            "mean": float(np.mean(fitness_values)),
            "std": float(np.std(fitness_values)),
            "min": float(np.min(fitness_values)),
            "max": float(np.max(fitness_values)),
        }
    
    def complete_generation(self) -> Dict[str, Any]:
        """Complete current generation and record stats.
        
        Returns:
            Generation summary
        """
        stats = self.get_generation_stats()
        
        summary = {
            "generation": self._current_generation,
            "population_size": self.population_size,
            "stats": stats,
            "best_fitness": max(self._fitness.values()) if self._fitness else 0,
        }
        
        self._generation_history.append(summary)
        
        # Call callbacks
        for callback in self._on_generation_complete:
            callback(summary)
        
        return summary
    
    def next_generation(self) -> int:
        """Advance to next generation.
        
        Returns:
            New generation number
        """
        self._current_generation += 1
        
        # Reset fitness
        for car_id in self._fitness:
            self._fitness[car_id] = 0.0
        
        return self._current_generation
    
    def reset_cars(
        self,
        start_positions: List[tuple] | None = None,
    ) -> None:
        """Reset all cars to starting positions.
        
        Args:
            start_positions: Optional positions for each car
        """
        for i, car in enumerate(self._population):
            if start_positions and i < len(start_positions):
                x, y, heading = start_positions[i]
                car.reset(x=x, y=y, heading=heading)
            else:
                car.reset()
    
    def add_generation_callback(self, callback: Callable) -> None:
        """Add callback for generation completion.
        
        Args:
            callback: Function taking generation summary dict
        """
        self._on_generation_complete.append(callback)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get generation history.
        
        Returns:
            List of generation summaries
        """
        return self._generation_history.copy()
    
    def step_all(
        self,
        action_provider: Callable[[int], CarInputs],
        dt: float,
    ) -> Dict[int, np.ndarray]:
        """Step all cars with provided actions.
        
        Args:
            action_provider: Function taking car_id, returning CarInputs
            dt: Time step
            
        Returns:
            Dictionary of car_id to observation arrays
        """
        observations = {}
        
        for car in self._population:
            inputs = action_provider(car.car_id)
            car.step(inputs, dt)
            observations[car.car_id] = car.get_observation()
        
        return observations
    
    def get_state(self) -> dict:
        """Get manager state.
        
        Returns:
            Dictionary with state
        """
        return {
            "population_size": self.population_size,
            "current_generation": self._current_generation,
            "generation_stats": self.get_generation_stats(),
            "history_length": len(self._generation_history),
        }
