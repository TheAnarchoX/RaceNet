#!/usr/bin/env python3
"""
Multi-Car Population Training Example

This example demonstrates how to:
1. Manage multiple cars for population-based training
2. Track fitness across generations
3. Run parallel evaluation
4. Analyze population statistics

Run with: python population_training.py
"""

import numpy as np
from racenet import Simulator
from racenet.track import TrackGenerator
from racenet.car import CarInputs
from racenet.ml import MultiCarManager
from racenet.ml.multi_car import MultiCarConfig


def evaluate_car(car, track, num_steps: int = 500, seed: int | None = None) -> float:
    """Evaluate a single car's fitness on the track.
    
    Fitness is based on:
    - Distance traveled
    - Average speed
    - Staying on track
    
    Args:
        car: Car to evaluate
        track: Track to evaluate on
        num_steps: Number of simulation steps
        seed: Random seed for reproducible behavior
        
    Returns:
        Fitness score (higher is better)
    """
    # Reset car to start
    start_pos = track.get_position_at_distance(0)
    start_heading = track.get_heading_at_distance(0)
    car.reset(x=start_pos[0], y=start_pos[1], heading=start_heading)
    
    total_distance = 0.0
    speeds = []
    off_track_steps = 0
    
    # Simple random controller (would be replaced with neural network)
    # Use dedicated RNG to avoid affecting global state
    rng = np.random.default_rng(seed if seed is not None else car.car_id)
    throttle_bias = rng.uniform(0.5, 1.0)
    steering_noise = rng.uniform(0.0, 0.2)
    
    for step in range(num_steps):
        # Simple controller
        speed = car.speed
        steering = rng.uniform(-steering_noise, steering_noise)
        
        if speed < 20:
            throttle = throttle_bias
            brake = 0.0
        else:
            throttle = throttle_bias * 0.7
            brake = 0.0
        
        inputs = CarInputs(throttle=throttle, brake=brake, steering=steering)
        car.step(inputs, dt=0.01)
        
        # Track metrics
        speeds.append(car.speed)
        
        # Check track position
        track_info = track.get_track_info_at_position(car.state.x, car.state.y)
        if abs(track_info['lateral_offset_m']) > track_info['width_m'] / 2:
            off_track_steps += 1
    
    # Calculate fitness
    avg_speed = np.mean(speeds) if speeds else 0
    on_track_ratio = 1 - (off_track_steps / num_steps)
    
    # Fitness = average speed * on-track bonus
    fitness = avg_speed * (0.5 + 0.5 * on_track_ratio)
    
    return fitness


def main():
    print("=" * 60)
    print("RaceNet Multi-Car Population Training Example")
    print("=" * 60)
    
    # Step 1: Setup track
    print("\n1. Generating track...")
    generator = TrackGenerator()
    track = generator.generate_with_seed(42, "Population Track")
    print(f"   Track: {track.config.name} ({track.length:.0f}m)")
    
    # Step 2: Create multi-car manager
    print("\n2. Setting up population...")
    config = MultiCarConfig(
        population_size=10,
        max_generations=5,
    )
    manager = MultiCarManager(config)
    
    # Initialize population
    start_positions = []
    for i in range(config.population_size):
        pos = track.get_position_at_distance(i * 10)  # 10m spacing
        heading = track.get_heading_at_distance(i * 10)
        start_positions.append((pos[0], pos[1], heading))
    
    manager.initialize_population(start_positions=start_positions)
    print(f"   Population size: {manager.population_size}")
    
    # Step 3: Run generational training
    print("\n3. Running generational training...")
    print("-" * 60)
    
    generation_bests = []
    
    for gen in range(5):
        # Evaluate each car
        for car in manager.population:
            fitness = evaluate_car(car, track, num_steps=300)
            manager.set_fitness(car.car_id, fitness)
        
        # Get generation statistics
        stats = manager.get_generation_stats()
        best_car = manager.get_best_car()
        best_fitness = manager.get_fitness(best_car.car_id)
        generation_bests.append(best_fitness)
        
        print(f"   Generation {gen}: "
              f"Best = {best_fitness:.2f}, "
              f"Mean = {stats['mean']:.2f}, "
              f"Std = {stats['std']:.2f}")
        
        # Complete generation and advance
        manager.complete_generation()
        manager.next_generation()
        
        # Reset cars for next generation
        manager.reset_cars(start_positions)
    
    # Step 4: Display results
    print("\n" + "-" * 60)
    print("\n4. Training Results:")
    print(f"   Total generations: {manager.current_generation}")
    print(f"   Best fitness achieved: {max(generation_bests):.2f}")
    print(f"   Fitness improvement: {generation_bests[-1] - generation_bests[0]:.2f}")
    
    # Step 5: Show ranked population
    print("\n5. Final Population Ranking:")
    ranked = manager.get_ranked_population()
    for i, (car, fitness) in enumerate(ranked[:5]):
        print(f"   #{i+1} Car {car.car_id}: Fitness = {fitness:.2f}")
    
    print("\n" + "=" * 60)
    print("Population training example complete!")
    print("=" * 60)
    print("\nNote: This example uses random controllers. For real training,")
    print("replace with neural network policies and proper genetic algorithms")
    print("or policy gradient methods for population-based training.")


if __name__ == "__main__":
    main()
