# Multi-Car Population Training Example

This example demonstrates population-based training with multiple cars evaluated in parallel.

## What You'll Learn

- How to manage multiple cars in a population
- How to track fitness across generations
- How to implement evaluation loops
- How to analyze population statistics

## Running the Example

```bash
python population_training.py
```

## Population-Based Training

Population-based training is useful for:
- Evolutionary algorithms (genetic algorithms, ES)
- Parallel evaluation of different policies
- Finding diverse solutions

## Multi-Car Manager

```python
from racenet.ml import MultiCarManager
from racenet.ml.multi_car import MultiCarConfig

config = MultiCarConfig(
    population_size=20,
    max_generations=100,
)

manager = MultiCarManager(config)
manager.initialize_population()
```

## Evaluation Loop

```python
for generation in range(num_generations):
    # Evaluate each car
    for car in manager.population:
        fitness = evaluate(car)
        manager.set_fitness(car.car_id, fitness)
    
    # Get statistics
    stats = manager.get_generation_stats()
    print(f"Gen {generation}: Best = {stats['max']:.2f}")
    
    # Selection and breeding (not shown - depends on algorithm)
    
    # Advance generation
    manager.complete_generation()
    manager.next_generation()
```

## Statistics Available

```python
stats = manager.get_generation_stats()
# Returns:
# {
#     'mean': 45.2,    # Average fitness
#     'std': 12.3,     # Standard deviation
#     'min': 23.1,     # Worst fitness
#     'max': 78.4,     # Best fitness
# }
```

## Ranked Population

```python
# Get population sorted by fitness (best first)
ranked = manager.get_ranked_population()
for car, fitness in ranked[:5]:
    print(f"Car {car.car_id}: {fitness}")
```

## Integration with Evolutionary Algorithms

For real evolutionary training, you might use:

### NEAT (NeuroEvolution of Augmenting Topologies)
```bash
pip install neat-python
```

### Evolution Strategies
```bash
pip install evotorch
```

### Custom Implementation
```python
def breed(parent1, parent2):
    """Create child from two parents."""
    child_weights = crossover(parent1.weights, parent2.weights)
    child_weights = mutate(child_weights)
    return child_weights
```

## No Collision Mode

Cars in RaceNet don't collide with each other, making parallel evaluation straightforward. Each car can be evaluated independently.

## Next Steps

- Implement proper evolutionary operators (crossover, mutation)
- Use neural network policies instead of random controllers
- Combine with [telemetry_analysis](../telemetry_analysis/) to analyze top performers
- Scale to larger populations for faster exploration
