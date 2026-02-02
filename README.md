# RaceNet 🏎️

A GT3-style racing simulation framework for machine learning experimentation. Built in pure Python, RaceNet provides a detailed physics-based racing simulation with realistic car models, procedural track generation, and comprehensive telemetry systems.

## Features

### Car Simulation
- **Realistic GT3-style car model** with components:
  - Engine (torque curves, rev limiter, thermal model)
  - 6-speed sequential transmission
  - Aerodynamics (downforce, drag, DRS capability)
  - Suspension (weight transfer, roll, pitch)
  - Tires (grip model, temperature, wear)
  - Electronics (TC, ABS, engine maps, brake bias)

### Track Generation
- **Procedural track generation** with realistic constraints
- Variable track width, banking, and elevation
- Automatic kerb placement
- Sector-based timing

### Telemetry & Scoring
- **Real-time telemetry** for all car systems
- Lap and sector timing with best time tracking
- Driving style scoring (smoothness, efficiency)
- Combined ML-ready reward calculation

### ML Integration
- Gymnasium-compatible environment interface
- Configurable observation and action spaces
- Multi-car/multi-generation population management
- No car-to-car collisions (simplified for ML training)

## Installation

```bash
# Clone the repository
git clone https://github.com/TheAnarchoX/RaceNet.git
cd RaceNet

# Install in development mode
pip install -e ".[dev]"

# Optional: Install ML dependencies
pip install -e ".[ml]"

# Optional: Install visualization dependencies
pip install -e ".[viz]"
```

## Quick Start

### Basic Simulation

```python
from racenet import Simulator, Car
from racenet.track import TrackGenerator
from racenet.car import CarInputs

# Generate a track
generator = TrackGenerator()
track = generator.generate("Test Circuit")

# Create simulator and add car
sim = Simulator()
sim.set_track(track)
car_id = sim.spawn_cars(1)[0]

# Run simulation
sim.start()
for _ in range(1000):
    inputs = CarInputs(throttle=0.5, steering=0.0)
    observations = sim.step({car_id: inputs})
    
    if not sim.is_running:
        break

# Get telemetry
telemetry = sim.get_telemetry()
```

### ML Training with Gymnasium

```python
from racenet.ml import RaceEnv

# Create environment
env = RaceEnv()

# Reset and get initial observation
obs, info = env.reset()

# Training loop
for _ in range(10000):
    action = env.action_space.sample()  # Replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Multi-Car Population Training

```python
from racenet.ml import MultiCarManager
from racenet.car import Car, CarInputs

# Create manager for population-based training
manager = MultiCarManager(config={"population_size": 20})
manager.initialize_population()

# Evaluate population
for car in manager.population:
    # Run simulation for this car
    fitness = evaluate_car(car)
    manager.set_fitness(car.car_id, fitness)

# Get statistics and advance generation
stats = manager.complete_generation()
print(f"Gen {manager.current_generation}: Best = {stats['best_fitness']:.2f}")
manager.next_generation()
```

## Project Structure

```
RaceNet/
├── src/racenet/
│   ├── car/              # Car components (engine, aero, tires, etc.)
│   ├── track/            # Track generation and features
│   ├── simulation/       # Core simulation loop and physics
│   ├── telemetry/        # Data recording and export
│   ├── scoring/          # Lap timing and style scoring
│   └── ml/               # ML integration (Gym env, spaces)
├── examples/             # Example scripts and use cases
│   ├── basic_simulation/ # Basic car simulation
│   ├── track_generation/ # Procedural track generation
│   ├── telemetry_analysis/ # Recording and analyzing telemetry
│   ├── ml_training/      # RL agent training
│   └── multi_car/        # Population-based training
├── tests/                # Unit tests
├── TASKS.md              # Implementation tasks for contributors
├── pyproject.toml        # Project configuration
└── README.md
```

## Examples

Check out the [examples/](examples/) directory for complete, runnable examples:

| Example | Description |
|---------|-------------|
| [basic_simulation](examples/basic_simulation/) | Run a car simulation with manual inputs |
| [track_generation](examples/track_generation/) | Generate procedural race tracks |
| [telemetry_analysis](examples/telemetry_analysis/) | Record and export telemetry data |
| [ml_training](examples/ml_training/) | Train an RL agent with Gymnasium |
| [multi_car](examples/multi_car/) | Population-based multi-car training |

## Dependencies

Core:
- Python >= 3.10
- NumPy >= 1.24.0
- SciPy >= 1.10.0

Optional (ML):
- Gymnasium >= 0.29.0
- Stable-Baselines3 >= 2.0.0
- PyTorch >= 2.0.0

Optional (Visualization):
- Matplotlib >= 3.7.0
- Pygame >= 2.5.0

## Contributing

See [TASKS.md](TASKS.md) for implementation tasks that need to be completed. Tasks are ordered by priority and include clear acceptance criteria.

## License

MIT License - See LICENSE file for details.
