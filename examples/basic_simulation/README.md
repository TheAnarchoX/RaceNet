# Basic Simulation Example

This example demonstrates the fundamental usage of RaceNet to run a car simulation.

## What You'll Learn

- How to generate a procedural race track
- How to create and configure a simulator
- How to spawn cars on the track
- How to control cars with inputs (throttle, brake, steering)
- How to access real-time telemetry data

## Running the Example

```bash
python run_simulation.py
```

## Expected Output

```
============================================================
RaceNet Basic Simulation Example
============================================================

1. Generating track...
   Track: Example Circuit
   Length: 4523 meters
   Segments: 12
   Sectors: 3

2. Setting up simulation...
   Spawned car with ID: 0

3. Running simulation (500 steps at 100Hz = 5 seconds)...
   Step 100: Speed = 42.3 km/h, Gear = 2, RPM = 5234
   Step 200: Speed = 78.6 km/h, Gear = 3, RPM = 6102
   ...

4. Final telemetry snapshot:
   Position: (234.5, 45.2)
   Speed: 65.4 km/h
   ...

============================================================
Simulation complete!
============================================================
```

## Code Walkthrough

### 1. Track Generation

```python
from racenet.track import TrackGenerator

generator = TrackGenerator()
track = generator.generate_with_seed(42, "Example Circuit")
```

The `TrackGenerator` creates procedural race tracks. Using a seed ensures reproducible results.

### 2. Simulator Setup

```python
from racenet import Simulator

sim = Simulator()
sim.set_track(track)
car_ids = sim.spawn_cars(1)
```

The simulator manages the physics loop and all cars on the track.

### 3. Running the Simulation

```python
from racenet.car import CarInputs

sim.start()
for step in range(500):
    inputs = CarInputs(throttle=0.5, brake=0.0, steering=0.0)
    observations = sim.step({car_id: inputs})
```

Each step advances the simulation by the configured time delta (default: 0.01s = 100Hz).

### 4. Accessing Telemetry

```python
car = sim.get_car(car_id)
telemetry = car.get_telemetry()
```

Telemetry provides real-time data for all car systems including:
- Position and velocity
- Engine state (RPM, temperature, throttle)
- Transmission state (gear, shift status)
- Tire conditions (temperature, wear, slip)
- G-forces and more

## Next Steps

- Try modifying the inputs to see how the car responds
- Experiment with different track seeds
- Check out the [telemetry_analysis](../telemetry_analysis/) example for data recording
