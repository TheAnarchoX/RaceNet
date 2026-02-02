# Telemetry Analysis Example

This example demonstrates how to record, analyze, and export car telemetry data.

## What You'll Learn

- How to set up a telemetry recorder
- How to record data during simulation
- How to access real-time statistics
- How to export data to CSV, JSON, and NumPy formats

## Running the Example

```bash
python record_telemetry.py
```

This will create an `output/` directory with exported telemetry files.

## Available Telemetry Channels

RaceNet provides many telemetry channels:

| Channel | Unit | Description |
|---------|------|-------------|
| `speed_kph` | km/h | Vehicle speed |
| `rpm` | RPM | Engine RPM |
| `throttle` | % | Throttle position |
| `brake` | % | Brake pressure |
| `gear` | - | Current gear |
| `lateral_g` | g | Lateral acceleration |
| `longitudinal_g` | g | Longitudinal acceleration |
| `tire_temp_fl/fr/rl/rr` | °C | Tire temperatures |
| `tc_active` | 0/1 | Traction control active |
| `abs_active` | 0/1 | ABS active |

## Recording Telemetry

```python
from racenet.telemetry import TelemetryRecorder

# Create recorder attached to a car
recorder = TelemetryRecorder(car=car)

# Record during simulation
for step in range(1000):
    sim.step({car_id: inputs})
    recorder.record(sim.time, car.get_telemetry())
```

## Accessing Statistics

```python
# Get statistics for all channels
stats = recorder.get_statistics()

print(f"Max speed: {stats['speed_kph']['max']}")
print(f"Avg RPM: {stats['rpm']['mean']}")
```

## Exporting Data

```python
from racenet.telemetry import TelemetryExporter

exporter = TelemetryExporter()

# Export to CSV (human-readable)
exporter.export_csv(recorder, "telemetry.csv")

# Export to JSON (with metadata)
exporter.export_json(recorder, "telemetry.json")

# Export to NumPy (for analysis)
exporter.export_numpy(recorder, "telemetry.npz")
```

## Using Exported Data

### CSV
```python
import pandas as pd
df = pd.read_csv("output/telemetry.csv")
print(df.describe())
```

### NumPy
```python
import numpy as np
data = np.load("output/telemetry.npz")
speeds = data["speed_kph_values"]
times = data["speed_kph_times"]
```

## Next Steps

- Combine with [ml_training](../ml_training/) to analyze agent performance
- Create custom visualizations from exported data
- Compare telemetry across different laps or runs
