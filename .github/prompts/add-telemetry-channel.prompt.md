# Add Telemetry Channel

Use this prompt when adding new telemetry channels to record car or simulation data.

## Instructions

1. Identify the data source (car component, track, simulation)
2. Define the channel in the appropriate component
3. Register the channel with the telemetry system
4. Add export support
5. Test the channel recording

## Reference Files

- [src/racenet/telemetry/channel.py](../../src/racenet/telemetry/channel.py) - Channel definitions
- [src/racenet/telemetry/recorder.py](../../src/racenet/telemetry/recorder.py) - Recording system
- [src/racenet/telemetry/exporter.py](../../src/racenet/telemetry/exporter.py) - Export formats

## Channel Naming Convention

Use dot notation to create hierarchical channel names:

```
<system>.<subsystem>.<measurement>

Examples:
- engine.rpm
- engine.temperature
- tire.fl.temperature (fl = front-left)
- tire.fr.slip_ratio
- aero.downforce.front
- position.x
- velocity.magnitude
```

## Adding a Channel

### Step 1: Define in Component

In the component's `get_telemetry()` method:

```python
def get_telemetry(self) -> dict:
    """Return current telemetry data."""
    return {
        "component.channel_name": self.state.value,
        "component.nested.channel": self.computed_value,
    }
```

### Step 2: Register Channel Metadata (Optional)

For enhanced channel information:

```python
from racenet.telemetry import TelemetryChannel

# Define channel with metadata
channel = TelemetryChannel(
    name="engine.rpm",
    unit="rpm",
    min_value=0.0,
    max_value=9000.0,
    description="Engine rotational speed"
)
```

### Step 3: Verify Recording

```python
# In a test
from racenet.telemetry import TelemetryRecorder

recorder = TelemetryRecorder()
recorder.start()

# Update car and record
car.update(dt, inputs)
recorder.record(car.get_telemetry())

# Export and verify
data = recorder.export_dict()
assert "component.channel_name" in data
```

## Channel Categories

### Car Channels
- `engine.*` - Engine data (rpm, torque, temperature)
- `transmission.*` - Gearbox data (gear, clutch)
- `tire.<position>.*` - Per-tire data (temperature, wear, slip)
- `suspension.<position>.*` - Per-corner suspension
- `aero.*` - Aerodynamic forces
- `electronics.*` - TC, ABS, other systems

### Position/Motion Channels
- `position.*` - x, y, z coordinates
- `velocity.*` - Speed components
- `acceleration.*` - G-forces
- `rotation.*` - Heading, pitch, roll

### Track Channels
- `track.distance` - Distance around track
- `track.progress` - Lap progress (0-1)
- `track.sector` - Current sector number
- `track.surface.*` - Surface conditions

### Timing Channels
- `lap.number` - Current lap
- `lap.time` - Current lap time
- `sector.*.time` - Sector times

## Checklist

- [ ] Choose appropriate channel name with dot notation
- [ ] Add to component's `get_telemetry()` method
- [ ] Use SI units (meters, seconds, Newtons, etc.)
- [ ] Add channel metadata if needed
- [ ] Verify channel appears in recorded data
- [ ] Test export to CSV/JSON/NumPy formats
- [ ] Document channel in component docstring
