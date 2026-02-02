# RaceNet - GitHub Copilot Instructions

This document provides repository-specific guidance for GitHub Copilot when working with RaceNet.

## Project Overview

RaceNet is a GT3-style racing simulation framework designed for machine learning experimentation. It provides:
- Realistic physics-based car simulation
- Procedural track generation
- Comprehensive telemetry systems
- ML-ready interfaces (Gymnasium-compatible)

## Code Style & Conventions

### Python Style
- Follow PEP 8 with 88-character line limit (Black formatting)
- Use type hints for all function signatures
- Use dataclasses for configuration and state objects
- Prefer NumPy for numerical operations

### Naming Conventions
- Classes: `PascalCase` (e.g., `TireModel`, `TrackGenerator`)
- Functions/methods: `snake_case` (e.g., `calculate_grip`, `get_telemetry`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_STEERING_ANGLE`, `DEFAULT_TIMESTEP`)
- Private methods: `_leading_underscore` (e.g., `_compute_forces`)
- Type aliases: `PascalCase` ending with descriptive suffix (e.g., `ForceVector`, `GripCoefficient`)

### File Organization
```
src/racenet/
├── car/           # Car components (engine, tires, suspension, etc.)
├── track/         # Track generation and features
├── simulation/    # Core simulation loop and physics
├── telemetry/     # Data recording and export
├── scoring/       # Lap timing and scoring systems
└── ml/            # Machine learning integration
```

## Architecture Guidelines

### Component Design
- Each car component (engine, tires, etc.) should be a separate class
- Components communicate through well-defined interfaces
- Use configuration dataclasses for component parameters
- State should be clearly separated from configuration

### Physics Implementation
- Use SI units consistently (meters, seconds, kilograms, Newtons)
- Store angles in radians internally, convert for display
- Timestep should be configurable (default: 10ms)
- Consider numerical stability for all calculations

### Telemetry
- All car state should be accessible through telemetry channels
- Channel names use dot notation: `engine.rpm`, `tire.fl.temperature`
- Export supports CSV, JSON, and NumPy formats
- Recording frequency should be configurable

## ML Integration Guidelines

### Environment Design
- Follow Gymnasium API conventions exactly
- Observation space should use `Box` with normalized values [-1, 1] where possible
- Action space should match realistic control inputs
- Include useful metadata in `info` dict

### Reward Design
- Combine lap time and driving style metrics
- Avoid sparse rewards (provide continuous feedback)
- Normalize rewards to reasonable ranges
- Document reward component weights

## Testing Guidelines

### Test Structure
```python
def test_component_specific_behavior():
    """Test that [component] does [expected behavior]."""
    # Arrange
    component = Component(config)
    
    # Act
    result = component.method(inputs)
    
    # Assert
    assert result == expected
```

### What to Test
- Unit tests for each component's core functionality
- Integration tests for component interactions
- Physics regression tests with known good values
- Edge cases (zero inputs, maximum values, etc.)

## Task Implementation

When implementing tasks from `TASKS.md`:

1. **Read the full task description** including requirements and acceptance criteria
2. **Check dependencies** - ensure prerequisite tasks are complete
3. **Create tests first** - write tests based on acceptance criteria
4. **Implement incrementally** - make small, testable changes
5. **Update telemetry** - ensure new state is accessible
6. **Update documentation** - add docstrings and update README if needed

## Common Patterns

### Creating a New Car Component
```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class ComponentConfig:
    """Configuration for the component."""
    param1: float = 1.0
    param2: float = 2.0

@dataclass
class ComponentState:
    """Current state of the component."""
    value1: float = 0.0
    value2: float = 0.0

class Component:
    """Component description."""
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        self.config = config or ComponentConfig()
        self.state = ComponentState()
    
    def update(self, dt: float, inputs: InputType) -> OutputType:
        """Update component state for one timestep."""
        # Implementation
        pass
    
    def reset(self) -> None:
        """Reset component to initial state."""
        self.state = ComponentState()
    
    def get_telemetry(self) -> dict:
        """Return current telemetry data."""
        return {
            "value1": self.state.value1,
            "value2": self.state.value2,
        }
```

### Adding Telemetry Channel
```python
# In the component class
def get_telemetry(self) -> dict:
    return {
        "channel.name": self.state.value,
        "channel.nested.name": self.state.nested_value,
    }

# In the telemetry recorder
recorder.add_channels([
    TelemetryChannel("channel.name", "unit", min_val, max_val),
])
```

### Track Feature Implementation
```python
@dataclass
class FeatureConfig:
    """Configuration for track feature."""
    intensity: float = 0.5
    
class Feature:
    """Track feature that affects car behavior."""
    
    def get_effect(self, position: np.ndarray) -> FeatureEffect:
        """Calculate effect at given position."""
        pass
```

## Physics Reference

### GT3 Car Characteristics
- Mass: ~1300 kg (with driver)
- Power: ~500-550 hp
- Downforce: ~1500 kg at 250 km/h
- Maximum cornering: ~1.5-1.6g
- 0-100 km/h: ~3.5 seconds
- Top speed: ~280-300 km/h

### Tire Behavior
- Peak slip angle: ~8-10 degrees
- Peak slip ratio: ~8-10%
- Optimal temperature: ~85-100°C
- Use Pacejka "Magic Formula" for accurate grip

### Aerodynamics
- Drag coefficient: ~0.35-0.40
- Lift coefficient: ~-2.5 to -3.5 (downforce)
- Balance shift with speed and DRS

## Useful References

- [TASKS.md](../TASKS.md) - Implementation tasks
- [examples/](../examples/) - Usage examples
- [tests/](../tests/) - Test patterns
