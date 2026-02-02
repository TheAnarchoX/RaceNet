# Add New Car Component

Use this prompt when adding a new component to the car model (e.g., brakes, differential, fuel system).

## Instructions

1. Create the component in `src/racenet/car/`
2. Follow the existing component patterns
3. Integrate with the main Car class
4. Add telemetry channels
5. Write comprehensive tests

## Reference Files

- [src/racenet/car/engine.py](../../src/racenet/car/engine.py) - Engine component pattern
- [src/racenet/car/tires.py](../../src/racenet/car/tires.py) - Tires component pattern
- [src/racenet/car/car.py](../../src/racenet/car/car.py) - Main car integration
- [tests/test_car.py](../../tests/test_car.py) - Test patterns

## Component Structure

```python
"""
New car component module.

This module implements [component description].
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ComponentConfig:
    """Configuration for [component].
    
    Attributes:
        param1: Description of param1.
        param2: Description of param2.
    """
    param1: float = 1.0
    param2: float = 2.0


@dataclass
class ComponentState:
    """Current state of [component].
    
    Attributes:
        value1: Description of value1.
        value2: Description of value2.
    """
    value1: float = 0.0
    value2: float = 0.0


class Component:
    """[Component] model for the car.
    
    This component [description of what it does and how it affects
    the car's behavior].
    
    Example:
        >>> config = ComponentConfig(param1=1.5)
        >>> component = Component(config)
        >>> component.update(0.01, inputs)
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the component.
        
        Args:
            config: Component configuration. Uses defaults if None.
        """
        self.config = config or ComponentConfig()
        self.state = ComponentState()
    
    def update(self, dt: float, inputs: "InputType") -> "OutputType":
        """Update component state for one simulation timestep.
        
        Args:
            dt: Timestep in seconds.
            inputs: Input values for this update.
            
        Returns:
            Output values from this component.
        """
        # Implementation here
        pass
    
    def reset(self) -> None:
        """Reset component to initial state."""
        self.state = ComponentState()
    
    def get_telemetry(self) -> dict:
        """Return current telemetry data.
        
        Returns:
            Dictionary of telemetry channel values.
        """
        return {
            "component.value1": self.state.value1,
            "component.value2": self.state.value2,
        }
```

## Integration Steps

1. **Create the component file**: `src/racenet/car/<component>.py`

2. **Export from package**: Add to `src/racenet/car/__init__.py`
   ```python
   from .component import Component, ComponentConfig, ComponentState
   ```

3. **Integrate with Car class**: Update `src/racenet/car/car.py`
   ```python
   # In CarConfig
   component: ComponentConfig = field(default_factory=ComponentConfig)
   
   # In Car.__init__
   self.component = Component(config.component)
   
   # In Car.update
   component_output = self.component.update(dt, inputs)
   
   # In Car.reset
   self.component.reset()
   
   # In Car.get_telemetry
   telemetry.update(self.component.get_telemetry())
   ```

4. **Add tests**: Create `tests/test_<component>.py`

## Checklist

- [ ] Create component file with Config, State, and main class
- [ ] Add comprehensive docstrings
- [ ] Add type hints to all functions
- [ ] Export from `car/__init__.py`
- [ ] Integrate with Car class
- [ ] Add telemetry channels
- [ ] Create test file
- [ ] Test component in isolation
- [ ] Test component integrated with Car
- [ ] Run full test suite: `pytest tests/`
