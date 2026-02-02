# RaceNet - Copilot Instructions

GT3-style racing simulation framework for ML experimentation. Physics-based car simulation + procedural tracks + Gymnasium-compatible RL environment.

## Commands
```bash
pip install -e ".[dev]"     # Dev install
pip install -e ".[ml]"      # Add ML deps (gymnasium, stable-baselines3)
pytest                       # Run tests (uses tests/ dir)
black src/ tests/           # Format (88 char limit)
ruff check src/             # Lint
python run_agent.py         # Run autonomous dev agent
```

## Architecture

**Data Flow**: `Simulator` → `World` (manages `Car[]` + `Track`) → `PhysicsEngine` → per-step telemetry

```
src/racenet/
├── car/         # Components: engine, tires, transmission, aero, suspension, chassis, electronics
├── track/       # TrackGenerator creates Track from Segments with Kerbs/Features
├── simulation/  # Simulator (main loop), World (state), PhysicsEngine
├── telemetry/   # TelemetryRecorder with channel system (see STANDARD_CHANNELS in recorder.py)
├── scoring/     # ScoringSystem combines LapTimer + DrivingStyleScorer for rewards
└── ml/          # RaceEnv (Gymnasium), ObservationSpace, ActionSpace, multi-car population
```

**Core Pattern**: Config dataclass → Component class with `update(dt)`, `reset()`, `get_telemetry()` methods.

## Key Conventions

- **Units**: SI everywhere (m, s, kg, N, rad). Angles in radians internally, degrees for display
- **Timestep**: 10ms default (`SimulatorConfig.fixed_dt = 0.01`)
- **Config/State Separation**: Every component has `ComponentConfig` (immutable params) + `ComponentState` (mutable runtime)
- **Type hints**: Required on all function signatures. Use `X | None` not `Optional[X]`
- **Telemetry channels**: Use underscores like `speed_kph`, `tire_temp_fl` (see `STANDARD_CHANNELS` dict in `telemetry/recorder.py`)

## Component Template
```python
@dataclass
class WidgetConfig:
    param: float = 1.0

class Widget:
    def __init__(self, config: WidgetConfig | None = None):
        self.config = config or WidgetConfig()
        self._state_value: float = 0.0
    
    def update(self, dt: float, inputs: ...) -> ...:
        """Update for one timestep."""
        pass
    
    def reset(self) -> None:
        self._state_value = 0.0
    
    def get_telemetry(self) -> dict:
        return {"widget_value": self._state_value}
```

## ML Environment Usage
```python
from racenet.ml import RaceEnv
env = RaceEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)  # action: [throttle, brake, steering]
```
- Observations normalized to [-1, 1] where possible
- Reward weights in `ScoringConfig`: speed (0.3), progress (0.3), style (0.2), time (0.2)

## Test Pattern
```python
class TestWidget:
    def test_widget_does_expected_thing(self):
        """Test that widget [expected behavior]."""
        widget = Widget(WidgetConfig(param=2.0))
        result = widget.update(0.01, inputs)
        assert result == expected
```

## Task Implementation (from TASKS.md)
1. Check task dependencies are complete
2. Write tests from acceptance criteria first
3. Add new state to component's `get_telemetry()` method
4. Verify physics uses SI units and handles edge cases

## Physics Reference (GT3 targets)
- Cornering grip: ~1.5-1.6g | Peak slip: 8-10% longitudinal, 8° lateral
- Tire temps: 85-100°C optimal | Use Pacejka "Magic Formula" for grip curves
