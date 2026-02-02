# Implement Task from TASKS.md

Use this prompt to implement a specific task from the [TASKS.md](../../TASKS.md) file.

## Instructions

1. Read the task specification completely
2. Check that all dependencies are satisfied
3. Write tests first based on acceptance criteria
4. Implement the feature incrementally
5. Ensure all tests pass
6. Update documentation as needed

## Context Files

- [TASKS.md](../../TASKS.md) - Task definitions
- [README.md](../../README.md) - Project overview
- [pyproject.toml](../../pyproject.toml) - Project configuration

## Task Implementation Template

When implementing, follow this structure:

```python
# 1. Add any new configuration dataclasses
@dataclass
class NewConfig:
    """Configuration for the new feature."""
    pass

# 2. Add any new state dataclasses
@dataclass  
class NewState:
    """State for the new feature."""
    pass

# 3. Implement the main class/functions
class NewFeature:
    """Implementation of the new feature."""
    pass

# 4. Add telemetry integration
def get_telemetry(self) -> dict:
    """Return telemetry data for this feature."""
    pass

# 5. Write comprehensive tests
def test_new_feature_basic():
    """Test basic functionality."""
    pass
```

## Checklist

- [ ] Read full task description from TASKS.md
- [ ] Verify dependencies are complete
- [ ] Create test file: `tests/test_<feature>.py`
- [ ] Write tests for each acceptance criterion
- [ ] Implement the feature
- [ ] Add type hints to all functions
- [ ] Add docstrings
- [ ] Add telemetry channels if applicable
- [ ] Run all tests: `pytest tests/`
- [ ] Update README if needed
