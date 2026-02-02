# Create New Example

Use this prompt when creating a new example script to showcase RaceNet functionality.

## Instructions

1. Create a new directory in `examples/`
2. Create the main script and README
3. Ensure the example is self-contained
4. Test that the example runs successfully
5. Update the main examples README

## Reference Examples

- [examples/basic_simulation/](../../examples/basic_simulation/) - Basic usage
- [examples/track_generation/](../../examples/track_generation/) - Track generation
- [examples/telemetry_analysis/](../../examples/telemetry_analysis/) - Telemetry recording
- [examples/ml_training/](../../examples/ml_training/) - ML environment
- [examples/multi_car/](../../examples/multi_car/) - Multi-car simulation

## Example Structure

```
examples/<example_name>/
├── README.md           # Documentation
├── <script_name>.py    # Main executable script
└── output/             # Generated output (git-ignored)
```

## README Template

```markdown
# Example: [Title]

[Brief description of what this example demonstrates.]

## What This Example Shows

- [Feature 1]
- [Feature 2]
- [Feature 3]

## Prerequisites

[List any additional dependencies or setup required]

## Running the Example

```bash
# From the repository root
cd examples/<example_name>
python <script_name>.py
```

## Expected Output

[Describe what the user should see when running]

## Key Code Concepts

### [Concept 1]

```python
# Relevant code snippet
```

### [Concept 2]

```python
# Relevant code snippet
```

## Next Steps

- [Link to related example or documentation]
- [Suggestion for what to try next]
```

## Script Template

```python
#!/usr/bin/env python3
"""
Example: [Title]

This example demonstrates [what it shows].

Usage:
    python <script_name>.py

Output:
    [Description of output]
"""

import sys
from pathlib import Path

# Add project root to path for development
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Standard imports
import numpy as np

# RaceNet imports
from racenet import Simulator
from racenet.track import TrackGenerator


def main():
    """Run the example."""
    print("=" * 60)
    print("RaceNet Example: [Title]")
    print("=" * 60)
    print()
    
    # Setup
    print("Setting up...")
    
    # Main example code
    print("Running example...")
    
    # Results
    print()
    print("Results:")
    print("-" * 40)
    
    # Summary
    print()
    print("Example complete!")


if __name__ == "__main__":
    main()
```

## Best Practices

1. **Self-contained**: Example should work without external data
2. **Educational**: Include comments explaining key concepts
3. **Progressive**: Start simple, add complexity
4. **Output directory**: Use `./output/` for generated files (git-ignored)
5. **Print progress**: Show what's happening during execution
6. **Handle errors**: Provide helpful error messages

## Checklist

- [ ] Create `examples/<name>/` directory
- [ ] Create main Python script
- [ ] Add shebang and module docstring
- [ ] Add path setup for development mode
- [ ] Implement the example with clear comments
- [ ] Create README.md with description
- [ ] Test that example runs successfully
- [ ] Verify output is correct
- [ ] Update `examples/README.md` to list new example
- [ ] Update main `README.md` if significant
