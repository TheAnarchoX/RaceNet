# RaceNet Examples

This directory contains example scripts demonstrating various features of the RaceNet racing simulation framework.

## Examples Overview

| Example | Description | Key Features |
|---------|-------------|--------------|
| [basic_simulation](basic_simulation/) | Basic car simulation on a track | Car controls, physics step, telemetry |
| [track_generation](track_generation/) | Procedural track generation | Track generator, configuration, seeding |
| [telemetry_analysis](telemetry_analysis/) | Recording and analyzing telemetry | Data recording, export, analysis |
| [ml_training](ml_training/) | Machine learning agent training | Gymnasium environment, RL training |
| [multi_car](multi_car/) | Multi-car population training | Population management, generations |

## Running Examples

First, ensure RaceNet is installed:

```bash
# From the repository root
pip install -e ".[dev]"
```

Then navigate to any example directory and run the script:

```bash
cd examples/basic_simulation
python run_simulation.py
```

## Example Structure

Each example follows a consistent structure:

```
example_name/
├── README.md           # Documentation for this example
└── example_script.py   # Main runnable script
```

## Prerequisites

- Python >= 3.10
- RaceNet installed (see Installation in main README)

For ML examples, also install:
```bash
pip install -e ".[ml]"
```
