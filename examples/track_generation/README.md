# Track Generation Example

This example demonstrates the procedural track generation capabilities of RaceNet.

## What You'll Learn

- How to generate random tracks with default settings
- How to use seeds for reproducible track generation
- How to configure track characteristics (length, corner tightness, complexity)
- How to inspect track segments and their properties

## Running the Example

```bash
python generate_tracks.py
```

## Track Configuration Options

The `GeneratorConfig` class allows you to customize track generation:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_length_m` | 4000.0 | Target track length in meters |
| `min_turn_radius_m` | 30.0 | Minimum corner radius (tightest turn) |
| `max_turn_radius_m` | 500.0 | Maximum corner radius (fastest sweeper) |
| `min_straight_length_m` | 100.0 | Minimum straight section length |
| `max_straight_length_m` | 800.0 | Maximum straight section length |
| `complexity` | 0.5 | Track complexity (0.0 = simple, 1.0 = complex) |
| `max_elevation_change_m` | 50.0 | Maximum total elevation change |
| `max_banking_deg` | 15.0 | Maximum corner banking angle |

## Example Configurations

### Sprint Track (Short & Simple)
```python
config = GeneratorConfig(
    target_length_m=2000.0,
    complexity=0.3,
)
```

### Technical Track (Tight Corners)
```python
config = GeneratorConfig(
    min_turn_radius_m=30.0,
    max_turn_radius_m=150.0,
    max_straight_length_m=400.0,
    complexity=0.8,
)
```

### High-Speed Track (Fast Sweepers)
```python
config = GeneratorConfig(
    min_turn_radius_m=100.0,
    max_turn_radius_m=500.0,
    min_straight_length_m=200.0,
    max_straight_length_m=800.0,
)
```

## Seeded Generation

For reproducible tracks (useful for ML training), use seeds:

```python
generator = TrackGenerator()
track = generator.generate_with_seed(42, "My Track")
```

The same seed will always produce the same track layout.

## Track Properties

After generation, you can inspect track properties:

```python
print(f"Length: {track.length} m")
print(f"Segments: {track.num_segments}")
print(f"Is closed: {track.is_closed}")

# Access individual segments
for segment in track.segments:
    print(f"Type: {segment.segment_type}")
    print(f"Length: {segment.length_m}")
    print(f"Curvature: {segment.curvature}")
```

## Next Steps

- Use generated tracks with the [basic_simulation](../basic_simulation/) example
- Explore different configurations to create varied training environments
- Combine with ML training for curriculum learning
