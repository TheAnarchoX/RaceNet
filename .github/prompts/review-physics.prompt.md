# Review Physics Implementation

Use this prompt when reviewing or improving physics implementations in RaceNet.

## Instructions

1. Review the physics model for correctness
2. Check units and dimensional consistency
3. Verify numerical stability
4. Compare with real-world GT3 data
5. Suggest improvements

## Reference Files

- [src/racenet/simulation/physics.py](../../src/racenet/simulation/physics.py) - Physics engine
- [src/racenet/car/tires.py](../../src/racenet/car/tires.py) - Tire model
- [src/racenet/car/aero.py](../../src/racenet/car/aero.py) - Aerodynamics
- [src/racenet/car/suspension.py](../../src/racenet/car/suspension.py) - Suspension
- [src/racenet/car/engine.py](../../src/racenet/car/engine.py) - Engine model

## GT3 Reference Data

### Vehicle Specifications (Typical GT3)
| Parameter | Value | Unit |
|-----------|-------|------|
| Mass | 1280-1350 | kg |
| Power | 500-550 | hp |
| Torque | 500-550 | Nm |
| Wheelbase | 2.4-2.5 | m |
| Track width | 1.6-1.7 | m |
| CG height | 0.4-0.5 | m |

### Performance Targets
| Metric | Value | Notes |
|--------|-------|-------|
| Max lateral G | 1.5-1.6 | Cornering |
| Max braking G | 1.4-1.6 | ABS threshold |
| Max acceleration G | 0.6-0.8 | Traction limited |
| 0-100 km/h | 3.0-3.5 | seconds |
| Top speed | 280-300 | km/h |

### Tire Characteristics
| Parameter | Typical Value |
|-----------|---------------|
| Peak slip angle | 8-10° |
| Peak slip ratio | 8-10% |
| Optimal temp | 85-100°C |
| Grip coefficient | 1.5-1.7 |

### Aerodynamics
| Parameter | Typical Value |
|-----------|---------------|
| Drag coefficient | 0.35-0.45 |
| Downforce at 250 km/h | 1200-1500 kg |
| Front/rear aero balance | 45-55% |

## Physics Review Checklist

### Units & Dimensions
- [ ] All calculations use SI units internally
- [ ] Angles stored in radians
- [ ] Time in seconds, distances in meters
- [ ] Forces in Newtons, torques in Nm
- [ ] Mass in kg, not weight

### Numerical Stability
- [ ] No division by zero possible
- [ ] Clamp values to physical limits
- [ ] Stable at varying timesteps (5-20ms)
- [ ] No NaN or Inf propagation

### Physical Correctness
- [ ] Force directions are correct
- [ ] Moments calculated about correct point
- [ ] Conservation laws respected
- [ ] Coordinate frames consistent

### Tire Model
- [ ] Slip angle calculation correct
- [ ] Slip ratio calculation correct
- [ ] Combined slip handled (friction ellipse)
- [ ] Load sensitivity implemented
- [ ] Peak grip at correct slip values

### Aerodynamics
- [ ] Downforce increases with speed²
- [ ] Drag increases with speed²
- [ ] Front/rear balance shifts appropriately
- [ ] Ground effect approximated

### Suspension
- [ ] Weight transfer calculated correctly
- [ ] Roll and pitch affect tire loads
- [ ] Suspension travel limited
- [ ] Anti-roll bars modeled

## Common Issues

### Issue 1: Incorrect Slip Calculation
```python
# Wrong: Using velocity instead of wheel speed
slip_ratio = (velocity - wheel_speed) / velocity

# Correct: Reference is wheel speed for braking, velocity for acceleration
if accelerating:
    slip_ratio = (wheel_speed - velocity) / wheel_speed
else:
    slip_ratio = (velocity - wheel_speed) / velocity
```

### Issue 2: Missing Load Sensitivity
```python
# Wrong: Constant grip coefficient
grip = base_grip

# Correct: Grip decreases with load
grip = base_grip * (load / nominal_load) ** load_sensitivity
```

### Issue 3: Coordinate Frame Errors
```python
# Wrong: Mixing world and body frames
force_x = tire_force * cos(heading)  # This mixes frames

# Correct: Transform properly
body_force = rotate_to_body(world_force, heading)
```

## Improvement Suggestions

When reviewing, consider:

1. **Accuracy vs Performance**: More complex models are slower
2. **Validation Data**: Can we validate against real data?
3. **Numerical Methods**: Is integration order appropriate?
4. **Edge Cases**: Behavior at extremes (stopped, max speed)
5. **ML Training**: Does physics provide useful gradients?

## Output Format

After review, provide:

```markdown
## Physics Review: [Component]

### Summary
[Brief assessment]

### Issues Found
1. [Issue with severity and location]
2. [Issue with severity and location]

### Recommendations
1. [Specific fix or improvement]
2. [Specific fix or improvement]

### Validation Tests
[Suggested tests to verify correctness]
```
