# RaceNet Implementation Tasks

This document defines tasks for implementing and improving the RaceNet racing simulation.
Tasks are ordered by priority and designed for AI agents or developers to pick up independently.

---

## How to Use This Document

Each task includes:
- **Priority**: Order of implementation (P1 = highest)
- **Difficulty**: Easy / Medium / Hard
- **Dependencies**: Tasks that should be completed first
- **Acceptance Criteria**: Definition of done
- **Files to Modify**: Starting points

---

## Phase 1: Core Physics & Simulation (P1)

### Task 1.1: Improve Tire Physics Model
**Priority**: P1  
**Difficulty**: Medium  
**Dependencies**: None  
**Estimated Time**: 2-4 hours

**Description**:
Enhance the tire physics model with a proper Pacejka "Magic Formula" implementation for more realistic grip behavior.

**Current State**:
- Basic slip-based grip calculation in `src/racenet/car/tires.py`
- Simplified combined slip handling

**Requirements**:
1. Implement Pacejka Magic Formula for longitudinal force
2. Implement Pacejka Magic Formula for lateral force
3. Add proper combined slip (friction ellipse) calculation
4. Add tire load sensitivity coefficients
5. Tune parameters for GT3-style tires

**Acceptance Criteria**:
- [x] Tire forces match expected GT3 behavior (~1.5-1.6g cornering)
- [x] Combined slip properly reduces available grip
- [x] Peak slip occurs at realistic values (~8-10% longitudinal, ~8° lateral)
- [x] All existing tests pass
- [x] Add unit tests for Pacejka implementation

**Files to Modify**:
- `src/racenet/car/tires.py`
- `tests/test_tires.py` (create)

---

### Task 1.2: Add Proper Vehicle Dynamics Integration
**Priority**: P1  
**Difficulty**: Hard  
**Dependencies**: Task 1.1  
**Estimated Time**: 4-6 hours

**Description**:
Replace simplified physics with proper rigid body dynamics integration including:
- Proper force/moment summation at CG
- Euler or RK4 integration for state
- Accurate coordinate frame transformations

**Current State**:
- Basic integration in `src/racenet/car/car.py`
- Simplified yaw moment calculation

**Requirements**:
1. Implement proper force accumulation at each wheel
2. Transform wheel forces to body frame
3. Calculate moments about CG correctly
4. Use proper integration (RK4 preferred)
5. Handle small timestep edge cases

**Acceptance Criteria**:
- [ ] Car follows realistic trajectory in corners
- [ ] Weight transfer affects individual wheel loads correctly
- [ ] Oversteer/understeer behavior matches GT3 characteristics
- [ ] Physics stable at varying timesteps (5ms to 20ms)
- [ ] Add integration tests for dynamics

**Files to Modify**:
- `src/racenet/car/car.py`
- `src/racenet/simulation/physics.py`
- `tests/test_dynamics.py` (create)

---

### Task 1.3: Implement Brake Temperature Model
**Priority**: P1  
**Difficulty**: Medium  
**Dependencies**: None  
**Estimated Time**: 2-3 hours

**Description**:
Add brake temperature simulation affecting braking performance.

**Requirements**:
1. Add brake temperature state to car model
2. Calculate heat generation from braking force
3. Calculate cooling based on speed
4. Reduce braking efficiency when cold/overheated
5. Add brake temperature to telemetry

**Acceptance Criteria**:
- [ ] Brake temps increase under heavy braking
- [ ] Brake temps cool at high speed
- [ ] Cold brakes have reduced efficiency
- [ ] Overheated brakes fade (reduced force)
- [ ] Telemetry includes brake temperatures

**Files to Modify**:
- `src/racenet/car/electronics.py` (or new `brakes.py`)
- `src/racenet/car/car.py`
- `tests/test_brakes.py` (create)

---

## Phase 2: Track Generation (P1)

### Task 2.1: Improve Track Closure Algorithm
**Priority**: P1  
**Difficulty**: Medium  
**Dependencies**: None  
**Estimated Time**: 2-3 hours

**Description**:
The current track generator sometimes creates tracks that don't close smoothly. Improve the closing algorithm.

**Current State**:
- Basic closing in `src/racenet/track/generator.py`
- Can create awkward final segments

**Requirements**:
1. Calculate optimal closing path with spline fitting
2. Ensure smooth curvature transitions at start/finish
3. Add safety margin for closure distance
4. Support both left and right closing directions

**Acceptance Criteria**:
- [ ] 100% of generated tracks close properly
- [ ] No abrupt curvature changes at start/finish
- [ ] Closing segment length is reasonable (not too long/short)
- [ ] Add test for track closure validation

**Files to Modify**:
- `src/racenet/track/generator.py`
- `tests/test_track_generator.py` (create)

---

### Task 2.2: Add Racing Line Calculation
**Priority**: P1  
**Difficulty**: Hard  
**Dependencies**: Task 2.1  
**Estimated Time**: 4-6 hours

**Description**:
Calculate optimal racing line for generated tracks to provide reference for ML training.

**Requirements**:
1. Implement racing line optimization algorithm
2. Consider track width constraints
3. Account for kerb usage
4. Store racing line as queryable spline
5. Provide distance-from-racing-line metric

**Acceptance Criteria**:
- [ ] Racing line stays within track limits
- [ ] Line uses track width appropriately in corners
- [ ] Line approaches corners from outside
- [ ] Can query racing line position at any track distance
- [ ] Can calculate lateral offset from racing line

**Files to Modify**:
- `src/racenet/track/racing_line.py` (create)
- `src/racenet/track/track.py`
- `tests/test_racing_line.py` (create)

---

### Task 2.3: Add Elevation Profile Generation
**Priority**: P2  
**Difficulty**: Medium  
**Dependencies**: Task 2.1  
**Estimated Time**: 2-3 hours

**Description**:
Improve elevation generation to create realistic hill profiles.

**Requirements**:
1. Generate smooth elevation changes
2. Ensure reasonable gradients (max ~12%)
3. Create interesting elevation features (crests, compressions)
4. Add elevation to position queries
5. Account for elevation in physics (gravity component)

**Acceptance Criteria**:
- [ ] Tracks have realistic elevation changes
- [ ] No impossible gradients
- [ ] Elevation affects car physics (acceleration on hills)
- [ ] Telemetry includes elevation data

**Files to Modify**:
- `src/racenet/track/generator.py`
- `src/racenet/track/segment.py`
- `src/racenet/car/car.py` (for physics effects)

---

## Phase 3: Telemetry & Analysis (P2)

### Task 3.1: Add Comparative Telemetry Analysis
**Priority**: P2  
**Difficulty**: Medium  
**Dependencies**: None  
**Estimated Time**: 2-3 hours

**Description**:
Add tools to compare telemetry between laps or cars.

**Requirements**:
1. Synchronize telemetry by distance (not time)
2. Calculate delta at each point
3. Identify gains/losses by corner
4. Generate comparison summary

**Acceptance Criteria**:
- [ ] Can compare any two laps
- [ ] Comparison aligned by track distance
- [ ] Shows time gain/loss at each sector
- [ ] Exports comparison data

**Files to Modify**:
- `src/racenet/telemetry/analyzer.py` (create)
- `tests/test_telemetry_analysis.py` (create)

---

### Task 3.2: Add Real-time Telemetry Dashboard Interface
**Priority**: P2  
**Difficulty**: Medium  
**Dependencies**: None  
**Estimated Time**: 3-4 hours

**Description**:
Create an interface for real-time telemetry visualization (data provider, not UI).

**Requirements**:
1. WebSocket-compatible data streaming
2. Configurable update rates
3. Channel subscription system
4. Data compression for efficiency

**Acceptance Criteria**:
- [ ] Can stream telemetry at 50Hz
- [ ] Clients can subscribe to specific channels
- [ ] Data format documented
- [ ] Example client implementation provided

**Files to Modify**:
- `src/racenet/telemetry/streaming.py` (create)
- `examples/telemetry_client.py` (create)

---

## Phase 4: ML Integration (P2)

### Task 4.1: Add Curriculum Learning Support
**Priority**: P2  
**Difficulty**: Medium  
**Dependencies**: None  
**Estimated Time**: 2-3 hours

**Description**:
Add support for curriculum learning with progressive difficulty.

**Requirements**:
1. Define difficulty parameters (track complexity, speed limits, etc.)
2. Create difficulty presets
3. Add methods to adjust difficulty dynamically
4. Track agent progress for automatic advancement

**Acceptance Criteria**:
- [ ] Can create easy/medium/hard tracks
- [ ] Can limit car speed during learning
- [ ] Automatic difficulty progression option
- [ ] Difficulty state included in environment info

**Files to Modify**:
- `src/racenet/ml/curriculum.py` (create)
- `src/racenet/ml/environment.py`
- `tests/test_curriculum.py` (create)

---

### Task 4.2: Add Multi-Agent Environment
**Priority**: P2  
**Difficulty**: Hard  
**Dependencies**: None  
**Estimated Time**: 4-6 hours

**Description**:
Create a multi-agent version of the environment for competitive training.

**Requirements**:
1. Multiple cars per environment instance
2. Shared observation space including other cars
3. Proper indexing for parallel agents
4. Compatible with RLlib/PettingZoo interfaces

**Acceptance Criteria**:
- [ ] Can run 10+ cars simultaneously
- [ ] Each agent has unique observation
- [ ] Other car positions visible in observation
- [ ] Compatible with standard multi-agent libraries

**Files to Modify**:
- `src/racenet/ml/multi_env.py` (create)
- `src/racenet/ml/spaces.py` (extend observations)
- `tests/test_multi_env.py` (create)

---

### Task 4.3: Implement Imitation Learning Data Collection
**Priority**: P2  
**Difficulty**: Medium  
**Dependencies**: Task 2.2 (Racing Line)  
**Estimated Time**: 3-4 hours

**Description**:
Add ability to collect demonstration data from reference racing line.

**Requirements**:
1. Create expert driver that follows racing line
2. Record state-action pairs during driving
3. Export in common IL formats (HDF5, pickle)
4. Add noise injection for data augmentation

**Acceptance Criteria**:
- [ ] Expert driver completes laps smoothly
- [ ] Data includes observations and actions
- [ ] Export compatible with behavior cloning
- [ ] Noise injection configurable

**Files to Modify**:
- `src/racenet/ml/expert.py` (create)
- `src/racenet/ml/data_collection.py` (create)
- `examples/collect_demonstrations.py` (create)

---

## Phase 5: Visualization (P3)

### Task 5.1: Add Track Visualization
**Priority**: P3  
**Difficulty**: Medium  
**Dependencies**: None  
**Estimated Time**: 2-3 hours

**Description**:
Create matplotlib-based track visualization.

**Requirements**:
1. Draw track outline with proper width
2. Show kerbs with colors
3. Mark start/finish and sectors
4. Overlay car positions

**Acceptance Criteria**:
- [ ] Clear track outline rendering
- [ ] Kerbs visible with appropriate colors
- [ ] Sector markers shown
- [ ] Can plot car trajectory on track

**Files to Modify**:
- `src/racenet/viz/track_plot.py` (create)
- `examples/visualize_track.py` (create)

---

### Task 5.2: Add Real-time Pygame Visualization
**Priority**: P3  
**Difficulty**: Hard  
**Dependencies**: Task 5.1  
**Estimated Time**: 4-6 hours

**Description**:
Create real-time visualization using Pygame.

**Requirements**:
1. Render track from top-down view
2. Show car(s) with orientation
3. Display telemetry overlay
4. Support camera following car

**Acceptance Criteria**:
- [ ] Smooth 60 FPS rendering
- [ ] Track clearly visible
- [ ] Car position/heading accurate
- [ ] Basic telemetry display (speed, gear, G-forces)

**Files to Modify**:
- `src/racenet/viz/pygame_renderer.py` (create)
- `examples/visualize_live.py` (create)

---

## Phase 6: Performance & Quality (P3)

### Task 6.1: Add Comprehensive Test Suite
**Priority**: P3  
**Difficulty**: Medium  
**Dependencies**: None  
**Estimated Time**: 3-4 hours

**Description**:
Create comprehensive unit tests for all modules.

**Requirements**:
1. Unit tests for each component
2. Integration tests for car + track
3. Regression tests for physics behavior
4. >80% code coverage

**Acceptance Criteria**:
- [ ] All modules have unit tests
- [ ] Tests run in <30 seconds
- [ ] Coverage > 80%
- [ ] CI pipeline configured

**Files to Modify**:
- `tests/` (multiple files)
- `pyproject.toml` (test configuration)

---

### Task 6.2: Optimize Simulation Performance
**Priority**: P3  
**Difficulty**: Medium  
**Dependencies**: None  
**Estimated Time**: 3-4 hours

**Description**:
Profile and optimize simulation for faster ML training.

**Requirements**:
1. Profile current bottlenecks
2. Optimize hot paths (tire physics, etc.)
3. Add optional NumPy vectorization
4. Consider Numba JIT for critical functions

**Acceptance Criteria**:
- [ ] >2x speedup from current baseline
- [ ] Can simulate 10,000 steps/second on single core
- [ ] No accuracy regression

**Files to Modify**:
- Various physics files
- `src/racenet/simulation/physics.py`

---

### Task 6.3: Add Configuration Serialization
**Priority**: P3  
**Difficulty**: Easy  
**Dependencies**: None  
**Estimated Time**: 1-2 hours

**Description**:
Add ability to save/load car and track configurations.

**Requirements**:
1. Serialize all config dataclasses to JSON/YAML
2. Load configurations from files
3. Validate loaded configurations
4. Include preset configurations

**Acceptance Criteria**:
- [ ] All configs can be saved to JSON
- [ ] Configs can be loaded and used
- [ ] Invalid configs raise clear errors
- [ ] Preset configs included (GT3, track types)

**Files to Modify**:
- `src/racenet/config.py` (create)
- `configs/` directory (create)

---

## Task Template (for adding new tasks)

```markdown
### Task X.Y: [Title]
**Priority**: P1/P2/P3  
**Difficulty**: Easy/Medium/Hard  
**Dependencies**: [List task IDs]  
**Estimated Time**: X-Y hours

**Description**:
[What needs to be done and why]

**Current State**:
[What exists now]

**Requirements**:
1. [Requirement 1]
2. [Requirement 2]
...

**Acceptance Criteria**:
- [ ] [Criterion 1]
- [ ] [Criterion 2]
...

**Files to Modify**:
- [File 1]
- [File 2]
...
```

---

## Contributing

1. Pick a task matching your skill level
2. Check dependencies are completed
3. Create a branch: `feature/task-X.Y-description`
4. Implement following the requirements
5. Ensure all acceptance criteria are met
6. Submit PR with task number in title
