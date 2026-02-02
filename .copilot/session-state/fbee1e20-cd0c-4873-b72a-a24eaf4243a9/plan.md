# Plan

## Problem
Enhance the tire physics model with a Pacejka Magic Formula implementation, combined slip friction ellipse, load sensitivity, and unit tests for GT3-style behavior.

## Approach
- Inspect existing tire and test patterns.
- Update tire config and force calculations to use Pacejka for longitudinal and lateral forces, add load sensitivity, and apply friction ellipse combined slip.
- Add unit tests to validate peak slip locations, combined slip reduction, and realistic peak force ranges.
- Run pytest before and after changes.

## Workplan
- [ ] Review existing tire code and test patterns
- [ ] Implement Pacejka-based longitudinal/lateral force models with load sensitivity
- [ ] Add combined slip friction ellipse scaling
- [ ] Create unit tests for Pacejka behavior
- [ ] Run pytest and fix any failures
