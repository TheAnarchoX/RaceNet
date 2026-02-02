"""Integration tests for vehicle dynamics."""

import numpy as np

from racenet.car.car import Car, CarInputs


def _run_cornering_simulation(
    dt: float,
    accel_time: float = 2.0,
    corner_time: float = 3.0,
) -> Car:
    car = Car()
    car.reset()
    # Build up speed before cornering.
    accel_steps = int(accel_time / dt)
    for _ in range(accel_steps):
        car.step(CarInputs(throttle=0.7, steering=0.0), dt)
    corner_steps = int(corner_time / dt)
    for _ in range(corner_steps):
        car.step(CarInputs(throttle=0.4, steering=0.35), dt)
    return car


def test_weight_transfer_in_cornering():
    """Weight transfer should bias outside wheels during a corner."""
    car = _run_cornering_simulation(dt=0.01)
    loads = car.suspension.get_wheel_load_newtons(car.chassis.total_mass)
    # Right turn -> higher load on left wheels.
    assert loads[0] > loads[1]
    assert loads[2] > loads[3]


def test_cornering_produces_yaw_response():
    """Cornering should produce a clear yaw response."""
    car = _run_cornering_simulation(dt=0.01)
    assert abs(car.state.yaw_rate) > 0.1
    assert abs(car.state.heading) > 0.1


def test_dynamics_stable_across_timesteps():
    """Cornering trajectory should be stable across timestep sizes."""
    car_fast = _run_cornering_simulation(dt=0.005)
    car_slow = _run_cornering_simulation(dt=0.02)

    displacement = np.hypot(
        car_fast.state.x - car_slow.state.x,
        car_fast.state.y - car_slow.state.y,
    )
    assert displacement < 10.0

    yaw_rate_diff = abs(car_fast.state.yaw_rate - car_slow.state.yaw_rate)
    assert yaw_rate_diff < 0.5
