"""Tests for the Pacejka tire model."""

from racenet.car.tires import Tire, TireConfig


def test_pacejka_longitudinal_peak_slip() -> None:
    """Peak longitudinal force should occur near 8-10% slip."""
    tire = Tire(TireConfig())
    load_n = 3500.0
    slips = [0.02, 0.06, 0.09, 0.12, 0.18]
    forces = [abs(tire._calculate_longitudinal_force(load_n, s)) for s in slips]
    peak_slip = slips[forces.index(max(forces))]

    assert 0.08 <= peak_slip <= 0.10


def test_pacejka_lateral_peak_slip_angle() -> None:
    """Peak lateral force should occur near 8 degrees."""
    tire = Tire(TireConfig())
    load_n = 3500.0
    angles = [2.0, 6.0, 8.0, 10.0, 14.0]
    forces = [abs(tire._calculate_lateral_force(load_n, a)) for a in angles]
    peak_angle = angles[forces.index(max(forces))]

    assert 7.0 <= peak_angle <= 9.0


def test_combined_slip_reduces_grip() -> None:
    """Combined slip should reduce available force."""
    tire = Tire(TireConfig())
    load_n = 3500.0

    tire.reset()
    fx_only, _ = tire.update(
        load_n=load_n,
        slip_ratio=0.10,
        slip_angle_deg=0.0,
        ground_speed_mps=50.0,
        dt=0.01,
    )
    tire.reset()
    _, fy_only = tire.update(
        load_n=load_n,
        slip_ratio=0.0,
        slip_angle_deg=8.0,
        ground_speed_mps=50.0,
        dt=0.01,
    )
    tire.reset()
    fx_combined, fy_combined = tire.update(
        load_n=load_n,
        slip_ratio=0.10,
        slip_angle_deg=8.0,
        ground_speed_mps=50.0,
        dt=0.01,
    )

    assert abs(fx_combined) < abs(fx_only)
    assert abs(fy_combined) < abs(fy_only)


def test_peak_lateral_force_matches_gt3() -> None:
    """Peak lateral force should be around 1.5-1.6g at nominal load."""
    tire = Tire(TireConfig())
    tire._temperature_c = tire.config.optimal_temp_c
    load_n = 3500.0
    fy_peak = abs(tire._calculate_lateral_force(load_n, 8.0))
    mu = fy_peak / load_n

    assert 1.45 <= mu <= 1.65
