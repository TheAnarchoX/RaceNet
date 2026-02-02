"""Basic tests for the RaceNet simulation module."""

import pytest
import numpy as np

from racenet.simulation.simulator import Simulator, SimulatorConfig
from racenet.simulation.world import World, EnvironmentConditions
from racenet.simulation.physics import PhysicsEngine
from racenet.car.car import CarInputs
from racenet.track import TrackGenerator


class TestPhysicsEngine:
    """Test physics engine."""
    
    def test_acceleration_calculation(self):
        """Test force to acceleration conversion."""
        physics = PhysicsEngine()
        
        force = np.array([1000.0, 0.0, 0.0])
        mass = 1000.0
        
        accel = physics.calculate_acceleration(force, mass)
        
        assert np.allclose(accel, [1.0, 0.0, 0.0])
        
    def test_velocity_integration(self):
        """Test velocity integration."""
        physics = PhysicsEngine()
        
        velocity = np.array([10.0, 0.0, 0.0])
        accel = np.array([1.0, 0.0, 0.0])
        
        new_vel = physics.integrate_velocity(velocity, accel, dt=1.0)
        
        assert new_vel[0] > velocity[0]
        
    def test_cornering_speed(self):
        """Test maximum cornering speed calculation."""
        physics = PhysicsEngine()
        
        speed = physics.calculate_maximum_speed_for_corner(
            radius=100.0,
            friction=1.5,
        )
        
        assert speed > 0
        assert speed < 100  # Reasonable for 100m radius


class TestWorld:
    """Test world state management."""
    
    def test_world_creation(self):
        """Test world initializes correctly."""
        world = World()
        
        assert world.car_count == 0
        assert world.time == 0.0
        
    def test_world_time_advance(self):
        """Test time advances correctly."""
        world = World()
        
        world.advance_time(0.1)
        assert world.time == 0.1
        assert world.frame == 1
        
    def test_world_with_track(self):
        """Test world with track."""
        generator = TrackGenerator()
        track = generator.generate()
        
        world = World(track=track)
        assert world.track is not None


class TestSimulator:
    """Test simulator."""
    
    def test_simulator_creation(self):
        """Test simulator initializes correctly."""
        sim = Simulator()
        
        assert not sim.is_running
        assert sim.time == 0.0
        
    def test_simulator_set_track(self):
        """Test setting track."""
        generator = TrackGenerator()
        track = generator.generate()
        
        sim = Simulator()
        sim.set_track(track)
        
        assert sim.track is not None
        
    def test_simulator_spawn_cars(self):
        """Test spawning cars."""
        generator = TrackGenerator()
        track = generator.generate()
        
        sim = Simulator()
        sim.set_track(track)
        
        car_ids = sim.spawn_cars(5)
        
        assert len(car_ids) == 5
        assert len(sim.cars) == 5
        
    def test_simulator_step(self):
        """Test simulation step."""
        generator = TrackGenerator()
        track = generator.generate()
        
        sim = Simulator()
        sim.set_track(track)
        car_ids = sim.spawn_cars(1)
        sim.start()
        
        inputs = {car_ids[0]: CarInputs(throttle=0.5)}
        observations = sim.step(inputs)
        
        assert len(observations) == 1
        assert car_ids[0] in observations
        
    def test_simulator_telemetry(self):
        """Test telemetry collection."""
        generator = TrackGenerator()
        track = generator.generate()
        
        sim = Simulator()
        sim.set_track(track)
        sim.spawn_cars(1)
        sim.start()
        
        for _ in range(10):
            sim.step({})
        
        telemetry = sim.get_telemetry()
        assert len(telemetry) > 0
        
    def test_simulator_reset(self):
        """Test simulator reset."""
        generator = TrackGenerator()
        track = generator.generate()
        
        sim = Simulator()
        sim.set_track(track)
        sim.spawn_cars(3)
        sim.start()
        
        for _ in range(100):
            sim.step({})
        
        sim.reset(keep_track=True)
        
        assert sim.time == 0.0
        assert len(sim.cars) == 0
