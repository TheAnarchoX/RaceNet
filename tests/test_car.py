"""Basic tests for the RaceNet car module."""

import pytest
import numpy as np

from racenet.car.engine import Engine, EngineConfig
from racenet.car.transmission import Transmission, TransmissionConfig, GearState
from racenet.car.aero import Aero, AeroConfig
from racenet.car.suspension import Suspension, SuspensionConfig
from racenet.car.chassis import Chassis, ChassisConfig
from racenet.car.electronics import Electronics, TractionControl, ABS
from racenet.car.tires import Tire, TireSet
from racenet.car.car import Car, CarInputs


class TestEngine:
    """Test engine component."""
    
    def test_engine_initialization(self):
        """Test engine initializes with default config."""
        engine = Engine()
        assert engine.rpm == engine.config.idle_rpm
        assert engine.throttle == 0.0
        
    def test_engine_torque_at_peak(self):
        """Test engine produces expected torque at peak RPM."""
        engine = Engine()
        engine.rpm = 6500  # Peak torque RPM
        engine.throttle = 1.0
        
        torque = engine.get_torque()
        assert torque > 0
        assert torque <= engine.config.max_torque_nm
        
    def test_engine_rev_limiter(self):
        """Test rev limiter cuts torque."""
        engine = Engine()
        engine.rpm = engine.config.rev_limiter_rpm + 100
        engine.throttle = 1.0
        
        torque = engine.get_torque()
        assert torque == 0.0
        assert engine.rev_limiter_active
        
    def test_engine_state(self):
        """Test engine state dictionary."""
        engine = Engine()
        state = engine.get_state()
        
        assert "rpm" in state
        assert "throttle" in state
        assert "torque_nm" in state


class TestTransmission:
    """Test transmission component."""
    
    def test_transmission_initialization(self):
        """Test transmission initializes in neutral."""
        trans = Transmission()
        assert trans.current_gear == int(GearState.NEUTRAL)
        
    def test_upshift(self):
        """Test upshift mechanics."""
        trans = Transmission()
        trans.set_gear(int(GearState.FIRST))
        
        assert trans.shift_up()
        assert trans.is_shifting
        
        # Complete shift
        for _ in range(10):
            trans.update(0.01)
        
        assert trans.current_gear == int(GearState.SECOND)
        
    def test_downshift(self):
        """Test downshift mechanics."""
        trans = Transmission()
        trans.set_gear(int(GearState.THIRD))
        
        assert trans.shift_down()
        
        # Complete shift
        for _ in range(10):
            trans.update(0.01)
        
        assert trans.current_gear == int(GearState.SECOND)
        
    def test_gear_ratios(self):
        """Test gear ratio calculations."""
        trans = Transmission()
        trans.set_gear(int(GearState.FIRST))
        
        ratio = trans.get_total_ratio()
        assert ratio > 0
        assert ratio == trans.get_gear_ratio() * trans.config.final_drive


class TestAero:
    """Test aerodynamics component."""
    
    def test_aero_at_rest(self):
        """Test no aero forces at zero speed."""
        aero = Aero()
        drag, downforce = aero.update(0.0)
        
        assert drag == 0.0
        assert downforce == 0.0
        
    def test_aero_increases_with_speed(self):
        """Test aero forces increase with speed squared."""
        aero = Aero()
        
        _, df1 = aero.update(50.0)
        _, df2 = aero.update(100.0)
        
        # Downforce at 100 m/s should be ~4x downforce at 50 m/s
        ratio = df2 / df1
        assert 3.5 < ratio < 4.5
        
    def test_drs_reduces_drag(self):
        """Test DRS reduces drag."""
        aero = Aero()
        
        aero.update(80.0)
        drag_closed = aero.current_drag
        
        aero.drs_active = True
        aero.update(80.0)
        drag_open = aero.current_drag
        
        assert drag_open < drag_closed


class TestTires:
    """Test tire component."""
    
    def test_tire_initialization(self):
        """Test tire initializes correctly."""
        tire = Tire(position="FL")
        
        assert tire.temperature < 30  # Cold start
        assert tire.wear == 0.0
        assert tire.radius > 0
        
    def test_tire_grip_with_load(self):
        """Test tire produces force with load and slip."""
        tire = Tire()
        
        fx, fy = tire.update(
            load_n=4000.0,
            slip_ratio=0.1,
            slip_angle_deg=0.0,
            ground_speed_mps=50.0,
            dt=0.01,
        )
        
        assert fx > 0  # Positive longitudinal force
        
    def test_tire_set(self):
        """Test tire set has all four tires."""
        tire_set = TireSet()
        
        assert len(tire_set.tires) == 4
        assert tire_set.fl is not None
        assert tire_set.rr is not None


class TestCar:
    """Test complete car simulation."""
    
    def test_car_initialization(self):
        """Test car initializes correctly."""
        car = Car()
        
        assert car.speed == 0.0
        assert car.transmission.current_gear == int(GearState.FIRST)
        
    def test_car_step(self):
        """Test car physics step."""
        car = Car()
        
        inputs = CarInputs(throttle=1.0, brake=0.0, steering=0.0)
        state = car.step(inputs, dt=0.01)
        
        assert state is not None
        
    def test_car_telemetry(self):
        """Test car telemetry output."""
        car = Car()
        telemetry = car.get_telemetry()
        
        assert "car_id" in telemetry
        assert "state" in telemetry
        assert "engine" in telemetry
        assert "transmission" in telemetry
        assert "tires" in telemetry
        
    def test_car_observation(self):
        """Test car ML observation vector."""
        car = Car()
        obs = car.get_observation()
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs) > 0
        
    def test_car_acceleration(self):
        """Test car accelerates with throttle."""
        car = Car()
        
        # Accelerate for 1 second
        inputs = CarInputs(throttle=1.0, brake=0.0, steering=0.0)
        for _ in range(100):
            car.step(inputs, dt=0.01)
        
        assert car.speed > 0


class TestElectronics:
    """Test electronics (TC, ABS)."""
    
    def test_tc_reduces_wheelspin(self):
        """Test TC reduces throttle during wheelspin."""
        tc = TractionControl()
        tc.level = 3
        
        # High slip ratio
        output = tc.process(1.0, wheel_slip_rear=0.2, dt=0.01)
        
        assert output < 1.0  # Throttle should be reduced
        assert tc.is_active
        
    def test_abs_modulates_brakes(self):
        """Test ABS modulates brake pressure."""
        abs_sys = ABS()
        abs_sys.level = 3
        
        # High wheel slip
        output = abs_sys.process(1.0, wheel_slip=0.2, dt=0.01)
        
        assert output <= 1.0
        assert abs_sys.is_active
