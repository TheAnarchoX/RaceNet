"""
Car module - GT3-style racing car simulation.

This module contains all car-related components:
- Engine: Power delivery, RPM, torque curves
- Transmission: Gear ratios, shifting logic
- Aero: Downforce and drag calculations
- Suspension: Spring rates, damping, weight transfer
- Chassis: Weight distribution, center of gravity
- Electronics: TC (Traction Control), ABS systems
- Tires: Grip model, temperature, wear
"""

from racenet.car.car import Car
from racenet.car.engine import Engine
from racenet.car.transmission import Transmission
from racenet.car.aero import Aero
from racenet.car.suspension import Suspension
from racenet.car.chassis import Chassis
from racenet.car.electronics import Electronics, TractionControl, ABS
from racenet.car.tires import Tire, TireSet

__all__ = [
    "Car",
    "Engine",
    "Transmission",
    "Aero",
    "Suspension",
    "Chassis",
    "Electronics",
    "TractionControl",
    "ABS",
    "Tire",
    "TireSet",
]
