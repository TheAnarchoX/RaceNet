#!/usr/bin/env python3
"""
Telemetry Analysis Example

This example demonstrates how to:
1. Record telemetry during simulation
2. Access real-time telemetry channels
3. Export telemetry to different formats
4. Analyze lap data

Run with: python record_telemetry.py
"""

import os
from pathlib import Path

from racenet import Simulator
from racenet.track import TrackGenerator
from racenet.car import CarInputs
from racenet.telemetry import TelemetryRecorder, TelemetryExporter


def main():
    print("=" * 60)
    print("RaceNet Telemetry Recording Example")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Setup simulation
    print("\n1. Setting up simulation...")
    generator = TrackGenerator()
    track = generator.generate_with_seed(42, "Telemetry Test Track")
    
    sim = Simulator()
    sim.set_track(track)
    car_ids = sim.spawn_cars(1)
    car_id = car_ids[0]
    car = sim.get_car(car_id)
    
    print(f"   Track: {track.config.name} ({track.length:.0f}m)")
    
    # Step 2: Create telemetry recorder
    print("\n2. Setting up telemetry recorder...")
    recorder = TelemetryRecorder(car=car)
    print(f"   Recording {len(recorder.channels)} channels")
    print(f"   Sample rate: {recorder.config.sample_rate_hz} Hz")
    
    # Step 3: Run simulation with telemetry recording
    print("\n3. Running simulation (1000 steps = 10 seconds)...")
    sim.start()
    
    # Simulate a driving pattern
    for step in range(1000):
        # Varied driving pattern
        phase = step % 300
        if phase < 150:
            # Accelerating
            inputs = CarInputs(throttle=1.0, brake=0.0, steering=0.0)
        elif phase < 200:
            # Braking into corner
            inputs = CarInputs(throttle=0.0, brake=0.8, steering=0.0)
        elif phase < 250:
            # Through corner
            inputs = CarInputs(throttle=0.3, brake=0.0, steering=0.3)
        else:
            # Exit corner
            inputs = CarInputs(throttle=0.7, brake=0.0, steering=0.1)
        
        sim.step({car_id: inputs})
        
        # Record telemetry
        recorder.record(sim.time, car.get_telemetry())
        
        # Print progress
        if (step + 1) % 250 == 0:
            print(f"   Step {step + 1}: {car.speed_kph:.1f} km/h")
    
    sim.stop()
    
    # Step 4: Display recorded statistics
    print("\n4. Telemetry Statistics:")
    stats = recorder.get_statistics()
    
    print("\n   Speed (km/h):")
    speed_stats = stats.get("speed_kph", {})
    print(f"     Min: {speed_stats.get('min', 0):.1f}")
    print(f"     Max: {speed_stats.get('max', 0):.1f}")
    print(f"     Mean: {speed_stats.get('mean', 0):.1f}")
    
    print("\n   Engine RPM:")
    rpm_stats = stats.get("rpm", {})
    print(f"     Min: {rpm_stats.get('min', 0):.0f}")
    print(f"     Max: {rpm_stats.get('max', 0):.0f}")
    print(f"     Mean: {rpm_stats.get('mean', 0):.0f}")
    
    print("\n   Tire Temperature (°C):")
    tire_stats = stats.get("tire_temp_fl", {})
    print(f"     FL: {tire_stats.get('last', 0):.1f}")
    
    # Step 5: Export telemetry
    print("\n5. Exporting telemetry...")
    from racenet.telemetry.exporter import ExporterConfig
    exporter = TelemetryExporter(ExporterConfig(output_dir=str(output_dir)))
    
    # Export to different formats
    csv_path = exporter.export_csv(recorder, "telemetry.csv")
    print(f"   CSV: {csv_path}")
    
    json_path = exporter.export_json(recorder, "telemetry.json")
    print(f"   JSON: {json_path}")
    
    npz_path = exporter.export_numpy(recorder, "telemetry.npz")
    print(f"   NumPy: {npz_path}")
    
    # Step 6: Show current channel values
    print("\n6. Current Channel Values:")
    current = recorder.get_current_values()
    for channel_name, value in list(current.items())[:8]:  # First 8 channels
        print(f"   {channel_name}: {value:.2f}")
    
    print("\n" + "=" * 60)
    print(f"Telemetry exported to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
