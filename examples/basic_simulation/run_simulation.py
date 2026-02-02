#!/usr/bin/env python3
"""
Basic Simulation Example

This example demonstrates how to:
1. Generate a random track
2. Create a car and place it on the track
3. Run a simulation loop with manual inputs
4. Access car telemetry data

Run with: python run_simulation.py
"""

from racenet import Simulator, Car
from racenet.track import TrackGenerator
from racenet.car import CarInputs


def main():
    print("=" * 60)
    print("RaceNet Basic Simulation Example")
    print("=" * 60)
    
    # Step 1: Generate a track
    print("\n1. Generating track...")
    generator = TrackGenerator()
    track = generator.generate_with_seed(42, "Example Circuit")
    
    print(f"   Track: {track.config.name}")
    print(f"   Length: {track.length:.0f} meters")
    print(f"   Segments: {track.num_segments}")
    print(f"   Sectors: {track.config.num_sectors}")
    
    # Step 2: Create simulator and spawn car
    print("\n2. Setting up simulation...")
    sim = Simulator()
    sim.set_track(track)
    car_ids = sim.spawn_cars(1)
    car_id = car_ids[0]
    
    print(f"   Spawned car with ID: {car_id}")
    
    # Step 3: Run simulation
    print("\n3. Running simulation (500 steps at 100Hz = 5 seconds)...")
    sim.start()
    
    # Simulate acceleration and steering
    for step in range(500):
        # Example: Full throttle, slight steering
        if step < 200:
            inputs = CarInputs(throttle=1.0, brake=0.0, steering=0.0)
        elif step < 300:
            # Light braking and turning
            inputs = CarInputs(throttle=0.3, brake=0.3, steering=0.2)
        else:
            # Accelerate out of corner
            inputs = CarInputs(throttle=0.8, brake=0.0, steering=0.1)
        
        observations = sim.step({car_id: inputs})
        
        # Print status every 100 steps
        if (step + 1) % 100 == 0:
            car = sim.get_car(car_id)
            print(f"   Step {step + 1}: Speed = {car.speed_kph:.1f} km/h, "
                  f"Gear = {car.transmission.current_gear}, "
                  f"RPM = {car.engine.rpm:.0f}")
    
    # Step 4: Get final telemetry
    print("\n4. Final telemetry snapshot:")
    car = sim.get_car(car_id)
    telemetry = car.get_telemetry()
    
    print(f"   Position: ({telemetry['state']['x']:.1f}, {telemetry['state']['y']:.1f})")
    print(f"   Speed: {telemetry['state']['speed_kph']:.1f} km/h")
    print(f"   Engine RPM: {telemetry['engine']['rpm']:.0f}")
    print(f"   Engine Temp: {telemetry['engine']['temperature_c']:.1f}°C")
    print(f"   Gear: {telemetry['transmission']['gear']}")
    print(f"   Lateral G: {telemetry['state']['lateral_g']:.2f}g")
    print(f"   Longitudinal G: {telemetry['state']['longitudinal_g']:.2f}g")
    print(f"   Tire Avg Temp: {telemetry['tires']['avg_temp_c']:.1f}°C")
    
    # Step 5: Get simulation statistics
    print("\n5. Simulation statistics:")
    print(f"   Simulation time: {sim.time:.2f} seconds")
    print(f"   Total telemetry frames: {len(sim.get_telemetry())}")
    
    sim.stop()
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
