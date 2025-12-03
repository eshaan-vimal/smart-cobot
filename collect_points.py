"""
Collects robot corner coordinates on the Raspberry Pi for workspace calibration.

Steps:
1. Run on the Pi connected to MyCobot
2. Manually move the end-effector to each workspace corner
3. Press Enter to record each position
4. Paste printed coordinates into calibration.py
"""

import time
from pymycobot.mycobot import MyCobot

# --- SERIAL CONFIGURATION ---
SERIAL_PORT = '/dev/ttyAMA0' 
BAUD_RATE = 1000000

print("="*60)
print("🤖 ROBOT CALIBRATION POINT COLLECTOR")
print("="*60)

# Initialize robot and enable manual positioning
print(f"\nInitializing robot on {SERIAL_PORT}...")
try:
    mc = MyCobot(SERIAL_PORT, BAUD_RATE)
    mc.power_on()
    time.sleep(2)
    
    if mc.is_power_on():
        print("✅ Robot powered on")
    
    # Let user freely move the arm
    print("\n🔓 Releasing servos - you can now manually move the arm")
    mc.release_all_servos()
    time.sleep(0.5)
    
except Exception as e:
    print(f"❌ Failed to initialize robot: {e}")
    exit()

# Corner labels and storage
corners = ["TOP-LEFT (TL)", "TOP-RIGHT (TR)", "BOTTOM-RIGHT (BR)", "BOTTOM-LEFT (BL)"]
collected_points = []

print("\n" + "="*60)
print("📋 INSTRUCTIONS")
print("="*60)
print("You need to collect 4 corner points of your workspace.")
print("For each corner:")
print("  1. Manually move the robot arm so the end-effector")
print("     is directly above that corner")
print("  2. Press ENTER to record the position")
print("  3. The robot will briefly lock to confirm the position")
print("")
print("The corners should match where you'll click in the camera view.")
print("="*60)

for i, corner in enumerate(corners):
    print(f"\n{'='*60}")
    print(f"📍 POINT {i+1}/4: {corner}")
    print("="*60)
    
    mc.release_all_servos()
    mc.set_color(255, 255, 0)  # Yellow = waiting for input
    
    input(f"Move robot to {corner} corner, then press ENTER...")
    
    # Briefly lock servo to stabilize reading
    mc.focus_servo(1)  # This might not work on all versions
    time.sleep(0.3)
    
    # Sample multiple readings and average for robustness
    readings = []
    for _ in range(3):
        coords = mc.get_coords()
        if coords and len(coords) >= 2:
            readings.append(coords)
        time.sleep(0.1)
    
    if readings:
        avg_x = sum(r[0] for r in readings) / len(readings)
        avg_y = sum(r[1] for r in readings) / len(readings)
        avg_z = sum(r[2] for r in readings) / len(readings)
        
        collected_points.append({
            'name': corner,
            'x': avg_x,
            'y': avg_y,
            'z': avg_z,
            'full': readings[-1]  # Keep last full reading
        })
        
        mc.set_color(0, 255, 0)  # Green = recorded
        print(f"✅ Recorded: X={avg_x:.1f}, Y={avg_y:.1f}, Z={avg_z:.1f}")
        
    else:
        print("❌ Failed to read coordinates!")
        mc.set_color(255, 0, 0)
    
    time.sleep(0.5)
    mc.release_all_servos()

# Print summary and helper snippet for calibration.py
print("\n" + "="*60)
print("📊 CALIBRATION POINTS COLLECTED")
print("="*60)

if len(collected_points) == 4:
    print("\n✅ All 4 points collected successfully!\n")
    
    print("Copy this into your calibration.py file:\n")
    print("-"*60)
    print("ROBOT_POINTS = np.float32([")
    for point in collected_points:
        print(f"    [{point['x']:.1f}, {point['y']:.1f}],    # {point['name']}")
    print("])")
    print("-"*60)
    
    print("\nFull coordinates (for reference):")
    for point in collected_points:
        full = point['full']
        print(f"  {point['name']}: [{full[0]:.1f}, {full[1]:.1f}, {full[2]:.1f}, {full[3]:.1f}, {full[4]:.1f}, {full[5]:.1f}]")
    
    print("\n💡 Note: The Z values above show the height at each corner.")
    print("   Use a consistent Z height in calibration_communication.py")
    print(f"   Suggested SAFE_Z_HEIGHT: {max(p['z'] for p in collected_points) + 50:.0f}mm")
    
else:
    print(f"\n❌ Only collected {len(collected_points)}/4 points")
    print("Please run again and collect all 4 corners.")

# Final status
mc.set_color(0, 0, 255)  # Blue = idle
print("\n✅ Done!")
