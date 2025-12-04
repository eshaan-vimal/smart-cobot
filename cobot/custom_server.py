"""
Robot Command TCP Server

This script runs on the Raspberry Pi connected to MyCobot.
It listens for commands from the laptop and controls robot movement.

Steps:
1. Run this script on the Pi
2. Connect from laptop using TCP socket on port 5000
3. Send JSON commands: {"cmd": "move", "coords": [...], "speed": 30}
4. The robot moves and returns status messages
"""

import socket
import json
import time
from pymycobot.mycobot280 import MyCobot280

# Robot + serial configuration
SERIAL_PORT = '/dev/ttyAMA0' 
BAUD_RATE = 1000000

print(f"Initializing MyCobot280 on {SERIAL_PORT}...")
try:
    # Initialize MyCobot280 (not MyCobot!)
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    
    mc.power_on()
    time.sleep(2)
    
    mc.set_fresh_mode(1)
    time.sleep(0.5)
    
    if mc.is_power_on():
        print("✅ Motors powered ON")
    else:
        print("⚠️ Motors not powered - trying again...")
        mc.power_on()
        time.sleep(2)
    
    mc.set_color(0, 0, 255)
    print("✅ Robot Ready & Powered On.")
    
    try:
        current_coords = mc.get_coords()
        print(f"   Current Position: {current_coords}")
    except:
        print("   (Could not read current position)")
    
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Safety limits for XYZ workspace
def is_valid_coords(coords):
    if len(coords) < 6:
        return False
    
    x, y, z = coords[0], coords[1], coords[2]
    
    X_MIN, X_MAX = -280, 280
    Y_MIN, Y_MAX = -280, 280
    Z_MIN, Z_MAX = -70, 280
    
    if not (X_MIN <= x <= X_MAX):
        print(f"⚠️ X={x} out of range [{X_MIN}, {X_MAX}]")
        return False
    if not (-280 <= y <= 280):
        print(f"⚠️ Y={y} out of range [-280, 280]")
        return False
    if not (-280 <= z <= 280):
        print(f"⚠️ Z={z} out of range [-280, 280]")
        return False
    if not (-314 <= rx <= 314):
        print(f"⚠️ RX={rx} out of range [-314, 314]")
        return False
    if not (-314 <= ry <= 314):
        print(f"⚠️ RY={ry} out of range [-314, 314]")
        return False
    if not (-314 <= rz <= 314):
        print(f"⚠️ RZ={rz} out of range [-314, 314]")
        return False
    
    return True

# TCP server configuration
HOST = "0.0.0.0" 
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"🚀 Server Listening on Port {PORT}...")

while True:
    print("Waiting for Laptop...")
    mc.set_color(0, 0, 255)
    
    conn, addr = server.accept()
    print(f"✅ Connected: {addr}")
    mc.set_color(0, 255, 255)

    while True:
        try:
            data = conn.recv(1024)
            if not data:
                print("Laptop disconnected.")
                break

            msg = data.decode().strip()
            command = json.loads(msg)
            
            if command["cmd"] == "move":
                coords = command["coords"]
                speed = command.get("speed", 30)
                mode = command.get("mode", 0)  # 0=angular, 1=linear
                
                print(f"\n📥 Received move command:")
                print(f"   Target: X={coords[0]:.1f}, Y={coords[1]:.1f}, Z={coords[2]:.1f}")
                print(f"   Angles: RX={coords[3]:.1f}, RY={coords[4]:.1f}, RZ={coords[5]:.1f}")
                print(f"   Speed: {speed}, Mode: {mode}")
                
                if not is_valid_coords(coords):
                    print(f"❌ Invalid coordinates")
                    conn.send(b"ERROR:OUT_OF_RANGE\n")
                    continue
                
                mc.set_color(255, 255, 0)
                print(f"🟡 Executing movement...")
                
                # Send target pose to robot
                mc.send_coords(coords, speed, mode)
                
                # Wait a bit for movement to start
                time.sleep(0.3)
                
                # Short wait before checking movement state
                time.sleep(0.5)
                
                # Wait for motion to finish, with timeout as safety
                timeout = 10  # seconds
                start_time = time.time()
                while mc.is_moving():
                    if time.time() - start_time > timeout:
                        print("⚠️ Movement timeout - continuing anyway")
                        break
                    time.sleep(0.1)
                
                print(f"✅ Movement complete!")
                mc.set_color(0, 255, 0)
                conn.send(b"OK:COMPLETE\n")
                
            elif command["cmd"] == "color":
                r = command.get("r", 0)
                g = command.get("g", 0)
                b = command.get("b", 0)
                mc.set_color(r, g, b)
                conn.send(b"OK\n")
                
            else:
                conn.send(b"ERROR:UNKNOWN_CMD\n")

        except json.JSONDecodeError as e:
            print(f"❌ JSON Error: {e}")
            conn.send(b"ERROR:INVALID_JSON\n")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            conn.send(f"ERROR:{str(e)}\n".encode())
            break
            
    conn.close()
    print("Connection closed.")
    mc.set_color(0, 0, 255)
