import socket
import json
import time
from pymycobot.mycobot280 import MyCobot280

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyAMA0' 
BAUD_RATE = 1000000

print(f"Initializing MyCobot280 on {SERIAL_PORT}...")
try:
    # Initialize MyCobot280 (not MyCobot!)
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    
    # CRITICAL: Check and set fresh mode EXACTLY as in documentation
    if mc.get_fresh_mode() != 1:
        mc.set_fresh_mode(1)
        time.sleep(0.5)
    
    # Power on the robot
    mc.power_on()
    time.sleep(2)
    
    # Check if powered on
    if mc.is_power_on():
        print("✅ Motors powered ON")
    else:
        print("⚠️ Motors not powered - trying again...")
        mc.power_on()
        time.sleep(2)
    
    # Set initial LED color
    mc.set_color(0, 0, 255)  # Blue = idle
    
    print("✅ Robot Ready & Powered On.")
    print(f"   Fresh Mode: {mc.get_fresh_mode()}")
    
    # Get and print current position
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

# --- SAFETY LIMITS ---
def is_valid_coords(coords):
    """Validate coordinates"""
    if len(coords) < 6:
        print(f"⚠️ Invalid coords length: {len(coords)}, need 6")
        return False
    
    x, y, z = coords[0], coords[1], coords[2]
    rx, ry, rz = coords[3], coords[4], coords[5]
    
    # From documentation: x,y,z range is -280 ~ 280, rx,ry,rz range is -314 ~ 314
    if not (-280 <= x <= 280):
        print(f"⚠️ X={x} out of range [-280, 280]")
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

# --- SERVER SETUP ---
HOST = "0.0.0.0" 
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"🚀 Server Listening on Port {PORT}...")

while True:
    print("\nWaiting for Laptop...")
    mc.set_color(0, 0, 255)  # Blue = waiting
    
    conn, addr = server.accept()
    print(f"✅ Connected: {addr}")
    
    mc.set_color(0, 255, 255)  # Cyan = connected

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
                
                print(f"\n📥 Move Command Received:")
                print(f"   Coords: {coords}")
                print(f"   Speed: {speed}")
                
                # Validate
                if not is_valid_coords(coords):
                    print(f"❌ Invalid coordinates")
                    conn.send(b"ERROR:OUT_OF_RANGE\n")
                    continue
                
                # Turn YELLOW
                mc.set_color(255, 255, 0)
                print(f"🟡 Executing movement...")
                
                # SEND COORDS - Mode 0 for angular interpolation
                # This is the EXACT way from documentation
                mc.send_coords(coords, speed, 0)
                
                # Wait a bit for movement to start
                time.sleep(0.3)
                
                # Check if robot is actually moving
                print(f"⏳ Checking movement status...")
                start_time = time.time()
                was_moving = False
                
                while time.time() - start_time < 15:  # 15 second timeout
                    if mc.is_moving():
                        if not was_moving:
                            print(f"   ✅ Robot started moving!")
                            was_moving = True
                        time.sleep(0.1)
                    elif was_moving:
                        # Was moving, now stopped = done
                        print(f"   ✅ Movement completed!")
                        break
                    else:
                        # Never started moving
                        time.sleep(0.1)
                
                if not was_moving:
                    print(f"   ⚠️ Robot did not move!")
                    print(f"   Possible reasons:")
                    print(f"   - Target same as current position")
                    print(f"   - Position unreachable (inverse kinematics failed)")
                    print(f"   - Motors not engaged")
                
                # Get final position
                try:
                    final = mc.get_coords()
                    print(f"   Final position: {final}")
                except:
                    pass
                
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
    mc.set_color(0, 0, 255)  # Back to blue