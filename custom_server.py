import socket
import json
import time
from pymycobot.mycobot import MyCobot

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyAMA0' 
BAUD_RATE = 1000000

print(f"Initializing Robot on {SERIAL_PORT}...")
try:
    mc = MyCobot(SERIAL_PORT, BAUD_RATE)
    
    # CRITICAL: Properly power on and initialize
    mc.power_on()
    time.sleep(2)  # Give motors time to engage
    
    # CRITICAL: Enable fresh mode for real-time control (mode 1)
    mc.set_fresh_mode(1)
    time.sleep(0.5)
    
    # Check if robot is powered
    if mc.is_power_on():
        print("✅ Motors powered ON")
    else:
        print("⚠️ Motors may not be powered!")
    
    # Set initial LED color
    mc.set_color(0, 0, 255)  # Blue = idle
    
    print("✅ Robot Ready & Powered On.")
    print(f"   Fresh Mode: Enabled")
    
    # Print current position for debugging
    try:
        current_coords = mc.get_coords()
        print(f"   Current Coords: {current_coords}")
    except:
        print("   (Could not read current position)")
    
except Exception as e:
    print(f"❌ Failed to connect to motors: {e}")
    exit()

# --- SAFETY LIMITS ---
def is_valid_coords(coords):
    """Validate coordinates are within safe workspace"""
    if len(coords) < 6:
        print(f"⚠️ Invalid coords length: {len(coords)}, need 6")
        return False
    
    x, y, z = coords[0], coords[1], coords[2]
    rx, ry, rz = coords[3], coords[4], coords[5]
    
    # MyCobot 280 safe workspace (from official docs)
    X_MIN, X_MAX = -280, 280
    Y_MIN, Y_MAX = -280, 280
    Z_MIN, Z_MAX = -280, 280
    
    # Rotation angles (in degrees)
    R_MIN, R_MAX = -180, 180
    
    if not (X_MIN <= x <= X_MAX):
        print(f"⚠️ X={x} out of range [{X_MIN}, {X_MAX}]")
        return False
    if not (Y_MIN <= y <= Y_MAX):
        print(f"⚠️ Y={y} out of range [{Y_MIN}, {Y_MAX}]")
        return False
    if not (Z_MIN <= z <= Z_MAX):
        print(f"⚠️ Z={z} out of range [{Z_MIN}, {Z_MAX}]")
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
    print("Waiting for Laptop...")
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
                
                print(f"\n📥 Received move command:")
                print(f"   Target: X={coords[0]:.1f}, Y={coords[1]:.1f}, Z={coords[2]:.1f}")
                print(f"   Angles: RX={coords[3]:.1f}, RY={coords[4]:.1f}, RZ={coords[5]:.1f}")
                print(f"   Speed: {speed}")
                
                # Validate coordinates
                if not is_valid_coords(coords):
                    print(f"❌ Rejected unsafe coordinates")
                    conn.send(b"ERROR:OUT_OF_RANGE\n")
                    continue
                
                # Turn YELLOW while moving
                mc.set_color(255, 255, 0)
                print(f"🟡 Sending movement command...")
                
                # CRITICAL FIX: send_coords() needs mode parameter!
                # mode=0: angular interpolation (smoother, recommended)
                # mode=1: linear interpolation (straight line path)
                mc.send_coords(coords, speed, 0)
                
                print(f"⏳ Waiting for movement to complete...")
                
                # Wait for movement to complete
                time.sleep(0.5)  # Small delay to let movement start
                
                while mc.is_moving():
                    time.sleep(0.1)
                
                print(f"✅ Movement complete!")
                
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
            print(f"JSON Error: {e}")
            conn.send(b"ERROR:INVALID_JSON\n")
        except Exception as e:
            print(f"❌ Error: {e}")
            conn.send(f"ERROR:{str(e)}\n".encode())
            break
            
    conn.close()
    print("Connection closed.")
    mc.set_color(0, 0, 255)  # Back to blue