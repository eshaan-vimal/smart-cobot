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
    mc.power_on()
    time.sleep(1)
    mc.set_color(0, 0, 255)  # Blue = idle
    print("Robot Ready & Powered On.")
except Exception as e:
    print(f"Failed to connect to motors: {e}")
    exit()

# --- SAFETY LIMITS ---
def is_valid_coords(coords):
    """Check basic workspace limits"""
    if len(coords) < 3: return False
    x, y, z = coords[0], coords[1], coords[2]
    X_MIN, X_MAX = -200, 200
    Y_MIN, Y_MAX = 0, 280
    Z_MIN, Z_MAX = 20, 250
    if not (X_MIN <= x <= X_MAX): return False
    if not (Y_MIN <= y <= Y_MAX): return False
    if not (Z_MIN <= z <= Z_MAX): return False
    return True

# --- SERVER ---
HOST = "0.0.0.0"
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"🚀 Server Listening on {PORT}...")

while True:
    print("Waiting for Laptop...")
    mc.set_color(0, 0, 255)
    
    conn, addr = server.accept()
    print(f"Connected: {addr}")
    mc.set_color(0, 255, 255)  # Cyan = connected

    while True:
        try:
            data = conn.recv(1024)
            if not data:
                print("Laptop disconnected.")
                break

            command = json.loads(data.decode().strip())

            if command["cmd"] == "move":
                coords = command["coords"]
                speed = command.get("speed", 30)

                if not is_valid_coords(coords):
                    conn.send(b"ERROR:OUT_OF_RANGE\n")
                    continue

                mc.set_color(255, 255, 0)  # Yellow = moving
                mc.send_coords(coords, speed, 1)

                while mc.is_moving():  # Wait for completion
                    time.sleep(0.1)

                conn.send(b"OK:COMPLETE\n")
                print("Move Complete.")

            elif command["cmd"] == "color":
                mc.set_color(command.get("r", 0),
                             command.get("g", 0),
                             command.get("b", 0))
                conn.send(b"OK\n")

            else:
                conn.send(b"ERROR:UNKNOWN_CMD\n")

        except json.JSONDecodeError:
            conn.send(b"ERROR:INVALID_JSON\n")
        except Exception as e:
            conn.send(f"ERROR:{e}\n".encode())
            break
            
    conn.close()
    mc.set_color(0, 0, 255)  # Blue = idle
    print("Connection closed.")
