import cv2
import numpy as np
import time
import os
import socket
import json
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
ROBOT_IP = os.getenv("ROBOT_IP") 
ROBOT_PORT = 9000

# --- CUSTOM CLIENT ---
class CustomRobotClient:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connected = False
    
    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10.0)
            self.sock.connect((self.ip, self.port))
            self.connected = True
            print("✅ Socket Connected!")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            raise

    def send_coords(self, coords, speed):
        cmd = {"cmd": "move", "coords": coords, "speed": speed}
        return self._send(cmd)

    def set_color(self, r, g, b):
        cmd = {"cmd": "color", "r": r, "g": g, "b": b}
        return self._send(cmd)

    def _send(self, data):
        if not self.connected:
            print("⚠️ Not connected to robot!")
            return False
        
        try:
            msg = json.dumps(data)
            self.sock.sendall(msg.encode())
            
            # Wait for response (blocks until movement is done!)
            response = self.sock.recv(1024).decode().strip()
            
            if response.startswith("OK"):
                return True
            elif response.startswith("ERROR"):
                print(f"❌ Robot error: {response}")
                return False
            else:
                print(f"⚠️ Unexpected response: {response}")
                return False
                
        except socket.timeout:
            print("⚠️ Robot response timeout!")
            self.connected = False
            return False
        except Exception as e:
            print(f"⚠️ Communication error: {e}")
            self.connected = False
            return False

# --- SETUP ---
print("="*60)
print("🧪 CALIBRATION & COMMUNICATION TEST")
print("="*60)
print("This script tests:")
print("  1. Perspective transformation (pixel → robot coords)")
print("  2. Socket communication to robot")
print("  3. Physical alignment (visual verification)")
print("="*60)

# Load calibration matrix
try:
    matrix = np.load("calibration_matrix.npy")
    print("✅ Calibration matrix loaded")
except:
    print("❌ Error: calibration_matrix.npy not found!")
    print("   Run calibration.py first!")
    exit()

# Connect to robot
print(f"\nConnecting to robot at {ROBOT_IP}:{ROBOT_PORT}...")
try:
    mc = CustomRobotClient(ROBOT_IP, ROBOT_PORT)
    mc.connect()
    mc.set_color(0, 255, 255)  # Cyan = Ready
except Exception as e:
    print(f"❌ Robot connection failed: {e}")
    print("\nYou can still test the coordinate transformation,")
    print("but robot movement will be skipped.")
    mc = None

# --- HELPER FUNCTIONS ---
def get_real_coords(u, v):
    """Convert pixel coordinates to robot coordinates"""
    pixel_point = np.array([[[u, v]]], dtype='float32')
    real = cv2.perspectiveTransform(pixel_point, matrix)
    # Convert numpy float32 to Python float for JSON serialization
    x, y = real[0][0]
    return float(x), float(y)

def is_within_workspace(x, y, z=150):
    """Check if coordinates are safe for the robot"""
    X_MIN, X_MAX = -200, 200
    Y_MIN, Y_MAX = 0, 250
    Z_MIN, Z_MAX = 20, 200
    
    in_bounds = (X_MIN <= x <= X_MAX and 
                 Y_MIN <= y <= Y_MAX and 
                 Z_MIN <= z <= Z_MAX)
    
    if not in_bounds:
        print(f"⚠️ WARNING: Coordinates ({x:.1f}, {y:.1f}, {z}) outside safe workspace!")
        print(f"   Safe ranges: X[{X_MIN}, {X_MAX}], Y[{Y_MIN}, {Y_MAX}], Z[{Z_MIN}, {Z_MAX}]")
    
    return in_bounds

def move_robot_to_point(rx, ry):
    """Move robot's end effector (6th axis/palm) to the specified coordinates"""
    if mc is None:
        print("⚠️ Robot not connected - skipping movement")
        return False
    
    # Check safety first
    if not is_within_workspace(rx, ry):
        print("❌ Unsafe coordinates - movement cancelled!")
        return False
    
    print(f"\n🤖 Moving robot to X={rx:.1f}mm, Y={ry:.1f}mm, Z=150mm")
    
    # Turn LED yellow during movement
    mc.set_color(255, 255, 0)
    
    # Move to the point (Z=150 for hover height, angles for straight down orientation)
    # Format: [X, Y, Z, rx, ry, rz] where rx, ry, rz are joint angles
    success = mc.send_coords([rx, ry, 150, -170, 0, -90], 40)
    
    if success:
        # Turn LED green when reached
        mc.set_color(0, 255, 0)
        print("✅ Robot reached target position!")
        time.sleep(2)  # Hold for visual verification
        
        # Back to cyan (ready)
        mc.set_color(0, 255, 255)
        return True
    else:
        # Turn LED red on error
        mc.set_color(255, 0, 0)
        time.sleep(1)
        mc.set_color(0, 255, 255)
        return False

# --- MOUSE CALLBACK ---
clicked_points = []
current_frame = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print("\n" + "="*60)
        print(f"📍 CLICK DETECTED")
        print("="*60)
        print(f"Pixel coordinates: ({x}, {y})")
        
        # Transform to robot coordinates
        rx, ry = get_real_coords(x, y)
        print(f"Robot coordinates: X={rx:.1f}mm, Y={ry:.1f}mm")
        
        # Store the point
        clicked_points.append({
            'pixel': (x, y),
            'robot': (rx, ry)
        })
        
        # Draw on frame
        if current_frame is not None:
            cv2.circle(current_frame, (x, y), 10, (0, 255, 0), 2)
            cv2.circle(current_frame, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(current_frame, f"({rx:.0f}, {ry:.0f})", 
                       (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Move robot to the point
        print("\nSending command to robot...")
        move_robot_to_point(rx, ry)
        
        print("="*60)
        print("💡 TIP: Check if the robot's palm aligns with where you clicked!")
        print("="*60)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Failed to open camera!")
    exit()

cv2.namedWindow("Calibration Test")
cv2.setMouseCallback("Calibration Test", mouse_callback)

print("\n" + "="*60)
print("🎯 INSTRUCTIONS")
print("="*60)
print("1. Click anywhere on the video feed")
print("2. The robot will move its palm to that location")
print("3. Verify visually if the robot's palm aligns with your click")
print("4. Click multiple points to test different areas")
print("5. Press 'Q' to quit")
print("6. Press 'C' to clear all marked points")
print("="*60)
print("\n⏳ Starting camera feed...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame")
        break
    
    current_frame = frame.copy()
    
    # Draw all previously clicked points
    for point_data in clicked_points:
        px, py = point_data['pixel']
        rx, ry = point_data['robot']
        cv2.circle(current_frame, (px, py), 10, (0, 255, 0), 2)
        cv2.circle(current_frame, (px, py), 3, (0, 255, 0), -1)
        cv2.putText(current_frame, f"({rx:.0f}, {ry:.0f})", 
                   (px+15, py-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add instructions overlay
    cv2.putText(current_frame, "Click to test | Q: Quit | C: Clear", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(current_frame, f"Points tested: {len(clicked_points)}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if mc and mc.connected:
        cv2.putText(current_frame, "Robot: Connected", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    else:
        cv2.putText(current_frame, "Robot: Disconnected", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    cv2.imshow("Calibration Test", current_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n👋 Shutting down...")
        break
    elif key == ord('c'):
        clicked_points = []
        print("\n🧹 Cleared all marked points")

cap.release()
cv2.destroyAllWindows()

if mc and mc.connected:
    mc.set_color(0, 0, 255)  # Blue = idle

print("\n" + "="*60)
print("📊 TEST SUMMARY")
print("="*60)
print(f"Total points tested: {len(clicked_points)}")
if clicked_points:
    print("\nTested coordinates:")
    for i, point_data in enumerate(clicked_points, 1):
        px, py = point_data['pixel']
        rx, ry = point_data['robot']
        print(f"  {i}. Pixel ({px}, {py}) → Robot ({rx:.1f}, {ry:.1f})")
print("="*60)
print("✅ Test complete!")