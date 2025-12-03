"""
Calibration & Communication Test

Uses the precomputed camera-to-robot calibration matrix to:
- Convert clicked camera pixels to robot coordinates
- Send movement commands to the robot over a TCP socket
- Visually verify alignment between clicks and robot end-effector
"""

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
ROBOT_PORT = 5000

# Safe Z height - adjust based on your setup
# This should be high enough that the robot can reach all X,Y positions
SAFE_Z_HEIGHT = 50  # mm above the base

# End-effector orientation (adjust if robot can't reach certain areas)
# These are Euler angles in degrees
# Try different values if robot struggles to reach positions
END_EFFECTOR_RX = 180   # Roll
END_EFFECTOR_RY = 0     # Pitch  
END_EFFECTOR_RZ = -90   # Yaw (pointing direction)

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
            self.sock.settimeout(30.0)  # Longer timeout for movements
            self.sock.connect((self.ip, self.port))
            self.connected = True
            print("✅ Socket Connected!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
    def reconnect(self):
        """Try to reconnect if disconnected"""
        print("🔄 Attempting to reconnect...")
        try:
            if self.sock:
                self.sock.close()
        except:
            pass
        return self.connect()

    def send_coords(self, coords, speed, mode=0):
        """
        Send coordinates to robot
        coords: [X, Y, Z, RX, RY, RZ]
        speed: 1-100
        mode: 0=angular interpolation, 1=linear interpolation
        """
        cmd = {"cmd": "move", "coords": coords, "speed": speed, "mode": mode}
        return self._send(cmd)

    def set_color(self, r, g, b):
        cmd = {"cmd": "color", "r": r, "g": g, "b": b}
        return self._send(cmd, timeout=5.0)
    
    def get_coords(self):
        """Get current robot coordinates"""
        cmd = {"cmd": "get_coords"}
        response = self._send_and_receive(cmd)
        if response and response.get("status") == "OK":
            return response.get("coords")
        return None
    
    def ping(self):
        """Check if robot is responsive"""
        cmd = {"cmd": "ping"}
        return self._send(cmd, timeout=5.0)

    def _send(self, data, timeout=30.0):
        if not self.connected:
            print("⚠️ Not connected to robot!")
            return False
        
        try:
            old_timeout = self.sock.gettimeout()
            self.sock.settimeout(timeout)
            
            msg = json.dumps(data)
            self.sock.sendall(msg.encode())
            
            # Wait for response
            response = self.sock.recv(1024).decode().strip()
            
            self.sock.settimeout(old_timeout)
            
            if response.startswith("OK"):
                return True
            elif response.startswith("ERROR"):
                print(f"❌ Robot error: {response}")
                return False
            else:
                print(f"⚠️ Unexpected response: {response}")
                return True  # Assume OK if not explicitly error
                
        except socket.timeout:
            print("⚠️ Robot response timeout (this is OK, movement may still complete)")
            # Don't set connected=False, just return True to continue
            return True
        except Exception as e:
            print(f"⚠️ Communication error: {e}")
            self.connected = False
            return False
    
    def _send_and_receive(self, data, timeout=10.0):
        """Send command and return parsed JSON response"""
        if not self.connected:
            return None
        
        try:
            old_timeout = self.sock.gettimeout()
            self.sock.settimeout(timeout)
            
            msg = json.dumps(data)
            self.sock.sendall(msg.encode())
            
            response = self.sock.recv(1024).decode().strip()
            self.sock.settimeout(old_timeout)
            
            return json.loads(response)
        except:
            return None

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
mc = None
try:
    mc = CustomRobotClient(ROBOT_IP, ROBOT_PORT)
    if mc.connect():
        mc.set_color(0, 255, 255)  # Cyan = Ready
        
        # Get and display current position
        current = mc.get_coords()
        if current:
            print(f"   Current position: X={current[0]:.1f}, Y={current[1]:.1f}, Z={current[2]:.1f}")
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
    x, y = real[0][0]
    return float(x), float(y)

def is_within_workspace(x, y, z=SAFE_Z_HEIGHT):
    """Check if coordinates are safe for the robot"""
    # Adjusted workspace - MyCobot 280 has ~280mm reach
    X_MIN, X_MAX = -250, 250
    Y_MIN, Y_MAX = -250, 250
    Z_MIN, Z_MAX = 50, 300
    
    in_bounds = (X_MIN <= x <= X_MAX and 
                 Y_MIN <= y <= Y_MAX and 
                 Z_MIN <= z <= Z_MAX)
    
    if not in_bounds:
        print(f"⚠️ WARNING: Coordinates ({x:.1f}, {y:.1f}, {z}) outside safe workspace!")
        print(f"   Safe ranges: X[{X_MIN}, {X_MAX}], Y[{Y_MIN}, {Y_MAX}], Z[{Z_MIN}, {Z_MAX}]")
    
    # Also check if point is within reach (rough circular check)
    reach = np.sqrt(x**2 + y**2)
    if reach > 280:
        print(f"⚠️ WARNING: Point may be outside robot reach (distance={reach:.1f}mm, max≈280mm)")
        return False
    
    if reach < 100:
        print(f"⚠️ WARNING: Point may be too close to robot base (distance={reach:.1f}mm)")
        return False
    
    return in_bounds

def move_robot_to_point(rx, ry):
    """Move robot's end effector to the specified coordinates"""
    global mc
    
    if mc is None or not mc.connected:
        print("⚠️ Robot not connected - skipping movement")
        return False
    
    # Check safety first
    if not is_within_workspace(rx, ry, SAFE_Z_HEIGHT):
        print("❌ Unsafe coordinates - movement cancelled!")
        return False
    
    print(f"\n🤖 Moving robot to X={rx:.1f}mm, Y={ry:.1f}mm, Z={SAFE_Z_HEIGHT}mm")
    print(f"   Orientation: RX={END_EFFECTOR_RX}, RY={END_EFFECTOR_RY}, RZ={END_EFFECTOR_RZ}")
    
    # Turn LED yellow during movement
    mc.set_color(255, 255, 0)
    
    # Build coordinate array [X, Y, Z, RX, RY, RZ]
    target_coords = [rx, ry, SAFE_Z_HEIGHT, END_EFFECTOR_RX, END_EFFECTOR_RY, END_EFFECTOR_RZ]
    
    # Send movement command
    # mode=0: angular interpolation (robot finds its own path)
    # mode=1: linear interpolation (straight line - may fail if obstacles)
    success = mc.send_coords(target_coords, speed=40, mode=0)
    
    if success:
        mc.set_color(0, 255, 0)  # Green = reached
        print("✅ Movement command sent!")
        time.sleep(1.5)  # Hold for visual verification
        mc.set_color(0, 255, 255)  # Back to cyan
        return True
    else:
        mc.set_color(255, 0, 0)  # Red = error
        print("❌ Movement failed!")
        time.sleep(1)
        
        # Try to reconnect if disconnected
        if not mc.connected:
            if mc.reconnect():
                mc.set_color(0, 255, 255)
            
        return False

# --- MOUSE CALLBACK ---
clicked_points = []
current_frame = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print("\n" + "="*60)
        print(f"🔍 CLICK DETECTED")
        print("="*60)
        print(f"Pixel coordinates: ({x}, {y})")
        
        # Transform to robot coordinates
        rx, ry = get_real_coords(x, y)
        print(f"Robot coordinates: X={rx:.1f}mm, Y={ry:.1f}mm")
        
        # Calculate distance from robot base
        reach = np.sqrt(rx**2 + ry**2)
        print(f"Distance from base: {reach:.1f}mm")
        
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
        print("💡 TIP: Check if the robot's end-effector aligns with where you clicked!")
        print("="*60)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Failed to open camera!")
    print("   Trying camera index 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera found!")
        exit()

cv2.namedWindow("Calibration Test")
cv2.setMouseCallback("Calibration Test", mouse_callback)

print("\n" + "="*60)
print("🎯 INSTRUCTIONS")
print("="*60)
print("1. Click anywhere on the video feed")
print("2. The robot will move its end-effector to that location")
print("3. Verify visually if alignment is correct")
print("4. Click multiple points to test different areas")
print("")
print("KEYS:")
print("  Q - Quit")
print("  C - Clear all marked points")
print("  R - Reconnect to robot")
print("  P - Print current robot position")
print("="*60)
print(f"\n📐 Using Z height: {SAFE_Z_HEIGHT}mm")
print(f"📐 End-effector angles: RX={END_EFFECTOR_RX}, RY={END_EFFECTOR_RY}, RZ={END_EFFECTOR_RZ}")
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
    cv2.putText(current_frame, "Click to test | Q:Quit | C:Clear | R:Reconnect", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(current_frame, f"Points tested: {len(clicked_points)}", 
               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Connection status
    if mc and mc.connected:
        cv2.putText(current_frame, "Robot: Connected", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(current_frame, "Robot: Disconnected (press R)", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imshow("Calibration Test", current_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n👋 Shutting down...")
        break
    elif key == ord('c'):
        clicked_points = []
        print("\n🧹 Cleared all marked points")
    elif key == ord('r'):
        print("\n🔄 Reconnecting to robot...")
        if mc:
            if mc.reconnect():
                mc.set_color(0, 255, 255)
                print("✅ Reconnected!")
            else:
                print("❌ Reconnection failed")
        else:
            mc = CustomRobotClient(ROBOT_IP, ROBOT_PORT)
            if mc.connect():
                mc.set_color(0, 255, 255)
                print("✅ Connected!")
    elif key == ord('p'):
        if mc and mc.connected:
            coords = mc.get_coords()
            if coords:
                print(f"\n📍 Current position: X={coords[0]:.1f}, Y={coords[1]:.1f}, Z={coords[2]:.1f}")
                print(f"   Orientation: RX={coords[3]:.1f}, RY={coords[4]:.1f}, RZ={coords[5]:.1f}")
            else:
                print("\n⚠️ Could not read position")

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
        reach = np.sqrt(rx**2 + ry**2)
        print(f"  {i}. Pixel ({px}, {py}) → Robot ({rx:.1f}, {ry:.1f}) [reach: {reach:.0f}mm]")
print("="*60)
print("✅ Test complete!")
