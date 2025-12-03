"""
Robot Vision + Voice Command Controller

Combines:
- Camera feed + Gemini vision to locate named objects
- Voice or typed commands to specify the target
- Homography calibration to convert pixels → robot coordinates
- Socket control of the robot to execute a pick-like motion
"""

import cv2
import numpy as np
import time
import os
import socket
import json
from PIL import Image
import speech_recognition as sr
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
ROBOT_IP = os.getenv("ROBOT_IP") 
ROBOT_PORT = 5000
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Robot movement parameters - adjust based on your setup
SAFE_Z_HEIGHT = 200      # Hover height (mm) - high enough to clear objects
PICK_Z_HEIGHT = 80       # Descent height for picking (mm)
MOVEMENT_SPEED = 40      # Speed for horizontal movements (1-100)
DESCENT_SPEED = 30       # Speed for vertical movements (1-100)

# End-effector orientation (Euler angles in degrees)
# Adjust if robot can't reach certain positions
END_EFFECTOR_RX = 180
END_EFFECTOR_RY = 0
END_EFFECTOR_RZ = -90

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
            print("⚠️ Robot response timeout (movement may still complete)")
            # Don't disconnect on timeout - movement might still be in progress
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

# --- SETUP SYSTEMS ---
print("="*60)
print("🤖 INITIALIZING ROBOT VISION SYSTEM")
print("="*60)

# Initialize Gemini client
print("\nInitializing Gemini...")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.0-flash"  # gemini-2.0-flash or gemini-robotics-er-1.5-preview
print(f"✅ Using model: {MODEL_ID}")

# Load calibration homography
print("\nLoading calibration...")
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
        mc.set_color(0, 255, 255)  # Cyan = Connected
        
        # Get and display current position
        current = mc.get_coords()
        if current:
            print(f"   Current position: X={current[0]:.1f}, Y={current[1]:.1f}, Z={current[2]:.1f}")
except Exception as e:
    print(f"❌ Robot connection failed: {e}")
    print("   System will run without robot control.")
    mc = None

# --- HELPER FUNCTIONS ---
def is_within_workspace(x, y, z=SAFE_Z_HEIGHT):
    """Check if coordinates are within safe robot workspace"""
    X_MIN, X_MAX = -250, 250
    Y_MIN, Y_MAX = -250, 250
    Z_MIN, Z_MAX = 50, 300
    
    in_bounds = (X_MIN <= x <= X_MAX and 
                 Y_MIN <= y <= Y_MAX and 
                 Z_MIN <= z <= Z_MAX)
    
    if not in_bounds:
        print(f"⚠️ Coordinates ({x:.1f}, {y:.1f}, {z}) outside safe workspace!")
        return False
    
    # Check if within robot's reach (circular workspace)
    reach = np.sqrt(x**2 + y**2)
    if reach > 280:
        print(f"⚠️ Point too far from robot base (reach={reach:.1f}mm, max≈280mm)")
        return False
    
    if reach < 100:
        print(f"⚠️ Point too close to robot base (reach={reach:.1f}mm)")
        return False
    
    return True

def scale_normalized_coords(y_norm, x_norm, height=480, width=640):
    """Convert normalized [0-1000] coordinates to pixel coordinates"""
    x_pixel = int((x_norm / 1000.0) * width)
    y_pixel = int((y_norm / 1000.0) * height)
    return x_pixel, y_pixel

def get_real_coords(u, v):
    """Convert pixel coordinates to robot XY using homography matrix"""
    try:
        pixel_point = np.array([[[u, v]]], dtype='float32')
        real = cv2.perspectiveTransform(pixel_point, matrix)
        x, y = float(real[0][0][0]), float(real[0][0][1])
        
        if is_within_workspace(x, y):
            return x, y
        else:
            return None
    except Exception as e:
        print(f"❌ Error transforming coordinates: {e}")
        return None

def find_object_robotics(frame, object_name):
    """Use Gemini to detect object center in the frame"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    prompt = f"""
    Find the {object_name} in this image. Return the center point.
    The answer should follow the json format: [{{"point": [y, x], "label": <label>}}].
    The points are in [y, x] format normalized to 0-1000.
    If the object is not found, return an empty list [].
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                temperature=0.5,
            )
        )
        
        response_text = response.text.strip()
        
        # Strip markdown code fences if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        
        results = json.loads(response_text)
        
        if results and len(results) > 0:
            result = results[0]
            y_norm, x_norm = result["point"]
            label = result.get("label", object_name)
            
            x_pixel, y_pixel = scale_normalized_coords(y_norm, x_norm)
            
            return {
                "found": True,
                "pixel_coords": (x_pixel, y_pixel),
                "normalized_coords": (y_norm, x_norm),
                "label": label
            }
        else:
            return {"found": False}
            
    except Exception as e:
        print(f"❌ Error in vision model: {e}")
        return {"found": False, "error": str(e)}

def listen_for_command():
    """Capture voice command using speech recognition"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("\n🎤 Listening... Speak now!")
        try:
            audio = r.listen(source, timeout=7, phrase_time_limit=5)
            text = r.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            print("⚠️ No speech detected")
            return None
        except sr.UnknownValueError:
            print("⚠️ Could not understand audio")
            return None
        except Exception as e:
            print(f"⚠️ Speech recognition error: {e}")
            return None

def execute_action(real_x, real_y):
    """Execute a pick-like motion at the target coordinates"""
    global mc
    
    if mc is None or not mc.connected:
        print("⚠️ Robot not connected!")
        return False

    print(f"\n🦾 Executing action at X={real_x:.1f}mm, Y={real_y:.1f}mm")
    
    # Build coordinate arrays with consistent orientation
    hover_coords = [real_x, real_y, SAFE_Z_HEIGHT, END_EFFECTOR_RX, END_EFFECTOR_RY, END_EFFECTOR_RZ]
    pick_coords = [real_x, real_y, PICK_Z_HEIGHT, END_EFFECTOR_RX, END_EFFECTOR_RY, END_EFFECTOR_RZ]
    
    # Step 1: Move to hover position above target
    print(f"   → Moving to hover position (Z={SAFE_Z_HEIGHT}mm)...")
    mc.set_color(255, 255, 0)  # Yellow = moving
    
    if not mc.send_coords(hover_coords, MOVEMENT_SPEED, mode=0):
        print("❌ Failed to move to hover position")
        mc.set_color(255, 0, 0)
        return False
    
    time.sleep(0.5)
    mc.set_color(0, 255, 0)  # Green = reached hover
    
    # Step 2: Descend to pick height
    print(f"   → Descending to pick height (Z={PICK_Z_HEIGHT}mm)...")
    mc.set_color(255, 165, 0)  # Orange = descending
    
    if not mc.send_coords(pick_coords, DESCENT_SPEED, mode=0):
        print("❌ Failed to descend")
        mc.set_color(255, 0, 0)
        return False
    
    time.sleep(0.5)
    
    # Step 3: Gripper action would go here
    # mc.set_gripper_state(1, 50)  # Close gripper
    # time.sleep(0.5)
    
    # Step 4: Lift back up
    print(f"   → Lifting (Z={SAFE_Z_HEIGHT}mm)...")
    mc.set_color(255, 255, 0)  # Yellow = moving
    
    if not mc.send_coords(hover_coords, MOVEMENT_SPEED, mode=0):
        print("❌ Failed to lift")
        mc.set_color(255, 0, 0)
        return False
    
    mc.set_color(0, 255, 255)  # Cyan = ready
    print("✅ Action complete!")
    return True

def try_reconnect():
    """Attempt to reconnect to the robot"""
    global mc
    if mc:
        if mc.reconnect():
            mc.set_color(0, 255, 255)
            print("✅ Reconnected to robot!")
            return True
    else:
        mc = CustomRobotClient(ROBOT_IP, ROBOT_PORT)
        if mc.connect():
            mc.set_color(0, 255, 255)
            print("✅ Connected to robot!")
            return True
    return False

def process_command(frame, cmd):
    """Process a voice or text command"""
    global mc
    
    # Set thinking color
    if mc and mc.connected:
        mc.set_color(255, 255, 255)  # White = thinking
    
    print("🔍 Analyzing image frame...")
    result = find_object_robotics(frame, cmd)
    
    if result.get("found"):
        x_pixel, y_pixel = result["pixel_coords"]
        y_norm, x_norm = result["normalized_coords"]
        label = result["label"]
        
        print(f"✅ Found: {label}")
        print(f"   Normalized: y={y_norm}, x={x_norm}")
        print(f"   Pixels: ({x_pixel}, {y_pixel})")
        
        # Draw target on frame
        display = frame.copy()
        cv2.circle(display, (x_pixel, y_pixel), 20, (0, 255, 0), 2)
        cv2.circle(display, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
        cv2.line(display, (x_pixel-30, y_pixel), (x_pixel+30, y_pixel), (0, 255, 0), 2)
        cv2.line(display, (x_pixel, y_pixel-30), (x_pixel, y_pixel+30), (0, 255, 0), 2)
        cv2.putText(display, label, (x_pixel+25, y_pixel-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Robot Vision", display)
        cv2.waitKey(1500)
        
        # Convert to robot coordinates
        robot_result = get_real_coords(x_pixel, y_pixel)
        if robot_result:
            rx, ry = robot_result
            reach = np.sqrt(rx**2 + ry**2)
            print(f"🎯 Robot target: X={rx:.1f}mm, Y={ry:.1f}mm (reach: {reach:.0f}mm)")
            
            execute_action(rx, ry)
        else:
            print("❌ Target outside safe workspace!")
            if mc and mc.connected:
                mc.set_color(255, 0, 0)  # Red = error
                time.sleep(1)
                mc.set_color(0, 255, 255)
    else:
        error_msg = result.get("error", "Object not found in image")
        print(f"❌ {error_msg}")
        if mc and mc.connected:
            mc.set_color(255, 0, 0)  # Red = not found
            time.sleep(1)
            mc.set_color(0, 255, 255)

# --- MAIN LOOP ---
print("\nInitializing camera...")
cap = cv2.VideoCapture(1)  # Try camera index 1 first (external camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("   Camera 1 not found, trying camera 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera found!")
        exit()

print("✅ Camera ready")

print("\n" + "="*60)
print("🤖 ROBOT VISION SYSTEM READY")
print("="*60)
print("Controls:")
print("  ENTER  - Activate voice command")
print("  T      - Type command manually")
print("  P      - Print current robot position")
print("  R      - Reconnect to robot")
print("  Q      - Quit")
print("="*60)
print(f"\nMovement parameters:")
print(f"  Hover height: {SAFE_Z_HEIGHT}mm")
print(f"  Pick height:  {PICK_Z_HEIGHT}mm")
print(f"  Orientation:  RX={END_EFFECTOR_RX}, RY={END_EFFECTOR_RY}, RZ={END_EFFECTOR_RZ}")
print("="*60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame")
        break
    
    display_frame = frame.copy()
    
    # Status overlay
    if mc and mc.connected:
        status_text = "Ready | ENTER: Voice | T: Type | Q: Quit"
        status_color = (0, 255, 0)
        cv2.putText(display_frame, "Robot: Connected", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        status_text = "Robot Disconnected | R: Reconnect | Q: Quit"
        status_color = (0, 0, 255)
        cv2.putText(display_frame, "Robot: Disconnected (press R)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.putText(display_frame, status_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    cv2.imshow("Robot Vision", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # ENTER key - voice command
    if key == 13:
        cmd = listen_for_command()
        if cmd:
            print(f"\n🔍 Command: '{cmd}'")
            process_command(frame, cmd)
    
    # T key - type command
    elif key == ord('t'):
        print("\n⌨️ Type your command:")
        cmd = input(">>> ").strip()
        if cmd:
            print(f"\n🔍 Command: '{cmd}'")
            process_command(frame, cmd)
    
    # P key - print position
    elif key == ord('p'):
        if mc and mc.connected:
            coords = mc.get_coords()
            if coords:
                print(f"\n📍 Current position:")
                print(f"   X={coords[0]:.1f}, Y={coords[1]:.1f}, Z={coords[2]:.1f}")
                print(f"   RX={coords[3]:.1f}, RY={coords[4]:.1f}, RZ={coords[5]:.1f}")
            else:
                print("\n⚠️ Could not read position")
        else:
            print("\n⚠️ Robot not connected")
    
    # R key - reconnect
    elif key == ord('r'):
        print("\n🔄 Reconnecting to robot...")
        try_reconnect()
    
    # Q key - quit
    elif key == ord('q'):
        print("\n👋 Shutting down...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

if mc and mc.connected:
    mc.set_color(0, 0, 255)  # Blue = idle

print("✅ System shutdown complete.")
