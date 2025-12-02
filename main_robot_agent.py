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
ROBOT_PORT = 5000  # Robot server port
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
            print("Socket Connected!")
        except Exception as e:
            print(f"Connection failed: {e}")
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
            
            # Waits until robot finishes and replies
            response = self.sock.recv(1024).decode().strip()
            
            if response.startswith("OK"):
                return True
            elif response.startswith("ERROR"):
                print(f"Robot error: {response}")
                return False
            else:
                print(f"Unexpected response: {response}")
                return False
                
        except socket.timeout:
            print("⚠️ Robot response timeout!")
            self.connected = False
            return False
        except Exception as e:
            print(f"⚠️ Communication error: {e}")
            self.connected = False
            return False

# --- SETUP SYSTEMS ---
print("Initializing Systems...")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-robotics-er-1.5-preview"
print("🤖 Using Gemini Robotics-ER 1.5 - Purpose-built for robotics!")

try:
    matrix = np.load("calibration_matrix.npy")
    print("✅ Calibration matrix loaded")
except:
    print("Error: Run calibration.py first!")
    exit()

print(f"Connecting to Robot at {ROBOT_IP}:{ROBOT_PORT}...")
try:
    mc = CustomRobotClient(ROBOT_IP, ROBOT_PORT)
    mc.connect()
    print("Robot Connected.")
    mc.set_color(0, 255, 255) # Cyan = Connected
except Exception as e:
    print(f"Robot Connection Failed: {e}")
    mc = None

# --- HELPER FUNCTIONS ---
def is_within_workspace(x, y, z=150):
    """Basic soft limits for safety."""
    X_MIN, X_MAX = -200, 200
    Y_MIN, Y_MAX = 0, 250
    Z_MIN, Z_MAX = 20, 200
    
    return (X_MIN <= x <= X_MAX and 
            Y_MIN <= y <= Y_MAX and 
            Z_MIN <= z <= Z_MAX)

def scale_normalized_coords(y_norm, x_norm, height=480, width=640):
    """[0–1000] → pixel coords."""
    x_pixel = int((x_norm / 1000.0) * width)
    y_pixel = int((y_norm / 1000.0) * height)
    return x_pixel, y_pixel

def get_real_coords(u, v):
    """Pixels → robot XY using homography."""
    try:
        pixel_point = np.array([[[u, v]]], dtype='float32')
        real = cv2.perspectiveTransform(pixel_point, matrix)
        x, y = real[0][0]
        
        if is_within_workspace(x, y):
            return x, y
        else:
            print(f"⚠️ Coordinates ({x:.1f}, {y:.1f}) outside safe workspace!")
            return None
    except Exception as e:
        print(f"Error transforming coordinates: {e}")
        return None

def find_object_robotics(frame, object_name):
    """Call Gemini Robotics-ER to detect object center."""
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
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        # Parse JSON from model text
        response_text = response.text.strip()
        
        # Strip markdown code fences if any
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        
        results = json.loads(response_text)
        
        if results and len(results) > 0:
            result = results[0]
            y_norm, x_norm = result["point"]
            label = result.get("label", object_name)
            
            # Normalized → pixels
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
        print(f"❌ Error in robotics model: {e}")
        return {"found": False, "error": str(e)}

def listen_for_command():
    """Capture short voice command."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("\n🎤 Speak now...")
        try:
            audio = r.listen(source, timeout=7, phrase_time_limit=5)
            return r.recognize_google(audio)
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
    """Simple pick-like motion at (x, y)."""
    if mc is None: 
        print("⚠️ Robot not connected!")
        return False

    print(f"🦾 Moving to X={real_x:.1f}, Y={real_y:.1f}")
    
    # Hover above target
    if not mc.send_coords([real_x, real_y, 150, -170, 0, -90], 40):
        print("❌ Failed to move to hover position")
        return False
    
    # Reached target
    mc.set_color(0, 255, 0)
    time.sleep(0.5)
    
    # Descend
    if not mc.send_coords([real_x, real_y, 50, -170, 0, -90], 30):
        print("❌ Failed to descend")
        return False
    
    time.sleep(0.5)
    
    # Lift
    if not mc.send_coords([real_x, real_y, 150, -170, 0, -90], 40):
        print("❌ Failed to lift")
        return False
    
    # Idle
    mc.set_color(0, 255, 255)
    print("✅ Action Complete.")
    return True

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Failed to open camera!")
    exit()

print("\n" + "="*50)
print("🤖 ROBOT VISION SYSTEM READY")
print("   Powered by Gemini Robotics-ER 1.5")
print("="*50)
print("Controls:")
print("  ENTER - Activate voice command")
print("  Q     - Quit")
print("="*50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame")
        break
    
    # Status overlay
    status_text = "Ready | Press ENTER to speak"
    if mc and not mc.connected:
        status_text = "⚠️ Robot Disconnected"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(frame, "Robotics-ER Model", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.imshow("Robot Eyes", frame)
    
    key = cv2.waitKey(1)
    
    if key == 13:  # ENTER key
        cmd = listen_for_command()
        if not cmd: 
            continue
        
        print(f"📝 You said: '{cmd}'")
        
        # Thinking state
        if mc: mc.set_color(255, 255, 255)
        
        print("🔍 Analyzing with Gemini Robotics-ER 1.5...")
        result = find_object_robotics(frame, cmd)
        
        if result.get("found"):
            x_pixel, y_pixel = result["pixel_coords"]
            y_norm, x_norm = result["normalized_coords"]
            label = result["label"]
            
            print(f"✅ Object found: {label}")
            print(f"   Normalized: y={y_norm}, x={x_norm} (0-1000)")
            print(f"   Pixels: ({x_pixel}, {y_pixel})")
            
            # Mark target on frame
            cv2.circle(frame, (x_pixel, y_pixel), 15, (0, 255, 0), 2)
            cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 255, 0), -1)
            cv2.line(frame, (x_pixel-25, y_pixel), (x_pixel+25, y_pixel), (0, 255, 0), 2)
            cv2.line(frame, (x_pixel, y_pixel-25), (x_pixel, y_pixel+25), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_pixel+20, y_pixel-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Robot Eyes", frame)
            cv2.waitKey(1000)
            
            # Pixels → robot
            robot_result = get_real_coords(x_pixel, y_pixel)
            if robot_result:
                rx, ry = robot_result
                print(f"📍 Robot coordinates: X={rx:.1f}mm, Y={ry:.1f}mm")
                execute_action(rx, ry)
            else:
                print("❌ Target outside safe workspace!")
                if mc: 
                    mc.set_color(255, 0, 0)
                    time.sleep(1)
                    mc.set_color(0, 255, 255)
        else:
            error_msg = result.get("error", "Not found")
            print(f"❌ {error_msg}")
            if mc: 
                mc.set_color(255, 0, 0)
                time.sleep(1)
                mc.set_color(0, 255, 255)
    
    elif key == ord('q'):
        print("\n👋 Shutting down...")
        break

cap.release()
cv2.destroyAllWindows()

if mc and mc.connected:
    mc.set_color(0, 0, 255)  # Blue = idle
    
print("✅ System shutdown complete.")
