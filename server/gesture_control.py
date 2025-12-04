"""
Gesture control for MyCobot 280 in velocity mode using laptop webcam + hand gestures.
Open palm = velocity control, fist = stop, pointing = pick, peace = home.
Keyboard: G toggle, H home, C corner test, +/- Z adjust, R reconnect, Q quit.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import socket
import json
from collections import deque
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
ROBOT_IP = os.getenv("ROBOT_IP")
ROBOT_PORT = 5000

# Gesture control parameters
SMOOTHING_WINDOW = 5          # Hand position smoothing
COMMAND_RATE_LIMIT = 0.1      # Command interval (s)
ACTIVATION_TIME = 0.5         # Open-palm hold time (s)
GESTURE_CONFIDENCE = 0.7      # Min detection confidence

# ============================================================================
# VELOCITY CONTROL PARAMETERS
# ============================================================================
DEAD_ZONE = 0.1               # Neutral zone around image center
MAX_VELOCITY = 15             # Max mm per update
VELOCITY_CURVE = "quadratic"  # "linear" or "quadratic"

# ============================================================================
# ROBOT WORKSPACE LIMITS
# ============================================================================
ROBOT_X_MIN, ROBOT_X_MAX = -200, 200
ROBOT_Y_MIN, ROBOT_Y_MAX = -150, 150
ROBOT_Z_DEFAULT = 180
ROBOT_Z_MIN, ROBOT_Z_MAX = 100, 280

# Starting position
HOME_X, HOME_Y = 100, 0

# End-effector orientation
END_EFFECTOR_RX = 180
END_EFFECTOR_RY = 0
END_EFFECTOR_RZ = -90

# Discrete action speed
ACTION_SPEED = 50

# ============================================================================
# AXIS MAPPING
# ============================================================================
X_DIRECTION = 1    # Flip X if needed
Y_DIRECTION = -1   # Flip Y if needed
SWAP_XY = False    # Swap X/Y axes

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# --- ROBOT CLIENT ---
class GestureRobotClient:
    """Non-blocking client for gesture-based velocity control."""
    
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connected = False
        self.last_command_time = 0
        self.current_x = HOME_X
        self.current_y = HOME_Y
        self.current_z = ROBOT_Z_DEFAULT
    
    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.ip, self.port))
            self.sock.setblocking(False)
            self.connected = True
            print("✅ Robot connected")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
    def reconnect(self):
        try:
            if self.sock:
                self.sock.close()
        except:
            pass
        return self.connect()
    
    def move_velocity(self, vx, vy):
        """Apply velocity step to current XY and send move."""
        if not self.connected:
            return False
        
        now = time.time()
        if now - self.last_command_time < COMMAND_RATE_LIMIT:
            return True
        
        new_x = self.current_x + vx
        new_y = self.current_y + vy
        
        new_x = max(ROBOT_X_MIN, min(new_x, ROBOT_X_MAX))
        new_y = max(ROBOT_Y_MIN, min(new_y, ROBOT_Y_MAX))
        
        if abs(new_x - self.current_x) < 0.1 and abs(new_y - self.current_y) < 0.1:
            return True
        
        self.current_x = new_x
        self.current_y = new_y
        
        coords = [
            float(self.current_x),
            float(self.current_y),
            float(self.current_z),
            float(END_EFFECTOR_RX),
            float(END_EFFECTOR_RY),
            float(END_EFFECTOR_RZ)
        ]
        
        cmd = {"cmd": "move", "coords": coords, "speed": 80, "mode": 0}
        
        try:
            self.sock.sendall(json.dumps(cmd).encode())
            self.last_command_time = now
            try:
                self.sock.recv(1024)
            except BlockingIOError:
                pass
            return True
        except Exception as e:
            print(f"⚠️ Send error: {e}")
            self.connected = False
            return False
    
    def send_coords_sync(self, coords, speed):
        """Blocking move for pick/home actions."""
        if not self.connected:
            return False
        
        self.current_x = coords[0]
        self.current_y = coords[1]
        self.current_z = coords[2]
        
        cmd = {"cmd": "move", "coords": [float(c) for c in coords], "speed": speed, "mode": 0}
        
        try:
            self.sock.setblocking(True)
            self.sock.settimeout(15.0)
            self.sock.sendall(json.dumps(cmd).encode())
            response = self.sock.recv(1024).decode().strip()
            self.sock.setblocking(False)
            return response.startswith("OK")
        except Exception as e:
            print(f"⚠️ Sync error: {e}")
            self.sock.setblocking(False)
            return False
    
    def set_color(self, r, g, b):
        """Set robot LED color."""
        if not self.connected:
            return
        try:
            cmd = {"cmd": "color", "r": r, "g": g, "b": b}
            self.sock.setblocking(True)
            self.sock.settimeout(2.0)
            self.sock.sendall(json.dumps(cmd).encode())
            try:
                self.sock.recv(1024)
            except:
                pass
            self.sock.setblocking(False)
        except:
            pass
    
    def go_home(self):
        """Go to predefined home pose."""
        coords = [HOME_X, HOME_Y, ROBOT_Z_DEFAULT, END_EFFECTOR_RX, END_EFFECTOR_RY, END_EFFECTOR_RZ]
        return self.send_coords_sync(coords, ACTION_SPEED)

# --- GESTURE DETECTION ---
class GestureDetector:
    """MediaPipe-based hand + gesture detector."""
    
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=GESTURE_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        self.x_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.y_buffer = deque(maxlen=SMOOTHING_WINDOW)
    
    def process_frame(self, frame):
        """Return smoothed palm, finger count, and gesture label."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            self.x_buffer.clear()
            self.y_buffer.clear()
            return None
        
        hand = results.multi_hand_landmarks[0]
        
        wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
        middle_mcp = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        palm_x = (wrist.x + middle_mcp.x) / 2
        palm_y = (wrist.y + middle_mcp.y) / 2
        
        self.x_buffer.append(palm_x)
        self.y_buffer.append(palm_y)
        
        smooth_x = sum(self.x_buffer) / len(self.x_buffer)
        smooth_y = sum(self.y_buffer) / len(self.y_buffer)
        
        fingers_up = self._count_fingers(hand)
        gesture = self._classify_gesture(fingers_up, hand)
        
        return {
            "landmarks": hand,
            "palm_x": smooth_x,
            "palm_y": smooth_y,
            "fingers_up": fingers_up,
            "gesture": gesture
        }
    
    def _count_fingers(self, hand):
        """Simple finger up/down count."""
        tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP]
        pips = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
                mp_hands.HandLandmark.PINKY_PIP]
        
        count = 0
        for i, (tip, pip) in enumerate(zip(tips, pips)):
            if i == 0:  # Thumb
                if abs(hand.landmark[tip].x - hand.landmark[pip].x) > 0.05:
                    count += 1
            else:
                if hand.landmark[tip].y < hand.landmark[pip].y - 0.02:
                    count += 1
        return count
    
    def _classify_gesture(self, fingers_up, hand):
        """Map finger pattern to basic gesture name."""
        if fingers_up >= 4:
            return "OPEN_PALM"
        elif fingers_up == 0:
            return "FIST"
        elif fingers_up == 2:
            index_up = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
                       hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            middle_up = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < \
                        hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            if index_up and middle_up:
                return "PEACE"
        elif fingers_up == 1:
            index_up = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
                       hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y - 0.02
            if index_up:
                return "POINTING"
        return "OTHER"
    
    def draw_hand(self, frame, hand_data):
        """Draw landmarks and palm center."""
        if hand_data is None:
            return
        mp_draw.draw_landmarks(frame, hand_data["landmarks"], mp_hands.HAND_CONNECTIONS,
                               mp_styles.get_default_hand_landmarks_style(),
                               mp_styles.get_default_hand_connections_style())
        h, w = frame.shape[:2]
        cx, cy = int(hand_data["palm_x"] * w), int(hand_data["palm_y"] * h)
        cv2.circle(frame, (cx, cy), 15, (0, 255, 255), 3)

# --- VELOCITY CALCULATION ---
def calculate_velocity(palm_x, palm_y):
    """Map palm offset from center to XY velocity."""
    offset_x = palm_x - 0.5
    offset_y = palm_y - 0.5
    
    if abs(offset_x) < DEAD_ZONE:
        offset_x = 0
    else:
        offset_x = (offset_x - np.sign(offset_x) * DEAD_ZONE) / (0.5 - DEAD_ZONE)
    
    if abs(offset_y) < DEAD_ZONE:
        offset_y = 0
    else:
        offset_y = (offset_y - np.sign(offset_y) * DEAD_ZONE) / (0.5 - DEAD_ZONE)
    
    if VELOCITY_CURVE == "quadratic":
        vx = np.sign(offset_x) * (offset_x ** 2) * MAX_VELOCITY
        vy = np.sign(offset_y) * (offset_y ** 2) * MAX_VELOCITY
    else:
        vx = offset_x * MAX_VELOCITY
        vy = offset_y * MAX_VELOCITY
    
    vx = vx * X_DIRECTION
    vy = vy * Y_DIRECTION
    
    if SWAP_XY:
        vx, vy = vy, vx
    
    return float(vx), float(vy)

# --- MAIN ---
def main():
    print("="*60)
    print("🖐️  GESTURE CONTROL - VELOCITY MODE")
    print("="*60)
    
    print(f"\nConnecting to robot at {ROBOT_IP}:{ROBOT_PORT}...")
    robot = GestureRobotClient(ROBOT_IP, ROBOT_PORT)
    if robot.connect():
        robot.set_color(0, 0, 255)
    else:
        print("⚠️ Running without robot")
    
    detector = GestureDetector()
    print("✅ MediaPipe ready")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Camera failed!")
        return
    print("✅ Camera ready")
    
    gesture_mode = False
    tracking_active = False
    palm_open_start = None
    
    fps_time = time.time()
    fps_counter = 0
    fps = 0
    
    print("\n" + "="*60)
    print("VELOCITY CONTROL MODE")
    print("  Hand in CENTER = Stop")
    print("  Hand offset from center = Move in that direction")
    print("  Further from center = Faster movement")
    print("")
    print("CONTROLS:")
    print("  G - Toggle gesture control")
    print("  H - Home position")
    print("  C - Test corners")
    print("  +/- - Adjust Z")
    print("  R - Reconnect")
    print("  Q - Quit")
    print("")
    print("GESTURES:")
    print("  ✋ Open Palm  - Move robot (velocity control)")
    print("  ✊ Fist       - Stop & hold position")
    print("  ☝️ Pointing   - Pick action (descend & lift)")
    print("  ✌️ Peace      - Go home")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        hand_data = detector.process_frame(frame)
        if hand_data:
            detector.draw_hand(frame, hand_data)
        
        # --- GESTURE CONTROL LOGIC ---
        if gesture_mode and hand_data:
            gesture = hand_data["gesture"]
            palm_x = hand_data["palm_x"]
            palm_y = hand_data["palm_y"]
            
            if gesture == "OPEN_PALM":
                if not tracking_active:
                    if palm_open_start is None:
                        palm_open_start = time.time()
                    elif time.time() - palm_open_start >= ACTIVATION_TIME:
                        tracking_active = True
                        robot.set_color(0, 255, 0)
                        print("🟢 Tracking ACTIVATED - move hand to control robot")
                        palm_open_start = None
                
                if tracking_active:
                    vx, vy = calculate_velocity(palm_x, palm_y)
                    if abs(vx) > 0.1 or abs(vy) > 0.1:
                        robot.move_velocity(vx, vy)
            
            elif gesture == "FIST":
                if tracking_active:
                    tracking_active = False
                    robot.set_color(255, 255, 0)
                    print(f"🟡 STOPPED at X={robot.current_x:.0f}, Y={robot.current_y:.0f}")
                    print("   Reposition hand, then open palm to continue")
                palm_open_start = None
            
            elif gesture == "POINTING" and not tracking_active:
                print("☝️ PICK action!")
                robot.set_color(255, 165, 0)
                
                pick_coords = [robot.current_x, robot.current_y, ROBOT_Z_MIN,
                               END_EFFECTOR_RX, END_EFFECTOR_RY, END_EFFECTOR_RZ]
                robot.send_coords_sync(pick_coords, 40)
                time.sleep(0.5)
                
                lift_coords = [robot.current_x, robot.current_y, robot.current_z,
                               END_EFFECTOR_RX, END_EFFECTOR_RY, END_EFFECTOR_RZ]
                robot.send_coords_sync(lift_coords, 40)
                
                robot.set_color(255, 255, 0)
                print("✅ Pick complete")
            
            elif gesture == "PEACE" and not tracking_active:
                print("✌️ Going HOME")
                robot.set_color(0, 255, 255)
                robot.go_home()
                robot.set_color(255, 255, 0)
            
            else:
                palm_open_start = None
        
        elif gesture_mode and not hand_data:
            if tracking_active:
                tracking_active = False
                robot.set_color(255, 255, 0)
                print("⚠️ Hand lost - stopped")
            palm_open_start = None
        
        # --- UI ---
        cv2.rectangle(frame, (0, 0), (w, 95), (0, 0, 0), -1)
        
        if gesture_mode:
            if tracking_active:
                mode_text = "TRACKING (velocity control)"
                mode_color = (0, 255, 0)
            else:
                mode_text = "GESTURE ON - show palm to start"
                mode_color = (0, 255, 255)
        else:
            mode_text = "GESTURE OFF (press G)"
            mode_color = (128, 128, 128)
        
        cv2.putText(frame, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        conn_color = (0, 255, 0) if robot.connected else (0, 0, 255)
        cv2.putText(frame, f"Robot: {'Connected' if robot.connected else 'Disconnected'}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conn_color, 1)
        
        cv2.putText(frame, f"Pos: X={robot.current_x:.0f} Y={robot.current_y:.0f} Z={robot.current_z:.0f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"FPS: {fps}", (w - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if hand_data:
            cv2.putText(frame, f"{hand_data['gesture']}", (w - 150, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Center / dead zone visuals
        center_x, center_y = w // 2, h // 2
        dead_zone_px = int(DEAD_ZONE * w)
        
        cv2.circle(frame, (center_x, center_y), dead_zone_px, (100, 100, 100), 1)
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
        
        if tracking_active and hand_data:
            vx, vy = calculate_velocity(hand_data["palm_x"], hand_data["palm_y"])
            arrow_scale = 5
            arrow_end_x = int(center_x + vx * arrow_scale)
            arrow_end_y = int(center_y + vy * arrow_scale)
            cv2.arrowedLine(frame, (center_x, center_y), (arrow_end_x, arrow_end_y), 
                           (0, 255, 0), 3, tipLength=0.3)
            cv2.putText(frame, f"V: ({vx:.1f}, {vy:.1f})", (w - 150, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Workspace mini-map
        map_size = 100
        map_x, map_y = w - map_size - 10, h - map_size - 10
        
        cv2.rectangle(frame, (map_x, map_y), (map_x + map_size, map_y + map_size), (40, 40, 40), -1)
        cv2.rectangle(frame, (map_x, map_y), (map_x + map_size, map_y + map_size), (100, 100, 100), 1)
        
        norm_x = (robot.current_x - ROBOT_X_MIN) / (ROBOT_X_MAX - ROBOT_X_MIN)
        norm_y = (robot.current_y - ROBOT_Y_MIN) / (ROBOT_Y_MAX - ROBOT_Y_MIN)
        dot_x = int(map_x + norm_x * map_size)
        dot_y = int(map_y + map_size - norm_y * map_size)
        cv2.circle(frame, (dot_x, dot_y), 5, (0, 255, 0), -1)
        
        home_norm_x = (HOME_X - ROBOT_X_MIN) / (ROBOT_X_MAX - ROBOT_X_MIN)
        home_norm_y = (HOME_Y - ROBOT_Y_MIN) / (ROBOT_Y_MAX - ROBOT_Y_MIN)
        home_dot_x = int(map_x + home_norm_x * map_size)
        home_dot_y = int(map_y + map_size - home_norm_y * map_size)
        cv2.drawMarker(frame, (home_dot_x, home_dot_y), (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
        
        if palm_open_start and not tracking_active:
            progress = min((time.time() - palm_open_start) / ACTIVATION_TIME, 1.0)
            bar_w = int(200 * progress)
            cv2.rectangle(frame, (w//2 - 100, h - 40), (w//2 - 100 + bar_w, h - 25), (0, 255, 0), -1)
            cv2.rectangle(frame, (w//2 - 100, h - 40), (w//2 + 100, h - 25), (255, 255, 255), 2)
        
        cv2.imshow("Gesture Control - Velocity Mode", frame)
        
        # --- KEY HANDLING ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('g'):
            gesture_mode = not gesture_mode
            tracking_active = False
            palm_open_start = None
            if gesture_mode:
                print("🖐️ Gesture control ENABLED")
                robot.set_color(255, 255, 0)
            else:
                print("⏹️ Gesture control DISABLED")
                robot.set_color(0, 0, 255)
        
        elif key == ord('h'):
            print("🏠 Going home...")
            robot.go_home()
            robot.set_color(0, 0, 255)
        
        elif key == ord('c'):
            print("\n🎯 CORNER TEST")
            corners = [
                ("Min X, Min Y", ROBOT_X_MIN + 20, ROBOT_Y_MIN + 20),
                ("Max X, Min Y", ROBOT_X_MAX - 20, ROBOT_Y_MIN + 20),
                ("Max X, Max Y", ROBOT_X_MAX - 20, ROBOT_Y_MAX - 20),
                ("Min X, Max Y", ROBOT_X_MIN + 20, ROBOT_Y_MAX - 20),
                ("Home", HOME_X, HOME_Y),
            ]
            for name, x, y in corners:
                print(f"   {name}: X={x}, Y={y}")
                coords = [x, y, robot.current_z, END_EFFECTOR_RX, END_EFFECTOR_RY, END_EFFECTOR_RZ]
                robot.send_coords_sync(coords, 40)
                time.sleep(1.5)
            print("✅ Corner test done")
        
        elif key == ord('+') or key == ord('='):
            robot.current_z = min(robot.current_z + 20, ROBOT_Z_MAX)
            print(f"📈 Z: {robot.current_z}mm")
        
        elif key == ord('-') or key == ord('_'):
            robot.current_z = max(robot.current_z - 20, ROBOT_Z_MIN)
            print(f"📉 Z: {robot.current_z}mm")
        
        elif key == ord('r'):
            print("🔄 Reconnecting...")
            robot.reconnect()
        
        elif key == ord('q'):
            print("\n👋 Shutting down...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    if robot.connected:
        robot.set_color(0, 0, 255)
    print("✅ Done")

if __name__ == "__main__":
    main()
