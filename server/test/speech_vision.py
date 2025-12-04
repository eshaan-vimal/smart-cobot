"""
Gemini Vision & Speech Test

Opens a camera feed and:
- Takes voice or typed commands describing an object
- Uses Gemini vision to localize that object in the image
- Optionally converts the pixel position to robot coordinates using calibration_matrix.npy
"""

import cv2
import numpy as np
import os
import json
from PIL import Image
import speech_recognition as sr
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- INIT ---
print("="*60)
print("🧪 GEMINI VISION & SPEECH TEST")
print("="*60)

print("\nInitializing Gemini...")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.0-flash"  # Updated to match main agent
print(f"✅ Using model: {MODEL_ID}")

# Load camera→robot matrix if available
print("\nChecking calibration...")
try:
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "..", "calibration_matrix.npy")
    matrix = np.load(path)
    print("✅ Calibration matrix loaded")
    has_calibration = True
except:
    print("⚠️ No calibration matrix found (robot coords won't be shown)")
    has_calibration = False

# --- UTILS ---
def scale_normalized_coords(y_norm, x_norm, height=480, width=640):
    """Convert 0-1000 normalized coords to image pixels"""
    return int((x_norm / 1000.0) * width), int((y_norm / 1000.0) * height)

def get_real_coords(u, v):
    """Convert pixels to robot XY coordinates"""
    if not has_calibration:
        return None
    try:
        p = np.array([[[u, v]]], dtype='float32')
        result = cv2.perspectiveTransform(p, matrix)[0][0]
        return float(result[0]), float(result[1])
    except Exception as e:
        print(f"⚠️ Transform error: {e}")
        return None

def listen_for_command():
    """Capture voice command"""
    r = sr.Recognizer()
    with sr.Microphone() as s:
        r.adjust_for_ambient_noise(s, duration=0.5)
        print("\n🎤 Listening... Speak now!")
        try:
            audio = r.listen(s, timeout=7, phrase_time_limit=5)
            return r.recognize_google(audio)
        except sr.WaitTimeoutError:
            print("⚠️ No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            print("⚠️ Could not understand audio")
            return None
        except Exception as e:
            print(f"⚠️ Speech error: {e}")
            return None

def find_object_robotics(frame, object_name):
    """Use Gemini to locate object in frame"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    prompt = f"""
    Find the {object_name} in this image and return center point.
    JSON format: [{{"point":[y,x], "label":<label>}}]
    Coordinates normalized 0-1000. If not found return [].
    """
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                temperature=0.5,
            )
        )

        text = response.text.strip()
        
        # Strip markdown code fences if present
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.replace("```", "").strip()

        results = json.loads(text)
        if results and len(results) > 0:
            r = results[0]
            y_norm, x_norm = r["point"]
            lbl = r.get("label", object_name)
            x_pix, y_pix = scale_normalized_coords(y_norm, x_norm)
            return {
                "found": True,
                "pixel_coords": (x_pix, y_pix),
                "normalized_coords": (y_norm, x_norm),
                "label": lbl
            }
        return {"found": False}

    except json.JSONDecodeError as e:
        return {"found": False, "error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"found": False, "error": str(e)}

# --- MAIN ---
print("\nInitializing camera...")
cap = cv2.VideoCapture(1)  # Try external camera first
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
print("🤖 VISION & SPEECH TEST MODE")
print("="*60)
print("Controls:")
print("  ENTER - Voice command")
print("  T     - Type command")
print("  Q     - Quit")
print("="*60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame")
        break
    
    display = frame.copy()
    
    # Overlay instructions
    cv2.putText(display, "ENTER: Voice | T: Type | Q: Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if has_calibration:
        cv2.putText(display, "Calibration: Loaded", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        cv2.putText(display, "Calibration: Not found", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    
    cv2.imshow("Vision Test", display)
    
    key = cv2.waitKey(1) & 0xFF
    
    # ENTER key - voice command
    if key == 13:
        cmd = listen_for_command()
        if not cmd:
            continue
        
        print("\n" + "="*60)
        print(f"📝 Command: '{cmd}'")
        print("="*60)
        print("🔍 Analyzing with Gemini...")
        
        result = find_object_robotics(frame, cmd)
        
        if result.get("found"):
            x_pix, y_pix = result["pixel_coords"]
            y_norm, x_norm = result["normalized_coords"]
            lbl = result["label"]

            print(f"✅ Found: {lbl}")
            print(f"   Normalized: y={y_norm}, x={x_norm}")
            print(f"   Pixels: ({x_pix}, {y_pix})")

            # Draw crosshair on target
            result_frame = frame.copy()
            cv2.circle(result_frame, (x_pix, y_pix), 20, (0, 255, 0), 2)
            cv2.circle(result_frame, (x_pix, y_pix), 5, (0, 255, 0), -1)
            cv2.line(result_frame, (x_pix-30, y_pix), (x_pix+30, y_pix), (0, 255, 0), 2)
            cv2.line(result_frame, (x_pix, y_pix-30), (x_pix, y_pix+30), (0, 255, 0), 2)
            cv2.putText(result_frame, lbl, (x_pix+25, y_pix-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show robot coordinates if calibration available
            if has_calibration:
                robot_coords = get_real_coords(x_pix, y_pix)
                if robot_coords:
                    rx, ry = robot_coords
                    reach = np.sqrt(rx**2 + ry**2)
                    print(f"🎯 Robot: X={rx:.1f}mm, Y={ry:.1f}mm (reach: {reach:.0f}mm)")
                    cv2.putText(result_frame, f"Robot: X={rx:.1f}, Y={ry:.1f}",
                                (x_pix+25, y_pix+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.imshow("Vision Test", result_frame)
            cv2.waitKey(3000)
        else:
            err = result.get("error", "Object not found in image")
            print(f"❌ {err}")
            
            result_frame = frame.copy()
            cv2.putText(result_frame, f"NOT FOUND: {cmd}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Vision Test", result_frame)
            cv2.waitKey(2000)
    
    # T key - type command
    elif key == ord('t'):
        print("\n⌨️ Type your command:")
        cmd = input(">>> ").strip()
        if not cmd:
            continue
        
        print("\n" + "="*60)
        print(f"📝 Command: '{cmd}'")
        print("="*60)
        print("🔍 Analyzing with Gemini...")
        
        result = find_object_robotics(frame, cmd)
        
        if result.get("found"):
            x_pix, y_pix = result["pixel_coords"]
            y_norm, x_norm = result["normalized_coords"]
            lbl = result["label"]

            print(f"✅ Found: {lbl}")
            print(f"   Normalized: y={y_norm}, x={x_norm}")
            print(f"   Pixels: ({x_pix}, {y_pix})")

            result_frame = frame.copy()
            cv2.circle(result_frame, (x_pix, y_pix), 20, (0, 255, 0), 2)
            cv2.circle(result_frame, (x_pix, y_pix), 5, (0, 255, 0), -1)
            cv2.line(result_frame, (x_pix-30, y_pix), (x_pix+30, y_pix), (0, 255, 0), 2)
            cv2.line(result_frame, (x_pix, y_pix-30), (x_pix, y_pix+30), (0, 255, 0), 2)
            cv2.putText(result_frame, lbl, (x_pix+25, y_pix-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if has_calibration:
                robot_coords = get_real_coords(x_pix, y_pix)
                if robot_coords:
                    rx, ry = robot_coords
                    reach = np.sqrt(rx**2 + ry**2)
                    print(f"🎯 Robot: X={rx:.1f}mm, Y={ry:.1f}mm (reach: {reach:.0f}mm)")
                    cv2.putText(result_frame, f"Robot: X={rx:.1f}, Y={ry:.1f}",
                                (x_pix+25, y_pix+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.imshow("Vision Test", result_frame)
            cv2.waitKey(3000)
        else:
            err = result.get("error", "Object not found in image")
            print(f"❌ {err}")
            
            result_frame = frame.copy()
            cv2.putText(result_frame, f"NOT FOUND: {cmd}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Vision Test", result_frame)
            cv2.waitKey(2000)

    # Q key - quit
    elif key == ord('q'):
        print("\n👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Test complete!")
