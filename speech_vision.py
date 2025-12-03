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
print("Initializing Systems...")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-robotics-er-1.5-preview"

# Load camera→robot matrix if available
try:
    matrix = np.load("calibration_matrix.npy")
    print("✅ Calibration matrix loaded")
    has_calibration = True
except:
    print("⚠️ No calibration matrix found")
    has_calibration = False

# --- UTILS ---
def scale_normalized_coords(y_norm, x_norm, height=480, width=640):
    """0-1000 → image pixels"""
    return int((x_norm / 1000.0) * width), int((y_norm / 1000.0) * height)

def get_real_coords(u, v):
    """Pixels → robot XY"""
    if not has_calibration: return None
    p = np.array([[[u, v]]], dtype='float32')
    return cv2.perspectiveTransform(p, matrix)[0][0]

def listen_for_command():
    r = sr.Recognizer()
    with sr.Microphone() as s:
        r.adjust_for_ambient_noise(s)
        print("\n🎤 Speak now...")
        try:
            return r.recognize_google(r.listen(s, timeout=5))
        except:
            return None

def find_object_robotics(frame, object_name):
    """Gemini Robotics-ER object localization"""
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
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )

        text = response.text.strip()
        if text.startswith("```"):  # strip code fences
            text = text.replace("```json", "").replace("```", "").strip()

        results = json.loads(text)
        if results:
            r = results[0]
            y_norm, x_norm = r["point"]
            lbl = r.get("label", object_name)
            x_pix, y_pix = scale_normalized_coords(y_norm, x_norm)
            return {"found": True, "pixel_coords": (x_pix, y_pix),
                    "normalized_coords": (y_norm, x_norm), "label": lbl}
        return {"found": False}

    except Exception as e:
        return {"found": False, "error": str(e)}

# --- MAIN ---
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

print("\n" + "="*60)
print("🤖 GEMINI ROBOTICS-ER TEST MODE")
print("="*60)
print("ENTER = voice command | Q = quit")
print("="*60)

while True:
    ret, frame = cap.read()
    
    cv2.putText(frame, "Press ENTER to speak", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, "Press Q to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("Robotics Vision Test", frame)
    
    if cv2.waitKey(1) == 13:
        cmd = listen_for_command()
        if not cmd:
            print("❌ No command heard")
            continue
        
        print("\n" + "="*60)
        print(f"📝 You said: '{cmd}'")
        print("="*60)
        print("🔍 Analyzing...")
        
        result = find_object_robotics(frame, cmd)
        if result.get("found"):
            x_pix, y_pix = result["pixel_coords"]
            y_norm, x_norm = result["normalized_coords"]
            lbl = result["label"]

            print(f"✅ Found: {lbl}")
            print(f"   Norm: y={y_norm}, x={x_norm}")
            print(f"   Pixel: ({x_pix}, {y_pix})")

            cv2.circle(frame, (x_pix, y_pix), 15, (0,255,0), 2)
            cv2.circle(frame, (x_pix, y_pix), 3, (0,255,0), -1)
            cv2.line(frame, (x_pix-25,y_pix),(x_pix+25,y_pix),(0,255,0),2)
            cv2.line(frame, (x_pix,y_pix-25),(x_pix,y_pix+25),(0,255,0),2)
            cv2.putText(frame, lbl, (x_pix+20,y_pix-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            if has_calibration:
                rx, ry = get_real_coords(x_pix, y_pix)
                print(f"📍 Robot: X={rx:.1f}mm, Y={ry:.1f}mm")
                cv2.putText(frame, f"Robot X={rx:.1f}, Y={ry:.1f}",
                            (x_pix+20,y_pix+20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

            cv2.imshow("Robotics Vision Test", frame)
            cv2.waitKey(3000)
        else:
            err = result.get("error", "Not found")
            print(f"❌ {err}")
            cv2.putText(frame, f"NOT FOUND: {cmd}", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            cv2.imshow("Robotics Vision Test", frame)
            cv2.waitKey(2000)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Test complete!")
