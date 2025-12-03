import cv2
import numpy as np

# Real robot coordinates of paper corners: TL, TR, BR, BL (mm)
ROBOT_POINTS = np.float32([
    [],   # TL [204.0, 140.5, -11.3, 170.92, 9.57, 167.87]
    [],  # TR [215.1, -122.2, -15.0, 176.84, 4.2, 90.99]
    [],    # BR  [24.4, -158.0, -57.6, -176.25, -2.38, 56.93]
    []      # BL [22.9, 145.0, -64.8, -174.72, -1.76, -138.83]

])

camera_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Captured Point: Pixel({x}, {y})")
        camera_points.append([x, y])

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse_callback)

print("--- CALIBRATION STARTED ---")
print("Click corners in order: TL, TR, BR, BL")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Show clicked points
    for i, point in enumerate(camera_points):
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(i+1), (int(point[0])+10, int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Calibration", frame)
    
    if len(camera_points) == 4:
        print("Calculating Homography...")
        pts_camera = np.float32(camera_points)
        matrix = cv2.getPerspectiveTransform(pts_camera, ROBOT_POINTS)  # Camera → Robot map
        np.save("calibration_matrix.npy", matrix)
        print("Saved to 'calibration_matrix.npy'. Exiting...")
        break
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
