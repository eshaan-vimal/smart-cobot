"""
Camera-to-Robot Calibration Script

This script creates a transformation matrix that maps camera pixels to robot coordinates.

BEFORE RUNNING THIS SCRIPT:
1. Run collect_calibration_points.py on the Pi to get robot coordinates
2. Update ROBOT_POINTS below with those values
3. Make sure your camera is positioned to see the same workspace

HOW IT WORKS:
- You click 4 corners on the camera feed (in order: TL, TR, BR, BL)
- The script creates a homography matrix that maps any pixel to robot X,Y coordinates
- The matrix is saved to calibration_matrix.npy
"""

import cv2
import numpy as np

# ============================================================================
# ROBOT COORDINATES - UPDATE THESE WITH VALUES FROM collect_calibration_points.py
# ============================================================================
# These are the X,Y coordinates (in mm) from the robot's base frame
# for each of the 4 corners of your workspace.
#
# Order: TL (Top-Left), TR (Top-Right), BR (Bottom-Right), BL (Bottom-Left)
# "Top" = further from robot base, "Bottom" = closer to robot base (typically)
#
# IMPORTANT: These must match the physical corners you'll click in the camera!

ROBOT_POINTS = np.float32([
    [204.0, 140.5],    # TL - Top-Left corner (robot X, Y in mm)
    [215.1, -122.2],   # TR - Top-Right corner
    [24.4, -158.0],    # BR - Bottom-Right corner  
    [22.9, 145.0]      # BL - Bottom-Left corner
])

# ============================================================================
# CAMERA SETUP
# ============================================================================
CAMERA_INDEX = 1  # Change to 0 if using built-in/first camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ============================================================================
# CALIBRATION CODE
# ============================================================================

camera_points = []
corner_names = ["TL (Top-Left)", "TR (Top-Right)", "BR (Bottom-Right)", "BL (Bottom-Left)"]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(camera_points) < 4:
            camera_points.append([x, y])
            idx = len(camera_points) - 1
            robot_pt = ROBOT_POINTS[idx]
            print(f"✅ Point {idx+1}/4 [{corner_names[idx]}]: Pixel({x}, {y}) → Robot({robot_pt[0]:.1f}, {robot_pt[1]:.1f})")
        else:
            print("⚠️ Already have 4 points!")

# Initialize camera
print("="*60)
print("📷 CAMERA-TO-ROBOT CALIBRATION")
print("="*60)
print("\nRobot corner coordinates loaded:")
for i, (name, pt) in enumerate(zip(corner_names, ROBOT_POINTS)):
    print(f"  {i+1}. {name}: X={pt[0]:.1f}mm, Y={pt[1]:.1f}mm")

print(f"\nOpening camera {CAMERA_INDEX}...")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"❌ Failed to open camera {CAMERA_INDEX}")
    print("   Trying camera 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera found!")
        exit()

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse_callback)

print("\n" + "="*60)
print("📋 INSTRUCTIONS")
print("="*60)
print("Click the 4 corners of your workspace IN ORDER:")
print("  1. TOP-LEFT     (TL) - typically far-left from camera view")
print("  2. TOP-RIGHT    (TR) - typically far-right from camera view")
print("  3. BOTTOM-RIGHT (BR) - typically near-right from camera view")
print("  4. BOTTOM-LEFT  (BL) - typically near-left from camera view")
print("")
print("These must correspond to the same physical corners where")
print("you recorded the robot coordinates!")
print("")
print("Press 'R' to reset and start over")
print("Press 'Q' to quit without saving")
print("="*60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    # Draw clicked points and connect them
    for i, point in enumerate(camera_points):
        # Draw point
        color = (0, 255, 0) if i < len(camera_points) else (128, 128, 128)
        cv2.circle(frame, (int(point[0]), int(point[1])), 8, color, -1)
        cv2.circle(frame, (int(point[0]), int(point[1])), 10, color, 2)
        
        # Label
        label = f"{i+1}: {corner_names[i].split()[0]}"
        cv2.putText(frame, label, (int(point[0])+15, int(point[1])-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw line to previous point
        if i > 0:
            cv2.line(frame, (int(camera_points[i-1][0]), int(camera_points[i-1][1])),
                    (int(point[0]), int(point[1])), (0, 200, 0), 2)
    
    # Close the quadrilateral if we have 4 points
    if len(camera_points) == 4:
        cv2.line(frame, (int(camera_points[3][0]), int(camera_points[3][1])),
                (int(camera_points[0][0]), int(camera_points[0][1])), (0, 200, 0), 2)
    
    # Instructions overlay
    if len(camera_points) < 4:
        next_corner = corner_names[len(camera_points)]
        cv2.putText(frame, f"Click: {next_corner}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "All points captured! Calculating...", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Points: {len(camera_points)}/4 | R:Reset | Q:Quit", 
               (10, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Calibration", frame)
    
    # Check if we have all 4 points
    if len(camera_points) == 4:
        print("\n" + "="*60)
        print("🧮 CALCULATING HOMOGRAPHY")
        print("="*60)
        
        pts_camera = np.float32(camera_points)
        
        # Calculate the perspective transform matrix
        # This maps camera pixels → robot coordinates
        matrix = cv2.getPerspectiveTransform(pts_camera, ROBOT_POINTS)
        
        print("\nTransformation matrix:")
        print(matrix)
        
        # Test the transformation on the clicked points
        print("\nVerification (clicked points → robot coords):")
        for i, (cam_pt, robot_pt) in enumerate(zip(camera_points, ROBOT_POINTS)):
            # Transform the camera point
            test_pt = np.array([[[cam_pt[0], cam_pt[1]]]], dtype='float32')
            result = cv2.perspectiveTransform(test_pt, matrix)
            rx, ry = result[0][0]
            
            # Calculate error
            error_x = abs(rx - robot_pt[0])
            error_y = abs(ry - robot_pt[1])
            
            print(f"  {corner_names[i]}: ({rx:.1f}, {ry:.1f}) vs expected ({robot_pt[0]:.1f}, {robot_pt[1]:.1f}) | Error: ({error_x:.2f}, {error_y:.2f})")
        
        # Save the matrix
        np.save("calibration_matrix.npy", matrix)
        print("\n✅ Saved transformation matrix to 'calibration_matrix.npy'")
        print("="*60)
        
        # Wait a moment to show the final result
        cv2.waitKey(2000)
        break
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n❌ Calibration cancelled by user")
        break
    elif key == ord('r'):
        camera_points = []
        print("\n🔄 Reset - click the 4 corners again")

cap.release()
cv2.destroyAllWindows()

if len(camera_points) == 4:
    print("\n🎉 Calibration complete!")
    print("   You can now run calibration_communication.py to test the mapping.")
else:
    print("\n⚠️ Calibration incomplete - no matrix saved")
