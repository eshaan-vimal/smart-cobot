import time
import os
import cv2
import numpy as np

from simulation import Simulation
from controller import RobotController
from utils import pixel_to_world, detect_object


def main_loop():
    """Main visual servoing loop with HSV-based red object detection."""
    sim = Simulation(headless=False)
    robot = None
    
    try:
        # Initialize simulation
        print("\n=== INITIALIZING ===")
        sim.load_assets()
        sim.setup_camera(width=640, height=480)
        robot = RobotController(sim.robot_id, sim.end_effector_link_index)
        
        # Get target ground truth position
        true_target_pos = sim.get_target_position()
        print(f"\nGround truth target position: {true_target_pos}\n")
        
        # Capture and save initial debug frame
        print("=== Capturing Initial Frame ===")
        rgb_img, depth_buf = sim.get_camera_image()
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "initial_view.jpg") 
        cv2.imwrite(save_path, bgr_img)
        print("Saved 'initial_view.jpg'.\n")
        
        print("=== Starting Visual Servoing Loop ===")
        print("The robot will move toward the red teddy bear.")
        print("Press Ctrl+C to stop.\n")
        
        frame_count = 0
        last_valid_target = None
        consecutive_failures = 0
        
        while True:
            frame_count += 1
            
            # 1. PERCEIVE: Capture image from camera
            rgb_image, depth_buffer = sim.get_camera_image()
            
            # 2. FIND: Use HSV to detect RED teddy bear (much more reliable)
            target_info = detect_object(rgb_image)
            
            if target_info is None:
                consecutive_failures += 1
                print(f"[Frame {frame_count}] Red object NOT detected. ({consecutive_failures} consecutive failures)")
                
                # If we detected it before, use last known position
                if last_valid_target is not None and consecutive_failures < 10:
                    print(f"[Frame {frame_count}] Using last known target position.")
                    target_info = last_valid_target
                else:
                    print(f"[Frame {frame_count}] Skipping motion.")
                    sim.step()
                    time.sleep(1.0 / 240.0)
                    continue
            else:
                consecutive_failures = 0
                last_valid_target = target_info
            
            # 3. TRANSFORM: Convert pixel to world coordinates
            u, v = target_info['center']
            depth_pixel = float(depth_buffer[min(v, depth_buffer.shape[0]-1), min(u, depth_buffer.shape[1]-1)])
            
            # Validate depth
            if depth_pixel <= 0.0 or depth_pixel > sim.camera_params['far']:
                print(f"[Frame {frame_count}] Invalid depth {depth_pixel:.3f}. Using estimated depth.")
                depth_pixel = 0.5  # Use mid-range depth estimate
            
            world_pos = pixel_to_world(u, v, depth_pixel, sim.camera_params)
            if world_pos is None:
                print(f"[Frame {frame_count}] Unprojection failed.")
                sim.step()
                time.sleep(1.0 / 240.0)
                continue
            
            print(f"[Frame {frame_count}] Pixel:({u},{v}) Depth:{depth_pixel:.3f} World:{world_pos}")
            
            # 4. ACT: Move robot toward target
            # Approach from above (add z-offset to avoid collision)
            z_offset = 0.2
            target_with_offset = (world_pos[0], world_pos[1], world_pos[2] + z_offset)
            
            success = robot.move_to_target(
                target_with_offset, 
                target_orientation_euler=[0, -np.pi, 0]
            )
            
            if success:
                # Get current end-effector position
                ee_pos = robot.get_end_effector_pos()
                distance = np.linalg.norm(np.array(ee_pos) - np.array(target_with_offset))
                print(f"  EE Position: {ee_pos}")
                print(f"  Distance to target: {distance:.3f}\n")
            else:
                print(f"  IK FAILED\n")
            
            # Step simulation multiple times for smoother motion
            for _ in range(10):
                sim.step()
                time.sleep(1.0 / 2400.0)
    
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")
    
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if sim:
            sim.close()


if __name__ == "__main__":
    main_loop()