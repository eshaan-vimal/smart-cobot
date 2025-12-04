import cv2
import numpy as np


def pixel_to_world(u, v, depth_pixel, camera_params):
    """Converts a 2D pixel coordinate (u, v) and its depth to a 3D world coordinate.
    
    Args:
        u, v: Pixel coordinates (0-based indexing)
        depth_pixel: Depth value at that pixel (0-1 normalized by PyBullet)
        camera_params: Dictionary with width, height, view_matrix, projection_matrix, near, far
        
    Returns:
        (x, y, z) in world coordinates or None if unprojection fails
    """
    try:
        width = camera_params['width']
        height = camera_params['height']
        
        # Reshape matrices from PyBullet (column-major order)
        proj_matrix_np = np.array(camera_params['projection_matrix']).reshape((4, 4), order='F')
        view_matrix_np = np.array(camera_params['view_matrix']).reshape((4, 4), order='F')
        
        near = camera_params['near']
        far = camera_params['far']
        
        # Convert pixel (u, v) to Normalized Device Coordinates (NDC)
        ndc_x = (2.0 * u / width) - 1.0
        ndc_y = 1.0 - (2.0 * v / height)
        
        # Clip coordinates
        clip_x = ndc_x
        clip_y = ndc_y
        clip_z = (2.0 * depth_pixel) - 1.0
        clip_w = 1.0
        
        clip_coords = np.array([clip_x, clip_y, clip_z, clip_w], dtype=np.float64)
        
        # Inverse projection: clip -> eye space
        inv_proj_matrix = np.linalg.inv(proj_matrix_np)
        eye_coords = inv_proj_matrix.dot(clip_coords)
        
        # Perspective divide
        if abs(eye_coords[3]) < 1e-8:
            return None
        
        eye_coords = eye_coords / eye_coords[3]
        
        # Transform to world coordinates using inverse view matrix
        inv_view_matrix = np.linalg.inv(view_matrix_np)
        world_coords = inv_view_matrix.dot(eye_coords)
        
        return float(world_coords[0]), float(world_coords[1]), float(world_coords[2])
    
    except Exception as e:
        print(f"pixel_to_world error: {e}")
        return None


def detect_object(rgb_image):
    """
    Args:
        rgb_image: numpy array (H, W, 3) in RGB format
        
    Returns:
        Dictionary {'center': (u, v), 'bbox': [x1, y1, x2, y2]} or None
    """
    try:
        # Convert RGB to HSV
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Define range for red color in HSV
        # Red wraps around, so we need two ranges
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Get the largest contour (assume that's the teddy bear)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Require minimum area (filter noise)
        if area < 100:
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Calculate center
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2
        
        print(f"Teddy Bear detected at pixel ({u}, {v}) area={area}")
        
        return {
            'center': (int(u), int(v)),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'conf': min(1.0, area / 5000.0)  # Fake confidence based on area
        }
    
    except Exception as e:
        print(f"Detection error: {e}")
        return None