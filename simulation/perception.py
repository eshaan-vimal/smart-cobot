from ultralytics import YOLO
import numpy as np


class Perception:
    def __init__(self, model_name='yolov8n.pt', target_class_id=75):
        """Initializes the YOLOv8 perception module.
        
        Args:
            model_name: Path or name of YOLO model
            target_class_id: Class ID to search for (default 75 is 'vase' in COCO, which the teddy is detected as)
        """
        print(f"Loading perception model '{model_name}'...")
        self.model = YOLO(model_name)
        self.target_class_id = int(target_class_id)
        print(f"Perception model loaded. Target class ID: {self.target_class_id}")

    def find_target(self, rgb_image):
        """Finds the target object in an RGB image.
        
        Args:
            rgb_image: numpy array (H, W, 3) in RGB format
            
        Returns:
            Dictionary {'center': (u, v), 'bbox': [x1, y1, x2, y2]} or None
        """
        try:
            results = self.model(rgb_image, verbose=False)
            
            for result in results:
                class_names = getattr(result, 'names', {})
                
                if len(result.boxes) == 0:
                    continue
                
                # Debug: print all detections this frame
                print("=== YOLO Detections This Frame ===")
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names.get(cls_id, 'Unknown')
                    conf = float(box.conf[0])
                    print(f"  Found: {cls_name} (ID {cls_id}) confidence={conf:.2f}")
                
                # Look for target class
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls_id == self.target_class_id:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy
                        
                        u = (x1 + x2) // 2
                        v = (y1 + y2) // 2
                        
                        cls_name = class_names.get(cls_id, 'Unknown')
                        print(f"*** TARGET FOUND: {cls_name} at pixel ({u}, {v}) conf={conf:.2f} ***")
                        
                        return {
                            'center': (int(u), int(v)),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'conf': conf
                        }
            
            print(f"... Target class ID {self.target_class_id} not found in this frame.")
            return None
        
        except Exception as e:
            print(f"Perception error: {e}")
            return None