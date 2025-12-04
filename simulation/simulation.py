import pybullet as p
import pybullet_data
import numpy as np


class Simulation:
    def __init__(self, headless=False):
        """Initializes the PyBullet simulation."""
        print("Initializing simulation...")
        if headless:
            self.client = p.connect(p.DIRECT)
        else:
            self.client = p.connect(p.GUI)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setRealTimeSimulation(0)
        
        self.robot_id = None
        self.target_id = None
        self.end_effector_link_index = None
        self.camera_params = {}

    def load_assets(self):
        """Loads the robot and target object into the simulation."""
        print("Loading assets...")
        
        # Load plane
        p.loadURDF("plane.urdf")
        
        # Load KUKA robot arm at origin
        robot_start_pos = [0, 0, 0]
        robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf", 
            robot_start_pos, 
            robot_start_orn, 
            useFixedBase=1
        )
        
        # Get end-effector link index
        num_joints = p.getNumJoints(self.robot_id)
        self.end_effector_link_index = num_joints - 1
        
        # Load target (teddy bear) CLOSER to the robot and LOWER
        # Place it in front of the robot at reachable distance
        target_start_pos = [0.5, 0.3, 0.15]  # X, Y, Z (much closer, lower)
        target_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.target_id = p.loadURDF(
            "teddy_vhacd.urdf", 
            target_start_pos, 
            target_start_orn, 
            globalScaling=3.0  # Make it a bit smaller so YOLO can see it better
        )
        
        print(f"Assets loaded.")
        print(f"  Robot ID: {self.robot_id}")
        print(f"  Target ID: {self.target_id}")
        print(f"  Target position: {target_start_pos}")

    def setup_camera(self, width=640, height=480):
        """Sets up the simulated camera to look directly at the teddy bear area."""
        fov = 60
        aspect = width / height
        near = 0.01
        far = 5.0
        
        # Camera positioned to look directly at the teddy bear
        # Place camera to the side, looking down at the workspace
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.3, -0.8, 0.8],      # Side and above the scene
            cameraTargetPosition=[0.5, 0.3, 0.15],   # Looking at teddy bear location
            cameraUpVector=[0, 0, 1]
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )
        
        self.camera_params = {
            'width': width,
            'height': height,
            'view_matrix': view_matrix,
            'projection_matrix': projection_matrix,
            'near': near,
            'far': far
        }
        print("Camera setup complete.")

    def get_camera_image(self):
        """Captures a single image from the simulated camera.
        
        Returns:
            rgb_image: numpy array (height, width, 3), dtype=uint8
            depth_buffer: numpy array (height, width), dtype=float32
        """
        try:
            img_data = p.getCameraImage(
                self.camera_params['width'],
                self.camera_params['height'],
                self.camera_params['view_matrix'],
                self.camera_params['projection_matrix'],
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
        except Exception:
            img_data = p.getCameraImage(
                self.camera_params['width'],
                self.camera_params['height'],
                self.camera_params['view_matrix'],
                self.camera_params['projection_matrix'],
                renderer=p.ER_TINY_RENDERER
            )
        
        # Extract RGBA and depth
        try:
            rgba = img_data[2]
            depth_raw = img_data[3]
        except Exception as e:
            raise RuntimeError(f"Unexpected camera image format: {e}")
        
        # Convert RGBA to RGB
        rgb_image = np.array(rgba, dtype=np.uint8)
        if rgb_image.ndim == 3 and rgb_image.shape[2] >= 3:
            rgb_image = rgb_image[:, :, :3]
        else:
            w = self.camera_params['width']
            h = self.camera_params['height']
            try:
                rgb_image = rgb_image.reshape((h, w, -1))[:, :, :3]
            except Exception:
                rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert depth to numpy array
        depth_np = np.array(depth_raw, dtype=np.float32)
        width = self.camera_params['width']
        height = self.camera_params['height']
        
        if depth_np.size == width * height:
            depth_buffer = depth_np.reshape((height, width))
        elif depth_np.shape == (height, width):
            depth_buffer = depth_np
        else:
            try:
                depth_buffer = depth_np.reshape((width, height)).T
            except Exception:
                raise RuntimeError(f"Unexpected depth buffer shape: {depth_np.shape}")
        
        return rgb_image, depth_buffer

    def get_target_position(self):
        """Get the ground-truth position of the teddy bear."""
        pos, orn = p.getBasePositionAndOrientation(self.target_id)
        return pos

    def step(self):
        """Steps the simulation forward."""
        p.stepSimulation()

    def close(self):
        """Disconnects from the simulation."""
        print("Disconnecting from simulation.")
        p.disconnect()