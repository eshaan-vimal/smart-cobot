import pybullet as p
import numpy as np


class RobotController:
    def __init__(self, robot_id, end_effector_link_index):
        """Initializes the robot controller with proper joint handling."""
        self.robot_id = int(robot_id)
        self.end_effector_link_index = int(end_effector_link_index)
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_infos = [p.getJointInfo(self.robot_id, i) for i in range(self.num_joints)]
        
        # Filter for movable joints only
        self.movable_indices = [
            i for i, info in enumerate(self.joint_infos) 
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
        ]
        
        print(f"Found {len(self.movable_indices)} movable joints: {self.movable_indices}")
        
        # Extract bounds
        self.lower_limits = [self.joint_infos[i][8] for i in self.movable_indices]
        self.upper_limits = [self.joint_infos[i][9] for i in self.movable_indices]
        self.joint_ranges = [ul - ll for ul, ll in zip(self.upper_limits, self.lower_limits)]
        self.rest_poses = [0.0] * len(self.movable_indices)
        
        # Initialize joints to rest pose
        self._reset_to_rest()
        
        # IK parameters
        self.max_num_iterations = 100
        self.residual_threshold = 1e-5
        
        print(f"Robot controller ready. Lower limits: {self.lower_limits}")
        print(f"Robot controller ready. Upper limits: {self.upper_limits}")

    def _reset_to_rest(self):
        """Reset the robot to rest pose."""
        for idx, joint_idx in enumerate(self.movable_indices):
            p.resetJointState(self.robot_id, joint_idx, self.rest_poses[idx])

    def move_to_target(self, target_position, target_orientation_euler=[0, -np.pi, 0], attempts=3):
        """Moves the robot's end-effector to the target position using IK.
        
        Args:
            target_position: [x, y, z] in world coordinates
            target_orientation_euler: [roll, pitch, yaw] in radians
            attempts: Number of IK attempts with different rest poses
            
        Returns:
            True if successful, False otherwise
        """
        try:
            target_orientation = p.getQuaternionFromEuler(target_orientation_euler)
            
            # Try IK with different rest poses
            best_solution = None
            
            for attempt in range(attempts):
                # Vary rest pose slightly each attempt
                current_rest = [p + 0.2 * np.sin(attempt) for p in self.rest_poses]
                
                try:
                    joint_poses = p.calculateInverseKinematics(
                        self.robot_id,
                        self.end_effector_link_index,
                        target_position,
                        target_orientation,
                        lowerLimits=self.lower_limits,
                        upperLimits=self.upper_limits,
                        jointRanges=self.joint_ranges,
                        restPoses=current_rest,
                        maxNumIterations=self.max_num_iterations,
                        residualThreshold=self.residual_threshold
                    )
                    
                    best_solution = joint_poses
                    break
                except Exception:
                    continue
            
            if best_solution is None:
                print(f"IK failed after {attempts} attempts for target {target_position}")
                return False
            
            # Extract only movable joint positions
            target_joint_poses = [float(best_solution[i]) for i in range(len(self.movable_indices))]
            
            # Clamp to joint limits
            target_joint_poses = [
                np.clip(target_joint_poses[i], self.lower_limits[i], self.upper_limits[i])
                for i in range(len(self.movable_indices))
            ]
            
            # ✅ CORRECT - Removed the invalid parameter
            p.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=self.movable_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_joint_poses,
                forces=[1000.0] * len(self.movable_indices)
            )

            
            print(f"IK SUCCESS: Moving to {target_position}")
            return True
        
        except Exception as e:
            print(f"IK calculation failed: {e}")
            return False

    def get_end_effector_pos(self):
        """Get current end-effector position."""
        state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        return state[0]  # Position