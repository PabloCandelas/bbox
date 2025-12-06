#!/usr/bin/env python3
"""
transforms.py - Coordinate Frame Transformations for BlueROV2
=============================================================

WHAT IS THIS FILE?
------------------
This file handles converting positions and orientations between different
"coordinate frames" (reference systems). Think of it like converting between
different measurement systems (meters to feet), but for 3D space.

WHY DO WE NEED THIS?
--------------------
Different sensors "see" the world differently:
- The camera thinks "forward" is its Z-axis
- The ROV body thinks "forward" is its X-axis
- The pool has its own coordinate system

We need to translate between these systems so all parts of the robot
can communicate using the same "language".

KEY CONCEPTS FOR BEGINNERS:
---------------------------

1. POSITION (Translation):
   A point in 3D space: (x, y, z)
   Example: The blackbox is at position (3.0, 5.0, 2.0) meters

2. ORIENTATION (Rotation):
   Which way something is facing. We use "quaternions" (4 numbers: x, y, z, w)
   instead of angles because they're mathematically better behaved.
   
   Don't worry if quaternions seem confusing - just know they represent rotation.
   
3. POSE = Position + Orientation
   A complete description of where something is AND which way it's facing.

4. HOMOGENEOUS TRANSFORMATION MATRIX (4x4):
   A mathematical tool that combines rotation and translation into one matrix.
   This lets us chain multiple transformations together easily.
   
   [R R R | tx]     R = 3x3 rotation matrix
   [R R R | ty]     t = translation vector (x, y, z)
   [R R R | tz]
   [0 0 0 | 1 ]

YOUR COORDINATE FRAMES:
-----------------------

CAMERA FRAME:           BODY FRAME:            WORLD/NET FRAME:
     Z (forward)             X (forward)           Y
     ↑                       ↑                     ↑
     │                       │                     │
     │                       │                     │
     └────→ X (right)        └────→ Y (right)      └────→ X
    ↙                       ↙                      
   Y (down)                Z (down)               Z (into pool/down)

The camera is 23cm behind the body center.
"""

import numpy as np
import math
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class Pose3D:
    """
    Represents a 3D pose (position + orientation).
    
    This is a simple container to hold pose data in a clean way.
    
    Attributes:
        x, y, z: Position in meters
        qx, qy, qz, qw: Orientation as quaternion (qw is the scalar part)
    
    Example:
        pose = Pose3D(x=1.0, y=2.0, z=3.0, qx=0, qy=0, qz=0, qw=1)
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0  # Identity rotation (no rotation)
    
    def position(self) -> np.ndarray:
        """Return position as numpy array [x, y, z]"""
        return np.array([self.x, self.y, self.z])
    
    def quaternion(self) -> np.ndarray:
        """Return quaternion as numpy array [qx, qy, qz, qw]"""
        return np.array([self.qx, self.qy, self.qz, self.qw])
    
    def __repr__(self):
        return f"Pose3D(pos=[{self.x:.3f}, {self.y:.3f}, {self.z:.3f}], quat=[{self.qx:.3f}, {self.qy:.3f}, {self.qz:.3f}, {self.qw:.3f}])"


class Transforms:
    """
    Utility class for coordinate transformations.
    
    This class provides all the math needed to convert between coordinate frames.
    
    IMPORTANT TRANSFORMS FOR YOUR PROJECT:
    1. Camera → Body: Account for camera position and tilt
    2. Body → World: Use ArUco markers to know where ROV is in pool
    3. World → Body: Plan paths in world, execute in body frame
    """
    
    # =====================================================================
    # PHYSICAL PARAMETERS - Measure these on your actual robot!
    # =====================================================================
    
    # Distance from body center to camera (meters)
    # The camera is 23cm in front of body center along body X-axis
    CAMERA_TO_BODY_OFFSET = np.array([0.23, 0.0, 0.0])  # [forward, right, down] in body frame
    
    # Maximum camera tilt angles (degrees)
    CAMERA_TILT_MIN = -45.0
    CAMERA_TILT_MAX = 45.0
    
    # =====================================================================
    # QUATERNION OPERATIONS
    # =====================================================================
    
    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions (chain rotations).
        
        Think of this like: "first rotate by q1, then rotate by q2"
        
        Args:
            q1: First quaternion [x, y, z, w]
            q2: Second quaternion [x, y, z, w]
            
        Returns:
            Combined rotation quaternion [x, y, z, w]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    @staticmethod
    def quaternion_inverse(q: np.ndarray) -> np.ndarray:
        """
        Compute inverse of a quaternion (reverse the rotation).
        
        Args:
            q: Quaternion [x, y, z, w]
            
        Returns:
            Inverse quaternion [x, y, z, w]
        """
        x, y, z, w = q
        norm_sq = x*x + y*y + z*z + w*w
        if norm_sq < 1e-10:
            return np.array([0, 0, 0, 1])
        return np.array([-x, -y, -z, w]) / norm_sq
    
    @staticmethod
    def quaternion_normalize(q: np.ndarray) -> np.ndarray:
        """
        Normalize quaternion to unit length.
        
        Quaternions must have length 1 to represent valid rotations.
        """
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([0, 0, 0, 1])
        return q / norm
    
    @staticmethod
    def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion.
        
        This is the ZYX convention (yaw first, then pitch, then roll).
        
        Args:
            roll: Rotation around X-axis (radians)
            pitch: Rotation around Y-axis (radians)
            yaw: Rotation around Z-axis (radians)
            
        Returns:
            Quaternion [x, y, z, w]
            
        Example:
            # 90 degrees yaw rotation (turn left)
            q = Transforms.quaternion_from_euler(0, 0, math.pi/2)
        """
        cr = math.cos(roll / 2)
        sr = math.sin(roll / 2)
        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)
        cy = math.cos(yaw / 2)
        sy = math.sin(yaw / 2)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw])
    
    @staticmethod
    def euler_from_quaternion(q: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Args:
            q: Quaternion [x, y, z, w]
            
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        x, y, z, w = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    # =====================================================================
    # ROTATION MATRIX OPERATIONS
    # =====================================================================
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.
        
        Args:
            q: Quaternion [x, y, z, w]
            
        Returns:
            3x3 rotation matrix
        """
        x, y, z, w = q
        
        # Normalize
        n = math.sqrt(x*x + y*y + z*z + w*w)
        if n < 1e-10:
            return np.eye(3)
        x, y, z, w = x/n, y/n, z/n, w/n
        
        # Build rotation matrix
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to quaternion.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion [x, y, z, w]
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return Transforms.quaternion_normalize(np.array([x, y, z, w]))
    
    # =====================================================================
    # HOMOGENEOUS TRANSFORMATION MATRICES (4x4)
    # =====================================================================
    
    @staticmethod
    def pose_to_matrix(pose: Pose3D) -> np.ndarray:
        """
        Convert a Pose3D to a 4x4 homogeneous transformation matrix.
        
        This matrix can be used to transform points from one frame to another.
        
        Args:
            pose: Pose3D object
            
        Returns:
            4x4 transformation matrix
        """
        R = Transforms.quaternion_to_rotation_matrix(pose.quaternion())
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = pose.position()
        return T
    
    @staticmethod
    def matrix_to_pose(T: np.ndarray) -> Pose3D:
        """
        Convert a 4x4 homogeneous transformation matrix to Pose3D.
        
        Args:
            T: 4x4 transformation matrix
            
        Returns:
            Pose3D object
        """
        q = Transforms.rotation_matrix_to_quaternion(T[0:3, 0:3])
        return Pose3D(
            x=T[0, 3], y=T[1, 3], z=T[2, 3],
            qx=q[0], qy=q[1], qz=q[2], qw=q[3]
        )
    
    @staticmethod
    def invert_transform(T: np.ndarray) -> np.ndarray:
        """
        Invert a 4x4 transformation matrix.
        
        If T transforms from frame A to frame B,
        then invert_transform(T) transforms from frame B to frame A.
        
        Args:
            T: 4x4 transformation matrix
            
        Returns:
            Inverted 4x4 transformation matrix
        """
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        
        R_inv = R.T  # For rotation matrices, inverse = transpose
        t_inv = -R_inv @ t
        
        T_inv = np.eye(4)
        T_inv[0:3, 0:3] = R_inv
        T_inv[0:3, 3] = t_inv
        return T_inv
    
    # =====================================================================
    # CAMERA <-> BODY FRAME TRANSFORMS
    # =====================================================================
    
    @staticmethod
    def get_camera_to_body_rotation() -> np.ndarray:
        """
        Get the rotation matrix from camera frame to body frame.
        
        Camera frame:  Z forward, Y down, X right
        Body frame:    X forward, Z down, Y right
        
        We need to map:
            Camera Z → Body X  (forward)
            Camera X → Body Y  (right)  
            Camera Y → Body Z  (down)
        
        Returns:
            3x3 rotation matrix R such that p_body = R @ p_camera
        """
        # This rotation matrix maps camera axes to body axes
        # Each column tells us where camera's X, Y, Z axes go in body frame
        #
        # Camera X (right)   → Body Y
        # Camera Y (down)    → Body Z  
        # Camera Z (forward) → Body X
        #
        # So the matrix is:
        # [body_x_from_cam_x, body_x_from_cam_y, body_x_from_cam_z]   [0, 0, 1]
        # [body_y_from_cam_x, body_y_from_cam_y, body_y_from_cam_z] = [1, 0, 0]
        # [body_z_from_cam_x, body_z_from_cam_y, body_z_from_cam_z]   [0, 1, 0]
        
        R_cam_to_body = np.array([
            [0, 0, 1],  # Body X comes from Camera Z
            [1, 0, 0],  # Body Y comes from Camera X
            [0, 1, 0]   # Body Z comes from Camera Y
        ])
        return R_cam_to_body
    
    @staticmethod
    def get_camera_tilt_rotation(tilt_angle_deg: float) -> np.ndarray:
        """
        Get rotation matrix for camera tilt.
        
        The camera tilts around its X-axis (roll in camera frame).
        Positive tilt = camera looks up.
        
        Args:
            tilt_angle_deg: Tilt angle in degrees (positive = up)
            
        Returns:
            3x3 rotation matrix for the tilt
        """
        angle_rad = math.radians(tilt_angle_deg)
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        
        # Rotation around X-axis
        R_tilt = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        return R_tilt
    
    @classmethod
    def camera_to_body_transform(cls, tilt_angle_deg: float = 0.0) -> np.ndarray:
        """
        Get the full 4x4 transform from camera frame to body frame.
        
        This accounts for:
        1. Camera tilt (rotation around camera X-axis)
        2. Frame rotation (camera axes → body axes)
        3. Translation offset (camera is 23cm in front of body center)
        
        Args:
            tilt_angle_deg: Current camera tilt angle in degrees
            
        Returns:
            4x4 transformation matrix T such that p_body = T @ p_camera
        """
        # Step 1: Apply camera tilt (in camera frame)
        R_tilt = cls.get_camera_tilt_rotation(tilt_angle_deg)
        
        # Step 2: Rotate from camera frame to body frame
        R_cam_to_body = cls.get_camera_to_body_rotation()
        
        # Combined rotation: first tilt, then frame change
        R_combined = R_cam_to_body @ R_tilt
        
        # Step 3: Translation (camera is offset from body center)
        # In body frame, camera is 23cm forward (along body X)
        # So to go from camera origin to body origin, we go -23cm in body X
        t_cam_to_body = -cls.CAMERA_TO_BODY_OFFSET
        
        # Build 4x4 matrix
        T = np.eye(4)
        T[0:3, 0:3] = R_combined
        T[0:3, 3] = t_cam_to_body
        
        return T
    
    @classmethod
    def body_to_camera_transform(cls, tilt_angle_deg: float = 0.0) -> np.ndarray:
        """
        Get the 4x4 transform from body frame to camera frame.
        
        This is the inverse of camera_to_body_transform.
        
        Args:
            tilt_angle_deg: Current camera tilt angle in degrees
            
        Returns:
            4x4 transformation matrix
        """
        return cls.invert_transform(cls.camera_to_body_transform(tilt_angle_deg))
    
    # =====================================================================
    # POINT TRANSFORMATIONS
    # =====================================================================
    
    @staticmethod
    def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
        Transform a 3D point using a 4x4 transformation matrix.
        
        Args:
            T: 4x4 transformation matrix
            point: 3D point [x, y, z]
            
        Returns:
            Transformed 3D point [x', y', z']
            
        Example:
            # Transform a point from camera frame to body frame
            p_cam = np.array([1.0, 0.0, 2.0])  # 2m forward, 1m right in camera
            T_cam_to_body = Transforms.camera_to_body_transform(0.0)
            p_body = Transforms.transform_point(T_cam_to_body, p_cam)
        """
        p_homo = np.array([point[0], point[1], point[2], 1.0])
        p_transformed = T @ p_homo
        return p_transformed[0:3]
    
    @staticmethod
    def transform_pose(T: np.ndarray, pose: Pose3D) -> Pose3D:
        """
        Transform a pose using a 4x4 transformation matrix.
        
        Args:
            T: 4x4 transformation matrix
            pose: Pose3D to transform
            
        Returns:
            Transformed Pose3D
        """
        # Convert pose to matrix
        pose_matrix = Transforms.pose_to_matrix(pose)
        
        # Chain transforms
        transformed_matrix = T @ pose_matrix
        
        # Convert back to pose
        return Transforms.matrix_to_pose(transformed_matrix)
    
    # =====================================================================
    # UTILITY FUNCTIONS
    # =====================================================================
    
    @staticmethod
    def distance_between_poses(pose1: Pose3D, pose2: Pose3D) -> float:
        """
        Calculate Euclidean distance between two poses (position only).
        
        Args:
            pose1: First pose
            pose2: Second pose
            
        Returns:
            Distance in meters
        """
        return np.linalg.norm(pose1.position() - pose2.position())
    
    @staticmethod
    def angle_between_quaternions(q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Calculate the angle between two quaternions.
        
        Args:
            q1: First quaternion [x, y, z, w]
            q2: Second quaternion [x, y, z, w]
            
        Returns:
            Angle in radians
        """
        # q1 · q2 = cos(θ/2) where θ is the rotation angle between them
        dot = abs(np.dot(q1, q2))
        dot = min(1.0, dot)  # Clamp for numerical stability
        return 2 * math.acos(dot)
    
    @staticmethod
    def interpolate_poses(pose1: Pose3D, pose2: Pose3D, t: float) -> Pose3D:
        """
        Linearly interpolate between two poses.
        
        Args:
            pose1: Start pose (t=0)
            pose2: End pose (t=1)
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated Pose3D
        """
        t = max(0.0, min(1.0, t))
        
        # Linear interpolation for position
        pos = (1 - t) * pose1.position() + t * pose2.position()
        
        # SLERP for quaternion (Spherical Linear Interpolation)
        q1 = pose1.quaternion()
        q2 = pose2.quaternion()
        
        # Ensure shortest path
        if np.dot(q1, q2) < 0:
            q2 = -q2
        
        dot = np.dot(q1, q2)
        dot = min(1.0, max(-1.0, dot))
        
        if abs(dot) > 0.9995:
            # Quaternions very close, use linear interpolation
            q = (1 - t) * q1 + t * q2
        else:
            theta = math.acos(dot)
            q = (math.sin((1 - t) * theta) * q1 + math.sin(t * theta) * q2) / math.sin(theta)
        
        q = Transforms.quaternion_normalize(q)
        
        return Pose3D(
            x=pos[0], y=pos[1], z=pos[2],
            qx=q[0], qy=q[1], qz=q[2], qw=q[3]
        )


# =========================================================================
# TESTING / DEMONSTRATION
# =========================================================================

if __name__ == "__main__":
    """Test the transforms module."""
    
    print("=" * 60)
    print("TRANSFORMS MODULE TEST")
    print("=" * 60)
    
    # Test 1: Camera to body rotation
    print("\n1. Camera to Body Frame Rotation:")
    R = Transforms.get_camera_to_body_rotation()
    print(f"   Camera Z (forward) [0,0,1] → Body: {R @ np.array([0, 0, 1])}")
    print(f"   Camera X (right) [1,0,0]   → Body: {R @ np.array([1, 0, 0])}")
    print(f"   Camera Y (down) [0,1,0]    → Body: {R @ np.array([0, 1, 0])}")
    
    # Test 2: Full camera to body transform
    print("\n2. Full Camera → Body Transform (no tilt):")
    T = Transforms.camera_to_body_transform(0.0)
    p_cam = np.array([0, 0, 1])  # 1m forward in camera frame
    p_body = Transforms.transform_point(T, p_cam)
    print(f"   Point 1m forward in camera → Body frame: {p_body}")
    
    # Test 3: With camera tilt
    print("\n3. Camera → Body Transform (30° up tilt):")
    T_tilted = Transforms.camera_to_body_transform(30.0)
    p_body_tilted = Transforms.transform_point(T_tilted, p_cam)
    print(f"   Point 1m forward in camera → Body frame: {p_body_tilted}")
    
    # Test 4: Quaternion operations
    print("\n4. Quaternion from Euler (90° yaw):")
    q = Transforms.quaternion_from_euler(0, 0, math.pi/2)
    roll, pitch, yaw = Transforms.euler_from_quaternion(q)
    print(f"   Quaternion: {q}")
    print(f"   Back to Euler: roll={math.degrees(roll):.1f}°, pitch={math.degrees(pitch):.1f}°, yaw={math.degrees(yaw):.1f}°")
    
    # Test 5: Pose interpolation
    print("\n5. Pose Interpolation:")
    pose1 = Pose3D(x=0, y=0, z=0, qx=0, qy=0, qz=0, qw=1)
    pose2 = Pose3D(x=2, y=4, z=0, qx=0, qy=0, qz=0.707, qw=0.707)
    pose_mid = Transforms.interpolate_poses(pose1, pose2, 0.5)
    print(f"   Start:  {pose1}")
    print(f"   End:    {pose2}")
    print(f"   Middle: {pose_mid}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
