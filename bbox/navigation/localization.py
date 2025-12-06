#!/usr/bin/env python3
"""
localization.py - Robot Localization for BlueROV2
=================================================

WHAT IS LOCALIZATION?
---------------------
Localization answers the question: "Where am I in the world?"

Your BlueROV2 needs to know its position and orientation in the pool to:
1. Navigate to waypoints
2. Avoid obstacles
3. Find and approach the blackbox
4. Return home after grabbing the blackbox

HOW DOES IT WORK?
-----------------
We use multiple information sources and combine them:

1. ArUco MARKERS (Primary source):
   - 9 markers on the pool floor at known positions
   - Camera detects markers → calculates camera pose relative to marker
   - Since we know marker positions in the pool, we can calculate ROV position
   
2. DEPTH SENSOR (Z-axis):
   - Very accurate depth measurement
   - Used to correct/improve Z coordinate from marker detection

3. IMU (Orientation):
   - Provides roll, pitch, yaw
   - Used between marker detections to track orientation changes

SENSOR FUSION:
--------------
We combine these sensors using a technique called "filtering":
- When we see multiple markers, we average their estimates
- We apply a low-pass filter to smooth out noise and jumps
- We use the depth sensor to improve vertical position accuracy

THE PROBLEM: "TOO CLOSE TO SEE"
-------------------------------
When the ROV gets very close to the blackbox, the markers may go out of view.
This is why we also store the last known good position and can use IMU 
dead-reckoning (estimating position from motion) for short periods.

COORDINATE FRAME REMINDER:
--------------------------
World/Net Frame:
- Origin: Corner of pool
- X, Y: Horizontal plane (pool floor)
- Z: Depth (positive = deeper into the water)

Your ArUco markers are at Z = 4.8m (on the pool floor at 4.8m depth).
"""

import numpy as np
import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from .transforms import Transforms, Pose3D


class LocalizationState(Enum):
    """
    States of the localization system.
    
    UNINITIALIZED: No valid pose yet
    TRACKING: Actively seeing markers, good localization
    DEAD_RECKONING: Lost markers, estimating from last known pose
    LOST: Been too long without markers, position unreliable
    """
    UNINITIALIZED = 0
    TRACKING = 1
    DEAD_RECKONING = 2
    LOST = 3


@dataclass
class LocalizationConfig:
    """
    Configuration parameters for the localization system.
    
    These are tuning parameters you can adjust based on your testing.
    """
    # Marker detection settings
    min_markers_for_good_fix: int = 1  # Minimum markers needed for "good" localization
    max_marker_distance: float = 5.0   # Ignore markers farther than this (meters)
    
    # Filtering parameters (0.0 = no smoothing, 1.0 = infinite smoothing)
    position_filter_alpha: float = 0.3  # How much to trust new measurements (lower = smoother)
    orientation_filter_alpha: float = 0.3
    
    # Depth sensor settings
    use_depth_sensor: bool = True
    depth_sensor_weight: float = 0.8  # Weight given to depth sensor vs markers for Z
    
    # Dead reckoning settings
    dead_reckoning_timeout: float = 2.0  # Seconds before marking as LOST
    max_dead_reckoning_drift: float = 0.5  # Max estimated drift before LOST
    
    # Outlier rejection
    max_position_jump: float = 1.0  # Reject measurements that jump more than this
    max_orientation_jump: float = 0.5  # Radians


@dataclass
class MarkerObservation:
    """
    A single observation of an ArUco marker.
    
    Attributes:
        marker_id: The ID of the observed marker (0-8 for pool markers)
        camera_to_marker: Transform from camera frame to marker frame
        timestamp: When this observation was made
        confidence: How confident we are in this detection (0-1)
    """
    marker_id: int
    camera_to_marker: np.ndarray  # 4x4 transform matrix
    timestamp: float
    confidence: float = 1.0


class Localization:
    """
    Main localization class for the BlueROV2.
    
    This class takes in sensor data (ArUco detections, depth, IMU) and outputs
    a filtered, smooth estimate of the robot's pose in the world frame.
    
    Usage:
        # Create localization system
        loc = Localization()
        
        # In your main loop:
        # 1. Update with marker observations
        loc.update_markers(marker_observations, camera_tilt_deg)
        
        # 2. Update depth if available
        loc.update_depth(depth_meters)
        
        # 3. Get current pose estimate
        pose = loc.get_pose()
        state = loc.get_state()
    """
    
    # Known positions of ArUco markers in the pool (World/Net frame)
    # Format: marker_id -> (x, y, z, qx, qy, qz, qw)
    MARKER_POSES = {
        0: (4.5, 7.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
        1: (1.5, 1.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
        2: (7.5, 4.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
        3: (7.5, 7.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
        4: (4.5, 4.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
        5: (1.5, 7.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
        6: (1.5, 4.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
        7: (4.5, 1.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
        8: (7.5, 1.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
    }
    
    def __init__(self, config: Optional[LocalizationConfig] = None):
        """
        Initialize the localization system.
        
        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self.config = config or LocalizationConfig()
        
        # Current state
        self._state = LocalizationState.UNINITIALIZED
        self._current_pose: Optional[Pose3D] = None
        self._raw_pose: Optional[Pose3D] = None  # Before filtering
        
        # Timing
        self._last_marker_time: float = 0.0
        self._last_update_time: float = 0.0
        
        # Depth sensor data
        self._current_depth: Optional[float] = None
        
        # IMU data for dead reckoning
        self._last_imu_orientation: Optional[np.ndarray] = None
        
        # Pre-compute marker pose matrices
        self._marker_matrices: Dict[int, np.ndarray] = {}
        for marker_id, pose_tuple in self.MARKER_POSES.items():
            x, y, z, qx, qy, qz, qw = pose_tuple
            pose = Pose3D(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw)
            self._marker_matrices[marker_id] = Transforms.pose_to_matrix(pose)
        
        # Statistics for debugging
        self.stats = {
            'markers_seen': 0,
            'updates': 0,
            'rejected_outliers': 0,
        }
    
    def update_markers(self, 
                       observations: List[MarkerObservation],
                       camera_tilt_deg: float = 0.0) -> bool:
        """
        Update localization with new marker observations.
        
        This is the main update function. Call it whenever you have new
        marker detections from the camera.
        
        Args:
            observations: List of MarkerObservation objects from aruco_pool_node
            camera_tilt_deg: Current camera tilt angle in degrees
            
        Returns:
            True if localization was updated successfully
            
        How it works:
        1. For each visible marker, calculate where the ROV must be
        2. Average all the estimates (they should be similar if markers are accurate)
        3. Apply filtering to smooth out noise
        4. Update the state machine (TRACKING, DEAD_RECKONING, etc.)
        """
        current_time = time.time()
        
        if not observations:
            # No markers seen - switch to dead reckoning
            self._handle_no_markers(current_time)
            return False
        
        # Calculate body pose from each marker observation
        body_pose_estimates: List[Pose3D] = []
        
        for obs in observations:
            if obs.marker_id not in self._marker_matrices:
                continue  # Unknown marker
            
            # Get marker's known position in world frame
            T_world_to_marker = self._marker_matrices[obs.marker_id]
            
            # Get camera to body transform (accounting for tilt)
            T_camera_to_body = Transforms.camera_to_body_transform(camera_tilt_deg)
            
            # The observation gives us: camera → marker transform
            # We need: world → body
            #
            # Chain of transforms:
            # world → marker → camera → body
            #
            # T_world_to_marker is known (marker map)
            # T_marker_to_camera = inverse of (T_camera_to_marker from observation)
            # T_camera_to_body is from our physical setup
            
            T_marker_to_camera = Transforms.invert_transform(obs.camera_to_marker)
            
            # Chain them: world → marker → camera → body
            T_world_to_camera = T_world_to_marker @ T_marker_to_camera
            T_world_to_body = T_world_to_camera @ T_camera_to_body
            
            # Convert to Pose3D
            body_pose = Transforms.matrix_to_pose(T_world_to_body)
            
            # Check for outliers (wild jumps)
            if self._current_pose is not None:
                dist = Transforms.distance_between_poses(body_pose, self._current_pose)
                if dist > self.config.max_position_jump:
                    self.stats['rejected_outliers'] += 1
                    continue
            
            body_pose_estimates.append(body_pose)
        
        if not body_pose_estimates:
            self._handle_no_markers(current_time)
            return False
        
        # Average all pose estimates
        raw_pose = self._average_poses(body_pose_estimates)
        self._raw_pose = raw_pose
        
        # Apply depth sensor correction
        if self.config.use_depth_sensor and self._current_depth is not None:
            raw_pose = self._fuse_depth(raw_pose)
        
        # Apply filtering (smooth out noise)
        if self._current_pose is None:
            # First measurement - no filtering
            self._current_pose = raw_pose
        else:
            self._current_pose = self._filter_pose(self._current_pose, raw_pose)
        
        # Update state and timing
        self._state = LocalizationState.TRACKING
        self._last_marker_time = current_time
        self._last_update_time = current_time
        
        self.stats['markers_seen'] += len(observations)
        self.stats['updates'] += 1
        
        return True
    
    def update_depth(self, depth_meters: float) -> None:
        """
        Update with new depth sensor reading.
        
        Args:
            depth_meters: Current depth in meters (positive = deeper)
        """
        self._current_depth = depth_meters
        
        # If we have a pose and depth, update the Z component
        if self._current_pose is not None and self.config.use_depth_sensor:
            # Blend current pose Z with depth sensor
            alpha = self.config.depth_sensor_weight
            new_z = alpha * depth_meters + (1 - alpha) * self._current_pose.z
            self._current_pose.z = new_z
    
    def update_imu(self, 
                   roll: float, 
                   pitch: float, 
                   yaw: float,
                   angular_velocity: Optional[np.ndarray] = None) -> None:
        """
        Update with IMU orientation data.
        
        This is useful for:
        1. Improving orientation estimates between marker detections
        2. Dead reckoning when markers are not visible
        
        Args:
            roll, pitch, yaw: Euler angles in radians
            angular_velocity: Optional [wx, wy, wz] angular velocities
        """
        q = Transforms.quaternion_from_euler(roll, pitch, yaw)
        self._last_imu_orientation = q
        
        # If in dead reckoning mode, use IMU to update orientation
        if self._state == LocalizationState.DEAD_RECKONING and self._current_pose is not None:
            # Update orientation from IMU
            alpha = self.config.orientation_filter_alpha
            current_q = self._current_pose.quaternion()
            filtered_q = self._slerp(current_q, q, alpha)
            self._current_pose.qx = filtered_q[0]
            self._current_pose.qy = filtered_q[1]
            self._current_pose.qz = filtered_q[2]
            self._current_pose.qw = filtered_q[3]
    
    def get_pose(self) -> Optional[Pose3D]:
        """
        Get the current estimated pose of the robot body in world frame.
        
        Returns:
            Pose3D if localized, None if not yet initialized
        """
        return self._current_pose
    
    def get_state(self) -> LocalizationState:
        """
        Get the current state of the localization system.
        
        Returns:
            LocalizationState enum value
        """
        return self._state
    
    def is_tracking(self) -> bool:
        """Check if actively tracking markers."""
        return self._state == LocalizationState.TRACKING
    
    def is_lost(self) -> bool:
        """Check if localization is lost."""
        return self._state == LocalizationState.LOST
    
    def get_position(self) -> Optional[np.ndarray]:
        """Get just the position [x, y, z] if available."""
        if self._current_pose is None:
            return None
        return self._current_pose.position()
    
    def get_yaw(self) -> Optional[float]:
        """Get current yaw angle in radians."""
        if self._current_pose is None:
            return None
        q = self._current_pose.quaternion()
        _, _, yaw = Transforms.euler_from_quaternion(q)
        return yaw
    
    def reset(self) -> None:
        """Reset the localization system."""
        self._state = LocalizationState.UNINITIALIZED
        self._current_pose = None
        self._raw_pose = None
        self._last_marker_time = 0.0
        self._current_depth = None
    
    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================
    
    def _handle_no_markers(self, current_time: float) -> None:
        """Handle the case when no markers are visible."""
        if self._state == LocalizationState.UNINITIALIZED:
            return  # Can't do anything yet
        
        time_since_markers = current_time - self._last_marker_time
        
        if time_since_markers < self.config.dead_reckoning_timeout:
            self._state = LocalizationState.DEAD_RECKONING
        else:
            self._state = LocalizationState.LOST
    
    def _average_poses(self, poses: List[Pose3D]) -> Pose3D:
        """
        Average multiple pose estimates.
        
        For position: simple arithmetic mean
        For orientation: quaternion averaging (normalized sum)
        """
        if len(poses) == 1:
            return poses[0]
        
        # Average positions
        positions = np.array([p.position() for p in poses])
        avg_position = positions.mean(axis=0)
        
        # Average quaternions (simple approach: sum and normalize)
        quaternions = np.array([p.quaternion() for p in poses])
        
        # Make sure quaternions are in same hemisphere
        for i in range(1, len(quaternions)):
            if np.dot(quaternions[0], quaternions[i]) < 0:
                quaternions[i] = -quaternions[i]
        
        avg_quaternion = quaternions.sum(axis=0)
        avg_quaternion = Transforms.quaternion_normalize(avg_quaternion)
        
        return Pose3D(
            x=avg_position[0], y=avg_position[1], z=avg_position[2],
            qx=avg_quaternion[0], qy=avg_quaternion[1], 
            qz=avg_quaternion[2], qw=avg_quaternion[3]
        )
    
    def _filter_pose(self, current: Pose3D, measurement: Pose3D) -> Pose3D:
        """
        Apply low-pass filter to smooth pose estimate.
        
        This is an Exponential Moving Average (EMA) filter:
        new_value = alpha * measurement + (1 - alpha) * current
        
        Lower alpha = smoother but more lag
        Higher alpha = more responsive but noisier
        """
        alpha_pos = self.config.position_filter_alpha
        alpha_ori = self.config.orientation_filter_alpha
        
        # Filter position
        new_pos = alpha_pos * measurement.position() + (1 - alpha_pos) * current.position()
        
        # Filter orientation using SLERP
        new_quat = self._slerp(current.quaternion(), measurement.quaternion(), alpha_ori)
        
        return Pose3D(
            x=new_pos[0], y=new_pos[1], z=new_pos[2],
            qx=new_quat[0], qy=new_quat[1], qz=new_quat[2], qw=new_quat[3]
        )
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical Linear Interpolation between quaternions.
        
        Args:
            q1: Start quaternion
            q2: End quaternion
            t: Interpolation factor [0, 1]
            
        Returns:
            Interpolated quaternion
        """
        # Ensure shortest path
        if np.dot(q1, q2) < 0:
            q2 = -q2
        
        dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
        
        if dot > 0.9995:
            # Very close - use linear interpolation
            result = q1 + t * (q2 - q1)
        else:
            theta = math.acos(dot)
            result = (math.sin((1 - t) * theta) * q1 + math.sin(t * theta) * q2) / math.sin(theta)
        
        return Transforms.quaternion_normalize(result)
    
    def _fuse_depth(self, pose: Pose3D) -> Pose3D:
        """
        Fuse marker-based Z estimate with depth sensor.
        
        The depth sensor is typically more accurate for Z than markers.
        """
        if self._current_depth is None:
            return pose
        
        alpha = self.config.depth_sensor_weight
        fused_z = alpha * self._current_depth + (1 - alpha) * pose.z
        
        return Pose3D(
            x=pose.x, y=pose.y, z=fused_z,
            qx=pose.qx, qy=pose.qy, qz=pose.qz, qw=pose.qw
        )
    
    def get_debug_info(self) -> dict:
        """Get debugging information about the localization state."""
        return {
            'state': self._state.name,
            'pose': self._current_pose,
            'raw_pose': self._raw_pose,
            'depth_sensor': self._current_depth,
            'time_since_markers': time.time() - self._last_marker_time if self._last_marker_time > 0 else -1,
            'stats': self.stats.copy(),
        }


# =============================================================================
# ROS2 NODE WRAPPER
# =============================================================================

class LocalizationNode:
    """
    ROS2 wrapper for the Localization class.
    
    This is a template showing how to integrate with your ROS2 system.
    You can either use this directly or integrate the Localization class
    into your existing nodes.
    """
    
    def __init__(self, node):
        """
        Initialize with a ROS2 node.
        
        Args:
            node: rclpy.node.Node instance
        """
        self.node = node
        self.localization = Localization()
        self.camera_tilt_deg = 0.0
        
        # You would set up subscribers here:
        # - Subscribe to aruco_pool/poses and aruco_pool/ids
        # - Subscribe to depth sensor
        # - Subscribe to IMU
        # - Subscribe to camera tilt
        
        # And publishers:
        # - Publish robot pose (geometry_msgs/PoseStamped)
        # - Publish localization status
    
    def marker_callback(self, poses_msg, ids_msg):
        """
        Example callback for marker detections.
        
        You would convert ROS messages to MarkerObservation objects
        and call self.localization.update_markers()
        """
        pass
    
    def depth_callback(self, depth_msg):
        """Example callback for depth sensor."""
        pass


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the localization module."""
    
    print("=" * 60)
    print("LOCALIZATION MODULE TEST")
    print("=" * 60)
    
    # Create localization system
    config = LocalizationConfig(
        position_filter_alpha=0.5,
        use_depth_sensor=True
    )
    loc = Localization(config)
    
    print(f"\nInitial state: {loc.get_state().name}")
    
    # Simulate a marker observation
    # Pretend we see marker 4 (at position 4.5, 4.0, 4.8)
    # And the camera-to-marker transform says marker is 2m away
    
    # Create a fake camera-to-marker transform
    # (In reality, this comes from aruco_pool_node)
    T_cam_to_marker = np.eye(4)
    T_cam_to_marker[0:3, 3] = [0, 0, 2.0]  # Marker 2m forward of camera
    
    obs = MarkerObservation(
        marker_id=4,
        camera_to_marker=T_cam_to_marker,
        timestamp=time.time(),
        confidence=1.0
    )
    
    print("\nSimulating marker observation...")
    success = loc.update_markers([obs], camera_tilt_deg=0.0)
    
    print(f"Update successful: {success}")
    print(f"State: {loc.get_state().name}")
    print(f"Pose: {loc.get_pose()}")
    
    # Simulate depth update
    loc.update_depth(2.5)  # ROV at 2.5m depth
    print(f"\nAfter depth update:")
    print(f"Pose: {loc.get_pose()}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
