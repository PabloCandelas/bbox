#!/usr/bin/env python3
"""
aruco_localization.py
ROS2 Node for BlueROV Localization using existing ArUco detections + Camera Tilt.

Updates:
- FEATURE: Multi-Marker Weighted Averaging.
  Calculates position based on Inverse Distance Weighting (1/dist).
  Closer markers (more reliable) contribute more to the final estimated position.
- FEATURE: Low-Pass Filter (Smoothing) with RESET LOGIC.
- RETAINED: Static marker publishing and Tilt correction.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Float64
from geometry_msgs.msg import TransformStamped, PoseArray
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import numpy as np
import math
from scipy.spatial.transform import Rotation as R_scipy

# ================= USER CONFIGURATION =================

# 1. KNOWN MARKER POSITIONS IN WORLD FRAME (map)
# Format: ID: [x, y, z, qx, qy, qz, qw]
KNOWN_MARKERS = {
    0: [4.5, 7.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328],
    1: [1.5, 1.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328],
    2: [7.5, 4.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328],
    3: [7.5, 7.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328],
    4: [4.5, 4.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328],
    5: [1.5, 7.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328],
    6: [1.5, 4.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328],
    7: [4.5, 1.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328],
    8: [7.5, 1.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328],
}

# 2. CAMERA MOUNTING OFFSET (Robot Center -> Camera Pivot)
CAM_OFFSET_X = 0.2  # meters forward
CAM_OFFSET_Y = 0.0  # meters left
CAM_OFFSET_Z = 0.0  # meters up

# 3. SMOOTHING FACTOR (0.0 to 1.0)
FILTER_ALPHA = 0.2 

# 4. FILTER RESET TIMEOUT (seconds)
# If no markers are seen for this duration, reset the filter to avoid "sliding"
FILTER_TIMEOUT_S = 1.0

# ======================================================

def get_tf_matrix_from_pose(pose):
    mat = np.eye(4)
    mat[0, 3] = pose.position.x
    mat[1, 3] = pose.position.y
    mat[2, 3] = pose.position.z
    r = R_scipy.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    mat[:3, :3] = r.as_matrix()
    return mat

def get_tf_matrix_from_values(tvec, quat):
    mat = np.eye(4)
    mat[:3, 3] = tvec
    r = R_scipy.from_quat(quat)
    mat[:3, :3] = r.as_matrix()
    return mat

def matrix_to_transform_msg(matrix, frame_id, child_frame_id, stamp):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id
    t.transform.translation.x = matrix[0, 3]
    t.transform.translation.y = matrix[1, 3]
    t.transform.translation.z = matrix[2, 3]
    r = R_scipy.from_matrix(matrix[:3, :3])
    q = r.as_quat()
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    return t

class ArucoLocalizationNode(Node):
    def __init__(self):
        super().__init__("aruco_localization_node")

        # Parameters
        self.declare_parameter("camera_frame", "camera_optical") 
        self.declare_parameter("base_frame", "base_link")        
        self.declare_parameter("world_frame", "map")             

        self.frame_camera = self.get_parameter("camera_frame").value
        self.frame_base = self.get_parameter("base_frame").value
        self.frame_world = self.get_parameter("world_frame").value

        # Internal state
        self.current_tilt_deg = 0.0
        self.latest_ids = []
        self.last_T_w_b = None 
        
        # Filtering State
        self.filtered_pos = None
        self.filtered_quat = None
        self.last_detection_time = self.get_clock().now()

        # Subscriptions
        qos = rclpy.qos.QoSProfile(depth=10)
        self.sub_tilt = self.create_subscription(Float64, "camera/tilt_angle", self.tilt_cb, 10)
        self.sub_ids = self.create_subscription(Int32MultiArray, "aruco_pool/ids", self.ids_cb, qos)
        self.sub_poses = self.create_subscription(PoseArray, "aruco_pool/poses", self.poses_cb, qos)

        # Broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Publish Static Markers immediately
        self.publish_static_markers()

        # Timer to publish Base->Camera TF continuously
        self.timer = self.create_timer(0.05, self.timer_tf_callback)
        
        self.get_logger().info("Localization Node Ready. Smoothing & Timeout Enabled.")

    def publish_static_markers(self):
        """ Publishes all KNOWN_MARKERS as static TFs relative to the map """
        static_transforms = []
        now = self.get_clock().now().to_msg()
        
        for mid, data in KNOWN_MARKERS.items():
            if len(data) == 7:
                tvec = data[:3]
                quat = data[3:]
            else:
                tvec = data[:3]
                mr, mp, myaw = data[3:]
                quat = R_scipy.from_euler('xyz', [mr, mp, myaw], degrees=False).as_quat()
            
            mat = get_tf_matrix_from_values(tvec, quat)
            tf_msg = matrix_to_transform_msg(
                mat, 
                self.frame_world,       
                f"fixed_marker_{mid}",  
                now
            )
            static_transforms.append(tf_msg)
        self.static_broadcaster.sendTransform(static_transforms)

    def tilt_cb(self, msg):
        self.current_tilt_deg = msg.data

    def ids_cb(self, msg):
        self.latest_ids = msg.data

    def get_base_to_optical_transform(self):
        T_base_mount = np.eye(4)
        T_base_mount[:3, 3] = [CAM_OFFSET_X, CAM_OFFSET_Y, CAM_OFFSET_Z]
        r_tilt = R_scipy.from_euler('y', self.current_tilt_deg, degrees=True)
        T_tilt = np.eye(4)
        T_tilt[:3, :3] = r_tilt.as_matrix()
        T_opt = np.eye(4)
        T_opt[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        return T_base_mount @ T_tilt @ T_opt

    def timer_tf_callback(self):
        now = self.get_clock().now().to_msg()
        
        # 1. Base -> Camera Optical
        T_base_opt = self.get_base_to_optical_transform()
        tf_cam = matrix_to_transform_msg(
            T_base_opt, 
            self.frame_base, 
            self.frame_camera, 
            now
        )
        self.tf_broadcaster.sendTransform(tf_cam)

        # 2. Map -> Base
        if self.last_T_w_b is not None:
            tf_map = matrix_to_transform_msg(
                self.last_T_w_b,
                self.frame_world,
                self.frame_base,
                now 
            )
            self.tf_broadcaster.sendTransform(tf_map)

    def poses_cb(self, msg: PoseArray):
        if not self.latest_ids or len(self.latest_ids) != len(msg.poses):
            return

        # 1. Static Transform: Base -> Camera Optical
        T_base_opt = self.get_base_to_optical_transform()
        T_opt_base = np.linalg.inv(T_base_opt)

        # Storage for all computed robot poses this frame
        measured_positions = []
        measured_quats = []
        measured_weights = []

        # 2. Loop through ALL detected markers
        for mid, pose in zip(self.latest_ids, msg.poses):
            if mid in KNOWN_MARKERS:
                marker_data = KNOWN_MARKERS[mid]
                if len(marker_data) == 7:
                    tvec = marker_data[:3]
                    quat_w_m = marker_data[3:]
                else:
                    tvec = marker_data[:3]
                    mr, mp, myaw = marker_data[3:]
                    quat_w_m = R_scipy.from_euler('xyz', [mr, mp, myaw], degrees=False).as_quat()

                # Calculate Distance for Weighting
                dist_sq = pose.position.x**2 + pose.position.y**2 + pose.position.z**2
                dist = math.sqrt(dist_sq)
                
                # Weight = 1 / Distance (Closer is better)
                # Add small epsilon to avoid div by zero, though unlikely
                weight = 1.0 / (dist + 0.1) 

                T_w_m = get_tf_matrix_from_values(tvec, quat_w_m)
                T_opt_m = get_tf_matrix_from_pose(pose)

                # Solve: Map -> Base (Candidate)
                T_opt_m_inv = np.linalg.inv(T_opt_m)
                T_w_opt = T_w_m @ T_opt_m_inv
                T_w_b = T_w_opt @ T_opt_base
                
                # Extract Translation
                trans = T_w_b[:3, 3]
                # Extract Rotation
                r = R_scipy.from_matrix(T_w_b[:3, :3])
                q = r.as_quat()

                measured_positions.append(trans)
                measured_quats.append(q)
                measured_weights.append(weight)

        if not measured_positions:
            return

        # 3. Weighted Average of measurements
        weights = np.array(measured_weights)
        
        # Weighted Position
        avg_pos = np.average(measured_positions, axis=0, weights=weights)
        
        # Weighted Quaternions (Handle sign flip)
        ref_quat = measured_quats[0]
        aligned_quats = []
        for q in measured_quats:
            if np.dot(q, ref_quat) < 0:
                aligned_quats.append(-q)
            else:
                aligned_quats.append(q)
        
        # Weighted Average of aligned quaternions
        avg_quat = np.average(aligned_quats, axis=0, weights=weights)
        norm = np.linalg.norm(avg_quat)
        if norm > 0:
            avg_quat /= norm

        # 4. Check Time Gap (Filter Reset Logic)
        now = self.get_clock().now()
        dt = (now - self.last_detection_time).nanoseconds / 1e9
        
        if self.filtered_pos is None or dt > FILTER_TIMEOUT_S:
            # RESET FILTER (Gap too long, jump to new position)
            self.filtered_pos = avg_pos
            self.filtered_quat = avg_quat
        else:
            # APPLY SMOOTHING (Continuous movement)
            self.filtered_pos = FILTER_ALPHA * avg_pos + (1.0 - FILTER_ALPHA) * self.filtered_pos
            # Slerp approximation for filter
            self.filtered_quat = FILTER_ALPHA * avg_quat + (1.0 - FILTER_ALPHA) * self.filtered_quat
            # Re-normalize after lerp
            self.filtered_quat /= np.linalg.norm(self.filtered_quat)
        
        # Update last valid detection time
        self.last_detection_time = now

        # 5. Reconstruct the final Smoothed Transform
        self.last_T_w_b = get_tf_matrix_from_values(self.filtered_pos, self.filtered_quat)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoLocalizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()