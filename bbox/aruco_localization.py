#!/usr/bin/env python3
"""
aruco_localization_node.py

- Loads a fixed map of aruco markers in the net frame.
- Listens to detection TFs (camera -> aruco_<id>) and camera tilt (camera_tilt -> camera).
- Computes net -> camera and net -> body (applying configurable offset and rotation).
- Broadcasts TFs for visualization and publishes a single PoseArray topic with the computed poses.

Topics / TFs used/produced:
 - reads: /tf (camera -> aruco_<id> from detector, camera_tilt -> camera from joy node)
 - publishes TFs (via /tf): net -> aruco_map_<id>, net -> camera_in_net, net -> body
 - publishes topic: /aruco_localization/poses (geometry_msgs/PoseArray) with header.frame_id = net_frame
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import numpy as np
import math

# --------- ADJUSTABLE TRANSFORMS / CONSTANTS ----------
NET_FRAME = "net"
BODY_FRAME = "body"
CAMERA_FRAME = "camera"
ARUCO_MAP_PREFIX = "aruco_map_"

CAMERA_TO_BODY_PRE_TRANSLATION_Z = -0.23  # meters
# Correct quaternion to map camera frame -> body frame (Z down, X forward, Y right)
CAMERA_TO_BODY_ROT_QUAT = np.array([0.5, 0.5, 0.5, 0.5])

LOOP_HZ = 10.0

# Marker poses in net frame: (tx, ty, tz, qx, qy, qz, qw)
marker_poses = [
    (4.5, 7.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
    (1.5, 1.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
    (7.5, 4.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
    (7.5, 7.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
    (4.5, 4.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
    (1.5, 7.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
    (1.5, 4.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
    (4.5, 1.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
    (7.5, 1.0, 4.8, 0.7068252, 0.7073883, 0.0000327, 0.0000328),
]

# ---------- helper math utilities ----------
def quat_to_matrix(q):
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        return np.eye(3)
    x /= n; y /= n; z /= n; w /= n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx+zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx+yy)]
    ])
    return R

def matrix_to_quat(R):
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
    return np.array([x, y, z, w])

def make_transform_matrix(t, q):
    R = quat_to_matrix(q)
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = t
    return M

def invert_transform_matrix(M):
    R = M[0:3,0:3]
    t = M[0:3,3]
    Rinv = R.T
    tinv = -Rinv.dot(t)
    Minv = np.eye(4)
    Minv[0:3,0:3] = Rinv
    Minv[0:3,3] = tinv
    return Minv

def transform_matrix_to_pose(M):
    t = M[0:3,3]
    q = matrix_to_quat(M[0:3,0:3])
    p = Pose()
    p.position.x = float(t[0])
    p.position.y = float(t[1])
    p.position.z = float(t[2])
    p.orientation.x = float(q[0])
    p.orientation.y = float(q[1])
    p.orientation.z = float(q[2])
    p.orientation.w = float(q[3])
    return p

def average_poses(pose_matrices):
    if len(pose_matrices) == 0:
        return None
    positions = np.array([M[0:3,3] for M in pose_matrices])
    pos_mean = positions.mean(axis=0)
    quats = np.array([matrix_to_quat(M[0:3,0:3]) for M in pose_matrices])
    qsum = quats.sum(axis=0)
    qnorm = qsum / np.linalg.norm(qsum)
    return make_transform_matrix(pos_mean, qnorm)

# ---------- Node ----------
class ArucoLocalization(Node):
    def __init__(self):
        super().__init__("aruco_localization")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseArray, "/aruco_localization/poses", 10)

        # Precompute marker map
        self.marker_map = {}
        for mid, data in enumerate(marker_poses):
            t = np.array(data[0:3])
            q = np.array(data[3:7])
            self.marker_map[mid] = make_transform_matrix(t, q)

        self.last_broadcast = 0.0
        self.broadcast_interval = 1.0

        self.net_frame = NET_FRAME
        self.camera_frame = CAMERA_FRAME
        self.body_frame = BODY_FRAME
        self.map_prefix = ARUCO_MAP_PREFIX

        self.camera_to_body_pre_t = np.array([0.0, 0.0, CAMERA_TO_BODY_PRE_TRANSLATION_Z])
        self.camera_to_body_rot_q = CAMERA_TO_BODY_ROT_QUAT

        self.timer = self.create_timer(1.0 / LOOP_HZ, self.timer_cb)
        self.get_logger().info("ArucoLocalization ready.")

    def broadcast_marker_map_frames(self):
        for mid, M in self.marker_map.items():
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.net_frame
            t.child_frame_id = f"{self.map_prefix}{mid}"
            t.transform.translation.x = float(M[0,3])
            t.transform.translation.y = float(M[1,3])
            t.transform.translation.z = float(M[2,3])
            q = matrix_to_quat(M[0:3,0:3])
            t.transform.rotation.x = float(q[0])
            t.transform.rotation.y = float(q[1])
            t.transform.rotation.z = float(q[2])
            t.transform.rotation.w = float(q[3])
            self.tf_broadcaster.sendTransform(t)

    def lookup_camera_to_aruco(self, mid):
        aruco_frame = f"aruco_{mid}"
        try:
            t = self.tf_buffer.lookup_transform(aruco_frame, self.camera_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            tx, ty, tz = t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            qx, qy, qz, qw = t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w
            return make_transform_matrix(np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]))
        except Exception:
            return None

    def timer_cb(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_broadcast > self.broadcast_interval:
            self.broadcast_marker_map_frames()
            self.last_broadcast = now

        candidate_camera_in_net = []
        candidate_body_in_net = []

        for mid, M_map in self.marker_map.items():
            M_cam_to_aruco = self.lookup_camera_to_aruco(mid)
            if M_cam_to_aruco is None:
                continue
            M_aruco_to_cam = invert_transform_matrix(M_cam_to_aruco)
            M_net_to_cam = M_map.dot(M_aruco_to_cam)
            candidate_camera_in_net.append(M_net_to_cam)

            T_pre = np.eye(4)
            T_pre[0:3,3] = self.camera_to_body_pre_t
            R_cb = quat_to_matrix(self.camera_to_body_rot_q)
            T_rot = np.eye(4)
            T_rot[0:3,0:3] = R_cb
            M_cam_to_body_local = T_rot.dot(T_pre)
            M_net_to_body = M_net_to_cam.dot(M_cam_to_body_local)
            candidate_body_in_net.append(M_net_to_body)

        pose_array = PoseArray()
        pose_array.header.frame_id = self.net_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()

        if candidate_camera_in_net:
            M_cam = average_poses(candidate_camera_in_net)
            tcam = TransformStamped()
            tcam.header.stamp = self.get_clock().now().to_msg()
            tcam.header.frame_id = self.net_frame
            tcam.child_frame_id = "camera_in_net"
            tc = M_cam[0:3,3]
            rc = matrix_to_quat(M_cam[0:3,0:3])
            tcam.transform.translation.x = float(tc[0])
            tcam.transform.translation.y = float(tc[1])
            tcam.transform.translation.z = float(tc[2])
            tcam.transform.rotation.x = float(rc[0])
            tcam.transform.rotation.y = float(rc[1])
            tcam.transform.rotation.z = float(rc[2])
            tcam.transform.rotation.w = float(rc[3])
            self.tf_broadcaster.sendTransform(tcam)
            pose_array.poses.append(transform_matrix_to_pose(M_cam))

        if candidate_body_in_net:
            M_body = average_poses(candidate_body_in_net)
            tbody = TransformStamped()
            tbody.header.stamp = self.get_clock().now().to_msg()
            tbody.header.frame_id = self.net_frame
            tbody.child_frame_id = self.body_frame
            tb = M_body[0:3,3]
            rb = matrix_to_quat(M_body[0:3,0:3])
            tbody.transform.translation.x = float(tb[0])
            tbody.transform.translation.y = float(tb[1])
            tbody.transform.translation.z = float(tb[2])
            tbody.transform.rotation.x = float(rb[0])
            tbody.transform.rotation.y = float(rb[1])
            tbody.transform.rotation.z = float(rb[2])
            tbody.transform.rotation.w = float(rb[3])
            self.tf_broadcaster.sendTransform(tbody)
            pose_array.poses.append(transform_matrix_to_pose(M_body))

        self.pose_pub.publish(pose_array)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
