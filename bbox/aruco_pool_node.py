#!/usr/bin/env python3
"""
aruco_pool_node.py
ROS2 Humble Python node for detecting multiple ArUco markers.

Updates:
- Feature: AUTO-SCALING of Camera Matrix. Detects if the calibration resolution 
  differs from the actual image resolution and scales the matrix automatically.
  (Fixes "axes not displaying where they should" / incorrect TF positions).
- REVERTED: TF timestamps now use "Current Time" (self.get_clock().now()) for better responsiveness.
- RETAINED: OpenCV 4.7+ compatibility fix (prevents crashes on newer systems).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import os
import sys

# ---------- Utility: convert rotation matrix to quaternion ----------
def rotation_matrix_to_quaternion(R):
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
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
    return (x, y, z, w)

# ---------- Load camera calibration ----------
def load_camera_calibration(npz_path):
    if not os.path.exists(npz_path):
        raise RuntimeError(f"Calibration file not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if "camera_matrix" in data and "dist_coeffs" in data:
        cam = data["camera_matrix"]
        dist = data["dist_coeffs"]
    elif "mtx" in data and "dist" in data:
        cam = data["mtx"]
        dist = data["dist"]
    else:
        cam = None
        dist = None
        for k in data.files:
            v = data[k]
            if v.shape == (3, 3) and cam is None:
                cam = v
            if (v.ndim == 1 or (v.ndim == 2 and (v.shape[0] == 1 or v.shape[1] in (4,5,8)))) and dist is None:
                dist = v
        if cam is None or dist is None:
            raise RuntimeError(f"Could not find camera matrix/dist coeffs in {npz_path}. Keys: {data.files}")
    return cam.astype(float), dist.astype(float)

# ---------- Convert rvec/tvec to geometry_msgs.Pose ----------
def rvec_tvec_to_pose_msg(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
    pose = Pose()
    pose.position.x = float(tvec[0])
    pose.position.y = float(tvec[1])
    pose.position.z = float(tvec[2])
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)
    return pose

# ---------- The Node ----------
class ArucoPoolNode(Node):
    def __init__(self):
        super().__init__("aruco_pool_node")

        # Parameters
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("camera_info_npz", "bbox/pablos_camera_calb.npz")
        self.declare_parameter("marker_size_m", 0.30)
        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("debug_image_topic", "")

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.npz_path = self.get_parameter("camera_info_npz").get_parameter_value().string_value
        self.marker_size_m = self.get_parameter("marker_size_m").get_parameter_value().double_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.publish_tf = self.get_parameter("publish_tf").get_parameter_value().bool_value
        self.debug_image_topic = self.get_parameter("debug_image_topic").get_parameter_value().string_value

        if not self.npz_path:
            self.get_logger().error("Parameter 'camera_info_npz' is required (path to .npz).")
            raise RuntimeError("camera_info_npz parameter missing")

        self.camera_matrix, self.dist_coeffs = load_camera_calibration(self.npz_path)
        self.get_logger().info(f"Loaded calibration from: {self.npz_path}")
        
        # Flag to check scaling on first frame
        self.matrix_scaled = False

        # Bridge & TF
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        qos = rclpy.qos.QoSProfile(depth=10)
        self.img_sub = self.create_subscription(Image, self.image_topic, self.image_cb, qos)
        self.pose_pub = self.create_publisher(PoseArray, "aruco_pool/poses", qos)
        self.ids_pub = self.create_publisher(Int32MultiArray, "aruco_pool/ids", qos)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, qos) if self.debug_image_topic else None

        # Dictionaries & detectors
        self.dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.dict_6x6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.params = cv2.aruco.DetectorParameters()
        
        # Improve detection stability
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.detector_4x4 = cv2.aruco.ArucoDetector(self.dict_4x4, self.params)
        self.detector_6x6 = cv2.aruco.ArucoDetector(self.dict_6x6, self.params)

        # Marker ID sets
        self.ids_set_A = set(range(0, 9))
        self.ids_set_B = {42}

        # --- WINDOW SETUP ---
        try:
            cv2.namedWindow("Aruco Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Aruco Debug", 960, 540)
        except Exception:
            pass 

        self.get_logger().info("ArucoPoolNode initialized: detecting 4x4 ids 0-8 and 6x6 id 42.")

    def check_and_scale_matrix(self, height, width):
        """ Scales the camera matrix if the image resolution doesn't match the calibration. """
        if self.matrix_scaled:
            return

        # Principal point (cx, cy) is usually at the center (width/2, height/2)
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        expected_w = cx * 2.0
        expected_h = cy * 2.0
        
        # Calculate scale factors
        scale_x = width / expected_w
        scale_y = height / expected_h
        
        # If mismatch is significant (>5%), scale the matrix
        if abs(scale_x - 1.0) > 0.05 or abs(scale_y - 1.0) > 0.05:
            self.get_logger().warn(
                f"Resolution Mismatch! Image: {width}x{height}, Calibration implies: {int(expected_w)}x{int(expected_h)}. "
                f"Scaling matrix by factor X:{scale_x:.2f}, Y:{scale_y:.2f}"
            )
            self.camera_matrix[0, 0] *= scale_x # fx
            self.camera_matrix[1, 1] *= scale_y # fy
            self.camera_matrix[0, 2] *= scale_x # cx
            self.camera_matrix[1, 2] *= scale_y # cy
        else:
            self.get_logger().info(f"Resolution Verified: Image matches Calibration ({width}x{height}).")
            
        self.matrix_scaled = True

    def image_cb(self, msg: Image):
        # Convert ROS Image to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        h, w = frame.shape[:2]
        self.check_and_scale_matrix(h, w)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []

        # --- 4x4 detection ---
        corners_4x4, ids_4x4, _ = self.detector_4x4.detectMarkers(gray)
        if ids_4x4 is not None and len(ids_4x4) > 0:
            for i, mid in enumerate(ids_4x4.flatten()):
                if int(mid) in self.ids_set_A:
                    # Pose Estimation
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners_4x4[i]], self.marker_size_m, self.camera_matrix, self.dist_coeffs
                    )
                    rvec = rvecs[0].flatten()  
                    tvec = tvecs[0].flatten()
                    detections.append((int(mid), corners_4x4[i], rvec, tvec))

        # --- 6x6 detection ---
        corners_6x6, ids_6x6, _ = self.detector_6x6.detectMarkers(gray)
        if ids_6x6 is not None and len(ids_6x6) > 0:
            for i, mid in enumerate(ids_6x6.flatten()):
                if int(mid) in self.ids_set_B:
                    # Pose Estimation
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners_6x6[i]], self.marker_size_m, self.camera_matrix, self.dist_coeffs
                    )
                    rvec = rvecs[0].flatten()
                    tvec = tvecs[0].flatten()
                    detections.append((int(mid), corners_6x6[i], rvec, tvec))

        # --- Prepare messages ---
        pose_array = PoseArray()
        # Use image header for PoseArray to match detection source
        pose_array.header = msg.header 
        
        if self.camera_frame:
             pose_array.header.frame_id = self.camera_frame

        ids_msg = Int32MultiArray()
        ids_msg.data = []
        debug_img = frame.copy()

        # --- Process detections ---
        for marker_id, corners, rvec, tvec in detections:
            pose = rvec_tvec_to_pose_msg(rvec, tvec)
            pose_array.poses.append(pose)
            ids_msg.data.append(int(marker_id))

            # Draw marker & axis (Robust to OpenCV version changes)
            try:
                cv2.aruco.drawDetectedMarkers(debug_img, [corners], np.array([[marker_id]]))
                
                # Try new API (OpenCV 4.7+)
                if hasattr(cv2, 'drawFrameAxes'):
                    cv2.drawFrameAxes(debug_img, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size_m * 0.5)
                # Fallback to old API (OpenCV <= 4.6)
                elif hasattr(cv2.aruco, 'drawAxis'):
                    cv2.aruco.drawAxis(debug_img, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size_m * 0.5)
            except Exception:
                # If both fail, just draw a green box so we know it was detected
                pts = corners.reshape((4, 2)).astype(int)
                cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Overlay readable pose
            c = np.mean(corners.reshape((4, 2)), axis=0).astype(int)
            tx, ty, tz = tvec[0], tvec[1], tvec[2]
            overlay = f"ID:{marker_id} x:{tx:.2f} y:{ty:.2f} z:{tz:.2f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            (w, h), _ = cv2.getTextSize(overlay, font, scale, thickness)
            text_org = (c[0] - w // 2, max(10, c[1] - 10))
            cv2.rectangle(debug_img, (text_org[0]-3, text_org[1]-h-3), (text_org[0]+w+3, text_org[1]+3), (0,0,0), -1)
            cv2.putText(debug_img, overlay, text_org, font, scale, (255,255,255), thickness, cv2.LINE_AA)

            # Publish TF (camera_frame -> aruco_<id>)
            if self.publish_tf:
                t = TransformStamped()
                
                # REVERTED: Use current time for responsiveness (fixes lag/extrapolation issues)
                t.header.stamp = self.get_clock().now().to_msg()
                
                t.header.frame_id = self.camera_frame
                t.child_frame_id = f"aruco_{int(marker_id)}"
                t.transform.translation.x = float(tvec[0])
                t.transform.translation.y = float(tvec[1])
                t.transform.translation.z = float(tvec[2])
                
                R, _ = cv2.Rodrigues(rvec)
                qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
                t.transform.rotation.x = float(qx)
                t.transform.rotation.y = float(qy)
                t.transform.rotation.z = float(qz)
                t.transform.rotation.w = float(qw)
                self.tf_broadcaster.sendTransform(t)

        # --- Publish messages ---
        self.pose_pub.publish(pose_array)
        self.ids_pub.publish(ids_msg)

        # Show debug image
        try:
            cv2.imshow("Aruco Debug", debug_img)
            cv2.waitKey(1)
        except Exception:
            pass

        if self.debug_pub:
            try:
                img_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
                img_msg.header = msg.header
                self.debug_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish debug image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()