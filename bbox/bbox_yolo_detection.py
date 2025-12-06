#!/usr/bin/env python3
# --- CRITICAL IMPORT ORDER FIX ---
# PyTorch must be imported BEFORE rclpy and cv2 to avoid "CUDA unknown error" 
import torch
import os
import sys

# Ensure CUDA is visible before other libraries load
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point 
from std_msgs.msg import String, Float32 # Added for new topics
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class BBoxYoloDetection(Node):
    def __init__(self):
        super().__init__('bbox_yolo_detection')

        # --- DIAGNOSTICS ---
        self.get_logger().info(f"PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            self.get_logger().info(f"✅ USING GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.get_logger().warn("⚠️  Running on CPU")

        # --- PARAMETERS ---
        self.declare_parameter('model_path', 'bbox/best.pt')
        self.declare_parameter('show_display', True)
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('camera_topic', '/bluerov2/camera/image')
        
        # Servoing Target Adjustments (Pixels from center)
        self.declare_parameter('target_offset_x', 0) 
        self.declare_parameter('target_offset_y', 0)

        # Read parameters
        model_path_param = self.get_parameter('model_path').get_parameter_value().string_value
        self.show_display = self.get_parameter('show_display').get_parameter_value().bool_value
        self.conf_thres = self.get_parameter('conf_thres').get_parameter_value().double_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

        # Internal State for Target (Initialized from params)
        self.target_off_x = self.get_parameter('target_offset_x').value
        self.target_off_y = self.get_parameter('target_offset_y').value
        self.img_w = 640 
        self.img_h = 480 

        # --- SMART MODEL LOADING ---
        final_model_path = os.path.expanduser(model_path_param)
        if not os.path.exists(final_model_path):
            self.get_logger().warn(f"Model not found at: {final_model_path}")
            workspace_fallback = os.path.join(os.path.expanduser("~"), "ros2_ws/src/bbox", model_path_param)
            if os.path.exists(workspace_fallback):
                self.get_logger().info(f"Found model in workspace: {workspace_fallback}")
                final_model_path = workspace_fallback
            else:
                self.get_logger().error(f"❌ Could not find model file!")
                return 

        self.get_logger().info(f"Loading YOLO model from: {final_model_path}")
        
        # --- GPU SETUP ---
        try:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                dummy = torch.zeros(1).to(self.device)
            else:
                self.device = 'cpu'
        except Exception as e:
            self.get_logger().error(f"CUDA Init warning: {e}. Falling back to CPU.")
            self.device = 'cpu'

        try:
            self.model = YOLO(final_model_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return

        self.bridge = CvBridge()
        qos_profile = rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        
        # --- PUBLISHERS ---
        self.detection_pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.pub_target = self.create_publisher(Point, '/yolo/target_point', 10)
        
        # Servoing Error
        self.pub_box_err = self.create_publisher(Point, '/yolo/box_error', 10)
        self.pub_hdl_err = self.create_publisher(Point, '/yolo/handle_error', 10)
        
        # Dimensions (x=width, y=height)
        self.pub_box_dim = self.create_publisher(Point, '/yolo/box_dim', 10)
        self.pub_hdl_dim = self.create_publisher(Point, '/yolo/handle_dim', 10)
        
        # Confidence
        self.pub_box_conf = self.create_publisher(Float32, '/yolo/box_conf', 10)
        self.pub_hdl_conf = self.create_publisher(Float32, '/yolo/handle_conf', 10)
        
        # Orientation
        self.pub_box_orient = self.create_publisher(String, '/yolo/box_orientation', 10)

        self.subscription = self.create_subscription(Image, camera_topic, self.image_callback, qos_profile)
        
        # --- WINDOW SETUP ---
        if self.show_display:
            try:
                cv2.namedWindow("YOLOv8 Servoing", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLOv8 Servoing", 960, 540)
                cv2.setMouseCallback("YOLOv8 Servoing", self.mouse_callback)
            except Exception:
                pass
        
        self.get_logger().info(f"Listening on {camera_topic} | Display: {self.show_display}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            center_x = self.img_w / 2
            center_y = self.img_h / 2
            self.target_off_x = int(x - center_x)
            self.target_off_y = int(y - center_y)
            self.get_logger().info(f"New Target Set: Offset({self.target_off_x}, {self.target_off_y})")

    def get_orientation(self, width, height):
        if width > height: return "Horizontal"
        else: return "Vertical"

    def analyze_handle_position(self, box_bbox, handle_bbox):
        bx, by, bw, bh = box_bbox
        hx, hy, hw, hh = handle_bbox
        
        if hy < (by - bh * 0.3): v_pos = "TOP"
        elif hy > (by + bh * 0.3): v_pos = "BOTTOM"
        else: v_pos = "CENTER_V"

        if hx < (bx - bw * 0.3): h_pos = "LEFT"
        elif hx > (bx + bw * 0.3): h_pos = "RIGHT"
        else: h_pos = "CENTER_H"

        return f"{v_pos}-{h_pos}"

    def draw_hud(self, img, box_data, handle_data, ref_pt):
        h_img, w_img = img.shape[:2]
        rx, ry = ref_pt
        
        lines = []
        
        # 1. Box Data
        if box_data:
            # box_data = (x, y, w, h, conf, box_obj)
            bx, by, bw, bh, b_conf, _ = box_data
            err_x = int(bx - rx)
            err_y = int(by - ry) 
            orient = self.get_orientation(bw, bh)
            lines.append(f"BOX Conf: {b_conf:.2f}")
            lines.append(f"BOX Dim: {bw:.0f}x{bh:.0f} ({orient})")
            lines.append(f"BOX Err: dx={err_x} dy={err_y}")
            cv2.line(img, (int(rx), int(ry)), (int(bx), int(by)), (255, 0, 255), 1)
            cv2.circle(img, (int(bx), int(by)), 4, (255, 0, 255), -1)
        else:
            lines.append("BOX: Not Found")

        # 2. Handle Data
        if handle_data:
            # handle_data = (x, y, w, h, conf, box_obj)
            hx, hy, hw, hh, h_conf, _ = handle_data
            err_x = int(hx - rx)
            err_y = int(hy - ry)
            lines.append(f"HDL Conf: {h_conf:.2f}")
            lines.append(f"HDL Dim: {hw:.0f}x{hh:.0f}")
            lines.append(f"HDL Err: dx={err_x} dy={err_y}")
            cv2.line(img, (int(rx), int(ry)), (int(hx), int(hy)), (0, 0, 255), 1)
            cv2.circle(img, (int(hx), int(hy)), 4, (0, 0, 255), -1)
        else:
            lines.append("HDL: Not Found")

        # Draw Text Block
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thick = 2
        y_start = 30
        max_w = 0
        for line in lines:
            (w, _), _ = cv2.getTextSize(line, font, scale, thick)
            max_w = max(max_w, w)
            
        x_start = w_img - max_w - 20
        sub_img = img[0:y_start + len(lines)*30, x_start-10:w_img]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        img[0:y_start + len(lines)*30, x_start-10:w_img] = res

        for i, line in enumerate(lines):
            y = y_start + i * 30
            color = (0, 255, 255) 
            if "BOX" in line: color = (255, 0, 255) 
            if "HDL" in line: color = (0, 0, 255)   
            cv2.putText(img, line, (x_start, y), font, scale, color, thick)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8").copy()
            self.img_h, self.img_w = cv_image.shape[:2]

            # --- SERVOING TARGET ---
            ref_x = int(self.img_w / 2 + self.target_off_x)
            ref_y = int(self.img_h / 2 + self.target_off_y)
            ref_pt = (ref_x, ref_y)

            # Publish Target Point
            pt_msg = Point()
            pt_msg.x = float(ref_x)
            pt_msg.y = float(ref_y)
            pt_msg.z = 0.0
            self.pub_target.publish(pt_msg)

            # --- INFERENCE ---
            results = self.model(cv_image, verbose=False, conf=self.conf_thres, device=self.device, half=False)

            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            # Temporary lists to find highest confidence
            found_boxes = []   # list of (conf, x, y, w, h, box_obj)
            found_handles = [] # list of (conf, x, y, w, h, box_obj)

            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes.cpu().numpy()

                for box in boxes:
                    x_c, y_c, w, h = box.xywh[0]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]

                    data_tuple = (conf, x_c, y_c, w, h, box)

                    if 'box' in name.lower() or 'bbox' in name.lower():
                        found_boxes.append(data_tuple)
                    elif 'handle' in name.lower():
                        found_handles.append(data_tuple)

            # --- FILTERING: PICK BEST ---
            best_box = None
            best_handle = None

            if found_boxes:
                # Sort by confidence descending and pick first
                found_boxes.sort(key=lambda x: x[0], reverse=True)
                best_box = found_boxes[0] # (conf, x, y, w, h, box_obj)
            
            if found_handles:
                found_handles.sort(key=lambda x: x[0], reverse=True)
                best_handle = found_handles[0]

            # --- PROCESS & PUBLISH BEST DETECTIONS ---
            # Helper to create detection msg
            def add_detection_msg(data):
                conf, x, y, w, h, box_obj = data
                d = Detection2D()
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(box_obj.cls[0]))
                hyp.hypothesis.score = conf
                d.results.append(hyp)
                d.bbox.center.position.x = float(x)
                d.bbox.center.position.y = float(y)
                d.bbox.size_x = float(w)
                d.bbox.size_y = float(h)
                detections_msg.detections.append(d)

            # 1. Process Best Box
            if best_box:
                conf, bx, by, bw, bh, box_obj = best_box
                add_detection_msg(best_box)
                
                # Publish Box Error
                err = Point()
                err.x = bx - ref_x
                err.y = by - ref_y
                self.pub_box_err.publish(err)
                
                # Publish Box Dim
                dim = Point()
                dim.x = float(bw)
                dim.y = float(bh)
                self.pub_box_dim.publish(dim)
                
                # Publish Box Conf
                c_msg = Float32()
                c_msg.data = conf
                self.pub_box_conf.publish(c_msg)
                
                # Publish Box Orientation
                orient_msg = String()
                orient_msg.data = self.get_orientation(bw, bh)
                self.pub_box_orient.publish(orient_msg)

                # Visualization (Only draw best)
                if self.show_display:
                    x1, y1, x2, y2 = map(int, box_obj.xyxy[0])
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    label = f"BOX {conf:.2f}"
                    # Move BOX label to the TOP
                    cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # 2. Process Best Handle
            if best_handle:
                conf, hx, hy, hw, hh, box_obj = best_handle
                add_detection_msg(best_handle)
                
                # Publish Handle Error
                err = Point()
                err.x = hx - ref_x
                err.y = hy - ref_y
                self.pub_hdl_err.publish(err)
                
                # Publish Handle Dim
                dim = Point()
                dim.x = float(hw)
                dim.y = float(hh)
                self.pub_hdl_dim.publish(dim)
                
                # Publish Handle Conf
                c_msg = Float32()
                c_msg.data = conf
                self.pub_hdl_conf.publish(c_msg)

                # Visualization (Only draw best)
                if self.show_display:
                    x1, y1, x2, y2 = map(int, box_obj.xyxy[0])
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"HANDLE {conf:.2f}"
                    # Move HANDLE label to the RIGHT
                    cv2.putText(cv_image, label, (x2 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            self.detection_pub.publish(detections_msg)

            # --- DRAW HUD ---
            if self.show_display:
                # Prepare clean data tuples for drawing (x, y, w, h) -> remove extra info if needed
                # draw_hud expects (x, y, w, h) or similar. 
                # best_box tuple is (conf, x, y, w, h, box_obj). 
                # We need to rearrange for draw_hud which I updated to handle the tuple order or modify call
                
                # Let's clean the data passed to draw_hud to: (x, y, w, h, conf, obj)
                b_data = (best_box[1], best_box[2], best_box[3], best_box[4], best_box[0], best_box[5]) if best_box else None
                h_data = (best_handle[1], best_handle[2], best_handle[3], best_handle[4], best_handle[0], best_handle[5]) if best_handle else None

                cv2.circle(cv_image, ref_pt, 15, (0, 255, 255), 2)
                cv2.line(cv_image, (ref_x - 25, ref_y), (ref_x + 25, ref_y), (0, 255, 255), 2)
                cv2.line(cv_image, (ref_x, ref_y - 25), (ref_x, ref_y + 25), (0, 255, 255), 2)

                self.draw_hud(cv_image, b_data, h_data, ref_pt)

                if best_box and best_handle:
                    # pass just the bounding box tuple (x,y,w,h) for logic
                    box_rect = (best_box[1], best_box[2], best_box[3], best_box[4])
                    hdl_rect = (best_handle[1], best_handle[2], best_handle[3], best_handle[4])
                    rel_pos = self.analyze_handle_position(box_rect, hdl_rect)
                    cv2.putText(cv_image, f"Rel: {rel_pos}", (20, self.img_h - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                cv2.imshow("YOLOv8 Servoing", cv_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BBoxYoloDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()