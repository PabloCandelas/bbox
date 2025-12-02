import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class BBoxYoloDetection(Node):
    def __init__(self):
        super().__init__('bbox_yolo_detection')

        # --- PARAMETERS ---
        self.declare_parameter('model_path', 'bbox/best.pt')
        self.declare_parameter('show_display', True)
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('camera_topic', '/bluerov2/camera/image')

        # Read parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.show_display = self.get_parameter('show_display').get_parameter_value().bool_value
        self.conf_thres = self.get_parameter('conf_thres').get_parameter_value().double_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return

        self.bridge = CvBridge()
        self.detection_pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        # --- WINDOW SETUP (NEW) ---
        if self.show_display:
            # WINDOW_NORMAL allows resizing
            cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
            # Set a starting size (e.g., half-HD) so it fits on your screen
            cv2.resizeWindow("YOLOv8 Detection", 960, 540)
        
        self.get_logger().info(f"Listening on {camera_topic} | Display: {self.show_display}")

    def get_orientation(self, width, height):
        """Returns 'Horizontal' or 'Vertical' based on aspect ratio."""
        if width > height:
            return "Horizontal"
        else:
            return "Vertical"

    def analyze_handle_position(self, box_bbox, handle_bbox):
        bx, by, bw, bh = box_bbox
        hx, hy, hw, hh = handle_bbox

        # 1. Check Vertical relative position
        if hy < (by - bh * 0.3): 
            v_pos = "TOP"
        elif hy > (by + bh * 0.3):
            v_pos = "BOTTOM"
        else:
            v_pos = "CENTER_V"

        # 2. Check Horizontal relative position
        if hx < (bx - bw * 0.3):
            h_pos = "LEFT"
        elif hx > (bx + bw * 0.3):
            h_pos = "RIGHT"
        else:
            h_pos = "CENTER_H"

        return f"{v_pos}-{h_pos}"

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model(cv_image, verbose=False, conf=self.conf_thres)

            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            found_box = None
            found_handle = None

            for r in results:
                for box in r.boxes:
                    # --- DATA EXTRACTION ---
                    x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    # Store for logic check
                    if 'box' in class_name.lower() or 'bbox' in class_name.lower():
                        found_box = (x_center, y_center, width, height)
                    elif 'handle' in class_name.lower():
                        found_handle = (x_center, y_center, width, height)

                    # --- CREATE MESSAGE ---
                    detection = Detection2D()
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(class_id)
                    hypothesis.hypothesis.score = conf
                    detection.results.append(hypothesis)

                    detection.bbox.center.position.x = float(x_center)
                    detection.bbox.center.position.y = float(y_center)
                    detection.bbox.size_x = float(width)
                    detection.bbox.size_y = float(height)
                    detections_msg.detections.append(detection)

                    # --- VISUALIZATION ---
                    if self.show_display:
                        # Logic for Colors
                        name_lower = class_name.lower()
                        orient = self.get_orientation(width, height)
                        
                        # Name, Confidence, and Orientation
                        display_text = f"{class_name} {conf:.2f} ({orient})"

                        if 'handle' in name_lower:
                            box_color = (0, 0, 255) # Red
                        elif 'box' in name_lower or 'bbox' in name_lower:
                            box_color = (255, 0, 255) # Purple
                        else:
                            box_color = (0, 255, 0)

                        # Draw Box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), box_color, 3) 
                        
                        # Header Background
                        (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(cv_image, (x1, y1 - 30), (x1 + w, y1), box_color, -1)
                        
                        # White Text
                        cv2.putText(cv_image, display_text, (x1, y1 - 8), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # --- RELATIONSHIP LOGIC ---
            if found_box and found_handle:
                rel_pos = self.analyze_handle_position(found_box, found_handle)
                box_orient = self.get_orientation(found_box[2], found_box[3])
                
                # Draw relationship on screen
                if self.show_display:
                    status_text = f"Box: {box_orient} | Handle: {rel_pos}"
                    cv2.putText(cv_image, status_text, (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            self.detection_pub.publish(detections_msg)

            if self.show_display:
                # Use the EXACT same name as in __init__
                cv2.imshow("YOLOv8 Detection", cv_image)
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