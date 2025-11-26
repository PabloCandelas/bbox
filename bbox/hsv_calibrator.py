# PDCP Color calibration for object tracking ROS2 version
# Last update 21/abril/2025

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class HSVCalibrationNode(Node):
    def __init__(self):
        super().__init__('hsv_calibrator')
        self.bridge = CvBridge()

        # Subscribe to image topic
        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            1
        )

        # Publisher for HSV slider values
        self.hsv_pub = self.create_publisher(Int32MultiArray, '/hsv_values', 10)

        self.latest_frame = None

        # Create slider window
        cv2.namedWindow("Color Tuner")
        cv2.createTrackbar("Lower H", "Color Tuner", 0, 179, lambda x: None)
        cv2.createTrackbar("Lower S", "Color Tuner", 0, 255, lambda x: None)
        cv2.createTrackbar("Lower V", "Color Tuner", 0, 255, lambda x: None)
        cv2.createTrackbar("Upper H", "Color Tuner", 179, 179, lambda x: None)
        cv2.createTrackbar("Upper S", "Color Tuner", 255, 255, lambda x: None)
        cv2.createTrackbar("Upper V", "Color Tuner", 255, 255, lambda x: None)

        # Timer for UI loop (~30 fps)
        self.timer = self.create_timer(0.03, self.update_display)

        print('\nHola!!\n')
        print("   Move the sliders to adjust the HSV values to define the object")
        print("\t- Click on the video to see the HSV of a pixel")
        print("\t- Press 'p' to publish current HSV slider values")
        print("\t- Press 'q' to quit\n")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def update_display(self):
        if self.latest_frame is None:
            return

        frame = cv2.resize(self.latest_frame, (300, 300))  # Adjust size if needed
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.namedWindow("Color Calibration")

        # Mouse callback for HSV pixel info
        cv2.setMouseCallback("Color Calibration", self.get_hsv_on_click, hsv)

        # Get slider values
        lh = cv2.getTrackbarPos("Lower H", "Color Tuner")
        ls = cv2.getTrackbarPos("Lower S", "Color Tuner")
        lv = cv2.getTrackbarPos("Lower V", "Color Tuner")
        uh = cv2.getTrackbarPos("Upper H", "Color Tuner")
        us = cv2.getTrackbarPos("Upper S", "Color Tuner")
        uv = cv2.getTrackbarPos("Upper V", "Color Tuner")

        lower = np.array([lh, ls, lv])
        upper = np.array([uh, us, uv])

        # Mask and visualization
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        combined = np.vstack((
            np.hstack((frame, mask_bgr)),
            np.hstack((result, np.zeros_like(result)))
        ))

        cv2.imshow("Color Calibration", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()
        elif key == ord('p'):
            hsv_msg = Int32MultiArray()
            hsv_msg.data = [lh, ls, lv, uh, us, uv]
            self.hsv_pub.publish(hsv_msg)
            print(f"\nPublished HSV Values:")
            print(f"Lower: H={lh}, S={ls}, V={lv}")
            print(f"Upper: H={uh}, S={us}, V={uv}")

    def get_hsv_on_click(self, event, x, y, flags, hsv_image):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y < hsv_image.shape[0] and x < hsv_image.shape[1]:
                hsv_value = hsv_image[y, x]
                print(f"Clicked HSV at ({x},{y}): H={hsv_value[0]}, S={hsv_value[1]}, V={hsv_value[2]}")

def main(args=None):
    rclpy.init(args=args)
    node = HSVCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    print("\n\tBYEEE\n")

if __name__ == '__main__':
    main()
