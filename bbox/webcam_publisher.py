import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create publisher for ROS 2 Image messages
        self.publisher = self.create_publisher(Image, 'camera/image', 10)

        # Open the webcam
        self.cap = cv2.VideoCapture(0)  # 0 for the default webcam

        if not self.cap.isOpened():
            self.get_logger().error("Could not open webcam.")
            return

        # Set up the timer for periodic image publishing (30 Hz)
        self.timer = self.create_timer(1/30.0, self.timer_callback)  # 30 fps

    def timer_callback(self):
        # Read a frame from the webcam
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to grab frame.")
            return

        # Resize the frame to a smaller size (optional)
        frame = cv2.resize(frame, (640, 480))

        # Convert the frame to a ROS 2 Image message
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        # Publish the image
        self.publisher.publish(img_msg)

        # Display the frame using OpenCV (GUI)
        cv2.imshow('Webcam Feed', frame)

        # Check for 'q' key press to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Closing webcam feed window")
            cv2.destroyAllWindows()

    def destroy(self):
        # Release the webcam on shutdown
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    webcam_publisher = WebcamPublisher()
    try:
        rclpy.spin(webcam_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        webcam_publisher.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
