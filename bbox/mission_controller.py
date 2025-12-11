import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# --- Message Imports ---
from sensor_msgs.msg import BatteryState, FluidPressure
from mavros_msgs.msg import SysStatus
from std_msgs.msg import Bool, String, Float32
from tf2_msgs.msg import TFMessage

import os
import time
from enum import Enum
from datetime import datetime

class MissionState(Enum):
    INIT = 0
    GROUND_CHECK = 1
    READY_TO_DEPLOY = 2         # Waiting for User to put in water and press A
    SEARCHING_BOX = 3           # Step 1: Just find the object reliably
    DETERMINING_ORIENTATION = 4 # Step 2: Once object is found, figure out angle
    VERIFY_TARGET = 5           # Step 3: Ask User: Is this right? (A/B)
    APPROACH_BOX = 6            # Step 4: Move towards it

class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')

        # --- Settings ---
        self.min_battery_voltage = 14.5
        
        # Detection Config
        self.required_conf_level = 0.80      
        self.box_lock_frames = 20            # ~2 seconds to confirm box existence
        self.orient_lock_frames = 50         # ~5 seconds to confirm orientation (slower, more robust)

        # --- Log File Setup ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_folder = os.path.join(script_dir, 'mission_logs')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_folder, f"mission_report_{timestamp_str}.txt")
        self.get_logger().info(f"Saving logs to: {self.log_file_path}")

        # --- State Management ---
        self.current_state = MissionState.INIT
        self.create_mission_log_file()

        # --- Data Buffers ---
        self.battery_data = None
        self.sys_status = None
        self.pressure_data = None
        
        # Inputs
        self.button_a_pressed = False
        self.button_b_pressed = False
        self.last_tf_time = 0.0
        
        # Yolo Data
        self.yolo_conf = 0.0
        self.yolo_orientation = "Unknown"

        # --- Accumulators ---
        self.box_stability_counter = 0       # Stage 1 Counter
        
        self.orient_stability_counter = 0    # Stage 2 Counter
        self.orient_candidate = "Unknown"    # The orientation we are currently testing

        # --- QoS Profile ---
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- Subscribers ---
        self.create_subscription(BatteryState, '/bluerov2/battery', self.battery_callback, qos_sensor)
        self.create_subscription(SysStatus, '/bluerov2/sys_status', self.sys_status_callback, qos_sensor)
        self.create_subscription(FluidPressure, '/bluerov2/imu/static_pressure', self.pressure_callback, qos_sensor)
        self.create_subscription(Bool, '/bluerov2/buttons/A', self.button_a_callback, qos_sensor)
        self.create_subscription(Bool, '/bluerov2/buttons/B', self.button_b_callback, qos_sensor)
        self.create_subscription(TFMessage, '/tf', self.tf_callback, qos_sensor) 
        self.create_subscription(Float32, '/yolo/box_conf', self.yolo_conf_callback, qos_sensor)
        self.create_subscription(String, '/yolo/box_orientation', self.yolo_orient_callback, qos_sensor)

        # --- Main Control Loop (10Hz) ---
        self.timer = self.create_timer(0.1, self.control_loop)

    # --- Callbacks ---
    def battery_callback(self, msg): self.battery_data = msg
    def sys_status_callback(self, msg): self.sys_status = msg
    def pressure_callback(self, msg): self.pressure_data = msg
    def button_a_callback(self, msg): 
        if msg.data: self.button_a_pressed = True
    def button_b_callback(self, msg): 
        if msg.data: self.button_b_pressed = True
    def tf_callback(self, msg): self.last_tf_time = time.time()
    def yolo_conf_callback(self, msg): self.yolo_conf = msg.data
    def yolo_orient_callback(self, msg): self.yolo_orientation = msg.data

    # --- Logging Helper ---
    def create_mission_log_file(self):
        with open(self.log_file_path, 'w') as f:
            f.write(f"MISSION REPORT - START: {datetime.now()}\n")
            f.write("--------------------------------------------------\n")

    def log_event(self, message, level='INFO'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        
        if level == 'ERROR': self.get_logger().error(message)
        elif level == 'WARN': self.get_logger().warn(message)
        else: self.get_logger().info(message)
            
        with open(self.log_file_path, 'a') as f:
            f.write(formatted_msg + "\n")

    # --- Main Logic ---
    def control_loop(self):
        # 0. Background: Localization Heartbeat
        if (time.time() - self.last_tf_time) < 0.2:
            # TF is active. 
            pass

        # 1. INIT
        if self.current_state == MissionState.INIT:
            if self.battery_data and self.sys_status:
                self.log_event("Sensors Connected. Starting Ground Checks.")
                self.current_state = MissionState.GROUND_CHECK

        # 2. GROUND CHECK
        elif self.current_state == MissionState.GROUND_CHECK:
            if self.perform_ground_checks():
                self.log_event(">>> GROUND CHECKS PASSED. <<<")
                self.log_event("Waiting for Deployment. PRESS 'A' WHEN READY.")
                self.button_a_pressed = False
                self.current_state = MissionState.READY_TO_DEPLOY

        # 3. READY TO DEPLOY
        elif self.current_state == MissionState.READY_TO_DEPLOY:
            if self.button_a_pressed:
                self.log_event(">>> MISSION START: SEARCHING FOR BOX. <<<")
                self.button_a_pressed = False
                self.reset_detection_counters()
                self.current_state = MissionState.SEARCHING_BOX

        # 4. SEARCHING BOX (Stage 1: Is something there?)
        elif self.current_state == MissionState.SEARCHING_BOX:
            # Increment if confident, Decay (decrement) if not
            if self.yolo_conf > self.required_conf_level:
                self.box_stability_counter += 1
            else:
                self.box_stability_counter = max(0, self.box_stability_counter - 1)

            # Check Threshold
            if self.box_stability_counter >= self.box_lock_frames:
                self.log_event("BOX DETECTED (Stable). Determining Orientation...")
                self.current_state = MissionState.DETERMINING_ORIENTATION
                # Initialize candidate with whatever we see right now
                self.orient_candidate = self.yolo_orientation 
                self.orient_stability_counter = 10 # Give it a small head start

        # 5. DETERMINING ORIENTATION (Stage 2: What is it?)
        elif self.current_state == MissionState.DETERMINING_ORIENTATION:
            # Safety: If box is lost completely, go back
            if self.yolo_conf < 0.5:
                self.log_event("Lost Box Visibility! Restarting Search...", level='WARN')
                self.reset_detection_counters()
                self.current_state = MissionState.SEARCHING_BOX
                return

            # Decay/Reinforce Logic
            if self.yolo_orientation == self.orient_candidate:
                self.orient_stability_counter += 1
            else:
                # Different orientation seen? Decay the counter.
                self.orient_stability_counter -= 1

            # Swapping Logic: If counter hits zero, the new orientation wins
            if self.orient_stability_counter <= 0:
                self.log_event(f"Orientation Changed candidate to: {self.yolo_orientation}")
                self.orient_candidate = self.yolo_orientation
                self.orient_stability_counter = 1

            # Success Logic
            if self.orient_stability_counter >= self.orient_lock_frames:
                self.log_event(f">>> ORIENTATION LOCKED: {self.orient_candidate} <<<")
                self.log_event("Please Verify: Press A (Confirm) or B (Reject)")
                self.button_a_pressed = False
                self.button_b_pressed = False
                self.current_state = MissionState.VERIFY_TARGET

        # 6. VERIFY TARGET (User Input)
        elif self.current_state == MissionState.VERIFY_TARGET:
            if self.button_a_pressed:
                self.log_event(f"USER CONFIRMED ({self.orient_candidate}). Moving to Approach.")
                self.button_a_pressed = False
                self.current_state = MissionState.APPROACH_BOX
                
            elif self.button_b_pressed:
                self.log_event("USER REJECTED TARGET. Resuming Search...", level='WARN')
                self.button_b_pressed = False
                self.reset_detection_counters()
                self.current_state = MissionState.SEARCHING_BOX

        # 7. APPROACH
        elif self.current_state == MissionState.APPROACH_BOX:
            # Next steps
            pass

    def reset_detection_counters(self):
        self.box_stability_counter = 0
        self.orient_stability_counter = 0
        self.orient_candidate = "Unknown"

    def perform_ground_checks(self):
        if self.battery_data.voltage < self.min_battery_voltage:
            self.log_event(f"BATTERY FAIL: {self.battery_data.voltage:.2f}V", level='ERROR')
            return False
        return True

def main(args=None):
    rclpy.init(args=args)
    node = MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.log_event("Mission Aborted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()