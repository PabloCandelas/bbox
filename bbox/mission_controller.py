#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# --- Message Imports ---
from sensor_msgs.msg import BatteryState, FluidPressure
from mavros_msgs.msg import SysStatus
from std_msgs.msg import Bool, String, Float32, Float64
from tf2_msgs.msg import TFMessage

import os
import time
from enum import Enum
from datetime import datetime

class MissionState(Enum):
    INIT = 0
    GROUND_CHECK = 1
    READY_TO_DEPLOY = 2         # Waiting for User to put in water and press B
    SEARCHING_BOX = 3           # Step 1: Just find the object reliably
    DETERMINING_ORIENTATION = 4 # Step 2: Once object is found, figure out angle
    VERIFY_TARGET = 5           # Step 3: Ask User: Is this right? (Press B)
    APPROACH_BOX = 6            # Step 4: Visual Servoing takes over (User holds A)
    VERIFY_GRASP = 7            # Step 5: Did we get it? (A=No, B=Yes)
    RESETTING_POSITION = 8      # Step 5b: Failed grasp, manual reposition
    RETURN_TO_SURFACE = 9       # Step 6: Return to surface
    FINAL_CONFIRMATION = 10     # Step 7: On surface, confirm recovery
    MISSION_COMPLETE = 11

class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')

        # --- Settings ---
        self.min_battery_voltage = 14.5
        
        # Detection Config
        self.required_conf_level = 0.80      
        self.box_lock_frames = 20            # ~2 seconds to confirm box existence
        self.orient_lock_frames = 50         # ~5 seconds to confirm orientation

        # --- Log File Setup ---
        # UPDATED: Use explicit path to source directory as requested
        # os.path.expanduser("~") resolves to /home/pablo
        home_dir = os.path.expanduser("~")
        log_folder = os.path.join(home_dir, 'ros2_ws/src/bbox/bbox/mission_logs')
        
        if not os.path.exists(log_folder):
            try:
                os.makedirs(log_folder)
            except Exception as e:
                self.get_logger().warn(f"Could not create log folder at {log_folder}: {e}")
                # Fallback to home if the deep path fails
                log_folder = os.path.join(home_dir, 'mission_logs')
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
        self.current_depth = 0.0
        
        # Inputs
        self.button_a_pressed = False
        self.button_b_pressed = False
        self.last_tf_time = 0.0
        
        # Yolo Data
        self.yolo_conf = 0.0
        self.yolo_orientation = "Unknown"
        
        # Servoing Data
        self.last_servoing_status = ""
        self.monitor_servoing = False # Gate to ignore servoing logs when not active

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

        # --- Publishers ---
        # NEW: Publish logs to a topic for live monitoring
        self.pub_mission_log = self.create_publisher(String, '/mission/log', 10)

        # --- Subscribers ---
        self.create_subscription(BatteryState, '/bluerov2/battery', self.battery_callback, qos_sensor)
        self.create_subscription(SysStatus, '/bluerov2/sys_status', self.sys_status_callback, qos_sensor)
        self.create_subscription(FluidPressure, '/bluerov2/imu/static_pressure', self.pressure_callback, qos_sensor)
        self.create_subscription(Bool, '/bluerov2/buttons/A', self.button_a_callback, qos_sensor)
        self.create_subscription(Bool, '/bluerov2/buttons/B', self.button_b_callback, qos_sensor)
        self.create_subscription(TFMessage, '/tf', self.tf_callback, qos_sensor) 
        self.create_subscription(Float32, '/yolo/box_conf', self.yolo_conf_callback, qos_sensor)
        self.create_subscription(String, '/yolo/box_orientation', self.yolo_orient_callback, qos_sensor)
        
        # Listen to visual servoing status & Depth
        self.create_subscription(String, '/servoing/status', self.servoing_status_callback, qos_sensor)
        self.create_subscription(Float64, '/bluerov2/global_position/rel_alt', self.depth_callback, qos_sensor)

        # --- Main Control Loop (10Hz) ---
        self.timer = self.create_timer(0.1, self.control_loop)

    # --- Callbacks ---
    def battery_callback(self, msg): self.battery_data = msg
    def sys_status_callback(self, msg): self.sys_status = msg
    def pressure_callback(self, msg): self.pressure_data = msg
    def depth_callback(self, msg): self.current_depth = msg.data
    
    def button_a_callback(self, msg): 
        if msg.data: self.button_a_pressed = True
        
    def button_b_callback(self, msg): 
        if msg.data: self.button_b_pressed = True
        
    def tf_callback(self, msg): self.last_tf_time = time.time()
    def yolo_conf_callback(self, msg): self.yolo_conf = msg.data
    def yolo_orient_callback(self, msg): self.yolo_orientation = msg.data

    def servoing_status_callback(self, msg):
        # Gate: Only process if we are actively monitoring servoing (during approach)
        if not self.monitor_servoing:
            return

        # Only log changes to keep the file clean
        current_status = msg.data
        if current_status != self.last_servoing_status:
            self.last_servoing_status = current_status
            self.log_event(f"SERVOING UPDATE: {current_status}")
            
            # Auto-detect completion
            # Updated triggers to match verbose output from visual servoing node ("Phase 7..." or "MISSION COMPLETE")
            trigger_phrases = ["MISSION COMPLETE", "Phase 7", "Phase 8"]
            
            if any(phrase in current_status for phrase in trigger_phrases) and self.current_state == MissionState.APPROACH_BOX:
                self.log_event(">>> APPROACH SEQUENCE FINISHED (Blind Surge Started). <<<")
                
                # STOP monitoring servoing updates to prevent log spam/state confusion during user interaction
                self.monitor_servoing = False
                
                self.button_a_pressed = False # Clear buffers
                self.button_b_pressed = False
                self.current_state = MissionState.VERIFY_GRASP

    # --- Logging Helper ---
    def create_mission_log_file(self):
        with open(self.log_file_path, 'w') as f:
            f.write(f"MISSION REPORT - START: {datetime.now()}\n")
            f.write("--------------------------------------------------\n")

    def log_event(self, message, level='INFO'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        
        # 1. Console Log
        if level == 'ERROR': self.get_logger().error(message)
        elif level == 'WARN': self.get_logger().warn(message)
        else: self.get_logger().info(message)
            
        # 2. File Log
        with open(self.log_file_path, 'a') as f:
            f.write(formatted_msg + "\n")

        # 3. Topic Publish (New)
        if hasattr(self, 'pub_mission_log'):
            msg = String()
            msg.data = formatted_msg
            self.pub_mission_log.publish(msg)

    # --- Main Logic ---
    def control_loop(self):
        # 0. Background: Localization Heartbeat
        if (time.time() - self.last_tf_time) < 0.2:
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
                self.log_event("Waiting for Deployment. PRESS 'B' WHEN READY TO START.")
                self.button_b_pressed = False
                self.current_state = MissionState.READY_TO_DEPLOY

        # 3. READY TO DEPLOY
        elif self.current_state == MissionState.READY_TO_DEPLOY:
            # Changed to Button B to avoid conflict with Deadman (A)
            if self.button_b_pressed:
                self.log_event(">>> MISSION START (User Button B). SEARCHING FOR BOX. <<<")
                self.button_b_pressed = False
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
                self.orient_candidate = self.yolo_orientation 
                self.orient_stability_counter = 10 

        # 5. DETERMINING ORIENTATION (Stage 2: What is it?)
        elif self.current_state == MissionState.DETERMINING_ORIENTATION:
            # Safety: If box is lost completely, go back
            if self.yolo_conf < 0.5:
                self.log_event("Lost Box Visibility! Restarting Search...", level='WARN')
                self.reset_detection_counters()
                self.current_state = MissionState.SEARCHING_BOX
                return

            if self.yolo_orientation == self.orient_candidate:
                self.orient_stability_counter += 1
            else:
                self.orient_stability_counter -= 1

            # Swapping Logic
            if self.orient_stability_counter <= 0:
                self.log_event(f"Orientation Changed candidate to: {self.yolo_orientation}")
                self.orient_candidate = self.yolo_orientation
                self.orient_stability_counter = 1

            # Success Logic
            if self.orient_stability_counter >= self.orient_lock_frames:
                self.log_event(f">>> ORIENTATION LOCKED: {self.orient_candidate} <<<")
                self.log_event("Target Found. PRESS 'B' TO CONFIRM AND START APPROACH.")
                self.button_b_pressed = False
                self.current_state = MissionState.VERIFY_TARGET

        # 6. VERIFY TARGET (User Input)
        elif self.current_state == MissionState.VERIFY_TARGET:
            # Changed to Button B. Button A is reserved for Deadman Switch.
            if self.button_b_pressed:
                self.log_event(f"USER CONFIRMED ({self.orient_candidate}). Switching to Approach Mode.")
                self.log_event("Phase 1: Going to approximate depth to initiate approach...")
                self.log_event("Please HOLD BUTTON A to engage thrusters for Visual Servoing.")
                self.button_b_pressed = False
                self.monitor_servoing = True # START LISTENING TO SERVOING STATUS
                self.current_state = MissionState.APPROACH_BOX

        # 7. APPROACH (Passive Log)
        elif self.current_state == MissionState.APPROACH_BOX:
            # Handled by servoing_status_callback
            pass

        # 8. VERIFY GRASP (User Input)
        elif self.current_state == MissionState.VERIFY_GRASP:
            # Only log once
            if not getattr(self, '_verify_log_sent', False):
                self.log_event("Did the robot grab the box?")
                self.log_event("PRESS 'B' to CONFIRM Grasp.")
                self.log_event("PRESS 'A' to REJECT and RETRY approach.")
                self._verify_log_sent = True

            if self.button_b_pressed:
                self.log_event("USER: Grasp Confirmed.")
                self.log_event("ACTION: Please release carabiner and return to recovery point.")
                self.button_b_pressed = False
                self._verify_log_sent = False
                self.current_state = MissionState.RETURN_TO_SURFACE
            
            elif self.button_a_pressed:
                self.log_event("USER: Grasp Failed / Rejected.")
                self.log_event("ACTION: Moving to start approach position. Please manually reposition.")
                self.log_event("Press 'A' again when ready to RETRY APPROACH.")
                self.button_a_pressed = False
                self._verify_log_sent = False
                self.current_state = MissionState.RESETTING_POSITION

        # 9. RESETTING POSITION (Retry Logic)
        elif self.current_state == MissionState.RESETTING_POSITION:
            if self.button_a_pressed:
                self.log_event("USER: Ready to Retry.")
                self.log_event("Restarting Visual Approach Sequence...")
                self.button_a_pressed = False
                self.monitor_servoing = True # RESUME LISTENING
                self.current_state = MissionState.APPROACH_BOX

        # 10. RETURN TO SURFACE
        elif self.current_state == MissionState.RETURN_TO_SURFACE:
            # Check Depth (closer to surface than -0.6)
            if self.current_depth > -0.6:
                self.log_event(f"Surface Reached (Depth: {self.current_depth:.2f}m).")
                self.log_event("ACTION: Recover bbox.")
                self.log_event("PRESS 'B' for Final Confirmation.")
                self.button_b_pressed = False
                self.current_state = MissionState.FINAL_CONFIRMATION

        # 11. FINAL CONFIRMATION
        elif self.current_state == MissionState.FINAL_CONFIRMATION:
            if self.button_b_pressed:
                self.log_event("USER: Final Confirmation Received (It is good).")
                
                # Check Battery one last time
                if self.battery_data:
                    self.log_event(f"Final Battery Level: {self.battery_data.voltage:.2f}V")
                
                self.log_event(f"End of mission log ready: {self.log_file_path}")
                self.log_event(">>> MISSION COMPLETE <<<")
                self.current_state = MissionState.MISSION_COMPLETE

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