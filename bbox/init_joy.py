#!/usr/bin/env python3
"""
joy_init: ROS2 joystick-based initialization and control node for BlueROV.

Updates:
- Feature: Added publishers for Button A and Button B states.
  - Topics: "buttons/A" and "buttons/B" (std_msgs/Bool).
  - Behavior: Publishes constantly (True/False) in the timer loop.
- FIXED: Moved Gripper Logic to Timer Loop for smooth incremental control.
- CHANGED: Gripper Toggle Button is now Button 8 (Logitech/Guide button).
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

# Added Bool for button state publishing
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import Joy, Imu
from mavros_msgs.msg import OverrideRCIn, MountControl
from mavros_msgs.srv import CommandLong, StreamRate, SetMode 
from geometry_msgs.msg import Twist

from time import sleep


class JoyInit(Node):
    """Joystick control and initialization node for the ROV."""

    def __init__(self):
        super().__init__("joy_init")

        # Parameters
        self.declare_parameter("run_initialization_test", False)
        self.run_initialization_test = self.get_parameter("run_initialization_test").value

        self.get_logger().info("Starting joy_init node")

        # Publishers
        self.pub_rc_override = self.create_publisher(OverrideRCIn, "rc/override", 10)
        self.pub_angle_deg = self.create_publisher(Twist, "angle_degree", 10)
        self.pub_ang_vel = self.create_publisher(Twist, "angular_velocity", 10)
        self.pub_mount = self.create_publisher(MountControl, "mount_control/command", 10)
        
        # New Camera Tilt Publisher
        self.pub_cam_tilt = self.create_publisher(Float64, "camera/tilt_angle", 10)

        # --- BUTTON STATE PUBLISHERS ---
        self.pub_btn_a = self.create_publisher(Bool, "buttons/A", 10)
        self.pub_btn_b = self.create_publisher(Bool, "buttons/B", 10)

        # --- SERVICE CLIENTS ---
        self.cli_command = self.create_client(CommandLong, "cmd/command")
        self.cli_set_mode = self.create_client(SetMode, "set_mode")
        self.cli_stream_rate = self.create_client(StreamRate, "set_stream_rate")

        # MAVROS Stream Rate
        self.set_stream_rate(25)

        # Internal State
        self.arming = False
        # [Manual, Auto (Depth Hold), Correction]
        self.set_mode = [True, False, False] 
        self.init_angles = True
        self.init_depth = True
        self.depth_ref = 0.0
        self.Correction_yaw = 1500
        self.Correction_depth = 1500
        
        # Current RC commands [Pitch, Roll, Throttle, Yaw, Forward, Lateral]
        self.current_rc = [1500, 1500, 1500, 1500, 1500, 1500]
        
        # Joystick States (Stored for Timer Loop)
        self.dpad_horizontal = 0.0
        self.dpad_vertical = 0.0
        self.btn_a_state = False
        self.btn_b_state = False

        # Lights
        self.light_pin = 12.0
        self.light_min = 1100.0
        self.light_max = 1900.0
        self.light_pwm = 1100.0

        # Camera tilt
        self.camera_pin = 16.0
        self.tilt = 0.0
        self.tilt_min = -60.0
        self.tilt_max = 60.0

        # --- GRIPPER SETTINGS ---
        self.gripper_mode = False  # False = Pitch/Roll, True = Gripper
        self.gripper_pin = 10.0    # Aux Pin 10
        self.gripper_open = 1900.0
        self.gripper_close = 1100.0
        self.gripper_val = 1500.0  # Current tracked gripper position
        self.gripper_last_sent = 0.0 # To avoid spamming service calls
        self.btn_gripper_toggle_prev = 0 # To detect button state change

        # Subscribers
        self.create_subscribers()

        # Timers
        self.timer = self.create_timer(0.05, self.timer_callback)

        # Initialization Test
        if self.run_initialization_test:
            self.run_init_test()

        self.get_logger().info("joy_init node ready.")

    # Initialization Test
    def run_init_test(self):
        self.get_logger().info("Running initialization test...")

        for pwm in [self.light_min, self.light_max, self.light_min]:
            self.send_servo_cmd(self.light_pin, pwm)
            sleep(0.5)

        for angle in [0, self.tilt_max, self.tilt_min, 0]:
            self.set_camera_tilt(angle)
            sleep(0.5)

        self.get_logger().info("Initialization test complete.")

    # MAVROS Commands
    def send_servo_cmd(self, pin, value):
        if self.cli_command.service_is_ready():
            req = CommandLong.Request()
            req.command = 183
            req.param1 = float(pin)
            req.param2 = float(value)
            self.cli_command.call_async(req)
        else:
            self.get_logger().warn("MAVROS Command service not ready")

    def set_stream_rate(self, rate):
        if not self.cli_stream_rate.wait_for_service(timeout_sec=2.0):
             self.get_logger().warn("StreamRate service not available (skipping)")
             return

        req = StreamRate.Request()
        req.stream_id = 0
        req.message_rate = rate
        req.on_off = True
        self.cli_stream_rate.call_async(req)

    # MODE SWITCHING
    def set_ardusub_mode(self, mode_name):
        if self.cli_set_mode.service_is_ready():
            req = SetMode.Request()
            req.custom_mode = mode_name
            future = self.cli_set_mode.call_async(req)
            future.add_done_callback(lambda future: self.get_logger().info(f"Mode switch to {mode_name} request sent."))
            return True
        else:
            self.get_logger().error(f"Cannot switch to {mode_name}: Service not ready!")
            return False

    def arm_disarm(self, arm):
        if self.cli_command.service_is_ready():
            req = CommandLong.Request()
            req.command = 400
            req.param1 = 1.0 if arm else 0.0
            self.cli_command.call_async(req)
            self.get_logger().info("Armed" if arm else "Disarmed")
        else:
            self.get_logger().error("Cannot Arm/Disarm: Service not ready")

    # Subscribers
    def create_subscribers(self):
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.create_subscription(Joy, "joy", self.joy_cb, qos)
        self.create_subscription(Twist, "cmd_vel", self.cmd_vel_cb, qos)
        self.create_subscription(Imu, "imu/data", self.imu_cb, qos)
        self.create_subscription(Float64, "global_position/rel_alt", self.depth_cb, qos)

    # Joystick callback
    def joy_cb(self, joy):
        btn_arm = joy.buttons[7]
        btn_disarm = joy.buttons[6]

        btn_manual = joy.buttons[3]
        btn_auto = joy.buttons[2]
        btn_corr = joy.buttons[0] # Button A

        btn_tilt_up = joy.buttons[4]
        btn_tilt_down = joy.buttons[5]
        btn_tilt_reset = joy.buttons[9]
        
        # --- GRIPPER MODE TOGGLE ---
        btn_gripper_toggle = joy.buttons[8] 

        lt = joy.axes[2]
        rt = joy.axes[5]
        
        # Store States for Timer Loop
        self.dpad_horizontal = joy.axes[6] # LEFT/RIGHT
        self.dpad_vertical = joy.axes[7]   # UP/DOWN
        
        # Store Button A (0) and B (1) states
        self.btn_a_state = bool(joy.buttons[0])
        self.btn_b_state = bool(joy.buttons[1])

        if btn_arm and not self.arming:
            self.arming = True
            self.arm_disarm(True)

        if btn_disarm and self.arming:
            self.arming = False
            self.arm_disarm(False)
            
        # Toggle Logic
        if btn_gripper_toggle and not self.btn_gripper_toggle_prev:
            self.gripper_mode = not self.gripper_mode
            mode_str = "GRIPPER MODE" if self.gripper_mode else "MOTION MODE"
            self.get_logger().info(f"Switched to: {mode_str}")
        self.btn_gripper_toggle_prev = btn_gripper_toggle

        # MODE LOGIC
        if btn_manual:
            if self.set_ardusub_mode("MANUAL"):
                self.set_mode = [True, False, False]
                self.get_logger().info("Switched to MANUAL MODE")

        if btn_auto:
            if self.set_ardusub_mode("ALT_HOLD"):
                self.set_mode = [False, True, False]
                self.get_logger().info("Switched to DEPTH HOLD (ALT_HOLD)")

        if btn_corr:
            self.set_mode = [False, False, True]
            self.init_angles = True
            self.init_depth = True
            self.get_logger().info("Mode: Correction")

        # Light Control
        if rt == -1 and self.light_pwm < self.light_max:
            self.light_pwm = min(self.light_pwm + 100, self.light_max)
            self.send_servo_cmd(self.light_pin, self.light_pwm)

        if lt == -1 and self.light_pwm > self.light_min:
            self.light_pwm = max(self.light_pwm - 100, self.light_min)
            self.send_servo_cmd(self.light_pin, self.light_pwm)

        # Tilt Control
        if btn_tilt_up:
            self.tilt = min(self.tilt + 5.0, self.tilt_max)
            self.set_camera_tilt(self.tilt)

        elif btn_tilt_down:
            self.tilt = max(self.tilt - 5.0, self.tilt_min)
            self.set_camera_tilt(self.tilt)

        elif btn_tilt_reset:
            self.tilt = 0.0
            self.set_camera_tilt(self.tilt)

    # CMD VEL UPDATES STATE ONLY
    def cmd_vel_cb(self, msg):
        # Helper with DEADZONE to prevent drift in ALT_HOLD
        def map_pwm(v):
            if abs(v) < 0.05:
                v = 0.0
            return int(np.clip(v * 400 + 1500, 1100, 1900))

        self.current_rc[0] = map_pwm(msg.angular.y)
        self.current_rc[1] = map_pwm(msg.angular.x)
        self.current_rc[2] = map_pwm(msg.linear.z)
        self.current_rc[3] = map_pwm(-msg.angular.z)
        self.current_rc[4] = map_pwm(msg.linear.x)
        self.current_rc[5] = map_pwm(-msg.linear.y)

    # RC override
    def send_rc_override(self, pitch, roll, throttle, yaw, forward, lateral):
        msg = OverrideRCIn()
        channels = [65535] * 18
        channels[0] = pitch
        channels[1] = roll
        channels[2] = throttle
        channels[3] = yaw
        channels[4] = forward
        channels[5] = lateral
        msg.channels = channels
        self.pub_rc_override.publish(msg)

    # IMU callback
    def imu_cb(self, imu):
        x, y, z, w = imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w

        sinr = 2 * (w * x + y * z)
        cosr = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny = 2 * (w * z + x * y)
        cosy = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny, cosy)

        if self.init_angles:
            self.roll0, self.pitch0, self.yaw0 = roll, pitch, yaw
            self.init_angles = False

        ang = Twist()
        ang.angular.x = math.degrees(roll - self.roll0)
        ang.angular.y = math.degrees(pitch - self.pitch0)
        ang.angular.z = math.degrees(yaw - self.yaw0)
        self.pub_angle_deg.publish(ang)

        vel = Twist()
        vel.angular = imu.angular_velocity
        self.pub_ang_vel.publish(vel)

    # Depth callback
    def depth_cb(self, msg):
        if self.init_depth:
            self.depth_ref = msg.data
            self.init_depth = False

    # Publish camera tilt Topic
    def publish_camera_tilt_topic(self):
        msg = Float64()
        msg.data = float(self.tilt)
        self.pub_cam_tilt.publish(msg)

    # Set camera tilt
    def set_camera_tilt(self, angle):
        msg = MountControl()
        msg.pitch = float(angle)
        msg.mode = 2
        self.pub_mount.publish(msg)
        self.publish_camera_tilt_topic()

    # MAIN TIMER LOOP
    def timer_callback(self):
        self.publish_camera_tilt_topic()

        # --- PUBLISH BUTTON STATES ---
        msg_a = Bool()
        msg_a.data = self.btn_a_state
        self.pub_btn_a.publish(msg_a)

        msg_b = Bool()
        msg_b.data = self.btn_b_state
        self.pub_btn_b.publish(msg_b)

        # --- GRIPPER CONTROL LOGIC (Moved here for continuous updates) ---
        if self.gripper_mode:
            # UP: Full Open
            if self.dpad_vertical > 0.5:
                self.gripper_val = self.gripper_open
                
            # DOWN: Full Close
            elif self.dpad_vertical < -0.5:
                self.gripper_val = self.gripper_close
            
            # LEFT: Step Close (Continuous if held)
            elif self.dpad_horizontal > 0.5:
                self.gripper_val -= 20.0 
            
            # RIGHT: Step Open (Continuous if held)
            elif self.dpad_horizontal < -0.5:
                self.gripper_val += 20.0 
            
            # Clamp limits
            self.gripper_val = float(np.clip(self.gripper_val, self.gripper_close, self.gripper_open))

            # Only send command if value changed
            if self.gripper_val != self.gripper_last_sent:
                self.send_servo_cmd(self.gripper_pin, self.gripper_val)
                self.gripper_last_sent = self.gripper_val

        # ---------- FIXED SECTION ----------
        
        # LOGIC:
        # If in Manual OR Depth Hold (ALT_HOLD), we send the full RC override.
        if self.set_mode[0] or self.set_mode[1]:
            # Create a copy of the commands to avoid modifying the global state
            rc_to_send = self.current_rc.copy()
            
            # --- GRIPPER SAFETY INTERLOCK ---
            # If Gripper Mode is active, we MUST disable Pitch and Roll 
            # to prevent the robot from moving while we use the arrows for the gripper.
            if self.gripper_mode:
                rc_to_send[0] = 1500 # Pitch Center
                rc_to_send[1] = 1500 # Roll Center

            self.send_rc_override(*rc_to_send)
        
        # -----------------------------------

        # CORRECTION MODE
        if self.set_mode[2]:
            self.send_rc_override(
                1500,
                1500,
                self.Correction_depth,
                self.Correction_yaw,
                1500,
                1500
            )


def main(args=None):
    rclpy.init(args=args)
    node = JoyInit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()