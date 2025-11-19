#!/usr/bin/env python3
"""
joy_init: ROS2 joystick-based initialization and control node for BlueROV.

Features:
- Arming / disarming
- Manual / Auto / Correction modes
- Light intensity control
- Camera tilt servo control
- Thruster RC override
- Basic IMU and depth processing
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import Joy, Imu
from mavros_msgs.msg import OverrideRCIn, MountControl
from mavros_msgs.srv import CommandLong, StreamRate
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

        # ------------ Publishers ------------
        self.pub_rc_override = self.create_publisher(OverrideRCIn, "rc/override", 10)
        self.pub_angle_deg = self.create_publisher(Twist, "angle_degree", 10)
        self.pub_ang_vel = self.create_publisher(Twist, "angular_velocity", 10)
        self.pub_mount = self.create_publisher(MountControl, "mount_control/command", 10)

        # ------------ MAVROS Stream Rate ------------
        self.set_stream_rate(25)

        # ------------ Internal State ------------
        self.arming = False
        self.set_mode = [True, False, False]  # Manual, Automatic, Correction
        self.init_angles = True
        self.init_depth = True
        self.depth_ref = 0.0
        self.Correction_yaw = 1500
        self.Correction_depth = 1500

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

        # ------------ Subscribers ------------
        self.create_subscribers()

        # ------------ Timers ------------
        self.timer = self.create_timer(0.05, self.timer_callback)

        # ------------ Initialization Test ------------
        if self.run_initialization_test:
            self.run_init_test()

        self.get_logger().info("Holaa joy_init node ready.")

    # =======================================================================
    # Initialization Test
    # =======================================================================

    def run_init_test(self):
        """Flash the light and sweep the camera tilt."""
        self.get_logger().info("Running initialization test...")

        # Light test
        for pwm in [self.light_min, self.light_max, self.light_min]:
            self.send_servo_cmd(self.light_pin, pwm)
            sleep(0.5)

        # Camera tilt test
        for angle in [0, self.tilt_max, self.tilt_min, 0]:
            self.set_camera_tilt(angle)
            sleep(0.5)

        self.get_logger().info("Initialization test complete.")

    # =======================================================================
    # MAVROS Commands
    # =======================================================================

    def send_servo_cmd(self, pin, value):
        """Send a MAV_CMD_DO_SET_SERVO command."""
        client = self.create_client(CommandLong, "cmd/command")
        client.wait_for_service()

        req = CommandLong.Request()
        req.command = 183          # MAV_CMD_DO_SET_SERVO
        req.param1 = float(pin)
        req.param2 = float(value)

        client.call_async(req)

    def set_stream_rate(self, rate):
        client = self.create_client(StreamRate, "set_stream_rate")
        client.wait_for_service()

        req = StreamRate.Request()
        req.stream_id = 0
        req.message_rate = rate
        req.on_off = True

        client.call_async(req)

    def arm_disarm(self, arm):
        """Arm or disarm the ROV."""
        client = self.create_client(CommandLong, "cmd/command")
        client.wait_for_service()

        req = CommandLong.Request()
        req.command = 400
        req.param1 = 1.0 if arm else 0.0

        client.call_async(req)
        self.get_logger().info("Armed" if arm else "Disarmed")

    # =======================================================================
    # Subscribers and Callbacks
    # =======================================================================

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

    # ---------------- Joystick ----------------

    def joy_cb(self, joy):
        btn_arm = joy.buttons[7]
        btn_disarm = joy.buttons[6]

        btn_manual = joy.buttons[3]
        btn_auto = joy.buttons[2]
        btn_corr = joy.buttons[0]

        btn_tilt_up = joy.buttons[4]
        btn_tilt_down = joy.buttons[5]
        btn_tilt_reset = joy.buttons[9]

        lt = joy.axes[2]
        rt = joy.axes[5]

        # Arming / Disarming
        if btn_arm and not self.arming:
            self.arming = True
            self.arm_disarm(True)

        if btn_disarm and self.arming:
            self.arming = False
            self.arm_disarm(False)

        # Mode switching
        if btn_manual:
            self.set_mode = [True, False, False]
            self.get_logger().info("Mode: Manual")

        if btn_auto:
            self.set_mode = [False, True, False]
            self.get_logger().info("Mode: Automatic")

        if btn_corr:
            self.set_mode = [False, False, True]
            self.init_angles = True
            self.init_depth = True
            self.get_logger().info("Mode: Correction")

        # Lights
        if rt == -1 and self.light_pwm < self.light_max:
            self.light_pwm = min(self.light_pwm + 100, self.light_max)
            self.send_servo_cmd(self.light_pin, self.light_pwm)

        if lt == -1 and self.light_pwm > self.light_min:
            self.light_pwm = max(self.light_pwm - 100, self.light_min)
            self.send_servo_cmd(self.light_pin, self.light_pwm)

        # Camera tilt
        if btn_tilt_up:
            self.tilt = min(self.tilt + 5.0, self.tilt_max)
            self.set_camera_tilt(self.tilt)

        elif btn_tilt_down:
            self.tilt = max(self.tilt - 5.0, self.tilt_min)
            self.set_camera_tilt(self.tilt)

        elif btn_tilt_reset:
            self.tilt = 0.0
            self.set_camera_tilt(self.tilt)

    # ---------------- Manual Velocity ----------------

    def cmd_vel_cb(self, msg):
        """Convert velocity commands to RC PWM in manual mode."""
        if not self.set_mode[0]:   # Only manual
            return

        def map_pwm(v):
            return int(np.clip(v * 400 + 1500, 1100, 1900))

        pitch = map_pwm(msg.angular.y)
        roll = map_pwm(msg.angular.x)
        throttle = map_pwm(msg.linear.z)
        yaw = map_pwm(-msg.angular.z)
        surge = map_pwm(msg.linear.x)
        sway = map_pwm(-msg.linear.y)

        self.send_rc_override(pitch, roll, throttle, yaw, surge, sway)

    # ---------------- RC Override ----------------

    def send_rc_override(self, pitch, roll, throttle, yaw, forward, lateral):
        """Send RC override with exactly 18 channels as required by MAVROS."""
        msg = OverrideRCIn()

        # Create full 18-channel array initialized to 0 (ignored)
        channels = [0] * 18

        # Map outputs to channels 1–6
        channels[0] = pitch
        channels[1] = roll
        channels[2] = throttle
        channels[3] = yaw
        channels[4] = forward
        channels[5] = lateral

        msg.channels = channels
        self.pub_rc_override.publish(msg)

    # ---------------- IMU ----------------

    def imu_cb(self, imu):
        """Convert IMU quaternion to roll/pitch/yaw (deg) and publish."""
        x, y, z, w = imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w

        # Convert quaternion → RPY
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

        # Publish orientation (deg)
        ang = Twist()
        ang.angular.x = math.degrees(roll - self.roll0)
        ang.angular.y = math.degrees(pitch - self.pitch0)
        ang.angular.z = math.degrees(yaw - self.yaw0)
        self.pub_angle_deg.publish(ang)

        # Angular velocities
        vel = Twist()
        vel.angular.x = imu.angular_velocity.x
        vel.angular.y = imu.angular_velocity.y
        vel.angular.z = imu.angular_velocity.z
        self.pub_ang_vel.publish(vel)

    # ---------------- Depth ----------------

    def depth_cb(self, msg):
        if self.init_depth:
            self.depth_ref = msg.data
            self.init_depth = False

    # =======================================================================
    # Camera Mount Control
    # =======================================================================

    def set_camera_tilt(self, angle):
        msg = MountControl()
        msg.pitch = float(angle)
        msg.mode = 2
        self.pub_mount.publish(msg)

    # =======================================================================
    # Timer
    # =======================================================================

    def timer_callback(self):
        """Runs at 20Hz for automatic/correction modes."""
        if self.set_mode[0]:
            return

        if self.set_mode[1]:  # Auto
            self.send_rc_override(1500, 1500, 1500, 1500, 1500, 1500)

        if self.set_mode[2]:  # Correction
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
