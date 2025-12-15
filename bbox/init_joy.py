#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import Joy, Imu, BatteryState  ### NEW: Added BatteryState
from mavros_msgs.msg import OverrideRCIn, MountControl
from mavros_msgs.srv import CommandLong, StreamRate, SetMode 
from geometry_msgs.msg import Twist
from time import sleep

class JoyInit(Node):
    def __init__(self):
        super().__init__("joy_init")

        # ... (Your existing Publishers) ...
        self.pub_rc_override = self.create_publisher(OverrideRCIn, "rc/override", 10)
        self.pub_angle_deg = self.create_publisher(Twist, "angle_degree", 10)
        self.pub_ang_vel = self.create_publisher(Twist, "angular_velocity", 10)
        self.pub_mount = self.create_publisher(MountControl, "mount_control/command", 10)
        self.pub_cam_tilt = self.create_publisher(Float64, "camera/tilt_angle", 10)
        self.pub_btn_a = self.create_publisher(Bool, "buttons/A", 10)
        self.pub_btn_b = self.create_publisher(Bool, "buttons/B", 10)
        self.pub_light_pct = self.create_publisher(Float64, "lights/percentage", 10)

        # ### NEW: Battery Percentage Publisher ###
        self.pub_bat_pct = self.create_publisher(Float64, "battery/smart_percentage", 10)
        
        # ... (Services & Params) ...
        self.cli_command = self.create_client(CommandLong, "cmd/command")
        self.cli_set_mode = self.create_client(SetMode, "set_mode")
        self.cli_stream_rate = self.create_client(StreamRate, "set_stream_rate")
        self.set_stream_rate(25)

        # Internal State
        self.arming = False
        self.set_mode = [True, False, False] 
        self.init_angles = True
        self.init_depth = True
        self.depth_ref = 0.0
        self.current_rc = [1500, 1500, 1500, 1500, 1500, 1500]
        
        # Joystick States
        self.dpad_horizontal = 0.0
        self.dpad_vertical = 0.0
        self.btn_a_state = False
        self.btn_b_state = False
        self.btn_gripper_toggle_prev = 0

        # Hardware Config
        self.light_pin = 12.0
        self.light_min = 1100.0
        self.light_max = 1900.0
        self.light_pwm = 1100.0

        self.camera_pin = 16.0
        self.tilt = 0.0
        self.tilt_min = -60.0
        self.tilt_max = 60.0

        self.gripper_mode = False 
        self.gripper_pin = 10.0    
        self.gripper_open = 1900.0
        self.gripper_close = 1100.0
        self.gripper_val = 1500.0  
        self.gripper_last_sent = 0.0 

        # ### NEW: BATTERY FILTER VARIABLES ###
        self.bat_voltage_filtered = 16.0  # Start high to avoid 0% alarm at boot
        self.bat_alpha = 0.05             # Filter strength (Lower = Smoother/Slower)
        self.bat_min_v = 13.5             # 0% Level (4S Battery)
        self.bat_max_v = 16.8             # 100% Level (4S Battery)

        self.create_subscribers()
        self.timer = self.create_timer(0.05, self.timer_callback)

    # ... (Keep existing methods: run_init_test, send_servo_cmd, etc.) ...
    def run_init_test(self): pass
    def send_servo_cmd(self, pin, value):
        if self.cli_command.service_is_ready():
            req = CommandLong.Request()
            req.command = 183
            req.param1 = float(pin)
            req.param2 = float(value)
            self.cli_command.call_async(req)
    def set_stream_rate(self, rate):
        if self.cli_stream_rate.wait_for_service(timeout_sec=1.0):
            req = StreamRate.Request()
            req.stream_id = 0
            req.message_rate = rate
            req.on_off = True
            self.cli_stream_rate.call_async(req)
    def set_ardusub_mode(self, mode_name):
        if self.cli_set_mode.service_is_ready():
            req = SetMode.Request()
            req.custom_mode = mode_name
            self.cli_set_mode.call_async(req)
            return True
        return False
    def arm_disarm(self, arm):
        if self.cli_command.service_is_ready():
            req = CommandLong.Request()
            req.command = 400
            req.param1 = 1.0 if arm else 0.0
            self.cli_command.call_async(req)

    def create_subscribers(self):
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(Joy, "joy", self.joy_cb, qos)
        self.create_subscription(Twist, "cmd_vel", self.cmd_vel_cb, qos)
        self.create_subscription(Imu, "imu/data", self.imu_cb, qos)
        self.create_subscription(Float64, "global_position/rel_alt", self.depth_cb, qos)
        
        # ### NEW: BATTERY SUBSCRIBER ###
        # Note: Mavros usually publishes battery status to mavros/battery
        self.create_subscription(BatteryState, "battery", self.battery_cb, qos)

    # ### NEW: BATTERY CALLBACK ###
    def battery_cb(self, msg):
        raw_voltage = msg.voltage
        
        # 1. Filter the voltage (Exponential Moving Average)
        # This ignores sudden drops when motors spin up
        self.bat_voltage_filtered = (self.bat_alpha * raw_voltage) + ((1 - self.bat_alpha) * self.bat_voltage_filtered)

        # 2. Calculate Percentage
        # Formula: (Current - Min) / (Max - Min) * 100
        pct = (self.bat_voltage_filtered - self.bat_min_v) / (self.bat_max_v - self.bat_min_v) * 100.0
        
        # 3. Clamp (Keep between 0 and 100)
        pct = float(np.clip(pct, 0.0, 100.0))

        # 4. Publish
        out_msg = Float64()
        out_msg.data = pct
        self.pub_bat_pct.publish(out_msg)

    # ... (Keep existing callbacks: joy_cb, cmd_vel_cb, etc.) ...
    def joy_cb(self, joy):
        # (Paste your existing joy_cb here exactly as before)
        btn_arm = joy.buttons[7]
        btn_disarm = joy.buttons[6]
        btn_manual = joy.buttons[3]
        btn_auto = joy.buttons[2]
        btn_tilt_up = joy.buttons[4]
        btn_tilt_down = joy.buttons[5]
        btn_tilt_reset = joy.buttons[9]
        btn_gripper_toggle = joy.buttons[8] 
        lt = joy.axes[2]
        rt = joy.axes[5]
        self.dpad_horizontal = joy.axes[6]
        self.dpad_vertical = joy.axes[7]
        self.btn_a_state = bool(joy.buttons[0])
        self.btn_b_state = bool(joy.buttons[1])
        if btn_arm and not self.arming:
            self.arming = True
            self.arm_disarm(True)
        if btn_disarm and self.arming:
            self.arming = False
            self.arm_disarm(False)
        if btn_gripper_toggle and not self.btn_gripper_toggle_prev:
            self.gripper_mode = not self.gripper_mode
            self.get_logger().info(f"Switched to: {'GRIPPER' if self.gripper_mode else 'MOTION'}")
        self.btn_gripper_toggle_prev = btn_gripper_toggle
        if btn_manual:
            if self.set_ardusub_mode("MANUAL"):
                self.set_mode = [True, False, False]
        if btn_auto:
            if self.set_ardusub_mode("ALT_HOLD"):
                self.set_mode = [False, True, False]
        if rt == -1 and self.light_pwm < self.light_max:
            self.light_pwm = min(self.light_pwm + 100, self.light_max)
            self.send_servo_cmd(self.light_pin, self.light_pwm)
        if lt == -1 and self.light_pwm > self.light_min:
            self.light_pwm = max(self.light_pwm - 100, self.light_min)
            self.send_servo_cmd(self.light_pin, self.light_pwm)
        if btn_tilt_up:
            self.tilt = min(self.tilt + 5.0, self.tilt_max)
            self.set_camera_tilt(self.tilt)
        elif btn_tilt_down:
            self.tilt = max(self.tilt - 5.0, self.tilt_min)
            self.set_camera_tilt(self.tilt)
        elif btn_tilt_reset:
            self.tilt = 0.0
            self.set_camera_tilt(self.tilt)

    def cmd_vel_cb(self, msg):
        def map_pwm(v):
            if abs(v) < 0.05: v = 0.0
            return int(np.clip(v * 400 + 1500, 1100, 1900))
        self.current_rc[0] = map_pwm(msg.angular.y)
        self.current_rc[1] = map_pwm(msg.angular.x)
        self.current_rc[2] = map_pwm(msg.linear.z)
        self.current_rc[3] = map_pwm(-msg.angular.z)
        self.current_rc[4] = map_pwm(msg.linear.x)
        self.current_rc[5] = map_pwm(-msg.linear.y)

    def send_rc_override(self, pitch, roll, throttle, yaw, forward, lateral):
        msg = OverrideRCIn()
        msg.channels = [65535] * 18
        msg.channels[0:6] = [pitch, roll, throttle, yaw, forward, lateral]
        self.pub_rc_override.publish(msg)

    def imu_cb(self, imu):
        # (Keep existing IMU logic)
        x, y, z, w = imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w
        sinr, cosr = 2 * (w * x + y * z), 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)
        siny, cosy = 2 * (w * z + x * y), 1 - 2 * (y * y + z * z)
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

    def depth_cb(self, msg):
        if self.init_depth:
            self.depth_ref = msg.data
            self.init_depth = False

    def publish_camera_tilt_topic(self):
        msg = Float64()
        msg.data = float(self.tilt)
        self.pub_cam_tilt.publish(msg)

    def set_camera_tilt(self, angle):
        msg = MountControl()
        msg.pitch = float(angle)
        msg.mode = 2
        self.pub_mount.publish(msg)
        self.publish_camera_tilt_topic()

    def timer_callback(self):
        self.publish_camera_tilt_topic()
        
        # Lights Percentage Logic
        percentage = (self.light_pwm - self.light_min) / (self.light_max - self.light_min) * 100.0
        light_msg = Float64()
        light_msg.data = percentage
        self.pub_light_pct.publish(light_msg)

        msg_a = Bool()
        msg_a.data = self.btn_a_state
        self.pub_btn_a.publish(msg_a)
        msg_b = Bool()
        msg_b.data = self.btn_b_state
        self.pub_btn_b.publish(msg_b)

        if self.gripper_mode:
            if self.dpad_vertical > 0.5: self.gripper_val = self.gripper_open
            elif self.dpad_vertical < -0.5: self.gripper_val = self.gripper_close
            elif self.dpad_horizontal > 0.5: self.gripper_val -= 20.0 
            elif self.dpad_horizontal < -0.5: self.gripper_val += 20.0 
            self.gripper_val = float(np.clip(self.gripper_val, self.gripper_close, self.gripper_open))
            if self.gripper_val != self.gripper_last_sent:
                self.send_servo_cmd(self.gripper_pin, self.gripper_val)
                self.gripper_last_sent = self.gripper_val

        if self.set_mode[0] or self.set_mode[1]:
            rc_to_send = self.current_rc.copy()
            if self.gripper_mode:
                rc_to_send[0] = 1500 
                rc_to_send[1] = 1500 
            self.send_rc_override(*rc_to_send)

def main(args=None):
    rclpy.init(args=args)
    node = JoyInit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()