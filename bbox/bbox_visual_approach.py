#!/usr/bin/env python3
"""
bbox_visual_approach.py
Visual Servoing Node for BlueROV.

Updates:
- FIXED: Strict State Machine (Vertical -> Horizontal -> Approach).
- FEATURE: 2-Second Delays between stages for stabilization.
- FEATURE: Clear console logging of actions and transitions.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Bool, Float32, String
from rclpy.duration import Duration

class BBoxVisualApproach(Node):
    def __init__(self):
        super().__init__('bbox_visual_approach')

        # --- PARAMETERS ---
        self.declare_parameter('kp_heave', 0.002) 
        self.declare_parameter('kp_yaw', 0.001)   
        self.declare_parameter('kp_surge', 0.005) 

        # Thresholds
        self.declare_parameter('center_deadzone_x', 20) 
        self.declare_parameter('center_deadzone_y', 20) 
        self.declare_parameter('target_width', 200.0)   

        self.kp_heave = self.get_parameter('kp_heave').value
        self.kp_yaw = self.get_parameter('kp_yaw').value
        self.kp_surge = self.get_parameter('kp_surge').value
        self.deadzone_x = self.get_parameter('center_deadzone_x').value
        self.deadzone_y = self.get_parameter('center_deadzone_y').value
        self.target_width = self.get_parameter('target_width').value

        # --- SUBSCRIBERS ---
        self.sub_box_err = self.create_subscription(Point, '/yolo/box_error', self.cb_box_err, 10)
        self.sub_box_dim = self.create_subscription(Point, '/yolo/box_dim', self.cb_box_dim, 10)
        self.sub_box_conf = self.create_subscription(Float32, '/yolo/box_conf', self.cb_box_conf, 10)
        self.sub_btn_a = self.create_subscription(Bool, 'buttons/A', self.cb_btn_a, 10)

        # --- PUBLISHERS ---
        self.pub_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub_status = self.create_publisher(String, '/servoing/status', 10)

        # --- INTERNAL STATE ---
        self.deadman_active = False 
        self.box_visible = False
        self.err_x = 0.0
        self.err_y = 0.0
        self.box_w = 0.0
        
        # State Machine
        # 0: IDLE
        # 1: ALIGN_VERTICAL
        # 2: ALIGN_HORIZONTAL
        # 3: APPROACH
        # 99: WAITING (Stabilization pause)
        self.state = 0 
        self.status_msg = "IDLE"
        
        # Wait Logic
        self.wait_deadline = None
        self.next_state_after_wait = 0

        # Timer (Control Loop - 20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Visual Approach Node Ready. Hold Button A to engage.")

    # --- CALLBACKS ---
    def cb_btn_a(self, msg):
        prev = self.deadman_active
        self.deadman_active = msg.data
        if self.deadman_active and not prev:
            self.get_logger().info(">>> VISUAL SERVOING ENGAGED <<<")
            self.state = 1 # Start with Vertical
        elif not self.deadman_active and prev:
            self.get_logger().info(">>> MANUAL CONTROL <<<")
            self.state = 0

    def cb_box_err(self, msg):
        self.err_x = msg.x
        self.err_y = msg.y

    def cb_box_dim(self, msg):
        self.box_w = msg.x
        self.box_visible = True

    def cb_box_conf(self, msg):
        if msg.data < 0.4:
            self.box_visible = False

    # --- HELPER: START WAIT ---
    def start_wait(self, seconds, next_state_id, msg="Stabilizing"):
        self.state = 99 # Wait state
        self.next_state_after_wait = next_state_id
        self.wait_deadline = self.get_clock().now() + Duration(seconds=seconds)
        self.get_logger().info(f"{msg}... Waiting {seconds}s")
        self.stop_robot()

    # --- CONTROL LOOP ---
    def control_loop(self):
        # 1. Safety Checks
        if not self.deadman_active:
            if self.status_msg != "WAITING FOR BUTTON A":
                self.status_msg = "WAITING FOR BUTTON A"
                self.pub_status_msg()
            return

        if not self.box_visible:
            self.status_msg = "NO TARGET DETECTED"
            self.stop_robot()
            return

        cmd = Twist()

        # 2. STATE MACHINE
        
        # --- STATE 99: WAITING ---
        if self.state == 99:
            time_left = (self.wait_deadline - self.get_clock().now()).nanoseconds / 1e9
            if time_left > 0:
                self.status_msg = f"STABILIZING ({time_left:.1f}s)"
                # Optional: Keep Holding Depth gently while waiting
                cmd.linear.z = -self.kp_heave * self.err_y
                self.pub_vel.publish(cmd)
                self.pub_status_msg()
                return
            else:
                self.state = self.next_state_after_wait
                self.get_logger().info(f"Wait Complete. Starting State {self.state}")

        # --- STATE 1: VERTICAL ALIGNMENT (Heave) ---
        if self.state == 1:
            self.status_msg = f"STAGE 1: VERTICAL (Err Y: {self.err_y:.0f})"
            
            # Check if aligned
            if abs(self.err_y) < self.deadzone_y:
                self.get_logger().info("✅ VERTICAL ALIGNED.")
                self.start_wait(2.0, 2, "Vertical Good")
                return

            # Control
            cmd.linear.z = -self.kp_heave * self.err_y
            cmd.linear.z = max(min(cmd.linear.z, 0.5), -0.5)

        # --- STATE 2: HORIZONTAL ALIGNMENT (Yaw) ---
        elif self.state == 2:
            self.status_msg = f"STAGE 2: HORIZONTAL (Err X: {self.err_x:.0f})"
            
            # Check if aligned
            if abs(self.err_x) < self.deadzone_x:
                self.get_logger().info("✅ HORIZONTAL ALIGNED.")
                self.start_wait(2.0, 3, "Horizontal Good")
                return

            # Control
            cmd.angular.z = -self.kp_yaw * self.err_x
            cmd.angular.z = max(min(cmd.angular.z, 0.5), -0.5)
            
            # Maintain Depth (Background)
            cmd.linear.z = -self.kp_heave * self.err_y

        # --- STATE 3: APPROACH (Surge) ---
        elif self.state == 3:
            width_err = self.target_width - self.box_width_safe()
            self.status_msg = f"STAGE 3: APPROACH (Dist Err: {width_err:.0f})"
            
            # Check if close enough
            if width_err <= 5:
                self.status_msg = "🎯 TARGET REACHED"
                cmd.linear.x = 0.0
            else:
                cmd.linear.x = self.kp_surge * width_err
                cmd.linear.x = max(min(cmd.linear.x, 0.4), -0.2) 

            # Active Correction (Keep it centered while moving)
            cmd.linear.z = -self.kp_heave * self.err_y
            cmd.angular.z = -self.kp_yaw * self.err_x

        # Publish
        self.pub_vel.publish(cmd)
        self.pub_status_msg()

    def box_width_safe(self):
        return self.box_w if self.box_w > 0 else 1.0

    def stop_robot(self):
        self.pub_vel.publish(Twist()) 
        self.pub_status_msg()

    def pub_status_msg(self):
        stat = String()
        stat.data = self.status_msg
        self.pub_status.publish(stat)

def main(args=None):
    rclpy.init(args=args)
    node = BBoxVisualApproach()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()