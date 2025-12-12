#!/usr/bin/env python3
"""
bbox_visual_approach.py
Visual Servoing Node for BlueROV.

Updates:
- LOGIC: Stage 6 now aligns 3 points: Box Center, Handle Center, and Target.
- FEATURE: Status messages are now descriptive "Phase X..." explanations 
  published to /servoing/status for UI/User feedback.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Bool, Float32, String, Float64
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration

class BBoxVisualApproach(Node):
    def __init__(self):
        super().__init__('bbox_visual_approach')

        # --- PARAMETERS ---
        self.declare_parameter('kp_heave', 0.5) 
        self.declare_parameter('ki_heave', 0.05)   
        self.declare_parameter('heave_i_max', 0.15) 
        
        self.declare_parameter('kp_yaw', 0.00045)    
        self.declare_parameter('kp_yaw_d', 0.00015)   
        self.declare_parameter('kp_surge', 0.01) 
        
        self.declare_parameter('kp_sway', 0.0025)
        self.declare_parameter('ki_sway', 0.0005)   
        self.declare_parameter('sway_i_max', 0.1)   

        # Final Stage Gains (Softer for Stage 6 & 7)
        self.declare_parameter('kp_yaw_final', 0.0002)   # Very soft yaw
        self.declare_parameter('kp_sway_final', 0.002)   # Soft sway
        self.declare_parameter('kp_surge_final', 0.005)  # Slow creep surge

        # Thresholds
        self.declare_parameter('center_deadzone_x', 40) 
        self.declare_parameter('altitude_tolerance', 0.08) 
        self.declare_parameter('surge_tolerance', 15.0) 
        self.declare_parameter('sway_tolerance', 10.0) 
        
        # New Phase 5/6 Params
        self.declare_parameter('sway_tolerance_final', 5.0) 
        self.declare_parameter('altitude_tolerance_final', 0.04) 
        self.declare_parameter('target_altitude_final', -4.79) 
        self.declare_parameter('final_surge_speed', 0.4) 
        
        # TARGETS
        self.declare_parameter('target_altitude', -4.70) 
        self.declare_parameter('target_height_px', 250.0) 
        self.declare_parameter('target_height_final_px', 450.0)   

        # DEBUG FLAG
        self.declare_parameter('debug_prints', True)

        # Load Params
        self.kp_heave = self.get_parameter('kp_heave').value
        self.ki_heave = self.get_parameter('ki_heave').value
        self.heave_i_max = self.get_parameter('heave_i_max').value
        
        self.kp_yaw = self.get_parameter('kp_yaw').value
        self.kp_yaw_d = self.get_parameter('kp_yaw_d').value
        self.kp_surge = self.get_parameter('kp_surge').value
        
        self.kp_sway = self.get_parameter('kp_sway').value
        self.ki_sway = self.get_parameter('ki_sway').value
        self.sway_i_max = self.get_parameter('sway_i_max').value
        
        self.kp_yaw_final = self.get_parameter('kp_yaw_final').value
        self.kp_sway_final = self.get_parameter('kp_sway_final').value
        self.kp_surge_final = self.get_parameter('kp_surge_final').value

        self.deadzone_x = self.get_parameter('center_deadzone_x').value
        self.alt_tol = self.get_parameter('altitude_tolerance').value
        self.surge_tol = self.get_parameter('surge_tolerance').value
        self.sway_tol = self.get_parameter('sway_tolerance').value
        
        # Final tolerances
        self.sway_tol_final = self.get_parameter('sway_tolerance_final').value
        self.alt_tol_final = self.get_parameter('altitude_tolerance_final').value
        
        self.target_alt = self.get_parameter('target_altitude').value
        self.target_alt_final = self.get_parameter('target_altitude_final').value
        self.target_height = self.get_parameter('target_height_px').value
        self.target_height_final = self.get_parameter('target_height_final_px').value
        self.final_surge_spd = self.get_parameter('final_surge_speed').value
        
        self.debug = self.get_parameter('debug_prints').value

        # --- QOS PROFILES ---
        qos_sensor = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        qos_standard = 10

        # --- SUBSCRIBERS ---
        self.sub_box_err = self.create_subscription(Point, '/yolo/box_error', self.cb_box_err, qos_standard)
        self.sub_box_dim = self.create_subscription(Point, '/yolo/box_dim', self.cb_box_dim, qos_standard)
        self.sub_box_conf = self.create_subscription(Float32, '/yolo/box_conf', self.cb_box_conf, qos_standard)
        
        self.sub_hdl_err = self.create_subscription(Point, '/yolo/handle_error', self.cb_hdl_err, qos_standard)
        self.sub_hdl_conf = self.create_subscription(Float32, '/yolo/handle_conf', self.cb_hdl_conf, qos_standard)

        self.sub_depth = self.create_subscription(Float64, '/bluerov2/global_position/rel_alt', self.cb_depth, qos_sensor)
        self.sub_btn_a = self.create_subscription(Bool, 'buttons/A', self.cb_btn_a, qos_standard)

        # --- PUBLISHERS ---
        self.pub_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub_status = self.create_publisher(String, '/servoing/status', 10)

        # --- INTERNAL STATE ---
        self.deadman_active = False 
        self.box_visible = False
        self.handle_visible = False
        
        self.err_box_x = 0.0
        self.err_box_y = 0.0 
        self.box_h = 0.0
        self.err_hdl_x = 0.0
        
        self.current_alt = 0.0
        self.prev_yaw_err = 0.0
        self.heave_integral = 0.0 
        self.sway_integral = 0.0 
        
        # State Machine
        self.state = 0 
        self.status_msg = "IDLE"
        self.surge_deadline = None

        # Timer
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info(f"Visual Approach Ready. Final Target: {self.target_alt_final}m")

    # --- CALLBACKS ---
    def cb_btn_a(self, msg):
        prev = self.deadman_active
        self.deadman_active = msg.data
        if self.deadman_active and not prev:
            self.get_logger().info(">>> SERVOING ENGAGED <<<")
            self.state = 1 
            self.prev_yaw_err = 0.0
            self.heave_integral = 0.0 
            self.sway_integral = 0.0 
        elif not self.deadman_active and prev:
            self.get_logger().info(">>> MANUAL CONTROL <<<")
            self.state = 0
            self.stop_robot()

    def cb_depth(self, msg):
        self.current_alt = msg.data

    def cb_box_err(self, msg):
        self.err_box_x = msg.x
        self.err_box_y = msg.y

    def cb_box_dim(self, msg):
        self.box_h = msg.y 
        self.box_visible = True

    def cb_box_conf(self, msg):
        if msg.data < 0.4:
            self.box_visible = False

    def cb_hdl_err(self, msg):
        self.err_hdl_x = msg.x

    def cb_hdl_conf(self, msg):
        self.handle_visible = (msg.data > 0.4)

    # --- HELPER: CALCULATE HEAVE (PI) ---
    def get_heave_cmd(self, target):
        alt_err = target - self.current_alt
        
        p_term = self.kp_heave * alt_err
        self.heave_integral += alt_err * self.dt
        
        # Anti-Windup
        i_term_val = self.ki_heave * self.heave_integral
        if i_term_val > self.heave_i_max:
            i_term_val = self.heave_i_max
            self.heave_integral = self.heave_i_max / self.ki_heave
        elif i_term_val < -self.heave_i_max:
            i_term_val = -self.heave_i_max
            self.heave_integral = -self.heave_i_max / self.ki_heave
            
        z_cmd = p_term + i_term_val
        return max(min(z_cmd, 0.5), -0.5)

    # --- HELPER: CALCULATE SWAY (PI) ---
    def get_sway_cmd(self, error_val, kp=None):
        if kp is None: kp = self.kp_sway
        
        # P-Term
        p_term = kp * error_val
        
        # I-Term
        self.sway_integral += error_val * self.dt
        
        # Anti-Windup
        i_term_val = self.ki_sway * self.sway_integral
        if i_term_val > self.sway_i_max:
            i_term_val = self.sway_i_max
            self.sway_integral = self.sway_i_max / self.ki_sway
        elif i_term_val < -self.sway_i_max:
            i_term_val = -self.sway_i_max
            self.sway_integral = -self.sway_i_max / self.ki_sway
            
        y_cmd = p_term + i_term_val
        return max(min(y_cmd, 0.3), -0.3) 

    # --- HELPER: CALCULATE YAW (PD) ---
    def get_yaw_cmd(self, error_val, kp=None, kd=None):
        if kp is None: kp = self.kp_yaw
        if kd is None: kd = self.kp_yaw_d

        P = kp * error_val
        d_err = (error_val - self.prev_yaw_err) / self.dt
        D = kd * d_err
        self.prev_yaw_err = error_val
        yaw_cmd = -(P + D)
        return max(min(yaw_cmd, 0.5), -0.5)

    # --- CONTROL LOOP ---
    def control_loop(self):
        if not self.deadman_active:
            if self.status_msg != "WAITING FOR BUTTON A":
                self.status_msg = "WAITING FOR BUTTON A"
                self.pub_status_msg()
            return

        cmd = Twist()

        # Target Check 
        if self.state < 7 and not self.box_visible:
            self.state = 0
            self.status_msg = "NO BOX DETECTED"
            self.stop_robot()
            return
        
        if self.state == 0 and self.box_visible:
            self.state = 1
            self.get_logger().info("Target found. Restarting Approach.")
            self.prev_yaw_err = self.err_box_x
            self.heave_integral = 0.0 
            self.sway_integral = 0.0

        # --- STATE 1: INITIAL DEPTH ---
        if self.state == 1:
            err = self.target_alt - self.current_alt
            self.status_msg = f"Phase 1: Going to approximate depth {self.target_alt}m (Curr: {self.current_alt:.2f})"
            if abs(err) < self.alt_tol:
                self.get_logger().info("✅ DEPTH REACHED.")
                self.state = 2 
                return
            cmd.linear.z = self.get_heave_cmd(self.target_alt)

        # --- STATE 2: YAW ALIGNMENT ---
        elif self.state == 2:
            self.status_msg = f"Phase 2: Aligning with BBox horizontally (Err: {self.err_box_x:.0f})"
            yaw_out = self.get_yaw_cmd(self.err_box_x)
            if abs(self.err_box_x) < self.deadzone_x:
                self.get_logger().info("✅ HORIZONTAL ALIGNED.")
                self.state = 3 
                return
            cmd.angular.z = yaw_out
            cmd.linear.z = self.get_heave_cmd(self.target_alt)

        # --- STATE 3: APPROACH ---
        elif self.state == 3:
            current_h = self.box_h if self.box_h > 0 else 1.0
            height_err = self.target_height - current_h
            self.status_msg = f"Phase 3: Approaching BBox to detect handle (Size: {current_h:.0f}/{self.target_height:.0f})"
            
            if height_err <= self.surge_tol:
                self.get_logger().info("✅ TARGET REACHED.")
                self.state = 4 
                # Reset Yaw D-term memory before switching target to Handle
                self.prev_yaw_err = self.err_hdl_x 
                return
            
            cmd.linear.x = self.kp_surge * height_err
            cmd.linear.x = max(min(cmd.linear.x, 0.4), -0.2) 

            if abs(self.err_box_x) < 10:
                cmd.angular.z = 0.0
            else:
                cmd.angular.z = self.get_yaw_cmd(self.err_box_x)
            cmd.linear.z = self.get_heave_cmd(self.target_alt)

        # --- STATE 4: COARSE HANDLE ALIGNMENT ---
        elif self.state == 4:
            if not self.handle_visible:
                self.status_msg = "Phase 4: Waiting for Handle detection..."
                cmd.linear.z = self.get_heave_cmd(self.target_alt)
                cmd.angular.z = self.get_yaw_cmd(self.err_box_x) 
            else:
                alignment_err = self.err_box_x - self.err_hdl_x
                self.status_msg = f"Phase 4: Coarse alignment BBox-Handle (Diff: {alignment_err:.0f})"
                
                if abs(alignment_err) < self.sway_tol:
                    self.get_logger().info("✅ HANDLE COARSE ALIGNED.")
                    self.state = 5 
                    return
                else:
                    cmd.linear.y = self.get_sway_cmd(alignment_err)

                cmd.angular.z = self.get_yaw_cmd(self.err_hdl_x)
                cmd.linear.z = self.get_heave_cmd(self.target_alt)

        # --- STATE 5: DESCEND & FINE ALIGN (BOX vs HANDLE) ---
        elif self.state == 5:
            err_depth = self.target_alt_final - self.current_alt
            cmd.linear.z = self.get_heave_cmd(self.target_alt_final)
            
            if not self.handle_visible:
                self.status_msg = "Phase 5: Handle Lost! Holding..."
            else:
                alignment_err = self.err_box_x - self.err_hdl_x
                self.status_msg = f"Phase 5: Descending to {self.target_alt_final}m & Aligning (D_Err: {err_depth:.2f} | A_Err: {alignment_err:.0f})"
                
                cmd.linear.y = self.get_sway_cmd(alignment_err)
                cmd.angular.z = self.get_yaw_cmd(self.err_hdl_x)

                depth_good = abs(err_depth) < self.alt_tol_final
                sway_good = abs(alignment_err) < self.sway_tol_final
                
                if self.debug:
                    self.get_logger().info(
                        f"[DEBUG S5] Depth: {depth_good} (Err: {err_depth:.3f}) | "
                        f"Sway: {sway_good} (Err: {alignment_err:.1f})"
                    )

                if depth_good and sway_good:
                    self.get_logger().info("✅ ALIGNED AT DEPTH. FINAL LOCK...")
                    self.state = 6
                    self.sway_integral = 0.0 # Reset I-term for final stage
                    return

        # --- STATE 6: FINAL STATION KEEPING (HANDLE vs TARGET) ---
        elif self.state == 6:
            err_depth = self.target_alt_final - self.current_alt
            cmd.linear.z = self.get_heave_cmd(self.target_alt_final)

            if not self.handle_visible or not self.box_visible:
                self.status_msg = "Phase 6: Visuals Unstable..."
            else:
                # 1. Perspective Alignment (Sway): Align Box to Handle
                persp_err = self.err_box_x - self.err_hdl_x
                cmd.linear.y = self.get_sway_cmd(persp_err, kp=self.kp_sway_final)

                # 2. Centering Alignment (Yaw): Align Handle to Target
                center_err = self.err_hdl_x
                cmd.angular.z = self.get_yaw_cmd(center_err, kp=self.kp_yaw_final)

                # 3. Surge (Maintain NEW LARGER Box Size)
                current_h = self.box_h
                height_err = self.target_height_final - current_h
                cmd.linear.x = self.kp_surge * height_err
                cmd.linear.x = max(min(cmd.linear.x, 0.2), -0.2) 

                # Exit Logic
                depth_good = abs(err_depth) < self.alt_tol_final
                persp_good = abs(persp_err) < self.sway_tol_final
                center_good = abs(center_err) < self.deadzone_x 
                surge_good = abs(height_err) < self.surge_tol 

                self.status_msg = f"Phase 6: Final 3-Point Station Keeping (P:{persp_good} C:{center_good} Sz:{surge_good})"
                
                if self.debug:
                    self.get_logger().info(f"[DEBUG S6] PerspErr:{persp_err:.1f} CenterErr:{center_err:.1f}")

                if depth_good and persp_good and center_good and surge_good:
                    self.get_logger().info("GOING >:)")
                    self.state = 7
                    self.surge_deadline = self.get_clock().now() + Duration(seconds=5.0)
                    return

        # --- STATE 7: FINAL SURGE ---
        elif self.state == 7:
            time_left = (self.surge_deadline - self.get_clock().now()).nanoseconds / 1e9
            
            if time_left > 0:
                self.status_msg = f"Phase 7: GOING >:) (Blind Surge: {time_left:.1f}s)"
                cmd.linear.x = self.final_surge_spd 
                
                # Active Sway Correction (User Request)
                if self.handle_visible:
                    # Keep Handle Centered while surging
                    alignment_err = self.err_hdl_x
                    cmd.linear.y = self.get_sway_cmd(-alignment_err, kp=self.kp_sway_final)
                else:
                    cmd.linear.y = 0.0
                    
                cmd.linear.z = 0.0
                cmd.angular.z = 0.0
            else:
                self.state = 8
                self.get_logger().info("MISSION COMPLETE")

        # --- STATE 8: DONE ---
        elif self.state == 8:
            self.status_msg = "MISSION COMPLETE"
            cmd = Twist() 

        # Publish
        self.pub_vel.publish(cmd)
        self.pub_status_msg()

    def stop_robot(self):
        self.heave_integral = 0.0 
        self.sway_integral = 0.0
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