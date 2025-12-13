#!/usr/bin/env python3
"""
navigation_node_real_v6.py - PID-CONTROLLED Straight-Line Approach
===================================================================

KEY IMPROVEMENTS:
-----------------
1. PID CONTROLLERS for yaw and heave (not just P)
   - Derivative term dampens oscillations
   - Integral term eliminates steady-state error
2. FASTER APPROACH - increased surge speeds
3. BETTER HANDLE TRACKING - corrections during approach
4. HANDLE FILLS SCREEN - clear stopping condition

STRATEGY:
---------
Phase 1: ALIGN - PID control to center handle (no forward movement)
Phase 2: APPROACH - Steady forward, PID corrections, straight line
Phase 3: FINAL - Handle fills screen → stop for gripper attachment
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import math
import time
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass
from collections import deque

from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Joy
from vision_msgs.msg import Detection2DArray


@dataclass
class Pose3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0


@dataclass 
class Detection:
    class_name: str
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float = 1.0
    timestamp: float = 0.0


class MissionMode(Enum):
    MANUAL = 0
    SEARCH = 1
    ALIGN_TO_HANDLE = 2
    APPROACH_STRAIGHT = 3
    FINAL_APPROACH = 4
    HOLD_POSITION = 5
    BACKUP = 6


class PIDController:
    """
    PID Controller with anti-windup and derivative filtering.
    
    The derivative term is crucial for damping oscillations!
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limit: float = 1.0,
                 integral_limit: float = 100.0,
                 derivative_filter: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.derivative_filter = derivative_filter  # Low-pass filter for derivative
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.last_time = None
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.last_time = None
    
    def compute(self, error: float, current_time: float = None) -> float:
        """
        Compute PID output.
        
        Args:
            error: Current error (setpoint - measured)
            current_time: Current timestamp (uses time.time() if None)
        
        Returns:
            Control output (limited to output_limit)
        """
        if current_time is None:
            current_time = time.time()
        
        # Calculate dt
        if self.last_time is None:
            dt = 0.05  # Default 20Hz
        else:
            dt = current_time - self.last_time
            dt = max(dt, 0.001)  # Prevent division by zero
        
        self.last_time = current_time
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term with low-pass filter (reduces noise)
        raw_derivative = (error - self.prev_error) / dt
        # Filter: new = alpha * raw + (1-alpha) * old
        derivative = (self.derivative_filter * raw_derivative + 
                     (1 - self.derivative_filter) * self.prev_derivative)
        self.prev_derivative = derivative
        d_term = self.kd * derivative
        
        self.prev_error = error
        
        # Total output
        output = p_term + i_term + d_term
        
        # Limit output
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        return output
    
    def set_gains(self, kp: float, ki: float, kd: float):
        """Update PID gains"""
        self.kp = kp
        self.ki = ki
        self.kd = kd


class NavigationNodeReal(Node):
    """
    PID-Controlled Straight-Line Approach Navigation
    """
    
    def __init__(self):
        super().__init__('navigation_node_real')
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Navigation Node v6 - PID CONTROLLED")
        self.get_logger().info("  Strategy: PID align → Straight approach → Stop at target")
        self.get_logger().info("=" * 60)
        
        self._declare_parameters()
        self._init_controllers()
        self._init_state()
        self._init_ros()
        self._print_startup_info()
    
    def _declare_parameters(self):
        self.declare_parameter('control_rate', 20.0)
        
        # === SPEED LIMITS ===
        self.declare_parameter('max_surge', 0.18)
        self.declare_parameter('max_heave', 0.25)
        self.declare_parameter('max_yaw_rate', 0.20)
        self.declare_parameter('backup_speed', 0.10)
        self.declare_parameter('final_approach_speed', 0.08)
        
        # === PID GAINS FOR YAW ===
        # Kp: Proportional - how hard to correct
        # Ki: Integral - eliminates steady-state error  
        # Kd: Derivative - dampens oscillations (CRITICAL!)
        self.declare_parameter('yaw_kp', 0.0015)
        self.declare_parameter('yaw_ki', 0.0001)
        self.declare_parameter('yaw_kd', 0.002)  # Important for damping!
        
        # === PID GAINS FOR HEAVE ===
        self.declare_parameter('heave_kp', 0.003)
        self.declare_parameter('heave_ki', 0.0002)
        self.declare_parameter('heave_kd', 0.004)  # Important for damping!
        
        # === APPROACH PHASE GAINS (lower for stability) ===
        self.declare_parameter('approach_yaw_kp', 0.0008)
        self.declare_parameter('approach_yaw_ki', 0.00005)
        self.declare_parameter('approach_yaw_kd', 0.001)
        
        self.declare_parameter('approach_heave_kp', 0.002)
        self.declare_parameter('approach_heave_ki', 0.0001)
        self.declare_parameter('approach_heave_kd', 0.002)
        
        # Sign inversion
        self.declare_parameter('yaw_sign', -1.0)
        self.declare_parameter('heave_sign', -1.0)
        
        # === BUOYANCY ===
        self.declare_parameter('buoyancy_compensation', 0.0)
        
        # === DETECTION ===
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('detection_timeout', 2.0)
        
        # === IMAGE PARAMETERS ===
        self.declare_parameter('image_center_x', 960.0)
        self.declare_parameter('image_center_y', 540.0)
        self.declare_parameter('image_width', 1920.0)
        self.declare_parameter('image_height', 1080.0)
        
        # === CENTERING TOLERANCES ===
        self.declare_parameter('align_tolerance_x', 60.0)
        self.declare_parameter('align_tolerance_y', 80.0)
        self.declare_parameter('approach_tolerance_x', 100.0)
        self.declare_parameter('approach_tolerance_y', 120.0)
        
        # === STABILITY ===
        self.declare_parameter('stability_count_required', 15)  # More frames for PID to settle
        
        # === HANDLE SIZE THRESHOLDS ===
        self.declare_parameter('handle_area_start', 3000.0)      # Min area to start approach
        self.declare_parameter('handle_area_close', 30000.0)     # Getting close
        self.declare_parameter('handle_area_final', 60000.0)     # Final approach
        self.declare_parameter('handle_area_stop', 100000.0)     # STOP - ready for gripper
        
        # Screen fill percentages
        self.declare_parameter('handle_fill_final', 0.04)   # 4% = start final approach
        self.declare_parameter('handle_fill_stop', 0.08)    # 8% = stop completely
        
        # === BOX THRESHOLDS ===
        self.declare_parameter('box_too_close_area', 500000.0)
        
        # === TIMEOUTS ===
        self.declare_parameter('max_approach_time', 120.0)
        self.declare_parameter('safety_timeout', 5.0)
        self.declare_parameter('align_timeout', 45.0)
    
    def _init_controllers(self):
        """Initialize PID controllers"""
        # Alignment phase controllers (higher gains for faster response)
        self._yaw_pid_align = PIDController(
            kp=self.get_parameter('yaw_kp').value,
            ki=self.get_parameter('yaw_ki').value,
            kd=self.get_parameter('yaw_kd').value,
            output_limit=self.get_parameter('max_yaw_rate').value,
            integral_limit=200.0,
            derivative_filter=0.2
        )
        
        self._heave_pid_align = PIDController(
            kp=self.get_parameter('heave_kp').value,
            ki=self.get_parameter('heave_ki').value,
            kd=self.get_parameter('heave_kd').value,
            output_limit=self.get_parameter('max_heave').value,
            integral_limit=150.0,
            derivative_filter=0.2
        )
        
        # Approach phase controllers (lower gains for stability)
        self._yaw_pid_approach = PIDController(
            kp=self.get_parameter('approach_yaw_kp').value,
            ki=self.get_parameter('approach_yaw_ki').value,
            kd=self.get_parameter('approach_yaw_kd').value,
            output_limit=self.get_parameter('max_yaw_rate').value * 0.6,
            integral_limit=100.0,
            derivative_filter=0.3
        )
        
        self._heave_pid_approach = PIDController(
            kp=self.get_parameter('approach_heave_kp').value,
            ki=self.get_parameter('approach_heave_ki').value,
            kd=self.get_parameter('approach_heave_kd').value,
            output_limit=self.get_parameter('max_heave').value * 0.8,
            integral_limit=100.0,
            derivative_filter=0.3
        )
    
    def _init_state(self):
        self._mode = MissionMode.MANUAL
        self._current_pose = Pose3D()
        self._hold_pose: Optional[Pose3D] = None
        
        # Detections
        self._box_detection: Optional[Detection] = None
        self._handle_detection: Optional[Detection] = None
        self._last_box_time: float = 0.0
        self._last_handle_time: float = 0.0
        self._latest_detections: List[Detection] = []
        
        # Timing
        self._last_control_time = time.time()
        self._button_debounce = {}
        self._control_count = 0
        self._approach_start_time = None
        self._align_start_time = None
        
        # Safety
        self._last_cmd_time = time.time()
        self._emergency_stop = False
        
        # Stability
        self._centered_count = 0
        self._aligned = False
        
        # Backup state
        self._backup_start_time = None
        
        # Last known handle error
        self._last_handle_error_x = 0.0
        self._last_handle_error_y = 0.0
        
        # For logging PID internals
        self._last_yaw_p = 0.0
        self._last_yaw_d = 0.0
    
    def _init_ros(self):
        self.pub_cmd_vel = self.create_publisher(Twist, '/bluerov2/cmd_vel', 10)
        self.pub_status = self.create_publisher(String, '/navigation/status', 10)
        self.pub_mode = self.create_publisher(String, '/navigation/mode', 10)
        
        sensor_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        # YOLO topics
        self.create_subscription(Point, '/yolo/box_error', self._box_error_cb, 10)
        self.create_subscription(Point, '/yolo/box_dim', self._box_dim_cb, 10)
        self.create_subscription(Float32, '/yolo/box_conf', self._box_conf_cb, 10)
        
        self.create_subscription(Point, '/yolo/handle_error', self._handle_error_cb, 10)
        self.create_subscription(Point, '/yolo/handle_dim', self._handle_dim_cb, 10)
        self.create_subscription(Float32, '/yolo/handle_conf', self._handle_conf_cb, 10)
        
        self.create_subscription(Detection2DArray, '/yolo/detections', self._yolo_cb, 10)
        
        # Joystick
        self.create_subscription(Joy, '/bluerov2/joy', self._joy_cb, sensor_qos)
        self.create_subscription(Joy, '/joy', self._joy_cb, 10)
        self.create_subscription(Bool, '/bluerov2/buttons/A', self._btn_a_cb, 10)
        
        # Timers
        rate = self.get_parameter('control_rate').value
        self.create_timer(1.0 / rate, self._control_loop)
        self.create_timer(1.0, self._status_loop)
        self.create_timer(0.5, self._safety_check)
        
        self.get_logger().info(f"Control loop at {rate} Hz")
    
    def _print_startup_info(self):
        self.get_logger().info("")
        self.get_logger().info("v6 FEATURES:")
        self.get_logger().info("  - PID controllers (not just P!)")
        self.get_logger().info("  - Derivative term dampens oscillations")
        self.get_logger().info("  - Integral term eliminates steady-state error")
        self.get_logger().info("  - Separate gains for ALIGN vs APPROACH phases")
        self.get_logger().info("")
        self.get_logger().info("PID GAINS:")
        self.get_logger().info(f"  Yaw (align):  Kp={self._yaw_pid_align.kp}, Ki={self._yaw_pid_align.ki}, Kd={self._yaw_pid_align.kd}")
        self.get_logger().info(f"  Heave (align): Kp={self._heave_pid_align.kp}, Ki={self._heave_pid_align.ki}, Kd={self._heave_pid_align.kd}")
        self.get_logger().info("")
        self.get_logger().info("CONTROLS: Y=Manual, X=Auto, A=Hold, B=Manual")
        self.get_logger().info("=" * 60)
    
    # =========================================================================
    # DETECTION CALLBACKS
    # =========================================================================
    
    def _box_error_cb(self, msg: Point):
        now = time.time()
        if self._box_detection is None:
            self._box_detection = Detection("box", 0, 0, 0, 0)
        center_x = self.get_parameter('image_center_x').value
        center_y = self.get_parameter('image_center_y').value
        self._box_detection.center_x = msg.x + center_x
        self._box_detection.center_y = msg.y + center_y
        self._box_detection.timestamp = now
        self._last_box_time = now
    
    def _box_dim_cb(self, msg: Point):
        if self._box_detection is None:
            self._box_detection = Detection("box", 0, 0, 0, 0)
        self._box_detection.width = msg.x
        self._box_detection.height = msg.y
    
    def _box_conf_cb(self, msg: Float32):
        if self._box_detection is None:
            self._box_detection = Detection("box", 0, 0, 0, 0)
        self._box_detection.confidence = msg.data
    
    def _handle_error_cb(self, msg: Point):
        now = time.time()
        if self._handle_detection is None:
            self._handle_detection = Detection("handle", 0, 0, 0, 0)
        center_x = self.get_parameter('image_center_x').value
        center_y = self.get_parameter('image_center_y').value
        self._handle_detection.center_x = msg.x + center_x
        self._handle_detection.center_y = msg.y + center_y
        self._handle_detection.timestamp = now
        self._last_handle_time = now
        
        self._last_handle_error_x = msg.x
        self._last_handle_error_y = msg.y
    
    def _handle_dim_cb(self, msg: Point):
        if self._handle_detection is None:
            self._handle_detection = Detection("handle", 0, 0, 0, 0)
        self._handle_detection.width = msg.x
        self._handle_detection.height = msg.y
    
    def _handle_conf_cb(self, msg: Float32):
        if self._handle_detection is None:
            self._handle_detection = Detection("handle", 0, 0, 0, 0)
        self._handle_detection.confidence = msg.data
    
    def _yolo_cb(self, msg: Detection2DArray):
        now = time.time()
        self._latest_detections = []
        for det in msg.detections:
            if not det.results:
                continue
            self._latest_detections.append(Detection(
                class_name=str(det.results[0].hypothesis.class_id),
                center_x=det.bbox.center.position.x,
                center_y=det.bbox.center.position.y,
                width=det.bbox.size_x,
                height=det.bbox.size_y,
                confidence=det.results[0].hypothesis.score,
                timestamp=now
            ))
    
    # =========================================================================
    # DETECTION GETTERS
    # =========================================================================
    
    def _get_box_detection(self) -> Optional[Detection]:
        timeout = self.get_parameter('detection_timeout').value
        min_conf = self.get_parameter('min_confidence').value
        now = time.time()
        
        if self._box_detection:
            age = now - self._last_box_time
            if age < timeout and self._box_detection.confidence >= min_conf:
                return self._box_detection
        return None
    
    def _get_handle_detection(self) -> Optional[Detection]:
        timeout = self.get_parameter('detection_timeout').value
        min_conf = self.get_parameter('min_confidence').value
        now = time.time()
        
        if self._handle_detection:
            age = now - self._last_handle_time
            if age < timeout and self._handle_detection.confidence >= min_conf:
                return self._handle_detection
        return None
    
    def _get_handle_fill_ratio(self, handle: Detection) -> float:
        """Get handle area as fraction of image"""
        img_width = self.get_parameter('image_width').value
        img_height = self.get_parameter('image_height').value
        image_area = img_width * img_height
        handle_area = handle.width * handle.height
        return handle_area / image_area
    
    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================
    
    def _control_loop(self):
        self._control_count += 1
        now = time.time()
        
        if self._emergency_stop:
            self._send_stop_command()
            return
        
        if self._mode == MissionMode.MANUAL:
            return
        
        buoyancy = self.get_parameter('buoyancy_compensation').value
        
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        if self._mode == MissionMode.ALIGN_TO_HANDLE:
            cmd = self._do_align(buoyancy)
        elif self._mode == MissionMode.APPROACH_STRAIGHT:
            cmd = self._do_approach_straight(buoyancy)
        elif self._mode == MissionMode.FINAL_APPROACH:
            cmd = self._do_final_approach(buoyancy)
        elif self._mode == MissionMode.BACKUP:
            cmd = self._do_backup(buoyancy)
        elif self._mode == MissionMode.HOLD_POSITION:
            cmd = self._do_hold(buoyancy)
        elif self._mode == MissionMode.SEARCH:
            cmd = self._do_search(buoyancy)
        
        self.pub_cmd_vel.publish(cmd)
        self._last_cmd_time = now
    
    # =========================================================================
    # PHASE 1: ALIGN TO HANDLE (PID CONTROLLED)
    # =========================================================================
    
    def _do_align(self, buoyancy: float) -> Twist:
        """
        ALIGN phase with PID control.
        Centers handle in view before approaching.
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
        # Initialize
        if self._align_start_time is None:
            self._align_start_time = now
            self._yaw_pid_align.reset()
            self._heave_pid_align.reset()
            self._centered_count = 0
            self.get_logger().info("Starting ALIGN phase with PID control...")
        
        # Timeout check
        align_timeout = self.get_parameter('align_timeout').value
        if now - self._align_start_time > align_timeout:
            self.get_logger().warn(f"Align timeout after {align_timeout}s - HOLD")
            self.set_mode(MissionMode.HOLD_POSITION)
            return cmd
        
        handle = self._get_handle_detection()
        box = self._get_box_detection()
        
        if not handle:
            if box:
                self.get_logger().info("No handle during ALIGN - searching...")
                cmd.angular.z = 0.08
                self._centered_count = 0
                self._yaw_pid_align.reset()
                return cmd
            else:
                self.get_logger().warn("Lost both - HOLD")
                self.set_mode(MissionMode.HOLD_POSITION)
                return cmd
        
        # Calculate errors (target = image center)
        center_x = self.get_parameter('image_center_x').value
        center_y = self.get_parameter('image_center_y').value
        
        error_x = handle.center_x - center_x
        error_y = handle.center_y - center_y
        
        # Get tolerances
        tol_x = self.get_parameter('align_tolerance_x').value
        tol_y = self.get_parameter('align_tolerance_y').value
        
        is_centered_x = abs(error_x) < tol_x
        is_centered_y = abs(error_y) < tol_y
        is_centered = is_centered_x and is_centered_y
        
        if is_centered:
            self._centered_count += 1
        else:
            self._centered_count = max(0, self._centered_count - 1)  # Decay slowly
        
        # PID control
        yaw_sign = self.get_parameter('yaw_sign').value
        heave_sign = self.get_parameter('heave_sign').value
        
        yaw_cmd = yaw_sign * self._yaw_pid_align.compute(error_x, now)
        heave_cmd = heave_sign * self._heave_pid_align.compute(error_y, now)
        
        cmd.angular.z = yaw_cmd
        cmd.linear.z = buoyancy + heave_cmd
        cmd.linear.x = 0.0  # No forward movement during align
        
        # Check if stable enough
        stability_required = self.get_parameter('stability_count_required').value
        
        if self._centered_count >= stability_required:
            self.get_logger().info("=" * 50)
            self.get_logger().info(">>> ALIGNED - STARTING STRAIGHT APPROACH <<<")
            self.get_logger().info("=" * 50)
            self._aligned = True
            self._approach_start_time = time.time()
            self._yaw_pid_approach.reset()
            self._heave_pid_approach.reset()
            self.set_mode(MissionMode.APPROACH_STRAIGHT)
            return cmd
        
        # Logging
        if self._control_count % 10 == 0:
            cx = "✓" if is_centered_x else " "
            cy = "✓" if is_centered_y else " "
            handle_area = handle.width * handle.height
            
            self.get_logger().info(
                f"ALIGN [{cx}{cy}] err=({error_x:+.0f},{error_y:+.0f}) "
                f"yaw={yaw_cmd:+.3f} heave={heave_cmd:+.3f} | "
                f"stable:{self._centered_count}/{stability_required} area={handle_area:.0f}"
            )
        
        return cmd
    
    # =========================================================================
    # PHASE 2: STRAIGHT-LINE APPROACH (PID + FORWARD)
    # =========================================================================
    
    def _do_approach_straight(self, buoyancy: float) -> Twist:
        """
        STRAIGHT APPROACH with PID corrections.
        Continues even if box is lost (as long as handle visible).
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
        handle = self._get_handle_detection()
        box = self._get_box_detection()
        
        if not handle:
            if box:
                box_area = box.width * box.height
                box_too_close = self.get_parameter('box_too_close_area').value
                
                if box_area > box_too_close:
                    self.get_logger().info("Lost handle, box too close - backing up...")
                    self.set_mode(MissionMode.BACKUP)
                    return cmd
                else:
                    self.get_logger().info("Lost handle - returning to ALIGN")
                    self._aligned = False
                    self._align_start_time = None
                    self.set_mode(MissionMode.ALIGN_TO_HANDLE)
                    return cmd
            else:
                self.get_logger().warn("Lost both - HOLD")
                self.set_mode(MissionMode.HOLD_POSITION)
                return cmd
        
        # Handle visible - check size
        handle_area = handle.width * handle.height
        fill_ratio = self._get_handle_fill_ratio(handle)
        
        fill_final = self.get_parameter('handle_fill_final').value
        fill_stop = self.get_parameter('handle_fill_stop').value
        
        # Check for final approach
        if fill_ratio > fill_final:
            self.get_logger().info("=" * 50)
            self.get_logger().info(f">>> HANDLE CLOSE (fill={fill_ratio*100:.1f}%) - FINAL APPROACH <<<")
            self.get_logger().info("=" * 50)
            self.set_mode(MissionMode.FINAL_APPROACH)
            return self._do_final_approach(buoyancy)
        
        # Calculate errors
        center_x = self.get_parameter('image_center_x').value
        center_y = self.get_parameter('image_center_y').value
        
        error_x = handle.center_x - center_x
        error_y = handle.center_y - center_y
        
        # PID control (approach gains - lower)
        yaw_sign = self.get_parameter('yaw_sign').value
        heave_sign = self.get_parameter('heave_sign').value
        
        yaw_cmd = yaw_sign * self._yaw_pid_approach.compute(error_x, now)
        heave_cmd = heave_sign * self._heave_pid_approach.compute(error_y, now)
        
        # Get tolerances
        tol_x = self.get_parameter('approach_tolerance_x').value
        tol_y = self.get_parameter('approach_tolerance_y').value
        
        is_centered_x = abs(error_x) < tol_x
        is_centered_y = abs(error_y) < tol_y
        
        # SURGE - faster when centered, slower when off
        max_surge = self.get_parameter('max_surge').value
        
        if is_centered_x and is_centered_y:
            surge_cmd = max_surge  # Full speed when centered
        elif is_centered_x or is_centered_y:
            surge_cmd = max_surge * 0.7  # Reduce if partially off
        else:
            surge_cmd = max_surge * 0.4  # Slow if both off, prioritize centering
        
        # Scale surge based on handle size (slow down as we get closer)
        area_close = self.get_parameter('handle_area_close').value
        if handle_area > area_close:
            surge_scale = max(0.5, 1.0 - (handle_area - area_close) / area_close)
            surge_cmd *= surge_scale
        
        cmd.angular.z = yaw_cmd
        cmd.linear.z = buoyancy + heave_cmd
        cmd.linear.x = surge_cmd
        
        # Status
        handle_only = not box
        handle_only_str = " [HANDLE-ONLY]" if handle_only else ""
        
        if self._control_count % 10 == 0:
            cx = "✓" if is_centered_x else " "
            cy = "✓" if is_centered_y else " "
            elapsed = now - self._approach_start_time if self._approach_start_time else 0
            
            self.get_logger().info(
                f"APPROACH [{cx}{cy}] err=({error_x:+.0f},{error_y:+.0f}) "
                f"surge={surge_cmd:.2f} yaw={yaw_cmd:+.3f}{handle_only_str} | "
                f"fill={fill_ratio*100:.1f}% T={elapsed:.0f}s"
            )
        
        return cmd
    
    # =========================================================================
    # PHASE 3: FINAL APPROACH
    # =========================================================================
    
    def _do_final_approach(self, buoyancy: float) -> Twist:
        """
        FINAL APPROACH: Handle fills significant portion of screen.
        Very slow, careful. Stop when fill ratio exceeds threshold.
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
        handle = self._get_handle_detection()
        
        if not handle:
            self.get_logger().info("Lost handle in final approach - HOLD")
            self.set_mode(MissionMode.HOLD_POSITION)
            return cmd
        
        # Calculate fill ratio
        fill_ratio = self._get_handle_fill_ratio(handle)
        fill_stop = self.get_parameter('handle_fill_stop').value
        
        # Check if should stop
        if fill_ratio > fill_stop:
            if self._control_count % 20 == 0:
                self.get_logger().info(f">>> HANDLE FILLS SCREEN ({fill_ratio*100:.1f}%) - HOLDING FOR GRIPPER <<<")
            
            # Hold position - tiny corrections only
            center_x = self.get_parameter('image_center_x').value
            center_y = self.get_parameter('image_center_y').value
            error_x = handle.center_x - center_x
            error_y = handle.center_y - center_y
            
            yaw_sign = self.get_parameter('yaw_sign').value
            heave_sign = self.get_parameter('heave_sign').value
            
            cmd.angular.z = yaw_sign * 0.0003 * error_x  # Tiny corrections
            cmd.linear.z = buoyancy + heave_sign * 0.0005 * error_y
            cmd.linear.x = 0.0  # STOP
            
            return cmd
        
        # Not at stop threshold - continue slowly
        center_x = self.get_parameter('image_center_x').value
        center_y = self.get_parameter('image_center_y').value
        error_x = handle.center_x - center_x
        error_y = handle.center_y - center_y
        
        yaw_sign = self.get_parameter('yaw_sign').value
        heave_sign = self.get_parameter('heave_sign').value
        
        # Use approach PID but with reduced output
        yaw_cmd = yaw_sign * self._yaw_pid_approach.compute(error_x, now) * 0.5
        heave_cmd = heave_sign * self._heave_pid_approach.compute(error_y, now) * 0.5
        
        final_speed = self.get_parameter('final_approach_speed').value
        
        # Scale speed based on how close to stop threshold
        speed_scale = max(0.3, 1.0 - (fill_ratio / fill_stop))
        surge_cmd = final_speed * speed_scale
        
        cmd.angular.z = yaw_cmd
        cmd.linear.z = buoyancy + heave_cmd
        cmd.linear.x = surge_cmd
        
        if self._control_count % 10 == 0:
            self.get_logger().info(
                f"FINAL err=({error_x:+.0f},{error_y:+.0f}) "
                f"surge={surge_cmd:.3f} fill={fill_ratio*100:.1f}%/{fill_stop*100:.0f}%"
            )
        
        return cmd
    
    # =========================================================================
    # BACKUP
    # =========================================================================
    
    def _do_backup(self, buoyancy: float) -> Twist:
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
        if self._backup_start_time is None:
            self._backup_start_time = now
            self.get_logger().info(">>> BACKING UP <<<")
        
        backup_duration = 2.5
        elapsed = now - self._backup_start_time
        
        handle = self._get_handle_detection()
        if handle and elapsed > 0.5:
            self.get_logger().info("Handle found while backing up - realigning")
            self._backup_start_time = None
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN_TO_HANDLE)
            return cmd
        
        if elapsed > backup_duration:
            self.get_logger().info("Backup complete - realigning")
            self._backup_start_time = None
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN_TO_HANDLE)
            return cmd
        
        # Back up while centering on box if visible
        box = self._get_box_detection()
        if box:
            center_x = self.get_parameter('image_center_x').value
            center_y = self.get_parameter('image_center_y').value
            error_x = box.center_x - center_x
            error_y = box.center_y - center_y
            
            yaw_sign = self.get_parameter('yaw_sign').value
            heave_sign = self.get_parameter('heave_sign').value
            
            cmd.angular.z = yaw_sign * 0.0008 * error_x
            cmd.linear.z = buoyancy + heave_sign * 0.002 * error_y
        
        cmd.linear.x = -self.get_parameter('backup_speed').value
        
        if self._control_count % 10 == 0:
            self.get_logger().info(f"Backing up... ({elapsed:.1f}/{backup_duration:.1f}s)")
        
        return cmd
    
    # =========================================================================
    # HOLD / SEARCH
    # =========================================================================
    
    def _do_hold(self, buoyancy: float) -> Twist:
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        handle = self._get_handle_detection()
        if handle:
            self.get_logger().info(">>> Handle detected - starting ALIGN <<<")
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN_TO_HANDLE)
            return cmd
        
        return cmd
    
    def _do_search(self, buoyancy: float) -> Twist:
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        handle = self._get_handle_detection()
        if handle:
            self.get_logger().info(">>> Handle found - starting ALIGN <<<")
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN_TO_HANDLE)
            return cmd
        
        box = self._get_box_detection()
        if box:
            self.get_logger().info(">>> Box found - starting ALIGN <<<")
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN_TO_HANDLE)
            return cmd
        
        cmd.angular.z = 0.10
        return cmd
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def _joy_cb(self, msg: Joy):
        buttons = msg.buttons
        now = time.time()
        
        if not hasattr(self, '_joy_count'):
            self._joy_count = 0
        self._joy_count += 1
        if self._joy_count % 20 == 1:
            self.get_logger().info(
                f"[DEBUG] Joy: buttons[0:4]={list(buttons[0:4]) if len(buttons)>=4 else buttons}"
            )
        
        def debounced(idx):
            if len(buttons) <= idx or buttons[idx] != 1:
                return False
            if now - self._button_debounce.get(idx, 0) < 0.3:
                return False
            self._button_debounce[idx] = now
            return True
        
        if debounced(3):  # Y
            self.get_logger().info("Y → MANUAL")
            self.set_mode(MissionMode.MANUAL)
        
        if debounced(2):  # X
            handle = self._get_handle_detection()
            if handle:
                self.get_logger().info("X → Handle visible, ALIGN")
                self._aligned = False
                self._align_start_time = None
                self.set_mode(MissionMode.ALIGN_TO_HANDLE)
            else:
                self.get_logger().info("X → No handle, SEARCH")
                self.set_mode(MissionMode.SEARCH)
        
        if debounced(0):  # A
            self.get_logger().info("A → HOLD")
            self.set_mode(MissionMode.HOLD_POSITION)
        
        if debounced(1):  # B
            self.get_logger().info("B → MANUAL")
            self.set_mode(MissionMode.MANUAL)
    
    def _btn_a_cb(self, msg: Bool):
        pass
    
    # =========================================================================
    # SAFETY / MODE
    # =========================================================================
    
    def _safety_check(self):
        timeout = self.get_parameter('safety_timeout').value
        if time.time() - self._last_cmd_time > timeout:
            if self._mode not in [MissionMode.MANUAL]:
                self.get_logger().error("Safety timeout → MANUAL")
                self.set_mode(MissionMode.MANUAL)
    
    def _send_stop_command(self):
        cmd = Twist()
        try:
            self.pub_cmd_vel.publish(cmd)
        except Exception:
            pass
    
    def set_mode(self, mode: MissionMode):
        if mode == self._mode:
            return
        
        old = self._mode
        self._mode = mode
        
        self.get_logger().info(f"MODE: {old.name} → {mode.name}")
        
        mode_msg = String()
        mode_msg.data = mode.name
        self.pub_mode.publish(mode_msg)
        
        if mode == MissionMode.HOLD_POSITION:
            self._centered_count = 0
            self._aligned = False
            self._yaw_pid_align.reset()
            self._heave_pid_align.reset()
            self._yaw_pid_approach.reset()
            self._heave_pid_approach.reset()
        elif mode == MissionMode.MANUAL:
            self._centered_count = 0
            self._aligned = False
            self._approach_start_time = None
            self._align_start_time = None
            self._backup_start_time = None
            self._yaw_pid_align.reset()
            self._heave_pid_align.reset()
            self._yaw_pid_approach.reset()
            self._heave_pid_approach.reset()
            self._send_stop_command()
    
    def _status_loop(self):
        box = self._get_box_detection()
        handle = self._get_handle_detection()
        
        box_str = f"BOX:{'Y' if box else 'N'}"
        handle_str = f"HDL:{'Y' if handle else 'N'}"
        
        if box:
            box_str += f"({box.width * box.height:.0f})"
        if handle:
            fill = self._get_handle_fill_ratio(handle) * 100
            handle_str += f"({handle.width * handle.height:.0f})[{fill:.1f}%]"
        
        aligned_str = "ALIGNED" if self._aligned else "NOT-ALIGNED"
        
        msg = String()
        msg.data = f"{self._mode.name} | {box_str} {handle_str} | {aligned_str}"
        self.pub_status.publish(msg)
        
        if self._mode != MissionMode.MANUAL:
            self.get_logger().info(f"[STATUS] {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNodeReal()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        try:
            node._send_stop_command()
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
