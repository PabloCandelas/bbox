#!/usr/bin/env python3
"""
<<<<<<< HEAD
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
=======
navigation_node_real_v7.py - FULL DOCKING SEQUENCE with Carabiner Attachment
=============================================================================

COMPLETE MISSION SEQUENCE:
--------------------------
1. ALIGN       - Center BOTH box and handle (ensure facing handle-face directly)
2. APPROACH    - Steady straight-line approach, minimal corrections
3. CLOSE       - Handle fills screen, box may be lost, continue carefully
4. BLIND_PUSH  - Timed forward push to attach carabiner (no visual feedback)
5. VERIFY      - Back up and check if box moved WITH robot (attachment test)
6. SUCCESS/RETRY - Either attached (mission complete) or retry

ALIGNMENT STRATEGY:
-------------------
- Box center and Handle center should be roughly vertically aligned
- This ensures we're facing the handle-face directly (not from the side)
- Robot centers on HANDLE, but checks that HANDLE is centered on BOX

BLIND PUSH:
-----------
- When handle fills screen beyond threshold, visual servoing becomes unreliable
- Robot does a timed forward push (e.g., 3 seconds at low speed)
- This pushes the carabiner onto the handle

VERIFICATION:
-------------
- After blind push, robot backs up slowly
- If attached: handle/box stays centered (moves with robot)
- If NOT attached: handle/box drifts away and gets smaller
- Compare handle position before/after backup to detect attachment
>>>>>>> ef6e778 (final version docs)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import math
import time
from enum import Enum
from typing import Optional, List, Tuple
from dataclasses import dataclass
from collections import deque

from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Joy
from vision_msgs.msg import Detection2DArray


# =============================================================================
# PID CONTROLLER
# =============================================================================

class PIDController:
    """PID Controller with derivative filtering and anti-windup"""
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_min: float = -1.0, output_max: float = 1.0,
                 integral_max: float = 100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max
        
        self._integral = 0.0
        self._last_error = None
        self._last_time = None
        self._derivative_filter = deque(maxlen=3)
    
    def reset(self):
        self._integral = 0.0
        self._last_error = None
        self._last_time = None
        self._derivative_filter.clear()
    
    def compute(self, error: float, dt: float = None) -> float:
        now = time.time()
        
        if dt is None:
            dt = 0.05 if self._last_time is None else max(0.001, now - self._last_time)
        
        # P term
        p_term = self.kp * error
        
        # I term with anti-windup
        self._integral = np.clip(self._integral + error * dt, 
                                  -self.integral_max, self.integral_max)
        i_term = self.ki * self._integral
        
        # D term with filtering
        if self._last_error is not None:
            raw_derivative = (error - self._last_error) / dt
            self._derivative_filter.append(raw_derivative)
            filtered_derivative = sum(self._derivative_filter) / len(self._derivative_filter)
            d_term = self.kd * filtered_derivative
        else:
            d_term = 0.0
        
        self._last_error = error
        self._last_time = now
        
        output = np.clip(p_term + i_term + d_term, self.output_min, self.output_max)
        return output


# =============================================================================
# DATA CLASSES
# =============================================================================

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
<<<<<<< HEAD
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
=======
    ALIGN = 2              # Align box and handle centers
    APPROACH = 3           # Straight-line approach
    CLOSE_APPROACH = 4     # Handle fills screen, careful approach
    BLIND_PUSH = 5         # Timed forward push to attach
    VERIFY_BACKUP = 6      # Back up to verify attachment
    VERIFY_CHECK = 7       # Check if attached
    ATTACHED = 8           # Successfully attached!
    HOLD_POSITION = 9
    BACKUP = 10


# =============================================================================
# MAIN NODE
# =============================================================================

class NavigationNodeReal(Node):
    """
    Full Docking Sequence Navigation for BlueROV
    
    Aligns box+handle centers → Approaches → Blind push → Verifies attachment
>>>>>>> ef6e778 (final version docs)
    """
    
    def __init__(self):
        super().__init__('navigation_node_real')
        
        self.get_logger().info("=" * 60)
<<<<<<< HEAD
        self.get_logger().info("Navigation Node v6 - PID CONTROLLED")
        self.get_logger().info("  Strategy: PID align → Straight approach → Stop at target")
=======
        self.get_logger().info("Navigation Node v7 - FULL DOCKING SEQUENCE")
        self.get_logger().info("  ALIGN → APPROACH → CLOSE → BLIND_PUSH → VERIFY")
>>>>>>> ef6e778 (final version docs)
        self.get_logger().info("=" * 60)
        
        self._declare_parameters()
        self._init_controllers()
        self._init_state()
        self._init_ros()
        self._print_startup_info()
    
    def _declare_parameters(self):
        self.declare_parameter('control_rate', 20.0)
        
        # === SPEED LIMITS ===
<<<<<<< HEAD
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
        
=======
        self.declare_parameter('max_surge', 0.12)
        self.declare_parameter('approach_surge', 0.10)
        self.declare_parameter('close_approach_surge', 0.06)
        self.declare_parameter('blind_push_surge', 0.08)
        self.declare_parameter('verify_backup_speed', 0.08)
        self.declare_parameter('max_heave', 0.20)
        self.declare_parameter('max_yaw_rate', 0.20)
        self.declare_parameter('backup_speed', 0.10)
        
        # === PID GAINS ===
        self.declare_parameter('yaw_kp', 0.0015)
        self.declare_parameter('yaw_ki', 0.0001)
        self.declare_parameter('yaw_kd', 0.0025)  # Higher D for less oscillation
        
        self.declare_parameter('heave_kp', 0.0020)
        self.declare_parameter('heave_ki', 0.0001)
        self.declare_parameter('heave_kd', 0.0018)
        
>>>>>>> ef6e778 (final version docs)
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
        
<<<<<<< HEAD
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
=======
        # === ALIGNMENT TOLERANCES ===
        self.declare_parameter('align_tolerance_x', 60.0)
        self.declare_parameter('align_tolerance_y', 80.0)
        
        # Box-Handle alignment: handle should be roughly centered on box
        self.declare_parameter('box_handle_align_tolerance', 150.0)
        
        # === APPROACH TOLERANCES ===
        self.declare_parameter('approach_tolerance_x', 120.0)
        self.declare_parameter('approach_tolerance_y', 150.0)
        
        # === STABILITY ===
        self.declare_parameter('align_stable_count', 15)
        
        # === HANDLE FILL THRESHOLDS ===
        # These control phase transitions based on how much screen the handle fills
        self.declare_parameter('handle_fill_close', 0.08)      # 8% = start CLOSE_APPROACH
        self.declare_parameter('handle_fill_blind_push', 0.15) # 15% = start BLIND_PUSH
        
        # === BLIND PUSH PARAMETERS ===
        self.declare_parameter('blind_push_duration', 3.0)     # Push forward for 3 seconds
        
        # === VERIFICATION PARAMETERS ===
        self.declare_parameter('verify_backup_duration', 2.0)  # Back up for 2 seconds
        self.declare_parameter('verify_check_duration', 1.5)   # Check for 1.5 seconds
        # If handle moves more than this after backup, NOT attached
        self.declare_parameter('attachment_drift_threshold', 200.0)  # pixels
        # If handle area decreases more than this ratio, NOT attached
        self.declare_parameter('attachment_area_decrease_threshold', 0.3)  # 30% decrease
>>>>>>> ef6e778 (final version docs)
        
        # === TIMEOUTS ===
        self.declare_parameter('max_approach_time', 120.0)
        self.declare_parameter('safety_timeout', 5.0)
        self.declare_parameter('align_timeout', 45.0)
<<<<<<< HEAD
    
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
=======
        
        # === APPROACH BEHAVIOR ===
        self.declare_parameter('approach_gain_reduction', 0.3)
    
    def _init_controllers(self):
        yaw_kp = self.get_parameter('yaw_kp').value
        yaw_ki = self.get_parameter('yaw_ki').value
        yaw_kd = self.get_parameter('yaw_kd').value
        max_yaw = self.get_parameter('max_yaw_rate').value
        
        self._yaw_pid = PIDController(
            kp=yaw_kp, ki=yaw_ki, kd=yaw_kd,
            output_min=-max_yaw, output_max=max_yaw
        )
        
        heave_kp = self.get_parameter('heave_kp').value
        heave_ki = self.get_parameter('heave_ki').value
        heave_kd = self.get_parameter('heave_kd').value
        max_heave = self.get_parameter('max_heave').value
        
        self._heave_pid = PIDController(
            kp=heave_kp, ki=heave_ki, kd=heave_kd,
            output_min=-max_heave, output_max=max_heave
        )
        
        self.get_logger().info(f"YAW PID: Kp={yaw_kp}, Ki={yaw_ki}, Kd={yaw_kd}")
        self.get_logger().info(f"HEAVE PID: Kp={heave_kp}, Ki={heave_ki}, Kd={heave_kd}")
>>>>>>> ef6e778 (final version docs)
    
    def _init_state(self):
        self._mode = MissionMode.MANUAL
        
        # Detections
        self._box_detection: Optional[Detection] = None
        self._handle_detection: Optional[Detection] = None
        self._last_box_time: float = 0.0
        self._last_handle_time: float = 0.0
        
        # Timing
        self._last_control_time = time.time()
        self._button_debounce = {}
        self._control_count = 0
        self._approach_start_time = None
        self._align_start_time = None
        
        # Safety
        self._last_cmd_time = time.time()
        
        # Stability
<<<<<<< HEAD
        self._centered_count = 0
=======
        self._stable_count = 0
>>>>>>> ef6e778 (final version docs)
        self._aligned = False
        
        # Backup state
        self._backup_start_time = None
        
<<<<<<< HEAD
        # Last known handle error
        self._last_handle_error_x = 0.0
        self._last_handle_error_y = 0.0
        
        # For logging PID internals
        self._last_yaw_p = 0.0
        self._last_yaw_d = 0.0
=======
        # Blind push state
        self._blind_push_start_time = None
        
        # Verification state
        self._verify_backup_start_time = None
        self._verify_check_start_time = None
        self._pre_backup_handle_center: Optional[Tuple[float, float]] = None
        self._pre_backup_handle_area: Optional[float] = None
        self._attachment_check_samples: List[Tuple[float, float, float]] = []  # (cx, cy, area)
        
        # Mission state
        self._attachment_verified = False
        self._attachment_attempts = 0
>>>>>>> ef6e778 (final version docs)
    
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
<<<<<<< HEAD
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
=======
        self.get_logger().info("v7 DOCKING SEQUENCE:")
        self.get_logger().info("  1. ALIGN    - Center handle, verify box-handle alignment")
        self.get_logger().info("  2. APPROACH - Straight-line, PID corrections")
        self.get_logger().info("  3. CLOSE    - Handle fills screen, careful approach")
        self.get_logger().info("  4. BLIND_PUSH - Timed forward push (3s)")
        self.get_logger().info("  5. VERIFY   - Back up, check if attached")
        self.get_logger().info("")
        self.get_logger().info("CONTROLS: Y=Manual, X=Start, A=Hold, B=Manual")
>>>>>>> ef6e778 (final version docs)
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
<<<<<<< HEAD
        
        self._last_handle_error_x = msg.x
        self._last_handle_error_y = msg.y
=======
>>>>>>> ef6e778 (final version docs)
    
    def _handle_dim_cb(self, msg: Point):
        if self._handle_detection is None:
            self._handle_detection = Detection("handle", 0, 0, 0, 0)
        self._handle_detection.width = msg.x
        self._handle_detection.height = msg.y
    
    def _handle_conf_cb(self, msg: Float32):
        if self._handle_detection is None:
            self._handle_detection = Detection("handle", 0, 0, 0, 0)
        self._handle_detection.confidence = msg.data
    
    # =========================================================================
    # DETECTION GETTERS
    # =========================================================================
    
    def _get_box(self) -> Optional[Detection]:
        timeout = self.get_parameter('detection_timeout').value
        min_conf = self.get_parameter('min_confidence').value
        now = time.time()
        
        if self._box_detection:
            age = now - self._last_box_time
            if age < timeout and self._box_detection.confidence >= min_conf:
                return self._box_detection
        return None
    
    def _get_handle(self) -> Optional[Detection]:
        timeout = self.get_parameter('detection_timeout').value
        min_conf = self.get_parameter('min_confidence').value
        now = time.time()
        
        if self._handle_detection:
            age = now - self._last_handle_time
            if age < timeout and self._handle_detection.confidence >= min_conf:
                return self._handle_detection
        return None
    
    def _get_handle_fill_ratio(self, handle: Detection) -> float:
<<<<<<< HEAD
        """Get handle area as fraction of image"""
        img_width = self.get_parameter('image_width').value
        img_height = self.get_parameter('image_height').value
        image_area = img_width * img_height
        handle_area = handle.width * handle.height
        return handle_area / image_area
=======
        img_w = self.get_parameter('image_width').value
        img_h = self.get_parameter('image_height').value
        return (handle.width * handle.height) / (img_w * img_h)
>>>>>>> ef6e778 (final version docs)
    
    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================
    
    def _control_loop(self):
        self._control_count += 1
        now = time.time()
        
        if self._mode == MissionMode.MANUAL:
            return
        
        buoyancy = self.get_parameter('buoyancy_compensation').value
        
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        if self._mode == MissionMode.ALIGN:
            cmd = self._do_align(buoyancy)
        elif self._mode == MissionMode.APPROACH:
            cmd = self._do_approach(buoyancy)
        elif self._mode == MissionMode.CLOSE_APPROACH:
            cmd = self._do_close_approach(buoyancy)
        elif self._mode == MissionMode.BLIND_PUSH:
            cmd = self._do_blind_push(buoyancy)
        elif self._mode == MissionMode.VERIFY_BACKUP:
            cmd = self._do_verify_backup(buoyancy)
        elif self._mode == MissionMode.VERIFY_CHECK:
            cmd = self._do_verify_check(buoyancy)
        elif self._mode == MissionMode.ATTACHED:
            cmd = self._do_attached(buoyancy)
        elif self._mode == MissionMode.BACKUP:
            cmd = self._do_backup(buoyancy)
        elif self._mode == MissionMode.HOLD_POSITION:
            cmd = self._do_hold(buoyancy)
        elif self._mode == MissionMode.SEARCH:
            cmd = self._do_search(buoyancy)
        
        self.pub_cmd_vel.publish(cmd)
        self._last_cmd_time = now
    
    # =========================================================================
<<<<<<< HEAD
    # PHASE 1: ALIGN TO HANDLE (PID CONTROLLED)
=======
    # PHASE 1: ALIGN (Center handle, verify box-handle alignment)
>>>>>>> ef6e778 (final version docs)
    # =========================================================================
    
    def _do_align(self, buoyancy: float) -> Twist:
        """
<<<<<<< HEAD
        ALIGN phase with PID control.
        Centers handle in view before approaching.
=======
        ALIGN: Center on handle AND verify handle is centered on box.
        This ensures we're facing the handle-face directly.
>>>>>>> ef6e778 (final version docs)
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
<<<<<<< HEAD
        # Initialize
        if self._align_start_time is None:
            self._align_start_time = now
            self._yaw_pid_align.reset()
            self._heave_pid_align.reset()
            self._centered_count = 0
            self.get_logger().info("Starting ALIGN phase with PID control...")
=======
        if self._align_start_time is None:
            self._align_start_time = now
            self._yaw_pid.reset()
            self._heave_pid.reset()
            self._stable_count = 0
            self.get_logger().info("ALIGN: Starting alignment...")
>>>>>>> ef6e778 (final version docs)
        
        # Timeout
        align_timeout = self.get_parameter('align_timeout').value
        if now - self._align_start_time > align_timeout:
            self.get_logger().warn(f"ALIGN timeout after {align_timeout}s")
            self.set_mode(MissionMode.HOLD_POSITION)
            return cmd
        
        handle = self._get_handle()
        box = self._get_box()
        
        if not handle:
            if box:
<<<<<<< HEAD
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
=======
                self.get_logger().info("ALIGN: No handle, searching...")
                cmd.angular.z = 0.06
                self._stable_count = 0
                return cmd
            else:
                self.get_logger().warn("ALIGN: Lost both - HOLD")
                self.set_mode(MissionMode.HOLD_POSITION)
                return cmd
        
        # Calculate handle error (from image center)
>>>>>>> ef6e778 (final version docs)
        center_x = self.get_parameter('image_center_x').value
        center_y = self.get_parameter('image_center_y').value
        
        error_x = handle.center_x - center_x
        error_y = handle.center_y - center_y
        
<<<<<<< HEAD
=======
        # Check box-handle alignment (handle should be roughly centered on box)
        box_handle_aligned = True
        if box:
            box_handle_offset = abs(handle.center_x - box.center_x)
            align_tol = self.get_parameter('box_handle_align_tolerance').value
            box_handle_aligned = box_handle_offset < align_tol
            
            if not box_handle_aligned and self._control_count % 20 == 0:
                self.get_logger().warn(
                    f"ALIGN: Box-Handle offset={box_handle_offset:.0f}px "
                    f"(need <{align_tol:.0f})"
                )
        
>>>>>>> ef6e778 (final version docs)
        # Get tolerances
        tol_x = self.get_parameter('align_tolerance_x').value
        tol_y = self.get_parameter('align_tolerance_y').value
        
        is_centered_x = abs(error_x) < tol_x
        is_centered_y = abs(error_y) < tol_y
        is_centered = is_centered_x and is_centered_y and box_handle_aligned
        
        if is_centered:
            self._stable_count += 1
        else:
<<<<<<< HEAD
            self._centered_count = max(0, self._centered_count - 1)  # Decay slowly
=======
            self._stable_count = max(0, self._stable_count - 2)
>>>>>>> ef6e778 (final version docs)
        
        # PID control
        yaw_sign = self.get_parameter('yaw_sign').value
        heave_sign = self.get_parameter('heave_sign').value
        
<<<<<<< HEAD
        yaw_cmd = yaw_sign * self._yaw_pid_align.compute(error_x, now)
        heave_cmd = heave_sign * self._heave_pid_align.compute(error_y, now)
        
        cmd.angular.z = yaw_cmd
        cmd.linear.z = buoyancy + heave_cmd
        cmd.linear.x = 0.0  # No forward movement during align
        
        # Check if stable enough
        stability_required = self.get_parameter('stability_count_required').value
=======
        yaw_cmd = yaw_sign * self._yaw_pid.compute(error_x)
        heave_cmd = heave_sign * self._heave_pid.compute(error_y)
        
        cmd.angular.z = yaw_cmd
        cmd.linear.z = buoyancy + heave_cmd
        cmd.linear.x = 0.0  # NO forward during align
        
        # Check stability
        stable_required = self.get_parameter('align_stable_count').value
>>>>>>> ef6e778 (final version docs)
        
        if self._stable_count >= stable_required:
            self.get_logger().info("=" * 50)
            self.get_logger().info(">>> ALIGNED - STARTING APPROACH <<<")
            self.get_logger().info("=" * 50)
            self._aligned = True
            self._approach_start_time = time.time()
<<<<<<< HEAD
            self._yaw_pid_approach.reset()
            self._heave_pid_approach.reset()
            self.set_mode(MissionMode.APPROACH_STRAIGHT)
=======
            self.set_mode(MissionMode.APPROACH)
>>>>>>> ef6e778 (final version docs)
            return cmd
        
        # Logging
        if self._control_count % 10 == 0:
            cx = "✓" if is_centered_x else " "
            cy = "✓" if is_centered_y else " "
<<<<<<< HEAD
            handle_area = handle.width * handle.height
            
            self.get_logger().info(
                f"ALIGN [{cx}{cy}] err=({error_x:+.0f},{error_y:+.0f}) "
                f"yaw={yaw_cmd:+.3f} heave={heave_cmd:+.3f} | "
                f"stable:{self._centered_count}/{stability_required} area={handle_area:.0f}"
=======
            ba = "✓" if box_handle_aligned else "✗"
            
            self.get_logger().info(
                f"ALIGN [{cx}{cy}|{ba}] err=({error_x:+.0f},{error_y:+.0f}) "
                f"yaw={yaw_cmd:+.3f} | stable:{self._stable_count}/{stable_required}"
>>>>>>> ef6e778 (final version docs)
            )
        
        return cmd
    
    # =========================================================================
<<<<<<< HEAD
    # PHASE 2: STRAIGHT-LINE APPROACH (PID + FORWARD)
=======
    # PHASE 2: APPROACH (Straight-line, PID corrections)
>>>>>>> ef6e778 (final version docs)
    # =========================================================================
    
    def _do_approach(self, buoyancy: float) -> Twist:
        """
<<<<<<< HEAD
        STRAIGHT APPROACH with PID corrections.
        Continues even if box is lost (as long as handle visible).
=======
        APPROACH: Steady straight-line approach with reduced PID corrections.
        Transitions to CLOSE_APPROACH when handle fills threshold.
>>>>>>> ef6e778 (final version docs)
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        handle = self._get_handle()
        box = self._get_box()
        
<<<<<<< HEAD
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
=======
        if not handle:
            if box:
                self.get_logger().info("APPROACH: Lost handle - re-align")
                self._aligned = False
                self._align_start_time = None
                self.set_mode(MissionMode.ALIGN)
                return cmd
            else:
                self.get_logger().warn("APPROACH: Lost everything - HOLD")
                self.set_mode(MissionMode.HOLD_POSITION)
                return cmd
        
        # Check fill ratio
        fill_ratio = self._get_handle_fill_ratio(handle)
        close_threshold = self.get_parameter('handle_fill_close').value
        
        if fill_ratio >= close_threshold:
            self.get_logger().info("=" * 50)
            self.get_logger().info(f">>> CLOSE APPROACH ({fill_ratio*100:.1f}% fill) <<<")
            self.get_logger().info("=" * 50)
            self.set_mode(MissionMode.CLOSE_APPROACH)
>>>>>>> ef6e778 (final version docs)
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
        
<<<<<<< HEAD
        yaw_sign = self.get_parameter('yaw_sign').value
        heave_sign = self.get_parameter('heave_sign').value
        
        # Use approach PID but with reduced output
        yaw_cmd = yaw_sign * self._yaw_pid_approach.compute(error_x, now) * 0.5
        heave_cmd = heave_sign * self._heave_pid_approach.compute(error_y, now) * 0.5
        
        final_speed = self.get_parameter('final_approach_speed').value
        
        # Scale speed based on how close to stop threshold
        speed_scale = max(0.3, 1.0 - (fill_ratio / fill_stop))
        surge_cmd = final_speed * speed_scale
=======
        # Check drift
        drift_tol_x = self.get_parameter('approach_tolerance_x').value
        drift_tol_y = self.get_parameter('approach_tolerance_y').value
        
        if abs(error_x) > drift_tol_x or abs(error_y) > drift_tol_y:
            self.get_logger().info("APPROACH: Drifted - re-align")
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN)
            return cmd
        
        # Reduced gain PID
        yaw_sign = self.get_parameter('yaw_sign').value
        heave_sign = self.get_parameter('heave_sign').value
        gain_reduction = self.get_parameter('approach_gain_reduction').value
        
        orig_kp = self._yaw_pid.kp
        self._yaw_pid.kp = orig_kp * gain_reduction
        yaw_cmd = yaw_sign * self._yaw_pid.compute(error_x)
        self._yaw_pid.kp = orig_kp
        
        orig_kp = self._heave_pid.kp
        self._heave_pid.kp = orig_kp * gain_reduction
        heave_cmd = heave_sign * self._heave_pid.compute(error_y)
        self._heave_pid.kp = orig_kp
        
        # Steady surge
        surge_cmd = self.get_parameter('approach_surge').value
        
        cmd.angular.z = yaw_cmd
        cmd.linear.z = buoyancy + heave_cmd
        cmd.linear.x = surge_cmd
        
        if self._control_count % 10 == 0:
            box_str = "+BOX" if box else "HDL-ONLY"
            self.get_logger().info(
                f"APPROACH err=({error_x:+.0f},{error_y:+.0f}) "
                f"surge={surge_cmd:.2f} fill={fill_ratio*100:.1f}% [{box_str}]"
            )
        
        return cmd
    
    # =========================================================================
    # PHASE 3: CLOSE APPROACH (Handle fills screen)
    # =========================================================================
    
    def _do_close_approach(self, buoyancy: float) -> Twist:
        """
        CLOSE APPROACH: Handle fills significant portion of screen.
        Continue carefully until ready for blind push.
        Box may be lost at this point - that's OK.
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        handle = self._get_handle()
        
        if not handle:
            self.get_logger().warn("CLOSE: Lost handle - backing up")
            self.set_mode(MissionMode.BACKUP)
            return cmd
        
        fill_ratio = self._get_handle_fill_ratio(handle)
        blind_push_threshold = self.get_parameter('handle_fill_blind_push').value
        
        # Ready for blind push?
        if fill_ratio >= blind_push_threshold:
            self.get_logger().info("=" * 50)
            self.get_logger().info(f">>> BLIND PUSH ({fill_ratio*100:.1f}% fill) <<<")
            self.get_logger().info(">>> Handle centered - pushing forward! <<<")
            self.get_logger().info("=" * 50)
            self._blind_push_start_time = None
            self.set_mode(MissionMode.BLIND_PUSH)
            return cmd
        
        # Center on handle
        center_x = self.get_parameter('image_center_x').value
        center_y = self.get_parameter('image_center_y').value
        
        error_x = handle.center_x - center_x
        error_y = handle.center_y - center_y
        
        # Very gentle corrections
        yaw_sign = self.get_parameter('yaw_sign').value
        heave_sign = self.get_parameter('heave_sign').value
        
        yaw_cmd = yaw_sign * 0.0005 * error_x
        heave_cmd = heave_sign * 0.0008 * error_y
        
        yaw_cmd = np.clip(yaw_cmd, -0.05, 0.05)
        heave_cmd = np.clip(heave_cmd, -0.10, 0.10)
        
        surge_cmd = self.get_parameter('close_approach_surge').value
>>>>>>> ef6e778 (final version docs)
        
        cmd.angular.z = yaw_cmd
        cmd.linear.z = buoyancy + heave_cmd
        cmd.linear.x = surge_cmd
        
        if self._control_count % 10 == 0:
            self.get_logger().info(
<<<<<<< HEAD
                f"FINAL err=({error_x:+.0f},{error_y:+.0f}) "
                f"surge={surge_cmd:.3f} fill={fill_ratio*100:.1f}%/{fill_stop*100:.0f}%"
=======
                f"CLOSE err=({error_x:+.0f},{error_y:+.0f}) "
                f"surge={surge_cmd:.2f} fill={fill_ratio*100:.1f}%"
>>>>>>> ef6e778 (final version docs)
            )
        
        return cmd
    
    # =========================================================================
    # PHASE 4: BLIND PUSH (Timed forward push)
    # =========================================================================
    
    def _do_blind_push(self, buoyancy: float) -> Twist:
        """
        BLIND PUSH: Timed forward movement to attach carabiner.
        No visual feedback - just push forward for set duration.
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
        if self._blind_push_start_time is None:
            self._blind_push_start_time = now
            self._attachment_attempts += 1
            self.get_logger().info(f"BLIND PUSH: Starting (attempt #{self._attachment_attempts})")
        
        duration = self.get_parameter('blind_push_duration').value
        elapsed = now - self._blind_push_start_time
        
        if elapsed >= duration:
            self.get_logger().info("=" * 50)
            self.get_logger().info(">>> BLIND PUSH COMPLETE - VERIFYING <<<")
            self.get_logger().info("=" * 50)
            
            # Record current handle state for comparison
            handle = self._get_handle()
            if handle:
                self._pre_backup_handle_center = (handle.center_x, handle.center_y)
                self._pre_backup_handle_area = handle.width * handle.height
            else:
                self._pre_backup_handle_center = None
                self._pre_backup_handle_area = None
            
            self._verify_backup_start_time = None
            self._attachment_check_samples = []
            self.set_mode(MissionMode.VERIFY_BACKUP)
            return cmd
        
        # Push forward (no corrections - blind!)
        surge_cmd = self.get_parameter('blind_push_surge').value
        cmd.linear.x = surge_cmd
        
        if self._control_count % 10 == 0:
            self.get_logger().info(f"BLIND PUSH: {elapsed:.1f}/{duration:.1f}s")
        
        return cmd
    
    # =========================================================================
    # PHASE 5: VERIFY BACKUP (Back up to check attachment)
    # =========================================================================
    
    def _do_verify_backup(self, buoyancy: float) -> Twist:
        """
        VERIFY BACKUP: Back up slowly to test attachment.
        If attached, handle/box should move WITH the robot.
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
        if self._verify_backup_start_time is None:
            self._verify_backup_start_time = now
            self.get_logger().info("VERIFY: Backing up to test attachment...")
        
        duration = self.get_parameter('verify_backup_duration').value
        elapsed = now - self._verify_backup_start_time
        
        if elapsed >= duration:
            self.get_logger().info("VERIFY: Backup complete - checking attachment")
            self._verify_check_start_time = None
            self.set_mode(MissionMode.VERIFY_CHECK)
            return cmd
        
        # Back up slowly
        cmd.linear.x = -self.get_parameter('verify_backup_speed').value
        
        # Collect samples during backup
        handle = self._get_handle()
        if handle:
            self._attachment_check_samples.append(
                (handle.center_x, handle.center_y, handle.width * handle.height)
            )
        
        if self._control_count % 10 == 0:
            self.get_logger().info(f"VERIFY BACKUP: {elapsed:.1f}/{duration:.1f}s")
        
        return cmd
    
    # =========================================================================
    # PHASE 6: VERIFY CHECK (Analyze if attached)
    # =========================================================================
    
    def _do_verify_check(self, buoyancy: float) -> Twist:
        """
        VERIFY CHECK: Analyze handle movement to determine attachment.
        
        If ATTACHED:
        - Handle stays roughly centered (moves with robot)
        - Handle area stays similar
        
        If NOT ATTACHED:
        - Handle drifts toward bottom of image (robot moved back, box didn't)
        - Handle area decreases significantly (box got further away)
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
        if self._verify_check_start_time is None:
            self._verify_check_start_time = now
            self.get_logger().info("VERIFY CHECK: Analyzing attachment...")
        
        duration = self.get_parameter('verify_check_duration').value
        elapsed = now - self._verify_check_start_time
        
        # Collect more samples
        handle = self._get_handle()
        if handle:
            self._attachment_check_samples.append(
                (handle.center_x, handle.center_y, handle.width * handle.height)
            )
        
        if elapsed >= duration:
            # Analyze samples
            attached = self._analyze_attachment()
            
            if attached:
                self.get_logger().info("=" * 50)
                self.get_logger().info(">>> ATTACHMENT VERIFIED! <<<")
                self.get_logger().info(">>> MISSION SUCCESS! <<<")
                self.get_logger().info("=" * 50)
                self._attachment_verified = True
                self.set_mode(MissionMode.ATTACHED)
            else:
                self.get_logger().warn("=" * 50)
                self.get_logger().warn(">>> NOT ATTACHED - RETRY <<<")
                self.get_logger().warn("=" * 50)
                
                # Reset and retry
                self._aligned = False
                self._align_start_time = None
                self._blind_push_start_time = None
                self.set_mode(MissionMode.ALIGN)
            
            return cmd
        
        if self._control_count % 10 == 0:
            self.get_logger().info(f"VERIFY CHECK: {elapsed:.1f}/{duration:.1f}s")
        
        return cmd
    
    def _analyze_attachment(self) -> bool:
        """
        Analyze collected samples to determine if attached.
        
        Returns True if likely attached, False otherwise.
        """
        if len(self._attachment_check_samples) < 5:
            self.get_logger().warn("VERIFY: Not enough samples")
            return False
        
        if self._pre_backup_handle_center is None or self._pre_backup_handle_area is None:
            self.get_logger().warn("VERIFY: No pre-backup data")
            return False
        
        # Calculate average position and area after backup
        recent_samples = self._attachment_check_samples[-10:]  # Last 10 samples
        avg_cx = sum(s[0] for s in recent_samples) / len(recent_samples)
        avg_cy = sum(s[1] for s in recent_samples) / len(recent_samples)
        avg_area = sum(s[2] for s in recent_samples) / len(recent_samples)
        
        pre_cx, pre_cy = self._pre_backup_handle_center
        pre_area = self._pre_backup_handle_area
        
        # Calculate drift and area change
        drift_x = abs(avg_cx - pre_cx)
        drift_y = abs(avg_cy - pre_cy)
        total_drift = math.sqrt(drift_x**2 + drift_y**2)
        
        area_change = (pre_area - avg_area) / pre_area if pre_area > 0 else 0
        
        drift_threshold = self.get_parameter('attachment_drift_threshold').value
        area_threshold = self.get_parameter('attachment_area_decrease_threshold').value
        
        self.get_logger().info(f"VERIFY: drift={total_drift:.0f}px (threshold={drift_threshold:.0f})")
        self.get_logger().info(f"VERIFY: area_decrease={area_change*100:.1f}% (threshold={area_threshold*100:.0f}%)")
        
        # If handle stayed centered AND area didn't decrease much → ATTACHED
        # If handle drifted away OR area decreased significantly → NOT ATTACHED
        
        is_attached = (total_drift < drift_threshold) and (area_change < area_threshold)
        
        return is_attached
    
    # =========================================================================
    # ATTACHED STATE (Mission complete!)
    # =========================================================================
    
    def _do_attached(self, buoyancy: float) -> Twist:
        """Successfully attached - hold position"""
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        if self._control_count % 40 == 0:
            self.get_logger().info("ATTACHED: Mission complete! Press Y for manual control.")
        
        return cmd
    
    # =========================================================================
    # BACKUP / HOLD / SEARCH
    # =========================================================================
    
    def _do_backup(self, buoyancy: float) -> Twist:
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
        if self._backup_start_time is None:
            self._backup_start_time = now
            self.get_logger().info("BACKUP: Backing up...")
        
<<<<<<< HEAD
        backup_duration = 2.5
        elapsed = now - self._backup_start_time
        
        handle = self._get_handle_detection()
        if handle and elapsed > 0.5:
            self.get_logger().info("Handle found while backing up - realigning")
=======
        duration = 2.5
        elapsed = now - self._backup_start_time
        
        handle = self._get_handle()
        if handle and elapsed > 0.5:
            self.get_logger().info("BACKUP: Handle found - re-align")
>>>>>>> ef6e778 (final version docs)
            self._backup_start_time = None
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN)
            return cmd
        
        if elapsed > duration:
            self.get_logger().info("BACKUP complete - re-align")
            self._backup_start_time = None
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN)
            return cmd
        
<<<<<<< HEAD
        # Back up while centering on box if visible
        box = self._get_box_detection()
=======
        box = self._get_box()
>>>>>>> ef6e778 (final version docs)
        if box:
            center_x = self.get_parameter('image_center_x').value
            center_y = self.get_parameter('image_center_y').value
            error_x = box.center_x - center_x
            error_y = box.center_y - center_y
            
            yaw_sign = self.get_parameter('yaw_sign').value
            heave_sign = self.get_parameter('heave_sign').value
            
            cmd.angular.z = yaw_sign * 0.0008 * error_x
<<<<<<< HEAD
            cmd.linear.z = buoyancy + heave_sign * 0.002 * error_y
=======
            cmd.linear.z = buoyancy + heave_sign * 0.001 * error_y
>>>>>>> ef6e778 (final version docs)
        
        cmd.linear.x = -self.get_parameter('backup_speed').value
        
        return cmd
    
    def _do_hold(self, buoyancy: float) -> Twist:
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        handle = self._get_handle()
        if handle:
            self.get_logger().info("HOLD: Handle detected - ALIGN")
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN)
        
        return cmd
    
    def _do_search(self, buoyancy: float) -> Twist:
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        handle = self._get_handle()
        if handle:
            self.get_logger().info("SEARCH: Handle found - ALIGN")
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN)
            return cmd
        
<<<<<<< HEAD
        box = self._get_box_detection()
        if box:
            self.get_logger().info(">>> Box found - starting ALIGN <<<")
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN_TO_HANDLE)
            return cmd
        
        cmd.angular.z = 0.10
=======
        cmd.angular.z = 0.08
>>>>>>> ef6e778 (final version docs)
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
        
<<<<<<< HEAD
        if debounced(2):  # X
            handle = self._get_handle_detection()
=======
        if debounced(2):  # X - Start mission
            self.get_logger().info("X → Starting docking sequence")
            self._attachment_verified = False
            self._attachment_attempts = 0
            self._aligned = False
            self._align_start_time = None
            
            handle = self._get_handle()
>>>>>>> ef6e778 (final version docs)
            if handle:
                self.set_mode(MissionMode.ALIGN)
            else:
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
            if self._mode not in [MissionMode.MANUAL, MissionMode.ATTACHED]:
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
        
<<<<<<< HEAD
        if mode == MissionMode.HOLD_POSITION:
            self._centered_count = 0
            self._aligned = False
            self._yaw_pid_align.reset()
            self._heave_pid_align.reset()
            self._yaw_pid_approach.reset()
            self._heave_pid_approach.reset()
        elif mode == MissionMode.MANUAL:
            self._centered_count = 0
=======
        if mode in [MissionMode.HOLD_POSITION, MissionMode.MANUAL]:
            self._stable_count = 0
            self._yaw_pid.reset()
            self._heave_pid.reset()
        
        if mode == MissionMode.MANUAL:
>>>>>>> ef6e778 (final version docs)
            self._aligned = False
            self._approach_start_time = None
            self._align_start_time = None
            self._backup_start_time = None
<<<<<<< HEAD
            self._yaw_pid_align.reset()
            self._heave_pid_align.reset()
            self._yaw_pid_approach.reset()
            self._heave_pid_approach.reset()
=======
            self._blind_push_start_time = None
            self._verify_backup_start_time = None
            self._verify_check_start_time = None
>>>>>>> ef6e778 (final version docs)
            self._send_stop_command()
    
    def _status_loop(self):
        box = self._get_box()
        handle = self._get_handle()
        
        box_str = f"BOX:{'Y' if box else 'N'}"
        handle_str = f"HDL:{'Y' if handle else 'N'}"
        
        if handle:
            fill = self._get_handle_fill_ratio(handle) * 100
<<<<<<< HEAD
            handle_str += f"({handle.width * handle.height:.0f})[{fill:.1f}%]"
=======
            handle_str += f"[{fill:.1f}%]"
>>>>>>> ef6e778 (final version docs)
        
        attempt_str = f"Attempt#{self._attachment_attempts}" if self._attachment_attempts > 0 else ""
        attached_str = "✓ATTACHED" if self._attachment_verified else ""
        
        msg = String()
        msg.data = f"{self._mode.name} | {box_str} {handle_str} | {attempt_str} {attached_str}"
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
