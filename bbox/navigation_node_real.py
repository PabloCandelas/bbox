#!/usr/bin/env python3
"""
navigation_node_real_v5.py - ALIGNED STRAIGHT-LINE Approach
============================================================

KEY REQUIREMENTS:
-----------------
1. Keep BOX center and HANDLE center aligned horizontally
2. Approach steadily in a STRAIGHT LINE (no oscillation)
3. If lose BOX but still see HANDLE → continue gentle approach
4. When HANDLE fills screen → we're close enough, HOLD

STRATEGY:
---------
Phase 1: ALIGN - Get handle centered, verify we're facing handle-face
Phase 2: APPROACH - Steady straight-line approach, small corrections only
Phase 3: FINAL - Handle fills screen, hold position for gripper

FIXES FROM v4:
--------------
1. MUCH HIGHER YAW GAIN when error is large (was stuck at err=-154)
2. Progressive gain: small errors = small corrections, large errors = fast correction
3. Alignment verification: ensure handle is near box center (not viewing from side)
4. Handle-fills-screen detection for final approach
5. Straight-line lock: once centered, minimal yaw corrections
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
    ALIGN_TO_HANDLE = 2      # Phase 1: Get centered on handle
    APPROACH_STRAIGHT = 3    # Phase 2: Straight-line approach
    FINAL_APPROACH = 4       # Phase 3: Handle fills screen, very close
    HOLD_POSITION = 5
    BACKUP = 6


class ErrorFilter:
    """Smooths error signals"""
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.error_x_history = deque(maxlen=window_size)
        self.error_y_history = deque(maxlen=window_size)
    
    def update(self, error_x: float, error_y: float) -> tuple:
        self.error_x_history.append(error_x)
        self.error_y_history.append(error_y)
        
        if len(self.error_x_history) == 0:
            return error_x, error_y
        
        smooth_x = sum(self.error_x_history) / len(self.error_x_history)
        smooth_y = sum(self.error_y_history) / len(self.error_y_history)
        
        return smooth_x, smooth_y
    
    def reset(self):
        self.error_x_history.clear()
        self.error_y_history.clear()


class NavigationNodeReal(Node):
    """
    ALIGNED STRAIGHT-LINE Approach Navigation
    
    Key: Keep handle centered, approach in straight line, handle screen-fill detection
    """
    
    def __init__(self):
        super().__init__('navigation_node_real')
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Navigation Node v5 - ALIGNED STRAIGHT-LINE")
        self.get_logger().info("  Strategy: Align → Approach straight → Final approach")
        self.get_logger().info("=" * 60)
        
        self._declare_parameters()
        self._init_state()
        self._init_ros()
        self._print_startup_info()
    
    def _declare_parameters(self):
        self.declare_parameter('control_rate', 20.0)
        
        # === SPEED LIMITS ===
        self.declare_parameter('max_surge', 0.15)        # Steady approach speed
        self.declare_parameter('max_heave', 0.20)
        self.declare_parameter('max_yaw_rate', 0.25)     # INCREASED for faster alignment
        self.declare_parameter('backup_speed', 0.10)
        self.declare_parameter('final_approach_speed', 0.05)  # Very slow final
        
        # === VISUAL SERVOING GAINS ===
        # Progressive gains - higher when error is large
        self.declare_parameter('vs_gain_yaw_base', 0.0012)    # INCREASED base gain
        self.declare_parameter('vs_gain_yaw_boost', 0.0008)   # Extra gain for large errors
        self.declare_parameter('vs_gain_heave', 0.002)
        
        # Error threshold for boost
        self.declare_parameter('large_error_threshold', 100.0)  # Pixels
        
        # Sign inversion (adjust if robot turns wrong way)
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
        self.declare_parameter('align_tolerance_x', 80.0)    # Tighter for alignment phase
        self.declare_parameter('align_tolerance_y', 100.0)
        self.declare_parameter('approach_tolerance_x', 120.0)  # Slightly looser during approach
        self.declare_parameter('approach_tolerance_y', 140.0)
        
        # === STABILITY ===
        self.declare_parameter('stability_count_required', 10)  # More frames for stable alignment
        
        # === HANDLE SIZE THRESHOLDS ===
        # Handle area relative to image for phase transitions
        self.declare_parameter('handle_area_for_approach', 5000.0)   # Start approach
        self.declare_parameter('handle_area_for_final', 20000.0)     # Very close
        self.declare_parameter('handle_fills_screen_ratio', 0.15)    # 15% of image = filled
        
        # === BOX THRESHOLDS ===
        self.declare_parameter('box_close_area', 80000.0)
        self.declare_parameter('box_too_close_area', 400000.0)  # Back up if larger
        
        # === ALIGNMENT CHECK ===
        # Handle should be roughly centered on box (not viewing from side)
        self.declare_parameter('handle_box_offset_tolerance', 200.0)  # Max X offset between centers
        
        # === TIMEOUTS ===
        self.declare_parameter('max_approach_time', 120.0)
        self.declare_parameter('safety_timeout', 5.0)
        self.declare_parameter('align_timeout', 30.0)  # Max time to align
        
        # === STRAIGHT LINE APPROACH ===
        self.declare_parameter('straight_line_yaw_limit', 0.05)  # Max yaw during straight approach
    
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
        self._aligned = False  # True when properly aligned to handle
        
        # Error filtering
        self._handle_error_filter = ErrorFilter(window_size=3)
        
        # Backup state
        self._backup_start_time = None
        
        # Last known handle error (for when handle is lost briefly)
        self._last_handle_error_x = 0.0
        self._last_handle_error_y = 0.0
    
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
        self.get_logger().info("v5 STRATEGY:")
        self.get_logger().info("  1. ALIGN: Center on handle, verify facing handle-face")
        self.get_logger().info("  2. APPROACH: Steady straight-line, minimal corrections")
        self.get_logger().info("  3. FINAL: Handle fills screen → hold for gripper")
        self.get_logger().info("")
        self.get_logger().info("  - If lose BOX but see HANDLE → continue gently")
        self.get_logger().info("  - Higher yaw gain for faster alignment")
        self.get_logger().info("  - Straight-line lock during approach phase")
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
        
        # Store last known error
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
    
    def _handle_fills_screen(self, handle: Detection) -> bool:
        """Check if handle fills significant portion of screen"""
        img_width = self.get_parameter('image_width').value
        img_height = self.get_parameter('image_height').value
        fill_ratio = self.get_parameter('handle_fills_screen_ratio').value
        
        image_area = img_width * img_height
        handle_area = handle.width * handle.height
        
        return (handle_area / image_area) > fill_ratio
    
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
    # PHASE 1: ALIGN TO HANDLE
    # =========================================================================
    
    def _do_align(self, buoyancy: float) -> Twist:
        """
        ALIGN phase: Get handle centered before approaching.
        Uses higher yaw gain for faster alignment.
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
        # Initialize align timer
        if self._align_start_time is None:
            self._align_start_time = now
            self._handle_error_filter.reset()
            self._centered_count = 0
            self.get_logger().info("Starting ALIGN phase...")
        
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
                # Have box but no handle - search for handle
                self.get_logger().info("No handle during ALIGN - searching...")
                cmd.angular.z = 0.08  # Slow rotation to find handle
                self._centered_count = 0
                return cmd
            else:
                self.get_logger().warn("Lost both box and handle - HOLD")
                self.set_mode(MissionMode.HOLD_POSITION)
                return cmd
        
        # Check if handle fills screen (already very close!)
        if self._handle_fills_screen(handle):
            self.get_logger().info("=" * 50)
            self.get_logger().info(">>> HANDLE FILLS SCREEN - FINAL APPROACH <<<")
            self.get_logger().info("=" * 50)
            self.set_mode(MissionMode.FINAL_APPROACH)
            return self._do_final_approach(buoyancy)
        
        # Check box alignment (are we viewing from correct face?)
        if box:
            handle_box_offset = abs(handle.center_x - box.center_x)
            offset_tolerance = self.get_parameter('handle_box_offset_tolerance').value
            
            if handle_box_offset > offset_tolerance:
                if self._control_count % 20 == 0:
                    self.get_logger().warn(
                        f"Handle-Box misaligned: offset={handle_box_offset:.0f}px "
                        f"(tolerance={offset_tolerance:.0f})"
                    )
                # TODO: Could add logic to reposition if viewing from wrong side
        
        # Calculate errors
        center_x = self.get_parameter('image_center_x').value
        center_y = self.get_parameter('image_center_y').value
        
        raw_error_x = handle.center_x - center_x
        raw_error_y = handle.center_y - center_y
        
        error_x, error_y = self._handle_error_filter.update(raw_error_x, raw_error_y)
        
        # Get tolerances for alignment
        tol_x = self.get_parameter('align_tolerance_x').value
        tol_y = self.get_parameter('align_tolerance_y').value
        
        is_centered_x = abs(error_x) < tol_x
        is_centered_y = abs(error_y) < tol_y
        is_centered = is_centered_x and is_centered_y
        
        if is_centered:
            self._centered_count += 1
        else:
            self._centered_count = 0
        
        # PROGRESSIVE YAW GAIN - faster correction for large errors
        gain_yaw_base = self.get_parameter('vs_gain_yaw_base').value
        gain_yaw_boost = self.get_parameter('vs_gain_yaw_boost').value
        large_error_thresh = self.get_parameter('large_error_threshold').value
        
        # Apply boost for large errors
        if abs(error_x) > large_error_thresh:
            gain_yaw = gain_yaw_base + gain_yaw_boost
        else:
            gain_yaw = gain_yaw_base
        
        gain_heave = self.get_parameter('vs_gain_heave').value
        max_yaw = self.get_parameter('max_yaw_rate').value
        max_heave = self.get_parameter('max_heave').value
        yaw_sign = self.get_parameter('yaw_sign').value
        heave_sign = self.get_parameter('heave_sign').value
        
        # Calculate commands
        yaw_cmd = yaw_sign * gain_yaw * error_x
        yaw_cmd = np.clip(yaw_cmd, -max_yaw, max_yaw)
        
        heave_cmd = heave_sign * gain_heave * error_y
        heave_cmd = np.clip(heave_cmd, -max_heave, max_heave)
        
        cmd.angular.z = yaw_cmd
        cmd.linear.z = buoyancy + heave_cmd
        cmd.linear.x = 0.0  # NO forward movement during align
        
        # Check if stable enough to transition to approach
        stability_required = self.get_parameter('stability_count_required').value
        
        if self._centered_count >= stability_required:
            self.get_logger().info("=" * 50)
            self.get_logger().info(">>> ALIGNED - STARTING STRAIGHT APPROACH <<<")
            self.get_logger().info("=" * 50)
            self._aligned = True
            self._approach_start_time = time.time()
            self.set_mode(MissionMode.APPROACH_STRAIGHT)
            return cmd
        
        # Logging
        if self._control_count % 10 == 0:
            cx = "✓" if is_centered_x else " "
            cy = "✓" if is_centered_y else " "
            
            self.get_logger().info(
                f"ALIGN [{cx}{cy}] err=({error_x:+.0f},{error_y:+.0f}) "
                f"yaw={yaw_cmd:+.3f} gain={gain_yaw:.4f} | "
                f"center:{self._centered_count}/{stability_required}"
            )
        
        return cmd
    
    # =========================================================================
    # PHASE 2: STRAIGHT-LINE APPROACH
    # =========================================================================
    
    def _do_approach_straight(self, buoyancy: float) -> Twist:
        """
        STRAIGHT APPROACH phase: 
        - Steady forward movement
        - Minimal yaw corrections (capped)
        - If lose box but see handle → continue gently
        - If handle fills screen → final approach
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        now = time.time()
        
        handle = self._get_handle_detection()
        box = self._get_box_detection()
        
        # Priority: Handle detection
        if not handle:
            # Lost handle
            if box:
                # Have box but lost handle - back up to regain view
                box_area = box.width * box.height
                box_too_close = self.get_parameter('box_too_close_area').value
                
                if box_area > box_too_close:
                    self.get_logger().info("Lost handle, box too close - backing up...")
                    self.set_mode(MissionMode.BACKUP)
                    return self._do_backup(buoyancy)
                else:
                    # Box visible but not huge - realign
                    self.get_logger().info("Lost handle - returning to ALIGN")
                    self._aligned = False
                    self._align_start_time = None
                    self.set_mode(MissionMode.ALIGN_TO_HANDLE)
                    return cmd
            else:
                # Lost both - hold
                self.get_logger().warn("Lost handle and box - HOLD")
                self.set_mode(MissionMode.HOLD_POSITION)
                return cmd
        
        # Handle is visible!
        handle_area = handle.width * handle.height
        
        # Check if handle fills screen
        if self._handle_fills_screen(handle):
            self.get_logger().info("=" * 50)
            self.get_logger().info(">>> HANDLE FILLS SCREEN - FINAL APPROACH <<<")
            self.get_logger().info("=" * 50)
            self.set_mode(MissionMode.FINAL_APPROACH)
            return self._do_final_approach(buoyancy)
        
        # Calculate errors
        center_x = self.get_parameter('image_center_x').value
        center_y = self.get_parameter('image_center_y').value
        
        raw_error_x = handle.center_x - center_x
        raw_error_y = handle.center_y - center_y
        
        error_x, error_y = self._handle_error_filter.update(raw_error_x, raw_error_y)
        
        # Get approach tolerances (slightly looser than align)
        tol_x = self.get_parameter('approach_tolerance_x').value
        tol_y = self.get_parameter('approach_tolerance_y').value
        
        is_centered_x = abs(error_x) < tol_x
        is_centered_y = abs(error_y) < tol_y
        
        # Gains
        gain_yaw_base = self.get_parameter('vs_gain_yaw_base').value
        gain_heave = self.get_parameter('vs_gain_heave').value
        max_heave = self.get_parameter('max_heave').value
        yaw_sign = self.get_parameter('yaw_sign').value
        heave_sign = self.get_parameter('heave_sign').value
        
        # STRAIGHT LINE: Cap yaw to prevent oscillation
        straight_yaw_limit = self.get_parameter('straight_line_yaw_limit').value
        
        yaw_cmd = yaw_sign * gain_yaw_base * error_x
        yaw_cmd = np.clip(yaw_cmd, -straight_yaw_limit, straight_yaw_limit)
        
        heave_cmd = heave_sign * gain_heave * error_y
        heave_cmd = np.clip(heave_cmd, -max_heave, max_heave)
        
        # SURGE: Steady forward, reduce if too off-center
        max_surge = self.get_parameter('max_surge').value
        
        # Calculate target area
        handle_final_area = self.get_parameter('handle_area_for_final').value
        area_ratio = handle_area / handle_final_area
        
        if is_centered_x and is_centered_y:
            # Well centered - full speed
            if area_ratio < 0.3:
                surge_cmd = max_surge
            elif area_ratio < 0.6:
                surge_cmd = max_surge * 0.8
            elif area_ratio < 0.9:
                surge_cmd = max_surge * 0.6
            else:
                surge_cmd = max_surge * 0.4
        else:
            # Off-center - reduce speed, prioritize centering
            surge_cmd = max_surge * 0.3
        
        # If very off-center, go back to align
        if abs(error_x) > tol_x * 2 or abs(error_y) > tol_y * 2:
            self.get_logger().info("Drifted off-center - returning to ALIGN")
            self._aligned = False
            self._align_start_time = None
            self.set_mode(MissionMode.ALIGN_TO_HANDLE)
            return cmd
        
        cmd.angular.z = yaw_cmd
        cmd.linear.z = buoyancy + heave_cmd
        cmd.linear.x = surge_cmd
        
        # Check if box is gone but handle visible (we're very close)
        handle_only_str = ""
        if not box and handle:
            handle_only_str = " [HANDLE-ONLY]"
        
        # Logging
        if self._control_count % 10 == 0:
            cx = "✓" if is_centered_x else " "
            cy = "✓" if is_centered_y else " "
            
            elapsed = now - self._approach_start_time if self._approach_start_time else 0
            
            self.get_logger().info(
                f"APPROACH [{cx}{cy}] err=({error_x:+.0f},{error_y:+.0f}) "
                f"surge={surge_cmd:.2f} yaw={yaw_cmd:+.3f}{handle_only_str} | "
                f"hdl_area={handle_area:.0f} T={elapsed:.0f}s"
            )
        
        return cmd
    
    # =========================================================================
    # PHASE 3: FINAL APPROACH
    # =========================================================================
    
    def _do_final_approach(self, buoyancy: float) -> Twist:
        """
        FINAL APPROACH: Handle fills most of screen.
        Very slow, careful movement. Ready for gripper.
        """
        cmd = Twist()
        cmd.linear.z = buoyancy
        
        handle = self._get_handle_detection()
        
        if not handle:
            self.get_logger().info("Lost handle in final approach - HOLD")
            self.set_mode(MissionMode.HOLD_POSITION)
            return cmd
        
        # Calculate errors
        center_x = self.get_parameter('image_center_x').value
        center_y = self.get_parameter('image_center_y').value
        
        error_x = handle.center_x - center_x
        error_y = handle.center_y - center_y
        
        # Very gentle corrections
        gain_yaw = self.get_parameter('vs_gain_yaw_base').value * 0.5
        gain_heave = self.get_parameter('vs_gain_heave').value * 0.5
        yaw_sign = self.get_parameter('yaw_sign').value
        heave_sign = self.get_parameter('heave_sign').value
        
        yaw_cmd = yaw_sign * gain_yaw * error_x
        yaw_cmd = np.clip(yaw_cmd, -0.03, 0.03)  # Very limited
        
        heave_cmd = heave_sign * gain_heave * error_y
        heave_cmd = np.clip(heave_cmd, -0.1, 0.1)
        
        # Very slow forward or stop
        final_speed = self.get_parameter('final_approach_speed').value
        
        handle_area = handle.width * handle.height
        img_area = self.get_parameter('image_width').value * self.get_parameter('image_height').value
        fill_ratio = handle_area / img_area
        
        if fill_ratio > 0.25:
            # Handle is huge - stop!
            surge_cmd = 0.0
            if self._control_count % 20 == 0:
                self.get_logger().info(">>> HANDLE VERY CLOSE - HOLDING FOR GRIPPER <<<")
        else:
            surge_cmd = final_speed
        
        cmd.angular.z = yaw_cmd
        cmd.linear.z = buoyancy + heave_cmd
        cmd.linear.x = surge_cmd
        
        if self._control_count % 10 == 0:
            self.get_logger().info(
                f"FINAL err=({error_x:+.0f},{error_y:+.0f}) "
                f"surge={surge_cmd:.2f} fill={fill_ratio*100:.1f}%"
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
        
        backup_duration = 2.0  # seconds
        elapsed = now - self._backup_start_time
        
        # Check if handle reappeared
        handle = self._get_handle_detection()
        if handle and elapsed > 0.5:  # Give it at least 0.5s
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
        
        # Back up while staying centered on box if visible
        box = self._get_box_detection()
        if box:
            center_x = self.get_parameter('image_center_x').value
            center_y = self.get_parameter('image_center_y').value
            error_x = box.center_x - center_x
            error_y = box.center_y - center_y
            
            yaw_sign = self.get_parameter('yaw_sign').value
            heave_sign = self.get_parameter('heave_sign').value
            gain_yaw = self.get_parameter('vs_gain_yaw_base').value
            gain_heave = self.get_parameter('vs_gain_heave').value
            
            cmd.angular.z = yaw_sign * gain_yaw * error_x
            cmd.linear.z = buoyancy + heave_sign * gain_heave * error_y
        
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
        
        # Rotate to search
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
        
        if debounced(2):  # X - Start mission
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
        elif mode == MissionMode.MANUAL:
            self._centered_count = 0
            self._aligned = False
            self._approach_start_time = None
            self._align_start_time = None
            self._backup_start_time = None
            self._send_stop_command()
    
    def _status_loop(self):
        box = self._get_box_detection()
        handle = self._get_handle_detection()
        
        box_str = f"BOX:{'Y' if box else 'N'}"
        handle_str = f"HDL:{'Y' if handle else 'N'}"
        
        if box:
            box_str += f"({box.width * box.height:.0f})"
        if handle:
            handle_str += f"({handle.width * handle.height:.0f})"
            
            # Show fill percentage
            img_area = self.get_parameter('image_width').value * self.get_parameter('image_height').value
            fill = (handle.width * handle.height) / img_area * 100
            handle_str += f"[{fill:.1f}%]"
        
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