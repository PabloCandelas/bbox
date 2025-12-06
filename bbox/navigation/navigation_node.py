#!/usr/bin/env python3
"""
navigation_node.py - ROS2 Navigation Node for BlueROV2
=======================================================

This is the main ROS2 node that connects the navigation library to your
perception system and thrusters.

WHAT THIS NODE DOES:
--------------------
1. SUBSCRIBES to perception topics:
   - ArUco marker detections (from aruco_pool_node)
   - YOLO detections (from bbox_yolo_detection)
   - Depth sensor (from MAVROS)
   - IMU data (from MAVROS)

2. RUNS navigation logic:
   - Localization (where am I?)
   - Path planning (how to get there?)
   - Waypoint navigation (follow the path)
   - Motion control (compute velocities)
   - Visual servoing (precise final approach)

3. PUBLISHES thruster commands:
   - RC override messages to MAVROS

HOW TO RUN:
-----------
    ros2 run bbox navigation_node

    # With parameters:
    ros2 run bbox navigation_node --ros-args -p mode:=search

TOPICS:
-------
Subscriptions:
    /aruco_pool/poses      - PoseArray from ArUco detection
    /aruco_pool/ids        - Int32MultiArray marker IDs
    /yolo/detections       - Detection2DArray from YOLO
    /global_position/rel_alt - Float64 depth
    /imu/data              - Imu orientation
    /joy                   - Joystick for mode switching

Publications:
    /rc/override           - OverrideRCIn thruster commands
    /navigation/status     - String status messages
    /navigation/target     - PoseStamped current target
    /navigation/path       - Path for visualization
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import math
import time
from enum import Enum
from typing import Optional, List

# ROS2 message types
from std_msgs.msg import String, Int32MultiArray, Float64
from geometry_msgs.msg import PoseArray, PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import Imu, Joy
from nav_msgs.msg import Path
from vision_msgs.msg import Detection2DArray
from mavros_msgs.msg import OverrideRCIn

# Navigation library imports
from .transforms import Transforms, Pose3D
from .localization import Localization, LocalizationConfig, MarkerObservation, LocalizationState
from .path_planner import PathPlanner, PlannerConfig, Waypoint
from .waypoint_navigator import WaypointNavigator, AdaptiveNavigator, NavigatorConfig, NavigatorState
from .motion_controller import MotionController, ControllerConfig, VelocityCommand, ControlMode
from .thruster_mapper import ThrusterMapper, ThrusterMapperConfig
from .visual_servoing import VisualServoing, VisualServoingConfig, Detection, ServoingState


class MissionMode(Enum):
    """High-level mission modes."""
    IDLE = 0           # Waiting for commands
    MANUAL = 1         # Manual joystick control
    SEARCH = 2         # Searching for blackbox
    APPROACH = 3       # Approaching blackbox
    ATTACH = 4         # Visual servoing to attach carabiner
    RETURN_HOME = 5    # Returning to surface
    HOLD_POSITION = 6  # Station keeping


class NavigationNode(Node):
    """
    Main ROS2 navigation node for BlueROV2.
    
    This node integrates all navigation components and connects them
    to the ROS2 perception and control systems.
    """
    
    def __init__(self):
        super().__init__('navigation_node')
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("Starting BlueROV2 Navigation Node")
        self.get_logger().info("=" * 50)
        
        # Declare parameters
        self._declare_parameters()
        
        # Initialize navigation components
        self._init_navigation_components()
        
        # Initialize ROS2 interfaces
        self._init_publishers()
        self._init_subscribers()
        self._init_timers()
        
        # State variables
        self._mode = MissionMode.IDLE
        self._blackbox_position: Optional[np.ndarray] = None
        self._camera_tilt_deg = 0.0
        self._armed = False
        
        # Last received data timestamps (for timeout detection)
        self._last_aruco_time = 0.0
        self._last_yolo_time = 0.0
        self._last_depth_time = 0.0
        
        self.get_logger().info("Navigation node initialized successfully!")
        self.get_logger().info(f"Initial mode: {self._mode.name}")
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        # Mode parameter
        self.declare_parameter('mode', 'idle')
        self.declare_parameter('control_rate', 20.0)
        
        # Localization parameters
        self.declare_parameter('position_filter_alpha', 0.3)
        self.declare_parameter('use_depth_sensor', True)
        
        # Motion control parameters
        self.declare_parameter('max_surge', 0.4)
        self.declare_parameter('max_sway', 0.4)
        self.declare_parameter('max_heave', 0.3)
        self.declare_parameter('max_yaw_rate', 0.4)
        
        # Visual servoing parameters
        self.declare_parameter('servoing_enabled', True)
        self.declare_parameter('approach_distance', 1.0)
        
        # Safety parameters
        self.declare_parameter('data_timeout', 2.0)  # seconds
    
    def _init_navigation_components(self):
        """Initialize all navigation library components."""
        
        # --- Localization ---
        loc_config = LocalizationConfig(
            position_filter_alpha=self.get_parameter('position_filter_alpha').value,
            use_depth_sensor=self.get_parameter('use_depth_sensor').value,
        )
        self.localization = Localization(loc_config)
        self.get_logger().info("Localization initialized")
        
        # --- Path Planner ---
        planner_config = PlannerConfig()
        self.planner = PathPlanner(planner_config)
        self.get_logger().info("Path planner initialized")
        
        # --- Waypoint Navigator ---
        nav_config = NavigatorConfig(
            position_tolerance=0.3,
            heading_tolerance=math.radians(15),
        )
        self.navigator = AdaptiveNavigator(nav_config)
        self.navigator.set_on_waypoint_reached(self._on_waypoint_reached)
        self.navigator.set_on_navigation_complete(self._on_navigation_complete)
        self.get_logger().info("Waypoint navigator initialized")
        
        # --- Motion Controller ---
        ctrl_config = ControllerConfig(
            max_surge=self.get_parameter('max_surge').value,
            max_sway=self.get_parameter('max_sway').value,
            max_heave=self.get_parameter('max_heave').value,
            max_yaw_rate=self.get_parameter('max_yaw_rate').value,
        )
        self.controller = MotionController(ctrl_config)
        self.get_logger().info("Motion controller initialized")
        
        # --- Thruster Mapper ---
        self.thruster_mapper = ThrusterMapper()
        self.get_logger().info("Thruster mapper initialized")
        
        # --- Visual Servoing ---
        servo_config = VisualServoingConfig()
        self.visual_servoing = VisualServoing(servo_config)
        self.visual_servoing.set_on_state_change(self._on_servoing_state_change)
        self.visual_servoing.set_on_completed(self._on_attachment_complete)
        self.get_logger().info("Visual servoing initialized")
    
    def _init_publishers(self):
        """Initialize ROS2 publishers."""
        
        # Thruster commands (to MAVROS)
        self.pub_rc_override = self.create_publisher(
            OverrideRCIn, 
            'rc/override', 
            10
        )
        
        # Navigation status (for monitoring)
        self.pub_status = self.create_publisher(
            String,
            'navigation/status',
            10
        )
        
        # Current target pose (for visualization)
        self.pub_target = self.create_publisher(
            PoseStamped,
            'navigation/target',
            10
        )
        
        # Current robot pose estimate
        self.pub_pose = self.create_publisher(
            PoseStamped,
            'navigation/pose',
            10
        )
        
        # Planned path (for visualization)
        self.pub_path = self.create_publisher(
            Path,
            'navigation/path',
            10
        )
        
        self.get_logger().info("Publishers initialized")
    
    def _init_subscribers(self):
        """Initialize ROS2 subscribers."""
        
        # QoS for sensor data (best effort for high-rate data)
        sensor_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        # ArUco marker detections
        self.sub_aruco_poses = self.create_subscription(
            PoseArray,
            'aruco_pool/poses',
            self._aruco_poses_callback,
            10
        )
        
        self.sub_aruco_ids = self.create_subscription(
            Int32MultiArray,
            'aruco_pool/ids',
            self._aruco_ids_callback,
            10
        )
        
        # YOLO detections (for blackbox and handle)
        self.sub_yolo = self.create_subscription(
            Detection2DArray,
            '/yolo/detections',
            self._yolo_callback,
            10
        )
        
        # Depth sensor (from MAVROS)
        self.sub_depth = self.create_subscription(
            Float64,
            'global_position/rel_alt',
            self._depth_callback,
            sensor_qos
        )
        
        # IMU (from MAVROS)
        self.sub_imu = self.create_subscription(
            Imu,
            'imu/data',
            self._imu_callback,
            sensor_qos
        )
        
        # Joystick for mode switching
        self.sub_joy = self.create_subscription(
            Joy,
            'joy',
            self._joy_callback,
            sensor_qos
        )
        
        self.get_logger().info("Subscribers initialized")
        
        # Storage for latest data
        self._latest_aruco_poses: Optional[PoseArray] = None
        self._latest_aruco_ids: Optional[List[int]] = None
        self._latest_yolo_detections: List[Detection] = []
    
    def _init_timers(self):
        """Initialize control loop timer."""
        control_rate = self.get_parameter('control_rate').value
        self.control_timer = self.create_timer(
            1.0 / control_rate,
            self._control_loop
        )
        self.get_logger().info(f"Control loop running at {control_rate} Hz")
    
    # =========================================================================
    # SUBSCRIBER CALLBACKS
    # =========================================================================
    
    def _aruco_poses_callback(self, msg: PoseArray):
        """Handle ArUco pose detections."""
        self._latest_aruco_poses = msg
        self._last_aruco_time = time.time()
    
    def _aruco_ids_callback(self, msg: Int32MultiArray):
        """Handle ArUco marker IDs."""
        self._latest_aruco_ids = list(msg.data)
        
        # Process ArUco data when we have both poses and IDs
        if self._latest_aruco_poses is not None and self._latest_aruco_ids is not None:
            self._process_aruco_detections()
    
    def _process_aruco_detections(self):
        """Convert ArUco detections to MarkerObservations and update localization."""
        if self._latest_aruco_poses is None or self._latest_aruco_ids is None:
            return
        
        observations = []
        
        for i, (pose, marker_id) in enumerate(zip(
            self._latest_aruco_poses.poses, 
            self._latest_aruco_ids
        )):
            # Convert ROS Pose to transform matrix
            # The aruco_pool_node publishes camera → marker transform
            T = self._pose_msg_to_matrix(pose)
            
            obs = MarkerObservation(
                marker_id=marker_id,
                camera_to_marker=T,
                timestamp=time.time(),
                confidence=1.0
            )
            observations.append(obs)
        
        # Update localization
        self.localization.update_markers(observations, self._camera_tilt_deg)
        
        # Clear data after processing
        self._latest_aruco_poses = None
        self._latest_aruco_ids = None
    
    def _yolo_callback(self, msg: Detection2DArray):
        """Handle YOLO detections."""
        self._last_yolo_time = time.time()
        
        self._latest_yolo_detections = []
        
        for det in msg.detections:
            # Extract detection info
            center_x = det.bbox.center.position.x
            center_y = det.bbox.center.position.y
            width = det.bbox.size_x
            height = det.bbox.size_y
            
            # Get class name and confidence
            if det.results:
                class_id = det.results[0].hypothesis.class_id
                confidence = det.results[0].hypothesis.score
            else:
                class_id = 'unknown'
                confidence = 0.0
            
            # Determine orientation
            orientation = 'Horizontal' if width > height else 'Vertical'
            
            detection = Detection(
                class_name=class_id,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
                confidence=confidence,
                orientation=orientation
            )
            self._latest_yolo_detections.append(detection)
            
            # Check if we found the blackbox
            if 'box' in class_id.lower() or 'bbox' in class_id.lower():
                self._on_blackbox_detected(detection)
    
    def _depth_callback(self, msg: Float64):
        """Handle depth sensor data."""
        self._last_depth_time = time.time()
        self.localization.update_depth(msg.data)
    
    def _imu_callback(self, msg: Imu):
        """Handle IMU data."""
        # Extract orientation
        q = msg.orientation
        quat = np.array([q.x, q.y, q.z, q.w])
        roll, pitch, yaw = Transforms.euler_from_quaternion(quat)
        
        # Update localization with IMU
        self.localization.update_imu(roll, pitch, yaw)
    
    def _joy_callback(self, msg: Joy):
        """Handle joystick input for mode switching."""
        # Button mappings (adjust for your controller)
        # Using same buttons as init_joy.py
        btn_manual = msg.buttons[3] if len(msg.buttons) > 3 else 0
        btn_auto = msg.buttons[2] if len(msg.buttons) > 2 else 0
        btn_search = msg.buttons[0] if len(msg.buttons) > 0 else 0
        btn_hold = msg.buttons[1] if len(msg.buttons) > 1 else 0
        
        if btn_manual:
            self.set_mode(MissionMode.MANUAL)
        elif btn_auto:
            self.set_mode(MissionMode.SEARCH)
            print("search mode activated")
        elif btn_hold:
            self.set_mode(MissionMode.HOLD_POSITION)
    
    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================
    
    def _control_loop(self):
        """
        Main control loop - runs at control_rate Hz.
        
        This is where all the navigation logic comes together.
        """
        # Check data timeouts
        if not self._check_data_health():
            self._publish_stop()
            return
        
        # Get current pose from localization
        current_pose = self.localization.get_pose()
        
        if current_pose is None and self._mode not in [MissionMode.IDLE, MissionMode.MANUAL]:
            self.get_logger().warn("No localization - cannot navigate", throttle_duration_sec=2.0)
            self._publish_stop()
            return
        
        # Publish current pose for visualization
        if current_pose is not None:
            self._publish_pose(current_pose)
        
        # Execute mode-specific logic
        velocity = VelocityCommand()
        
        if self._mode == MissionMode.IDLE:
            velocity = VelocityCommand()
            
        elif self._mode == MissionMode.MANUAL:
            # Manual mode - don't output anything, let init_joy handle it
            return
            
        elif self._mode == MissionMode.SEARCH:
            velocity = self._execute_search(current_pose)
            
        elif self._mode == MissionMode.APPROACH:
            velocity = self._execute_approach(current_pose)
            
        elif self._mode == MissionMode.ATTACH:
            velocity = self._execute_attach()
            
        elif self._mode == MissionMode.RETURN_HOME:
            velocity = self._execute_return_home(current_pose)
            
        elif self._mode == MissionMode.HOLD_POSITION:
            velocity = self._execute_hold_position(current_pose)
        
        # Convert velocity to PWM and publish
        self._publish_velocity(velocity)
        
        # Publish status
        self._publish_status()
    
    def _check_data_health(self) -> bool:
        """Check if we're receiving data from sensors."""
        timeout = self.get_parameter('data_timeout').value
        current_time = time.time()
        
        # In certain modes, we need localization data
        if self._mode in [MissionMode.SEARCH, MissionMode.APPROACH, MissionMode.RETURN_HOME]:
            if current_time - self._last_aruco_time > timeout:
                loc_state = self.localization.get_state()
                if loc_state == LocalizationState.LOST:
                    self.get_logger().warn("Localization lost!", throttle_duration_sec=2.0)
                    return False
        
        return True
    
    # =========================================================================
    # MODE EXECUTION
    # =========================================================================
    
    def _execute_search(self, current_pose: Pose3D) -> VelocityCommand:
        """Execute search pattern to find blackbox."""
        
        # Generate search pattern if not already navigating
        if not self.navigator.is_navigating():
            self.get_logger().info("Generating search pattern...")
            search_path = self.planner.generate_search_pattern()
            self.navigator.set_path(search_path)
            self._publish_path(search_path)
        
        # Update navigator
        self.navigator.update(current_pose)
        
        # Get target and compute velocity
        target = self.navigator.get_target_pose()
        if target is not None:
            self.controller.set_target(target)
            self._publish_target(target)
        
        return self.controller.update(current_pose)
    
    def _execute_approach(self, current_pose: Pose3D) -> VelocityCommand:
        """Approach the detected blackbox."""
        
        if self._blackbox_position is None:
            self.get_logger().warn("No blackbox position for approach")
            self.set_mode(MissionMode.SEARCH)
            return VelocityCommand()
        
        # Plan approach path if needed
        if not self.navigator.is_navigating():
            approach_distance = self.get_parameter('approach_distance').value
            approach_path = self.planner.plan_approach(
                current_pose,
                self._blackbox_position,
                approach_distance=approach_distance
            )
            
            if approach_path:
                self.navigator.set_path(approach_path)
                self._publish_path(approach_path)
            else:
                self.get_logger().error("Failed to plan approach path")
                return VelocityCommand()
        
        # Update navigator
        self.navigator.update(current_pose)
        
        # Check if we've reached approach point
        if self.navigator.get_state() == NavigatorState.COMPLETED:
            self.get_logger().info("Approach complete - switching to visual servoing")
            self.set_mode(MissionMode.ATTACH)
            return VelocityCommand()
        
        # Compute velocity to target
        target = self.navigator.get_target_pose()
        if target is not None:
            self.controller.set_target(target)
            self._publish_target(target)
        
        return self.controller.update(current_pose)
    
    def _execute_attach(self) -> VelocityCommand:
        """Execute visual servoing for carabiner attachment."""
        
        if not self.visual_servoing.is_active():
            self.get_logger().info("Starting visual servoing for attachment")
            self.visual_servoing.start()
        
        # Update visual servoing with YOLO detections
        velocity = self.visual_servoing.update(self._latest_yolo_detections)
        
        return velocity
    
    def _execute_return_home(self, current_pose: Pose3D) -> VelocityCommand:
        """Return to home/surface position."""
        
        if not self.navigator.is_navigating():
            home = self.planner.get_home_position()
            home_pose = home.to_pose()
            
            path = self.planner.plan_path(current_pose, home_pose)
            if path:
                self.navigator.set_path(path)
                self._publish_path(path)
        
        self.navigator.update(current_pose)
        
        target = self.navigator.get_target_pose()
        if target is not None:
            self.controller.set_target(target)
        
        return self.controller.update(current_pose)
    
    def _execute_hold_position(self, current_pose: Pose3D) -> VelocityCommand:
        """Hold current position."""
        self.controller.hold_position(current_pose)
        return self.controller.update(current_pose)
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_blackbox_detected(self, detection: Detection):
        """Handle blackbox detection from YOLO."""
        # Only process if we're searching
        if self._mode != MissionMode.SEARCH:
            return
        
        current_pose = self.localization.get_pose()
        if current_pose is None:
            return
        
        # Estimate blackbox position (simplified - assumes it's in front)
        # In a real system, you'd use the detection + depth to triangulate
        distance_estimate = 2.0  # meters (could be computed from bbox size)
        
        yaw = Transforms.euler_from_quaternion(current_pose.quaternion())[2]
        
        blackbox_x = current_pose.x + distance_estimate * math.cos(yaw)
        blackbox_y = current_pose.y + distance_estimate * math.sin(yaw)
        blackbox_z = current_pose.z  # Same depth
        
        self._blackbox_position = np.array([blackbox_x, blackbox_y, blackbox_z])
        
        self.get_logger().info(f"Blackbox detected! Estimated position: {self._blackbox_position}")
        self.set_mode(MissionMode.APPROACH)
    
    def _on_waypoint_reached(self, index: int, waypoint: Waypoint):
        """Handle reaching a waypoint."""
        self.get_logger().info(f"Reached waypoint {index}: {waypoint.position}")
    
    def _on_navigation_complete(self):
        """Handle navigation completion."""
        self.get_logger().info("Navigation path completed")
        
        if self._mode == MissionMode.RETURN_HOME:
            self.get_logger().info("Arrived home!")
            self.set_mode(MissionMode.IDLE)
    
    def _on_servoing_state_change(self, state: ServoingState):
        """Handle visual servoing state changes."""
        self.get_logger().info(f"Visual servoing state: {state.name}")
    
    def _on_attachment_complete(self):
        """Handle successful carabiner attachment."""
        self.get_logger().info("=" * 50)
        self.get_logger().info("ATTACHMENT COMPLETE!")
        self.get_logger().info("=" * 50)
        
        # Switch to return home
        self.set_mode(MissionMode.RETURN_HOME)
    
    # =========================================================================
    # MODE CONTROL
    # =========================================================================
    
    def set_mode(self, mode: MissionMode):
        """Change the mission mode."""
        if mode == self._mode:
            return
        
        old_mode = self._mode
        self._mode = mode
        
        self.get_logger().info(f"Mode changed: {old_mode.name} -> {mode.name}")
        
        # Reset components when changing modes
        if mode == MissionMode.SEARCH:
            self.navigator.clear_path()
            self._blackbox_position = None
            
        elif mode == MissionMode.IDLE:
            self.navigator.clear_path()
            self.controller.disable()
            self.visual_servoing.stop()
            
        elif mode == MissionMode.HOLD_POSITION:
            pose = self.localization.get_pose()
            if pose:
                self.controller.hold_position(pose)
    
    # =========================================================================
    # PUBLISHERS
    # =========================================================================
    
    def _publish_velocity(self, velocity: VelocityCommand):
        """Convert velocity to PWM and publish to MAVROS."""
        pwm = self.thruster_mapper.velocity_to_pwm(velocity)
        
        msg = OverrideRCIn()
        msg.channels = pwm
        self.pub_rc_override.publish(msg)
    
    def _publish_stop(self):
        """Publish stop command (all thrusters neutral)."""
        pwm = self.thruster_mapper.stop()
        msg = OverrideRCIn()
        msg.channels = pwm
        self.pub_rc_override.publish(msg)
    
    def _publish_status(self):
        """Publish navigation status."""
        loc_state = self.localization.get_state()
        nav_state = self.navigator.get_state()
        servo_state = self.visual_servoing.get_state()
        
        status = (f"Mode: {self._mode.name} | "
                  f"Loc: {loc_state.name} | "
                  f"Nav: {nav_state.name} | "
                  f"Servo: {servo_state.name}")
        
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)
    
    def _publish_pose(self, pose: Pose3D):
        """Publish current pose estimate."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'net'
        msg.pose = self._pose3d_to_msg(pose)
        self.pub_pose.publish(msg)
    
    def _publish_target(self, target: Pose3D):
        """Publish current target pose."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'net'
        msg.pose = self._pose3d_to_msg(target)
        self.pub_target.publish(msg)
    
    def _publish_path(self, waypoints: List[Waypoint]):
        """Publish planned path for visualization."""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'net'
        
        for wp in waypoints:
            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = self._pose3d_to_msg(wp.to_pose())
            msg.poses.append(pose_stamped)
        
        self.pub_path.publish(msg)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _pose_msg_to_matrix(self, pose: Pose) -> np.ndarray:
        """Convert ROS Pose message to 4x4 transform matrix."""
        q = np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ])
        R = Transforms.quaternion_to_rotation_matrix(q)
        
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z
        
        return T
    
    def _pose3d_to_msg(self, pose: Pose3D) -> Pose:
        """Convert Pose3D to ROS Pose message."""
        msg = Pose()
        msg.position.x = pose.x
        msg.position.y = pose.y
        msg.position.z = pose.z
        msg.orientation.x = pose.qx
        msg.orientation.y = pose.qy
        msg.orientation.z = pose.qz
        msg.orientation.w = pose.qw
        return msg


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = NavigationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down navigation node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
