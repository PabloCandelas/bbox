"""
Navigation Submodule for BlueROV2 Underwater SAR Project
=========================================================

This submodule contains all navigation-related components for the BlueROV2
underwater Search and Rescue (SAR) system.

Architecture Overview (for beginners):
--------------------------------------

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         PERCEPTION LAYER                            │
    │   (aruco_pool_node.py, bbox_yolo_detection.py - already exists)     │
    │         Detects: ArUco markers, BlackBox, Handle                     │
    └─────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         LOCALIZATION                                 │
    │   transforms.py:   Coordinate frame conversions                      │
    │   localization.py: Fuses markers + depth → smooth world pose         │
    └─────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         PLANNING                                     │
    │   path_planner.py:       Creates obstacle-free paths                 │
    │   waypoint_navigator.py: Sequences waypoints for execution           │
    └─────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         CONTROL                                      │
    │   visual_servoing.py:  Image-based precise alignment                 │
    │   motion_controller.py: Converts poses → velocity commands           │
    │   thruster_mapper.py:   Converts velocities → thruster PWMs          │
    └─────────────────────────────────────────────────────────────────────┘

Coordinate Frames (IMPORTANT - memorize these!):
------------------------------------------------

1. CAMERA FRAME (what the camera sees):
   - Z points FORWARD (out of the lens)
   - Y points DOWN
   - X points RIGHT
   - Origin: at the camera lens

2. BODY/ROV FRAME (robot's perspective):
   - X points FORWARD (nose of the ROV)
   - Z points DOWN
   - Y points RIGHT
   - Origin: center of the ROV

3. WORLD/NET FRAME (pool coordinates):
   - X, Y are horizontal (pool floor plane)
   - Z is depth (positive = deeper)
   - Origin: corner of the pool

The camera is 23cm from the body center and can tilt ±45°.

Usage:
------
    from bbox.navigation import (
        Localization,
        PathPlanner,
        WaypointNavigator,
        MotionController,
        ThrusterMapper,
        VisualServoing,
        Transforms
    )

Author: Your Team
Date: 2025
"""

# Import all navigation components for easy access
from .transforms import Transforms, Pose3D
from .localization import Localization, LocalizationConfig, MarkerObservation, LocalizationState
from .path_planner import PathPlanner, PlannerConfig, Waypoint, Obstacle
from .waypoint_navigator import WaypointNavigator, AdaptiveNavigator, NavigatorConfig, NavigatorState
from .motion_controller import MotionController, ControllerConfig, VelocityCommand, ControlMode
from .thruster_mapper import ThrusterMapper, ThrusterMapperConfig
from .visual_servoing import VisualServoing, VisualServoingConfig, Detection, ServoingState

__all__ = [
    # Core classes
    'Transforms',
    'Pose3D',
    'Localization',
    'LocalizationConfig',
    'MarkerObservation',
    'LocalizationState',
    'PathPlanner',
    'PlannerConfig',
    'Waypoint',
    'Obstacle',
    'WaypointNavigator',
    'AdaptiveNavigator',
    'NavigatorConfig',
    'NavigatorState',
    'MotionController',
    'ControllerConfig',
    'VelocityCommand',
    'ControlMode',
    'ThrusterMapper',
    'ThrusterMapperConfig',
    'VisualServoing',
    'VisualServoingConfig',
    'Detection',
    'ServoingState',
]

# Note: nav_manager_node is a ROS2 node, not imported here
# Run it with: ros2 run bbox nav_manager_node
