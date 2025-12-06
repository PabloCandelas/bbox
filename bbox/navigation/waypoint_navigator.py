#!/usr/bin/env python3
"""
waypoint_navigator.py - Waypoint Navigation for BlueROV2
=========================================================

WHAT IS WAYPOINT NAVIGATION?
----------------------------
A waypoint navigator takes a planned path (list of waypoints) and executes
it step by step. It's like having a list of GPS coordinates and driving
through them one at a time.

Responsibilities:
1. Track which waypoint we're currently heading to
2. Determine when we've "reached" a waypoint (within tolerance)
3. Advance to the next waypoint
4. Handle special actions at waypoints
5. Report progress and status

STATE MACHINE:
--------------
The navigator operates as a state machine:

    IDLE → NAVIGATING → WAYPOINT_REACHED → ... → COMPLETED
                ↓
           PAUSED/ABORTED

States:
- IDLE: No path loaded
- NAVIGATING: Moving towards current waypoint  
- WAYPOINT_REACHED: Just arrived at a waypoint
- PAUSED: Navigation paused by user
- COMPLETED: Reached final waypoint
- ABORTED: Navigation cancelled or failed

TOLERANCES:
-----------
"Reaching" a waypoint doesn't mean hitting it exactly. We use tolerances:
- Position tolerance: How close in meters (e.g., 0.3m)
- Heading tolerance: How aligned in degrees (e.g., 10°)

You can adjust these based on mission requirements:
- Search pattern: Loose tolerances (0.5m)
- Precise approach: Tight tolerances (0.1m)
"""

import numpy as np
import math
from typing import List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time

from .transforms import Pose3D, Transforms
from .path_planner import Waypoint


class NavigatorState(Enum):
    """States of the waypoint navigator."""
    IDLE = 0           # No path loaded, waiting
    NAVIGATING = 1     # Moving towards current waypoint
    WAYPOINT_REACHED = 2  # Just reached a waypoint
    PAUSED = 3         # Navigation paused
    COMPLETED = 4      # All waypoints reached
    ABORTED = 5        # Navigation cancelled


@dataclass
class NavigatorConfig:
    """
    Configuration parameters for the waypoint navigator.
    
    These tolerances define what "reaching a waypoint" means.
    """
    # Position tolerance (meters)
    position_tolerance: float = 0.3
    
    # Heading tolerance (radians)
    heading_tolerance: float = math.radians(15)
    
    # Speed settings
    default_speed: float = 0.3  # m/s
    approach_speed: float = 0.15  # m/s for careful approach
    
    # Timeout settings
    waypoint_timeout: float = 60.0  # seconds to reach a waypoint before warning
    
    # Behavior settings
    stop_at_waypoints: bool = False  # Pause briefly at each waypoint
    waypoint_dwell_time: float = 0.5  # Time to pause if stop_at_waypoints is True
    
    # Accept waypoint even if heading is not aligned yet
    ignore_heading: bool = False


class WaypointNavigator:
    """
    Waypoint navigation controller for BlueROV2.
    
    This class takes a list of waypoints and guides the robot through them.
    It works together with the MotionController to actually move the robot.
    
    Usage:
        navigator = WaypointNavigator()
        
        # Load a path
        navigator.set_path(waypoints)
        
        # In your control loop:
        target = navigator.get_target_pose()
        state = navigator.update(current_pose)
        
        if state == NavigatorState.COMPLETED:
            print("Path completed!")
    
    Typical Integration:
        path = planner.plan_path(start, goal)
        navigator.set_path(path)
        
        while navigator.get_state() != NavigatorState.COMPLETED:
            current_pose = localization.get_pose()
            navigator.update(current_pose)
            target = navigator.get_target_pose()
            
            # Send target to motion controller
            motion_controller.set_target(target)
    """
    
    def __init__(self, config: Optional[NavigatorConfig] = None):
        """Initialize the waypoint navigator."""
        self.config = config or NavigatorConfig()
        
        # Path and state
        self._waypoints: List[Waypoint] = []
        self._current_index: int = 0
        self._state: NavigatorState = NavigatorState.IDLE
        
        # Timing
        self._navigation_start_time: float = 0.0
        self._waypoint_start_time: float = 0.0
        self._waypoint_reached_time: float = 0.0
        
        # Callbacks
        self._on_waypoint_reached: Optional[Callable[[int, Waypoint], None]] = None
        self._on_navigation_complete: Optional[Callable[[], None]] = None
        
        # Progress tracking
        self._total_distance: float = 0.0
        self._distance_traveled: float = 0.0
        self._last_position: Optional[np.ndarray] = None
    
    # =========================================================================
    # PATH MANAGEMENT
    # =========================================================================
    
    def set_path(self, waypoints: List[Waypoint]) -> bool:
        """
        Load a new path to follow.
        
        Args:
            waypoints: List of Waypoint objects to follow
            
        Returns:
            True if path was loaded successfully
        """
        if not waypoints:
            self._state = NavigatorState.IDLE
            return False
        
        self._waypoints = waypoints.copy()
        self._current_index = 0
        self._state = NavigatorState.NAVIGATING
        
        self._navigation_start_time = time.time()
        self._waypoint_start_time = time.time()
        
        # Calculate total path distance
        self._total_distance = self._calculate_path_distance(waypoints)
        self._distance_traveled = 0.0
        self._last_position = None
        
        return True
    
    def clear_path(self) -> None:
        """Clear the current path."""
        self._waypoints = []
        self._current_index = 0
        self._state = NavigatorState.IDLE
    
    def _calculate_path_distance(self, waypoints: List[Waypoint]) -> float:
        """Calculate total distance of a path."""
        if len(waypoints) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(waypoints) - 1):
            dist = np.linalg.norm(waypoints[i+1].position - waypoints[i].position)
            total += dist
        return total
    
    # =========================================================================
    # NAVIGATION CONTROL
    # =========================================================================
    
    def update(self, current_pose: Pose3D) -> NavigatorState:
        """
        Update navigation state based on current robot pose.
        
        Call this in your main control loop with the robot's current pose.
        
        Args:
            current_pose: Current pose from localization
            
        Returns:
            Current navigator state
        """
        if self._state not in [NavigatorState.NAVIGATING, NavigatorState.WAYPOINT_REACHED]:
            return self._state
        
        # Track distance traveled
        if self._last_position is not None:
            self._distance_traveled += np.linalg.norm(
                current_pose.position() - self._last_position
            )
        self._last_position = current_pose.position().copy()
        
        # Handle waypoint reached state (brief pause)
        if self._state == NavigatorState.WAYPOINT_REACHED:
            if self.config.stop_at_waypoints:
                if time.time() - self._waypoint_reached_time < self.config.waypoint_dwell_time:
                    return self._state
            
            # Advance to next waypoint
            self._advance_waypoint()
            return self._state
        
        # Check if we've reached current waypoint
        target_waypoint = self._waypoints[self._current_index]
        
        if self._is_waypoint_reached(current_pose, target_waypoint):
            self._handle_waypoint_reached()
        
        return self._state
    
    def _is_waypoint_reached(self, current_pose: Pose3D, waypoint: Waypoint) -> bool:
        """
        Check if we've reached the target waypoint.
        
        A waypoint is considered "reached" when:
        1. Position is within tolerance
        2. Heading is within tolerance (optional)
        """
        # Position check
        current_pos = current_pose.position()
        target_pos = waypoint.position
        
        distance = np.linalg.norm(current_pos - target_pos)
        
        if distance > self.config.position_tolerance:
            return False
        
        # Heading check (if required and specified)
        if not self.config.ignore_heading and waypoint.yaw is not None:
            current_yaw = Transforms.euler_from_quaternion(current_pose.quaternion())[2]
            heading_error = abs(self._angle_diff(current_yaw, waypoint.yaw))
            
            if heading_error > self.config.heading_tolerance:
                return False
        
        return True
    
    def _handle_waypoint_reached(self) -> None:
        """Handle reaching a waypoint."""
        waypoint = self._waypoints[self._current_index]
        
        self._state = NavigatorState.WAYPOINT_REACHED
        self._waypoint_reached_time = time.time()
        
        # Call callback if registered
        if self._on_waypoint_reached:
            self._on_waypoint_reached(self._current_index, waypoint)
    
    def _advance_waypoint(self) -> None:
        """Move to the next waypoint."""
        self._current_index += 1
        
        if self._current_index >= len(self._waypoints):
            # Path completed
            self._state = NavigatorState.COMPLETED
            if self._on_navigation_complete:
                self._on_navigation_complete()
        else:
            # Continue to next waypoint
            self._state = NavigatorState.NAVIGATING
            self._waypoint_start_time = time.time()
    
    def pause(self) -> None:
        """Pause navigation."""
        if self._state == NavigatorState.NAVIGATING:
            self._state = NavigatorState.PAUSED
    
    def resume(self) -> None:
        """Resume paused navigation."""
        if self._state == NavigatorState.PAUSED:
            self._state = NavigatorState.NAVIGATING
            self._waypoint_start_time = time.time()  # Reset timeout
    
    def abort(self) -> None:
        """Abort navigation."""
        self._state = NavigatorState.ABORTED
    
    def skip_waypoint(self) -> bool:
        """
        Skip the current waypoint and move to next.
        
        Useful if a waypoint is unreachable.
        
        Returns:
            True if skipped, False if no more waypoints
        """
        if self._current_index < len(self._waypoints) - 1:
            self._current_index += 1
            self._waypoint_start_time = time.time()
            return True
        return False
    
    # =========================================================================
    # GETTERS
    # =========================================================================
    
    def get_state(self) -> NavigatorState:
        """Get current navigator state."""
        return self._state
    
    def get_target_pose(self) -> Optional[Pose3D]:
        """
        Get the current target pose.
        
        Returns:
            Target Pose3D if navigating, None otherwise
        """
        if not self._waypoints or self._current_index >= len(self._waypoints):
            return None
        
        return self._waypoints[self._current_index].to_pose()
    
    def get_target_waypoint(self) -> Optional[Waypoint]:
        """Get the current target waypoint."""
        if not self._waypoints or self._current_index >= len(self._waypoints):
            return None
        return self._waypoints[self._current_index]
    
    def get_current_index(self) -> int:
        """Get index of current target waypoint."""
        return self._current_index
    
    def get_total_waypoints(self) -> int:
        """Get total number of waypoints."""
        return len(self._waypoints)
    
    def get_remaining_waypoints(self) -> int:
        """Get number of remaining waypoints."""
        return max(0, len(self._waypoints) - self._current_index)
    
    def get_progress(self) -> float:
        """
        Get navigation progress as percentage (0-100).
        
        Returns:
            Progress percentage based on waypoints completed
        """
        if not self._waypoints:
            return 0.0
        
        # Waypoint-based progress
        return 100.0 * self._current_index / len(self._waypoints)
    
    def get_distance_to_target(self, current_pose: Pose3D) -> float:
        """Get distance to current target waypoint."""
        target = self.get_target_waypoint()
        if target is None:
            return 0.0
        return np.linalg.norm(current_pose.position() - target.position)
    
    def get_heading_to_target(self, current_pose: Pose3D) -> float:
        """
        Get heading angle towards current target.
        
        Returns:
            Angle in radians
        """
        target = self.get_target_waypoint()
        if target is None:
            return 0.0
        
        current_pos = current_pose.position()
        target_pos = target.position
        
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        return math.atan2(dy, dx)
    
    def is_complete(self) -> bool:
        """Check if navigation is complete."""
        return self._state == NavigatorState.COMPLETED
    
    def is_navigating(self) -> bool:
        """Check if actively navigating."""
        return self._state in [NavigatorState.NAVIGATING, NavigatorState.WAYPOINT_REACHED]
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def set_on_waypoint_reached(self, callback: Callable[[int, Waypoint], None]) -> None:
        """
        Set callback for when a waypoint is reached.
        
        Args:
            callback: Function(index, waypoint) called when reaching each waypoint
        """
        self._on_waypoint_reached = callback
    
    def set_on_navigation_complete(self, callback: Callable[[], None]) -> None:
        """
        Set callback for when navigation is complete.
        
        Args:
            callback: Function() called when all waypoints are reached
        """
        self._on_navigation_complete = callback
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    @staticmethod
    def _angle_diff(angle1: float, angle2: float) -> float:
        """
        Calculate the shortest difference between two angles.
        
        Args:
            angle1, angle2: Angles in radians
            
        Returns:
            Shortest angular difference in radians (-π to π)
        """
        diff = angle2 - angle1
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    def get_debug_info(self) -> dict:
        """Get debugging information."""
        return {
            'state': self._state.name,
            'current_index': self._current_index,
            'total_waypoints': len(self._waypoints),
            'progress': self.get_progress(),
            'time_on_waypoint': time.time() - self._waypoint_start_time,
            'total_time': time.time() - self._navigation_start_time if self._navigation_start_time > 0 else 0,
            'distance_traveled': self._distance_traveled,
        }


class AdaptiveNavigator(WaypointNavigator):
    """
    Enhanced navigator with adaptive behaviors.
    
    Features:
    - Adjusts tolerances based on waypoint type
    - Slows down for precise waypoints
    - Handles special actions at waypoints
    """
    
    def __init__(self, config: Optional[NavigatorConfig] = None):
        super().__init__(config)
        
        # Adaptive tolerances for different waypoint types
        self.tolerance_presets = {
            'search': {'position': 0.5, 'heading': math.radians(20)},
            'approach': {'position': 0.15, 'heading': math.radians(5)},
            'approach_ready': {'position': 0.2, 'heading': math.radians(10)},
            'home': {'position': 0.3, 'heading': math.radians(15)},
            'default': {'position': 0.3, 'heading': math.radians(15)},
        }
    
    def _is_waypoint_reached(self, current_pose: Pose3D, waypoint: Waypoint) -> bool:
        """Override to use adaptive tolerances based on waypoint action."""
        # Get tolerances for this waypoint type
        preset_key = waypoint.action if waypoint.action in self.tolerance_presets else 'default'
        tolerances = self.tolerance_presets[preset_key]
        
        # Position check
        distance = np.linalg.norm(current_pose.position() - waypoint.position)
        if distance > tolerances['position']:
            return False
        
        # Heading check
        if not self.config.ignore_heading and waypoint.yaw is not None:
            current_yaw = Transforms.euler_from_quaternion(current_pose.quaternion())[2]
            heading_error = abs(self._angle_diff(current_yaw, waypoint.yaw))
            if heading_error > tolerances['heading']:
                return False
        
        return True
    
    def get_recommended_speed(self) -> float:
        """
        Get recommended speed for current waypoint.
        
        Returns slower speed for precise waypoints.
        """
        waypoint = self.get_target_waypoint()
        if waypoint is None:
            return self.config.default_speed
        
        if waypoint.action in ['approach', 'approach_ready']:
            return self.config.approach_speed
        
        return waypoint.speed if waypoint.speed > 0 else self.config.default_speed


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the waypoint navigator module."""
    
    print("=" * 60)
    print("WAYPOINT NAVIGATOR MODULE TEST")
    print("=" * 60)
    
    # Create navigator
    config = NavigatorConfig(
        position_tolerance=0.3,
        heading_tolerance=math.radians(15)
    )
    navigator = AdaptiveNavigator(config)
    
    # Create test waypoints
    waypoints = [
        Waypoint(position=np.array([1, 1, 2]), yaw=0.0, action='search'),
        Waypoint(position=np.array([3, 1, 2]), yaw=0.0, action='search'),
        Waypoint(position=np.array([3, 3, 2]), yaw=math.pi/2, action='search'),
        Waypoint(position=np.array([5, 5, 2]), yaw=math.pi, action='approach_ready'),
    ]
    
    # Set up callbacks
    def on_waypoint(idx, wp):
        print(f"   → Reached waypoint {idx}: {wp.position}")
    
    def on_complete():
        print("   → Navigation complete!")
    
    navigator.set_on_waypoint_reached(on_waypoint)
    navigator.set_on_navigation_complete(on_complete)
    
    # Load path
    print("\n1. Loading path...")
    navigator.set_path(waypoints)
    print(f"   Total waypoints: {navigator.get_total_waypoints()}")
    print(f"   State: {navigator.get_state().name}")
    
    # Simulate navigation
    print("\n2. Simulating navigation...")
    
    # Simulate reaching each waypoint
    for i, wp in enumerate(waypoints):
        # Create pose at waypoint position
        q = Transforms.quaternion_from_euler(0, 0, wp.yaw if wp.yaw else 0)
        pose = Pose3D(
            x=wp.position[0], y=wp.position[1], z=wp.position[2],
            qx=q[0], qy=q[1], qz=q[2], qw=q[3]
        )
        
        # Update navigator
        state = navigator.update(pose)
        
        # Process WAYPOINT_REACHED state
        if state == NavigatorState.WAYPOINT_REACHED:
            state = navigator.update(pose)
    
    print(f"\n   Final state: {navigator.get_state().name}")
    print(f"   Progress: {navigator.get_progress():.1f}%")
    
    # Test debug info
    print("\n3. Debug Info:")
    debug = navigator.get_debug_info()
    for key, value in debug.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
