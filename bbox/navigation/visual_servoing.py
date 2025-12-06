#!/usr/bin/env python3
"""
visual_servoing.py - Visual Servoing for BlueROV2
==================================================

WHAT IS VISUAL SERVOING?
------------------------
Visual servoing (also called "vision-based control") uses camera images
directly to control robot movement. Instead of:

    Position → Controller → Movement

We do:

    Image Features → Controller → Movement

This is especially useful for your mission when:
1. You're close to the blackbox and markers are out of view
2. You need precise alignment with the handle for carabiner attachment
3. GPS/marker-based localization isn't accurate enough

THE "TOO CLOSE TO SEE" PROBLEM:
-------------------------------
When the ROV gets very close to the blackbox:
- ArUco markers on the pool floor go out of camera view
- Normal localization fails
- But you can still see the HANDLE in the camera!

Visual servoing solves this by using the handle position in the image
to guide the robot, not the world position.

HOW IT WORKS:
-------------
1. YOLO detection finds the handle in the image
2. We calculate where the handle is relative to image center
3. We command the robot to move so the handle stays centered
4. When handle is centered and at right size → we're aligned!

IMAGE-BASED VISUAL SERVOING (IBVS):
-----------------------------------
We control based on "image features" - things we can measure in the image:

- Feature position (u, v): Pixel coordinates of the handle
- Feature size: Bounding box area (indicates distance)
- Feature orientation: Handle angle in image

The goal is to make:
- (u, v) = (image_center_x, image_center_y)  → Handle is centered
- size = target_size                          → Robot is at right distance
- orientation = 0                             → Handle is properly aligned

CONTROL LAW:
------------
error_x = target_u - current_u   (horizontal error in image)
error_y = target_v - current_v   (vertical error in image)
error_z = target_size - current_size  (distance error)

velocity_sway = Kp * error_x    (move left/right to center horizontally)
velocity_heave = Kp * error_y   (move up/down to center vertically)
velocity_surge = Kp * error_z   (move forward/back for right distance)

BLIND PUSH BEHAVIOR:
--------------------
At the very end, when the carabiner is about to touch the handle:
1. Handle may go completely out of view
2. We switch to "blind push" - move forward at slow constant speed
3. Push for a set duration to ensure carabiner clips onto handle
4. Success is detected by force feedback (if available) or timeout
"""

import numpy as np
import math
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum
import time

from .transforms import Pose3D, Transforms
from .motion_controller import VelocityCommand


class ServoingState(Enum):
    """States of the visual servoing system."""
    IDLE = 0              # Not active
    SEARCHING = 1         # Looking for target
    CENTERING = 2         # Centering target in image
    APPROACHING = 3       # Moving towards target
    FINAL_APPROACH = 4    # Very close, careful alignment
    BLIND_PUSH = 5        # Target lost, pushing forward blindly
    COMPLETED = 6         # Attachment complete
    FAILED = 7            # Failed to complete


@dataclass
class Detection:
    """
    A detection from the YOLO detector.
    
    Attributes:
        class_name: 'handle', 'bbox', etc.
        center_x: X coordinate of bounding box center (pixels)
        center_y: Y coordinate of bounding box center (pixels)
        width: Bounding box width (pixels)
        height: Bounding box height (pixels)
        confidence: Detection confidence (0-1)
        orientation: 'Horizontal' or 'Vertical' (from your YOLO node)
    """
    class_name: str
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float = 1.0
    orientation: str = 'Unknown'
    
    @property
    def area(self) -> float:
        """Bounding box area in pixels."""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio."""
        return self.width / max(self.height, 1)


@dataclass
class VisualServoingConfig:
    """
    Configuration for visual servoing.
    
    These parameters need to be tuned for your specific setup!
    """
    # Image dimensions (should match your camera)
    image_width: int = 1920
    image_height: int = 1080
    
    # Target position in image (where we want the handle)
    # Default: center of image, slightly below center (for camera tilt)
    target_u: float = 960   # Center horizontally
    target_v: float = 600   # Slightly below center
    
    # Target bounding box area for "correct distance"
    # Tune this based on your handle size and desired approach distance
    target_area: float = 15000  # pixels² when at correct distance
    target_area_tolerance: float = 2000  # ± tolerance
    
    # Minimum detection confidence to use
    min_confidence: float = 0.5
    
    # Control gains (start low, increase carefully!)
    # These map pixel errors to velocity commands
    gain_horizontal: float = 0.002   # pixels → m/s for sway
    gain_vertical: float = 0.002     # pixels → m/s for heave
    gain_distance: float = 0.00005   # area error → m/s for surge
    gain_yaw: float = 0.003          # pixels → rad/s for rotation
    
    # Velocity limits during servoing (slower = safer)
    max_surge: float = 0.15   # m/s
    max_sway: float = 0.15    # m/s
    max_heave: float = 0.1    # m/s
    max_yaw: float = 0.2      # rad/s
    
    # Centering tolerance (how centered is "good enough")
    center_tolerance_x: float = 50   # pixels
    center_tolerance_y: float = 50   # pixels
    
    # Approach parameters
    approach_speed: float = 0.08     # m/s for final approach
    blind_push_speed: float = 0.05   # m/s during blind push
    blind_push_duration: float = 3.0 # seconds to push blindly
    
    # Target loss timeout
    target_lost_timeout: float = 2.0  # seconds before switching to blind push
    
    # Completion criteria
    completion_area: float = 50000   # Handle area when "arrived" (very close)


class VisualServoing:
    """
    Visual servoing controller for precise handle approach.
    
    This class takes over control when the robot is close to the blackbox
    and needs to precisely align with the handle for carabiner attachment.
    
    Usage:
        servoing = VisualServoing()
        
        # Activate when ready for final approach
        servoing.start()
        
        # In control loop:
        detections = get_yolo_detections()  # From your YOLO node
        velocity = servoing.update(detections)
        
        if servoing.get_state() == ServoingState.COMPLETED:
            print("Handle attached!")
    
    Integration with your bbox_yolo_detection.py:
        The detections should come from your YOLO detection node.
        Convert Detection2DArray messages to Detection objects.
    """
    
    def __init__(self, config: Optional[VisualServoingConfig] = None):
        """Initialize visual servoing controller."""
        self.config = config or VisualServoingConfig()
        
        # State
        self._state = ServoingState.IDLE
        self._last_detection: Optional[Detection] = None
        self._last_detection_time: float = 0.0
        
        # Blind push tracking
        self._blind_push_start_time: float = 0.0
        
        # Statistics
        self._start_time: float = 0.0
        self._approach_count: int = 0
        
        # Image center
        self._image_center_x = self.config.image_width / 2
        self._image_center_y = self.config.image_height / 2
        
        # Callbacks
        self._on_state_change: Optional[Callable[[ServoingState], None]] = None
        self._on_completed: Optional[Callable[[], None]] = None
    
    # =========================================================================
    # STATE CONTROL
    # =========================================================================
    
    def start(self) -> None:
        """
        Start visual servoing.
        
        Call this when you want to begin the final approach.
        """
        self._state = ServoingState.SEARCHING
        self._start_time = time.time()
        self._last_detection = None
        self._approach_count = 0
        self._notify_state_change()
    
    def stop(self) -> None:
        """Stop visual servoing."""
        self._state = ServoingState.IDLE
        self._notify_state_change()
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._state = ServoingState.IDLE
        self._last_detection = None
        self._approach_count = 0
    
    def get_state(self) -> ServoingState:
        """Get current state."""
        return self._state
    
    def is_active(self) -> bool:
        """Check if servoing is active."""
        return self._state not in [ServoingState.IDLE, ServoingState.COMPLETED, ServoingState.FAILED]
    
    def _set_state(self, new_state: ServoingState) -> None:
        """Set state and notify."""
        if self._state != new_state:
            self._state = new_state
            self._notify_state_change()
    
    def _notify_state_change(self) -> None:
        """Notify callback of state change."""
        if self._on_state_change:
            self._on_state_change(self._state)
    
    # =========================================================================
    # MAIN UPDATE FUNCTION
    # =========================================================================
    
    def update(self, detections: List[Detection]) -> VelocityCommand:
        """
        Update visual servoing based on current detections.
        
        This is the main function - call it in your control loop.
        
        Args:
            detections: List of Detection objects from YOLO
            
        Returns:
            VelocityCommand for the motion controller
        """
        if self._state == ServoingState.IDLE:
            return VelocityCommand()
        
        if self._state in [ServoingState.COMPLETED, ServoingState.FAILED]:
            return VelocityCommand()
        
        # Find the handle detection
        handle = self._find_handle(detections)
        
        current_time = time.time()
        
        if handle is not None:
            # Handle found!
            self._last_detection = handle
            self._last_detection_time = current_time
            
            # Process based on current state
            if self._state == ServoingState.SEARCHING:
                self._set_state(ServoingState.CENTERING)
            
            if self._state == ServoingState.BLIND_PUSH:
                # Regained sight of handle, go back to approaching
                self._set_state(ServoingState.APPROACHING)
            
            return self._compute_servo_velocity(handle)
        else:
            # Handle not found
            return self._handle_target_lost(current_time)
    
    def _find_handle(self, detections: List[Detection]) -> Optional[Detection]:
        """
        Find the handle detection from the list.
        
        Prioritizes:
        1. Detections with 'handle' in class name
        2. Higher confidence
        3. Larger bounding box (closer = more reliable)
        """
        handle_detections = []
        
        for det in detections:
            # Check if this is a handle
            if 'handle' in det.class_name.lower():
                if det.confidence >= self.config.min_confidence:
                    handle_detections.append(det)
        
        if not handle_detections:
            return None
        
        # Sort by confidence * area (prefer high confidence AND close)
        handle_detections.sort(key=lambda d: d.confidence * d.area, reverse=True)
        
        return handle_detections[0]
    
    def _handle_target_lost(self, current_time: float) -> VelocityCommand:
        """Handle the case when target is not detected."""
        
        time_since_detection = current_time - self._last_detection_time
        
        if self._state == ServoingState.SEARCHING:
            # Still searching, perform search pattern
            return self._search_velocity()
        
        if self._state == ServoingState.BLIND_PUSH:
            # Already in blind push, continue
            return self._blind_push_velocity(current_time)
        
        # Check if we should switch to blind push
        if time_since_detection > self.config.target_lost_timeout:
            if self._last_detection is not None:
                # We had the target before, now lost - might be very close
                if self._last_detection.area > self.config.completion_area * 0.5:
                    # Target was big (close) before losing it
                    self._set_state(ServoingState.BLIND_PUSH)
                    self._blind_push_start_time = current_time
                    return self._blind_push_velocity(current_time)
        
        # Target recently lost, hold position briefly
        return VelocityCommand()
    
    # =========================================================================
    # VELOCITY COMPUTATION
    # =========================================================================
    
    def _compute_servo_velocity(self, handle: Detection) -> VelocityCommand:
        """
        Compute velocity command to servo towards handle.
        
        This is the core visual servoing algorithm.
        """
        # Calculate errors
        error_x = self.config.target_u - handle.center_x  # Positive = move right
        error_y = self.config.target_v - handle.center_y  # Positive = move down
        error_area = self.config.target_area - handle.area  # Positive = move forward
        
        # Check if centered
        is_centered_x = abs(error_x) < self.config.center_tolerance_x
        is_centered_y = abs(error_y) < self.config.center_tolerance_y
        is_at_distance = abs(error_area) < self.config.target_area_tolerance
        
        # Update state based on centering
        if self._state == ServoingState.CENTERING:
            if is_centered_x and is_centered_y:
                self._set_state(ServoingState.APPROACHING)
        
        if self._state == ServoingState.APPROACHING:
            if is_centered_x and is_centered_y and is_at_distance:
                self._set_state(ServoingState.FINAL_APPROACH)
        
        # Check for completion (handle very large = very close)
        if handle.area > self.config.completion_area:
            # We're very close - time for blind push
            self._set_state(ServoingState.BLIND_PUSH)
            self._blind_push_start_time = time.time()
            return self._blind_push_velocity(time.time())
        
        # Compute velocities using proportional control
        # Note: sway moves robot left/right
        #       error_x positive = handle is left of target = move right (positive sway)
        # Note: heave moves robot up/down
        #       error_y positive = handle is above target = move down (positive heave)
        
        sway = self.config.gain_horizontal * error_x
        heave = self.config.gain_vertical * error_y
        surge = self.config.gain_distance * error_area
        
        # Limit velocities
        sway = np.clip(sway, -self.config.max_sway, self.config.max_sway)
        heave = np.clip(heave, -self.config.max_heave, self.config.max_heave)
        surge = np.clip(surge, -self.config.max_surge, self.config.max_surge)
        
        # In final approach, reduce velocities further
        if self._state == ServoingState.FINAL_APPROACH:
            sway *= 0.5
            heave *= 0.5
            surge = self.config.approach_speed  # Constant slow approach
        
        # Only move forward if centered (prioritize centering)
        if not (is_centered_x and is_centered_y):
            surge *= 0.3  # Reduce forward speed if not centered
        
        return VelocityCommand(
            surge=surge,
            sway=sway,
            heave=heave,
            yaw=0.0  # No yaw during visual servoing
        )
    
    def _search_velocity(self) -> VelocityCommand:
        """
        Generate velocity for searching for the target.
        
        Performs a small sweeping motion to find the handle.
        """
        # Simple left-right sweep
        t = time.time() - self._start_time
        yaw = 0.1 * math.sin(t * 0.5)  # Slow sweep
        
        return VelocityCommand(yaw=yaw)
    
    def _blind_push_velocity(self, current_time: float) -> VelocityCommand:
        """
        Generate velocity for blind push phase.
        
        This is the final push to attach the carabiner.
        """
        elapsed = current_time - self._blind_push_start_time
        
        if elapsed > self.config.blind_push_duration:
            # Push complete!
            self._set_state(ServoingState.COMPLETED)
            if self._on_completed:
                self._on_completed()
            return VelocityCommand()
        
        # Constant slow forward push
        return VelocityCommand(surge=self.config.blind_push_speed)
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def set_on_state_change(self, callback: Callable[[ServoingState], None]) -> None:
        """Set callback for state changes."""
        self._on_state_change = callback
    
    def set_on_completed(self, callback: Callable[[], None]) -> None:
        """Set callback for completion."""
        self._on_completed = callback
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_debug_info(self) -> dict:
        """Get debugging information."""
        info = {
            'state': self._state.name,
            'active': self.is_active(),
            'time_active': time.time() - self._start_time if self._start_time > 0 else 0,
        }
        
        if self._last_detection:
            info['last_detection'] = {
                'center': (self._last_detection.center_x, self._last_detection.center_y),
                'area': self._last_detection.area,
                'confidence': self._last_detection.confidence,
            }
            info['error_x'] = self.config.target_u - self._last_detection.center_x
            info['error_y'] = self.config.target_v - self._last_detection.center_y
            info['error_area'] = self.config.target_area - self._last_detection.area
        
        return info
    
    def visualize_target(self, image: np.ndarray) -> np.ndarray:
        """
        Draw visual servoing overlay on image.
        
        Useful for debugging and visualization.
        
        Args:
            image: Input image (will be modified)
            
        Returns:
            Image with overlay
        """
        import cv2
        
        h, w = image.shape[:2]
        
        # Draw target crosshair
        target_u = int(self.config.target_u)
        target_v = int(self.config.target_v)
        
        # Crosshair
        cv2.line(image, (target_u - 30, target_v), (target_u + 30, target_v), (0, 255, 0), 2)
        cv2.line(image, (target_u, target_v - 30), (target_u, target_v + 30), (0, 255, 0), 2)
        
        # Tolerance box
        tol_x = int(self.config.center_tolerance_x)
        tol_y = int(self.config.center_tolerance_y)
        cv2.rectangle(image, 
                      (target_u - tol_x, target_v - tol_y),
                      (target_u + tol_x, target_v + tol_y),
                      (0, 255, 0), 1)
        
        # Draw last detection
        if self._last_detection:
            det = self._last_detection
            cx, cy = int(det.center_x), int(det.center_y)
            
            # Detection center
            cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)
            
            # Line from target to detection
            cv2.line(image, (target_u, target_v), (cx, cy), (255, 0, 0), 2)
            
            # Error text
            error_x = self.config.target_u - det.center_x
            error_y = self.config.target_v - det.center_y
            cv2.putText(image, f"Error: ({error_x:.0f}, {error_y:.0f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # State text
        cv2.putText(image, f"State: {self._state.name}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image


class HandleAlignmentChecker:
    """
    Utility class to check if handle is properly aligned for attachment.
    
    Checks:
    1. Handle orientation (should be horizontal for your carabiner)
    2. Handle is centered
    3. Handle is at correct distance
    4. Handle is stable (not moving much between frames)
    """
    
    def __init__(self, config: Optional[VisualServoingConfig] = None):
        """Initialize alignment checker."""
        self.config = config or VisualServoingConfig()
        self._history: List[Detection] = []
        self._max_history = 10
    
    def update(self, detection: Optional[Detection]) -> None:
        """Add a detection to history."""
        if detection:
            self._history.append(detection)
            if len(self._history) > self._max_history:
                self._history.pop(0)
    
    def is_aligned(self) -> Tuple[bool, dict]:
        """
        Check if handle is aligned for attachment.
        
        Returns:
            Tuple of (is_aligned, details_dict)
        """
        if len(self._history) < 3:
            return False, {'reason': 'Not enough history'}
        
        latest = self._history[-1]
        
        checks = {}
        
        # Check 1: Centered horizontally
        error_x = abs(self.config.target_u - latest.center_x)
        checks['centered_x'] = error_x < self.config.center_tolerance_x
        
        # Check 2: Centered vertically
        error_y = abs(self.config.target_v - latest.center_y)
        checks['centered_y'] = error_y < self.config.center_tolerance_y
        
        # Check 3: At correct distance
        error_area = abs(self.config.target_area - latest.area)
        checks['at_distance'] = error_area < self.config.target_area_tolerance
        
        # Check 4: Handle is horizontal (preferred orientation for your setup)
        checks['horizontal'] = latest.orientation.lower() == 'horizontal'
        
        # Check 5: Stable (not moving much)
        if len(self._history) >= 3:
            recent = self._history[-3:]
            positions = [(d.center_x, d.center_y) for d in recent]
            max_movement = max(
                math.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                          (positions[i][1] - positions[i-1][1])**2)
                for i in range(1, len(positions))
            )
            checks['stable'] = max_movement < 20  # pixels
        else:
            checks['stable'] = False
        
        # All checks must pass
        is_aligned = all(checks.values())
        
        return is_aligned, checks
    
    def clear(self) -> None:
        """Clear history."""
        self._history.clear()


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def detection_from_yolo_msg(detection_2d) -> Detection:
    """
    Convert a vision_msgs/Detection2D message to our Detection format.
    
    This helper function converts ROS messages from your bbox_yolo_detection
    node to the Detection format used by visual servoing.
    
    Usage in your ROS node:
        from vision_msgs.msg import Detection2DArray
        
        def detection_callback(msg: Detection2DArray):
            detections = []
            for det in msg.detections:
                d = detection_from_yolo_msg(det)
                detections.append(d)
            
            velocity = servoing.update(detections)
    """
    # Extract data from Detection2D message
    # Note: This assumes your YOLO node populates these fields
    
    center_x = detection_2d.bbox.center.position.x
    center_y = detection_2d.bbox.center.position.y
    width = detection_2d.bbox.size_x
    height = detection_2d.bbox.size_y
    
    # Get class name and confidence from results
    if detection_2d.results:
        class_id = detection_2d.results[0].hypothesis.class_id
        confidence = detection_2d.results[0].hypothesis.score
    else:
        class_id = 'unknown'
        confidence = 0.0
    
    # Determine orientation from aspect ratio
    orientation = 'Horizontal' if width > height else 'Vertical'
    
    return Detection(
        class_name=class_id,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
        confidence=confidence,
        orientation=orientation
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the visual servoing module."""
    
    print("=" * 60)
    print("VISUAL SERVOING MODULE TEST")
    print("=" * 60)
    
    # Create visual servoing controller
    config = VisualServoingConfig(
        image_width=1920,
        image_height=1080,
        target_area=15000,
    )
    servoing = VisualServoing(config)
    
    # Set up state change callback
    def on_state_change(state):
        print(f"   State changed to: {state.name}")
    
    servoing.set_on_state_change(on_state_change)
    
    # Test 1: Start servoing
    print("\n1. Starting visual servoing...")
    servoing.start()
    
    # Test 2: Simulate handle detection sequence
    print("\n2. Simulating handle detection sequence...")
    
    # Simulate handle moving from left side to center
    test_positions = [
        (400, 500, 50, 30),   # Far left, small
        (600, 520, 80, 50),   # Moving right, bigger
        (800, 550, 120, 70),  # Getting closer
        (900, 580, 150, 90),  # Almost centered
        (960, 600, 180, 100), # Centered!
    ]
    
    for i, (cx, cy, w, h) in enumerate(test_positions):
        det = Detection(
            class_name='handle',
            center_x=cx,
            center_y=cy,
            width=w,
            height=h,
            confidence=0.9,
            orientation='Horizontal' if w > h else 'Vertical'
        )
        
        velocity = servoing.update([det])
        
        print(f"\n   Frame {i+1}: Handle at ({cx}, {cy}), size={w*h}")
        print(f"   Velocity: surge={velocity.surge:.3f}, sway={velocity.sway:.3f}, heave={velocity.heave:.3f}")
        
        time.sleep(0.1)  # Small delay for state transitions
    
    # Test 3: Simulate target loss and blind push
    print("\n3. Simulating target loss (handle out of view)...")
    
    # Make last detection look close
    servoing._last_detection = Detection(
        class_name='handle',
        center_x=960,
        center_y=600,
        width=300,
        height=180,
        confidence=0.9,
        orientation='Horizontal'
    )
    servoing._last_detection_time = time.time() - 3.0  # Pretend lost 3 seconds ago
    
    velocity = servoing.update([])  # No detections
    print(f"   State: {servoing.get_state().name}")
    print(f"   Velocity: surge={velocity.surge:.3f} (blind push)")
    
    # Test 4: Debug info
    print("\n4. Debug Information:")
    debug = servoing.get_debug_info()
    for key, value in debug.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
