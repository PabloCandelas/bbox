#!/usr/bin/env python3
"""
motion_controller.py - Motion Control for BlueROV2
===================================================

WHAT IS A MOTION CONTROLLER?
----------------------------
The motion controller takes a TARGET POSE (where you want to be) and the
CURRENT POSE (where you are), then calculates VELOCITY COMMANDS to move
from current to target.

It's the bridge between:
    Planning (where to go) → Control (how to move) → Thrusters (actually moving)

CONTROL LOOP (simplified):
    1. Navigator says: "Go to position (5, 3, 2)"
    2. Localization says: "You're at position (2, 1, 2)"
    3. Motion Controller calculates: "Move at 0.3 m/s forward, 0.2 m/s right"
    4. Thruster Mapper converts: "Set thruster PWMs to [1550, 1450, ...]"
    5. Repeat at 20-50 Hz

PID CONTROL:
------------
We use PID (Proportional-Integral-Derivative) controllers. Here's what each means:

P (Proportional): How hard to push based on how far off you are
    - Far from target → Push hard
    - Close to target → Push gently
    - Problem: May overshoot or oscillate

I (Integral): Accumulated error over time
    - If you've been off-target for a while, push harder
    - Helps eliminate steady-state error
    - Problem: Can cause overshoot if too high (windup)

D (Derivative): Rate of change of error
    - If error is decreasing fast, ease off to prevent overshoot
    - Acts as a "damper" to smooth motion
    - Problem: Amplifies noise

A well-tuned PID controller balances all three terms.

6 DEGREES OF FREEDOM (6-DOF):
-----------------------------
Your BlueROV can move in 6 ways:

LINEAR (Translation):
- Surge (X): Forward/Backward
- Sway (Y): Left/Right  
- Heave (Z): Up/Down

ANGULAR (Rotation):
- Roll: Tilt left/right (rarely controlled)
- Pitch: Tilt nose up/down (rarely controlled)
- Yaw: Turn left/right

Typically we control: Surge, Sway, Heave, Yaw
Roll and Pitch are often passively stable (ROV naturally levels out)
"""

import numpy as np
import math
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .transforms import Pose3D, Transforms


class ControlMode(Enum):
    """Different control modes."""
    DISABLED = 0        # No control output
    POSITION = 1        # Control to target position
    VELOCITY = 2        # Direct velocity control
    HOLD_POSITION = 3   # Stay at current position


@dataclass
class PIDGains:
    """
    PID controller gains.
    
    Start with these guidelines and tune:
    - P: Start low (0.5), increase until responsive but not oscillating
    - I: Start at 0, increase slowly if there's steady-state error
    - D: Start at 0, increase if there's overshoot
    
    Attributes:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        max_integral: Limit on integral term to prevent windup
        output_limit: Maximum controller output
    """
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    max_integral: float = 1.0
    output_limit: float = 1.0


@dataclass
class ControllerConfig:
    """Configuration for the motion controller."""
    
    # Control loop rate (Hz)
    control_rate: float = 20.0
    
    # Linear velocity limits (m/s)
    max_surge: float = 0.5    # Forward/backward
    max_sway: float = 0.5     # Left/right
    max_heave: float = 0.3    # Up/down
    
    # Angular velocity limits (rad/s)
    max_yaw_rate: float = 0.5  # Turn rate
    
    # Default PID gains (tune these for your robot!)
    surge_gains: PIDGains = None
    sway_gains: PIDGains = None
    heave_gains: PIDGains = None
    yaw_gains: PIDGains = None
    
    # Position tolerances (when to stop controlling)
    position_deadband: float = 0.05  # meters
    yaw_deadband: float = math.radians(2)  # radians
    
    def __post_init__(self):
        """Set default gains if not provided."""
        if self.surge_gains is None:
            self.surge_gains = PIDGains(kp=0.8, ki=0.05, kd=0.1)
        if self.sway_gains is None:
            self.sway_gains = PIDGains(kp=0.8, ki=0.05, kd=0.1)
        if self.heave_gains is None:
            self.heave_gains = PIDGains(kp=1.0, ki=0.1, kd=0.15)
        if self.yaw_gains is None:
            self.yaw_gains = PIDGains(kp=1.5, ki=0.05, kd=0.2)


@dataclass
class VelocityCommand:
    """
    Velocity command output from motion controller.
    
    All values in body frame:
    - surge: Forward (positive) / Backward (negative) m/s
    - sway: Right (positive) / Left (negative) m/s
    - heave: Down (positive) / Up (negative) m/s
    - yaw: Turn right (positive) / Turn left (negative) rad/s
    """
    surge: float = 0.0
    sway: float = 0.0
    heave: float = 0.0
    yaw: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Return as [surge, sway, heave, yaw] array."""
        return np.array([self.surge, self.sway, self.heave, self.yaw])
    
    def is_zero(self, threshold: float = 0.01) -> bool:
        """Check if command is essentially zero."""
        return (abs(self.surge) < threshold and 
                abs(self.sway) < threshold and
                abs(self.heave) < threshold and
                abs(self.yaw) < threshold)


class PIDController:
    """
    Single-axis PID controller.
    
    This is a basic implementation of a PID controller for one degree of freedom.
    
    Usage:
        pid = PIDController(gains)
        
        # In control loop:
        output = pid.compute(error, dt)
    """
    
    def __init__(self, gains: PIDGains):
        """Initialize PID controller with gains."""
        self.gains = gains
        
        # Internal state
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False
    
    def compute(self, error: float, dt: float) -> float:
        """
        Compute PID output.
        
        Args:
            error: Current error (target - current)
            dt: Time step in seconds
            
        Returns:
            Control output
        """
        if dt <= 0:
            return 0.0
        
        # Proportional term
        p_term = self.gains.kp * error
        
        # Integral term (with anti-windup)
        self._integral += error * dt
        self._integral = np.clip(self._integral, 
                                 -self.gains.max_integral, 
                                 self.gains.max_integral)
        i_term = self.gains.ki * self._integral
        
        # Derivative term
        if self._initialized:
            derivative = (error - self._prev_error) / dt
        else:
            derivative = 0.0
            self._initialized = True
        d_term = self.gains.kd * derivative
        
        self._prev_error = error
        
        # Sum and limit output
        output = p_term + i_term + d_term
        output = np.clip(output, -self.gains.output_limit, self.gains.output_limit)
        
        return output
    
    def reset(self):
        """Reset controller state (call when switching modes)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False


class MotionController:
    """
    Main motion controller for BlueROV2.
    
    Converts target poses to velocity commands using PID control.
    
    COORDINATE FRAMES:
    - Input target pose: World frame (from planning)
    - Input current pose: World frame (from localization)
    - Output velocities: Body frame (for thruster mapper)
    
    Usage:
        controller = MotionController()
        
        # Set target
        controller.set_target(target_pose)
        
        # In control loop:
        velocity = controller.update(current_pose)
        
        # Send velocity to thruster mapper
        thruster_mapper.set_velocity(velocity)
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        """Initialize motion controller."""
        self.config = config or ControllerConfig()
        
        # Current state
        self._mode = ControlMode.DISABLED
        self._target_pose: Optional[Pose3D] = None
        self._hold_pose: Optional[Pose3D] = None
        
        # PID controllers for each axis
        self._surge_pid = PIDController(self.config.surge_gains)
        self._sway_pid = PIDController(self.config.sway_gains)
        self._heave_pid = PIDController(self.config.heave_gains)
        self._yaw_pid = PIDController(self.config.yaw_gains)
        
        # Timing
        self._last_update_time = 0.0
        
        # For monitoring/debugging
        self._last_errors = {'surge': 0.0, 'sway': 0.0, 'heave': 0.0, 'yaw': 0.0}
    
    # =========================================================================
    # MODE CONTROL
    # =========================================================================
    
    def enable(self) -> None:
        """Enable position control mode."""
        if self._target_pose is not None:
            self._mode = ControlMode.POSITION
        else:
            self._mode = ControlMode.HOLD_POSITION
    
    def disable(self) -> None:
        """Disable controller (output zero velocities)."""
        self._mode = ControlMode.DISABLED
        self._reset_pids()
    
    def hold_position(self, pose: Pose3D) -> None:
        """
        Switch to hold position mode.
        
        The robot will try to stay at the given position.
        """
        self._mode = ControlMode.HOLD_POSITION
        self._hold_pose = pose
        self._reset_pids()
    
    def _reset_pids(self) -> None:
        """Reset all PID controllers."""
        self._surge_pid.reset()
        self._sway_pid.reset()
        self._heave_pid.reset()
        self._yaw_pid.reset()
    
    # =========================================================================
    # TARGET MANAGEMENT
    # =========================================================================
    
    def set_target(self, target: Pose3D) -> None:
        """
        Set target pose (in world frame).
        
        Args:
            target: Target Pose3D to move towards
        """
        self._target_pose = target
        if self._mode == ControlMode.DISABLED:
            self._mode = ControlMode.POSITION
    
    def set_velocity_direct(self, velocity: VelocityCommand) -> None:
        """
        Set velocity directly (bypasses PID control).
        
        Useful for manual control or visual servoing.
        
        Args:
            velocity: Direct velocity command
        """
        self._mode = ControlMode.VELOCITY
        self._direct_velocity = velocity
    
    def get_target(self) -> Optional[Pose3D]:
        """Get current target pose."""
        return self._target_pose
    
    # =========================================================================
    # MAIN UPDATE FUNCTION
    # =========================================================================
    
    def update(self, current_pose: Pose3D) -> VelocityCommand:
        """
        Compute velocity command to move towards target.
        
        This is the main function - call it in your control loop.
        
        Args:
            current_pose: Current robot pose from localization (world frame)
            
        Returns:
            VelocityCommand in body frame
        """
        current_time = time.time()
        dt = current_time - self._last_update_time if self._last_update_time > 0 else 0.05
        self._last_update_time = current_time
        
        # Limit dt to prevent huge jumps
        dt = min(dt, 0.2)
        
        if self._mode == ControlMode.DISABLED:
            return VelocityCommand()
        
        if self._mode == ControlMode.VELOCITY:
            return self._direct_velocity
        
        # Get target pose
        if self._mode == ControlMode.HOLD_POSITION:
            target = self._hold_pose
        else:
            target = self._target_pose
        
        if target is None:
            return VelocityCommand()
        
        # Compute velocity command
        return self._compute_control(current_pose, target, dt)
    
    def _compute_control(self, 
                         current: Pose3D, 
                         target: Pose3D, 
                         dt: float) -> VelocityCommand:
        """
        Compute PID control output.
        
        Steps:
        1. Calculate errors in world frame
        2. Transform XY errors to body frame
        3. Apply PID control to each axis
        4. Limit outputs
        """
        # === STEP 1: Calculate errors in WORLD frame ===
        
        # Position errors
        error_x = target.x - current.x
        error_y = target.y - current.y
        error_z = target.z - current.z
        
        # Yaw error (need to handle angle wraparound)
        current_yaw = Transforms.euler_from_quaternion(current.quaternion())[2]
        target_yaw = Transforms.euler_from_quaternion(target.quaternion())[2]
        error_yaw = self._angle_diff(current_yaw, target_yaw)
        
        # === STEP 2: Transform XY errors to BODY frame ===
        #
        # World X and Y need to be rotated into body surge and sway
        # based on current heading.
        #
        # If robot is facing north (yaw=0):
        #   World +X = Body surge
        #   World +Y = Body sway
        #
        # If robot is facing east (yaw=90°):
        #   World +X = Body sway (negative)
        #   World +Y = Body surge
        
        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)
        
        # Rotate world error to body frame
        error_surge = error_x * cos_yaw + error_y * sin_yaw
        error_sway = -error_x * sin_yaw + error_y * cos_yaw
        error_heave = error_z  # Z is same in both frames
        
        # Store errors for debugging
        self._last_errors = {
            'surge': error_surge,
            'sway': error_sway,
            'heave': error_heave,
            'yaw': error_yaw
        }
        
        # === STEP 3: Apply deadband ===
        # Don't control if error is very small (prevents jitter)
        
        if abs(error_surge) < self.config.position_deadband:
            error_surge = 0.0
        if abs(error_sway) < self.config.position_deadband:
            error_sway = 0.0
        if abs(error_heave) < self.config.position_deadband:
            error_heave = 0.0
        if abs(error_yaw) < self.config.yaw_deadband:
            error_yaw = 0.0
        
        # === STEP 4: Apply PID control ===
        
        surge_cmd = self._surge_pid.compute(error_surge, dt)
        sway_cmd = self._sway_pid.compute(error_sway, dt)
        heave_cmd = self._heave_pid.compute(error_heave, dt)
        yaw_cmd = self._yaw_pid.compute(error_yaw, dt)
        
        # === STEP 5: Limit outputs ===
        
        surge_cmd = np.clip(surge_cmd, -self.config.max_surge, self.config.max_surge)
        sway_cmd = np.clip(sway_cmd, -self.config.max_sway, self.config.max_sway)
        heave_cmd = np.clip(heave_cmd, -self.config.max_heave, self.config.max_heave)
        yaw_cmd = np.clip(yaw_cmd, -self.config.max_yaw_rate, self.config.max_yaw_rate)
        
        return VelocityCommand(
            surge=surge_cmd,
            sway=sway_cmd,
            heave=heave_cmd,
            yaw=yaw_cmd
        )
    
    @staticmethod
    def _angle_diff(angle1: float, angle2: float) -> float:
        """Calculate shortest angle difference (handles wraparound)."""
        diff = angle2 - angle1
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    # =========================================================================
    # STATUS AND DEBUGGING
    # =========================================================================
    
    def get_mode(self) -> ControlMode:
        """Get current control mode."""
        return self._mode
    
    def is_at_target(self, 
                     current_pose: Pose3D,
                     position_tol: float = 0.2,
                     yaw_tol: float = 0.15) -> bool:
        """
        Check if robot has reached target.
        
        Args:
            current_pose: Current robot pose
            position_tol: Position tolerance in meters
            yaw_tol: Yaw tolerance in radians
            
        Returns:
            True if at target within tolerances
        """
        if self._target_pose is None:
            return True
        
        # Position error
        pos_error = np.linalg.norm(
            current_pose.position() - self._target_pose.position()
        )
        
        if pos_error > position_tol:
            return False
        
        # Yaw error
        current_yaw = Transforms.euler_from_quaternion(current_pose.quaternion())[2]
        target_yaw = Transforms.euler_from_quaternion(self._target_pose.quaternion())[2]
        yaw_error = abs(self._angle_diff(current_yaw, target_yaw))
        
        return yaw_error <= yaw_tol
    
    def get_errors(self) -> dict:
        """Get current error values (for debugging/tuning)."""
        return self._last_errors.copy()
    
    def get_debug_info(self) -> dict:
        """Get debugging information."""
        return {
            'mode': self._mode.name,
            'target': self._target_pose,
            'errors': self._last_errors,
        }


class VelocityLimiter:
    """
    Limits velocity commands for smooth, safe operation.
    
    Features:
    - Rate limiting (prevents sudden acceleration)
    - Smooth deceleration near target
    - Emergency stop capability
    """
    
    def __init__(self,
                 max_accel: float = 0.3,  # m/s²
                 max_decel: float = 0.5,  # m/s²
                 smooth_distance: float = 1.0):  # Start slowing at this distance
        """Initialize velocity limiter."""
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.smooth_distance = smooth_distance
        
        self._last_velocity = VelocityCommand()
    
    def limit(self, 
              cmd: VelocityCommand, 
              dt: float,
              distance_to_target: float = float('inf')) -> VelocityCommand:
        """
        Apply velocity limits.
        
        Args:
            cmd: Desired velocity command
            dt: Time step
            distance_to_target: Distance to target (for smooth deceleration)
            
        Returns:
            Limited velocity command
        """
        # Apply distance-based speed limit
        if distance_to_target < self.smooth_distance:
            scale = max(0.2, distance_to_target / self.smooth_distance)
            cmd = VelocityCommand(
                surge=cmd.surge * scale,
                sway=cmd.sway * scale,
                heave=cmd.heave * scale,
                yaw=cmd.yaw * scale
            )
        
        # Apply rate limiting (acceleration limit)
        if dt > 0:
            max_delta = self.max_accel * dt
            
            cmd = VelocityCommand(
                surge=self._rate_limit(cmd.surge, self._last_velocity.surge, max_delta),
                sway=self._rate_limit(cmd.sway, self._last_velocity.sway, max_delta),
                heave=self._rate_limit(cmd.heave, self._last_velocity.heave, max_delta),
                yaw=self._rate_limit(cmd.yaw, self._last_velocity.yaw, max_delta)
            )
        
        self._last_velocity = cmd
        return cmd
    
    def _rate_limit(self, target: float, current: float, max_delta: float) -> float:
        """Limit rate of change."""
        delta = target - current
        if abs(delta) > max_delta:
            return current + math.copysign(max_delta, delta)
        return target
    
    def reset(self):
        """Reset limiter state."""
        self._last_velocity = VelocityCommand()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the motion controller module."""
    
    print("=" * 60)
    print("MOTION CONTROLLER MODULE TEST")
    print("=" * 60)
    
    # Create controller with custom config
    config = ControllerConfig(
        max_surge=0.4,
        max_yaw_rate=0.3,
    )
    controller = MotionController(config)
    
    # Create test poses
    current = Pose3D(x=0, y=0, z=2, qx=0, qy=0, qz=0, qw=1)
    target = Pose3D(x=3, y=2, z=2.5, qx=0, qy=0, qz=0.383, qw=0.924)  # 45° yaw
    
    print("\n1. Initial Setup:")
    print(f"   Current: pos=({current.x}, {current.y}, {current.z})")
    print(f"   Target:  pos=({target.x}, {target.y}, {target.z})")
    
    # Set target
    controller.set_target(target)
    controller.enable()
    
    # Simulate several control cycles
    print("\n2. Control Loop Simulation:")
    
    for i in range(5):
        velocity = controller.update(current)
        errors = controller.get_errors()
        
        print(f"\n   Cycle {i+1}:")
        print(f"     Velocity: surge={velocity.surge:.3f}, sway={velocity.sway:.3f}, "
              f"heave={velocity.heave:.3f}, yaw={velocity.yaw:.3f}")
        print(f"     Errors: surge={errors['surge']:.3f}, sway={errors['sway']:.3f}, "
              f"heave={errors['heave']:.3f}, yaw={errors['yaw']:.3f}")
        
        # Simulate movement (simplified)
        current.x += velocity.surge * 0.05
        current.y += velocity.sway * 0.05
        current.z += velocity.heave * 0.05
    
    # Test velocity limiter
    print("\n3. Velocity Limiter Test:")
    limiter = VelocityLimiter(max_accel=0.5)
    
    sudden_cmd = VelocityCommand(surge=1.0, sway=0.5, heave=0, yaw=0.3)
    limited_cmd = limiter.limit(sudden_cmd, dt=0.05, distance_to_target=2.0)
    
    print(f"   Sudden command: surge={sudden_cmd.surge:.3f}")
    print(f"   Limited output: surge={limited_cmd.surge:.3f}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
