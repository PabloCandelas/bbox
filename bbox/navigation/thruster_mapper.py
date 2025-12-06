#!/usr/bin/env python3
"""
thruster_mapper.py - Thruster Allocation for BlueROV2
======================================================

WHAT IS THRUSTER MAPPING?
-------------------------
Thruster mapping (also called "thruster allocation" or "control allocation")
converts velocity commands (surge, sway, heave, yaw) into individual thruster
signals (PWM values).

Your BlueROV2 has 8 thrusters:
- 4 horizontal thrusters for surge, sway, and yaw
- 4 vertical thrusters for heave

THE CHALLENGE:
--------------
We want to move in 4 directions (surge, sway, heave, yaw), but we have 8 thrusters.
This means multiple thruster combinations can achieve the same motion.
Thruster mapping finds the right combination.

BLUEROV2 THRUSTER CONFIGURATION:
--------------------------------
Looking from above (top view):

        FRONT
          ↑
    T1 ╲     ╱ T2     (Horizontal thrusters, angled 45°)
        ╲   ╱
         ╲ ╱
    T3 ─── ───  T4    (Horizontal thrusters, angled 45°)
         ╱ ╲
        ╱   ╲
    T5 ╱     ╲ T6     (Not actually like this - see below)
        
Actually, BlueROV2 has a different layout. Let me correct:

Standard BlueROV2 Heavy (8 thrusters):
- Thrusters 1-4: Horizontal, angled at 45° for omnidirectional movement
- Thrusters 5-8: Vertical, for depth control

          FRONT
            ↑
     T1 ↖     ↗ T2    (angled 45° towards center)
          
           □           (ROV body)
          
     T4 ↙     ↘ T3    (angled 45° towards center)

Vertical thrusters (T5-T8) are at each corner pointing up/down.

PWM VALUES:
-----------
Thrusters are controlled via PWM (Pulse Width Modulation):
- 1500 µs = Neutral (no thrust)
- 1100 µs = Full reverse
- 1900 µs = Full forward
- Range: 1100 to 1900 µs

THRUST ALLOCATION MATRIX:
-------------------------
We use a matrix that maps [surge, sway, heave, yaw] to [T1, T2, T3, T4, T5, T6, T7, T8].

Each row represents how one thruster contributes to each motion axis.
"""

import numpy as np
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .motion_controller import VelocityCommand


@dataclass
class ThrusterConfig:
    """
    Configuration for a single thruster.
    
    Attributes:
        channel: RC channel number (1-indexed for MAVROS)
        direction: 1 for normal, -1 for reversed
        max_thrust: Maximum thrust force (normalized, 0-1)
    """
    channel: int
    direction: int = 1
    max_thrust: float = 1.0


@dataclass
class ThrusterMapperConfig:
    """
    Configuration for the thruster mapper.
    
    You may need to adjust these based on your specific ROV setup!
    """
    # PWM limits
    pwm_neutral: int = 1500
    pwm_min: int = 1100
    pwm_max: int = 1900
    
    # Deadband (PWM values this close to neutral are set to neutral)
    pwm_deadband: int = 25
    
    # Thrust scaling factors (adjust for desired response)
    surge_scale: float = 1.0
    sway_scale: float = 1.0
    heave_scale: float = 1.0
    yaw_scale: float = 0.8
    
    # Individual thruster configurations
    # BlueROV2 Heavy has 8 thrusters
    # Channels typically: 1=pitch, 2=roll, 3=throttle, 4=yaw, 5=forward, 6=lateral
    # But for RC override: channels 0-5 are the 6 motion channels
    
    # If any thruster is reversed, set direction to -1
    thruster_directions: List[int] = None
    
    def __post_init__(self):
        if self.thruster_directions is None:
            # Default: all thrusters in normal direction
            self.thruster_directions = [1, 1, 1, 1, 1, 1, 1, 1]


class ThrusterMapper:
    """
    Maps velocity commands to thruster PWM values for BlueROV2.
    
    This class handles the conversion from desired velocities to actual
    thruster commands that can be sent to the ROV.
    
    Usage:
        mapper = ThrusterMapper()
        
        # Get PWM values
        pwms = mapper.velocity_to_pwm(velocity_command)
        
        # Or get forces
        forces = mapper.velocity_to_forces(velocity_command)
    
    Integration with MAVROS:
        The output is designed to work with mavros_msgs/OverrideRCIn.
        Channels [0-5] map to [pitch, roll, throttle, yaw, forward, lateral].
    """
    
    def __init__(self, config: Optional[ThrusterMapperConfig] = None):
        """Initialize thruster mapper."""
        self.config = config or ThrusterMapperConfig()
        
        # Build thrust allocation matrix
        self._build_allocation_matrix()
    
    def _build_allocation_matrix(self):
        """
        Build the thrust allocation matrix.
        
        For BlueROV2 with MAVROS, we use a simplified approach:
        - The flight controller (Pixhawk running ArduSub) handles the actual
          thruster mixing internally
        - We just need to send the 6 virtual "channels": 
          [pitch, roll, throttle, yaw, forward, lateral]
        
        So our allocation is straightforward:
        - Surge → Forward channel
        - Sway → Lateral channel  
        - Heave → Throttle channel
        - Yaw → Yaw channel
        """
        # For MAVROS RC Override, channels are:
        # 0: Pitch
        # 1: Roll
        # 2: Throttle (heave)
        # 3: Yaw
        # 4: Forward (surge)
        # 5: Lateral (sway)
        
        # This maps [surge, sway, heave, yaw] to [ch0, ch1, ch2, ch3, ch4, ch5]
        self._channel_map = {
            'pitch': 0,
            'roll': 1,
            'throttle': 2,  # Heave
            'yaw': 3,
            'forward': 4,   # Surge
            'lateral': 5,   # Sway
        }
    
    def velocity_to_pwm(self, velocity: VelocityCommand) -> List[int]:
        """
        Convert velocity command to PWM values.
        
        Args:
            velocity: VelocityCommand with surge, sway, heave, yaw
            
        Returns:
            List of 18 PWM values for MAVROS RC Override
            (most will be 0, meaning "no override")
        """
        # Initialize all channels to 0 (no override)
        pwm = [0] * 18
        
        # Scale velocities to [-1, 1] range
        surge_scaled = velocity.surge * self.config.surge_scale
        sway_scaled = velocity.sway * self.config.sway_scale
        heave_scaled = velocity.heave * self.config.heave_scale
        yaw_scaled = velocity.yaw * self.config.yaw_scale
        
        # Clamp to [-1, 1]
        surge_scaled = np.clip(surge_scaled, -1.0, 1.0)
        sway_scaled = np.clip(sway_scaled, -1.0, 1.0)
        heave_scaled = np.clip(heave_scaled, -1.0, 1.0)
        yaw_scaled = np.clip(yaw_scaled, -1.0, 1.0)
        
        # Convert to PWM
        pwm[self._channel_map['forward']] = self._value_to_pwm(surge_scaled)
        pwm[self._channel_map['lateral']] = self._value_to_pwm(-sway_scaled)  # Note: may need to flip sign
        pwm[self._channel_map['throttle']] = self._value_to_pwm(heave_scaled)
        pwm[self._channel_map['yaw']] = self._value_to_pwm(-yaw_scaled)  # Note: may need to flip sign
        
        # Pitch and roll channels (not controlled, set to neutral)
        pwm[self._channel_map['pitch']] = self.config.pwm_neutral
        pwm[self._channel_map['roll']] = self.config.pwm_neutral
        
        return pwm
    
    def _value_to_pwm(self, value: float) -> int:
        """
        Convert normalized value [-1, 1] to PWM [1100, 1900].
        
        Args:
            value: Normalized value (-1 to 1)
            
        Returns:
            PWM value (1100 to 1900)
        """
        # Map [-1, 1] to [1100, 1900]
        # value = -1 → 1100
        # value = 0  → 1500
        # value = 1  → 1900
        
        pwm_range = (self.config.pwm_max - self.config.pwm_min) / 2
        pwm = self.config.pwm_neutral + int(value * pwm_range)
        
        # Apply deadband
        if abs(pwm - self.config.pwm_neutral) < self.config.pwm_deadband:
            pwm = self.config.pwm_neutral
        
        # Clamp to valid range
        pwm = max(self.config.pwm_min, min(self.config.pwm_max, pwm))
        
        return pwm
    
    def pwm_to_value(self, pwm: int) -> float:
        """
        Convert PWM to normalized value [-1, 1].
        
        Args:
            pwm: PWM value (1100 to 1900)
            
        Returns:
            Normalized value (-1 to 1)
        """
        pwm_range = (self.config.pwm_max - self.config.pwm_min) / 2
        return (pwm - self.config.pwm_neutral) / pwm_range
    
    def stop(self) -> List[int]:
        """
        Generate stop command (all thrusters neutral).
        
        Returns:
            List of neutral PWM values
        """
        pwm = [0] * 18
        for channel in self._channel_map.values():
            pwm[channel] = self.config.pwm_neutral
        return pwm
    
    def get_channel_map(self) -> dict:
        """Get the channel mapping dictionary."""
        return self._channel_map.copy()


class DirectThrusterMapper:
    """
    Direct thruster control (bypasses flight controller mixing).
    
    Use this if you want full control over individual thrusters.
    
    WARNING: This is more complex and requires correct configuration
    of your specific BlueROV2 setup!
    
    BlueROV2 Heavy Thruster Layout:
    
    Top View:
                      FRONT
                        ↑
           T1(CCW)  ╲       ╱  T2(CW)
                     ╲     ╱
                      ╲   ╱
                       ╲ ╱
           T4(CW)   ─── ● ───   T3(CCW)
                       ╱ ╲
                      ╱   ╲
                     ╱     ╲
           T5(CW)   ↑       ↑   T6(CCW)
                     
    Side View:
           T5  ○───────────○  T6
              │           │
              │           │
           T7  ○───────────○  T8
           
    Vertical thrusters (T5-T8) point upward for positive heave.
    """
    
    # Allocation matrix for BlueROV2 Heavy
    # Rows: [T1, T2, T3, T4, T5, T6, T7, T8]
    # Cols: [Surge, Sway, Heave, Yaw]
    ALLOCATION_MATRIX = np.array([
        # Surge  Sway   Heave  Yaw
        [ 0.707, -0.707,  0.0,  1.0],  # T1 - Front left horizontal
        [ 0.707,  0.707,  0.0, -1.0],  # T2 - Front right horizontal
        [-0.707,  0.707,  0.0,  1.0],  # T3 - Rear right horizontal
        [-0.707, -0.707,  0.0, -1.0],  # T4 - Rear left horizontal
        [  0.0,    0.0,   1.0,  0.0],  # T5 - Front left vertical
        [  0.0,    0.0,   1.0,  0.0],  # T6 - Front right vertical
        [  0.0,    0.0,   1.0,  0.0],  # T7 - Rear left vertical
        [  0.0,    0.0,   1.0,  0.0],  # T8 - Rear right vertical
    ])
    
    def __init__(self, config: Optional[ThrusterMapperConfig] = None):
        """Initialize direct thruster mapper."""
        self.config = config or ThrusterMapperConfig()
        
        # Apply thruster directions
        self.allocation = self.ALLOCATION_MATRIX.copy()
        for i, direction in enumerate(self.config.thruster_directions[:8]):
            self.allocation[i] *= direction
    
    def velocity_to_thrust(self, velocity: VelocityCommand) -> np.ndarray:
        """
        Convert velocity command to normalized thrust values.
        
        Args:
            velocity: VelocityCommand
            
        Returns:
            Array of 8 normalized thrust values [-1, 1]
        """
        # Create input vector
        inputs = np.array([
            velocity.surge * self.config.surge_scale,
            velocity.sway * self.config.sway_scale,
            velocity.heave * self.config.heave_scale,
            velocity.yaw * self.config.yaw_scale,
        ])
        
        # Compute thrust values
        thrusts = self.allocation @ inputs
        
        # Normalize if any thrust exceeds limits
        max_thrust = np.max(np.abs(thrusts))
        if max_thrust > 1.0:
            thrusts /= max_thrust
        
        return thrusts
    
    def velocity_to_pwm(self, velocity: VelocityCommand) -> List[int]:
        """
        Convert velocity command to PWM values for 8 thrusters.
        
        Args:
            velocity: VelocityCommand
            
        Returns:
            List of 8 PWM values [T1, T2, ..., T8]
        """
        thrusts = self.velocity_to_thrust(velocity)
        
        pwms = []
        for thrust in thrusts:
            pwm_range = (self.config.pwm_max - self.config.pwm_min) / 2
            pwm = self.config.pwm_neutral + int(thrust * pwm_range)
            
            # Apply deadband
            if abs(pwm - self.config.pwm_neutral) < self.config.pwm_deadband:
                pwm = self.config.pwm_neutral
            
            # Clamp
            pwm = max(self.config.pwm_min, min(self.config.pwm_max, pwm))
            pwms.append(pwm)
        
        return pwms


class ThrusterTest:
    """
    Utility class for testing individual thrusters.
    
    SAFETY WARNING: Only use in water with proper safety measures!
    Start with low thrust values.
    """
    
    def __init__(self, mapper: ThrusterMapper):
        """Initialize with a thruster mapper."""
        self.mapper = mapper
    
    def test_surge(self, thrust: float = 0.3) -> List[int]:
        """Generate forward thrust command."""
        return self.mapper.velocity_to_pwm(VelocityCommand(surge=thrust))
    
    def test_sway(self, thrust: float = 0.3) -> List[int]:
        """Generate sideways thrust command."""
        return self.mapper.velocity_to_pwm(VelocityCommand(sway=thrust))
    
    def test_heave(self, thrust: float = 0.3) -> List[int]:
        """Generate vertical thrust command."""
        return self.mapper.velocity_to_pwm(VelocityCommand(heave=thrust))
    
    def test_yaw(self, thrust: float = 0.3) -> List[int]:
        """Generate rotation thrust command."""
        return self.mapper.velocity_to_pwm(VelocityCommand(yaw=thrust))
    
    def generate_test_sequence(self) -> List[Tuple[str, VelocityCommand]]:
        """
        Generate a sequence of test commands.
        
        Returns:
            List of (description, VelocityCommand) tuples
        """
        return [
            ("Surge forward", VelocityCommand(surge=0.3)),
            ("Surge backward", VelocityCommand(surge=-0.3)),
            ("Sway right", VelocityCommand(sway=0.3)),
            ("Sway left", VelocityCommand(sway=-0.3)),
            ("Heave down", VelocityCommand(heave=0.3)),
            ("Heave up", VelocityCommand(heave=-0.3)),
            ("Yaw right", VelocityCommand(yaw=0.3)),
            ("Yaw left", VelocityCommand(yaw=-0.3)),
            ("Stop", VelocityCommand()),
        ]


# =============================================================================
# ROS2 INTEGRATION HELPER
# =============================================================================

def create_rc_override_msg(pwm_values: List[int]):
    """
    Create a MAVROS RC Override message.
    
    This is a helper function showing how to use the mapper output
    with MAVROS.
    
    Usage (in your ROS2 node):
        from mavros_msgs.msg import OverrideRCIn
        
        msg = OverrideRCIn()
        msg.channels = mapper.velocity_to_pwm(velocity)
        rc_override_pub.publish(msg)
    
    Args:
        pwm_values: List of PWM values from ThrusterMapper
        
    Returns:
        Dictionary mimicking OverrideRCIn message structure
    """
    return {
        'channels': pwm_values,
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the thruster mapper module."""
    
    print("=" * 60)
    print("THRUSTER MAPPER MODULE TEST")
    print("=" * 60)
    
    # Create mapper
    mapper = ThrusterMapper()
    
    # Test basic velocity commands
    print("\n1. Basic Velocity Commands:")
    
    test_commands = [
        ("Forward", VelocityCommand(surge=0.5)),
        ("Backward", VelocityCommand(surge=-0.5)),
        ("Right", VelocityCommand(sway=0.5)),
        ("Down", VelocityCommand(heave=0.5)),
        ("Turn right", VelocityCommand(yaw=0.5)),
        ("Stop", VelocityCommand()),
    ]
    
    for name, cmd in test_commands:
        pwm = mapper.velocity_to_pwm(cmd)
        relevant_pwm = pwm[:6]  # First 6 channels
        print(f"   {name:12s}: PWM = {relevant_pwm}")
    
    # Test combined movements
    print("\n2. Combined Movements:")
    
    combined = VelocityCommand(surge=0.3, sway=0.2, heave=-0.1, yaw=0.15)
    pwm = mapper.velocity_to_pwm(combined)
    print(f"   Surge=0.3, Sway=0.2, Heave=-0.1, Yaw=0.15")
    print(f"   PWM: pitch={pwm[0]}, roll={pwm[1]}, throttle={pwm[2]}, "
          f"yaw={pwm[3]}, forward={pwm[4]}, lateral={pwm[5]}")
    
    # Test direct thruster mapper
    print("\n3. Direct Thruster Mapper (8 thrusters):")
    
    direct_mapper = DirectThrusterMapper()
    
    thrusts = direct_mapper.velocity_to_thrust(VelocityCommand(surge=0.5))
    print(f"   Surge=0.5 thrust distribution:")
    for i, t in enumerate(thrusts):
        print(f"     T{i+1}: {t:+.3f}")
    
    # Test PWM conversion
    print("\n4. PWM Range Test:")
    
    for value in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        pwm = mapper._value_to_pwm(value)
        back = mapper.pwm_to_value(pwm)
        print(f"   Value {value:+.1f} → PWM {pwm} → Back {back:+.3f}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
