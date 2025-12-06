#!/usr/bin/env python3
"""
Launch file for BlueROV2 Navigation System.

This launches all the nodes needed for autonomous navigation:
1. Video streaming (camera input)
2. ArUco detection (localization markers)
3. YOLO detection (blackbox and handle)
4. Navigation node (path planning, control, visual servoing)

Usage:
    ros2 launch bbox run_navigation.launch.py

    # With search mode
    ros2 launch bbox run_navigation.launch.py mode:=search

    # With custom control rate
    ros2 launch bbox run_navigation.launch.py control_rate:=30.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description."""
    
    # Declare arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='idle',
        description='Initial navigation mode (idle, search, hold_position)'
    )
    
    control_rate_arg = DeclareLaunchArgument(
        'control_rate',
        default_value='20.0',
        description='Control loop rate in Hz'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image',
        description='Camera image topic'
    )
    
    # Video node (camera streaming)
    video_node = Node(
        package='bbox',
        executable='video',
        name='video',
        output='screen',
        parameters=[{
            'port': 5600,
        }]
    )
    
    # ArUco detection node (localization)
    aruco_node = Node(
        package='bbox',
        executable='aruco_pool_node',
        name='aruco_pool_node',
        output='screen',
        parameters=[{
            'image_topic': LaunchConfiguration('camera_topic'),
            'camera_info_npz': 'bbox/pablos_camera_calb.npz',
            'marker_size_m': 0.30,
            'publish_tf': True,
        }]
    )
    
    # YOLO detection node (blackbox and handle)
    yolo_node = Node(
        package='bbox',
        executable='bbox_yolo_detection',
        name='bbox_yolo_detection',
        output='screen',
        parameters=[{
            'model_path': 'bbox/best.pt',
            'camera_topic': LaunchConfiguration('camera_topic'),
            'conf_thres': 0.5,
            'show_display': True,
        }]
    )
    
    # Navigation node (the main controller)
    navigation_node = Node(
        package='bbox',
        executable='navigation_node',
        name='navigation_node',
        output='screen',
        parameters=[{
            'mode': LaunchConfiguration('mode'),
            'control_rate': LaunchConfiguration('control_rate'),
            
            # Localization
            'position_filter_alpha': 0.3,
            'use_depth_sensor': True,
            
            # Motion control
            'max_surge': 0.4,
            'max_sway': 0.4,
            'max_heave': 0.3,
            'max_yaw_rate': 0.4,
            
            # Visual servoing
            'servoing_enabled': True,
            'approach_distance': 1.0,
            
            # Safety
            'data_timeout': 2.0,
        }]
    )
    
    return LaunchDescription([
        # Arguments
        mode_arg,
        control_rate_arg,
        camera_topic_arg,
        
        # Info
        LogInfo(msg='Starting BlueROV2 Navigation System...'),
        LogInfo(msg=['Mode: ', LaunchConfiguration('mode')]),
        LogInfo(msg=['Control rate: ', LaunchConfiguration('control_rate'), ' Hz']),
        
        # Nodes
        video_node,
        aruco_node,
        yolo_node,
        navigation_node,
    ])
