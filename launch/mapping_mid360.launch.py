#!/usr/bin/python3
# -- coding: utf-8 --**

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def _parse_launch_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _create_bag_play_action(context):
    play_bag = _parse_launch_bool(LaunchConfiguration("play_bag").perform(context))
    if not play_bag:
        return []

    bag_path = os.path.expanduser(LaunchConfiguration("bag_path").perform(context))
    bag_clock = _parse_launch_bool(LaunchConfiguration("bag_clock").perform(context))
    bag_loop = _parse_launch_bool(LaunchConfiguration("bag_loop").perform(context))

    cmd = ["ros2", "bag", "play", bag_path]
    if bag_clock:
        cmd.append("--clock")
    if bag_loop:
        cmd.append("--loop")

    return [
        ExecuteProcess(
            cmd=cmd,
            output="screen",
        )
    ]


def generate_launch_description():
    
    # Find path
    config_file_dir = os.path.join(get_package_share_directory("fast_livo"), "config")
    rviz_config_file = os.path.join(get_package_share_directory("fast_livo"), "rviz_cfg", "fast_livo2.rviz")

    # Load MID360 lidar + camera + MambaPose parameters.
    mid360_config_cmd = os.path.join(config_file_dir, "mid360.yaml")
    camera_config_cmd = os.path.join(config_file_dir, "camera_pinhole.yaml")
    mamba_pose_config_cmd = os.path.join(config_file_dir, "mamba_pose_onnx_normal_test.yaml")

    # 打开 use_rviz
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="True",
        description="Whether to launch Rviz2",
    )

    mid360_config_arg = DeclareLaunchArgument(
        'mid360_params_file',
        default_value=mid360_config_cmd,
        description='Full path to the ROS2 parameters file to use for fast_livo2 nodes',
    )

    camera_config_arg = DeclareLaunchArgument(
        'camera_params_file',
        default_value=camera_config_cmd,
        description='Full path to the ROS2 parameters file to use for vikit_ros nodes',
    )

    mamba_pose_config_arg = DeclareLaunchArgument(
        'mamba_pose_params_file',
        default_value=mamba_pose_config_cmd,
        description='Full path to the ROS2 parameters file to use for mamba pose test settings',
    )

    play_bag_arg = DeclareLaunchArgument(
        'play_bag',
        default_value='True',
        description='Whether to automatically play a rosbag when launching MID360 mapping.',
    )

    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value='',
        description='Full path to the rosbag to play when play_bag is enabled.',
    )

    bag_loop_arg = DeclareLaunchArgument(
        'bag_loop',
        default_value='False',
        description='Whether to loop rosbag playback when play_bag is enabled.',
    )

    bag_clock_arg = DeclareLaunchArgument(
        'bag_clock',
        default_value='True',
        description='Whether to add --clock to rosbag playback when play_bag is enabled.',
    )

    use_respawn_arg = DeclareLaunchArgument(
        'use_respawn', 
        default_value='True',
        description='Whether to respawn if a node crashes. Applied when composition is disabled.')

    mid360_params_file = LaunchConfiguration('mid360_params_file')
    camera_params_file = LaunchConfiguration('camera_params_file')
    mamba_pose_params_file = LaunchConfiguration('mamba_pose_params_file')
    use_respawn = LaunchConfiguration('use_respawn')

    return LaunchDescription([
        use_rviz_arg,
        mid360_config_arg,
        camera_config_arg,
        mamba_pose_config_arg,
        play_bag_arg,
        bag_path_arg,
        bag_loop_arg,
        bag_clock_arg,
        use_respawn_arg,

        OpaqueFunction(function=_create_bag_play_action),

        Node(
            package="image_transport",
            executable="republish",
            name="republish",
            arguments=[ 
                'compressed', 
                'raw',
            ],
            remappings=[
                #("in",  "/image_raw/compressed"), 
                #("out", "/image_raw/compressed")
                ("in",  "/image_raw"), 
                ("out", "/image_raw")
            ],
            output="screen",
            respawn=use_respawn,
        ),
        
        Node(
            package="fast_livo",
            executable="fastlivo_mapping",
            name="laserMapping",
            parameters=[
                mid360_params_file,
                camera_params_file,
                mamba_pose_params_file,
            ],
            output="screen"
        ),

        Node(
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config_file],
            output="screen"
        ),
    ])
