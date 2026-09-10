#!/usr/bin/python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("fast_livo")
    default_parameters = os.path.join(package_share, "config", "m3dgr_mid360_lio.yaml")
    default_camera = os.path.join(package_share, "config", "camera_pinhole_m3dgr.yaml")
    rviz_configuration = os.path.join(package_share, "rviz_cfg", "fast_livo2.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_parameters),
            DeclareLaunchArgument("camera_params_file", default_value=default_camera),
            DeclareLaunchArgument("sequence_name", default_value="M3DGR_LIO"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            Node(
                package="fast_livo",
                executable="fastlivo_mapping",
                name="laserMapping",
                parameters=[
                    LaunchConfiguration("params_file"),
                    LaunchConfiguration("camera_params_file"),
                    {
                        "common.img_en": 0,
                        "evo.pose_output_en": True,
                        "evo.seq_name": LaunchConfiguration("sequence_name"),
                        "pcd_save.pcd_save_en": False,
                    },
                ],
                output="screen",
            ),
            Node(
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_configuration],
                output="screen",
            ),
        ]
    )
