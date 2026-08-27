#!/usr/bin/python3
"""Run the M3DGR MID360+IMU LIO export without camera or ONNX corrections."""

import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _create_runtime_actions(context):
    sequence = LaunchConfiguration("sequence").perform(context).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sequence):
        raise RuntimeError("sequence may contain only letters, digits, '_' and '-'.")
    bag_path = os.path.abspath(
        os.path.expanduser(LaunchConfiguration("bag_path").perform(context))
    )
    if not bag_path:
        raise RuntimeError("bag_path is required.")
    output_root = os.path.abspath(
        os.path.expanduser(LaunchConfiguration("output_root").perform(context))
    )
    covariance_profile = LaunchConfiguration("covariance_profile").perform(context).strip()
    covariance_overrides = {
        "current": {"imu.acc_cov": 0.5, "imu.gyr_cov": 0.3},
        "kalibr_variance": {
            "imu.acc_cov": 1.316155576238733e-04,
            "imu.gyr_cov": 1.756828497459329e-06,
        },
    }
    if covariance_profile not in covariance_overrides:
        raise RuntimeError(
            "covariance_profile must be 'current' or 'kalibr_variance'."
        )
    run_suffix = "" if covariance_profile == "current" else f"__{covariance_profile}"
    run_dir = os.path.join(output_root, f"m3dgr_{sequence}{run_suffix}")
    export_csv = os.path.join(run_dir, "mamba_pose_train_data.csv")
    allow_append = _as_bool(LaunchConfiguration("allow_append").perform(context))
    if os.path.exists(export_csv) and os.path.getsize(export_csv) > 0 and not allow_append:
        raise RuntimeError(
            f"Refusing to append to existing export CSV: {export_csv}. "
            "Use a clean run directory or set allow_append:=true explicitly."
        )

    mid360_params = LaunchConfiguration("mid360_params_file").perform(context)
    camera_params = LaunchConfiguration("camera_params_file").perform(context)
    mamba_params = LaunchConfiguration("mamba_pose_params_file").perform(context)
    actions = [
        Node(
            package="fast_livo",
            executable="fastlivo_mapping",
            name="laserMapping",
            parameters=[
                mid360_params,
                camera_params,
                mamba_params,
                {
                    "mamba_pose/export_train_data_path": export_csv,
                    "mamba_pose/apply_correction_en": False,
                    **covariance_overrides[covariance_profile],
                },
            ],
            output="screen",
        )
    ]
    if _as_bool(LaunchConfiguration("play_bag").perform(context)):
        command = ["ros2", "bag", "play", bag_path]
        if _as_bool(LaunchConfiguration("bag_clock").perform(context)):
            command.append("--clock")
        actions.append(ExecuteProcess(cmd=command, output="screen"))
    return actions


def generate_launch_description():
    package_share = get_package_share_directory("fast_livo")
    config_dir = os.path.join(package_share, "config")
    rviz_config = os.path.join(package_share, "rviz_cfg", "fast_livo2.rviz")
    default_output_root = "/home/liu/fast_livo2/src/FAST-LIVO2/Log/runs"
    return LaunchDescription(
        [
            DeclareLaunchArgument("sequence", default_value="Varying-illu02"),
            DeclareLaunchArgument("bag_path", default_value=""),
            DeclareLaunchArgument("play_bag", default_value="true"),
            DeclareLaunchArgument("bag_clock", default_value="true"),
            DeclareLaunchArgument("allow_append", default_value="false"),
            DeclareLaunchArgument(
                "covariance_profile",
                default_value="current",
                description=(
                    "Use current FAST-LIVO2 covariance values or the M3DGR "
                    "Kalibr-density-squared candidate. Select/freeze on train only."
                ),
            ),
            DeclareLaunchArgument("output_root", default_value=default_output_root),
            DeclareLaunchArgument(
                "mid360_params_file",
                default_value=os.path.join(config_dir, "m3dgr_mid360_lio.yaml"),
            ),
            DeclareLaunchArgument(
                "camera_params_file",
                default_value=os.path.join(config_dir, "camera_pinhole.yaml"),
            ),
            DeclareLaunchArgument(
                "mamba_pose_params_file",
                default_value=os.path.join(
                    config_dir, "mamba_pose_m3dgr_train_export_dummy.yaml"
                ),
            ),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            OpaqueFunction(function=_create_runtime_actions),
            Node(
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                output="screen",
            ),
        ]
    )
