#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Top-level ROS2 launch file that brings up the UR3e driver, MoveIt, RealSense, and all application nodes.
"""
master.launch.py

Top-level launch file for the UR3e object scanning system.

This launch file is responsible for:
- bringing up the UR robot driver,
- starting MoveIt for motion planning,
- launching the Intel RealSense camera stack,
- publishing the fixed hand-eye and optical-frame TFs,
- starting the RGB-D capture node,
- starting the robot movement node,
- starting the main system supervisor node,
- starting the GUI.

Design notes:
- Hardware-specific nodes are only started when real hardware is enabled.
- MoveIt is delayed slightly so the robot driver has time to initialise cleanly.
- Application-layer nodes are delayed until the robot and planning stack are ready.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ------------------------------------------------------------------
    # Package directories
    # ------------------------------------------------------------------
    ur_driver_share_dir = get_package_share_directory("ur_robot_driver")
    rs2_share_dir = get_package_share_directory("rs2")
    realsense_share_dir = get_package_share_directory("realsense2_camera")
    ur_description_share_dir = get_package_share_directory("ur_description")

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    reverse_ip = LaunchConfiguration("reverse_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    kinematics_params_file = LaunchConfiguration("kinematics_params_file")

    # Common condition: only launch hardware-dependent nodes on real hardware
    real_hw_only = IfCondition(
        PythonExpression(["'", use_fake_hardware, "' == 'false'"])
    )

    # ------------------------------------------------------------------
    # UR robot driver
    # ------------------------------------------------------------------
    driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_driver_share_dir, "launch", "ur_control.launch.py")
        ),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "reverse_ip": reverse_ip,
            "use_fake_hardware": use_fake_hardware,
            "kinematics_params_file": kinematics_params_file,
            "description_package": "rs2",
            "description_file": "rs2_world.urdf.xacro",


            "joint_limit_params": os.path.join(
                ur_description_share_dir, "config", "ur3e", "joint_limits.yaml"
            ),
            "physical_params": os.path.join(
                ur_description_share_dir, "config", "ur3e", "physical_parameters.yaml"
            ),
            "visual_params": os.path.join(
                ur_description_share_dir, "config", "ur3e", "visual_parameters.yaml"
            ),



            "launch_rviz": "false",
            "initial_joint_controller": "scaled_joint_trajectory_controller",
            "activate_joint_controller": "true",
        }.items(),
    )

    # ------------------------------------------------------------------
    # MoveIt motion-planning stack
    # ------------------------------------------------------------------
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rs2_share_dir, "launch", "moveItLaunch.py")
        ),
        launch_arguments={
            "ur_type": ur_type,
            "description_package": "rs2",
            "description_file": "rs2_world.urdf.xacro",
            "kinematics_params_file": kinematics_params_file,
            "launch_rviz": "true",
            "use_sim_time": "false",
        }.items(),
    )

    # ------------------------------------------------------------------
    # RealSense camera stack
    # ------------------------------------------------------------------
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_share_dir, "launch", "rs_launch.py")
        ),
        launch_arguments={
            "camera_name": "d435i",
            "camera_namespace": "camera",
            "initial_reset": "true",
            "enable_color": "true",
            "enable_depth": "true",
            "align_depth.enable": "true",
            "pointcloud.enable": "true",
            "pointcloud.stream_filter": "2",
            "pointcloud.stream_index_filter": "0",
            "enable_sync": "true",
            "filters": "spatial,hole_filling",
        }.items(),
        condition=real_hw_only,
    )

    # ------------------------------------------------------------------
    # Static TF: hand-eye calibration
    # tool0 -> d435i_link
    #
    # The easy_handeye2 calibration was run with
    #   tracking_base_frame: d435i_color_optical_frame
    # so its output is T_tool0->d435i_color_optical_frame (small/near-identity
    # rotation because tool0 and the optical frame point in similar directions).
    #
    # We need T_tool0->d435i_link here so the driver's own
    # d435i_link->d435i_color_optical_frame transform completes the chain
    # without double-applying the body-to-optical rotation.
    #
    # Corrected values computed as:
    #   T_tool0_to_d435i_link = T_calib * inv(T_d435i_link_to_optical)
    # where T_d435i_link_to_optical = roll=-π/2, pitch=0, yaw=-π/2
    # (the standard RealSense body-to-optical rotation published by the driver).
    # Translation is unchanged; only the rotation quaternion changes.
    # ------------------------------------------------------------------
    handeye_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "-0.022588736608667",
            "--y", "0.010379646224213",
            "--z", "0.051145708964794",
            "--qx", "0.510642621",
            "--qy", "-0.493534731",
            "--qz", "0.493613808",
            "--qw", "0.502008956",
            "--frame-id", "tool0",
            "--child-frame-id", "d435i_link",
        ],
        output="screen",
        condition=real_hw_only,
    )

    # ------------------------------------------------------------------
    # Main control node
    # Supervises the major subsystems and can react to heartbeat failures,
    # scanner failures, or movement faults.
    # ------------------------------------------------------------------
    main_control_node = Node(
        package='rs2',
        executable='main_control_node.py',
        name='main_control_node',
        output='screen',
        parameters=[{
            'use_fake_hardware': use_fake_hardware,
        }],
    )

    # ------------------------------------------------------------------
    # Movement node
    # Renamed from main_movement.py -> movement.py
    # Owns robot waypoint execution and motion-specific state.
    # ------------------------------------------------------------------
    movement_node = Node(
        package="rs2",
        executable="movement_node.py",
        name="movement_node",
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    # ------------------------------------------------------------------
    # GUI node
    # Operator-facing interface for scan control, status, and visualisation.
    # ------------------------------------------------------------------
    gui_node = Node(
        package="rs2",
        executable="gui_node.py",
        name="gui_node",
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    # ------------------------------------------------------------------
    # Scan node
    # Subscribes to aligned RGB-D data and saves scan captures on request.
    # ------------------------------------------------------------------
    scan_node = Node(
        package="rs2",
        executable="scan_node.py",
        name="scan_node",
        output="screen",
        parameters=[{
            'use_fake_hardware':    use_fake_hardware,
            'auto_reconstruct':     True,
            'recon_voxel_size':     0.003,
            'recon_icp':            True,
            'recon_icp_threshold':  0.005,
        }],
    )

    # ------------------------------------------------------------------
    # Launch description
    # Startup order:
    # 1. Robot driver immediately
    # 2. RealSense / TF / RGB-D capture immediately for real hardware
    # 3. MoveIt after a short delay
    # 4. Application nodes after the robot stack is stable
    # ------------------------------------------------------------------
    return LaunchDescription([
        DeclareLaunchArgument("ur_type", default_value="ur3e"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.0.194"),
        DeclareLaunchArgument("reverse_ip", default_value="192.168.0.100"),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
        DeclareLaunchArgument(
            "kinematics_params_file",
            default_value=os.path.join(
                rs2_share_dir,
                "config",
                "my_robot_calibration.yaml",
            ),
        ),

        # Core robot stack
        driver_launch,

        # Real hardware camera / TF / capture stack
        realsense_launch,
        handeye_tf,

        # Planning stack after driver startup
        TimerAction(period=4.0, actions=[moveit_launch]),

        # Application layer after robot + planning are available
        TimerAction(
            period=2.0,
            actions=[
                main_control_node
            ],
        ),

        # Application layer after robot + planning are available
        TimerAction(
            period=3.0,
            actions=[
                gui_node,
            ],
        ),

        # Application layer after robot + planning are available
        TimerAction(
            period=6.0,
            actions=[
                movement_node,
                scan_node,
            ],
        ),
    ])