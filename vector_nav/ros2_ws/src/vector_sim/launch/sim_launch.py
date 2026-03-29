"""
sim_launch.py
──────────────
Launches the complete simulation stack:
  1. Gazebo Harmonic with vector_arena.sdf
  2. gz_bridge: /model/vectorN/pose → ROS2  and  /vectorN/cmd_vel ← ROS2
  3. (Optional) RViz2 for visualisation

Usage:
  ros2 launch vector_sim sim_launch.py n_agents:=2
  ros2 launch vector_sim sim_launch.py n_agents:=2 rviz:=true
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_bridge_params(n_agents: int):
    """Generate gz_bridge topic list for N agents."""
    params = []
    for i in range(n_agents):
        robot = f"vector{i}"
        # Pose: Gazebo → ROS2  (read robot position)
        params.append(
            f"/model/{robot}/pose@geometry_msgs/msg/Pose[gz.msgs.Pose"
        )
        # cmd_vel: ROS2 → Gazebo  (matches diff-drive plugin topic)
        params.append(
            f"/{robot}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"
        )
        # Odometry: Gazebo → ROS2
        params.append(
            f"/{robot}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry"
        )
        # Goal poses (static, read once)
    for g in [f"goal{i}" for i in range(n_agents)]:
        params.append(
            f"/model/{g}/pose@geometry_msgs/msg/Pose[gz.msgs.Pose"
        )
    return params


def launch_setup(context, *args, **kwargs):
    n_agents = int(LaunchConfiguration("n_agents").perform(context))
    rviz     = LaunchConfiguration("rviz").perform(context)
    pkg_dir  = get_package_share_directory("vector_sim")
    world    = os.path.join(pkg_dir, "worlds", "vector_arena.sdf")

    # Directory containing models/vector, vector0, vector1, vector2 subdirs.
    # GZ_SIM_RESOURCE_PATH must be set in os.environ BEFORE Gazebo starts —
    # Gazebo resolves model:// URIs at SDF parse time, so additional_env
    # (which is applied after the process forks) is too late.
    models_dir = os.path.join(pkg_dir, "models")
    existing = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_path = f"{models_dir}:{existing}" if existing else models_dir
    os.environ["GZ_SIM_RESOURCE_PATH"] = resource_path

    actions = []

    # ── 1. Gazebo Harmonic ────────────────────────────────────────────
    actions.append(
        ExecuteProcess(
            cmd=["gz", "sim", world, "-r"],
            output="screen",
        )
    )

    # ── 2. gz_bridge ─────────────────────────────────────────────────
    bridge_topics = generate_bridge_params(n_agents)
    actions.append(
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="gz_bridge",
            arguments=bridge_topics,
            output="screen",
        )
    )

    # ── 3. RViz2 (optional) ───────────────────────────────────────────
    if rviz.lower() == "true":
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("n_agents", default_value="2",
                              description="Number of Vector robots"),
        DeclareLaunchArgument("rviz", default_value="false",
                              description="Launch RViz2"),
        OpaqueFunction(function=launch_setup),
    ])
