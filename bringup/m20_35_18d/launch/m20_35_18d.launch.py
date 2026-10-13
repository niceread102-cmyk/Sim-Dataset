# SPDX-License-Identifier: Apache-2.0
"""m20_35_18d integrated launch — robot_state_publisher + move_group + RViz + ik_node, which receives an ee target and moves.
(for bringing everything up at once, like lite6's `ros2 launch xarm_moveit_config moveit_fake_lite6.launch.py add_gripper:=true`)

The official FANUC fanuc_driver has no MoveIt config for this model (it supports CRX only), so this is a hand-made config.
The controller that move_group sends trajectories to is the FollowJointTrajectory action provided by
isaacsim/robot_sim.py (/m20_35_18d_traj_controller/follow_joint_trajectory), and the current robot pose is taken as it is
from the /joint_states published by robot_sim.py — the Isaac Sim side is left untouched.

Only files inside this Simulation folder are read (neither fanuc_ws nor colcon build is needed):
  - config/, rviz/ : read directly from paths relative to this file. Edit them and launch again to apply the change right away.
  - URDF + meshes  : Simulation/control/m20_35_18d (the same files that are loaded in Isaac Sim).
    The meshes in the URDF use relative paths like '../meshes/x.stl', but move_group/RViz only read package:// or file://,
    so they are converted to file:// absolute paths when read and passed on (the URDF file itself is not modified).

Preparation:
    source /opt/ros/jazzy/setup.bash
    # Start the Isaac Sim side controller (/m20_35_18d_traj_controller) first in a separate terminal:
    #   conda activate sim && cd isaacsim && lab robot_sim.py

Run (from /home/v/project/Simulation):
    ros2 launch bringup/m20_35_18d/launch/m20_35_18d.launch.py   # robot_state_publisher + move_group + RViz + ik_node
    python3 control/ee_6d.py                                     # enter an ee target: x y z [roll pitch yaw]

The /target_pose sent by ee_6d is received by ik_node (control/ik_node.py, parameters in config/ik_node.yaml), which solves IK and moves via the MoveGroup action.
  - position = TCP (gripper tip, 0.4693 m from the flange along the tool z), orientation = fanuc_flange. Omitted rpy = 180 0 0 (tool pointing down)
Options:
    sequence:=true   Start control/robot_sequence.py instead of ik_node. It goes to the home pose at start, and when a target (TCP)
                     is sent with ee_6d it runs the sequence approach -> insert -> open gripper -> rise -> home -> place pose
                     (above the basket) -> lower into the basket -> close gripper -> lift -> home. Its parameters are in
                     isaacsim/configs/sequence/robot_sequence.yaml (the basket itself is scene: basket: of isaacsim/configs/map/map.yaml).
    ik_node:=false   Start no control node at all (when running another control node by hand).
ik_node and robot_sequence.py receive the same /target_pose, so running them at the same time puts two MoveGroup goals
into move_group for one target and it dies (segfault). That is why sequence:=true automatically leaves ik_node out.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_param_builder import load_xacro, load_yaml
from launch_ros.actions import Node

BRINGUP_DIR = Path(__file__).resolve().parents[1]  # Simulation/bringup/m20_35_18d/
SIM_DIR = BRINGUP_DIR.parents[1]                   # Simulation/
CONFIG_DIR = BRINGUP_DIR / "config"
RVIZ_FILE = BRINGUP_DIR / "rviz" / "demo.rviz"
URDF_FILE = SIM_DIR / "assets" / "m20_35_18d" / "urdf" / "m20_35_18d.urdf"
IK_NODE = SIM_DIR / "control" / "ik_node.py"  # generic, the robot is in config/ik_node.yaml
SEQUENCE_NODE = SIM_DIR / "control" / "robot_sequence.py"


def load_urdf(urdf_file):
    """Read the URDF and return it as an XML string with the mesh paths converted to file:// absolute paths (the file itself is left unchanged)."""
    root = ET.parse(urdf_file).getroot()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if "://" not in filename:  # '../meshes/visual/j1.dae' -> resolved relative to the URDF file location into an absolute path
            mesh.set("filename", "file://" + str((urdf_file.parent / filename).resolve()))
    return ET.tostring(root, encoding="unicode")


def launch_setup(context, *args, **kwargs):
    # Assemble by hand the parameters that MoveItConfigsBuilder used to build (same keys and values).
    # The Builder only looks for files inside the share folder of a ROS package installed with colcon, so it cannot
    # read config/ of this folder, which is not a package. The same helpers (load_xacro/load_yaml) are used to read them directly instead.
    robot_description = {"robot_description": load_urdf(URDF_FILE)}
    robot_description_semantic = {"robot_description_semantic": load_xacro(CONFIG_DIR / "m20_35_18d.srdf")}
    robot_description_kinematics = {"robot_description_kinematics": load_yaml(CONFIG_DIR / "kinematics.yaml")}
    joint_limits = {"robot_description_planning": load_yaml(CONFIG_DIR / "joint_limits.yaml")}
    planning_pipelines = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": load_yaml(CONFIG_DIR / "ompl_planning.yaml"),
    }
    trajectory_execution = {  # moveit_controller_manager / moveit_simple_controller_manager come from here
        "moveit_manage_controllers": True,
        **load_yaml(CONFIG_DIR / "moveit_controllers.yaml"),
    }
    planning_scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # move_group has died with exit code -11 right after a second MoveGroup goal arrived while a trajectory was still
    # executing (e.g. RViz Plan & Execute pressed during an ee_6d move). RViz then keeps printing "MoveGroup action
    # client/server not ready" and nothing moves until the whole launch is restarted, so it is started again by itself.
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            **robot_description,
            **robot_description_semantic,
            **robot_description_kinematics,
            **planning_pipelines,
            **trajectory_execution,
            **planning_scene_monitor,
            **joint_limits,
        }],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="both",
        arguments=["--display-config", str(RVIZ_FILE)],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            planning_pipelines,
            joint_limits,
        ],
    )

    # Control node that receives the ee target (/target_pose) and moves. Both receive the same /target_pose, so only one is started:
    #   sequence:=true  -> robot_sequence.py (home -> approach/insert -> gripper -> rise -> place -> basket -> home sequence)
    #   otherwise ik_node:=true (default) -> ik_node.py (solves one target with IK and moves)
    # They are scripts, not package executables, so they are started with ExecuteProcess (sys.executable = the system
    # python that runs ros2 launch: it has rclpy/scipy).
    use_sequence = LaunchConfiguration("sequence").perform(context).lower() == "true"
    use_ik_node = LaunchConfiguration("ik_node").perform(context).lower() == "true" and not use_sequence
    controllers = []
    if use_ik_node:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(IK_NODE), "--config", str(CONFIG_DIR / "ik_node.yaml")], name="ik_node", output="screen"))
    if use_sequence:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(SEQUENCE_NODE), "--robot", "m20_35_18d"], name="robot_sequence", output="screen"))

    return [GroupAction([robot_state_publisher, move_group, rviz, *controllers])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ik_node", default_value="true",
                              description="Also run ik_node.py, which receives the ee target (/target_pose) and moves"),
        DeclareLaunchArgument("sequence", default_value="false",
                              description="Run control/robot_sequence.py (home -> approach/insert -> gripper -> rise -> place -> basket -> home sequence) instead of ik_node"),
        OpaqueFunction(function=launch_setup),
    ])
