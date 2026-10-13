# SPDX-License-Identifier: Apache-2.0
"""Integrated launch of ANY robot — robot_state_publisher + move_group + RViz + ik_node, which receives an ee target and moves.
(one launch instead of bringup/<robot>/launch/<robot>.launch.py: the robot is only an argument and changes the paths)

    ros2 launch robot.launch.py robot:=ur5e                                       (from this folder)
    ros2 launch bringup/integration/launch/robot.launch.py robot:=ur5e            (from /home/v/project/Simulation)

robot:=<name> picks the files of that robot, the same ones its own launch used:
  - bringup/<name>/config/ : <name>.srdf, kinematics, joint_limits (MoveIt speed limits), ompl_planning, moveit_controllers
                             (the controller of the sim), ik_node.yaml (base frame, ee_link, tcp_offset, IK seed, speed scaling)
  - bringup/<name>/rviz/   : RViz layout
  - assets/<name>/urdf/<name>.urdf (or <name>.urdf in another assets/*/urdf/) + meshes: the same file that is loaded in Isaac Sim
The Isaac Sim side reads isaacsim/configs/robots/<name>.yaml (stiffness, damping, speed, controller; the URDF effort is the force
limit): select the same robot there with `robot:` in isaacsim/configs/robots/select_robot.yaml or `lab robot_sim.py robot:=<name>`.
A new robot = those three folders/files (copy an existing bringup/<name>/ and edit it).

Preparation:
    source /opt/ros/jazzy/setup.bash
    # Start the Isaac Sim side controller of the robot first in a separate terminal:
    #   conda activate sim && cd isaacsim && lab robot_sim.py

The /target_pose sent by ee_6d (python3 control/ee_6d.py: x y z [roll pitch yaw]) is received by ik_node (control/ik_node.py,
parameters in the robot's config/ik_node.yaml), which solves IK and moves via the MoveGroup action.
  - position = TCP (ee_link + tcp_offset of ik_node.yaml), orientation = ee_link. Omitted rpy = 180 0 0 (tool pointing down)
Options:
    robot:=<name>    default: `robot:` of isaacsim/configs/robots/select_robot.yaml
    sequence:=true   Start control/robot_sequence.py instead of ik_node (home -> approach -> insert -> open gripper -> rise -> home ->
                     place pose -> lower into the basket -> close gripper -> lift -> home). Its parameters are
                     isaacsim/configs/sequence/robot_sequence.yaml (+ <robot>.yaml for what differs per robot).
    ik_node:=false   Start no control node at all (when running another control node by hand).
    rviz:=false      No RViz window.
ik_node and robot_sequence.py receive the same /target_pose, so running them at the same time puts two MoveGroup goals
into move_group for one target and it dies (segfault). That is why sequence:=true automatically leaves ik_node out.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_param_builder import load_xacro, load_yaml
from launch_ros.actions import Node

BRINGUP_DIR = Path(__file__).resolve().parents[2]  # Simulation/bringup/
SIM_DIR = BRINGUP_DIR.parent                       # Simulation/
SELECT_FILE = SIM_DIR / "isaacsim" / "configs" / "robots" / "select_robot.yaml"
IK_NODE = SIM_DIR / "control" / "ik_node.py"       # generic, the robot is in config/ik_node.yaml
SEQUENCE_NODE = SIM_DIR / "control" / "robot_sequence.py"


def robot_name(choice):
    """robot:=<name>; empty = the robot of select_robot.yaml (the one `lab robot_sim.py` starts)."""
    name = choice or str((yaml.safe_load(open(SELECT_FILE)) or {}).get("robot") or "")
    robots = sorted(d.name for d in BRINGUP_DIR.iterdir() if (d / "config").is_dir())
    if not name:
        raise ValueError(f"no robot: pass robot:=<name> or set `robot:` in {SELECT_FILE}; robots: {robots}")
    if name not in robots:
        raise ValueError(f"no bringup/{name}/config; robots: {robots}")
    return name


def find_urdf(robot):
    """assets/<robot>/urdf/<robot>.urdf, or <robot>.urdf in another assets/*/urdf/ folder (several URDFs of one robot share a folder)."""
    own = SIM_DIR / "assets" / robot / "urdf" / f"{robot}.urdf"
    return own if own.is_file() else next(iter(sorted((SIM_DIR / "assets").glob(f"*/urdf/{robot}.urdf"))), own)


def load_urdf(urdf_file):
    """Read the URDF and return it as an XML string with the mesh paths converted to file:// absolute paths (the file itself is left unchanged)."""
    root = ET.parse(urdf_file).getroot()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if "://" not in filename:  # '../meshes/visual/j1.dae' -> resolved relative to the URDF file location into an absolute path
            mesh.set("filename", "file://" + str((urdf_file.parent / filename).resolve()))
    return ET.tostring(root, encoding="unicode")


def launch_setup(context, *args, **kwargs):
    robot = robot_name(LaunchConfiguration("robot").perform(context))
    config_dir = BRINGUP_DIR / robot / "config"
    rviz_file = BRINGUP_DIR / robot / "rviz" / "demo.rviz"
    urdf_file = find_urdf(robot)

    # Assemble by hand the parameters that MoveItConfigsBuilder used to build (same keys and values).
    # The Builder only looks for files inside the share folder of a ROS package installed with colcon, so it cannot
    # read config/ of this folder, which is not a package. The same helpers (load_xacro/load_yaml) are used to read them directly instead.
    robot_description = {"robot_description": load_urdf(urdf_file)}
    robot_description_semantic = {"robot_description_semantic": load_xacro(config_dir / f"{robot}.srdf")}
    robot_description_kinematics = {"robot_description_kinematics": load_yaml(config_dir / "kinematics.yaml")}
    joint_limits = {"robot_description_planning": load_yaml(config_dir / "joint_limits.yaml")}
    planning_pipelines = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": load_yaml(config_dir / "ompl_planning.yaml"),
    }
    trajectory_execution = {  # moveit_controller_manager / moveit_simple_controller_manager come from here
        "moveit_manage_controllers": True,
        **load_yaml(config_dir / "moveit_controllers.yaml"),
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
        arguments=["--display-config", str(rviz_file)],
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
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(IK_NODE), "--config", str(config_dir / "ik_node.yaml")], name="ik_node", output="screen"))
    if use_sequence:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(SEQUENCE_NODE), "--robot", robot], name="robot_sequence", output="screen"))

    show_rviz = LaunchConfiguration("rviz").perform(context).lower() == "true"
    return [GroupAction([LogInfo(msg=f"robot {robot}: {urdf_file.relative_to(SIM_DIR)} + bringup/{robot}/config"),
                         robot_state_publisher, move_group, *([rviz] if show_rviz else []), *controllers])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="",
                              description="Robot name = its folder bringup/<name>/ and assets/<name>/; empty = the robot of select_robot.yaml"),
        DeclareLaunchArgument("ik_node", default_value="true",
                              description="Also run ik_node.py, which receives the ee target (/target_pose) and moves"),
        DeclareLaunchArgument("sequence", default_value="false",
                              description="Run control/robot_sequence.py (home -> approach/insert -> gripper -> rise -> place -> basket -> home sequence) instead of ik_node"),
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz"),
        OpaqueFunction(function=launch_setup),
    ])
