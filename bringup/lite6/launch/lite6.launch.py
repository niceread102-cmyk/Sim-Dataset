# SPDX-License-Identifier: Apache-2.0
"""lite6 integrated launch — robot_state_publisher + move_group + RViz + ik_node, which receives an ee target and moves.
(the same structure as bringup/m20_35_18d and bringup/ur5e, for the UFactory Lite6 with its gripper of the simulation)

The controller that move_group sends trajectories to is the FollowJointTrajectory action provided by isaacsim/robot_sim.py
(/lite6_traj_controller/follow_joint_trajectory) or run_isaacsim.py (see CONTROLLERS there), and the current robot pose is taken as
it is from the /joint_states they publish — the Isaac Sim side is left untouched. xarm_ros2 is not needed.

Only files inside this Simulation folder are read:
  - config/, rviz/ : read directly from paths relative to this file. Edit them and launch again to apply the change right away.
  - URDF + meshes  : Simulation/assets/lite6 (the same file that is loaded in Isaac Sim).
    The meshes in the URDF use relative paths like '../meshes/visual/link1.stl', but move_group/RViz only read package:// or file://,
    so they are converted to file:// absolute paths when read and passed on (the URDF file itself is not modified).

Preparation:
    source /opt/ros/jazzy/setup.bash
    # Start the Isaac Sim side controller (/lite6_traj_controller) first in a separate terminal (robot: scene.robot of configs/map/map.yaml):
    #   conda activate sim && cd isaacsim && lab robot_sim.py robot:=lite6

Run (from /home/v/project/Simulation):
    ros2 launch bringup/lite6/launch/lite6.launch.py   # robot_state_publisher + move_group + RViz + ik_node
    python3 control/ee_6d.py                           # enter an ee target: x y z [roll pitch yaw]

The /target_pose sent by ee_6d is received by ik_node (control/ik_node.py, parameters in config/ik_node.yaml), which solves IK and moves via the MoveGroup action.
  - position = link_tcp (between the gripper fingers), orientation = link_tcp. Omitted rpy = 180 0 0 (tool pointing down)
Options:
    ik_node:=false   Start no control node at all (when running another control node by hand).
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_param_builder import load_xacro, load_yaml
from launch_ros.actions import Node

BRINGUP_DIR = Path(__file__).resolve().parents[1]  # Simulation/bringup/lite6/
SIM_DIR = BRINGUP_DIR.parents[1]                   # Simulation/
CONFIG_DIR = BRINGUP_DIR / "config"
RVIZ_FILE = BRINGUP_DIR / "rviz" / "demo.rviz"
URDF_FILE = SIM_DIR / "assets" / "robot" / "lite6" / "urdf" / "lite6.urdf"
IK_NODE = SIM_DIR / "control" / "ik_node.py"  # generic, the robot is in config/ik_node.yaml


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
    robot_description_semantic = {"robot_description_semantic": load_xacro(CONFIG_DIR / "lite6.srdf")}
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

    # Control node that receives the ee target (/target_pose) and moves (ik_node:=false leaves it out). It is a script, not a package
    # executable, so it is started with ExecuteProcess (sys.executable = the system python that runs ros2 launch: it has rclpy/scipy).
    controllers = []
    if LaunchConfiguration("ik_node").perform(context).lower() == "true":
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(IK_NODE), "--config", str(CONFIG_DIR / "ik_node.yaml")], name="ik_node", output="screen"))

    return [GroupAction([robot_state_publisher, move_group, rviz, *controllers])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ik_node", default_value="true",
                              description="Also run ik_node.py, which receives the ee target (/target_pose) and moves"),
        OpaqueFunction(function=launch_setup),
    ])
