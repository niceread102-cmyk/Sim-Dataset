# SPDX-License-Identifier: Apache-2.0
"""Integrated launch of ANY robot — robot_state_publisher + move_group + RViz + ik_node, which receives an ee target and moves.
(one launch instead of bringup/<robot>/launch/<robot>.launch.py: the robot is only an argument and changes the paths)

    ros2 launch robot.launch.py robot:=ur5e                                       (from this folder)
    ros2 launch bringup/integration/launch/robot.launch.py robot:=ur5e            (from /home/v/project/Simulation)

robot:=<name> picks the files of that robot, the same ones its own launch used:
  - bringup/<name>/config/ : <name>.srdf, kinematics, joint_limits (MoveIt speed limits), ompl_planning, moveit_controllers
                             (the controller of the sim), ik_node.yaml (base frame, ee_link, tcp_offset, IK seed, speed scaling)
  - bringup/<name>/rviz/   : RViz layout
  - assets/robot/<name>/urdf/<name>.urdf (or <name>.urdf in another assets/robot/*/urdf/) + meshes: the same file that is loaded in Isaac Sim
The Isaac Sim side reads isaacsim/configs/robots/<name>.yaml (stiffness, damping, speed, controller; the URDF effort is the force
limit): select the same robot there with `scene.robot` in isaacsim/configs/map/map.yaml or `lab robot_sim.py robot:=<name>`.
A new robot = those three folders/files (copy an existing bringup/<name>/ and edit it).

World frame: MoveIt works in the frame `world` of Isaac Sim. The pose of the robot base in it (pos_m / rpy_deg of isaacsim/configs/robots/<name>.yaml,
the calibration world -> robot base) is put into the URDF given to MoveIt, so /target_pose, the bin, basket and floor are world coordinates
for a robot at any x y z roll pitch yaw (ik_node.yaml: base_frame: world).

Preparation:
    source /opt/ros/jazzy/setup.bash
    # Start the Isaac Sim side controller of the robot first in a separate terminal:
    #   conda activate sim && cd isaacsim && lab robot_sim.py

The /target_pose sent by ee_6d (python3 control/ee_6d.py: x y z [roll pitch yaw]) is received by ik_node (control/ik_node.py,
parameters in the robot's config/ik_node.yaml), which solves IK and moves via the MoveGroup action.
  - position = TCP (ee_link + tcp_offset of ik_node.yaml), orientation = ee_link. Omitted rpy = 180 0 0 (tool pointing down)
Options:
    robot:=<name>    default: `scene.robot` of isaacsim/configs/map/map.yaml
    sequence:=<name> Start control/robot_sequence.py with the sequence isaacsim/configs/sequence/<name>.yaml instead of ik_node (home ->
                     approach -> insert -> open gripper -> rise -> home -> place pose -> close gripper -> home). The sequence is made for
                     one robot (its `robot:` has to be the robot:= of the launch) and has the poses and the trajectory of that robot;
                     the rest is robot_sequence.yaml. Nothing of the scene is read. Without sequence:= only ik_node runs (the robot
                     moves to the /target_pose you send); sequence:=true = the sequence with the name of the robot.
    ik_node:=false   Start no control node at all (when running another control node by hand).
    laser_marker:=false  With sequence:=<name> the laser marker cycle (isaacsim/machine/laser_marker.py) runs too: when the robot has put a turned-over plate on
                     the flipper and is at home again, the marker flips, holds, lets go and moves its belt one pitch. false = not started.
    rviz:=false      No RViz window.
ik_node and robot_sequence.py receive the same /target_pose, so running them at the same time puts two MoveGroup goals
into move_group for one target and it dies (segfault). That is why sequence:=<name> automatically leaves ik_node out.
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_param_builder import load_xacro, load_yaml
from launch_ros.actions import Node
from scipy.spatial.transform import Rotation

BRINGUP_DIR = Path(__file__).resolve().parents[2]  # Simulation/bringup/
SIM_DIR = BRINGUP_DIR.parent                       # Simulation/
ROBOTS_DIR = SIM_DIR / "isaacsim" / "configs" / "robots"
MAP_FILE = SIM_DIR / "isaacsim" / "configs" / "map" / "map.yaml"
IK_NODE = SIM_DIR / "control" / "ik_node.py"       # generic, the robot is in config/ik_node.yaml
SEQUENCE_NODE = SIM_DIR / "control" / "robot_sequence.py"
SEQUENCE_DIR = SIM_DIR / "isaacsim" / "configs" / "sequence"
CONVEYOR_STEP = SIM_DIR / "isaacsim" / "machine" / "conveyor.py"        # moves the chain of the conveyor one step when the left hands of the laser marker have let go of their plates
LASER_MARKER = SIM_DIR / "isaacsim" / "machine" / "laser_marker.py"   # the flipper of the laser marker: runs when the sequence has put a turned-over plate on it


def laser_marker_enabled(context):
    """laser_marker:=false leaves isaacsim/machine/laser_marker.py out (it only does something with the laser marker of the map in the simulation)"""
    return LaunchConfiguration("laser_marker").perform(context).lower() != "false"


def robot_name(choice):
    """robot:=<name>; empty = scene.robot of map.yaml (the one `lab robot_sim.py` starts)."""
    name = choice or str(((yaml.safe_load(open(MAP_FILE)) or {}).get("scene") or {}).get("robot") or "")
    robots = sorted(d.name for d in BRINGUP_DIR.iterdir() if (d / "config").is_dir())
    if not name:
        raise ValueError(f"no robot: pass robot:=<name> or set `scene.robot` in {MAP_FILE}; robots: {robots}")
    if name not in robots:
        raise ValueError(f"no bringup/{name}/config; robots: {robots}")
    return name


def find_urdf(robot):
    """assets/robot/<robot>/urdf/<robot>.urdf, or <robot>.urdf in another assets/robot/*/urdf/ folder (several URDFs of one robot share a folder)."""
    own = SIM_DIR / "assets" / "robot" / robot / "urdf" / f"{robot}.urdf"
    return own if own.is_file() else next(iter(sorted((SIM_DIR / "assets" / "robot").glob(f"*/urdf/{robot}.urdf"))), own)


def robot_pose(robot):
    """Where the robot base stands in the Isaac Sim world: pos_m (m) and rpy_deg of isaacsim/configs/robots/<robot>.yaml, the pose
    robot_sim.py places it at (= the calibration world -> robot base, any x y z roll pitch yaw). Zeros without the file / keys."""
    f = ROBOTS_DIR / f"{robot}.yaml"
    cfg = (yaml.safe_load(open(f)) or {}) if f.is_file() else {}
    pos = np.array(cfg.get("pos_m") or [0, 0, 0], float)
    pf = ((yaml.safe_load(open(MAP_FILE)) or {}).get("scene") or {}).get("platform")     # a robot on the platform stands on top of it,
    if pf and pf.get("size_m"):                                                          # like in robot_sim.py: pos_m z counts from the top
        px, py, pz = (float(v) for v in pf.get("pos_m", [0, 0, 0]))
        sx, sy, sz = (float(v) for v in pf["size_m"])
        yaw = math.radians(float((pf.get("rpy_deg") or [0, 0, 0])[2]))
        dx, dy = pos[0] - px, pos[1] - py
        u, v = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        if abs(u) <= sx / 2 and abs(v) <= sy / 2:
            pos[2] += pz + sz
    return pos, np.array(cfg.get("rpy_deg") or [0, 0, 0], float)


def place_in_world(root, pos, rpy_deg):
    """Moves the robot of the URDF `root` to pos / rpy_deg of the world (roll about X, pitch about Y, yaw about Z, all fixed axes =
    URDF and robot_sim.py): the origin of the fixed joint that holds the robot below the empty root link (`world`) becomes that pose
    (composed with the origin it had). Then MoveIt's frame `world` is the one of Isaac Sim: target poses, bin, basket and floor are world
    coordinates, whatever the pose of the robot. Returns False when the URDF has no such root (the robot then stays where its root is)."""
    links = {ln.get("name"): ln for ln in root.findall("link")}
    joints = root.findall("joint")
    top = next(n for n in links if n not in {j.find("child").get("link") for j in joints})
    held = [j for j in joints if j.find("parent").get("link") == top]
    if any(links[top].find(t) is not None for t in ("visual", "collision", "inertial")) or len(held) != 1 or held[0].get("type") != "fixed":
        return False
    o = held[0].find("origin")
    if o is None:
        o = ET.SubElement(held[0], "origin")
    R0 = Rotation.from_euler("xyz", [float(v) for v in (o.get("rpy") or "0 0 0").split()])
    p0 = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
    R = Rotation.from_euler("xyz", rpy_deg, degrees=True)
    o.set("xyz", " ".join(repr(float(v)) for v in R.apply(p0) + pos))
    o.set("rpy", " ".join(repr(float(v)) for v in (R * R0).as_euler("xyz")))
    return True


def load_urdf(urdf_file, pose=None):
    """Read the URDF and return it as an XML string with the mesh paths converted to file:// absolute paths (the file itself is left unchanged).
    pose = (pos_m, rpy_deg) of the robot in the world, see place_in_world; None or zeros = as the URDF says."""
    root = ET.parse(urdf_file).getroot()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if "://" not in filename:  # '../meshes/visual/j1.dae' -> resolved relative to the URDF file location into an absolute path
            mesh.set("filename", "file://" + str((urdf_file.parent / filename).resolve()))
    if pose is not None and (np.any(pose[0]) or np.any(pose[1])) and not place_in_world(root, *pose):
        print(f"[WARN] {urdf_file.name}: no empty root link with one fixed joint to the robot: the pose {pose[0].tolist()} m / "
              f"{pose[1].tolist()} deg is not applied in MoveIt", flush=True)
    return ET.tostring(root, encoding="unicode")


def launch_setup(context, *args, **kwargs):
    robot = robot_name(LaunchConfiguration("robot").perform(context))
    config_dir = BRINGUP_DIR / robot / "config"
    rviz_file = BRINGUP_DIR / robot / "rviz" / "demo.rviz"
    urdf_file = find_urdf(robot)

    # Assemble by hand the parameters that MoveItConfigsBuilder used to build (same keys and values).
    # The Builder only looks for files inside the share folder of a ROS package installed with colcon, so it cannot
    # read config/ of this folder, which is not a package. The same helpers (load_xacro/load_yaml) are used to read them directly instead.
    pose = robot_pose(robot)
    robot_description = {"robot_description": load_urdf(urdf_file, pose)}
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
    #   sequence:=<name> -> robot_sequence.py with that sequence (home -> approach/insert -> gripper -> rise -> place -> home)
    #   otherwise ik_node:=true (default) -> ik_node.py (solves one target with IK and moves)
    # They are scripts, not package executables, so they are started with ExecuteProcess (sys.executable = the system
    # python that runs ros2 launch: it has rclpy/scipy).
    seq = LaunchConfiguration("sequence").perform(context).strip()
    seq = "" if seq.lower() in ("", "false", "none") else robot if seq.lower() == "true" else seq
    if seq:
        names = sorted(f.stem for f in SEQUENCE_DIR.glob("*.yaml") if f.name != "robot_sequence.yaml")
        if seq not in names:
            raise ValueError(f"sequence:={seq}: {SEQUENCE_DIR / (seq + '.yaml')} not found; sequences: {', '.join(names) or '-'}")
        declared = (yaml.safe_load(open(SEQUENCE_DIR / f"{seq}.yaml")) or {}).get("robot")
        if declared and declared != robot:
            raise ValueError(f"sequence:={seq} is the sequence of the robot {declared}, but robot:={robot}")
    use_sequence = bool(seq)
    use_ik_node = LaunchConfiguration("ik_node").perform(context).lower() == "true" and not use_sequence
    controllers = []
    if use_ik_node:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(IK_NODE), "--config", str(config_dir / "ik_node.yaml")], name="ik_node", output="screen"))
    if use_sequence:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(SEQUENCE_NODE), "--robot", robot, "--sequence", seq], name="robot_sequence", output="screen"))
        if laser_marker_enabled(context):
            controllers.append(ExecuteProcess(cmd=[sys.executable, str(LASER_MARKER), "--sequence", seq], name="laser_marker", output="screen"))
            controllers.append(ExecuteProcess(cmd=[sys.executable, str(CONVEYOR_STEP)], name="conveyor_step", output="screen"))

    show_rviz = LaunchConfiguration("rviz").perform(context).lower() == "true"
    return [GroupAction([LogInfo(msg=f"robot {robot}: {urdf_file.relative_to(SIM_DIR)} + bringup/{robot}/config, base in the world at "
                                     f"{pose[0].tolist()} m, rpy {pose[1].tolist()} deg (isaacsim/configs/robots/{robot}.yaml)"),
                         robot_state_publisher, move_group, *([rviz] if show_rviz else []), *controllers])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="",
                              description="Robot name = its folder bringup/<name>/ and assets/robot/<name>/; empty = scene.robot of map.yaml"),
        DeclareLaunchArgument("ik_node", default_value="true",
                              description="Also run ik_node.py, which receives the ee target (/target_pose) and moves"),
        DeclareLaunchArgument("sequence", default_value="",
                              description="Name of a sequence (isaacsim/configs/sequence/<name>.yaml, made for one robot): run control/robot_sequence.py with it instead of ik_node. Empty = only ik_node"),
        DeclareLaunchArgument("laser_marker", default_value="true",
                              description="With sequence:=<name>: also run isaacsim/machine/laser_marker.py (flip, hold and move the belt of the laser marker after a plate was placed) and conveyor.py (one step of the conveyor chain after the left hands let go). false = leave them out"),
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz"),
        OpaqueFunction(function=launch_setup),
    ])
