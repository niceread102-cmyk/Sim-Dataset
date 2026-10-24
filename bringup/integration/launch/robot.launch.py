# SPDX-License-Identifier: Apache-2.0
"""Integrated launch of ANY robot in ANY map — robot_state_publisher + move_group + RViz + ik_node, which receives an ee target and moves.
(the robot and the map are only arguments: they change the paths)

    ros2 launch robot.launch.py robot:=ur5e                                       (from this folder; no map: the robot at the origin)
    ros2 launch robot.launch.py map:=SIDE_PLATE_MAP [robot:=<name>]               (the robot of the map, where the map puts it)
    ros2 launch bringup/integration/launch/robot.launch.py robot:=ur5e            (from /home/v/project/Simulation)

robot:=<name> picks the files of that robot:
  - bringup/<name>/config/ : <name>.srdf, kinematics, joint_limits (MoveIt speed limits), ompl_planning, moveit_controllers
                             (the controller of the sim), ik_node.yaml (base frame, ee_link, tcp_offset, IK seed)
  - bringup/<name>/rviz/   : RViz layout
  - assets/robot/<name>/urdf/<name>.urdf (or <name>.urdf in another assets/robot/*/urdf/) + meshes: the same file that is loaded in Isaac Sim
  - isaacsim/configs/robot/<name>.yaml: the speed of the motions (velocity_scaling, acceleration_scaling) and narrower joint limits
    (joint_limits_deg: MoveIt plans and solves the IK inside them); the Isaac Sim side reads the rest of it (stiffness, damping, speed, controller)
A new robot = python3 isaacsim/make_robot_config.py robot:=<name> (writes these files from the URDF).
map:=<map> (isaacsim/configs/map/<map>.yaml, isaacsim/sim_config.py reads it) gives the robot when robot:= is not given (scene.robot), where the
robot stands (scene.robot_pose), the sequence of sequence:=true (scene.sequence) and the machines whose cycle runs with a sequence (scene.machines).
Start Isaac Sim with the same map: lab robot_sim.py map:=<map> [robot:=<name>].

World frame: MoveIt works in the frame `world` of Isaac Sim. The pose of the robot base in it (scene.robot_pose of the map times pos_m / rpy_deg
of the robot yaml, which are 0) is put into the URDF given to MoveIt, so /target_pose, the bin, basket and floor are world coordinates
for a robot at any x y z roll pitch yaw (ik_node.yaml: base_frame: world).

Preparation:
    source /opt/ros/jazzy/setup.bash
    # Start the Isaac Sim side controller of the robot first in a separate terminal:
    #   conda activate sim && cd isaacsim && lab robot_sim.py robot:=<name> [map:=<map>]

The /target_pose sent by ee_6d (python3 control/ee_6d.py: x y z [roll pitch yaw]) is received by ik_node (control/ik_node.py,
parameters in the robot's config/ik_node.yaml and configs/robot/<name>.yaml), which solves IK and moves via the MoveGroup action.
  - position = TCP (ee_link + tcp_offset of ik_node.yaml), orientation = ee_link. Omitted rpy = 180 0 0 (tool pointing down)
Options:
    robot:=<name>    default: the robot of the running robot_sim.py, else scene.robot of the map
    di:=<id>         the robot of the map with this ROS domain id (scene.robots: - {name, di, pose}): MoveIt and the control nodes of this launch run in
                     that domain, so every robot of the map has a launch of its own (ros2 launch robot.launch.py di:=0, ... di:=1)
    map:=<map>       default: the map of the running robot_sim.py (isaacsim/assets_cache/current_sim.yaml), none = the robot at the origin
    sequence:=<name> Start control/robot_sequence.py with the sequence isaacsim/configs/sequence/<name>.yaml instead of ik_node (home ->
                     approach -> insert -> open gripper -> rise -> home -> place pose -> close gripper -> home). A sequence may be made for
                     one robot (its `robot:` has to be the robot:= of the launch); the rest is robot_sequence.yaml. Without sequence:= only
                     ik_node runs (the robot moves to the /target_pose you send); sequence:=true = scene.sequence of the map, else the
                     sequence with the name of the robot.
    machines:=false  With a sequence the cycle of every machine of the map that has one (isaacsim/machine/<name>.py, e.g. the laser marker:
                     flip, hold, let go, move the belt; the conveyor: one step of the chain) runs too. false = none of them.
    ik_node:=false   Start no control node at all (when running another control node by hand).
    rviz:=false      No RViz window.
ik_node and robot_sequence.py receive the same /target_pose, so running them at the same time puts two MoveGroup goals
into move_group for one target and it dies (segfault). That is why sequence:=<name> automatically leaves ik_node out.
"""
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_param_builder import load_xacro, load_yaml
from launch_ros.actions import Node
from scipy.spatial.transform import Rotation

BRINGUP_DIR = Path(__file__).resolve().parents[2]  # Simulation/bringup/
SIM_DIR = BRINGUP_DIR.parent                       # Simulation/
sys.path.insert(0, str(SIM_DIR / "isaacsim"))
import sim_config  # noqa: E402  the maps, robots and machines of isaacsim/configs/

IK_NODE = SIM_DIR / "control" / "ik_node.py"       # generic, the robot is in config/ik_node.yaml + configs/robot/<robot>.yaml
SEQUENCE_NODE = SIM_DIR / "control" / "robot_sequence.py"
SEQUENCE_DIR = Path(sim_config.SEQUENCE_DIR)


def robot_name(choice, scene):
    """robot:=<name>; empty = scene.robot of the map (the one `lab robot_sim.py map:=<map>` starts)."""
    robots = sorted(d.name for d in BRINGUP_DIR.iterdir() if (d / "config").is_dir())
    name = sim_config.pick_robot(choice, scene)
    if name not in robots:
        raise ValueError(f"no bringup/{name}/config; robots: {robots}")
    return name


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
    """Read the URDF and return it as an XML string with the mesh paths as absolute file:// paths (relative, file:// and package:// paths, found
    like robot_sim.py finds them; the file itself is left unchanged). pose = (pos_m, rpy_deg) of the robot in the world, see place_in_world;
    None or zeros = as the URDF says."""
    root = ET.parse(urdf_file).getroot()
    for mesh in root.iter("mesh"):
        found = sim_config.find_mesh(mesh.get("filename"), str(urdf_file.parent))
        if found:   # '../meshes/visual/j1.dae' -> an absolute path (move_group / RViz read package:// or file:// only)
            mesh.set("filename", "file://" + str(Path(found).resolve()))
    if pose is not None and (np.any(pose[0]) or np.any(pose[1])) and not place_in_world(root, *pose):
        print(f"[WARN] {urdf_file.name}: no empty root link with one fixed joint to the robot: the pose {pose[0].tolist()} m / "
              f"{pose[1].tolist()} deg is not applied in MoveIt", flush=True)
    return ET.tostring(root, encoding="unicode")


def planning_limits(config_dir, robot_cfg):
    """robot_description_planning: config/joint_limits.yaml (velocity and acceleration of the URDF) with the narrower position range of every joint
    in joint_limits_deg of the robot yaml (MoveIt's robot model then plans, solves the IK and checks the start state inside it)"""
    limits = load_yaml(config_dir / "joint_limits.yaml")
    for joint, (lo, hi) in (sim_config.robot_motion(robot_cfg)["joint_limits_deg"] or {}).items():
        entry = limits.setdefault("joint_limits", {}).setdefault(joint, {})
        entry.update(has_position_limits=True, min_position=math.radians(float(lo)), max_position=math.radians(float(hi)))
    return limits


def launch_setup(context, *args, **kwargs):
    given = LaunchConfiguration("map").perform(context).strip()
    user_robot = LaunchConfiguration("robot").perform(context).strip() or None
    di = LaunchConfiguration("di").perform(context).strip() or None
    robot_arg, map_name = sim_config.with_current(user_robot, given or None)    # without map:= the map (and robot) of the robot_sim.py running now
    if di is not None and user_robot is None:
        robot_arg = None                         # di:= picks the robot of the map
    map_name = map_name or ""
    scene = sim_config.scene_of(map_name)
    entry = sim_config.robot_entry(scene, robot_arg, di)   # {name, di, pose}: the robot with ROS domain di (di:=), or robot:=, or the first of the map
    robot = robot_name(entry["name"], scene)
    di = di if di is not None else entry.get("di")       # every node of this launch runs in the ROS domain of the robot
    robot_cfg = sim_config.load_robot(robot)
    config_dir = BRINGUP_DIR / robot / "config"
    rviz_file = BRINGUP_DIR / robot / "rviz" / "demo.rviz"
    urdf_file = Path(sim_config.find_urdf(robot))

    # Assemble by hand the parameters that MoveItConfigsBuilder used to build (same keys and values).
    # The Builder only looks for files inside the share folder of a ROS package installed with colcon, so it cannot
    # read config/ of this folder, which is not a package. The same helpers (load_xacro/load_yaml) are used to read them directly instead.
    pose = tuple(np.array(v, float) for v in sim_config.robot_pose(robot_cfg, {**scene, "robot_pose": entry["pose"]}))   # where the robot stands in the map (the same pose as in robot_sim.py)
    robot_description = {"robot_description": load_urdf(urdf_file, pose)}
    robot_description_semantic = {"robot_description_semantic": load_xacro(config_dir / f"{robot}.srdf")}
    robot_description_kinematics = {"robot_description_kinematics": load_yaml(config_dir / "kinematics.yaml")}
    joint_limits = {"robot_description_planning": planning_limits(config_dir, robot_cfg)}
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
    seq = "" if seq.lower() in ("", "false", "none") else str(scene.get("sequence") or robot) if seq.lower() == "true" else seq
    if seq:
        names = sorted(f.stem for f in SEQUENCE_DIR.glob("*.yaml") if f.name != "robot_sequence.yaml")
        if seq not in names:
            raise ValueError(f"sequence:={seq}: {SEQUENCE_DIR / (seq + '.yaml')} not found; sequences: {', '.join(names) or '-'}")
        declared = (yaml.safe_load(open(SEQUENCE_DIR / f"{seq}.yaml")) or {}).get("robot")
        if declared and declared != robot:
            raise ValueError(f"sequence:={seq} is the sequence of the robot {declared}, but robot:={robot}")
    use_sequence = bool(seq)
    use_ik_node = LaunchConfiguration("ik_node").perform(context).lower() == "true" and not use_sequence
    robot_yaml = sim_config.robot_config_path(robot)
    controllers = []
    if use_ik_node:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(IK_NODE), "--config", str(config_dir / "ik_node.yaml"), "--robot", robot_yaml],
                                          name="ik_node", output="screen"))
    machines = []
    if use_sequence:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(SEQUENCE_NODE), "--robot", robot, "--sequence", seq] + (["--map", map_name] if map_name else []),
                                          name="robot_sequence", output="screen"))
        if LaunchConfiguration("machines").perform(context).lower() != "false":   # the cycles of the machines of the map (isaacsim/machine/<name>.py)
            for m in scene.get("machines") or []:
                script = os.path.join(sim_config.MACHINE_SCRIPTS, f"{m['name']}.py")
                if os.path.isfile(script):
                    machines.append(m["name"])
                    controllers.append(ExecuteProcess(cmd=[sys.executable, script], name=m["name"], output="screen"))

    show_rviz = LaunchConfiguration("rviz").perform(context).lower() == "true"
    domain = [SetEnvironmentVariable("ROS_DOMAIN_ID", str(int(di)))] if di is not None else []
    return [GroupAction([*domain, LogInfo(msg=f"robot {robot}{f' (ROS domain {int(di)})' if di is not None else ''}: {urdf_file.relative_to(SIM_DIR)} + bringup/{robot}/config, "
                                     f"{'map ' + map_name if map_name else 'no map'}{'' if given else ' (of the running robot_sim.py)'}: base in the world at {pose[0].tolist()} m, rpy {pose[1].tolist()} deg"
                                     + (f", sequence {seq}" if seq else "") + (f", machine cycles {machines}" if machines else "")),
                         robot_state_publisher, move_group, *([rviz] if show_rviz else []), *controllers])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="",
                              description="Robot name = its folder bringup/<name>/ and assets/robot/<name>/; empty = scene.robot of the map"),
        DeclareLaunchArgument("di", default_value="",
                              description="ROS domain id of the robot (scene.robots of the map: {name, di, pose}): picks that robot and runs every node of this launch in its domain"),
        DeclareLaunchArgument("map", default_value="",
                              description="Map name = isaacsim/configs/map/<map>.yaml (its robot, where the robot stands, its sequence and machines); empty = no map"),
        DeclareLaunchArgument("ik_node", default_value="true",
                              description="Also run ik_node.py, which receives the ee target (/target_pose) and moves"),
        DeclareLaunchArgument("sequence", default_value="",
                              description="Name of a sequence (isaacsim/configs/sequence/<name>.yaml): run control/robot_sequence.py with it instead of ik_node; "
                                          "true = scene.sequence of the map. Empty = only ik_node"),
        DeclareLaunchArgument("machines", default_value="true",
                              description="With a sequence: also run the cycle of every machine of the map that has one (isaacsim/machine/<name>.py: the laser marker, the conveyor). false = none"),
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz"),
        OpaqueFunction(function=launch_setup),
    ])
