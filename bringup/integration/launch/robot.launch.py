"""robot.launch.py: one launch for any robot in any map: robot_state_publisher + move_group + RViz + control/ik_node.py,
and with a sequence also control/robot_sequence.py and the cycles of the machines of the map (isaacsim/machine/<name>.py).

    ros2 launch robot.launch.py [robot:=<name>] [map:=<map>] [sequence:=<name>|true] [id:=<domain>] [ik_node:=false] [machines:=false] [rviz:=false]

The robot stands where the map puts it (scene.robot_pose); everything robot specific is in bringup/<robot>/config and isaacsim/configs/robot/<robot>.yaml.
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

BRINGUP_DIR = Path(__file__).resolve().parents[2]
SIM_DIR = BRINGUP_DIR.parent
sys.path.insert(0, str(SIM_DIR / "isaacsim"))
import sim_config

IK_NODE = SIM_DIR / "control" / "ik_node.py"
SEQUENCE_NODE = SIM_DIR / "control" / "robot_sequence.py"
SEQUENCE_DIR = Path(sim_config.SEQUENCE_DIR)


def robot_name(choice, scene):
    robots = sorted(d.name for d in BRINGUP_DIR.iterdir() if (d / "config").is_dir())
    name = sim_config.pick_robot(choice, scene)
    if name not in robots:
        raise ValueError(f"no bringup/{name}/config; robots: {robots}")
    return name


def place_in_world(root, pos, rpy_deg):
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
    root = ET.parse(urdf_file).getroot()
    for mesh in root.iter("mesh"):
        found = sim_config.find_mesh(mesh.get("filename"), str(urdf_file.parent))
        if found:
            mesh.set("filename", "file://" + str(Path(found).resolve()))
    if pose is not None and (np.any(pose[0]) or np.any(pose[1])) and not place_in_world(root, *pose):
        print(f"[WARN] {urdf_file.name}: no empty root link with one fixed joint to the robot: the pose {pose[0].tolist()} m / "
              f"{pose[1].tolist()} deg is not applied in MoveIt", flush=True)
    return ET.tostring(root, encoding="unicode")


def planning_limits(config_dir, robot_cfg):
    limits = load_yaml(config_dir / "joint_limits.yaml")
    for joint, (lo, hi) in (sim_config.robot_motion(robot_cfg)["joint_limits_deg"] or {}).items():
        entry = limits.setdefault("joint_limits", {}).setdefault(joint, {})
        entry.update(has_position_limits=True, min_position=math.radians(float(lo)), max_position=math.radians(float(hi)))
    return limits


def launch_setup(context, *args, **kwargs):
    given = LaunchConfiguration("map").perform(context).strip()
    user_robot = LaunchConfiguration("robot").perform(context).strip() or None
    id = LaunchConfiguration("id").perform(context).strip() or None
    robot_arg, map_name = sim_config.with_current(user_robot, given or None)
    if id is not None and user_robot is None:
        robot_arg = None
    map_name = map_name or ""
    scene = sim_config.scene_of(map_name)
    entry = sim_config.robot_entry(scene, robot_arg, id)
    robot = robot_name(entry["name"], scene)
    id = id if id is not None else entry.get("id")
    robot_cfg = sim_config.load_robot(robot)
    config_dir = BRINGUP_DIR / robot / "config"
    rviz_file = BRINGUP_DIR / robot / "rviz" / "demo.rviz"
    urdf_file = Path(sim_config.find_urdf(robot))

    pose = tuple(np.array(v, float) for v in sim_config.robot_pose(robot_cfg, {**scene, "robot_pose": entry["pose"]}))
    robot_description = {"robot_description": load_urdf(urdf_file, pose)}
    robot_description_semantic = {"robot_description_semantic": load_xacro(config_dir / f"{robot}.srdf")}
    robot_description_kinematics = {"robot_description_kinematics": load_yaml(config_dir / "kinematics.yaml")}
    joint_limits = {"robot_description_planning": planning_limits(config_dir, robot_cfg)}
    planning_pipelines = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": load_yaml(config_dir / "ompl_planning.yaml"),
    }
    trajectory_execution = {
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
    use_ik_node = LaunchConfiguration("ik_node").perform(context).lower() == "true"
    robot_yaml = sim_config.robot_config_path(robot)
    controllers = []
    if use_ik_node:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(IK_NODE), "--config", str(config_dir / "ik_node.yaml"), "--robot", robot_yaml] + (["--no-target-pose"] if use_sequence else []),
                                          name="ik_node", output="screen"))
    machines = []
    if use_sequence:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(SEQUENCE_NODE), "--robot", robot, "--sequence", seq] + (["--map", map_name] if map_name else []),
                                          name="robot_sequence", output="screen"))
        if LaunchConfiguration("machines").perform(context).lower() != "false":
            for m in scene.get("machines") or []:
                script = os.path.join(sim_config.MACHINE_SCRIPTS, f"{m['name']}.py")
                if os.path.isfile(script):
                    machines.append(m["name"])
                    controllers.append(ExecuteProcess(cmd=[sys.executable, script], name=m["name"], output="screen"))

    show_rviz = LaunchConfiguration("rviz").perform(context).lower() == "true"
    domain = [SetEnvironmentVariable("ROS_DOMAIN_ID", str(int(id)))] if id is not None else []
    return [GroupAction([*domain, LogInfo(msg=f"robot {robot}{f' (ROS domain {int(id)})' if id is not None else ''}: {urdf_file.relative_to(SIM_DIR)} + bringup/{robot}/config, "
                                     f"{'map ' + map_name if map_name else 'no map'}{'' if given else ' (of the running robot_sim.py)'}: base in the world at {pose[0].tolist()} m, rpy {pose[1].tolist()} deg"
                                     + (f", sequence {seq}" if seq else "") + (f", machine cycles {machines}" if machines else "")),
                         robot_state_publisher, move_group, *([rviz] if show_rviz else []), *controllers])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="",
                              description="Robot name = its folder bringup/<name>/ and assets/robot/<name>/; empty = scene.robot of the map"),
        DeclareLaunchArgument("id", default_value="",
                              description="ROS domain id of the robot (scene.robots of the map: {name, id, pose}): picks that robot and runs every node of this launch in its domain"),
        DeclareLaunchArgument("map", default_value="",
                              description="Map name = isaacsim/configs/map/<map>.yaml (its robot, where the robot stands, its sequence and machines); empty = no map"),
        DeclareLaunchArgument("ik_node", default_value="true",
                              description="Also run ik_node.py, which receives the ee target (/target_pose) and moves"),
        DeclareLaunchArgument("sequence", default_value="",
                              description="Name of a sequence (isaacsim/configs/sequence/<name>.yaml): also run control/robot_sequence.py with it (ik_node then only follows its steps); "
                                          "true = scene.sequence of the map. Empty = only ik_node"),
        DeclareLaunchArgument("machines", default_value="true",
                              description="With a sequence: also run the cycle of every machine of the map that has one (isaacsim/machine/<name>.py: the laser marker, the conveyor). false = none"),
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz"),
        OpaqueFunction(function=launch_setup),
    ])
