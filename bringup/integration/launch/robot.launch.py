"""robot.launch.py: one launch for any robot: robot_state_publisher + move_group + RViz + control/ik_node.py, and with a sequence also control/robot_sequence.py.

    ros2 launch robot.launch.py [robot:=<name>] [sequence:=<name>] [pose:="x y z roll pitch yaw"] [id:=<domain>] [ik_node:=false] [rviz:=false] [joints:=false]

Needs nothing of the simulation: it only talks to it (or to the real robot) over the topics.
A small window shows the current angles of the arm joints in degrees; joints:=false does not open it.
The robot stands where the sequence says (robot_pose of control/sequence/<name>.yaml) or where pose:= says; everything robot specific is in bringup/<robot>/config.
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_param_builder import load_xacro, load_yaml
from launch_ros.actions import Node
from scipy.spatial.transform import Rotation

BRINGUP_DIR = Path(__file__).resolve().parents[2]
SIM_DIR = BRINGUP_DIR.parent
sys.path.insert(0, str(SIM_DIR / "control"))
import cell_config

IK_NODE = SIM_DIR / "control" / "ik_node.py"
SEQUENCE_NODE = SIM_DIR / "control" / "robot_sequence.py"
SEQUENCE_DIR = cell_config.SEQUENCE_DIR
JOINT_WINDOW = """
import math, sys, threading, tkinter, rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
names = sys.argv[1:]
rclpy.init()
node = Node('joint_window')
latest = {}
node.create_subscription(JointState, '/joint_states', lambda m: latest.update(zip(m.name, m.position)), 10)
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
root = tkinter.Tk()
root.title('Joints (deg)')
root.attributes('-topmost', True)
labels = {}
for row, n in enumerate(names):
    tkinter.Label(root, text=n, font=('monospace', 18, 'bold'), width=4).grid(row=row, column=0, padx=10, pady=4)
    labels[n] = tkinter.Label(root, text='--', font=('monospace', 18), width=9, anchor='e')
    labels[n].grid(row=row, column=1, padx=10)
def tick():
    for n in names:
        if n in latest:
            labels[n].config(text=f'{math.degrees(latest[n]):8.1f}')
    root.after(100, tick)
tick()
root.mainloop()
"""


def robot_name(choice):
    robots = sorted(d.name for d in BRINGUP_DIR.iterdir() if (d / "config").is_dir())
    name = cell_config.pick_robot(choice)
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
        found = cell_config.find_mesh(mesh.get("filename"), str(urdf_file.parent))
        if found:
            mesh.set("filename", "file://" + str(Path(found).resolve()))
    if pose is not None and (np.any(pose[0]) or np.any(pose[1])) and not place_in_world(root, *pose):
        print(f"[WARN] {urdf_file.name}: no empty root link with one fixed joint to the robot: the pose {pose[0].tolist()} m / "
              f"{pose[1].tolist()} deg is not applied in MoveIt", flush=True)
    return ET.tostring(root, encoding="unicode")


def planning_limits(config_dir, robot_cfg):
    limits = load_yaml(config_dir / "joint_limits.yaml")
    for joint, (lo, hi) in (cell_config.robot_motion(robot_cfg)["joint_limits_deg"] or {}).items():
        entry = limits.setdefault("joint_limits", {}).setdefault(joint, {})
        entry.update(has_position_limits=True, min_position=math.radians(float(lo)), max_position=math.radians(float(hi)))
    return limits


def launch_setup(context, *args, **kwargs):
    user_robot = LaunchConfiguration("robot").perform(context).strip() or None
    id = LaunchConfiguration("id").perform(context).strip() or None
    seq = LaunchConfiguration("sequence").perform(context).strip()
    seq = "" if seq.lower() in ("", "false", "none") else seq
    if seq:
        names = cell_config.sequence_names()
        if seq not in names:
            raise ValueError(f"sequence:={seq}: {SEQUENCE_DIR / (seq + '.yaml')} not found; sequences: {', '.join(names) or '-'}")
        declared = cell_config.load_sequence_file(seq).get("robot")
        if declared and user_robot and declared != cell_config.canonical_robot(user_robot):
            raise ValueError(f"sequence:={seq} is the sequence of the robot {declared}, but robot:={user_robot}")
        user_robot = user_robot or declared
    robot = robot_name(user_robot)
    robot_cfg = cell_config.load_robot(robot)
    config_dir = BRINGUP_DIR / robot / "config"
    rviz_file = BRINGUP_DIR / robot / "rviz" / "demo.rviz"
    urdf_file = Path(cell_config.find_urdf(robot))

    given_pose = LaunchConfiguration("pose").perform(context).split()
    base = [float(v) for v in given_pose] if given_pose else (cell_config.sequence_value(seq, robot, "robot_pose") if seq else None)
    pose = tuple(np.array(v, float) for v in cell_config.base_pose(robot_cfg, base))
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

    use_sequence = bool(seq)
    use_ik_node = LaunchConfiguration("ik_node").perform(context).lower() == "true"
    controllers = []
    if use_ik_node:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(IK_NODE), "--robot-config", str(config_dir / "robot.yaml")], name="ik_node", output="screen"))
    if LaunchConfiguration("joints").perform(context).lower() != "false":
        arm_joints = [str(j) for j in (load_yaml(config_dir / "ik_node.yaml") or {}).get("arm_joints") or []]
        if arm_joints:
            controllers.append(ExecuteProcess(cmd=["/usr/bin/python3", "-c", JOINT_WINDOW, *arm_joints], name="joint_window", output="screen"))
    if use_sequence:
        controllers.append(ExecuteProcess(cmd=[sys.executable, str(SEQUENCE_NODE), "--robot", robot, "--sequence", seq],
                                          name="robot_sequence", output="screen"))

    show_rviz = LaunchConfiguration("rviz").perform(context).lower() == "true"
    domain = [SetEnvironmentVariable("ROS_DOMAIN_ID", str(int(id)))] if id is not None else []
    return [GroupAction([*domain, LogInfo(msg=f"robot {robot}{f' (ROS domain {int(id)})' if id is not None else ''}: {urdf_file.relative_to(SIM_DIR)} + bringup/{robot}/config, "
                                     f"base in the world at {pose[0].tolist()} m, rpy {pose[1].tolist()} deg"
                                     + (f", sequence {seq}" if seq else "")),
                         robot_state_publisher, move_group, *([rviz] if show_rviz else []), *controllers])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="",
                              description="Robot name = its folder bringup/<name>/ and assets/robot/<name>/; empty = the robot of the sequence"),
        DeclareLaunchArgument("id", default_value="",
                              description="ROS domain id: every node of this launch runs in that domain"),
        DeclareLaunchArgument("pose", default_value="",
                              description="Where the robot base stands in the world: \"x y z roll pitch yaw\" (m, deg); empty = robot_pose of the sequence, else 0"),
        DeclareLaunchArgument("ik_node", default_value="true",
                              description="Also run ik_node.py, which receives the ee target (/target_pose) and moves"),
        DeclareLaunchArgument("sequence", default_value="",
                              description="Name of a sequence (control/sequence/<name>.yaml): also run control/robot_sequence.py with it (ik_node then only follows its steps). Empty = only ik_node"),
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz"),
        DeclareLaunchArgument("joints", default_value="true", description="Open a small window with the joint angles of the arm (deg); false = do not"),
        OpaqueFunction(function=launch_setup),
    ])
