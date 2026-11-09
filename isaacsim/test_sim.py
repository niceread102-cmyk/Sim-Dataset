"""test_sim.py: a robot-only Isaac Sim scene to study how a robot reaches targets (singularities, gimbal lock, joint turns): the robot on a floor, nothing else.

    lab test_sim.py robot:=<robot> [--tip fanuc_flange] [--tcp <link>] [--controller <name>] [--start J2=15 J5=-75]        (or a URDF file instead of robot:=<robot>)
    ros2 launch robot.launch.py robot:=<robot> map:=default        (same base pose as here: the origin)
    python3 control/ee_6d.py                                        (x y z roll pitch yaw of the TCP)

Every /target_pose becomes a red ball at the target; when the TCP is there (position and the orientation of the tip link) it turns green and is removed 1 s later.
While the TCP travels to the target it leaves small sky-blue dots (one per centimetre of TCP path); they are removed together with the green ball (a new target removes them at once). The floor and the lights are dark so the dots stand out.
A new target removes a red ball that was not reached. The robot is moved by ik_node (robot.launch.py), the sim only follows it: it answers the MoveIt controller <urdf name>_traj_controller (--controller changes it) and publishes /joint_states.
--tip = last link of the arm (its orientation is the orientation of the target), --tcp = link of the tool centre point (default: the only link named *_tcp in the URDF, else the tip),
--start = start angles in degrees. <robot> = the URDF file assets/robot/*/urdf/<robot>.urdf.
"""
import argparse
import glob
import math
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('robot')
parser.add_argument('--tip', default='fanuc_flange')
parser.add_argument('--tcp')
parser.add_argument('--controller')
parser.add_argument('--start', nargs='*', default=[])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.experience = f"{HOME}/IsaacSim/_build/linux-aarch64/release/apps/isaacsim.exp.base.kit"
args_cli.visualizer = ["kit"]
robot = args_cli.robot[len('robot:='):] if args_cli.robot.startswith('robot:=') else args_cli.robot
urdf = os.path.abspath(robot) if os.path.isfile(robot) else next(iter(sorted(glob.glob(os.path.join(os.path.dirname(HERE), 'assets', 'robot', '*', 'urdf', f'{robot}.urdf')))), None)
if urdf is None:
    sys.exit(f'Not Found URDF: {robot} (a URDF file, or a robot name with assets/robot/*/urdf/<robot>.urdf)')
name = os.path.splitext(os.path.basename(urdf))[0]
controller = args_cli.controller or f'{name}_traj_controller'

URDF = ET.parse(urdf).getroot()
PARENT = {j.find("child").get("link"): j for j in URDF.findall("joint")}
TCP_LINKS = [ln.get("name") for ln in URDF.findall("link") if ln.get("name").endswith("_tcp")]
TCP = args_cli.tcp or (TCP_LINKS[0] if len(TCP_LINKS) == 1 else args_cli.tip)


def chain_to(link):
    chain = []
    while link in PARENT:
        chain.append(PARENT[link])
        link = PARENT[link].find("parent").get("link")
    return chain[::-1]


def ghost_root():
    children = {j.find("child").get("link") for j in URDF.findall("joint")}
    root = next(ln for ln in URDF.findall("link") if ln.get("name") not in children)
    return all(root.find(t) is None for t in ("visual", "collision", "inertial"))


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import threading
from collections import deque

import numpy as np
import omni.timeline
import omni.usd
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaacsim.core.experimental.prims import Articulation
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

ROBOT_PATH = '/World/robot'
PHYSICS_HZ = 240
STIFFNESS = 20000.0
DAMPING = 1000.0
BALL_RADIUS = 0.02
RED, GREEN, SKY = Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.35, 0.8, 1.0)
TRAIL_RADIUS = 0.004
TRAIL_STEP = 0.01
FLOOR_COLOR = Gf.Vec3f(0.04, 0.04, 0.05)
DOME_INTENSITY = 250.0
SUN_INTENSITY = 700.0
POSITION_TOLERANCE = 0.005
ANGLE_TOLERANCE = math.radians(2.0)
GREEN_SECONDS = 1.0

converter = UrdfConverter(UrdfConverterCfg(
    asset_path=urdf,
    usd_dir=os.path.join(HERE, 'assets_cache', name),
    fix_base=not ghost_root(),
    merge_fixed_joints=False,
))
usd_path = converter.usd_path
print(f'[INFO] {urdf} -> {usd_path}')


def rpy_matrix(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_matrix(axis, angle):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def quaternion_matrix(q):
    x, y, z, w = np.array([q.x, q.y, q.z, q.w]) / np.linalg.norm([q.x, q.y, q.z, q.w])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def forward(chain, q):
    T = np.eye(4)
    for joint in chain:
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0.0] * 3
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0.0] * 3
        T[:3, 3] += T[:3, :3] @ xyz
        T[:3, :3] = T[:3, :3] @ rpy_matrix(rpy)
        axis = joint.find("axis")
        axis = np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else np.array([1.0, 0.0, 0.0])
        angle = q.get(joint.get("name"), 0.0)
        if joint.get("type") in ("revolute", "continuous"):
            T[:3, :3] = T[:3, :3] @ axis_matrix(axis / np.linalg.norm(axis), angle)
        elif joint.get("type") == "prismatic":
            T[:3, 3] += T[:3, :3] @ (axis / np.linalg.norm(axis) * angle)
    return T


def read_joints():
    joints = {}
    for j in URDF.findall("joint"):
        if j.get("type") not in ("revolute", "continuous", "prismatic"):
            continue
        lim = j.find("limit")
        joints[j.get("name")] = (j.get("type"), float(lim.get("effort", 0)) if lim is not None else 0.0)
    return joints


def set_drives(stage, joints):
    for prim in stage.Traverse():
        if prim.GetName() not in joints:
            continue
        joint_type, effort = joints[prim.GetName()]
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear" if joint_type == "prismatic" else "angular")
        drive.CreateStiffnessAttr().Set(STIFFNESS)
        drive.CreateDampingAttr().Set(DAMPING)
        drive.CreateTargetPositionAttr().Set(0.0)
        if effort > 0:
            drive.CreateMaxForceAttr().Set(effort)


def build_stage():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx.CreateTimeStepsPerSecondAttr(PHYSICS_HZ)
    physx.CreateSolverTypeAttr("TGS")

    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor_xf = UsdGeom.Xformable(floor)
    floor_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    floor_xf.AddScaleOp().Set(Gf.Vec3f(6.0, 6.0, 0.01))
    floor.CreateDisplayColorAttr([FLOOR_COLOR])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(DOME_INTENSITY)
    UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(SUN_INTENSITY)

    stage.DefinePrim(ROBOT_PATH, "Xform").GetReferences().AddReference(usd_path)
    set_drives(stage, read_joints())
    return stage


class Markers:
    def __init__(self, stage, tip, tcp):
        self.stage = stage
        self.tip_chain = chain_to(tip)
        self.tcp_chain = chain_to(tcp)
        self.count = 0
        self.active = None
        self.reached = {}
        self.trail = []
        self.last_dot = None
        UsdGeom.Xform.Define(stage, "/World/markers")

    def drop_dot(self, position):
        path = f'/World/markers/trail_{self.count}_{len(self.trail)}'
        dot = UsdGeom.Sphere.Define(self.stage, path)
        dot.CreateRadiusAttr(TRAIL_RADIUS)
        dot.CreateDisplayColorAttr([SKY])
        UsdGeom.Xformable(dot).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
        self.trail.append(path)
        self.last_dot = position

    def clear_trail(self):
        for path in self.trail:
            self.stage.RemovePrim(path)
        self.trail = []
        self.last_dot = None

    def add(self, pose):
        if self.active is not None:
            self.stage.RemovePrim(self.active['path'])
            print(f"[INFO] target {self.active['id']} was replaced before the TCP reached it")
        self.clear_trail()
        self.count += 1
        path = f'/World/markers/target_{self.count}'
        ball = UsdGeom.Sphere.Define(self.stage, path)
        ball.CreateRadiusAttr(BALL_RADIUS)
        ball.CreateDisplayColorAttr([RED])
        UsdGeom.Xformable(ball).AddTranslateOp().Set(Gf.Vec3d(pose.position.x, pose.position.y, pose.position.z))
        self.active = {'id': self.count, 'path': path, 'ball': ball, 'position': np.array([pose.position.x, pose.position.y, pose.position.z]),
                       'rotation': quaternion_matrix(pose.orientation)}
        print(f"[INFO] target {self.count}: {pose.position.x:.3f} {pose.position.y:.3f} {pose.position.z:.3f} (red until the TCP is there)")

    def step(self, q, now):
        if self.active is not None:
            tip, tcp = forward(self.tip_chain, q), forward(self.tcp_chain, q)
            if self.last_dot is None or np.linalg.norm(tcp[:3, 3] - self.last_dot) >= TRAIL_STEP:
                self.drop_dot(tcp[:3, 3].copy())
            distance = float(np.linalg.norm(tcp[:3, 3] - self.active['position']))
            turn = math.acos(min(1.0, max(-1.0, (np.trace(tip[:3, :3].T @ self.active['rotation']) - 1.0) / 2.0)))
            if distance < POSITION_TOLERANCE and turn < ANGLE_TOLERANCE:
                self.active['ball'].GetDisplayColorAttr().Set([GREEN])
                self.reached[self.active['path']] = (now, self.trail)
                self.trail = []
                self.last_dot = None
                print(f"[INFO] target {self.active['id']} reached: position error {distance * 1000:.1f} mm, orientation error {math.degrees(turn):.1f} deg")
                self.active = None
        for path, (since, dots) in list(self.reached.items()):
            if now - since >= GREEN_SECONDS:
                self.stage.RemovePrim(path)
                for dot in dots:
                    self.stage.RemovePrim(dot)
                del self.reached[path]


class Trajectory:
    def __init__(self, idx, times, positions):
        self.idx = idx
        self.times = times
        self.positions = positions
        self.t0 = None
        self.done = threading.Event()
        self.replaced = False


class RobotBridge(Node):
    def __init__(self, dof_names, controller):
        super().__init__("robot_sim")
        self.dof_names = list(dof_names)
        self._lock = threading.Lock()
        self._traj = None
        self.targets = deque()
        self.state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_subscription(Pose, "/target_pose", lambda msg: self.targets.append(msg), 10)
        self.server = ActionServer(self, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory",
                                   execute_callback=self._execute, cancel_callback=lambda _: CancelResponse.ACCEPT,
                                   callback_group=ReentrantCallbackGroup())

    def _execute(self, goal_handle):
        msg = goal_handle.request.trajectory
        job = Trajectory(
            idx=[self.dof_names.index(n) for n in msg.joint_names],
            times=np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points]),
            positions=np.array([p.positions for p in msg.points]))
        with self._lock:
            if self._traj is not None:
                self._traj.replaced = True
                self._traj.done.set()
            self._traj = job
        job.done.wait()
        goal_handle.succeed()
        return FollowJointTrajectory.Result()

    def command(self, t, cmd):
        with self._lock:
            job = self._traj
        if job is None:
            return cmd
        if job.t0 is None:
            job.t0 = t
        s = t - job.t0
        cmd = cmd.copy()
        for k, i in enumerate(job.idx):
            cmd[i] = np.interp(s, job.times, job.positions[:, k])
        if s >= job.times[-1]:
            with self._lock:
                if self._traj is job:
                    self._traj = None
            job.done.set()
        return cmd

    def publish_state(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.dof_names
        msg.position = [float(p) for p in positions]
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    stage = build_stage()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    art = Articulation(ROBOT_PATH)
    cmd = np.zeros(len(art.dof_names))
    for item in args_cli.start:
        joint, degrees = item.split('=')
        cmd[art.dof_names.index(joint)] = math.radians(float(degrees))
    art.set_dof_positions(cmd)
    art.set_dof_position_targets(cmd)

    markers = Markers(stage, args_cli.tip, TCP)
    bridge = RobotBridge(art.dof_names, controller)
    executor = MultiThreadedExecutor()
    executor.add_node(bridge)
    spin = threading.Thread(target=executor.spin)
    spin.start()
    print(f'[INFO] joints {art.dof_names}; tip {args_cli.tip}, tcp {TCP}; /{controller}/follow_joint_trajectory, /joint_states, /target_pose')

    try:
        while simulation_app.is_running():
            while bridge.targets:
                markers.add(bridge.targets.popleft())
            cmd = bridge.command(timeline.get_current_time(), cmd)
            art.set_dof_position_targets(cmd)
            simulation_app.update()
            q = art.get_dof_positions().numpy()[0]
            bridge.publish_state(q)
            markers.step(dict(zip(art.dof_names, q)), time.monotonic())
    finally:
        executor.shutdown()
        spin.join()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
