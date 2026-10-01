"""Robot from its URDF in Isaac Sim, controlled over ROS 2 (same bridge as the lite6 test_depthcam_obj_pose_fix.py).

    lab scripts/robot_sim.py [configs/robots/robot_sim.yaml] [--name m20_12_23d] [--gripper jegg2008_300mm|none]
                             [--force-convert] [--headless]

robot_assets/urdf/<name>.urdf is converted to USD on the first run and cached in assets_cache/robot_<name>/ (converted
again when the .urdf changes; --force-convert after changing only meshes). Meshes: relative to the URDF or
package://<pkg>/meshes/... (found in robot_assets/meshes/), so a URDF straight out of xacro works.
With a gripper (gripper_assets/urdf/<gripper>.urdf, same layout) its root link (e.g. flange) is joined to the robot's
link of that name first; the joined URDF and its USD are cached in assets_cache/robot_<name>_<gripper>/. Gripper joints
are driven like the robot's (joint_pos, ROS commands); URDF <mimic> joints follow their leader joint.
`objects:` drops rigid parts from configs/parts/<part>.yaml (the dataset part configs: mesh, mass, collision, material;
USD cached in assets_cache like the dataset pipeline) at pos_m (part center) / rpy_deg.
The robot starts at joint_pos and follows commands at `speed` x the URDF joint velocity limits (targets are moved
towards the command a step per frame, no jumps):
  subscribes  <controller>/joint_trajectory          trajectory_msgs/JointTrajectory (last point)
  action      <controller>/follow_joint_trajectory   control_msgs/FollowJointTrajectory (MoveIt "Execute")
  subscribes  <gripper_topic> (with a gripper)       std_msgs/Float64: 0 = closed, 1 = open (in between: partly open)
  publishes   /joint_states                          sensor_msgs/JointState (measured position, velocity)
Run it where ROS 2 is sourced. Separate from the dataset pipeline (own stage; touches no dataset config or output).
"""
import argparse
import importlib.util
import math
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("config", nargs="?", default=os.path.join(REPO, "configs", "robots", "robot_sim.yaml"))
parser.add_argument("--name", help="robot name (default: name in the config)")
parser.add_argument("--gripper", help="gripper name in gripper_assets/urdf/, none = no gripper (default: gripper in the config)")
parser.add_argument("--force-convert", action="store_true", help="convert the URDF again even if the cache is current")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def find_mesh(f, base):
    """Mesh path as the converter resolves it: relative to the URDF, file://..., or package://<pkg>/<path> searched in
    the URDF folder and every folder above it. None if missing."""
    if f.startswith("package://"):
        rel, d = f[len("package://"):].partition("/")[2], base
        while rel:
            if os.path.isfile(os.path.join(d, rel)):
                return os.path.join(d, rel)
            if os.path.dirname(d) == d:
                return None
            d = os.path.dirname(d)
        return None
    p = f[len("file://"):] if f.startswith("file://") else os.path.join(base, f)
    return p if os.path.isfile(p) else None


def attach_gripper(robot_urdf, gripper_urdf, out):
    """Robot URDF + gripper URDF joined at the gripper's root link (a robot link, e.g. flange) -> out. Mesh paths
    become absolute (the joined file lives in assets_cache); rewritten only when the content changes."""
    robot, grip = ET.parse(robot_urdf).getroot(), ET.parse(gripper_urdf).getroot()
    for root, src in ((robot, robot_urdf), (grip, gripper_urdf)):
        for m in root.iter("mesh"):
            m.set("filename", find_mesh(m.get("filename"), os.path.dirname(src)))
    names = lambda r, tag: {e.get("name") for e in r.findall(tag)}  # noqa: E731
    children = {j.find("child").get("link") for j in grip.findall("joint")}
    mount = next(ln.get("name") for ln in grip.findall("link") if ln.get("name") not in children)
    if mount not in names(robot, "link"):
        sys.exit(f"gripper root link '{mount}' of {gripper_urdf} is not a link of {robot_urdf}")
    clash = (names(grip, "link") - {mount}) & names(robot, "link") | names(grip, "joint") & names(robot, "joint")
    if clash:
        sys.exit(f"{gripper_urdf} and {robot_urdf} share link/joint names: {sorted(clash)}")
    robot.extend(e for e in grip if not (e.tag == "link" and e.get("name") == mount))
    text = ET.tostring(robot, encoding="unicode")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.isfile(out) or open(out).read() != text:
        with open(out, "w") as f:
            f.write(text)
    return out


def load_settings(path, name, gripper):
    s = yaml.safe_load(open(path)) or {}
    s["name"] = name or s.get("name")
    if not s["name"]:
        sys.exit(f"no robot name: set `name:` in {path} or pass --name")
    g = s.get("gripper") if gripper is None else gripper
    s["gripper"] = None if g is None or str(g).lower() in ("", "none", "null") else str(g)
    urdfs = [os.path.join(REPO, "robot_assets", "urdf", f"{s['name']}.urdf")]
    if s["gripper"]:
        urdfs.append(os.path.join(REPO, "gripper_assets", "urdf", f"{s['gripper']}.urdf"))
    for u in urdfs:
        if not os.path.isfile(u):
            sys.exit(f"URDF not found: {u}")
        miss = [m.get("filename") for m in ET.parse(u).getroot().iter("mesh")
                if not find_mesh(m.get("filename", ""), os.path.dirname(u))]
        if miss:
            sys.exit(f"{len(miss)} mesh(es) of {u} not found: {miss[:5]}")
    s["model"] = "_".join([s["name"]] + ([s["gripper"]] if s["gripper"] else []))  # USD cache: assets_cache/robot_<model>
    s["urdf"] = (attach_gripper(*urdfs, os.path.join(REPO, "assets_cache", f"robot_{s['model']}", f"{s['model']}.urdf"))
                 if s["gripper"] else urdfs[0])
    root = ET.parse(s["urdf"]).getroot()
    s["joints"] = {}  # movable joints: name -> (type, lower, upper, max velocity)
    for j in root.findall("joint"):
        if j.get("type") in ("revolute", "continuous", "prismatic"):
            lim = j.find("limit")
            g = (lambda k, d: float(lim.get(k)) if lim is not None and lim.get(k) else d)
            cont = j.get("type") == "continuous"
            s["joints"][j.get("name")] = (j.get("type"), -math.inf if cont else g("lower", 0.0),
                                          math.inf if cont else g("upper", 0.0), g("velocity", 1.0))
    # URDF <mimic>: joint -> (leader, multiplier, offset). The converter writes NewtonMimicAPI, which PhysX does not
    # use, so the follower's drive target is set from the leader's here
    s["mimic"] = {j.get("name"): (m.get("joint"), float(m.get("multiplier", 1.0)), float(m.get("offset", 0.0)))
                  for j in root.findall("joint") if (m := j.find("mimic")) is not None and j.get("name") in s["joints"]}
    # gripper joints driven by <gripper_topic> (0..1 -> lower..upper); mimic followers come along via s["mimic"]
    s["gripper_joints"] = [n for n in (j.get("name") for j in ET.parse(urdfs[1]).getroot().findall("joint"))
                           if n in s["joints"] and n not in s["mimic"]] if s["gripper"] else []
    from binsim.config import _load_yaml_tree  # same loader as the dataset's part files (mesh path resolved)
    objects = []
    for o in s.get("objects") or []:
        f = os.path.join(REPO, "configs", "parts", f"{o['part']}.yaml")
        if not os.path.isfile(f):
            sys.exit(f"part config not found: {f}")
        part = _load_yaml_tree(f)
        if not os.path.isfile(part.get("mesh") or ""):
            sys.exit(f"mesh of {f} not found: {part.get('mesh')}")
        objects.append({**o, "cfg": part})
    s["objects"] = objects
    return s


robot = load_settings(os.path.abspath(args.config), args.name, args.gripper)  # checked before Isaac Sim starts
if importlib.util.find_spec("rclpy") is None:
    sys.exit("rclpy not found: source ROS 2 first (source /opt/ros/jazzy/setup.bash)")
args.experience = args.experience or os.path.expanduser(
    "~/IsaacSim/_build/linux-aarch64/release/apps/isaacsim.exp.base.kit")  # same kit as generate_isaacsim.py
if not args.headless:
    args.visualizer = args.visualizer or ["kit"]
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import rclpy  # noqa: E402
from control_msgs.action import FollowJointTrajectory  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402
from trajectory_msgs.msg import JointTrajectory  # noqa: E402

from binsim.transforms import mat_to_quat, rpy_to_mat  # noqa: E402

APP_HZ = 60.0  # physics steps per app.update() in the default physics scene (as in generate_isaacsim.py)
ROBOT_PATH = "/World/robot"


class RobotBridge(Node):
    """Joint targets from ROS (topic or MoveIt action); the sim loop reads them and reports the measured state."""

    def __init__(self, names, q0, controller, mimic, gripper, gripper_topic):
        super().__init__("robot_sim")
        self.names = list(names)
        self._idx = {n: i for i, n in enumerate(self.names)}
        self._mimic = {n: m for n, m in mimic.items() if n in self._idx and m[0] in self._idx}
        self._gripper = {n: lim for n, lim in gripper.items() if n in self._idx}  # joint -> (closed, open) position
        if self._gripper:
            self.create_subscription(Float64, gripper_topic, self._on_gripper, 10)
        self._lock = threading.Lock()
        self._targets = list(q0)
        self._current = list(q0)
        self.create_subscription(JointTrajectory, f"{controller}/joint_trajectory", self._on_traj, 10)
        self._action = ActionServer(
            self, FollowJointTrajectory, f"{controller}/follow_joint_trajectory",
            execute_callback=self._execute, goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=ReentrantCallbackGroup())
        self._state_pub = self.create_publisher(JointState, "/joint_states", 10)

    def _set(self, names, positions):
        with self._lock:
            for n, p in zip(names, positions):
                if n in self._idx:
                    self._targets[self._idx[n]] = float(p)
            for n, (lead, k, off) in self._mimic.items():
                self._targets[self._idx[n]] = self._targets[self._idx[lead]] * k + off

    def _on_traj(self, msg):
        if msg.points:
            self._set(msg.joint_names, msg.points[-1].positions)

    def _on_gripper(self, msg):
        v = min(max(float(msg.data), 0.0), 1.0)
        self._set(list(self._gripper), [lo + v * (hi - lo) for lo, hi in self._gripper.values()])
        self.get_logger().info(f"gripper {v:.2f} ({'closed' if v == 0 else 'open' if v == 1 else 'partly open'})")

    def _execute(self, goal):
        """Waypoints paced by their time_from_start (the path shape, not a jump to the goal), then wait for arrival."""
        traj = goal.request.trajectory
        if not traj.points:
            goal.abort()
            return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.INVALID_GOAL)
        prev = 0.0
        for pt in traj.points[:-1]:
            if goal.is_cancel_requested:
                goal.canceled()
                return FollowJointTrajectory.Result()
            self._set(traj.joint_names, pt.positions)
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            time.sleep(max(0.0, min(t - prev, 0.5)))
            prev = t
        self._set(traj.joint_names, traj.points[-1].positions)
        target = self.targets()
        fb = FollowJointTrajectory.Feedback()
        fb.joint_names = self.names
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if goal.is_cancel_requested:
                goal.canceled()
                return FollowJointTrajectory.Result()
            with self._lock:
                current = list(self._current)
            fb.actual.positions, fb.desired.positions = current, target
            goal.publish_feedback(fb)
            if all(abs(c - t) < 0.01 for c, t in zip(current, target)):
                break
            time.sleep(0.05)
        goal.succeed()
        return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.SUCCESSFUL)

    def targets(self):
        with self._lock:
            return list(self._targets)

    def report(self, q, v):
        with self._lock:
            self._current = [float(x) for x in q]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position, msg.velocity = self.names, [float(x) for x in q], [float(x) for x in v]
        self._state_pub.publish(msg)


def update(n=1):
    for _ in range(n):
        simulation_app.update()


def ghost_root(urdf):
    """True if the URDF's root link is empty (e.g. `world`): the importer then fixes the robot to the world itself."""
    root = ET.parse(urdf).getroot()
    children = {j.find("child").get("link") for j in root.findall("joint")}
    for ln in root.findall("link"):
        if ln.get("name") not in children:
            return all(ln.find(t) is None for t in ("visual", "collision", "inertial"))
    return False


def robot_usd(s, force):
    """URDF -> USD, cached in assets_cache/robot_<model>/ (redone when the .urdf file changes)."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    conv = UrdfConverter(UrdfConverterCfg(
        asset_path=s["urdf"], usd_dir=os.path.join(REPO, "assets_cache", f"robot_{s['model']}"),
        fix_base=not ghost_root(s["urdf"]), merge_fixed_joints=False, force_usd_conversion=force))
    return conv.usd_path


def build_stage(usd_path, s, q0):
    ctx = omni.usd.get_context()
    ctx.new_stage()
    update(5)
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.36)])
    fx = UsdGeom.Xformable(floor)
    fx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    fx.AddScaleOp().Set(Gf.Vec3f(6.0, 6.0, 0.02))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(1000.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 0.0, 30.0))

    def place(path, usd, pos, rpy):
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(usd)
        x = UsdGeom.Xformable(prim)
        x.ClearXformOpOrder()
        x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
        q = mat_to_quat(rpy_to_mat(rpy))
        x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))

    place(ROBOT_PATH, usd_path, s.get("pos_m", [0, 0, 0]), s.get("rpy_deg", [0, 0, 0]))
    for i, o in enumerate(s["objects"]):
        place(f"/World/objects/{o['part']}_{i}", o["usd"], o.get("pos_m", [0, 0, 0]), o.get("rpy_deg", [0, 0, 0]))
    update(5)

    # drives + start pose on the joint prims before physics starts (USD units as in the lite6 script: N*m/deg, deg),
    # so the robot is stiff from the first frame and does not jump
    for jp in stage.Traverse():
        info = s["joints"].get(jp.GetName())
        if info is None or not jp.IsA(UsdPhysics.Joint) or not str(jp.GetPath()).startswith(ROBOT_PATH + "/"):
            continue
        kind = "linear" if info[0] == "prismatic" else "angular"
        start = q0[jp.GetName()] if kind == "linear" else math.degrees(q0[jp.GetName()])
        drv = UsdPhysics.DriveAPI.Apply(jp, kind)
        drv.CreateStiffnessAttr().Set(float(s.get("stiffness", 20000.0)))
        drv.CreateDampingAttr().Set(float(s.get("damping", 1000.0)))
        drv.CreateTargetPositionAttr().Set(start)
        PhysxSchema.JointStateAPI.Apply(jp, kind).CreatePositionAttr().Set(start)


def main():
    rclpy.init()
    node = None
    try:
        usd_path = robot_usd(robot, args.force_convert)  # before the stage is built: the converter uses its own stages
        print(f"[INFO] {robot['urdf']} -> {usd_path}", flush=True)
        if robot["objects"]:
            from binsim.assets import prepare_part
            for o in robot["objects"]:
                o["usd"] = prepare_part(o["cfg"], os.path.join(REPO, "assets_cache")).usd_path
                print(f"[INFO] object {o['part']} at {o.get('pos_m', [0, 0, 0])} m, rpy {o.get('rpy_deg', [0, 0, 0])} deg"
                      f" -> {o['usd']}", flush=True)
        jp = robot.get("joint_pos") or {}
        q0 = {n: min(max(next((float(v) for k, v in jp.items() if re.fullmatch(k, n)), 0.0), lo), hi)
              for n, (_, lo, hi, _) in robot["joints"].items()}
        for n, (lead, k, off) in robot["mimic"].items():
            if lead in q0:
                q0[n] = q0[lead] * k + off
        build_stage(usd_path, robot, q0)

        root_path = Articulation.fetch_articulation_root_api_prim_paths(ROBOT_PATH)[0]
        if root_path is None:
            print(f"[ERROR] no articulation root under {ROBOT_PATH} in {usd_path}", flush=True)
            return
        art = Articulation(root_path)
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        update(2)

        names = list(art.dof_names)
        cmd = np.array([q0.get(n, 0.0) for n in names])
        step = np.array([robot["joints"].get(n, (None, 0, 0, 1.0))[3] for n in names]) * float(robot.get("speed", 0.3)) / APP_HZ
        controller = "/" + str(robot.get("controller", f"{robot['name']}_controller")).strip("/")
        gripper = {n: robot["joints"][n][1:3] for n in robot["gripper_joints"]}
        gripper_topic = str(robot.get("gripper_topic") or "/gripper/command")
        node = RobotBridge(names, cmd, controller, robot["mimic"], gripper, gripper_topic)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        threading.Thread(target=executor.spin, daemon=True).start()
        print(f"[INFO] robot {robot['name']} ready (gripper: {robot['gripper']}): joints {names}, "
              f"max speed {robot.get('speed', 0.3)} x URDF limits", flush=True)
        print(f"[INFO] ROS 2: {controller}/joint_trajectory, {controller}/follow_joint_trajectory -> sim, "
              f"sim -> /joint_states", flush=True)
        if gripper:
            print(f"[INFO] gripper {robot['gripper']}: {gripper_topic} (std_msgs/Float64) 0 = closed, 1 = open "
                  f"-> {list(gripper)}", flush=True)

        while simulation_app.is_running():
            cmd = cmd + np.clip(np.array(node.targets()) - cmd, -step, step)  # rate-limited towards the command
            art.set_dof_position_targets(cmd)
            update(1)
            node.report(art.get_dof_positions().numpy()[0], art.get_dof_velocities().numpy()[0])
        timeline.stop()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
