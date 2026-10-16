"""Robot from its URDF in Isaac Sim, controlled over ROS 2 (same bridge as the lite6 test_depthcam_obj_pose_fix.py).

    lab robot_sim.py [robot:=<name> | robot:=configs/robots/<name>.yaml] [--name m20_35_18d] [--gripper none]
                     [--force-convert] [--robot-only] [--headless]

The robot is chosen by its config file: one yaml per robot in configs/robots/ (name = the folder of the URDF in ../assets,
controller, gripper, start pose, drive gains). Set `robot: <name>` in configs/robots/select_robot.yaml and run `lab robot_sim.py`,
or name it on the command line (`lab robot_sim.py robot:=ur5e` like the bringup launch, which wins over select_robot.yaml; a bare
`ur5e` works too). A name is found as configs/robots/<name>.yaml or as the config whose `name:` it is.
What stands around the robot (objects, bin with a pile, basket) is not part of the robot: it is the `scene:` section of
configs/map/map.yaml (the rest of that file is the dataset config for the data collection) and stays when the robot changes.

../assets/<name>/urdf/<name>.urdf (the same file the MoveIt bringup uses) is converted to USD on the first
run and cached in assets_cache/robot_<name>/ (converted again when the .urdf changes; --force-convert after changing
only meshes). Meshes: relative to the URDF or package://<pkg>/meshes/... (searched in the URDF folder and every folder
above it), so a URDF straight out of xacro works.
A gripper that is part of the robot URDF (lite6) is driven by <gripper_topic> too: list its joints as `gripper_joints:` in the config.
With a gripper (../assets/<gripper>/urdf/<gripper>.urdf, same layout) its root link (e.g. flange) is joined to the robot's
link of that name first; the joined URDF and its USD are cached in assets_cache/robot_<name>_<gripper>/. Gripper joints
are driven like the robot's (joint_pos, ROS commands); URDF <mimic> joints follow their leader joint.
`objects:` (scene: of configs/map/map.yaml) drops rigid parts from configs/parts/<part>.yaml (the dataset part configs: mesh, mass, collision, material;
USD cached in assets_cache like the dataset pipeline) at pos_m (part center) / rpy_deg.
`bin:` puts the dataset's bin (`bin:` + `loading:` of configs/map/map.yaml) at pos_m / rpy_deg in the world and
drops a pile of `pile.count` parts into it with the dataset's planner (`pile.seed: random` = a different random pile on every
start, the seed is printed; a number = always the same pile); bin and parts are separate world-frame prims
(/World/bin, /World/pile/<part>_NNN), so part poses are robot-base coordinates.
`basket:` (scene: of configs/map/map.yaml) an open box (floor + 4 walls, kinematic; the dataset's procedural bin, no STL) at pos_m (bottom centre) /
rpy_deg in the world (/World/basket): where the part is released. control/robot_sequence.py reads its pos_m and
wall_thickness_m to know how far down the gripper goes.
`physics_hz` (default 60) is the physics step rate of the scene (the dataset generator uses 200: falling parts sink into each other less).
Drives use `stiffness` / `damping` and the URDF <limit effort> as force limit (the real joint torque / gripper force).
The robot starts at joint_pos and follows commands at `speed` x the URDF joint velocity limits (targets are moved
towards the command a step per frame, no jumps):
  subscribes  <controller>/joint_trajectory          trajectory_msgs/JointTrajectory (last point)
  action      <controller>/follow_joint_trajectory   control_msgs/FollowJointTrajectory (MoveIt "Execute")
  subscribes  <gripper_topic> (with a gripper)       std_msgs/Float64: 0 = closed, 1 = open (in between: partly open)
  publishes   /joint_states                          sensor_msgs/JointState (measured position, velocity)
Run it where ROS 2 is sourced. Separate from the dataset pipeline (own stage; touches no dataset config or output).
"""
import argparse
import glob
import importlib.util
import math
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))  # Simulation/isaacsim: configs/, assets/, binsim/, assets_cache/
ROBOTS = os.path.join(os.path.dirname(REPO), "assets")  # Simulation/assets/<name>/urdf/<name>.urdf (+ meshes)
MAP_FILE = os.path.join(REPO, "configs", "map", "map.yaml")  # `scene:` = objects, bin + pile, basket (the same for every robot) + dataset config
SELECT_FILE = os.path.join(REPO, "configs", "robots", "select_robot.yaml")  # `robot: <name>`
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("config", nargs="?", default=None, metavar="robot:=<name>",
                    type=lambda a: a[len("robot:="):] if a.startswith("robot:=") else a,   # ROS style `robot:=ur5e`; a bare name works too
                    help="robot name (ur5e) or robot config file; default: `robot:` of configs/robots/select_robot.yaml")
parser.add_argument("--name", help="robot name (default: name in the config)")
parser.add_argument("--gripper", help="gripper name in ../assets/<gripper>/urdf/, none = no gripper (default: gripper in the config)")
parser.add_argument("--force-convert", action="store_true", help="convert the URDF again even if the cache is current")
parser.add_argument("--robot-only", action="store_true",
                    help="only the robot and the floor: ignore the `scene:` section of configs/map/map.yaml (no objects, bin, pile, basket)")

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


def find_urdf(name):
    """../assets/<name>/urdf/<name>.urdf, or <name>.urdf in another assets/*/urdf/ folder (several URDFs of one robot share a folder)."""
    own = os.path.join(ROBOTS, name, "urdf", f"{name}.urdf")
    found = sorted(glob.glob(os.path.join(ROBOTS, "*", "urdf", f"{name}.urdf")))
    return own if os.path.isfile(own) or not found else found[0]


def load_settings(path, name, gripper):
    s = yaml.safe_load(open(path)) or {}
    s["name"] = name or s.get("name")
    if not s["name"]:
        sys.exit(f"no robot name: set `name:` in {path} or pass --name")
    g = s.get("gripper") if gripper is None else gripper
    s["gripper"] = None if g is None or str(g).lower() in ("", "none", "null") else str(g)
    urdfs = [find_urdf(s["name"])]
    if s["gripper"]:
        urdfs.append(os.path.join(ROBOTS, s["gripper"], "urdf", f"{s['gripper']}.urdf"))
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
    s["joints"] = {}  # movable joints: name -> (type, lower, upper, max velocity, max effort; 0 = none in the URDF)
    for j in root.findall("joint"):
        if j.get("type") in ("revolute", "continuous", "prismatic"):
            lim = j.find("limit")
            g = (lambda k, d: float(lim.get(k)) if lim is not None and lim.get(k) else d)
            cont = j.get("type") == "continuous"
            s["joints"][j.get("name")] = (j.get("type"), -math.inf if cont else g("lower", 0.0),
                                          math.inf if cont else g("upper", 0.0), g("velocity", 1.0), g("effort", 0.0))
    # URDF <mimic>: joint -> (leader, multiplier, offset). The converter writes NewtonMimicAPI, which PhysX does not
    # use, so the follower's drive target is set from the leader's here
    s["mimic"] = {j.get("name"): (m.get("joint"), float(m.get("multiplier", 1.0)), float(m.get("offset", 0.0)))
                  for j in root.findall("joint") if (m := j.find("mimic")) is not None and j.get("name") in s["joints"]}
    # gripper joints driven by <gripper_topic> (0..1 -> lower..upper); mimic followers come along via s["mimic"]. Either the joints of
    # the separate gripper URDF, or `gripper_joints:` of the config for a gripper that is already part of the robot URDF (lite6)
    builtin = list(s.get("gripper_joints") or [])
    bad = [n for n in builtin if n not in s["joints"]]
    if bad:
        sys.exit(f"gripper_joints of {path} are not movable joints of {s['urdf']}: {bad}")
    s["gripper_joints"] = [n for n in (j.get("name") for j in ET.parse(urdfs[1]).getroot().findall("joint"))
                           if n in s["joints"] and n not in s["mimic"]] if s["gripper"] else builtin
    # the gripper's own links (all but its root, which is the robot's flange): gripper_material is applied to them
    g_root = ET.parse(urdfs[1]).getroot() if s["gripper"] else None
    s["gripper_links"] = [ln.get("name") for ln in g_root.findall("link")
                          if ln.get("name") in {j.find("child").get("link") for j in g_root.findall("joint")}] if s["gripper"] else []
    from binsim.config import _load_yaml_tree, load_config, to_plain  # same loaders as the dataset's part files / config
    ds, scene = None, {}
    if not args.robot_only:
        try:
            ds = load_config(MAP_FILE)  # the map: its `scene:` section + the dataset's `bin:` and `loading:` (bin shape, pile planner)
            scene = to_plain(ds.get("scene") or {})
        except Exception as e:  # noqa: BLE001  the scene is optional: a map that does not load (missing part file, ...) = robot only
            print(f"[WARN] {MAP_FILE} not usable ({type(e).__name__}: {e}): starting the robot without objects, bin and basket", flush=True)
    objects = []
    for o in scene.get("objects") or []:
        f = os.path.join(REPO, "configs", "objects", f"{o['part']}.yaml")
        if not os.path.isfile(f):
            print(f"[WARN] object {o['part']} skipped: part config not found: {f}", flush=True)
            continue
        part = _load_yaml_tree(f)
        if not os.path.isfile(part.get("mesh") or ""):
            print(f"[WARN] object {o['part']} skipped: mesh of {f} not found: {part.get('mesh')}", flush=True)
            continue
        objects.append({**o, "cfg": part})
    s["objects"] = objects
    # bin + pile for picking tests: the dataset's bin and loading settings, placed in the world
    b, s["bin"] = scene.get("bin") or None, None
    if b:
        pile = b.get("pile") or {}
        seed = pile.get("seed", "random")  # "random" / null: a new pile on every start (printed below), a number: always the same pile
        seed = int.from_bytes(os.urandom(4), "big") if seed is None or str(seed).lower() == "random" else int(seed)
        part, count = None, int(pile.get("count", 0))
        if pile.get("part"):
            f = os.path.join(REPO, "configs", "objects", f"{pile['part']}.yaml")
            if os.path.isfile(f):
                part = _load_yaml_tree(f)
            else:
                print(f"[WARN] pile of {pile['part']} skipped (the bin stays empty): part config not found: {f}", flush=True)
                count = 0
        s["bin"] = {"cfg": ds.bin, "loading": ds.loading, "pos_m": b.get("pos_m", [0, 0, 0]),
                    "rpy_deg": b.get("rpy_deg", [0, 0, 0]), "part": part, "count": count,
                    "seed": seed}
    s["basket"] = k = scene.get("basket") or None  # open box to release the part into: a procedural bin in the world
    if k and any(key not in k for key in ("pos_m", "size_m", "wall_thickness_m")):
        print(f"[WARN] basket skipped: basket of {MAP_FILE} (scene:) needs pos_m, size_m and wall_thickness_m", flush=True)
        s["basket"] = None
    return s


def robot_config(choice):
    """Config file of a robot: a file path, or a robot name = configs/robots/<name>.yaml or the file whose `name:` is it."""
    if os.path.isfile(choice):
        return choice
    d = os.path.join(REPO, "configs", "robots")
    configs = {f: yaml.safe_load(open(os.path.join(d, f))) or {} for f in sorted(os.listdir(d))
               if f.endswith(".yaml") and f != os.path.basename(SELECT_FILE)}
    for f, c in configs.items():
        if f == f"{choice}.yaml" or c.get("name") == choice:
            return os.path.join(d, f)
    sys.exit(f"robot config not found: {choice}; robots: " + ", ".join(sorted({c.get("name", f[:-5]) for f, c in configs.items()})))


if args.config is None:  # nothing on the command line: the robot named in configs/robots/select_robot.yaml
    selected = (yaml.safe_load(open(SELECT_FILE)) or {}).get("robot")
    if not selected:
        sys.exit(f"no robot selected: set `robot: <name>` in {SELECT_FILE} or pass a robot name")
    args.config = str(selected)
args.config = robot_config(args.config)
print(f"[INFO] robot config: {os.path.relpath(args.config, REPO)}", flush=True)
robot = load_settings(os.path.abspath(args.config), args.name, args.gripper)  # checked before Isaac Sim starts
if importlib.util.find_spec("rclpy") is None:
    sys.exit("rclpy not found: source ROS 2 first (source /opt/ros/jazzy/setup.bash)")
args.experience = args.experience or os.path.expanduser(
    "~/IsaacSim/apps/isaacsim.exp.base.kit")  # same kit as generate_isaacsim.py
# Isaac Lab 3.0.0's KitLauncher has no --headless CLI flag: HEADLESS env var forces headless
headless = bool(int(os.environ.get("HEADLESS", "0")))
args.headless = headless
if not headless:
    args.visualizer = args.visualizer or ["kit"]
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import rclpy  # noqa: E402
from control_msgs.action import FollowJointTrajectory  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402
from trajectory_msgs.msg import JointTrajectory  # noqa: E402

from binsim.transforms import make_T, mat_to_quat, quat_to_mat, rpy_to_mat  # noqa: E402

APP_HZ = 60.0  # physics steps per app.update() in the default physics scene (as in generate_isaacsim.py)
ROBOT_PATH = "/World/robot"
# end of a trajectory: done when its joints are within ARRIVE_TOL of the goal, or when they have stopped short of it (no joint
# moved more than STALL_MOVE for STALL_TIME, e.g. blocked by the bin, or a part held in the gripper), or after
# ARRIVE_DEADLINE at the latest - the last two with a warning. MoveIt takes no new goal until this one is done.
ARRIVE_TOL, STALL_MOVE, STALL_TIME, ARRIVE_DEADLINE = 0.01, 0.01, 1.0, 15.0   # rad, rad, s, s


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
        """Waypoints paced by their time_from_start (the path shape, not a jump to the goal), then wait until the joints of
        the trajectory have arrived or stopped short of the goal (see ARRIVE_TOL)."""
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
        idx = [self._idx[n] for n in traj.joint_names if n in self._idx]   # only the joints of this trajectory (not the gripper)
        fb = FollowJointTrajectory.Feedback()
        fb.joint_names = self.names
        start = time.monotonic()
        moved_at, ref, smooth = start, None, None       # when the joints last moved by more than STALL_MOVE, from where; smoothed positions
        while True:
            if goal.is_cancel_requested:
                goal.canceled()
                return FollowJointTrajectory.Result()
            with self._lock:
                current = list(self._current)
            fb.actual.positions, fb.desired.positions = current, target
            goal.publish_feedback(fb)
            err = [abs(current[i] - target[i]) for i in idx]
            if max(err) < ARRIVE_TOL:
                break
            now, pos = time.monotonic(), [current[i] for i in idx]
            smooth = pos if smooth is None else [s + 0.2 * (p - s) for s, p in zip(smooth, pos)]   # contact jitter must not reset the timer
            if ref is None or max(abs(a - b) for a, b in zip(smooth, ref)) > STALL_MOVE:
                moved_at, ref = now, list(smooth)
            if now - moved_at > STALL_TIME or now - start > ARRIVE_DEADLINE:
                k = max(range(len(idx)), key=lambda i: err[i])
                why = "stopped" if now - moved_at > STALL_TIME else "still not there after %.0f s" % ARRIVE_DEADLINE
                self.get_logger().warn(f"arm {why} {math.degrees(err[k]):.2f} deg short of the goal at {self.names[idx[k]]} "
                                       f"(blocked?) -> trajectory reported as done")
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


def part_root(path):
    """/World/pile/side_plate_078/geometry -> /World/pile/side_plate_078 (a part of the pile or a single object), else None"""
    p = path.split("/")
    return "/".join(p[:4]) if len(p) >= 4 and p[1] == "World" and p[2] in ("pile", "objects") else None


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


def plan_pile(b, cache_dir):
    """Bin USD and the pile's world poses: the dataset's planner (binsim.loading.plan_scene with the dataset's loading
    settings) places b["count"] parts in the bin frame; each pose is moved to the world with the bin's pos_m / rpy_deg.
    Returns (bin usd, [(part usd, world position, world rotation matrix), ...], loading mode)."""
    from binsim.assets import prepare_bin, prepare_part
    from binsim.loading import bin_geom_from_cfg, part_geoms_from_cfg, plan_scene
    bin_usd = prepare_bin(b["cfg"], cache_dir).usd_path
    if not b["part"] or b["count"] <= 0:
        return bin_usd, [], None
    pa = prepare_part(b["part"], cache_dir)
    geoms = part_geoms_from_cfg([b["part"]], [pa.extents], [pa.radius], [pa.mesh])
    pl, mode = plan_scene([b["count"]], geoms, bin_geom_from_cfg(b["cfg"]), b["loading"],
                          np.random.default_rng(b["seed"]))
    T = make_T(rpy_to_mat(b["rpy_deg"]), b["pos_m"])
    return bin_usd, [(pa.usd_path, T[:3, :3] @ p.pos + T[:3, 3], T[:3, :3] @ quat_to_mat(p.quat)) for p in pl], mode


def set_gripper_material(stage, links, gm):
    """gripper_material: {metallic, roughness} on every visual material of the gripper links. The converter turns URDF
    colours into UsdPreviewSurface materials with metallic 0 (a plastic look) inside instanced meshes, so the instances
    under these links are made editable first (only on this stage; the cached USD is unchanged). Returns the count."""
    roots = [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)) if p.GetName() in links]
    while True:                                   # instances can be nested (instanced mesh -> instanced material)
        inst = [p for r in roots for p in Usd.PrimRange(r) if p.IsInstance()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)
    mats = {UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0].GetPath() for r in roots for p in Usd.PrimRange(r)
            if p.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(p).ComputePurpose() == UsdGeom.Tokens.default_}
    mats.discard(Sdf.Path())                      # gprims without a material
    for mp in mats:
        m = UsdShade.Material(stage.GetPrimAtPath(mp))
        for k in ("metallic", "roughness"):
            if gm.get(k) is not None:
                m.CreateInput(k, Sdf.ValueTypeNames.Float).Set(float(gm[k]))
    return len(mats)


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
    # GPU contact buffers: with a pile (100 parts x 32 convex hulls) the default found/lost aggregate pair capacity is
    # too small - PhysX: "increase ...foundLostAggregatePairsCapacity ..., otherwise the simulation will miss interactions"
    psa = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    psa.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(2 ** 21)
    psa.CreateGpuTotalAggregatePairsCapacityAttr().Set(2 ** 21)
    # physics rate (1 app update = 1/60 s of simulated time): at 60 Hz a part falling out of the pile column (up to 0.5 m, ~3 m/s)
    # moves ~5 cm per step, more than the size of a plate, and can sink into the parts below. The dataset generator steps at 200 Hz
    scene.GetPrim().CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(int(s.get("physics_hz", 60)))
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

    def place(path, usd, pos, rpy=None, R=None):
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(usd)
        x = UsdGeom.Xformable(prim)
        x.ClearXformOpOrder()
        x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
        q = mat_to_quat(rpy_to_mat(rpy) if R is None else R)
        x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))

    place(ROBOT_PATH, usd_path, s.get("pos_m", [0, 0, 0]), s.get("rpy_deg", [0, 0, 0]))
    for i, o in enumerate(s["objects"]):
        place(f"/World/objects/{o['part']}_{i}", o["usd"], o.get("pos_m", [0, 0, 0]), o.get("rpy_deg", [0, 0, 0]))
    if s.get("bin"):                              # bin and pile: separate prims, world poses (not under the bin)
        place("/World/bin", s["bin"]["usd"], s["bin"]["pos_m"], s["bin"]["rpy_deg"])
        for i, (usd, pos, R) in enumerate(s["bin"]["pile"]):
            place(f"/World/pile/{s['bin']['part']['name']}_{i:03d}", usd, pos, R=R)
    if s.get("basket"):                           # basket: world pose, kinematic like the bin
        place("/World/basket", s["basket"]["usd"], s["basket"]["pos_m"], s["basket"].get("rpy_deg", [0, 0, 0]))
    update(5)
    if s["gripper_links"] and s.get("gripper_material"):
        n = set_gripper_material(stage, set(s["gripper_links"]), s["gripper_material"])
        print(f"[INFO] gripper material {dict(s['gripper_material'])} on {n} materials", flush=True)

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
        if info[4] > 0:  # URDF effort = the drive's force limit (N*m, prismatic N): the gripper jaws push with the real
            drv.CreateMaxForceAttr().Set(float(info[4]))  # gripper force (JEGG-2510 75 N per jaw), the arm with its real
                                                          # torque - without it both push with unlimited force through parts
        PhysxSchema.JointStateAPI.Apply(jp, kind).CreatePositionAttr().Set(start)


def main():
    rclpy.init(args=[])   # the command line (robot:=<name>, Isaac Lab flags) is not for ROS: rcl would read robot:=ur5e as a remap rule
    node = None
    try:
        usd_path = robot_usd(robot, args.force_convert)  # before the stage is built: the converter uses its own stages
        print(f"[INFO] {robot['urdf']} -> {usd_path}", flush=True)
        # objects, bin + pile and basket are optional: whatever cannot be prepared (missing file, bad config) is skipped with a warning
        # and the robot starts without it
        if robot["objects"]:
            from binsim.assets import prepare_part
            ready = []
            for o in robot["objects"]:
                try:
                    o["usd"] = prepare_part(o["cfg"], os.path.join(REPO, "assets_cache")).usd_path
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] object {o['part']} skipped: {type(e).__name__}: {e}", flush=True)
                    continue
                ready.append(o)
                print(f"[INFO] object {o['part']} at {o.get('pos_m', [0, 0, 0])} m, rpy {o.get('rpy_deg', [0, 0, 0])} deg"
                      f" -> {o['usd']}", flush=True)
            robot["objects"] = ready
        if robot["bin"]:
            b = robot["bin"]
            try:
                b["usd"], b["pile"], mode = plan_pile(b, os.path.join(REPO, "assets_cache"))
                print(f"[INFO] bin at {b['pos_m']} m (world), pile: {len(b['pile'])} x "
                      f"{b['part']['name'] if b['part'] else '-'} ({mode}, seed {b['seed']}) -> /World/pile", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] bin and pile skipped: {type(e).__name__}: {e}", flush=True)
                robot["bin"] = None
        if robot.get("basket"):
            from binsim.assets import prepare_bin
            k = robot["basket"]
            try:
                k["usd"] = prepare_bin(k, os.path.join(REPO, "assets_cache")).usd_path
                print(f"[INFO] basket {k['size_m']} m at {k['pos_m']} m (world), floor top at z = "
                      f"{k['pos_m'][2] + k['wall_thickness_m']:.3f} m -> /World/basket", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] basket skipped: {type(e).__name__}: {e}", flush=True)
                robot["basket"] = None
        jp = robot.get("joint_pos") or {}
        q0 = {n: min(max(next((float(v) for k, v in jp.items() if re.fullmatch(k, n)), 0.0), lo), hi)
              for n, (_, lo, hi, _, _) in robot["joints"].items()}
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
              f"max speed {robot.get('speed', 0.3)} x URDF limits, physics {int(robot.get('physics_hz', 60))} Hz", flush=True)
        print(f"[INFO] ROS 2: {controller}/joint_trajectory, {controller}/follow_joint_trajectory -> sim, "
              f"sim -> /joint_states", flush=True)
        if gripper:
            print(f"[INFO] gripper {robot['gripper']}: {gripper_topic} (std_msgs/Float64) 0 = closed, 1 = open "
                  f"-> {list(gripper)}", flush=True)

        # click a part in the viewport -> its pose is printed as "x y z roll pitch yaw", the Translate and Rotate X Y Z of Isaac's Property
        # panel (m, deg; read live from the physics, the panel itself may keep showing the start values). The event only remembers
        # the clicked parts, the pose is read in the loop below
        clicked, views = [], {}

        def on_stage_event(event):
            if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
                paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
                clicked[:] = sorted({part_root(x) for x in paths} - {None})
        stage_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(on_stage_event, name="robot_sim click")
        print("[INFO] click a part in the viewport: its pose is printed as x y z roll pitch yaw (m, deg, like the Property panel)", flush=True)

        while simulation_app.is_running():
            cmd = cmd + np.clip(np.array(node.targets()) - cmd, -step, step)  # rate-limited towards the command
            art.set_dof_position_targets(cmd)
            update(1)
            node.report(art.get_dof_positions().numpy()[0], art.get_dof_velocities().numpy()[0])
            while clicked:
                path = clicked.pop(0)
                try:
                    view = views.get(path) or views.setdefault(path, RigidPrim(path))
                    pos, quat = view.get_world_poses()
                    (x, y, z), (qw, qx, qy, qz) = pos.numpy()[0], quat.numpy()[0]
                    roll, pitch, yaw = Gf.Rotation(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz)))).Decompose(
                        Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())  # the Rotate X Y Z of the panel (checked against it)
                    print(f"{x:.3f} {y:.3f} {z:.3f} {roll + 0.0:.1f} {pitch + 0.0:.1f} {yaw + 0.0:.1f}", flush=True)
                except Exception as e:
                    print(f"[WARN] pose of {path} not available: {type(e).__name__}: {e}", flush=True)
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
