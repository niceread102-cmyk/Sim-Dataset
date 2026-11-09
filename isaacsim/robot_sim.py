"""robot_sim.py: the robot (URDF) in Isaac Sim, controlled over ROS 2, in a map (configs/map/<map>.yaml) or an empty one.

    lab robot_sim.py [robot:=<name>] [map:=<map>] [obj:=<part> ...] [cycles:=false] [--gripper none] [--force-convert] [--headless] [--export-layout FILE.usd]

ROS: FollowJointTrajectory of the controller + /joint_states, gripper command / result, vacuum command / result, wrist camera capture. The map brings the machines
(laser marker, conveyor), pallets of crates with parts and cameras; a click on a part prints its pose and the TCP that grips it.
The cycle of every machine of the map that has one (machine/<name>.py, e.g. pump_cover_pnp) starts with the simulation; cycles:=false leaves them off.
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import sim_config

ASSETS = sim_config.ASSETS
MACHINES = sim_config.MACHINE_ASSETS
GRIPPERS = sim_config.GRIPPER_ASSETS

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("options", nargs="*", metavar="robot:=<name> map:=<map>",
                    help="ROS style `robot:=ur5e` (robot name or robot config file; a bare name works too; default: scene.robot of the map) "
                         "and `map:=SIDE_PLATE_MAP` (configs/map/<map>.yaml; without it: only the robot and the floor)")
parser.add_argument("--name", help="robot name (default: name in the config)")
parser.add_argument("--gripper", help="gripper name in ../assets/grippper/<gripper>/urdf/, none = no gripper (default: gripper in the config)")
parser.add_argument("--export-layout", metavar="FILE.usd", help="build the scene of the map, save its layout (the map USD with map_edits, the pallets with their crates, the machines, platform, "
                    "floor; no robot, no parts) as FILE.usd and exit")
parser.add_argument("--force-convert", action="store_true", help="convert the URDF again even if the cache is current")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.config, args.map, args.objects, args.cycles = None, None, [], True
for _a in args.options:
    if _a.startswith("cycles:="):
        args.cycles = _a[len("cycles:="):].strip().lower() not in ("false", "0", "no", "off")
    elif _a.startswith("map:="):
        args.map = _a[len("map:="):]
    elif _a.startswith("obj:="):
        args.objects.append(_a[len("obj:="):])
    elif _a.startswith("robot:="):
        args.config = _a[len("robot:="):] or None
    elif _a not in ("robot", "map"):
        args.config = _a
try:
    MAP_FILE = sim_config.map_path(args.map)
    MAP = sim_config.load_map(args.map)
except ValueError as e:
    sys.exit(str(e))


def start_machine_cycles():
    import subprocess
    distro = next((d for d in ("jazzy", "humble", "rolling") if os.path.isfile(f"/opt/ros/{d}/setup.bash")), None)
    if not args.cycles or distro is None:
        return []
    keep = ("HOME", "USER", "LANG", "DISPLAY", "ROS_DOMAIN_ID", "ROS_AUTOMATIC_DISCOVERY_RANGE", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PATH"] = "/usr/bin:/bin"
    procs = []
    for m in (MAP.get("scene") or {}).get("machines") or []:
        script = os.path.join(sim_config.MACHINE_SCRIPTS, f"{m['name']}.py")
        if os.path.isfile(script):
            procs.append(subprocess.Popen(["bash", "-c", f"source /opt/ros/{distro}/setup.bash && exec /usr/bin/python3 {script}"], env=env, cwd=os.path.dirname(script)))
            print(f"[INFO] machine cycle {m['name']} started ({script}, pid {procs[-1].pid}); cycles:=false leaves it off", flush=True)
    return procs


def stop_machine_cycles(procs):
    import signal
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()


def attach_gripper(robot_urdf, gripper_urdf, out):
    robot, grip = ET.parse(robot_urdf).getroot(), ET.parse(gripper_urdf).getroot()
    for root, src in ((robot, robot_urdf), (grip, gripper_urdf)):
        for m in root.iter("mesh"):
            m.set("filename", sim_config.find_mesh(m.get("filename"), os.path.dirname(src)))
    names = lambda r, tag: {e.get("name") for e in r.findall(tag)}
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


def urdf_joints(urdf):
    root = ET.parse(urdf).getroot()
    joints = {}
    for j in root.findall("joint"):
        if j.get("type") in ("revolute", "continuous", "prismatic"):
            lim = j.find("limit")
            g = (lambda k, d: float(lim.get(k)) if lim is not None and lim.get(k) else d)
            cont = j.get("type") == "continuous"
            joints[j.get("name")] = (j.get("type"), -math.inf if cont else g("lower", 0.0),
                                     math.inf if cont else g("upper", 0.0), g("velocity", 1.0), g("effort", 0.0))
    mimic = {j.get("name"): (m.get("joint"), float(m.get("multiplier", 1.0)), float(m.get("offset", 0.0)))
             for j in root.findall("joint") if (m := j.find("mimic")) is not None and j.get("name") in joints}
    return joints, mimic


def missing_meshes(urdf):
    return [m.get("filename") for m in ET.parse(urdf).getroot().iter("mesh") if not sim_config.find_mesh(m.get("filename", ""), os.path.dirname(urdf))]


def any_parts(counts):
    return any(any_parts(c) for c in counts) if isinstance(counts, (list, tuple)) else bool(counts)


def load_settings(path, name, gripper, pose=None):
    raw = yaml.safe_load(open(path)) or {}
    s = sim_config.deep_merge({k: v for k, v in raw.items() if k != "sim"}, raw.get("sim") or {})
    s["name"] = name or s.get("name")
    if not s["name"]:
        sys.exit(f"no robot name: set `name:` in {path} or pass --name")
    g = s.get("gripper") if gripper is None else gripper
    s["gripper"] = None if g is None or str(g).lower() in ("", "none", "null") else str(g)
    urdfs = [sim_config.find_urdf(s["name"])]
    if s["gripper"]:
        urdfs.append(os.path.join(GRIPPERS, s["gripper"], "urdf", f"{s['gripper']}.urdf"))
    for u in urdfs:
        if not os.path.isfile(u):
            sys.exit(f"URDF not found: {u}")
        miss = missing_meshes(u)
        if miss:
            sys.exit(f"{len(miss)} mesh(es) of {u} not found: {miss[:5]}")
    s["model"] = "_".join([s["name"]] + ([s["gripper"]] if s["gripper"] else []))
    s["urdf"] = (attach_gripper(*urdfs, os.path.join(REPO, "assets_cache", f"robot_{s['model']}", f"{s['model']}.urdf"))
                 if s["gripper"] else urdfs[0])
    s["joints"], s["mimic"] = urdf_joints(s["urdf"])
    builtin = list(s.get("gripper_joints") or [])
    bad = [n for n in builtin if n not in s["joints"]]
    if bad:
        sys.exit(f"gripper_joints of {path} are not movable joints of {s['urdf']}: {bad}")
    s["gripper_joints"] = [n for n in (j.get("name") for j in ET.parse(urdfs[1]).getroot().findall("joint"))
                           if n in s["joints"] and n not in s["mimic"]] if s["gripper"] else builtin
    g_root = ET.parse(urdfs[1]).getroot() if s["gripper"] else None
    s["gripper_links"] = [ln.get("name") for ln in g_root.findall("link")
                          if ln.get("name") in {j.find("child").get("link") for j in g_root.findall("joint")}] if s["gripper"] else []
    s["sim_consts"] = {k: t(s[key]) for k, key, t in (("ARRIVE_TOL", "arrive_tolerance_rad", float), ("SETTLE_FRAMES", "arrive_settle_frames", int),
                                                        ("GRIP_GRACE", "gripper_grace_frames", int), ("GRIP_WINDOW", "gripper_window_frames", int),
                                                        ("ACCEL_S", "accel_s", float)) if s.get(key) is not None}
    from binsim.config import _load_yaml_tree
    scene = MAP.get("scene") or {}
    s["pos_m"], s["rpy_deg"] = sim_config.robot_pose(s, scene if pose is None else {**scene, "robot_pose": pose})
    objects = []
    for o in list(scene.get("objects") or []) + [{"part": name} for name in args.objects]:
        f = os.path.join(sim_config.OBJECTS_DIR, f"{o['part']}.yaml")
        if not os.path.isfile(f):
            print(f"[WARN] object {o['part']} skipped: part config not found: {f}", flush=True)
            continue
        part = _load_yaml_tree(f)
        if not os.path.isfile(part.get("mesh") or ""):
            print(f"[WARN] object {o['part']} skipped: mesh of {f} not found: {part.get('mesh')}", flush=True)
            continue
        objects.append({"pos_m": part.get("pos_m", [0, 0, 0]), "rpy_deg": part.get("rpy_deg", [0, 0, 0]), **o, "cfg": part})
    s["objects"] = objects
    b, s["bin"] = scene.get("bin") or None, None
    if b and not (MAP.get("bin") and MAP.get("loading")):
        print(f"[WARN] bin skipped: scene.bin of {MAP_FILE} needs the dataset's `bin:` (e.g. defaults: [../object/bin]) and `loading:` sections in the map", flush=True)
    elif b:
        pile = b.get("pile") or {}
        seed = pile.get("seed", "random")
        seed = int.from_bytes(os.urandom(4), "big") if seed is None or str(seed).lower() == "random" else int(seed)
        part, count = None, int(pile.get("count", 0))
        if pile.get("part"):
            f = os.path.join(sim_config.OBJECTS_DIR, f"{pile['part']}.yaml")
            if os.path.isfile(f):
                part = _load_yaml_tree(f)
            else:
                print(f"[WARN] pile of {pile['part']} skipped (the bin stays empty): part config not found: {f}", flush=True)
                count = 0
        s["bin"] = {"cfg": MAP["bin"], "loading": MAP["loading"], "pos_m": b.get("pos_m", [0, 0, 0]),
                    "rpy_deg": b.get("rpy_deg", [0, 0, 0]), "part": part, "count": count,
                    "seed": seed}
    choice = scene.get("map")
    s["map"] = None if not choice or str(choice).lower() in ("none", "null") else map_file(str(choice))
    s["platform"] = pf = scene.get("platform") or None
    if pf and not pf.get("size_m"):
        print(f"[WARN] platform skipped: platform of {MAP_FILE} (scene:) needs size_m [x, y, z]", flush=True)
        s["platform"] = None
    s["map_edits"] = scene.get("map_edits") or []
    s["map_collision"] = scene.get("map_collision") or []
    s["map_materials"] = scene.get("map_materials") or []
    s["props"] = scene.get("props") or []
    s["fence"] = scene.get("fence") or None
    s["click_refine"] = scene.get("click_refine", True)
    s["floor_color"] = scene.get("floor_color")
    s["pallets"] = pc = scene.get("pallets") or None
    if pc:
        try:
            from binsim.pallets import DEFAULTS as PALLET_DEFAULTS
            crate = dict(pc["crate"])
            usd = None
            if crate.get("config"):
                bf = os.path.join(sim_config.OBJECTS_DIR, f"{crate['config']}.yaml")
                if not os.path.isfile(bf):
                    raise FileNotFoundError(f"crate config not found: {bf}")
                box = _load_yaml_tree(bf)["bin"]
                crate.setdefault("size_m", box["size_m"])
                crate.setdefault("inner_size_m", list(box["inner_size_m"][:2]))
                crate.setdefault("floor_z_m", box["floor_z_m"])
                crate["bin_cfg"] = box
            else:
                usd = crate["usd"] if os.path.isabs(crate["usd"]) else os.path.normpath(os.path.join(os.path.dirname(MAP_FILE), crate["usd"]))
                if not os.path.isfile(usd):
                    raise FileNotFoundError(f"crate USD not found: {usd}")
            missing = [k for k in ("size_m", "inner_size_m", "floor_z_m") if k not in crate] + [k for k in ("items",) if not pc.get(k)]
            if not pc.get("part") and any_parts(pc.get("parts_per_crate", PALLET_DEFAULTS["parts_per_crate"])):
                missing.append("part (configs/object/<part>.yaml for parts_per_crate; parts_per_crate: 0 = empty crates)")
            if missing:
                raise ValueError(f"pallets needs {', '.join(missing)}")
            pc["part_cfg"] = None
            if pc.get("part"):
                f = os.path.join(sim_config.OBJECTS_DIR, f"{pc['part']}.yaml")
                if not os.path.isfile(f):
                    raise FileNotFoundError(f"part config not found: {f}")
                pc["part_cfg"] = _load_yaml_tree(f)
            seed = pc.get("seed", "random")
            pc["seed"] = int.from_bytes(os.urandom(4), "big") if seed is None or str(seed).lower() == "random" else int(seed)
            pc["crate"] = {**crate, "usd": usd}
            s["pallet_physics"] = {(int(pl_.get("pallet", 0)), int(ly.get("layer", 0))): (bool(ly.get("physics", True)), None if ly.get("crates") is None else {int(i) for i in ly["crates"]}, bool(ly.get("switch")))
                                   for pl_ in (pc.get("physics") or []) if isinstance(pl_, dict) for ly in (pl_.get("layers") or []) if isinstance(ly, dict)}
        except Exception as e:
            print(f"[WARN] pallets skipped: {type(e).__name__}: {e}", flush=True)
            s["pallets"] = None
    s["basket"] = k = scene.get("basket") or None
    if k and any(key not in k for key in ("pos_m", "size_m", "wall_thickness_m")):
        print(f"[WARN] basket skipped: basket of {MAP_FILE} (scene:) needs pos_m, size_m and wall_thickness_m", flush=True)
        s["basket"] = None
    s["cameras"] = [dict(c) for c in scene.get("cameras") or []]
    s["machines"] = []
    for entry in scene.get("machines") or []:
        m = sim_config.machine_config(entry)
        urdf = (os.path.join(MACHINES, str(m["urdf"])) if m.get("urdf")
                else os.path.join(MACHINES, str(m.get("name")), "urdf", f"{m.get('name')}.urdf"))
        if not os.path.isfile(urdf):
            print(f"[WARN] machine {m.get('name')} skipped: URDF not found: {urdf}", flush=True)
            continue
        miss = missing_meshes(urdf)
        if miss:
            print(f"[WARN] machine {m['name']} skipped: {len(miss)} mesh(es) of {urdf} not found: {miss[:3]}", flush=True)
            continue
        joints, mimic = urdf_joints(urdf)
        s["machines"].append({**m, "entry": entry, "model": f"machine_{m['name']}", "urdf": urdf, "joints": joints, "mimic": mimic,
                              "friction_override": (m.get("fixtures") or {}).get("friction")})
    return s


def map_file(choice):
    if os.path.isfile(choice):
        return os.path.abspath(choice)
    d = os.path.join(ASSETS, "map")
    for ext in ("", ".usd", ".usda", ".usdc"):
        if os.path.isfile(os.path.join(d, choice + ext)):
            return os.path.join(d, choice + ext)
    names = sorted({os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith((".usd", ".usda", ".usdc"))}) if os.path.isdir(d) else []
    sys.exit(f"map USD not found: {choice}; USDs in {d}: {', '.join(names) or '-'}")


try:
    ENTRIES = sim_config.scene_robots(MAP.get("scene") or {}, args.config)
    CONFIGS = [sim_config.robot_config_path(e["name"]) for e in ENTRIES]
except ValueError as e:
    sys.exit(str(e))
print(f"[INFO] map: {os.path.relpath(MAP_FILE, REPO) if MAP_FILE else 'none (the robot and the floor)'}; robots: "
      + ", ".join(f"{os.path.relpath(c, REPO)} (id {e['id'] if e['id'] is not None else 'of the environment'}, pose {e['pose']})" for e, c in zip(ENTRIES, CONFIGS)), flush=True)
robots = []
for _i, (_e, _c) in enumerate(zip(ENTRIES, CONFIGS)):
    _r = load_settings(os.path.abspath(_c), args.name, args.gripper, _e["pose"])
    _r["id"], _r["prim"] = _e["id"], "/World/robot" if len(ENTRIES) == 1 else f"/World/robot_{_i}"
    robots.append(_r)
robot = robots[0]
sim_config.write_current([{"name": r["name"], "id": r["id"]} for r in robots],
                         os.path.splitext(os.path.basename(MAP_FILE))[0] if MAP_FILE else None)
if importlib.util.find_spec("rclpy") is None:
    sys.exit("rclpy not found: source ROS 2 first (source /opt/ros/jazzy/setup.bash)")
args.experience = args.experience or os.path.expanduser(
    "~/IsaacSim/_build/linux-aarch64/release/apps/isaacsim.exp.full.kit")
if not args.headless:
    args.visualizer = args.visualizer or ["kit"]
simulation_app = AppLauncher(args).app

import numpy as np
import omni.timeline
import omni.ui as ui
import omni.usd
import rclpy
from control_msgs.action import FollowJointTrajectory
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty, Float64, Int32, String
from trajectory_msgs.msg import JointTrajectory

from binsim.camera import Intrinsics
from binsim.lens import LensWarp, lens_intrinsics, pixel_rays, render_intrinsics
from binsim.transforms import make_T, mat_to_quat, quat_to_mat, rpy_to_mat
sys.path.insert(0, os.path.join(os.path.dirname(REPO), "control"))
import compute_grasp_tcp_pose as grasp_tcp

APP_HZ = 60.0
ROBOT_PATH = "/World/robot"
ARRIVE_TOL, STALL_MOVE, STALL_FRAMES, SETTLE_MOVE, SETTLE_FRAMES = 0.01, 0.01, 60, 0.0005, 18
STALL_ACCEPT = 0.035
ACCEL_S, RAMP_FLOOR = 0.0, 0.03
GRIP_TOL, GRIP_GRACE, GRIP_WINDOW = 0.0002, 18, 18
MACHINE_STIFFNESS = {"angular": 100.0, "linear": 5000.0}
MACHINE_DAMPING = {"angular": 2.5, "linear": 100.0}
MACHINE_ACCEL = {"angular": 3.0, "linear": 0.5}
MACHINE_LEAD = 0.08
CARRIER_ACCEL = 0.5
PART_TOPIC = "/part_turned_over"
TARGET_RADIUS = 0.04
STIFF_KEEP = 2
CORNER_TOL, CORNER_FRAMES = 0.015, 45
NEAR_TOL = 0.03
CLICK_HULLS = 1000
CLICK_VOXELS = 3000000
HULLS_ATTR = "physxConvexDecompositionCollision:maxConvexHulls"
VOXELS_ATTR, VOXELS_DEFAULT = "physxConvexDecompositionCollision:voxelResolution", 500000
DECOMPOSITION_ATTRS = {"max_hulls": ("maxConvexHulls", "Int"), "voxel_resolution": ("voxelResolution", "Int"), "error_percentage": ("errorPercentage", "Float"),
                       "min_thickness": ("minThickness", "Float"), "hull_vertex_limit": ("hullVertexLimit", "Int"), "shrink_wrap": ("shrinkWrap", "Bool")}
CLICK_KEEP = 10
CARRIED_M = 0.05
CLICK_PHYSX = {"CreateSolverPositionIterationCountAttr": (16, 4), "CreateSolverVelocityIterationCountAttr": (2, 1),
               "CreateEnableCCDAttr": (True, False), "CreateMaxDepenetrationVelocityAttr": (0.5, 1.0)}
PART_PHYSX = {"CreateLinearDampingAttr": 0.2, "CreateAngularDampingAttr": 2.0, "CreateSleepThresholdAttr": 0.01, "CreateStabilizationThresholdAttr": 0.005,
              "CreateMaxDepenetrationVelocityAttr": 1.0}
CONTACT_OFFSET_M, REST_OFFSET_M = 0.003, 0.0


class RobotBridge(Node):

    def __init__(self, names, q0, controller, mimic, gripper, gripper_topic, limits, context=None):
        super().__init__("robot_sim", context=context)
        self.names = list(names)
        self._lo = [lo for lo, _ in limits]
        self._hi = [hi for _, hi in limits]
        self._idx = {n: i for i, n in enumerate(self.names)}
        self._mimic = {n: m for n, m in mimic.items() if n in self._idx and m[0] in self._idx}
        self._gripper = {n: lim for n, lim in gripper.items() if n in self._idx}
        if self._gripper:
            self.create_subscription(Float64, gripper_topic, self._on_gripper, 10)
        self.slow = False
        self.create_subscription(Bool, "/motion/slow", lambda m: setattr(self, "slow", bool(m.data)), 10)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.frame = 0
        self._targets = list(q0)
        self._current = list(q0)
        self._wps = deque()
        self.grip_events = deque()
        self._grip_v = None
        self._hand = np.array([n in self._gripper or n in self._mimic for n in self.names])
        self._share = 0.0
        self._corner_frames = 0
        self.events = deque(maxlen=40)
        self._goal = None
        self._grip = None
        self._grip_pub = self.create_publisher(String, f"{gripper_topic}/result", 10) if self._gripper else None
        self._part_pub = self.create_publisher(Bool, PART_TOPIC, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(JointTrajectory, f"{controller}/joint_trajectory", self._on_traj, 10)
        self._action = ActionServer(
            self, FollowJointTrajectory, f"{controller}/follow_joint_trajectory",
            execute_callback=self._execute, goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=ReentrantCallbackGroup())
        for other in self.sibling_controllers(controller):
            self.create_subscription(JointTrajectory, f"{other}/joint_trajectory", self._on_traj, 10)
            self._aliases.append(ActionServer(
                self, FollowJointTrajectory, f"{other}/follow_joint_trajectory",
                execute_callback=self._execute, goal_callback=lambda _: GoalResponse.ACCEPT,
                cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=ReentrantCallbackGroup()))
        self._state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.capture_requested = threading.Event()
        self.create_subscription(Empty, "/wrist_camera/capture", lambda _: self.capture_requested.set(), 10)
        self.pictures = deque()
        self.create_subscription(String, "/vision_pick/picture", lambda m: self.pictures.append(m.data), QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.targets_seen = deque()
        self.create_subscription(Pose, "/target_pose", lambda m: self.targets_seen.append((m.position.x, m.position.y, m.position.z)), 10)

    def sibling_controllers(self, controller):
        self._aliases = []
        found = []
        for f in sorted(glob.glob(os.path.join(REPO, "..", "bringup", "*", "config", "moveit_controllers.yaml"))):
            cfg = (yaml.safe_load(open(f)) or {}).get("moveit_simple_controller_manager") or {}
            for name in cfg.get("controller_names") or []:
                joints = (cfg.get(name) or {}).get("joints") or []
                path = "/" + str(name).strip("/")
                if joints and set(joints) <= set(self.names) and path != controller and path not in found:
                    found.append(path)
        return found

    def _set(self, names, positions):
        with self._lock:
            self._apply(names, positions)

    def _apply(self, names, positions):
        for n, p in zip(names, positions):
            if n in self._idx:
                self._targets[self._idx[n]] = float(p)
        for n, (lead, k, off) in self._mimic.items():
            self._targets[self._idx[n]] = self._targets[self._idx[lead]] * k + off

    def _on_traj(self, msg):
        if msg.points:
            self._set(msg.joint_names, msg.points[-1].positions)

    def publish_turned_over(self, flipped):
        self._part_pub.publish(Bool(data=bool(flipped)))

    def hold_gripper(self):
        with self._lock:
            self._apply(list(self._gripper), [self._current[self._idx[n]] for n in self._gripper])

    def regrip(self):
        if self._grip_v is not None:
            with self._lock:
                self._apply(list(self._gripper), [lo + self._grip_v * (hi - lo) for lo, hi in self._gripper.values()])

    def _on_gripper(self, msg):
        v = min(max(float(msg.data), 0.0), 1.0)
        joint = next(iter(self._gripper))
        lo, hi = self._gripper[joint]
        with self._lock:
            self._apply(list(self._gripper), [lo + v * (hi - lo) for lo, hi in self._gripper.values()])
            self._grip = {"idx": self._idx[joint], "target": lo + v * (hi - lo), "frame": self.frame, "window": deque(maxlen=GRIP_WINDOW if v > 0.5 else max(GRIP_WINDOW, 45)), "v": v}
            self._grip_v = v
            self.grip_events.append(("command", v))
        self.events.append((time.time(), f"gripper command {v:.2f} ({'close' if v == 0 else 'open' if v == 1 else 'partly open'})"))
        self.get_logger().info(f"gripper {v:.2f} ({'closed' if v == 0 else 'open' if v == 1 else 'partly open'})")

    def _execute(self, goal):
        traj = goal.request.trajectory
        self.events.append((time.time(), f"trajectory of {len(traj.points)} waypoints"))
        if not traj.points:
            goal.abort()
            return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.INVALID_GOAL)
        idx = [self._idx[n] for n in traj.joint_names if n in self._idx]
        mine = {"idx": idx, "final": [p for n, p in zip(traj.joint_names, traj.points[-1].positions) if n in self._idx],
                "window": deque(maxlen=SETTLE_FRAMES), "ref": None, "ref_frame": 0, "outcome": None, "near": False}
        with self._cond:
            self._wps = deque((list(traj.joint_names), list(pt.positions)) for pt in traj.points)
            self._goal = mine
            while mine["outcome"] is None and self._goal is mine:
                if goal.is_cancel_requested:
                    break
                self._cond.wait(1.0)
            if self._goal is mine:
                self._goal = None
                self._wps.clear()
        if mine["outcome"] is None:
            if goal.is_cancel_requested:
                goal.canceled()
                return FollowJointTrajectory.Result()
            goal.abort()
            return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.INVALID_GOAL)
        if mine["outcome"] == "stalled":
            k = max(range(len(idx)), key=lambda i: mine["error"][i])
            self.get_logger().warn(f"arm stopped {math.degrees(mine['error'][k]):.2f} deg short of the goal at {self.names[idx[k]]} (blocked?) -> trajectory aborted")
            goal.abort()
            return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED)
        self.events.append((time.time(), "trajectory arrived"))
        goal.succeed()
        return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.SUCCESSFUL)

    def _track_arm(self, q):
        g = self._goal
        if g is None or g["outcome"] is not None or len(self._wps) > 1:
            return
        pos = [q[i] for i in g["idx"]]
        error = [abs(p - f) for p, f in zip(pos, g["final"])]
        if not g["near"] and max(error) < NEAR_TOL:
            g["near"] = True
            self.grip_events.append(("near", True))
        g["window"].append(pos)
        if g["ref"] is None or max(abs(a - b) for a, b in zip(pos, g["ref"])) > STALL_MOVE:
            g["ref"], g["ref_frame"] = pos, self.frame
        at_rest = len(g["window"]) == SETTLE_FRAMES and max(max(c) - min(c) for c in zip(*g["window"])) < SETTLE_MOVE
        if max(error) < ARRIVE_TOL and (at_rest or self.frame - g["ref_frame"] > STALL_FRAMES):
            g["outcome"] = "arrived"
        elif self.frame - g["ref_frame"] > STALL_FRAMES and max(error) < STALL_ACCEPT:
            g["outcome"] = "arrived"
            k = max(range(len(error)), key=lambda i: error[i])
            self.events.append((time.time(), f"arrived {math.degrees(error[k]):.2f} deg short at {self.names[g['idx'][k]]} (within the goal tolerance)"))
        elif self.frame - g["ref_frame"] > STALL_FRAMES:
            g["outcome"], g["error"] = "stalled", error

    def _track_gripper(self, q):
        g = self._grip
        if g is None:
            return
        pos = q[g["idx"]]
        g["window"].append(pos)
        if abs(pos - g["target"]) < GRIP_TOL:
            outcome = "reached"
        elif (self.frame - g["frame"] > (GRIP_GRACE if g["v"] > 0.5 else max(GRIP_GRACE, 30)) + g["window"].maxlen and len(g["window"]) == g["window"].maxlen
              and max(g["window"]) - min(g["window"]) < GRIP_TOL):
            outcome = "blocked"
        else:
            return
        self._grip = None
        self.grip_events.append(("done", g["v"], float(pos), outcome))
        self.get_logger().info(f"gripper {outcome}: position {pos:.4f} m (command {g['target']:.4f} m)")
        self._grip_pub.publish(String(data=outcome))

    def limit(self, cmd, target, step):
        delta = np.asarray(target, float) - cmd
        out = np.clip(delta, -step, step)
        arm = ~self._hand
        k = max(1.0, float(np.max(np.abs(delta[arm]) / np.maximum(step[arm], 1e-12)))) if arm.any() else 1.0
        out[arm] = delta[arm] / k
        return cmd + out

    def ramp(self, cmd, step):
        if ACCEL_S <= 0:
            return step
        with self._lock:
            if not self._wps:
                self._share = 0.0
                return step
            wps, k = list(self._wps), 0
            while k + 1 < len(wps) - 1 and wps[k + 1][1] != wps[k][1]:
                k += 1
            joints = [self._idx[n] for n in wps[0][0] if n in self._idx]
            path = np.array([[p for n, p in zip(w[0], w[1]) if n in self._idx] for w in wps[:k + 2]])
            left = np.abs(path[0] - np.array(cmd)[joints]) + np.abs(np.diff(path, axis=0)).sum(axis=0)
            seconds = float(np.max(left / np.maximum(np.array(step)[joints] * APP_HZ, 1e-9)))
            self._share = min(self._share + 1.0 / (ACCEL_S * APP_HZ), math.sqrt(2.0 * seconds / ACCEL_S), 1.0)
            return step * max(self._share, RAMP_FLOOR)

    def targets(self, cmd=None, step=0.0):
        with self._lock:
            while self._wps:
                names, pos = self._wps[0]
                self._apply(names, pos)
                if cmd is None or len(self._wps) == 1:
                    break
                joints = [self._idx[n] for n in names if n in self._idx]
                if self._wps[1][1] == pos:
                    err = np.abs(np.array(self._targets)[joints] - np.array(self._current)[joints])
                    self._corner_frames += 1
                    if (err > CORNER_TOL).any() and self._corner_frames < CORNER_FRAMES:
                        break
                    self._corner_frames = 0
                    self._wps.popleft()
                    last = not any(self._wps[i][1] == self._wps[i + 1][1] for i in range(len(self._wps) - 1))
                    self.grip_events.append(("corner", last))
                    continue
                if (np.abs(np.array(self._targets)[joints] - np.array(cmd)[joints]) > np.array(step)[joints]).any():
                    break
                self._wps.popleft()
            return list(self._targets)

    def report(self, q, v):
        q = [min(max(float(x), lo), hi) for x, lo, hi in zip(q, self._lo, self._hi)]
        with self._cond:
            self._current = [float(x) for x in q]
            self.frame += 1
            self._track_arm(self._current)
            self._track_gripper(self._current)
            self._cond.notify_all()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position, msg.velocity = self.names, q, [float(x) for x in v]
        self._state_pub.publish(msg)


def part_root(path):
    p = path.split("/")
    if len(p) >= 4 and p[1] == "World" and p[2] in ("pile", "objects"):
        return "/".join(p[:4])
    if len(p) >= 5 and p[1] == "World" and p[2] == "pallets" and re.fullmatch(r"L\d+_C\d+_.+", p[4]):
        return "/".join(p[:5])
    return None


def update(n=1):
    for _ in range(n):
        simulation_app.update()


def ghost_root(urdf):
    root = ET.parse(urdf).getroot()
    children = {j.find("child").get("link") for j in root.findall("joint")}
    for ln in root.findall("link"):
        if ln.get("name") not in children:
            return all(ln.find(t) is None for t in ("visual", "collision", "inertial"))
    return False


def meshes_newer(urdf, usd_dir):
    made = max((os.path.getmtime(os.path.join(d, f)) for d, _, fs in os.walk(usd_dir) for f in fs if f.endswith((".usd", ".usda"))), default=0.0)
    meshes = (sim_config.find_mesh(m.get("filename"), os.path.dirname(urdf)) for m in ET.parse(urdf).getroot().iter("mesh"))
    return any(p and os.path.getmtime(p) > made for p in meshes)


def robot_usd(s, force):
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    usd_dir = os.path.join(REPO, "assets_cache", f"robot_{s['model']}")
    if not force and meshes_newer(s["urdf"], usd_dir):
        print("[INFO] a mesh of the robot URDF changed: converting the robot to USD again", flush=True)
        force = True
    conv = UrdfConverter(UrdfConverterCfg(
        asset_path=s["urdf"], usd_dir=usd_dir,
        fix_base=not ghost_root(s["urdf"]), merge_fixed_joints=False, force_usd_conversion=force,
        collision_from_visuals="collision_type" in s, collision_type=str(s.get("collision_type", "Convex Hull"))))
    return conv.usd_path


def read_stl(path):
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw[80:84], "<u4")[0])
    rec = np.frombuffer(raw[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"].astype(np.float32), rec["n"].astype(np.float32)


def urdf_chain(urdf):
    root = ET.parse(urdf).getroot()
    xyz = lambda e: [float(v) for v in (e.find("origin").get("xyz") or "0 0 0").split()] if e.find("origin") is not None else [0.0, 0.0, 0.0]
    joints = []
    for j in root.findall("joint"):
        ax = j.find("axis")
        joints.append({"name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                       "xyz": xyz(j), "axis": [float(v) for v in ax.get("xyz").split()] if ax is not None else [1.0, 0.0, 0.0]})
    colors, links = {}, {}
    for ln in root.findall("link"):
        vis = []
        for v in ln.findall("visual"):
            mesh, mat = v.find("geometry/mesh"), v.find("material")
            if mesh is None:
                continue
            col = mat.find("color") if mat is not None else None
            if col is not None:
                colors[mat.get("name")] = [float(c) for c in col.get("rgba").split()]
            vis.append({"file": mesh.get("filename"), "xyz": xyz(v), "material": mat.get("name") if mat is not None else None})
        links[ln.get("name")] = vis
    for vis in links.values():
        for v in vis:
            v["rgba"] = colors.get(v["material"], [0.5, 0.5, 0.5, 1.0])
    return links, joints


def kinematic_usd(s):
    links, joints = urdf_chain(s["urdf"])
    out_dir = os.path.join(REPO, "assets_cache", f"robot_{s['model']}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{s['name']}_kinematic.usdc")
    files = {v["file"]: sim_config.find_mesh(v["file"], os.path.dirname(s["urdf"])) for vis in links.values() for v in vis}
    if os.path.isfile(path) and os.path.getmtime(path) > max([os.path.getmtime(s["urdf"])] + [os.path.getmtime(f) for f in files.values()]):
        return path
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    top = f"/{s['name']}"
    UsdGeom.Xform.Define(stage, top)
    stage.SetDefaultPrim(stage.GetPrimAtPath(top))
    base = next(n for n in links if n not in {j["child"] for j in joints})
    paths = {base: f"{top}/{base}"}
    UsdGeom.Xform.Define(stage, paths[base])
    for j in joints:
        paths[j["child"]] = f"{paths[j['parent']]}/{j['child']}"
        UsdGeom.Xform.Define(stage, paths[j["child"]]).AddTransformOp().Set(Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*j["xyz"])))
    mats = {}

    def material(rgba):
        key = tuple(rgba[:3])
        if key not in mats:
            mp = f"{top}/Looks/m{len(mats)}"
            mat, sh = UsdShade.Material.Define(stage, mp), UsdShade.Shader.Define(stage, mp + "/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*key))
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            mats[key] = mat
        return mats[key]
    for name, vis in links.items():
        for k, v in enumerate(vis):
            UsdGeom.Xform.Define(stage, f"{paths[name]}/visual_{k}").AddTransformOp().Set(Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*v["xyz"])))
            tri, nrm = read_stl(files[v["file"]])
            mesh = UsdGeom.Mesh.Define(stage, f"{paths[name]}/visual_{k}/mesh")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(tri.reshape(-1, 3)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tri), 3, np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(3 * len(tri), dtype=np.int32)))
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(nrm, 3, axis=0)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            mesh.CreateSubdivisionSchemeAttr("none")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material(v["rgba"]))
    stage.GetRootLayer().Save()
    return path


class NumpyBox:

    def __init__(self, a):
        self.a = a

    def numpy(self):
        return self.a


class KinArticulation:

    def __init__(self, m, root):
        self.root = root
        _, joints = urdf_chain(m["urdf"])
        self.joints = [j for j in joints if j["type"] in ("prismatic", "revolute", "continuous")]
        self.dof_names = [j["name"] for j in self.joints]
        self.q = np.array([m["q0"].get(n, 0.0) for n in self.dof_names], float)
        self.ops = None
        self.shown = None

    def apply(self):
        if self.ops is None:
            prims = {q.GetName(): q for q in Usd.PrimRange(omni.usd.get_context().get_stage().GetPrimAtPath(self.root))}
            self.ops = [UsdGeom.Xformable(prims[j["child"]]).GetOrderedXformOps()[0] for j in self.joints]
        for op, j, v in zip(self.ops, self.joints, self.q):
            motion = Gf.Matrix4d(1.0)
            if j["type"] == "prismatic":
                motion.SetTranslate(Gf.Vec3d(*[a * float(v) for a in j["axis"]]))
            else:
                motion.SetRotate(Gf.Rotation(Gf.Vec3d(*j["axis"]), math.degrees(float(v))))
            op.Set(motion * Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*j["xyz"])))
        self.shown = self.q.copy()

    def set_dof_position_targets(self, cmd):
        self.q = np.array(cmd, float)
        if self.shown is None or not np.array_equal(self.shown, self.q):
            self.apply()

    def set_dof_positions(self, *a, **k):
        pass

    def set_dof_velocities(self, *a, **k):
        pass

    def get_dof_positions(self):
        return NumpyBox(self.q[None].copy())

    def get_dof_velocities(self):
        return NumpyBox(np.zeros((1, len(self.q))))


def plan_pile(b, cache_dir):
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


def plan_pallet_crates(pc, cache_dir):
    from binsim.assets import prepare_bin, prepare_part
    from binsim.pallets import plan_pallets
    if pc["crate"].get("bin_cfg"):
        pc["crate"]["usd"] = prepare_bin(pc["crate"]["bin_cfg"], cache_dir).usd_path
    pa = prepare_part(pc["part_cfg"], cache_dir) if pc.get("part_cfg") else None
    cfg = {k: ({kk: vv for kk, vv in v.items() if kk != "bin_cfg"} if k == "crate" else v) for k, v in pc.items() if k not in ("part", "part_cfg", "seed", "physics")}
    if pa is None:
        cfg["parts_per_crate"] = 0
    plan = plan_pallets(cfg, pa.extents if pa else [0.01, 0.01, 0.01], (pc.get("part_cfg") or {}).get("layer_rpy_deg"), np.random.default_rng(pc["seed"]),
                         pa.mesh.bounds.mean(axis=0) if pa else (0.0, 0.0, 0.0))
    plan["part_usd"] = pa.usd_path if pa else None
    return plan


def wrist_lens(cfg):
    if "_lens" not in cfg:
        w, h = (int(v) for v in cfg["resolution"])
        intr = Intrinsics(w, h, float(cfg["fx"]), float(cfg["fy"]), (w - 1) / 2.0, (h - 1) / 2.0)
        lens = lens_intrinsics(intr, {"resolution": [w, h], "lens": cfg.get("lens")})
        if lens is None:
            cfg["_lens"] = (intr, intr, None)
        else:
            render = render_intrinsics(lens)
            cfg["_lens"] = (lens, render, LensWarp(lens, render))
    return cfg["_lens"]


def make_wrist_camera(stage, cfg, root=ROBOT_PATH):
    link = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.GetName() == str(cfg["link"])]
    if not link:
        print(f"[WARN] wrist_camera skipped: no link {cfg['link']} in the robot (is it in the URDF?)", flush=True)
        return None
    _, render, _ = wrist_lens(cfg)
    w, h = render.width, render.height
    cam = UsdGeom.Camera.Define(stage, f"{link[0].GetPath()}/wrist_camera")
    focal = 24.0
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(float(w * focal / float(cfg["fx"])))
    cam.CreateVerticalApertureAttr(float(h * focal / float(cfg["fy"])))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    UsdGeom.Xformable(cam).AddOrientOp().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))
    return cam.GetPath().pathString


def make_fixed_camera(stage, cfg):
    name = str(cfg["name"])
    pos, tgt = np.array(cfg["pos_m"], float), np.array(cfg["look_at_m"], float)
    f = (tgt - pos) / np.linalg.norm(tgt - pos)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([1.0, 0.0, 0.0])
    u = np.cross(r, f)
    M = Gf.Matrix4d(*r, 0.0, *u, 0.0, *(-f), 0.0, *pos, 1.0)
    root = f"/World/cameras/{name}"
    UsdGeom.Xform.Define(stage, "/World/cameras")
    xf = UsdGeom.Xform.Define(stage, root)
    xf.AddTransformOp().Set(M)
    w, h = (int(v) for v in cfg.get("resolution", [1280, 960]))
    focal = 24.0
    cam = UsdGeom.Camera.Define(stage, f"{root}/camera")
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(float(2.0 * focal * math.tan(math.radians(float(cfg.get("fov_deg", 40.0))) / 2.0)))
    cam.CreateVerticalApertureAttr(float(2.0 * focal * math.tan(math.radians(float(cfg.get("fov_deg", 40.0))) / 2.0) * h / w))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 10.0))
    return cam.GetPath().pathString


def capture_fixed_camera(cfg, cam_path):
    import omni.replicator.core as rep
    from PIL import Image
    w, h = (int(v) for v in cfg.get("resolution", [1280, 960]))
    out = os.path.expanduser(str(cfg.get("output_dir") or os.path.join(REPO, "assets_cache", "cameras", str(cfg["name"]))))
    os.makedirs(out, exist_ok=True)
    product = rep.create.render_product(cam_path, (w, h))
    rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_a.attach(product)
    try:
        update(int(cfg.get("warmup_frames", 8)))
        rgb = np.asarray(rgb_a.get_data())
        for _ in range(60):
            if rgb.size:
                break
            update(1)
            rgb = np.asarray(rgb_a.get_data())
        if not rgb.size:
            raise RuntimeError("the camera annotator returned no data")
    finally:
        rgb_a.detach()
        product.destroy()
    n = len([f for f in os.listdir(out) if f.endswith(".png")])
    path = os.path.join(out, f"{cfg['name']}_{n:06d}.png")
    Image.fromarray(np.ascontiguousarray(rgb[..., :3], np.uint8)).save(path)
    return path


def urdf_link_pose(urdf, link, q, base_T):
    root = ET.parse(urdf).getroot()
    parent = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain, name = [], link
    while name in parent:
        chain.append(parent[name])
        name = parent[name].find("parent").get("link")
    T = np.array(base_T, np.float64)
    for j in reversed(chain):
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
        rpy = [math.degrees(float(v)) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
        T = T @ make_T(rpy_to_mat(rpy), xyz)
        kind, v = j.get("type"), float(q.get(j.get("name"), 0.0))
        axis = np.array([float(a) for a in j.find("axis").get("xyz").split()]) if j.find("axis") is not None else np.array([1.0, 0.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        if kind in ("revolute", "continuous"):
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            T = T @ make_T(np.eye(3) + math.sin(v) * K + (1 - math.cos(v)) * (K @ K), [0.0, 0.0, 0.0])
        elif kind == "prismatic":
            T = T @ make_T(np.eye(3), axis * v)
    return T


def capture_wrist_camera(cfg, cam_path, T_world_cam):
    import omni.replicator.core as rep
    from PIL import Image
    sensor, render, warp = wrist_lens(cfg)
    w, h = sensor.width, sensor.height
    out = os.path.expanduser(str(cfg["output_dir"]))
    os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out, "depth"), exist_ok=True)
    os.makedirs(os.path.join(out, "pose"), exist_ok=True)
    cam_file = os.path.join(out, "camera.json")
    if os.path.isfile(cam_file) and os.listdir(os.path.join(out, "rgb")):
        old = json.load(open(cam_file))
        if (("undistort_map" in old) != (warp is not None) or tuple(old.get("dist", ())) != tuple(sensor.dist)
                or (old.get("width"), old.get("height"), old.get("fx"), old.get("fy")) != (w, h, float(cfg["fx"]), float(cfg["fy"]))):
            raise RuntimeError(f"{out} holds frames of another camera setting (lens / size / focal length differ from camera.json): "
                               "move them away or set another wrist_camera.output_dir, nothing was captured")
    product = rep.create.render_product(cam_path, (render.width, render.height))
    rgb_a, depth_a = (rep.AnnotatorRegistry.get_annotator(n) for n in ("rgb", "distance_to_image_plane"))
    for a in (rgb_a, depth_a):
        a.attach(product)
    try:
        for _ in range(int(cfg.get("warmup_frames", 8))):
            update(1)
        rgb, depth = np.asarray(rgb_a.get_data()), np.asarray(depth_a.get_data(), np.float32)
        for _ in range(60):
            if rgb.size and depth.size:
                break
            update(1)
            rgb, depth = np.asarray(rgb_a.get_data()), np.asarray(depth_a.get_data(), np.float32)
        if not (rgb.size and depth.size):
            raise RuntimeError("the camera annotators returned no data")
    finally:
        for a in (rgb_a, depth_a):
            a.detach()
        product.destroy()
    scale = float(cfg["depth_scale"])
    lo, hi = (float(v) for v in cfg["depth_range_m"])
    depth = np.squeeze(depth)
    rgb = rgb[..., :3]
    if warp is not None:
        rgb, depth = warp.image(np.ascontiguousarray(rgb, np.uint8)), warp.labels(depth)
    depth = np.where(np.isfinite(depth) & (depth >= lo) & (depth <= hi), depth, 0.0)
    n = len([f for f in os.listdir(os.path.join(out, "rgb")) if f.endswith(".png")])
    Image.fromarray(np.ascontiguousarray(rgb, np.uint8)).save(os.path.join(out, "rgb", f"{n:06d}.png"))
    Image.fromarray(np.clip(np.round(depth * 1000.0 / scale), 0, 65535).astype(np.uint16)).save(os.path.join(out, "depth", f"{n:06d}.png"))
    with open(os.path.join(out, "pose", f"{n:06d}.json"), "w") as f:
        json.dump({"T_world_cam": np.asarray(T_world_cam, np.float64).tolist()}, f)
    info = {"fx": float(cfg["fx"]), "fy": float(cfg["fy"]), "cx": (w - 1) / 2.0, "cy": (h - 1) / 2.0, "width": w, "height": h,
            "depth_scale": scale}
    map_file = os.path.join(out, "undistort_map.npy")
    if warp is not None:
        rays = pixel_rays(sensor)
        np.save(map_file, np.stack([rays[..., 0] * sensor.fx + sensor.cx, rays[..., 1] * sensor.fy + sensor.cy], -1).astype(np.float32))
        info.update(undistort_map="undistort_map.npy", dist=list(sensor.dist))
    with open(os.path.join(out, "camera.json"), "w") as f:
        json.dump(info, f, indent=1)
    return os.path.join(out, "rgb", f"{n:06d}.png"), float((depth > 0).mean())


def make_editable(roots):
    while True:
        inst = [p for r in roots for p in Usd.PrimRange(r) if p.IsInstance()]
        if not inst:
            return
        for p in inst:
            p.SetInstanceable(False)


def solid_rules(lo, hi, rules):
    dims, c, height = np.sort(hi - lo, axis=1), (lo + hi) / 2, hi[:, 2] - lo[:, 2]
    pick = np.full(len(lo), -1)
    for k, rule in enumerate(rules):
        if rule.get("of") is not None:
            continue
        ok = pick < 0
        if rule.get("region_m") is not None:
            x0, x1, y0, y1 = (float(v) for v in rule["region_m"])
            ok &= (c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1)
        if rule.get("top_max_m") is not None:
            ok &= hi[:, 2] <= float(rule["top_max_m"])
        if rule.get("height_min_m") is not None:
            ok &= height >= float(rule["height_min_m"])
        if rule.get("dims_min_m") is not None:
            ok &= (dims >= np.asarray(rule["dims_min_m"], float)).all(1)
        if rule.get("dims_max_m") is not None:
            ok &= (dims <= np.asarray(rule["dims_max_m"], float)).all(1)
        pick[ok] = k
    return pick


def vacuum_tcp(qw, qx, qy, qz):
    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]]) @ np.diag([1.0, -1.0, -1.0])
    return (math.degrees(math.atan2(-R[1, 2], R[2, 2])), math.degrees(math.asin(max(-1.0, min(1.0, R[0, 2])))), math.degrees(math.atan2(-R[0, 1], R[0, 0])),
            1.0 - 2.0 * (qx * qx + qy * qy) < 0.0)


def paint_material(stage, name, rule):
    path = f"/World/map_looks/{name}"
    if stage.GetPrimAtPath(path).IsValid():
        return UsdShade.Material(stage.GetPrimAtPath(path))
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(float(c) for c in rule.get("color", [0.66, 0.67, 0.69]))))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(rule.get("metallic", 0.2)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rule.get("roughness", 0.5)))
    if rule.get("opacity") is not None:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(rule["opacity"]))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def fence_props(spec):
    if not spec:
        return []
    frame = {"look": "fence_frame", "color": [0.42, 0.31, 0.02], "metallic": 0.15, "roughness": 0.55, **(spec.get("frame") or {})}
    panel = {"look": "fence_panel", "color": [0.035, 0.035, 0.04], "metallic": 0.0, "roughness": 0.85, "opacity": 0.25, **(spec.get("panel") or {})}
    rails = sorted(float(z) for z in spec.get("rails_z_m", [0.1625, 1.0125, 1.2125, 2.0875]))
    post, bar, gap = float(spec.get("post_m", 0.044)), float(spec.get("bar_m", 0.025)), float(spec.get("gap_m", 0.035))
    post_h, plate = float(spec.get("post_height_m", 2.05)), float(spec.get("plate_m", 0.12))
    zc, zh = (rails[0] + rails[-1]) / 2.0, rails[-1] - rails[0]
    out, posts = [], {}
    for r, run in enumerate(spec.get("runs") or []):
        for (ax, ay), (bx, by) in zip(run[:-1], run[1:]):
            length = math.hypot(bx - ax, by - ay)
            n = max(1, math.ceil(length / float(spec.get("panel_max_m", 1.5))))
            ux, uy, yaw = (bx - ax) / length, (by - ay) / length, math.degrees(math.atan2(by - ay, bx - ax))
            w = length / n
            for i in range(n + 1):
                posts.setdefault((round(ax + ux * w * i, 4), round(ay + uy * w * i, 4)), yaw)
            for i in range(n):
                t0, t1 = i * w + post / 2.0 + gap, (i + 1) * w - post / 2.0 - gap
                at = lambda t, z: [ax + ux * t, ay + uy * t, z]
                name = f"fence_{r}_{len(out)}"
                out += [{**frame, "name": f"{name}_side_a", "pos_m": at(t0 + bar / 2.0, zc), "size_m": [bar, bar, zh + bar], "rpy_deg": [0, 0, yaw]},
                        {**frame, "name": f"{name}_side_b", "pos_m": at(t1 - bar / 2.0, zc), "size_m": [bar, bar, zh + bar], "rpy_deg": [0, 0, yaw]},
                        *({**frame, "name": f"{name}_rail_{j}", "pos_m": at((t0 + t1) / 2.0, z), "size_m": [t1 - t0, bar, bar], "rpy_deg": [0, 0, yaw]}
                          for j, z in enumerate(rails)),
                        {**panel, "name": f"{name}_panel", "pos_m": at((t0 + t1) / 2.0, zc), "size_m": [t1 - t0 - 2.0 * bar, 0.004, zh - bar], "rpy_deg": [0, 0, yaw]}]
    for k, ((x, y), yaw) in enumerate(posts.items()):
        out += [{**frame, "name": f"fence_post_{k}", "pos_m": [x, y, (post_h + 0.01) / 2.0], "size_m": [post, post, post_h - 0.01], "rpy_deg": [0, 0, yaw]},
                {**frame, "name": f"fence_plate_{k}", "pos_m": [x, y, 0.005], "size_m": [plate, plate, 0.01], "rpy_deg": [0, 0, yaw]}]
    print(f"[INFO] fence: {len(spec.get('runs') or [])} run(s), {len(posts)} posts, {len(out) - 2 * len(posts)} frame/panel pieces", flush=True)
    return out


def paint_solids(stage, mesh_prim, rules, tag):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    mesh = UsdGeom.Mesh(mesh_prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [])
    if not len(counts) or (counts != 3).any():
        print(f"[WARN] map_materials: {mesh_prim.GetPath()} is not a triangle mesh: not painted", flush=True)
        return 0
    M = np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).T
    W = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64) @ M[:3, :3].T + M[:3, 3]
    tri = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    q = np.round(W / 2e-4).astype(np.int64) + (1 << 20)
    weld = np.unique((q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2], return_inverse=True)[1].ravel()[tri]
    n = int(weld.max()) + 1
    a, b = np.r_[weld[:, 0], weld[:, 1], weld[:, 2]], np.r_[weld[:, 1], weld[:, 2], weld[:, 0]]
    solid = connected_components(coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n)), directed=False)[1][weld[:, 0]]
    T = W[tri]
    order = np.argsort(solid, kind="stable")
    cut = np.r_[0, np.flatnonzero(np.diff(solid[order])) + 1]
    lo, hi = np.minimum.reduceat(T.min(1)[order], cut, axis=0), np.maximum.reduceat(T.max(1)[order], cut, axis=0)
    pick = solid_rules(lo, hi, rules)[solid]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    for k, rule in enumerate(rules):
        if rule.get("of") is not None:
            parent = next(i for i, r in enumerate(rules) if r.get("name") == rule["of"])
            inside = np.ones(len(pick), bool)
            if rule.get("face_box_m") is not None:
                x0, x1, y0, y1, z0, z1 = (float(v) for v in rule["face_box_m"])
                c = T.mean(1)
                inside = (c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1) & (c[:, 2] >= z0) & (c[:, 2] <= z1)
            if rule.get("face_area_min_m2") is not None:
                inside &= area >= float(rule["face_area_min_m2"])
            if rule.get("face_size_max_m") is not None:
                inside &= (T.max(1) - T.min(1)).max(1) <= float(rule["face_size_max_m"])
            pick[(pick == parent) & inside & (area <= float(rule.get("face_area_max_m2", np.inf)))] = k
    done = []
    for k, rule in enumerate(rules):
        faces = np.flatnonzero(pick == k)
        if not len(faces):
            continue
        name = re.sub(r"\W", "_", f"paint_{tag}_{rule.get('name', k)}")
        subset = UsdGeom.Subset.CreateGeomSubset(mesh, name, UsdGeom.Tokens.face, Vt.IntArray.FromNumpy(faces.astype(np.int32)), "materialBind", UsdGeom.Tokens.nonOverlapping)
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(paint_material(stage, name, rule))
        done.append(f"{rule.get('name', k)} {len(faces)}")
    print(f"[INFO] map_materials: {mesh_prim.GetPath()}: {len(cut)} solid(s), faces painted: {', '.join(done) or 'none'}", flush=True)
    return int((pick >= 0).sum())


def apply_decomposition(prim, options=None):
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
    prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
    for key, value in (options or {}).items():
        name, kind = DECOMPOSITION_ATTRS[key]
        prim.CreateAttribute(f"physxConvexDecompositionCollision:{name}", getattr(Sdf.ValueTypeNames, kind)).Set({"Int": int, "Float": float, "Bool": bool}[kind](value))


def set_collision_decomposition(stage, patterns, root=ROBOT_PATH):
    spec = dict(patterns) if isinstance(patterns, dict) else {q: None for q in patterns}
    rx = [(re.compile(str(q)), n) for q, n in spec.items()]
    roots = [(p, next(n for r, n in rx if r.fullmatch(p.GetName()))) for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if any(r.fullmatch(p.GetName()) for r, _ in rx)]
    make_editable([r for r, _ in roots])
    done = set()
    for r, spec_value in roots:
        options = {"max_hulls": spec_value} if isinstance(spec_value, (int, float)) and spec_value else dict(spec_value or {})
        sdf = options.pop("sdf", None)
        bad = [k for k in options if k not in DECOMPOSITION_ATTRS]
        if bad:
            sys.exit(f"collision_decomposition: unknown option {bad} (known: {', '.join(DECOMPOSITION_ATTRS)})")
        for p in Usd.PrimRange(r):
            if not (p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)) or p.GetPath() in done:
                continue
            if sdf:
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("sdf")
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(p).CreateSdfResolutionAttr(int(sdf))
            else:
                apply_decomposition(p, options)
            done.add(p.GetPath())
    return len(done)


def set_colliders(prim, on):
    n = 0
    for q in Usd.PrimRange(prim):
        if q.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(on)
            n += 1
    return n


def set_collision_ccd(stage, patterns, root):
    rx = [re.compile(str(q)) for q in patterns]
    done = set()
    for p in list(Usd.PrimRange(stage.GetPrimAtPath(root))):
        if not any(r.fullmatch(p.GetName()) for r in rx):
            continue
        body = p
        while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if body.IsValid() and body.GetPath() not in done and str(body.GetPath()) != "/":
            PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateEnableCCDAttr().Set(True)
            done.add(body.GetPath())
    return len(done)


def set_collision_friction(stage, patterns, root, name="machine", override=None, combine="min"):
    rx = [(re.compile(str(q)), float(f) if override is None else float(override), i) for i, (q, f) in enumerate(dict(patterns).items())]
    n, mats = 0, []
    for p in list(Usd.PrimRange(stage.GetPrimAtPath(root))):
        hit = next(((f, i) for r, f, i in rx if r.fullmatch(p.GetName())), None)
        if hit is None:
            continue
        f, i = hit
        mat_path = f"/World/PhysicsMaterials/{name}_friction_{i}"
        if mat_path not in mats:
            mats.append(mat_path)
        if not stage.GetPrimAtPath(mat_path).IsValid():
            UsdShade.Material.Define(stage, mat_path)
            m = stage.GetPrimAtPath(mat_path)
            pm = UsdPhysics.MaterialAPI.Apply(m)
            pm.CreateStaticFrictionAttr(f)
            pm.CreateDynamicFrictionAttr(f)
            pm.CreateRestitutionAttr(0.0)
            PhysxSchema.PhysxMaterialAPI.Apply(m).CreateFrictionCombineModeAttr(combine)
        mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
        for q in Usd.PrimRange(p):
            if q.IsA(UsdGeom.Mesh) and q.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(q).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
                n += 1
    return n, mats


def set_materials(stage, root, rules):
    count = 0
    for rule in rules:
        rx = [re.compile(str(q)) for q in rule["links"]]
        roots = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if any(r.fullmatch(p.GetName()) for r in rx)]
        make_editable(roots)
        mats = {UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0].GetPath() for r in roots for p in Usd.PrimRange(r)
                if p.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(p).ComputePurpose() == UsdGeom.Tokens.default_}
        mats.discard(Sdf.Path())
        for mp in mats:
            m = UsdShade.Material(stage.GetPrimAtPath(mp))
            if rule.get("color") is not None:
                m.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(float(c) for c in rule["color"])))
            for k in ("metallic", "roughness"):
                if rule.get(k) is not None:
                    m.CreateInput(k, Sdf.ValueTypeNames.Float).Set(float(rule[k]))
        count += len(mats)
    return count


class PicturePanel:

    def __init__(self):
        self.window = None
        self.image = None
        self.label = None

    def show(self, path):
        if self.window is None:
            self.window = ui.Window("Vision pick: score ranking", width=760, height=700)
            with self.window.frame:
                with ui.VStack(spacing=4):
                    self.label = ui.Label("", height=20)
                    self.image = ui.Image("", fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT)
        self.image.source_url = path
        self.label.text = os.path.basename(path)
        self.window.visible = True
        print(f"[INFO] picture window: {path}", flush=True)


class MachineBridge(Node):

    def __init__(self, name, names, q0, limits, cfg=None, context=None):
        super().__init__(f"{name}_bridge", context=context)
        self.names = list(names)
        self._idx = {n: i for i, n in enumerate(self.names)}
        self._limits = list(limits)
        self._targets = [float(x) for x in q0]
        self._lock = threading.Lock()
        self._warned = set()
        self.create_subscription(JointState, f"/{name}/joint_command", self._on_command, 10)
        self._pub = self.create_publisher(JointState, f"/{name}/joint_states", 10)
        cfg = cfg or {}
        self.hands_pub = self.create_publisher(Int32, str((cfg.get("hands") or {}).get("topic") or f"/{name}/hand_plates"), 10) if cfg.get("hands") else None
        self.part_pose_pub = self.create_publisher(Pose, str((cfg.get("part_pose") or {}).get("topic") or "/target_part_pose"), 10) if cfg.get("part_pose") else None
        self.sensors_pub = self.create_publisher(Int32, str((cfg.get("sensors") or {}).get("topic") or f"/{name}/sensors"), 10) if cfg.get("sensors") else None
        self.attach_requests = deque()
        trigger = (cfg.get("attach") or {}).get("trigger_topic") or cfg.get("trigger_topic")
        if cfg.get("attach") and trigger:
            self.create_subscription(Empty, str(trigger), lambda _: self.attach_requests.append(time.time()), 10)
        self.vacuum_req, self.vacuum_pub = deque(), None
        if cfg.get("vacuum"):
            vtopic = str(cfg["vacuum"].get("topic") or f"/{name}/vacuum")
            self.create_subscription(Int32, vtopic, lambda m: self.vacuum_req.append(int(m.data)), 10)
            self.vacuum_pub = self.create_publisher(Int32, vtopic + "/held", 10)

    def _on_command(self, msg):
        with self._lock:
            for n, p in zip(msg.name, msg.position):
                i = self._idx.get(n)
                if i is None:
                    if n not in self._warned:
                        self._warned.add(n)
                        self.get_logger().warn(f"no joint {n} (joints: {self.names})")
                    continue
                lo, hi = self._limits[i]
                self._targets[i] = min(max(float(p), lo), hi)

    def targets(self):
        with self._lock:
            return np.array(self._targets)

    def report(self, q, v):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position, msg.velocity = self.names, [float(x) for x in q], [float(x) for x in v]
        self._pub.publish(msg)

    def publish_part_pose(self, pos, quat_wxyz):
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = (float(v) for v in pos)
        msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = (float(v) for v in quat_wxyz)
        self.part_pose_pub.publish(msg)


class CarrierBridge(Node):

    def __init__(self, name, topic, joints, slots, window, speed, accel=None, context=None, reverse=False):
        super().__init__(f"{name}_carriers", context=context)
        self.sign = -1.0 if reverse else 1.0
        self.accel = float(accel) if accel else CARRIER_ACCEL
        self.joints, self.slots, self.lo, self.hi = list(joints), [float(y) for y in slots], float(window[0]), float(window[1])
        self.loop = self.hi - self.lo
        self.pitch = self.loop / len(self.slots)
        self.speed = float(speed)
        self.v = 0.0
        self.s = self.goal = 0.0
        self.moving = False
        self._lock = threading.Lock()
        self.create_subscription(Int32, topic, self._on_command, 10)
        self._pub = self.create_publisher(Bool, f"{topic}/moving", QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._pub.publish(Bool(data=False))

    def _on_command(self, msg):
        with self._lock:
            n = int(msg.data)
            if n > 0:
                self.goal = max(self.goal, self.s) + n * self.pitch
            elif n == 0:
                self.goal = self.s
        self.get_logger().info(f"carriers: {'+%d pitch' % n if n > 0 else 'stop' if n == 0 else 'negative step ignored'}")

    def positions(self):
        with self._lock:
            return [self.lo + (y - self.sign * self.s - self.lo) % self.loop - y for y in self.slots]

    def step(self, dt):
        with self._lock:
            if self.s < self.goal:
                v = min(self.speed, self.v + self.accel * dt, math.sqrt(2.0 * self.accel * (self.goal - self.s)))
                self.v = max(v, 0.01)
                self.s = min(self.goal, self.s + self.v * dt)
            else:
                self.v = 0.0
            moving = self.s < self.goal
        if moving != self.moving:
            self.moving = moving
            self._pub.publish(Bool(data=moving))
        return self.positions()


def export_layout(robots, path):
    stage = omni.usd.get_context().get_stage()
    crate_frame = re.compile(r"L\d+_C\d+")
    drop = [r["prim"] for r in robots] + ["/World/pile", "/World/objects"]
    for pallet in stage.GetPrimAtPath("/World/pallets").GetChildren() if stage.GetPrimAtPath("/World/pallets").IsValid() else []:
        drop += [str(c.GetPath()) for c in pallet.GetChildren() if not crate_frame.fullmatch(c.GetName())]
    mats = stage.GetPrimAtPath("/World/PhysicsMaterials")
    drop += [str(c.GetPath()) for c in (mats.GetChildren() if mats.IsValid() else []) if any(c.GetName().startswith(f"{r['name']}_robot_") for r in robots)]
    for d in drop:
        if stage.GetPrimAtPath(d).IsValid():
            stage.RemovePrim(d)
    stage.GetRootLayer().Export(path)
    print(f"[INFO] layout saved: {path} ({len(drop)} prims left out: robot, its materials, parts)", flush=True)


def build_stage(robots):
    s = robots[0]
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
    psa = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    psa.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(2 ** 21)
    psa.CreateGpuTotalAggregatePairsCapacityAttr().Set(2 ** 21)
    psa.CreateGpuMaxRigidPatchCountAttr().Set(2 ** 18)
    psa.CreateGpuMaxRigidContactCountAttr().Set(2 ** 21)
    psa.CreateGpuFoundLostPairsCapacityAttr().Set(2 ** 22)
    psa.CreateGpuHeapCapacityAttr().Set(2 ** 28)
    psa.CreateGpuTempBufferCapacityAttr().Set(2 ** 28)
    psa.CreateEnableCCDAttr().Set(True)
    psa.CreateGpuCollisionStackSizeAttr().Set(2 ** 29)
    scene.GetPrim().CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(int(s.get("physics_hz", 60)))
    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    floor.CreateDisplayColorAttr([Gf.Vec3f(*(float(c) for c in (s.get("floor_color") or [0.35, 0.35, 0.36])))])
    fx = UsdGeom.Xformable(floor)
    fx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    fx.AddScaleOp().Set(Gf.Vec3f(30.0, 30.0, 0.02))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(1000.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 0.0, 30.0))

    def place(path, usd, pos, rpy=None, R=None):
        prim = stage.DefinePrim(path, "Xform")
        if usd is not None:
            prim.GetReferences().AddReference(usd)
        x = UsdGeom.Xformable(prim)
        x.ClearXformOpOrder()
        x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
        q = mat_to_quat(rpy_to_mat(rpy) if R is None else R)
        x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))

    for r in robots:
        place(r["prim"], r["usd"], r.get("pos_m", [0, 0, 0]), r.get("rpy_deg", [0, 0, 0]))
    for m in s.get("machines") or []:
        place(f"/World/machines/{m['name']}", m["usd"], m.get("pos_m", [0, 0, 0]), m.get("rpy_deg", [0, 0, 0]))
    for i, o in enumerate(s["objects"]):
        place(f"/World/objects/{o['part']}_{i}", o["usd"], o.get("pos_m", [0, 0, 0]), o.get("rpy_deg", [0, 0, 0]))
    if s.get("bin"):
        place("/World/bin", s["bin"]["usd"], s["bin"]["pos_m"], s["bin"]["rpy_deg"])
        for i, (usd, pos, R) in enumerate(s["bin"]["pile"]):
            place(f"/World/pile/{s['bin']['part']['name']}_{i:03d}", usd, pos, R=R)
    if s.get("basket"):
        place("/World/basket", s["basket"]["usd"], s["basket"]["pos_m"], s["basket"].get("rpy_deg", [0, 0, 0]))
    if s.get("pallets"):
        pc, pl = s["pallets"], s["pallets"]["plan"]
        pname = (pc.get("part_cfg") or {}).get("name", "part")
        settling = float(pc.get("settle_s", 0.0)) > 0
        for c in pl["crates"]:
            frame = f"/World/pallets/pallet_{c['pallet']}/L{c['layer']}_C{c['idx']}"
            place(frame, None, c["pos"], R=c["R"])
            crate = stage.DefinePrim(frame + "/crate")
            crate.GetReferences().AddReference(pc["crate"]["usd"])
            crate.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if s.get("pallet_physics") and not c["dynamic"]:
                set_colliders(crate, False)
            approx = "off" if pc["crate"].get("collision") is False else str(pc["crate"].get("collision") or "none")
            if approx == "off":
                for q in list(Usd.PrimRange(crate)):
                    if q.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
                    if q.HasAPI(UsdPhysics.RigidBodyAPI):
                        q.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    if q.IsA(UsdPhysics.Joint):
                        q.SetActive(False)
            elif approx != "none":
                for q in Usd.PrimRange(crate):
                    if q.IsA(UsdGeom.Mesh) and q.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr(approx)
                        if approx == "sdf":
                            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(q).CreateSdfResolutionAttr(int(pc["crate"].get("sdf_resolution", 256)))
            if c.get("switch_off") and not settling:
                set_colliders(crate, False)
        for p in pl["parts"]:
            path = (f"/World/pile/{pname}_p{p['pallet']}_L{p['layer']}_C{p['crate']}_{p['j']:02d}" if p["dynamic"]
                    else f"/World/pallets/pallet_{p['pallet']}/L{p['layer']}_C{p['crate']}_{pname}_{p['j']:02d}")
            pos = list(p["pos"])
            if p["dynamic"] and str(pc["crate"].get("collision", "none")) == "convexHull":
                pos[2] += float(pc["crate"]["size_m"][2]) - float(pc["crate"]["floor_z_m"])
            place(path, pl["part_usd"], pos, R=p["R"])
            if p["dynamic"]:
                api_ = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path))
                for method_, value_ in PART_PHYSX.items():
                    getattr(api_, method_)().Set(value_)
                col_ = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(f"{path}/geometry"))
                col_.CreateContactOffsetAttr().Set(CONTACT_OFFSET_M)
                col_.CreateRestOffsetAttr().Set(REST_OFFSET_M)
            if p["dynamic"] and p["layer"] >= int(pc.get("layers", 6)) - int(pc.get("sdf_layers", 0)):
                gp = stage.GetPrimAtPath(f"{path}/geometry")
                UsdPhysics.MeshCollisionAPI.Apply(gp).CreateApproximationAttr("sdf")
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(gp).CreateSdfResolutionAttr(int(pc.get("sdf_resolution", 256)))
            if p.get("switch_off") and not settling:
                set_colliders(stage.GetPrimAtPath(path), False)
                UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).CreateKinematicEnabledAttr(True)
            if not p["dynamic"]:
                stage.GetPrimAtPath(path).RemoveAPI(UsdPhysics.RigidBodyAPI)
                if s.get("pallet_physics"):
                    set_colliders(stage.GetPrimAtPath(path), False)
    if s.get("map"):
        layer = Sdf.Layer.FindOrOpen(s["map"])
        UsdGeom.Xform.Define(stage, "/World/map")
        for spec in layer.rootPrims:
            if spec.name.startswith(("Render", "Omni")) or spec.typeName in ("Camera",) or spec.typeName.endswith("Light"):
                continue
            stage.DefinePrim(f"/World/map/{spec.name}").GetReferences().AddReference(s["map"], f"/{spec.name}")
        for m in s.get("machines") or []:
            if m.get("replaces"):
                old = stage.GetPrimAtPath("/World/map/" + str(m["replaces"]).strip("/"))
                if old.IsValid():
                    old.SetActive(False)
                else:
                    print(f"[WARN] machine {m['name']}: {m['replaces']} not found in the map: the static copy stays", flush=True)
    for mc_ in s.get("map_collision") or []:
        root = stage.GetPrimAtPath("/World/map/" + str(mc_["prim"]).strip("/"))
        if not root.IsValid():
            print(f"[WARN] map_collision: {mc_['prim']} not found in the map", flush=True)
            continue
        n = 0
        for q in Usd.PrimRange(root):
            if q.IsA(UsdGeom.Mesh) and q.HasAPI(UsdPhysics.CollisionAPI):
                mode = str(mc_.get("collision", "convexHull"))
                if mode == "off":
                    UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
                elif mode == "convexDecomposition":
                    apply_decomposition(q, mc_.get("options"))
                else:
                    UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr(mode)
                n += 1
        print(f"[INFO] map_collision: {mc_['prim']}: {n} collider(s) -> {mc_.get('collision', 'convexHull')}", flush=True)
    for ed in s.get("map_edits") or []:
        root = stage.GetPrimAtPath("/World/map/" + str(ed["prim"]).strip("/"))
        if not root.IsValid():
            print(f"[WARN] map_edits: {ed['prim']} not found in the map", flush=True)
            continue
        n = 0
        for q in Usd.PrimRange(root):
            if not q.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(q)
            P = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
            M = np.array(UsdGeom.Xformable(q).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).T
            W = P @ M[:3, :3].T + M[:3, 3]
            if ed.get("scale_z") is not None:
                W[:, 2] *= float(ed["scale_z"])
            if ed.get("dz_m") is not None:
                W[W[:, 2] > float(ed["above_z_m"]), 2] += float(ed["dz_m"])
            P2 = (W - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P2.astype(np.float32)))
            mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in P2.min(0)]), Gf.Vec3f(*[float(v) for v in P2.max(0)])]))
            n += 1
        print(f"[INFO] map_edits: {ed['prim']}: {n} mesh(es)" + (f" scaled x{ed['scale_z']} in z" if ed.get("scale_z") is not None else "")
              + (f" stretched by {ed['dz_m']} m above z {ed['above_z_m']} m" if ed.get("dz_m") is not None else ""), flush=True)
    for mm in s.get("map_materials") or []:
        where = str(mm["prim"])
        root = stage.GetPrimAtPath(where if where.startswith("/") else "/World/map/" + where.strip("/"))
        if not root.IsValid():
            print(f"[WARN] map_materials: {mm['prim']} not found in the stage", flush=True)
            continue
        make_editable([root])
        for q in [q for q in Usd.PrimRange(root) if q.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(q).ComputePurpose() == UsdGeom.Tokens.default_]:
            paint_solids(stage, q, mm.get("rules") or [mm], root.GetName())
    for k, pr in enumerate((s.get("props") or []) + fence_props(s.get("fence"))):
        name = re.sub(r"\W", "_", str(pr.get("name", f"prop_{k}")))
        path = f"/World/props/{name}"
        UsdGeom.Xform.Define(stage, "/World/props")
        sx, sy, sz = (float(v) for v in pr["size_m"])
        if str(pr.get("shape", "box")) == "cylinder":
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateRadiusAttr(sx / 2.0)
            geom.CreateHeightAttr(sz)
            scale = None
        else:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            scale = Gf.Vec3f(sx, sy, sz)
        fx = UsdGeom.Xformable(geom)
        px, py, pz = (float(v) for v in pr["pos_m"])
        fx.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
        q = mat_to_quat(rpy_to_mat(pr.get("rpy_deg", [0, 0, 0])))
        fx.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
        if scale is not None:
            fx.AddScaleOp().Set(scale)
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(paint_material(stage, f"prop_{pr.get('look', name)}", pr))
    if s.get("platform"):
        pf = s["platform"]
        sx, sy, sz = (float(v) for v in pf["size_m"])
        px, py, pz = (float(v) for v in pf.get("pos_m", [0, 0, 0]))
        cube = UsdGeom.Cube.Define(stage, "/World/platform")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*map(float, pf.get("color", [0.55, 0.55, 0.58])))])
        fx = UsdGeom.Xformable(cube)
        fx.AddTranslateOp().Set(Gf.Vec3d(px, py, pz + sz / 2))
        q = mat_to_quat(rpy_to_mat(pf.get("rpy_deg", [0, 0, 0])))
        fx.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
        fx.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    update(5)
    for r in robots:
        if r["gripper_links"] and r.get("gripper_material"):
            n = set_materials(stage, r["prim"], [{"links": [re.escape(ln) for ln in r["gripper_links"]],
                                                   **{k: r["gripper_material"].get(k) for k in ("metallic", "roughness")}}])
            print(f"[INFO] gripper material {dict(r['gripper_material'])} on {n} materials", flush=True)
        if r.get("materials"):
            n = set_materials(stage, r["prim"], r["materials"])
            print(f"[INFO] materials of the robot {r['prim']}: {n} set (metal look)", flush=True)
    for m in s.get("machines") or []:
        if m.get("materials"):
            n = set_materials(stage, f"/World/machines/{m['name']}", m["materials"])
            print(f"[INFO] machine {m['name']}: {n} materials set (metal look)", flush=True)
    for r in robots:
        if r.get("collision_decomposition"):
            n = set_collision_decomposition(stage, r["collision_decomposition"], r["prim"])
            print(f"[INFO] {r['prim']}: collision_decomposition {dict(r['collision_decomposition']) if isinstance(r['collision_decomposition'], dict) else list(r['collision_decomposition'])}: "
                  f"{n} colliders -> convexDecomposition", flush=True)
    for r in robots:
        if r.get("collision_friction"):
            n, _ = set_collision_friction(stage, r["collision_friction"], r["prim"], f"{r['name']}_robot", combine="max")
            print(f"[INFO] {r['prim']}: collision_friction {dict(r['collision_friction'])} (combine max): {n} colliders", flush=True)
    for r in robots:
        if r.get("collision_ccd"):
            n = set_collision_ccd(stage, r["collision_ccd"], r["prim"])
            print(f"[INFO] {r['prim']}: collision_ccd {list(r['collision_ccd'])}: {n} bodies", flush=True)
    for m in s.get("machines") or []:
        if m.get("collision_decomposition"):
            n = set_collision_decomposition(stage, m["collision_decomposition"], f"/World/machines/{m['name']}")
            print(f"[INFO] machine {m['name']}: collision_decomposition {list(m['collision_decomposition'])}: {n} colliders -> convexDecomposition", flush=True)
        att = m.get("attach") or {}
        if att.get("enabled", True) and att.get("jig_links") and att.get("jig_collision") is False:
            rx = [re.compile(str(q)) for q in att["jig_links"]]
            n = 0
            for p in Usd.PrimRange(stage.GetPrimAtPath(f"/World/machines/{m['name']}")):
                if any(r.fullmatch(p.GetName()) for r in rx):
                    n += set_colliders(p, False)
            print(f"[INFO] machine {m['name']}: jigs {list(att['jig_links'])} without collider ({n}): the parts put on them are removed", flush=True)
        if m.get("collision") is False:
            n = set_colliders(stage.GetPrimAtPath(f"/World/machines/{m['name']}"), False)
            print(f"[INFO] machine {m['name']}: collision false: {n} colliders off (only to look at)", flush=True)
        for p_ in Usd.PrimRange(stage.GetPrimAtPath(f"/World/machines/{m['name']}")):
            if p_.GetName().startswith("seat_"):
                for q_ in Usd.PrimRange(p_):
                    if q_.IsA(UsdGeom.Mesh) and q_.HasAPI(UsdPhysics.CollisionAPI):
                        col_ = PhysxSchema.PhysxCollisionAPI.Apply(q_)
                        col_.CreateContactOffsetAttr().Set(CONTACT_OFFSET_M)
                        col_.CreateRestOffsetAttr().Set(REST_OFFSET_M)
        if m.get("collision_friction_max"):
            n, _ = set_collision_friction(stage, m["collision_friction_max"], f"/World/machines/{m['name']}", f"{m['name']}_max", combine="max")
            print(f"[INFO] machine {m['name']}: collision_friction_max {dict(m['collision_friction_max'])} (combine max): {n} colliders", flush=True)
        if m.get("collision_friction"):
            n, mats = set_collision_friction(stage, m["collision_friction"], f"/World/machines/{m['name']}", m["name"], m.get("friction_override"))
            s.setdefault("friction_mats", {})[m["name"]] = mats
            print(f"[INFO] machine {m['name']}: collision_friction {dict(m['collision_friction'])}"
                  f"{' -> ' + str(m['friction_override']) + ' (fixtures.friction of the machine yaml)' if m.get('friction_override') is not None else ''}: {n} colliders", flush=True)
    for r in robots:
        if r.get("wrist_camera"):
            r["wrist_camera_path"] = make_wrist_camera(stage, r["wrist_camera"], r["prim"])
            if r["wrist_camera_path"]:
                print(f"[INFO] wrist camera {r['wrist_camera_path']}: publish std_msgs/Empty on /wrist_camera/capture for one frame -> "
                      f"{r['wrist_camera']['output_dir']}", flush=True)
    for c in s.get("cameras") or []:
        c["path"] = make_fixed_camera(stage, c)
        print(f"[INFO] camera {c['name']} at {c['pos_m']} looking at {c['look_at_m']}: publish std_msgs/Empty on /{c['name']}/capture for one picture", flush=True)

    def drives(prefix, joints, q_start, stiffness, damping):
        for jp in stage.Traverse():
            info = joints.get(jp.GetName())
            if info is None or not jp.IsA(UsdPhysics.Joint) or not str(jp.GetPath()).startswith(prefix + "/"):
                continue
            kind = "linear" if info[0] == "prismatic" else "angular"
            start = q_start[jp.GetName()] if kind == "linear" else math.degrees(q_start[jp.GetName()])
            drv = UsdPhysics.DriveAPI.Apply(jp, kind)
            drv.CreateStiffnessAttr().Set(float(stiffness[kind]))
            drv.CreateDampingAttr().Set(float(damping[kind]))
            drv.CreateTargetPositionAttr().Set(start)
            if info[4] > 0:
                drv.CreateMaxForceAttr().Set(float(info[4]))
            PhysxSchema.JointStateAPI.Apply(jp, kind).CreatePositionAttr().Set(start)

    for r in robots:
        k, d = float(r.get("stiffness", 20000.0)), float(r.get("damping", 1000.0))
        drives(r["prim"], r["joints"], r["q0"], {"linear": k, "angular": k}, {"linear": d, "angular": d})
    for m in s.get("machines") or []:
        prefix = f"/World/machines/{m['name']}"
        drives(prefix, m["joints"], m["q0"], MACHINE_STIFFNESS, {**MACHINE_DAMPING, **({"angular": float(m["rotor_damping"])} if m.get("rotor_damping") else {})})
        for p in Usd.PrimRange(stage.GetPrimAtPath(prefix)):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                art_api = PhysxSchema.PhysxArticulationAPI.Apply(p)
                art_api.CreateEnabledSelfCollisionsAttr().Set(False)
                if m.get("solver_iterations"):
                    art_api.CreateSolverPositionIterationCountAttr(int(m["solver_iterations"].get("position", 32)))
                    art_api.CreateSolverVelocityIterationCountAttr(int(m["solver_iterations"].get("velocity", 8)))


def main():
    globals().update(robot.get("sim_consts") or {})
    contexts = {}

    def ros(id):
        if id not in contexts:
            ctx = rclpy.Context()
            rclpy.init(args=[], context=ctx, **({} if id is None else {"domain_id": int(id)}))
            contexts[id] = (ctx, MultiThreadedExecutor(context=ctx))
        return contexts[id]
    units = []
    try:
        for r in robots:
            r["usd"] = robot_usd(r, args.force_convert)
            print(f"[INFO] {r['urdf']} -> {r['usd']}", flush=True)
        if robot["objects"]:
            from binsim.assets import prepare_part
            ready = []
            for o in robot["objects"]:
                try:
                    o["usd"] = prepare_part(o["cfg"], os.path.join(REPO, "assets_cache")).usd_path
                except Exception as e:
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
            except Exception as e:
                print(f"[WARN] bin and pile skipped: {type(e).__name__}: {e}", flush=True)
                robot["bin"] = None
        if robot.get("pallets"):
            pc = robot["pallets"]
            try:
                pc["plan"] = plan_pallet_crates(pc, os.path.join(REPO, "assets_cache"))
                pl = pc["plan"]
                phys = robot.get("pallet_physics") or {}
                if phys:
                    def has_physics(pallet, layer, crate, default):
                        ph = phys.get((pallet, layer))
                        return default if ph is None else (ph[0] and (ph[1] is None or crate in ph[1] or ph[2]))

                    def switched_off(pallet, layer, crate):
                        ph = phys.get((pallet, layer))
                        return bool(ph and ph[0] and ph[2] and ph[1] is not None and crate not in ph[1])
                    for c in pl["crates"]:
                        c["dynamic"] = has_physics(c["pallet"], c["layer"], c["idx"], c["dynamic"])
                        c["switch_off"] = switched_off(c["pallet"], c["layer"], c["idx"])
                    for p in pl["parts"]:
                        p["dynamic"] = has_physics(p["pallet"], p["layer"], p["crate"], p["dynamic"])
                        p["switch_off"] = switched_off(p["pallet"], p["layer"], p["crate"])
                n_dyn = sum(p["dynamic"] for p in pl["parts"])
                print(f"[INFO] pallets: {sum(bool(i.get('enabled', True)) for i in pc['items'])} of {len(pc['items'])} on x {int(pc.get('layers', 6))} layers, {len(pl['crates'])} crates, {len(pl['parts'])} x "
                      f"{(pc.get('part_cfg') or {}).get('name', '-')} ({n_dyn} movable, seed {pc['seed']}); grids per layer (nx x ny): "
                      + " | ".join(",".join(f"{a}x{b}" for a, b in g) for g in pl["grids"]), flush=True)
            except Exception as e:
                print(f"[WARN] pallets skipped: {type(e).__name__}: {e}", flush=True)
                robot["pallets"] = None
        if robot.get("map"):
            print(f"[INFO] map {os.path.relpath(robot['map'], ASSETS)} -> /World/map (a big USD takes a while to load)", flush=True)
        if robot.get("platform"):
            print(f"[INFO] platform {robot['platform']['size_m']} m at {robot['platform'].get('pos_m', [0, 0, 0])} m (bottom centre) -> /World/platform"
                  f", top at z = {float(robot['platform'].get('pos_m', [0, 0, 0])[2]) + float(robot['platform']['size_m'][2]):.3f} m", flush=True)
        if robot.get("basket"):
            from binsim.assets import prepare_bin
            k = robot["basket"]
            try:
                k["usd"] = prepare_bin(k, os.path.join(REPO, "assets_cache")).usd_path
                print(f"[INFO] basket {k['size_m']} m at {k['pos_m']} m (world), floor top at z = "
                      f"{k['pos_m'][2] + k['wall_thickness_m']:.3f} m -> /World/basket", flush=True)
            except Exception as e:
                print(f"[WARN] basket skipped: {type(e).__name__}: {e}", flush=True)
                robot["basket"] = None
        for r in robots:
            jp = r.get("joint_pos") or {}
            r["q0"] = {n: min(max(next((float(v) for k, v in jp.items() if re.fullmatch(k, n)), 0.0), lo), hi)
                       for n, (_, lo, hi, _, _) in r["joints"].items()}
            for n, (lead, k, off) in r["mimic"].items():
                if lead in r["q0"]:
                    r["q0"][n] = r["q0"][lead] * k + off
        ready = []
        for m in robot["machines"]:
            try:
                m["usd"] = kinematic_usd(m) if m.get("kinematic") else robot_usd(m, args.force_convert)
            except Exception as e:
                print(f"[WARN] machine {m['name']} skipped: {type(e).__name__}: {e}", flush=True)
                continue
            jp0 = m.get("joint_pos") or {}
            m["q0"] = {n: min(max(float(jp0.get(n, 0.0)), lo), hi) for n, (_, lo, hi, _, _) in m["joints"].items()}
            ready.append(m)
            print(f"[INFO] machine {m['name']} at {m.get('pos_m', [0, 0, 0])} m, rpy {m.get('rpy_deg', [0, 0, 0])} deg -> {m['usd']}", flush=True)
        robot["machines"] = ready
        build_stage(robots)
        if args.export_layout:
            export_layout(robots, os.path.abspath(args.export_layout))
            return

        for r in robots:
            root_path = Articulation.fetch_articulation_root_api_prim_paths(r["prim"])[0]
            if root_path is None:
                print(f"[ERROR] no articulation root under {r['prim']} in {r['usd']}", flush=True)
                return
            units.append({"r": r, "root": root_path, "art": Articulation(root_path)})
        machines = []
        for m in robot["machines"]:
            mpath = f"/World/machines/{m['name']}"
            if m.get("kinematic"):
                machines.append({"m": m, "root": mpath, "art": KinArticulation(m, mpath)})
                continue
            mroot = Articulation.fetch_articulation_root_api_prim_paths(mpath)[0]
            if mroot is None:
                print(f"[WARN] machine {m['name']}: no articulation root under {mpath}: its joints cannot be moved", flush=True)
                continue
            machines.append({"m": m, "root": mroot, "art": Articulation(mroot)})
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        update(2)

        for u in units:
            r = u["r"]
            u["names"] = list(u["art"].dof_names)
            u["cmd"] = np.array([r["q0"].get(n, 0.0) for n in u["names"]])
            u["cmd0"] = u["cmd"].copy()
            u["step"] = np.array([r["joints"].get(n, (None, 0, 0, 1.0))[3] * float(r.get("gripper_speed", r.get("speed", 0.3)) if n in r["gripper_joints"] or n in r["mimic"] else r.get("speed", 0.3))
                                  for n in u["names"]]) / APP_HZ
            u["step_slow"] = u["step"] * np.array([float(r["slow_speed"]) / float(r.get("speed", 0.3)) if r.get("slow_speed") and not (n in r["gripper_joints"] or n in r["mimic"]) else 1.0
                                                   for n in u["names"]])
            controller = "/" + str(r.get("controller", f"{r['name']}_controller")).strip("/")
            gripper = {n: r["joints"][n][1:3] for n in r["gripper_joints"]}
            gripper_topic = str(r.get("gripper_topic") or "/gripper/command")
            limits = [r["joints"][n][1:3] if n in r["joints"] else (-math.inf, math.inf) for n in u["names"]]
            ctx, ex = ros(r.get("id"))
            u["node"] = RobotBridge(u["names"], u["cmd"], controller, r["mimic"], gripper, gripper_topic, limits, context=ctx)
            ex.add_node(u["node"])
            print(f"[INFO] robot {r['name']} at {r['prim']} (ROS domain {r['id'] if r.get('id') is not None else os.environ.get('ROS_DOMAIN_ID', '0')}) ready "
                  f"(gripper: {r['gripper']}): joints {u['names']}, max speed {r.get('speed', 0.3)} x URDF limits" + (f", {r['slow_speed']} x while /motion/slow (the final descents)" if r.get("slow_speed") else ""), flush=True)
            print(f"[INFO]   ROS 2: {controller}/joint_trajectory, {controller}/follow_joint_trajectory -> sim, sim -> /joint_states"
                  + (f"; gripper {r['gripper']}: {gripper_topic} (std_msgs/Float64) 0 = closed, 1 = open -> {list(gripper)}" if gripper else ""), flush=True)
        node = units[0]["node"]
        mctx, executor = ros(robot.get("id"))
        for mc in machines:
            m = mc["m"]
            mc["names"] = list(mc["art"].dof_names)
            mc["cmd"] = np.array([m["q0"].get(n, 0.0) for n in mc["names"]])
            mc["cmd0"] = mc["cmd"].copy()
            speeds = [(str(rx), float(f)) for rx, f in (m.get("joint_speeds") or {}).items()]
            mc["step"] = np.array([m["joints"][n][3] * next((f for rx, f in speeds if re.fullmatch(rx, n)), float(m.get("speed", 1.0)))
                                   for n in mc["names"]]) / APP_HZ
            mc["vel"] = np.zeros(len(mc["names"]))
            mc["rot"] = np.array([m["joints"][n][0] != "prismatic" for n in mc["names"]])
            acc_rot, acc_lin = float(m.get("rotor_accel_rad_s2") or MACHINE_ACCEL["angular"]), float(m.get("jaw_accel_m_s2") or MACHINE_ACCEL["linear"])
            mc["acc"] = np.where(mc["rot"], acc_rot, acc_lin)
            mc["mname"] = m["name"]
            damping = float(m.get("rotor_damping") or MACHINE_DAMPING["angular"])
            effort = max((info[4] for info in m["joints"].values() if info[0] != "prismatic"), default=0.0)
            print(f"[INFO] machine {m['name']}: speed {float(m.get('speed', 1.0)):g} x URDF limit, joint_speeds {dict(speeds)}; acceleration rotors {acc_rot:g} rad/s2, "
                  f"jaws {acc_lin:g} m/s2; rotor damping {damping:g}" + (f" (speed limit by the drive force {effort / damping:.0f} deg/s)" if effort else ""), flush=True)
            mc["q"] = mc["cmd"].copy()
            mc["carrier"] = np.zeros(len(mc["names"]), bool)
            mc["node"] = MachineBridge(m["name"], mc["names"], mc["cmd"], [m["joints"][n][1:3] for n in mc["names"]], m, context=mctx)
            executor.add_node(mc["node"])
            mc["last_target"] = mc["node"].targets()
            c = m.get("carriers")
            mc["carriers"] = None
            if c:
                bad = [n for n in c["joints"] if n not in mc["names"]]
                if bad:
                    print(f"[WARN] machine {m['name']}: carriers of its yaml name joints that do not exist ({bad}): the carriers stay where they are", flush=True)
                else:
                    mc["carriers"] = {"node": CarrierBridge(m["name"], c.get("topic") or f"/{m['name']}/carriers", c["joints"], c["slots_m"], c["window_m"], c.get("speed_mps", 0.1), c.get("accel_mps2"), context=mctx, reverse=bool(c.get("reverse"))),
                                      "idx": [mc["names"].index(n) for n in c["joints"]]}
                    mc["carriers"]["last"] = mc["carriers"]["node"].positions()
                    mc["carrier"][mc["carriers"]["idx"]] = True
                    executor.add_node(mc["carriers"]["node"])
                    cn = mc["carriers"]["node"]
                    print(f"[INFO] machine {m['name']}: carriers {c['joints']}: std_msgs/Int32 on {c.get('topic') or '/' + m['name'] + '/carriers'} (n > 0 = n pitches of {cn.pitch:.2f} m, "
                          f"0 = stop), {cn.speed:g} m/s, state on .../moving", flush=True)
            print(f"[INFO] machine {m['name']}: /{m['name']}/joint_command (sensor_msgs/JointState: name, position in rad / m) -> {mc['names']}, "
                  f"/{m['name']}/joint_states <- sim", flush=True)
        for _, ex in contexts.values():
            threading.Thread(target=ex.spin, daemon=True).start()
        print(f"[INFO] physics {int(robot.get('physics_hz', 60))} Hz", flush=True)

        clicked, views = [], {}
        picture_panel = PicturePanel()
        part_view = [None, None]


        def rebuild_views(why, while_stopped=None):
            print(f"[WARN] a prim was deleted: the physics views are invalid ({why}); restarting the physics from the start state", flush=True)
            views.clear()
            part_view[0] = None
            timeline.stop()
            update(2)
            if while_stopped is not None:
                while_stopped()
            timeline.play()
            update(3)
            for u in units:
                u["art"] = Articulation(u["root"])
                u["cmd"] = u["cmd0"].copy()
                u["node"]._set(u["names"], u["cmd"])
            for mc in machines:
                if not isinstance(mc["art"], KinArticulation):
                    mc["art"] = Articulation(mc["root"])
                mc["cmd"] = mc["cmd0"].copy()
                if mc["carriers"]:
                    mc["carriers"]["node"].s = mc["carriers"]["node"].goal = 0.0
                    mc["carriers"]["last"] = mc["carriers"]["node"].positions()
            omni.usd.get_context().get_selection().clear_selected_prim_paths()

        def on_stage_event(event):
            if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
                paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
                clicked[:] = sorted({part_root(x) for x in paths} - {None})
                if paths and not clicked:
                    print(f"[INFO] {paths[0]} is not a part that can be picked (only the plates of the top layer / the objects move): no pose printed", flush=True)
        stage_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(on_stage_event, name="robot_sim click")
        refined = {}
        refined_at = {}

        def pile_positions():
            view, paths = pile_view()
            return dict(zip(paths, np.array(view.get_world_poses()[0].numpy()))) if view is not None else {}

        def remember_at(path):
            at = pile_positions().get(path)
            if at is not None:
                refined_at[path] = at

        def carried():
            pos = pile_positions()
            return {p for p, at in refined_at.items() if p in pos and np.linalg.norm(pos[p] - at) > CARRIED_M}

        def set_click_physx(prim, stiff):
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            kinematic = bool(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
            for method, (on, off) in CLICK_PHYSX.items():
                if not (stiff and kinematic and method == "CreateEnableCCDAttr"):
                    getattr(api, method)().Set(on if stiff else off)

        stiff_order = []
        plate_ev = {}

        def note(path, text):
            plate_ev.setdefault(path, []).append((time.time(), text))
            del plate_ev[path][:-6]

        def make_stiff(path):
            st_ = omni.usd.get_context().get_stage()
            set_click_physx(st_.GetPrimAtPath(path), True)
            note(path, "stiff contact settings ON (solver 16 / CCD)")
            if path in stiff_order:
                stiff_order.remove(path)
            stiff_order.append(path)
            busy = carried()
            idle = [p for p in stiff_order if p not in busy]
            for old_path in idle[:max(0, len(idle) - STIFF_KEEP)]:
                stiff_order.remove(old_path)
                old = st_.GetPrimAtPath(old_path)
                if old.IsValid():
                    set_click_physx(old, False)
                    note(old_path, "stiff contact settings off")

        def refine_collision(path):
            stage_now = omni.usd.get_context().get_stage()
            busy = carried()
            idle = [(old, nv) for old, nv in refined.items() if old not in busy]
            for old, (n, voxels) in idle[:max(0, len(idle) - CLICK_KEEP + 1)]:
                if old != path and n is None:
                    set_click_physx(stage_now.GetPrimAtPath(old), False)
                    refined.pop(old)
                    refined_at.pop(old, None)
                elif old != path and stage_now.GetPrimAtPath(old).IsValid():
                    geom = stage_now.GetPrimAtPath(f"{old}/geometry")
                    note(old, f"collider back to {n} hulls (cooked again)")
                    geom.GetAttribute(HULLS_ATTR).Set(n)
                    geom.CreateAttribute(VOXELS_ATTR, Sdf.ValueTypeNames.Int).Set(voxels)
                    set_click_physx(stage_now.GetPrimAtPath(old), False)
                    print(f"[INFO] {old}: back to {n} convex hulls, {voxels} voxels", flush=True)
                    refined.pop(old)
                    refined_at.pop(old, None)
            if str(UsdPhysics.MeshCollisionAPI(stage_now.GetPrimAtPath(f"{path}/geometry")).GetApproximationAttr().Get()) == "sdf" and path not in refined:
                refined[path] = (None, None)
                remember_at(path)
                make_stiff(path)
                print(f"[INFO] {path}: exact shape (SDF), stiff contact settings", flush=True)
                return
            attr = stage_now.GetPrimAtPath(f"{path}/geometry").GetAttribute(HULLS_ATTR)
            if not attr.IsValid() or path in refined:
                return
            geom = stage_now.GetPrimAtPath(f"{path}/geometry")
            voxels = geom.GetAttribute(VOXELS_ATTR)
            refined[path] = (int(attr.Get()), int(voxels.Get()) if voxels.IsValid() and voxels.Get() else VOXELS_DEFAULT)
            remember_at(path)
            note(path, f"collider {refined[path][0]} -> {CLICK_HULLS} hulls (cooked again)")
            attr.Set(CLICK_HULLS)
            geom.CreateAttribute(VOXELS_ATTR, Sdf.ValueTypeNames.Int).Set(CLICK_VOXELS)
            make_stiff(path)
            print(f"[INFO] {path}: {CLICK_HULLS} convex hulls instead of {refined[path][0]}, {CLICK_VOXELS} voxels instead of {refined[path][1]} (the collider is cooked again, this can take a moment)", flush=True)

        def refine_target(pos):
            view, paths = pile_view()
            if view is None or not paths:
                return
            centres = view.get_world_poses()[0].numpy()
            dist = np.linalg.norm(centres - np.array(pos), axis=1)
            i = int(np.argmin(dist))
            if dist[i] > TARGET_RADIUS:
                print(f"[INFO] target {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}: no part of the pile within {TARGET_RADIUS * 100:.0f} cm (nearest {dist[i] * 100:.1f} cm): nothing refined", flush=True)
                return
            print(f"[INFO] target {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}: the part is {paths[i]} ({dist[i] * 100:.1f} cm away)", flush=True)
            own = next((sw for sw in switch if paths[i] in sw["parts"]), None)
            if own is not None and not own["on"]:
                print(f"[WARN] {paths[i]} lies in {own['name']}, which has no physics: switched on now", flush=True)
                for sw in switch:
                    if sw is not own and sw["on"]:
                        switch_crate(sw, False)
                switch_crate(own, True)
            refine_collision(paths[i])

        print("[INFO] click a part in the viewport: its pose is printed as x y z roll pitch yaw (m, deg, like the Property panel), (= the target of a sequence with `target: object`), and below it the TCP pose that grips it "
              "(compute_grasp_tcp_pose.py) and the state of the part (standard | turn_over) for ik_node / a sequence with `target: tcp`, to paste into ee_6d.py", flush=True)

        fx_state = {"t": 0.0, "mtime": {}, "fx": {mc["mname"]: dict(mc["m"].get("fixtures") or {}) for mc in machines}, "last": {}}
        for mc in machines:
            f = mc["m"].get("config_file")
            fx_state["mtime"][mc["mname"]] = os.path.getmtime(f) if f and os.path.isfile(f) else 0.0

        def fixture_friction():
            st = omni.usd.get_context().get_stage()
            for mc in machines:
                car, name = mc.get("carriers"), mc.get("mname")
                if not car or name not in (robot.get("friction_mats") or {}):
                    continue
                moving = bool(car["node"].moving)
                if fx_state["last"].get(name) == moving:
                    continue
                fx_state["last"][name] = moving
                cfg = fx_state["fx"].get(name) or {}
                v = cfg.get("friction_moving", cfg.get("friction")) if moving else cfg.get("friction")
                if v is None:
                    continue
                for mp in robot["friction_mats"][name]:
                    api = UsdPhysics.MaterialAPI(st.GetPrimAtPath(mp))
                    api.GetStaticFrictionAttr().Set(float(v))
                    api.GetDynamicFrictionAttr().Set(float(v))
                print(f"[INFO] friction of the jigs of {name}: {float(v):g} ({'moving' if moving else 'at rest'})", flush=True)

        def reload_fixtures():
            for mc in machines:
                path, name = mc["m"].get("config_file"), mc["mname"]
                if not path or not os.path.isfile(path) or os.path.getmtime(path) == fx_state["mtime"].get(name):
                    continue
                fx_state["mtime"][name] = os.path.getmtime(path)
                try:
                    fx_state["fx"][name] = dict(sim_config.machine_config(mc["m"]["entry"]).get("fixtures") or {})
                    fx_state["last"].pop(name, None)
                    print(f"[INFO] fixtures of {os.path.basename(path)} read again (no restart): {json.dumps(fx_state['fx'][name])}", flush=True)
                except Exception as e:
                    print(f"[WARN] fixtures of {path} not read: {type(e).__name__}: {e}", flush=True)

        rebuilds = 0
        attaches = [{"mc": mc, "cfg": mc["m"]["attach"], "B": None, "bound": {}}
                    for mc in machines if mc["m"].get("attach") and mc["m"]["attach"].get("enabled", True) and mc.get("carriers")]
        PARK = (0.0, 0.0, -50.0)
        removed, layer_gone = set(), set()

        def bound_parts():
            return {path for att in attaches for path in att["bound"]} | removed | layer_gone

        def retire(st_, path, why):
            part = st_.GetPrimAtPath(path)
            set_colliders(part, False)
            set_click_physx(part, False)
            PhysxSchema.PhysxRigidBodyAPI.Apply(part).CreateDisableGravityAttr(True)
            UsdGeom.Imageable(part).MakeInvisible()
            note(path, why)
            refined.pop(path, None)
            refined_at.pop(path, None)
            if path in stiff_order:
                stiff_order.remove(path)

        def park(view, idx):
            view.set_world_poses(np.tile(np.array(PARK, dtype=np.float32), (len(idx), 1)), np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(idx), 1)), indices=idx)
            view.set_velocities(np.zeros((len(idx), 3), dtype=np.float32), np.zeros((len(idx), 3), dtype=np.float32), indices=idx)

        def jig_matrix(att, joint):
            mc = att["mc"]
            if att.get("B") is None:
                root = omni.usd.get_context().get_stage().GetPrimAtPath(mc["root"])
                base = next(q_ for q_ in Usd.PrimRange(root) if q_.GetName() == str(att["cfg"].get("base_link", "base_link")))
                att["B"] = Gf.Matrix4d(UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            d = float(mc["q"][mc["names"].index(joint)])
            ax = [float(v) for v in (mc["m"].get("carriers") or {}).get("axis", [1, 0, 0])]
            T = Gf.Matrix4d(1.0)
            T.SetTranslate(Gf.Vec3d(d * ax[0], d * ax[1], d * ax[2]))
            return T * att["B"]

        def pile_view():
            st_ = omni.usd.get_context().get_stage()
            pile = st_.GetPrimAtPath("/World/pile")
            paths = [str(c.GetPath()) for c in pile.GetChildren()] if pile.IsValid() else []
            if not paths:
                return None, []
            if part_view[0] is None or part_view[1] != paths:
                part_view[0], part_view[1] = RigidPrim(paths), paths
            return part_view[0], part_view[1]

        def attach_parts(att):
            st_ = omni.usd.get_context().get_stage()
            view, paths = pile_view()
            if view is None:
                return
            pos_, quat_ = view.get_world_poses()
            pn, qn = pos_.numpy(), quat_.numpy()
            jig_matrix(att, att["mc"]["carriers"]["node"].joints[0])
            Binv, name = att["B"].GetInverse(), att["mc"]["mname"]
            (x0, x1), (y0, y1), (z0, z1) = ([float(v) for v in r] for r in att["cfg"]["remove_box_m"])
            gone = []
            for i, p_ in enumerate(pn):
                path = paths[i]
                if path in att["bound"]:
                    continue
                x, y, z = Binv.Transform(Gf.Vec3d(float(p_[0]), float(p_[1]), float(p_[2])))
                if not (x0 < x < x1 and y0 < y < y1 and z0 < z < z1):
                    continue
                att["bound"][path] = name
                rides.pop(path, None)
                retire(st_, path, f"removed on {name}: hidden, collider off, parked")
                gone.append(i)
            if not gone:
                print(f"[INFO] {name}: no part above it (attach.remove_box_m)", flush=True)
            if gone:
                park(view, gone)
                print(f"[INFO] {len(gone)} part(s) on the jigs of {name} removed from the simulation, {len(att['bound'])} in all", flush=True)

        grip_holds = {}
        grip_views = {}

        def grip_view(path):
            if path not in grip_views:
                grip_views[path] = RigidPrim(path)
            return grip_views[path]

        def grip_release(k, u, why, regrip):
            h = grip_holds.pop(k, None)
            if h is None:
                return
            if regrip:
                u["node"].regrip()
            print(f"[INFO] grip hold: {h['path'].rsplit('/', 1)[-1]} let go of ({why})", flush=True)

        def grip_hold(k, u, cfg, jaw):
            lo, hi = (float(v) for v in cfg.get("stroke_m", [-1.0, 1.0]))
            if not lo <= jaw <= hi:
                print(f"[INFO] grip hold: the jaws stopped at {jaw * 1000:.1f} mm, outside {lo * 1000:.1f}..{hi * 1000:.1f} mm: no grip, nothing held", flush=True)
                return
            st_ = omni.usd.get_context().get_stage()
            r = u["r"]
            jp = next((q for q in st_.Traverse() if q.GetName() == r["gripper_joints"][0] and q.IsA(UsdPhysics.Joint) and str(q.GetPath()).startswith(r["prim"] + "/")), None)
            body0 = UsdPhysics.Joint(jp).GetBody0Rel().GetTargets() if jp else []
            view, paths = pile_view()
            if not body0 or view is None:
                print("[WARN] grip hold: no palm body of the gripper joint / no parts: nothing held", flush=True)
                return
            palm = str(body0[0])
            p0, q0 = (a.numpy()[0] for a in grip_view(palm).get_world_poses())
            R0 = quat_to_mat(q0)
            tcp = np.zeros(3)
            for j in ET.parse(r["urdf"]).getroot().findall("joint"):
                if "tcp" in j.find("child").get("link") and j.find("parent").get("link") == palm.rsplit("/", 1)[-1] and j.find("origin") is not None:
                    tcp = np.array([float(v) for v in j.find("origin").get("xyz", "0 0 0").split()])
            pos, quat = (a.numpy() for a in view.get_world_poses())
            d = np.linalg.norm(pos - (p0 + R0 @ tcp), axis=1)
            i = int(np.argmin(d))
            radius = float(cfg.get("radius_m", 0.04))
            if d[i] > radius:
                print(f"[INFO] grip hold: no part within {radius * 100:.0f} cm of the TCP (nearest {d[i] * 100:.1f} cm): nothing held", flush=True)
                return
            grip_holds[k] = {"path": paths[i], "palm": palm, "rel_p": R0.T @ (pos[i] - p0), "rel_R": R0.T @ quat_to_mat(quat[i])}
            u["node"].hold_gripper()
            print(f"[INFO] grip hold: {paths[i].rsplit('/', 1)[-1]} ({d[i] * 100:.1f} cm from the TCP, jaws at {jaw * 1000:.1f} mm) is one body with the hand now", flush=True)

        kin_roots = tuple(f"/World/machines/{mc['mname']}/" for mc in machines if isinstance(mc["art"], KinArticulation))

        def palm_state(path):
            if path.startswith(kin_roots):
                T = UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                return np.array(T.ExtractTranslation(), float), np.array(T.ExtractRotationMatrix(), float).T, np.zeros(3), np.zeros(3)
            p0, q0 = (a.numpy()[0] for a in grip_view(path).get_world_poses())
            lin, ang = (a.numpy()[0] for a in grip_view(path).get_velocities())
            return p0, quat_to_mat(q0), lin, ang

        def carry(h):
            view, paths = pile_view()
            if view is None or h["path"] not in paths:
                return False
            p0, R0, lin, ang = palm_state(h["palm"])
            at = p0 + R0 @ h["rel_p"]
            idx = [paths.index(h["path"])]
            view.set_world_poses(at[None].astype(np.float32), mat_to_quat(R0 @ h["rel_R"])[None].astype(np.float32), indices=idx)
            view.set_velocities((lin + np.cross(ang, at - p0))[None].astype(np.float32), ang[None].astype(np.float32), indices=idx)
            return True

        def grip_step():
            for k, u in enumerate(units):
                cfg = u["r"].get("grip_hold")
                while u["node"].grip_events:
                    ev = u["node"].grip_events.popleft()
                    if ev[0] == "command":
                        grip_release(k, u, "gripper command", False)
                    elif ev[0] == "corner" and ev[1] and str((cfg or {}).get("release", "corner")).lower() == "corner":
                        grip_release(k, u, "last corner of the path: the jaws press on it again", True)
                    elif ev[0] == "near" and str((cfg or {}).get("release", "corner")).lower() == "near":
                        grip_release(k, u, "the arm is within 0.03 rad of the end of the path", False)
                    elif ev[0] == "done" and cfg and ev[3] == "blocked" and (ev[1] > 0.5) == (str(cfg.get("when", "open")).lower() == "open"):
                        grip_hold(k, u, cfg, ev[2])
                h = grip_holds.get(k)
                if h is not None and not carry(h):
                    grip_holds.pop(k, None)

        vac_holds, vac_req, vac_result = {}, deque(), {}
        rides, ride_links = {}, {}
        wake = []

        def colliders_on(prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                wake.append(str(prim.GetPath()))
            set_colliders(prim, True)

        def wake_step():
            st_ = omni.usd.get_context().get_stage()
            while wake:
                prim = st_.GetPrimAtPath(wake.pop())
                if prim.IsValid():
                    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(False)

        for k, u in enumerate(units):
            vc = u["r"].get("vacuum")
            if vc:
                vtopic = str(vc.get("topic") or "/vacuum/command")
                u["node"].create_subscription(Bool, vtopic, lambda m, k=k: vac_req.append((k, bool(m.data))), 10)
                vac_result[k] = u["node"].create_publisher(String, f"{vtopic}/result", 10)
                print(f"[INFO] vacuum of {u['r']['name']}: {vtopic} (std_msgs/Bool) true = suction on, false = off; answer on {vtopic}/result (held | empty | released)", flush=True)

        def vacuum_on(k, u, vc):
            r = u["r"]
            st_ = omni.usd.get_context().get_stage()
            body = next((q for q in st_.Traverse() if q.GetName() == vc["body"] and str(q.GetPath()).startswith(r["prim"] + "/")), None)
            view, paths = pile_view()
            if body is None or view is None:
                print(f"[WARN] vacuum: no body {vc['body']} under {r['prim']} / no parts: nothing held", flush=True)
                return "empty"
            palm = str(body.GetPath())
            p0, q0 = (a.numpy()[0] for a in grip_view(palm).get_world_poses())
            R0 = quat_to_mat(q0)
            T = np.linalg.inv(urdf_link_pose(r["urdf"], vc["body"], {}, np.eye(4))) @ urdf_link_pose(r["urdf"], vc["tcp_link"], {}, np.eye(4))
            tcp = p0 + R0 @ T[:3, 3]
            pos, quat = (a.numpy() for a in view.get_world_poses())
            taken = bound_parts() | {h["path"] for h in grip_holds.values()} | set(rides)
            d = np.linalg.norm(pos - tcp, axis=1)
            d[[i for i, p_ in enumerate(paths) if p_ in taken]] = np.inf
            i = int(np.argmin(d))
            radius = float(vc.get("radius_m", 0.05))
            if d[i] > radius:
                print(f"[INFO] vacuum: no part within {radius * 100:.0f} cm of the TCP (nearest {d[i] * 100:.1f} cm): nothing held", flush=True)
                return "empty"
            part = st_.GetPrimAtPath(paths[i])
            if UsdPhysics.RigidBodyAPI(part).GetKinematicEnabledAttr().Get():
                sw = next((s_ for s_ in switch if paths[i] in s_["parts"]), None)
                if sw is not None and not sw["on"]:
                    switch_crate(sw, True)
                else:
                    set_colliders(part, True)
                    UsdPhysics.RigidBodyAPI(part).CreateKinematicEnabledAttr(False)
            vac_holds[k] = {"path": paths[i], "palm": palm, "rel_p": R0.T @ (pos[i] - p0), "rel_R": R0.T @ quat_to_mat(quat[i]), "fresh": True}
            set_colliders(part, False)
            print(f"[INFO] vacuum: {paths[i].rsplit('/', 1)[-1]} ({d[i] * 100:.1f} cm from the TCP) sucked on", flush=True)
            return "held"

        def attach_on_release(h, cfg):
            st_ = omni.usd.get_context().get_stage()
            view, paths = pile_view()
            if view is None or h["path"] not in paths:
                return None
            links_key = tuple(str(q) for q in cfg.get("links", []))
            if links_key not in ride_links:
                rx = [re.compile(q) for q in links_key]
                ride_links[links_key] = [str(q.GetPath()) for q in st_.Traverse() if q.HasAPI(UsdPhysics.RigidBodyAPI) and any(r.fullmatch(q.GetName()) for r in rx)]
            i = paths.index(h["path"])
            pos, quat = (a.numpy()[i] for a in view.get_world_poses())
            best = None
            for link in ride_links[links_key]:
                p0 = grip_view(link).get_world_poses()[0].numpy()[0]
                d = float(np.linalg.norm(pos - p0))
                if d < float(cfg.get("radius_m", 0.1)) and (best is None or d < best[0]):
                    best = (d, link)
            if best is None:
                return None
            p0, q0 = (a.numpy()[0] for a in grip_view(best[1]).get_world_poses())
            R0 = quat_to_mat(q0)
            rides[h["path"]] = {"path": h["path"], "palm": best[1], "rel_p": R0.T @ (pos - p0), "rel_R": R0.T @ quat_to_mat(quat)}
            return best[1].rsplit("/", 1)[-1]

        def vacuum_step():
            while vac_req:
                k, on = vac_req.popleft()
                vc = units[k]["r"]["vacuum"]
                try:
                    if on:
                        state = "held" if k in vac_holds else vacuum_on(k, units[k], vc)
                    else:
                        h = vac_holds.pop(k, None)
                        state = "released" if h else "empty"
                        link = attach_on_release(h, vc["attach_on_release"]) if h and vc.get("attach_on_release") else None
                        if h and not link:
                            colliders_on(omni.usd.get_context().get_stage().GetPrimAtPath(h["path"]))
                        print(f"[INFO] vacuum: {h['path'].rsplit('/', 1)[-1] + (' stays on ' + link + ' (follows it, no collider)' if link else ' let go of') if h else 'off, nothing was held'}", flush=True)
                except Exception as e:
                    print(f"[WARN] vacuum {'on' if on else 'off'}: {type(e).__name__}: {e}", flush=True)
                    state = "empty"
                vac_result[k].publish(String(data=state))
            for k, h in list(vac_holds.items()):
                if h.pop("fresh", False):
                    continue
                if not carry(h):
                    vac_holds.pop(k)
            for path, h in list(rides.items()):
                if not carry(h):
                    rides.pop(path)

        def cup_link(mc, name):
            st_ = omni.usd.get_context().get_stage()
            return next((str(q.GetPath()) for q in st_.Traverse() if q.GetName() == name and str(q.GetPath()).startswith(f"/World/machines/{mc['mname']}/")), None)

        def cup_hold(mc, cup, vc):
            view, paths = pile_view()
            palm = cup_link(mc, cup)
            if view is None or palm is None:
                return None
            p0, R0, _, _ = palm_state(palm)
            pos, quat = (a.numpy() for a in view.get_world_poses())
            d = np.linalg.norm(pos - p0, axis=1)
            taken = bound_parts() | {h["path"] for h in grip_holds.values()} | {h["path"] for h in vac_holds.values()}
            d[[i for i, p_ in enumerate(paths) if p_ in taken]] = np.inf
            i = int(np.argmin(d))
            if d[i] > float(vc.get("radius_m", 0.05)):
                print(f"[INFO] {mc['mname']}: no part within {float(vc.get('radius_m', 0.05)) * 100:.0f} cm of {cup} (nearest {d[i] * 100:.1f} cm): nothing held", flush=True)
                return None
            rides.pop(paths[i], None)
            set_colliders(omni.usd.get_context().get_stage().GetPrimAtPath(paths[i]), False)
            print(f"[INFO] {mc['mname']}: {paths[i].rsplit('/', 1)[-1]} ({d[i] * 100:.1f} cm from {cup}) sucked on", flush=True)
            return {"path": paths[i], "palm": palm, "rel_p": R0.T @ (pos[i] - p0), "rel_R": R0.T @ quat_to_mat(quat[i]), "fresh": True}

        def machine_vacuum_step():
            for mc in machines:
                vc = mc["m"].get("vacuum")
                node_ = mc["node"]
                if not vc or not node_.vacuum_req:
                    continue
                while node_.vacuum_req:
                    mask = node_.vacuum_req.popleft()
                    for i, cup in enumerate(vc["cups"]):
                        key = (mc["mname"], i)
                        try:
                            if mask >> i & 1 and key not in vac_holds:
                                h = cup_hold(mc, cup, vc)
                                if h:
                                    vac_holds[key] = h
                            elif not mask >> i & 1 and key in vac_holds:
                                h = vac_holds.pop(key)
                                link = attach_on_release(h, vc["attach_on_release"]) if vc.get("attach_on_release") else None
                                if not link:
                                    UsdPhysics.RigidBodyAPI(omni.usd.get_context().get_stage().GetPrimAtPath(h["path"])).CreateKinematicEnabledAttr(True)
                                print(f"[INFO] {mc['mname']}: {h['path'].rsplit('/', 1)[-1] + (' stays on ' + link + ' (follows it, no collider)' if link else ' put down where it is (no jig within attach_on_release.radius_m): kinematic, no collider')}", flush=True)
                        except Exception as e:
                            print(f"[WARN] {mc['mname']} {cup}: {type(e).__name__}: {e}", flush=True)
                node_.vacuum_pub.publish(Int32(data=sum(1 << i for i in range(len(vc["cups"])) if (mc["mname"], i) in vac_holds)))

        jaw_holds, jaw_still = {}, {}

        def jaw_hold(mc, gc):
            view, paths = pile_view()
            palm = cup_link(mc, gc["link"])
            if view is None or palm is None:
                return None
            m = mc["m"]
            pose = (m.get("pos_m") or [0, 0, 0], m.get("rpy_deg") or [0, 0, 0])
            wx, wy, _ = sim_config.to_world(pose, [float(v) for v in gc["point_m"][:2]] + [0.0])
            zlo, zhi = (float(v) + float(pose[0][2]) for v in gc.get("z_m", [0.9, 1.2]))
            pos, quat = (a.numpy() for a in view.get_world_poses())
            d = np.hypot(pos[:, 0] - wx, pos[:, 1] - wy)
            taken = bound_parts() | {h["path"] for h in list(grip_holds.values()) + list(vac_holds.values()) + list(jaw_holds.values())}
            d[[i for i, p_ in enumerate(paths) if p_ in taken or not zlo < pos[i, 2] < zhi]] = np.inf
            i = int(np.argmin(d))
            if d[i] > float(gc.get("radius_m", 0.03)):
                print(f"[INFO] {mc['mname']}: the jaws {gc['jaws']} stopped on something, but no part within {float(gc.get('radius_m', 0.03)) * 100:.0f} cm of {gc['point_m']}: nothing held", flush=True)
                return None
            p0, R0, _, _ = palm_state(palm)
            print(f"[INFO] {mc['mname']}: {paths[i].rsplit('/', 1)[-1]} ({d[i] * 100:.1f} cm from {gc['point_m']}) is held by the jaws (follows {gc['link']})", flush=True)
            return {"path": paths[i], "palm": palm, "rel_p": R0.T @ (pos[i] - p0), "rel_R": R0.T @ quat_to_mat(quat[i])}

        def machine_grip_step():
            for mc in machines:
                for g, gc in enumerate(mc["m"].get("grip_holds") or []):
                    key = (mc["mname"], g)
                    try:
                        idx = [mc["names"].index(j) for j in gc["jaws"]]
                        q, cmd = np.asarray(mc["q"], float)[idx], np.asarray(mc["node"].targets(), float)[idx]
                        h = jaw_holds.get(key)
                        if h is not None:
                            if np.any(cmd > q + 0.001) or not carry(h):
                                jaw_holds.pop(key)
                                link = attach_on_release(h, gc["attach_on_release"]) if gc.get("attach_on_release") else None
                                if link:
                                    set_colliders(omni.usd.get_context().get_stage().GetPrimAtPath(h["path"]), False)
                                print(f"[INFO] {mc['mname']}: {h['path'].rsplit('/', 1)[-1]} let go of (the jaws open)" + (f", stays on {link} (follows it, no collider)" if link else ""), flush=True)
                            continue
                        if q.min() > float(gc.get("reopen_m", 0.006)):
                            jaw_still[key] = False
                        elif not jaw_still.get(key) and q.max() < float(gc.get("engage_m", 0.002)) and np.all(cmd < q + 0.001):
                            jaw_still[key] = True
                            h = jaw_hold(mc, gc)
                            if h:
                                jaw_holds[key] = h
                    except Exception as e:
                        print(f"[WARN] {mc['mname']} grip_holds {g}: {type(e).__name__}: {e}", flush=True)

        def attach_step():
            for att in attaches:
                try:
                    if att["mc"]["node"].attach_requests:
                        att["mc"]["node"].attach_requests.clear()
                        att["due"] = timeline.get_current_time() + float(att["cfg"].get("delay_s", 0.0))
                    if att.get("due") is not None and timeline.get_current_time() >= att["due"]:
                        att["due"] = None
                        attach_parts(att)
                except Exception as e:
                    print(f"[WARN] attach to {att['mc']['mname']}: {type(e).__name__}: {str(e)[:100]}", flush=True)

        watchers = [mc for mc in machines if mc["m"].get("hands") or mc["m"].get("part_pose") or mc["m"].get("sensors")]

        def watch_machine(mc, pn, qn):
            m = mc["m"]
            pose = (m.get("pos_m") or [0, 0, 0], m.get("rpy_deg") or [0, 0, 0])
            hands = m.get("hands") or {}
            if hands and mc["node"].hands_pub is not None:
                car = m.get("carriers") or {}
                ax = [float(v) for v in car.get("axis", [1, 0, 0])]
                o = [float(v) for v in hands.get("origin_m", [0.0, 0.0, 0.0])]
                zlo, zhi = (float(v) + float(pose[0][2]) for v in hands.get("z_m", [0.9, 1.2]))
                mask = 0
                for i, k in enumerate(hands.get("slots") or []):
                    at = float(car["slots_m"][int(k)])
                    hx, hy, _ = sim_config.to_world(pose, [o[0] + ax[0] * at, o[1] + ax[1] * at, o[2] + ax[2] * at])
                    if np.any((np.hypot(pn[:, 0] - hx, pn[:, 1] - hy) < float(hands.get("radius_m", 0.03))) & (pn[:, 2] > zlo) & (pn[:, 2] < zhi)):
                        mask |= 1 << i
                mc["node"].hands_pub.publish(Int32(data=int(mask)))
            sensors = m.get("sensors") or {}
            if sensors and mc["node"].sensors_pub is not None:
                zlo, zhi = (float(v) + float(pose[0][2]) for v in sensors.get("z_m", [0.9, 1.2]))
                mask = 0
                for i, (sx, sy) in enumerate((sensors.get("points_m") or {}).values()):
                    wx, wy, _ = sim_config.to_world(pose, [float(sx), float(sy), 0.0])
                    if len(pn) and np.any((np.hypot(pn[:, 0] - wx, pn[:, 1] - wy) < float(sensors.get("radius_m", 0.03))) & (pn[:, 2] > zlo) & (pn[:, 2] < zhi)):
                        mask |= 1 << i
                mc["node"].sensors_pub.publish(Int32(data=int(mask)))
            seat = m.get("part_pose") or {}
            if seat and mc["node"].part_pose_pub is not None and len(pn):
                sx, sy = (float(v) for v in (seat.get("seat_m") or [0.0, 0.0])[:2])
                wx, wy, _ = sim_config.to_world(pose, [sx, sy, 0.0])
                d = np.hypot(pn[:, 0] - wx, pn[:, 1] - wy)
                k = int(np.argmin(d))
                if d[k] < float(seat.get("radius_m", 0.08)):
                    mc["node"].publish_part_pose(pn[k], qn[k])

        stat = {"t": time.time(), "f": 0, "sim": timeline.get_current_time()}
        BOOM_SPEED = 2.0
        boom = {"t": 0.0, "seen": {}, "n": 0}

        def boom_step():
            now = time.time()
            if now - boom["t"] < 0.1 or part_view[0] is None or boom["n"] >= 40:
                return
            boom["t"] = now
            try:
                lin = part_view[0].get_velocities()[0].numpy()
                speed = np.linalg.norm(lin, axis=1)
                fast = np.where(speed > BOOM_SPEED)[0]
                if not len(fast):
                    return
                pos = part_view[0].get_world_poses()[0].numpy()
            except Exception:
                return
            for i in fast[:3]:
                path = part_view[1][int(i)]
                if path in bound_parts() or now - boom["seen"].get(path, 0.0) < 5.0:
                    continue
                boom["seen"][path] = now
                boom["n"] += 1
                near = [part_view[1][int(j)].rsplit("/", 1)[-1] for j in np.where((np.hypot(pos[:, 0] - pos[i, 0], pos[:, 1] - pos[i, 1]) < 0.06) & (np.abs(pos[:, 2] - pos[i, 2]) < 0.04))[0] if int(j) != int(i)]
                ev = "; ".join(f"{now - t:5.1f}s ago: {text}" for t, text in plate_ev.get(path, [])[-4:]) or "nothing"
                rb = "; ".join(f"{now - t:4.1f}s ago: {text}" for t, text in list(node.events)[-6:] if now - t < 8.0) or "nothing in the last 8 s"
                print(f"[BOOM] {path.rsplit('/', 1)[-1]} {speed[i]:.1f} m/s at {pos[i, 0]:+.3f} {pos[i, 1]:+.3f} {pos[i, 2]:+.3f} | plates within 6 cm: {near or 'none'} | "
                      f"collider events: {ev} | robot: {rb}", flush=True)

        def stat_step():
            now = time.time()
            if now - stat["t"] < 30.0:
                return
            fps = (node.frame - stat["f"]) / (now - stat["t"])
            sim_now = timeline.get_current_time()
            per_frame = (sim_now - stat["sim"]) / max(1, node.frame - stat["f"])
            stat["t"], stat["f"], stat["sim"] = now, node.frame, sim_now
            rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
            moving = ""
            try:
                view_, _ = pile_view()
                sp = np.linalg.norm(view_.get_velocities()[0].numpy(), axis=1) if view_ is not None else np.zeros(0)
                mv = sp[sp > 0.0005]
                moving = f", parts moving {len(mv)} of {len(sp)}" + (f" (median {np.median(mv) * 1000:.1f} mm/s, 95 % {np.percentile(mv, 95) * 1000:.1f} mm/s)" if len(mv) else "")
            except Exception:
                pass
            print(f"[STAT] {fps:5.1f} fps ({per_frame * 1000:.1f} ms of simulated time per frame), rss {rss:6.0f} MB, fine collider {len(refined)}, stiff {len(stiff_order)}, removed {len(bound_parts())}{moving}", flush=True)

        switch, switch_req, clear_req, cleared = [], deque(), deque(), set()
        parts_req, parts_pub, remove_req = deque(), None, deque()
        if robot.get("pallets") and robot["pallets"].get("plan") and any(len(v) > 2 and v[2] for v in (robot.get("pallet_physics") or {}).values()):
            pc_ = robot["pallets"]
            pname_ = (pc_.get("part_cfg") or {}).get("name", "part")
            for c in pc_["plan"]["crates"]:
                ph = robot["pallet_physics"].get((c["pallet"], c["layer"]))
                if ph and ph[0] and ph[2]:
                    switch.append({"name": f"pallet {c['pallet']} layer {c['layer']} crate {c['idx']}", "pos": [float(v) for v in c["pos"]],
                                   "on": not c.get("switch_off") or float(pc_.get("settle_s", 0.0)) > 0, "start_off": bool(c.get("switch_off")),
                                   "pallet": c["pallet"], "layer": c["layer"],
                                   "crate": f"/World/pallets/pallet_{c['pallet']}/L{c['layer']}_C{c['idx']}/crate",
                                   "parts": [f"/World/pile/{pname_}_p{p['pallet']}_L{p['layer']}_C{p['crate']}_{p['j']:02d}" for p in pc_["plan"]["parts"]
                                             if (p["pallet"], p["layer"], p["crate"]) == (c["pallet"], c["layer"], c["idx"])]})
            r_in = math.hypot(*[float(v) for v in pc_["crate"]["inner_size_m"][:2]]) / 2.0
            topic = str(pc_.get("switch_topic") or "/crate_active")
            node.create_subscription(String, topic, lambda m: switch_req.append(m.data), 10)
            node.create_subscription(String, str(pc_.get("clear_topic") or "/crate_clear"), lambda m: clear_req.append(m.data), 10)
            parts_pub = node.create_publisher(String, str(pc_.get("parts_topic") or "/crate_parts"), 10)
            node.create_subscription(String, str(pc_.get("parts_request_topic") or "/crate_parts_request"), lambda m: parts_req.append(m.data), 10)
            node.create_subscription(String, str(pc_.get("parts_remove_topic") or "/crate_parts_remove"), lambda m: remove_req.append(m.data), 10)
            print(f"[INFO] pallets: {len(switch)} crates switched one at a time ({sum(sw['on'] for sw in switch)} on now) from {topic}", flush=True)

        def switch_crate(sw, on, quiet=False):
            st_ = omni.usd.get_context().get_stage()
            set_colliders(st_.GetPrimAtPath(sw["crate"]), on)
            pos, n = pile_positions(), 0
            gone = bound_parts() | {h["path"] for h in list(grip_holds.values()) + list(vac_holds.values()) + list(jaw_holds.values())} | set(rides)
            for path in sw["parts"]:
                at = pos.get(path)
                if path in gone or at is None or math.hypot(at[0] - sw["pos"][0], at[1] - sw["pos"][1]) > r_in or abs(at[2] - sw["pos"][2]) > 0.3:
                    continue
                part = st_.GetPrimAtPath(path)
                if not on:
                    set_click_physx(part, False)
                set_colliders(part, on)
                UsdPhysics.RigidBodyAPI(part).CreateKinematicEnabledAttr(not on)
                n += 1
            sw["on"] = on
            if not quiet:
                print(f"[INFO] {sw['name']}: physics {'on' if on else 'off'} (the crate and {n} part(s))", flush=True)

        taken = set()

        def take_away(pallet, layer):
            st_ = omni.usd.get_context().get_stage()
            root = st_.GetPrimAtPath(f"/World/pallets/pallet_{pallet}")
            n = 0
            for q in (root.GetChildren() if root.IsValid() else []):
                if q.GetName().startswith(f"L{layer}_"):
                    UsdGeom.Imageable(q).MakeInvisible()
                    set_colliders(q, False)
                    n += 1
            for sw in switch:
                if (sw["pallet"], sw["layer"]) == (pallet, layer):
                    if sw["on"]:
                        switch_crate(sw, False)
                    for path in sw["parts"]:
                        part = st_.GetPrimAtPath(path)
                        if part.IsValid() and path not in bound_parts():
                            UsdGeom.Imageable(part).MakeInvisible()
                            set_colliders(part, False)
            taken.add((pallet, layer))
            print(f"[INFO] pallet {pallet} layer {layer} is done: taken away ({n} crates / parts hidden, no colliders)", flush=True)

        layer_req, layer_hidden = deque(), {}
        node.create_subscription(Int32, "/crate_layer_toggle", lambda m: layer_req.append(int(m.data)), 10)

        def layer_snapshot(prim):
            vis = UsdGeom.Imageable(prim).GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible
            cols = [(q.GetPath(), bool(UsdPhysics.CollisionAPI(q).GetCollisionEnabledAttr().Get() is not False)) for q in Usd.PrimRange(prim) if q.HasAPI(UsdPhysics.CollisionAPI)]
            return vis, cols

        def layer_step():
            while layer_req:
                number = layer_req.popleft()
                try:
                    pc = robot.get("pallets") or {}
                    plan, layers = pc.get("plan"), int(pc.get("layers", 0))
                    if not plan or not 1 <= number <= layers:
                        print(f"[WARN] /crate_layer_toggle {number}: the layers are 1 (bottom) .. {layers} (top)" + ("" if plan else ", but this map has no pallets"), flush=True)
                        continue
                    st_ = omni.usd.get_context().get_stage()
                    layer = number - 1
                    if layer in layer_hidden:
                        for path, vis, cols, kin in layer_hidden.pop(layer):
                            layer_gone.discard(str(path))
                            prim = st_.GetPrimAtPath(path)
                            if not prim.IsValid():
                                continue
                            UsdGeom.Imageable(prim).MakeVisible() if vis else UsdGeom.Imageable(prim).MakeInvisible()
                            for cpath, on in cols:
                                UsdPhysics.CollisionAPI(st_.GetPrimAtPath(cpath)).CreateCollisionEnabledAttr(on)
                            if kin is not None:
                                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(kin)
                        print(f"[INFO] layer {number}: the crates and the parts are back as they were", flush=True)
                        continue
                    saved, n_crates, n_parts = [], 0, 0
                    pname = (pc.get("part_cfg") or {}).get("name", "part")
                    root = st_.GetPrimAtPath("/World/pallets")
                    for pallet in (root.GetChildren() if root.IsValid() else []):
                        for q in pallet.GetChildren():
                            if q.GetName().startswith(f"L{layer}_"):
                                vis, cols = layer_snapshot(q)
                                saved.append((q.GetPath(), vis, cols, None))
                                UsdGeom.Imageable(q).MakeInvisible()
                                set_colliders(q, False)
                                n_crates += 1
                    gone = bound_parts() | {h["path"] for h in list(grip_holds.values()) + list(vac_holds.values()) + list(jaw_holds.values())} | set(rides)
                    for p_ in plan["parts"]:
                        path = f"/World/pile/{pname}_p{p_['pallet']}_L{p_['layer']}_C{p_['crate']}_{p_['j']:02d}"
                        part = st_.GetPrimAtPath(path)
                        if p_["layer"] != layer or path in gone or not part.IsValid():
                            continue
                        vis, cols = layer_snapshot(part)
                        saved.append((part.GetPath(), vis, cols, bool(UsdPhysics.RigidBodyAPI(part).GetKinematicEnabledAttr().Get())))
                        layer_gone.add(path)
                        UsdGeom.Imageable(part).MakeInvisible()
                        set_colliders(part, False)
                        UsdPhysics.RigidBodyAPI(part).CreateKinematicEnabledAttr(True)
                        n_parts += 1
                    layer_hidden[layer] = saved
                    print(f"[INFO] layer {number}: {n_crates} crates and {n_parts} parts hidden (no colliders); the same number again brings them back", flush=True)
                except Exception as e:
                    print(f"[WARN] /crate_layer_toggle {number}: {type(e).__name__}: {e}", flush=True)

        def switch_step():
            while switch_req:
                raw = switch_req.popleft()
                try:
                    msg = json.loads(raw)
                    cx, cy = (float(v) for v in msg["centre"][:2])
                    fz = msg.get("floor_z")
                    floor = float(robot["pallets"]["crate"]["floor_z_m"])

                    def off(sw):
                        return math.hypot(sw["pos"][0] - cx, sw["pos"][1] - cy) + (abs(sw["pos"][2] + floor - float(fz)) if fz is not None else 0.0)
                    target = min(switch, key=off)
                    if off(target) > 0.2:
                        print(f"[WARN] /crate_active {raw}: no switched crate there", flush=True)
                        continue
                    for lay in sorted({sw["layer"] for sw in switch if sw["pallet"] == target["pallet"] and sw["layer"] > target["layer"]}):
                        if (target["pallet"], lay) not in taken:
                            take_away(target["pallet"], lay)
                    for sw in switch:
                        if sw is not target and sw["on"] and bool((robot.get("pallets") or {}).get("freeze_after_settle", True)):
                            switch_crate(sw, False)
                    if not target["on"]:
                        switch_crate(target, True)
                except Exception as e:
                    print(f"[WARN] /crate_active {raw}: {type(e).__name__}: {e}", flush=True)

        def clear_step():
            while clear_req:
                raw = clear_req.popleft()
                try:
                    msg = json.loads(raw)
                    cx, cy = (float(v) for v in msg["centre"][:2])
                    fz = msg.get("floor_z")
                    floor = float(robot["pallets"]["crate"]["floor_z_m"])

                    def off(sw):
                        return math.hypot(sw["pos"][0] - cx, sw["pos"][1] - cy) + (abs(sw["pos"][2] + floor - float(fz)) if fz is not None else 0.0)
                    target = min(switch, key=off)
                    if off(target) > 0.2:
                        print(f"[WARN] /crate_clear {raw}: no switched crate there", flush=True)
                        continue
                    cleared.add(target["name"])
                    same = [sw for sw in switch if (sw["pallet"], sw["layer"]) == (target["pallet"], target["layer"])]
                    if (target["pallet"], target["layer"]) not in taken and all(sw["name"] in cleared for sw in same):
                        take_away(target["pallet"], target["layer"])
                except Exception as e:
                    print(f"[WARN] /crate_clear {raw}: {type(e).__name__}: {e}", flush=True)

        def parts_step():
            while parts_req:
                raw = parts_req.popleft()
                try:
                    msg = json.loads(raw)
                    cx, cy = (float(v) for v in msg["centre"][:2])
                    fz = msg.get("floor_z")
                    floor = float(robot["pallets"]["crate"]["floor_z_m"])

                    def off(sw):
                        return math.hypot(sw["pos"][0] - cx, sw["pos"][1] - cy) + (abs(sw["pos"][2] + floor - float(fz)) if fz is not None else 0.0)
                    target = min(switch, key=off)
                    if off(target) > 0.2:
                        print(f"[WARN] /crate_parts_request {raw}: no switched crate there", flush=True)
                        parts_pub.publish(String(data=json.dumps({"index": msg.get("index"), "parts": [], "missing": True})))
                        continue
                    view, paths = pile_view()
                    out = []
                    if view is not None:
                        pos_, quat_ = view.get_world_poses()
                        pn, qn = pos_.numpy(), quat_.numpy()
                        row, gone = {p_: i_ for i_, p_ in enumerate(paths)}, bound_parts()
                        for path in target["parts"]:
                            i = row.get(path)
                            if i is None or path in gone:
                                continue
                            x, y, z = (float(v) for v in pn[i])
                            if math.hypot(x - target["pos"][0], y - target["pos"][1]) > r_in or abs(z - target["pos"][2]) > 0.3:
                                continue
                            qw, qx, qy, qz = (float(v) for v in qn[i])
                            roll, pitch, yaw = Gf.Rotation(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz))).Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
                            if units[0]["r"].get("vacuum"):
                                tcp_r, tcp_p, tcp_y, flipped = vacuum_tcp(qw, qx, qy, qz)
                            else:
                                tcp_r, tcp_p, tcp_y, flipped = grasp_tcp.tcp_rpy(roll, pitch, yaw)
                            state = "turn_over" if flipped else "standard"
                            flat = abs(1.0 - 2.0 * (qx * qx + qy * qy))
                            out.append({"path": path, "xyz_m": [x, y, z], "state": state, "flat": flat, "tilt_deg": math.degrees(math.acos(min(1.0, flat))),
                                        "ee_6d": f"{x:.4f} {y:.4f} {z:.4f} {tcp_r + 0.0:.1f} {tcp_p + 0.0:.1f} {tcp_y + 0.0:.2f} {state}"})
                    parts_pub.publish(String(data=json.dumps({"index": msg.get("index"), "parts": out})))
                    print(f"[INFO] {target['name']}: {len(out)} part(s) in the crate sent to /crate_parts ({sum(p_['state'] == 'turn_over' for p_ in out)} turned over)", flush=True)
                except Exception as e:
                    print(f"[WARN] /crate_parts_request {raw}: {type(e).__name__}: {e}", flush=True)

        def remove_step():
            while remove_req:
                raw = remove_req.popleft()
                try:
                    msg = json.loads(raw)
                    pts, radius = np.array(msg["xyz_m"], dtype=float).reshape(-1, 3), float(msg.get("radius_m", 0.08))
                    view, paths = pile_view()
                    if view is None:
                        continue
                    pn, st_ = view.get_world_poses()[0].numpy(), omni.usd.get_context().get_stage()
                    keep = bound_parts() | {h["path"] for h in grip_holds.values()}
                    out = [i for i, p_ in enumerate(pn) if paths[i] not in keep and np.linalg.norm(pts - p_, axis=1).min() < radius]
                    for i in out:
                        retire(st_, paths[i], "removed at a place pose: hidden, collider off, parked")
                        removed.add(paths[i])
                        rides.pop(paths[i], None)
                    if out:
                        park(view, out)
                    print(f"[INFO] /crate_parts_remove: {len(out)} part(s) within {radius * 100:.0f} cm of the place poses removed from the simulation, {len(removed)} in all", flush=True)
                except Exception as e:
                    print(f"[WARN] /crate_parts_remove {raw}: {type(e).__name__}: {e}", flush=True)

        cam_req = deque()
        for c in robot.get("cameras") or []:
            node.create_subscription(Empty, f"/{c['name']}/capture", lambda _, c=c: cam_req.append(c), 10)

        settle_s = float((robot.get("pallets") or {}).get("settle_s", 0.0))
        fixture_friction()
        if settle_s > 0:
            t_settle, frames, quiet = time.time(), 0, 0
            cap_v = float((robot.get("pallets") or {}).get("settle_max_speed_mps", 1.0))
            st_cap = omni.usd.get_context().get_stage()
            for path_ in pile_view()[1]:
                PhysxSchema.PhysxRigidBodyAPI.Apply(st_cap.GetPrimAtPath(path_)).CreateMaxLinearVelocityAttr().Set(cap_v)
            at_start = {sw["name"]: not sw["start_off"] for sw in switch}
            for sw in switch:
                if not sw["on"]:
                    switch_crate(sw, True, quiet=True)
            print(f"[INFO] letting the pile settle (up to {settle_s:g} s of simulated time, ends as soon as nothing moves)"
                  + (f", all {len(switch)} switched crates with physics on meanwhile" if switch else "") + " ...", flush=True)
            while simulation_app.is_running() and frames < int(settle_s * APP_HZ):
                update(1)
                frames += 1
                if frames % 30 == 0:
                    try:
                        view_, _ = pile_view()
                        if view_ is None:
                            break
                        sp_ = np.linalg.norm(view_.get_velocities()[0].numpy(), axis=1)
                        z_ = view_.get_world_poses()[0].numpy()[:, 2]
                        print(f"[INFO] settle {frames / APP_HZ:.1f} s: fastest part {sp_.max():.2f} m/s, {int((sp_ > 1.0).sum())} faster than 1 m/s, {int((sp_ > 0.0005).sum())} moving, z {z_.min():.2f}..{z_.max():.2f} m", flush=True)
                        quiet = 0 if (sp_ > 0.0005).any() else quiet + 1
                    except Exception:
                        break
                    if quiet >= 2:
                        break
            print(f"[INFO] the pile settled after {frames / APP_HZ:.1f} s of simulated time ({time.time() - t_settle:.0f} s)", flush=True)
            for path_ in pile_view()[1]:
                PhysxSchema.PhysxRigidBodyAPI.Apply(st_cap.GetPrimAtPath(path_)).CreateMaxLinearVelocityAttr().Set(1.0e6)
            freeze = bool((robot.get("pallets") or {}).get("freeze_after_settle", True))
            for sw in switch:
                if freeze and not at_start[sw["name"]]:
                    switch_crate(sw, False, quiet=True)
            if switch and freeze:
                print(f"[INFO] the settled poses are frozen: physics of the {len(switch)} crates off again, a crate gets it back (SDF) when /crate_active names it", flush=True)
            elif switch:
                print(f"[INFO] the settled poses are kept: the {len(switch)} crates stay on (pallets.freeze_after_settle: false), /crate_active switches the others off", flush=True)
            if any(not isinstance(mc["art"], KinArticulation) for mc in machines):
                view_, paths_ = pile_view()
                if view_ is not None:
                    st_bake = omni.usd.get_context().get_stage()
                    pos_b, quat_b = (a.numpy().copy() for a in view_.get_world_poses())
                    paths_b = list(paths_)

                    def bake_poses():
                        for i_, path_ in enumerate(paths_b):
                            ops_ = UsdGeom.Xformable(st_bake.GetPrimAtPath(path_)).GetOrderedXformOps()
                            if len(ops_) >= 2:
                                ops_[0].Set(Gf.Vec3d(*(float(v) for v in pos_b[i_])))
                                ops_[1].Set(Gf.Quatd(float(quat_b[i_][0]), Gf.Vec3d(*(float(v) for v in quat_b[i_][1:]))))
                        print(f"[INFO] the settled poses of {len(paths_b)} parts are written into the stage: the physics restarts from them (the machines start again, the GPU crashed with a machine articulation that lived through the settle)", flush=True)
                    rebuild_views("settled poses baked", bake_poses)

        cycles = start_machine_cycles()
        while simulation_app.is_running():
            wake_step()
            switch_step()
            layer_step()
            clear_step()
            parts_step()
            remove_step()
            if time.time() - fx_state["t"] > 1.0:
                fx_state["t"] = time.time()
                reload_fixtures()
            fixture_friction()
            attach_step()
            grip_step()
            vacuum_step()
            machine_vacuum_step()
            machine_grip_step()
            stat_step()
            boom_step()
            if watchers and time.time() - fx_state.get("tp", 0.0) > 0.2:
                fx_state["tp"] = time.time()
                try:
                    if fx_state.get("pile") is None:
                        st_ = omni.usd.get_context().get_stage()
                        fx_state["pile"] = RigidPrim([str(q.GetPath()) for root_ in ("/World/pile", "/World/objects") if st_.GetPrimAtPath(root_).IsValid() for q in st_.GetPrimAtPath(root_).GetChildren()])
                    pos_, quat_ = fx_state["pile"].get_world_poses()
                    pn, qn = pos_.numpy(), quat_.numpy()
                    for mc in watchers:
                        watch_machine(mc, pn, qn)
                except Exception:
                    fx_state["pile"] = None
            for u in units:
                step = u["node"].ramp(u["cmd"], u["step_slow"] if u["node"].slow else u["step"])
                u["cmd"] = u["node"].limit(u["cmd"], u["node"].targets(u["cmd"], step), step)
            try:
                for u in units:
                    u["art"].set_dof_position_targets(u["cmd"])
                for mc in machines:
                    target = mc["node"].targets()
                    new = target != mc["last_target"]
                    mc["cmd"] = np.where(new & ~mc["carrier"], mc["q"], mc["cmd"])
                    mc["vel"] = np.where(new, 0.0, mc["vel"])
                    mc["last_target"] = target
                    err = target - mc["cmd"]
                    v_want = np.sign(err) * np.minimum(mc["step"] * APP_HZ, np.sqrt(2.0 * mc["acc"] * np.abs(err)))
                    mc["vel"] = mc["vel"] + np.clip(v_want - mc["vel"], -mc["acc"] / APP_HZ, mc["acc"] / APP_HZ)
                    mc["cmd"] = mc["cmd"] + np.clip(mc["vel"] / APP_HZ, -np.abs(err), np.abs(err))
                    mc["cmd"] = np.where(mc["rot"], np.clip(mc["cmd"], mc["q"] - MACHINE_LEAD, mc["q"] + MACHINE_LEAD), mc["cmd"])
                    if mc["carriers"]:
                        cr = mc["carriers"]
                        q = cr["node"].step(1.0 / APP_HZ)
                        jumped = [i for i, (a, b) in zip(cr["idx"], zip(cr["last"], q)) if abs(b - a) > cr["node"].loop / 2]
                        if jumped:
                            mc["art"].set_dof_positions(np.array([[q[cr["idx"].index(i)] for i in jumped]]), dof_indices=jumped)
                            mc["art"].set_dof_velocities(np.zeros((1, len(jumped))), dof_indices=jumped)
                        cr["last"] = q
                        for i, v in zip(cr["idx"], q):
                            mc["cmd"][i] = v
                    mc["art"].set_dof_position_targets(mc["cmd"])
                update(1)
                for u in units:
                    u["node"].report(u["art"].get_dof_positions().numpy()[0], u["art"].get_dof_velocities().numpy()[0])
                for mc in machines:
                    mc["q"] = mc["art"].get_dof_positions().numpy()[0]
                    mc["node"].report(mc["q"], mc["art"].get_dof_velocities().numpy()[0])
                rebuilds = 0
            except Exception as e:
                rebuilds += 1
                if rebuilds > 3:
                    raise
                rebuild_views(f"{type(e).__name__}: {str(e)[:90]}")
                continue
            while clicked:
                path = clicked.pop(0)
                try:
                    if not omni.usd.get_context().get_stage().GetPrimAtPath(path).IsValid():
                        views.pop(path, None)
                        continue
                    if not path.startswith("/World/pallets/") and robot.get("click_refine", True):
                        refine_collision(path)
                    if path.startswith("/World/pallets/"):
                        T = Gf.Transform(UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                        (x, y, z), q = T.GetTranslation(), T.GetRotation().GetQuat()
                        qw, (qx, qy, qz) = q.GetReal(), q.GetImaginary()
                    else:
                        view = views.get(path) or views.setdefault(path, RigidPrim(path))
                        pos, quat = view.get_world_poses()
                        (x, y, z), (qw, qx, qy, qz) = pos.numpy()[0], quat.numpy()[0]
                    roll, pitch, yaw = Gf.Rotation(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz)))).Decompose(
                        Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
                    print(f"{x:.3f} {y:.3f} {z:.3f} {roll + 0.0:.1f} {pitch + 0.0:.1f} {yaw + 0.0:.1f}", flush=True)
                    if units[0]["r"].get("vacuum"):
                        tcp_r, tcp_p, tcp_y, flipped = vacuum_tcp(float(qw), float(qx), float(qy), float(qz))
                    else:
                        tcp_r, tcp_p, tcp_y, flipped = grasp_tcp.tcp_rpy(roll, pitch, yaw)
                    print(f"{x:.4f} {y:.4f} {z:.4f} {tcp_r + 0.0:.1f} {tcp_p + 0.0:.1f} {tcp_y + 0.0:.2f} {'turn_over' if flipped else 'standard'}", flush=True)
                    for u in units:
                        u["node"].publish_turned_over(flipped)
                except Exception as e:
                    print(f"[WARN] pose of {path} not available: {type(e).__name__}: {e}", flush=True)
            for u in units:
                un, r = u["node"], u["r"]
                while un.pictures:
                    path = un.pictures.popleft()
                    if not un.pictures and os.path.isfile(path):
                        try:
                            picture_panel.show(path)
                        except Exception as e:
                            print(f"[WARN] picture {path} not shown: {type(e).__name__}: {e}", flush=True)
                while un.targets_seen:
                    pos = un.targets_seen.popleft()
                    try:
                        refine_target(pos)
                    except Exception as e:
                        print(f"[WARN] part at the target {pos} not refined: {type(e).__name__}: {e}", flush=True)
                if un.capture_requested.is_set():
                    un.capture_requested.clear()
                    try:
                        if not r.get("wrist_camera_path"):
                            raise RuntimeError("no wrist camera: set `wrist_camera:` in the robot yaml (and the link in the URDF)")
                        q_now = dict(zip(u["names"], (float(v) for v in u["art"].get_dof_positions().numpy()[0])))
                        T_base = make_T(rpy_to_mat(r.get("rpy_deg", [0, 0, 0])), r.get("pos_m", [0, 0, 0]))
                        f, valid = capture_wrist_camera(r["wrist_camera"], r["wrist_camera_path"],
                                                        urdf_link_pose(r["urdf"], str(r["wrist_camera"]["link"]), q_now, T_base))
                        print(f"[INFO] wrist camera frame -> {f} ({valid * 100:.0f} % of the pixels have depth)", flush=True)
                    except Exception as e:
                        print(f"[WARN] wrist camera capture failed: {type(e).__name__}: {e}", flush=True)
            while cam_req:
                c = cam_req.popleft()
                try:
                    f = capture_fixed_camera(c, c["path"])
                    print(f"[INFO] camera {c['name']} picture -> {f}", flush=True)
                    picture_panel.show(f)
                except Exception as e:
                    print(f"[WARN] camera {c['name']} capture failed: {type(e).__name__}: {e}", flush=True)
        timeline.stop()
    except KeyboardInterrupt:
        pass
    finally:
        stop_machine_cycles(locals().get("cycles") or [])
        for u in units:
            if u.get("node") is not None:
                u["node"].destroy_node()
        for ctx, _ in contexts.values():
            ctx.try_shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
