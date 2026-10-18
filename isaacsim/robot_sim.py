"""Robot from its URDF in Isaac Sim, controlled over ROS 2 (same bridge as the lite6 test_depthcam_obj_pose_fix.py).

    lab robot_sim.py [robot:=<name> | robot:=configs/robots/<name>.yaml] [map:=<name>] [--name m20_35_18d] [--gripper none]
                     [--force-convert] [--robot-only] [--headless]

The robot is chosen by its config file: one yaml per robot in configs/robots/ (name = the folder of the URDF in ../assets/robot,
controller, gripper, start pose, drive gains). Set `scene.robot: <name>` in configs/map/map.yaml and run `lab robot_sim.py`,
or name it on the command line (`lab robot_sim.py robot:=ur5e` like the bringup launch, which wins over map.yaml; a bare
`ur5e` works too). A name is found as configs/robots/<name>.yaml or as the config whose `name:` it is.
What stands around the robot (objects, bin with a pile, basket) is not part of the robot: it is the `scene:` section of
configs/map/map.yaml (the rest of that file is the dataset config for the data collection) and stays when the robot changes.

../assets/robot/<name>/urdf/<name>.urdf (the same file the MoveIt bringup uses) is converted to USD on the first
run and cached in assets_cache/robot_<name>/ (converted again when the .urdf changes; --force-convert after changing
only meshes). Meshes: relative to the URDF or package://<pkg>/meshes/... (searched in the URDF folder and every folder
above it), so a URDF straight out of xacro works.
A gripper that is part of the robot URDF (lite6) is driven by <gripper_topic> too: list its joints as `gripper_joints:` in the config.
With a gripper (../assets/grippper/<gripper>/urdf/<gripper>.urdf, same layout) its root link (e.g. flange) is joined to the robot's
link of that name first; the joined URDF and its USD are cached in assets_cache/robot_<name>_<gripper>/. Gripper joints
are driven like the robot's (joint_pos, ROS commands); URDF <mimic> joints follow their leader joint.
`objects:` (scene: of configs/map/map.yaml) drops rigid parts from configs/objects/<part>.yaml (the dataset part configs: mesh, mass, collision, material;
USD cached in assets_cache like the dataset pipeline) at pos_m (part center) / rpy_deg.
`bin:` puts the dataset's bin (`bin:` + `loading:` of configs/map/map.yaml) at pos_m / rpy_deg in the world and
drops a pile of `pile.count` parts into it with the dataset's planner (`pile.seed: random` = a different random pile on every
start, the seed is printed; a number = always the same pile); bin and parts are separate world-frame prims
(/World/bin, /World/pile/<part>_NNN), so part poses are robot-base coordinates.
`basket:` (scene: of configs/map/map.yaml) an open box (floor + 4 walls, kinematic; the dataset's procedural bin, no STL) at pos_m (bottom centre) /
rpy_deg in the world (/World/basket): only an object of the scene (control/robot_sequence.py does not read it).
`map:=<name>`: if configs/map/<name>.yaml exists it is the scene (e.g. PUMP_COVER_MAP.yaml: map.yaml with another box and part), else a USD below.
`map:=<name>` (or `scene.map` of map.yaml; `map:=none` = no map) loads an environment made in the Isaac Sim GUI: ../assets/map/<name>.usd (or a path),
every top-level prim of it except what the Kit stage saved around the scene (render settings, viewport cameras, light rig), as /World/map/<prim>, at the world origin. Its colliders (rigid / collision
set in the GUI) are used as they are.
`platform:` (scene: of configs/map/map.yaml) a static box (pedestal) at pos_m (bottom centre) / rpy_deg, size_m [x, y, z] (/World/platform).
A robot whose pos_m x y is on it stands on top: the z of its yaml (pos_m) is measured from the top of the platform (robot.launch.py does the same).
Default (no arguments): the scene of map.yaml = map, robot and platform; `bin:` / `basket:` / `objects:` of it only when switched on there.
`pallets:` (scene: of map.yaml) stacks crates (assets/objects/usd/SIDE_PLATE_BOX.usd) on the pallets of the map, `layers` layers of a 2x3 or 3x2 grid
each, `parts_per_crate` random parts in every crate (binsim/pallets.py). The parts of the top layer move (/World/pile/..), the rest only fills the crates.
`machines:` (scene: of map.yaml) puts articulated machines (assets/machine/<name>/urdf/<name>.urdf, e.g. the laser marker) at pos_m / rpy_deg (the base_link) as
/World/machines/<name>; `replaces:` switches off the static copy of it in the map. Their joints follow /<name>/joint_command (sensor_msgs/JointState, name + position
in rad / m, a few joints at a time are fine) at `speed` x the URDF velocity limit and are published on /<name>/joint_states.
Every /target_pose (geometry_msgs/Pose, the TCP = the centre of the part to pick) makes the part of the pile there (within 4 cm) as exact as a clicked one: CLICK_HULLS
convex hulls and the stiff contact settings, so the gripper does not sink into its bore wall. Only one part has them at a time.
`physics_hz` (default 60) is the physics step rate of the scene (the dataset generator uses 200: falling parts sink into each other less).
Drives use `stiffness` / `damping` and the URDF <limit effort> as force limit (the real joint torque / gripper force).
The robot starts at joint_pos and follows commands at `speed` x the URDF joint velocity limits (targets are moved
towards the command a step per frame, no jumps):
  subscribes  <controller>/joint_trajectory          trajectory_msgs/JointTrajectory (last point)
  action      <controller>/follow_joint_trajectory   control_msgs/FollowJointTrajectory (MoveIt "Execute")
  subscribes  <gripper_topic> (with a gripper)       std_msgs/Float64: 0 = closed, 1 = open (in between: partly open)
  publishes   /joint_states                          sensor_msgs/JointState (measured position, velocity)
  subscribes  /wrist_camera/capture (with `wrist_camera:` in the robot yaml)   std_msgs/Empty: one RGB + depth frame of the camera on the robot
              to wrist_camera.output_dir (camera/ layout of gdrnpp's camera_convert_data.py, which turns it into the BOP folder of infer.py).
              wrist_camera.lens = the barrel distortion of the training-data camera: the saved images are the distorted ones, undistort_map.npy next to them
Run it where ROS 2 is sourced. Separate from the dataset pipeline (own stage; touches no dataset config or output).
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
import xml.etree.ElementTree as ET
from collections import deque

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))  # Simulation/isaacsim: configs/, assets/, binsim/, assets_cache/
ASSETS = os.path.join(os.path.dirname(REPO), "assets")
ROBOTS = os.path.join(ASSETS, "robot")        # Simulation/assets/robot/<name>/urdf/<name>.urdf (+ meshes)
MACHINES = os.path.join(ASSETS, "machine")    # Simulation/assets/machine/<name>/urdf/<name>.urdf (+ meshes): articulated machines of the scene
GRIPPERS = os.path.join(ASSETS, "grippper")   # Simulation/assets/grippper/<gripper>/urdf/<gripper>.urdf (the folder is spelled like this on disk)
MAP_FILE = os.path.join(REPO, "configs", "map", "map.yaml")  # `scene:` = objects, bin + pile, basket (the same for every robot) + dataset config
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("options", nargs="*", metavar="robot:=<name> map:=<name>",
                    help="ROS style `robot:=ur5e` (robot name or robot config file; default: `scene.robot` of configs/map/map.yaml; a bare name "
                         "works too, `robot` / `robot:=` alone = none named) and `map:=SIDE_PLATE_MAP` (a USD in ../assets/map/ or a path; "
                         "default: `scene.map` of map.yaml; `map:=none` = no map)")
parser.add_argument("--name", help="robot name (default: name in the config)")
parser.add_argument("--gripper", help="gripper name in ../assets/grippper/<gripper>/urdf/, none = no gripper (default: gripper in the config)")
parser.add_argument("--force-convert", action="store_true", help="convert the URDF again even if the cache is current")
parser.add_argument("--robot-only", action="store_true",
                    help="only the robot and the floor: ignore the `scene:` section of configs/map/map.yaml (no objects, bin, pile, basket)")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.config, args.map = None, None
for _a in args.options:
    if _a.startswith("map:="):
        args.map = _a[len("map:="):]
        if os.path.isfile(os.path.join(REPO, "configs", "map", args.map + ".yaml")) and args.map != "map":
            MAP_FILE, args.map = os.path.join(REPO, "configs", "map", args.map + ".yaml"), None   # a scene of its own (e.g. PUMP_COVER_MAP.yaml, built on map.yaml)
    elif _a.startswith("robot:="):
        args.config = _a[len("robot:="):] or None
    elif _a not in ("robot", "map"):
        args.config = _a                                     # a bare name is a robot


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
    """../assets/robot/<name>/urdf/<name>.urdf, or <name>.urdf in another assets/robot/*/urdf/ folder (several URDFs of one robot share a folder)."""
    own = os.path.join(ROBOTS, name, "urdf", f"{name}.urdf")
    found = sorted(glob.glob(os.path.join(ROBOTS, "*", "urdf", f"{name}.urdf")))
    return own if os.path.isfile(own) or not found else found[0]


def urdf_joints(urdf):
    """(movable joints, mimic joints) of a URDF. Joints: name -> (type, lower, upper, max velocity, max effort; 0 = none in the URDF).
    Mimic: joint -> (leader, multiplier, offset); the converter writes NewtonMimicAPI, which PhysX does not use, so the follower's
    drive target is set from the leader's by the sim."""
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


def load_settings(path, name, gripper):
    s = yaml.safe_load(open(path)) or {}
    s["name"] = name or s.get("name")
    if not s["name"]:
        sys.exit(f"no robot name: set `name:` in {path} or pass --name")
    g = s.get("gripper") if gripper is None else gripper
    s["gripper"] = None if g is None or str(g).lower() in ("", "none", "null") else str(g)
    urdfs = [find_urdf(s["name"])]
    if s["gripper"]:
        urdfs.append(os.path.join(GRIPPERS, s["gripper"], "urdf", f"{s['gripper']}.urdf"))
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
    s["joints"], s["mimic"] = urdf_joints(s["urdf"])
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
    choice = args.map if args.map is not None else scene.get("map")   # an environment made in the GUI: a USD (command line wins)
    s["map"] = None if args.robot_only or not choice or str(choice).lower() in ("none", "null") else map_file(str(choice))
    s["platform"] = pf = scene.get("platform") or None   # pedestal: a static box in the world
    if pf and not pf.get("size_m"):
        print(f"[WARN] platform skipped: platform of {MAP_FILE} (scene:) needs size_m [x, y, z]", flush=True)
        s["platform"] = None
    if s["platform"]:                              # the robot stands on the platform when its x y is on it: pos_m z counts from the top
        pos = [float(v) for v in (s.get("pos_m") or [0, 0, 0])]
        top = platform_top(s["platform"], pos)
        if top:
            s["pos_m"] = [pos[0], pos[1], pos[2] + top]
            print(f"[INFO] the robot stands on the platform: z = {pos[2]:g} + {top:g} m", flush=True)
    s["pallets"] = pc = scene.get("pallets") or None   # crates of parts stacked on the pallets of the map (binsim/pallets.py)
    if pc:
        try:
            crate = dict(pc["crate"])
            usd = None
            if crate.get("config"):                  # the crate is a box of configs/objects/<config>.yaml (STL, like the bin): size, inner size, floor from there
                bf = os.path.join(REPO, "configs", "objects", f"{crate['config']}.yaml")
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
            missing = [k for k in ("size_m", "inner_size_m", "floor_z_m") if k not in crate] + [k for k in ("items", "part") if not pc.get(k)]
            if missing:
                raise ValueError(f"pallets needs {', '.join(missing)}")
            f = os.path.join(REPO, "configs", "objects", f"{pc['part']}.yaml")
            if not os.path.isfile(f):
                raise FileNotFoundError(f"part config not found: {f}")
            seed = pc.get("seed", "random")
            pc["seed"] = int.from_bytes(os.urandom(4), "big") if seed is None or str(seed).lower() == "random" else int(seed)
            pc["crate"] = {**crate, "usd": usd}
            pc["part_cfg"] = _load_yaml_tree(f)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] pallets skipped: {type(e).__name__}: {e}", flush=True)
            s["pallets"] = None
    s["basket"] = k = scene.get("basket") or None  # open box to release the part into: a procedural bin in the world
    if k and any(key not in k for key in ("pos_m", "size_m", "wall_thickness_m")):
        print(f"[WARN] basket skipped: basket of {MAP_FILE} (scene:) needs pos_m, size_m and wall_thickness_m", flush=True)
        s["basket"] = None
    s["machines"] = []                             # articulated machines standing in the map (assets/machine/<name>/urdf/<name>.urdf)
    for m in scene.get("machines") or []:
        urdf = os.path.join(MACHINES, str(m.get("name")), "urdf", f"{m.get('name')}.urdf")
        if not os.path.isfile(urdf):
            print(f"[WARN] machine {m.get('name')} skipped: URDF not found: {urdf}", flush=True)
            continue
        miss = [f for f in (x.get("filename", "") for x in ET.parse(urdf).getroot().iter("mesh")) if not find_mesh(f, os.path.dirname(urdf))]
        if miss:
            print(f"[WARN] machine {m['name']} skipped: {len(miss)} mesh(es) of {urdf} not found: {miss[:3]}", flush=True)
            continue
        joints, mimic = urdf_joints(urdf)
        s["machines"].append({**m, "model": f"machine_{m['name']}", "urdf": urdf, "joints": joints, "mimic": mimic})
    return s


def platform_top(pf, pos):
    """z of the top of the platform if pos (x, y) is on its footprint (yaw only), else 0: a robot there stands on it"""
    px, py, pz = (float(v) for v in pf.get("pos_m", [0, 0, 0]))
    sx, sy, sz = (float(v) for v in pf["size_m"])
    yaw = math.radians(float((pf.get("rpy_deg") or [0, 0, 0])[2]))
    dx, dy = pos[0] - px, pos[1] - py
    u, v = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    return pz + sz if abs(u) <= sx / 2 and abs(v) <= sy / 2 else 0.0


def map_file(choice):
    """USD of the map: a file path, or a name = ../assets/map/<name>.usd (.usda, .usdc)"""
    if os.path.isfile(choice):
        return os.path.abspath(choice)
    d = os.path.join(ASSETS, "map")
    for ext in ("", ".usd", ".usda", ".usdc"):
        if os.path.isfile(os.path.join(d, choice + ext)):
            return os.path.join(d, choice + ext)
    names = sorted({os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith((".usd", ".usda", ".usdc"))}) if os.path.isdir(d) else []
    sys.exit(f"map not found: {choice}; maps in {d}: {', '.join(names) or '-'}")


def robot_config(choice):
    """Config file of a robot: a file path, or a robot name = configs/robots/<name>.yaml or the file whose `name:` is it."""
    if os.path.isfile(choice):
        return choice
    d = os.path.join(REPO, "configs", "robots")
    configs = {f: yaml.safe_load(open(os.path.join(d, f))) or {} for f in sorted(os.listdir(d))
               if f.endswith(".yaml")}
    for f, c in configs.items():
        if f == f"{choice}.yaml" or c.get("name") == choice:
            return os.path.join(d, f)
    sys.exit(f"robot config not found: {choice}; robots: " + ", ".join(sorted({c.get("name", f[:-5]) for f, c in configs.items()})))


if args.config is None:  # nothing on the command line: the robot of the scene in configs/map/map.yaml
    selected = None
    for f in (MAP_FILE, os.path.join(REPO, "configs", "map", "map.yaml")):    # a scene yaml built on map.yaml (defaults:) may leave the robot to it
        selected = selected or ((yaml.safe_load(open(f)) or {}).get("scene") or {}).get("robot")
    if not selected:
        sys.exit(f"no robot selected: set `scene.robot: <name>` in {MAP_FILE} or pass a robot name")
    args.config = str(selected)
args.config = robot_config(args.config)
print(f"[INFO] robot config: {os.path.relpath(args.config, REPO)}", flush=True)
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
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from geometry_msgs.msg import Pose  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import Bool, Empty, Float64, Int32, String  # noqa: E402
from trajectory_msgs.msg import JointTrajectory  # noqa: E402

from binsim.camera import Intrinsics  # noqa: E402
from binsim.lens import LensWarp, lens_intrinsics, pixel_rays, render_intrinsics  # noqa: E402
from binsim.transforms import make_T, mat_to_quat, quat_to_mat, rpy_to_mat  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(REPO), "control"))
import compute_grasp_tcp_pose as grasp_tcp  # noqa: E402  object pose -> the pose of the TCP that grips it (control/, also a script of its own)

APP_HZ = 60.0  # physics steps per app.update() in the default physics scene (as in generate_isaacsim.py)
ROBOT_PATH = "/World/robot"
# Nothing here waits for the clock: every limit counts physics frames (1 app update = 1/60 s of simulated time), so a slow sim
# gives the same result as a fast one. A trajectory ends with the result of the action (MoveIt takes no new goal until then):
# succeeded = the joints are within ARRIVE_TOL of the goal and no joint moved more than SETTLE_MOVE during SETTLE_FRAMES;
# aborted = they stopped short of it (no joint moved more than STALL_MOVE for STALL_FRAMES, e.g. blocked by the bin).
ARRIVE_TOL, STALL_MOVE, STALL_FRAMES, SETTLE_MOVE, SETTLE_FRAMES = 0.01, 0.01, 60, 0.0005, 18   # rad, rad, frames, rad, frames
# A gripper command is answered on <gripper topic>/result (std_msgs/String): "reached" = the joint is within GRIP_TOL (m) of the
# commanded position, "blocked" = it stopped short of it (no more than GRIP_TOL of travel during GRIP_WINDOW frames, counted
# after GRIP_GRACE frames of start-up), e.g. the jaws pressing against the part.
GRIP_TOL, GRIP_GRACE, GRIP_WINDOW = 0.0002, 18, 18   # m, frames, frames
# The part that is clicked in the viewport gets this many convex hulls (every other part keeps the max_convex_hulls of its object yaml, 32):
# only the part the fingers go into needs the exact shape of its bore, and ~900 parts with hundreds of hulls overflow the PhysX GPU buffers.
# Drives of the machines (scene.machines): a small machine, not an arm. USD units: N*m/deg and N*m*s/deg (angular), N/m and N*s/m (linear).
# The damping times the URDF speed limit stays below the URDF effort (flip: 5 x 34 deg/s = 170 < 300 N*m, jaws: 100 x 0.03 m/s = 3 < 100 N).
MACHINE_STIFFNESS = {"angular": 100.0, "linear": 5000.0}
MACHINE_DAMPING = {"angular": 5.0, "linear": 100.0}
# The part that is clicked is published as std_msgs/Bool on PART_TOPIC: True = turned over (the word `turn_over` of the line printed for ee_6d.py), False = standard.
# Latched (transient local), so a control node that starts later still gets the last click. control/robot_sequence.py takes the place pose of the part from it.
PART_TOPIC = "/part_turned_over"
TARGET_RADIUS = 0.04   # m: the part of the pile this close to a /target_pose is the one the robot goes to pick
CLICK_HULLS = 1000
HULLS_ATTR = "physxConvexDecompositionCollision:maxConvexHulls"
# The clicked part (the one the fingers push against) also gets the stiff contact settings, so a finger pushing hard on the thin
# hole wall does not sink through it. Every other part keeps the PhysX defaults: with ~120 free parts these cost far too much.
CLICK_PHYSX = {"CreateSolverPositionIterationCountAttr": (16, 4), "CreateSolverVelocityIterationCountAttr": (2, 1),
               "CreateEnableCCDAttr": (True, False), "CreateMaxDepenetrationVelocityAttr": (0.5, 3.0)}   # (clicked, default)


class RobotBridge(Node):
    """Joint targets from ROS (topic or MoveIt action); the sim loop reads them and reports the measured state."""

    def __init__(self, names, q0, controller, mimic, gripper, gripper_topic, limits):
        super().__init__("robot_sim")
        self.names = list(names)
        self._lo = [lo for lo, _ in limits]   # URDF position limits per joint: the measured angle is reported inside them (see report)
        self._hi = [hi for _, hi in limits]
        self._idx = {n: i for i, n in enumerate(self.names)}
        self._mimic = {n: m for n, m in mimic.items() if n in self._idx and m[0] in self._idx}
        self._gripper = {n: lim for n, lim in gripper.items() if n in self._idx}  # joint -> (closed, open) position
        if self._gripper:
            self.create_subscription(Float64, gripper_topic, self._on_gripper, 10)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)   # notified by report() once per physics frame
        self.frame = 0                                  # physics frames reported so far
        self._targets = list(q0)
        self._current = list(q0)
        self._wps = deque()                             # waypoints of the running trajectory that the command has not passed yet
        self._goal = None                               # the running trajectory (see _execute / _track_arm)
        self._grip = None                               # the gripper command that has not been answered yet (see _track_gripper)
        self._grip_pub = self.create_publisher(String, f"{gripper_topic}/result", 10) if self._gripper else None
        self._part_pub = self.create_publisher(Bool, PART_TOPIC, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(JointTrajectory, f"{controller}/joint_trajectory", self._on_traj, 10)
        self._action = ActionServer(
            self, FollowJointTrajectory, f"{controller}/follow_joint_trajectory",
            execute_callback=self._execute, goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=ReentrantCallbackGroup())
        self._state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.capture_requested = threading.Event()   # set by the topic, handled in the sim loop (Kit calls only on the main thread)
        self.create_subscription(Empty, "/wrist_camera/capture", lambda _: self.capture_requested.set(), 10)
        self.targets_seen = deque()                  # (x, y, z) of every /target_pose: the part there is handled like a clicked one (see refine_target in the sim loop)
        self.create_subscription(Pose, "/target_pose", lambda m: self.targets_seen.append((m.position.x, m.position.y, m.position.z)), 10)

    def _set(self, names, positions):
        with self._lock:
            self._apply(names, positions)

    def _apply(self, names, positions):   # the caller holds the lock
        for n, p in zip(names, positions):
            if n in self._idx:
                self._targets[self._idx[n]] = float(p)
        for n, (lead, k, off) in self._mimic.items():
            self._targets[self._idx[n]] = self._targets[self._idx[lead]] * k + off

    def _on_traj(self, msg):
        if msg.points:
            self._set(msg.joint_names, msg.points[-1].positions)

    def publish_turned_over(self, flipped):
        """the state of the clicked part: True = turned over, False = standard (PART_TOPIC)"""
        self._part_pub.publish(Bool(data=bool(flipped)))

    def _on_gripper(self, msg):
        v = min(max(float(msg.data), 0.0), 1.0)
        joint = next(iter(self._gripper))                                # the first joint tells when the gripper is done
        lo, hi = self._gripper[joint]
        with self._lock:
            self._apply(list(self._gripper), [lo + v * (hi - lo) for lo, hi in self._gripper.values()])
            self._grip = {"idx": self._idx[joint], "target": lo + v * (hi - lo), "frame": self.frame, "window": deque(maxlen=GRIP_WINDOW)}
        self.get_logger().info(f"gripper {v:.2f} ({'closed' if v == 0 else 'open' if v == 1 else 'partly open'})")

    def _execute(self, goal):
        """The waypoints are followed one after the other by the sim loop (see targets: the path shape, not a jump to the goal);
        the goal ends with the result of the trajectory joints: arrived and at rest -> succeeded, stopped short of the goal -> aborted."""
        traj = goal.request.trajectory
        if not traj.points:
            goal.abort()
            return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.INVALID_GOAL)
        idx = [self._idx[n] for n in traj.joint_names if n in self._idx]   # only the joints of this trajectory (not the gripper)
        mine = {"idx": idx, "final": [p for n, p in zip(traj.joint_names, traj.points[-1].positions) if n in self._idx],
                "window": deque(maxlen=SETTLE_FRAMES), "ref": None, "ref_frame": 0, "outcome": None}
        with self._cond:
            self._wps = deque((list(traj.joint_names), list(pt.positions)) for pt in traj.points)
            self._goal = mine
            while mine["outcome"] is None and self._goal is mine:
                if goal.is_cancel_requested:
                    break
                self._cond.wait(1.0)                       # woken by every frame; the timeout only re-checks for a cancel
            if self._goal is mine:
                self._goal = None
                self._wps.clear()
        if mine["outcome"] is None:
            if goal.is_cancel_requested:
                goal.canceled()
                return FollowJointTrajectory.Result()
            goal.abort()                                   # a newer trajectory took over
            return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.INVALID_GOAL)
        if mine["outcome"] == "stalled":
            k = max(range(len(idx)), key=lambda i: mine["error"][i])
            self.get_logger().warn(f"arm stopped {math.degrees(mine['error'][k]):.2f} deg short of the goal at {self.names[idx[k]]} (blocked?) -> trajectory aborted")
            goal.abort()
            return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED)
        goal.succeed()
        return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.SUCCESSFUL)

    def _track_arm(self, q):   # once per frame (report), the caller holds the lock
        g = self._goal
        if g is None or g["outcome"] is not None or len(self._wps) > 1:    # only the last waypoint is waited for
            return
        pos = [q[i] for i in g["idx"]]
        error = [abs(p - f) for p, f in zip(pos, g["final"])]
        g["window"].append(pos)
        if g["ref"] is None or max(abs(a - b) for a, b in zip(pos, g["ref"])) > STALL_MOVE:
            g["ref"], g["ref_frame"] = pos, self.frame
        at_rest = len(g["window"]) == SETTLE_FRAMES and max(max(c) - min(c) for c in zip(*g["window"])) < SETTLE_MOVE
        if max(error) < ARRIVE_TOL and (at_rest or self.frame - g["ref_frame"] > STALL_FRAMES):
            g["outcome"] = "arrived"
        elif self.frame - g["ref_frame"] > STALL_FRAMES:
            g["outcome"], g["error"] = "stalled", error

    def _track_gripper(self, q):   # once per frame (report), the caller holds the lock
        g = self._grip
        if g is None:
            return
        pos = q[g["idx"]]
        g["window"].append(pos)
        if abs(pos - g["target"]) < GRIP_TOL:
            outcome = "reached"
        elif (self.frame - g["frame"] > GRIP_GRACE + GRIP_WINDOW and len(g["window"]) == GRIP_WINDOW
              and max(g["window"]) - min(g["window"]) < GRIP_TOL):
            outcome = "blocked"
        else:
            return
        self._grip = None
        self.get_logger().info(f"gripper {outcome}: position {pos:.4f} m (command {g['target']:.4f} m)")
        self._grip_pub.publish(String(data=outcome))

    def targets(self, cmd=None, step=0.0):
        """The joint targets. cmd = the rate-limited command of the sim loop, step = its largest move per frame: the waypoints of a
        running trajectory that the command can reach within this frame are passed, so the command follows the path at full speed"""
        with self._lock:
            while self._wps:
                names, pos = self._wps[0]
                self._apply(names, pos)
                if cmd is None or len(self._wps) == 1:
                    break
                joints = [self._idx[n] for n in names if n in self._idx]
                if (np.abs(np.array(self._targets)[joints] - np.array(cmd)[joints]) > np.array(step)[joints]).any():
                    break
                self._wps.popleft()
            return list(self._targets)

    def report(self, q, v):
        # PhysX holds a joint at its limit with ~1e-6 rad of play, and MoveIt refuses a start state outside the URDF bounds
        # (START_STATE_INVALID, 'Joint J1 ... outside bounds'): a joint resting on its limit is reported exactly on it
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
    """/World/pile/side_plate_078/geometry -> /World/pile/side_plate_078 (a part of the pile or a single object), or
    /World/pallets/pallet_1/L0_C0_side_plate_01/geometry -> /World/pallets/pallet_1/L0_C0_side_plate_01 (a static part in a crate), else None"""
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
    """True if the URDF's root link is empty (e.g. `world`): the importer then fixes the robot to the world itself."""
    root = ET.parse(urdf).getroot()
    children = {j.find("child").get("link") for j in root.findall("joint")}
    for ln in root.findall("link"):
        if ln.get("name") not in children:
            return all(ln.find(t) is None for t in ("visual", "collision", "inertial"))
    return False


def meshes_newer(urdf, usd_dir):
    """True if a mesh file of the URDF changed after the cached USD was made (the converter only looks at the .urdf, not at its meshes)"""
    made = max((os.path.getmtime(os.path.join(d, f)) for d, _, fs in os.walk(usd_dir) for f in fs if f.endswith((".usd", ".usda"))), default=0.0)
    meshes = (os.path.normpath(os.path.join(os.path.dirname(urdf), m.get("filename"))) for m in ET.parse(urdf).getroot().iter("mesh"))
    return any(os.path.isfile(p) and os.path.getmtime(p) > made for p in meshes)


def robot_usd(s, force):
    """URDF -> USD, cached in assets_cache/robot_<model>/ (redone when the .urdf file or one of its meshes changes)."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    usd_dir = os.path.join(REPO, "assets_cache", f"robot_{s['model']}")
    if not force and meshes_newer(s["urdf"], usd_dir):
        print("[INFO] a mesh of the robot URDF changed: converting the robot to USD again", flush=True)
        force = True
    conv = UrdfConverter(UrdfConverterCfg(
        asset_path=s["urdf"], usd_dir=usd_dir,
        fix_base=not ghost_root(s["urdf"]), merge_fixed_joints=False, force_usd_conversion=force,
        collision_from_visuals="collision_type" in s, collision_type=str(s.get("collision_type", "Convex Hull"))))     # Convex Decomposition = the exact shape of the fingers
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


def plan_pallet_crates(pc, cache_dir):
    """Crates and parts of the pallets (binsim.pallets.plan_pallets): world poses + the USD of the part"""
    from binsim.assets import prepare_bin, prepare_part
    from binsim.pallets import plan_pallets
    if pc["crate"].get("bin_cfg"):                   # a box made of an STL: its USD like the bin's
        pc["crate"]["usd"] = prepare_bin(pc["crate"]["bin_cfg"], cache_dir).usd_path
    pa = prepare_part(pc["part_cfg"], cache_dir)
    plan = plan_pallets({k: ({kk: vv for kk, vv in v.items() if kk != "bin_cfg"} if k == "crate" else v) for k, v in pc.items() if k not in ("part", "part_cfg", "seed")}, pa.extents,
                        pc["part_cfg"].get("layer_rpy_deg"), np.random.default_rng(pc["seed"]))
    plan["part_usd"] = pa.usd_path
    return plan


def wrist_lens(cfg):
    """(intrinsics of the image, intrinsics of the pinhole render it is made from, LensWarp or None) of wrist_camera. With wrist_camera.lens
    (the lens of the training-data camera, configs/cameras/phoxi_1952x1460.yaml: barrel distortion, the "convex" look) the renderer, an ideal
    pinhole camera, draws a bit more than the image and binsim/lens.py bends it into the image the real lens takes (like generate_isaacsim.py)."""
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


def make_wrist_camera(stage, cfg):
    """wrist_camera: a USD camera on the link `cfg.link` of the robot (an OpenCV optical frame: x right, y down, z forward), with the
    intrinsics of cfg (resolution, fx, fy; centred principal point; the size of the pinhole render when there is a lens, see wrist_lens).
    Returns the camera prim path, or None when the link is missing."""
    link = [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)) if p.GetName() == str(cfg["link"])]
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
    UsdGeom.Xformable(cam).AddOrientOp().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))   # a USD camera looks along -z with y up: turned 180 deg about x it is the optical frame
    return cam.GetPath().pathString


def urdf_link_pose(urdf, link, q, base_T):
    """World pose (4 x 4, m) of a URDF link: base_T (the pose of the URDF root in the world) times the chain of joint origins down to
    `link`, revolute / prismatic joints at the positions q (dict joint name -> rad / m). Read from the URDF, not from the stage: what
    the camera sees is the measured state of the joints."""
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
    """One RGB + depth frame of the wrist camera in the `camera/` layout of gdrnpp's camera_convert_data.py (rgb/<n>.png, depth/<n>.png,
    pose/<n>.json = T_world_cam, the pose of the optical frame in the Isaac world (m), camera.json), which its --to-bop turns into the BOP
    folder infer.py reads. depth: uint16 in depth_scale mm, 0 = no depth (also outside
    depth_range_m, like the PhoXi). With wrist_camera.lens the images are the distorted ones of the lens, like the raw captures of the real
    PhoXi, and undistort_map.npy (the pinhole pixel of every image pixel, named in camera.json) is saved for --to-bop to undistort them.
    The render product only exists for the capture, so the sim does not render the camera every frame."""
    import omni.replicator.core as rep
    from PIL import Image
    sensor, render, warp = wrist_lens(cfg)
    w, h = sensor.width, sensor.height
    out = os.path.expanduser(str(cfg["output_dir"]))
    os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out, "depth"), exist_ok=True)
    os.makedirs(os.path.join(out, "pose"), exist_ok=True)
    cam_file = os.path.join(out, "camera.json")
    if os.path.isfile(cam_file) and os.listdir(os.path.join(out, "rgb")):     # one camera.json (and one undistort map) serves all frames of the folder
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
        for _ in range(60):                                   # the first frames of a new render product can be empty
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
    if warp is not None:                                   # the pinhole render -> the image of the lens (depth = z along the axis: nearest, no mixing)
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
    if warp is not None:                                   # K above = the pinhole the images are undistorted onto
        rays = pixel_rays(sensor)
        np.save(map_file, np.stack([rays[..., 0] * sensor.fx + sensor.cx, rays[..., 1] * sensor.fy + sensor.cy], -1).astype(np.float32))
        info.update(undistort_map="undistort_map.npy", dist=list(sensor.dist))
    with open(os.path.join(out, "camera.json"), "w") as f:
        json.dump(info, f, indent=1)
    return os.path.join(out, "rgb", f"{n:06d}.png"), float((depth > 0).mean())


def set_collision_decomposition(stage, patterns, root=ROBOT_PATH):
    """collision_decomposition: link-name regexes of the robot yaml (or of a machine, below its prim `root`). The colliders under these links (the URDF collision meshes, which the converter
    makes convexHull) become convexDecomposition, so the fingers keep their exact shape. The instances under them are made editable first (only on
    this stage; the cached USD is unchanged). Returns the number of colliders."""
    rx = [re.compile(str(q)) for q in patterns]
    roots = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if any(r.fullmatch(p.GetName()) for r in rx)]
    while True:                                   # instances can be nested (instanced mesh -> instanced material)
        inst = [p for r in roots for p in Usd.PrimRange(r) if p.IsInstance()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)
    done = set()                                  # the link roots are nested (gripper_mount contains gripper_motor, ...): every collider once
    for r in roots:
        for p in Usd.PrimRange(r):
            if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI) and p.GetPath() not in done:
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexDecomposition")
                p.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
                done.add(p.GetPath())
    return len(done)


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


class MachineBridge(Node):
    """Joint commands of one machine of the scene (scene.machines of map.yaml) from ROS; the sim loop reads them and reports the measured state."""

    def __init__(self, name, names, q0, limits):
        super().__init__(f"{name}_bridge")
        self.names = list(names)
        self._idx = {n: i for i, n in enumerate(self.names)}
        self._limits = list(limits)
        self._targets = [float(x) for x in q0]
        self._lock = threading.Lock()
        self._warned = set()
        self.create_subscription(JointState, f"/{name}/joint_command", self._on_command, 10)
        self._pub = self.create_publisher(JointState, f"/{name}/joint_states", 10)

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


class CarrierBridge(Node):
    """The carriers of a machine's belt (machines: carriers: of map.yaml) move in steps of one pitch, stop-and-go, and run round: what leaves at the -y end
    of the loop comes back at the +y end. One topic controls it: std_msgs/Int32 on <topic>: n > 0 = advance n pitches (added to what is still to do),
    0 = stop where it is. <topic>/moving (std_msgs/Bool, latched) is True while the carriers are moving. The sim loop calls step() once per frame."""

    def __init__(self, name, topic, joints, slots, window, speed):
        super().__init__(f"{name}_carriers")
        self.joints, self.slots, self.lo, self.hi = list(joints), [float(y) for y in slots], float(window[0]), float(window[1])
        self.loop = self.hi - self.lo                       # length of the loop = number of carriers x pitch
        self.pitch = self.loop / len(self.slots)
        self.speed = float(speed)                           # m/s
        self.s = self.goal = 0.0                            # travelled (m, towards -y) and where it is going; multiples of the pitch = every carrier at a slot
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
        """joint positions (m, relative to their slots) of the carriers at the travelled distance s: every carrier stays inside the loop window"""
        with self._lock:
            return [self.lo + (y - self.s - self.lo) % self.loop - y for y in self.slots]

    def step(self, dt):
        """advance by dt s of simulated time towards the goal; returns the joint positions"""
        with self._lock:
            if self.s < self.goal:
                self.s = min(self.goal, self.s + self.speed * dt)
            moving = self.s < self.goal
        if moving != self.moving:
            self.moving = moving
            self._pub.publish(Bool(data=moving))
        return self.positions()


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
    psa.CreateGpuMaxRigidPatchCountAttr().Set(2 ** 18)    # default 81920: the piles need ~90k contact patches while they settle ("Patch buffer overflow", contacts are dropped)
    psa.CreateGpuMaxRigidContactCountAttr().Set(2 ** 21)  # default 524288 ("Contact buffer overflow ... at least 621121" crashed the GPU once)
    psa.CreateGpuCollisionStackSizeAttr().Set(2 ** 29)   # the default 64 MB overflows with the pallets (~720 free parts, ~300 MB seen): PhysX drops contacts then
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
        if usd is not None:
            prim.GetReferences().AddReference(usd)
        x = UsdGeom.Xformable(prim)
        x.ClearXformOpOrder()
        x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
        q = mat_to_quat(rpy_to_mat(rpy) if R is None else R)
        x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))

    place(ROBOT_PATH, usd_path, s.get("pos_m", [0, 0, 0]), s.get("rpy_deg", [0, 0, 0]))
    for m in s.get("machines") or []:
        place(f"/World/machines/{m['name']}", m["usd"], m.get("pos_m", [0, 0, 0]), m.get("rpy_deg", [0, 0, 0]))
    for i, o in enumerate(s["objects"]):
        place(f"/World/objects/{o['part']}_{i}", o["usd"], o.get("pos_m", [0, 0, 0]), o.get("rpy_deg", [0, 0, 0]))
    if s.get("bin"):                              # bin and pile: separate prims, world poses (not under the bin)
        place("/World/bin", s["bin"]["usd"], s["bin"]["pos_m"], s["bin"]["rpy_deg"])
        for i, (usd, pos, R) in enumerate(s["bin"]["pile"]):
            place(f"/World/pile/{s['bin']['part']['name']}_{i:03d}", usd, pos, R=R)
    if s.get("basket"):                           # basket: world pose, kinematic like the bin
        place("/World/basket", s["basket"]["usd"], s["basket"]["pos_m"], s["basket"].get("rpy_deg", [0, 0, 0]))
    if s.get("pallets"):                          # crates of parts on the pallets: the crate USD keeps its own unit ops, so it hangs below a frame
        pc, pl = s["pallets"], s["pallets"]["plan"]
        pname = pc["part_cfg"]["name"]
        for c in pl["crates"]:
            frame = f"/World/pallets/pallet_{c['pallet']}/L{c['layer']}_C{c['idx']}"
            place(frame, None, c["pos"], R=c["R"])
            crate = stage.DefinePrim(frame + "/crate")
            crate.GetReferences().AddReference(pc["crate"]["usd"])
            crate.RemoveAPI(UsdPhysics.RigidBodyAPI)       # the crate is a rigid body in its file: here it stands still (a static triangle mesh)
        for p in pl["parts"]:                       # the parts of the top layers move (/World/pile: clickable like the pile), the ones below only fill the crates
            path = (f"/World/pile/{pname}_p{p['pallet']}_L{p['layer']}_C{p['crate']}_{p['j']:02d}" if p["dynamic"]
                    else f"/World/pallets/pallet_{p['pallet']}/L{p['layer']}_C{p['crate']}_{pname}_{p['j']:02d}")
            place(path, pl["part_usd"], p["pos"], R=p["R"])
            if not p["dynamic"]:
                stage.GetPrimAtPath(path).RemoveAPI(UsdPhysics.RigidBodyAPI)
    if s.get("map"):                              # environment made in the GUI: its top-level prims (not the render settings / cameras / lights its stage saved)
        layer = Sdf.Layer.FindOrOpen(s["map"])
        UsdGeom.Xform.Define(stage, "/World/map")
        for spec in layer.rootPrims:
            if spec.name.startswith(("Render", "Omni")) or spec.typeName in ("Camera",) or spec.typeName.endswith("Light"):
                continue                          # what the Kit stage saved around the scene: render settings, viewport cameras, light rig
            stage.DefinePrim(f"/World/map/{spec.name}").GetReferences().AddReference(s["map"], f"/{spec.name}")
        for m in s.get("machines") or []:         # the machine takes the place of the static copy (an STL) of it in the map: that one is switched off
            if m.get("replaces"):
                old = stage.GetPrimAtPath("/World/map/" + str(m["replaces"]).strip("/"))
                if old.IsValid():
                    old.SetActive(False)
                else:
                    print(f"[WARN] machine {m['name']}: {m['replaces']} not found in the map: the static copy stays", flush=True)
    if s.get("platform"):                         # pedestal: static box, pos_m = bottom centre
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
    if s["gripper_links"] and s.get("gripper_material"):
        n = set_gripper_material(stage, set(s["gripper_links"]), s["gripper_material"])
        print(f"[INFO] gripper material {dict(s['gripper_material'])} on {n} materials", flush=True)
    if s.get("collision_decomposition"):
        n = set_collision_decomposition(stage, s["collision_decomposition"])
        print(f"[INFO] collision_decomposition {list(s['collision_decomposition'])}: {n} colliders -> convexDecomposition", flush=True)
    for m in s.get("machines") or []:
        if m.get("collision_decomposition"):
            n = set_collision_decomposition(stage, m["collision_decomposition"], f"/World/machines/{m['name']}")
            print(f"[INFO] machine {m['name']}: collision_decomposition {list(m['collision_decomposition'])}: {n} colliders -> convexDecomposition", flush=True)
    if s.get("wrist_camera"):
        s["wrist_camera_path"] = make_wrist_camera(stage, s["wrist_camera"])
        if s["wrist_camera_path"]:
            print(f"[INFO] wrist camera {s['wrist_camera_path']}: publish std_msgs/Empty on /wrist_camera/capture for one frame -> "
                  f"{s['wrist_camera']['output_dir']}", flush=True)

    # drives + start pose on the joint prims before physics starts (USD units as in the lite6 script: N*m/deg, deg),
    # so the robot is stiff from the first frame and does not jump
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
            if info[4] > 0:  # URDF effort = the drive's force limit (N*m, prismatic N): the gripper jaws push with the real
                drv.CreateMaxForceAttr().Set(float(info[4]))  # gripper force (JEGG-2510 75 N per jaw), the arm with its real
                                                              # torque - without it both push with unlimited force through parts
            PhysxSchema.JointStateAPI.Apply(jp, kind).CreatePositionAttr().Set(start)

    k, d = float(s.get("stiffness", 20000.0)), float(s.get("damping", 1000.0))
    drives(ROBOT_PATH, s["joints"], q0, {"linear": k, "angular": k}, {"linear": d, "angular": d})
    for m in s.get("machines") or []:
        prefix = f"/World/machines/{m['name']}"
        drives(prefix, m["joints"], m["q0"], MACHINE_STIFFNESS, MACHINE_DAMPING)
        for p in Usd.PrimRange(stage.GetPrimAtPath(prefix)):    # the collision boxes of the frame overlap the rotors and jaws: their links must not collide with each other
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(p).CreateEnabledSelfCollisionsAttr().Set(False)


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
        if robot.get("pallets"):
            pc = robot["pallets"]
            try:
                pc["plan"] = plan_pallet_crates(pc, os.path.join(REPO, "assets_cache"))
                pl = pc["plan"]
                n_dyn = sum(p["dynamic"] for p in pl["parts"])
                print(f"[INFO] pallets: {len(pc['items'])} x {int(pc.get('layers', 6))} layers, {len(pl['crates'])} crates, {len(pl['parts'])} x "
                      f"{pc['part_cfg']['name']} ({n_dyn} movable in the top layer, seed {pc['seed']}); grids per layer (nx x ny): "
                      + " | ".join(",".join(f"{a}x{b}" for a, b in g) for g in pl["grids"]), flush=True)
            except Exception as e:  # noqa: BLE001
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
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] basket skipped: {type(e).__name__}: {e}", flush=True)
                robot["basket"] = None
        jp = robot.get("joint_pos") or {}
        q0 = {n: min(max(next((float(v) for k, v in jp.items() if re.fullmatch(k, n)), 0.0), lo), hi)
              for n, (_, lo, hi, _, _) in robot["joints"].items()}
        for n, (lead, k, off) in robot["mimic"].items():
            if lead in q0:
                q0[n] = q0[lead] * k + off
        ready = []
        for m in robot["machines"]:               # machines are optional like the objects: whatever cannot be converted is skipped with a warning
            try:
                m["usd"] = robot_usd(m, args.force_convert)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] machine {m['name']} skipped: {type(e).__name__}: {e}", flush=True)
                continue
            jp0 = m.get("joint_pos") or {}
            m["q0"] = {n: min(max(float(jp0.get(n, 0.0)), lo), hi) for n, (_, lo, hi, _, _) in m["joints"].items()}
            ready.append(m)
            print(f"[INFO] machine {m['name']} at {m.get('pos_m', [0, 0, 0])} m, rpy {m.get('rpy_deg', [0, 0, 0])} deg -> {m['usd']}", flush=True)
        robot["machines"] = ready
        build_stage(usd_path, robot, q0)

        root_path = Articulation.fetch_articulation_root_api_prim_paths(ROBOT_PATH)[0]
        if root_path is None:
            print(f"[ERROR] no articulation root under {ROBOT_PATH} in {usd_path}", flush=True)
            return
        art = Articulation(root_path)
        machines = []                             # articulated machines of the scene: the articulation, its ROS bridge, the rate-limited command
        for m in robot["machines"]:
            mpath = f"/World/machines/{m['name']}"
            mroot = Articulation.fetch_articulation_root_api_prim_paths(mpath)[0]
            if mroot is None:
                print(f"[WARN] machine {m['name']}: no articulation root under {mpath}: its joints cannot be moved", flush=True)
                continue
            machines.append({"m": m, "root": mroot, "art": Articulation(mroot)})
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        update(2)

        names = list(art.dof_names)
        cmd = np.array([q0.get(n, 0.0) for n in names])
        step = np.array([robot["joints"].get(n, (None, 0, 0, 1.0))[3] for n in names]) * float(robot.get("speed", 0.3)) / APP_HZ
        controller = "/" + str(robot.get("controller", f"{robot['name']}_controller")).strip("/")
        gripper = {n: robot["joints"][n][1:3] for n in robot["gripper_joints"]}
        gripper_topic = str(robot.get("gripper_topic") or "/gripper/command")
        limits = [robot["joints"][n][1:3] if n in robot["joints"] else (-math.inf, math.inf) for n in names]
        node = RobotBridge(names, cmd, controller, robot["mimic"], gripper, gripper_topic, limits)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        for mc in machines:
            m = mc["m"]
            mc["names"] = list(mc["art"].dof_names)
            mc["cmd"] = np.array([m["q0"].get(n, 0.0) for n in mc["names"]])
            mc["cmd0"] = mc["cmd"].copy()
            mc["step"] = np.array([m["joints"][n][3] for n in mc["names"]]) * float(m.get("speed", 1.0)) / APP_HZ
            mc["node"] = MachineBridge(m["name"], mc["names"], mc["cmd"], [m["joints"][n][1:3] for n in mc["names"]])
            executor.add_node(mc["node"])
            c = m.get("carriers")
            mc["carriers"] = None
            if c:
                bad = [n for n in c["joints"] if n not in mc["names"]]
                if bad:
                    print(f"[WARN] machine {m['name']}: carriers of map.yaml name joints that do not exist ({bad}): the carriers stay where they are", flush=True)
                else:
                    mc["carriers"] = {"node": CarrierBridge(m["name"], c.get("topic") or f"/{m['name']}/carriers", c["joints"], c["slots_m"], c["window_m"], c.get("speed_mps", 0.1)),
                                      "idx": [mc["names"].index(n) for n in c["joints"]]}
                    mc["carriers"]["last"] = mc["carriers"]["node"].positions()
                    executor.add_node(mc["carriers"]["node"])
                    cn = mc["carriers"]["node"]
                    print(f"[INFO] machine {m['name']}: carriers {c['joints']}: std_msgs/Int32 on {c.get('topic') or '/' + m['name'] + '/carriers'} (n > 0 = n pitches of {cn.pitch:.2f} m, "
                          f"0 = stop), {cn.speed:g} m/s, state on .../moving", flush=True)
            print(f"[INFO] machine {m['name']}: /{m['name']}/joint_command (sensor_msgs/JointState: name, position in rad / m) -> {mc['names']}, "
                  f"/{m['name']}/joint_states <- sim", flush=True)
        threading.Thread(target=executor.spin, daemon=True).start()
        print(f"[INFO] robot {robot['name']} ready (gripper: {robot['gripper']}): joints {names}, "
              f"max speed {robot.get('speed', 0.3)} x URDF limits, physics {int(robot.get('physics_hz', 60))} Hz", flush=True)
        print(f"[INFO] ROS 2: {controller}/joint_trajectory, {controller}/follow_joint_trajectory -> sim, "
              f"sim -> /joint_states", flush=True)
        if gripper:
            print(f"[INFO] gripper {robot['gripper']}: {gripper_topic} (std_msgs/Float64) 0 = closed, 1 = open "
                  f"-> {list(gripper)}", flush=True)

        # click a part in the viewport -> its pose is printed as "x y z roll pitch yaw" (and the TCP pose that grips it below), the Translate and Rotate X Y Z of Isaac's Property
        # panel (m, deg; read live from the physics, the panel itself may keep showing the start values). The event only remembers
        # the clicked parts, the pose is read in the loop below
        clicked, views = [], {}
        part_view = [None, None]                 # [RigidPrim over all parts of /World/pile, their paths]: the parts nearest to a /target_pose

        cmd0 = cmd.copy()                       # the start pose, where the arm is again after a restart of the physics

        def rebuild_views(why):
            """Deleting a prim in the GUI (a plate) invalidates the physics tensor views (PhysX: "Simulation view object is invalidated") and
            they can only be made again by playing the timeline anew: the physics restarts from the start state (the arm and the parts that
            moved are back where they began, the deleted prims stay deleted) instead of the app closing"""
            nonlocal art, cmd
            print(f"[WARN] a prim was deleted: the physics views are invalid ({why}); restarting the physics from the start state", flush=True)
            views.clear()
            part_view[0] = None
            timeline.stop()
            update(2)
            timeline.play()
            update(3)
            art = Articulation(root_path)
            cmd = cmd0.copy()
            node._set(names, cmd)
            for mc in machines:
                mc["art"] = Articulation(mc["root"])
                mc["cmd"] = mc["cmd0"].copy()
                if mc["carriers"]:                                            # the physics starts from the start state: the carriers are at their slots again
                    mc["carriers"]["node"].s = mc["carriers"]["node"].goal = 0.0
                    mc["carriers"]["last"] = mc["carriers"]["node"].positions()
            omni.usd.get_context().get_selection().clear_selected_prim_paths()     # so that clicking the same part again fires the event

        def on_stage_event(event):
            if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
                paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
                clicked[:] = sorted({part_root(x) for x in paths} - {None})
                if paths and not clicked:
                    print(f"[INFO] {paths[0]} is not a part that can be picked (only the plates of the top layer / the objects move): no pose printed", flush=True)
        stage_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(on_stage_event, name="robot_sim click")
        refined = {}     # part path -> its original max_convex_hulls, for the part that has CLICK_HULLS now (only one at a time)

        def set_click_physx(prim, stiff):
            """CLICK_PHYSX contact settings on the rigid body of a part: stiff = the clicked part, else the PhysX defaults"""
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            for method, (on, off) in CLICK_PHYSX.items():
                getattr(api, method)().Set(on if stiff else off)

        def refine_collision(path):
            """The clicked part gets CLICK_HULLS convex hulls (PhysX cooks its collider again), the one clicked before goes back to its own number"""
            stage_now = omni.usd.get_context().get_stage()
            for old, n in list(refined.items()):
                if old != path and stage_now.GetPrimAtPath(old).IsValid():
                    stage_now.GetPrimAtPath(f"{old}/geometry").GetAttribute(HULLS_ATTR).Set(n)
                    set_click_physx(stage_now.GetPrimAtPath(old), False)
                    print(f"[INFO] {old}: back to {n} convex hulls", flush=True)
                    refined.pop(old)
            attr = stage_now.GetPrimAtPath(f"{path}/geometry").GetAttribute(HULLS_ATTR)
            if not attr.IsValid() or path in refined:
                return                                                       # not a convex decomposition part, or already fine
            refined[path] = int(attr.Get())
            attr.Set(CLICK_HULLS)
            set_click_physx(stage_now.GetPrimAtPath(path), True)
            print(f"[INFO] {path}: {CLICK_HULLS} convex hulls instead of {refined[path]} (the collider is cooked again, this can take a moment)", flush=True)

        def refine_target(pos):
            """The part the robot is sent to (/target_pose = the TCP = the part's centre) gets what a clicked part gets: the exact collider (CLICK_HULLS) and the
            stiff contact settings (CLICK_PHYSX). Without it the fingers sink into the thin wall of the bore and the part sticks to them"""
            stage_now = omni.usd.get_context().get_stage()
            pile = stage_now.GetPrimAtPath("/World/pile")
            paths = [str(c.GetPath()) for c in pile.GetChildren()] if pile.IsValid() else []
            if not paths:
                return
            if part_view[0] is None or part_view[1] != paths:
                part_view[0], part_view[1] = RigidPrim(paths), paths
            centres = part_view[0].get_world_poses()[0].numpy()
            dist = np.linalg.norm(centres - np.array(pos), axis=1)
            i = int(np.argmin(dist))
            if dist[i] > TARGET_RADIUS:
                print(f"[INFO] target {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}: no part of the pile within {TARGET_RADIUS * 100:.0f} cm (nearest {dist[i] * 100:.1f} cm): nothing refined", flush=True)
                return
            print(f"[INFO] target {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}: the part is {paths[i]} ({dist[i] * 100:.1f} cm away)", flush=True)
            refine_collision(paths[i])

        print("[INFO] click a part in the viewport: its pose is printed as x y z roll pitch yaw (m, deg, like the Property panel), (= the target of a sequence with `target: object`), and below it the TCP pose that grips it "
              "(compute_grasp_tcp_pose.py) and the state of the part (standard | turn_over) for ik_node / a sequence with `target: tcp`, to paste into ee_6d.py", flush=True)

        rebuilds = 0
        while simulation_app.is_running():
            cmd = cmd + np.clip(np.array(node.targets(cmd, step)) - cmd, -step, step)  # rate-limited towards the command
            try:
                art.set_dof_position_targets(cmd)
                for mc in machines:
                    mc["cmd"] = mc["cmd"] + np.clip(mc["node"].targets() - mc["cmd"], -mc["step"], mc["step"])
                    if mc["carriers"]:                                        # the carriers follow the belt, not joint_command
                        cr = mc["carriers"]
                        q = cr["node"].step(1.0 / APP_HZ)
                        jumped = [i for i, (a, b) in zip(cr["idx"], zip(cr["last"], q)) if b - a > cr["node"].loop / 2]   # left at the -y end: placed at the +y end
                        if jumped:
                            mc["art"].set_dof_positions(np.array([[q[cr["idx"].index(i)] for i in jumped]]), dof_indices=jumped)
                            mc["art"].set_dof_velocities(np.zeros((1, len(jumped))), dof_indices=jumped)
                        cr["last"] = q
                        for i, v in zip(cr["idx"], q):
                            mc["cmd"][i] = v
                    mc["art"].set_dof_position_targets(mc["cmd"])
                update(1)
                node.report(art.get_dof_positions().numpy()[0], art.get_dof_velocities().numpy()[0])
                for mc in machines:
                    mc["node"].report(mc["art"].get_dof_positions().numpy()[0], mc["art"].get_dof_velocities().numpy()[0])
                rebuilds = 0
            except Exception as e:  # noqa: BLE001
                rebuilds += 1
                if rebuilds > 3:
                    raise
                rebuild_views(f"{type(e).__name__}: {str(e)[:90]}")
                continue
            while clicked:
                path = clicked.pop(0)
                try:
                    if not omni.usd.get_context().get_stage().GetPrimAtPath(path).IsValid():     # deleted while it was selected
                        views.pop(path, None)
                        continue
                    if not path.startswith("/World/pallets/"):                   # a part that moves: exact collider for the fingers
                        refine_collision(path)
                    if path.startswith("/World/pallets/"):                       # static part: no physics view, the pose is in the stage
                        T = Gf.Transform(UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                        (x, y, z), q = T.GetTranslation(), T.GetRotation().GetQuat()
                        qw, (qx, qy, qz) = q.GetReal(), q.GetImaginary()
                    else:
                        view = views.get(path) or views.setdefault(path, RigidPrim(path))
                        pos, quat = view.get_world_poses()
                        (x, y, z), (qw, qx, qy, qz) = pos.numpy()[0], quat.numpy()[0]
                    roll, pitch, yaw = Gf.Rotation(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz)))).Decompose(
                        Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())  # the Rotate X Y Z of the panel (checked against it)
                    print(f"{x:.3f} {y:.3f} {z:.3f} {roll + 0.0:.1f} {pitch + 0.0:.1f} {yaw + 0.0:.1f}", flush=True)
                    tcp_r, tcp_p, tcp_y, flipped = grasp_tcp.tcp_rpy(roll, pitch, yaw)    # the line to paste into ee_6d.py: same x y z, the TCP angles that grip it, the state of the part
                    print(f"{x:.3f} {y:.3f} {z:.3f} {tcp_r + 0.0:.1f} {tcp_p + 0.0:.1f} {tcp_y + 0.0:.1f} {'turn_over' if flipped else 'standard'}", flush=True)
                    node.publish_turned_over(flipped)
                except Exception as e:
                    print(f"[WARN] pose of {path} not available: {type(e).__name__}: {e}", flush=True)
            while node.targets_seen:                                         # /target_pose: the part the robot goes to
                pos = node.targets_seen.popleft()
                try:
                    refine_target(pos)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] part at the target {pos} not refined: {type(e).__name__}: {e}", flush=True)
            if node.capture_requested.is_set():                              # /wrist_camera/capture
                node.capture_requested.clear()
                try:
                    if not robot.get("wrist_camera_path"):
                        raise RuntimeError("no wrist camera: set `wrist_camera:` in the robot yaml (and the link in the URDF)")
                    q_now = dict(zip(names, (float(v) for v in art.get_dof_positions().numpy()[0])))    # the measured joints = what the camera sees
                    T_base = make_T(rpy_to_mat(robot.get("rpy_deg", [0, 0, 0])), robot.get("pos_m", [0, 0, 0]))
                    f, valid = capture_wrist_camera(robot["wrist_camera"], robot["wrist_camera_path"],
                                                    urdf_link_pose(robot["urdf"], str(robot["wrist_camera"]["link"]), q_now, T_base))
                    print(f"[INFO] wrist camera frame -> {f} ({valid * 100:.0f} % of the pixels have depth)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] wrist camera capture failed: {type(e).__name__}: {e}", flush=True)
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
