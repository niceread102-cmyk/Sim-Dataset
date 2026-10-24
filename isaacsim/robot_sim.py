"""Robot from its URDF in Isaac Sim, controlled over ROS 2, in a map of its own or in an empty map.

    lab robot_sim.py robot:=<name> [map:=<map>] [--name m20_35_18d] [--gripper none] [--force-convert] [--headless]
    lab robot_sim.py map:=<map>                    (the robot of the map: its scene.robot)
    lab robot_sim.py robot:=<name> obj:=<part> [obj:=<part> ...]   (objects of configs/object/<part>.yaml at the pos_m / rpy_deg of their yaml)

The robot (robot:=<name>; a bare name works too, or robot:=configs/robot/<name>.yaml) is chosen by its config file: one yaml per robot in
configs/robot/ (the URDF in ../assets/robot, controller, gripper, start pose, drive gains, speeds, joint limits), the same in every map. A name
is found as configs/robot/<name>.yaml or as the config whose `name:` it is; robot:= wins over the robot of the map.
The map (map:=<map>) is configs/map/<map>.yaml, a map of its own: everything that stands around the robot (`scene:` below) and where the robot
stands in it (scene.robot_pose [x, y, z, roll, pitch, yaw], times pos_m / rpy_deg of the robot yaml, which are 0). Without map:= the robot stands
at the origin of an empty map: the floor only. (configs/map/map.yaml is the config of the dataset generator, not a map.) sim_config.py finds the
files; the launch (robot.launch.py map:=<map>) reads the same map, so start both with the same map:=.

../assets/robot/<name>/urdf/<name>.urdf (the same file the MoveIt bringup uses) is converted to USD on the first
run and cached in assets_cache/robot_<name>/ (converted again when the .urdf changes; --force-convert after changing
only meshes). Meshes: relative to the URDF or package://<pkg>/meshes/... (searched in the URDF folder and every folder
above it), so a URDF straight out of xacro works.
A gripper that is part of the robot URDF (lite6) is driven by <gripper_topic> too: list its joints as `gripper_joints:` in the config.
With a gripper (../assets/grippper/<gripper>/urdf/<gripper>.urdf, same layout) its root link (e.g. flange) is joined to the robot's
link of that name first; the joined URDF and its USD are cached in assets_cache/robot_<name>_<gripper>/. Gripper joints
are driven like the robot's (joint_pos, ROS commands); URDF <mimic> joints follow their leader joint.
The `scene:` of the map:
`objects:` drops rigid parts from configs/object/<part>.yaml (the dataset part configs: mesh, mass, collision, material;
USD cached in assets_cache like the dataset pipeline) at pos_m (part center) / rpy_deg.
`bin:` puts the dataset's bin (the `bin:` + `loading:` sections of the map, e.g. `defaults: [../object/bin]`) at pos_m / rpy_deg in the world and
drops a pile of `pile.count` parts into it with the dataset's planner (`pile.seed: random` = a different random pile on every
start, the seed is printed; a number = always the same pile); bin and parts are separate world-frame prims
(/World/bin, /World/pile/<part>_NNN), so part poses are world coordinates.
`basket:` an open box (floor + 4 walls, kinematic; the dataset's procedural bin, no STL) at pos_m (bottom centre) /
rpy_deg in the world (/World/basket): only an object of the scene (control/robot_sequence.py does not read it).
`map:` loads an environment made in the Isaac Sim GUI: ../assets/map/<name>.usd (or a path; null = none),
every top-level prim of it except what the Kit stage saved around the scene (render settings, viewport cameras, light rig), as /World/map/<prim>, at the
world origin. Its colliders (rigid / collision set in the GUI) are used as they are; `map_collision:` / `map_edits:` change parts of it.
`platform:` a static box (pedestal) at pos_m (bottom centre) / rpy_deg, size_m [x, y, z] (/World/platform).
A robot whose x y is on it stands on top: its z is measured from the top of the platform (robot.launch.py does the same).
`pallets:` stacks crates (a USD, or a box of configs/object/<config>.yaml) on the pallets of the map, `layers` layers of a grid
each, `parts_per_crate` random parts in every crate (binsim/pallets.py; 0 = empty crates, no `part:` needed), `physics:` per pallet and layer.
The parts of the top `dynamic_layers` layers move (/World/pile/..), the rest only fills the crates.
`machines:` puts articulated machines at pos_m / rpy_deg (the base_link) as /World/machines/<name>: configs/machine/<name>.yaml is the machine
itself (URDF of assets/machine/, materials, collision, carriers, speeds, fixtures, hands, part_pose, attach), the map entry says where it stands
(any key of the machine yaml there wins in that map); `replaces:` switches off the static copy of it in the map. Their joints follow /<name>/joint_command
(sensor_msgs/JointState, name + position in rad / m, a few joints at a time are fine) at `speed` / `joint_speeds` x the URDF velocity limit and are
published on /<name>/joint_states.
Every /target_pose (geometry_msgs/Pose, the TCP = the centre of the part to pick) makes the part of the pile there (within 4 cm) as exact as a clicked one: CLICK_HULLS
convex hulls and the stiff contact settings, so the gripper does not sink into its bore wall. Only one part has them at a time.
`physics_hz` (default 60) is the physics step rate of the scene (the dataset generator uses 200: falling parts sink into each other less).
Drives use `stiffness` / `damping` and the URDF <limit effort> as force limit (the real joint torque / gripper force).
The robot starts at joint_pos and follows commands at `speed` x the URDF joint velocity limits (targets are moved
towards the command a step per frame, no jumps); arrive_tolerance_rad / arrive_settle_frames / gripper_grace_frames / gripper_window_frames of
the robot yaml change when a move or a gripper command counts as done (ARRIVE_TOL, SETTLE_FRAMES, GRIP_GRACE, GRIP_WINDOW below):
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

REPO = os.path.dirname(os.path.abspath(__file__))  # Simulation/isaacsim: configs/, binsim/, assets_cache/
sys.path.insert(0, REPO)
import sim_config  # noqa: E402  the maps, robots and machines of configs/

ASSETS = sim_config.ASSETS
MACHINES = sim_config.MACHINE_ASSETS    # Simulation/assets/machine/<name>/urdf/<name>.urdf (+ meshes): articulated machines of the scene
GRIPPERS = sim_config.GRIPPER_ASSETS    # Simulation/assets/grippper/<gripper>/urdf/<gripper>.urdf (the folder is spelled like this on disk)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("options", nargs="*", metavar="robot:=<name> map:=<map>",
                    help="ROS style `robot:=ur5e` (robot name or robot config file; a bare name works too; default: scene.robot of the map) "
                         "and `map:=SIDE_PLATE_MAP` (configs/map/<map>.yaml; without it: only the robot and the floor)")
parser.add_argument("--name", help="robot name (default: name in the config)")
parser.add_argument("--gripper", help="gripper name in ../assets/grippper/<gripper>/urdf/, none = no gripper (default: gripper in the config)")
parser.add_argument("--force-convert", action="store_true", help="convert the URDF again even if the cache is current")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.config, args.map, args.objects = None, None, []
for _a in args.options:
    if _a.startswith("map:="):
        args.map = _a[len("map:="):]
    elif _a.startswith("obj:="):
        args.objects.append(_a[len("obj:="):])              # an object on the command line, on top of the objects of the map
    elif _a.startswith("robot:="):
        args.config = _a[len("robot:="):] or None
    elif _a not in ("robot", "map"):
        args.config = _a                                     # a bare name is a robot
try:
    MAP_FILE = sim_config.map_path(args.map)                 # configs/map/<map>.yaml; None = no map: the robot and the floor
    MAP = sim_config.load_map(args.map)                      # the whole map (scene: and, for a bin, the dataset's bin: / loading:)
except ValueError as e:
    sys.exit(str(e))


def attach_gripper(robot_urdf, gripper_urdf, out):
    """Robot URDF + gripper URDF joined at the gripper's root link (a robot link, e.g. flange) -> out. Mesh paths
    become absolute (the joined file lives in assets_cache); rewritten only when the content changes."""
    robot, grip = ET.parse(robot_urdf).getroot(), ET.parse(gripper_urdf).getroot()
    for root, src in ((robot, robot_urdf), (grip, gripper_urdf)):
        for m in root.iter("mesh"):
            m.set("filename", sim_config.find_mesh(m.get("filename"), os.path.dirname(src)))
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


def missing_meshes(urdf):
    """the mesh references of a URDF whose file is not there"""
    return [m.get("filename") for m in ET.parse(urdf).getroot().iter("mesh") if not sim_config.find_mesh(m.get("filename", ""), os.path.dirname(urdf))]


def any_parts(counts):
    """parts_per_crate: one number or the nested list [pallet][layer][row][col]: is there a part in any crate?"""
    return any(any_parts(c) for c in counts) if isinstance(counts, (list, tuple)) else bool(counts)


def load_settings(path, name, gripper, pose=None):
    s = yaml.safe_load(open(path)) or {}
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
    # settling checks of the arm and of its gripper (module constants, set in main())
    s["sim_consts"] = {k: t(s[key]) for k, key, t in (("ARRIVE_TOL", "arrive_tolerance_rad", float), ("SETTLE_FRAMES", "arrive_settle_frames", int),
                                                        ("GRIP_GRACE", "gripper_grace_frames", int), ("GRIP_WINDOW", "gripper_window_frames", int)) if s.get(key) is not None}
    from binsim.config import _load_yaml_tree  # the same loader as the dataset's part files
    scene = MAP.get("scene") or {}
    s["pos_m"], s["rpy_deg"] = sim_config.robot_pose(s, scene if pose is None else {**scene, "robot_pose": pose})   # where the robot base stands in the map: its pose there x pos_m / rpy_deg of the robot yaml (on the platform)
    objects = []
    for o in list(scene.get("objects") or []) + [{"part": name} for name in args.objects]:   # obj:= of the command line: at the pose of its yaml
        f = os.path.join(sim_config.OBJECTS_DIR, f"{o['part']}.yaml")
        if not os.path.isfile(f):
            print(f"[WARN] object {o['part']} skipped: part config not found: {f}", flush=True)
            continue
        part = _load_yaml_tree(f)
        if not os.path.isfile(part.get("mesh") or ""):
            print(f"[WARN] object {o['part']} skipped: mesh of {f} not found: {part.get('mesh')}", flush=True)
            continue
        objects.append({"pos_m": part.get("pos_m", [0, 0, 0]), "rpy_deg": part.get("rpy_deg", [0, 0, 0]), **o, "cfg": part})   # the map entry wins over the pose of the part yaml
    s["objects"] = objects
    # bin + pile for picking tests: the dataset's bin and loading settings (sections of the map), placed in the world
    b, s["bin"] = scene.get("bin") or None, None
    if b and not (MAP.get("bin") and MAP.get("loading")):
        print(f"[WARN] bin skipped: scene.bin of {MAP_FILE} needs the dataset's `bin:` (e.g. defaults: [../object/bin]) and `loading:` sections in the map", flush=True)
    elif b:
        pile = b.get("pile") or {}
        seed = pile.get("seed", "random")  # "random" / null: a new pile on every start (printed below), a number: always the same pile
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
    choice = scene.get("map")                       # an environment made in the GUI: a USD
    s["map"] = None if not choice or str(choice).lower() in ("none", "null") else map_file(str(choice))
    s["platform"] = pf = scene.get("platform") or None   # pedestal: a static box in the world (the robot on it: sim_config.robot_pose)
    if pf and not pf.get("size_m"):
        print(f"[WARN] platform skipped: platform of {MAP_FILE} (scene:) needs size_m [x, y, z]", flush=True)
        s["platform"] = None
    s["map_edits"] = scene.get("map_edits") or []
    s["map_collision"] = scene.get("map_collision") or []
    s["pallets"] = pc = scene.get("pallets") or None   # crates of parts stacked on the pallets of the map (binsim/pallets.py)
    if pc:
        try:
            from binsim.pallets import DEFAULTS as PALLET_DEFAULTS
            crate = dict(pc["crate"])
            usd = None
            if crate.get("config"):                  # the crate is a box of configs/object/<config>.yaml (STL, like the bin): size, inner size, floor from there
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
                                   for pl_ in (pc.get("physics") or []) if isinstance(pl_, dict) for ly in (pl_.get("layers") or []) if isinstance(ly, dict)}   # physics true / false per pallet and layer (0 = bottom)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] pallets skipped: {type(e).__name__}: {e}", flush=True)
            s["pallets"] = None
    s["basket"] = k = scene.get("basket") or None  # open box to release the part into: a procedural bin in the world
    if k and any(key not in k for key in ("pos_m", "size_m", "wall_thickness_m")):
        print(f"[WARN] basket skipped: basket of {MAP_FILE} (scene:) needs pos_m, size_m and wall_thickness_m", flush=True)
        s["basket"] = None
    s["cameras"] = [dict(c) for c in scene.get("cameras") or []]   # fixed 2D cameras of the map (make_fixed_camera): a picture on /<name>/capture
    s["machines"] = []                             # articulated machines standing in the map: configs/machine/<name>.yaml + where the map puts it
    for entry in scene.get("machines") or []:
        m = sim_config.machine_config(entry)
        urdf = (os.path.join(MACHINES, str(m["urdf"])) if m.get("urdf")      # `urdf:` = a path below assets/machine/ (a folder or file that is not named like the machine)
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
    """USD of the map: a file path, or a name = ../assets/map/<name>.usd (.usda, .usdc)"""
    if os.path.isfile(choice):
        return os.path.abspath(choice)
    d = os.path.join(ASSETS, "map")
    for ext in ("", ".usd", ".usda", ".usdc"):
        if os.path.isfile(os.path.join(d, choice + ext)):
            return os.path.join(d, choice + ext)
    names = sorted({os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith((".usd", ".usda", ".usdc"))}) if os.path.isdir(d) else []
    sys.exit(f"map USD not found: {choice}; USDs in {d}: {', '.join(names) or '-'}")


try:                                               # robot:= on the command line = that robot alone, else the robots of the map (scene.robots: each with its ROS domain di)
    ENTRIES = sim_config.scene_robots(MAP.get("scene") or {}, args.config)
    CONFIGS = [sim_config.robot_config_path(e["name"]) for e in ENTRIES]
except ValueError as e:
    sys.exit(str(e))
print(f"[INFO] map: {os.path.relpath(MAP_FILE, REPO) if MAP_FILE else 'none (the robot and the floor)'}; robots: "
      + ", ".join(f"{os.path.relpath(c, REPO)} (di {e['di'] if e['di'] is not None else 'of the environment'}, pose {e['pose']})" for e, c in zip(ENTRIES, CONFIGS)), flush=True)
robots = []                                        # checked before Isaac Sim starts
for _i, (_e, _c) in enumerate(zip(ENTRIES, CONFIGS)):
    _r = load_settings(os.path.abspath(_c), args.name, args.gripper, _e["pose"])
    _r["di"], _r["prim"] = _e["di"], "/World/robot" if len(ENTRIES) == 1 else f"/World/robot_{_i}"
    robots.append(_r)
robot = robots[0]                                  # the scene of the map (machines, pallets, objects, ...) comes with the first robot
sim_config.write_current([{"name": r["name"], "di": r["di"]} for r in robots],                 # the launch and the tools started without robot:= / map:= follow this
                         os.path.splitext(os.path.basename(MAP_FILE))[0] if MAP_FILE else None)
if importlib.util.find_spec("rclpy") is None:
    sys.exit("rclpy not found: source ROS 2 first (source /opt/ros/jazzy/setup.bash)")
args.experience = args.experience or os.path.expanduser(
    "~/IsaacSim/_build/linux-aarch64/release/apps/isaacsim.exp.full.kit")  # same kit as generate_isaacsim.py
if not args.headless:
    args.visualizer = args.visualizer or ["kit"]
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
import rclpy  # noqa: E402
from control_msgs.action import FollowJointTrajectory  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402
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
# The damping times the highest joint speed (the URDF speed limit x `speed` of the machine, 2: flip 1.2 rad/s = 69 deg/s, jaws 0.06 m/s) stays below the URDF effort
# (flip: 2.5 x 69 deg/s = 172 < 300 N*m, jaws: 100 x 0.06 m/s = 6 < 100 N).
MACHINE_STIFFNESS = {"angular": 100.0, "linear": 5000.0}
MACHINE_DAMPING = {"angular": 2.5, "linear": 100.0}
# The joints of a machine speed up and slow down smoothly (the command is a velocity ramp, not a jump to full speed) and the command of a rotor never runs
# further ahead of the measured angle than MACHINE_LEAD: a rotor that is held (a jaw on the plate, the conveyor) cannot wind up a big error and then shoot off.
MACHINE_ACCEL = {"angular": 3.0, "linear": 0.5}   # rad/s^2, m/s^2
MACHINE_LEAD = 0.08                               # rad
CARRIER_ACCEL = 0.5                               # m/s^2: the belt starts and stops softly, so the plates on the carriers do not slide (friction 0.5 holds up to 4.9)
# The part that is clicked is published as std_msgs/Bool on PART_TOPIC: True = turned over (the word `turn_over` of the line printed for ee_6d.py), False = standard.
# Latched (transient local), so a control node that starts later still gets the last click. control/robot_sequence.py takes the place pose of the part from it.
PART_TOPIC = "/part_turned_over"
TARGET_RADIUS = 0.04   # m: the part of the pile this close to a /target_pose is the one the robot goes to pick
STIFF_KEEP = 2                                    # how many of the parts that still lie where they were refined keep the stiff contact settings (solver 16 / CCD); a part the robot carried away (CARRIED_M) keeps them until it is removed on a jig (attach:)
CORNER_TOL, CORNER_FRAMES = 0.015, 45            # rad, frames: how close the arm has to be to the corner of a path (a joint that sags more than this is let go after CORNER_FRAMES frames)
CLICK_HULLS = 1000
CLICK_VOXELS = 3000000   # voxel resolution of the same collider (PhysX default 500000): the hulls follow the thin wall of the bore closely
HULLS_ATTR = "physxConvexDecompositionCollision:maxConvexHulls"
VOXELS_ATTR, VOXELS_DEFAULT = "physxConvexDecompositionCollision:voxelResolution", 500000
DECOMPOSITION_ATTRS = {"max_hulls": ("maxConvexHulls", "Int"), "voxel_resolution": ("voxelResolution", "Int"), "error_percentage": ("errorPercentage", "Float"),
                       "min_thickness": ("minThickness", "Float"), "hull_vertex_limit": ("hullVertexLimit", "Int"), "shrink_wrap": ("shrinkWrap", "Bool")}   # option -> attribute of PhysxConvexDecompositionCollisionAPI
# The clicked part (the one the fingers push against) also gets the stiff contact settings, so a finger pushing hard on the thin
# hole wall does not sink through it. Every other part keeps the PhysX defaults: with ~120 free parts these cost far too much.
CLICK_KEEP = 10                                   # how many clicked / targeted parts that still lie where they were refined keep the fine collider; the oldest beyond it go back to their own
CARRIED_M = 0.05                                  # m: a refined part this far from where it lay when it was refined was carried by the robot (seat, belt, hands): it keeps the fine collider and the stiff
                                                  # settings until it is removed on a jig (attach:). Taking them back from a plate resting on a seat re-cooked its collider around the pin and threw it off
CLICK_PHYSX = {"CreateSolverPositionIterationCountAttr": (16, 4), "CreateSolverVelocityIterationCountAttr": (2, 1),
               "CreateEnableCCDAttr": (True, False), "CreateMaxDepenetrationVelocityAttr": (0.5, 3.0)}   # (clicked, default)


class RobotBridge(Node):
    """Joint targets from ROS (topic or MoveIt action); the sim loop reads them and reports the measured state."""

    def __init__(self, names, q0, controller, mimic, gripper, gripper_topic, limits, context=None):
        super().__init__("robot_sim", context=context)                 # context = the ROS domain of this robot (di)
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
        self._corner_frames = 0                         # frames the head waypoint (a corner of the path) has been waited for
        self.events = deque(maxlen=40)                  # (time, text) of the last commands of the robot: the BOOM line (boom_step in the sim loop) shows the ones just before a plate is thrown
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
        self.pictures = deque()                      # paths of the pictures of control/vision_pick.py (the sim loop shows the newest, PicturePanel)
        self.create_subscription(String, "/vision_pick/picture", lambda m: self.pictures.append(m.data), QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
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
        self.events.append((time.time(), f"gripper command {v:.2f} ({'close' if v == 0 else 'open' if v == 1 else 'partly open'})"))
        self.get_logger().info(f"gripper {v:.2f} ({'closed' if v == 0 else 'open' if v == 1 else 'partly open'})")

    def _execute(self, goal):
        """The waypoints are followed one after the other by the sim loop (see targets: the path shape, not a jump to the goal);
        the goal ends with the result of the trajectory joints: arrived and at rest -> succeeded, stopped short of the goal -> aborted."""
        traj = goal.request.trajectory
        self.events.append((time.time(), f"trajectory of {len(traj.points)} waypoints"))
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
        self.events.append((time.time(), "trajectory arrived"))
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
                if self._wps[1][1] == pos:                       # a repeated waypoint = a corner (robot_sequence.py, fuse_moves): the ARM has to be there, the command alone cutting the corner would
                    err = np.abs(np.array(self._targets)[joints] - np.array(self._current)[joints])      # make a diagonal move into the bore
                    self._corner_frames += 1
                    if (err > CORNER_TOL).any() and self._corner_frames < CORNER_FRAMES:
                        break
                    self._corner_frames = 0
                    self._wps.popleft()
                    continue
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
    meshes = (sim_config.find_mesh(m.get("filename"), os.path.dirname(urdf)) for m in ET.parse(urdf).getroot().iter("mesh"))
    return any(p and os.path.getmtime(p) > made for p in meshes)


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
    """Crates and parts of the pallets (binsim.pallets.plan_pallets): world poses + the USD of the part (None for empty crates: no `part:`)"""
    from binsim.assets import prepare_bin, prepare_part
    from binsim.pallets import plan_pallets
    if pc["crate"].get("bin_cfg"):                   # a box made of an STL: its USD like the bin's
        pc["crate"]["usd"] = prepare_bin(pc["crate"]["bin_cfg"], cache_dir).usd_path
    pa = prepare_part(pc["part_cfg"], cache_dir) if pc.get("part_cfg") else None
    cfg = {k: ({kk: vv for kk, vv in v.items() if kk != "bin_cfg"} if k == "crate" else v) for k, v in pc.items() if k not in ("part", "part_cfg", "seed", "physics")}
    if pa is None:
        cfg["parts_per_crate"] = 0
    plan = plan_pallets(cfg, pa.extents if pa else [0.01, 0.01, 0.01], (pc.get("part_cfg") or {}).get("layer_rpy_deg"), np.random.default_rng(pc["seed"]))
    plan["part_usd"] = pa.usd_path if pa else None
    return plan


def wrist_lens(cfg):
    """(intrinsics of the image, intrinsics of the pinhole render it is made from, LensWarp or None) of wrist_camera. With wrist_camera.lens
    (the lens of the training-data camera, configs/camera/phoxi_1952x1460.yaml: barrel distortion, the "convex" look) the renderer, an ideal
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


def make_wrist_camera(stage, cfg, root=ROBOT_PATH):
    """wrist_camera: a USD camera on the link `cfg.link` of the robot (an OpenCV optical frame: x right, y down, z forward), with the
    intrinsics of cfg (resolution, fx, fy; centred principal point; the size of the pinhole render when there is a lens, see wrist_lens).
    Returns the camera prim path, or None when the link is missing."""
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
    UsdGeom.Xformable(cam).AddOrientOp().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))   # a USD camera looks along -z with y up: turned 180 deg about x it is the optical frame
    return cam.GetPath().pathString


def make_fixed_camera(stage, cfg):
    """cameras: of the map: a fixed 2D camera at pos_m looking at look_at_m (world, m) with resolution [w, h] and a horizontal fov_deg, on a grey pole from the
    floor (pole_m: its x y, default under the camera) - only to look at, no collider. Returns the camera prim path"""
    name = str(cfg["name"])
    pos, tgt = np.array(cfg["pos_m"], float), np.array(cfg["look_at_m"], float)
    f = (tgt - pos) / np.linalg.norm(tgt - pos)                        # viewing direction
    r = np.cross(f, [0.0, 0.0, 1.0])
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([1.0, 0.0, 0.0])
    u = np.cross(r, f)
    M = Gf.Matrix4d(*r, 0.0, *u, 0.0, *(-f), 0.0, *pos, 1.0)        # a USD camera looks along its -z with +y up (rows = its axes in the world)
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
    grey = [Gf.Vec3f(0.18, 0.19, 0.21)]
    body = UsdGeom.Cube.Define(stage, f"{root}/body")                  # the housing, behind the lens (+z of the camera = backwards)
    body.CreateSizeAttr(1.0)
    body.CreateDisplayColorAttr(grey)
    bx = UsdGeom.Xformable(body)
    bx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.05))
    bx.AddScaleOp().Set(Gf.Vec3f(0.07, 0.05, 0.09))
    px, py = (float(v) for v in (cfg.get("pole_m") or cfg["pos_m"][:2]))
    top = float(pos[2]) - 0.03
    pole = UsdGeom.Cube.Define(stage, f"/World/cameras/{name}_pole")
    pole.CreateSizeAttr(1.0)
    pole.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.56, 0.58)])
    plx = UsdGeom.Xformable(pole)
    plx.AddTranslateOp().Set(Gf.Vec3d(px, py, top / 2.0))
    plx.AddScaleOp().Set(Gf.Vec3f(0.04, 0.04, top))
    return cam.GetPath().pathString


def capture_fixed_camera(cfg, cam_path):
    """one RGB picture of a fixed camera (cameras: of the map) -> <output_dir>/<name>_<n>.png; the render product only exists for the picture"""
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
        for _ in range(60):                                   # the first frames of a new render product can be empty
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


def make_editable(roots):
    """The instances below these prims are made editable (instanceable off, nested ones too: instanced mesh -> instanced material), only on this stage:
    the cached USD is unchanged. The converter puts the meshes and materials of a URDF into instances, which cannot be changed otherwise."""
    while True:
        inst = [p for r in roots for p in Usd.PrimRange(r) if p.IsInstance()]
        if not inst:
            return
        for p in inst:
            p.SetInstanceable(False)


def apply_decomposition(prim, options=None):
    """convexDecomposition collider on a mesh prim, with the options of DECOMPOSITION_ATTRS (max_hulls, voxel_resolution, error_percentage, ...)"""
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
    prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
    for key, value in (options or {}).items():
        name, kind = DECOMPOSITION_ATTRS[key]
        prim.CreateAttribute(f"physxConvexDecompositionCollision:{name}", getattr(Sdf.ValueTypeNames, kind)).Set({"Int": int, "Float": float, "Bool": bool}[kind](value))


def set_collision_decomposition(stage, patterns, root=ROBOT_PATH):
    """collision_decomposition: link-name regexes of the robot yaml (or of a machine, below its prim `root`), or {regex: options} with the options of the
    decomposition: a number = most convex hulls, or a mapping of DECOMPOSITION_ATTRS (max_hulls, voxel_resolution, error_percentage, ...; PhysX defaults:
    32 hulls, 500000 voxels, 10 % error). The colliders under these links (the URDF collision meshes, which the converter
    makes convexHull) become convexDecomposition, so the fingers keep their exact shape. Returns the number of colliders."""
    spec = dict(patterns) if isinstance(patterns, dict) else {q: None for q in patterns}
    rx = [(re.compile(str(q)), n) for q, n in spec.items()]
    roots = [(p, next(n for r, n in rx if r.fullmatch(p.GetName()))) for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if any(r.fullmatch(p.GetName()) for r, _ in rx)]
    make_editable([r for r, _ in roots])
    done = set()                                  # the link roots are nested (gripper_mount contains gripper_motor, ...): every collider once
    for r, spec_value in roots:
        options = {"max_hulls": spec_value} if isinstance(spec_value, (int, float)) and spec_value else dict(spec_value or {})
        sdf = options.pop("sdf", None)               # {sdf: 512}: the exact shape of the mesh as a signed distance field (resolution along the longest side), not hulls: thin prongs and windows stay
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


def set_collision_friction(stage, patterns, root, name="machine", override=None, combine="min"):
    """collision_friction: {link-name regex: friction} below the prim `root` (a machine): the colliders of these links get a physics material with this friction
    (static = dynamic) that wins over the one of the part touching them (friction combine mode `min`): 0.02 = oiled, a part slides down a sloped fixture.
    combine "max" (a robot's fingertips): the higher one wins, a grip with more friction than the part's own
    Set before the physics starts. Returns the number of colliders."""
    rx = [(re.compile(str(q)), float(f) if override is None else float(override), i) for i, (q, f) in enumerate(dict(patterns).items())]
    n, mats = 0, []
    for p in list(Usd.PrimRange(stage.GetPrimAtPath(root))):
        hit = next(((f, i) for r, f, i in rx if r.fullmatch(p.GetName())), None)
        if hit is None:
            continue
        f, i = hit
        mat_path = f"/World/PhysicsMaterials/{name}_friction_{i}"                     # one material per machine and pattern: its friction can be changed while the simulation runs
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
    """materials: [{links: [regex], color: [r, g, b], metallic, roughness}, ...] of a robot yaml / a machine yaml (gripper_material of the robot yaml = one rule
    for the links of the gripper): the visual materials under the links whose prim name matches get that look (real metal: colour, metallic 1, a low
    roughness = gloss). The converter turns the URDF colours into plastic-like materials (metallic 0, roughness 0.5). The rules are applied in order and a
    link contains the links below it, so a rule for a link and then one for a part of it works. Returns the number of materials."""
    count = 0
    for rule in rules:
        rx = [re.compile(str(q)) for q in rule["links"]]
        roots = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if any(r.fullmatch(p.GetName()) for r in rx)]
        make_editable(roots)
        mats = {UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0].GetPath() for r in roots for p in Usd.PrimRange(r)
                if p.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(p).ComputePurpose() == UsdGeom.Tokens.default_}
        mats.discard(Sdf.Path())                  # gprims without a material
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
    """A window of Isaac Sim with the last picture of control/vision_pick.py (all parts with their score ranking, the best one marked): the path of every new picture comes on
    /vision_pick/picture. It is made when the first picture arrives and comes back if it was closed."""

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
    """Joint commands of one machine of the scene (scene.machines of the map) from ROS; the sim loop reads them and reports the measured state.
    The topics of its machine yaml: hands.topic (the sim loop publishes which hand pairs have a part in front of them), part_pose.topic (the pose of
    the part on its seat), trigger_topic with attach: (parts were put on its jigs: the sim loop removes them)."""

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
        self.attach_requests = deque()               # trigger_topic: parts were put on the jigs (attach_step in the sim loop)
        trigger = (cfg.get("attach") or {}).get("trigger_topic") or cfg.get("trigger_topic")   # attach.trigger_topic, else the trigger_topic of the machine
        if cfg.get("attach") and trigger:
            self.create_subscription(Empty, str(trigger), lambda _: self.attach_requests.append(time.time()), 10)

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
    """The carriers of a machine's belt (carriers: of its machine yaml) move in steps of one pitch, stop-and-go, and run round: what leaves at the -y end
    of the loop comes back at the +y end. One topic controls it: std_msgs/Int32 on <topic>: n > 0 = advance n pitches (added to what is still to do),
    0 = stop where it is. <topic>/moving (std_msgs/Bool, latched) is True while the carriers are moving. The sim loop calls step() once per frame."""

    def __init__(self, name, topic, joints, slots, window, speed, accel=None, context=None):
        super().__init__(f"{name}_carriers", context=context)
        self.accel = float(accel) if accel else CARRIER_ACCEL    # m/s^2
        self.joints, self.slots, self.lo, self.hi = list(joints), [float(y) for y in slots], float(window[0]), float(window[1])
        self.loop = self.hi - self.lo                       # length of the loop = number of carriers x pitch
        self.pitch = self.loop / len(self.slots)
        self.speed = float(speed)                           # m/s
        self.v = 0.0                                        # m/s now
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
                v = min(self.speed, self.v + self.accel * dt, math.sqrt(2.0 * self.accel * (self.goal - self.s)))    # up to speed, and slow enough to stop at the goal
                self.v = max(v, 0.01)                                                                                      # (a creep, so that the goal is reached)
                self.s = min(self.goal, self.s + self.v * dt)
            else:
                self.v = 0.0
            moving = self.s < self.goal
        if moving != self.moving:
            self.moving = moving
            self._pub.publish(Bool(data=moving))
        return self.positions()


def build_stage(robots):
    """the stage: the floor, the lights, every robot (its usd, q0 and prim) and the scene of the map (it comes with the first robot)"""
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
    # GPU contact buffers: with a pile (100 parts x 32 convex hulls) the default found/lost aggregate pair capacity is
    # too small - PhysX: "increase ...foundLostAggregatePairsCapacity ..., otherwise the simulation will miss interactions"
    psa = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    psa.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(2 ** 21)
    psa.CreateGpuTotalAggregatePairsCapacityAttr().Set(2 ** 21)
    psa.CreateGpuMaxRigidPatchCountAttr().Set(2 ** 18)    # default 81920: the piles need ~90k contact patches while they settle ("Patch buffer overflow", contacts are dropped)
    psa.CreateGpuMaxRigidContactCountAttr().Set(2 ** 21)  # default 524288 ("Contact buffer overflow ... at least 621121" crashed the GPU once)
    psa.CreateGpuFoundLostPairsCapacityAttr().Set(2 ** 22)        # default 2^18: hundreds of hulls per fine plate, 26 SDF jigs
    psa.CreateGpuHeapCapacityAttr().Set(2 ** 28)                  # default 64 MB
    psa.CreateGpuTempBufferCapacityAttr().Set(2 ** 28)            # default 16 MB
    psa.CreateGpuCollisionStackSizeAttr().Set(2 ** 29)   # the default 64 MB overflows with the pallets (~720 free parts, ~300 MB seen): PhysX drops contacts then
    # physics rate (1 app update = 1/60 s of simulated time): at 60 Hz a part falling out of the pile column (up to 0.5 m, ~3 m/s)
    # moves ~5 cm per step, more than the size of a plate, and can sink into the parts below. The dataset generator steps at 200 Hz
    scene.GetPrim().CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(int(s.get("physics_hz", 60)))
    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.36)])
    fx = UsdGeom.Xformable(floor)
    fx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    fx.AddScaleOp().Set(Gf.Vec3f(30.0, 30.0, 0.02))      # 30 x 30 m
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
    if s.get("bin"):                              # bin and pile: separate prims, world poses (not under the bin)
        place("/World/bin", s["bin"]["usd"], s["bin"]["pos_m"], s["bin"]["rpy_deg"])
        for i, (usd, pos, R) in enumerate(s["bin"]["pile"]):
            place(f"/World/pile/{s['bin']['part']['name']}_{i:03d}", usd, pos, R=R)
    if s.get("basket"):                           # basket: world pose, kinematic like the bin
        place("/World/basket", s["basket"]["usd"], s["basket"]["pos_m"], s["basket"].get("rpy_deg", [0, 0, 0]))
    if s.get("pallets"):                          # crates of parts on the pallets: the crate USD keeps its own unit ops, so it hangs below a frame
        pc, pl = s["pallets"], s["pallets"]["plan"]
        pname = (pc.get("part_cfg") or {}).get("name", "part")
        for c in pl["crates"]:
            frame = f"/World/pallets/pallet_{c['pallet']}/L{c['layer']}_C{c['idx']}"
            place(frame, None, c["pos"], R=c["R"])
            crate = stage.DefinePrim(frame + "/crate")
            crate.GetReferences().AddReference(pc["crate"]["usd"])
            crate.RemoveAPI(UsdPhysics.RigidBodyAPI)       # the crate is a rigid body in its file: here it stands still (a static triangle mesh)
            if s.get("pallet_physics") and not c["dynamic"]:   # pallets.physics of the map: the crates of a layer with physics: false have no collider (not shown in the collision viewer, not cooked)
                for q in Usd.PrimRange(crate):
                    if q.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
            # collision of the crate mesh: meshSimplification | convexHull | convexDecomposition | none (the triangle mesh of the file, default) | off (no collider at all: only to look at; yaml reads a bare `off` as False)
            approx = "off" if pc["crate"].get("collision") is False else str(pc["crate"].get("collision") or "none")
            if approx == "off":                            # only to look at: no collider, no rigid body (the hinged wire handles would fall off), no joint
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
            if c.get("switch_off"):                        # pallets.physics switch: not its turn yet: no collider (switched on from /crate_active, see switch_crate)
                for q in Usd.PrimRange(crate):
                    if q.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
        for p in pl["parts"]:                       # the parts of the top layers move (/World/pile: clickable like the pile), the ones below only fill the crates
            path = (f"/World/pile/{pname}_p{p['pallet']}_L{p['layer']}_C{p['crate']}_{p['j']:02d}" if p["dynamic"]
                    else f"/World/pallets/pallet_{p['pallet']}/L{p['layer']}_C{p['crate']}_{pname}_{p['j']:02d}")
            pos = list(p["pos"])
            if p["dynamic"] and str(pc["crate"].get("collision", "none")) == "convexHull":   # the hull of a crate is solid up to its rim: the parts that move rest on top of it, not on the floor inside
                pos[2] += float(pc["crate"]["size_m"][2]) - float(pc["crate"]["floor_z_m"])
            place(path, pl["part_usd"], pos, R=p["R"])
            if p["dynamic"] and p["layer"] >= int(pc.get("layers", 6)) - int(pc.get("sdf_layers", 0)):     # the parts the robot picks: their exact shape (SDF), set before the physics starts (a change of the collision type later restarts the physics)
                gp = stage.GetPrimAtPath(f"{path}/geometry")
                UsdPhysics.MeshCollisionAPI.Apply(gp).CreateApproximationAttr("sdf")
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(gp).CreateSdfResolutionAttr(int(pc.get("sdf_resolution", 256)))
            if p.get("switch_off"):                        # a body already (switching it on later must not add one: that would restart the physics), but frozen and without collider
                for q in Usd.PrimRange(stage.GetPrimAtPath(path)):
                    if q.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
                UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).CreateKinematicEnabledAttr(True)
            if not p["dynamic"]:
                stage.GetPrimAtPath(path).RemoveAPI(UsdPhysics.RigidBodyAPI)
                if s.get("pallet_physics"):                    # pallets.physics of the map: a layer with physics: false has no collision at all (not even a static one): the robot never gets there
                    for q in Usd.PrimRange(stage.GetPrimAtPath(path)):
                        if q.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
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
    for mc_ in s.get("map_collision") or []:      # scene.map_collision: the collision of parts of the map (it is as made in the GUI otherwise): convexHull | convexDecomposition | meshSimplification | none (triangle mesh) | off = no collider
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
                elif mode == "convexDecomposition":                          # options: {max_hulls, voxel_resolution, error_percentage, ...} as for the machines
                    apply_decomposition(q, mc_.get("options"))
                else:
                    UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr(mode)
                n += 1
        print(f"[INFO] map_collision: {mc_['prim']}: {n} collider(s) -> {mc_.get('collision', 'convexHull')}", flush=True)
    for ed in s.get("map_edits") or []:           # scene.map_edits: parts of the map stretched in height: every vertex of the meshes below `prim` that is higher than above_z_m (world) moves up by dz_m
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
            M = np.array(UsdGeom.Xformable(q).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).T   # column vectors: world = M[:3,:3] p + M[:3,3]
            W = P @ M[:3, :3].T + M[:3, 3]
            W[W[:, 2] > float(ed["above_z_m"]), 2] += float(ed["dz_m"])
            P2 = (W - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P2.astype(np.float32)))
            mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in P2.min(0)]), Gf.Vec3f(*[float(v) for v in P2.max(0)])]))
            n += 1
        print(f"[INFO] map_edits: {ed['prim']}: {n} mesh(es) stretched by {ed['dz_m']} m above z {ed['above_z_m']} m", flush=True)
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
    for r in robots:
        if r["gripper_links"] and r.get("gripper_material"):   # {metallic, roughness} on every visual material of the gripper links
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
        if r.get("collision_friction"):                  # {link-name regex: friction} of the robot yaml (the fingertips): combine mode max = it wins over the part's 0.5
            n, _ = set_collision_friction(stage, r["collision_friction"], r["prim"], f"{r['name']}_robot", combine="max")
            print(f"[INFO] {r['prim']}: collision_friction {dict(r['collision_friction'])} (combine max): {n} colliders", flush=True)
    for m in s.get("machines") or []:
        if m.get("collision_decomposition"):
            n = set_collision_decomposition(stage, m["collision_decomposition"], f"/World/machines/{m['name']}")
            print(f"[INFO] machine {m['name']}: collision_decomposition {list(m['collision_decomposition'])}: {n} colliders -> convexDecomposition", flush=True)
        att = m.get("attach") or {}
        if att.get("enabled", True) and att.get("jig_links") and att.get("jig_collision") is False:   # the jigs are only to look at: the parts put on them are removed
            rx = [re.compile(str(q)) for q in att["jig_links"]]
            n = 0
            for p in Usd.PrimRange(stage.GetPrimAtPath(f"/World/machines/{m['name']}")):
                if any(r.fullmatch(p.GetName()) for r in rx):
                    for q in Usd.PrimRange(p):
                        if q.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
                            n += 1
            print(f"[INFO] machine {m['name']}: jigs {list(att['jig_links'])} without collider ({n}): the parts put on them are removed", flush=True)
        if m.get("collision") is False:                                  # the whole machine only to look at: no collider on any of its links (its joints still move)
            n = 0
            for q in Usd.PrimRange(stage.GetPrimAtPath(f"/World/machines/{m['name']}")):
                if q.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
                    n += 1
            print(f"[INFO] machine {m['name']}: collision false: {n} colliders off (only to look at)", flush=True)
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

    for r in robots:
        k, d = float(r.get("stiffness", 20000.0)), float(r.get("damping", 1000.0))
        drives(r["prim"], r["joints"], r["q0"], {"linear": k, "angular": k}, {"linear": d, "angular": d})
    for m in s.get("machines") or []:
        prefix = f"/World/machines/{m['name']}"
        drives(prefix, m["joints"], m["q0"], MACHINE_STIFFNESS, {**MACHINE_DAMPING, **({"angular": float(m["rotor_damping"])} if m.get("rotor_damping") else {})})
        for p in Usd.PrimRange(stage.GetPrimAtPath(prefix)):    # the collision boxes of the frame overlap the rotors and jaws: their links must not collide with each other
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(p).CreateEnabledSelfCollisionsAttr().Set(False)


def main():
    globals().update(robot.get("sim_consts") or {})   # ARRIVE_TOL, SETTLE_FRAMES, GRIP_GRACE, GRIP_WINDOW from the robot yaml
    contexts = {}                                # ROS domain id of a robot (di; None = the one of the environment) -> (its rclpy context, its executor)

    def ros(di):
        """the rclpy context and executor of one ROS domain: the nodes of a robot live in its domain, so every robot is controlled on its own"""
        if di not in contexts:
            ctx = rclpy.Context()
            rclpy.init(args=[], context=ctx, **({} if di is None else {"domain_id": int(di)}))   # (not the command line: rcl would read robot:=ur5e as a remap rule)
            contexts[di] = (ctx, MultiThreadedExecutor(context=ctx))
        return contexts[di]
    units = []                                   # per robot: its settings, articulation, ROS bridge and rate-limited command
    try:
        for r in robots:
            r["usd"] = robot_usd(r, args.force_convert)  # before the stage is built: the converter uses its own stages
            print(f"[INFO] {r['urdf']} -> {r['usd']}", flush=True)
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
                phys = robot.get("pallet_physics") or {}
                if phys:                                                            # pallets.physics of the map, per pallet and layer: the parts of a layer with physics: false are fixed in their crates (static, no cost)
                    def has_physics(pallet, layer, crate, default):             # the layer has physics, and (when `crates:` names some) this crate is one of them (switch: every crate of the layer)
                        ph = phys.get((pallet, layer))
                        return default if ph is None else (ph[0] and (ph[1] is None or crate in ph[1] or ph[2]))

                    def switched_off(pallet, layer, crate):                     # switch: true: a crate of the layer that is not in `crates:` starts switched off (no collider, its parts frozen)
                        ph = phys.get((pallet, layer))
                        return bool(ph and ph[0] and ph[2] and ph[1] is not None and crate not in ph[1])
                    for c in pl["crates"]:
                        c["dynamic"] = has_physics(c["pallet"], c["layer"], c["idx"], c["dynamic"])
                        c["switch_off"] = switched_off(c["pallet"], c["layer"], c["idx"])
                    for p in pl["parts"]:
                        p["dynamic"] = has_physics(p["pallet"], p["layer"], p["crate"], p["dynamic"])
                        p["switch_off"] = switched_off(p["pallet"], p["layer"], p["crate"])
                n_dyn = sum(p["dynamic"] for p in pl["parts"])
                print(f"[INFO] pallets: {len(pc['items'])} x {int(pc.get('layers', 6))} layers, {len(pl['crates'])} crates, {len(pl['parts'])} x "
                      f"{(pc.get('part_cfg') or {}).get('name', '-')} ({n_dyn} movable, seed {pc['seed']}); grids per layer (nx x ny): "
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
        for r in robots:
            jp = r.get("joint_pos") or {}
            r["q0"] = {n: min(max(next((float(v) for k, v in jp.items() if re.fullmatch(k, n)), 0.0), lo), hi)
                       for n, (_, lo, hi, _, _) in r["joints"].items()}
            for n, (lead, k, off) in r["mimic"].items():
                if lead in r["q0"]:
                    r["q0"][n] = r["q0"][lead] * k + off
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
        build_stage(robots)

        for r in robots:
            root_path = Articulation.fetch_articulation_root_api_prim_paths(r["prim"])[0]
            if root_path is None:
                print(f"[ERROR] no articulation root under {r['prim']} in {r['usd']}", flush=True)
                return
            units.append({"r": r, "root": root_path, "art": Articulation(root_path)})
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

        for u in units:                          # every robot: its ROS bridge in its own domain (di)
            r = u["r"]
            u["names"] = list(u["art"].dof_names)
            u["cmd"] = np.array([r["q0"].get(n, 0.0) for n in u["names"]])
            u["cmd0"] = u["cmd"].copy()          # the start pose, where the arm is again after a restart of the physics
            u["step"] = np.array([r["joints"].get(n, (None, 0, 0, 1.0))[3] for n in u["names"]]) * float(r.get("speed", 0.3)) / APP_HZ
            controller = "/" + str(r.get("controller", f"{r['name']}_controller")).strip("/")
            gripper = {n: r["joints"][n][1:3] for n in r["gripper_joints"]}
            gripper_topic = str(r.get("gripper_topic") or "/gripper/command")
            limits = [r["joints"][n][1:3] if n in r["joints"] else (-math.inf, math.inf) for n in u["names"]]
            ctx, ex = ros(r.get("di"))
            u["node"] = RobotBridge(u["names"], u["cmd"], controller, r["mimic"], gripper, gripper_topic, limits, context=ctx)
            ex.add_node(u["node"])
            print(f"[INFO] robot {r['name']} at {r['prim']} (ROS domain {r['di'] if r.get('di') is not None else os.environ.get('ROS_DOMAIN_ID', '0')}) ready "
                  f"(gripper: {r['gripper']}): joints {u['names']}, max speed {r.get('speed', 0.3)} x URDF limits", flush=True)
            print(f"[INFO]   ROS 2: {controller}/joint_trajectory, {controller}/follow_joint_trajectory -> sim, sim -> /joint_states"
                  + (f"; gripper {r['gripper']}: {gripper_topic} (std_msgs/Float64) 0 = closed, 1 = open -> {list(gripper)}" if gripper else ""), flush=True)
        node = units[0]["node"]                  # the first robot: the log lines of the loop (frames, last commands)
        mctx, executor = ros(robot.get("di"))    # the machines live in the domain of the first robot
        for mc in machines:
            m = mc["m"]
            mc["names"] = list(mc["art"].dof_names)
            mc["cmd"] = np.array([m["q0"].get(n, 0.0) for n in mc["names"]])
            mc["cmd0"] = mc["cmd"].copy()
            speeds = [(str(rx), float(f)) for rx, f in (m.get("joint_speeds") or {}).items()]   # joint_speeds of the machine yaml: {joint-name regex: x URDF limit}
            mc["step"] = np.array([m["joints"][n][3] * next((f for rx, f in speeds if re.fullmatch(rx, n)), float(m.get("speed", 1.0)))
                                   for n in mc["names"]]) / APP_HZ
            mc["vel"] = np.zeros(len(mc["names"]))                                  # velocity of the command (rad/s, m/s)
            mc["rot"] = np.array([m["joints"][n][0] != "prismatic" for n in mc["names"]])
            acc_rot, acc_lin = float(m.get("rotor_accel_rad_s2") or MACHINE_ACCEL["angular"]), float(m.get("jaw_accel_m_s2") or MACHINE_ACCEL["linear"])
            mc["acc"] = np.where(mc["rot"], acc_rot, acc_lin)
            mc["mname"] = m["name"]
            damping = float(m.get("rotor_damping") or MACHINE_DAMPING["angular"])
            effort = max((info[4] for info in m["joints"].values() if info[0] != "prismatic"), default=0.0)   # the strongest rotor drive (URDF effort, N*m)
            print(f"[INFO] machine {m['name']}: speed {float(m.get('speed', 1.0)):g} x URDF limit, joint_speeds {dict(speeds)}; acceleration rotors {acc_rot:g} rad/s2, "
                  f"jaws {acc_lin:g} m/s2; rotor damping {damping:g}" + (f" (speed limit by the drive force {effort / damping:.0f} deg/s)" if effort else ""), flush=True)
            mc["q"] = mc["cmd"].copy()                                              # measured positions of the last frame
            mc["carrier"] = np.zeros(len(mc["names"]), bool)                        # the joints of the belt: not driven by joint_command (set below)
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
                    mc["carriers"] = {"node": CarrierBridge(m["name"], c.get("topic") or f"/{m['name']}/carriers", c["joints"], c["slots_m"], c["window_m"], c.get("speed_mps", 0.1), c.get("accel_mps2"), context=mctx),
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

        # click a part in the viewport -> its pose is printed as "x y z roll pitch yaw" (and the TCP pose that grips it below), the Translate and Rotate X Y Z of Isaac's Property
        # panel (m, deg; read live from the physics, the panel itself may keep showing the start values). The event only remembers
        # the clicked parts, the pose is read in the loop below
        clicked, views = [], {}
        picture_panel = PicturePanel()
        part_view = [None, None]                 # [RigidPrim over all parts of /World/pile, their paths]: the parts nearest to a /target_pose


        def rebuild_views(why):
            """Deleting a prim in the GUI (a plate) invalidates the physics tensor views (PhysX: "Simulation view object is invalidated") and
            they can only be made again by playing the timeline anew: the physics restarts from the start state (the arm and the parts that
            moved are back where they began, the deleted prims stay deleted) instead of the app closing"""
            print(f"[WARN] a prim was deleted: the physics views are invalid ({why}); restarting the physics from the start state", flush=True)
            views.clear()
            part_view[0] = None
            timeline.stop()
            update(2)
            timeline.play()
            update(3)
            for u in units:
                u["art"] = Articulation(u["root"])
                u["cmd"] = u["cmd0"].copy()
                u["node"]._set(u["names"], u["cmd"])
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
        refined = {}     # part path -> (its original max_convex_hulls, its original voxel resolution), for the part that has CLICK_HULLS / CLICK_VOXELS now (only one at a time)
        refined_at = {}  # part path -> where it lay when it was refined (see CARRIED_M)

        def pile_positions():
            """part path -> its world position now (m), for every part of the pile"""
            view, paths = pile_view()
            return dict(zip(paths, np.array(view.get_world_poses()[0].numpy()))) if view is not None else {}

        def remember_at(path):
            at = pile_positions().get(path)
            if at is not None:
                refined_at[path] = at

        def carried():
            """the refined parts the robot carried away from where they were refined: they keep the fine collider and the stiff settings"""
            pos = pile_positions()
            return {p for p, at in refined_at.items() if p in pos and np.linalg.norm(pos[p] - at) > CARRIED_M}

        def set_click_physx(prim, stiff):
            """CLICK_PHYSX contact settings on the rigid body of a part: stiff = the clicked part, else the PhysX defaults"""
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            kinematic = bool(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
            for method, (on, off) in CLICK_PHYSX.items():
                if not (stiff and kinematic and method == "CreateEnableCCDAttr"):   # PhysX: kinematic bodies with CCD are not supported
                    getattr(api, method)().Set(on if stiff else off)

        stiff_order = []     # paths of the parts that have the stiff contact settings now, the newest last
        plate_ev = {}        # plate path -> [(time, what happened to its collider / physics)], the last ones: the BOOM line (boom_step) tells them

        def note(path, text):
            plate_ev.setdefault(path, []).append((time.time(), text))
            del plate_ev[path][:-6]

        def make_stiff(path):
            """CLICK_PHYSX stiff settings for the part the fingers go into; of the parts that still lie where they were refined only the STIFF_KEEP newest keep them (16 solver iterations + CCD
            on every clicked part are heavy). A part the robot carried keeps them until attach_parts removes it on a jig"""
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
            """The clicked part gets CLICK_HULLS convex hulls on a CLICK_VOXELS voxel grid (PhysX cooks its collider again); of the parts that still lie where they were refined
            the oldest beyond CLICK_KEEP go back to their own (a part the robot carried keeps it, see CARRIED_M)"""
            stage_now = omni.usd.get_context().get_stage()
            busy = carried()
            idle = [(old, nv) for old, nv in refined.items() if old not in busy]
            for old, (n, voxels) in idle[:max(0, len(idle) - CLICK_KEEP + 1)]:
                if old != path and n is None:                                # an SDF part: only its contact settings were changed
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
                refined[path] = (None, None)                                 # already its exact shape: nothing to cook, only the stiff contact settings
                remember_at(path)
                make_stiff(path)
                print(f"[INFO] {path}: exact shape (SDF), stiff contact settings", flush=True)
                return
            attr = stage_now.GetPrimAtPath(f"{path}/geometry").GetAttribute(HULLS_ATTR)
            if not attr.IsValid() or path in refined:
                return                                                       # not a convex decomposition part, or already fine
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
            """The part the robot is sent to (/target_pose = the TCP = the part's centre) gets what a clicked part gets: the exact collider (CLICK_HULLS) and the
            stiff contact settings (CLICK_PHYSX). Without it the fingers sink into the thin wall of the bore and the part sticks to them"""
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
            own = next((sw for sw in switch if paths[i] in sw["parts"]), None)    # a part of a switched-off crate (its /crate_active was missed): that crate gets the physics now
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
        for mc in machines:                      # fixtures: of the machine yamls are read again when the file has changed (reload_fixtures)
            f = mc["m"].get("config_file")
            fx_state["mtime"][mc["mname"]] = os.path.getmtime(f) if f and os.path.isfile(f) else 0.0

        def fixture_friction():
            """the friction of the jigs of a machine whose belt / chain moves is `friction_moving` (the plates ride along), at rest `friction` (a plate slides down the prongs into the windows)"""
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
            """fixtures: of a machine yaml again when the file has changed: the friction of the jigs is changed while the simulation runs (a material attribute, no restart)"""
            for mc in machines:
                path, name = mc["m"].get("config_file"), mc["mname"]
                if not path or not os.path.isfile(path) or os.path.getmtime(path) == fx_state["mtime"].get(name):
                    continue
                fx_state["mtime"][name] = os.path.getmtime(path)
                try:
                    fx_state["fx"][name] = dict(sim_config.machine_config(mc["m"]["entry"]).get("fixtures") or {})
                    fx_state["last"].pop(name, None)                         # fixture_friction() sets the new values in the next frame
                    print(f"[INFO] fixtures of {os.path.basename(path)} read again (no restart): {json.dumps(fx_state['fx'][name])}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] fixtures of {path} not read: {type(e).__name__}: {e}", flush=True)

        rebuilds = 0
        # attach: of a machine yaml (the conveyor): the parts that the hands put on its jigs are taken out of the simulation (hidden, no collider, no gravity, parked far below
        # the floor). A deleted prim would invalidate every physics view (rebuild_views restarts the physics), and parts that stay on the chain cost contacts / kinematic updates
        attaches = [{"mc": mc, "cfg": mc["m"]["attach"], "B": None, "bound": {}}   # bound: part path -> the carrier joint it was put on (removed)
                    for mc in machines if mc["m"].get("attach") and mc["m"]["attach"].get("enabled", True) and mc.get("carriers")]
        PARK = (0.0, 0.0, -50.0)   # m: where the removed parts wait (no collider, no gravity: nothing is calculated for them)

        def bound_parts():
            return {path for att in attaches for path in att["bound"]}

        def jig_matrix(att, joint):
            """link -> world matrix of a carrier (a jig pair of the conveyor): the base of the machine (fixed: its USD pose is right) and the joint (prismatic along carriers.axis)
            from the MEASURED joint position: the physics result is not written back to the USD stage, so the pose of a moving link in USD is the one of the start"""
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
            """the ONE physics view of all plates of the pile (the one of refine_target): a new RigidPrim per call would add a tensor view (and its GPU buffers) every cycle"""
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
            jig_matrix(att, att["mc"]["carriers"]["node"].joints[0])           # (sets att["B"], the base of the machine in the world)
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
                part = st_.GetPrimAtPath(path)
                for q_ in Usd.PrimRange(part):
                    if q_.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(q_).CreateCollisionEnabledAttr(False)
                set_click_physx(part, False)
                PhysxSchema.PhysxRigidBodyAPI.Apply(part).CreateDisableGravityAttr(True)
                UsdGeom.Imageable(part).MakeInvisible()
                note(path, f"removed on {name}: hidden, collider off, parked")
                refined.pop(path, None)
                refined_at.pop(path, None)
                if path in stiff_order:
                    stiff_order.remove(path)
                gone.append(i)
            if not gone:
                print(f"[INFO] {name}: no part above it (attach.remove_box_m)", flush=True)
            if gone:
                view.set_world_poses(np.tile(np.array(PARK, dtype=np.float32), (len(gone), 1)), np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(gone), 1)), indices=gone)
                view.set_velocities(np.zeros((len(gone), 3), dtype=np.float32), np.zeros((len(gone), 3), dtype=np.float32), indices=gone)
                print(f"[INFO] {len(gone)} part(s) on the jigs of {name} removed from the simulation, {len(att['bound'])} in all", flush=True)

        def attach_step():
            for att in attaches:
                try:
                    if att["mc"]["node"].attach_requests:                  # trigger_topic of the machine yaml: the hands are about to let go over its jigs
                        att["mc"]["node"].attach_requests.clear()
                        att["due"] = timeline.get_current_time() + float(att["cfg"].get("delay_s", 0.0))   # simulated time
                    if att.get("due") is not None and timeline.get_current_time() >= att["due"]:
                        att["due"] = None
                        attach_parts(att)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] attach to {att['mc']['mname']}: {type(e).__name__}: {str(e)[:100]}", flush=True)

        watchers = [mc for mc in machines if mc["m"].get("hands") or mc["m"].get("part_pose")]   # machines that look at the parts (hands: / part_pose: of their yaml)

        def watch_machine(mc, pn, qn):
            """for a machine with hands: / part_pose: in its yaml (positions in the frame of the machine): a bit per hand pair when a part (z in the range) lies within radius_m of
            the carrier slot in front of that pair (hands.topic), and the pose of the part nearest to the seat within radius_m (part_pose.topic; none there: nothing is published)"""
            m = mc["m"]
            pose = (m.get("pos_m") or [0, 0, 0], m.get("rpy_deg") or [0, 0, 0])
            hands = m.get("hands") or {}
            if hands and mc["node"].hands_pub is not None:
                car = m.get("carriers") or {}
                ax = [float(v) for v in car.get("axis", [1, 0, 0])]
                zlo, zhi = (float(v) + float(pose[0][2]) for v in hands.get("z_m", [0.9, 1.2]))
                mask = 0
                for i, k in enumerate(hands.get("slots") or []):
                    at = float(car["slots_m"][int(k)])
                    hx, hy, _ = sim_config.to_world(pose, [ax[0] * at, ax[1] * at, ax[2] * at])
                    if np.any((np.hypot(pn[:, 0] - hx, pn[:, 1] - hy) < float(hands.get("radius_m", 0.03))) & (pn[:, 2] > zlo) & (pn[:, 2] < zhi)):
                        mask |= 1 << i
                mc["node"].hands_pub.publish(Int32(data=int(mask)))
            seat = m.get("part_pose") or {}
            if seat and mc["node"].part_pose_pub is not None and len(pn):
                sx, sy = (float(v) for v in (seat.get("seat_m") or [0.0, 0.0])[:2])
                wx, wy, _ = sim_config.to_world(pose, [sx, sy, 0.0])
                d = np.hypot(pn[:, 0] - wx, pn[:, 1] - wy)
                k = int(np.argmin(d))
                if d[k] < float(seat.get("radius_m", 0.08)):
                    mc["node"].publish_part_pose(pn[k], qn[k])

        stat = {"t": time.time(), "f": 0, "sim": timeline.get_current_time()}
        BOOM_SPEED = 2.0                                       # m/s: a plate this fast is thrown (the pile settles at < 0.1 m/s, an arm carries a plate at ~1 m/s: those are not looked at)
        boom = {"t": 0.0, "seen": {}, "n": 0}

        def boom_step():
            """10 times a second: a plate that moves faster than BOOM_SPEED (and is not in the gripper) is written down with everything that was done to it / around it just before"""
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
            except Exception:  # noqa: BLE001
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
            """every 30 s one line: how the frame rate and the memory of this process change while the sim runs (is it a leak or the physics that gets heavier?)"""
            now = time.time()
            if now - stat["t"] < 30.0:
                return
            fps = (node.frame - stat["f"]) / (now - stat["t"])
            sim_now = timeline.get_current_time()
            per_frame = (sim_now - stat["sim"]) / max(1, node.frame - stat["f"])
            stat["t"], stat["f"], stat["sim"] = now, node.frame, sim_now
            rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
            print(f"[STAT] {fps:5.1f} fps ({per_frame * 1000:.1f} ms of simulated time per frame), rss {rss:6.0f} MB, fine collider {len(refined)}, stiff {len(stiff_order)}, removed on a jig {len(bound_parts())}", flush=True)

        # pallets.physics `switch: true` (a layer of the map): one crate of the layer at a time has physics (its collider, its parts free); the others are only to look at
        # (no collider, parts kinematic). /crate_active (pallets.switch_topic; std_msgs/String json {"centre": [x, y]}, vision_pick.py: the crate it picks from now) switches
        switch, switch_req = [], deque()
        if robot.get("pallets") and robot["pallets"].get("plan") and any(len(v) > 2 and v[2] for v in (robot.get("pallet_physics") or {}).values()):
            pc_ = robot["pallets"]
            pname_ = (pc_.get("part_cfg") or {}).get("name", "part")
            for c in pc_["plan"]["crates"]:
                ph = robot["pallet_physics"].get((c["pallet"], c["layer"]))
                if ph and ph[0] and ph[2]:
                    switch.append({"name": f"pallet {c['pallet']} layer {c['layer']} crate {c['idx']}", "pos": [float(v) for v in c["pos"]], "on": not c.get("switch_off"),
                                   "pallet": c["pallet"], "layer": c["layer"],
                                   "crate": f"/World/pallets/pallet_{c['pallet']}/L{c['layer']}_C{c['idx']}/crate",
                                   "parts": [f"/World/pile/{pname_}_p{p['pallet']}_L{p['layer']}_C{p['crate']}_{p['j']:02d}" for p in pc_["plan"]["parts"]
                                             if (p["pallet"], p["layer"], p["crate"]) == (c["pallet"], c["layer"], c["idx"])]})
            r_in = math.hypot(*[float(v) for v in pc_["crate"]["inner_size_m"][:2]]) / 2.0     # a part this close to the crate centre (x y) still lies in it
            topic = str(pc_.get("switch_topic") or "/crate_active")
            node.create_subscription(String, topic, lambda m: switch_req.append(m.data), 10)
            print(f"[INFO] pallets: {len(switch)} crates switched one at a time ({sum(sw['on'] for sw in switch)} on now) from {topic}", flush=True)

        def switch_crate(sw, on):
            """physics of one crate of a switch layer: its collider, and its parts (on = free with collider; off = kinematic without collider, only the ones still in it)"""
            st_ = omni.usd.get_context().get_stage()
            for q in Usd.PrimRange(st_.GetPrimAtPath(sw["crate"])):
                if q.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(on)
            pos, gone, n = pile_positions(), bound_parts(), 0
            for path in sw["parts"]:
                at = pos.get(path)
                if path in gone or (not on and (at is None or math.hypot(at[0] - sw["pos"][0], at[1] - sw["pos"][1]) > r_in or abs(at[2] - sw["pos"][2]) > 0.3)):
                    continue                                                 # removed, or (off) picked and carried away: left alone. On: every part of it (frozen since the start)
                part = st_.GetPrimAtPath(path)
                if not on:
                    set_click_physx(part, False)                             # CCD off before it is kinematic
                for q in Usd.PrimRange(part):
                    if q.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(on)
                UsdPhysics.RigidBodyAPI(part).CreateKinematicEnabledAttr(not on)
                n += 1
            sw["on"] = on
            print(f"[INFO] {sw['name']}: physics {'on' if on else 'off'} (the crate and {n} part(s))", flush=True)

        taken = set()        # (pallet, layer) taken away: the layers above the crate that has the physics now

        def take_away(pallet, layer):
            """a layer that is done (the robot works in the layer below it now): its crates and the parts left in them are hidden and lose their colliders
            (a deleted prim would restart the physics); the moving parts that are still in it are frozen too"""
            st_ = omni.usd.get_context().get_stage()
            root = st_.GetPrimAtPath(f"/World/pallets/pallet_{pallet}")
            n = 0
            for q in (root.GetChildren() if root.IsValid() else []):
                if q.GetName().startswith(f"L{layer}_"):                   # the crate frames L<layer>_C<i> and the fixed parts L<layer>_C<i>_<part>_<j>
                    UsdGeom.Imageable(q).MakeInvisible()
                    for c in Usd.PrimRange(q):
                        if c.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(c).CreateCollisionEnabledAttr(False)
                    n += 1
            for sw in switch:
                if (sw["pallet"], sw["layer"]) == (pallet, layer):
                    if sw["on"]:
                        switch_crate(sw, False)
                    for path in sw["parts"]:
                        part = st_.GetPrimAtPath(path)
                        if part.IsValid() and path not in bound_parts():
                            UsdGeom.Imageable(part).MakeInvisible()
                            for c in Usd.PrimRange(part):
                                if c.HasAPI(UsdPhysics.CollisionAPI):
                                    UsdPhysics.CollisionAPI(c).CreateCollisionEnabledAttr(False)
            taken.add((pallet, layer))
            print(f"[INFO] pallet {pallet} layer {layer} is done: taken away ({n} crates / parts hidden, no colliders)", flush=True)

        def switch_step():
            while switch_req:
                raw = switch_req.popleft()
                try:
                    msg = json.loads(raw)
                    cx, cy = (float(v) for v in msg["centre"][:2])
                    fz = msg.get("floor_z")                                 # the floor of the crate: which layer of the stack (the centre is the same in every layer)
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
                        if sw is not target and sw["on"]:
                            switch_crate(sw, False)
                    if not target["on"]:
                        switch_crate(target, True)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] /crate_active {raw}: {type(e).__name__}: {e}", flush=True)

        cam_req = deque()    # cameras: of the map: the ones asked for a picture (/<name>/capture)
        for c in robot.get("cameras") or []:
            node.create_subscription(Empty, f"/{c['name']}/capture", lambda _, c=c: cam_req.append(c), 10)

        while simulation_app.is_running():
            switch_step()
            if time.time() - fx_state["t"] > 1.0:
                fx_state["t"] = time.time()
                reload_fixtures()
            fixture_friction()
            attach_step()
            stat_step()
            boom_step()
            if watchers and time.time() - fx_state.get("tp", 0.0) > 0.2:     # 5 Hz: the machines that look at the parts (hands, the part on a seat)
                fx_state["tp"] = time.time()
                try:
                    if fx_state.get("pile") is None:                       # all parts of the pile and the objects in one view
                        st_ = omni.usd.get_context().get_stage()
                        fx_state["pile"] = RigidPrim([str(q.GetPath()) for root_ in ("/World/pile", "/World/objects") if st_.GetPrimAtPath(root_).IsValid() for q in st_.GetPrimAtPath(root_).GetChildren()])
                    pos_, quat_ = fx_state["pile"].get_world_poses()
                    pn, qn = pos_.numpy(), quat_.numpy()
                    for mc in watchers:
                        watch_machine(mc, pn, qn)
                except Exception:  # noqa: BLE001
                    fx_state["pile"] = None
            for u in units:
                u["cmd"] = u["cmd"] + np.clip(np.array(u["node"].targets(u["cmd"], u["step"])) - u["cmd"], -u["step"], u["step"])  # rate-limited towards the command
            try:
                for u in units:
                    u["art"].set_dof_position_targets(u["cmd"])
                for mc in machines:
                    target = mc["node"].targets()
                    new = target != mc["last_target"]                                       # a new command: the move starts from where the joint really is (a joint that was held
                    mc["cmd"] = np.where(new & ~mc["carrier"], mc["q"], mc["cmd"])          # against something has its command far ahead of it: no waiting, then a jump)
                    mc["vel"] = np.where(new, 0.0, mc["vel"])
                    mc["last_target"] = target
                    err = target - mc["cmd"]                                               # towards the commanded position on a velocity ramp
                    v_want = np.sign(err) * np.minimum(mc["step"] * APP_HZ, np.sqrt(2.0 * mc["acc"] * np.abs(err)))
                    mc["vel"] = mc["vel"] + np.clip(v_want - mc["vel"], -mc["acc"] / APP_HZ, mc["acc"] / APP_HZ)
                    mc["cmd"] = mc["cmd"] + np.clip(mc["vel"] / APP_HZ, -np.abs(err), np.abs(err))
                    mc["cmd"] = np.where(mc["rot"], np.clip(mc["cmd"], mc["q"] - MACHINE_LEAD, mc["q"] + MACHINE_LEAD), mc["cmd"])    # a held rotor does not wind up
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
                for u in units:
                    u["node"].report(u["art"].get_dof_positions().numpy()[0], u["art"].get_dof_velocities().numpy()[0])
                for mc in machines:
                    mc["q"] = mc["art"].get_dof_positions().numpy()[0]
                    mc["node"].report(mc["q"], mc["art"].get_dof_velocities().numpy()[0])
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
                    for u in units:
                        u["node"].publish_turned_over(flipped)
                except Exception as e:
                    print(f"[WARN] pose of {path} not available: {type(e).__name__}: {e}", flush=True)
            for u in units:
                un, r = u["node"], u["r"]
                while un.pictures:                                           # /vision_pick/picture: show the newest one in a window of Isaac Sim
                    path = un.pictures.popleft()
                    if not un.pictures and os.path.isfile(path):
                        try:
                            picture_panel.show(path)
                        except Exception as e:  # noqa: BLE001
                            print(f"[WARN] picture {path} not shown: {type(e).__name__}: {e}", flush=True)
                while un.targets_seen:                                       # /target_pose: the part the robot goes to
                    pos = un.targets_seen.popleft()
                    try:
                        refine_target(pos)
                    except Exception as e:  # noqa: BLE001
                        print(f"[WARN] part at the target {pos} not refined: {type(e).__name__}: {e}", flush=True)
                if un.capture_requested.is_set():                            # /wrist_camera/capture
                    un.capture_requested.clear()
                    try:
                        if not r.get("wrist_camera_path"):
                            raise RuntimeError("no wrist camera: set `wrist_camera:` in the robot yaml (and the link in the URDF)")
                        q_now = dict(zip(u["names"], (float(v) for v in u["art"].get_dof_positions().numpy()[0])))    # the measured joints = what the camera sees
                        T_base = make_T(rpy_to_mat(r.get("rpy_deg", [0, 0, 0])), r.get("pos_m", [0, 0, 0]))
                        f, valid = capture_wrist_camera(r["wrist_camera"], r["wrist_camera_path"],
                                                        urdf_link_pose(r["urdf"], str(r["wrist_camera"]["link"]), q_now, T_base))
                        print(f"[INFO] wrist camera frame -> {f} ({valid * 100:.0f} % of the pixels have depth)", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"[WARN] wrist camera capture failed: {type(e).__name__}: {e}", flush=True)
            while cam_req:                                                   # /<name>/capture of a fixed camera
                c = cam_req.popleft()
                try:
                    f = capture_fixed_camera(c, c["path"])
                    print(f"[INFO] camera {c['name']} picture -> {f}", flush=True)
                    picture_panel.show(f)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] camera {c['name']} capture failed: {type(e).__name__}: {e}", flush=True)
        timeline.stop()
    except KeyboardInterrupt:
        pass
    finally:
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
