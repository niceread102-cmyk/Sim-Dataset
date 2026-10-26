"""Where the config files of the simulation are and how they are read: the maps, robots and machines of isaacsim/configs/.
One place for robot_sim.py, check_reach.py, isaacsim/machine/*.py, control/*.py and the launch (bringup/integration/launch/robot.launch.py),
so they all read the same map, the same robot pose and the same machine settings.

  configs/map/<map>.yaml         a map, chosen with map:=<map>: what stands around the robot (map USD, machines, pallets, objects) in the
                                 world frame of the map, the robot of the map (scene.robot) and where it stands (scene.robot_pose). Every
                                 map is a file of its own. No map:= = an empty map: only the robot and the floor.
                                 (map.yaml of that folder is the config of the dataset generator, generate_isaacsim.py: not a map)
  configs/robot/<robot>.yaml    a robot, chosen with robot:=<robot> (default: scene.robot of the map): the same in every map
  configs/machine/<name>.yaml   a machine: the same in every map; a map puts it somewhere (scene.machines: name, pos_m, rpy_deg, ...)
Only os, math and yaml are used (and binsim.config for the `defaults:` of a yaml), so the Isaac Sim python, the launch and the ROS nodes
can all import it: put this folder (Simulation/isaacsim) on sys.path first.
"""
import math
import os

import yaml

from binsim.config import _load_yaml_tree, deep_merge  # noqa: F401  (deep_merge is used by the importers too)

ISAAC_DIR = os.path.dirname(os.path.abspath(__file__))     # Simulation/isaacsim
SIM_DIR = os.path.dirname(ISAAC_DIR)                        # Simulation
CONFIGS = os.path.join(ISAAC_DIR, "configs")
MAPS_DIR = os.path.join(CONFIGS, "map")
ROBOTS_DIR = os.path.join(CONFIGS, "robot")
MACHINES_DIR = os.path.join(CONFIGS, "machine")
SEQUENCE_DIR = os.path.join(CONFIGS, "sequence")
OBJECTS_DIR = os.path.join(CONFIGS, "object")
ASSETS = os.path.join(SIM_DIR, "assets")
ROBOT_ASSETS = os.path.join(ASSETS, "robot")                # <name>/urdf/<name>.urdf + meshes
MACHINE_ASSETS = os.path.join(ASSETS, "machine")            # <name>/urdf/<name>.urdf + meshes
GRIPPER_ASSETS = os.path.join(ASSETS, "grippper")           # <gripper>/urdf/<gripper>.urdf (the folder is spelled like this on disk)
MACHINE_SCRIPTS = os.path.join(ISAAC_DIR, "machine")        # <name>.py: the ROS node that runs the cycle of a machine (started by the launch)
BRINGUP_DIR = os.path.join(SIM_DIR, "bringup")
DATASET_CONFIG = "map"                                      # configs/map/map.yaml = the dataset generator config, not a map
DEFAULT_MAP = "default"                                     # configs/map/default.yaml: the map without map:= (map:=none = no map at all)
# MoveIt speed and joint limits of a robot whose yaml does not set them (see configs/robot/*.yaml)
MOTION_DEFAULTS = {"velocity_scaling": 0.15, "acceleration_scaling": 0.05, "joint_limits_deg": {}}


def option(argv, key):
    """The value of `key:=value` among the command line words argv (the last one wins), None when it is not there"""
    found = None
    for a in argv:
        if a.startswith(key + ":="):
            found = a[len(key) + 2:]
    return found


def _match(name, names):
    """`name` as it is spelled in names, whatever its case (an exact spelling wins); `name` itself when no single name matches"""
    name = str(name)
    if name in names:
        return name
    same = sorted({n for n in names if n.lower() == name.lower()})
    return same[0] if len(same) == 1 else name


# ---------------------------------------------------------------- maps

def map_names():
    return sorted(f[:-5] for f in os.listdir(MAPS_DIR) if f.endswith(".yaml") and f[:-5] != DATASET_CONFIG)


def map_path(name):
    """configs/map/<name>.yaml of map:=<name> (or the path of a yaml); not given / empty = configs/map/default.yaml (if it is there);
    None for no map (none, null)"""
    if name is None or not str(name).strip():
        if not os.path.isfile(os.path.join(MAPS_DIR, f"{DEFAULT_MAP}.yaml")):
            return None
        name = DEFAULT_MAP
    if str(name).strip().lower() in ("none", "null"):
        return None
    name = os.path.expanduser(str(name).strip())
    if name.endswith((".yaml", ".yml")) and os.path.isfile(name):
        return os.path.abspath(name)
    stem = _match(name[:-5] if name.endswith(".yaml") else name, map_names() + [DATASET_CONFIG])   # map:=side_plate_map = SIDE_PLATE_MAP
    if stem == DATASET_CONFIG:
        raise ValueError(f"map:={name}: configs/map/{DATASET_CONFIG}.yaml is the config of the dataset generator, not a map; "
                         f"maps: {', '.join(map_names()) or '-'}")
    path = os.path.join(MAPS_DIR, f"{stem}.yaml")
    if not os.path.isfile(path):
        raise ValueError(f"map:={name}: {path} not found; maps: {', '.join(map_names()) or '-'}")
    return path


def load_map(name):
    """The map of map:=<name> as one tree (its `defaults:` merged in, the paths of mesh / usd keys made absolute); {} for no map"""
    path = map_path(name)
    return _load_yaml_tree(path) if path else {}


def scene_of(name):
    """The `scene:` of the map of map:=<name> ({} for no map)"""
    return load_map(name).get("scene") or {}


# ---------------------------------------------------------------- robots

def robot_names():
    names = set()
    for f in sorted(os.listdir(ROBOTS_DIR)):
        if f.endswith(".yaml"):
            names.add(str((yaml.safe_load(open(os.path.join(ROBOTS_DIR, f))) or {}).get("name") or f[:-5]))
    return sorted(names)


def canonical_robot(name):
    """A robot name as its config file spells it (robot:=M20_35_18D works too); the name itself when it is a path or no single name matches"""
    name = str(name)
    if os.path.sep in name or name.endswith((".yaml", ".yml")):
        return name
    stems = [f[:-5] for f in os.listdir(ROBOTS_DIR) if f.endswith(".yaml")]
    return _match(name, stems + robot_names())


def robot_config_path(choice):
    """The yaml of a robot: the path of a yaml, configs/robot/<choice>.yaml, or the file of that folder whose `name:` is choice (the case of the name does not matter)"""
    if os.path.isfile(choice):
        return os.path.abspath(choice)
    choice = canonical_robot(choice)
    own = os.path.join(ROBOTS_DIR, f"{choice}.yaml")
    if os.path.isfile(own):
        return own
    for f in sorted(os.listdir(ROBOTS_DIR)):
        if f.endswith(".yaml") and (yaml.safe_load(open(os.path.join(ROBOTS_DIR, f))) or {}).get("name") == choice:
            return os.path.join(ROBOTS_DIR, f)
    raise ValueError(f"robot config not found: {choice}; robots: {', '.join(robot_names())}")


def pick_robot(robot=None, scene=None):
    """robot:=<name> when it is given, else scene.robot of the map"""
    name = robot or (scene or {}).get("robot")
    if not name:
        raise ValueError(f"no robot: pass robot:=<name> (or a map:= whose scene.robot names one), or start robot_sim.py first; robots: {', '.join(robot_names())}")
    return canonical_robot(name)


# ---------------------------------------------------------------- the robot and the map robot_sim.py runs

CURRENT_FILE = os.path.join(ISAAC_DIR, "assets_cache", "current_sim.yaml")   # written by robot_sim.py at start


def write_current(robots, map_name):
    """robot_sim.py: the robots ([{name, di}]) and the map (None = no map) it was started with, for the tools started without robot:= / map:="""
    os.makedirs(os.path.dirname(CURRENT_FILE), exist_ok=True)
    with open(CURRENT_FILE, "w") as f:
        yaml.safe_dump({"robot": robots[0]["name"] if robots else None, "robots": robots, "map": map_name or "none"}, f)


def with_current(robot=None, map_name=None):
    """(robot, map name) of a tool: what is given wins; without map:= the map (and, if robot:= is not given either, the robot) of the
    robot_sim.py started last (CURRENT_FILE). The robot may still be None: then the map names it (pick_robot)"""
    if map_name:
        return robot or None, map_name
    cur = (yaml.safe_load(open(CURRENT_FILE)) or {}) if os.path.isfile(CURRENT_FILE) else {}
    return robot or cur.get("robot"), cur.get("map")


def scene_robots(scene, robot=None):
    """The robots of a map: [{name, di, pose}] (di = the ROS domain id of the robot, None = the one of the environment; pose = [x, y, z, roll, pitch, yaw]).
    robot:= alone = that robot at scene.robot_pose; else scene.robots: [{name, di, pose}, ...]; else scene.robot at scene.robot_pose"""
    zero = [0, 0, 0, 0, 0, 0]
    if robot:
        return [{"name": canonical_robot(robot), "di": None, "pose": scene.get("robot_pose") or zero}]
    if scene.get("robots"):
        return [{"name": canonical_robot(r["name"]), "di": r.get("di"), "pose": r.get("pose") or zero} for r in scene["robots"]]
    if scene.get("robot"):
        return [{"name": canonical_robot(scene["robot"]), "di": None, "pose": scene.get("robot_pose") or zero}]
    raise ValueError(f"no robot: pass robot:=<name> (or a map:= with scene.robots / scene.robot), or start robot_sim.py first; robots: {', '.join(robot_names())}")


def robot_entry(scene, robot=None, di=None):
    """The one robot of the map a tool (the launch, ee_6d) is for: the one with ROS domain di (di:=), else the one named robot:=, else the first"""
    rows = scene_robots(scene) if (scene.get("robots") or scene.get("robot")) else []
    robot = canonical_robot(robot) if robot else robot
    if di not in (None, ""):
        hit = [r for r in rows if r["di"] is not None and int(r["di"]) == int(di)]
        if hit:
            return {**hit[0], "name": str(robot or hit[0]["name"])}
        if not robot:
            raise ValueError(f"di:={di}: no robot with that di in the map (robots: {[(r['name'], r['di']) for r in rows]})")
    if robot:
        hit = [r for r in rows if r["name"] == str(robot)]
        return hit[0] if hit else {"name": str(robot), "di": di, "pose": scene.get("robot_pose") or [0, 0, 0, 0, 0, 0]}
    if rows:
        return rows[0]
    return scene_robots(scene)[0]


def expand_crates(crate, crates):
    """`crates:` of a sequence (the crates of the top layer, in the order they are emptied) for every layer of the stack down to the bottom one:
    crate.layers (default 1) times the list, each layer crate.layer_pitch_m lower (its scan pose `home` and its floor_z go down with it).
    Returns [{centre, home, floor_z, level}, ...] in the order of work (level 0 = the top layer)"""
    crate = crate or {}
    n, pitch = int(crate.get("layers", 1)), float(crate.get("layer_pitch_m", 0.0))
    out = []
    for level in range(n):
        for c in crates or []:
            home = [float(v) for v in c["home"]]
            home[2] -= level * pitch
            out.append({**c, "home": home, "floor_z": float(c.get("floor_z", crate.get("floor_z", 0.0))) - level * pitch, "level": level})
    return out


def load_robot(name):
    """configs/robot/<name>.yaml as a dict ({} when there is none)"""
    try:
        return yaml.safe_load(open(robot_config_path(name))) or {}
    except ValueError:
        return {}


def robot_motion(robot_cfg):
    """MoveIt speed and joint limits of a robot yaml: velocity_scaling, acceleration_scaling, joint_limits_deg ({joint: [min, max]} deg)"""
    return {k: robot_cfg[k] if robot_cfg.get(k) is not None else v for k, v in MOTION_DEFAULTS.items()}


def find_urdf(name):
    """assets/robot/<name>/urdf/<name>.urdf, or <name>.urdf in another assets/robot/*/urdf/ folder (several URDFs of one robot share a folder)"""
    own = os.path.join(ROBOT_ASSETS, name, "urdf", f"{name}.urdf")
    if os.path.isfile(own) or not os.path.isdir(ROBOT_ASSETS):
        return own
    found = sorted(os.path.join(ROBOT_ASSETS, d, "urdf", f"{name}.urdf") for d in os.listdir(ROBOT_ASSETS)
                   if os.path.isfile(os.path.join(ROBOT_ASSETS, d, "urdf", f"{name}.urdf")))
    return found[0] if found else own


def find_mesh(f, base):
    """The file of a URDF mesh: relative to the URDF folder `base`, file://..., or package://<pkg>/<path> searched in the URDF folder and
    every folder above it (so a URDF straight out of xacro works). None if missing."""
    f = f or ""
    if f.startswith("package://"):
        rel, d = f[len("package://"):].partition("/")[2], base
        while rel:
            if os.path.isfile(os.path.join(d, rel)):
                return os.path.join(d, rel)
            if os.path.dirname(d) == d:
                break
            d = os.path.dirname(d)
        parts = rel.split("/")                      # not there: the meshes folder next to the urdf folder, with fewer leading folders
        for k in range(len(parts)):                 # (package://pkg/meshes/m710id_70/visual/base.dae -> ../meshes/visual/base.dae)
            p = os.path.join(os.path.dirname(base), "meshes", *[q for q in parts[k:] if q != "meshes"])
            if os.path.isfile(p):
                return p
        return None
    p = f[len("file://"):] if f.startswith("file://") else os.path.join(base, f)
    return p if f and os.path.isfile(p) else None


# ---------------------------------------------------------------- machines

def machine_file(name):
    return os.path.join(MACHINES_DIR, f"{name}.yaml")


def load_machine(name):
    """configs/machine/<name>.yaml as a dict ({} when there is none)"""
    path = machine_file(name)
    return (yaml.safe_load(open(path)) or {}) if os.path.isfile(path) else {}


def machine_config(entry):
    """A machine of scene.machines: configs/machine/<name>.yaml (the machine itself: URDF, materials, collision, carriers, speeds, fixtures, ...)
    with the keys of the map entry on top (where it stands in this map: pos_m, rpy_deg, replaces, or anything else that differs there).
    `config:` of the entry names another file than <name>.yaml. `config_file` = the path of that file (None without one)."""
    name = str(entry.get("config") or entry["name"])
    cfg = deep_merge(load_machine(name), {k: v for k, v in entry.items() if k != "config"})
    cfg["config_file"] = machine_file(name) if os.path.isfile(machine_file(name)) else None
    return cfg


# ---------------------------------------------------------------- poses (rpy = roll about x, pitch about y, yaw about z, fixed axes: URDF)

def rpy_matrix(rpy_deg):
    r, p, y = (math.radians(float(v)) for v in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def matrix_rpy(R):
    if math.hypot(R[0][0], R[1][0]) < 1e-9:                     # pitch +-90 deg: roll and yaw are one angle, yaw = 0
        return [math.degrees(math.atan2(-R[1][2], R[1][1])), math.degrees(math.atan2(-R[2][0], 0.0)), 0.0]
    return [math.degrees(math.atan2(R[2][1], R[2][2])), math.degrees(math.atan2(-R[2][0], math.hypot(R[0][0], R[1][0]))),
            math.degrees(math.atan2(R[1][0], R[0][0]))]


def compose(a, b):
    """(pos, rpy_deg) of the pose b given in the frame of the pose a, in the frame a is given in"""
    (pa, ra), (pb, rb) = a, b
    Ra, Rb = rpy_matrix(ra), rpy_matrix(rb)
    pos = [float(pa[i]) + sum(Ra[i][k] * float(pb[k]) for k in range(3)) for i in range(3)]
    R = [[sum(Ra[i][k] * Rb[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return [round(v, 9) + 0.0 for v in pos], [round(v, 9) + 0.0 for v in matrix_rpy(R)]


def to_world(pose, local):
    """A point given in the frame of pose (pos, rpy_deg), in the world"""
    return compose(pose, (local, [0.0, 0.0, 0.0]))[0]


def platform_top(pf, pos):
    """z of the top of the platform (scene.platform) if pos (x, y) is on its footprint (yaw only), else 0: a robot there stands on it"""
    px, py, pz = (float(v) for v in pf.get("pos_m", [0, 0, 0]))
    sx, sy, sz = (float(v) for v in pf["size_m"])
    yaw = math.radians(float((pf.get("rpy_deg") or [0, 0, 0])[2]))
    dx, dy = pos[0] - px, pos[1] - py
    u, v = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    return pz + sz if abs(u) <= sx / 2 and abs(v) <= sy / 2 else 0.0


def robot_pose(robot_cfg, scene):
    """Where the robot base stands in the world of the map: scene.robot_pose [x, y, z, roll, pitch, yaw] of the map (default 0) times
    pos_m / rpy_deg of the robot yaml (0 for every robot), on top of scene.platform when it stands on one. (pos [m], rpy [deg])"""
    rp = [float(v) for v in (scene.get("robot_pose") or [0, 0, 0, 0, 0, 0])]
    if len(rp) != 6:
        raise ValueError(f"scene.robot_pose needs [x, y, z, roll, pitch, yaw], got {rp}")
    pos, rpy = compose((rp[:3], rp[3:]), (robot_cfg.get("pos_m") or [0, 0, 0], robot_cfg.get("rpy_deg") or [0, 0, 0]))
    pf = scene.get("platform")
    if pf and pf.get("size_m"):
        pos[2] += platform_top(pf, pos)
    return pos, rpy
