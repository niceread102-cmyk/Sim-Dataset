"""sim_config.py: where the config files are and how they are read (maps, robots, machines, objects of isaacsim/configs/), shared by robot_sim.py,
the tools, control/*.py and the launch, so they all read the same map, robot pose and machine settings.
"""
import math
import os

import yaml

from binsim.config import _load_yaml_tree, deep_merge

ISAAC_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(ISAAC_DIR)
CONFIGS = os.path.join(ISAAC_DIR, "configs")
MAPS_DIR = os.path.join(CONFIGS, "map")
ROBOTS_DIR = os.path.join(CONFIGS, "robot")
MACHINES_DIR = os.path.join(CONFIGS, "machine")
SEQUENCE_DIR = os.path.join(CONFIGS, "sequence")
OBJECTS_DIR = os.path.join(CONFIGS, "object")
ASSETS = os.path.join(SIM_DIR, "assets")
ROBOT_ASSETS = os.path.join(ASSETS, "robot")
MACHINE_ASSETS = os.path.join(ASSETS, "machine")
GRIPPER_ASSETS = os.path.join(ASSETS, "grippper")
MACHINE_SCRIPTS = os.path.join(ISAAC_DIR, "machine")
BRINGUP_DIR = os.path.join(SIM_DIR, "bringup")
DATASET_CONFIG = "map"
DEFAULT_MAP = "default"
MOTION_DEFAULTS = {"velocity_scaling": 0.15, "acceleration_scaling": 0.05, "joint_limits_deg": {}}


def option(argv, key):
    found = None
    for a in argv:
        if a.startswith(key + ":="):
            found = a[len(key) + 2:]
    return found


def _match(name, names):
    name = str(name)
    if name in names:
        return name
    same = sorted({n for n in names if n.lower() == name.lower()})
    return same[0] if len(same) == 1 else name


def map_names():
    return sorted(f[:-5] for f in os.listdir(MAPS_DIR) if f.endswith(".yaml") and f[:-5] != DATASET_CONFIG)


def map_path(name):
    if name is None or not str(name).strip():
        if not os.path.isfile(os.path.join(MAPS_DIR, f"{DEFAULT_MAP}.yaml")):
            return None
        name = DEFAULT_MAP
    if str(name).strip().lower() in ("none", "null"):
        return None
    name = os.path.expanduser(str(name).strip())
    if name.endswith((".yaml", ".yml")) and os.path.isfile(name):
        return os.path.abspath(name)
    stem = _match(name[:-5] if name.endswith(".yaml") else name, map_names() + [DATASET_CONFIG])
    if stem == DATASET_CONFIG:
        raise ValueError(f"map:={name}: configs/map/{DATASET_CONFIG}.yaml is the config of the dataset generator, not a map; "
                         f"maps: {', '.join(map_names()) or '-'}")
    path = os.path.join(MAPS_DIR, f"{stem}.yaml")
    if not os.path.isfile(path):
        raise ValueError(f"map:={name}: {path} not found; maps: {', '.join(map_names()) or '-'}")
    return path


def load_map(name):
    path = map_path(name)
    return _load_yaml_tree(path) if path else {}


def scene_of(name):
    return load_map(name).get("scene") or {}


def robot_names():
    names = set()
    for f in sorted(os.listdir(ROBOTS_DIR)):
        if f.endswith(".yaml"):
            names.add(str((yaml.safe_load(open(os.path.join(ROBOTS_DIR, f))) or {}).get("name") or f[:-5]))
    return sorted(names)


def canonical_robot(name):
    name = str(name)
    if os.path.sep in name or name.endswith((".yaml", ".yml")):
        return name
    stems = [f[:-5] for f in os.listdir(ROBOTS_DIR) if f.endswith(".yaml")]
    return _match(name, stems + robot_names())


def robot_config_path(choice):
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
    name = robot or (scene or {}).get("robot")
    if not name:
        raise ValueError(f"no robot: pass robot:=<name> (or a map:= whose scene.robot names one), or start robot_sim.py first; robots: {', '.join(robot_names())}")
    return canonical_robot(name)


CURRENT_FILE = os.path.join(ISAAC_DIR, "assets_cache", "current_sim.yaml")


def single_instance(name, why):
    import fcntl
    lock = open(f'/tmp/{name}_{os.environ.get("ROS_DOMAIN_ID", "0")}.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(why) from None
    return lock


def write_current(robots, map_name):
    os.makedirs(os.path.dirname(CURRENT_FILE), exist_ok=True)
    with open(CURRENT_FILE, "w") as f:
        yaml.safe_dump({"robot": robots[0]["name"] if robots else None, "robots": robots, "map": map_name or "none"}, f)


def with_current(robot=None, map_name=None):
    if map_name:
        return robot or None, map_name
    cur = (yaml.safe_load(open(CURRENT_FILE)) or {}) if os.path.isfile(CURRENT_FILE) else {}
    return robot or cur.get("robot"), cur.get("map")


def scene_robots(scene, robot=None):
    zero = [0, 0, 0, 0, 0, 0]
    if robot:
        return [{"name": canonical_robot(robot), "id": None, "pose": scene.get("robot_pose") or zero}]
    if scene.get("robots"):
        return [{"name": canonical_robot(r["name"]), "id": r.get("id"), "pose": r.get("pose") or zero} for r in scene["robots"]]
    if scene.get("robot"):
        return [{"name": canonical_robot(scene["robot"]), "id": None, "pose": scene.get("robot_pose") or zero}]
    raise ValueError(f"no robot: pass robot:=<name> (or a map:= with scene.robots / scene.robot), or start robot_sim.py first; robots: {', '.join(robot_names())}")


def robot_entry(scene, robot=None, id=None):
    rows = scene_robots(scene) if (scene.get("robots") or scene.get("robot")) else []
    robot = canonical_robot(robot) if robot else robot
    if id not in (None, ""):
        hit = [r for r in rows if r["id"] is not None and int(r["id"]) == int(id)]
        if hit:
            return {**hit[0], "name": str(robot or hit[0]["name"])}
        if not robot:
            raise ValueError(f"id:={id}: no robot with that id in the map (robots: {[(r['name'], r['id']) for r in rows]})")
    if robot:
        hit = [r for r in rows if r["name"] == str(robot)]
        return hit[0] if hit else {"name": str(robot), "id": id, "pose": scene.get("robot_pose") or [0, 0, 0, 0, 0, 0]}
    if rows:
        return rows[0]
    return scene_robots(scene)[0]


def expand_crates(crate, crates):
    crate = crate or {}
    n, pitch = int(crate.get("layers", 1)), float(crate.get("layer_pitch_m", 0.0))
    crates = list(crates or [])
    per = int(crate.get("per_pallet") or 0)
    groups = [crates[i:i + per] for i in range(0, len(crates), per)] if per > 0 else [crates]
    out = []
    for g, group in enumerate(groups):
        for level in range(n):
            for c in group:
                home = [float(v) for v in c["home"]]
                home[2] -= level * pitch
                out.append({**c, "home": home, "floor_z": float(c.get("floor_z", crate.get("floor_z", 0.0))) - level * pitch, "level": level, "group": g})
    return out


def load_robot(name):
    try:
        return yaml.safe_load(open(robot_config_path(name))) or {}
    except ValueError:
        return {}


def robot_motion(robot_cfg):
    return {k: robot_cfg[k] if robot_cfg.get(k) is not None else v for k, v in MOTION_DEFAULTS.items()}


def find_urdf(name):
    own = os.path.join(ROBOT_ASSETS, name, "urdf", f"{name}.urdf")
    if os.path.isfile(own) or not os.path.isdir(ROBOT_ASSETS):
        return own
    found = sorted(os.path.join(ROBOT_ASSETS, d, "urdf", f"{name}.urdf") for d in os.listdir(ROBOT_ASSETS)
                   if os.path.isfile(os.path.join(ROBOT_ASSETS, d, "urdf", f"{name}.urdf")))
    return found[0] if found else own


def find_mesh(f, base):
    f = f or ""
    if f.startswith("package://"):
        rel, d = f[len("package://"):].partition("/")[2], base
        while rel:
            if os.path.isfile(os.path.join(d, rel)):
                return os.path.join(d, rel)
            if os.path.dirname(d) == d:
                break
            d = os.path.dirname(d)
        parts = rel.split("/")
        for k in range(len(parts)):
            p = os.path.join(os.path.dirname(base), "meshes", *[q for q in parts[k:] if q != "meshes"])
            if os.path.isfile(p):
                return p
        return None
    p = f[len("file://"):] if f.startswith("file://") else os.path.join(base, f)
    return p if f and os.path.isfile(p) else None


def machine_file(name):
    return os.path.join(MACHINES_DIR, f"{name}.yaml")


def load_machine(name):
    path = machine_file(name)
    return (yaml.safe_load(open(path)) or {}) if os.path.isfile(path) else {}


def machine_config(entry):
    name = str(entry.get("config") or entry["name"])
    cfg = deep_merge(load_machine(name), {k: v for k, v in entry.items() if k != "config"})
    cfg["config_file"] = machine_file(name) if os.path.isfile(machine_file(name)) else None
    return cfg


def rpy_matrix(rpy_deg):
    r, p, y = (math.radians(float(v)) for v in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def matrix_rpy(R):
    if math.hypot(R[0][0], R[1][0]) < 1e-9:
        return [math.degrees(math.atan2(-R[1][2], R[1][1])), math.degrees(math.atan2(-R[2][0], 0.0)), 0.0]
    return [math.degrees(math.atan2(R[2][1], R[2][2])), math.degrees(math.atan2(-R[2][0], math.hypot(R[0][0], R[1][0]))),
            math.degrees(math.atan2(R[1][0], R[0][0]))]


def compose(a, b):
    (pa, ra), (pb, rb) = a, b
    Ra, Rb = rpy_matrix(ra), rpy_matrix(rb)
    pos = [float(pa[i]) + sum(Ra[i][k] * float(pb[k]) for k in range(3)) for i in range(3)]
    R = [[sum(Ra[i][k] * Rb[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return [round(v, 9) + 0.0 for v in pos], [round(v, 9) + 0.0 for v in matrix_rpy(R)]


def to_world(pose, local):
    return compose(pose, (local, [0.0, 0.0, 0.0]))[0]


def platform_top(pf, pos):
    px, py, pz = (float(v) for v in pf.get("pos_m", [0, 0, 0]))
    sx, sy, sz = (float(v) for v in pf["size_m"])
    yaw = math.radians(float((pf.get("rpy_deg") or [0, 0, 0])[2]))
    dx, dy = pos[0] - px, pos[1] - py
    u, v = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    return pz + sz if abs(u) <= sx / 2 and abs(v) <= sy / 2 else 0.0


def robot_pose(robot_cfg, scene):
    rp = [float(v) for v in (scene.get("robot_pose") or [0, 0, 0, 0, 0, 0])]
    if len(rp) != 6:
        raise ValueError(f"scene.robot_pose needs [x, y, z, roll, pitch, yaw], got {rp}")
    pos, rpy = compose((rp[:3], rp[3:]), (robot_cfg.get("pos_m") or [0, 0, 0], robot_cfg.get("rpy_deg") or [0, 0, 0]))
    pf = scene.get("platform")
    if pf and pf.get("size_m"):
        pos[2] += platform_top(pf, pos)
    return pos, rpy
