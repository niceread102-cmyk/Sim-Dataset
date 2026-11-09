"""cell_config.py: paths and yaml loaders of the robot control part (control/ + bringup/); nothing here needs Isaac Sim.

Robot: bringup/<robot>/config/robot.yaml (+ ik_node.yaml, MoveIt files).  Sequence: control/sequence/<name>.yaml (robot, robot_pose of its base, poses, crate_list).
The URDF and meshes are read from assets/robot/<robot>/urdf (folder of the assets: environment variable ROBOT_ASSETS_DIR).
"""
import copy
import math
import os
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

CONTROL_DIR = Path(__file__).resolve().parent
SIM_DIR = CONTROL_DIR.parent
BRINGUP_DIR = SIM_DIR / 'bringup'
SEQUENCE_DIR = CONTROL_DIR / 'sequence'
ASSETS = Path(os.environ.get('ROBOT_ASSETS_DIR') or SIM_DIR / 'assets')
ROBOT_ASSETS = ASSETS / 'robot'
GRIPPER_ASSETS = ASSETS / 'grippper'
MOTION_DEFAULTS = {'velocity_scaling': 0.15, 'acceleration_scaling': 0.05, 'joint_limits_deg': {}}


def option(argv, key):
    found = None
    for a in argv:
        if a.startswith(key + ':='):
            found = a[len(key) + 2:]
    return found


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def robot_names():
    return sorted(d.name for d in BRINGUP_DIR.iterdir() if (d / 'config' / 'robot.yaml').is_file())


def canonical_robot(name):
    names = robot_names()
    name = str(name)
    if name in names:
        return name
    same = sorted({n for n in names if n.lower() == name.lower()})
    return same[0] if len(same) == 1 else name


def pick_robot(robot=None):
    if not robot:
        raise ValueError(f'no robot: pass robot:=<name> or a sequence that names one; robots: {", ".join(robot_names())}')
    return canonical_robot(robot)


def robot_config_path(name):
    path = BRINGUP_DIR / canonical_robot(name) / 'config' / 'robot.yaml'
    if not path.is_file():
        raise ValueError(f'robot config not found: {path}; robots: {", ".join(robot_names())}')
    return path


def load_robot(name):
    try:
        return yaml.safe_load(open(robot_config_path(name))) or {}
    except ValueError:
        return {}


def robot_motion(robot_cfg):
    return {k: robot_cfg[k] if robot_cfg.get(k) is not None else v for k, v in MOTION_DEFAULTS.items()}


def sequence_names():
    return sorted(f.stem for f in SEQUENCE_DIR.glob('*.yaml') if f.name != 'robot_sequence.yaml')


def load_sequence_file(name):
    path = SEQUENCE_DIR / f'{name}.yaml'
    if not path.is_file():
        raise ValueError(f'sequence {name!r} not found: {path} (sequences: {", ".join(sequence_names()) or "-"})')
    return yaml.safe_load(open(path)) or {}


def sequence_value(name, robot, key):
    mine = load_sequence_file(name)
    chain = []
    if mine.get('extends'):
        parent = load_sequence_file(mine['extends'])
        chain += [parent, (parent.get('robots') or {}).get(robot) or {}]
    chain += [mine, (mine.get('robots') or {}).get(robot) or {}]
    found = None
    for part in chain:
        if part.get(key) is not None:
            found = part[key]
    return found


def sequence_of(name=None):
    name = name or os.environ.get('ROBOT_SEQUENCE')
    if not name:
        raise ValueError(f'no sequence: pass --sequence <name> (or set ROBOT_SEQUENCE); sequences: {", ".join(sequence_names()) or "-"}')
    return str(name)


def find_urdf(name):
    own = ROBOT_ASSETS / name / 'urdf' / f'{name}.urdf'
    if own.is_file() or not ROBOT_ASSETS.is_dir():
        return str(own)
    found = sorted(d / 'urdf' / f'{name}.urdf' for d in ROBOT_ASSETS.iterdir() if (d / 'urdf' / f'{name}.urdf').is_file())
    return str(found[0] if found else own)


def find_mesh(f, base):
    f = f or ''
    if f.startswith('package://'):
        rel, d = f[len('package://'):].partition('/')[2], base
        while rel:
            if os.path.isfile(os.path.join(d, rel)):
                return os.path.join(d, rel)
            if os.path.dirname(d) == d:
                break
            d = os.path.dirname(d)
        parts = rel.split('/')
        for k in range(len(parts)):
            p = os.path.join(os.path.dirname(base), 'meshes', *[q for q in parts[k:] if q != 'meshes'])
            if os.path.isfile(p):
                return p
        return None
    p = f[len('file://'):] if f.startswith('file://') else os.path.join(base, f)
    return p if f and os.path.isfile(p) else None


def expand_crates(crate, crates):
    crate = crate or {}
    n, pitch = int(crate.get('layers', 1)), float(crate.get('layer_pitch_m', 0.0))
    crates = list(crates or [])
    per = int(crate.get('per_pallet') or 0)
    groups = [crates[i:i + per] for i in range(0, len(crates), per)] if per > 0 else [crates]
    out = []
    for g, group in enumerate(groups):
        for level in range(n):
            for c in group:
                home = [float(v) for v in c['home']]
                home[2] -= level * pitch
                out.append({**c, 'home': home, 'floor_z': float(c.get('floor_z', crate.get('floor_z', 0.0))) - level * pitch, 'level': level, 'group': g})
    return out


def base_pose(robot_cfg, pose):
    pose = [float(v) for v in (pose or [0, 0, 0, 0, 0, 0])]
    if len(pose) != 6:
        raise ValueError(f'robot_pose needs [x, y, z, roll, pitch, yaw], got {pose}')
    own = Rotation.from_euler('xyz', robot_cfg.get('rpy_deg') or [0, 0, 0], degrees=True)
    rot = Rotation.from_euler('xyz', pose[3:], degrees=True)
    pos = np.array(pose[:3]) + rot.apply(robot_cfg.get('pos_m') or [0, 0, 0])
    rpy = (rot * own).as_euler('xyz', degrees=True)
    return [round(float(v), 9) + 0.0 for v in pos], [round(float(v), 9) + 0.0 for v in rpy]
