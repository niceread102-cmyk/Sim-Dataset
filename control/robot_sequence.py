"""robot_sequence.py: the pick and place sequence: which poses, in which order; control/ik_node.py moves the arm.

    python3 control/robot_sequence.py --sequence <name> [--robot <robot>] [--map <map>]       (robot.launch.py sequence:=<name> starts it)

On /target_pose (TCP of the part): approach -> insert -> open gripper -> rise -> place pose of the part's state -> lower -> close gripper -> home.
A robot with `vacuum:` in its yaml has no gripper joints: open gripper = suction on, close gripper = suction off (/vacuum/command, answer on .../result).
The task is isaacsim/configs/sequence/<name>.yaml (+ robot_sequence.yaml); crates (vision_pick.py) and the transfer to the conveyor are optional.
"""
import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, Empty, Float64, Int32, String

import compute_grasp_tcp_pose as grasp
from ik_node import (COMMAND_TOPIC, CONFIG_FILE, READY_TOPIC, RESULT_TOPIC, dumps, joints, line, move, offset_pose, pack, robot_part,
                     sim_config, tcp_to_flange, via_steps as via_pieces, xyzrpy_to_pose)


REQUIRED_KEYS = {
    'robot':     ['planning_group', 'arm_joints', 'base_link', 'tip_link', 'tool_axis', 'tcp_length'],
    'poses':     ['home', 'place'],
    'approach':  ['distance'],
    'line':      ['service', 'action', 'max_step', 'revolute_jump_threshold', 'min_fraction', 'ik_step', 'ik_angle_step'],
    'ik':        ['service', 'seed', 'timeout', 'retry_timeout', 'avoid_collisions'],
    'startup':   ['poll', 'retries'],
    'tolerance': ['joint', 'waypoint_position'],
    'motion':    ['velocity_scaling', 'acceleration_scaling', 'planning_time', 'planning_attempts', 'retries', 'timeout'],
    'gripper':   ['topic', 'open', 'close', 'enabled'],
    'topics':    ['target_pose', 'move_action', 'joint_states'],
}


def gripper_joint(robot, robot_cfg):
    names = list(robot_cfg.get('gripper_joints') or [])
    gripper = str(robot_cfg.get('gripper') or 'none')
    if gripper.lower() not in ('none', 'null'):
        urdf = Path(sim_config.GRIPPER_ASSETS) / gripper / 'urdf' / f'{gripper}.urdf'
    elif names:
        urdf = Path(sim_config.find_urdf(robot))
    else:
        return None
    joints = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint') if j.get('type') in ('revolute', 'continuous', 'prismatic')}
    names = names or [n for n, j in joints.items() if j.find('mimic') is None]
    if not names or names[0] not in joints:
        raise ValueError(f'gripper joint {names[:1] or "?"} not found in {urdf}')
    lim = joints[names[0]].find('limit')
    return names[0], float(lim.get('lower', 0.0)), float(lim.get('upper', 0.0))


def sequence_names(folder=CONFIG_FILE.parent):
    return sorted(f.stem for f in Path(folder).glob('*.yaml') if f.name != CONFIG_FILE.name)


def load_config(path=CONFIG_FILE, robot=None, sequence=None, map_name=None):
    scene = sim_config.scene_of(map_name)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    own = Path(path).parent / f'{sequence or sim_config.pick_robot(robot, scene)}.yaml'
    if sequence and not own.is_file():
        raise ValueError(f'sequence {sequence!r} not found: {own} (sequences: {", ".join(sequence_names(own.parent)) or "-"})')
    declared = None
    if own.is_file():
        mine = yaml.safe_load(open(own)) or {}
        base = mine.pop('extends', None)
        if base:
            base_file = own.parent / f'{base}.yaml'
            if not base_file.is_file():
                raise ValueError(f'{own.name}: extends {base!r} but {base_file} does not exist (sequences: {", ".join(sequence_names(own.parent))})')
            parent = yaml.safe_load(open(base_file)) or {}
            parent.pop('robot', None)
            cfg = sim_config.deep_merge(cfg, {k: v for k, v in parent.items() if k != 'robots'})
            mine = sim_config.deep_merge({'robots': parent.get('robots') or {}}, mine)
        declared = mine.pop('robot', None)
        per_robot = mine.pop('robots', None) or {}
        cfg = sim_config.deep_merge(cfg, mine)
    else:
        per_robot = {}
    cfg.pop('robot', None)
    if robot and declared and robot != declared:
        raise ValueError(f'{own.name} is the sequence of the robot {declared}, not {robot}')
    robot = sim_config.pick_robot(robot or declared, scene)
    if per_robot.get(robot):
        cfg = sim_config.deep_merge(cfg, per_robot[robot])
    robot_cfg = robot_part(cfg, robot)
    entry = sim_config.robot_entry(scene, robot)
    cfg['robot'].update(base_xy=[float(v) for v in entry['pose'][:2]], base_yaw=math.radians(float(entry['pose'][5])))

    g = cfg.setdefault('gripper', {})
    g['topic'] = str(robot_cfg.get('gripper_topic') or g.get('topic') or '/gripper/command')
    grip = gripper_joint(robot, robot_cfg)
    g['enabled'] = grip is not None
    g['result_topic'] = g['topic'] + '/result'
    cfg['slow_labels'] = ['insert', 'lower'] if robot_cfg.get('slow_speed') else []
    cfg['vacuum'] = None
    if robot_cfg.get('vacuum'):
        topic = str(robot_cfg['vacuum'].get('topic') or '/vacuum/command')
        cfg['vacuum'] = {'topic': topic, 'result_topic': topic + '/result', 'timeout': float(robot_cfg['vacuum'].get('answer_timeout_s', 5.0))}

    poses = cfg.setdefault('poses', {})
    poses.setdefault('home', 'ready')
    poses.setdefault('place', 'home')
    poses.setdefault('release', None)
    poses.setdefault('place_height', None)
    poses.setdefault('place_offset', None)
    missing = [f'{section}.{key}' for section, keys in REQUIRED_KEYS.items() for key in keys
               if key not in (cfg.get(section) or {})]
    if missing:
        raise ValueError(f'{path}: missing keys: {", ".join(missing)}')
    if isinstance(poses['place'], dict):
        poses['place'] = {str(k).strip().lower().replace(' ', '_'): v for k, v in poses['place'].items()}
    for key, value in poses.items():
        for v in (value.values() if isinstance(value, dict) else [value]):
            if v is not None and not isinstance(v, str) and key in ('home', 'place', 'release') and len(v) != 6:
                raise ValueError(f'{path}: poses.{key} needs [x, y, z, roll, pitch, yaw]')
    if poses['place_offset'] is not None:
        if poses['release'] is not None or poses['place_height'] is not None:
            raise ValueError(f'{path}: poses.place_offset, poses.place_height and poses.release all say where the arm goes down to: use one of them')
        try:
            poses['place_offset'] = float(poses['place_offset'])
        except (TypeError, ValueError):
            raise ValueError(f'{path}: poses.place_offset needs a height in metres, got {poses["place_offset"]!r}') from None
        if poses['place_offset'] <= 0:
            raise ValueError(f'{path}: poses.place_offset must be above 0 (the arm comes from above), got {poses["place_offset"]}')
    t = cfg.get('transfer')
    if t is not None:
        for key in ('pick', 'place'):
            if not isinstance(t.get(key), (list, tuple)) or len(t[key]) != 6:
                raise ValueError(f'{path}: transfer.{key} needs [x, y, z, roll, pitch, yaw]')
    height = poses['place_height']
    if height is not None:
        if poses['release'] is not None:
            raise ValueError(f'{path}: poses.place_height and poses.release both say where the arm goes down to: use one of them')
        if str(height).strip().lower() == 'home':
            poses['place_height'] = 'home'
        else:
            try:
                poses['place_height'] = float(height)
            except (TypeError, ValueError):
                raise ValueError(f'{path}: poses.place_height needs home or a z in metres, got {height!r}') from None
    return cfg


def grip(label, value, settle=0.0):
    return {'label': label, 'kind': 'grip', 'value': value, 'search': False, 'settle': settle}


def via_steps(label, start, end, max_step, max_angle):
    return [{'label': st['label'], 'kind': 'ik', 'value': st['value'], 'search': True} for st in via_pieces(label, start, end, max_step, max_angle)]


def heading(rot):
    m = rot.as_matrix()
    return math.atan2(m[1, 0], m[0, 0])


def in_band(angle, ref):
    return ref + (angle - ref + math.pi) % (2 * math.pi) - math.pi


def wrist_angle(pose, base_xy):
    q = pose.orientation
    return math.atan2(pose.position.y - base_xy[1], pose.position.x - base_xy[0]) - heading(Rotation.from_quat([q.x, q.y, q.z, q.w]))


def pan_steps(label, start, end, base_xy, max_step, gap=None, rise_to=None, rise_angle=0.0, ref=None, wrist=False):
    bx, by = base_xy
    a, b = start.position, end.position
    ta, tb = math.atan2(a.y - by, a.x - bx), math.atan2(b.y - by, b.x - bx)
    dt = (tb - ta + math.pi) % (2 * math.pi) - math.pi
    if gap is not None:
        way = 1.0 if dt >= 0 else -1.0
        if ((gap - ta) * way) % (2 * math.pi) < abs(dt):
            dt -= way * 2 * math.pi
    ra, rb = math.hypot(a.x - bx, a.y - by), math.hypot(b.x - bx, b.y - by)
    qa, qb = start.orientation, end.orientation
    rot_a, rot_b = Rotation.from_quat([qa.x, qa.y, qa.z, qa.w]), Rotation.from_quat([qb.x, qb.y, qb.z, qb.w])
    if ref is None:
        dz = 0.0
    elif wrist:
        dz = dt - (in_band(wrist_angle(end, base_xy), ref) - in_band(wrist_angle(start, base_xy), ref))
    else:
        dz = in_band(heading(rot_b), ref) - in_band(heading(rot_a), ref)
    rest = (rot_a.inv() * Rotation.from_euler('z', -dz) * rot_b).as_rotvec()
    angle = abs(dz) + math.sqrt(sum(v * v for v in rest))
    if abs(dt) < 1e-3:
        rise_to = None
    top = a.z if rise_to is None else rise_to
    rise_angle = min(max(rise_angle, 1e-6), abs(dt) * 0.8)
    length = math.hypot(max(ra, rb) * abs(dt), rb - ra, abs(top - a.z) + abs(b.z - top))
    n = max(1, math.ceil(length / max_step), math.ceil(angle / (math.pi / 2)))
    steps, prev = [], start
    for i in range(1, n + 1):
        f = i / n
        t, r = ta + dt * f, ra + (rb - ra) * f
        q = (Rotation.from_euler('z', dz * f) * rot_a * Rotation.from_rotvec(rest * f)).as_quat()
        turned = abs(dt) * f
        if rise_to is None:
            z = a.z + (b.z - a.z) * f
        elif turned < rise_angle:
            z = a.z + (rise_to - a.z) * turned / rise_angle
        else:
            z = rise_to + (b.z - rise_to) * (turned - rise_angle) / (abs(dt) - rise_angle)
        pose = Pose(position=Point(x=bx + r * math.cos(t), y=by + r * math.sin(t), z=z),
                    orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))
        steps.append(line(f'{label} {i}/{n}', pose, prev))
        prev = pose
    return steps


def is_pan(cfg, mode):
    return str((cfg.get('trajectory') or {}).get(mode, '')).lower() == 'pan'


def lift_pose(cfg, approach, carry):
    lz, cz = (cfg.get('trajectory') or {}).get('lift_z'), carry_height(cfg)
    if lz is None or carry is None or cz is None:
        return None
    dz = carry.position.z - (cz - float(lz)) - approach.position.z
    return raised(approach, dz) if dz > 1e-3 else None


def approach_pan(cfg, approach, home, carry):
    lift = lift_pose(cfg, approach, carry)
    if lift is None:
        return pan_down(cfg, 'approach', approach, home, home.position.z)
    return [*pan_down(cfg, 'approach', lift, home, home.position.z), line('approach', approach, lift)]


def pan_down(cfg, label, pose, start, high_z):
    forward = pan(cfg, label, start, pose, rise_to=high_z)
    points = [pose] + [st['value'] for st in forward]
    points.reverse()
    n = len(forward)
    return [line(f'{label} {i}/{n}', points[i], points[i - 1]) for i in range(1, n + 1)]


def pan(cfg, label, pose, start, rise_to=None):
    robot, traj = cfg['robot'], cfg.get('trajectory') or {}
    turn = str(traj.get('pan_turn', 'heading')).lower()
    if turn not in ('heading', 'wrist'):
        raise ValueError(f'trajectory.pan_turn must be heading | wrist, got {turn!r}')
    home = tcp_to_flange(xyzrpy_to_pose(*cfg['poses']['home']), cfg)
    q = home.orientation
    ref = wrist_angle(home, robot['base_xy']) if turn == 'wrist' else heading(Rotation.from_quat([q.x, q.y, q.z, q.w]))
    return pan_steps(label, start, pose, robot['base_xy'], float(traj.get('pan_step', 0.10)), robot['base_yaw'] + math.pi,
                     rise_to, math.radians(float(traj.get('climb_deg', 60.0))), ref, turn == 'wrist')


MOTION_KINDS = ('line', 'move', 'joints', 'ik')


def fuse_steps(steps):
    out, group = [], []

    def flush():
        nonlocal group
        real = [st for st in group if st['kind'] != 'ik']
        if len(real) > 1:
            out.append({'label': '+'.join(st['label'] for st in real), 'kind': 'path', 'steps': group,
                        'settle': group[-1].get('settle', 0.0), 'search': False})
        else:
            out.extend(group)
        group = []
    for st in steps:
        if st['kind'] in MOTION_KINDS and not st.get('slow'):
            group.append(st)
        else:
            flush()
            out.append(st)
    flush()
    return out


def travel(cfg, name):
    return 'line' if str((cfg.get('trajectory') or {}).get(name, 'joint')).lower() == 'line' else 'joint'


def go(cfg, label, pose, start, mode):
    if mode == 'home' and cfg.get('home_joints'):
        return joints(label, cfg['home_joints'])
    return line(label, pose, start) if travel(cfg, mode) == 'line' else move(label, pose, search=True)


def approach_step(cfg, approach, home):
    return go(cfg, 'approach', approach, home, 'approach') if travel(cfg, 'approach') == 'line' else move('approach', approach)


def vac(label, text, on, settle=0.0):
    return {'label': label, 'text': text, 'kind': 'vac', 'value': on, 'search': False, 'settle': settle}


def grips(cfg, *a, **k):
    if cfg.get('vacuum'):
        label, value = a[0], a[1]
        on = abs(float(value) - float(cfg['gripper']['open'])) < 1e-9
        return [vac(label, f"vacuum {'on' if on else 'off'}" + (' (start)' if 'start' in label else ''), on, k.get('settle', 0.0))]
    return [grip(*a, **k)] if cfg['gripper']['enabled'] else []


def place_pose(cfg, variant=None):
    place = cfg['poses']['place']
    if not isinstance(place, dict):
        return place
    return place.get(variant) or place.get('standard') or next(iter(place.values()))


def stays_over_standard(cfg, variant):
    place, t = cfg['poses']['place'], cfg.get('transfer') or {}
    return bool(t.get('stay_over_standard')) and isinstance(place, dict) and (variant != 'turn_over' or bool(t.get('stay_over_turn_over')))


def standard_above(cfg):
    poses = cfg['poses']
    chosen = place_pose(cfg, 'standard')
    if poses.get('place_height') is not None:
        z = poses['home'][2] if poses['place_height'] == 'home' else poses['place_height']
    else:
        z = chosen[2] + float(poses.get('place_offset') or 0.0)
    return tcp_to_flange(xyzrpy_to_pose(chosen[0], chosen[1], z, *chosen[3:]), cfg)


def carry_height(cfg):
    cz = cfg['poses'].get('carry_z')
    if cz is None:
        return None
    return float(cfg['poses']['home'][2]) if str(cz).strip().lower() == 'home' else float(cz)


def raised(pose, dz):
    p = pose.position
    return Pose(position=Point(x=p.x, y=p.y, z=p.z + dz), orientation=pose.orientation)


def home_high(cfg):
    poses = cfg['poses']
    home = tcp_to_flange(xyzrpy_to_pose(*poses['home']), cfg)
    cz = carry_height(cfg)
    return home, (raised(home, cz - float(poses['home'][2])) if cz is not None and cz > float(poses['home'][2]) + 1e-3 else None)


def high_way_home(cfg, here, here_z, home, home_hi, label):
    up = raised(here, carry_height(cfg) - here_z)
    across = pan(cfg, 'above home', home_hi, up) if is_pan(cfg, 'home') else [line('above home', home_hi, up)]
    return [*([line('up to the carry height', up, here)] if carry_height(cfg) > here_z + 1e-3 else []),
            *across, line(label, home, home_hi)]


def make_sequence(target, cfg, variant=None):
    gripper, poses = cfg['gripper'], cfg['poses']
    release_settle = float(gripper.get('release_settle', 0.0))
    traj = cfg.get('trajectory') or {}
    chosen   = place_pose(cfg, variant)
    place_tcp = poses['home'] if chosen == 'home' else chosen
    release_tcp = poses.get('release')
    if chosen != 'home' and poses.get('place_height') is not None:
        z = poses['home'][2] if poses['place_height'] == 'home' else poses['place_height']
        place_tcp, release_tcp = [*chosen[:2], z, *chosen[3:]], chosen
    elif chosen != 'home' and poses.get('place_offset') is not None:
        place_tcp, release_tcp = [chosen[0], chosen[1], chosen[2] + poses['place_offset'], *chosen[3:]], chosen
    flange   = tcp_to_flange(target, cfg)
    approach = offset_pose(flange, -cfg['approach']['distance'], cfg['robot']['tool_axis'])
    carry    = None
    cz = carry_height(cfg)
    if cz is not None and cz > target.position.z + cfg['approach']['distance'] + 1e-3:
        carry = tcp_to_flange(Pose(position=Point(x=target.position.x, y=target.position.y, z=cz), orientation=target.orientation), cfg)
    home, home_hi = home_high(cfg)
    via      = [tcp_to_flange(xyzrpy_to_pose(*v), cfg) for v in (traj.get('via') or [])]
    place    = tcp_to_flange(xyzrpy_to_pose(*place_tcp), cfg)
    release  = tcp_to_flange(xyzrpy_to_pose(*release_tcp), cfg) if release_tcp else None
    at_home  = chosen == 'home'
    stay     = stays_over_standard(cfg, variant) and not at_home
    to_std   = stay and variant == 'turn_over'

    path = [] if at_home else [(f'via {i + 1}', v) for i, v in enumerate(via)] + [('place', place)]
    spiral = carry is not None and not at_home and is_pan(cfg, 'place')
    lift = lift_pose(cfg, approach, carry) if spiral else None
    out, here = [], (lift or approach if spiral else carry or home)
    for label, pose in path:
        if label == 'place' and is_pan(cfg, 'place'):
            out.extend(pan(cfg, label, pose, here, rise_to=carry.position.z if spiral else None))
        else:
            out.append(go(cfg, label, pose, here, 'place' if label == 'place' else 'via'))
        here = pose
    back, there = [], here
    for label, pose in reversed(path[:-1]):
        back.append(go(cfg, 'return ' + label, pose, there, 'via'))
        there = pose
    return [
        *grips(cfg, 'close gripper (start)', gripper['close']),
        *(approach_pan(cfg, approach, home, carry) if is_pan(cfg, 'approach') else
          [*via_steps('approach', home, approach, cfg['line']['ik_step'], cfg['line']['ik_angle_step']),
           approach_step(cfg, approach, home)]),
        line('insert',         flange,   approach),
        *grips(cfg, 'open gripper', gripper['open']),
        line('rise',           approach, flange),
        *([line('lift', lift, approach)] if lift else []),
        *([] if spiral else [line('carry', carry, approach)] if carry else [go(cfg, 'home', home, approach, 'home')]),
        *out,
        *([line('lower',       release,  here),
           *grips(cfg, 'close gripper', gripper['close'], settle=release_settle),
           line('lift',        here,     release)]
          if release else [*grips(cfg, 'close gripper', gripper['close'], settle=release_settle)]),
        *([line('wait above standard', standard_above(cfg), here)] if to_std else []),
        *([] if stay else back),
        *([] if (at_home or stay) else (high_way_home(cfg, there, float(place_tcp[2]), home, home_hi, 'return home') if home_hi is not None
                                        else pan(cfg, 'return home', home, there) if is_pan(cfg, 'home') and not cfg.get('home_joints')
                                        else [go(cfg, 'return home', home, there, 'home')])),
    ]


def make_transfer(cfg, place_tcp=None, above=None):
    head, tail = transfer_steps(cfg, place_tcp, above)
    return head + [{'label': 'measure yaw', 'kind': 'align', 'value': None, 'search': False}] + tail


def transfer_steps(cfg, place_tcp=None, above=None):
    gripper, poses, t = cfg['gripper'], cfg['poses'], cfg['transfer']
    release_settle = float(gripper.get('release_settle', 0.0))
    step, angle = cfg['line']['ik_step'], cfg['line']['ik_angle_step']
    pick     = tcp_to_flange(xyzrpy_to_pose(*t['pick']), cfg)
    approach = offset_pose(pick, -cfg['approach']['distance'], cfg['robot']['tool_axis'])
    x, y, z, *rest = place_tcp or t['place']
    up       = tcp_to_flange(xyzrpy_to_pose(x, y, z + float(poses.get('place_offset') or cfg['approach']['distance']), *rest), cfg)
    down     = tcp_to_flange(xyzrpy_to_pose(x, y, z, *rest), cfg)
    home, home_hi = home_high(cfg)

    insert_a = dict(line('insert', pick, approach), follow_part=True)
    insert_b = dict(line('insert', pick, above), follow_part=True) if above is not None else None
    start    = home_hi or home
    head = [
        *grips(cfg, 'close gripper (start)', gripper['close']),
        *([line('above home', home_hi, home)] if home_hi is not None else []),
        *via_steps('approach', start, approach, step, angle),
        approach_step(cfg, approach, start),
        insert_a,
        *grips(cfg, 'open gripper', gripper['open']),
        dict(line('rise', approach, pick), follow_part=True),
    ] if above is None else [
        insert_b,
        *grips(cfg, 'open gripper', gripper['open']),
        dict(line('rise', approach, pick), follow_part=True),
    ]
    tail = [
        *via_steps('place', approach, up, step, angle),
        go(cfg, 'place', up, approach, 'place'),
        line('lower', down, up),
        *grips(cfg, 'close gripper', gripper['close'], settle=release_settle),
        line('lift', up, down),
        *(high_way_home(cfg, up, z + float(poses.get('place_offset') or cfg['approach']['distance']), home, home_hi, 'home') if home_hi is not None
          else [go(cfg, 'home', home, up, 'home')]),
    ]
    return head, tail


LABEL_WAITS = {'close gripper (start)': 'close_gripper_start', 'approach': 'approach', 'insert': 'insert', 'open gripper': 'open_gripper', 'rise': 'rise',
               'carry': 'carry', 'home': 'home', 'place': 'place', 'lower': 'lower', 'close gripper': 'release', 'lift': 'lift', 'return home': 'return_home'}


def apply_waits(steps, waits):
    for st in steps:
        key = LABEL_WAITS.get(st['label'])
        if key in waits:
            st['settle'] = float(waits[key])
    return steps


class SequenceNode(Node):
    def __init__(self, cfg):
        super().__init__('robot_sequence')
        self.cfg         = cfg
        self.target_sub  = self.create_subscription(Pose, cfg['topics']['target_pose'], self.cb_target, 10)
        self.variant     = None
        self.seq_variant = ''
        self.done_pub    = self.create_publisher(String, cfg['topics'].get('sequence_done', '/sequence_done'), 10)
        self.variant_sub = self.create_subscription(String, cfg['topics'].get('place_variant', '/place_variant'), self.cb_variant, 10)
        self.part_pose   = None
        self.follow      = None
        self.create_subscription(Pose, cfg['topics'].get('target_part_pose', '/target_part_pose'), lambda m: setattr(self, 'part_pose', (m, time.time())), 10)
        self.over_standard = False
        self.transfer_pending = False
        self.ready_sub   = self.create_subscription(Empty, cfg['topics'].get('machine_ready', '/laser_marker/ready'), self.cb_ready, 10) if cfg.get('transfer') else None
        self.transfer_pub = self.create_publisher(Empty, cfg['topics'].get('transfer_done', '/transfer_done'), 10)
        crate = cfg.get('crate') or {}
        self.crates = sim_config.expand_crates(crate, cfg['poses'].get('crates') or cfg.get('crates'))
        self.crate_idx, self.crate_pending = None, None
        self.home0 = None
        self.crate_ready_pub = None
        if self.crates and crate.get('next_topic'):
            self.create_subscription(Int32, crate['next_topic'], self.cb_crate, 10)
            self.crate_ready_pub = self.create_publisher(Int32, crate.get('ready_topic', '/crate_ready'), 10)
        self.turned_over = None
        self.part_sub = self.create_subscription(Bool, cfg['topics'].get('part_turned_over', '/part_turned_over'), self.cb_part,
                                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.cmd_pub     = self.create_publisher(String, COMMAND_TOPIC, 10)
        self.create_subscription(String, RESULT_TOPIC, self.cb_result, 10)
        self.create_subscription(String, READY_TOPIC, self.cb_motion_ready, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.gripper_pub = self.create_publisher(Float64, cfg['gripper']['topic'], 10) if cfg['gripper']['enabled'] else None
        self.ready_seed  = [math.radians(d) for d in cfg['ik']['seed']]
        self.motion_ready = False
        self.moving      = False
        self.cur_settle  = 0.0
        self.steps       = []
        self.name        = ''
        self.init_tries  = 0
        self.cmd_id      = 0
        self.reset_seed  = False
        self.gripper_pending = None
        if self.gripper_pub:
            self.create_subscription(String, cfg['gripper']['result_topic'], self.cb_gripper_result, 10)
        self.vac_pub = self.create_publisher(Bool, cfg['vacuum']['topic'], 10) if cfg.get('vacuum') else None
        self.vac_pending, self.vac_id = None, 0
        if self.vac_pub:
            self.create_subscription(String, cfg['vacuum']['result_topic'], self.cb_vacuum_result, 10)

    def cb_motion_ready(self, msg):
        if self.moving:
            return
        self.motion_ready = True
        if isinstance(self.cfg['poses']['home'], str):
            self.cfg['poses']['home'] = json.loads(msg.data)['home']
            self.cfg['home_joints'] = list(self.ready_seed)
            self.get_logger().info('home pose = TCP of the start pose of the robot: ' + ' '.join(f'{v:.3f}' for v in self.cfg['poses']['home']))
        self.begin_init()

    def begin_init(self):
        self.init_tries += 1
        home = tcp_to_flange(xyzrpy_to_pose(*self.cfg['poses']['home']), self.cfg)
        steps = [grip('close gripper (init)', self.cfg['gripper']['close'])] if self.cfg['gripper']['enabled'] else []
        steps += [vac('close gripper (init)', 'vacuum off (init)', False)] if self.cfg.get('vacuum') else []
        self.start('initialization', steps + [joints('home (init)', self.cfg['home_joints']) if self.cfg.get('home_joints') else move('home (init)', home, search=True)])

    def cb_gripper_result(self, msg):
        done, self.gripper_pending = self.gripper_pending, None
        if done is None:
            return
        self.get_logger().info(f'gripper {msg.data}')
        done()

    def cb_vacuum_result(self, msg):
        pending, self.vac_pending = self.vac_pending, None
        if pending is None:
            return
        on, step = pending
        self.get_logger().info(f'vacuum {msg.data}')
        if on and msg.data != 'held':
            self.fail('vacuum: no part within reach of the cup -> remaining steps cancelled')
            return
        self.later(step.get('settle', 0.0), self.run_next)

    def vacuum_silent(self, vid):
        if self.vac_pending is not None and vid == self.vac_id and self.moving:
            self.vac_pending = None
            self.fail('vacuum: no answer on ' + self.cfg['vacuum']['result_topic'] + ' (is robot_sim.py running this robot?)')

    def cb_variant(self, msg):
        self.variant = str(msg.data).strip().lower().replace(' ', '_') or None
        self.get_logger().info(f'state of the part received: {self.variant}')

    def cb_ready(self, _):
        if self.moving or isinstance(self.cfg['poses']['home'], str):
            self.get_logger().info('the part lies at the standard coordinate; the arm is busy -> the transfer follows when it is free')
            self.transfer_pending = True
            return
        self.start_transfer()

    def start_transfer(self):
        self.follow = None
        t = self.cfg['transfer']
        above = standard_above(self.cfg) if self.over_standard else None
        self.over_standard = False
        self.get_logger().info(f"transfer: {'above the standard coordinate' if above is not None else 'home'} -> pick at the standard coordinate {t['pick']} (bore expanded) -> read the yaw of the plate in the hand, turn the wrist -> put it on the carrier of the belt {t['place']}")
        self.start('transfer', make_transfer(self.cfg, None, above))

    def align_yaw(self):
        t = self.cfg['transfer']
        place = list(t['place'])
        p = self.part_pose[0] if self.part_pose and time.time() - self.part_pose[1] < 2.0 else None
        if t.get('align_yaw', True) and p is not None and math.hypot(p.position.x - t['pick'][0], p.position.y - t['pick'][1]) < 0.08:
            roll, pitch, yaw = Rotation.from_quat([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]).as_euler('XYZ', degrees=True)
            yaw = (yaw + 90.0) % 180.0 - 90.0
            place[5] = float(place[5]) + float(yaw)
            self.get_logger().info(f"yaw of the plate in the hand: {yaw:.1f} deg (rpy {roll:.1f} {pitch:.1f} {yaw:.1f}) -> the tool yaw changes by {yaw:+.1f} (the wrist turns by {-yaw:.1f} deg about the world z), the plate comes down at yaw 0")
        else:
            self.get_logger().warn('no pose of the plate in the hand from the simulation (/target_part_pose): the yaw is not corrected')
        waits = dict(self.cfg.get('waits') or {})
        waits.update(waits.get('transfer') or {})
        self.steps = apply_waits(transfer_steps(self.cfg, place)[1], {k: v for k, v in waits.items() if k in LABEL_WAITS.values()})
        if self.cfg.get('fuse_moves'):
            self.steps = fuse_steps(self.steps)
        self.run_next()

    def cb_crate(self, msg):
        i = int(msg.data)
        if not -1 <= i < len(self.crates):
            self.get_logger().error(f'crate {i}: crates: of the sequence has {len(self.crates)} -> ignored')
            self.crate_ready_pub.publish(Int32(data=-1))
            return
        if self.moving or isinstance(self.cfg['poses']['home'], str):
            self.get_logger().info(f'crate {i}: the arm is busy -> it goes there when it is free')
            self.crate_pending = i
            return
        self.start_crate(i)

    def start_crate(self, i):
        old, old_hi = home_high(self.cfg)
        if self.home0 is None:
            self.home0 = (list(self.cfg['poses']['home']), self.cfg.get('home_joints'))
        if i < 0:
            home, label, final = list(self.home0[0]), 'home', ([joints('home', self.home0[1])] if self.home0[1] else [])
            if self.home0[1]:
                self.cfg['home_joints'] = self.home0[1]
            self.get_logger().info('all crates are done: back to the home pose ' + ' '.join(f'{v:g}' for v in home))
        else:
            home, label, final = [float(v) for v in self.crates[i]['home']], f'crate {i}', []
            self.cfg.pop('home_joints', None)
            self.get_logger().info(f"crate {i} (centre {self.crates[i].get('centre')}): home pose = its scan pose {' '.join(f'{v:g}' for v in home)}")
        self.cfg['poses']['home'] = home
        self.crate_idx = i
        new, new_hi = home_high(self.cfg)
        if is_pan(self.cfg, 'crate'):
            steps = pan(self.cfg, label, new, old) + final
            self.start('crate', fuse_steps(steps) if self.cfg.get('fuse_moves') else steps)
            return
        if carry_height(self.cfg) is None:
            self.start('crate', [move(f'{label} scan pose', new, search=True)] + final)
            return
        up, over = old_hi or old, new_hi or new
        self.start('crate', ([line('up from the crate', up, old)] if old_hi is not None else []) + [line(f'over {label}', over, up)]
                   + ([line(f'{label} scan pose', new, over)] if new_hi is not None else []) + final)

    def cb_part(self, msg):
        self.turned_over = bool(msg.data)
        self.get_logger().info(f"part from the simulation: {'turn_over' if self.turned_over else 'standard'}")

    def cb_target(self, msg):
        if self.moving:
            self.get_logger().warn('busy -> new target ignored')
            return
        if not self.motion_ready:
            self.get_logger().error('ik_node.py is not ready yet (no /motion/ready: move_group or /joint_states is not up)')
            return
        if isinstance(self.cfg['poses']['home'], str):
            self.get_logger().warn('the home pose is not known yet (initialization has not started) -> target ignored')
            return
        place, variant, self.variant = self.cfg['poses']['place'], self.variant, None
        if str(self.cfg.get('target', 'tcp')).lower() == 'object':
            q = msg.orientation
            roll, pitch, yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('XYZ', degrees=True)
            tcp_r, tcp_p, tcp_y, flipped = grasp.tcp_rpy(roll, pitch, yaw)
            msg = xyzrpy_to_pose(msg.position.x, msg.position.y, msg.position.z, tcp_r, tcp_p, tcp_y)
            variant = variant or ('turn_over' if flipped else 'standard')
            self.get_logger().info(f"object rpy {roll:.1f} {pitch:.1f} {yaw:.1f} -> {'turned over' if flipped else 'standard'}, TCP rpy {tcp_r:.1f} {tcp_p:.1f} {tcp_y:.1f}")
        if isinstance(place, dict):
            if self.turned_over is not None and str(self.cfg.get('target', 'tcp')).lower() != 'object':
                sim = 'turn_over' if self.turned_over else 'standard'
                if variant and variant != sim:
                    self.get_logger().warn(f"ee_6d.py says {variant!r}, the simulation says {sim!r} -> {sim}")
                variant = sim
            if not variant:
                self.get_logger().warn(f"no state of the part received (click the part in the simulation, or the word after the pose in ee_6d.py: {' | '.join(place)}) -> using standard")
            if variant and variant not in place:
                self.get_logger().warn(f"no place pose for {variant!r} (poses.place: {', '.join(place)}) -> using standard")
            self.get_logger().info(f"part: {variant if variant in place else 'standard'} -> place {place_pose(self.cfg, variant)}")
        self.seq_variant = ('turn_over' if variant == 'turn_over' and 'turn_over' in place else 'standard') if isinstance(place, dict) else ''
        self.start('sequence', make_sequence(msg, self.cfg, variant))

    def start(self, name, steps):
        for st in steps:
            if st['label'] in self.cfg.get('slow_labels', []):
                st['slow'] = True
        waits = dict(self.cfg.get('waits') or {})
        if name == 'transfer':
            waits.update(waits.get('transfer') or {})
        if name in ('sequence', 'transfer'):
            steps = apply_waits(steps, {k: v for k, v in waits.items() if k in LABEL_WAITS.values()})
            if self.cfg.get('fuse_moves'):
                steps = fuse_steps(steps)
        self.name   = name
        self.steps  = steps
        self.reset_seed = True
        self.moving = True
        self.run_next()

    def run_next(self):
        if not self.steps:
            self.get_logger().info(f'{self.name} done')
            self.moving = False
            self.over_standard = self.name == 'sequence' and stays_over_standard(self.cfg, self.seq_variant)
            if self.name == 'sequence':
                self.done_pub.publish(String(data=self.seq_variant))
            elif self.name == 'transfer':
                self.transfer_pub.publish(Empty())
            elif self.name == 'recovery':
                self.done_pub.publish(String(data='failed'))
            elif self.name == 'crate':
                self.crate_ready_pub.publish(Int32(data=self.crate_idx))
            if self.transfer_pending and self.name in ('sequence', 'transfer', 'recovery', 'crate'):
                self.transfer_pending = False
                self.start_transfer()
            elif self.crate_pending is not None:
                i, self.crate_pending = self.crate_pending, None
                self.start_crate(i)
            return
        step = self.steps.pop(0)
        self.execute(step)

    def later(self, seconds, fn):
        if seconds <= 0:
            fn()
            return
        timer = None

        def fire():
            timer.cancel()
            self.destroy_timer(timer)
            fn()
        timer = self.create_timer(seconds, fire)

    def fail(self, message):
        self.get_logger().error(message)
        self.over_standard = False
        self.steps  = []
        self.moving = False
        self.gripper_pending = None
        self.vac_pending = None
        if self.name == 'initialization':
            retries = self.cfg['startup']['retries']
            if self.init_tries < retries:
                self.get_logger().warn(f'trying the initialization again ({self.init_tries}/{retries})')
                self.later(self.cfg['startup']['poll'], self.begin_init)
            else:
                self.get_logger().error('initialization failed: could not reach the home pose (a target still starts the sequence)')
        elif self.name in ('sequence', 'transfer') and self.cfg['motion'].get('recover_home', True):
            self.recover()
        elif self.name == 'recovery':
            self.get_logger().error('recovery failed: the arm could not go back to the home pose (a target still starts the sequence)')
        elif self.name == 'crate':
            self.get_logger().error(f'crate {self.crate_idx}: its scan pose was not reached')
            self.crate_ready_pub.publish(Int32(data=-1))

    def recover(self):
        self.get_logger().warn('the sequence failed -> back to the home pose')
        if self.cfg.get('home_joints'):
            step = joints('home (recovery)', self.cfg['home_joints'])
        else:
            step = move('home (recovery)', tcp_to_flange(xyzrpy_to_pose(*self.cfg['poses']['home']), self.cfg), search=True)
        self.start('recovery', [step])

    def execute(self, step):
        self.cur_settle = step.get('settle', 0.0)
        if step['kind'] == 'align':
            self.align_yaw()
            return
        if step['kind'] == 'grip':
            self.get_logger().info(f"[{step['label']}]")
            self.gripper_pending = lambda: self.later(step.get('settle', 0.0), self.run_next)
            self.gripper_pub.publish(Float64(data=step['value']))
            return
        if step['kind'] == 'vac':
            self.get_logger().info(f"[{step['text']}]")
            self.vac_id += 1
            self.vac_pending = (bool(step['value']), step)
            self.vac_pub.publish(Bool(data=bool(step['value'])))
            vid = self.vac_id
            self.later(self.cfg['vacuum']['timeout'], lambda: self.vacuum_silent(vid))
            return
        self.send_motion(step)

    def send_motion(self, step):
        if self.cmd_pub.get_subscription_count() == 0:
            self.fail('no ik_node.py is listening on ' + COMMAND_TOPIC)
            return
        if step['kind'] == 'path':
            step = {**step, 'steps': [self.follow_part(s) for s in step['steps']]}
        else:
            step = self.follow_part(step)
        self.cmd_id += 1
        self.cmd_pub.publish(String(data=dumps({'id': self.cmd_id, 'reset_seed': self.reset_seed, 'step': pack(step)})))
        self.reset_seed = False

    def cb_result(self, msg):
        data = json.loads(msg.data)
        if data['id'] != self.cmd_id or not self.moving:
            return
        if not data['ok']:
            self.fail(f"{data['msg']} -> remaining steps cancelled")
            return
        self.get_logger().info('arrived' + (f' (waiting {self.cur_settle:g} s)' if self.cur_settle > 0 else ''))
        self.later(self.cur_settle, self.run_next)

    def follow_part(self, step, queue=None):
        t = self.cfg.get('transfer') or {}
        if not step.get('follow_part') or not t.get('follow_part_xy'):
            return step
        if self.follow is None:
            self.follow = (0.0, 0.0)
            p = self.part_pose[0] if self.part_pose and time.time() - self.part_pose[1] < 2.0 else None
            if p is None:
                self.get_logger().warn('no pose of the plate at the standard coordinate (/target_part_pose): it is picked at transfer.pick')
            else:
                dx, dy = p.position.x - float(t['pick'][0]), p.position.y - float(t['pick'][1])
                if math.hypot(dx, dy) > 0.03:
                    self.get_logger().warn(f'the plate lies {math.hypot(dx, dy) * 1000:.0f} mm from transfer.pick: more than 30 mm, it is picked at transfer.pick')
                else:
                    self.follow = (dx, dy)
                    self.get_logger().info(f'the plate at the standard coordinate lies {dx * 1000:+.1f} mm (x) / {dy * 1000:+.1f} mm (y) from transfer.pick -> picked there')
        dx, dy = self.follow
        v = step['value']
        return {**step, 'follow_part': False, 'value': Pose(position=Point(x=v.position.x + dx, y=v.position.y + dy, z=v.position.z), orientation=v.orientation)}


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot', help='robot name (bringup/<robot>/ and isaacsim/configs/robot/<robot>.yaml); default: the robot of the sequence, '
                                        'else scene.robot of the map')
    parser.add_argument('--sequence', help='sequence name = isaacsim/configs/sequence/<name>.yaml (poses, trajectory and what else differs for a robot)')
    parser.add_argument('--map', help='map name = isaacsim/configs/map/<map>.yaml: only its robot is read (scene.robot, the default of --robot)')
    cli, ros_args = parser.parse_known_args(args)
    map_name = cli.map or sim_config.with_current(None, None)[1]
    cfg = load_config(robot=cli.robot, sequence=cli.sequence, map_name=map_name)
    rclpy.init(args=ros_args)
    node = SequenceNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
