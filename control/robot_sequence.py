"""robot_sequence.py: the pick and place task: which coordinates, in which order; control/path_planner.py draws the paths, control/ik_node.py moves the arm.

    python3 control/robot_sequence.py --sequence <name> [--robot <robot>]       (robot.launch.py sequence:=<name> starts it)

On /target_pose (TCP of the part): approach -> insert -> open gripper -> rise -> place pose of the part's state -> lower -> close gripper -> home.
A robot with `vacuum:` in its yaml has no gripper joints: open gripper = suction on, close gripper = suction off (/vacuum/command, answer on .../result).
The coordinates are control/sequence/<name>.yaml (+ robot_sequence.yaml); crates (vision_pick.py) and the transfer to the conveyor are optional.
"""
import argparse
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, Empty, Float64, Int32, String

import compute_grasp_tcp_pose as grasp
import cell_config
import path_planner as pp
from ik_node import COMMAND_TOPIC, READY_TOPIC, RESULT_TOPIC, dumps, move, pack, xyzrpy_to_pose

CONFIG_FILE = cell_config.SEQUENCE_DIR / 'robot_sequence.yaml'


REQUIRED_KEYS = {
    'poses':     ['home', 'place'],
    'approach':  ['distance'],
    'startup':   ['poll', 'retries'],
    'gripper':   ['topic', 'open', 'close', 'enabled'],
    'topics':    ['target_pose'],
}
PATH_KEYS = ('trajectory', 'fuse_moves', 'fuse_joint_step')


def gripper_joint(robot, robot_cfg):
    names = list(robot_cfg.get('gripper_joints') or [])
    gripper = str(robot_cfg.get('gripper') or 'none')
    if gripper.lower() not in ('none', 'null'):
        urdf = Path(cell_config.GRIPPER_ASSETS) / gripper / 'urdf' / f'{gripper}.urdf'
    elif names:
        urdf = Path(cell_config.find_urdf(robot))
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


def load_config(path=CONFIG_FILE, robot=None, sequence=None):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    own = Path(path).parent / f'{sequence or cell_config.pick_robot(robot)}.yaml'
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
            cfg = cell_config.deep_merge(cfg, {k: v for k, v in parent.items() if k != 'robots'})
            mine = cell_config.deep_merge({'robots': parent.get('robots') or {}}, mine)
        declared = mine.pop('robot', None)
        per_robot = mine.pop('robots', None) or {}
        cfg = cell_config.deep_merge(cfg, mine)
    else:
        per_robot = {}
    cfg.pop('robot', None)
    if robot and declared and robot != declared:
        raise ValueError(f'{own.name} is the sequence of the robot {declared}, not {robot}')
    robot = cell_config.pick_robot(robot or declared)
    if per_robot.get(robot):
        cfg = cell_config.deep_merge(cfg, per_robot[robot])
    mixed = [k for k in PATH_KEYS if k in cfg]
    if mixed:
        raise ValueError(f'{own.name}: {mixed} describe the shape of the paths: they belong to control/config/path_planner.yaml, a sequence only gives coordinates')
    robot_cfg = cell_config.load_robot(robot)
    pose = cfg.pop('robot_pose', None) or [0, 0, 0, 0, 0, 0]
    cfg['robot'] = {'name': robot, 'base_xy': [float(v) for v in pose[:2]], 'base_yaw': math.radians(float(pose[5]))}
    cfg['trajectory'] = pp.load_settings()
    cfg['trajectory'].update(lift_z=(cfg.get('poses') or {}).get('lift_z'), via=(cfg.get('poses') or {}).get('via'),
                             crate_tuck_radius_m=(cfg.get('cell') or {}).get('tuck_radius_m'))
    if (cfg.get('crate') or {}).get('home_from_crate') and cfg.get('crate_list'):
        cfg['poses']['home'] = [float(v) for v in cfg['crate_list'][0]['home']]

    g = cfg.setdefault('gripper', {})
    g['topic'] = str(robot_cfg.get('gripper_topic') or g.get('topic') or '/gripper/command')
    grip = gripper_joint(robot, robot_cfg)
    g['enabled'] = grip is not None
    g['result_topic'] = g['topic'] + '/result'
    cfg['slow_labels'] = [str(v) for v in robot_cfg.get('slow_steps') or []]
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
        self.machine_state, self.target_pending, self.target_wait = None, None, None
        self.zone, self.zone_pub = None, None
        if (cfg.get('transfer') or {}).get('zone_topic'):
            self.zone_pub = self.create_publisher(String, str(cfg['transfer']['zone_topic']),
                                                  QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.set_zone([])
        if (cfg.get('transfer') or {}).get('state_topic'):
            self.create_subscription(String, str(cfg['transfer']['state_topic']), self.cb_machine_state,
                                     QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.transfer_pub = self.create_publisher(Empty, cfg['topics'].get('transfer_done', '/transfer_done'), 10)
        crate = cfg.get('crate') or {}
        self.crates = cfg.get('crate_list') or cell_config.expand_crates(crate, cfg['poses'].get('crates') or cfg.get('crates'))
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
        self.motion_ready = False
        self.moving      = False
        self.cur_settle  = 0.0
        self.cur_check   = False
        self.pick_args, self.alternatives = None, []
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
            self.get_logger().info('home pose = TCP of the start pose of the robot: ' + ' '.join(f'{v:.3f}' for v in self.cfg['poses']['home']))
        self.begin_init()

    def begin_init(self):
        self.init_tries += 1
        home = xyzrpy_to_pose(*self.cfg['poses']['home'])
        self.start('initialization', [pp.tool('close gripper (init)', False), move('home (init)', home)])

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
            self.fail('vacuum: no answer on ' + self.cfg['vacuum']['result_topic'] + ' (is the vacuum driver running: robot_sim.py in the simulation, the I/O node on the real cell?)')

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
        above = pp.standard_above(self.cfg) if self.over_standard else None
        self.over_standard = False
        self.get_logger().info(f"transfer: {'above the standard coordinate' if above is not None else 'home'} -> pick at the standard coordinate {t['pick']} (bore expanded) -> read the yaw of the plate in the hand, turn the wrist -> put it on the carrier of the belt {t['place']}")
        self.start('transfer', pp.make_transfer(self.cfg, None, above))

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
        self.steps = apply_waits(pp.transfer_steps(self.cfg, place)[1], {k: v for k, v in waits.items() if k in LABEL_WAITS.values()})
        if self.cfg['trajectory'].get('fuse_moves'):
            self.steps = pp.fuse_steps(self.steps)
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
        self.crate_from = (self.crate_idx, list(self.cfg['poses']['home']) if not isinstance(self.cfg['poses']['home'], str) else self.cfg['poses']['home'])
        old, old_hi = pp.home_high(self.cfg)
        if self.home0 is None:
            self.home0 = list(self.cfg['poses']['home'])
        if i < 0:
            home, label = list(self.home0), 'home'
            self.get_logger().info('all crates are done: back to the home pose ' + ' '.join(f'{v:g}' for v in home))
        else:
            home, label = [float(v) for v in self.crates[i]['home']], f'crate {i}'
            self.get_logger().info(f"crate {i} (centre {self.crates[i].get('centre')}): home pose = its scan pose {' '.join(f'{v:g}' for v in home)}")
        self.cfg['poses']['home'] = home
        self.crate_idx = i
        new, new_hi = pp.home_high(self.cfg)
        self.start('crate', pp.crate_steps(self.cfg, label, old, old_hi, new, new_hi, []))

    def cb_part(self, msg):
        self.turned_over = bool(msg.data)
        self.get_logger().info(f"part from the simulation: {'turn_over' if self.turned_over else 'standard'}")

    def cb_machine_state(self, msg):
        self.machine_state = json.loads(msg.data)
        if not self.moving:
            if self.over_standard and self.machine_state.get('standard_free') and self.machine_state.get('turn_over_free'):
                self.get_logger().info('no plate on the turn_over or standard seat and the flipper is idle: nothing to wait for -> the next pick')
                self.over_standard = False
            self.try_target()

    def cb_target(self, msg):
        if self.moving:
            self.get_logger().info('busy -> the new target waits until the arm is free')
        self.target_pending, self.target_wait = (msg, self.variant, self.turned_over), None
        self.variant = None
        self.try_target()

    def crate_for(self, pose):
        crate = self.cfg.get('crate') or {}
        if not self.crates or not pp.is_pan(self.cfg, 'crate'):
            return None
        p = pose.position
        lift = float(crate.get('part_above_floor_m', 0.027))
        cur = 0 if self.crate_idx is None else self.crate_idx
        if 0 <= cur < len(self.crates):
            c = self.crates[cur]
            if (math.hypot(c['centre'][0] - p.x, c['centre'][1] - p.y) < float(crate.get('same_crate_xy_m', 0.35))
                    and abs(float(c['floor_z']) + lift - p.z) < float(crate.get('same_crate_z_m', 0.45))):
                return None
        near = min(range(len(self.crates)), key=lambda i: math.hypot(self.crates[i]['centre'][0] - p.x, self.crates[i]['centre'][1] - p.y)
                   + abs(float(self.crates[i]['floor_z']) + lift - p.z))
        return near if near != cur else None

    def try_target(self):
        if self.target_pending is None or self.moving or self.over_standard:
            return
        if not self.motion_ready:
            self.target_pending = None
            self.get_logger().error('ik_node.py is not ready yet (no /motion/ready: move_group or /joint_states is not up) -> target ignored')
            return
        if isinstance(self.cfg['poses']['home'], str):
            self.target_pending = None
            self.get_logger().warn('the home pose is not known yet (initialization has not started) -> target ignored')
            return
        msg, variant, turned_over = self.target_pending
        place, notes = self.cfg['poses']['place'], []
        if str(self.cfg.get('target', 'tcp')).lower() == 'object':
            q = msg.orientation
            roll, pitch, yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('XYZ', degrees=True)
            tcp_r, tcp_p, tcp_y, flipped = grasp.tcp_rpy(roll, pitch, yaw)
            msg = xyzrpy_to_pose(msg.position.x, msg.position.y, msg.position.z, tcp_r, tcp_p, tcp_y)
            variant = variant or ('turn_over' if flipped else 'standard')
            notes.append(('info', f"object rpy {roll:.1f} {pitch:.1f} {yaw:.1f} -> {'turned over' if flipped else 'standard'}, TCP rpy {tcp_r:.1f} {tcp_p:.1f} {tcp_y:.1f}"))
        if isinstance(place, dict):
            if turned_over is not None and str(self.cfg.get('target', 'tcp')).lower() != 'object':
                sim = 'turn_over' if turned_over else 'standard'
                if variant and variant != sim:
                    notes.append(('warn', f"ee_6d.py says {variant!r}, the simulation says {sim!r} -> {sim}"))
                variant = sim
            if not variant:
                notes.append(('warn', f"no state of the part received (click the part in the simulation, or the word after the pose in ee_6d.py: {' | '.join(place)}) -> using standard"))
            if variant and variant not in place:
                notes.append(('warn', f"no place pose for {variant!r} (poses.place: {', '.join(place)}) -> using standard"))
            notes.append(('info', f"part: {variant if variant in place else 'standard'} -> place {pp.place_pose(self.cfg, variant)}"))
        seat = ('turn_over' if variant == 'turn_over' and 'turn_over' in place else 'standard') if isinstance(place, dict) else ''
        if seat and self.machine_state is not None and not self.machine_state.get(f'{seat}_free', True):
            if self.target_wait != seat:
                self.target_wait = seat
                self.get_logger().info(f'the {seat} seat is not free (machine state {self.machine_state}) -> the pick waits')
            return
        near = self.crate_for(msg)
        if near is not None:
            self.get_logger().info(f"the target is far from the crate the arm is at -> going to crate {near} ({self.crates[near].get('centre')}) first, then the pick")
            self.start_crate(near)
            return
        for level, text in notes:
            getattr(self.get_logger(), level)(text)
        self.target_pending, self.target_wait, self.seq_variant = None, None, seat
        self.pick_args = (msg, variant)
        self.alternatives = pp.alternatives(self.cfg)
        self.start('sequence', pp.make_sequence(msg, self.cfg, variant))

    def start(self, name, steps):
        strict = [str(v) for v in (self.cfg.get('trajectory') or {}).get('no_joint_fallback') or []]
        for st in steps:
            if st['label'] in self.cfg.get('slow_labels', []):
                st['slow'] = True
            if name in ('sequence', 'transfer') and st['kind'] == 'line' and re.split(r'[ /]', st['label'])[0] in strict:
                st['strict'] = True
            if st['label'] == 'wait above standard':
                st['single'] = True
        waits = dict(self.cfg.get('waits') or {})
        if name == 'transfer':
            waits.update(waits.get('transfer') or {})
        if name in ('sequence', 'transfer'):
            steps = apply_waits(steps, {k: v for k, v in waits.items() if k in LABEL_WAITS.values()})
            if self.cfg['trajectory'].get('fuse_moves'):
                steps = pp.fuse_steps(steps)
            chain = []
            for st in steps:
                if st['kind'] == 'path':
                    chain += st['steps']
                elif st['kind'] in pp.MOTION_KINDS:
                    chain.append(st)
            if name == 'sequence' and (self.cfg.get('trajectory') or {}).get('reach_check') and chain:
                steps = [{'label': 'reach check', 'kind': 'path', 'steps': chain, 'dry': True, 'settle': 0.0, 'search': False}, *steps]
        self.name   = name
        self.steps  = steps
        self.reset_seed = True
        self.moving = True
        if name != 'recovery':
            self.set_zone([self.seq_variant] if name == 'sequence' and self.seq_variant else ['standard', 'belt'] if name == 'transfer' else [])
        self.run_next()

    def set_zone(self, seats):
        if self.zone_pub is not None and seats != self.zone:
            self.zone = list(seats)
            self.zone_pub.publish(String(data=json.dumps(self.zone)))

    def run_next(self):
        if not self.steps:
            self.get_logger().info(f'{self.name} done')
            self.moving = False
            self.set_zone([])
            self.over_standard = self.name == 'sequence' and pp.stays_over_standard(self.cfg, self.seq_variant)
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
            else:
                self.try_target()
            return
        step = self.steps.pop(0)
        if step.get('label') == 'wait above standard':
            self.set_zone([])
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
        if not (self.name in ('sequence', 'transfer') and self.cfg['motion'].get('recover_home', True)):
            self.set_zone([])
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
            self.crate_idx, self.cfg['poses']['home'] = self.crate_from
            self.target_pending = None
            self.crate_ready_pub.publish(Int32(data=-1))

    def recover(self):
        self.get_logger().warn('the sequence failed -> back to the home pose')
        self.start('recovery', [move('home (recovery)', xyzrpy_to_pose(*self.cfg['poses']['home']))])

    def unreachable(self, message):
        self.get_logger().warn(f'{message}')
        if self.alternatives:
            target, alt, text = pp.with_alternative(self.cfg, self.pick_args[0], self.alternatives.pop(0))
            self.get_logger().info('reach check failed -> trying again with ' + text)
            self.start('sequence', pp.make_sequence(target, alt, self.pick_args[1]))
            return
        self.get_logger().warn('no way found to this part (straight lines, singularity margin): skipped, the arm did not move')
        self.steps, self.moving = [], False
        self.set_zone([])
        self.done_pub.publish(String(data='unreachable'))
        self.try_target()

    def execute(self, step):
        self.cur_settle = step.get('settle', 0.0)
        self.cur_check = bool(step.get('dry'))
        if step['kind'] == 'align':
            self.align_yaw()
            return
        if step['kind'] == 'tool':
            self.tool(step)
            return
        self.send_motion(step)

    def tool(self, step):
        on = bool(step['value'])
        if self.vac_pub:
            self.get_logger().info(f"[vacuum {'on' if on else 'off'}: {step['label']}]")
            self.vac_id += 1
            self.vac_pending = (on, step)
            self.vac_pub.publish(Bool(data=on))
            vid = self.vac_id
            self.later(self.cfg['vacuum']['timeout'], lambda: self.vacuum_silent(vid))
        elif self.gripper_pub:
            self.get_logger().info(f"[{step['label']}]")
            self.gripper_pending = lambda: self.later(step.get('settle', 0.0), self.run_next)
            self.gripper_pub.publish(Float64(data=float(self.cfg['gripper']['open' if on else 'close'])))
        else:
            self.later(step.get('settle', 0.0), self.run_next)

    def send_motion(self, step):
        if self.cmd_pub.get_subscription_count() == 0:
            self.fail('no ik_node.py is listening on ' + COMMAND_TOPIC)
            return
        if step['kind'] == 'path':
            step = {**step, 'steps': [self.follow_part(s) for s in step['steps']]}
        else:
            step = self.follow_part(step)
        self.cmd_id += 1
        self.cmd_pub.publish(String(data=dumps({'id': self.cmd_id, 'reset_seed': self.reset_seed, 'dry': bool(step.get('dry')), 'step': pack(step)})))
        self.reset_seed = self.reset_seed and bool(step.get('dry'))

    def cb_result(self, msg):
        data = json.loads(msg.data)
        if data['id'] != self.cmd_id or not self.moving:
            return
        if not data['ok'] and self.cur_check:
            self.unreachable(data['msg'])
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
    parser.add_argument('--robot', help='robot name (bringup/<robot>/); default: the robot of the sequence')
    parser.add_argument('--sequence', help='sequence name = control/sequence/<name>.yaml (the coordinates of a map)')
    cli, ros_args = parser.parse_known_args(args)
    cfg = load_config(robot=cli.robot, sequence=cell_config.sequence_of(cli.sequence))
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
