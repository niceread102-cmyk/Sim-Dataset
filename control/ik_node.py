#!/usr/bin/env python3
"""ik_node.py: moves the arm with MoveIt (IK, straight lines, joint goals, one trajectory for several moves); where to go comes from elsewhere.

    python3 control/ik_node.py --robot <robot> [--no-target-pose]        (robot.launch.py starts it)

/target_pose (ee_6d.py): one TCP pose -> IK from the current joints -> joint goal -> /ik_done.
/motion/command (robot_sequence.py, json): one step (move | line | joints | ik | path) -> /motion/result; /motion/ready (latched) gives the home pose.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus, GoalStatusArray
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPositionIK
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

SIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_DIR / 'isaacsim'))
import sim_config

CONFIG_FILE = Path(sim_config.SEQUENCE_DIR) / 'robot_sequence.yaml'
COMMAND_TOPIC, RESULT_TOPIC, READY_TOPIC, SLOW_TOPIC = '/motion/command', '/motion/result', '/motion/ready', '/motion/slow'
ERROR_NAMES = {v: k for k, v in vars(MoveItErrorCodes).items() if k.isupper() and isinstance(v, int)}
CORNER_LABELS = ('insert', 'lower')
REQUIRED_KEYS = ('base_frame', 'group_name', 'ee_link', 'arm_joints', 'ik_seed_deg')
DEFAULTS = {
    'tcp_offset':           [0.0, 0.0, 0.0],
    'velocity_scaling':     0.15,
    'acceleration_scaling': 0.05,
    'planning_time':        5.0,
    'planning_attempts':    10,
    'goal_tolerance':       0.001,
    'ik_timeout':           2.0,
    'busy_timeout':         60.0,
    'joint_limits_deg':     {},
    'wrap_joints_deg':      {},
}
TURN = 2 * math.pi
MOVEIT_ACTIONS = ('/move_action', '/execute_trajectory')
ACTIVE_STATUS = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING)


def offset_pose(pose, dist, axis):
    q, p = pose.orientation, pose.position
    d = Rotation.from_quat([q.x, q.y, q.z, q.w]).apply(axis)
    return Pose(position=Point(x=p.x + dist * d[0], y=p.y + dist * d[1], z=p.z + dist * d[2]), orientation=q)


def shift_toward_base(pose, dist):
    p = pose.position
    norm = math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
    return Pose(position=Point(x=p.x - dist * p.x / norm, y=p.y - dist * p.y / norm, z=p.z - dist * p.z / norm), orientation=pose.orientation)


def xyzrpy_to_pose(x, y, z, roll, pitch, yaw):
    q = Rotation.from_euler('XYZ', [roll, pitch, yaw], degrees=True).as_quat()
    p = Pose(position=Point(x=float(x), y=float(y), z=float(z)))
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p


def tcp_to_flange(pose, cfg):
    return offset_pose(pose, -cfg['robot']['tcp_length'], cfg['robot']['tool_axis'])


def pack(step):
    out = {}
    for key, value in step.items():
        if isinstance(value, Pose):
            p, q = value.position, value.orientation
            out[key] = {'pose': [p.x, p.y, p.z, q.x, q.y, q.z, q.w]}
        elif key == 'steps':
            out[key] = [pack(s) for s in value]
        else:
            out[key] = value
    return out


def unpack(data):
    out = {}
    for key, value in data.items():
        if isinstance(value, dict) and 'pose' in value:
            x, y, z, qx, qy, qz, qw = value['pose']
            pose = Pose(position=Point(x=x, y=y, z=z))
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
            out[key] = pose
        elif key == 'steps':
            out[key] = [unpack(s) for s in value]
        else:
            out[key] = value
    return out


def dumps(obj):
    return json.dumps(obj, separators=(',', ':'))


def move(label, pose, search=False):
    return {'label': label, 'kind': 'move', 'value': pose, 'search': search}


def line(label, pose, start):
    return {'label': label, 'kind': 'line', 'value': pose, 'start': start, 'search': False}


def joints(label, q):
    return {'label': label, 'kind': 'joints', 'value': list(q), 'search': False}


def ik_only(label, pose):
    return {'label': label, 'kind': 'ik', 'value': pose, 'search': True}


def via_steps(label, start, end, max_step, max_angle):
    a, b = start.position, end.position
    qa, qb = start.orientation, end.orientation
    ra, rb = Rotation.from_quat([qa.x, qa.y, qa.z, qa.w]), Rotation.from_quat([qb.x, qb.y, qb.z, qb.w])
    turn = ra.inv() * rb
    n = max(1, math.ceil(max(math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) / max_step, math.degrees(turn.magnitude()) / max_angle)))
    steps = []
    for i in range(1, n):
        q = (ra * Rotation.from_rotvec(turn.as_rotvec() * i / n)).as_quat()
        pose = Pose(position=Point(x=a.x + (b.x - a.x) * i / n, y=a.y + (b.y - a.y) * i / n, z=a.z + (b.z - a.z) * i / n),
                    orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))
        steps.append(ik_only(f'{label} {i}/{n}', pose))
    return steps


def robot_part(cfg, robot):
    ik_file = Path(sim_config.BRINGUP_DIR) / robot / 'config' / 'ik_node.yaml'
    if not ik_file.is_file():
        raise ValueError(f'{ik_file} not found: python3 isaacsim/make_robot_config.py robot:={robot}')
    ik = yaml.safe_load(open(ik_file))
    robot_cfg = sim_config.load_robot(robot)
    motion = sim_config.robot_motion(robot_cfg)
    cfg.setdefault('motion', {}).update(velocity_scaling=float(motion['velocity_scaling']), acceleration_scaling=float(motion['acceleration_scaling']))
    tcp = [float(x) for x in ik.get('tcp_offset', [0.0, 0.0, 0.0])]
    length = math.sqrt(sum(x * x for x in tcp))
    cfg['robot'] = {'name': robot, 'planning_group': ik['group_name'], 'arm_joints': list(ik['arm_joints']),
                    'base_link': ik['base_frame'], 'tip_link': ik['ee_link'],
                    'tool_axis': [x / length for x in tcp] if length > 1e-9 else [0.0, 0.0, 1.0], 'tcp_length': length}
    cfg.setdefault('ik', {})
    if 'seed' not in cfg['ik']:
        jp = robot_cfg.get('joint_pos') or {}
        cfg['ik']['seed'] = ([math.degrees(next((float(v) for k, v in jp.items() if re.fullmatch(str(k), n)), 0.0)) for n in cfg['robot']['arm_joints']]
                             if jp else list(ik['ik_seed_deg']))
    if len(cfg['ik']['seed']) != len(cfg['robot']['arm_joints']):
        raise ValueError(f'ik.seed needs one value per arm joint ({len(cfg["robot"]["arm_joints"])}): {cfg["robot"]["arm_joints"]}')
    return robot_cfg


def load_motion_config(robot):
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f) or {}
    robot_part(cfg, robot)
    return cfg


def ik_request(group, link, frame, joint_names, pose, seed, timeout, avoid_collisions=True):
    stamped = PoseStamped()
    stamped.header.frame_id = frame
    stamped.pose = pose
    req = GetPositionIK.Request()
    req.ik_request.group_name              = group
    req.ik_request.ik_link_name            = link
    req.ik_request.pose_stamped            = stamped
    req.ik_request.robot_state.is_diff     = True
    req.ik_request.robot_state.joint_state = JointState(name=list(joint_names), position=list(seed))
    req.ik_request.avoid_collisions        = avoid_collisions
    req.ik_request.timeout                 = Duration(sec=int(timeout), nanosec=int((timeout % 1.0) * 1e9))
    return req


def joint_goal(group, joint_names, positions, tolerance, attempts, planning_time, velocity, acceleration, scene_diff=False):
    constraints = Constraints()
    for name, position in zip(joint_names, positions):
        constraints.joint_constraints.append(JointConstraint(joint_name=name, position=position, weight=1.0, tolerance_above=tolerance, tolerance_below=tolerance))
    req = MotionPlanRequest()
    req.group_name                      = group
    req.goal_constraints                = [constraints]
    req.num_planning_attempts           = attempts
    req.allowed_planning_time           = planning_time
    req.max_velocity_scaling_factor     = velocity
    req.max_acceleration_scaling_factor = acceleration
    if scene_diff:
        req.start_state.is_diff = True
    goal = MoveGroup.Goal()
    goal.request = req
    goal.planning_options.plan_only = False
    if scene_diff:
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
    return goal


def line_request(cfg, pose, start_q=None):
    robot, line_cfg, motion = cfg['robot'], cfg['line'], cfg['motion']
    req = GetCartesianPath.Request()
    req.header.frame_id                 = robot['base_link']
    req.start_state.is_diff             = True
    if start_q is not None:
        req.start_state.joint_state     = JointState(name=list(robot['arm_joints']), position=list(start_q))
    req.group_name                      = robot['planning_group']
    req.link_name                       = robot['tip_link']
    req.waypoints                       = [pose]
    req.max_step                        = line_cfg['max_step']
    req.jump_threshold                  = 0.0
    req.revolute_jump_threshold         = line_cfg['revolute_jump_threshold']
    req.avoid_collisions                = cfg['ik']['avoid_collisions']
    req.max_velocity_scaling_factor     = motion['velocity_scaling']
    req.max_acceleration_scaling_factor = motion['acceleration_scaling']
    return req


def load_config(path, robot_yaml=None):
    cfg = yaml.safe_load(open(path)) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f'{path}: missing {missing}')
    cfg = {**DEFAULTS, **cfg}
    if robot_yaml:
        cfg.update(sim_config.robot_motion(yaml.safe_load(open(robot_yaml)) or {}))
    n = len(cfg['arm_joints'])
    if len(cfg['ik_seed_deg']) != n:
        raise ValueError(f'{path}: ik_seed_deg needs one value per arm_joints entry ({n})')
    if len(cfg['tcp_offset']) != 3:
        raise ValueError(f'{path}: tcp_offset needs 3 values [x, y, z]')
    unknown = [j for j in cfg['joint_limits_deg'] if j not in cfg['arm_joints']]
    if unknown:
        raise ValueError(f'{path}: joint_limits_deg names joints that are not in arm_joints: {unknown}')
    unknown = [j for j in cfg['wrap_joints_deg'] if j not in cfg['arm_joints']]
    if unknown:
        raise ValueError(f'{path}: wrap_joints_deg names joints that are not in arm_joints: {unknown}')
    bad = [j for j, r in cfg['wrap_joints_deg'].items() if len(r) != 2 or not r[0] < r[1]]
    if bad:
        raise ValueError(f'{path}: wrap_joints_deg needs [min, max] with min < max per joint: {bad}')
    return cfg


def build_ik_request(cfg, pose: Pose, seed=None):
    R = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix()
    ee_pos = np.array([pose.position.x, pose.position.y, pose.position.z]) - R @ np.array(cfg['tcp_offset'])
    flange = Pose(position=Point(x=float(ee_pos[0]), y=float(ee_pos[1]), z=float(ee_pos[2])), orientation=pose.orientation)
    seed = list(seed) if seed is not None else [math.radians(d) for d in cfg['ik_seed_deg']]
    return ik_request(cfg['group_name'], cfg['ee_link'], cfg['base_frame'], cfg['arm_joints'], flange, seed, cfg['ik_timeout'])


def outside_limits(cfg, q):
    out = []
    for name, (lo, hi) in cfg['joint_limits_deg'].items():
        deg = math.degrees(q[name])
        if not lo <= deg <= hi:
            out.append(f'{name}={deg:.1f} not in [{lo}, {hi}]')
    return out


def nearest_turn(q, ref, lo, hi):
    k = round((ref - q) / TURN)
    k_min = math.ceil((lo - q) / TURN - 1e-9) if lo > -math.inf else -10**9
    k_max = math.floor((hi - q) / TURN + 1e-9) if hi < math.inf else 10**9
    if k_min > 0 or k_max < 0:
        return q
    return q + TURN * min(max(k, k_min), k_max)


def wrap_to_current(cfg, q, current):
    out = dict(q)
    for name, (lo, hi) in cfg['wrap_joints_deg'].items():
        if name in q and name in current:
            narrow = cfg['joint_limits_deg'].get(name)
            if narrow:
                lo, hi = max(lo, narrow[0]), min(hi, narrow[1])
            out[name] = nearest_turn(q[name], current[name], math.radians(lo), math.radians(hi))
    return out


def build_joint_goal(cfg, joint_state: JointState):
    pairs = [(n, p) for n, p in zip(joint_state.name, joint_state.position) if n in cfg['arm_joints']]
    return joint_goal(cfg['group_name'], [n for n, _ in pairs], [p for _, p in pairs], cfg['goal_tolerance'], cfg['planning_attempts'],
                      cfg['planning_time'], cfg['velocity_scaling'], cfg['acceleration_scaling'])


class IKNode(Node):
    def __init__(self, cfg):
        super().__init__('ik_node')
        self.cfg      = cfg
        self.sub      = self.create_subscription(Pose, '/target_pose', self._cb_target_pose, 10)
        self.done_pub = self.create_publisher(Bool, '/ik_done', 10)
        self._client  = ActionClient(self, MoveGroup, '/move_action')
        self._ik      = self.create_client(GetPositionIK, '/compute_ik')
        self._busy    = False
        self._busy_since = 0.0
        self._seq     = 0
        self._own_goal_ids  = set()
        self._others_active = {}
        self._q_now   = {}
        self.create_subscription(JointState, '/joint_states', self._cb_joint_states, qos_profile_sensor_data)
        status_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for action in MOVEIT_ACTIONS:
            self.create_subscription(GoalStatusArray, f'{action}/_action/status', lambda msg, a=action: self._on_status(a, msg), status_qos)
        self.get_logger().info(f"ik_node ready ({cfg['group_name']}, {len(cfg['arm_joints'])} joints, ee_link {cfg['ee_link']}) - waiting for /target_pose")

    def _cb_joint_states(self, msg: JointState):
        self._q_now.update({n: p for n, p in zip(msg.name, msg.position) if n in self.cfg['arm_joints']})

    def _on_status(self, action: str, msg: GoalStatusArray):
        self._others_active[action] = any(
            st.status in ACTIVE_STATUS and bytes(st.goal_info.goal_id.uuid) not in self._own_goal_ids
            for st in msg.status_list)

    def _cb_target_pose(self, msg: Pose):
        if self._busy:
            waited = time.monotonic() - self._busy_since
            if self._client.server_is_ready() and waited < self.cfg['busy_timeout']:
                self.get_logger().warn(f'Already moving ({waited:.0f} s), ignoring')
                return
            self.get_logger().warn(f'No result for previous goal after {waited:.0f} s (move_group died/restarted?) - dropping it')
            self._busy = False
            self._seq += 1

        others = [a for a, on in self._others_active.items() if on]
        if others:
            self.get_logger().warn(f'move_group is executing another request on {", ".join(others)} (RViz Plan & Execute?) - ignoring target, send it again after that motion ends')
            return

        self._busy       = True
        self._busy_since = time.monotonic()
        self._seq       += 1
        self._solve_ik(msg, self._seq)

    def _finish(self, ok: bool, seq: int):
        if seq != self._seq:
            return
        self._busy = False
        self.done_pub.publish(Bool(data=ok))

    def _solve_ik(self, pose: Pose, seq: int):
        if not self._ik.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('/compute_ik not available')
            self._finish(False, seq)
            return

        names = self.cfg['arm_joints']
        if all(n in self._q_now for n in names):
            seed = [self._q_now[n] for n in names]
        else:
            seed = None
            self.get_logger().warn('no /joint_states of the arm yet: IK starts from ik_seed_deg, joint angles are not turned')
        self._ik.call_async(build_ik_request(self.cfg, pose, seed)).add_done_callback(lambda f: self._ik_cb(f, seq))

    def _ik_cb(self, future, seq: int):
        if seq != self._seq:
            return
        res = future.result()
        if res.error_code.val != 1:
            self.get_logger().error(f'[FAIL] IK error_code={res.error_code.val} (target unreachable or in collision)')
            self._finish(False, seq)
            return

        names = self.cfg['arm_joints']
        js = res.solution.joint_state
        q  = {n: p for n, p in zip(js.name, js.position) if n in names}
        self.get_logger().info('IK: ' + ' '.join(f'{n}={math.degrees(q[n]):.1f}' for n in names))
        if self.cfg['wrap_joints_deg']:
            turned = wrap_to_current(self.cfg, q, self._q_now)
            changed = [n for n in names if turned[n] != q[n]]
            if changed:
                self.get_logger().info('full turn to the angle nearest to the current one: ' + ' '.join(
                    f'{n} {math.degrees(q[n]):.1f}->{math.degrees(turned[n]):.1f}' for n in changed))
                js.position = [turned.get(n, p) for n, p in zip(js.name, js.position)]
                q = turned
        bad = outside_limits(self.cfg, q)
        if bad:
            self.get_logger().error('[FAIL] IK solution outside joint_limits_deg: ' + ', '.join(bad))
            self._finish(False, seq)
            return
        self._send_goal(build_joint_goal(self.cfg, js), seq)

    def _send_goal(self, goal_msg, seq: int):
        if not self._client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('MoveGroup not available')
            self._finish(False, seq)
            return

        self._client.send_goal_async(goal_msg).add_done_callback(lambda f: self._goal_cb(f, seq))

    def _goal_cb(self, future, seq: int):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected')
            self._finish(False, seq)
            return
        self._own_goal_ids.add(bytes(handle.goal_id.uuid))
        handle.get_result_async().add_done_callback(lambda f: self._result_cb(f, seq))

    def _result_cb(self, future, seq: int):
        if seq != self._seq:
            return
        result = future.result().result
        code   = result.error_code.val
        ok     = (code == 1)
        if ok:
            self.get_logger().info('[OK] Move completed')
        else:
            self.get_logger().error(f'[FAIL] error_code={code} message="{result.error_code.message}" source="{result.error_code.source}"')
        self._finish(ok, seq)


def ik_candidates(cfg, step):
    candidates = [(step['value'], 0.0)]
    if step['search']:
        radius = cfg['tolerance']['waypoint_position']
        candidates += [(shift_toward_base(step['value'], radius * f), radius * f) for f in (0.25, 0.5, 1.0)]
    return candidates


class MotionServer(Node):
    def __init__(self, cfg):
        super().__init__('motion_server')
        self.cfg         = cfg
        self.move_client = ActionClient(self, MoveGroup, cfg['topics']['move_action'])
        self.ik_client   = self.create_client(GetPositionIK, cfg['ik']['service'])
        self.line_client = self.create_client(GetCartesianPath, cfg['line']['service'])
        self.exec_client = ActionClient(self, ExecuteTrajectory, cfg['line']['action'])
        self.fk_client   = self.create_client(GetPositionFK, '/compute_fk')
        self.result_pub  = self.create_publisher(String, RESULT_TOPIC, 10)
        self.slow_pub    = self.create_publisher(Bool, SLOW_TOPIC, 10)
        self.ready_pub   = self.create_publisher(String, READY_TOPIC, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, COMMAND_TOPIC, self.cb_command, 10)
        self.create_subscription(JointState, cfg['topics']['joint_states'], self.cb_joint_states, 10)
        self.ready_seed  = [math.radians(d) for d in cfg['ik']['seed']]
        self.seed        = list(self.ready_seed)
        self.busy        = False
        self.cmd_id      = None
        self.queue       = []
        self.goal_q      = []
        self.move_tries  = 0
        self.retry       = None
        self.move_id     = 0
        self.move_handle = None
        self.move_timer  = self.create_timer(cfg['motion']['timeout'], self.cb_move_timeout)
        self.move_timer.cancel()
        self.joint_pos   = {}
        self.path        = None
        self.init_tries  = 0
        self.init_timer  = self.create_timer(cfg['startup']['poll'], self.cb_init)
        self.get_logger().info(f"motion server ready ({cfg['robot']['planning_group']}, {len(cfg['robot']['arm_joints'])} joints): {COMMAND_TOPIC} -> {RESULT_TOPIC}")

    def servers_ready(self):
        return (self.move_client.server_is_ready() and self.ik_client.service_is_ready() and self.line_client.service_is_ready()
                and self.exec_client.server_is_ready() and self.fk_client.service_is_ready())

    def cb_joint_states(self, msg):
        self.joint_pos = dict(zip(msg.name, msg.position))

    def cb_init(self):
        if not self.servers_ready() or not self.joint_pos:
            return
        self.init_timer.cancel()
        self.init_tries += 1
        req = GetPositionFK.Request()
        req.header.frame_id = self.cfg['robot']['base_link']
        req.fk_link_names = [self.cfg['robot']['tip_link']]
        req.robot_state.joint_state.name = list(self.cfg['robot']['arm_joints'])
        req.robot_state.joint_state.position = list(self.ready_seed)
        self.fk_client.call_async(req).add_done_callback(self.cb_home)

    def cb_home(self, future):
        res = future.result()
        if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS or not res.pose_stamped:
            self.get_logger().error('forward kinematics of the ready pose failed -> no home pose')
            if self.init_tries < self.cfg['startup']['retries']:
                self.init_timer.reset()
            return
        robot = self.cfg['robot']
        tcp = offset_pose(res.pose_stamped[0].pose, robot['tcp_length'], robot['tool_axis'])
        q, p = tcp.orientation, tcp.position
        rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('XYZ', degrees=True)
        home = [p.x, p.y, p.z, *[float(a) for a in rpy]]
        self.get_logger().info('home pose = TCP of the start pose of the robot: ' + ' '.join(f'{v:.3f}' for v in home))
        self.ready_pub.publish(String(data=dumps({'home': home})))

    def cb_command(self, msg):
        data = json.loads(msg.data)
        if self.busy:
            self.get_logger().warn(f"busy -> command {data['id']} ignored")
            self.result_pub.publish(String(data=dumps({'id': data['id'], 'ok': False, 'msg': 'busy'})))
            return
        self.busy, self.cmd_id = True, data['id']
        if data.get('reset_seed'):
            self.seed = list(self.ready_seed)
        step = unpack(data['step'])
        self.slow_pub.publish(Bool(data=bool(step.get('slow'))))
        if step['kind'] == 'path':
            self.queue = []
            self.run_path(step)
        else:
            self.queue = [step]
            self.run_next()

    def reply(self, ok, message=''):
        self.result_pub.publish(String(data=dumps({'id': self.cmd_id, 'ok': ok, 'msg': message})))
        self.busy, self.cmd_id, self.queue, self.path = False, None, [], None

    def run_next(self):
        if not self.queue:
            self.reply(True)
            return
        self.execute(self.queue.pop(0))

    def fail(self, message):
        self.cancel_move_timer()
        self.get_logger().error(message)
        self.reply(False, message)

    def cancel_move_timer(self):
        self.move_timer.cancel()

    def cb_move_timeout(self):
        self.cancel_move_timer()
        self.move_id += 1
        if self.move_handle is not None:
            self.move_handle.cancel_goal_async()
        self.fail(f"move not finished after {self.cfg['motion']['timeout']} s -> cancelled")

    def execute(self, step):
        if step['kind'] == 'line':
            self.move_tries = 0
            self.plan_line(step)
            return
        if step['kind'] == 'joints':
            self.seed, self.goal_q, self.move_tries = list(step['value']), list(step['value']), 0
            self.get_logger().info(f"[{step['label']}] joints: " + ' '.join(f'{j}={math.degrees(v):.1f}' for j, v in zip(self.cfg['robot']['arm_joints'], step['value'])))
            self.send_move()
            return
        self.try_ik(step, ik_candidates(self.cfg, step), 0)

    def run_path(self, step):
        names = self.cfg['robot']['arm_joints']
        q0 = [self.joint_pos.get(j) for j in names]
        if any(v is None for v in q0):
            self.path_abort(step, 'no /joint_states yet', list(self.seed))
            return
        self.move_tries = 0
        self.retry = lambda: self.run_path(step)
        self.path = {'step': step, 'queue': list(step['steps']), 'q': q0, 'seed0': list(self.seed), 'moves': 0, 't': 0.0,
                     'points': [JointTrajectoryPoint(positions=q0, time_from_start=Duration(sec=0, nanosec=0))]}
        self.path_next()

    def path_next(self):
        p = self.path
        if not p['queue']:
            self.path_send()
            return
        sub = p['queue'].pop(0)
        if sub['label'] in CORNER_LABELS and len(p['points']) > 1:
            self.path_point(p['q'], p['t'] + 0.001)
        if sub['kind'] == 'joints':
            self.seed = list(sub['value'])
            self.path_add_joint_move(list(sub['value']))
            self.path_next()
        elif sub['kind'] == 'line':
            future = self.line_client.call_async(line_request(self.cfg, sub['value'], p['q']))
            future.add_done_callback(lambda f: self.cb_path_line(f, sub))
        else:
            self.path_ik(sub, ik_candidates(self.cfg, sub), 0)

    def path_ik(self, sub, candidates, index):
        if index >= len(candidates):
            self.path_abort(self.path['step'], f"[{sub['label']}] no IK solution", self.path['seed0'])
            return
        self.ask_ik(candidates[index][0], index).add_done_callback(lambda f: self.cb_path_ik(f, sub, candidates, index))

    def cb_path_ik(self, future, sub, candidates, index):
        res = future.result()
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.path_ik(sub, candidates, index + 1)
            return
        names = self.cfg['robot']['arm_joints']
        solution = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        if any(j not in solution for j in names):
            self.path_abort(self.path['step'], f"[{sub['label']}] IK result is missing joints", self.path['seed0'])
            return
        q = [solution[j] for j in names]
        self.seed = q
        if sub['kind'] != 'ik':
            self.path_add_joint_move(q)
        self.path_next()

    def cb_path_line(self, future, sub):
        res, line_cfg, p = future.result(), self.cfg['line'], self.path
        traj = res.solution.joint_trajectory
        if res.error_code.val != MoveItErrorCodes.SUCCESS or res.fraction < line_cfg['min_fraction'] or len(traj.points) < 2:
            self.path_abort(p['step'], f"[{sub['label']}] straight line only {res.fraction * 100:.0f}% possible", p['seed0'])
            return
        names = self.cfg['robot']['arm_joints']
        t0 = p['t']
        for pt in traj.points[1:]:
            pos = dict(zip(traj.joint_names, pt.positions))
            t = t0 + pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            self.path_point([pos[j] for j in names], t)
        self.seed = list(p['q'])
        p['moves'] += 1
        self.path_next()

    def path_point(self, q, t):
        p = self.path
        p['t'] = max(t, p['t'] + 0.001)
        p['points'].append(JointTrajectoryPoint(positions=list(q), time_from_start=Duration(sec=int(p['t']), nanosec=int((p['t'] % 1.0) * 1e9))))
        p['q'] = list(q)

    def path_add_joint_move(self, q):
        p = self.path
        step = float(self.cfg.get('fuse_joint_step', 0.05))
        d = max(abs(a - b) for a, b in zip(q, p['q']))
        n = max(1, math.ceil(d / step))
        q0 = list(p['q'])
        for i in range(1, n + 1):
            self.path_point([a + (b - a) * i / n for a, b in zip(q0, q)], p['t'] + max(0.02, d / n / 0.6))
        p['moves'] += 1

    def path_abort(self, step, why, seed):
        self.get_logger().warn(f"[{step['label']}] {why} -> the pieces are executed one by one")
        self.seed = list(seed)
        self.queue[0:0] = list(step['steps'])
        self.path = None
        self.run_next()

    def path_send(self):
        p = self.path
        jt = JointTrajectory(joint_names=list(self.cfg['robot']['arm_joints']), points=p['points'])
        self.get_logger().info(f"[{p['step']['label']}] one path: {p['moves']} moves, {len(p['points'])} points, {p['t']:.1f} s")
        self.path = None
        self.send_goal(self.exec_client, ExecuteTrajectory.Goal(trajectory=RobotTrajectory(joint_trajectory=jt)))

    def plan_line(self, step):
        future = self.line_client.call_async(line_request(self.cfg, step['value']))
        future.add_done_callback(lambda f: self.cb_line(f, step))

    def cb_line(self, future, step):
        res, line_cfg = future.result(), self.cfg['line']
        traj = res.solution.joint_trajectory
        if res.error_code.val != MoveItErrorCodes.SUCCESS or res.fraction < line_cfg['min_fraction'] or not traj.points:
            self.get_logger().warn(f"[{step['label']}] straight line only {res.fraction * 100:.0f}% possible "
                                   f"({ERROR_NAMES.get(res.error_code.val, res.error_code.val)}) -> joint-space move instead")
            self.queue[0:0] = (via_steps(step['label'], step['start'], step['value'], line_cfg['ik_step'], line_cfg['ik_angle_step'])
                               + [move(step['label'], step['value'], search=True)])
            self.run_next()
            return
        names = self.cfg['robot']['arm_joints']
        end   = dict(zip(traj.joint_names, traj.points[-1].positions))
        self.seed = [end[j] for j in names]
        t = traj.points[-1].time_from_start
        self.get_logger().info(f"[{step['label']}] straight line: {len(traj.points)} points, {t.sec + t.nanosec * 1e-9:.1f} s, "
                               f"ends at " + ' '.join(f'{j}={math.degrees(end[j]):.1f}' for j in names))
        self.retry = lambda: self.plan_line(step)
        self.send_goal(self.exec_client, ExecuteTrajectory.Goal(trajectory=res.solution))

    def ask_ik(self, pose, index):
        ik, robot = self.cfg['ik'], self.cfg['robot']
        req = ik_request(robot['planning_group'], robot['tip_link'], robot['base_link'], robot['arm_joints'], pose, self.seed,
                         ik['timeout'] if index == 0 else ik['retry_timeout'], ik['avoid_collisions'])
        return self.ik_client.call_async(req)

    def try_ik(self, step, candidates, index):
        if index >= len(candidates):
            self.fail(f"[{step['label']}] no IK solution")
            return
        self.ask_ik(candidates[index][0], index).add_done_callback(lambda f: self.cb_ik(f, step, candidates, index))

    def cb_ik(self, future, step, candidates, index):
        res = future.result()
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.try_ik(step, candidates, index + 1)
            return
        names    = self.cfg['robot']['arm_joints']
        solution = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        if any(j not in solution for j in names):
            self.fail(f"[{step['label']}] IK result is missing joints: {sorted(solution)}")
            return
        q       = [solution[j] for j in names]
        self.seed = q
        shifted = candidates[index][1]
        note    = f' (pose moved {shifted * 1000:.1f} mm toward the base)' if shifted else ''
        log     = self.get_logger().debug if step['kind'] == 'ik' else self.get_logger().info
        log(f"[{step['label']}] IK: " + ' '.join(f'{j}={math.degrees(v):.1f}' for j, v in zip(names, q)) + note)
        if step['kind'] == 'ik':
            self.run_next()
            return
        self.goal_q     = q
        self.move_tries = 0
        self.send_move()

    def send_move(self):
        self.retry = self.send_move
        cfg, motion = self.cfg, self.cfg['motion']
        goal = joint_goal(cfg['robot']['planning_group'], cfg['robot']['arm_joints'], self.goal_q, cfg['tolerance']['joint'], motion['planning_attempts'],
                          motion['planning_time'], motion['velocity_scaling'], motion['acceleration_scaling'], scene_diff=True)
        self.send_goal(self.move_client, goal)

    def send_goal(self, client, goal):
        self.move_id += 1
        move_id = self.move_id
        self.move_handle = None
        client.send_goal_async(goal).add_done_callback(lambda future: self.cb_goal_response(future, move_id))
        self.move_timer.reset()

    def cb_goal_response(self, future, move_id):
        if move_id != self.move_id:
            return
        handle = future.result()
        if not handle.accepted:
            self.fail('move_group rejected the goal')
            return
        self.move_handle = handle
        handle.get_result_async().add_done_callback(lambda future: self.cb_result(future, move_id))

    def cb_result(self, future, move_id):
        if move_id != self.move_id:
            return
        self.cancel_move_timer()
        code = future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            retries = self.cfg['motion']['retries']
            if self.move_tries >= retries:
                self.fail(f'failed: {ERROR_NAMES.get(code, code)}')
                return
            self.move_tries += 1
            self.get_logger().warn(f'{ERROR_NAMES.get(code, code)} -> trying the move again ({self.move_tries}/{retries})')
            self.retry()
            return
        self.get_logger().info('arrived')
        self.run_next()


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot', help='robot name, e.g. ur5e, or the path of its robot yaml (isaacsim/configs/robot/ur5e.yaml)')
    parser.add_argument('--config', help='ik_node yaml of the robot (default: bringup/<robot>/config/ik_node.yaml)')
    parser.add_argument('--no-target-pose', action='store_true', help='do not listen to /target_pose (a sequence sends the targets)')
    cli, ros_args = parser.parse_known_args(args)
    if not cli.robot and not cli.config:
        parser.error('give --robot <name> (or --config <ik_node yaml>)')
    robot_yaml = sim_config.robot_config_path(cli.robot) if cli.robot else None
    config = Path(cli.config) if cli.config else Path(sim_config.BRINGUP_DIR) / Path(robot_yaml).stem / 'config' / 'ik_node.yaml'
    cfg = load_config(config, robot_yaml)
    motion_cfg = load_motion_config(Path(robot_yaml).stem if robot_yaml else config.parent.parent.name)

    rclpy.init(args=ros_args)
    nodes = [MotionServer(motion_cfg)] + ([] if cli.no_target_pose else [IKNode(cfg)])
    executor = SingleThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
