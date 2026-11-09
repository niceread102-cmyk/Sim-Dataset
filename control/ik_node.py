#!/usr/bin/env python3
"""ik_node.py: the robot control: MoveIt IK with the choice of the arm branch, wrist singularity avoidance (J4 held, |J5| kept off 0) and forearm twist checks, straight lines, joint moves and one-trajectory paths, reach checks without moving.

    python3 control/ik_node.py [--robot-config bringup/<robot>/config/robot.yaml] [--tcp-link <link>]

/motion/command (json): one step (move | line | joints | ik | path) with TCP poses -> /motion/result; /motion/ready (latched) gives the home pose = TCP of the pose at start.
Joints, limits, frame, tip link and TCP come from the URDF and the SRDF that move_group holds; the limits of this control from `control:` of the robot yaml.
Knows nothing of the task, the map or the simulation: the same node runs on the real controller.
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET

import yaml

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPositionIK
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

COMMAND_TOPIC, RESULT_TOPIC, READY_TOPIC, SLOW_TOPIC = '/motion/command', '/motion/result', '/motion/ready', '/motion/slow'
ERROR_NAMES = {v: k for k, v in vars(MoveItErrorCodes).items() if k.isupper() and isinstance(v, int)}
CORNER_LABELS = ('insert', 'lower')
TURN = 2 * math.pi
P = {
    'velocity_scaling': 0.15, 'acceleration_scaling': 0.05, 'planning_time': 5.0, 'planning_attempts': 1, 'joint_tolerance': 0.001,
    'retries': 2, 'timeout': 120.0, 'ik_timeout': 1.0, 'line_max_step': 0.005, 'revolute_jump': 0.3, 'min_fraction': 0.999,
    'path_joint_step': 0.05, 'wrist_margin_deg': 1.0, 'wrist_margin_run_deg': 0.5, 'strict_forearm_turn_deg': 10.0, 'strict_forearm_turn_run_deg': 15.0, 'flip_forearm_deg': 90.0, 'flip_elbow_deg': 120.0, 'singularity_avoidance': True, 'wrist_offset_deg': 0.5, 'wrist_band_deg': 10.0, 'wrist_hold_tilt_deg': 0.5, 'wrist_turn_step_deg': 1.0, 'poll': 1.0, 'branch_timeout': 0.2,
    'own_line_step': 0.01, 'own_line_angle_deg': 2.0, 'own_line_speed': 0.15, 'joint_speed': 0.6,
    'own_line_pos_tol': 0.0005, 'own_line_tilt_deg': 2.0, 'dls_iterations': 200, 'dls_damping': 0.01, 'dls_rot_weight': 0.3, 'dls_rot_weight_held': 0.02,
    'branch_j1_deg': (-150, -30, 30, 150), 'branch_j3_deg': (-40, 100), 'branch_j5_deg': (-50, 50), 'branch_j2_deg': 20,
    'reach_forearm_turn_deg': 60.0,
}


def offset_pose(pose, dist, axis):
    q, p = pose.orientation, pose.position
    d = Rotation.from_quat([q.x, q.y, q.z, q.w]).apply(axis)
    return Pose(position=Point(x=p.x + dist * d[0], y=p.y + dist * d[1], z=p.z + dist * d[2]), orientation=q)


def xyzrpy_to_pose(x, y, z, roll, pitch, yaw):
    q = Rotation.from_euler('XYZ', [roll, pitch, yaw], degrees=True).as_quat()
    p = Pose(position=Point(x=float(x), y=float(y), z=float(z)))
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p


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


def move(label, pose):
    return {'label': label, 'kind': 'move', 'value': pose}


def line(label, pose, start):
    return {'label': label, 'kind': 'line', 'value': pose, 'start': start}


def joints(label, q):
    return {'label': label, 'kind': 'joints', 'value': list(q)}


def joint_origin(j):
    T, o = np.eye(4), j.find('origin')
    if o is not None:
        T[:3, :3] = Rotation.from_euler('xyz', [float(v) for v in (o.get('rpy') or '0 0 0').split()]).as_matrix()
        T[:3, 3] = [float(v) for v in (o.get('xyz') or '0 0 0').split()]
    return T


def fixed_transform(parent_of, a, b):
    def up(n):
        out = [n]
        while n in parent_of:
            n = parent_of[n].find('parent').get('link')
            out.append(n)
        return out

    def frame(n, top):
        T = np.eye(4)
        while n != top:
            j = parent_of[n]
            if j.get('type') != 'fixed':
                return None
            T = joint_origin(j) @ T
            n = j.find('parent').get('link')
        return T
    pa, pb = up(a), up(b)
    top = next((n for n in pa if n in pb), None)
    Ta, Tb = (frame(a, top), frame(b, top)) if top is not None else (None, None)
    return None if Ta is None or Tb is None else np.linalg.inv(Ta) @ Tb


def read_robot(urdf_xml, srdf_xml, tcp_link=None):
    urdf, srdf = ET.fromstring(urdf_xml), ET.fromstring(srdf_xml)
    group = next((g for g in srdf.findall('group') if g.find('chain') is not None), None)
    if group is None:
        raise ValueError('the SRDF has no planning group with a chain')
    base, tip = group.find('chain').get('base_link'), group.find('chain').get('tip_link')
    parent_of = {j.find('child').get('link'): j for j in urdf.findall('joint')}
    chain, link = [], tip
    while link != base:
        chain.append(parent_of[link])
        link = chain[-1].find('parent').get('link')
    arm = [j for j in reversed(chain) if j.get('type') != 'fixed']
    wrap = {}
    for j in arm:
        lim = j.find('limit')
        if j.get('type') == 'continuous' or lim is None or lim.get('lower') is None:
            wrap[j.get('name')] = (-math.inf, math.inf)
        elif float(lim.get('upper')) - float(lim.get('lower')) > TURN + 1e-6:
            wrap[j.get('name')] = (float(lim.get('lower')), float(lim.get('upper')))
    root = next(l.get('name') for l in urdf.findall('link') if l.get('name') not in parent_of)
    full, link = [], tip
    while link != root:
        j = parent_of[link]
        if j.get('type') in ('revolute', 'continuous'):
            a = j.find('axis')
            axis = np.array([float(v) for v in (a.get('xyz') if a is not None else '1 0 0').split()])
            full.append((joint_origin(j), axis / np.linalg.norm(axis)))
        elif j.get('type') == 'fixed':
            full.append((joint_origin(j), None))
        else:
            full = None
            break
        link = j.find('parent').get('link')
    limits = []
    for j in arm:
        lim = j.find('limit')
        free = j.get('type') == 'continuous' or lim is None or lim.get('lower') is None
        limits.append((-math.inf, math.inf) if free else (float(lim.get('lower')), float(lim.get('upper'))))
    names = [l.get('name') for l in urdf.findall('link')]
    tcp = tcp_link or next((n for n in names if 'tcp' in n.lower() and n != tip), None)
    T = fixed_transform(parent_of, tip, tcp) if tcp else None
    offset = T[:3, 3] if T is not None else np.zeros(3)
    length = float(np.linalg.norm(offset))
    return {'group': group.get('name'), 'frame': root, 'tip': tip, 'joints': [j.get('name') for j in arm], 'wrap': wrap, 'limits': limits,
            'chain': list(reversed(full)) if full is not None else None, 'tcp_offset': offset.tolist(),
            'tcp_length': length, 'tool_axis': (offset / length).tolist() if length > 1e-9 else [0.0, 0.0, 1.0]}


def ik_request(robot, pose, seed, timeout):
    req = GetPositionIK.Request()
    req.ik_request.group_name = robot['group']
    req.ik_request.ik_link_name = robot['tip']
    req.ik_request.pose_stamped = PoseStamped(pose=pose)
    req.ik_request.pose_stamped.header.frame_id = robot['frame']
    req.ik_request.robot_state.is_diff = True
    req.ik_request.robot_state.joint_state = JointState(name=list(robot['joints']), position=list(seed))
    req.ik_request.avoid_collisions = True
    req.ik_request.timeout = Duration(sec=int(timeout), nanosec=int((timeout % 1.0) * 1e9))
    return req


def fk_request(robot, q):
    req = GetPositionFK.Request()
    req.header.frame_id = robot['frame']
    req.fk_link_names = [robot['tip']]
    req.robot_state.joint_state.name = list(robot['joints'])
    req.robot_state.joint_state.position = list(q)
    return req


def line_request(robot, pose, start_q=None):
    req = GetCartesianPath.Request()
    req.header.frame_id = robot['frame']
    req.start_state.is_diff = True
    if start_q is not None:
        req.start_state.joint_state = JointState(name=list(robot['joints']), position=list(start_q))
    req.group_name = robot['group']
    req.link_name = robot['tip']
    req.waypoints = [pose]
    req.max_step = P['line_max_step']
    req.jump_threshold = 0.0
    req.revolute_jump_threshold = P['revolute_jump']
    req.avoid_collisions = True
    req.max_velocity_scaling_factor = P['velocity_scaling']
    req.max_acceleration_scaling_factor = P['acceleration_scaling']
    return req


def joint_goal(robot, positions):
    constraints = Constraints()
    for name, position in zip(robot['joints'], positions):
        constraints.joint_constraints.append(JointConstraint(joint_name=name, position=position, weight=1.0,
                                                             tolerance_above=P['joint_tolerance'], tolerance_below=P['joint_tolerance']))
    req = MotionPlanRequest()
    req.group_name = robot['group']
    req.goal_constraints = [constraints]
    req.num_planning_attempts = P['planning_attempts']
    req.allowed_planning_time = P['planning_time']
    req.max_velocity_scaling_factor = P['velocity_scaling']
    req.max_acceleration_scaling_factor = P['acceleration_scaling']
    req.start_state.is_diff = True
    goal = MoveGroup.Goal()
    goal.request = req
    goal.planning_options.plan_only = False
    goal.planning_options.planning_scene_diff.is_diff = True
    goal.planning_options.planning_scene_diff.robot_state.is_diff = True
    return goal


def forearm_turn(joints, traj):
    if len(joints) != 6 or len(traj.points) < 2:
        return 0.0
    k = traj.joint_names.index(joints[3])
    turns = [pt.positions[k] for pt in traj.points]
    return math.degrees(max(turns) - min(turns))


def near_turn(v, ref):
    return v + TURN * round((ref - v) / TURN)


def same_wrist(q, prev):
    if len(q) != 6:
        return q
    best = None
    for flip in (False, True):
        c = list(q)
        d = math.pi if flip else 0.0
        c[3], c[4], c[5] = near_turn(q[3] + d, prev[3]), -q[4] if flip else q[4], near_turn(q[5] + d, prev[5])
        cost = abs(c[3] - prev[3]) + abs(c[4] - prev[4]) + abs(c[5] - prev[5])
        if best is None or cost < best[0]:
            best = (cost, c)
    return best[1]


def keep_off_zero(q, prev=None):
    offset = math.radians(P['wrist_offset_deg'])
    if len(q) == 6 and P['singularity_avoidance'] and abs(q[4]) < offset:
        q[4] = math.copysign(offset, q[4] if q[4] else (prev[4] if prev else 1.0) or 1.0)
    return q


def chain_fk(chain, q):
    T, i, axes = np.eye(4), 0, []
    for origin, axis in chain:
        T = T @ origin
        if axis is not None:
            axes.append((T[:3, :3] @ axis, T[:3, 3].copy()))
            R = np.eye(4)
            R[:3, :3] = Rotation.from_rotvec(axis * q[i]).as_matrix()
            T, i = T @ R, i + 1
    J = np.zeros((6, len(q)))
    for i, (z, p) in enumerate(axes):
        J[:3, i], J[3:, i] = np.cross(z, T[:3, 3] - p), z
    return T, J


def pose_error(goal, T):
    return float(np.linalg.norm(goal[:3, 3] - T[:3, 3])), math.degrees(np.linalg.norm(Rotation.from_matrix(goal[:3, :3] @ T[:3, :3].T).as_rotvec()))


def dls_ik(chain, q, goal, locked, rot_weight):
    q = np.array(q, float)
    w = np.ones(len(q))
    w[list(locked)] = 0.0
    s = np.sqrt(w)
    for _ in range(P['dls_iterations']):
        T, J = chain_fk(chain, q)
        e = np.concatenate([goal[:3, 3] - T[:3, 3], rot_weight * Rotation.from_matrix(goal[:3, :3] @ T[:3, :3].T).as_rotvec()])
        J[3:] *= rot_weight
        Js = J * s
        dq = s * (Js.T @ np.linalg.solve(Js @ Js.T + P['dls_damping'] ** 2 * np.eye(6), e))
        q += dq
        if np.max(np.abs(dq)) < 1e-9:
            break
    return q, pose_error(goal, chain_fk(chain, q)[0])


def wrist_line(robot, q0, goal_pose):
    chain, names = robot['chain'], robot['joints']
    if chain is None or sum(axis is not None for _, axis in chain) != len(names):
        return None, 'no plain revolute chain from the URDF root to the tip'
    o, p = goal_pose.orientation, goal_pose.position
    goal = np.eye(4)
    goal[:3, :3], goal[:3, 3] = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix(), [p.x, p.y, p.z]
    start = chain_fk(chain, q0)[0]
    dist = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
    ends = Rotation.from_matrix([start[:3, :3], goal[:3, :3]])
    turn = Slerp([0.0, 1.0], ends)
    n = max(1, math.ceil(dist / P['own_line_step']), math.ceil(math.degrees((ends[0].inv() * ends[1]).magnitude()) / P['own_line_angle_deg']))
    six = len(names) == 6
    band, offset, step4 = math.radians(P['wrist_band_deg']), math.radians(P['wrist_offset_deg']), math.radians(P['wrist_turn_step_deg'])
    pts, t, out, worst = [np.array(q0, float)], 0.0, [(list(q0), 0.0)], (0.0, 0.0)
    for i in range(1, n + 1):
        way = np.eye(4)
        way[:3, :3], way[:3, 3] = turn([i / n])[0].as_matrix(), start[:3, 3] + (goal[:3, 3] - start[:3, 3]) * i / n
        prev = pts[-1]
        q, err = None, None
        if six and abs(prev[4]) < band:
            q, err = dls_ik(chain, prev, way, [3], P['dls_rot_weight_held'])
            if err[1] > P['wrist_hold_tilt_deg']:
                free = dls_ik(chain, prev, way, [], P['dls_rot_weight'])[0]
                q = np.array(prev, float)
                q[3] += float(np.clip(free[3] - prev[3], -step4, step4))
                q, err = dls_ik(chain, q, way, [3], P['dls_rot_weight_held'])
        else:
            q, err = dls_ik(chain, prev, way, [], P['dls_rot_weight'])
        if six and abs(q[4]) < offset:
            q[4] = math.copysign(offset, q[4] if q[4] else (prev[4] or 1.0))
            q, err = dls_ik(chain, q, way, [3, 4], P['dls_rot_weight_held'])
        if err[0] > P['own_line_pos_tol'] or err[1] > P['own_line_tilt_deg']:
            return None, f'{100 * i / n:.0f}% of the line is off by {err[0] * 1000:.2f} mm / {err[1]:.2f} deg'
        for j, (lo, hi) in enumerate(robot['limits']):
            if not lo - 1e-9 <= q[j] <= hi + 1e-9:
                return None, f'{names[j]} would leave its limit ({math.degrees(q[j]):.1f} deg) at {100 * i / n:.0f}% of the line'
        jump = float(np.max(np.abs(q - prev)))
        if jump > P['revolute_jump']:
            return None, f'joint jump of {math.degrees(jump):.0f} deg at {100 * i / n:.0f}% of the line'
        worst = (max(worst[0], err[0]), max(worst[1], err[1]))
        t += max(0.02, jump / P['joint_speed'], dist / n / P['own_line_speed'])
        pts.append(q)
        out.append((q.tolist(), t))
    low = min(abs(q[4]) for q in pts[1:]) if six else 0.0
    turned = math.degrees(max(q[3] for q in pts) - min(q[3] for q in pts)) if six else 0.0
    return out, (f'{n} points, {names[3]} turns {turned:.1f} deg, {names[4]} >= {math.degrees(low):.2f} deg, '
                 f'off the line <= {worst[0] * 1000:.2f} mm / {worst[1]:.2f} deg, {t:.1f} s')


def lowest_turns(wrap, joints, q, seed):
    out = list(q)
    for i, joint in enumerate(joints):
        if joint not in wrap or abs(q[i] - seed[i]) <= TURN / 2:
            continue
        lo, hi = wrap[joint]
        candidates = [q[i] + TURN * k for k in range(-4, 5) if lo - 1e-9 <= q[i] + TURN * k <= hi + 1e-9]
        out[i] = min(candidates, key=abs) if candidates else q[i]
    return out


def near_singular(margin, joints, q):
    for joint, limit in margin.items():
        if abs(q[joints.index(joint)]) < limit:
            return f'wrist singularity, {joint} = {math.degrees(q[joints.index(joint)]):.1f} deg (margin {math.degrees(limit):g} deg)'
    return None


def trajectory_singular(margin, joints, traj):
    if len(traj.points) < 2:
        return None
    first = dict(zip(traj.joint_names, traj.points[0].positions))
    margin = {j: min(limit, abs(first[j])) for j, limit in margin.items()}
    for pt in traj.points[1:]:
        pos = dict(zip(traj.joint_names, pt.positions))
        why = near_singular(margin, joints, [pos[j] for j in joints])
        if why:
            return why
    return None


class MotionServer(Node):
    def __init__(self, tcp_link=None):
        super().__init__('motion_server')
        self.tcp_link    = tcp_link
        self.robot       = None
        self.margin      = {}
        self.run_margin  = {}
        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.ik_client   = self.create_client(GetPositionIK, '/compute_ik')
        self.line_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.fk_client   = self.create_client(GetPositionFK, '/compute_fk')
        self.param_client = self.create_client(GetParameters, '/move_group/get_parameters')
        self.result_pub  = self.create_publisher(String, RESULT_TOPIC, 10)
        self.slow_pub    = self.create_publisher(Bool, SLOW_TOPIC, 10)
        self.ready_pub   = self.create_publisher(String, READY_TOPIC, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, COMMAND_TOPIC, self.cb_command, 10)
        self.create_subscription(JointState, '/joint_states', self.cb_joint_states, 10)
        self.seed        = []
        self.busy        = False
        self.dry         = False
        self.cmd_id      = None
        self.queue       = []
        self.goal_q      = []
        self.move_tries  = 0
        self.retry       = None
        self.move_id     = 0
        self.move_handle = None
        self.move_timer  = self.create_timer(P['timeout'], self.cb_move_timeout)
        self.move_timer.cancel()
        self.joint_pos   = {}
        self.path        = None
        self.asked       = False
        self.init_timer  = self.create_timer(P['poll'], self.cb_init)
        self.get_logger().info(f'motion server: waiting for move_group (URDF, SRDF), /joint_states: {COMMAND_TOPIC} -> {RESULT_TOPIC}')

    def servers_ready(self):
        return (self.move_client.server_is_ready() and self.ik_client.service_is_ready() and self.line_client.service_is_ready()
                and self.exec_client.server_is_ready() and self.fk_client.service_is_ready())

    def cb_joint_states(self, msg):
        self.joint_pos = dict(zip(msg.name, msg.position))

    def cb_init(self):
        if self.robot is None:
            if not self.asked and self.param_client.service_is_ready():
                self.asked = True
                self.param_client.call_async(GetParameters.Request(names=['robot_description', 'robot_description_semantic'])).add_done_callback(self.cb_robot)
            return
        if not self.servers_ready() or any(j not in self.joint_pos for j in self.robot['joints']):
            return
        self.init_timer.cancel()
        self.fk_client.call_async(fk_request(self.robot, self.current())).add_done_callback(self.cb_home)

    def cb_robot(self, future):
        res = future.result()
        try:
            self.robot = read_robot(res.values[0].string_value, res.values[1].string_value, self.tcp_link)
        except (ValueError, KeyError, ET.ParseError, IndexError) as e:
            self.get_logger().error(f'cannot read the robot from the URDF and SRDF of move_group: {e}')
            self.asked = False
            return
        joints_ = self.robot['joints']
        self.margin = {joints_[4]: math.radians(P['wrist_margin_deg'])} if len(joints_) == 6 else {}
        self.run_margin = {joints_[4]: math.radians(P['wrist_margin_run_deg'])} if len(joints_) == 6 else {}
        self.get_logger().info(f"robot: group {self.robot['group']}, frame {self.robot['frame']}, tip {self.robot['tip']}, joints {joints_}, "
                               f"TCP {self.robot['tcp_length']:.3f} m along {np.round(self.robot['tool_axis'], 3).tolist()}, turn joints {list(self.robot['wrap'])}")

    def cb_home(self, future):
        res = future.result()
        if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS or not res.pose_stamped:
            self.get_logger().error('forward kinematics of the start pose failed -> no home pose')
            self.init_timer.reset()
            return
        tcp = offset_pose(res.pose_stamped[0].pose, self.robot['tcp_length'], self.robot['tool_axis'])
        q, p = tcp.orientation, tcp.position
        rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('XYZ', degrees=True)
        home = [p.x, p.y, p.z, *[float(a) for a in rpy]]
        self.get_logger().info('home pose = TCP of the start pose of the robot: ' + ' '.join(f'{v:.3f}' for v in home))
        self.ready_pub.publish(String(data=dumps({'home': home})))

    def flange_of(self, pose):
        q = pose.orientation
        d = Rotation.from_quat([q.x, q.y, q.z, q.w]).apply(self.robot['tcp_offset'])
        p = pose.position
        return Pose(position=Point(x=p.x - d[0], y=p.y - d[1], z=p.z - d[2]), orientation=q)

    def to_flange(self, step):
        out = {k: (self.flange_of(v) if isinstance(v, Pose) else v) for k, v in step.items()}
        if 'steps' in out:
            out['steps'] = [self.to_flange(s) for s in out['steps']]
        return out

    def current(self):
        return [self.joint_pos[j] for j in self.robot['joints']]

    def cb_command(self, msg):
        data = json.loads(msg.data)
        if self.robot is None or any(j not in self.joint_pos for j in self.robot['joints']):
            self.result_pub.publish(String(data=dumps({'id': data['id'], 'ok': False, 'msg': 'not ready'})))
            return
        if self.busy:
            self.get_logger().warn(f"busy -> command {data['id']} ignored")
            self.result_pub.publish(String(data=dumps({'id': data['id'], 'ok': False, 'msg': 'busy'})))
            return
        self.busy, self.cmd_id, self.dry = True, data['id'], bool(data.get('dry'))
        if data.get('reset_seed') or not self.seed:
            self.seed = self.current()
        step = self.to_flange(unpack(data['step']))
        self.slow_pub.publish(Bool(data=bool(step.get('slow'))))
        if self.dry and step['kind'] != 'path':
            self.reply(True)
        elif step['kind'] == 'path':
            self.queue = []
            self.run_path(step)
        else:
            self.queue = [step]
            self.run_next()

    def reply(self, ok, message=''):
        self.result_pub.publish(String(data=dumps({'id': self.cmd_id, 'ok': ok, 'msg': message})))
        self.busy, self.cmd_id, self.queue, self.path, self.dry = False, None, [], None, False

    def run_next(self):
        if not self.queue:
            self.reply(True)
            return
        self.execute(self.queue.pop(0))

    def fail(self, message):
        self.move_timer.cancel()
        self.get_logger().error(message)
        self.reply(False, message)

    def cb_move_timeout(self):
        self.move_timer.cancel()
        self.move_id += 1
        if self.move_handle is not None:
            self.move_handle.cancel_goal_async()
        self.fail(f"move not finished after {P['timeout']} s -> cancelled")

    def execute(self, step):
        if step['kind'] == 'line':
            self.move_tries = 0
            self.plan_line(step)
        elif step['kind'] == 'joints':
            self.seed, self.goal_q, self.move_tries = list(step['value']), list(step['value']), 0
            self.get_logger().info(f"[{step['label']}] joints: " + ' '.join(f'{j}={math.degrees(v):.1f}' for j, v in zip(self.robot['joints'], step['value'])))
            self.send_move()
        elif step['kind'] == 'move':
            self.solve_best(step)
        else:
            self.ask_ik(step['value']).add_done_callback(lambda f: self.cb_ik(f, step))

    def branch_seeds(self):
        if len(self.robot['joints']) != 6:
            return []
        return [[math.radians(v) for v in (j1, P['branch_j2_deg'], j3, 0.0, j5, 0.0)]
                for j1 in P['branch_j1_deg'] for j3 in P['branch_j3_deg'] for j5 in P['branch_j5_deg']]

    def solve_best(self, step):
        self.best_ik(step['value'], self.current(), lambda q, n: self.after_best(step, q, n))

    def best_ik(self, pose, start, done):
        self.branch_next({'pose': pose, 'seeds': [list(start)] + self.branch_seeds(), 'index': 0, 'found': [], 'start': list(start), 'done': done})

    def branch_next(self, pending):
        i = pending['index']
        if i >= len(pending['seeds']):
            names, start = self.robot['joints'], pending['start']

            def cost(q):
                odd = len(q) == 6 and (abs(q[3]) > math.radians(P['flip_forearm_deg']) or q[2] > math.radians(P['flip_elbow_deg']))
                return sum(abs(a - b) for a, b in zip(q, start)) + (100.0 if near_singular(self.margin, names, q) else 0.0) + (50.0 if odd else 0.0)
            pending['done'](min(pending['found'], key=cost) if pending['found'] else None, len(pending['found']))
            return
        pending['index'] += 1
        req = ik_request(self.robot, pending['pose'], pending['seeds'][i], P['ik_timeout'] if i == 0 else P['branch_timeout'])
        self.ik_client.call_async(req).add_done_callback(lambda f: self.cb_branch(f, pending))

    def cb_branch(self, future, pending):
        res = future.result()
        if res is not None and res.error_code.val == MoveItErrorCodes.SUCCESS:
            names = self.robot['joints']
            solution = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
            pending['found'].append(lowest_turns(self.robot['wrap'], names, [solution[j] for j in names], pending['start']))
        self.branch_next(pending)

    def after_best(self, step, q, n):
        if q is None:
            self.fail(f"[{step['label']}] no IK solution")
            return
        self.seed = q
        self.get_logger().info(f"[{step['label']}] IK: " + ' '.join(f'{j}={math.degrees(v):.1f}' for j, v in zip(self.robot['joints'], q))
                               + f" (nearest of {n} solutions)")
        self.goal_q = q
        self.move_tries = 0
        self.send_move()

    def cb_path_best(self, sub, q):
        if q is None:
            self.path_abort(self.path['step'], f"[{sub['label']}] no IK solution", self.path['seed0'])
            return
        self.seed = q
        self.path_add_joint_move(q)
        self.path_next()

    def run_path(self, step, again=False):
        if not again:
            self.move_tries = 0
        self.retry = lambda: self.run_path(step, again=True)
        q0 = self.current()
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
            self.line_client.call_async(line_request(self.robot, sub['value'], p['q'])).add_done_callback(lambda f: self.cb_path_line(f, sub))
        elif sub['kind'] == 'move' and not sub.get('near'):
            self.best_ik(sub['value'], list(p['q']), lambda q, n: self.cb_path_best(sub, q))
        else:
            self.ask_ik(sub['value']).add_done_callback(lambda f: self.cb_path_ik(f, sub))

    def cb_path_ik(self, future, sub):
        res = future.result()
        if res.error_code.val != MoveItErrorCodes.SUCCESS and sub['kind'] == 'ik':
            self.path_next()
            return
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.path_abort(self.path['step'], f"[{sub['label']}] no IK solution", self.path['seed0'])
            return
        solution = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        q = lowest_turns(self.robot['wrap'], self.robot['joints'], [solution[j] for j in self.robot['joints']], self.seed)
        if sub.get('near'):
            q = same_wrist(q, self.seed)
        if sub.get('strict') and near_singular(self.margin, self.robot['joints'], q):
            self.path_abort(self.path['step'], f"[{sub['label']}] IK solution at a wrist singularity", self.path['seed0'])
            return
        self.seed = q
        if sub['kind'] != 'ik':
            self.path_add_joint_move(q)
        self.path_next()

    def line_problem(self, res, strict, dry):
        traj = res.solution.joint_trajectory
        if res.error_code.val != MoveItErrorCodes.SUCCESS or res.fraction < P['min_fraction'] or len(traj.points) < 2:
            return f"only {res.fraction * 100:.0f}% possible ({ERROR_NAMES.get(res.error_code.val, res.error_code.val)})"
        if strict:
            why = self.strict_problem(traj, dry)
            if why:
                return why
        if len(self.robot['joints']) == 6:
            k = traj.joint_names.index(self.robot['joints'][4])
            low = min(abs(pt.positions[k]) for pt in traj.points[1:])
            if low < math.radians(P['wrist_band_deg']):
                return f"{self.robot['joints'][4]} comes to {math.degrees(low):.2f} deg"
        return None

    def cb_path_line(self, future, sub):
        res, p = future.result(), self.path
        traj = res.solution.joint_trajectory
        if res.error_code.val == MoveItErrorCodes.SUCCESS and res.fraction >= P['min_fraction'] and len(traj.points) < 2:
            self.path_next()
            return
        problem = self.line_problem(res, sub.get('strict'), self.dry)
        if problem:
            if not P['singularity_avoidance']:
                self.path_abort(p['step'], f"[{sub['label']}] straight line {problem}", p['seed0'])
                return
            self.own_line(p['q'], sub['value'], lambda pts, note: self.cb_path_own(sub, problem, pts, note))
            return
        t0 = p['t']
        for pt in traj.points[1:]:
            pos = dict(zip(traj.joint_names, pt.positions))
            self.path_point([pos[j] for j in self.robot['joints']], t0 + pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9)
        self.seed = list(p['q'])
        p['moves'] += 1
        self.path_next()

    def cb_path_own(self, sub, problem, pts, note):
        p = self.path
        why = note if pts is None else self.own_refused(pts, sub.get('strict'), self.dry)
        if why:
            self.path_abort(p['step'], f"[{sub['label']}] straight line {problem}; with the wrist correction: {why}", p['seed0'])
            return
        t0 = p['t']
        for q, t in pts[1:]:
            self.path_point(q, t0 + t)
        self.seed = list(p['q'])
        p['moves'] += 1
        (self.get_logger().debug if self.dry else self.get_logger().info)(f"[{sub['label']}] wrist correction ({problem}): {note}")
        self.path_next()

    def own_refused(self, pts, strict, dry):
        limit = P['strict_forearm_turn_deg'] if dry else P['strict_forearm_turn_run_deg']
        if strict and len(self.robot['joints']) == 6:
            turns = [q[3] for q, _ in pts]
            if math.degrees(max(turns) - min(turns)) > limit:
                return f"wrist twist, {self.robot['joints'][3]} turns {math.degrees(max(turns) - min(turns)):.0f} deg (limit {limit:g} deg)"
        return None

    def own_line(self, q0, goal, done):
        pts, note = wrist_line(self.robot, q0, goal)
        done(pts, note)

    def path_point(self, q, t):
        p = self.path
        q = keep_off_zero(list(q), p['q'])
        p['t'] = max(t, p['t'] + 0.001)
        p['points'].append(JointTrajectoryPoint(positions=list(q), time_from_start=Duration(sec=int(p['t']), nanosec=int((p['t'] % 1.0) * 1e9))))
        p['q'] = list(q)

    def path_add_joint_move(self, q):
        p = self.path
        d = max(abs(a - b) for a, b in zip(q, p['q']))
        n = max(1, math.ceil(d / P['path_joint_step']))
        q0 = list(p['q'])
        for i in range(1, n + 1):
            self.path_point([a + (b - a) * i / n for a, b in zip(q0, q)], p['t'] + max(0.02, d / n / P['joint_speed']))
        p['moves'] += 1

    def path_abort(self, step, why, seed):
        if self.dry:
            self.fail(f"[{step['label']}] reach check: {why}")
            return
        self.seed = list(seed)
        self.fail(f"[{step['label']}] {why} -> stopped here, no improvised joint-space move")

    def path_send(self):
        p = self.path
        if self.dry and len(self.robot['joints']) == 6:
            turns = [pt.positions[3] for pt in p['points']]
            spread = math.degrees(max(turns) - min(turns))
            if spread > P['reach_forearm_turn_deg']:
                self.fail(f"[{p['step']['label']}] reach check: {self.robot['joints'][3]} would turn {spread:.0f} deg on the way (wrist twist: limit {P['reach_forearm_turn_deg']:g} deg)")
                return
        if self.dry:
            self.get_logger().info(f"[{p['step']['label']}] reach check: the whole path is possible ({p['moves']} moves, {p['t']:.1f} s)")
            self.reply(True)
            return
        jt = JointTrajectory(joint_names=list(self.robot['joints']), points=p['points'])
        self.get_logger().info(f"[{p['step']['label']}] one path: {p['moves']} moves, {len(p['points'])} points, {p['t']:.1f} s")
        self.path = None
        self.send_goal(self.exec_client, ExecuteTrajectory.Goal(trajectory=RobotTrajectory(joint_trajectory=jt)))

    def strict_problem(self, traj, dry):
        why = trajectory_singular(self.margin if dry else self.run_margin, self.robot['joints'], traj)
        if why:
            return why
        limit = P['strict_forearm_turn_deg'] if dry else P['strict_forearm_turn_run_deg']
        turn = forearm_turn(self.robot['joints'], traj)
        return f"wrist twist, {self.robot['joints'][3]} turns {turn:.0f} deg on this straight line (limit {limit:g} deg)" if turn > limit else None

    def plan_line(self, step):
        self.line_client.call_async(line_request(self.robot, step['value'])).add_done_callback(lambda f: self.cb_line(f, step))

    def cb_line(self, future, step):
        res = future.result()
        traj = res.solution.joint_trajectory
        if res.error_code.val == MoveItErrorCodes.SUCCESS and res.fraction >= P['min_fraction'] and len(traj.points) < 2:
            self.run_next()
            return
        problem = self.line_problem(res, step.get('strict'), False)
        if problem:
            if not P['singularity_avoidance']:
                self.fail(f"[{step['label']}] straight line {problem} -> stopped here, no improvised joint-space move")
                return
            self.own_line(self.current(), step['value'], lambda pts, note: self.cb_line_own(step, problem, pts, note))
            return
        end = dict(zip(traj.joint_names, traj.points[-1].positions))
        self.seed = [end[j] for j in self.robot['joints']]
        t = traj.points[-1].time_from_start
        self.get_logger().info(f"[{step['label']}] straight line: {len(traj.points)} points, {t.sec + t.nanosec * 1e-9:.1f} s, "
                               f"ends at " + ' '.join(f'{j}={math.degrees(end[j]):.1f}' for j in self.robot['joints']))
        self.retry = lambda: self.plan_line(step)
        self.send_goal(self.exec_client, ExecuteTrajectory.Goal(trajectory=res.solution))

    def cb_line_own(self, step, problem, pts, note):
        why = note if pts is None else self.own_refused(pts, step.get('strict'), False)
        if why:
            self.fail(f"[{step['label']}] straight line {problem}; with the wrist correction: {why} -> stopped here, no improvised joint-space move")
            return
        self.seed = list(pts[-1][0])
        self.get_logger().info(f"[{step['label']}] wrist correction ({problem}): {note}")
        jt = JointTrajectory(joint_names=list(self.robot['joints']),
                             points=[JointTrajectoryPoint(positions=list(q), time_from_start=Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))) for q, t in pts])
        self.retry = lambda: self.plan_line(step)
        self.send_goal(self.exec_client, ExecuteTrajectory.Goal(trajectory=RobotTrajectory(joint_trajectory=jt)))

    def ask_ik(self, pose):
        return self.ik_client.call_async(ik_request(self.robot, pose, self.seed, P['ik_timeout']))

    def cb_ik(self, future, step):
        res = future.result()
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.fail(f"[{step['label']}] no IK solution")
            return
        names = self.robot['joints']
        solution = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        found = [solution[j] for j in names]
        q = lowest_turns(self.robot['wrap'], names, found, self.seed)
        self.seed = q
        turned = ''.join(f' {j} {math.degrees(a):.1f}->{math.degrees(b):.1f}' for j, a, b in zip(names, found, q) if abs(a - b) > 1e-6)
        log = self.get_logger().debug if step['kind'] == 'ik' else self.get_logger().info
        log(f"[{step['label']}] IK: " + ' '.join(f'{j}={math.degrees(v):.1f}' for j, v in zip(names, q)) + (f' (lowest turn:{turned})' if turned else ''))
        if step['kind'] == 'ik':
            self.run_next()
            return
        self.goal_q = q
        self.move_tries = 0
        self.send_move()

    def send_move(self):
        self.retry = self.send_move
        self.send_goal(self.move_client, joint_goal(self.robot, self.goal_q))

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
        self.move_timer.cancel()
        code = future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            if self.move_tries >= P['retries']:
                self.fail(f'failed: {ERROR_NAMES.get(code, code)}')
                return
            self.move_tries += 1
            self.get_logger().warn(f"{ERROR_NAMES.get(code, code)} -> trying the move again ({self.move_tries}/{P['retries']})")
            self.retry()
            return
        self.get_logger().info('arrived')
        self.run_next()


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tcp-link', help='link of the TCP (default: the link named *tcp* under the tip link, else the tip link itself)')
    parser.add_argument('--robot-config', help='robot yaml whose `control:` section overrides the limits of this node')
    cli, ros_args = parser.parse_known_args(args)
    if cli.robot_config:
        own = (yaml.safe_load(open(cli.robot_config)) or {}).get('control') or {}
        unknown = [k for k in own if k not in P]
        if unknown:
            raise SystemExit(f'{cli.robot_config}: control: has unknown keys {unknown} (known: {sorted(P)})')
        P.update(own)
    rclpy.init(args=ros_args)
    node = MotionServer(cli.tcp_link)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
