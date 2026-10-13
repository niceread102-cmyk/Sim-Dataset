"""Pick & place sequence control for any robot: /target_pose (Pose, TCP target in base_link) -> MoveIt2 /move_action + /gripper/command

    python3 control/robot_sequence.py [--robot ur5e]          (default: the robot of isaacsim/configs/robots/select_robot.yaml)

The task parameters (approach distance, delays, tolerances, topics, ...) are in isaacsim/configs/sequence/robot_sequence.yaml, the
same for every robot; isaacsim/configs/sequence/<robot>.yaml (optional) overrides what differs for one robot. Everything about the
robot comes from its own files: planning group, arm joints, base frame, end frame, TCP length and IK seed from
bringup/<robot>/config/ik_node.yaml (made by isaacsim/make_robot_config.py), the gripper topic and joint from
isaacsim/configs/robots/<robot>.yaml and the gripper URDF (a robot without a gripper skips the gripper steps), the basket from
isaacsim/configs/map/map.yaml. The home pose is the TCP of the ready pose (ik seed) and the place pose is above the basket,
unless poses.home / poses.place give a pose.

At start (once move_group is up) the gripper is closed and the robot moves to the home pose to initialize, then it
waits for /target_pose. On /target_pose it moves in the order below and ends at the home pose again:

    close the gripper (a cancelled sequence may have left it open) -> approach -> insert (after delays.descend) -> open the gripper (after delays.open)
    -> rise back to the approach pose (after delays.rise) -> home pose -> place pose (above the basket)
    -> lower into the basket (down to basket.release_height above its floor) -> close the gripper (lets go of the part)
    -> lift back to the place pose (after delays.gripper_wait) -> home pose

The basket is the one of the Isaac Sim scene: basket.map_config points at the map (isaacsim/configs/map/map.yaml) whose `scene: basket:` section creates it.
Its position and wall thickness give the height of its floor, and poses.place has to be above it (checked at start).

The moves along the tool axis (insert / rise at the target, lower / lift in the basket) are straight lines: MoveIt's
/compute_cartesian_path solves the IK every line.max_step along the line from where the arm is, and /execute_trajectory
runs it. A joint goal would interpolate the joints, and the tool would follow a slight curve. If less than
line.min_fraction of a line can be computed (out of reach, the arm would jump to another configuration), the step falls
back to a joint goal (with a warning).

The other moves (approach, home, place) are joint goals solved by /compute_ik with a seed (ik.seed for the first step of a
sequence, the end of the previous step after that). A plain pose goal lets MoveIt pick any IK solution, and a pose like
the home pose has two very different ones (natural "L" shape and a folded "Z" shape of the arm); the seeded IK always takes
the one nearest to the seed. That only holds for a seed that is near the target: a joint-space fallback of a long line
first solves the IK at intermediate poses (line.ik_step apart, no motion), so it stays on the same arm configuration.
The approach pose is just as far from the ready pose (about 1 m lower, the tool turned and tilted to the part): one IK call
from there returned a different, folded arm (elbow over the top, base turned the wrong way) on different calls. So the
approach is solved the same way: IK-only steps from the home pose, line.ik_step m and line.ik_angle_step deg apart.

The delays start when the real state in /joint_states says the previous action is finished (the arm has stopped after
a move, the gripper joint has reached its position or stopped short of it because the part blocks the jaws), not when
MoveIt reports success. The log shows how long that took ("arm stopped", "gripper done") and where the gripper stopped.

The first move (to the home pose) waits until move_group, /compute_ik and /joint_states are up and is tried again if
move_group rejects it right after launch (startup.retries). Any other move that fails (TIMED_OUT, CONTROL_FAILED, ...)
is tried again once the arm has stopped (motion.retries; a straight line is computed again from where the arm is) before
the sequence is cancelled. MoveIt itself has no time limit
(execution duration monitoring is off in the bringup config): a move that has not finished after motion.timeout (e.g.
move_group died) is cancelled here and stops the sequence.

Target value: send the rpy that compute_grasp_tcp_pose.py converted from the object orientation to a TCP orientation
with ee_6d.py (x y z roll pitch yaw).
It receives the same /target_pose as ik_node.py, so the two must not run together
(sequence:=true of the launch leaves ik_node out and starts this node).
"""
import argparse
import math
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import rclpy
import yaml
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation

SIM_DIR = Path(__file__).resolve().parents[1]
ISAAC_DIR = SIM_DIR / 'isaacsim'
CONFIG_FILE = ISAAC_DIR / 'configs' / 'sequence' / 'robot_sequence.yaml'   # the task, the same for every robot
ROBOTS_DIR = ISAAC_DIR / 'configs' / 'robots'
SELECT_FILE = ROBOTS_DIR / 'select_robot.yaml'

# Keys of the merged config (checked by load_config). robot, ik.seed, the gripper and the poses are filled in from the files of the robot
REQUIRED_KEYS = {
    'robot':     ['planning_group', 'arm_joints', 'base_link', 'tip_link', 'tool_axis', 'tcp_length'],
    'poses':     ['home', 'place'],
    'approach':  ['distance'],
    'basket':    ['map_config', 'release_height'],
    'line':      ['service', 'action', 'max_step', 'revolute_jump_threshold', 'min_fraction', 'ik_step', 'ik_angle_step'],
    'ik':        ['service', 'seed', 'timeout', 'retry_timeout', 'avoid_collisions'],
    'delays':    ['descend', 'open', 'rise', 'gripper_wait'],
    'startup':   ['poll', 'retries'],
    'settle':    ['topic', 'arm_window', 'arm_tolerance', 'gripper_tolerance', 'gripper_grace', 'gripper_window', 'timeout'],
    'tolerance': ['joint', 'waypoint_position'],
    'motion':    ['velocity_scaling', 'acceleration_scaling', 'planning_time', 'planning_attempts', 'retries', 'timeout'],
    'gripper':   ['topic', 'open', 'close', 'enabled'],
    'topics':    ['target_pose', 'move_action'],
}

ERROR_NAMES = {v: k for k, v in vars(MoveItErrorCodes).items() if k.isupper() and isinstance(v, int)}


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def robot_name(choice=None):
    """The robot of the sequence: --robot, else `robot:` of select_robot.yaml (the one `lab robot_sim.py` starts)."""
    name = choice or str((yaml.safe_load(open(SELECT_FILE)) or {}).get('robot') or '')
    if not name:
        raise ValueError(f'no robot: pass --robot <name> or set `robot:` in {SELECT_FILE}')
    return name


def gripper_joint(robot, robot_cfg):
    """(joint, lower, upper) of the gripper of the robot (positions in m), None for a robot without one: the first joint
    that is not a mimic joint of the gripper URDF (`gripper:` of the robot yaml), or the first of `gripper_joints:` (a
    gripper that is part of the robot URDF)."""
    names = list(robot_cfg.get('gripper_joints') or [])
    gripper = str(robot_cfg.get('gripper') or 'none')
    if gripper.lower() not in ('none', 'null'):
        urdf = SIM_DIR / 'assets' / gripper / 'urdf' / f'{gripper}.urdf'
    elif names:                                      # assets/<robot>/urdf/<robot>.urdf, or <robot>.urdf in another assets/*/urdf/
        urdf = SIM_DIR / 'assets' / robot / 'urdf' / f'{robot}.urdf'
        urdf = urdf if urdf.is_file() else next(iter(sorted((SIM_DIR / 'assets').glob(f'*/urdf/{robot}.urdf'))), urdf)
    else:
        return None
    joints = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint') if j.get('type') in ('revolute', 'continuous', 'prismatic')}
    names = names or [n for n, j in joints.items() if j.find('mimic') is None]
    if not names or names[0] not in joints:
        raise ValueError(f'gripper joint {names[:1] or "?"} not found in {urdf}')
    lim = joints[names[0]].find('limit')
    return names[0], float(lim.get('lower', 0.0)), float(lim.get('upper', 0.0))


def load_config(path=CONFIG_FILE, robot=None):
    """The task parameters (+ the overrides of the robot) with everything about the robot filled in from its own files. Raises
    a clear error if a key is missing, the seed does not match the joints or poses.place is not above the basket. Adds
    basket.floor_z (top of the basket floor, m, from the basket of the Isaac Sim scene)"""
    robot = robot_name(robot)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    own = Path(path).parent / f'{robot}.yaml'                              # optional: what differs for this robot
    if own.is_file():
        cfg = deep_merge(cfg, yaml.safe_load(open(own)) or {})
    ik_file = SIM_DIR / 'bringup' / robot / 'config' / 'ik_node.yaml'
    if not ik_file.is_file():
        raise ValueError(f'{ik_file} not found: python3 isaacsim/make_robot_config.py robot:={robot}')
    ik = yaml.safe_load(open(ik_file))
    robot_file = ROBOTS_DIR / f'{robot}.yaml'
    robot_cfg = (yaml.safe_load(open(robot_file)) or {}) if robot_file.is_file() else {}

    tcp = [float(x) for x in ik.get('tcp_offset', [0.0, 0.0, 0.0])]       # TCP in the end frame: length and direction
    length = math.sqrt(sum(x * x for x in tcp))
    cfg['robot'] = {'name': robot, 'planning_group': ik['group_name'], 'arm_joints': list(ik['arm_joints']),
                    'base_link': ik['base_frame'], 'tip_link': ik['ee_link'],
                    'tool_axis': [x / length for x in tcp] if length > 1e-9 else [0.0, 0.0, 1.0], 'tcp_length': length}
    cfg.setdefault('ik', {}).setdefault('seed', list(ik['ik_seed_deg']))
    if len(cfg['ik']['seed']) != len(cfg['robot']['arm_joints']):
        raise ValueError(f'ik.seed needs one value per arm joint ({len(cfg["robot"]["arm_joints"])}): {cfg["robot"]["arm_joints"]}')

    g = cfg.setdefault('gripper', {})
    g['topic'] = str(robot_cfg.get('gripper_topic') or g.get('topic') or '/gripper/command')
    grip = gripper_joint(robot, robot_cfg)
    g['enabled'] = grip is not None
    cfg.setdefault('settle', {})['gripper_joint'] = None
    if grip:
        cfg['settle']['gripper_joint'], g['position_closed'], g['position_open'] = grip

    sim_file = Path(path).parent / cfg['basket']['map_config']
    with open(sim_file) as f:
        sim = ((yaml.safe_load(f) or {}).get('scene') or {}).get('basket') or {}
    lacking = [key for key in ('pos_m', 'size_m', 'wall_thickness_m') if key not in sim]
    if lacking:
        raise ValueError(f'{sim_file}: the basket: section is missing {", ".join(lacking)}')
    cfg['basket']['floor_z'] = sim['pos_m'][2] + sim['wall_thickness_m']   # top of the basket floor
    poses = cfg.setdefault('poses', {})
    poses.setdefault('home', 'ready')                                      # the TCP of the ready pose (asked from MoveIt at start)
    if poses.get('place', 'auto') == 'auto':                               # above the middle of the basket
        poses['place'] = [sim['pos_m'][0], sim['pos_m'][1], sim['pos_m'][2] + sim['size_m'][2] + float(poses.get('place_clearance', 0.45)), 180, 0, 0]
    missing = [f'{section}.{key}' for section, keys in REQUIRED_KEYS.items() for key in keys
               if key not in (cfg.get(section) or {})]
    if missing:
        raise ValueError(f'{path}: missing keys: {", ".join(missing)}')
    inside = min(sim['size_m'][:2]) / 2 - sim['wall_thickness_m']          # radius of the circle between the walls
    x, y = poses['place'][:2]
    if math.hypot(x - sim['pos_m'][0], y - sim['pos_m'][1]) > inside:
        raise ValueError(f'{path}: poses.place (x {x}, y {y}) is not above the basket '
                         f'(centre {sim["pos_m"][:2]}, {inside:.3f} m between the walls) of {sim_file}')
    return cfg


def offset_pose(pose, dist, axis):
    """pose moved by dist (m) along its own tool axis (axis, expressed in the tip_link frame). Orientation unchanged"""
    q = pose.orientation
    direction = Rotation.from_quat([q.x, q.y, q.z, q.w]).apply(axis)  # tool axis in base_link
    p = pose.position
    return Pose(position=Point(x=p.x + dist * direction[0], y=p.y + dist * direction[1], z=p.z + dist * direction[2]),
                orientation=q)


def shift_toward_base(pose, dist):
    """pose moved by dist (m) toward the origin of the base frame. Orientation unchanged"""
    p = pose.position
    norm = math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
    return Pose(position=Point(x=p.x - dist * p.x / norm, y=p.y - dist * p.y / norm, z=p.z - dist * p.z / norm),
                orientation=pose.orientation)


def xyzrpy_to_pose(x, y, z, roll, pitch, yaw):
    """x y z (m) + roll pitch yaw (deg) -> Pose. Same conversion as ee_6d.py"""
    q = Rotation.from_euler('XYZ', [roll, pitch, yaw], degrees=True).as_quat()  # [x, y, z, w]
    p = Pose(position=Point(x=float(x), y=float(y), z=float(z)))
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p


def tcp_to_flange(pose, cfg):
    """TCP (gripper tip) target -> flange target sent to MoveIt (pulled back by tcp_length against the TCP z axis)"""
    return offset_pose(pose, -cfg['robot']['tcp_length'], cfg['robot']['tool_axis'])


def build_ik_request(pose, seed, cfg, timeout):
    """GetPositionIK request for the tip_link pose; the solver starts from the joint values in seed (rad)"""
    robot, ik = cfg['robot'], cfg['ik']
    stamped = PoseStamped()
    stamped.header.frame_id = robot['base_link']
    stamped.pose = pose
    req = GetPositionIK.Request()
    req.ik_request.group_name              = robot['planning_group']
    req.ik_request.ik_link_name            = robot['tip_link']
    req.ik_request.pose_stamped            = stamped
    req.ik_request.robot_state.is_diff     = True   # only the seed joints are given, the rest stays as move_group knows it
    req.ik_request.robot_state.joint_state = JointState(name=list(robot['arm_joints']), position=list(seed))
    req.ik_request.avoid_collisions        = ik['avoid_collisions']
    req.ik_request.timeout                 = Duration(sec=int(timeout), nanosec=int((timeout % 1.0) * 1e9))
    return req


def build_joint_goal(positions, cfg):
    """MoveGroup goal (plan + execute) that brings every arm joint to positions (rad)"""
    tol, motion = cfg['tolerance']['joint'], cfg['motion']
    constraints = Constraints()
    for name, position in zip(cfg['robot']['arm_joints'], positions):
        constraints.joint_constraints.append(JointConstraint(joint_name=name, position=position, weight=1.0,
                                                             tolerance_above=tol, tolerance_below=tol))
    req = MotionPlanRequest()
    req.group_name                      = cfg['robot']['planning_group']
    req.goal_constraints                = [constraints]
    req.num_planning_attempts           = motion['planning_attempts']
    req.allowed_planning_time           = motion['planning_time']
    req.max_velocity_scaling_factor     = motion['velocity_scaling']
    req.max_acceleration_scaling_factor = motion['acceleration_scaling']
    # Empty state + is_diff=True = "start from the pose move_group knows now" (move_group subscribes to /joint_states itself)
    req.start_state.is_diff = True
    goal = MoveGroup.Goal()
    goal.request = req
    goal.planning_options.planning_scene_diff.is_diff             = True
    goal.planning_options.planning_scene_diff.robot_state.is_diff = True
    # False = plan + execute. move_group sends the execution to the controller of the robot (FollowJointTrajectory)
    goal.planning_options.plan_only = False
    return goal


def build_line_request(pose, cfg):
    """GetCartesianPath request: straight line of the tip_link from the pose move_group knows now to pose"""
    robot, line, motion = cfg['robot'], cfg['line'], cfg['motion']
    req = GetCartesianPath.Request()
    req.header.frame_id                 = robot['base_link']
    req.start_state.is_diff             = True   # empty + is_diff = start where the arm is now
    req.group_name                      = robot['planning_group']
    req.link_name                       = robot['tip_link']
    req.waypoints                       = [pose]
    req.max_step                        = line['max_step']
    req.jump_threshold                  = 0.0    # relative check off, the absolute one below is used
    req.revolute_jump_threshold         = line['revolute_jump_threshold']
    req.avoid_collisions                = cfg['ik']['avoid_collisions']
    req.max_velocity_scaling_factor     = motion['velocity_scaling']
    req.max_acceleration_scaling_factor = motion['acceleration_scaling']
    return req


def move(label, pose, delay=0.0, search=False):
    """Move step: wait delay s, then move to the flange pose. search=True: if the exact pose has no IK solution, try
    poses moved toward the base by up to tolerance.waypoint_position (for waypoints that need not be exact)"""
    return {'label': label, 'kind': 'move', 'value': pose, 'delay': delay, 'search': search}


def line(label, pose, start, delay=0.0):
    """Straight-line step: wait delay s, then move the flange along a straight line to pose (start: the pose the previous
    step ends at, only used by the joint-space fallback)"""
    return {'label': label, 'kind': 'line', 'value': pose, 'start': start, 'delay': delay, 'search': False}


def grip(label, value, delay=0.0):
    """Gripper step: wait delay s, then publish value to the gripper topic (0 = closed, 1 = open)"""
    return {'label': label, 'kind': 'grip', 'value': value, 'delay': delay, 'search': False}


def ik_only(label, pose):
    """IK-only step: solves the pose with the current seed and keeps the solution as the seed of the next step, no motion"""
    return {'label': label, 'kind': 'ik', 'value': pose, 'delay': 0.0, 'search': True}


def via_steps(label, start, end, max_step, max_angle):
    """IK-only steps on the way from start to end (position on a straight line, orientation turned evenly), at most max_step (m)
    and max_angle (deg) apart. They carry the IK seed along the way: a single IK call over a long distance or a big rotation can
    land on a different arm configuration (base turned by 180 deg, wrist flipped, elbow folded over), the nearest solution of a
    short step does not"""
    a, b = start.position, end.position
    qa, qb = start.orientation, end.orientation
    ra, rb = Rotation.from_quat([qa.x, qa.y, qa.z, qa.w]), Rotation.from_quat([qb.x, qb.y, qb.z, qb.w])
    turn = ra.inv() * rb                                           # the rotation still to do, seen from the start orientation
    n = max(1, math.ceil(max(math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) / max_step, math.degrees(turn.magnitude()) / max_angle)))
    steps = []
    for i in range(1, n):
        q = (ra * Rotation.from_rotvec(turn.as_rotvec() * i / n)).as_quat()   # [x, y, z, w]
        pose = Pose(position=Point(x=a.x + (b.x - a.x) * i / n, y=a.y + (b.y - a.y) * i / n, z=a.z + (b.z - a.z) * i / n),
                    orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))
        steps.append(ik_only(f'{label} {i}/{n}', pose))
    return steps


def make_sequence(target, cfg):
    """All steps for one /target_pose (TCP)"""
    delays, gripper, basket = cfg['delays'], cfg['gripper'], cfg['basket']
    place_tcp = cfg['poses']['place']
    release_tcp = place_tcp[:2] + [basket['floor_z'] + basket['release_height']] + place_tcp[3:]  # same x y rpy, down in the basket
    flange   = tcp_to_flange(target, cfg)
    approach = offset_pose(flange, -cfg['approach']['distance'], cfg['robot']['tool_axis'])  # above the target (against the TCP z axis)
    home     = tcp_to_flange(xyzrpy_to_pose(*cfg['poses']['home']), cfg)
    place    = tcp_to_flange(xyzrpy_to_pose(*place_tcp), cfg)             # above the basket
    release  = tcp_to_flange(xyzrpy_to_pose(*release_tcp), cfg)           # in the basket, just above its floor
    g = (lambda *a, **k: [grip(*a, **k)]) if gripper['enabled'] else (lambda *a, **k: [])    # a robot without a gripper skips these steps
    return [
        *g('close gripper (start)', gripper['close']),                                # a sequence cancelled earlier may have left it open
        *via_steps('approach', home, approach, cfg['line']['ik_step'], cfg['line']['ik_angle_step']),  # IK only, no motion: the seed follows home -> approach
        move('approach',      approach),
        line('insert',         flange,   approach, delay=delays['descend']),          # straight down after the descend delay
        *g('open gripper',    gripper['open'], delay=delays['open']),                 # opens after the open delay
        line('rise',           approach, flange,   delay=delays['rise']),             # straight back up after the rise delay
        move('home',           home,     search=True),
        move('place',          place,    search=True),
        line('lower',          release,  place),                                      # straight down into the basket
        *g('close gripper',   gripper['close']),                                      # lets go of the part
        line('lift',           place,    release,  delay=delays['gripper_wait']),     # straight up out of the basket after the gripper wait
        move('return home',    home,     search=True),
    ]


class SequenceNode(Node):
    def __init__(self, cfg):
        super().__init__('moveit2_action')
        self.cfg         = cfg
        self.target_sub  = self.create_subscription(Pose, cfg['topics']['target_pose'], self.cb_target, 10)
        self.move_client = ActionClient(self, MoveGroup, cfg['topics']['move_action'])
        self.ik_client   = self.create_client(GetPositionIK, cfg['ik']['service'])
        self.line_client = self.create_client(GetCartesianPath, cfg['line']['service'])
        self.exec_client = ActionClient(self, ExecuteTrajectory, cfg['line']['action'])
        self.fk_client   = self.create_client(GetPositionFK, '/compute_fk')   # the home pose = the TCP of the ready pose
        self.gripper_pub = self.create_publisher(Float64, cfg['gripper']['topic'], 10) if cfg['gripper']['enabled'] else None
        self.ready_seed  = [math.radians(d) for d in cfg['ik']['seed']]
        self.seed        = list(self.ready_seed)  # IK seed of the next move (rad)
        self.moving      = False
        self.steps       = []  # remaining steps (in order, first one next)
        self.name        = ''  # name of what is running now (for the log when it finishes)
        self.init_tries  = 0
        self.goal_q      = []  # joint goal of the move that is running (kept to send it again if it fails)
        self.move_tries  = 0   # how often that move was sent again
        self.retry       = None  # sends the running move again (joint goal: the same goal, line: computed again)
        self.move_id     = 0   # number of the move that is running (late answers of an older move are ignored)
        self.move_handle = None  # goal handle of that move
        self.move_timer  = None  # time limit of that move (motion.timeout)
        self.init_timer  = self.create_timer(cfg['startup']['poll'], self.cb_init)  # checks until everything is up
        # Real joint state, to tell when the arm has stopped / the gripper has finished
        self.joint_pos   = {}       # latest position of every joint in /joint_states
        self.arm_history = deque()  # (receive time, arm joint positions) of the last few windows
        self.gripper_history = deque()  # (receive time, gripper joint position) of the last few windows
        self.gripper_command_time = 0.0  # when the last gripper command was published
        self.gripper_note = ''      # how the last gripper wait ended (reached the position / stopped short of it)
        self.state_time  = 0.0      # receive time of the latest /joint_states
        self.create_subscription(JointState, cfg['settle']['topic'], self.cb_joint_states, 10)

    def servers_ready(self):
        return (self.move_client.server_is_ready() and self.ik_client.service_is_ready()
                and self.line_client.service_is_ready() and self.exec_client.server_is_ready()
                and (not isinstance(self.cfg['poses']['home'], str) or self.fk_client.service_is_ready()))

    def cb_joint_states(self, msg):
        now = time.monotonic()
        self.joint_pos  = dict(zip(msg.name, msg.position))
        self.state_time = now
        settle     = self.cfg['settle']
        arm_joints = self.cfg['robot']['arm_joints']
        if all(j in self.joint_pos for j in arm_joints):
            self.arm_history.append((now, [self.joint_pos[j] for j in arm_joints]))
        while self.arm_history and now - self.arm_history[0][0] > 3 * settle['arm_window']:
            self.arm_history.popleft()
        if settle['gripper_joint'] and settle['gripper_joint'] in self.joint_pos:
            self.gripper_history.append((now, self.joint_pos[settle['gripper_joint']]))
        while self.gripper_history and now - self.gripper_history[0][0] > 3 * settle['gripper_window']:
            self.gripper_history.popleft()

    def arm_stopped(self):
        """True when no arm joint moved more than arm_tolerance during the last arm_window seconds"""
        settle, now = self.cfg['settle'], time.monotonic()
        recent = [(t, p) for t, p in self.arm_history if now - t <= settle['arm_window']]
        if len(recent) < 2 or now - self.state_time > 0.2:    # not enough data, or /joint_states is not coming in
            return False
        if now - recent[0][0] < 0.8 * settle['arm_window']:   # the data does not cover the window yet
            return False
        return max(max(c) - min(c) for c in zip(*[p for _, p in recent])) < settle['arm_tolerance']

    def gripper_done(self, value):
        """True when the gripper joint is at the position that the command value (0 = closed, 1 = open) asks for, or
        when it has stopped short of it (blocked, e.g. the jaws pressing against the part). Sets gripper_note"""
        settle, gripper, now = self.cfg['settle'], self.cfg['gripper'], time.monotonic()
        target = gripper['position_closed'] + value * (gripper['position_open'] - gripper['position_closed'])
        pos    = self.joint_pos.get(settle['gripper_joint'])
        if pos is None or now - self.state_time > 0.2:             # no /joint_states coming in
            return False
        self.gripper_note = f"position {pos:.4f} m (command {target:.4f} m)"
        if abs(pos - target) < settle['gripper_tolerance']:
            return True
        if now - self.gripper_command_time < settle['gripper_grace'] + settle['gripper_window']:
            return False                                           # the joint may not have started moving yet
        recent = [p for t, p in self.gripper_history if now - t <= settle['gripper_window']]
        if len(recent) >= 2 and max(recent) - min(recent) < settle['gripper_tolerance']:
            self.gripper_note += ' - stopped short of the target (contact)'
            return True
        return False

    def wait_until(self, condition, then, what):
        """Call then() as soon as condition() is true (checked every 50 ms). After settle.timeout seconds it goes on anyway"""
        start = time.monotonic()

        def check():
            waited = time.monotonic() - start
            if condition():
                timer.cancel()
                self.destroy_timer(timer)
                self.get_logger().info(f'{what} (after {waited:.2f} s)')
                then()
            elif waited > self.cfg['settle']['timeout']:
                timer.cancel()
                self.destroy_timer(timer)
                self.get_logger().warn(f'{what}: not confirmed after {waited:.1f} s, going on anyway')
                then()
        timer = self.create_timer(0.05, check)

    def cb_init(self):
        """Once everything is up (move_group, /compute_ik, fresh /joint_states): move to the home pose to initialize"""
        if self.moving:                                  # a target came in first: skip the initialization
            self.init_timer.cancel()
            return
        if not self.servers_ready() or time.monotonic() - self.state_time > 0.5:
            return
        self.init_timer.cancel()
        self.init_tries += 1
        if isinstance(self.cfg['poses']['home'], str):   # poses.home: ready -> the TCP of the ready pose first
            self.resolve_home()
        else:
            self.begin_init()

    def resolve_home(self):
        """poses.home = ready: the TCP pose of the ready pose (ik.seed), from the forward kinematics of MoveIt"""
        req = GetPositionFK.Request()
        req.header.frame_id = self.cfg['robot']['base_link']
        req.fk_link_names = [self.cfg['robot']['tip_link']]
        req.robot_state.joint_state.name = list(self.cfg['robot']['arm_joints'])
        req.robot_state.joint_state.position = list(self.ready_seed)
        self.fk_client.call_async(req).add_done_callback(self.cb_home)

    def cb_home(self, future):
        res = future.result()
        if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS or not res.pose_stamped:
            retries = self.cfg['startup']['retries']
            self.get_logger().error('forward kinematics of the ready pose failed -> no home pose')
            if self.init_tries < retries:
                self.destroy_timer(self.init_timer)
                self.init_timer = self.create_timer(self.cfg['startup']['poll'], self.cb_init)
            return
        robot = self.cfg['robot']
        tcp = offset_pose(res.pose_stamped[0].pose, robot['tcp_length'], robot['tool_axis'])   # the TCP is tcp_length along the tool axis from the end frame
        q, p = tcp.orientation, tcp.position
        rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('XYZ', degrees=True)
        self.cfg['poses']['home'] = [p.x, p.y, p.z, *[float(a) for a in rpy]]
        self.get_logger().info('home pose = TCP of the ready pose: ' + ' '.join(f'{v:.3f}' for v in self.cfg['poses']['home']))
        self.begin_init()

    def begin_init(self):
        home = tcp_to_flange(xyzrpy_to_pose(*self.cfg['poses']['home']), self.cfg)
        steps = [grip('close gripper (init)', self.cfg['gripper']['close'])] if self.cfg['gripper']['enabled'] else []
        self.start('initialization', steps + [move('home (init)', home, search=True)])

    def cb_target(self, msg):
        if self.moving:
            self.get_logger().warn('busy -> new target ignored')
            return
        if not self.servers_ready():
            self.get_logger().error(f"{self.cfg['topics']['move_action']} or {self.cfg['ik']['service']} not available -> check that move_group is running")
            return
        if isinstance(self.cfg['poses']['home'], str):
            self.get_logger().warn('the home pose is not known yet (initialization has not started) -> target ignored')
            return
        self.start('sequence', make_sequence(msg, self.cfg))

    def start(self, name, steps):
        self.name   = name
        self.steps  = steps
        self.seed   = list(self.ready_seed)  # the first IK of every sequence starts from the ready pose
        self.moving = True
        self.run_next()

    def run_next(self):
        if not self.steps:
            self.get_logger().info(f'{self.name} done')
            self.moving = False
            return
        step = self.steps.pop(0)
        self.after(step['delay'], lambda: self.execute(step))

    def after(self, delay, fn):
        """Run fn once after delay seconds (immediately if 0)"""
        if delay <= 0.0:
            fn()
            return

        def once():
            timer.cancel()
            self.destroy_timer(timer)
            fn()
        timer = self.create_timer(delay, once)

    def cancel_move_timer(self):
        if self.move_timer is not None:
            self.move_timer.cancel()
            self.destroy_timer(self.move_timer)
            self.move_timer = None

    def cb_move_timeout(self, move_id):
        """The move has not finished within motion.timeout (e.g. move_group died): cancel it and stop the sequence"""
        self.cancel_move_timer()
        if move_id != self.move_id:
            return
        self.move_id += 1                                         # a late answer of this move is ignored
        if self.move_handle is not None:
            self.move_handle.cancel_goal_async()
        self.fail(f"move not finished after {self.cfg['motion']['timeout']} s -> cancelled, remaining steps cancelled")

    def fail(self, message):
        self.cancel_move_timer()
        self.get_logger().error(message)
        self.steps  = []
        self.moving = False
        if self.name == 'initialization':                         # right after launch the first move can fail: try it again
            retries = self.cfg['startup']['retries']
            if self.init_tries < retries:
                self.get_logger().warn(f'trying the initialization again ({self.init_tries}/{retries})')
                self.destroy_timer(self.init_timer)
                self.init_timer = self.create_timer(self.cfg['startup']['poll'], self.cb_init)
            else:
                self.get_logger().error('initialization failed: could not reach the home pose (a target still starts the sequence)')

    def execute(self, step):
        if step['kind'] == 'grip':
            self.get_logger().info(f"[{step['label']}]")
            self.gripper_command_time = time.monotonic()
            self.gripper_pub.publish(Float64(data=step['value']))
            self.wait_until(lambda: self.gripper_done(step['value']), self.gripper_finished, 'gripper done')
            return
        if step['kind'] == 'line':
            self.move_tries = 0
            self.plan_line(step)
            return
        # candidate poses for the IK: the exact pose first, then (waypoints only) poses moved toward the base
        candidates = [(step['value'], 0.0)]
        if step['search']:
            radius = self.cfg['tolerance']['waypoint_position']
            candidates += [(shift_toward_base(step['value'], radius * f), radius * f) for f in (0.25, 0.5, 1.0)]
        self.try_ik(step, candidates, 0)

    def plan_line(self, step):
        """Ask MoveIt for the straight line from where the arm is now to the pose of the step"""
        future = self.line_client.call_async(build_line_request(step['value'], self.cfg))
        future.add_done_callback(lambda f: self.cb_line(f, step))

    def cb_line(self, future, step):
        res, line = future.result(), self.cfg['line']
        traj = res.solution.joint_trajectory
        if res.error_code.val != MoveItErrorCodes.SUCCESS or res.fraction < line['min_fraction'] or not traj.points:
            self.get_logger().warn(f"[{step['label']}] straight line only {res.fraction * 100:.0f}% possible "
                                   f"({ERROR_NAMES.get(res.error_code.val, res.error_code.val)}) -> joint-space move instead")
            self.steps[0:0] = (via_steps(step['label'], step['start'], step['value'], line['ik_step'], line['ik_angle_step'])
                               + [move(step['label'], step['value'], search=True)])
            self.run_next()
            return
        joints = self.cfg['robot']['arm_joints']
        end    = dict(zip(traj.joint_names, traj.points[-1].positions))
        self.seed = [end[j] for j in joints]                   # the next IK starts from where the line ends
        t = traj.points[-1].time_from_start
        self.get_logger().info(f"[{step['label']}] straight line: {len(traj.points)} points, {t.sec + t.nanosec * 1e-9:.1f} s, "
                               f"ends at " + ' '.join(f'{j}={math.degrees(end[j]):.1f}' for j in joints))
        self.retry = lambda: self.plan_line(step)               # a retry starts the line again from where the arm is
        self.send_goal(self.exec_client, ExecuteTrajectory.Goal(trajectory=res.solution))

    def gripper_finished(self):
        self.get_logger().info(f'gripper {self.gripper_note}')
        self.run_next()

    def try_ik(self, step, candidates, index):
        """Ask /compute_ik for candidates[index] (seeded with self.seed); on failure try the next candidate"""
        if index >= len(candidates):
            self.fail(f"[{step['label']}] no IK solution -> remaining steps cancelled")
            return
        ik      = self.cfg['ik']
        timeout = ik['timeout'] if index == 0 else ik['retry_timeout']
        future  = self.ik_client.call_async(build_ik_request(candidates[index][0], self.seed, self.cfg, timeout))
        future.add_done_callback(lambda f: self.cb_ik(f, step, candidates, index))

    def cb_ik(self, future, step, candidates, index):
        res = future.result()
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.try_ik(step, candidates, index + 1)
            return
        joints   = self.cfg['robot']['arm_joints']
        solution = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        if any(j not in solution for j in joints):
            self.fail(f"[{step['label']}] IK result is missing joints: {sorted(solution)} -> remaining steps cancelled")
            return
        q       = [solution[j] for j in joints]
        self.seed = q                                          # the next move starts its IK from this solution
        shifted = candidates[index][1]
        note    = f' (pose moved {shifted * 1000:.1f} mm toward the base)' if shifted else ''
        log     = self.get_logger().debug if step['kind'] == 'ik' else self.get_logger().info   # the via steps stay quiet
        log(f"[{step['label']}] IK: " + ' '.join(f'{j}={math.degrees(v):.1f}' for j, v in zip(joints, q)) + note)
        if step['kind'] == 'ik':                               # only the seed moved along, the arm does not
            self.run_next()
            return
        self.goal_q     = q
        self.move_tries = 0
        self.send_move()

    def send_move(self):
        """Joint goal of the running move (self.goal_q) to MoveGroup"""
        self.retry = self.send_move
        self.send_goal(self.move_client, build_joint_goal(self.goal_q, self.cfg))

    def send_goal(self, client, goal):
        """Send an action goal (MoveGroup or ExecuteTrajectory); motion.timeout limits how long it may take"""
        self.move_id += 1
        move_id = self.move_id
        self.move_handle = None
        client.send_goal_async(goal).add_done_callback(lambda future: self.cb_goal_response(future, move_id))
        self.move_timer = self.create_timer(self.cfg['motion']['timeout'], lambda: self.cb_move_timeout(move_id))

    def cb_goal_response(self, future, move_id):
        if move_id != self.move_id:
            return
        handle = future.result()
        if not handle.accepted:
            self.fail(f'move_group rejected the goal -> {self.name} cancelled')
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
                self.fail(f'failed: {ERROR_NAMES.get(code, code)} -> remaining steps cancelled')
                return
            self.move_tries += 1
            self.get_logger().warn(f'{ERROR_NAMES.get(code, code)} -> trying the move again once the arm has stopped ({self.move_tries}/{retries})')
            self.wait_until(self.arm_stopped, self.retry, 'arm stopped')       # the arm may still be moving after a failure
            return
        self.get_logger().info('arrived (MoveIt reports success)')
        self.wait_until(self.arm_stopped, self.run_next, 'arm stopped')


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot', help='robot name (bringup/<robot>/ and isaacsim/configs/robots/<robot>.yaml); default: `robot:` of select_robot.yaml')
    cli, ros_args = parser.parse_known_args(args)
    cfg = load_config(robot=cli.robot)
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
