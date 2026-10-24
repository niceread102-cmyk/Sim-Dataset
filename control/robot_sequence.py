"""Pick & place sequence control for any robot: /target_pose (Pose, TCP target in the world frame of ik_node.yaml base_frame) -> MoveIt2 /move_action + /gripper/command

    python3 control/robot_sequence.py --sequence side_plate [--robot m20_35_18d_mhzl2_20d] [--map SIDE_PLATE_MAP]
    ros2 launch robot.launch.py map:=SIDE_PLATE_MAP sequence:=side_plate       (without sequence:= only ik_node runs)

A sequence is one yaml of isaacsim/configs/sequence/<name>.yaml: a task on a map (where the part is let go of, the approach distance, ...,
in the world coordinates of the map). `robots:` in it holds what differs for one robot (key = robot name; any key of the yaml, e.g.
trajectory: the way the arm travels between the poses), `robot:` (optional) limits it to one robot, `extends: <sequence>` builds on another
sequence (e.g. a robot's own yaml with its trajectory on top of the task of the map). Everything else (tolerances,
topics, ...) is the same for every task and comes from isaacsim/configs/sequence/robot_sequence.yaml. Everything about the
robot comes from its own files: planning group, arm joints, base frame, end frame and TCP length from
bringup/<robot>/config/ik_node.yaml (made by isaacsim/make_robot_config.py), the start pose (joint_pos), the speed of the motions
(velocity_scaling, acceleration_scaling), the gripper topic and joint from isaacsim/configs/robot/<robot>.yaml and the gripper URDF (a robot
without a gripper skips the gripper steps). Of the map (--map) only its robot is read (scene.robot, when --robot and the sequence name none):
the sequence belongs to the robot alone. The home pose is the TCP of the start pose of the robot
(joint_pos of its yaml; ik_seed_deg of ik_node.yaml if it has none), unless poses.home gives a pose. poses.place (default: the home pose) is
where the part is let go of: one pose, or one pose per state of the part (`standard`, `turn_over`: the word after the pose in ee_6d.py,
published on topics.place_variant; with `target: object` the sequence decides it itself); poses.release (optional) is a pose below it that the gripper goes down to first.
poses.place_height (optional) does that for every place pose without a release pose: the arm first travels to the x y of the place pose at that height
(`home` = the height of the home pose, or a z in m), then goes straight down to the z of the place pose to let go of the part.
When a sequence is done (the robot is at home again) the state of the part it has placed (standard | turn_over) is published on topics.sequence_done
(std_msgs/String): isaacsim/machine/laser_marker.py runs the laser marker from it.
The state of the part comes from the simulation: robot_sim.py publishes the clicked part on topics.part_turned_over (std_msgs/Bool, True = turned over, latched); the word
after the pose in ee_6d.py (topics.place_variant) is only used while nothing has been clicked.
poses.place_offset (optional, instead of place_height) is the same with a height relative to the place pose, like the approach of a pick: the arm first goes to the place
pose raised by that many m in z, then straight down by the same distance (the `lower` line), lets go of the part and comes back up.

At start (once move_group is up) the gripper is closed and the robot moves to the home pose to initialize, then it
waits for /target_pose. On /target_pose it moves in the order below and ends at the home pose again:

    close the gripper (a cancelled sequence may have left it open) -> approach -> insert -> open the gripper
    -> rise back to the approach pose -> home pose -> place pose (unless it is the home pose)
    -> [lower to poses.release] -> close the gripper (lets go of the part)
    -> [lift back to the place pose] -> home pose

The moves along the tool axis (insert / rise at the target, lower / lift to poses.release) are straight lines: MoveIt's
/compute_cartesian_path solves the IK every line.max_step along the line from where the arm is, and /execute_trajectory
runs it. A joint goal would interpolate the joints, and the tool would follow a slight curve. If less than
line.min_fraction of a line can be computed (out of reach, the arm would jump to another configuration), the step falls
back to a joint goal (with a warning).

The way the arm travels between the poses is a property of the robot (reach, joint limits, mounting): `trajectory:` of its sequence
yaml says for every travel move (approach, home, via, place) whether the tool goes along a straight line (`line`, with the joint-space
fallback below) or the joints go straight to the goal (`joint`), and `via:` lists TCP poses the arm passes on its way from the home pose to
the place pose (and back), e.g. up to a safe height first, so that the path does not swing through the work area.
The other moves (approach, home, place) are by default joint goals solved by /compute_ik with a seed (the start pose of the robot for the first step of a
sequence, the end of the previous step after that). A plain pose goal lets MoveIt pick any IK solution, and a pose like
the home pose has two very different ones (natural "L" shape and a folded "Z" shape of the arm); the seeded IK always takes
the one nearest to the seed. That only holds for a seed that is near the target: a joint-space fallback of a long line
first solves the IK at intermediate poses (line.ik_step apart, no motion), so it stays on the same arm configuration.
The approach pose is just as far from the ready pose (about 1 m lower, the tool turned and tilted to the part): one IK call
from there returned a different, folded arm (elbow over the top, base turned the wrong way) on different calls. So the
approach is solved the same way: IK-only steps from the home pose, line.ik_step m and line.ik_angle_step deg apart.

Nothing is timed: a step starts when robot_sim.py has answered the one before. A move ends with the result of the trajectory
action (succeeded = the arm is at the goal and at rest, aborted = it stopped short of it, e.g. blocked), a gripper command with
the answer on <gripper topic>/result (std_msgs/String: `reached` = the jaws are at the commanded position, `blocked` = they stopped
short of it, e.g. pressing against the part: both mean the gripper is done). Both limits count physics frames in robot_sim.py, so a slow
simulation changes nothing.

The first move (to the home pose) waits until move_group, /compute_ik and /joint_states are up and is tried again if
move_group rejects it right after launch (startup.retries). Any other move that fails (TIMED_OUT, CONTROL_FAILED, ...)
is tried again (motion.retries; a straight line is computed again from where the arm is) before the sequence is cancelled.
When the sequence is cancelled (the arm is stuck, e.g. the gripper or the extension against a crate wall: MoveIt knows no crates) the arm goes back to the
home pose by itself (motion.recover_home, default on) and is ready for the next target; /sequence_done then says `failed` instead of the state of the part.
MoveIt itself has no time limit (execution duration monitoring is off in the bringup config): only a move that has not finished
after motion.timeout (move_group died) is cancelled here by a watchdog and stops the sequence.

Target value (ee_6d.py: x y z roll pitch yaw): with `target: tcp` (default) the pose of the TCP, i.e. the rpy that
compute_grasp_tcp_pose.py converted from the object orientation to a TCP orientation. With `target: object` the pose of the object itself
(the numbers robot_sim.py prints when you click a part): the sequence turns it into the TCP pose with compute_grasp_tcp_pose.py and
finds out whether the part is upside down (turn_over) or not (standard), which picks the place pose of poses.place.
It receives the same /target_pose as ik_node.py, so the two must not run together
(sequence:=true of the launch leaves ik_node out and starts this node).
"""
import argparse
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
import yaml
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty, Float64, Int32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial.transform import Rotation

import compute_grasp_tcp_pose as grasp    # object pose -> the TCP that grips it, and whether the object is upside down

SIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_DIR / 'isaacsim'))
import sim_config  # noqa: E402  the maps, robots and machines of isaacsim/configs/

CONFIG_FILE = Path(sim_config.SEQUENCE_DIR) / 'robot_sequence.yaml'   # the task, the same for every robot

# Keys of the merged config (checked by load_config). robot, ik.seed, the gripper and the poses are filled in from the files of the robot
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

ERROR_NAMES = {v: k for k, v in vars(MoveItErrorCodes).items() if k.isupper() and isinstance(v, int)}


def gripper_joint(robot, robot_cfg):
    """(joint, lower, upper) of the gripper of the robot (positions in m), None for a robot without one: the first joint
    that is not a mimic joint of the gripper URDF (`gripper:` of the robot yaml), or the first of `gripper_joints:` (a
    gripper that is part of the robot URDF)."""
    names = list(robot_cfg.get('gripper_joints') or [])
    gripper = str(robot_cfg.get('gripper') or 'none')
    if gripper.lower() not in ('none', 'null'):
        urdf = Path(sim_config.GRIPPER_ASSETS) / gripper / 'urdf' / f'{gripper}.urdf'
    elif names:                                      # assets/robot/<robot>/urdf/<robot>.urdf, or <robot>.urdf in another assets/robot/*/urdf/
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
    """The task parameters (+ the overrides of the robot) with everything about the robot filled in from its own files. Raises
    a clear error if a key is missing or the seed does not match the joints. Of the map (map_name) only its robot is read (scene.robot,
    the default of --robot when the sequence names none), nothing else of the scene."""
    scene = sim_config.scene_of(map_name)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    own = Path(path).parent / f'{sequence or sim_config.pick_robot(robot, scene)}.yaml'      # what differs for this robot (optional when only the robot is named)
    if sequence and not own.is_file():
        raise ValueError(f'sequence {sequence!r} not found: {own} (sequences: {", ".join(sequence_names(own.parent)) or "-"})')
    declared = None
    if own.is_file():
        mine = yaml.safe_load(open(own)) or {}
        base = mine.pop('extends', None)                                   # another sequence this one builds on (its poses, ...): the task of a map
        if base:
            base_file = own.parent / f'{base}.yaml'
            if not base_file.is_file():
                raise ValueError(f'{own.name}: extends {base!r} but {base_file} does not exist (sequences: {", ".join(sequence_names(own.parent))})')
            parent = yaml.safe_load(open(base_file)) or {}
            parent.pop('robot', None)
            cfg = sim_config.deep_merge(cfg, {k: v for k, v in parent.items() if k != 'robots'})
            mine = sim_config.deep_merge({'robots': parent.get('robots') or {}}, mine)
        declared = mine.pop('robot', None)                                 # the robot this sequence is limited to (optional)
        per_robot = mine.pop('robots', None) or {}                         # what differs for one robot: {robot name: {any key of this yaml}}
        cfg = sim_config.deep_merge(cfg, mine)
    else:
        per_robot = {}
    cfg.pop('robot', None)
    if robot and declared and robot != declared:
        raise ValueError(f'{own.name} is the sequence of the robot {declared}, not {robot}')
    robot = sim_config.pick_robot(robot or declared, scene)
    if per_robot.get(robot):
        cfg = sim_config.deep_merge(cfg, per_robot[robot])
    ik_file = Path(sim_config.BRINGUP_DIR) / robot / 'config' / 'ik_node.yaml'
    if not ik_file.is_file():
        raise ValueError(f'{ik_file} not found: python3 isaacsim/make_robot_config.py robot:={robot}')
    ik = yaml.safe_load(open(ik_file))
    robot_cfg = sim_config.load_robot(robot)
    motion = sim_config.robot_motion(robot_cfg)                            # the speed of the arm is the robot's (configs/robot/<robot>.yaml)
    cfg.setdefault('motion', {}).update(velocity_scaling=float(motion['velocity_scaling']), acceleration_scaling=float(motion['acceleration_scaling']))

    tcp = [float(x) for x in ik.get('tcp_offset', [0.0, 0.0, 0.0])]       # TCP in the end frame: length and direction
    length = math.sqrt(sum(x * x for x in tcp))
    cfg['robot'] = {'name': robot, 'planning_group': ik['group_name'], 'arm_joints': list(ik['arm_joints']),
                    'base_link': ik['base_frame'], 'tip_link': ik['ee_link'],
                    'tool_axis': [x / length for x in tcp] if length > 1e-9 else [0.0, 0.0, 1.0], 'tcp_length': length}
    cfg.setdefault('ik', {})
    if 'seed' not in cfg['ik']:                                           # the start pose of the robot = its ready pose: home pose, first IK seed
        jp = robot_cfg.get('joint_pos') or {}                              # (rad, a joint name or a pattern of it, like robot_sim.py; unlisted = 0)
        cfg['ik']['seed'] = ([math.degrees(next((float(v) for k, v in jp.items() if re.fullmatch(str(k), n)), 0.0)) for n in cfg['robot']['arm_joints']]
                             if jp else list(ik['ik_seed_deg']))
    if len(cfg['ik']['seed']) != len(cfg['robot']['arm_joints']):
        raise ValueError(f'ik.seed needs one value per arm joint ({len(cfg["robot"]["arm_joints"])}): {cfg["robot"]["arm_joints"]}')

    g = cfg.setdefault('gripper', {})
    g['topic'] = str(robot_cfg.get('gripper_topic') or g.get('topic') or '/gripper/command')
    grip = gripper_joint(robot, robot_cfg)
    g['enabled'] = grip is not None
    g['result_topic'] = g['topic'] + '/result'                             # robot_sim.py answers every gripper command here (reached | blocked)

    poses = cfg.setdefault('poses', {})
    poses.setdefault('home', 'ready')                                      # the TCP of the start pose of the robot (asked from MoveIt at start)
    poses.setdefault('place', 'home')                                      # where the part is let go of: the home pose, or [x, y, z, roll, pitch, yaw]
    poses.setdefault('release', None)                                      # optional: a pose below place the gripper goes down to before it lets go
    poses.setdefault('place_height', None)                                 # optional: travel to the x y of place at this z (home | metres), then straight down to place
    poses.setdefault('place_offset', None)                                 # optional: travel to place raised by this many metres in z, then straight down to place
    missing = [f'{section}.{key}' for section, keys in REQUIRED_KEYS.items() for key in keys
               if key not in (cfg.get(section) or {})]
    if missing:
        raise ValueError(f'{path}: missing keys: {", ".join(missing)}')
    if isinstance(poses['place'], dict):                                   # one place pose per state of the part: normalized names (turn over = turn_over)
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
    t = cfg.get('transfer')                                                # optional: after the part lies at the standard coordinate, pick it up again and put it on the conveyor
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


def build_line_request(pose, cfg, start_q=None):
    """GetCartesianPath request: straight line of the tip_link from the pose move_group knows now to pose
    (start_q: the arm joints (rad) the line starts from instead, for a path that is planned in pieces before the arm moves)"""
    robot, line, motion = cfg['robot'], cfg['line'], cfg['motion']
    req = GetCartesianPath.Request()
    req.header.frame_id                 = robot['base_link']
    req.start_state.is_diff             = True   # empty + is_diff = start where the arm is now
    if start_q is not None:
        req.start_state.joint_state     = JointState(name=list(robot['arm_joints']), position=list(start_q))   # is_diff: only these joints are replaced
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


def move(label, pose, search=False):
    """Move step: wait delay s, then move to the flange pose. search=True: if the exact pose has no IK solution, try
    poses moved toward the base by up to tolerance.waypoint_position (for waypoints that need not be exact)"""
    return {'label': label, 'kind': 'move', 'value': pose, 'search': search}


def line(label, pose, start):
    """Straight-line step: wait delay s, then move the flange along a straight line to pose (start: the pose the previous
    step ends at, only used by the joint-space fallback)"""
    return {'label': label, 'kind': 'line', 'value': pose, 'start': start, 'search': False}


def joints(label, q):
    """Joint step: the joints go straight to q (rad, arm_joints order), no IK: the home pose of a robot is its start pose, so the arm always
    comes back in the same configuration"""
    return {'label': label, 'kind': 'joints', 'value': list(q), 'search': False}


def grip(label, value, settle=0.0):
    """Gripper step: publish value to the gripper topic (0 = closed, 1 = open), wait for the answer of robot_sim.py, then wait settle s more"""
    return {'label': label, 'kind': 'grip', 'value': value, 'search': False, 'settle': settle}


def ik_only(label, pose):
    """IK-only step: solves the pose with the current seed and keeps the solution as the seed of the next step, no motion"""
    return {'label': label, 'kind': 'ik', 'value': pose, 'search': True}


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


MOTION_KINDS = ('line', 'move', 'joints', 'ik')
CORNER_LABELS = ('insert', 'lower')        # the straight moves down into the bore / onto the seat: the pose above them is a corner the arm has to reach exactly


def fuse_steps(steps):
    """fuse_moves: the motion steps between two gripper steps (or an align step) become ONE `path` step: every piece is planned from the end of the
    one before, the pieces are joined into a single trajectory and executed once, so the arm goes through the poses in between (home, above the
    place pose, ...) without stopping. The IK-only steps of such a group only carry the IK seed along"""
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
        if st['kind'] in MOTION_KINDS:
            group.append(st)
        else:
            flush()
            out.append(st)
    flush()
    return out


def travel(cfg, name):
    """how the arm goes to a pose of its path (trajectory.<name> of the sequence): 'line' = a straight line of the tool, else 'joint'"""
    return 'line' if str((cfg.get('trajectory') or {}).get(name, 'joint')).lower() == 'line' else 'joint'


def go(cfg, label, pose, start, mode):
    """travel step to pose from where the previous step ended (start): home = the start pose of the robot (its own joint angles, no IK),
    else a straight line or a joint goal (trajectory.<mode>)"""
    if mode == 'home' and cfg.get('home_joints'):
        return joints(label, cfg['home_joints'])
    return line(label, pose, start) if travel(cfg, mode) == 'line' else move(label, pose, search=True)


def approach_step(cfg, approach, home):
    """the move to the approach pose above the part: a straight line from home (trajectory.approach: line) or an IK goal without the waypoint search"""
    return go(cfg, 'approach', approach, home, 'approach') if travel(cfg, 'approach') == 'line' else move('approach', approach)


def grips(cfg, *a, **k):
    """a gripper step as a list: empty for a robot without a gripper (it skips these steps)"""
    return [grip(*a, **k)] if cfg['gripper']['enabled'] else []


def ik_candidates(cfg, step):
    """the poses the IK is asked for, in order: the exact pose, then (waypoints only: step['search']) poses moved toward the base"""
    candidates = [(step['value'], 0.0)]
    if step['search']:
        radius = cfg['tolerance']['waypoint_position']
        candidates += [(shift_toward_base(step['value'], radius * f), radius * f) for f in (0.25, 0.5, 1.0)]
    return candidates


def place_pose(cfg, variant=None):
    """poses.place for the state of the part (variant): the pose itself, or the entry of the mapping (standard when there is no such state)"""
    place = cfg['poses']['place']
    if not isinstance(place, dict):
        return place
    return place.get(variant) or place.get('standard') or next(iter(place.values()))


def stays_over_standard(cfg, variant):
    """transfer.stay_over_standard: after the part is put on the standard coordinate the arm only lifts (poses.place_offset) and stays there; the transfer goes straight down again.
    transfer.stay_over_turn_over: after a part put on the turn_over coordinate too: the arm lifts and goes on to above the standard coordinate, where the flipper brings the part"""
    place, t = cfg['poses']['place'], cfg.get('transfer') or {}
    return bool(t.get('stay_over_standard')) and isinstance(place, dict) and (variant != 'turn_over' or bool(t.get('stay_over_turn_over')))


def standard_above(cfg):
    """flange pose above the standard coordinate where the arm stays after the lift (the same raised pose as make_sequence puts the part down from)"""
    poses = cfg['poses']
    chosen = place_pose(cfg, 'standard')
    if poses.get('place_height') is not None:
        z = poses['home'][2] if poses['place_height'] == 'home' else poses['place_height']
    else:
        z = chosen[2] + float(poses.get('place_offset') or 0.0)
    return tcp_to_flange(xyzrpy_to_pose(chosen[0], chosen[1], z, *chosen[3:]), cfg)


def carry_height(cfg):
    """poses.carry_z: the TCP height the arm travels at between a crate and the place poses / home (home = the z of the home pose); None = none"""
    cz = cfg['poses'].get('carry_z')
    if cz is None:
        return None
    return float(cfg['poses']['home'][2]) if str(cz).strip().lower() == 'home' else float(cz)


def raised(pose, dz):
    """a flange pose moved up by dz in world z (the tool points down: the TCP goes up with it)"""
    p = pose.position
    return Pose(position=Point(x=p.x, y=p.y, z=p.z + dz), orientation=pose.orientation)


def home_high(cfg):
    """(home flange pose, the same pose at carry_height or None when home is already that high): the way to / from a home pose in a lower layer goes
    straight down / up from the carry height, never across the laser marker at a low height"""
    poses = cfg['poses']
    home = tcp_to_flange(xyzrpy_to_pose(*poses['home']), cfg)
    cz = carry_height(cfg)
    return home, (raised(home, cz - float(poses['home'][2])) if cz is not None and cz > float(poses['home'][2]) + 1e-3 else None)


def high_way_home(cfg, here, here_z, home, home_hi, label):
    """from `here` (flange pose, its TCP at here_z) to a low home: straight up to the carry height, across at that height, straight down - three short
    lines instead of one long diagonal (that one was not fully possible, its joint-space fallback swept past the laser marker table)"""
    up = raised(here, carry_height(cfg) - here_z)
    return [*([line('up to the carry height', up, here)] if carry_height(cfg) > here_z + 1e-3 else []),
            line('above home', home_hi, up), line(label, home, home_hi)]


def make_sequence(target, cfg, variant=None):
    """All steps for one /target_pose (TCP); variant = the state of the part (key of poses.place)"""
    gripper, poses = cfg['gripper'], cfg['poses']
    release_settle = float(gripper.get('release_settle', 0.0))                    # s the jaws get to be fully back before the arm lifts away from the part
    traj = cfg.get('trajectory') or {}
    chosen   = place_pose(cfg, variant)
    place_tcp = poses['home'] if chosen == 'home' else chosen
    release_tcp = poses.get('release')                                     # below place, optional
    if chosen != 'home' and poses.get('place_height') is not None:         # travel at place_height to above the place pose, then straight down to it
        z = poses['home'][2] if poses['place_height'] == 'home' else poses['place_height']
        place_tcp, release_tcp = [*chosen[:2], z, *chosen[3:]], chosen
    elif chosen != 'home' and poses.get('place_offset') is not None:       # the place pose raised by place_offset, then straight down to it (like approach + insert of a pick)
        place_tcp, release_tcp = [chosen[0], chosen[1], chosen[2] + poses['place_offset'], *chosen[3:]], chosen
    flange   = tcp_to_flange(target, cfg)
    approach = offset_pose(flange, -cfg['approach']['distance'], cfg['robot']['tool_axis'])  # above the target (against the TCP z axis)
    carry    = None                                                        # poses.carry_z: with the part straight up to this height first, then straight on to above the place pose
    cz = carry_height(cfg)                                                 #   (not via home: the home of a lower layer is low, the way from there would cut past the laser marker)
    if cz is not None and cz > target.position.z + cfg['approach']['distance'] + 1e-3:
        carry = tcp_to_flange(Pose(position=Point(x=target.position.x, y=target.position.y, z=cz), orientation=target.orientation), cfg)
    home, home_hi = home_high(cfg)
    via      = [tcp_to_flange(xyzrpy_to_pose(*v), cfg) for v in (traj.get('via') or [])]  # between home and place
    place    = tcp_to_flange(xyzrpy_to_pose(*place_tcp), cfg)
    release  = tcp_to_flange(xyzrpy_to_pose(*release_tcp), cfg) if release_tcp else None
    at_home  = chosen == 'home'                                            # the part is let go of at the home pose: no extra moves
    stay     = stays_over_standard(cfg, variant) and not at_home           # transfer.stay_over_standard: the sequence ends above the standard coordinate (the transfer goes straight down again)
    to_std   = stay and variant == 'turn_over'                             # transfer.stay_over_turn_over: from above the turn_over seat on to above the standard one (the flipper brings the part there)

    path = [] if at_home else [(f'via {i + 1}', v) for i, v in enumerate(via)] + [('place', place)]   # home -> via ... -> place
    out, here = [], (carry or home)
    for label, pose in path:
        out.append(go(cfg, label, pose, here, 'place' if label == 'place' else 'via'))
        here = pose
    back, there = [], here                                                 # the way back: the via poses reversed, then home
    for label, pose in reversed(path[:-1]):
        back.append(go(cfg, 'return ' + label, pose, there, 'via'))
        there = pose
    return [
        *grips(cfg, 'close gripper (start)', gripper['close']),                       # a sequence cancelled earlier may have left it open
        *via_steps('approach', home, approach, cfg['line']['ik_step'], cfg['line']['ik_angle_step']),  # IK only, no motion: the seed follows home -> approach
        approach_step(cfg, approach, home),
        line('insert',         flange,   approach),                                   # straight down
        *grips(cfg, 'open gripper', gripper['open']),
        line('rise',           approach, flange),                                     # straight back up
        *([line('carry',       carry,    approach)] if carry else [go(cfg, 'home', home, approach, 'home')]),   # straight on up to poses.carry_z (else to home)
        *out,
        *([line('lower',       release,  here),                                       # straight down to poses.release
           *grips(cfg, 'close gripper', gripper['close'], settle=release_settle),    # lets go of the part
           line('lift',        here,     release)]                                    # straight back up
          if release else [*grips(cfg, 'close gripper', gripper['close'], settle=release_settle)]),   # lets go of the part at the place pose
        *([line('wait above standard', standard_above(cfg), here)] if to_std else []),  # straight over to above the standard coordinate: the transfer goes down from there
        *([] if stay else back),
        *([] if (at_home or stay) else (high_way_home(cfg, there, float(place_tcp[2]), home, home_hi, 'return home') if home_hi is not None
                                        else [go(cfg, 'return home', home, there, 'home')])),
    ]


def make_transfer(cfg, place_tcp=None, above=None):
    """The part lies at the standard coordinate: from home to it, pick it up again (approach, insert, open = the bore is expanded, rise), 'measure yaw' (the yaw of the plate in
    the hand is read from the simulation and the wrist is turned by it on the way: the steps after it are made again by transfer_steps), then put it on the carrier of the belt"""
    head, tail = transfer_steps(cfg, place_tcp, above)
    return head + [{'label': 'measure yaw', 'kind': 'align', 'value': None, 'search': False}] + tail


def transfer_steps(cfg, place_tcp=None, above=None):
    """(steps up to the rise at the standard coordinate, steps from there to the carrier and home): the second part for a place pose [x, y, z, r, p, y] (default transfer.place).
    above = flange pose of the arm when it already hangs above the standard coordinate (transfer.stay_over_standard): straight down from there, no home / approach"""
    gripper, poses, t = cfg['gripper'], cfg['poses'], cfg['transfer']
    release_settle = float(gripper.get('release_settle', 0.0))
    step, angle = cfg['line']['ik_step'], cfg['line']['ik_angle_step']
    pick     = tcp_to_flange(xyzrpy_to_pose(*t['pick']), cfg)
    approach = offset_pose(pick, -cfg['approach']['distance'], cfg['robot']['tool_axis'])
    x, y, z, *rest = place_tcp or t['place']
    up       = tcp_to_flange(xyzrpy_to_pose(x, y, z + float(poses.get('place_offset') or cfg['approach']['distance']), *rest), cfg)   # raised
    down     = tcp_to_flange(xyzrpy_to_pose(x, y, z, *rest), cfg)
    home, home_hi = home_high(cfg)

    insert_a = dict(line('insert', pick, approach), follow_part=True)        # follow_part: goes down at the xy where the plate really lies (transfer.follow_part_xy)
    insert_b = dict(line('insert', pick, above), follow_part=True) if above is not None else None
    start    = home_hi or home                                             # from a low home (a lower layer): straight up first
    head = [
        *grips(cfg, 'close gripper (start)', gripper['close']),
        *([line('above home', home_hi, home)] if home_hi is not None else []),
        *via_steps('approach', start, approach, step, angle),
        approach_step(cfg, approach, start),
        insert_a,
        *grips(cfg, 'open gripper', gripper['open']),
        dict(line('rise', approach, pick), follow_part=True),                  # straight up where it went down
    ] if above is None else [
        insert_b,                                                              # the arm hangs above the standard coordinate: straight down to the part
        *grips(cfg, 'open gripper', gripper['open']),
        dict(line('rise', approach, pick), follow_part=True),
    ]
    tail = [
        *via_steps('place', approach, up, step, angle),                       # IK seeds along the way to the belt (the wrist turns on the way)
        go(cfg, 'place', up, approach, 'place'),
        line('lower', down, up),
        *grips(cfg, 'close gripper', gripper['close'], settle=release_settle),   # lets go of the part on the carrier
        line('lift', up, down),
        *(high_way_home(cfg, up, z + float(poses.get('place_offset') or cfg['approach']['distance']), home, home_hi, 'home') if home_hi is not None
          else [go(cfg, 'home', home, up, 'home')]),
    ]
    return head, tail


LABEL_WAITS = {'close gripper (start)': 'close_gripper_start', 'approach': 'approach', 'insert': 'insert', 'open gripper': 'open_gripper', 'rise': 'rise',
               'carry': 'carry', 'home': 'home', 'place': 'place', 'lower': 'lower', 'close gripper': 'release', 'lift': 'lift', 'return home': 'return_home'}   # step label -> key of `waits:` of the sequence yaml


def apply_waits(steps, waits):
    """waits: {key: s}: the seconds to wait after the step (when it is done) before the next one starts; steps without a key keep what they have"""
    for st in steps:
        key = LABEL_WAITS.get(st['label'])
        if key in waits:
            st['settle'] = float(waits[key])
    return steps


class SequenceNode(Node):
    def __init__(self, cfg):
        super().__init__('moveit2_action')
        self.cfg         = cfg
        self.target_sub  = self.create_subscription(Pose, cfg['topics']['target_pose'], self.cb_target, 10)
        self.variant     = None                                          # state of the part for the next target (ee_6d.py: the word after the pose)
        self.seq_variant = ''                                            # state of the part of the sequence that is running (published when it is done)
        self.done_pub    = self.create_publisher(String, cfg['topics'].get('sequence_done', '/sequence_done'), 10)
        self.variant_sub = self.create_subscription(String, cfg['topics'].get('place_variant', '/place_variant'), self.cb_variant, 10)
        self.part_pose   = None                                          # (Pose, time) of the part the robot goes for: robot_sim.py publishes it (the plate that lies at the standard coordinate is the one)
        self.follow      = None                                          # transfer: (dx, dy) of the plate at the standard coordinate against transfer.pick, found at its first followed step
        self.create_subscription(Pose, cfg['topics'].get('target_part_pose', '/target_part_pose'), lambda m: setattr(self, 'part_pose', (m, time.time())), 10)
        self.over_standard = False                                       # the arm hangs above the standard coordinate (transfer.stay_over_standard): the transfer starts from there, not from home
        self.transfer_pending = False                                    # the machine said the part lies at the standard coordinate while the arm was busy: transfer it afterwards
        self.ready_sub   = self.create_subscription(Empty, cfg['topics'].get('machine_ready', '/laser_marker/ready'), self.cb_ready, 10) if cfg.get('transfer') else None
        self.transfer_pub = self.create_publisher(Empty, cfg['topics'].get('transfer_done', '/transfer_done'), 10)
        crate = cfg.get('crate') or {}
        self.crates = sim_config.expand_crates(crate, cfg.get('crates'))  # crates: of the sequence (side_plate_test) for every layer (crate.layers): the scan pose (home) of every crate, emptied in turn by vision_pick.py
        self.crate_idx, self.crate_pending = None, None                 # the crate whose home is the home pose now; the one asked for while the arm was busy
        self.crate_ready_pub = None
        if self.crates and crate.get('next_topic'):
            self.create_subscription(Int32, crate['next_topic'], self.cb_crate, 10)
            self.crate_ready_pub = self.create_publisher(Int32, crate.get('ready_topic', '/crate_ready'), 10)
        self.turned_over = None                                          # state of the part clicked in the simulation (robot_sim.py): True = turned over, False = standard
        self.part_sub = self.create_subscription(Bool, cfg['topics'].get('part_turned_over', '/part_turned_over'), self.cb_part,
                                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.move_client = ActionClient(self, MoveGroup, cfg['topics']['move_action'])
        self.ik_client   = self.create_client(GetPositionIK, cfg['ik']['service'])
        self.line_client = self.create_client(GetCartesianPath, cfg['line']['service'])
        self.exec_client = ActionClient(self, ExecuteTrajectory, cfg['line']['action'])
        self.fk_client   = self.create_client(GetPositionFK, '/compute_fk')   # the home pose = the TCP of the ready pose
        self.gripper_pub = self.create_publisher(Float64, cfg['gripper']['topic'], 10) if cfg['gripper']['enabled'] else None
        self.ready_seed  = [math.radians(d) for d in cfg['ik']['seed']]
        self.seed        = list(self.ready_seed)  # IK seed of the next move (rad)
        self.moving      = False
        self.cur_settle  = 0.0
        self.steps       = []  # remaining steps (in order, first one next)
        self.name        = ''  # name of what is running now (for the log when it finishes)
        self.init_tries  = 0
        self.goal_q      = []  # joint goal of the move that is running (kept to send it again if it fails)
        self.move_tries  = 0   # how often that move was sent again
        self.retry       = None  # sends the running move again (joint goal: the same goal, line: computed again)
        self.move_id     = 0   # number of the move that is running (late answers of an older move are ignored)
        self.move_handle = None  # goal handle of that move
        self.move_timer  = None  # watchdog of that move (motion.timeout): only for a move_group that died, a normal move ends with its result
        self.init_timer  = self.create_timer(cfg['startup']['poll'], self.cb_init)  # checks until everything is up
        self.joint_pos   = {}       # latest position of every joint in /joint_states (empty until robot_sim.py is up)
        self.path        = None     # the path that is being planned in pieces (fuse_moves)
        self.gripper_pending = None  # called with the answer of robot_sim.py to the gripper command that is running
        if self.gripper_pub:
            self.create_subscription(String, cfg['gripper']['result_topic'], self.cb_gripper_result, 10)
        self.create_subscription(JointState, cfg['topics']['joint_states'], self.cb_joint_states, 10)

    def servers_ready(self):
        return (self.move_client.server_is_ready() and self.ik_client.service_is_ready()
                and self.line_client.service_is_ready() and self.exec_client.server_is_ready()
                and (not isinstance(self.cfg['poses']['home'], str) or self.fk_client.service_is_ready()))

    def cb_joint_states(self, msg):
        self.joint_pos = dict(zip(msg.name, msg.position))

    def cb_gripper_result(self, msg):
        """robot_sim.py: the gripper reached its position ('reached') or stopped short of it, e.g. the jaws pressing against the part ('blocked')"""
        done, self.gripper_pending = self.gripper_pending, None
        if done is None:                                       # not waiting for an answer: an old one
            return
        self.get_logger().info(f'gripper {msg.data}')
        done()

    def cb_init(self):
        """Once everything is up (move_group, /compute_ik, fresh /joint_states): move to the home pose to initialize"""
        if self.moving:                                  # a target came in first: skip the initialization
            self.init_timer.cancel()
            return
        if not self.servers_ready() or not self.joint_pos:
            return
        self.init_timer.cancel()
        self.init_tries += 1
        if isinstance(self.cfg['poses']['home'], str):   # poses.home: ready -> the TCP of the start pose first
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
        self.get_logger().info('home pose = TCP of the start pose of the robot: ' + ' '.join(f'{v:.3f}' for v in self.cfg['poses']['home']))
        self.cfg['home_joints'] = list(self.ready_seed)    # the home moves go to these joint angles themselves (no IK): the same configuration every time
        self.begin_init()

    def begin_init(self):
        home = tcp_to_flange(xyzrpy_to_pose(*self.cfg['poses']['home']), self.cfg)
        steps = [grip('close gripper (init)', self.cfg['gripper']['close'])] if self.cfg['gripper']['enabled'] else []
        self.start('initialization', steps + [joints('home (init)', self.cfg['home_joints']) if self.cfg.get('home_joints') else move('home (init)', home, search=True)])

    def cb_variant(self, msg):
        self.variant = str(msg.data).strip().lower().replace(' ', '_') or None
        self.get_logger().info(f'state of the part received: {self.variant}')

    def cb_ready(self, _):
        """the machine (laser_marker.py) says the part lies at the standard coordinate (placed there, or turned over to it): pick it up again and put it on the belt"""
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
        """the 'measure yaw' step: the plate is in the hand (the bore is expanded, it was lifted straight up): its yaw in the world is read from the simulation, the wrist is turned by it,
        so that the plate comes down on the carrier at yaw 0 (the place pose): the steps from here to home are made again with that place pose"""
        t = self.cfg['transfer']
        place = list(t['place'])
        p = self.part_pose[0] if self.part_pose and time.time() - self.part_pose[1] < 2.0 else None
        if t.get('align_yaw', True) and p is not None and math.hypot(p.position.x - t['pick'][0], p.position.y - t['pick'][1]) < 0.08:
            roll, pitch, yaw = Rotation.from_quat([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]).as_euler('XYZ', degrees=True)
            yaw = (yaw + 90.0) % 180.0 - 90.0                                    # the windows of the plate repeat every 180 deg: at most a quarter turn
            place[5] = float(place[5]) + float(yaw)                              # the plate is fixed in the hand. The tool points down (roll 180): its rpy yaw turns it about the world z the other way round,
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
        """crate.next_topic (vision_pick.py: the crate before is empty): the home pose becomes the scan pose of crate msg.data, the arm goes there"""
        i = int(msg.data)
        if not 0 <= i < len(self.crates):
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
        home = [float(v) for v in self.crates[i]['home']]
        self.cfg['poses']['home'] = home                                   # every sequence / transfer / recovery returns here from now on
        self.cfg.pop('home_joints', None)
        self.crate_idx = i
        new, new_hi = home_high(self.cfg)
        self.get_logger().info(f"crate {i} (centre {self.crates[i].get('centre')}): home pose = its scan pose {' '.join(f'{v:g}' for v in home)}")
        if carry_height(self.cfg) is None:
            self.start('crate', [move(f'crate {i} scan pose', new, search=True)])
            return
        up, over = old_hi or old, new_hi or new                            # up to the carry height, over, straight down: not across the crates of the layer
        self.start('crate', ([line('up from the crate', up, old)] if old_hi is not None else []) + [line(f'over crate {i}', over, up)]
                   + ([line(f'crate {i} scan pose', new, over)] if new_hi is not None else []))

    def cb_part(self, msg):
        self.turned_over = bool(msg.data)
        self.get_logger().info(f"part from the simulation: {'turn_over' if self.turned_over else 'standard'}")

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
        place, variant, self.variant = self.cfg['poses']['place'], self.variant, None    # a state applies to one target only
        if str(self.cfg.get('target', 'tcp')).lower() == 'object':                  # the target is the part: its pose -> the TCP that grips it + its state
            q = msg.orientation
            roll, pitch, yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('XYZ', degrees=True)
            tcp_r, tcp_p, tcp_y, flipped = grasp.tcp_rpy(roll, pitch, yaw)
            msg = xyzrpy_to_pose(msg.position.x, msg.position.y, msg.position.z, tcp_r, tcp_p, tcp_y)
            variant = variant or ('turn_over' if flipped else 'standard')
            self.get_logger().info(f"object rpy {roll:.1f} {pitch:.1f} {yaw:.1f} -> {'turned over' if flipped else 'standard'}, TCP rpy {tcp_r:.1f} {tcp_p:.1f} {tcp_y:.1f}")
        if isinstance(place, dict):
            if self.turned_over is not None and str(self.cfg.get('target', 'tcp')).lower() != 'object':    # the state of the clicked part in the simulation decides (the word of ee_6d.py is only the fallback)
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
        waits = dict(self.cfg.get('waits') or {})                            # waits: of the sequence yaml (all the waiting times of the sequence in one place)
        if name == 'transfer':
            waits.update(waits.get('transfer') or {})
        if name in ('sequence', 'transfer'):
            steps = apply_waits(steps, {k: v for k, v in waits.items() if k in LABEL_WAITS.values()})
            if self.cfg.get('fuse_moves'):                                   # fuse_moves: of the sequence yaml: several moves in one trajectory (no stop in between)
                steps = fuse_steps(steps)
        self.name   = name
        self.steps  = steps
        self.seed   = list(self.ready_seed)  # the first IK of every sequence starts from the ready pose
        self.moving = True
        self.run_next()

    def run_next(self):
        if not self.steps:
            self.get_logger().info(f'{self.name} done')
            self.moving = False
            self.over_standard = self.name == 'sequence' and stays_over_standard(self.cfg, self.seq_variant)   # the sequence of a part put on the standard coordinate ends above it
            if self.name == 'sequence':                                      # at home again: the machines of the scene (isaacsim/machine/) go on from here
                self.done_pub.publish(String(data=self.seq_variant))
            elif self.name == 'transfer':                                    # the part lies on the carrier of the belt and the arm is at home: the belt moves one pitch (laser_marker.py)
                self.transfer_pub.publish(Empty())
            elif self.name == 'recovery':                                    # back at home after a failed sequence: ready for the next target
                self.done_pub.publish(String(data='failed'))
            elif self.name == 'crate':                                       # at the scan pose of the next crate: vision_pick.py takes its picture
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
        """fn once, after seconds (at once for 0): the node goes on receiving while it waits"""
        if seconds <= 0:
            fn()
            return
        timer = None

        def fire():
            timer.cancel()
            self.destroy_timer(timer)
            fn()
        timer = self.create_timer(seconds, fire)

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
        self.over_standard = False
        self.steps  = []
        self.moving = False
        self.gripper_pending = None
        if self.name == 'initialization':                         # right after launch the first move can fail: try it again
            retries = self.cfg['startup']['retries']
            if self.init_tries < retries:
                self.get_logger().warn(f'trying the initialization again ({self.init_tries}/{retries})')
                self.destroy_timer(self.init_timer)
                self.init_timer = self.create_timer(self.cfg['startup']['poll'], self.cb_init)
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
        """A move of the sequence failed for good (the arm is stuck, e.g. the gripper / extension against a crate): back to the home pose, so that the
        next target starts from there. The way back is a joint move from wherever the arm is; the gripper stays as it is (a sequence closes it first)."""
        self.get_logger().warn('the sequence failed -> back to the home pose')
        if self.cfg.get('home_joints'):
            step = joints('home (recovery)', self.cfg['home_joints'])
        else:
            step = move('home (recovery)', tcp_to_flange(xyzrpy_to_pose(*self.cfg['poses']['home']), self.cfg), search=True)
        self.start('recovery', [step])

    def execute(self, step):
        self.cur_settle = step.get('settle', 0.0)                              # waited after a move when it has arrived (a gripper step waits it itself)
        if step['kind'] == 'align':
            self.align_yaw()
            return
        if step['kind'] == 'grip':
            self.get_logger().info(f"[{step['label']}]")
            self.gripper_pending = lambda: self.later(step.get('settle', 0.0), self.run_next)    # the next step starts with the answer of robot_sim.py (+ settle s)
            self.gripper_pub.publish(Float64(data=step['value']))
            return
        if step['kind'] == 'path':
            self.run_path(step)
            return
        if step['kind'] == 'line':
            self.move_tries = 0
            self.plan_line(self.follow_part(step))
            return
        if step['kind'] == 'joints':
            self.seed, self.goal_q, self.move_tries = list(step['value']), list(step['value']), 0
            self.get_logger().info(f"[{step['label']}] joints: " + ' '.join(f'{j}={math.degrees(v):.1f}' for j, v in zip(self.cfg['robot']['arm_joints'], step['value'])))
            self.send_move()
            return
        self.try_ik(step, ik_candidates(self.cfg, step), 0)

    # ---- one trajectory for several steps (fuse_moves) ----
    def run_path(self, step):
        """Plan every piece of step['steps'] from the end of the one before and execute them as one trajectory (see fuse_steps)"""
        joints = self.cfg['robot']['arm_joints']
        q0 = [self.joint_pos.get(j) for j in joints]
        if any(v is None for v in q0):
            self.path_abort(step, 'no /joint_states yet', list(self.seed))
            return
        self.move_tries = 0
        self.retry = lambda: self.run_path(step)               # a retry plans the whole path again from where the arm is
        self.path = {'step': step, 'queue': list(step['steps']), 'q': q0, 'seed0': list(self.seed), 'moves': 0, 't': 0.0,
                     'points': [JointTrajectoryPoint(positions=q0, time_from_start=Duration(sec=0, nanosec=0))]}
        self.path_next()

    def path_next(self):
        p = self.path
        if not p['queue']:
            self.path_send()
            return
        sub = p['queue'].pop(0)
        if sub['label'] in CORNER_LABELS and len(p['points']) > 1:   # a corner in front of a straight move down: the last point is repeated, robot_sim.py makes the arm itself reach it (no cutting the corner)
            self.path_point(p['q'], p['t'] + 0.001)
        if sub['kind'] == 'joints':
            self.seed = list(sub['value'])
            self.path_add_joint_move(list(sub['value']))
            self.path_next()
        elif sub['kind'] == 'line':
            sub = self.follow_part(sub, p['queue'])
            future = self.line_client.call_async(build_line_request(sub['value'], self.cfg, p['q']))
            future.add_done_callback(lambda f: self.cb_path_line(f, sub))
        else:                                                   # move / ik: the same candidate poses as execute()
            self.path_ik(sub, ik_candidates(self.cfg, sub), 0)

    def path_ik(self, sub, candidates, index):
        if index >= len(candidates):
            self.path_abort(self.path['step'], f"[{sub['label']}] no IK solution", self.path['seed0'])
            return
        ik = self.cfg['ik']
        future = self.ik_client.call_async(build_ik_request(candidates[index][0], self.seed, self.cfg, ik['timeout'] if index == 0 else ik['retry_timeout']))
        future.add_done_callback(lambda f: self.cb_path_ik(f, sub, candidates, index))

    def cb_path_ik(self, future, sub, candidates, index):
        res = future.result()
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.path_ik(sub, candidates, index + 1)
            return
        joints = self.cfg['robot']['arm_joints']
        solution = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        if any(j not in solution for j in joints):
            self.path_abort(self.path['step'], f"[{sub['label']}] IK result is missing joints", self.path['seed0'])
            return
        q = [solution[j] for j in joints]
        self.seed = q                                           # the next piece starts its IK from this solution
        if sub['kind'] != 'ik':
            self.path_add_joint_move(q)
        self.path_next()

    def cb_path_line(self, future, sub):
        res, line, p = future.result(), self.cfg['line'], self.path
        traj = res.solution.joint_trajectory
        if res.error_code.val != MoveItErrorCodes.SUCCESS or res.fraction < line['min_fraction'] or len(traj.points) < 2:
            self.path_abort(p['step'], f"[{sub['label']}] straight line only {res.fraction * 100:.0f}% possible", p['seed0'])
            return
        joints = self.cfg['robot']['arm_joints']
        t0 = p['t']
        for pt in traj.points[1:]:                              # the first point is where the piece starts = the end of the one before
            pos = dict(zip(traj.joint_names, pt.positions))
            t = t0 + pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            self.path_point([pos[j] for j in joints], t)
        self.seed = list(p['q'])                                # the next IK starts from where the line ends
        p['moves'] += 1
        self.path_next()

    def path_point(self, q, t):
        p = self.path
        p['t'] = max(t, p['t'] + 0.001)
        p['points'].append(JointTrajectoryPoint(positions=list(q), time_from_start=Duration(sec=int(p['t']), nanosec=int((p['t'] % 1.0) * 1e9))))
        p['q'] = list(q)

    def path_add_joint_move(self, q):
        """a joint-space move from the end of the path so far to q, in steps of at most fuse_joint_step rad (the sim follows the waypoints in joint space)"""
        p = self.path
        step = float(self.cfg.get('fuse_joint_step', 0.05))
        d = max(abs(a - b) for a, b in zip(q, p['q']))
        n = max(1, math.ceil(d / step))
        q0 = list(p['q'])
        for i in range(1, n + 1):
            self.path_point([a + (b - a) * i / n for a, b in zip(q0, q)], p['t'] + max(0.02, d / n / 0.6))
        p['moves'] += 1

    def path_abort(self, step, why, seed):
        """a piece of the path could not be planned: the pieces go one by one as they would without fuse_moves (nothing has moved yet)"""
        self.get_logger().warn(f"[{step['label']}] {why} -> the pieces are executed one by one")
        self.seed = list(seed)
        self.steps[0:0] = list(step['steps'])
        self.run_next()

    def path_send(self):
        p = self.path
        jt = JointTrajectory(joint_names=list(self.cfg['robot']['arm_joints']), points=p['points'])
        self.get_logger().info(f"[{p['step']['label']}] one path: {p['moves']} moves, {len(p['points'])} points, {p['t']:.1f} s")
        self.send_goal(self.exec_client, ExecuteTrajectory.Goal(trajectory=RobotTrajectory(joint_trajectory=jt)))

    def follow_part(self, step, queue=None):
        """transfer.follow_part_xy: the plate on the seat of the standard coordinate does not lie exactly at transfer.pick (it slides into the seat when it is let go):
        the straight move down goes to where it really is (/target_part_pose, a few mm at most), the offset is logged. The gripper is not turned here: the plate is
        turned to the jig when it is carried to the belt (align_yaw)"""
        t = self.cfg.get('transfer') or {}
        if not step.get('follow_part') or not t.get('follow_part_xy'):
            return step
        if self.follow is None:                                               # the first followed step of this transfer: where the plate lies
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
            self.get_logger().warn(f'{ERROR_NAMES.get(code, code)} -> trying the move again ({self.move_tries}/{retries})')
            self.retry()
            return
        self.get_logger().info('arrived' + (f' (waiting {self.cur_settle:g} s)' if self.cur_settle > 0 else ''))   # robot_sim.py reports success once the arm is there and at rest
        self.later(self.cur_settle, self.run_next)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot', help='robot name (bringup/<robot>/ and isaacsim/configs/robot/<robot>.yaml); default: the robot of the sequence, '
                                        'else scene.robot of the map')
    parser.add_argument('--sequence', help='sequence name = isaacsim/configs/sequence/<name>.yaml (poses, trajectory and what else differs for a robot)')
    parser.add_argument('--map', help='map name = isaacsim/configs/map/<map>.yaml: only its robot is read (scene.robot, the default of --robot)')
    cli, ros_args = parser.parse_known_args(args)
    map_name = cli.map or sim_config.with_current(None, None)[1]        # without --map: the map of the running robot_sim.py
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
