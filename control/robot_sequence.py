"""m20_35_18d sequence control: /target_pose (Pose, TCP target in base_link) -> MoveIt2 /move_action + /gripper/command

All parameters (home / place poses, approach distance, IK seed, delays, tolerances, topics, ...) are in
robot_sequence.yaml next to this file.

At start (once move_group is up) the gripper is closed and the robot moves to the home pose to initialize, then it
waits for /target_pose. On /target_pose it moves in the order below and ends at the home pose again:

    close the gripper (a cancelled sequence may have left it open) -> approach -> insert (after delays.descend) -> open the gripper (after delays.open)
    -> rise back to the approach pose (after delays.rise) -> home pose -> place pose (above the basket)
    -> lower into the basket (down to basket.release_height above its floor) -> close the gripper (lets go of the part)
    -> lift back to the place pose (after delays.gripper_wait) -> home pose

The basket is the one of the Isaac Sim scene: basket.sim_config points at the robot_sim.yaml whose `basket:` section creates it.
Its position and wall thickness give the height of its floor, and poses.place has to be above it (checked at start).

Every move is a joint goal solved by /compute_ik with a seed (ik.seed for the first step of a sequence, the solution of the
previous step after that). A plain pose goal lets MoveIt pick any IK solution, and a pose like the home pose has two very
different ones (natural "L" shape and a folded "Z" shape of the arm); the seeded IK always takes the one nearest to the seed.
That only holds for a seed that is near the target: for the long way down into the basket (and back up) the IK is first
solved at intermediate poses (basket.ik_step apart, no motion), so the solution stays on the same arm configuration.

The delays start when the real state in /joint_states says the previous action is finished (the arm has stopped after
a move, the gripper joint has reached its position or stopped short of it because the part blocks the jaws), not when
MoveIt reports success. The log shows how long that took ("arm stopped", "gripper done") and where the gripper stopped.

The first move (to the home pose) waits until move_group, /compute_ik and /joint_states are up and is tried again if
move_group rejects it right after launch (startup.retries). Any other move that fails (TIMED_OUT, CONTROL_FAILED, ...)
is sent again once the arm has stopped (motion.retries) before the sequence is cancelled.

Target value: send the rpy that compute_grasp_tcp_pose.py converted from the object orientation to a TCP orientation
with ee_6d.py (x y z roll pitch yaw).
It receives the same /target_pose as ik_node_m20.py, so the two must not run together
(sequence:=true of the launch leaves ik_node out and starts this node).
"""
import math
import time
from collections import deque
from pathlib import Path

import rclpy
import yaml
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped, Point
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation

CONFIG_FILE = Path(__file__).with_name('robot_sequence.yaml')  # all parameters live in this file

# Keys that robot_sequence.yaml must contain (checked by load_config)
REQUIRED_KEYS = {
    'robot':     ['planning_group', 'arm_joints', 'base_link', 'tip_link', 'tool_axis', 'tcp_length'],
    'poses':     ['home', 'place'],
    'approach':  ['distance'],
    'basket':    ['sim_config', 'release_height', 'ik_step'],
    'ik':        ['service', 'seed', 'timeout', 'retry_timeout', 'avoid_collisions'],
    'delays':    ['descend', 'open', 'rise', 'gripper_wait'],
    'startup':   ['poll', 'retries'],
    'settle':    ['topic', 'arm_window', 'arm_tolerance', 'gripper_joint', 'gripper_tolerance', 'gripper_grace',
                  'gripper_window', 'timeout'],
    'tolerance': ['joint', 'waypoint_position'],
    'motion':    ['velocity_scaling', 'acceleration_scaling', 'planning_time', 'planning_attempts', 'retries'],
    'gripper':   ['topic', 'open', 'close', 'position_closed', 'position_open'],
    'topics':    ['target_pose', 'move_action'],
}

ERROR_NAMES = {v: k for k, v in vars(MoveItErrorCodes).items() if k.isupper() and isinstance(v, int)}


def load_config(path=CONFIG_FILE):
    """Read the parameter file; raise a clear error if a key is missing, the seed does not match the joints, or the place
    pose is not above the basket. Adds basket.floor_z (top of the basket floor, m, from the basket of the Isaac Sim scene)"""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    missing = [f'{section}.{key}' for section, keys in REQUIRED_KEYS.items() for key in keys
               if key not in (cfg.get(section) or {})]
    if missing:
        raise ValueError(f'{path}: missing keys: {", ".join(missing)}')
    if len(cfg['ik']['seed']) != len(cfg['robot']['arm_joints']):
        raise ValueError(f'{path}: ik.seed needs one value per robot.arm_joints entry ({len(cfg["robot"]["arm_joints"])})')
    sim_file = Path(path).parent / cfg['basket']['sim_config']
    with open(sim_file) as f:
        sim = (yaml.safe_load(f) or {}).get('basket') or {}
    lacking = [key for key in ('pos_m', 'size_m', 'wall_thickness_m') if key not in sim]
    if lacking:
        raise ValueError(f'{sim_file}: the basket: section is missing {", ".join(lacking)}')
    cfg['basket']['floor_z'] = sim['pos_m'][2] + sim['wall_thickness_m']   # top of the basket floor
    inside = min(sim['size_m'][:2]) / 2 - sim['wall_thickness_m']          # radius of the circle between the walls
    x, y = cfg['poses']['place'][:2]
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
    # False = plan + execute. move_group sends the execution to m20_traj_controller (FollowJointTrajectory)
    goal.planning_options.plan_only = False
    return goal


def move(label, pose, delay=0.0, search=False):
    """Move step: wait delay s, then move to the flange pose. search=True: if the exact pose has no IK solution, try
    poses moved toward the base by up to tolerance.waypoint_position (for waypoints that need not be exact)"""
    return {'label': label, 'kind': 'move', 'value': pose, 'delay': delay, 'search': search}


def grip(label, value, delay=0.0):
    """Gripper step: wait delay s, then publish value to the gripper topic (0 = closed, 1 = open)"""
    return {'label': label, 'kind': 'grip', 'value': value, 'delay': delay, 'search': False}


def ik_only(label, pose):
    """IK-only step: solves the pose with the current seed and keeps the solution as the seed of the next step, no motion"""
    return {'label': label, 'kind': 'ik', 'value': pose, 'delay': 0.0, 'search': True}


def via_steps(label, start, end, max_step):
    """IK-only steps on the straight line from start to end (same orientation), at most max_step (m) apart. They carry the IK
    seed along the way: a single IK call over a long distance can land on a different arm configuration (base turned by
    180 deg, wrist flipped), the nearest solution of a short step does not"""
    a, b = start.position, end.position
    n = max(1, math.ceil(math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) / max_step))
    return [ik_only(f'{label} {i}/{n}', Pose(position=Point(x=a.x + (b.x - a.x) * i / n, y=a.y + (b.y - a.y) * i / n,
                                                             z=a.z + (b.z - a.z) * i / n), orientation=start.orientation))
            for i in range(1, n)]


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
    return [
        grip('close gripper (start)', gripper['close']),                                  # a sequence cancelled earlier may have left it open
        move('approach',      approach),
        move('insert',         flange,   delay=delays['descend']),                    # goes down after the descend delay
        grip('open gripper',    gripper['open'], delay=delays['open']),                # opens after the open delay
        move('rise',           approach, delay=delays['rise']),                       # rises to the approach pose after the rise delay
        move('home',           home,     search=True),
        move('place',          place,    search=True),
        *via_steps('lower', place, release, basket['ik_step']),
        move('lower',          release,  search=True),                                # goes down into the basket
        grip('close gripper',   gripper['close']),                                    # lets go of the part
        *via_steps('lift', release, place, basket['ik_step']),
        move('lift',           place,    delay=delays['gripper_wait'], search=True),  # comes up out of the basket after the gripper wait
        move('return home',    home,     search=True),
    ]


class SequenceNode(Node):
    def __init__(self, cfg):
        super().__init__('moveit2_action')
        self.cfg         = cfg
        self.target_sub  = self.create_subscription(Pose, cfg['topics']['target_pose'], self.cb_target, 10)
        self.move_client = ActionClient(self, MoveGroup, cfg['topics']['move_action'])
        self.ik_client   = self.create_client(GetPositionIK, cfg['ik']['service'])
        self.gripper_pub = self.create_publisher(Float64, cfg['gripper']['topic'], 10)
        self.ready_seed  = [math.radians(d) for d in cfg['ik']['seed']]
        self.seed        = list(self.ready_seed)  # IK seed of the next move (rad)
        self.moving      = False
        self.steps       = []  # remaining steps (in order, first one next)
        self.name        = ''  # name of what is running now (for the log when it finishes)
        self.init_tries  = 0
        self.goal_q      = []  # joint goal of the move that is running (kept to send it again if it fails)
        self.move_tries  = 0   # how often that move was sent again
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
        return self.move_client.server_is_ready() and self.ik_client.service_is_ready()

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
        if settle['gripper_joint'] in self.joint_pos:
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
        home = tcp_to_flange(xyzrpy_to_pose(*self.cfg['poses']['home']), self.cfg)
        self.start('initialization', [grip('close gripper (init)', self.cfg['gripper']['close']),
                                  move('home (init)', home, search=True)])

    def cb_target(self, msg):
        if self.moving:
            self.get_logger().warn('busy -> new target ignored')
            return
        if not self.servers_ready():
            self.get_logger().error(f"{self.cfg['topics']['move_action']} or {self.cfg['ik']['service']} not available -> check that move_group is running")
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

    def fail(self, message):
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
        # candidate poses for the IK: the exact pose first, then (waypoints only) poses moved toward the base
        candidates = [(step['value'], 0.0)]
        if step['search']:
            radius = self.cfg['tolerance']['waypoint_position']
            candidates += [(shift_toward_base(step['value'], radius * f), radius * f) for f in (0.25, 0.5, 1.0)]
        self.try_ik(step, candidates, 0)

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
        self.move_client.send_goal_async(build_joint_goal(self.goal_q, self.cfg)).add_done_callback(self.cb_goal_response)

    def cb_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.fail(f'move_group rejected the goal -> {self.name} cancelled')
            return
        handle.get_result_async().add_done_callback(self.cb_result)

    def cb_result(self, future):
        code = future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            retries = self.cfg['motion']['retries']
            if self.move_tries >= retries:
                self.fail(f'failed: {ERROR_NAMES.get(code, code)} -> remaining steps cancelled')
                return
            self.move_tries += 1
            self.get_logger().warn(f'{ERROR_NAMES.get(code, code)} -> sending the same move again once the arm has stopped ({self.move_tries}/{retries})')
            self.wait_until(self.arm_stopped, self.send_move, 'arm stopped')   # the arm may still be moving after a TIMED_OUT
            return
        self.get_logger().info('arrived (MoveIt reports success)')
        self.wait_until(self.arm_stopped, self.run_next, 'arm stopped')


def main(args=None):
    cfg = load_config()
    rclpy.init(args=args)
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
