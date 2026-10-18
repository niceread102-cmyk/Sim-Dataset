#!/usr/bin/env python3
"""End-effector target node for any robot arm (any number of joints); everything robot specific is in a yaml.

    python3 control/ik_node.py --config bringup/<robot>/config/ik_node.yaml

ee_6d.py -> /target_pose -> this node: /compute_ik (starting from the current joint angles of /joint_states, so the solution is
  the one nearest to where the arm stands; ik_seed_deg only until /joint_states has arrived; a joint that can turn a full turn
  is then set to the angle nearest to the current one, see wrap_joints_deg) -> joint goal to /move_action (plan+execute)
  -> the FollowJointTrajectory action of the robot (the controller of bringup/<robot>/config/moveit_controllers.yaml).
bringup/<robot>/launch/<robot>.launch.py and bringup/integration/launch/robot.launch.py (robot:=<name>) start it with
move_group (ik_node:=false leaves it out).
It receives the same /target_pose as control/robot_sequence.py, so run only one of the two.

/target_pose: position = TCP, orientation = ee_link orientation (z axis = tool direction, so rpy 180 0 0 = tool pointing down), in
base_frame of the yaml = `world`, the Isaac Sim world: the robot may stand anywhere in it (its pos_m / rpy_deg are in MoveIt's URDF).
The IK is solved for ee_link by subtracting tcp_offset (the TCP in the ee_link frame).

Customizing a robot (the yaml, see bringup/ur5e/config/ik_node.yaml):
  speed          velocity_scaling / acceleration_scaling = fraction of the limits MoveIt uses (bringup/<robot>/config/joint_limits.yaml
                 and the URDF), so the speed of one joint is set there and the speed of the whole motion here
  joint limits   joint_limits_deg = optional narrower range per joint: an IK solution outside it is refused (MoveIt's own
                 position limits come from the URDF / joint_limits.yaml)
  full turns     wrap_joints_deg = the joints whose URDF range is wider than 360 deg, with that range (made by make_robot_config.py).
                 q and q +- 360 are the same pose there and the IK may return any of them (its restarts from a random seed keep the
                 turn), so the arm would spin once round to reach the angle. Each of these joints is set to the equivalent angle
                 nearest to its current one (/joint_states) inside its range (and joint_limits_deg). The other joints, and a robot
                 without the key, are sent exactly as the IK solved them.
"""
import argparse
import math
import time

import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus, GoalStatusArray
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

REQUIRED_KEYS = ('base_frame', 'group_name', 'ee_link', 'arm_joints', 'ik_seed_deg')
DEFAULTS = {
    'tcp_offset':           [0.0, 0.0, 0.0],  # m, TCP in the ee_link frame
    'velocity_scaling':     0.15,             # fraction of the velocity limits
    'acceleration_scaling': 0.05,             # fraction of the acceleration limits
    'planning_time':        5.0,              # s
    'planning_attempts':    10,
    'goal_tolerance':       0.001,            # rad, every joint
    'ik_timeout':           2.0,              # s
    'busy_timeout':         60.0,             # s, see IKNode._cb_target_pose
    'joint_limits_deg':     {},               # joint name -> [min, max]
    'wrap_joints_deg':      {},               # joint name -> [min, max] of the joints that can turn a full turn (see above)
}
TURN = 2 * math.pi

# While executing, if a trajectory from another client (RViz Plan & Execute etc.) overlaps, move_group rejects it
# (CONTROL_FAILED), then its internal state breaks, every later goal fails and it finally dies with a segfault
# (Jazzy, reproduced). So nothing is sent while another execution is in progress.
MOVEIT_ACTIONS = ('/move_action', '/execute_trajectory')
ACTIVE_STATUS  = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING)


def load_config(path):
    cfg = yaml.safe_load(open(path)) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f'{path}: missing {missing}')
    cfg = {**DEFAULTS, **cfg}
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
    R       = Rotation.from_quat([pose.orientation.x, pose.orientation.y,
                                   pose.orientation.z, pose.orientation.w]).as_matrix()
    tcp_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
    ee_pos  = tcp_pos - R @ np.array(cfg['tcp_offset'])

    ps = PoseStamped()
    ps.header.frame_id = cfg['base_frame']
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = ee_pos.tolist()
    ps.pose.orientation = pose.orientation

    # IK seed (rad, arm_joints order): the current joint angles, so the Newton iteration of the IK ends in the solution nearest
    # to where the arm stands. From a far seed it does not converge, restarts from random angles and returns any of the (up to 8)
    # solutions of a 6 axis arm (shoulder behind, elbow flipped, ...): the arm swings round to reach it. Before the first
    # /joint_states, the ready pose of the yaml.
    seed = list(seed) if seed is not None else [math.radians(d) for d in cfg['ik_seed_deg']]

    req = GetPositionIK.Request()
    req.ik_request.group_name                 = cfg['group_name']
    req.ik_request.ik_link_name               = cfg['ee_link']
    req.ik_request.pose_stamped               = ps
    req.ik_request.robot_state.is_diff        = True
    req.ik_request.robot_state.joint_state    = JointState(name=list(cfg['arm_joints']), position=seed)
    req.ik_request.avoid_collisions           = True
    timeout = cfg['ik_timeout']
    req.ik_request.timeout                    = Duration(sec=int(timeout), nanosec=int((timeout % 1.0) * 1e9))
    return req


def outside_limits(cfg, q):
    """Joints of the IK solution q (name -> rad) that are outside joint_limits_deg, as messages"""
    out = []
    for name, (lo, hi) in cfg['joint_limits_deg'].items():
        deg = math.degrees(q[name])
        if not lo <= deg <= hi:
            out.append(f'{name}={deg:.1f} not in [{lo}, {hi}]')
    return out


def nearest_turn(q, ref, lo, hi):
    """q (rad) turned by whole turns to the angle nearest to ref inside [lo, hi] (the bounds may be infinite). q itself, unchanged,
    when it is outside the range: that is not for this function to hide."""
    k = round((ref - q) / TURN)                                       # whole turns that bring q nearest to ref
    k_min = math.ceil((lo - q) / TURN - 1e-9) if lo > -math.inf else -10**9      # fewest / most turns that stay inside
    k_max = math.floor((hi - q) / TURN + 1e-9) if hi < math.inf else 10**9
    if k_min > 0 or k_max < 0:
        return q
    return q + TURN * min(max(k, k_min), k_max)


def wrap_to_current(cfg, q, current):
    """IK solution q (joint name -> rad) with every joint of wrap_joints_deg turned to the angle nearest to its current one (name ->
    rad) inside its range and joint_limits_deg. A joint without a current angle, and every other joint, stays as it is."""
    out = dict(q)
    for name, (lo, hi) in cfg['wrap_joints_deg'].items():
        if name in q and name in current:
            narrow = cfg['joint_limits_deg'].get(name)
            if narrow:
                lo, hi = max(lo, narrow[0]), min(hi, narrow[1])
            out[name] = nearest_turn(q[name], current[name], math.radians(lo), math.radians(hi))
    return out


def build_joint_goal(cfg, joint_state: JointState):
    constraints = Constraints()
    for name, pos in zip(joint_state.name, joint_state.position):
        if name not in cfg['arm_joints']:
            continue
        constraints.joint_constraints.append(JointConstraint(
            joint_name      = name,
            position        = pos,
            tolerance_above = cfg['goal_tolerance'],
            tolerance_below = cfg['goal_tolerance'],
            weight          = 1.0,
        ))

    req                                 = MotionPlanRequest()
    req.group_name                      = cfg['group_name']
    req.goal_constraints                = [constraints]
    req.num_planning_attempts           = cfg['planning_attempts']
    req.allowed_planning_time           = cfg['planning_time']
    req.max_velocity_scaling_factor     = cfg['velocity_scaling']
    req.max_acceleration_scaling_factor = cfg['acceleration_scaling']

    goal_msg                             = MoveGroup.Goal()
    goal_msg.request                     = req
    goal_msg.planning_options.plan_only  = False
    return goal_msg


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
        self._seq     = 0   # to ignore late callbacks of dropped goals
        self._own_goal_ids  = set()
        self._others_active = {}
        self._q_now   = {}   # latest angle (rad) of every arm joint, from /joint_states: the IK seed, the angle to turn to
        self.create_subscription(JointState, '/joint_states', self._cb_joint_states, qos_profile_sensor_data)
        status_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for action in MOVEIT_ACTIONS:
            self.create_subscription(GoalStatusArray, f'{action}/_action/status',
                                     lambda msg, a=action: self._on_status(a, msg), status_qos)
        self.get_logger().info(f"ik_node ready ({cfg['group_name']}, {len(cfg['arm_joints'])} joints, ee_link {cfg['ee_link']})"
                               ' — waiting for /target_pose')

    def _cb_joint_states(self, msg: JointState):
        self._q_now.update({n: p for n, p in zip(msg.name, msg.position) if n in self.cfg['arm_joints']})

    def _on_status(self, action: str, msg: GoalStatusArray):
        self._others_active[action] = any(
            st.status in ACTIVE_STATUS and bytes(st.goal_info.goal_id.uuid) not in self._own_goal_ids
            for st in msg.status_list)

    def _cb_target_pose(self, msg: Pose):
        # If the result of the previous goal does not arrive within busy_timeout, or move_group disappears (dies / restarts),
        # the previous goal is dropped and a new goal is accepted. Without it busy would never clear after move_group dies.
        if self._busy:
            waited = time.monotonic() - self._busy_since
            if self._client.server_is_ready() and waited < self.cfg['busy_timeout']:
                self.get_logger().warn(f'Already moving ({waited:.0f} s), ignoring')
                return
            self.get_logger().warn(f'No result for previous goal after {waited:.0f} s '
                                   '(move_group died/restarted?) — dropping it')
            self._busy = False
            self._seq += 1

        others = [a for a, on in self._others_active.items() if on]
        if others:
            self.get_logger().warn(f'move_group is executing another request on {", ".join(others)} '
                                   '(RViz Plan & Execute?) — ignoring target, send it again after that motion ends')
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

        joints = self.cfg['arm_joints']
        if all(n in self._q_now for n in joints):
            seed = [self._q_now[n] for n in joints]
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

        joints = self.cfg['arm_joints']
        js = res.solution.joint_state
        q  = {n: p for n, p in zip(js.name, js.position) if n in joints}
        self.get_logger().info('IK: ' + ' '.join(f'{n}={math.degrees(q[n]):.1f}' for n in joints))
        if self.cfg['wrap_joints_deg']:
            turned = wrap_to_current(self.cfg, q, self._q_now)
            changed = [n for n in joints if turned[n] != q[n]]
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
            self.get_logger().error(
                f'[FAIL] error_code={code} message="{result.error_code.message}" '
                f'source="{result.error_code.source}"')
        self._finish(ok, seq)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='robot yaml, e.g. bringup/ur5e/config/ik_node.yaml')
    cli, ros_args = parser.parse_known_args(args)
    cfg = load_config(cli.config)

    rclpy.init(args=ros_args)
    node = IKNode(cfg)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
