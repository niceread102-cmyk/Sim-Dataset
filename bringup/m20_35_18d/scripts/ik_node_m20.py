#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from action_msgs.msg import GoalStatus, GoalStatusArray
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation
import numpy as np

# FANUC M-20iD/35 (m20_35_18d) + bringup/m20_35_18d (move_group) + isaacsim/robot_sim.py.
# ee_6d.py -> /target_pose -> this node: /compute_ik (starting from SEED_Q) -> joint goal to /move_action (plan+execute)
#   -> /m20_traj_controller/follow_joint_trajectory
# launch/m20_35_18d.launch.py starts it together with move_group (it can be turned off with ik_node:=false). Standalone:
#   python3 bringup/m20_35_18d/scripts/ik_node_m20.py (needs move_group + robot_sim.py).
#   It receives the same /target_pose as control/robot_sequence.py, so run only one of the two.
# fanuc_flange: the flange front-face frame (z axis = tool direction), so like lite6 link_eef, rpy (180, 0, 0) = tool pointing down.
# /target_pose position = TCP (gripper finger tip), orientation = fanuc_flange orientation. IK is solved for fanuc_flange by subtracting TCP_OFFSET.
EEF_LINK   = 'fanuc_flange'
BASE_FRAME = 'base_link'
GROUP_NAME = 'arm'    # bringup/m20_35_18d/config/m20_35_18d.srdf
# TCP position (m) in the fanuc_flange frame. gripper_tcp of the JEGG-2510N straight tip +300mm
# (isaacsim/gripper_assets/urdf/jegg2510n_straight_300mm.urdf) = adapter 8 mm + motor rear to finger tip 466.516 mm
# = 474.516 mm from the flange face (farthest point of the mesh, the tip centre is on the axis).
# With the JEGG-2008 straight tip use (0.0, 0.0, 0.469316), with the bent tip jegg2008_300mm (-0.00035, 0.0, 0.469316), without a gripper 0.
TCP_OFFSET = (0.0, 0.0, 0.474516)
ARM_JOINTS = ('J1', 'J2', 'J3', 'J4', 'J5', 'J6')

# IK seed: giving a pose goal straight to move_group uses a random-seed IK, which can return elbow-down / flipped-wrist solutions.
# Start from the ready pose (elbow up, tool pointing down, J5 = -90 deg) so the same side of the solutions is always picked.
SEED_Q = (0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0)

# If the result of the previous goal does not arrive within this time, or move_group disappears (dies / restarts),
# the previous goal is dropped and a new goal is accepted. Without it busy would never clear after move_group dies
# and the node would keep saying 'Already moving'.
BUSY_TIMEOUT_S = 60.0

# While executing, if a trajectory from another client (RViz Plan & Execute etc.) overlaps, move_group rejects it
# (CONTROL_FAILED), then its internal state breaks, every later goal fails and it finally dies with a segfault
# (Jazzy, reproduced). So nothing is sent while another execution is in progress.
MOVEIT_ACTIONS = ('/move_action', '/execute_trajectory')
ACTIVE_STATUS  = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING)


def build_ik_request(pose: Pose, timeout_sec=2.0):
    R       = Rotation.from_quat([pose.orientation.x, pose.orientation.y,
                                   pose.orientation.z, pose.orientation.w]).as_matrix()
    tcp_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
    eef_pos = tcp_pos - R @ np.array(TCP_OFFSET)

    ps = PoseStamped()
    ps.header.frame_id = BASE_FRAME
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = eef_pos.tolist()
    ps.pose.orientation = pose.orientation

    req = GetPositionIK.Request()
    req.ik_request.group_name                 = GROUP_NAME
    req.ik_request.ik_link_name               = EEF_LINK
    req.ik_request.pose_stamped               = ps
    req.ik_request.robot_state.is_diff        = True
    req.ik_request.robot_state.joint_state    = JointState(name=list(ARM_JOINTS), position=list(SEED_Q))
    req.ik_request.avoid_collisions           = True
    req.ik_request.timeout                    = Duration(sec=int(timeout_sec))
    return req


def build_joint_goal(joint_state: JointState, vel_scale=0.15, acc_scale=0.05):
    constraints = Constraints()
    for name, pos in zip(joint_state.name, joint_state.position):
        if name not in ARM_JOINTS:
            continue
        constraints.joint_constraints.append(JointConstraint(
            joint_name      = name,
            position        = pos,
            tolerance_above = 0.001,
            tolerance_below = 0.001,
            weight          = 1.0,
        ))

    req                                 = MotionPlanRequest()
    req.group_name                      = GROUP_NAME
    req.goal_constraints                = [constraints]
    req.num_planning_attempts           = 10
    req.allowed_planning_time           = 5.0
    req.max_velocity_scaling_factor     = vel_scale
    req.max_acceleration_scaling_factor = acc_scale

    goal_msg                             = MoveGroup.Goal()
    goal_msg.request                     = req
    goal_msg.planning_options.plan_only  = False
    return goal_msg


class IK_Node_M20(Node):
    def __init__(self):
        super().__init__('IK_Node_M20')
        self.sub      = self.create_subscription(Pose, '/target_pose', self._cb_target_pose, 10)
        self.done_pub = self.create_publisher(Bool, '/ik_done', 10)
        self._client  = ActionClient(self, MoveGroup, '/move_action')
        self._ik      = self.create_client(GetPositionIK, '/compute_ik')
        self._busy    = False
        self._busy_since = 0.0
        self._seq     = 0   # to ignore late callbacks of dropped goals
        self._own_goal_ids  = set()
        self._others_active = {}
        status_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for action in MOVEIT_ACTIONS:
            self.create_subscription(GoalStatusArray, f'{action}/_action/status',
                                     lambda msg, a=action: self._on_status(a, msg), status_qos)
        self.get_logger().info('IK_Node_M20 ready — waiting for /target_pose')

    def _on_status(self, action: str, msg: GoalStatusArray):
        self._others_active[action] = any(
            st.status in ACTIVE_STATUS and bytes(st.goal_info.goal_id.uuid) not in self._own_goal_ids
            for st in msg.status_list)

    def _cb_target_pose(self, msg: Pose):
        if self._busy:
            waited = time.monotonic() - self._busy_since
            if self._client.server_is_ready() and waited < BUSY_TIMEOUT_S:
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

        self._ik.call_async(build_ik_request(pose)).add_done_callback(lambda f: self._ik_cb(f, seq))

    def _ik_cb(self, future, seq: int):
        if seq != self._seq:
            return
        res = future.result()
        if res.error_code.val != 1:
            self.get_logger().error(f'[FAIL] IK error_code={res.error_code.val} (target unreachable or in collision)')
            self._finish(False, seq)
            return

        js = res.solution.joint_state
        q  = {n: p for n, p in zip(js.name, js.position) if n in ARM_JOINTS}
        self.get_logger().info('IK: ' + ' '.join(f'{n}={math.degrees(q[n]):.1f}' for n in ARM_JOINTS))
        self._send_goal(build_joint_goal(js), seq)

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
    rclpy.init(args=args)
    node = IK_Node_M20()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
