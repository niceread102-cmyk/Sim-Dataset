"""Cycle of the laser marker's side plate flipper and belt (robot_sim.py machines: laser_marker, joints and carriers over ROS 2).

    python3 isaacsim/machine/laser_marker.py [--now [standard|turn_over]]      (ros2 launch robot.launch.py ... sequence:=<name> starts it with the sequence)

The robot puts a plate down at its place pose and goes back to its home pose: control/robot_sequence.py then publishes the state of the part it has
placed on /sequence_done (std_msgs/String, `standard` | `turn_over`; `failed` = nothing was placed).
  standard   the plate lies at the standard coordinate (x -0.612): the belt moves one pitch, that is all.
  turn_over  the plate lies at the turn_over coordinate (x -0.412, on the attachment): the flipper brings it to the standard coordinate first, then the belt moves
             one pitch. The rotor of the flipper rests at 90 deg (standing up, out of the way of the robot and the plates); the flipper:

    1. flip the rotor 90 -> 180 deg       FLIP_GRAB: the open jaws come round the plate at the turn_over coordinate
    2. close the jaws                     hold the plate by the rim of the flange
    3. flip the rotor 180 -> 0 deg        FLIP_RELEASE: the plate is at the standard coordinate (turned over)
    4. open the jaws                      let go of it
    5. flip the rotor back to 90 deg      the rest pose
    then the belt moves one pitch         the carriers go on (/laser_marker/carriers, 1 pitch)

Every step waits until the machine has done it (/laser_marker/joint_states, /laser_marker/carriers/moving) before the next one starts. A step that does
not finish in time stops the cycle with an error. --now runs the cycle once right away for a part of that state (default turn_over; without waiting for the
robot) and ends. Only one of these nodes may run per ROS domain (two would move the belt twice).
"""
import argparse
import fcntl
import os
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, String

MACHINE = 'laser_marker'
FLIP = 'right_flip_joint'
FLIP_REST, FLIP_GRAB, FLIP_RELEASE = 1.5708, 3.14159, 0.0             # rad: rest (standing up, = joint_pos of the machine in map.yaml), where it grabs (turn_over coordinate), where it lets go (standard coordinate)
FLIP_FOR = {'standard': False, 'turn_over': True}                       # state of the placed part (/sequence_done) -> does the flipper bring it to the standard coordinate first? (every state moves the belt)
JAWS = ['right_gripper_jaw_a_joint', 'right_gripper_jaw_b_joint']
JAW_CLOSED, JAW_OPEN = 0.0, 0.01                                       # m: 0 = the 55.5 mm pocket (holds the plate), 0.01 = open (upper limit)
TOPIC_DONE = '/sequence_done'                                          # std_msgs/String from control/robot_sequence.py
TOPIC_COMMAND = f'/{MACHINE}/joint_command'                            # sensor_msgs/JointState (name, position)
TOPIC_STATE = f'/{MACHINE}/joint_states'
TOPIC_CARRIERS = f'/{MACHINE}/carriers'                                # std_msgs/Int32: n > 0 = n pitches; .../moving (Bool): True while it moves
FLIP_TOL, JAW_TOL = 0.01, 0.0005                                       # rad, m: "arrived"
STALL_TIME, STALL_MOVE = 1.5, 0.0002                                   # s, m or rad: no more move than this for that long = stopped (the jaws on the plate)
TIMEOUT = {'flip': 90.0, 'jaws': 30.0, 'belt': 120.0, 'machine': 60.0}  # s per step (wall clock: the simulation may run slower than real time)


class LaserMarker(Node):
    def __init__(self):
        super().__init__('laser_marker_cycle')
        self.q = {}                                                     # joint -> measured position
        self.moving = None                                              # belt moving (None until the first message)
        self.busy = False
        self.create_subscription(JointState, TOPIC_STATE, self.cb_state, 10)
        self.create_subscription(Bool, f'{TOPIC_CARRIERS}/moving', self.cb_moving,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, TOPIC_DONE, self.cb_done, 10)
        self.joint_pub = self.create_publisher(JointState, TOPIC_COMMAND, 10)
        self.belt_pub = self.create_publisher(Int32, TOPIC_CARRIERS, 10)

    def cb_state(self, msg):
        self.q.update(zip(msg.name, msg.position))

    def cb_moving(self, msg):
        self.moving = bool(msg.data)

    def cb_done(self, msg):
        state = msg.data.strip().lower()
        if state not in FLIP_FOR:
            self.get_logger().info(f'sequence done with a {state or "?"} part: nothing to do for the machine')
            return
        if self.busy:
            self.get_logger().warn('the machine is still running its cycle -> /sequence_done ignored')
            return
        self.get_logger().info(f'a {state} part was placed: ' + ('flipper, then the belt' if FLIP_FOR[state] else 'the belt'))
        threading.Thread(target=self.cycle, args=(FLIP_FOR[state],), daemon=True).start()

    # -- steps --------------------------------------------------------------------------------------------------------------------------------------
    def wait_machine(self):
        """until the simulation answers: joint states are coming in and the machine takes commands"""
        end = time.time() + TIMEOUT['machine']
        while not (self.q and self.joint_pub.get_subscription_count() and self.belt_pub.get_subscription_count()):
            if time.time() > end:
                raise TimeoutError(f'no {TOPIC_STATE} / nobody listening on {TOPIC_COMMAND}: is robot_sim.py running with the {MACHINE}?')
            time.sleep(0.1)

    def command(self, names, positions):
        msg = JointState()
        msg.name, msg.position = list(names), [float(p) for p in positions]
        for _ in range(3):                                              # the same command again: harmless, and one of them is surely received
            self.joint_pub.publish(msg)
            time.sleep(0.1)

    def wait_joints(self, label, names, targets, tol, timeout):
        """until every joint is within tol of its target, or has stopped short of it (blocked by the part: the jaws on the plate)"""
        start = time.time()
        still_since, last = None, None
        while True:
            pos = [self.q.get(n) for n in names]
            if all(p is not None and abs(p - t) < tol for p, t in zip(pos, targets)):
                self.get_logger().info(f'{label}: done ({", ".join(f"{p:.4f}" for p in pos)})')
                return
            if time.time() - start > TIMEOUT['machine'] and any(p is None for p in pos):
                raise TimeoutError(f'{label}: no state of {names}')
            if time.time() - start > timeout:
                raise TimeoutError(f'{label}: not at {targets} after {timeout:.0f} s (now {pos})')
            if None not in pos:
                if last is None or max(abs(p - l) for p, l in zip(pos, last)) > STALL_MOVE:
                    last, still_since = list(pos), time.time()
                elif time.time() - still_since > STALL_TIME and time.time() - start > 1.0:
                    self.get_logger().info(f'{label}: stopped short of the target ({", ".join(f"{p:.4f}" for p in pos)}): blocked, e.g. holding the part')
                    return
            time.sleep(0.05)

    def step_belt(self):
        """one pitch of the belt: wait until it moves and until it has stopped again"""
        self.belt_pub.publish(Int32(data=1))                            # once: every message adds a pitch
        end = time.time() + 10.0
        while self.moving is not True:
            if time.time() > end:
                raise TimeoutError(f'the belt did not start ({TOPIC_CARRIERS}/moving stays {self.moving})')
            time.sleep(0.05)
        end = time.time() + TIMEOUT['belt']
        while self.moving is not False:
            if time.time() > end:
                raise TimeoutError('the belt did not stop')
            time.sleep(0.05)
        self.get_logger().info('belt: one pitch done')

    def cycle(self, flip=True):
        """the flipper (flip = True: the plate is at the turn_over coordinate) and then one pitch of the belt"""
        self.busy = True
        log = self.get_logger()
        try:
            self.wait_machine()
            if flip:
                self.flipper(log)
            log.info('the belt moves one pitch')
            self.step_belt()
            log.info('laser marker cycle done')
        except TimeoutError as e:
            log.error(f'laser marker cycle stopped: {e}')
        finally:
            self.busy = False

    def flipper(self, log):
        """turn_over coordinate -> standard coordinate with the jaws of the flipper; the rotor ends at its rest pose"""
        if any(self.q.get(j, JAW_OPEN) < JAW_OPEN - JAW_TOL for j in JAWS):    # the jaws have to be open when the rotor goes to the plate
            log.info('open the jaws first')
            self.command(JAWS, [JAW_OPEN] * len(JAWS))
            self.wait_joints('open', JAWS, [JAW_OPEN] * len(JAWS), JAW_TOL, TIMEOUT['jaws'])
        log.info(f'flip the rotor to {FLIP_GRAB:.3f} rad (grab)')
        self.command([FLIP], [FLIP_GRAB])
        self.wait_joints('flip to grab', [FLIP], [FLIP_GRAB], FLIP_TOL, TIMEOUT['flip'])
        log.info('close the jaws')
        self.command(JAWS, [JAW_CLOSED] * len(JAWS))
        self.wait_joints('close', JAWS, [JAW_CLOSED] * len(JAWS), JAW_TOL, TIMEOUT['jaws'])
        log.info(f'flip the rotor to {FLIP_RELEASE:.3f} rad (release)')
        self.command([FLIP], [FLIP_RELEASE])
        self.wait_joints('flip to release', [FLIP], [FLIP_RELEASE], FLIP_TOL, TIMEOUT['flip'])
        log.info('open the jaws')
        self.command(JAWS, [JAW_OPEN] * len(JAWS))
        self.wait_joints('open', JAWS, [JAW_OPEN] * len(JAWS), JAW_TOL, TIMEOUT['jaws'])
        log.info(f'flip the rotor back to {FLIP_REST:.3f} rad (rest)')
        self.command([FLIP], [FLIP_REST])
        self.wait_joints('flip to rest', [FLIP], [FLIP_REST], FLIP_TOL, TIMEOUT['flip'])


def single_instance():
    """Two of these nodes would both run the cycle on one /sequence_done (the belt would move two pitches): only one per ROS domain. The lock goes with the process."""
    lock = open(f'/tmp/laser_marker_cycle_{os.environ.get("ROS_DOMAIN_ID", "0")}.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit('laser_marker.py is already running (started by the launch, or by hand before): a second one would run every cycle twice')
    return lock


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--now', nargs='?', const='turn_over', choices=sorted(FLIP_FOR), metavar='STATE',
                    help='run the cycle once right away for a part of this state (default turn_over: flipper + belt; standard: belt only), without waiting for /sequence_done, and end')
    args, ros_args = ap.parse_known_args()
    lock = None if args.now else single_instance()                      # (keeps the file open for the life of the process)
    rclpy.init(args=ros_args)
    node = LaserMarker()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        if args.now:
            worker = threading.Thread(target=node.cycle, args=(FLIP_FOR[args.now],), daemon=True)
            worker.start()
            while worker.is_alive():
                executor.spin_once(timeout_sec=0.05)
        else:
            node.get_logger().info(f'waiting for {TOPIC_DONE} (the robot has put the plate down and is at home again): standard = belt, turn_over = flipper + belt')
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
