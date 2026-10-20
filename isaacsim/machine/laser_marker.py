"""Cycle of the laser marker's side plate flipper and belt (robot_sim.py machines: laser_marker, joints and carriers over ROS 2).

    python3 isaacsim/machine/laser_marker.py [--now [standard|turn_over]]      (ros2 launch robot.launch.py ... sequence:=<name> starts it with the sequence)

The robot puts a plate down at its place pose and goes back to its home pose: control/robot_sequence.py then publishes the state of the part it has
placed on /sequence_done (std_msgs/String, `standard` | `turn_over`; `failed` = nothing was placed).
  standard   the plate lies at the standard coordinate (the station that is to correct its angle, not built yet): /laser_marker/ready at once.
  turn_over  the plate lies at the turn_over coordinate (on the seat of the flipper): the flipper brings it to the standard coordinate first, then /laser_marker/ready.
             The rotor of the flipper rests at 90 deg (standing up, out of the way of the robot and the plates); the flipper:

    1. flip the rotor 90 -> 180 deg       FLIP_GRAB: the open jaws come round the plate at the turn_over coordinate
    2. close the jaws                     hold the plate by the rim of the flange
    3. flip the rotor 180 -> 0 deg        FLIP_RELEASE: the plate is at the standard coordinate (turned over)
    4. open the jaws                      let go of it
    5. flip the rotor back to 90 deg      the rest pose

  /laser_marker/ready: the robot picks the plate up again at the standard coordinate and puts it on the carrier of the belt next to it (slot 3), then it publishes
  /transfer_done (std_msgs/Empty): the belt moves one pitch (/laser_marker/carriers, 1 pitch).
The plates ride the belt to the left, one slot (0.36 m) per belt step, and the node counts them: when, after a belt step, a plate lies in each of the two hands of
the left station (slots 0 and 1), the left rotor does the same as the flipper (90 deg rest -> 0 deg grab -> 180 deg let go -> 90 deg) and publishes /laser_marker/hands_done (conveyor.py moves the chain of the
conveyor, where the plates were put, one step); the belt does not move after that.
The count starts empty when the node starts: restart it together with the simulation.

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
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Int32, String

MACHINE = 'laser_marker'
FLIP = 'right_flip_joint'
FLIP_REST, FLIP_GRAB, FLIP_RELEASE = 1.5708, 3.14159, 0.0               # rad: rest (standing up, = joint_pos of the machine in map.yaml), where it grabs (turn_over coordinate), where it lets go (standard coordinate)
FLIP_FOR = {'standard': False, 'turn_over': True}                       # state of the placed part (/sequence_done) -> does the flipper bring it to the standard coordinate first? (every state moves the belt)
JAWS = ['right_gripper_jaw_a_joint', 'right_gripper_jaw_b_joint']
LEFT_FLIP = 'left_flip_joint'                                           # the two pairs of hands of the left station move together on this rotor
LEFT_JAWS = ['left_gripper_1_jaw_a_joint', 'left_gripper_1_jaw_b_joint', 'left_gripper_2_jaw_a_joint', 'left_gripper_2_jaw_b_joint']
LEFT_REST, LEFT_GRAB, LEFT_RELEASE = 1.5708, 0.0, 3.14159               # rad: rest (= joint_pos of the machine in map.yaml), where they grab (the plates on the belt), where they let go
HAND_STILL = 1.0                                                        # s the two plates have to lie in front of the hands before they take them (the belt has stopped, the plates are at rest)
SLOTS, SLOT_PLACE, HANDS = 4, 3, (0, 1)                                 # carriers of the belt (slot 0 = leftmost, y -0.83 m ... 3 = y 0.25 m: where the robot puts the plate down); the slots under the two left hands
JAW_CLOSED, JAW_OPEN = -0.005, 0.01                                     # m: closed = 5 mm PAST the 55.5 mm pocket (0 = the pocket), so that the jaws on the plate (55.56 mm: they stop at ~0.5-1.5 mm) press with their drive (5000 N/m x ~5.5 mm = ~27 N each); -0.002 (10 N) was not enough, the plate was lost. 0.01 = open (upper limit)
FLIPPER_CLOSED = LEFT_CLOSED = JAW_CLOSED                                   # per gripper: speeds.laser_marker.flipper_close_mm / two_arm_close_mm of the sequence yaml
TOPIC_HAND_PLATES = '/laser_marker/hand_plates'                         # std_msgs/Int32 from robot_sim.py: bit i = a plate lies in front of hand pair i
TOPIC_READY = '/laser_marker/ready'                                     # std_msgs/Empty: the plate lies at the standard coordinate (the robot picks it up again)
TOPIC_TRANSFER_DONE = '/transfer_done'                                  # std_msgs/Empty from control/robot_sequence.py: the plate lies on the carrier of the belt (slot 3)
TOPIC_HANDS_DONE = '/laser_marker/hands_done'                           # std_msgs/Empty: the left hands have let go of their plates on the conveyor (conveyor.py then moves its chain one step)
TOPIC_DONE = '/sequence_done'                                           # std_msgs/String from control/robot_sequence.py
TOPIC_COMMAND = f'/{MACHINE}/joint_command'                             # sensor_msgs/JointState (name, position)
TOPIC_STATE = f'/{MACHINE}/joint_states'
TOPIC_CARRIERS = f'/{MACHINE}/carriers'                                 # std_msgs/Int32: n > 0 = n pitches; .../moving (Bool): True while it moves
FLIP_TOL, JAW_TOL = 0.01, 0.0005                                        # rad, m: "arrived"
STALL_FRAMES, STALL_MOVE = 90, 0.0002                                   # simulation frames (joint_states: one per frame, so a slow simulation does not count as a stop), m or rad:
                                                                        # no more move than this for that many frames = stopped (the jaws on the plate)
TIMEOUT = {'flip': 90.0, 'jaws': 30.0, 'belt': 120.0, 'machine': 60.0}  # s per step (wall clock: the simulation may run slower than real time)


class LaserMarker(Node):
    def __init__(self):
        super().__init__('laser_marker_cycle')
        self.q = {}                                                     # joint -> measured position
        self.frames = 0                                                 # joint_states received = frames of the simulation
        self.moving = None                                              # belt moving (None until the first message)
        self.busy = False
        self.slots = [False] * SLOTS                                    # which carrier of the belt has a plate (counted: a plate is put at slot 4, every belt step moves all one slot to the left)
        self.create_subscription(JointState, TOPIC_STATE, self.cb_state, 10)
        self.create_subscription(Bool, f'{TOPIC_CARRIERS}/moving', self.cb_moving,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, TOPIC_DONE, self.cb_done, 10)
        self.joint_pub = self.create_publisher(JointState, TOPIC_COMMAND, 10)
        self.belt_pub = self.create_publisher(Int32, TOPIC_CARRIERS, 10)
        self.hands_done_pub = self.create_publisher(Empty, TOPIC_HANDS_DONE, 10)
        self.ready_pub = self.create_publisher(Empty, TOPIC_READY, 10)
        self.create_subscription(Empty, TOPIC_TRANSFER_DONE, self.cb_transfer_done, 10)
        self.create_subscription(Int32, TOPIC_HAND_PLATES, self.cb_hand_plates, 10)
        self.hand_mask, self.hand_since, self.hands_wait_clear = 0, 0.0, False    # plates in front of the hands (the two plates are there when the mask is 3), since when, waiting for them to be taken away

    def cb_state(self, msg):
        self.q.update(zip(msg.name, msg.position))
        self.frames += 1

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
        self.get_logger().info(f'a {state} part was placed: ' + ('flipper, then ' if FLIP_FOR[state] else '') + 'the robot takes it to the belt')
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

    def wait_joints(self, label, names, targets, tol, timeout, stall=True):
        """until every joint is within tol of its target, or (stall) has stopped short of it (blocked by the part: the jaws on the plate, a rotor on the belt).
        stall=False waits for the target, however slowly the move starts: a rotor that has to get out of the way before the belt moves"""
        start = time.time()
        still_frame, last = 0, None
        while True:
            pos = [self.q.get(n) for n in names]
            if all(p is not None and abs(p - t) < tol for p, t in zip(pos, targets)):
                self.get_logger().info(f'{label}: done ({", ".join(f"{p:.4f}" for p in pos)})')
                return
            if time.time() - start > TIMEOUT['machine'] and any(p is None for p in pos):
                raise TimeoutError(f'{label}: no state of {names}')
            if time.time() - start > timeout:
                raise TimeoutError(f'{label}: not at {targets} after {timeout:.0f} s (now {pos})')
            if None not in pos and stall:
                if last is None or max(abs(p - l) for p, l in zip(pos, last)) > STALL_MOVE:
                    last, still_frame = list(pos), self.frames
                elif self.frames - still_frame > STALL_FRAMES:
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
        """a plate was put down: the flipper (flip = True: it is at the turn_over coordinate) brings it to the standard coordinate, then the robot is told that
        the plate lies there (/laser_marker/ready): it picks it up again and puts it on the carrier of the belt (cb_transfer_done)"""
        self.busy = True
        log = self.get_logger()
        try:
            self.wait_machine()
            if flip:
                self.flipper(log, FLIP, JAWS, FLIP_REST, FLIP_GRAB, FLIP_RELEASE, FLIPPER_CLOSED)
            log.info('the plate lies at the standard coordinate: the robot takes it to the belt')
            self.ready_pub.publish(Empty())
            log.info('laser marker cycle done')
        except TimeoutError as e:
            log.error(f'laser marker cycle stopped: {e}')
        finally:
            self.busy = False

    def cb_hand_plates(self, msg):
        """a plate in front of each of the two hand pairs (and they lie still for HAND_STILL s: the belt has stopped): the hands take both over to the conveyor, then wait for the next two"""
        mask = int(msg.data)
        if mask != self.hand_mask:
            self.hand_mask, self.hand_since = mask, time.time()
        if mask != 3:
            self.hands_wait_clear = False                                        # they are away: the next two may come
            return
        if self.hands_wait_clear or self.busy or self.moving or time.time() - self.hand_since < HAND_STILL:
            return
        self.hands_wait_clear = True
        self.get_logger().info('a plate in front of each of the two hand pairs: the hands take them over to the conveyor')
        threading.Thread(target=self.hands_cycle, daemon=True).start()

    def hands_cycle(self):
        self.busy = True
        log = self.get_logger()
        try:
            self.wait_machine()
            self.hands(log)
            log.info('hands cycle done: waiting for the next two plates')
        except TimeoutError as e:
            log.error(f'hands cycle stopped: {e}')
        finally:
            self.busy = False

    def cb_transfer_done(self, _):
        """the robot has put the plate on the carrier at slot SLOT_PLACE: the belt moves one pitch; when both left hands have a plate then, the left hands
        do the same as the flipper (grab, turn, let go); the belt does not move again after them"""
        if self.busy:
            self.get_logger().warn('the machine is still running its cycle -> /transfer_done ignored')
            return
        threading.Thread(target=self.after_transfer, daemon=True).start()

    def after_transfer(self):
        self.busy = True
        log = self.get_logger()
        try:
            self.wait_machine()
            self.belt(log)                                                      # the plate on the carrier at slot 3 comes one pitch closer to the hands; the hands start by themselves when a plate lies in front of each pair (cb_hand_plates)
            log.info('belt cycle done')
        except TimeoutError as e:
            log.error(f'belt cycle stopped: {e}')
        finally:
            self.busy = False

    def hands(self, log):
        """the two pairs of left hands do what the flipper does (rest 90 deg -> grab -> let go -> rest); the belt stays where it is"""
        self.flipper(log, LEFT_FLIP, LEFT_JAWS, LEFT_REST, LEFT_GRAB, LEFT_RELEASE, LEFT_CLOSED)
        self.hands_done_pub.publish(Empty())                                    # the plates lie on the conveyor and the hands are out of the way: it moves one step (conveyor.py)
        for i in HANDS:
            self.slots[i] = False                                               # the plates are taken off the belt (the belt does not move after the hands: it moves only when the robot has put a plate on it)

    def belt(self, log):
        log.info('the belt moves one pitch')
        self.step_belt()
        self.slots = self.slots[1:] + [False]                                   # everything on the belt is one slot further to the left (what leaves at slot 0 is not counted any more)
        log.info('belt slots (left to right): ' + ''.join('X' if s else '.' for s in self.slots))

    def flipper(self, log, flip, jaws, rest, grab, release, closed=None):
        """a rotor with its jaws: grab (the jaws close on the plate), turn to the release angle, let go, back to rest. The rotor ends at its rest pose"""
        if any(self.q.get(j, JAW_OPEN) < JAW_OPEN - JAW_TOL for j in jaws):    # the jaws have to be open when the rotor goes to the plate
            log.info('open the jaws first')
            self.command(jaws, [JAW_OPEN] * len(jaws))
            self.wait_joints('open', jaws, [JAW_OPEN] * len(jaws), JAW_TOL, TIMEOUT['jaws'])
        log.info(f'{flip}: to {grab:.3f} rad (grab)')
        self.command([flip], [grab])
        self.wait_joints('flip to grab', [flip], [grab], FLIP_TOL, TIMEOUT['flip'])
        closed = JAW_CLOSED if closed is None else closed
        log.info('close the jaws')
        self.command(jaws, [closed] * len(jaws))
        self.wait_joints('close', jaws, [closed] * len(jaws), JAW_TOL, TIMEOUT['jaws'])
        log.info(f'{flip}: to {release:.3f} rad (release)')
        self.command([flip], [release])
        self.wait_joints('flip to release', [flip], [release], FLIP_TOL, TIMEOUT['flip'])
        held = [self.q.get(j) for j in jaws]
        log.info('jaws before they let go: ' + ', '.join(f'{p:.4f}' for p in held if p is not None)
                 + (f' (about {closed:.4f} = closed on nothing: the plate was lost on the way)' if all(p is not None and p < closed + 0.0005 for p in held) else ' (still on the plate)'))
        log.info('open the jaws')
        self.command(jaws, [JAW_OPEN] * len(jaws))
        self.wait_joints('open', jaws, [JAW_OPEN] * len(jaws), JAW_TOL, TIMEOUT['jaws'])
        log.info(f'{flip}: back to {rest:.3f} rad (rest)')
        self.command([flip], [rest])
        self.wait_joints('flip to rest', [flip], [rest], FLIP_TOL, TIMEOUT['flip'], stall=False)     # the belt must not move before the rotor is up and out of the way


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
    ap.add_argument('--now', nargs='?', const='turn_over', choices=sorted(FLIP_FOR) + ['hands'], metavar='STATE',
                    help='run once right away, without waiting for /sequence_done, and end: for a part of the state turn_over (default: the flipper, then ready for the robot) or standard (ready at once), '
                         'or `hands` = only the left hands (grab, turn, let go, belt), as if a plate were in each of them')
    ap.add_argument('--sequence', default=None, help='sequence yaml (configs/sequence/<name>.yaml): speeds.laser_marker.stall_frames is read from it')
    args, ros_args = ap.parse_known_args()
    if args.sequence:
        global STALL_FRAMES
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'sequence', f'{args.sequence}.yaml')
        sp = ((yaml.safe_load(open(f)) or {}).get('speeds') or {}).get('laser_marker') or {} if os.path.isfile(f) else {}
        STALL_FRAMES = int(sp.get('stall_frames', STALL_FRAMES))
        global FLIPPER_CLOSED, LEFT_CLOSED
        FLIPPER_CLOSED = -float(sp.get('flipper_close_mm', -FLIPPER_CLOSED * 1000)) / 1000      # how far past the pocket the jaws are told to close (mm): the harder they press
        LEFT_CLOSED = -float(sp.get('two_arm_close_mm', -LEFT_CLOSED * 1000)) / 1000
    lock = None if args.now else single_instance()                      # (keeps the file open for the life of the process)
    rclpy.init(args=ros_args)
    node = LaserMarker()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        if args.now:
            def once():
                if args.now != 'hands':
                    return node.cycle(FLIP_FOR[args.now])
                node.busy = True
                try:
                    node.wait_machine()
                    node.hands(node.get_logger())
                except TimeoutError as e:
                    node.get_logger().error(f'the left hands stopped: {e}')
                finally:
                    node.busy = False
            worker = threading.Thread(target=once, daemon=True)
            worker.start()
            while worker.is_alive():
                executor.spin_once(timeout_sec=0.05)
        else:
            node.get_logger().info(f'waiting for {TOPIC_DONE} (the robot has put the plate down and is at home again): standard = ready for the robot, turn_over = flipper, then ready')
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
