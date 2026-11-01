"""laser_marker.py: cycle of the laser marker: the flipper turns a turn_over plate onto the standard seat and tells the robot (/laser_marker/ready),
the belt moves after /transfer_done, the left hands put the plates on the conveyor.

    python3 isaacsim/machine/laser_marker.py [--now [standard|turn_over|hands]]        (robot.launch.py with a sequence starts it; settings: configs/machine/laser_marker.yaml)
"""
import argparse
import os
import sys
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Int32, String

MACHINE = 'laser_marker'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sim_config

CFG = sim_config.load_machine(MACHINE)


def need(path):
    node = CFG
    for k in path.split('.'):
        if not isinstance(node, dict) or k not in node:
            raise SystemExit(f'{sim_config.machine_file(MACHINE)}: missing {path}')
        node = node[k]
    return node


FLIP = str(need('cycle.flipper.rotor'))
FLIP_REST, FLIP_GRAB, FLIP_RELEASE = (float(need(f'cycle.flipper.{k}')) for k in ('rest', 'grab', 'release'))
FLIP_FOR = {'standard': False, 'turn_over': True}
JAWS = [str(j) for j in need('cycle.flipper.jaws')]
LEFT_FLIP = str(need('cycle.hands.rotor'))
LEFT_JAWS = [str(j) for j in need('cycle.hands.jaws')]
LEFT_REST, LEFT_GRAB, LEFT_RELEASE = (float(need(f'cycle.hands.{k}')) for k in ('rest', 'grab', 'release'))
HAND_STILL = float(need('cycle.hands.still_s'))
SLOTS, SLOT_PLACE = int(need('cycle.slots')), int(need('cycle.slot_place'))
HANDS = tuple(int(i) for i in need('hands.slots'))
JAW_OPEN = float(need('cycle.jaw_open'))
FLIPPER_CLOSED = -float(need('cycle.flipper.close_mm')) / 1000.0
LEFT_CLOSED = -float(need('cycle.hands.close_mm')) / 1000.0
TOPIC_HAND_PLATES = str(need('hands.topic'))
TOPIC_READY = str(need('cycle.topics.ready'))
TOPIC_TRANSFER_DONE = str(need('cycle.topics.transfer_done'))
TOPIC_HANDS_DONE = str(need('cycle.topics.hands_done'))
TOPIC_HANDS_RELEASE = str(need('cycle.topics.hands_release'))
TOPIC_DONE = str(need('cycle.topics.sequence_done'))
TOPIC_COMMAND = f'/{MACHINE}/joint_command'
TOPIC_STATE = f'/{MACHINE}/joint_states'
TOPIC_CARRIERS = str(need('carriers.topic'))
FLIP_TOL, JAW_TOL = float(need('cycle.flip_tol')), float(need('cycle.jaw_tol'))
CLAMP_WAIT = 0.3
RELEASE_TOL = float(need('cycle.release_tol'))
STALL_FRAMES, STALL_MOVE = int(need('cycle.stall_frames')), float(need('cycle.stall_move'))
TIMEOUT = {k: float(v) for k, v in need('cycle.timeouts_s').items()}


class LaserMarker(Node):
    def __init__(self):
        super().__init__('laser_marker_cycle')
        self.q = {}
        self.frames = 0
        self.moving = None
        self.busy = False
        self.lock = threading.Lock()
        self.slots = [False] * SLOTS
        self.create_subscription(JointState, TOPIC_STATE, self.cb_state, 10)
        self.create_subscription(Bool, f'{TOPIC_CARRIERS}/moving', self.cb_moving,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, TOPIC_DONE, self.cb_done, 10)
        self.joint_pub = self.create_publisher(JointState, TOPIC_COMMAND, 10)
        self.belt_pub = self.create_publisher(Int32, TOPIC_CARRIERS, 10)
        self.hands_done_pub = self.create_publisher(Empty, TOPIC_HANDS_DONE, 10)
        self.hands_release_pub = self.create_publisher(Empty, TOPIC_HANDS_RELEASE, 10)
        self.ready_pub = self.create_publisher(Empty, TOPIC_READY, 10)
        self.create_subscription(Empty, TOPIC_TRANSFER_DONE, self.cb_transfer_done, 10)
        self.create_subscription(Int32, TOPIC_HAND_PLATES, self.cb_hand_plates, 10)
        self.hand_mask, self.hand_since, self.hands_wait_clear = 0, 0.0, False

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
            self.get_logger().info('the machine is still running its cycle: /sequence_done waits for its end')
        self.get_logger().info(f'a {state} part was placed: ' + ('flipper, then ' if FLIP_FOR[state] else '') + 'the robot takes it to the belt')
        threading.Thread(target=self.cycle, args=(FLIP_FOR[state],), daemon=True).start()

    def wait_machine(self):
        end = time.time() + TIMEOUT['machine']
        while not (self.q and self.joint_pub.get_subscription_count() and self.belt_pub.get_subscription_count()):
            if time.time() > end:
                raise TimeoutError(f'no {TOPIC_STATE} / nobody listening on {TOPIC_COMMAND}: is robot_sim.py running with the {MACHINE}?')
            time.sleep(0.1)

    def command(self, names, positions):
        msg = JointState()
        msg.name, msg.position = list(names), [float(p) for p in positions]
        for _ in range(3):
            self.joint_pub.publish(msg)
            time.sleep(0.1)

    def wait_joints(self, label, names, targets, tol, timeout, stall=True):
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
            time.sleep(0.01)

    def step_belt(self):
        self.belt_pub.publish(Int32(data=1))
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
        def work(log):
            sent = []

            def ready():
                if not sent:
                    sent.append(True)
                    log.info('the plate lies at the standard coordinate: the robot takes it to the belt')
                    self.ready_pub.publish(Empty())
            if flip:
                self.flipper(log, FLIP, JAWS, FLIP_REST, FLIP_GRAB, FLIP_RELEASE, FLIPPER_CLOSED, after_open=ready)
            ready()
        self.run_cycle('laser marker cycle', work, 'laser marker cycle done')

    def run_cycle(self, name, work, done):
        self.lock.acquire()
        self.busy = True
        log = self.get_logger()
        try:
            self.wait_machine()
            work(log)
            log.info(done)
        except TimeoutError as e:
            log.error(f'{name} stopped: {e}')
        finally:
            self.busy = False
            self.lock.release()

    def cb_hand_plates(self, msg):
        mask = int(msg.data)
        if mask != self.hand_mask:
            self.hand_mask, self.hand_since = mask, time.time()
        if mask != 3:
            self.hands_wait_clear = False
            return
        if self.hands_wait_clear or self.busy or self.moving or time.time() - self.hand_since < HAND_STILL:
            return
        self.hands_wait_clear = True
        self.get_logger().info('a plate in front of each of the two hand pairs: the hands take them over to the conveyor')
        threading.Thread(target=self.hands_cycle, daemon=True).start()

    def hands_cycle(self):
        self.run_cycle('hands cycle', self.hands, 'hands cycle done: waiting for the next two plates')

    def cb_transfer_done(self, _):
        if self.busy:
            self.get_logger().info('the machine is still running its cycle: /transfer_done waits for its end')
        threading.Thread(target=self.after_transfer, daemon=True).start()

    def after_transfer(self):
        self.run_cycle('belt cycle', self.belt, 'belt cycle done')

    def hands(self, log):
        self.flipper(log, LEFT_FLIP, LEFT_JAWS, LEFT_REST, LEFT_GRAB, LEFT_RELEASE, LEFT_CLOSED,
                     before_open=lambda: (self.hands_release_pub.publish(Empty()), time.sleep(0.3)))
        self.hands_done_pub.publish(Empty())
        for i in HANDS:
            self.slots[i] = False

    def belt(self, log):
        log.info('the belt moves one pitch')
        self.step_belt()
        self.slots = self.slots[1:] + [False]
        log.info('belt slots (left to right): ' + ''.join('X' if s else '.' for s in self.slots))

    def flipper(self, log, flip, jaws, rest, grab, release, closed, before_open=None, after_open=None):
        if any(self.q.get(j, JAW_OPEN) < JAW_OPEN - JAW_TOL for j in jaws):
            log.info('open the jaws first')
            self.command(jaws, [JAW_OPEN] * len(jaws))
            self.wait_joints('open', jaws, [JAW_OPEN] * len(jaws), JAW_TOL, TIMEOUT['jaws'])
        log.info(f'{flip}: to {grab:.3f} rad (grab)')
        self.command([flip], [grab])
        self.wait_joints('flip to grab', [flip], [grab], FLIP_TOL, TIMEOUT['flip'])
        log.info('close the jaws')
        self.command(jaws, [closed] * len(jaws))
        self.wait_joints('close', jaws, [closed] * len(jaws), JAW_TOL, TIMEOUT['jaws'])
        time.sleep(CLAMP_WAIT)
        if all(self.q.get(j) is not None and self.q[j] < closed + 0.003 for j in jaws):
            log.warn('the jaws closed on nothing: no plate at the seat; they open, wait 1 s and close once more')
            self.command(jaws, [JAW_OPEN] * len(jaws))
            self.wait_joints('open', jaws, [JAW_OPEN] * len(jaws), JAW_TOL, TIMEOUT['jaws'])
            time.sleep(1.0)
            self.command(jaws, [closed] * len(jaws))
            self.wait_joints('close', jaws, [closed] * len(jaws), JAW_TOL, TIMEOUT['jaws'])
        log.info(f'{flip}: to {release:.3f} rad (release)')
        self.command([flip], [release])
        self.wait_joints('flip to release', [flip], [release], RELEASE_TOL, TIMEOUT['flip'])
        held = [self.q.get(j) for j in jaws]
        log.info('jaws before they let go: ' + ', '.join(f'{p:.4f}' for p in held if p is not None)
                 + (f' (about {closed:.4f} = closed on nothing: the plate was lost on the way)' if all(p is not None and p < closed + 0.003 for p in held) else ' (still on the plate)'))
        if before_open is not None:
            before_open()
        log.info('open the jaws')
        self.command(jaws, [JAW_OPEN] * len(jaws))
        self.wait_joints('open', jaws, [JAW_OPEN] * len(jaws), JAW_TOL, TIMEOUT['jaws'])
        if after_open is not None:
            after_open()
        log.info(f'{flip}: back to {rest:.3f} rad (rest)')
        self.command([flip], [rest])
        self.wait_joints('flip to rest', [flip], [rest], FLIP_TOL, TIMEOUT['flip'], stall=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--now', nargs='?', const='turn_over', choices=sorted(FLIP_FOR) + ['hands'], metavar='STATE',
                    help='run once right away, without waiting for /sequence_done, and end: for a part of the state turn_over (default: the flipper, then ready for the robot) or standard (ready at once), '
                         'or `hands` = only the left hands (grab, turn, let go, belt), as if a plate were in each of them')
    args, ros_args = ap.parse_known_args()
    lock = None if args.now else sim_config.single_instance('laser_marker_cycle', 'laser_marker.py is already running (started by the launch, or by hand before): a second one would run every cycle twice')
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
