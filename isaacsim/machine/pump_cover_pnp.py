"""pump_cover_pnp.py: cycle of the pump cover PNP: when the two end jigs of conveyor 1 carry a cover (the hand plates topic of robot_sim.py), the cups go down and suck,
go up, move over conveyor 2 turned by 90 deg, go down and let go, conveyor 2 moves a step, the PNP goes back.

    python3 isaacsim/machine/pump_cover_pnp.py [--now]        (robot.launch.py with a sequence starts it; settings: configs/machine/pump_cover_pnp.yaml; --now = one cycle at once, without waiting for the covers)
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
from std_msgs.msg import Bool, Empty, Int32

MACHINE = 'pump_cover_pnp'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sim_config

CFG = sim_config.load_machine(MACHINE)
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def need(path):
    node = CFG
    for k in path.split('.'):
        if not isinstance(node, dict) or k not in node:
            raise SystemExit(f'{sim_config.machine_file(MACHINE)}: missing {path}')
        node = node[k]
    return node


JOINT = {k: str(need(f'cycle.joints.{k}')) for k in ('slide', 'z', 'turn')}
UP_Z = float(need('cycle.up_z'))
PICK = {k: float(need(f'cycle.pick.{k}')) for k in ('slide', 'z', 'turn')}
PLACE = {k: float(need(f'cycle.place.{k}')) for k in ('slide', 'z', 'turn')}
TOL = {k: float(need(f'cycle.tol.{k}')) for k in ('slide', 'z', 'turn')}
SETTLE = float(need('cycle.settle_s'))
TOPIC_PLATES = str(need('cycle.source.plates_topic'))
TOPIC_SOURCE_MOVING = str(need('cycle.source.moving_topic'))
MASK = int(need('cycle.source.mask'))
STABLE = float(need('cycle.source.stable_s'))
TOPIC_CARRIERS = str(need('cycle.target.carriers_topic'))
TOPIC_TARGET_MOVING = str(need('cycle.target.moving_topic'))
TOPIC_REMOVE = str(need('cycle.target.remove_topic'))
STEPS = int(need('cycle.target.steps'))
TIMEOUT = {k: float(v) for k, v in need('cycle.timeouts_s').items()}
TOPIC_VACUUM = str(need('vacuum.topic'))
ALL_CUPS = (1 << len(need('vacuum.cups'))) - 1
TOPIC_COMMAND = f'/{MACHINE}/joint_command'
TOPIC_STATE = f'/{MACHINE}/joint_states'


class Pnp(Node):
    def __init__(self):
        super().__init__(f'{MACHINE}_cycle')
        self.q, self.plates, self.held = {}, 0, 0
        self.source_moving, self.target_moving = None, None
        self.command = self.create_publisher(JointState, TOPIC_COMMAND, 10)
        self.vacuum = self.create_publisher(Int32, TOPIC_VACUUM, 10)
        self.carriers = self.create_publisher(Int32, TOPIC_CARRIERS, 10)
        self.remove = self.create_publisher(Empty, TOPIC_REMOVE, 10)
        self.create_subscription(JointState, TOPIC_STATE, self.cb_state, 10)
        self.create_subscription(Int32, TOPIC_PLATES, lambda m: setattr(self, 'plates', int(m.data)), 10)
        self.create_subscription(Int32, TOPIC_VACUUM + '/held', lambda m: setattr(self, 'held', int(m.data)), 10)
        self.create_subscription(Bool, TOPIC_SOURCE_MOVING, lambda m: setattr(self, 'source_moving', bool(m.data)), LATCHED)
        self.create_subscription(Bool, TOPIC_TARGET_MOVING, lambda m: setattr(self, 'target_moving', bool(m.data)), LATCHED)

    def cb_state(self, msg):
        self.q = dict(zip(msg.name, msg.position))

    def wait(self, test, seconds, what):
        end = time.time() + seconds
        while not test():
            if time.time() > end:
                raise TimeoutError(what)
            time.sleep(0.05)

    def wait_machine(self):
        self.wait(lambda: bool(self.q) and self.command.get_subscription_count() > 0 and self.vacuum.get_subscription_count() > 0 and self.target_moving is not None,
                  TIMEOUT['machine'], f'nobody answers on {TOPIC_STATE}: is robot_sim.py running with the machine {MACHINE}?')

    def wait_source(self):
        since = None
        while True:
            if self.plates & MASK == MASK and self.source_moving is False:
                since = since or time.time()
                if time.time() - since >= STABLE:
                    return
            else:
                since = None
            time.sleep(0.1)

    def go(self, **goal):
        msg = JointState()
        msg.name, msg.position = [JOINT[k] for k in goal], [float(v) for v in goal.values()]
        self.command.publish(msg)
        self.wait(lambda: all(abs(self.q.get(JOINT[k], 1e9) - v) < TOL[k] for k, v in goal.items()), TIMEOUT['move'], f'{MACHINE} did not reach {goal}')

    def suck(self, mask):
        self.vacuum.publish(Int32(data=mask))
        self.wait(lambda: self.held == mask, TIMEOUT['vacuum'], f'vacuum {mask}: the cups hold {self.held}')

    def cycle(self):
        log = self.get_logger()
        self.go(z=PICK['z'])
        time.sleep(SETTLE)
        self.suck(ALL_CUPS)
        log.info('the covers are sucked: up, over conveyor 2')
        time.sleep(SETTLE)
        self.go(z=UP_Z)
        self.go(slide=PLACE['slide'], turn=PLACE['turn'])
        self.wait(lambda: self.target_moving is False, TIMEOUT['conveyor'], f'{TOPIC_CARRIERS} keeps moving')
        self.go(z=PLACE['z'])
        time.sleep(SETTLE)
        self.suck(0)
        time.sleep(SETTLE)
        self.go(z=UP_Z)
        self.carriers.publish(Int32(data=STEPS))
        self.go(slide=PICK['slide'], turn=PICK['turn'])
        self.wait(lambda: self.target_moving is False, TIMEOUT['conveyor'], f'{TOPIC_CARRIERS} keeps moving')
        time.sleep(SETTLE)
        self.remove.publish(Empty())

    def run(self, now):
        log = self.get_logger()
        try:
            self.wait_machine()
            self.go(slide=PICK['slide'], turn=PICK['turn'], z=UP_Z)
            while True:
                if not now:
                    log.info(f'waiting for the covers on the end jigs of conveyor 1 ({TOPIC_PLATES} & {MASK})')
                    self.wait_source()
                self.cycle()
                log.info('cycle done')
                if now:
                    return
        except TimeoutError as e:
            log.error(f'pump cover PNP stopped: {e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--now', action='store_true', help='one cycle at once, without waiting for the covers on conveyor 1, and end')
    args, ros_args = ap.parse_known_args()
    lock = None if args.now else sim_config.single_instance(f'{MACHINE}_cycle', f'pump_cover_pnp.py is already running (started by the launch, or by hand before): a second one would move the PNP twice')
    rclpy.init(args=ros_args)
    node = Pnp()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    worker = threading.Thread(target=node.run, args=(args.now,), daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
