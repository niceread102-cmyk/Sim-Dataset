"""conveyor.py: steps the conveyor chain of the map, one pitch per trigger_topic (the conveyor: /laser_marker/hands_done; robot_sim.py moves it on carriers.topic).

    python3 isaacsim/machine/conveyor.py [--machine <name>] [--now [N]]        (robot.launch.py with a sequence starts the one of the machine named conveyor; settings: configs/machine/<name>.yaml)
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
from std_msgs.msg import Bool, Empty, Int32

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sim_config

NAME = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == '--machine'), 'conveyor')
CFG = sim_config.load_machine(NAME)
try:
    TOPIC_TRIGGER = str(CFG['trigger_topic'])
    TOPIC_CARRIERS = str(CFG['carriers']['topic'])
    TIMEOUT = {k: float(v) for k, v in CFG['cycle']['timeouts_s'].items()}
except (KeyError, TypeError) as e:
    raise SystemExit(f"{sim_config.machine_file(NAME)}: missing {e} (trigger_topic, carriers.topic, cycle.timeouts_s)") from None


class Conveyor(Node):
    def __init__(self):
        super().__init__(f'{NAME}_step')
        self.moving = None
        self.pending = 0
        self.lock = threading.Lock()
        self.worker = None
        self.create_subscription(Bool, f'{TOPIC_CARRIERS}/moving', self.cb_moving,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Empty, TOPIC_TRIGGER, self.cb_trigger, 10)
        self.pub = self.create_publisher(Int32, TOPIC_CARRIERS, 10)

    def cb_moving(self, msg):
        self.moving = bool(msg.data)

    def cb_trigger(self, _):
        self.get_logger().info(f'{TOPIC_TRIGGER}: {NAME} moves one step')
        self.request(1)

    def request(self, n):
        with self.lock:
            self.pending += n
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self.run, daemon=True)
                self.worker.start()

    def wait_machine(self):
        end = time.time() + TIMEOUT['machine']
        while self.pub.get_subscription_count() == 0 or self.moving is None:
            if time.time() > end:
                raise TimeoutError(f'nobody listens on {TOPIC_CARRIERS}: is robot_sim.py running with the machine {NAME}?')
            time.sleep(0.1)

    def step(self):
        self.pub.publish(Int32(data=1))
        end = time.time() + TIMEOUT['start']
        while self.moving is not True:
            if time.time() > end:
                raise TimeoutError(f'the chain did not start ({TOPIC_CARRIERS}/moving stays {self.moving})')
            time.sleep(0.05)
        end = time.time() + TIMEOUT['step']
        while self.moving is not False:
            if time.time() > end:
                raise TimeoutError('the chain did not stop')
            time.sleep(0.05)

    def run(self):
        log = self.get_logger()
        try:
            self.wait_machine()
            while True:
                with self.lock:
                    if self.pending <= 0:
                        return
                    self.pending -= 1
                self.step()
                log.info('conveyor: one step done')
        except TimeoutError as e:
            with self.lock:
                self.pending = 0
            log.error(f'conveyor stopped: {e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--machine', default='conveyor', help='name of the machine (configs/machine/<name>.yaml) whose chain is stepped (default conveyor)')
    ap.add_argument('--now', nargs='?', const=1, type=int, metavar='N', help='move the chain N pitches (default 1) right away, without waiting for the hands, and end')
    args, ros_args = ap.parse_known_args()
    lock = None if args.now else sim_config.single_instance(f'{NAME}_step', f'conveyor.py --machine {NAME} is already running (started by the launch, or by hand before): a second one would move the chain twice for every step')
    rclpy.init(args=ros_args)
    node = Conveyor()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        if args.now:
            node.request(args.now)
            time.sleep(0.2)
            while node.worker is not None and node.worker.is_alive():
                executor.spin_once(timeout_sec=0.05)
        else:
            node.get_logger().info(f'waiting for {TOPIC_TRIGGER}: one step of the chain of {NAME} each')
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
