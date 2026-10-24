"""Step of the conveyor of the map (robot_sim.py machines: conveyor, its chain of attachments over ROS 2).

    python3 isaacsim/machine/conveyor.py [--now [N]]      (ros2 launch robot.launch.py map:=<map> sequence:=<name> starts it in a map with the conveyor)
Its topics and time limits are trigger_topic, carriers.topic and cycle: of configs/machine/conveyor.yaml.

The two pairs of left hands of the laser marker put a plate each on the attachments of the conveyor (one in each of its two rows). When the left rotor is back at rest,
laser_marker.py publishes /laser_marker/hands_done (std_msgs/Empty): the chain then moves one pitch (0.1 m), so that the next pair of attachments is under the hands
for the next two plates. The chain is a loop: what leaves at the -x end of the straight run comes back at the +x end.

  /conveyor/carriers          std_msgs/Int32   n > 0 = advance n pitches, 0 = stop (robot_sim.py)
  /conveyor/carriers/moving   std_msgs/Bool    True while the chain moves (latched)

Every /laser_marker/hands_done is one step; one that comes while the chain still moves waits for its turn. --now moves the chain N pitches (default 1) right away and ends.
Only one of these nodes may run per ROS domain (two would move the chain twice).
"""
import argparse
import fcntl
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
import sim_config  # noqa: E402  configs/machine/conveyor.yaml

CFG = sim_config.load_machine('conveyor')                              # configs/machine/conveyor.yaml (carriers.topic: the one robot_sim.py moves the chain on)
try:
    TOPIC_TRIGGER = str(CFG['trigger_topic'])                          # std_msgs/Empty from laser_marker.py: the left hands have let go of their plates
    TOPIC_CARRIERS = str(CFG['carriers']['topic'])
    TIMEOUT = {k: float(v) for k, v in CFG['cycle']['timeouts_s'].items()}   # s: machine, start, step (wall clock: the simulation may run slower than real time)
except (KeyError, TypeError) as e:
    raise SystemExit(f"{sim_config.machine_file('conveyor')}: missing {e} (trigger_topic, carriers.topic, cycle.timeouts_s)") from None


class Conveyor(Node):
    def __init__(self):
        super().__init__('conveyor_step')
        self.moving = None                                              # chain moving (None until the first message)
        self.pending = 0                                                # steps still to do (one per /laser_marker/hands_done)
        self.lock = threading.Lock()
        self.worker = None
        self.create_subscription(Bool, f'{TOPIC_CARRIERS}/moving', self.cb_moving,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Empty, TOPIC_TRIGGER, self.cb_trigger, 10)
        self.pub = self.create_publisher(Int32, TOPIC_CARRIERS, 10)

    def cb_moving(self, msg):
        self.moving = bool(msg.data)

    def cb_trigger(self, _):
        self.get_logger().info('the left hands have let go: the conveyor moves one step')
        self.request(1)

    def request(self, n):
        """n more steps; one worker thread does them one after the other"""
        with self.lock:
            self.pending += n
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self.run, daemon=True)
                self.worker.start()

    def wait_machine(self):
        end = time.time() + TIMEOUT['machine']
        while self.pub.get_subscription_count() == 0 or self.moving is None:
            if time.time() > end:
                raise TimeoutError(f'nobody listens on {TOPIC_CARRIERS}: is robot_sim.py running with the conveyor?')
            time.sleep(0.1)

    def step(self):
        """one pitch: send it, wait until the chain moves and until it has stopped again"""
        self.pub.publish(Int32(data=1))                                 # once: every message adds a pitch
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


def single_instance():
    """only one per ROS domain: the lock goes with the process"""
    lock = open(f'/tmp/conveyor_step_{os.environ.get("ROS_DOMAIN_ID", "0")}.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit('conveyor.py is already running (started by the launch, or by hand before): a second one would move the chain twice for every step')
    return lock


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--now', nargs='?', const=1, type=int, metavar='N', help='move the chain N pitches (default 1) right away, without waiting for the hands, and end')
    args, ros_args = ap.parse_known_args()
    lock = None if args.now else single_instance()                      # (keeps the file open for the life of the process)
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
            node.get_logger().info(f'waiting for {TOPIC_TRIGGER} (the left hands of the laser marker have let go of their plates): one step of the chain each')
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
