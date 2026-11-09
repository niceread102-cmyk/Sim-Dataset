"""auto_control.py: pick and place crate after crate: the poses of the parts of a crate come from the part provider (/crate_parts_request -> /crate_parts json
{index, parts: [{xyz_m, ee_6d, state, ...}]}: robot_sim.py in the simulation, the vision node on the real cell), the best part goes to robot_sequence.py (/target_pose),
it is put at the standard or the turn_over coordinate (poses.place of the sequence yaml) and then removed (/crate_parts_remove, crate.remove_placed) and/or the conveyor
moves one pitch (crate.conveyor_topic), then the next part of the same crate; an empty crate -> the next crate; after the last crate the arm goes back home.

    python3 control/auto_control.py sequence:=<name> [--start <crate>] [--max-fails 3] [robot:=<name>]      (--sequence <name> works too)

The part provider (simulation or vision node) and robot.launch.py sequence:=<name> run first; the two only talk over the topics.
"""
import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, Int32, String

import cell_config
from pick_filter import rank_world

TOPIC_TARGET, TOPIC_PART, TOPIC_DONE = '/target_pose', '/part_turned_over', '/sequence_done'
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
TIMEOUT = {'listener': 300.0, 'crate': 600.0, 'parts': 120.0, 'sequence': 1800.0}


def crate_config(robot=None, sequence=None):
    try:
        cfg = cell_config.load_sequence_file(cell_config.sequence_of(sequence))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    name = cell_config.canonical_robot(robot or cfg.get('robot')) if (robot or cfg.get('robot')) else None
    mine = (cfg.get('robots') or {}).get(name) or {}
    crate = mine.get('crate') or cfg.get('crate')
    crates = (mine.get('poses') or {}).get('crates') or (cfg.get('poses') or {}).get('crates') or mine.get('crates') or cfg.get('crates') or []
    listed = mine.get('crate_list') or cfg.get('crate_list')
    place = (mine.get('poses') or {}).get('place') or (cfg.get('poses') or {}).get('place') or {}
    place = place.values() if isinstance(place, dict) else [place]
    if not crate or not (crates or listed):
        raise SystemExit(f'sequence {sequence}: no `crate:` or no `crate_list:` / `poses.crates:`')
    expanded = [dict(c) for c in listed] if listed else cell_config.expand_crates(crate, crates)
    for c in expanded:
        c.setdefault('off', False)
    return crate, expanded, [[float(v) for v in p[:3]] for p in place]


class AutoControl(Node):
    def __init__(self, crate, crates, place):
        super().__init__('auto_control')
        self.crate, self.crates, self.place = crate, crates, place
        self.ready, self.parts, self.done, self.want, self.missing = None, None, None, None, False
        self.next_pub = self.create_publisher(Int32, crate.get('next_topic', '/crate_next'), 10)
        self.active_pub = self.create_publisher(String, crate.get('active_topic', '/crate_active'), LATCHED)
        self.clear_pub = self.create_publisher(String, crate.get('clear_topic', '/crate_clear'), LATCHED)
        self.request_pub = self.create_publisher(String, crate.get('parts_request_topic', '/crate_parts_request'), 10)
        self.remove_pub = self.create_publisher(String, crate.get('parts_remove_topic', '/crate_parts_remove'), 10)
        self.conveyor_pub = self.create_publisher(Int32, crate['conveyor_topic'], 10) if crate.get('conveyor_topic') else None
        self.part_pub = self.create_publisher(Bool, TOPIC_PART, LATCHED)
        self.target_pub = self.create_publisher(Pose, TOPIC_TARGET, 10)
        self.create_subscription(Int32, crate.get('ready_topic', '/crate_ready'), lambda m: setattr(self, 'ready', int(m.data)), 10)
        self.create_subscription(String, crate.get('parts_topic', '/crate_parts'), self.cb_parts, 10)
        self.create_subscription(String, TOPIC_DONE, lambda m: setattr(self, 'done', m.data), 10)

    def cb_parts(self, msg):
        reply = json.loads(msg.data)
        if reply.get('index') == self.want:
            self.parts, self.missing = reply['parts'], bool(reply.get('missing'))

    def spin_until(self, test, seconds):
        end = time.time() + seconds
        while not test() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return test()

    def wait_listener(self, pub, who):
        if not self.spin_until(lambda: pub.get_subscription_count() > 0, 5.0):
            self.get_logger().info(f'waiting for {who} ...')
            if not self.spin_until(lambda: pub.get_subscription_count() > 0, TIMEOUT['listener']):
                raise TimeoutError(f'{who} is not running')

    def crate_msg(self, i):
        c = self.crates[i]
        return String(data=json.dumps({'index': i, 'centre': list(c['centre']), 'floor_z': float(c['floor_z'])}))

    def go_crate(self, i):
        if i >= 0:
            self.active_pub.publish(self.crate_msg(i))
        self.ready = None
        self.next_pub.publish(Int32(data=i))
        if not self.spin_until(lambda: self.ready is not None, TIMEOUT['crate']):
            raise TimeoutError(f'no /crate_ready for crate {i} from robot_sequence.py')
        if self.ready != i:
            raise RuntimeError('the arm did not reach ' + (f'the scan pose of crate {i}' if i >= 0 else 'the home pose'))

    def request(self, i):
        self.want, self.parts = i, None
        end = time.time() + TIMEOUT['parts']
        while time.time() < end:
            self.request_pub.publish(self.crate_msg(i))
            if self.spin_until(lambda: self.parts is not None, 5.0):
                return self.parts
        raise TimeoutError(f'no parts of crate {i} from the part provider (/crate_parts)')

    def pick(self, best):
        x, y, z, roll, pitch, yaw = (float(v) for v in best['ee_6d'].split()[:6])
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(v) for v in Rotation.from_euler('XYZ', [roll, pitch, yaw], degrees=True).as_quat())
        self.done = None
        self.part_pub.publish(Bool(data=best['state'] == 'turn_over'))
        self.spin_until(lambda: False, 0.2)
        self.target_pub.publish(pose)
        if not self.spin_until(lambda: self.done is not None, TIMEOUT['sequence']):
            raise TimeoutError('no /sequence_done from robot_sequence.py')
        if self.place and self.crate.get('remove_placed', True) and self.done != 'unreachable':
            self.remove_pub.publish(String(data=json.dumps({'xyz_m': self.place, 'radius_m': float(self.crate.get('remove_radius_m', 0.08))})))
        if self.conveyor_pub and self.done not in ('failed', 'unreachable'):
            self.conveyor_pub.publish(Int32(data=1))
        return self.done

    def run(self, start, max_fails):
        self.wait_listener(self.next_pub, 'robot_sequence.py (robot.launch.py ... sequence:=true)')
        self.wait_listener(self.request_pub, 'the part provider (robot_sim.py / vision node)')
        fails, skip = 0, float(self.crate.get('skip_radius_m', 0.05))
        for i in range(start, len(self.crates)):
            c = self.crates[i]
            if c.get('off'):
                continue
            self.request(i)
            if self.missing:
                self.get_logger().warn(f"crate {i}/{len(self.crates) - 1} (centre {c['centre']}, floor z {c['floor_z']:.4f}): no such crate in the simulation -> skipped, the next one")
                continue
            self.get_logger().info(f"crate {i}/{len(self.crates) - 1} (centre {c['centre']}, layer {c['level']} from the top): the arm goes there")
            self.go_crate(i)
            self.spin_until(lambda: False, float(self.crate.get('settle_s', 1.0)))
            failed = []
            while True:
                parts = self.request(i)
                left = [p for p in parts if all(math.hypot(p['xyz_m'][0] - fx, p['xyz_m'][1] - fy) > skip for fx, fy in failed)]
                ranked = rank_world(left, {**self.crate, 'centre': c['centre'], 'floor_z': c['floor_z']})
                self.get_logger().info(f"crate {i}: {len(parts)} parts ({sum(p['state'] == 'turn_over' for p in parts)} turned over)"
                                       + (f", {len(parts) - len(left)} left out near a failed pick" if len(left) < len(parts) else ''))
                if not ranked:
                    self.clear_pub.publish(self.crate_msg(i))
                    self.get_logger().info(f'crate {i} is done -> /crate_clear')
                    self.spin_until(lambda: False, 0.2)
                    break
                best = ranked[0]
                self.get_logger().info(f"pick: {best['ee_6d']}  (tilt {best['tilt_deg']:.1f} deg, rank {best['rank_score']:.2f})")
                outcome = self.pick(best)
                if outcome == 'unreachable':
                    failed.append(best['xyz_m'][:2])
                    self.get_logger().warn('this part cannot be reached (reach check): left out, the next one')
                    continue
                if outcome != 'failed':
                    fails = 0
                    self.get_logger().info(f"placed at the {best['state']} coordinate")
                    continue
                fails += 1
                failed.append(best['xyz_m'][:2])
                self.get_logger().warn(f'the pick failed ({fails}/{max_fails}): this part is left out')
                if fails >= max_fails:
                    self.get_logger().warn(f'crate {i}: {fails} failed picks in a row -> this crate is left, the next one')
                    fails = 0
                    break
        self.get_logger().info('all crates are done: the arm goes back home')
        self.go_crate(-1)
        self.get_logger().info('done')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', type=int, default=0, help='index of the first crate (default 0)')
    ap.add_argument('--max-fails', type=int, default=3, help='stop after this many failed picks in a row (default 3)')
    ap.add_argument('--robot', help='robot name (default: the robot of the sequence)')
    ap.add_argument('--sequence', help='sequence name = control/sequence/<name>.yaml (or the environment variable ROBOT_SEQUENCE)')
    args, ros_args = ap.parse_known_args()
    crate, crates, place = crate_config(args.robot or cell_config.option(ros_args, 'robot'), args.sequence or cell_config.option(ros_args, 'sequence'))
    rclpy.init(args=ros_args)
    node = AutoControl(crate, crates, place)
    try:
        node.run(args.start, args.max_fails)
    except (TimeoutError, RuntimeError) as e:
        node.get_logger().error(str(e))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
