"""Capture -> 6D pose -> pick: the part with the highest score of the wrist camera image is sent to the robot as the next target.

    python3 control/vision_pick.py [--dry-run] [--loop] [--robot <name>]

What it does (the steps you would do by hand):
  1. publishes std_msgs/Empty on /wrist_camera/capture: robot_sim.py writes one RGB + depth frame to wrist_camera.output_dir of the robot yaml
  2. copies only that new frame into a work folder and runs gdrnpp's camera_convert_data.py --to-bop and infer.py on it (its own .venv python)
  3. reads <frame>_pose6d.json: the confident poses, sorted by score, highest first; the best one is the part to pick
  4. publishes its state on /part_turned_over (std_msgs/Bool, True = turned over, latched: the sequence takes the place pose from it) and its pose on
     /target_pose (geometry_msgs/Pose, the TCP that grips it) = the line `ee_6d` of the json = what you paste into ee_6d.py
control/robot_sequence.py (ros2 launch robot.launch.py ... sequence:=<name>) then picks and places it like any other target. The robot must stand where the
wrist camera sees the parts when the picture is taken.
--dry-run only prints the chosen part (nothing is published except the capture request), --loop waits for /sequence_done (the robot is at home again)
and starts the next capture, until no confident part is left. A pick that fails (the arm is stuck, e.g. the gripper against a crate wall: /sequence_done says
`failed` after the robot has gone back home) is not tried again: the parts within --skip-radius of it are left out of the next choices; --max-fails failures
in a row stop the loop. Needs the ROS 2 environment (rclpy) and robot_sim.py running.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, Empty, String

SIM_DIR = Path(__file__).resolve().parent.parent                       # Simulation/
GDRNPP = Path.home() / 'project' / 'gdrnpp_bin_6DOF_estimation'
MODELS = Path.home() / 'IsaacSim' / 'captures' / 'side_plate_10k' / 'shard_00' / 'models'    # CAD models copied by --to-bop (the part of the pose network)
TOPIC_CAPTURE, TOPIC_TARGET = '/wrist_camera/capture', '/target_pose'
TOPIC_PART, TOPIC_DONE = '/part_turned_over', '/sequence_done'
TIMEOUT = {'capture': 180.0, 'pipeline': 1200.0, 'sequence': 1800.0}   # s: frame written by the simulation, convert + infer, one pick and place


def camera_dir(robot=None):
    """wrist_camera.output_dir of the robot yaml (the folder robot_sim.py writes the frames to)"""
    scene = (yaml.safe_load(open(SIM_DIR / 'isaacsim' / 'configs' / 'map' / 'map.yaml')) or {}).get('scene') or {}
    name = robot or scene.get('robot')
    cfg = yaml.safe_load(open(SIM_DIR / 'isaacsim' / 'configs' / 'robots' / f'{name}.yaml')) or {}
    out = (cfg.get('wrist_camera') or {}).get('output_dir')
    if not out:
        raise SystemExit(f'robot {name}: no wrist_camera.output_dir in its yaml')
    return Path(os.path.expanduser(str(out)))


def frames(folder):
    rgb = folder / 'rgb'
    return {p.stem for p in rgb.glob('*.png')} if rgb.is_dir() else set()


def pose_from_line(line):
    """`x y z roll pitch yaw state` (m, deg, intrinsic XYZ: the line of ee_6d.py) -> (Pose, state)"""
    v = line.split()
    x, y, z, roll, pitch, yaw = (float(a) for a in v[:6])
    q = Rotation.from_euler('XYZ', [roll, pitch, yaw], degrees=True).as_quat()    # the same as TargetPublisher.publish_target of ee_6d.py
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = x, y, z
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (float(a) for a in q)
    return msg, (v[6] if len(v) > 6 else 'standard')


class VisionPick(Node):
    def __init__(self):
        super().__init__('vision_pick')
        self.done = None                                                # last /sequence_done (state of the part) not yet used
        self.last_xy = None                                             # x y of the part that was sent to the robot
        self.failed = []                                                # x y of the parts whose pick failed
        self.capture_pub = self.create_publisher(Empty, TOPIC_CAPTURE, 10)
        self.target_pub = self.create_publisher(Pose, TOPIC_TARGET, 10)
        self.part_pub = self.create_publisher(Bool, TOPIC_PART, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, TOPIC_DONE, self.cb_done, 10)

    def cb_done(self, msg):
        self.done = msg.data

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_for(self, publisher, name, seconds=10.0):
        end = time.time() + seconds
        while publisher.get_subscription_count() == 0 and time.time() < end:
            self.spin_for(0.2)
        if publisher.get_subscription_count() == 0:
            self.get_logger().warn(f'nobody listens on {name} (yet): is it running?')

    def capture(self, cam):
        """request one frame, wait until the simulation has written it; returns its name (000003)"""
        before = frames(cam)
        self.wait_for(self.capture_pub, TOPIC_CAPTURE)
        self.capture_pub.publish(Empty())
        end = time.time() + TIMEOUT['capture']
        while time.time() < end:
            self.spin_for(0.5)
            new = sorted(frames(cam) - before)
            if new and all((cam / sub / f'{new[-1]}{ext}').is_file() for sub, ext in (('rgb', '.png'), ('depth', '.png'), ('pose', '.json'))):
                time.sleep(0.5)                                         # the files are complete
                return new[-1]
        raise TimeoutError(f'no new frame in {cam} after {TIMEOUT["capture"]:.0f} s: is robot_sim.py running with a wrist_camera?')

    def estimate(self, cam, frame, args, work):
        """convert + infer for the one frame; returns the poses of its pose6d.json"""
        sub_cam = work / 'camera'
        for sub, ext in (('rgb', '.png'), ('depth', '.png'), ('pose', '.json')):
            (sub_cam / sub).mkdir(parents=True)
            shutil.copy(cam / sub / f'{frame}{ext}', sub_cam / sub / f'000000{ext}')
        for name in ('camera.json', 'undistort_map.npy'):
            if (cam / name).is_file():
                shutil.copy(cam / name, sub_cam / name)
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH')}    # the ROS environment must not leak into the torch venv
        python = str(args.gdrnpp / '.venv' / 'bin' / 'python')
        steps = [
            [python, 'camera_convert_data.py', '--out', str(sub_cam), '--to-bop', str(work / 'bop'), '--keep-distortion', '--models', str(args.models)],
            [python, 'infer.py', '--set', f'data.root={work / "bop"}', '--scenes', '000000', '--view-stride', '1', '--out', str(work / 'run')],
        ]
        for cmd in steps:
            self.get_logger().info(f'running {Path(cmd[1]).name} ...')
            r = subprocess.run(cmd, cwd=args.gdrnpp, env=env, capture_output=True, text=True, timeout=TIMEOUT['pipeline'])
            if r.returncode:
                raise RuntimeError(f'{Path(cmd[1]).name} failed ({r.returncode}):\n' + '\n'.join((r.stdout + r.stderr).strip().splitlines()[-15:]))
        files = sorted((work / 'run').glob('*_pose6d.json'))
        if not files:
            raise RuntimeError(f'infer.py wrote no *_pose6d.json in {work / "run"}')
        return json.load(open(files[-1]))['poses']

    def pick_once(self, args):
        """one capture + estimate + target. Returns False when there is nothing (confident) to pick"""
        cam = args.camera_dir
        frame = self.capture(cam)
        self.get_logger().info(f'frame {frame} captured')
        with tempfile.TemporaryDirectory(prefix='vision_pick_') as tmp:
            poses = self.estimate(cam, frame, args, Path(tmp))
        good = sorted((p for p in poses if p.get('confident')), key=lambda p: -p['score'])
        left = [p for p in good if all(math.hypot(p['xyz_m'][0] - x, p['xyz_m'][1] - y) > args.skip_radius for x, y in self.failed)]
        if len(left) < len(good):
            self.get_logger().info(f'{len(good) - len(left)} confident parts left out: near a pick that failed')
        good = left
        self.get_logger().info(f'{len(poses)} parts found, {len(good)} confident; highest scores: ' +
                               ', '.join(f"{p['score']:.4f} ({p['state']})" for p in good[:3]))
        if not good:
            self.get_logger().warn('no confident part: nothing to pick')
            return False
        best = good[0]
        pose, state = pose_from_line(best['ee_6d'])
        self.last_xy = (best['xyz_m'][0], best['xyz_m'][1])
        self.get_logger().info(f"pick: {best['ee_6d']}  (score {best['score']:.4f})")
        if args.dry_run:
            return True
        self.wait_for(self.target_pub, TOPIC_TARGET)
        self.done = None
        self.part_pub.publish(Bool(data=(state == 'turn_over')))          # the state first: the sequence reads it when the target comes
        self.spin_for(0.5)
        self.target_pub.publish(pose)
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--robot', help='robot name (default: scene.robot of map.yaml): its wrist_camera.output_dir is where the frames are')
    ap.add_argument('--camera-dir', type=Path, help='folder of the captured frames (default: wrist_camera.output_dir of the robot yaml)')
    ap.add_argument('--gdrnpp', type=Path, default=GDRNPP, help='the gdrnpp_bin_6DOF_estimation folder (with its .venv)')
    ap.add_argument('--models', type=Path, default=MODELS, help='CAD models folder given to camera_convert_data.py --to-bop')
    ap.add_argument('--dry-run', action='store_true', help='only print the part that would be picked, send no target')
    ap.add_argument('--skip-radius', type=float, default=0.05, help='m: parts this close to a failed pick are not tried again (default 0.05)')
    ap.add_argument('--max-fails', type=int, default=3, help='--loop: stop after this many failed picks in a row (default 3)')
    ap.add_argument('--loop', action='store_true', help='after every pick wait for /sequence_done and capture again, until nothing confident is left')
    args, ros_args = ap.parse_known_args()
    args.camera_dir = args.camera_dir or camera_dir(args.robot)
    rclpy.init(args=ros_args)
    node = VisionPick()
    try:
        fails = 0
        while True:
            if not node.pick_once(args) or args.dry_run or not args.loop:
                break
            end = time.time() + TIMEOUT['sequence']
            while node.done is None and time.time() < end:
                node.spin_for(0.5)
            if node.done is None:
                node.get_logger().error('the robot did not report /sequence_done: stopping')
                break
            if node.done == 'failed':                                   # stuck on the way: the robot is back at home, the part is left alone
                node.failed.append(node.last_xy)
                fails += 1
                node.get_logger().warn(f'the pick failed ({fails}/{args.max_fails}): this part is left out, the next capture follows')
                if fails >= args.max_fails:
                    node.get_logger().error('too many failed picks in a row: stopping')
                    break
            else:
                fails = 0
    except (TimeoutError, RuntimeError, subprocess.TimeoutExpired) as e:
        node.get_logger().error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
