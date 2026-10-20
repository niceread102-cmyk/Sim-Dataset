"""Capture -> 6D pose -> pick: the part with the highest score of the wrist camera image is sent to the robot as the next target.

    python3 control/vision_pick.py [--dry-run] [--loop] [--robot <name>]

What it does (the steps you would do by hand):
  1. publishes std_msgs/Empty on /wrist_camera/capture: robot_sim.py writes one RGB + depth frame to wrist_camera.output_dir of the robot yaml
  2. copies only that new frame into a work folder and runs gdrnpp's camera_convert_data.py --to-bop and infer.py on it (its own .venv python)
  3. reads <frame>_pose6d.json: the confident poses, sorted by score, highest first; the best one is the part to pick
  4. saves a picture of all parts with their labels, boxes and the score ranking (the best one marked) in runs/vision_pick/ (--blur n blurs the parts) and publishes its path on /vision_pick/picture:
     robot_sim.py shows it in a window of Isaac Sim
  5. publishes its state on /part_turned_over (std_msgs/Bool, True = turned over, latched: the sequence takes the place pose from it) and its pose on
     /target_pose (geometry_msgs/Pose, the TCP that grips it) = the line `ee_6d` of the json = what you paste into ee_6d.py
control/robot_sequence.py (ros2 launch robot.launch.py ... sequence:=<name>) then picks and places it like any other target. The robot must stand where the
wrist camera sees the parts when the picture is taken.
--dry-run only prints the chosen part (nothing is published except the capture request), --loop waits for /sequence_done (the plate lies at the standard coordinate) and then for /transfer_done (the robot put it on the conveyor and is at home again: the last step)
and only then starts the next capture, until no confident part is left (--no-transfer: without the second wait). A pick that fails (the arm is stuck, e.g. the gripper against a crate wall: /sequence_done says
`failed` after the robot has gone back home) is not tried again: the parts within --skip-radius of it are left out of the next choices; --max-fails failures
in a row stop the loop. Needs the ROS 2 environment (rclpy) and robot_sim.py running.
"""
import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.spatial import ConvexHull
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, Empty, String

SIM_DIR = Path(__file__).resolve().parent.parent                       # Simulation/
GDRNPP = Path.home() / 'project' / 'gdrnpp_bin_6DOF_estimation'
MODELS = Path.home() / 'IsaacSim' / 'captures' / 'side_plate_10k' / 'shard_00' / 'models'    # CAD models copied by --to-bop (the part of the pose network)
OUT_DIR = GDRNPP / 'runs' / 'vision_pick'                              # the pictures and the poses of every capture
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]    # of the 8 box corners (the order of bin6d/viz.py)
TOPIC_CAPTURE, TOPIC_TARGET = '/wrist_camera/capture', '/target_pose'
TOPIC_PART, TOPIC_DONE, TOPIC_TRANSFER = '/part_turned_over', '/sequence_done', '/transfer_done'
TOPIC_PICTURE = '/vision_pick/picture'                                 # std_msgs/String: the path of every new picture; robot_sim.py shows it in a window of Isaac Sim
TIMEOUT = {'capture': 180.0, 'pipeline': 1200.0, 'sequence': 1800.0, 'transfer': 1800.0}   # s: frame written by the simulation, convert + infer, one pick and place


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


def box_corners(models):
    """8 corners (mm, model frame) of the CAD box, in the order of gdrnpp's bin6d/viz.py: from models_info.json of the models folder"""
    m = next(iter(json.load(open(Path(models) / 'models_info.json')).values()))
    lo, size = np.array([m['min_x'], m['min_y'], m['min_z']]), np.array([m['size_x'], m['size_y'], m['size_z']])
    return np.array([lo + size * np.array(c) for c in itertools.product((0, 1), repeat=3)])[[0, 2, 4, 6, 1, 3, 5, 7]]


def project(K, pts):
    uv = pts @ K.T
    return uv[:, :2] / uv[:, 2:3]


def blur_parts(rgb, poses, K, corners, radius):
    """The parts are the product of a customer and must not be recognisable in a picture that is shown: the image inside the box of every detected part (the hull of its
    projected box, a few pixels wider) is blurred with a Gaussian of `radius` pixels, the edge of the blur is soft. radius 0 = nothing is blurred"""
    img = rgb.convert('RGB')
    if radius <= 0 or not poses:
        return img
    mask = Image.new('L', img.size, 0)
    md = ImageDraw.Draw(mask)
    for p in poses:
        uv = project(K, corners @ np.array(p['R']).T + np.array(p['t']))
        try:
            hull = uv[ConvexHull(uv).vertices]
        except Exception:  # noqa: BLE001  (degenerate box)
            continue
        md.polygon([tuple(v) for v in hull], fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(2 * int(max(3, radius * 0.5)) + 1)).filter(ImageFilter.GaussianBlur(max(2.0, radius * 0.4)))
    img.paste(img.filter(ImageFilter.GaussianBlur(radius)), mask=mask)
    return img


def draw_ranking(rgb, poses, ranked, K, corners, title, top_n=5, blur=9.0):
    """The picture of the frame with every detected part (a thin box, its label number: green = confident, orange = not) and the score ranking on it: the best part (rank 1,
    the one that is picked) in thick magenta with its axes, the next top_n - 1 in cyan, every one with `#rank score` under its box; a panel with the ranking and the target of
    the pick. The parts themselves are blurred (blur_parts, `blur` pixels). poses: all parts, ranked: the confident ones, best first"""
    img = blur_parts(rgb, poses, K, corners, blur)
    img = Image.blend(img, Image.new('RGB', img.size, (0, 0, 0)), 0.3)           # darkened, so that the drawing stands out
    d = ImageDraw.Draw(img)
    font, big = ImageFont.load_default(size=16), ImageFont.load_default(size=26)
    for p in poses:                                                              # all parts: a thin box and the label number
        uv = project(K, corners @ np.array(p['R']).T + np.array(p['t']))
        color = (40, 220, 90) if p.get('confident') else (255, 165, 0)
        for a, b in EDGES:
            d.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=2)
        x, y = uv.min(0)
        d.text((x, max(y - 24, 0)), str(p['label']), fill=(255, 235, 0), font=ImageFont.load_default(size=22), stroke_width=3, stroke_fill=(0, 0, 0))
    for rank, p in reversed(list(enumerate(ranked[:top_n], 1))):                 # the best one is drawn last: on top
        R, t = np.array(p['R']), np.array(p['t'])
        uv = project(K, corners @ R.T + t)
        color, width = ((255, 0, 255), 5) if rank == 1 else ((0, 220, 255), 3)
        for a, b in EDGES:
            d.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=width)
        o = project(K, t[None])[0]
        if rank == 1:
            for ax, c in enumerate(((255, 60, 60), (60, 255, 60), (70, 140, 255))):    # x red, y green, z blue
                e = project(K, (t + R[:, ax] * 30.0)[None])[0]
                d.line([tuple(o), tuple(e)], fill=c, width=4)
            d.ellipse([o[0] - 6, o[1] - 6, o[0] + 6, o[1] + 6], outline=(255, 255, 255), width=2)
        x, y = uv.min(0)[0], uv.max(0)[1]                                        # under the box (the label number of infer.py is above it)
        d.text((x, min(y + 4, img.height - 34)), f"#{rank}  {p['score']:.6f}", fill=color, font=big if rank == 1 else font, stroke_width=3, stroke_fill=(0, 0, 0))
    lines = [title]
    if ranked:
        b = ranked[0]
        lines += [f"PICK  part {b['label']}  score {b['score']:.6f}  {b['state']}", f"target  {b['ee_6d']}"]
        lines += [f"{r}. part {p['label']:>3}  {p['score']:.6f}  {p['state']}" for r, p in enumerate(ranked[:top_n], 1)]
    else:
        lines.append('no confident part')
    h = 24 * len(lines) + 12
    panel = Image.new('RGBA', (520, h), (0, 0, 0, 200))
    img.paste(panel, (8, img.height - h - 8), panel)
    for i, ln in enumerate(lines):
        d.text((16, img.height - h + 2 + 24 * i), ln, fill=(255, 0, 255) if i == 1 else (255, 255, 255), font=font)
    return img


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
        self.picture_pub = self.create_publisher(String, TOPIC_PICTURE, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, TOPIC_DONE, self.cb_done, 10)
        self.transfer_done = False                                      # /transfer_done (std_msgs/Empty): the plate lies on the belt and the arm is at home again = the last step of a pick
        self.create_subscription(Empty, TOPIC_TRANSFER, lambda _: setattr(self, 'transfer_done', True), 10)

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
        """convert + infer for the one frame; returns (the poses of its pose6d.json, the picture infer.py made of them, K of the camera)"""
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
        overlay = Image.open(sub_cam / 'rgb' / '000000.png').convert('RGB')                  # the camera image itself: the picture is drawn on it (and blurred) here
        overlay.load()
        cam_json = json.load(open(sub_cam / 'camera.json'))
        K = np.array([[cam_json['fx'], 0.0, cam_json['cx']], [0.0, cam_json['fy'], cam_json['cy']], [0.0, 0.0, 1.0]])
        return json.load(open(files[-1]))['poses'], overlay, K, json.load(open(files[-1]))

    def pick_once(self, args):
        """one capture + estimate + target. Returns False when there is nothing (confident) to pick"""
        cam = args.camera_dir
        frame = args.frame or self.capture(cam)
        self.get_logger().info(f'frame {frame} ' + ('(already there)' if args.frame else 'captured'))
        with tempfile.TemporaryDirectory(prefix='vision_pick_') as tmp:
            poses, overlay, K, raw = self.estimate(cam, frame, args, Path(tmp))
        good = sorted((p for p in poses if p.get('confident')), key=lambda p: -p['score'])
        left = [p for p in good if all(math.hypot(p['xyz_m'][0] - x, p['xyz_m'][1] - y) > args.skip_radius for x, y in self.failed)]
        if len(left) < len(good):
            self.get_logger().info(f'{len(good) - len(left)} confident parts left out: near a pick that failed')
        good = left
        args.out.mkdir(parents=True, exist_ok=True)                    # the picture with the ranking, and the poses it is made of
        picture = args.out / f'{frame}_pick.png'
        draw_ranking(overlay, poses, good, K, box_corners(args.models), f'frame {frame}: {len(poses)} parts, {len(good)} confident', blur=args.blur).save(picture)
        json.dump(raw, open(args.out / f'{frame}_pose6d.json', 'w'), indent=1)
        self.get_logger().info(f'picture: {picture}')
        self.picture_pub.publish(String(data=str(picture)))              # shown in Isaac Sim (robot_sim.py)
        self.get_logger().info(f'{len(poses)} parts found, {len(good)} confident; highest scores: ' +
                               ', '.join(f"{p['score']:.6f} ({p['state']})" for p in good[:3]))
        if not good:
            self.get_logger().warn('no confident part: nothing to pick')
            return False
        best = good[0]
        pose, state = pose_from_line(best['ee_6d'])
        self.last_xy = (best['xyz_m'][0], best['xyz_m'][1])
        self.get_logger().info(f"pick: {best['ee_6d']}  (score {best['score']:.6f})")
        if args.dry_run:
            return True
        self.wait_for(self.target_pub, TOPIC_TARGET)
        self.done = None
        self.transfer_done = False
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
    ap.add_argument('--out', type=Path, default=OUT_DIR, help='folder for <frame>_pick.png (all parts with the score ranking, the best one marked) and <frame>_pose6d.json')
    ap.add_argument('--blur', type=float, default=0.0, help='pixels: blur the detected parts in the picture (e.g. 9: they are the product of a customer); 0 = not blurred (default)')
    ap.add_argument('--frame', help='use this frame of the camera folder (e.g. 000012) instead of capturing a new one, for looking at an earlier capture')
    ap.add_argument('--dry-run', action='store_true', help='only print the part that would be picked, send no target')
    ap.add_argument('--skip-radius', type=float, default=0.05, help='m: parts this close to a failed pick are not tried again (default 0.05)')
    ap.add_argument('--max-fails', type=int, default=3, help='--loop: stop after this many failed picks in a row (default 3)')
    ap.add_argument('--loop', action='store_true', help='after every pick wait for /sequence_done and then for /transfer_done (the plate was put on the conveyor and the arm is at home again) and capture again, until nothing confident is left')
    ap.add_argument('--no-transfer', action='store_true', help='--loop: do not wait for /transfer_done (a sequence without a transfer of the plate to the conveyor)')
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
                if not args.no_transfer:                                # the last step: the robot picks the plate up again at the standard coordinate and puts it on the conveyor, then it is at home again
                    node.done, node.transfer_done = None, False         # (a transfer that fails ends with /sequence_done `failed` after the way back home)
                    node.get_logger().info('waiting for the transfer of the plate to the conveyor (/transfer_done): the next picture is taken when the arm is at home again')
                    end = time.time() + TIMEOUT['transfer']
                    while not node.transfer_done and node.done is None and time.time() < end:
                        node.spin_for(0.5)
                    if node.done == 'failed':
                        fails += 1
                        node.get_logger().warn(f'the transfer failed ({fails}/{args.max_fails}): the next capture follows')
                        if fails >= args.max_fails:
                            node.get_logger().error('too many failed picks in a row: stopping')
                            break
                    elif not node.transfer_done:
                        node.get_logger().error('the robot did not report /transfer_done: stopping (--no-transfer if this sequence has no transfer)')
                        break
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
