"""vision_pick.py: wrist camera picture -> 6D poses (gdrnpp) -> the best part (pick_filter.py) -> /target_pose for robot_sequence.py, crate by crate.

    python3 control/vision_pick.py [model:=side_plate|pump_cover] [--dry-run] [--loop] [--robot <name>] [--map <map>]

model:= (or --model) = the part of isaacsim/configs/object/<part>.yaml: its `vision:` gives the detector, the pose net (paths in the gdrnpp folder) and the CAD models folder;
default = the part of the pallets of the map.
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

import cv2
import numpy as np
import rclpy
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.spatial import ConvexHull
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, Empty, Int32, String

SIM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM_DIR / 'isaacsim'))
import compute_grasp_tcp_pose as grasp_tcp
import sim_config
from pick_filter import mask_outside, rank, roi_polygon
GDRNPP = Path.home() / 'project' / 'gdrnpp_bin_6DOF_estimation'
OUT_DIR = GDRNPP / 'runs' / 'vision_pick'
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
TOPIC_CAPTURE, TOPIC_TARGET = '/wrist_camera/capture', '/target_pose'
TOPIC_PART, TOPIC_DONE, TOPIC_TRANSFER = '/part_turned_over', '/sequence_done', '/transfer_done'
TOPIC_PICTURE = '/vision_pick/picture'
TIMEOUT = {'capture': 180.0, 'pipeline': 1200.0, 'sequence': 1800.0, 'transfer': 1800.0}


def vision_of(model=None, map_name=None, robot=None):
    try:
        robot, map_name = sim_config.with_current(robot, map_name)
        scene = sim_config.scene_of(map_name)
    except ValueError:
        scene = {}
    part = model or (scene.get('pallets') or {}).get('part') or 'side_plate'
    path = Path(sim_config.OBJECTS_DIR) / f'{part}.yaml'
    if not path.is_file():
        raise SystemExit(f'model:={part}: {path} not found (parts: {", ".join(sorted(p.stem for p in Path(sim_config.OBJECTS_DIR).glob("*.yaml")))})')
    vision = sim_config._load_yaml_tree(str(path)).get('vision')
    if not vision or not all(vision.get(k) for k in ('detector', 'pose', 'models')):
        raise SystemExit(f'{path}: no `vision:` with detector, pose and models (the trained models of this part)')
    return part, vision


def grip_axis(model=None, map_name=None):
    part = model or (sim_config.scene_of(map_name).get('pallets') or {}).get('part') or 'side_plate'
    cfg = sim_config._load_yaml_tree(str(Path(sim_config.OBJECTS_DIR) / f'{part}.yaml'))
    origin = np.array(cfg.get('axis_origin_mm') or [0.0, 0.0, 0.0], float) / 1000.0
    from binsim.transforms import rpy_to_mat
    return rpy_to_mat(cfg.get('axis_rpy_deg') or [0.0, 0.0, 0.0]), origin


def apply_axis(poses, T_world_cam, axis):
    R_axis, origin = axis
    for p in poses:
        R_cam, t_cam = np.array(p['R']), np.array(p['t'])
        p['grip_R'] = (R_cam @ R_axis).tolist()
        p['grip_t'] = (t_cam + R_cam @ (origin * 1000.0)).tolist()
        R = T_world_cam[:3, :3] @ np.array(p['grip_R'])
        xyz = T_world_cam[:3, :3] @ (np.array(p['grip_t']) / 1000.0) + T_world_cam[:3, 3]
        flipped = bool(R[2, 2] < 0)
        tcp = grasp_tcp.R_to_rpy(R @ (grasp_tcp.FLIPPED_TCP_R if flipped else grasp_tcp.DEFAULT_TCP_R))
        state = 'turn_over' if flipped else 'standard'
        p['xyz_m'], p['rpy_deg'], p['state'] = xyz.tolist(), grasp_tcp.R_to_rpy(R).tolist(), state
        p['world'] = ' '.join([f'{v:.3f}' for v in xyz] + [f'{a + 0.0:.1f}' for a in p['rpy_deg']])
        p['ee_6d'] = ' '.join([f'{v:.3f}' for v in xyz] + [f'{a + 0.0:.1f}' for a in tcp]) + ' ' + state
    return poses


def camera_dir(robot=None, map_name=None):
    try:
        robot, map_name = sim_config.with_current(robot, map_name)
        name = sim_config.pick_robot(robot, sim_config.scene_of(map_name))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    cfg = sim_config.load_robot(name)
    out = (cfg.get('wrist_camera') or {}).get('output_dir')
    if not out:
        raise SystemExit(f'robot {name}: no wrist_camera.output_dir in its yaml')
    return Path(os.path.expanduser(str(out)))


def has_transfer(robot=None, map_name=None):
    try:
        robot, map_name = sim_config.with_current(robot, map_name)
        scene = sim_config.scene_of(map_name)
        name = sim_config.pick_robot(robot, scene)
    except ValueError:
        return True
    path = Path(sim_config.SEQUENCE_DIR) / f"{scene.get('sequence')}.yaml"
    if not scene.get('sequence') or not path.is_file():
        return True
    cfg = sim_config._load_yaml_tree(str(path))
    return bool(((cfg.get('robots') or {}).get(name) or {}).get('transfer') or cfg.get('transfer'))


def crate_of(robot=None, map_name=None):
    try:
        robot, map_name = sim_config.with_current(robot, map_name)
        scene = sim_config.scene_of(map_name)
        name = sim_config.pick_robot(robot, scene)
    except ValueError:
        return None, []
    path = Path(sim_config.SEQUENCE_DIR) / f"{scene.get('sequence')}.yaml"
    if not scene.get('sequence') or not path.is_file():
        return None, []
    cfg = sim_config._load_yaml_tree(str(path))
    mine = (cfg.get('robots') or {}).get(name) or {}
    crate = mine.get('crate') or cfg.get('crate')
    if (crate or {}).get('from_map'):
        return crate, sim_config.crates_from_map(scene, crate)
    crates = (mine.get('poses') or {}).get('crates') or (cfg.get('poses') or {}).get('crates') or mine.get('crates') or cfg.get('crates') or []
    return crate, sim_config.expand_crates(crate, crates)


def frames(folder):
    rgb = folder / 'rgb'
    return {p.stem for p in rgb.glob('*.png')} if rgb.is_dir() else set()


def pose_from_line(line):
    v = line.split()
    x, y, z, roll, pitch, yaw = (float(a) for a in v[:6])
    q = Rotation.from_euler('XYZ', [roll, pitch, yaw], degrees=True).as_quat()
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = x, y, z
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (float(a) for a in q)
    return msg, (v[6] if len(v) > 6 else 'standard')


def box_corners(models):
    m = next(iter(json.load(open(Path(models) / 'models_info.json')).values()))
    lo, size = np.array([m['min_x'], m['min_y'], m['min_z']]), np.array([m['size_x'], m['size_y'], m['size_z']])
    return np.array([lo + size * np.array(c) for c in itertools.product((0, 1), repeat=3)])[[0, 2, 4, 6, 1, 3, 5, 7]]


def project(K, pts):
    uv = pts @ K.T
    return uv[:, :2] / uv[:, 2:3]


def blur_parts(rgb, poses, K, corners, radius):
    img = rgb.convert('RGB')
    if radius <= 0 or not poses:
        return img
    mask = Image.new('L', img.size, 0)
    md = ImageDraw.Draw(mask)
    for p in poses:
        uv = project(K, corners @ np.array(p['R']).T + np.array(p['t']))
        try:
            hull = uv[ConvexHull(uv).vertices]
        except Exception:
            continue
        md.polygon([tuple(v) for v in hull], fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(2 * int(max(3, radius * 0.5)) + 1)).filter(ImageFilter.GaussianBlur(max(2.0, radius * 0.4)))
    img.paste(img.filter(ImageFilter.GaussianBlur(radius)), mask=mask)
    return img


def draw_ranking(rgb, poses, ranked, K, corners, title, top_n=5, blur=9.0):
    img = blur_parts(rgb, poses, K, corners, blur)
    img = Image.blend(img, Image.new('RGB', img.size, (0, 0, 0)), 0.3)
    d = ImageDraw.Draw(img)
    font, big = ImageFont.load_default(size=16), ImageFont.load_default(size=26)
    for p in poses:
        uv = project(K, corners @ np.array(p['R']).T + np.array(p['t']))
        color = (40, 220, 90) if p.get('confident') else (255, 165, 0)
        for a, b in EDGES:
            d.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=2)
        x, y = uv.min(0)
        d.text((x, max(y - 24, 0)), str(p['label']), fill=(255, 235, 0), font=ImageFont.load_default(size=22), stroke_width=3, stroke_fill=(0, 0, 0))
    for rank, p in reversed(list(enumerate(ranked[:top_n], 1))):
        R, t = np.array(p['R']), np.array(p['t'])
        uv = project(K, corners @ R.T + t)
        color, width = ((255, 0, 255), 5) if rank == 1 else ((0, 220, 255), 3)
        for a, b in EDGES:
            d.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=width)
        Rg, tg = np.array(p.get('grip_R', R)), np.array(p.get('grip_t', t))
        o = project(K, tg[None])[0]
        if rank == 1:
            for ax, c in enumerate(((255, 60, 60), (60, 255, 60), (70, 140, 255))):
                e = project(K, (tg + Rg[:, ax] * 30.0)[None])[0]
                d.line([tuple(o), tuple(e)], fill=c, width=4)
            d.ellipse([o[0] - 6, o[1] - 6, o[0] + 6, o[1] + 6], outline=(255, 255, 255), width=2)
        x, y = uv.min(0)[0], uv.max(0)[1]
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
    def __init__(self, crate=None, crates=(), axis=None):
        super().__init__('vision_pick')
        self.axis = axis or (np.eye(3), np.zeros(3))
        self.crates, self.idx = list(crates), 0
        self.crate = {**crate, 'centre': self.crates[0]['centre'], 'floor_z': self.crates[0]['floor_z']} if crate and self.crates else crate
        self.empty = False
        self.done = None
        self.last_xy = None
        self.failed = []
        self.capture_pub = self.create_publisher(Empty, TOPIC_CAPTURE, 10)
        self.target_pub = self.create_publisher(Pose, TOPIC_TARGET, 10)
        self.part_pub = self.create_publisher(Bool, TOPIC_PART, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.conveyor_pub = self.create_publisher(Int32, crate['conveyor_topic'], 10) if crate and crate.get('conveyor_topic') else None
        self.picture_pub = self.create_publisher(String, TOPIC_PICTURE, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, TOPIC_DONE, self.cb_done, 10)
        self.transfer_done = False
        self.create_subscription(Empty, TOPIC_TRANSFER, lambda _: setattr(self, 'transfer_done', True), 10)
        self.clear_topic = (crate or {}).get('clear_topic')
        self.clear_pub = self.create_publisher(String, self.clear_topic, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                                                   durability=DurabilityPolicy.TRANSIENT_LOCAL)) if self.clear_topic else None
        nxt, ready = (crate or {}).get('next_topic'), (crate or {}).get('ready_topic')
        self.next_pub = self.create_publisher(Int32, nxt, 10) if nxt and len(self.crates) > 1 else None
        active = (crate or {}).get('active_topic')
        self.active_pub = self.create_publisher(String, active, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                                           durability=DurabilityPolicy.TRANSIENT_LOCAL)) if active and self.crate and 'centre' in self.crate else None
        self.crate_ready = None
        if self.next_pub is not None and ready:
            self.create_subscription(Int32, ready, lambda m: setattr(self, 'crate_ready', int(m.data)), 10)

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
        before = frames(cam)
        self.wait_for(self.capture_pub, TOPIC_CAPTURE)
        self.capture_pub.publish(Empty())
        end = time.time() + TIMEOUT['capture']
        while time.time() < end:
            self.spin_for(0.5)
            new = sorted(frames(cam) - before)
            if new and all((cam / sub / f'{new[-1]}{ext}').is_file() for sub, ext in (('rgb', '.png'), ('depth', '.png'), ('pose', '.json'))):
                time.sleep(0.5)
                return new[-1]
        raise TimeoutError(f'no new frame in {cam} after {TIMEOUT["capture"]:.0f} s: is robot_sim.py running with a wrist_camera?')

    def estimate(self, cam, frame, args, work):
        sub_cam = work / 'camera'
        for sub, ext in (('rgb', '.png'), ('depth', '.png'), ('pose', '.json')):
            (sub_cam / sub).mkdir(parents=True)
            shutil.copy(cam / sub / f'{frame}{ext}', sub_cam / sub / f'000000{ext}')
        for name in ('camera.json', 'undistort_map.npy'):
            if (cam / name).is_file():
                shutil.copy(cam / name, sub_cam / name)
        cam_json = json.load(open(sub_cam / 'camera.json'))
        K = np.array([[cam_json['fx'], 0.0, cam_json['cx']], [0.0, cam_json['fy'], cam_json['cy']], [0.0, 0.0, 1.0]])
        dist = cam_json.get('dist') or [0.0] * 5
        T_world_cam = np.array(json.load(open(sub_cam / 'pose' / '000000.json'))['T_world_cam'])
        poly = None
        if self.crate:
            poly = roi_polygon(self.crate, T_world_cam, K, dist)
            rgb = cv2.imread(str(sub_cam / 'rgb' / '000000.png'))
            depth = cv2.imread(str(sub_cam / 'depth' / '000000.png'), cv2.IMREAD_UNCHANGED)
            rgb, depth = mask_outside(rgb, depth, roi_polygon(self.crate, T_world_cam, K, dist, margin=0.0))
            cv2.imwrite(str(sub_cam / 'rgb' / '000000.png'), rgb)
            cv2.imwrite(str(sub_cam / 'depth' / '000000.png'), depth)
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH')}
        python = str(args.gdrnpp / '.venv' / 'bin' / 'python')
        steps = [
            [python, 'camera_convert_data.py', '--out', str(sub_cam), '--to-bop', str(work / 'bop'), '--keep-distortion', '--models', str(args.models)],
            [python, 'infer.py', '--detector', str(args.detector), '--pose', str(args.pose), '--set', f'data.root={work / "bop"}', '--scenes', '000000', '--view-stride', '1', '--out', str(work / 'run')],
        ]
        for cmd in steps:
            self.get_logger().info(f'running {Path(cmd[1]).name} ...')
            r = subprocess.run(cmd, cwd=args.gdrnpp, env=env, capture_output=True, text=True, timeout=TIMEOUT['pipeline'])
            if r.returncode:
                raise RuntimeError(f'{Path(cmd[1]).name} failed ({r.returncode}):\n' + '\n'.join((r.stdout + r.stderr).strip().splitlines()[-15:]))
        files = sorted((work / 'run').glob('*_pose6d.json'))
        overlay = Image.open(sub_cam / 'rgb' / '000000.png').convert('RGB')
        overlay.load()
        raw = json.load(open(files[-1])) if files else {'frame': '000000', 'poses': []}
        return raw['poses'], overlay, K, raw, T_world_cam, dist, poly

    def activate(self):
        if self.active_pub is not None:
            self.active_pub.publish(String(data=json.dumps({'index': self.idx, 'centre': list(self.crate['centre']), 'floor_z': float(self.crate['floor_z'])})))

    def first_crate(self):
        if self.next_pub is None:
            return True
        self.wait_for(self.next_pub, 'the next crate topic')
        if self.next_pub.get_subscription_count() == 0:
            return True
        self.crate_ready = None
        self.next_pub.publish(Int32(data=0))
        self.get_logger().info(f"crate 0 (centre {self.crates[0]['centre']}, layer {self.crates[0]['level']} from the top): the arm goes to its scan pose")
        end = time.time() + TIMEOUT['sequence']
        while self.crate_ready is None and time.time() < end:
            self.spin_for(0.5)
        if self.crate_ready != 0:
            self.get_logger().error(f'the arm did not reach the scan pose of crate 0 ({"no answer" if self.crate_ready is None else "it failed"}): stopping')
            return False
        time.sleep(float(self.crate.get('settle_s', 1.0)))
        return True

    def next_crate(self):
        if self.next_pub is None or self.idx + 1 >= len(self.crates):
            self.get_logger().info('the last crate (of the bottom layer) is empty: done' if self.crates else 'done')
            if self.next_pub is not None and self.crates:
                self.crate_ready = None
                self.wait_for(self.next_pub, 'the next crate topic')
                self.next_pub.publish(Int32(data=-1))
                self.get_logger().info('all crates are empty: the arm goes back to the home pose')
                end = time.time() + TIMEOUT['sequence']
                while self.crate_ready is None and time.time() < end:
                    self.spin_for(0.5)
            return False
        i = self.idx + 1
        self.crate_ready = None
        self.wait_for(self.next_pub, 'the next crate topic')
        self.next_pub.publish(Int32(data=i))
        new_layer = self.crates[i]['level'] != self.crates[self.idx]['level']
        self.idx, self.crate = i, {**self.crate, 'centre': self.crates[i]['centre'], 'floor_z': self.crates[i]['floor_z']}
        self.activate()
        self.get_logger().info(f"next crate {i} (centre {self.crates[i]['centre']}, layer {self.crates[i]['level']} from the top, floor z {self.crates[i]['floor_z']:.4f})"
                               + (": the layer above is done, it is taken away" if new_layer else "") + ": the arm goes to its scan pose")
        end = time.time() + TIMEOUT['sequence']
        while self.crate_ready is None and time.time() < end:
            self.spin_for(0.5)
        if self.crate_ready != i:
            self.get_logger().error(f'the arm did not reach the scan pose of crate {i} ({"no answer" if self.crate_ready is None else "it failed"}): stopping')
            return False
        self.failed = []
        time.sleep(float(self.crate.get('settle_s', 1.0)))
        return True

    def clear_crate(self, args):
        if self.clear_pub is not None and not args.dry_run:
            self.clear_pub.publish(String(data=json.dumps({'index': self.idx, **{k: v for k, v in self.crate.items() if not k.endswith('_topic')}})))
            self.spin_for(1.0)
            self.get_logger().info(f'crate clear -> {self.clear_topic}')

    def pick_once(self, args):
        cam = args.camera_dir
        frame = args.frame or self.capture(cam)
        self.get_logger().info(f'frame {frame} ' + ('(already there)' if args.frame else 'captured'))
        with tempfile.TemporaryDirectory(prefix='vision_pick_') as tmp:
            poses, overlay, K, raw, T_world_cam, dist, poly = self.estimate(cam, frame, args, Path(tmp))
        apply_axis(poses, T_world_cam, self.axis)
        good = rank(poses, T_world_cam, K, dist, poly, self.crate)
        left = [p for p in good if all(math.hypot(p['xyz_m'][0] - x, p['xyz_m'][1] - y) > args.skip_radius for x, y in self.failed)]
        if len(left) < len(good):
            self.get_logger().info(f'{len(good) - len(left)} confident parts left out: near a pick that failed')
        good = left
        args.out.mkdir(parents=True, exist_ok=True)
        picture = args.out / f'{frame}_pick.png'
        if poly is not None:
            ImageDraw.Draw(overlay).polygon([tuple(int(v) for v in q) for q in poly], outline=(255, 255, 0), width=3)
        draw_ranking(overlay, poses, good, K, box_corners(args.models), f'frame {frame}: {len(poses)} parts, {len(good)} confident', blur=args.blur).save(picture)
        json.dump(raw, open(args.out / f'{frame}_pose6d.json', 'w'), indent=1)
        self.get_logger().info(f'picture: {picture}')
        self.picture_pub.publish(String(data=str(picture)))
        self.get_logger().info(f'{len(poses)} parts found, {len(good)} confident; highest scores: ' +
                               ', '.join(f"{p['score']:.6f} ({p['state']})" for p in good[:3]))
        self.empty = not poses
        if not poses:
            self.get_logger().info(f'no part detected: crate {self.idx} is empty')
            self.clear_crate(args)
            return False
        if not good:
            self.get_logger().warn(f'no confident part: {len(poses)} part(s) are left that cannot be picked (low score, outside the crate floor or near a failed pick): crate {self.idx} is taken as done')
            self.empty = True
            self.clear_crate(args)
            return False
        best = good[0]
        pose, state = pose_from_line(best['ee_6d'])
        self.last_xy = (best['xyz_m'][0], best['xyz_m'][1])
        self.get_logger().info(f"pick: {best['ee_6d']}  (score {best['score']:.6f}, tilt {best.get('tilt_deg', 0.0):.1f} deg, rank {best.get('rank_score', 0.0):.2f})")
        if args.dry_run:
            return True
        self.wait_for(self.target_pub, TOPIC_TARGET)
        self.done = None
        self.transfer_done = False
        self.part_pub.publish(Bool(data=(state == 'turn_over')))
        self.spin_for(0.5)
        self.target_pub.publish(pose)
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--robot', help='robot name (default: the robot of the running robot_sim.py): its wrist_camera.output_dir is where the frames are')
    ap.add_argument('--map', help='map name = isaacsim/configs/map/<map>.yaml: its robot (scene.robot) when --robot is not given')
    ap.add_argument('--camera-dir', type=Path, help='folder of the captured frames (default: wrist_camera.output_dir of the robot yaml)')
    ap.add_argument('--gdrnpp', type=Path, default=GDRNPP, help='the gdrnpp_bin_6DOF_estimation folder (with its .venv)')
    ap.add_argument('--model', help='part whose trained models are used (model:=<part> does the same): isaacsim/configs/object/<part>.yaml `vision:` (default: the part of the pallets of the map)')
    ap.add_argument('--detector', help='detector weights (default: vision.detector of the part), relative to the gdrnpp folder')
    ap.add_argument('--pose', help='pose net weights (default: vision.pose of the part), relative to the gdrnpp folder')
    ap.add_argument('--models', type=Path, help='CAD models folder given to camera_convert_data.py --to-bop (default: vision.models of the part)')
    ap.add_argument('--out', type=Path, default=OUT_DIR, help='folder for <frame>_pick.png (all parts with the score ranking, the best one marked) and <frame>_pose6d.json')
    ap.add_argument('--blur', type=float, default=0.0, help='pixels: blur the detected parts in the picture (e.g. 9: they are the product of a customer); 0 = not blurred (default)')
    ap.add_argument('--frame', help='use this frame of the camera folder (e.g. 000012) instead of capturing a new one, for looking at an earlier capture')
    ap.add_argument('--dry-run', action='store_true', help='only print the part that would be picked, send no target')
    ap.add_argument('--skip-radius', type=float, default=0.05, help='m: parts this close to a failed pick are not tried again (default 0.05)')
    ap.add_argument('--max-fails', type=int, default=3, help='--loop: stop after this many failed picks in a row (default 3)')
    ap.add_argument('--loop', action='store_true', help='after every pick wait for /sequence_done and then for /transfer_done (the plate was put on the conveyor and the arm is at home again) and capture again, until nothing confident is left')
    ap.add_argument('--no-transfer', action='store_true', help='--loop: do not wait for /transfer_done (a sequence without a transfer of the plate to the conveyor)')
    args, ros_args = ap.parse_known_args()
    short = sim_config.option(ros_args, 'model')
    ros_args = [a for a in ros_args if not a.startswith('model:=')]
    args.model = args.model or short
    part, vision = vision_of(args.model, args.map, args.robot)
    args.detector = args.detector or vision['detector']
    args.pose = args.pose or vision['pose']
    args.models = args.models or Path(os.path.expanduser(str(vision['models'])))
    for weights in (args.detector, args.pose):
        if not (args.gdrnpp / weights).is_file():
            raise SystemExit(f'model {part}: {args.gdrnpp / weights} not found')
    print(f'[INFO] model {part}: detector {args.detector}, pose {args.pose}, CAD models {args.models}', flush=True)
    args.camera_dir = args.camera_dir or camera_dir(args.robot, args.map)
    crate, crates = crate_of(args.robot, args.map)
    args.no_transfer = args.no_transfer or not has_transfer(args.robot, args.map)
    rclpy.init(args=ros_args)
    node = VisionPick(crate, crates, grip_axis(part, args.map))
    node.get_logger().info(f'crate (ROI): {node.crate}' + (f', {len(crates)} crates in turn' if crates else '') if crate else 'no `crate:` in the sequence yaml: the whole image is used')
    node.activate()
    try:
        if crates and not args.frame and not node.first_crate():
            sys.exit(1)
        fails = 0
        while True:
            if not node.pick_once(args):
                if node.empty and args.loop and not args.dry_run and node.next_crate():
                    continue
                break
            if args.dry_run or not args.loop:
                break
            end = time.time() + TIMEOUT['sequence']
            while node.done is None and time.time() < end:
                node.spin_for(0.5)
            if node.done is None:
                node.get_logger().error('the robot did not report /sequence_done: stopping')
                break
            if node.done in ('failed', 'unreachable'):
                node.failed.append(node.last_xy)
                fails += 1
                node.get_logger().warn(f'the pick failed ({fails}/{args.max_fails}): this part is left out, the next capture follows')
                if fails >= args.max_fails:
                    node.get_logger().error('too many failed picks in a row: stopping')
                    break
            else:
                fails = 0
                if node.conveyor_pub:
                    node.conveyor_pub.publish(Int32(data=1))
                    node.get_logger().info(f'placed: the conveyor moves one pitch ({node.crate["conveyor_topic"]})')
                if not args.no_transfer:
                    node.done, node.transfer_done = None, False
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
