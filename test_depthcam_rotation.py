"""Isaac Sim 6.0.0 — Phoxi camera multi-pose LINEMOD/FFB6D dataset capture (no robot).

Camera holds a random angle for SHOTS_PER_POSE frames, then jumps to a new one.
Per frame, saves:
  rgb/xxxxxx_rgb.png     : RGB image
  depth/xxxxxx_depth.png : depth, 16-bit PNG in mm
  mask/xxxxxx_<gear>.png : per-gear binary mask (0/255)
  gt/xxxxxx.json         : each visible gear's 6D pose in camera frame (R 3x3, t mm, OpenCV convention)
cam_K.json (camera intrinsics) is saved once.

Check after the first run:
  1) instance_segmentation's info dict key names vary across Isaac Sim versions.
     Raw info is printed on the first frame — fix extract_id_to_label() if masks/gt look wrong.
  2) Verify gt.json's R,t by projecting a 3D bbox onto the image before FFB6D training —
     one wrong axis in the USD->OpenCV convention silently breaks everything.
"""
import argparse
import json
import os
import random
import time

import numpy as np

from isaaclab.app import AppLauncher

HOME = os.path.expanduser("~")

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.experience = f"{HOME}/IsaacSim/_build/linux-aarch64/release/apps/isaacsim.exp.full.kit"
args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
import omni.timeline
import omni.replicator.core as rep
from pxr import UsdGeom, Gf, Semantics
from isaacsim.core.experimental.prims import RigidPrim

MAP_USD    = f"{HOME}/IsaacSim/assets/f2x_lab_get_data.usd"
OUTPUT_DIR = f"{HOME}/IsaacSim/captures/phoxi_ffb6d_dataset"

GEAR_PATHS = [
    "/f2x_lab/f2x_gear_black",
    "/f2x_lab/f2x_gear_cyan",
    "/f2x_lab/f2x_gear_pink",
    "/f2x_lab/f2x_gear_yellow",
    "/f2x_lab/f2x_gear_blue",
    "/f2x_lab/f2x_gear_green",
]

BOX_PATH = "/f2x_lab/f2x_box"
GEAR_BOX_WALL_MARGIN_M = 0.03  # reset a gear once it gets this close to a box wall

RENDER_WIDTH  = 640
RENDER_HEIGHT = 480

ANGLE_RANGE_DEG = 10.0
SHOTS_PER_POSE  = 200  # frames captured before the camera moves to a new random angle
CAMERA_SETTLE_FRAMES = 1  # wait frames after a camera move, before capturing again

GEAR_DRIFT_LINEAR_MPS  = 0.15  # random XY linear velocity re-applied every shot
GEAR_DRIFT_ANGULAR_RPS = 3.0   # random angular velocity re-applied every shot
GEAR_REORIENT_EVERY    = 10    # force a fully random gear orientation every N shots
GEAR_BOX_CHECK_EVERY   = 50    # extra out-of-box check every N shots

_USD_TO_CV = np.diag([1.0, -1.0, -1.0])


def _matrix4d_to_numpy(m):
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def compute_camera_intrinsics(cam_prim, width, height):
    cam = UsdGeom.Camera(cam_prim)
    focal = cam.GetFocalLengthAttr().Get()
    h_ap = cam.GetHorizontalApertureAttr().Get()
    fx = width * focal / h_ap
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]}


def compute_pose_in_camera(xform_cache, obj_prim, cam_prim, debug=False):
    """obj_prim's pose relative to cam_prim, OpenCV convention (R 3x3, t mm)."""

    obj_world_raw = xform_cache.GetLocalToWorldTransform(obj_prim)
    cam_world_raw = xform_cache.GetLocalToWorldTransform(cam_prim)
    if debug:
        print(f"[DEBUG] obj_world translation (raw, stage units) = "
              f"({obj_world_raw[3][0]:.4f}, {obj_world_raw[3][1]:.4f}, {obj_world_raw[3][2]:.4f})", flush=True)
        print(f"[DEBUG] cam_world translation (raw, stage units) = "
              f"({cam_world_raw[3][0]:.4f}, {cam_world_raw[3][1]:.4f}, {cam_world_raw[3][2]:.4f})", flush=True)
    obj_world = _matrix4d_to_numpy(obj_world_raw.GetOrthonormalized())
    cam_world = _matrix4d_to_numpy(cam_world_raw.GetOrthonormalized())
    cam_world_inv = np.linalg.inv(cam_world)

    # Gf.Matrix4d is row-vector convention (p_row @ M) -> world->camera_local order.
    obj_in_cam_row = obj_world @ cam_world_inv

    R_row = obj_in_cam_row[:3, :3]
    t_row = obj_in_cam_row[3, :3]

    # row-vector rotation -> standard column-vector (R@p+t) convention: transpose.
    R_usdcam = R_row.T
    t_usdcam = t_row

    # USD camera-local -> OpenCV camera-local
    R_cv = _USD_TO_CV @ R_usdcam
    t_cv = _USD_TO_CV @ t_usdcam

    # USD stage is in meters; LINEMOD convention stores mm.
    t_cv_mm = t_cv * 1000.0
    return R_cv, t_cv_mm


def random_quaternions(n):
    """n uniformly-random rotation quaternions (Shoemake's method, wxyz order)."""
    u1 = np.random.uniform(0.0, 1.0, size=n)
    u2 = np.random.uniform(0.0, 1.0, size=n)
    u3 = np.random.uniform(0.0, 1.0, size=n)
    q = np.empty((n, 4), dtype=np.float64)
    q[:, 0] = np.sqrt(u1) * np.cos(2 * np.pi * u3)      # w
    q[:, 1] = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)  # x
    q[:, 2] = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)  # y
    q[:, 3] = np.sqrt(u1) * np.sin(2 * np.pi * u3)      # z
    return q


def extract_id_to_label(seg_info):
    """Pull {instance_id(int): label(str)} out of instance_segmentation's info dict.
    Key names vary across Isaac Sim versions, so a few candidates are tried."""
    id_to_labels = None
    for key in ("idToSemantics", "idToLabels"):
        if key in seg_info:
            id_to_labels = seg_info[key]
            break
    if id_to_labels is None:
        print(f"[WARN] no id->label mapping found in info dict. raw info: {seg_info}", flush=True)
        return {}

    result = {}
    for k, v in id_to_labels.items():
        label = v if isinstance(v, str) else v.get("class", v.get("semantic", str(v)))
        try:
            result[int(k)] = label
        except (TypeError, ValueError):
            result[k] = label
    return result


def main():
    omni.usd.get_context().open_stage(MAP_USD)
    for _ in range(5):
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()

    cam_path = None
    for prim in stage.Traverse():
        if prim.GetName() == "phoxi_camera":
            cam_path = str(prim.GetPath())
            break
    if cam_path is None:
        print("[ERROR] phoxi_camera prim not found!", flush=True)
        simulation_app.close()
        return
    print(f"[INFO] phoxi_camera: {cam_path}", flush=True)

    cam_prim = stage.GetPrimAtPath(cam_path)
    cam_xform = UsdGeom.Xformable(cam_prim)

    rotate_op = None
    op_kind = None
    base_rotation = None
    quat_ctor = Gf.Quatd
    for op in cam_xform.GetOrderedXformOps():
        t = op.GetOpType()
        if t == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op
            op_kind = "eulerXYZ"
            val = op.Get()
            base_rotation = Gf.Vec3d(val[0], val[1], val[2]) if val is not None else Gf.Vec3d(0, 0, 0)
            break
        elif t == UsdGeom.XformOp.TypeOrient:
            rotate_op = op
            op_kind = "quat"
            val = op.Get()
            if val is not None:
                quat_ctor = Gf.Quatf if isinstance(val, Gf.Quatf) else Gf.Quatd
                base_rotation = Gf.Rotation(Gf.Quatd(val))
            else:
                base_rotation = Gf.Rotation(Gf.Quatd(1, 0, 0, 0))
            break

    if rotate_op is None:
        print("[ERROR] phoxi_camera has no rotate xformOp!", flush=True)
        simulation_app.close()
        return

    gear_prims = {}
    gear_paths_valid = []
    for gear_path in GEAR_PATHS:
        gp = stage.GetPrimAtPath(gear_path)
        if not gp.IsValid():
            print(f"[WARN] {gear_path} prim missing — skipping", flush=True)
            continue
        label = gp.GetName()
        sem_api = Semantics.SemanticsAPI.Apply(gp, "Semantics")
        sem_api.CreateSemanticTypeAttr().Set("class")
        sem_api.CreateSemanticDataAttr().Set(label)
        gear_prims[label] = gp
        gear_paths_valid.append(gear_path)
    print(f"[INFO] gears with semantic labels: {list(gear_prims.keys())}", flush=True)

    gear_rigid = RigidPrim(gear_paths_valid)
    n_gears = len(gear_paths_valid)
    zero_vel = np.zeros((n_gears, 3), dtype=np.float64)

    # Drift never resets position, so it can random-walk a gear through a wall over
    # time. Remember the spawn pose as the safe fallback fix_box_escapes() restores.
    base_gear_pos, base_gear_quat = gear_rigid.get_world_poses()
    base_gear_pos = np.array(base_gear_pos, dtype=np.float64)
    base_gear_quat = np.array(base_gear_quat, dtype=np.float64)

    box_prim = stage.GetPrimAtPath(BOX_PATH)
    if not box_prim.IsValid():
        print(f"[WARN] {BOX_PATH} prim not found — box-escape correction disabled", flush=True)
        box_bounds = None
    else:
        bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_], useExtentsHint=True)
        box_range = bbox_cache.ComputeWorldBound(box_prim).ComputeAlignedRange()
        bmin, bmax = box_range.GetMin(), box_range.GetMax()
        box_bounds = (
            bmin[0] + GEAR_BOX_WALL_MARGIN_M, bmax[0] - GEAR_BOX_WALL_MARGIN_M,
            bmin[1] + GEAR_BOX_WALL_MARGIN_M, bmax[1] - GEAR_BOX_WALL_MARGIN_M,
        )
        print(f"[INFO] box-escape correction range — X=[{box_bounds[0]:.3f},{box_bounds[1]:.3f}] "
              f"Y=[{box_bounds[2]:.3f},{box_bounds[3]:.3f}]", flush=True)

    def reorient_gears():
        """Force a fully random orientation (position unchanged)."""
        pos, _ = gear_rigid.get_world_poses()
        pos = np.array(pos, dtype=np.float64)
        gear_rigid.set_world_poses(pos, random_quaternions(n_gears))

    def drift_gears():
        """Re-apply small random linear/angular velocity to every gear."""
        lin_vel = np.random.uniform(-GEAR_DRIFT_LINEAR_MPS, GEAR_DRIFT_LINEAR_MPS, size=(n_gears, 3))
        lin_vel[:, 2] = 0.0  # leave Z to gravity
        ang_vel = np.random.uniform(-GEAR_DRIFT_ANGULAR_RPS, GEAR_DRIFT_ANGULAR_RPS, size=(n_gears, 3))
        gear_rigid.set_velocities(lin_vel, ang_vel)

    def fix_box_escapes():
        """Teleport any gear that drifted outside the box back to its spawn pose."""
        if box_bounds is None:
            return
        x_lo, x_hi, y_lo, y_hi = box_bounds
        pos, quat = gear_rigid.get_world_poses()
        pos = np.array(pos, dtype=np.float64)
        quat = np.array(quat, dtype=np.float64)
        out = (pos[:, 0] < x_lo) | (pos[:, 0] > x_hi) | (pos[:, 1] < y_lo) | (pos[:, 1] > y_hi)
        if np.any(out):
            print(f"[WARN] {int(out.sum())} gear(s) escaped the box — resetting to spawn pose", flush=True)
            pos[out] = base_gear_pos[out]
            quat[out] = base_gear_quat[out]
            gear_rigid.set_world_poses(pos, quat)
            gear_rigid.set_velocities(zero_vel, zero_vel)

    rgb_dir = os.path.join(OUTPUT_DIR, "rgb")
    depth_dir = os.path.join(OUTPUT_DIR, "depth")
    mask_dir = os.path.join(OUTPUT_DIR, "mask")
    gt_dir = os.path.join(OUTPUT_DIR, "gt")
    for d in (rgb_dir, depth_dir, mask_dir, gt_dir):
        os.makedirs(d, exist_ok=True)

    render_product = rep.create.render_product(cam_path, (RENDER_WIDTH, RENDER_HEIGHT))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach(render_product)
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_annotator.attach(render_product)
    seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False})
    seg_annotator.attach(render_product)

    # Camera intrinsics only change if focal/aperture change, so save once.
    intrinsics = compute_camera_intrinsics(cam_prim, RENDER_WIDTH, RENDER_HEIGHT)
    with open(os.path.join(OUTPUT_DIR, "cam_K.json"), "w") as f:
        json.dump(intrinsics, f, indent=2)
    print(f"[INFO] camera K = {intrinsics['K']}  (paste into common.py:145)", flush=True)

    xform_cache = UsdGeom.XformCache()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        simulation_app.update()

    def rand_angles():
        return Gf.Vec3d(
            random.uniform(-ANGLE_RANGE_DEG, ANGLE_RANGE_DEG),
            random.uniform(-ANGLE_RANGE_DEG, ANGLE_RANGE_DEG),
            random.uniform(-ANGLE_RANGE_DEG, ANGLE_RANGE_DEG),
        )

    def apply_camera_angles(vec3d):
        roll, pitch, yaw = vec3d
        if op_kind == "eulerXYZ":
            new_val = base_rotation + Gf.Vec3d(roll, pitch, yaw)
            rotate_op.Set(Gf.Vec3f(new_val[0], new_val[1], new_val[2]))
        else:
            delta = (
                Gf.Rotation(Gf.Vec3d.XAxis(), roll)
                * Gf.Rotation(Gf.Vec3d.YAxis(), pitch)
                * Gf.Rotation(Gf.Vec3d.ZAxis(), yaw)
            )
            new_rotation = delta * base_rotation
            new_quat = new_rotation.GetQuat()
            rotate_op.Set(quat_ctor(new_quat.GetReal(), new_quat.GetImaginary()))
        return roll, pitch, yaw

    def randomize_camera():
        return apply_camera_angles(rand_angles())

    reorient_gears()
    drift_gears()
    roll, pitch, yaw = randomize_camera()
    settle_count = 0
    shot = 0
    first_capture = True
    rate_window_start = time.monotonic()
    rate_window_shots = 0

    print(f"[INFO] camera holds each pose (±{ANGLE_RANGE_DEG:.0f}°) for {SHOTS_PER_POSE} shots, "
          f"gears drift every shot. Close window or Ctrl+C to stop.", flush=True)
    while simulation_app.is_running():
        simulation_app.update()

        if settle_count < CAMERA_SETTLE_FRAMES:
            settle_count += 1
            continue

        from PIL import Image as PILImage

        rgb = rgb_annotator.get_data()
        depth = depth_annotator.get_data()
        seg = seg_annotator.get_data()

        tag = f"{shot:06d}"

        if rgb is not None and rgb.size > 0:
            PILImage.fromarray(rgb[:, :, :3], mode="RGB").save(
                os.path.join(rgb_dir, f"{tag}_rgb.png")
            )

        if depth is not None and depth.size > 0:
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            PILImage.fromarray(depth_mm).save(
                os.path.join(depth_dir, f"{tag}_depth.png")
            )

        seg_data = seg.get("data") if isinstance(seg, dict) else None
        seg_info = seg.get("info", {}) if isinstance(seg, dict) else {}
        id_to_label = extract_id_to_label(seg_info)
        if first_capture:
            print(f"[DEBUG] seg type={type(seg)} keys={list(seg.keys()) if isinstance(seg, dict) else 'N/A'}", flush=True)
            print(f"[DEBUG] instance_segmentation raw info: {seg_info}", flush=True)
            print(f"[DEBUG] id_to_label (extracted)={id_to_label}", flush=True)
            print(f"[DEBUG] gear_prims keys (compare against)={list(gear_prims.keys())}", flush=True)
            if seg_data is not None:
                print(f"[DEBUG] seg_data shape={seg_data.shape} dtype={seg_data.dtype} "
                      f"unique values={np.unique(seg_data)}", flush=True)
            else:
                print("[DEBUG] seg_data is None", flush=True)

        xform_cache.Clear()

        frame_gt = {}
        if seg_data is not None:
            for inst_id, label in id_to_label.items():
                if label not in gear_prims:
                    continue
                mask = (seg_data == inst_id)
                if not mask.any():
                    continue
                PILImage.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
                    os.path.join(mask_dir, f"{tag}_{label}.png")
                )
                R_cv, t_cv_mm = compute_pose_in_camera(
                    xform_cache, gear_prims[label], cam_prim, debug=first_capture
                )
                frame_gt[label] = {
                    "cam_R_m2c": R_cv.flatten().tolist(),
                    "cam_t_m2c": t_cv_mm.tolist(),
                }

        with open(os.path.join(gt_dir, f"{tag}.json"), "w") as f:
            json.dump(frame_gt, f, indent=2)

        print(f"[INFO] shot {shot}: cam roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} "
              f"visible gears={list(frame_gt.keys())} captured", flush=True)
        shot += 1
        rate_window_shots += 1
        first_capture = False

        now = time.monotonic()
        if now - rate_window_start >= 5.0:
            fps = rate_window_shots / (now - rate_window_start)
            print(f"[INFO] capture rate = {fps:.2f} shots/sec", flush=True)
            rate_window_start = now
            rate_window_shots = 0

        drift_gears()
        if shot % GEAR_REORIENT_EVERY == 0:
            reorient_gears()
            fix_box_escapes()
        elif shot % GEAR_BOX_CHECK_EVERY == 0:
            fix_box_escapes()

        if shot % SHOTS_PER_POSE == 0:
            roll, pitch, yaw = randomize_camera()
            settle_count = 0
            print(f"[INFO] === camera rotated -> roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}, "
                  f"next {SHOTS_PER_POSE} shots held at this pose ===", flush=True)

    print(f"[INFO] stopped — {shot} frames captured total, {OUTPUT_DIR}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
