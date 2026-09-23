"""Generate a bin-picking dataset shard with the same Isaac Sim calls as legacy/test_depthcam_rotation.py
(Isaac Sim Full experience, RigidPrim, Replicator annotators, timeline + app.update), which is the path that
runs on the GB10 machine, plus binsim's placement, sensor model, BOP/COCO/grasp writers and viz.

    lab scripts/generate_isaacsim.py --config configs/default.yaml [--headless] [key=value ...]

One bin per process (run.num_envs is ignored). Output: <run.output_dir>/shard_XX (same layout as generate.py).
"""
import argparse
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from binsim.config import load_config  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
parser.add_argument("overrides", nargs="*", help="dotted config overrides, e.g. run.num_scenes=10")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cfg = load_config(args.config, args.overrides)
args.experience = os.path.expanduser(cfg.sim.get("experience")
                                     or "~/IsaacSim/_build/linux-aarch64/release/apps/isaacsim.exp.full.kit")
args.headless = bool(args.headless or cfg.sim.get("headless", False))
if not args.headless:
    args.visualizer = ["kit"]
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from pxr import Gf, Semantics, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from binsim import grasps as G  # noqa: E402
from binsim.assets import git_hash, prepare_all  # noqa: E402
from binsim.camera import intrinsics_from_cfg, rim_height, sample_view  # noqa: E402
from binsim.config import as_range, dump_config, to_plain  # noqa: E402
from binsim.loading import (bin_geom_from_cfg, in_bin, part_geoms_from_cfg, plan_scene, sample_counts,  # noqa: E402
                            shake_schedule)
from binsim.postprocess import SceneCapture, SceneContext, ViewCapture, process_scene  # noqa: E402
from binsim.transforms import inv_T, make_T, mat_to_quat, quat_to_mat, rpy_to_mat  # noqa: E402
from binsim.writers.bop import write_camera_json, write_models  # noqa: E402

APP_HZ = 60.0                          # physics steps per app.update() in the default Isaac Sim physics scene
CV_TO_USD_CAM = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes <-> USD camera axes (looks down -Z, +Y up)
PARTS_ROOT = "/World/parts"
CAM_PATH = "/World/phoxi"
_PART_RE = re.compile(r"(?:^|/)part_(\d+)$")


def update(n=1):
    for _ in range(n):
        simulation_app.update()


def set_pose(prim, T):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, T[:3, 3])))
    q = mat_to_quat(T[:3, :3])
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))


def world_T(prim):
    m = UsdGeom.XformCache().GetLocalToWorldTransform(prim).GetOrthonormalized()
    M = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    return M.T  # Gf matrices are row-vector convention


def id_to_slot(seg_info) -> dict:
    """{instance id -> part index} from instance_segmentation info (key names differ across versions)."""
    lut = {}
    for key in ("idToSemantics", "idToLabels"):
        for k, v in (seg_info.get(key) or {}).items():
            label = v if isinstance(v, str) else (v.get("class") or v.get("semantic") or "")
            m = _PART_RE.search(str(label).strip())
            if m:
                lut[int(k)] = int(m.group(1))
    return lut


def build_static_scene(stage, bin_asset, T_w_bin, intr):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)

    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.36)])
    fx = UsdGeom.Xformable(floor)
    fx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
    fx.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.05))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    bin_prim = stage.DefinePrim("/World/bin", "Xform")
    bin_prim.GetReferences().AddReference(bin_asset.usd_path)
    set_pose(bin_prim, T_w_bin)

    UsdLux.DomeLight.Define(stage, "/World/dome_light").CreateIntensityAttr(800.0)
    key = UsdLux.SphereLight.Define(stage, "/World/key_light")
    key.CreateRadiusAttr(0.15)
    key.CreateIntensityAttr(5000.0)
    UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0.3, 0.3, 1.6))

    cam = UsdGeom.Camera.Define(stage, CAM_PATH)
    focal = 24.0
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(float(intr.width * focal / intr.fx))
    cam.CreateVerticalApertureAttr(float(intr.height * focal / intr.fy))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    return cam.GetPrim(), key.GetPrim()


def spawn_parts(stage, placements, part_assets, T_w_bin):
    if stage.GetPrimAtPath(PARTS_ROOT):
        stage.RemovePrim(PARTS_ROOT)
    UsdGeom.Xform.Define(stage, PARTS_ROOT)
    paths = []
    for k, p in enumerate(placements):
        path = f"{PARTS_ROOT}/part_{k:03d}"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(part_assets[p.type_idx].usd_path)
        set_pose(prim, T_w_bin @ make_T(quat_to_mat(p.quat), p.pos))
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set(f"part_{k:03d}")
        paths.append(path)
    return paths


def settle(rigid, n, rng):
    """Step until every part is (nearly) at rest. Step counts in the config are at sim.physics_dt; converted to
    app updates (APP_HZ)."""
    lcfg = cfg.loading
    st = lcfg.get("settle", {})
    k = float(cfg.sim.get("physics_dt", 0.005)) * APP_HZ
    min_u, max_u = int(int(st.get("min_steps", 60)) * k), max(int(int(st.get("max_steps", 1200)) * k), 1)
    lin_thr, ang_thr = float(st.get("lin_vel_thresh", 0.005)), float(st.get("ang_vel_thresh", 0.05))
    hold_need = max(int(int(st.get("hold_steps", 10)) * k), 2)
    kicks = sorted({int(s * k) for s in shake_schedule(lcfg, rng)})
    sh = lcfg.get("shake", {})
    hold = u = 0
    for u in range(max_u):
        if u in kicks:
            lin = rng.uniform(-1, 1, (n, 3)) * float(sh.get("lin_vel_mps", 0.15))
            lin[:, 2] = np.abs(lin[:, 2])
            rigid.set_velocities(lin, rng.uniform(-1, 1, (n, 3)) * float(sh.get("ang_vel_rps", 2.0)))
        update()
        if u >= min_u and u > max(kicks, default=-1):
            lin, ang = (a.numpy() for a in rigid.get_velocities())
            still = np.linalg.norm(lin, axis=1).max() < lin_thr and np.linalg.norm(ang, axis=1).max() < ang_thr
            hold = hold + 1 if still else 0
            if hold >= hold_need:
                break
    return u + 1


def main():
    run = cfg.run
    shard = int(run.get("shard_id", 0))
    out = os.path.join(os.path.expanduser(run.output_dir), f"shard_{shard:02d}")
    os.makedirs(os.path.join(out, "viz"), exist_ok=True)
    rng = np.random.default_rng([int(cfg.sim.get("seed", 0)), shard])

    part_assets, bin_asset = prepare_all(cfg)
    geoms = part_geoms_from_cfg(cfg.parts, [a.extents for a in part_assets], [a.radius for a in part_assets])
    bin_geom = bin_geom_from_cfg(cfg.bin)
    intr = intrinsics_from_cfg(cfg.camera)
    T_w_bin = make_T(rpy_to_mat(cfg.bin.get("rpy_deg", [0, 0, 0])), cfg.bin.get("pos_m", [0, 0, 0]))
    T_bin_w = inv_T(T_w_bin)
    grng = np.random.default_rng(0)
    libraries = [G.build_library(p, a.mesh, cfg.gripper, grng) for p, a in zip(cfg.parts, part_assets)]
    npts = int(cfg.get("grasps", {}).get("point_samples", 1500))
    points = [G.sample_part_points(a.mesh, npts, seed=i) for i, a in enumerate(part_assets)]
    ctx = SceneContext(cfg, intr, list(cfg.parts), part_assets, points, libraries, bin_geom,
                       rim_height(cfg.bin, bin_geom), out)
    dump_config(cfg, os.path.join(out, "config_resolved.yaml"))
    write_models(out, cfg.parts, part_assets)
    write_camera_json(out, intr, float(cfg.outputs.get("depth_scale", 0.1)))
    info = {"git": git_hash(REPO), "backend": "isaacsim (legacy-style)", "shard": shard,
            "intrinsics": intr.to_json(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    usd = omni.usd.get_context()
    usd.new_stage()
    update(5)
    stage = usd.get_stage()
    cam_prim, key_prim = build_static_scene(stage, bin_asset, T_w_bin, intr)
    update(5)

    rgb_a = depth_a = seg_a = None
    timeline = omni.timeline.get_timeline_interface()
    lights = cfg.get("randomization", {}).get("lights", {}) or {}
    V = int(run.views_per_scene)
    total = int(run.num_scenes)
    viz_every = int(run.get("save_viz_every", 1) or 0)
    margin = float(cfg.loading.get("settle", {}).get("escape_margin_m", 0.02))
    subframes = max(int(cfg.camera.get("render_subframes", 2)), 1) + 1
    print(f"[INFO] generating {total} scenes x {V} views -> {out}", flush=True)
    t0 = time.monotonic()
    done = 0
    for idx in range(total):
        if not simulation_app.is_running():
            break
        ts = time.monotonic()
        timeline.stop()
        update(2)
        counts = sample_counts(cfg.parts, rng)
        placements, mode = plan_scene(counts, geoms, bin_geom, cfg.loading, rng)
        paths = spawn_parts(stage, placements, part_assets, T_w_bin)
        if seg_a is None:
            # attach after labelled prims exist (as the legacy script does); attached to an unlabelled stage, the
            # instance segmentation of the first scene comes back as a single id
            render_product = rep.create.render_product(CAM_PATH, (intr.width, intr.height))
            rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
            depth_a = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            seg_a = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False})
            for a in (rgb_a, depth_a, seg_a):
                a.attach(render_product)
        if "key_intensity" in lights:
            UsdLux.SphereLight(key_prim).GetIntensityAttr().Set(float(rng.uniform(*as_range(lights["key_intensity"]))))
        off = float(lights.get("key_offset_m", 0.8))
        UsdGeom.Xformable(key_prim).GetOrderedXformOps()[0].Set(
            Gf.Vec3d(rng.uniform(-off, off), rng.uniform(-off, off), rng.uniform(*as_range(lights.get("key_height_m", [1.2, 2.0])))))
        update(2)
        rigid = RigidPrim(paths)
        timeline.play()
        update(2)
        steps = settle(rigid, len(paths), rng)

        pos, quat = (a.numpy().astype(np.float64) for a in rigid.get_world_poses())
        pos_bin = pos @ T_bin_w[:3, :3].T + T_bin_w[:3, 3]
        active = in_bin(pos_bin, bin_geom, margin)
        for k in np.flatnonzero(~active):
            UsdGeom.Imageable(stage.GetPrimAtPath(paths[k])).MakeInvisible()
        timeline.pause()
        update(2)

        views = []
        for v in range(V):
            T_bin_cam = sample_view(cfg.camera, cfg.bin, bin_geom, rng, jitter=(v > 0 or V == 1))
            T_w_cam = T_w_bin @ T_bin_cam
            T_usd = T_w_cam.copy()
            T_usd[:3, :3] = T_w_cam[:3, :3] @ CV_TO_USD_CAM
            set_pose(cam_prim, T_usd)
            update(subframes)
            for _ in range(10):
                rgb, depth, seg = rgb_a.get_data(), depth_a.get_data(), seg_a.get_data()
                lut = id_to_slot(seg.get("info", {}))
                if lut or not active.any():
                    break
                update(3)  # the first scene's new semantic labels can take a few frames to reach the annotator
            seg_ids = np.asarray(seg["data"])
            inst = np.full(seg_ids.shape, -1, np.int32)
            for i, k in lut.items():
                inst[seg_ids == i] = k
            T_read = world_T(cam_prim)
            T_read[:3, :3] = T_read[:3, :3] @ CV_TO_USD_CAM
            if idx == 0 and v == 0:
                d = np.asarray(depth)
                print(f"[DEBUG] rgb {np.asarray(rgb).shape}, depth finite {np.isfinite(d).mean():.2f}, "
                      f"seg ids {np.unique(seg_ids)[:12].tolist()} -> parts mapped {len(lut)}, "
                      f"camera readback |dpos|={np.abs(T_read[:3, 3] - T_w_cam[:3, 3]).max() * 1000:.2f} mm", flush=True)
            views.append(ViewCapture(np.asarray(rgb), np.asarray(depth, np.float32), inst, T_bin_w @ T_read))

        slots = np.flatnonzero(active).tolist()
        sc = SceneCapture(scene_id=shard * 100000 + idx, mode=mode, slots=slots,
                          types=[placements[k].type_idx for k in slots],
                          T_bin_part=[T_bin_w @ make_T(quat_to_mat(quat[k]), pos[k]) for k in slots],
                          views=views, save_viz=bool(viz_every) and idx % viz_every == 0,
                          seed=int(rng.integers(1 << 31)), removed=int((~active).sum()))
        s = process_scene(ctx, sc)
        done += 1
        v0 = s["views"][0] if s and s.get("views") else {}
        print(f"[INFO] scene {idx + 1}/{total}: {len(slots)} parts ({sc.removed} escaped), mode={mode}, "
              f"settle {steps} updates, {time.monotonic() - ts:.1f}s | mask consistency={v0.get('mask_consistency')}, "
              f"valid depth={v0.get('valid_depth_frac')}, topmost=#{v0.get('topmost')}", flush=True)
        if idx == 0 and v0.get("table"):
            print("\n".join(v0["table"]), flush=True)

    timeline.stop()
    info.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "scenes": done, "images": done * V,
                 "seconds": round(time.monotonic() - t0, 1)})
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump(to_plain(info), f, indent=2)
    print(f"[INFO] done: {done} scenes / {done * V} images in {time.monotonic() - t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
