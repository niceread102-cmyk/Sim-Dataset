"""Generate a bin-picking dataset shard with the same Isaac Sim calls as legacy/test_depthcam_rotation.py
(Isaac Sim Full experience, RigidPrim, Replicator annotators, timeline + app.update), which is the path that
runs on the GB10 machine, plus binsim's placement, sensor model, BOP/COCO/grasp writers and viz.

    lab generate_isaacsim.py [--config configs/map/map.yaml] [--headless] [key=value ...]   (generate_supervisor.py runs it shard by shard)
    lab generate_isaacsim.py obj:=support_input_shaft obj_num:=10 out:=~/Downloads/shaft_data
        obj:=  part yaml of configs/parts/ (with or without .yaml; make_object_config.py makes one) instead of map.yaml's first part
        obj_num:=  how many of it per bin (N or [min,max]; default: the `count:` of the part yaml, else map.yaml's)
        out:=  run.output_dir
        scenes:=  run.num_scenes (piles = camera cycles)     views:=  run.views_per_scene (images per pile): scenes x views images

One bin per process (run.num_envs is ignored). Output: <run.output_dir>/shard_XX.
"""
import argparse
import ctypes
import faulthandler
import json
import os
import re
import signal
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))  # Simulation/isaacsim: configs/, binsim/, assets_cache/
sys.path.insert(0, REPO)

from binsim.config import expand_shorthand, load_config  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=os.path.join(REPO, "configs", "map", "map.yaml"))
parser.add_argument("overrides", nargs="*", help="dotted config overrides, e.g. run.num_scenes=10; or the short words obj:=<part> "
                    "obj_num:=N out:=DIR (see the top of this file)")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.overrides = expand_shorthand(args.overrides, args.config)
cfg = load_config(args.config, args.overrides)
_out = os.path.join(os.path.expanduser(cfg.run.output_dir), f"shard_{int(cfg.run.get('shard_id', 0)):02d}", "train_pbr")
if os.path.isdir(_out) and any(d.isdigit() for d in os.listdir(_out)) and not cfg.run.get("overwrite", False):
    sys.exit(f"[ERROR] {os.path.dirname(_out)} already has scenes - they would be overwritten. Use another run.shard_id "
             f"(generate_supervisor.py picks a new one itself), delete the folder, or set run.overwrite=true")
args.experience = os.path.expanduser(cfg.sim.get("experience")
                                     or "~/IsaacSim/apps/isaacsim.exp.base.kit")
args.headless = bool(args.headless or cfg.sim.get("headless", False))
if not args.headless:
    args.visualizer = ["kit"]
_kit_args = " ".join(a for a in (getattr(args, "kit_args", "") or "", str(cfg.sim.get("kit_args") or "")) if a.strip())
if _kit_args:
    args.kit_args = _kit_args  # e.g. the RTX descriptor pool sizes (sim.kit_args)
simulation_app = AppLauncher(args).app

# hang diagnosis: `kill -USR1 <pid>` prints every Python thread's stack; and let a watchdog attach py-spy / gdb
# although kernel.yama.ptrace_scope=1 (generate_supervisor.py does both when a shard stops making progress)
faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, all_threads=True)
try:
    ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)  # PR_SET_PTRACER, PR_SET_PTRACER_ANY
except Exception:
    pass

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from pxr import Gf, Sdf, Semantics, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

from binsim import grasps as G  # noqa: E402
from binsim.assets import git_hash, prepare_all, set_omnipbr  # noqa: E402
from binsim.camera import centred, intrinsics_from_cfg, mount_eye_target, rim_height, sample_view  # noqa: E402
from binsim.lens import LensWarp, lens_intrinsics, render_intrinsics  # noqa: E402
from binsim.config import as_range, dump_config, to_plain  # noqa: E402
from binsim.loading import (bin_geom_from_cfg, in_bin, part_geoms_from_cfg, plan_scene, sample_counts,  # noqa: E402
                            shake_schedule)
from binsim.postprocess import SceneCapture, SceneContext, ViewCapture, process_scene  # noqa: E402
from binsim.transforms import inv_T, make_T, mat_to_quat, quat_to_mat, rpy_to_mat  # noqa: E402
from binsim.writers.async_io import AsyncWriter  # noqa: E402
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


def map_ids(seg_ids: np.ndarray, lut: dict) -> np.ndarray:
    """Instance-segmentation ids -> part index per pixel (-1 = not a part), in one pass."""
    if not lut:
        return np.full(seg_ids.shape, -1, np.int32)
    keys = np.fromiter(lut.keys(), np.int64, len(lut))
    vals = np.fromiter(lut.values(), np.int32, len(lut))
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]
    pos = np.clip(np.searchsorted(keys, seg_ids), 0, len(keys) - 1)
    return np.where(keys[pos] == seg_ids, vals[pos], -1).astype(np.int32)


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

    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    sp = scene.GetPrim()
    sp.AddAppliedSchema("PhysxSceneAPI")
    sp.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(
        int(round(1.0 / float(cfg.sim.get("physics_dt", 0.005)))))
    sp.CreateAttribute("physxScene:enableCCD", Sdf.ValueTypeNames.Bool).Set(bool(cfg.sim.get("enable_ccd", False)))

    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    floor.CreateDisplayColorAttr([Gf.Vec3f(*(cfg.get("floor", {}) or {}).get("color", [0.35, 0.35, 0.36]))])
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

    # the PhoXi lights the scene with its own projector (next to the camera): a light at the projector position,
    # child of the camera so it moves with every view (camera.noise.projector.baseline_m, OpenCV camera axes)
    lights = cfg.get("randomization", {}).get("lights", {}) or {}
    proj = None
    if "projector_intensity" in lights:
        base = (cfg.camera.get("noise", {}).get("projector", {}) or {}).get("baseline_m", [0.24, 0.0, 0.0])
        proj = UsdLux.SphereLight.Define(stage, f"{CAM_PATH}/projector_light")
        proj.CreateRadiusAttr(float(lights.get("projector_radius_m", 0.03)))
        proj.CreateIntensityAttr(float(np.mean(as_range(lights["projector_intensity"]))))
        UsdGeom.Xformable(proj).AddTranslateOp().Set(Gf.Vec3d(*(CV_TO_USD_CAM @ np.asarray(base, float))))
    return cam.GetPrim(), key.GetPrim(), proj


def sample_lights(lights, rng) -> dict:
    """This scene's light setup (randomization.lights)."""
    off = float(lights.get("key_offset_m", 0.8))
    draw = (lambda k: float(rng.uniform(*as_range(lights[k]))) if k in lights else None)
    return {"dome": draw("dome_intensity"), "key": draw("key_intensity"), "proj": draw("projector_intensity"),
            "key_pos": np.array([rng.uniform(-off, off), rng.uniform(-off, off),
                                 rng.uniform(*as_range(lights.get("key_height_m", [1.2, 2.0])))])}


def apply_lights(stage, key_prim, proj_light, L, rng=None, jitter=0.0, move=0.0):
    """Set the scene's lights; with jitter / move (a new pile state after a shake) every intensity varies by
    +- jitter (relative) and the key light moves by up to +- move m around the scene's values."""
    f = (lambda: float(rng.uniform(1 - jitter, 1 + jitter))) if jitter else (lambda: 1.0)
    if L["dome"] is not None:
        UsdLux.DomeLight(stage.GetPrimAtPath("/World/dome_light")).GetIntensityAttr().Set(L["dome"] * f())
    if L["key"] is not None:
        UsdLux.SphereLight(key_prim).GetIntensityAttr().Set(L["key"] * f())
    if proj_light is not None and L["proj"] is not None:
        proj_light.GetIntensityAttr().Set(L["proj"] * f())
    pos = L["key_pos"] + (rng.uniform(-move, move, 3) if move else 0.0)
    UsdGeom.Xformable(key_prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float, pos)))


def randomize_materials(stage, paths, placements, rng):
    """Per-scene appearance jitter (randomization.part_materials / bin_materials): every part gets its own colour,
    roughness and metallic around its part config (roughness / metallic may be a value or a [lo, hi] range).
    randomization.part_color: fixed = the part's colour as configured | gray = same hue, only lighter/darker
    (+- 2 x color_jitter) | color = per-channel jitter (hue changes too)."""
    rc = cfg.get("randomization", {}) or {}
    jit = float(rc.get("color_jitter", 0.08))
    mode = str(rc.get("part_color", "gray"))
    if mode not in ("fixed", "gray", "color"):
        raise ValueError(f"randomization.part_color must be fixed | gray | color, got {mode!r}")

    def value(v, j):
        lo, hi = as_range(v)
        return float(np.clip(rng.uniform(lo, hi) + rng.uniform(-j, j), 0.0, 1.0))

    if rc.get("part_materials", True):
        for path, pl in zip(paths, placements):
            sh = UsdShade.Shader(stage.GetPrimAtPath(f"{path}/Looks/Visual/Shader"))
            if not sh:
                continue
            p = cfg.parts[pl.type_idx]
            base = np.asarray(p.get("color", [0.7, 0.7, 0.7]), float)
            if mode == "fixed":
                col = base
            elif mode == "gray":
                col = np.clip(base * rng.uniform(1 - jit * 2, 1 + jit * 2), 0, 1)
            else:
                col = np.clip(base * rng.uniform(1 - jit * 2, 1 + jit * 2) + rng.normal(0, jit / 2, 3), 0, 1)
            set_omnipbr(sh, col, p.get("color_tint", [1.0, 1.0, 1.0]),
                        value(p.get("roughness", 0.5), float(rc.get("roughness_jitter", 0.0))),
                        value(p.get("metallic", 0.0), float(rc.get("metallic_jitter", 0.0))))
    if rc.get("bin_materials", True):
        sh = UsdShade.Shader(stage.GetPrimAtPath("/World/bin/Looks/Visual/Shader"))
        if sh:
            base = np.asarray(cfg.bin.get("color", [0.2, 0.3, 0.6]), float)
            set_omnipbr(sh, np.clip(base + rng.normal(0, jit, 3), 0, 1), [1.0, 1.0, 1.0], rng.uniform(0.4, 0.9), 0.0)


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


_PARKED = {"n": 0, "idx": set()}  # parts parked in the current scene (each gets its own spot on the floor)


def park(rigid, idx, T_w_bin):
    """Move parts (escaped the bin / caught on its rim / picked) far to the side of the bin, outside every view, at
    rest, each on its own floor spot (spots are counted per scene so later calls never stack parts)."""
    idx = np.asarray([i for i in np.asarray(idx, int) if int(i) not in _PARKED["idx"]], int)  # never move one twice
    if not len(idx):
        return
    _PARKED["idx"].update(int(i) for i in idx)
    n = _PARKED["n"] + np.arange(len(idx))
    _PARKED["n"] += len(idx)
    p_bin = np.stack([1.2 + 0.08 * (n % 8), -1.5 + 0.08 * (n // 8), np.full(len(idx), 0.01)], 1)  # on the floor
    rigid.set_world_poses(p_bin @ T_w_bin[:3, :3].T + T_w_bin[:3, 3], np.tile([1.0, 0.0, 0.0, 0.0], (len(idx), 1)),
                          indices=idx)
    rigid.set_velocities(np.zeros((len(idx), 3)), np.zeros((len(idx), 3)), indices=idx)


def remove_out_of_bin(stage, rigid, paths, keep, T_w_bin, T_bin_w, bin_geom, ups=None):
    """Parts among `keep` that left the bin or got caught on its rim (loading.settle.escape_margin_m /
    rim_clearance_m), or got turned over (loading.settle.max_tilt_deg: their up axis `ups`, in the part frame, tilted
    further than this from the bin's +Z): hidden and parked off to the side. Returns the mask of parts still in the
    bin."""
    st = cfg.loading.get("settle", {})
    pos, quat = (a.numpy().astype(np.float64) for a in rigid.get_world_poses())
    inside = in_bin(pos @ T_bin_w[:3, :3].T + T_bin_w[:3, 3], bin_geom, float(st.get("escape_margin_m", 0.0)),
                    st.get("rim_clearance_m"))
    if st.get("max_tilt_deg") is not None and ups is not None:
        up_z = np.einsum("j,njk,nk->n", T_bin_w[2, :3], quat_to_mat(quat), ups)  # bin-frame z of each part's up axis
        inside &= up_z >= np.cos(np.deg2rad(float(st["max_tilt_deg"])))
    parked = np.zeros(len(keep), bool)
    parked[list(_PARKED["idx"])] = True
    out = np.flatnonzero(keep & ~inside & ~parked)  # parts already parked (picked / removed earlier) stay where they are
    for k in out:
        UsdGeom.Imageable(stage.GetPrimAtPath(paths[k])).MakeInvisible()
    park(rigid, out, T_w_bin)
    return keep & inside


def pick_plan(n_start, n_states, pk, rng):
    """Parts to remove at each of the n_states pile-state changes of a scene (loading.pick). [] = no picking.
    end_left_pct: the bin goes from full down to this % of the parts the scene started with (drawn per scene), spread over ALL
    states with +-jitter per state, so the pile thins out evenly over the views whatever views_per_scene / every_views / the
    part count are (the last state has the fewest parts, not the second half of the views).
    end_left (without end_left_pct): the same with a count of parts left."""
    if n_states <= 0:
        return []
    if pk.get("end_left_pct"):
        lo, hi = as_range(pk["end_left_pct"])
        n_end = min(n_start, max(1, int(round(n_start * rng.uniform(lo, hi) / 100))))
    elif pk.get("end_left"):
        lo, hi = as_range(pk["end_left"])
        n_end = int(min(rng.integers(int(lo), int(hi) + 1), n_start))
    else:
        return []
    w = rng.uniform(1 - float(pk.get("jitter", 0.3)), 1 + float(pk.get("jitter", 0.3)), n_states)
    q = np.floor(np.cumsum(w) / w.sum() * (n_start - n_end) + 1e-9).astype(int)
    return np.diff(np.r_[0, q]).tolist()


def pick_top(stage, rigid, paths, keep, T_bin_w, T_w_bin, bin_geom, k):
    """Bin picking in progress: remove the k highest parts still in the bin. Picked parts are hidden and parked; they
    stay in the scene's GT list (not visible). Returns how many were picked."""
    if k <= 0:
        return 0
    pos = rigid.get_world_poses()[0].numpy().astype(np.float64) @ T_bin_w[:3, :3].T + T_bin_w[:3, 3]
    cand = np.flatnonzero(keep & in_bin(pos, bin_geom, 0.0, None))
    k = min(k, len(cand))
    if k <= 0:
        return 0
    top = cand[np.argsort(-pos[cand, 2])[:k]]
    for j in top:
        UsdGeom.Imageable(stage.GetPrimAtPath(paths[j])).MakeInvisible()
    park(rigid, top, T_w_bin)
    return k


def parts_in_bin(rigid, keep, T_bin_w, bin_geom):
    """(N, 3) bin-frame centres (m) of the parts still in the bin (current pile state)."""
    pos = rigid.get_world_poses()[0].numpy().astype(np.float64) @ T_bin_w[:3, :3].T + T_bin_w[:3, 3]
    return pos[keep & in_bin(pos, bin_geom, 0.0, None)]


def camera_offset(pile, eye_z0, bin_geom, rng, half_fov=(0.4128, 0.3088)):
    """camera.random_height: a share camera.near_share of the views gets a camera height drawn between the mount height
    and min_distance_m above the highest part, the other views are taken from the mount height (the fixed PhoXi's own
    view) - so full and sparse bins are both seen from near and far. The lower the camera, the more it moves over a
    randomly chosen part (camera.aim_at_parts; a close-up of the bin centre would miss sparse parts), but only so far
    that at most camera.aim_max_outside of the image (per axis) looks past the inner walls. half_fov = (W/2/fx, H/2/fy).
    Returns (height offset m, xy offset m) for sample_view."""
    if not cfg.camera.get("random_height", False) or not len(pile):
        return 0.0, np.zeros(2)
    if rng.random() >= float(cfg.camera.get("near_share", 1.0)):
        return 0.0, np.zeros(2)
    z_lo = float(pile[:, 2].max()) + float(cfg.camera.get("min_distance_m", 0.5))
    if z_lo >= eye_z0:
        return 0.0, np.zeros(2)
    z = rng.uniform(z_lo, eye_z0)
    close = (eye_z0 - z) / (eye_z0 - z_lo)  # 0 = mount height, 1 = closest
    xy = np.zeros(2)
    if cfg.camera.get("aim_at_parts", True):
        ix, iy = bin_geom.inner_size[:2]
        # image footprint on the bin floor at this height; the view centre may move only so far that the part of the
        # footprint beyond the inner walls stays below aim_max_outside of the image
        d = z - bin_geom.floor_z
        hw, hh = d * half_fov[0], d * half_fov[1]
        keep = 1.0 - 2.0 * float(cfg.camera.get("aim_max_outside", 0.15))
        lim = np.array([max(ix / 2 - keep * hw, 0.0), max(iy / 2 - keep * hh, 0.0)])
        xy = np.clip(close * pile[rng.integers(len(pile)), :2], -lim, lim)
    return z - eye_z0, xy


def push_out(rigid, idx, T_w_bin, sp, rng, bin_geom):
    """One outward push that spreads a heap (bunched up in the middle by shaking) back into the corners: parts get an
    outward (from the pile's xy centre) + slightly upward velocity that grows with their height in the pile (lowest
    ~30 % not pushed), aimed from the pile's current shape. Parts near a wall are pushed less (spread.wall_fade_m) so
    they do not climb onto the rim."""
    R = T_w_bin[:3, :3]
    speed = float(sp.get("speed_mps", 0.0))
    p = (rigid.get_world_poses()[0].numpy().astype(np.float64) - T_w_bin[:3, 3]) @ R  # bin frame
    # only the parts still in the bin: picked / escaped parts are parked far outside it and would drag the pile
    # centre (and so every push) towards the parking area
    idx = idx[in_bin(p[idx], bin_geom, 0.0, None)]
    if not len(idx):
        return
    lin, ang = (a.numpy().astype(np.float64) for a in rigid.get_velocities())
    z = p[idx, 2]
    lo = np.percentile(z, 30)
    w = np.clip((z - lo) / max(z.max() - lo, 1e-6), 0.0, 1.0)
    ix, iy = bin_geom.inner_size[:2]
    wall = np.minimum(ix / 2 - np.abs(p[idx, 0]), iy / 2 - np.abs(p[idx, 1]))  # distance to the nearest wall
    w *= np.clip(wall / max(float(sp.get("wall_fade_m", 0.05)), 1e-6), 0.0, 1.0)
    d = p[idx, :2] - p[idx, :2].mean(axis=0)
    r = np.linalg.norm(d, axis=1, keepdims=True)
    rnd = rng.normal(size=d.shape)
    d = np.where(r > 1e-3, d / np.maximum(r, 1e-9), rnd / np.linalg.norm(rnd, axis=1, keepdims=True))
    v = np.zeros_like(lin)
    v[idx, :2] = speed * w[:, None] * d
    v[idx, 2] = float(sp.get("lift", 0.25)) * speed * w
    rigid.set_velocities(lin + v @ R.T, ang)


def vibrate(bin_t_op, T_w_bin, vib, rng, rigid, active, bin_geom):
    """One motion: shake the kinematic bin in xy (smooth fade in/out, random direction); while the shaking dies down
    (50-80 % of it) the heap is pushed outward (vibrate.spread) so it lies flat into the corners, then it rests.
    The bin ends exactly where it started. Timeline must be playing (1 app update = 1/APP_HZ s)."""
    amp = float(rng.uniform(*as_range(vib.get("amp_m", 0.003))))  # value or [lo, hi]: a different shake each time
    freq = float(rng.uniform(*as_range(vib.get("freq_hz", 8.0))))
    dur = max(float(vib.get("duration_s", 0.3)), 1.0 / APP_HZ)
    n = int(round(dur * APP_HZ))
    sp = vib.get("spread", {}) or {}
    pulses = int(sp.get("pulses", 3)) if float(sp.get("speed_mps", 0.0)) > 0 else 0
    # pushes between 50 % and 80 % of the shaking: the pile still gets the dying-down shaking + rest after the last one
    push_at = set(np.linspace(0.5 * n, 0.8 * n, pulses).round().astype(int).tolist()) if pulses else set()
    idx = np.flatnonzero(active)
    phase = rng.uniform(0, 2 * np.pi, 2)
    p0 = T_w_bin[:3, 3].astype(float)
    for i in range(1, n + 1):
        t = i / APP_HZ
        off = amp * 0.5 * (1 - np.cos(2 * np.pi * min(t / dur, 1.0))) * np.sin(2 * np.pi * freq * t + phase)
        bin_t_op.Set(Gf.Vec3d(p0[0] + off[0], p0[1] + off[1], p0[2]))
        if i in push_at:
            push_out(rigid, idx, T_w_bin, sp, rng, bin_geom)
        update()
    bin_t_op.Set(Gf.Vec3d(*p0))
    update(max(int(round(float(vib.get("settle_s", 0.3)) * APP_HZ)), 1))


def main():
    run = cfg.run
    shard = int(run.get("shard_id", 0))
    offset = int(run.get("scene_offset", 0))  # scene ids start here (generate_supervisor.py run.num_parts: part k -> k * num_scenes)
    # generate_supervisor.py sets scene_id_base: ids continue exactly where the previous process of the part stopped
    # (otherwise id = shard * 100000 + offset + index, unique for launch_multi_gpu.py shards)
    id_base = run.get("scene_id_base")
    id_base = None if id_base is None else int(id_base)
    out = os.path.join(os.path.expanduser(run.output_dir), f"shard_{shard:02d}")
    os.makedirs(os.path.join(out, "viz"), exist_ok=True)
    # the offset is part of the seed: another part (same shard ids) must not repeat the same piles
    rng = np.random.default_rng([int(cfg.sim.get("seed", 0)), shard] + ([offset] if offset else [])
                                + ([id_base] if id_base else []))
    import carb.settings
    _s = carb.settings.get_settings()
    print(f"[INFO] rtx reservedDescriptors={_s.get('/rtx/reservedDescriptors')} "
          f"descriptorSets={_s.get('/rtx/descriptorSets')}", flush=True)

    part_assets, bin_asset = prepare_all(cfg)
    geoms = part_geoms_from_cfg(cfg.parts, [a.extents for a in part_assets], [a.radius for a in part_assets],
                                [a.mesh for a in part_assets])  # mesh: loading.mode stack rests parts on their shape
    bin_geom = bin_geom_from_cfg(cfg.bin)
    intr = intrinsics_from_cfg(cfg.camera)
    lens = lens_intrinsics(intr, cfg.camera)  # camera.lens: K + distortion of the real lens, None = ideal pinhole
    if lens is not None:
        # the renderer is a pinhole camera with the optical axis at the image centre: render a bit more than the image
        # (render_intr) and bend it into the image of the real lens (warp); labels are projected through the same lens
        intr, render_intr = lens, render_intrinsics(lens)
        warp = LensWarp(intr, render_intr)
        print(f"[INFO] lens: k1,k2,p1,p2,k3={tuple(round(k, 5) for k in intr.dist)}, principal point "
              f"({intr.cx:.1f}, {intr.cy:.1f}); rendering {render_intr.width}x{render_intr.height} -> image "
              f"{intr.width}x{intr.height}", flush=True)
    else:
        ren = centred(intr)  # K of the rendered images (the renderer ignores cx/cy) - written to scene_camera.json
        if max(abs(ren.cx - intr.cx), abs(ren.cy - intr.cy)) > 0.01:
            print(f"[WARN] camera.intrinsics cx/cy = ({intr.cx:.2f}, {intr.cy:.2f}) at the render size, but the renderer "
                  f"puts the principal point at the image centre: labels use cx={ren.cx}, cy={ren.cy}", flush=True)
        intr = render_intr = ren
        warp = None
    T_w_bin = make_T(rpy_to_mat(cfg.bin.get("rpy_deg", [0, 0, 0])), cfg.bin.get("pos_m", [0, 0, 0]))
    T_bin_w = inv_T(T_w_bin)
    grng = np.random.default_rng(0)
    libraries = [G.build_library(p, a.mesh, cfg.gripper, grng) for p, a in zip(cfg.parts, part_assets)]
    npts = int(cfg.get("grasps", {}).get("point_samples", 1500))
    points = [G.sample_part_points(a.mesh, npts, seed=i) for i, a in enumerate(part_assets)]
    ctx = SceneContext(cfg, intr, list(cfg.parts), part_assets, points, libraries, bin_geom,
                       rim_height(cfg.bin, bin_geom), out, render_intr, warp)
    dump_config(cfg, os.path.join(out, "config_resolved.yaml"))
    write_models(out, cfg.parts, part_assets)
    write_camera_json(out, intr, float(cfg.outputs.get("depth_scale", 0.1)))
    info = {"git": git_hash(REPO), "backend": "isaacsim (legacy-style)", "shard": shard,
            "intrinsics": intr.to_json(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if warp is not None:
        info["render_intrinsics"] = render_intr.to_json()

    usd = omni.usd.get_context()
    usd.new_stage()
    update(5)
    stage = usd.get_stage()
    cam_prim, key_prim, proj_light = build_static_scene(stage, bin_asset, T_w_bin, render_intr)
    # the Kit viewport renders its own view every frame, also headless (Kit log: 'Simulating 2 sensor(s)' - Replicator +
    # ViewportTexture_0): twice the GPU work and RTX resources for nothing. sim.viewport_updates: auto = off headless
    vp_mode = cfg.sim.get("viewport_updates", "auto")
    if vp_mode is False or (vp_mode == "auto" and args.headless):
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            if vp is not None:
                vp.updates_enabled = False
                print("[INFO] viewport rendering off: only the dataset camera is rendered", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] could not turn the viewport rendering off: {e}", flush=True)
    bin_t_op = UsdGeom.Xformable(stage.GetPrimAtPath("/World/bin")).GetOrderedXformOps()[0]  # translate (set_pose)
    update(5)

    rgb_a = depth_a = seg_a = None
    timeline = omni.timeline.get_timeline_interface()
    lights = cfg.get("randomization", {}).get("lights", {}) or {}
    V = int(run.views_per_scene)
    total = int(run.num_scenes)
    viz_every = int(run.get("save_viz_every", 1) or 0)
    subframes = max(int(cfg.camera.get("render_subframes", 2)), 1) + 1
    warmup = int(cfg.camera.get("warmup_frames", 30))  # extra RTX frames before the first view of the process: its first rendered frame is not converged (garbled rgb)
    vib = cfg.loading.get("vibrate", {}) or {}
    vib_every = int(vib.get("every_views", 0) or 0)
    pick_cfg = cfg.loading.get("pick", {}) or {}  # picked at the same moments as the shakes (every_views)
    eye_z0 = float(mount_eye_target(cfg.camera, cfg.bin, bin_geom)[0][2])  # nominal camera height (bin frame)
    writer = AsyncWriter(int(run.get("writer_threads", 2)), max_pending=int(run.get("writer_threads", 2)))

    def save(sc, idx):
        ts = time.monotonic()
        s = process_scene(ctx, sc)
        v0 = s["views"][0] if s and s.get("views") else {}
        print(f"[INFO] scene {idx + 1} saved ({len(sc.views)} views) in {time.monotonic() - ts:.1f}s | "
              f"mask consistency={v0.get('mask_consistency')}, valid depth={v0.get('valid_depth_frac')}, "
              f"topmost=#{v0.get('topmost')}", flush=True)
        if idx == 0 and v0.get("table"):
            print("\n".join(v0["table"]), flush=True)
    print(f"[INFO] generating {total} scenes x {V} views -> {out}", flush=True)
    t0 = time.monotonic()
    done = 0
    for idx in range(total):
        if not simulation_app.is_running():
            break
        ts = time.monotonic()
        timeline.stop()
        update(2)
        counts = sample_counts(cfg.parts, rng, cfg.loading)
        placements, mode = plan_scene(counts, geoms, bin_geom, cfg.loading, rng)
        ups = np.array([geoms[p.type_idx].layer_R.T @ [0.0, 0.0, 1.0] for p in placements])  # up when lying flat
        paths = spawn_parts(stage, placements, part_assets, T_w_bin)
        randomize_materials(stage, paths, placements, rng)
        if seg_a is None:
            # attach after labelled prims exist (as the legacy script does); attached to an unlabelled stage, the
            # instance segmentation of the first scene comes back as a single id
            render_product = rep.create.render_product(CAM_PATH, (render_intr.width, render_intr.height))
            rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
            depth_a = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            seg_a = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False})
            for a in (rgb_a, depth_a, seg_a):
                a.attach(render_product)
        scene_lights = sample_lights(lights, rng)
        apply_lights(stage, key_prim, proj_light, scene_lights)
        update(2)
        rigid = RigidPrim(paths)
        timeline.play()
        update(2)
        t_settle = time.monotonic()
        steps = settle(rigid, len(paths), rng)
        t_settle = time.monotonic() - t_settle

        _PARKED["n"] = 0
        _PARKED["idx"] = set()
        active = remove_out_of_bin(stage, rigid, paths, np.ones(len(paths), bool), T_w_bin, T_bin_w, bin_geom, ups)
        update(1)  # the parked parts' new poses reach the physics readback
        pos, quat = (a.numpy().astype(np.float64) for a in rigid.get_world_poses())
        pile = parts_in_bin(rigid, active, T_bin_w, bin_geom)  # current pile state (camera.random_height)
        n_in = int((active & in_bin(pos @ T_bin_w[:3, :3].T + T_bin_w[:3, 3], bin_geom, 0.0, None)).sum())
        plan = pick_plan(n_in, len([u for u in range(1, V) if vib_every and u % vib_every == 0]), pick_cfg, rng)
        timeline.pause()
        update(2)

        t_cap = time.monotonic()
        views = []
        n_picked = 0
        T_parts = None  # None = the scene poses read above
        for v in range(V):
            if vib_every and v and v % vib_every == 0:
                timeline.play()
                n_picked += pick_top(stage, rigid, paths, active, T_bin_w, T_w_bin, bin_geom,
                                     plan[v // vib_every - 1] if plan else 0)
                vibrate(bin_t_op, T_w_bin, vib, rng, rigid, active, bin_geom)
                # parts that climbed onto the rim (or turned over): hidden + parked; they stay in this scene's GT
                # list, not visible
                n_parked = _PARKED["n"]
                remove_out_of_bin(stage, rigid, paths, active, T_w_bin, T_bin_w, bin_geom, ups)
                if _PARKED["n"] > n_parked:  # newly parked parts: let their poses reach the physics readback
                    update(1)
                apply_lights(stage, key_prim, proj_light, scene_lights, rng, float(lights.get("state_jitter", 0.0)),
                             float(lights.get("state_key_move_m", 0.0)))
                p_v, q_v = (a.numpy().astype(np.float64) for a in rigid.get_world_poses())
                T_parts = [T_bin_w @ make_T(quat_to_mat(q_v[k]), p_v[k]) for k in np.flatnonzero(active)]
                pile = parts_in_bin(rigid, active, T_bin_w, bin_geom)
                timeline.pause()
                update(2)
            cam_dz, cam_xy = camera_offset(pile, eye_z0, bin_geom, rng, (intr.width / 2 / intr.fx, intr.height / 2 / intr.fy))
            T_bin_cam = sample_view(cfg.camera, cfg.bin, bin_geom, rng, jitter=(v > 0 or V == 1), height_offset=cam_dz,
                                    xy_offset=cam_xy)
            T_w_cam = T_w_bin @ T_bin_cam
            T_usd = T_w_cam.copy()
            T_usd[:3, :3] = T_w_cam[:3, :3] @ CV_TO_USD_CAM
            set_pose(cam_prim, T_usd)
            update(subframes + (warmup if idx == 0 and v == 0 else 0))
            for _ in range(10):
                rgb, depth, seg = rgb_a.get_data(), depth_a.get_data(), seg_a.get_data()
                lut = id_to_slot(seg.get("info", {}))
                if lut or not active.any():
                    break
                update(3)  # the first scene's new semantic labels can take a few frames to reach the annotator
            else:
                print(f"[WARN] scene {idx + 1} view {v}: no part ids in the instance segmentation (labels empty)",
                      flush=True)
            if np.asarray(depth).size == 0:
                raise RuntimeError("camera annotators returned no data")
            seg_ids = np.asarray(seg["data"])
            inst = map_ids(seg_ids, lut)
            T_read = world_T(cam_prim)
            T_read[:3, :3] = T_read[:3, :3] @ CV_TO_USD_CAM
            if idx == 0 and v == 0:
                d = np.asarray(depth)
                print(f"[DEBUG] rgb {np.asarray(rgb).shape}, depth finite {np.isfinite(d).mean():.2f}, "
                      f"seg ids {np.unique(seg_ids)[:12].tolist()} -> parts mapped {len(lut)}, "
                      f"camera readback |dpos|={np.abs(T_read[:3, 3] - T_w_cam[:3, 3]).max() * 1000:.2f} mm", flush=True)
            views.append(ViewCapture(np.asarray(rgb), np.asarray(depth, np.float32), inst, T_bin_w @ T_read, T_parts))

        slots = np.flatnonzero(active).tolist()
        sc = SceneCapture(scene_id=(id_base + idx) if id_base is not None else shard * 100000 + offset + idx,
                          mode=mode, slots=slots,
                          types=[placements[k].type_idx for k in slots],
                          T_bin_part=[T_bin_w @ make_T(quat_to_mat(quat[k]), pos[k]) for k in slots],
                          views=views, save_viz=bool(viz_every) and idx % viz_every == 0,
                          seed=int(rng.integers(1 << 31)), removed=int((~active).sum()), picked=n_picked)
        t_cap = time.monotonic() - t_cap
        writer.submit(save, sc, idx)  # labels + files on a writer thread while the next pile settles
        done += 1
        el = time.monotonic() - t0
        print(f"[INFO] scene {idx + 1}/{total}: {len(slots)} parts ({sc.removed} escaped, {n_picked} picked), mode={mode}, "
              f"settle {steps} updates {t_settle:.1f}s, capture {V} views {t_cap:.1f}s, scene {time.monotonic() - ts:.1f}s"
              f" | {done * V} images in {el:.0f}s = {done * V / el:.2f} img/s", flush=True)

    writer.close()
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
