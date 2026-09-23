"""Turn one settled scene's raw renders + GT poses into dataset files. Simulator-independent,
runs on the writer thread pool (and in tests with synthetic data).

Inputs per view: RGB, perfect depth (m), instance map already resolved to pool-slot indices,
camera pose in the bin frame. Outputs: BOP files, COCO, grasp labels, per-part pose tables, viz.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import grasps as G
from .noise import apply_depth_noise, to_gray
from .poses import backproject, bbox_from_mask, model_to_cam, rasterize_silhouette
from .transforms import inv_T, mat_to_quat, mat_to_rpy
from .viz import bin_panel, part_table_lines
from .writers.bop import SceneWriter
from .writers.coco import CocoScene


@dataclass
class SceneContext:
    """Shared, read-only state for all scenes of a run."""
    cfg: dict
    intr: object                   # camera.Intrinsics
    part_cfgs: list
    assets: list                   # assets.MeshAsset per part type
    points: list                   # (N,3) surface samples per part type (grasp collision)
    libraries: list                # grasps.GraspLibrary per part type
    bin_geom: object
    rim_z: float
    out_root: str

    @property
    def K(self):
        return self.intr.K


@dataclass
class ViewCapture:
    rgb: np.ndarray                # (H,W,3|4) uint8
    depth: np.ndarray              # (H,W) float32, meters (distance to image plane)
    inst: np.ndarray               # (H,W) int32 pool-slot index, -1 = not a part
    T_bin_cam: np.ndarray          # 4x4 camera (OpenCV axes) -> bin frame


@dataclass
class SceneCapture:
    scene_id: int
    mode: str
    slots: list                    # pool slot per part in the scene
    types: list                    # part type index per part
    T_bin_part: list               # 4x4 per part, bin frame
    views: list = field(default_factory=list)
    save_viz: bool = False
    seed: int = 0
    removed: int = 0               # parts that escaped the bin and were removed before capture


def _write_ply(path, pts, cols):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = pts, cols
        f.write(rec.tobytes())


def process_scene(ctx: SceneContext, sc: SceneCapture) -> dict:
    cfg = ctx.cfg
    out_cfg = cfg.get("outputs", {})
    cam_cfg = cfg["camera"]
    gcfg = cfg.get("grasps", {})
    rng = np.random.default_rng(sc.seed)
    K = ctx.K
    W, H = ctx.intr.width, ctx.intr.height
    writer = SceneWriter(ctx.out_root, sc.scene_id, float(out_cfg.get("depth_scale", 0.1)),
                         bool(out_cfg.get("save_clean_depth", False)))
    coco = CocoScene([{"id": i + 1, "name": p["name"], "supercategory": "part"} for i, p in enumerate(ctx.part_cfgs)])
    n = len(sc.slots)
    lut_size = max(sc.slots) + 2 if sc.slots else 1
    slot_to_gt = np.full(lut_size + 1, -1, np.int32)
    for k, s in enumerate(sc.slots):
        slot_to_gt[s] = k

    scene_grasps = []
    if gcfg.get("enabled", True) and n:
        scene_grasps = G.evaluate_scene(sc.T_bin_part, sc.types, ctx.libraries, ctx.points, ctx.bin_geom, ctx.rim_z,
                                        cfg["gripper"], gcfg, rng=rng)
    valid_per_part = np.zeros(n, int)
    for g in scene_grasps:
        valid_per_part[g.part] += int(g.valid)

    summary = {"scene_id": sc.scene_id, "mode": sc.mode, "num_parts": n, "removed": sc.removed, "views": []}
    for im_id, v in enumerate(sc.views):
        depth_clean = np.asarray(v.depth, np.float64)
        depth = apply_depth_noise(depth_clean, K, cam_cfg, rng)
        rgb = np.ascontiguousarray(v.rgb[..., :3])
        if cam_cfg.get("image_mode", "rgb") == "gray":
            rgb = to_gray(rgb)
        inst_slot = np.asarray(v.inst)
        inst = np.where((inst_slot >= 0) & (inst_slot < len(slot_to_gt)), slot_to_gt[np.clip(inst_slot, 0, len(slot_to_gt) - 1)], -1)
        T_cam_bin = inv_T(v.T_bin_cam)

        gts, infos, m_full, m_vis, parts_view = [], [], [], [], []
        visib = np.zeros(n)
        consistency = []
        for k in range(n):
            ti = sc.types[k]
            R, t_mm = model_to_cam(v.T_bin_cam, sc.T_bin_part[k])
            verts = ctx.assets[ti].mesh.vertices @ R.T + t_mm / 1000.0
            full = rasterize_silhouette(verts, ctx.assets[ti].mesh.faces, K, W, H)
            vis = inst == k
            px_all, px_vis = int(full.sum()), int(vis.sum())
            vf = min(px_vis / px_all, 1.0) if px_all else 0.0
            visib[k] = vf
            if px_vis > 200:
                consistency.append(float((vis & full).sum() / px_vis))
            infos.append({"bbox_obj": bbox_from_mask(full), "bbox_visib": bbox_from_mask(vis), "px_count_all": px_all,
                          "px_count_valid": int((vis & (depth > 0)).sum()), "px_count_visib": px_vis,
                          "visib_fract": vf})
            gts.append({"obj_id": ti + 1, "cam_R_m2c": R.flatten().tolist(), "cam_t_m2c": t_mm.tolist()})
            m_full.append(full)
            m_vis.append(vis)
            coco.add_instance(im_id, ti + 1, vis, infos[-1]["bbox_visib"], vf)
            Tb = sc.T_bin_part[k]
            parts_view.append({"label": k, "slot": sc.slots[k], "name": ctx.part_cfgs[ti]["name"], "obj_id": ti + 1,
                               "R": R, "t_mm": t_mm, "R_bin": Tb[:3, :3], "t_bin_mm": Tb[:3, 3] * 1000.0,
                               "visib": vf, "axis_len_m": 0.6 * float(ctx.assets[ti].extents.max()),
                               "n_valid_grasps": int(valid_per_part[k])})
        coco.add_image(im_id, W, H)

        top, top_g = G.pick_topmost(sc.T_bin_part, sc.types, ctx.points, visib, scene_grasps, float(gcfg.get("min_visib", 0.6)))
        top_g_cam = (T_cam_bin @ top_g.T_w) if top_g is not None else None

        # grasp labels in the camera frame
        glist = []
        for g in scene_grasps:
            Tc = T_cam_bin @ g.T_w
            glist.append({"gt_idx": g.part, "obj_id": sc.types[g.part] + 1, "type": cfg["gripper"].get("type"),
                          "cam_R": Tc[:3, :3].flatten().tolist(), "cam_t_mm": (Tc[:3, 3] * 1000).tolist(),
                          "width_m": g.width, "approach_angle_deg": round(g.approach_angle_deg, 2),
                          "valid": bool(g.valid and visib[g.part] >= float(gcfg.get("min_visib", 0.6))),
                          "collision_free": g.valid, "reason": g.reason, "part_visib": round(float(visib[g.part]), 4)})
        with open(os.path.join(writer.dir, "grasps", f"{im_id:06d}.json"), "w") as f:
            json.dump({"topmost_gt_idx": top, "grasps": glist}, f)

        # per-part 6DOF table
        rows = []
        for p in parts_view:
            q_c, q_b = mat_to_quat(p["R"]), mat_to_quat(p["R_bin"])
            rc, rb = mat_to_rpy(p["R"]), mat_to_rpy(p["R_bin"])
            rows.append({"gt_idx": p["label"], "obj_id": p["obj_id"], "name": p["name"], "visib_fract": round(p["visib"], 4),
                         "cam_x_mm": p["t_mm"][0], "cam_y_mm": p["t_mm"][1], "cam_z_mm": p["t_mm"][2],
                         "cam_qw": q_c[0], "cam_qx": q_c[1], "cam_qy": q_c[2], "cam_qz": q_c[3],
                         "cam_roll_deg": rc[0], "cam_pitch_deg": rc[1], "cam_yaw_deg": rc[2],
                         "bin_x_mm": p["t_bin_mm"][0], "bin_y_mm": p["t_bin_mm"][1], "bin_z_mm": p["t_bin_mm"][2],
                         "bin_qw": q_b[0], "bin_qx": q_b[1], "bin_qy": q_b[2], "bin_qz": q_b[3],
                         "bin_roll_deg": rb[0], "bin_pitch_deg": rb[1], "bin_yaw_deg": rb[2],
                         "n_valid_grasps": p["n_valid_grasps"], "is_topmost": p["label"] == top})
        if rows:
            with open(os.path.join(writer.dir, "poses", f"{im_id:06d}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow({k: (f"{v:.5f}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})

        writer.add_view(im_id, rgb, depth, K, T_cam_bin[:3, :3], T_cam_bin[:3, 3] * 1000.0, gts, infos, m_full, m_vis, depth_clean)

        if out_cfg.get("pointcloud_ply", False):
            pts = backproject(depth, K)
            ok = depth > 0
            _write_ply(os.path.join(writer.dir, "depth", f"{im_id:06d}.ply"), pts[ok].astype(np.float32), rgb[ok])

        if sc.save_viz and out_cfg.get("viz", True):
            vdir = os.path.join(ctx.out_root, "viz")
            os.makedirs(vdir, exist_ok=True)
            title = f"scene {sc.scene_id:06d} view {im_id}  mode={sc.mode}  parts={n}  removed={sc.removed}"
            panel = bin_panel(rgb, depth, inst, K, parts_view, top, top_g_cam, title, cam_cfg.get("depth_range_m"))
            panel.save(os.path.join(vdir, f"{sc.scene_id:06d}_{im_id:02d}_bin.jpg"), quality=90)

        summary["views"].append({"im_id": im_id, "topmost": top, "mean_visib": float(visib.mean()) if n else 0.0,
                                 "mask_consistency": float(np.mean(consistency)) if consistency else None,
                                 "valid_depth_frac": float((depth > 0).mean()),
                                 "table": part_table_lines(parts_view) if im_id == 0 else None})
    writer.finalize()
    if out_cfg.get("coco", True):
        writer.write_json("scene_gt_coco.json", coco.data)
    writer.write_json("scene_info.json", {"mode": sc.mode, "slots": sc.slots, "types": sc.types, "removed": sc.removed,
                                          "seed": sc.seed, "num_valid_grasps": int(valid_per_part.sum())})
    return summary
