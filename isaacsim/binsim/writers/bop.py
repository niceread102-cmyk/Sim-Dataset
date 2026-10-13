"""BOP-format writer (https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).

<shard>/
  camera.json, models/{obj_000001.ply, models_info.json}
  train_pbr/<scene_id:06d>/
    rgb/<im:06d>.png  depth/<im:06d>.png (uint16, * depth_scale = mm)  [depth_clean/ ...]
    mask/<im>_<gt:06d>.png  mask_visib/<im>_<gt:06d>.png  (outputs.full_masks / outputs.mask_format: per_part)
    mask_visib_inst/<im>.png  (outputs.mask_format: instance; uint16, pixel = gt index + 1, 0 = no part)
    scene_camera.json  scene_gt.json  scene_gt_info.json  [scene_gt_coco.json]
    grasps/<im>.json   poses/<im>.csv      (binsim extras)
The scene's "world" frame is the BIN frame (bottom center of the bin, z up).
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
from PIL import Image

from ..poses import symmetry_transforms
from ..viz import depth_preview, depth_preview_phoxi, depth_preview_relief


def save_png(path: str, arr: np.ndarray, compress: int = 3):
    Image.fromarray(arr).save(path, compress_level=compress)


def _json(path: str, obj, indent: int | None = 1):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def model_diameter(vertices: np.ndarray, max_pts: int = 3000) -> float:
    v = vertices
    if len(v) > max_pts:
        v = v[np.random.default_rng(0).choice(len(v), max_pts, replace=False)]
    best = 0.0
    for i in range(0, len(v), 500):
        d = np.linalg.norm(v[i:i + 500, None] - v[None], axis=-1).max()
        best = max(best, float(d))
    return best


def write_models(root: str, parts_cfg, assets):
    d = os.path.join(root, "models")
    os.makedirs(d, exist_ok=True)
    info = {}
    for i, (p, a) in enumerate(zip(parts_cfg, assets)):
        obj_id = i + 1
        shutil.copyfile(a.ply_mm_path, os.path.join(d, f"obj_{obj_id:06d}.ply"))
        v = a.mesh.vertices * 1000.0
        lo, hi = v.min(0), v.max(0)
        entry = {"diameter": model_diameter(v), "min_x": lo[0], "min_y": lo[1], "min_z": lo[2],
                 "size_x": hi[0] - lo[0], "size_y": hi[1] - lo[1], "size_z": hi[2] - lo[2], "name": p["name"]}
        entry.update(symmetry_transforms(p.get("symmetry"), a.extents))
        info[str(obj_id)] = {k: (float(v_) if isinstance(v_, (np.floating, float)) else v_) for k, v_ in entry.items()}
    _json(os.path.join(d, "models_info.json"), info)


def write_camera_json(root: str, intr, depth_scale: float):
    cam = {"cx": intr.cx, "cy": intr.cy, "fx": intr.fx, "fy": intr.fy,
           "width": intr.width, "height": intr.height, "depth_scale": depth_scale}
    if getattr(intr, "dist", ()):
        cam["dist"] = list(intr.dist)  # lens distortion k1, k2, p1, p2, k3 (OpenCV): the images are NOT undistorted
    _json(os.path.join(root, "camera.json"), cam)


class SceneWriter:
    def __init__(self, root: str, scene_id: int, depth_scale: float, save_clean_depth: bool = False,
                 folders: dict | None = None, depth_range_m=(0.458, 1.118), depth_vis_style: str = "phoxi", dist=()):
        """folders: per-image folders to write ({"rgb": bool, "depth": bool, ...}, missing = True); a folder is only
        created when something is written into it. depth_vis_style: phoxi = 16-bit per-image min-max like the PhoXi
        sample viewer (viz.depth_preview_phoxi) | fixed = 8-bit, one mapping over depth_range_m (viz.depth_preview).
        dist: lens distortion of the images (k1, k2, p1, p2, k3), written as cam_dist next to cam_K."""
        self.dir = os.path.join(root, "train_pbr", f"{scene_id:06d}")
        self.dist = tuple(dist)
        self.depth_scale = depth_scale
        self.save_clean_depth = save_clean_depth
        self.folders = dict(folders or {})
        self.depth_range_m = tuple(float(x) for x in depth_range_m)
        if depth_vis_style not in ("phoxi", "fixed"):
            raise ValueError(f"outputs.depth_vis_style must be phoxi or fixed, got {depth_vis_style!r}")
        self.depth_vis_style = depth_vis_style
        os.makedirs(self.dir, exist_ok=True)
        self._made = set()
        self.scene_camera, self.scene_gt, self.scene_gt_info = {}, {}, {}

    def enabled(self, folder: str) -> bool:
        return bool(self.folders.get(folder, True))

    def _sub(self, name: str) -> str:
        d = os.path.join(self.dir, name)
        if name not in self._made:
            os.makedirs(d, exist_ok=True)
            self._made.add(name)
        return d

    def _depth_u16(self, depth_m):
        return np.clip(np.round(depth_m * 1000.0 / self.depth_scale), 0, 65535).astype(np.uint16)

    def add_view(self, im_id: int, rgb, depth_m, K, T_cam_w_R, T_cam_w_t_mm, gts, infos, masks_full, masks_visib,
                 depth_clean_m=None, inst_visib=None):
        """masks_full / masks_visib: per-gt (H, W) bool masks (None or None entries = not written);
        inst_visib: (H, W) gt index per pixel (-1 = no part), written as one uint16 PNG (gt index + 1)."""
        tag = f"{im_id:06d}"
        if self.enabled("rgb"):
            save_png(os.path.join(self._sub("rgb"), f"{tag}.png"), rgb)
        d16 = self._depth_u16(depth_m)
        if self.enabled("depth"):
            save_png(os.path.join(self._sub("depth"), f"{tag}.png"), d16)
        if self.folders.get("depth_vis", False):  # preview of exactly the stored values (off unless asked for)
            d_m = d16 * self.depth_scale / 1000.0
            vis = depth_preview_phoxi(d_m) if self.depth_vis_style == "phoxi" else depth_preview(d_m, *self.depth_range_m)
            save_png(os.path.join(self._sub("depth_vis"), f"{tag}.png"), vis)
        if self.folders.get("depth_relief", False):  # local-contrast preview: small height differences (relief of the parts) visible
            save_png(os.path.join(self._sub("depth_relief"), f"{tag}.png"), depth_preview_relief(d16 * self.depth_scale / 1000.0))
        if self.save_clean_depth and depth_clean_m is not None:
            save_png(os.path.join(self._sub("depth_clean"), f"{tag}.png"), self._depth_u16(depth_clean_m))
        if inst_visib is not None and self.enabled("mask_visib_inst"):
            save_png(os.path.join(self._sub("mask_visib_inst"), f"{tag}.png"), (np.asarray(inst_visib) + 1).astype(np.uint16))
        for sub, masks in (("mask", masks_full), ("mask_visib", masks_visib)):
            for k, m in enumerate(masks or []):
                if m is not None:
                    save_png(os.path.join(self._sub(sub), f"{tag}_{k:06d}.png"), m.astype(np.uint8) * 255, 1)
        self.scene_camera[str(im_id)] = {"cam_K": np.asarray(K).flatten().tolist(), "depth_scale": self.depth_scale,
                                         "cam_R_w2c": np.asarray(T_cam_w_R).flatten().tolist(),
                                         "cam_t_w2c": np.asarray(T_cam_w_t_mm).tolist()}
        if self.dist:
            self.scene_camera[str(im_id)]["cam_dist"] = list(self.dist)
        self.scene_gt[str(im_id)] = gts
        self.scene_gt_info[str(im_id)] = infos

    def write_json(self, name: str, obj, indent: int | None = 1):
        _json(os.path.join(self.dir, name), obj, indent)

    def finalize(self):
        _json(os.path.join(self.dir, "scene_camera.json"), self.scene_camera)
        _json(os.path.join(self.dir, "scene_gt.json"), self.scene_gt)
        _json(os.path.join(self.dir, "scene_gt_info.json"), self.scene_gt_info)
