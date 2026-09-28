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
    _json(os.path.join(root, "camera.json"), {"cx": intr.cx, "cy": intr.cy, "fx": intr.fx, "fy": intr.fy,
                                               "width": intr.width, "height": intr.height, "depth_scale": depth_scale})


class SceneWriter:
    def __init__(self, root: str, scene_id: int, depth_scale: float, save_clean_depth: bool = False):
        self.dir = os.path.join(root, "train_pbr", f"{scene_id:06d}")
        self.depth_scale = depth_scale
        self.save_clean_depth = save_clean_depth
        subdirs = ["rgb", "depth", "grasps", "poses"] + (["depth_clean"] if save_clean_depth else [])
        for s in subdirs:
            os.makedirs(os.path.join(self.dir, s), exist_ok=True)
        self._made = set(subdirs)
        self.scene_camera, self.scene_gt, self.scene_gt_info = {}, {}, {}

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
        save_png(os.path.join(self.dir, "rgb", f"{tag}.png"), rgb)
        save_png(os.path.join(self.dir, "depth", f"{tag}.png"), self._depth_u16(depth_m))
        if self.save_clean_depth and depth_clean_m is not None:
            save_png(os.path.join(self.dir, "depth_clean", f"{tag}.png"), self._depth_u16(depth_clean_m))
        if inst_visib is not None:
            save_png(os.path.join(self._sub("mask_visib_inst"), f"{tag}.png"), (np.asarray(inst_visib) + 1).astype(np.uint16))
        for sub, masks in (("mask", masks_full), ("mask_visib", masks_visib)):
            for k, m in enumerate(masks or []):
                if m is not None:
                    save_png(os.path.join(self._sub(sub), f"{tag}_{k:06d}.png"), m.astype(np.uint8) * 255, 1)
        self.scene_camera[str(im_id)] = {"cam_K": np.asarray(K).flatten().tolist(), "depth_scale": self.depth_scale,
                                         "cam_R_w2c": np.asarray(T_cam_w_R).flatten().tolist(),
                                         "cam_t_w2c": np.asarray(T_cam_w_t_mm).tolist()}
        self.scene_gt[str(im_id)] = gts
        self.scene_gt_info[str(im_id)] = infos

    def write_json(self, name: str, obj, indent: int | None = 1):
        _json(os.path.join(self.dir, name), obj, indent)

    def finalize(self):
        _json(os.path.join(self.dir, "scene_camera.json"), self.scene_camera)
        _json(os.path.join(self.dir, "scene_gt.json"), self.scene_gt)
        _json(os.path.join(self.dir, "scene_gt_info.json"), self.scene_gt_info)
