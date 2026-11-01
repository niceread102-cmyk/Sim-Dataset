"""Reads back a binsim / BOP shard (for visualize.py, verify_poses.py and tests)."""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh
from PIL import Image


class BopScene:
    def __init__(self, shard_dir: str, scene_id: int):
        self.root = shard_dir
        self.dir = os.path.join(shard_dir, "train_pbr", f"{scene_id:06d}")
        self.scene_id = scene_id
        with open(os.path.join(self.dir, "scene_camera.json")) as f:
            self.cam = json.load(f)
        with open(os.path.join(self.dir, "scene_gt.json")) as f:
            self.gt = json.load(f)
        with open(os.path.join(self.dir, "scene_gt_info.json")) as f:
            self.gt_info = json.load(f)

    @property
    def im_ids(self):
        return sorted(int(k) for k in self.cam)

    def K(self, im_id):
        return np.array(self.cam[str(im_id)]["cam_K"]).reshape(3, 3)

    def rgb(self, im_id):
        return np.array(Image.open(os.path.join(self.dir, "rgb", f"{im_id:06d}.png")).convert("RGB"))

    def depth_m(self, im_id):
        d = np.array(Image.open(os.path.join(self.dir, "depth", f"{im_id:06d}.png"))).astype(np.float64)
        return d * float(self.cam[str(im_id)]["depth_scale"]) / 1000.0

    def mask(self, im_id, k, visib=True):
        sub = "mask_visib" if visib else "mask"
        p = os.path.join(self.dir, sub, f"{im_id:06d}_{k:06d}.png")
        if visib and not os.path.exists(p):
            return self.instance_map(im_id) == k
        return np.array(Image.open(p)) > 127

    def instance_map(self, im_id):
        p = os.path.join(self.dir, "mask_visib_inst", f"{im_id:06d}.png")
        if os.path.exists(p):
            if getattr(self, "_inst_cache", (None,))[0] != im_id:
                self._inst_cache = (im_id, np.array(Image.open(p)).astype(np.int32) - 1)
            return self._inst_cache[1]
        n = len(self.gt[str(im_id)])
        inst = None
        for k in range(n):
            m = self.mask(im_id, k)
            if inst is None:
                inst = np.full(m.shape, -1, np.int32)
            inst[m] = k
        return inst

    def grasps(self, im_id):
        p = os.path.join(self.dir, "grasps", f"{im_id:06d}.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    def cam_T_w2c(self, im_id):
        c = self.cam[str(im_id)]
        T = np.eye(4)
        T[:3, :3] = np.array(c["cam_R_w2c"]).reshape(3, 3)
        T[:3, 3] = np.array(c["cam_t_w2c"]) / 1000.0
        return T


def list_scenes(shard_dir: str) -> list[int]:
    tp = os.path.join(shard_dir, "train_pbr")
    return sorted(int(d) for d in os.listdir(tp) if d.isdigit()) if os.path.isdir(tp) else []


def load_models(shard_dir: str) -> dict:
    d = os.path.join(shard_dir, "models")
    with open(os.path.join(d, "models_info.json")) as f:
        info = json.load(f)
    models = {}
    for k in info:
        m = trimesh.load(os.path.join(d, f"obj_{int(k):06d}.ply"), force="mesh")
        m.apply_scale(1e-3)
        models[int(k)] = m
    return models, info
