"""Camera model from config: intrinsics at render resolution + viewpoint sampling around the mount."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loading import BinGeom
from .transforms import look_at_cv, make_T


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def to_json(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy, "K": self.K.tolist()}


def intrinsics_from_cfg(cam_cfg) -> Intrinsics:
    """K for the RENDERED image (resolution * render_scale). Pixel centers follow the OpenCV
    convention; scaling uses (c + 0.5) * s - 0.5 so the principal point stays consistent."""
    W, H = cam_cfg["resolution"]
    s = float(cam_cfg.get("render_scale", 1.0))
    w, h = int(round(W * s)), int(round(H * s))
    intr = cam_cfg["intrinsics"]
    if "hfov_deg" in intr:
        fx = (W / 2) / np.tan(np.deg2rad(intr["hfov_deg"]) / 2)
        fy = fx if "vfov_deg" not in intr else (H / 2) / np.tan(np.deg2rad(intr["vfov_deg"]) / 2)
        cx, cy = (W - 1) / 2, (H - 1) / 2
    else:
        fx, fy = float(intr["fx"]), float(intr["fy"])
        cx, cy = float(intr.get("cx", (W - 1) / 2)), float(intr.get("cy", (H - 1) / 2))
    return Intrinsics(w, h, fx * w / W, fy * h / H, (cx + 0.5) * w / W - 0.5, (cy + 0.5) * h / H - 0.5)


def rim_height(bin_cfg, bin_geom: BinGeom) -> float:
    if not bin_cfg.get("mesh") and bin_cfg.get("size_m"):
        return float(bin_cfg["size_m"][2])
    return bin_geom.floor_z + float(bin_geom.inner_size[2])


def mount_eye_target(cam_cfg, bin_cfg, bin_geom: BinGeom):
    """Nominal camera eye/target in the BIN frame."""
    m = cam_cfg.get("mount", {})
    eye = np.array([*m.get("offset_m", [0, 0, 0])[:2], 0.0])
    eye[2] = rim_height(bin_cfg, bin_geom) + float(m.get("height_above_rim_m", 0.5)) + float(m.get("offset_m", [0, 0, 0])[2])
    target = np.asarray(m.get("look_at_m", [0, 0, 0]), float) + np.array([0, 0, bin_geom.floor_z])
    return eye, target


def sample_view(cam_cfg, bin_cfg, bin_geom: BinGeom, rng, jitter: bool = True) -> np.ndarray:
    """4x4 camera-to-bin transform (OpenCV camera axes) for one view."""
    eye, target = mount_eye_target(cam_cfg, bin_cfg, bin_geom)
    up = cam_cfg.get("mount", {}).get("image_up_world", [1, 0, 0])
    roll = 0.0
    if jitter:
        j = cam_cfg.get("pose_jitter", {}) or {}
        eye = eye + rng.uniform(-1, 1, 3) * float(j.get("pos_m", 0.0))
        target = target + np.r_[rng.uniform(-1, 1, 2) * float(j.get("look_at_m", 0.0)), 0.0]
        roll = rng.uniform(-1, 1) * float(j.get("roll_deg", 0.0))
    return make_T(look_at_cv(eye, target, up, roll), eye)
