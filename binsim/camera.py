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


def centred(intr: Intrinsics) -> Intrinsics:
    """The same camera with the principal point at the image centre. The USD camera (focal length + apertures) of the
    Isaac Sim renderer always projects the optical axis to the centre, whatever cx/cy the config gives, so this is the
    K of the rendered images (checked: labels re-rendered with it match the saved masks 100 %, the config K 63 %)."""
    return Intrinsics(intr.width, intr.height, intr.fx, intr.fy, (intr.width - 1) / 2.0, (intr.height - 1) / 2.0)


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
    """Nominal camera eye, look-at target and image-up direction in the BIN frame.

    `mount.T_bin_cam` (measured camera pose, e.g. from hand-eye calibration: 4x4 row-major, meters, OpenCV camera
    axes, see scripts/camera_mount_from_calib.py) overrides height_above_rim_m / offset_m / look_at_m."""
    m = cam_cfg.get("mount", {})
    if m.get("T_bin_cam") is not None:
        T = np.asarray(m["T_bin_cam"], float).reshape(4, 4)
        eye, z = T[:3, 3].copy(), T[:3, 2]
        s = (bin_geom.floor_z - eye[2]) / z[2] if abs(z[2]) > 1e-6 else 1.0  # where the optical axis meets the floor
        return eye, eye + z * s, -T[:3, 1]
    eye = np.array([*m.get("offset_m", [0, 0, 0])[:2], 0.0])
    eye[2] = rim_height(bin_cfg, bin_geom) + float(m.get("height_above_rim_m", 0.5)) + float(m.get("offset_m", [0, 0, 0])[2])
    target = np.asarray(m.get("look_at_m", [0, 0, 0]), float) + np.array([0, 0, bin_geom.floor_z])
    return eye, target, np.asarray(m.get("image_up_world", [1, 0, 0]), float)


def sample_view(cam_cfg, bin_cfg, bin_geom: BinGeom, rng, jitter: bool = True, height_offset: float = 0.0,
                xy_offset=(0.0, 0.0)) -> np.ndarray:
    """4x4 camera-to-bin transform (OpenCV camera axes) for one view. height_offset (m) moves the camera eye up/down,
    xy_offset (m, bin frame) moves eye and look-at point sideways (camera.follow_pile)."""
    eye, target, up = mount_eye_target(cam_cfg, bin_cfg, bin_geom)
    shift = np.array([float(xy_offset[0]), float(xy_offset[1]), 0.0])
    eye = eye + shift + np.array([0.0, 0.0, float(height_offset)])
    target = target + shift
    roll = 0.0
    if jitter:
        j = cam_cfg.get("pose_jitter", {}) or {}
        eye = eye + rng.uniform(-1, 1, 3) * float(j.get("pos_m", 0.0))
        target = target + np.r_[rng.uniform(-1, 1, 2) * float(j.get("look_at_m", 0.0)), 0.0]
        roll = rng.uniform(-1, 1) * float(j.get("roll_deg", 0.0))
    return make_T(look_at_cv(eye, target, up, roll), eye)
