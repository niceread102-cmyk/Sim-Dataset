"""Structured-light depth sensor artifacts (PhoXi-style), applied to the renderer's perfect depth.

Real scanners (unlike the clean `distance_to_image_plane` the legacy script saved) produce:
  - holes where the projector can't illuminate the surface (projector shadow, beside tall parts)
  - no return on surfaces seen at grazing angles
  - invalid / flying pixels along depth discontinuities (removed by the sensor's filters)
  - axial noise growing ~ z^2 and depth quantization
  - nothing outside the scanning range
Invalid pixels are 0, as in PhoXi/BOP depth maps. Every stage is toggled from `camera.noise`.
"""
from __future__ import annotations

import numpy as np

from .poses import backproject


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    out = mask.copy()
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            src = mask[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
            out[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)] |= src
    return out


def surface_normals(pts: np.ndarray) -> np.ndarray:
    """(H,W,3) organized point cloud -> unit normals (camera-facing)."""
    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dy[1:-1] = pts[2:] - pts[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    flip = (n * pts).sum(-1) > 0
    n[flip] *= -1
    return n


def projector_shadow(pts: np.ndarray, valid: np.ndarray, K: np.ndarray, baseline_m, tol_m: float,
                     downsample: int = 2) -> np.ndarray:
    """True where the surface point is occluded as seen from the projector.

    The projector is modeled as a pinhole at `baseline_m` in the camera frame, same orientation
    and intrinsics as the camera. A z-buffer is built from the projector's viewpoint; points that
    are farther than the nearest surface along their projector ray are in shadow.
    """
    H, W = valid.shape
    s = max(int(downsample), 1)
    P = pts - np.asarray(baseline_m, float)          # points in projector frame
    z = P[..., 2]
    ok = valid & (z > 1e-6)
    u = np.floor((P[..., 0] / np.where(ok, z, 1) * K[0, 0] + K[0, 2]) / s).astype(np.int64)
    v = np.floor((P[..., 1] / np.where(ok, z, 1) * K[1, 1] + K[1, 2]) / s).astype(np.int64)
    Wp, Hp = (W + s - 1) // s, (H + s - 1) // s
    ok &= (u >= 0) & (u < Wp) & (v >= 0) & (v < Hp)
    lin = np.where(ok, v * Wp + u, 0)
    zbuf = np.full(Hp * Wp, np.inf)
    np.minimum.at(zbuf, lin[ok], z[ok])
    shadow = np.zeros_like(valid)
    shadow[ok] = z[ok] > zbuf[lin[ok]] + tol_m
    return shadow


def apply_depth_noise(depth_m: np.ndarray, K: np.ndarray, cam_cfg, rng: np.random.Generator) -> np.ndarray:
    depth = np.nan_to_num(depth_m.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = cam_cfg.get("depth_range_m", [0.0, 1e9])
    valid = (depth >= lo) & (depth <= hi)
    ncfg = cam_cfg.get("noise", {}) or {}
    if not ncfg.get("enabled", False):
        return np.where(valid, depth, 0.0)

    pts = backproject(depth, K)
    drop = ~valid

    max_inc = ncfg.get("max_incidence_deg")
    if max_inc:
        n = surface_normals(pts)
        ray = pts / (np.linalg.norm(pts, axis=-1, keepdims=True) + 1e-12)
        cos_inc = np.abs((n * ray).sum(-1))
        drop |= cos_inc < np.cos(np.deg2rad(float(max_inc)))

    proj = ncfg.get("projector", {}) or {}
    if proj.get("enabled", False):
        drop |= projector_shadow(pts, valid, K, proj.get("baseline_m", [0.2, 0, 0]),
                                 float(proj.get("tolerance_mm", 2.0)) / 1000.0, int(proj.get("downsample", 2)))

    grad_thr = ncfg.get("edge_grad_mm")
    if grad_thr:
        gx = np.zeros_like(depth)
        gy = np.zeros_like(depth)
        gx[:, 1:] = np.abs(np.diff(depth, axis=1))
        gy[1:] = np.abs(np.diff(depth, axis=0))
        edges = _dilate(np.maximum(gx, gy) * 1000.0 > float(grad_thr), int(ncfg.get("edge_dilate_px", 1)))
        drop |= edges & (rng.random(depth.shape) < float(ncfg.get("edge_dropout_prob", 1.0)))

    p = float(ncfg.get("random_dropout", 0.0))
    if p > 0:
        drop |= rng.random(depth.shape) < p

    sigma = float(ncfg.get("sigma_mm_at_1m", 0.0)) / 1000.0
    if sigma > 0:
        depth = depth + rng.standard_normal(depth.shape) * sigma * depth ** 2
    q = float(ncfg.get("quantize_mm", 0.0)) / 1000.0
    if q > 0:
        depth = np.round(depth / q) * q
    return np.where(drop, 0.0, depth)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """PhoXi texture is a grayscale intensity image; keep 3 channels for tooling compatibility."""
    g = (rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).clip(0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)
