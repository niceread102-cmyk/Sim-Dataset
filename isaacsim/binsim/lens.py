"""Lens of the real camera: Brown-Conrady distortion (OpenCV k1 k2 p1 p2 k3) + the real, off-centre principal point.

The renderer is an ideal pinhole camera whose optical axis is the image centre. With a `camera.lens` section the renders
are made a little bigger than the image (render_intrinsics) and then bent into the image the real lens would have taken
(LensWarp): every per-pixel output - RGB, depth, instance map - is sampled from the ideal render at the position the lens
maps to each pixel. Everything that projects 3D points (poses.project / backproject / rasterize_silhouette) takes the same
`dist`, so labels and images agree.

dist = (k1, k2, p1, p2, k3) in OpenCV's order and convention: the pixel of a camera-frame point (X, Y, Z) is
    x = X/Z, y = Y/Z, r2 = x^2 + y^2
    xd = x (1 + k1 r2 + k2 r2^2 + k3 r2^3) + 2 p1 x y + p2 (r2 + 2 x^2)
    yd = y (1 + k1 r2 + k2 r2^2 + k3 r2^3) + p1 (r2 + 2 y^2) + 2 p2 x y
    u = fx xd + cx,  v = fy yd + cy
() = ideal pinhole.
"""
from __future__ import annotations

import numpy as np

KEYS = ("k1", "k2", "p1", "p2", "k3")  # OpenCV order


def distortion_from_cfg(lens) -> tuple:
    """(k1, k2, p1, p2, k3) from camera.lens.distortion (mapping with those keys, missing = 0); () when all zero / off."""
    d = (lens or {}).get("distortion") or {}
    t = tuple(float(d.get(k, 0.0)) for k in KEYS)
    return t if any(t) else ()


def distort(x, y, dist):
    """Normalised ideal coordinates -> normalised distorted coordinates."""
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x), y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y


def undistort(xd, yd, dist, iters: int = 40):
    """Inverse of distort (fixed-point iteration like cv2.undistortPoints; converges for any realistic lens)."""
    k1, k2, p1, p2, k3 = dist
    x, y = np.array(xd, np.float64), np.array(yd, np.float64)
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        nx = (xd - (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))) / rad
        ny = (yd - (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)) / rad
        done = max(float(np.abs(nx - x).max()), float(np.abs(ny - y).max())) < 1e-12
        x, y = nx, ny
        if done:
            break
    return x, y


def lens_intrinsics(intr, cam_cfg):
    """K + distortion of the images of the real lens, or None when camera.lens is off (ideal pinhole, centred K).
    camera.lens: {enabled: true, cx, cy (principal point at camera.resolution, px), distortion: {k1, k2, p1, p2, k3}}."""
    from .camera import Intrinsics
    lens = cam_cfg.get("lens") or {}
    if not lens or lens.get("enabled", True) is False:
        return None
    dist = distortion_from_cfg(lens)
    if not dist and "cx" not in lens and "cy" not in lens:
        return None
    W, H = cam_cfg["resolution"]
    cx = (float(lens["cx"]) + 0.5) * intr.width / W - 0.5 if "cx" in lens else intr.cx  # scaled like intrinsics_from_cfg
    cy = (float(lens["cy"]) + 0.5) * intr.height / H - 0.5 if "cy" in lens else intr.cy
    return Intrinsics(intr.width, intr.height, intr.fx, intr.fy, cx, cy, dist)


def render_intrinsics(dst):
    """The ideal pinhole render (same pixel size, principal point at its centre) from which the whole image `dst` of
    the lens can be sampled: the image is surrounded by a margin as wide as the lens pulls content in at the border."""
    from .camera import Intrinsics
    w, h = dst.width, dst.height
    u = np.r_[np.arange(w), np.arange(w), np.zeros(h), np.full(h, w - 1)].astype(np.float64)  # the border pixels
    v = np.r_[np.zeros(w), np.full(w, h - 1), np.arange(h), np.arange(h)].astype(np.float64)
    xu, yu = _rays(u, v, dst)
    su, sv = xu * dst.fx + (w - 1) / 2.0, yu * dst.fy + (h - 1) / 2.0  # where they land in a centred pinhole of the same size
    mx = int(np.ceil(max(0.0, -su.min(), su.max() - (w - 1)))) + 2
    my = int(np.ceil(max(0.0, -sv.min(), sv.max() - (h - 1)))) + 2
    return Intrinsics(w + 2 * mx, h + 2 * my, dst.fx, dst.fy, (w + 2 * mx - 1) / 2.0, (h + 2 * my - 1) / 2.0)


def _rays(u, v, intr):
    """Pixels of an image of this lens -> normalised ideal (undistorted) coordinates x = X/Z, y = Y/Z."""
    xd, yd = (u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy
    return undistort(xd, yd, intr.dist) if intr.dist else (xd, yd)


_RAYS = {}


def pixel_rays(intr):
    """(H, W, 2) normalised ideal coordinates of every pixel of an image of this lens (cached): backprojection of a
    depth map is z * (x, y, 1), exactly what the PhoXi XYZ map holds."""
    key = (intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy, tuple(intr.dist))
    if key not in _RAYS:
        u, v = np.meshgrid(np.arange(intr.width, dtype=np.float64), np.arange(intr.height, dtype=np.float64))
        _RAYS.clear()
        _RAYS[key] = np.stack(_rays(u, v, intr), -1)
    return _RAYS[key]


class LensWarp:
    """Ideal pinhole render `src` -> image `dst` of the real lens (see the module docstring). Maps are built once."""

    def __init__(self, dst, src):
        self.src_shape = (src.height, src.width)
        self.dst_shape = (dst.height, dst.width)
        u, v = np.meshgrid(np.arange(dst.width, dtype=np.float64), np.arange(dst.height, dtype=np.float64))
        xu, yu = _rays(u, v, dst)
        sx, sy = xu * src.fx + src.cx, yu * src.fy + src.cy            # sampling position in the ideal render
        hs, ws = self.src_shape
        self.valid = (sx >= 0) & (sx <= ws - 1) & (sy >= 0) & (sy <= hs - 1)
        x0 = np.clip(np.floor(sx), 0, ws - 1).astype(np.int64)
        y0 = np.clip(np.floor(sy), 0, hs - 1).astype(np.int64)
        x1, y1 = np.minimum(x0 + 1, ws - 1), np.minimum(y0 + 1, hs - 1)
        fx_, fy_ = (sx - x0).clip(0, 1).astype(np.float32), (sy - y0).clip(0, 1).astype(np.float32)
        self._i = [(y0 * ws + x0).ravel(), (y0 * ws + x1).ravel(), (y1 * ws + x0).ravel(), (y1 * ws + x1).ravel()]
        self._w = [((1 - fx_) * (1 - fy_)).ravel(), (fx_ * (1 - fy_)).ravel(), ((1 - fx_) * fy_).ravel(), (fx_ * fy_).ravel()]
        self._nearest = (np.clip(np.rint(sy), 0, hs - 1).astype(np.int64) * ws + np.clip(np.rint(sx), 0, ws - 1).astype(np.int64)).ravel()
        self._invalid = ~self.valid.ravel()

    def _check(self, a):
        if a.shape[:2] != self.src_shape:
            raise ValueError(f"LensWarp expects the {self.src_shape[1]}x{self.src_shape[0]} render, got {a.shape[1]}x{a.shape[0]}")

    def image(self, a):
        """Bilinear (RGB / intensities); outside the render = 0. Same dtype as the input."""
        a = np.asarray(a)
        self._check(a)
        flat = a.reshape(self.src_shape[0] * self.src_shape[1], -1).astype(np.float32)
        out = np.zeros((self._i[0].shape[0], flat.shape[1]), np.float32)
        for i, w in zip(self._i, self._w):
            out += flat[i] * w[:, None]
        out[self._invalid] = 0
        if np.issubdtype(a.dtype, np.integer):
            out = np.clip(np.rint(out), np.iinfo(a.dtype).min, np.iinfo(a.dtype).max)
        return out.astype(a.dtype).reshape(*self.dst_shape, *a.shape[2:])

    def labels(self, a, fill=0):
        """Nearest neighbour (depth, instance ids, masks: values must not mix); outside the render = fill."""
        a = np.asarray(a)
        self._check(a)
        out = a.reshape(-1)[self._nearest]
        out[self._invalid] = fill
        return out.reshape(self.dst_shape)
