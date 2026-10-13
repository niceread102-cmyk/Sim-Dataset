"""Ground-truth pose / projection utilities (numpy only).

Convention (same as BOP / OpenCV / the legacy script's output):
  p_cam = R_m2c @ p_model + t_m2c, camera x right, y down, z forward; t stored in mm.
The camera pose is sampled directly in OpenCV axes (binsim/camera.py) and applied to the
simulator in its "ros" convention (identical axes), so no USD->CV axis flip is needed here —
unlike the legacy script's `_USD_TO_CV`, which converted from the USD camera prim's -Z-forward frame.
"""
from __future__ import annotations

import numpy as np

from .lens import distort, pixel_rays
from .transforms import inv_T


def model_to_cam(T_w_cam: np.ndarray, T_w_obj: np.ndarray):
    """-> (R 3x3, t mm (3,)) of the object in the (OpenCV) camera frame."""
    T = inv_T(T_w_cam) @ T_w_obj
    return T[:3, :3], T[:3, 3] * 1000.0


def project(K: np.ndarray, pts_cam: np.ndarray, dist=()) -> np.ndarray:
    """(N,3) camera-frame points (any unit) -> (N,2) pixel coords. dist = lens distortion (binsim/lens.py), () = pinhole."""
    z = np.clip(pts_cam[:, 2:3], 1e-9, None)
    xy = pts_cam[:, :2] / z
    if dist:
        xy = np.stack(distort(xy[:, 0], xy[:, 1], dist), -1)
    return xy * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def backproject(depth_m: np.ndarray, K: np.ndarray, dist=()) -> np.ndarray:
    """(H,W) depth -> (H,W,3) camera-frame points in meters. With a lens (dist) the depth map is in the distorted pixel
    grid: every pixel looks along its own undistorted ray."""
    H, W = depth_m.shape
    if dist:
        from types import SimpleNamespace
        rays = pixel_rays(SimpleNamespace(width=W, height=H, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], dist=dist))
        return np.stack([rays[..., 0] * depth_m, rays[..., 1] * depth_m, depth_m], axis=-1)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) / K[0, 0] * depth_m
    y = (v - K[1, 2]) / K[1, 1] * depth_m
    return np.stack([x, y, depth_m], axis=-1)


def rasterize_silhouette(vertices_cam: np.ndarray, faces: np.ndarray, K: np.ndarray, width: int, height: int,
                         dist=()) -> np.ndarray:
    """Unoccluded silhouette of a mesh (BOP `mask/`), bool (H,W). Uses OpenCV when available,
    else a vectorized numpy edge-function rasterizer over each triangle's bbox. dist = lens distortion: the vertices are
    projected through the lens (triangle edges stay straight, fine for the small triangles of the part meshes)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(faces) == 0 or np.all(vertices_cam[:, 2] <= 0):
        return mask.astype(bool)
    uv = project(K, vertices_cam, dist)
    tri = uv[faces]                                           # (F,3,2)
    front = vertices_cam[faces][:, :, 2].min(axis=1) > 1e-4   # drop triangles behind the camera
    tri = tri[front]
    # cull triangles fully outside the image
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    keep = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < width) & (lo[:, 1] < height)
    tri = tri[keep]
    try:
        import cv2
        pts = np.round(tri * 16).astype(np.int32)             # 4 fractional bits
        # one fillConvexPoly per triangle: fillPoly over many overlapping polygons can leave
        # even-odd holes where triangles overlap
        for p in pts:
            cv2.fillConvexPoly(mask, p, 1, lineType=cv2.LINE_8, shift=4)
        return mask.astype(bool)
    except ImportError:
        pass
    out = mask.astype(bool)
    for t in tri:
        x0, y0 = np.floor(np.maximum(t.min(0), 0)).astype(int)
        x1, y1 = np.ceil(np.minimum(t.max(0), [width - 1, height - 1])).astype(int)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        px, py = xs + 0.0, ys + 0.0
        (ax, ay), (bx, by), (cx, cy) = t
        e0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        e1 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
        e2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        out[y0:y1 + 1, x0:x1 + 1] |= inside
    return out


def bbox_from_mask(mask: np.ndarray):
    """BOP bbox [x, y, w, h] or [-1, -1, -1, -1] if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [-1, -1, -1, -1]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def symmetry_transforms(sym_cfg, extents_m) -> dict:
    """BOP models_info symmetry entries from a {axis, order, flip} spec (translations in mm)."""
    if not sym_cfg:
        return {}
    axis = np.asarray(sym_cfg.get("axis", [0, 0, 1]), float)
    axis /= np.linalg.norm(axis)
    order = sym_cfg.get("order", 1)
    flip = bool(sym_cfg.get("flip", False))
    from .transforms import axis_angle_to_mat
    out = {}
    perp = np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0])
    perp /= np.linalg.norm(perp)
    if order in ("inf", "continuous", float("inf")):
        out["symmetries_continuous"] = [{"axis": axis.tolist(), "offset": [0, 0, 0]}]
        base = [np.eye(3)]
    else:
        n = int(order)
        base = [axis_angle_to_mat(axis, 2 * np.pi * k / n) for k in range(n)]
    discrete = []
    for R in base:
        for f in ([np.eye(3), axis_angle_to_mat(perp, np.pi)] if flip else [np.eye(3)]):
            M = R @ f
            if np.allclose(M, np.eye(3)):
                continue
            T = np.eye(4)
            T[:3, :3] = M
            discrete.append(T.flatten().tolist())
    if discrete:
        out["symmetries_discrete"] = discrete
    return out
