"""Ground-truth pose and projection helpers (numpy only, BOP / OpenCV camera convention, t in mm)."""
from __future__ import annotations

import numpy as np

from .lens import distort, pixel_rays
from .transforms import inv_T


def model_to_cam(T_w_cam: np.ndarray, T_w_obj: np.ndarray):
    T = inv_T(T_w_cam) @ T_w_obj
    return T[:3, :3], T[:3, 3] * 1000.0


def project(K: np.ndarray, pts_cam: np.ndarray, dist=()) -> np.ndarray:
    z = np.clip(pts_cam[:, 2:3], 1e-9, None)
    xy = pts_cam[:, :2] / z
    if dist:
        xy = np.stack(distort(xy[:, 0], xy[:, 1], dist), -1)
    return xy * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def backproject(depth_m: np.ndarray, K: np.ndarray, dist=()) -> np.ndarray:
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
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(faces) == 0 or np.all(vertices_cam[:, 2] <= 0):
        return mask.astype(bool)
    uv = project(K, vertices_cam, dist)
    tri = uv[faces]
    front = vertices_cam[faces][:, :, 2].min(axis=1) > 1e-4
    tri = tri[front]
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    keep = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < width) & (lo[:, 1] < height)
    tri = tri[keep]
    try:
        import cv2
        pts = np.round(tri * 16).astype(np.int32)
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
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [-1, -1, -1, -1]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def symmetry_transforms(sym_cfg, extents_m) -> dict:
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
