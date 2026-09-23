"""Tiny CPU z-buffer renderer standing in for Isaac in tests: depth + instance ids + flat RGB."""
import numpy as np

from binsim.poses import project


def render(meshes_cam, K, W, H, floor=None):
    """meshes_cam: list of (vertices_cam (V,3) m, faces (F,3)). floor: (n_cam, d) plane n.x + d = 0.
    Returns depth (H,W) m (inf where empty), inst (H,W) int (-1 background), rgb (H,W,3)."""
    depth = np.full((H, W), np.inf)
    inst = np.full((H, W), -1, np.int32)
    if floor is not None:
        n, d = floor
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
        denom = rays @ n
        z = np.where(np.abs(denom) > 1e-9, -d / denom, np.inf)
        depth = np.where(z > 0, z, np.inf)
    for idx, (V, F) in enumerate(meshes_cam):
        uv = project(K, V)
        for f in F:
            t = uv[f]
            zs = V[f, 2]
            if np.any(zs <= 0):
                continue
            x0, y0 = np.floor(np.maximum(t.min(0), 0)).astype(int)
            x1, y1 = np.ceil(np.minimum(t.max(0), [W - 1, H - 1])).astype(int)
            if x1 < x0 or y1 < y0:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            (ax, ay), (bx, by), (cx, cy) = t
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-12:
                continue
            w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
            w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            # perspective-correct depth: interpolate 1/z
            z = 1.0 / (w0 / zs[0] + w1 / zs[1] + w2 / zs[2])
            sub_d = depth[y0:y1 + 1, x0:x1 + 1]
            closer = inside & (z < sub_d)
            sub_d[closer] = z[closer]
            inst[y0:y1 + 1, x0:x1 + 1][closer] = idx
    rgb = np.full((H, W, 3), 90, np.uint8)
    rng = np.random.default_rng(0)
    cols = rng.integers(60, 255, (max(len(meshes_cam), 1), 3))
    rgb[inst >= 0] = cols[inst[inst >= 0]]
    depth[~np.isfinite(depth)] = 0.0
    return depth, inst, rgb
