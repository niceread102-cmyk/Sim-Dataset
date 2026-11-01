"""Crates of parts stacked on the pallets of a map (scene.pallets): world poses of every crate and part (numpy only)."""
from __future__ import annotations

import math

import numpy as np

from .transforms import rpy_to_mat

DEFAULTS = {
    "layers": 6,
    "grids": [[2, 3], [3, 2]],
    "grid_mode": "layer",
    "parts_per_crate": 10,
    "dynamic_layers": 1,
    "layer_gap_m": 0.0005,
    "margin_m": 0.004,
    "tilt_deg": None,
    "parallel": False,
    "allow_flipped": True,
}


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _tilt(az: float, angle: float) -> np.ndarray:
    ax, ay = math.cos(az), math.sin(az)
    K = np.array([[0.0, 0.0, ay], [0.0, 0.0, -ax], [-ay, ax, 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _quad(xy, yaw: float, hx: float, hy: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    return np.asarray(xy, float) + d @ np.array([[c, s], [-s, c]])


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True


def _contains(quad: np.ndarray, pt) -> bool:
    cross = [(quad[(i + 1) % 4] - quad[i])[0] * (pt[1] - quad[i][1]) - (quad[(i + 1) % 4] - quad[i])[1] * (pt[0] - quad[i][0]) for i in range(4)]
    return all(c >= 0 for c in cross) or all(c <= 0 for c in cross)


def _air_place(rng, hx, hy, lift, inner, max_top, placed_air, gap: float = 0.012, room: float = 0.6, tries: int = 400):
    for attempt in range(tries):
        xy = rng.uniform([-inner[0], -inner[1]], [inner[0], inner[1]]) * 0.9
        yaw = float(rng.uniform(0.0, 2.0 * math.pi))
        q = _quad(xy, yaw, hx, hy)
        under = [top for pq, top in placed_air if _overlap(q, pq)]
        if under and attempt < 100:
            continue
        base = (max(under) + gap) if under else max_top + gap
        if base + 2.0 * lift > max_top + room:
            continue
        placed_air.append((q, base + 2.0 * lift))
        return xy, yaw, base
    xy, yaw = rng.uniform([-inner[0], -inner[1]], [inner[0], inner[1]]) * 0.9, float(rng.uniform(0.0, 2.0 * math.pi))
    placed_air.append((_quad(xy, yaw, hx, hy), max_top + room + 2.0 * lift))
    return xy, yaw, max_top + room


def fill_crate(rng, n: int, foot, height: float, inner, floor_z: float, max_top: float, tries: int = 1200, floor_tries: int = 300, tilt=None, allow_flipped: bool = True):
    hx, hy = foot[0] / 2, foot[1] / 2
    placed = []
    placed_air = []
    out = []
    radius = 0.5 * max(foot)
    for _ in range(n):
        flip = bool(rng.integers(2)) and allow_flipped
        th =math.radians(float(rng.uniform(*tilt["flipped" if flip else "upright"]))) if tilt else 0.0
        az = float(rng.uniform(0.0, 2.0 * math.pi))
        lift = 0.5 * height * math.cos(th) + radius * math.sin(th)
        done = None
        for attempt in range(tries):
            xy = rng.uniform([-inner[0], -inner[1]], [inner[0], inner[1]])
            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
            q = _quad(xy, yaw, hx, hy)
            if np.abs(q[:, 0]).max() > inner[0] or np.abs(q[:, 1]).max() > inner[1]:
                continue
            under = [(top, pq) for pq, top in placed if _overlap(q, pq)]
            if under:
                if attempt < floor_tries:
                    continue
                top, pq = max(under, key=lambda t: t[0])
                if attempt < tries - 200 and not _contains(pq, xy):
                    continue
                base = top
            else:
                base = floor_z
            if base + 2.0 * lift > max_top:
                continue
            done = (xy, yaw, base)
            break
        air = done is None
        if air:
            done = _air_place(rng, hx, hy, lift, inner, max_top, placed_air)
        xy, yaw, base = done
        if not air:
            placed.append((_quad(xy, yaw, hx, hy), base + 2.0 * lift))
        out.append((float(xy[0]), float(xy[1]), base + lift, yaw, flip, th, az))
    return out


def layout(grid, crate_xy, gap: float):
    nx, ny = grid
    ex, ey = crate_xy
    best = None
    for yaw, (cx, cy) in ((0.0, (ex, ey)), (math.pi / 2, (ey, ex))):
        wx, wy = nx * cx + (nx - 1) * gap, ny * cy + (ny - 1) * gap
        if best is None or max(wx, wy) < max(best[2]) - 1e-9:
            best = (yaw, (cx, cy), (wx, wy))
    yaw, (cx, cy), (wx, wy) = best
    pts = [((i - (nx - 1) / 2) * (cx + gap), (j - (ny - 1) / 2) * (cy + gap)) for i in range(nx) for j in range(ny)]
    return yaw, pts, (wx, wy)


def pick_grids(cfg: dict, rng) -> list[list[tuple[int, int]]]:
    grids = [tuple(int(v) for v in g) for g in cfg["grids"]]
    mode, layers = str(cfg["grid_mode"]), int(cfg["layers"])
    out = []
    for _ in cfg["items"]:
        if mode == "pallet":
            out.append([grids[int(rng.integers(len(grids)))]] * layers)
        elif mode == "alternate":
            out.append([grids[l % len(grids)] for l in range(layers)])
        elif mode == "layer":
            out.append([grids[int(rng.integers(len(grids)))] for _ in range(layers)])
        else:
            raise ValueError(f"pallets.grid_mode must be layer | pallet | alternate, got {mode!r}")
    return out


def crate_count(counts, pallet: int, layer: int, idx: int, grid) -> int:
    if not isinstance(counts, (list, tuple)):
        return int(counts)
    nx, ny = grid
    col, row = divmod(idx, ny)
    try:
        return int(counts[pallet][layer][row][col])
    except (IndexError, TypeError):
        raise ValueError(f"pallets.parts_per_crate has no number for pallet {pallet}, layer {layer}, row {row}, col {col}: it needs a list "
                         f"[pallet][layer][row (y)][col (x)] (layer {layer} of pallet {pallet} is a grid of {nx} cols x {ny} rows)") from None


def plan_pallets(cfg: dict, part_ext, part_layer_rpy_deg, rng, part_centre=(0.0, 0.0, 0.0)) -> dict:
    cfg = {**DEFAULTS, **cfg}
    crate = cfg["crate"]
    cx, cy, cz = (float(v) for v in crate["size_m"])
    gap = float(crate.get("gap_m", 0.01))
    inner = [float(v) / 2 - float(cfg["margin_m"]) for v in crate["inner_size_m"]]
    floor_z = float(crate["floor_z_m"])
    layers = int(cfg["layers"])
    Rb = rpy_to_mat(part_layer_rpy_deg or [0, 0, 0])
    ext = np.abs(Rb) @ np.asarray(part_ext, float)
    foot, height = (float(ext[0]), float(ext[1])), float(ext[2])
    tilt = None if cfg.get("parallel") else cfg.get("tilt_deg")
    if tilt is not None and not isinstance(tilt, dict):
        tilt = {"upright": tilt, "flipped": tilt}
    if tilt:
        tilt = {key: [float(v) for v in tilt[key]] for key in ("upright", "flipped")}
    grids = pick_grids(cfg, rng)
    crates, parts = [], []
    for k, item in enumerate(cfg["items"]):
        if not item.get("enabled", True):
            continue
        px, py, pz = (float(v) for v in item["pos_m"])
        pyaw = math.radians(float(item.get("yaw_deg", 0.0)))
        for layer in range(layers):
            cyaw, pts, _ = layout(grids[k][layer], (cx, cy), gap)
            z0 = pz + layer * (cz + float(cfg["layer_gap_m"]))
            dynamic = layer >= layers - int(cfg["dynamic_layers"])
            for idx, (u, v) in enumerate(pts):
                Rc = _rz(pyaw + cyaw)
                pos = np.array([px, py, z0]) + _rz(pyaw) @ np.array([u, v, 0.0])
                crates.append({"pallet": k, "layer": layer, "idx": idx, "pos": pos, "R": Rc, "grid": grids[k][layer], "dynamic": dynamic})
                for j, (x, y, z, yaw, flip, th, az) in enumerate(fill_crate(rng, crate_count(cfg["parts_per_crate"], k, layer, idx, grids[k][layer]), foot, height, inner, floor_z, cz - 0.01, tilt=tilt or None, allow_flipped=bool(cfg["allow_flipped"]))):
                    Rp = Rc @ _rz(yaw) @ _tilt(az, th) @ (_rx(math.pi) if flip else np.eye(3)) @ Rb
                    parts.append({"pallet": k, "layer": layer, "crate": idx, "j": j, "pos": pos + Rc @ np.array([x, y, z]) - Rp @ np.asarray(part_centre, float), "R": Rp, "dynamic": dynamic})
    return {"crates": crates, "parts": parts, "grids": grids}
