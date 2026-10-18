"""Crates of parts stacked on pallets: where every crate and every part goes. numpy only (no Isaac Sim), the poses are in the world.

    pallets:                        scene.pallets of configs/map/map.yaml
      items: [{pos_m: [x, y, deck_top_z], yaw_deg}, ...]      the pallets of the map that get crates
      crate: {size_m, inner_size_m, floor_z_m, gap_m, usd}    the crate (outer size x y z, usable floor x y, top of its floor)
      layers, grids, grid_mode, parts_per_crate, part, dynamic_layers, seed

Every pallet gets `layers` layers of crates. A layer is a grid of nx x ny crates (grids: [[2, 3], [3, 2]] = one of them) in the frame of the
pallet (x, y, yaw_deg of its item); the crates stand with their long side along the axis that has fewer of them, so the block of a layer is
compact (2 x 3 or 3 x 2 crates of 0.24 x 0.38 m = 0.76 x 0.72 m). grid_mode: `layer` = a new random grid for every layer, `pallet` = one random
grid per pallet, `alternate` = the grids in turn, layer by layer.
`tilt_deg` ({upright: [lo, hi], flipped: [lo, hi]}, deg; default none = flat) tilts every part from lying flat about a horizontal axis of a random
direction, so that a pile does not consist of perfectly level plates (a flipped one stands on its hub: it leans before the physics settles it).
Every crate holds `parts_per_crate` parts (one number for all crates, or a list [pallet][layer][row = y][col = x] of numbers: one per crate, in
the frame of the pallet, layer 0 = the bottom layer, e.g. a different count for the top layer): each part is dropped straight down at a random place and yaw, lying flat (its thinnest side up or
down, 50 % each), on a free place of the floor if there is one, else on the parts it overlaps (its centre over the plate below).
Parts in the top `dynamic_layers` layers are meant to move (the robot picks them), the others are only decoration under the crates above.
"""
from __future__ import annotations

import math

import numpy as np

from .transforms import rpy_to_mat

DEFAULTS = {
    "layers": 6,
    "grids": [[2, 3], [3, 2]],
    "grid_mode": "layer",           # layer | pallet | alternate
    "parts_per_crate": 10,
    "dynamic_layers": 1,
    "layer_gap_m": 0.0005,          # between a crate and the crate below it
    "margin_m": 0.004,              # parts keep this far from the walls of the crate
    "tilt_deg": None,               # None = every part lies flat; {upright: [lo, hi], flipped: [lo, hi]} = a random tilt (deg) from lying flat, about a horizontal axis
}


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _tilt(az: float, angle: float) -> np.ndarray:
    """Rotation by angle (rad) about the horizontal axis (cos az, sin az, 0)"""
    ax, ay = math.cos(az), math.sin(az)
    K = np.array([[0.0, 0.0, ay], [0.0, 0.0, -ax], [-ay, ax, 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _quad(xy, yaw: float, hx: float, hy: float) -> np.ndarray:
    """Corners (4, 2) of the rectangle of half sizes hx, hy at xy, turned by yaw"""
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    return np.asarray(xy, float) + d @ np.array([[c, s], [-s, c]])


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Do two convex quadrilaterals overlap (separating axis test)?"""
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


def fill_crate(rng, n: int, foot, height: float, inner, floor_z: float, max_top: float, tries: int = 1200, floor_tries: int = 300, tilt=None):
    """n parts dropped into a crate. foot = (fx, fy) footprint of a part lying flat, height = its thickness (m); inner = (hx, hy) half size of
    the usable floor, floor_z = top of the floor, max_top = the highest a part may reach (rim of the crate). A part takes a free place on the
    floor if it finds one (floor_tries), else it lies on the parts below it. tilt = {upright: [lo, hi], flipped: [lo, hi]} (deg) tilts every part
    from lying flat about a horizontal axis of a random direction (its centre is lifted so that its edge rests on the floor / the part below, a
    disc of the radius of the foot; the physics settles it from there). Returns [(x, y, z centre, yaw, flipped, tilt rad, tilt direction rad)]."""
    hx, hy = foot[0] / 2, foot[1] / 2
    placed = []                                              # (quad, top z)
    out = []
    radius = 0.5 * max(foot)
    for _ in range(n):
        flip = bool(rng.integers(2))
        th = math.radians(float(rng.uniform(*tilt["flipped" if flip else "upright"]))) if tilt else 0.0
        az = float(rng.uniform(0.0, 2.0 * math.pi))
        lift = 0.5 * height * math.cos(th) + radius * math.sin(th)       # centre above the support (= height / 2 when flat)
        done = None
        for attempt in range(tries):
            xy = rng.uniform([-inner[0], -inner[1]], [inner[0], inner[1]])
            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
            q = _quad(xy, yaw, hx, hy)
            if np.abs(q[:, 0]).max() > inner[0] or np.abs(q[:, 1]).max() > inner[1]:
                continue
            under = [(top, pq) for pq, top in placed if _overlap(q, pq)]
            if under:
                if attempt < floor_tries:                    # first look for a free place on the floor
                    continue
                top, pq = max(under, key=lambda t: t[0])
                if attempt < tries - 200 and not _contains(pq, xy):   # the centre has to be over the plate below (relaxed in the last tries)
                    continue
                base = top
            else:
                base = floor_z
            if base + 2.0 * lift > max_top:
                continue
            done = (xy, yaw, base)
            break
        if done is None:                                     # nothing found: overlapping is better than losing the part
            xy = rng.uniform([-inner[0], -inner[1]], [inner[0], inner[1]]) * 0.5
            done = (xy, float(rng.uniform(0.0, 2.0 * math.pi)), floor_z)
        xy, yaw, base = done
        placed.append((_quad(xy, yaw, hx, hy), base + 2.0 * lift))
        out.append((float(xy[0]), float(xy[1]), base + lift, yaw, flip, th, az))
    return out


def layout(grid, crate_xy, gap: float):
    """Crates of one layer in the frame of the pallet. Returns (crate yaw relative to the pallet, [(x, y), ...], (width x, width y)):
    the crate stands with its long side along the axis with fewer crates, unless the other way gives the smaller block."""
    nx, ny = grid
    ex, ey = crate_xy                                        # crate size along its own x and y (y is the long side)
    best = None
    for yaw, (cx, cy) in ((0.0, (ex, ey)), (math.pi / 2, (ey, ex))):
        wx, wy = nx * cx + (nx - 1) * gap, ny * cy + (ny - 1) * gap
        if best is None or max(wx, wy) < max(best[2]) - 1e-9:
            best = (yaw, (cx, cy), (wx, wy))
    yaw, (cx, cy), (wx, wy) = best
    pts = [((i - (nx - 1) / 2) * (cx + gap), (j - (ny - 1) / 2) * (cy + gap)) for i in range(nx) for j in range(ny)]
    return yaw, pts, (wx, wy)


def pick_grids(cfg: dict, rng) -> list[list[tuple[int, int]]]:
    """The grid of every layer of every pallet: [[(nx, ny), ...] per layer] per pallet"""
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
    """Number of parts of one crate: `counts` is parts_per_crate, one number or the nested list [pallet][layer][row][col] (idx = col * rows + row)"""
    if not isinstance(counts, (list, tuple)):
        return int(counts)
    nx, ny = grid
    col, row = divmod(idx, ny)
    try:
        return int(counts[pallet][layer][row][col])
    except (IndexError, TypeError):
        raise ValueError(f"pallets.parts_per_crate has no number for pallet {pallet}, layer {layer}, row {row}, col {col}: it needs a list "
                         f"[pallet][layer][row (y)][col (x)] (layer {layer} of pallet {pallet} is a grid of {nx} cols x {ny} rows)") from None


def plan_pallets(cfg: dict, part_ext, part_layer_rpy_deg, rng) -> dict:
    """Poses of all crates and parts. part_ext = size of the part model (m, as the part config's mesh), part_layer_rpy_deg = its layer_rpy_deg
    (thinnest side up). Returns {"crates": [...], "parts": [...], "grids": ...}; every entry has the world position `pos` and rotation matrix `R`,
    its pallet / layer index and `dynamic` (a part in the top dynamic_layers layers)."""
    cfg = {**DEFAULTS, **cfg}
    crate = cfg["crate"]
    cx, cy, cz = (float(v) for v in crate["size_m"])
    gap = float(crate.get("gap_m", 0.01))
    inner = [float(v) / 2 - float(cfg["margin_m"]) for v in crate["inner_size_m"]]
    floor_z = float(crate["floor_z_m"])
    layers = int(cfg["layers"])
    Rb = rpy_to_mat(part_layer_rpy_deg or [0, 0, 0])         # the part lying flat, thinnest side up
    ext = np.abs(Rb) @ np.asarray(part_ext, float)           # its size in that pose (exact for turns of 90 degrees)
    foot, height = (float(ext[0]), float(ext[1])), float(ext[2])
    tilt = cfg.get("tilt_deg")
    if tilt is not None and not isinstance(tilt, dict):                  # one range for upright and flipped parts
        tilt = {"upright": tilt, "flipped": tilt}
    if tilt:
        tilt = {key: [float(v) for v in tilt[key]] for key in ("upright", "flipped")}
    grids = pick_grids(cfg, rng)
    crates, parts = [], []
    for k, item in enumerate(cfg["items"]):
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
                for j, (x, y, z, yaw, flip, th, az) in enumerate(fill_crate(rng, crate_count(cfg["parts_per_crate"], k, layer, idx, grids[k][layer]), foot, height, inner, floor_z, cz - 0.01, tilt=tilt or None)):
                    Rp = Rc @ _rz(yaw) @ _tilt(az, th) @ (_rx(math.pi) if flip else np.eye(3)) @ Rb
                    parts.append({"pallet": k, "layer": layer, "crate": idx, "j": j, "pos": pos + Rc @ np.array([x, y, z]), "R": Rp, "dynamic": dynamic})
    return {"crates": crates, "parts": parts, "grids": grids}
