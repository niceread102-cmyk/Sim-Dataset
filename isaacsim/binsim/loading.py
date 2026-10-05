"""Initial part placement for a bin (pure numpy, simulator-independent).

All positions are in the BIN frame: origin at the bin's bottom center, +Z up, inner volume
x in [-ix/2, ix/2], y in [-iy/2, iy/2], z from `floor_z` up. Physics does the rest (drop + settle).

Three modes:
  random : parts spawn above the floor with no initial overlap (vertical-cylinder bounds),
           then fall into a cluster.
  layers : ordered grid stacks in each part's `layer_rpy_deg` orientation with jitter,
           partially filled top layer; overflow parts are dropped on top.
  stack  : parts placed (not dropped) in loose layers, each in its `layer_rpy_deg` orientation
           (e.g. front side up) with a random yaw at a random free spot; a full layer starts the
           next one on top. Upper parts bridge the gaps below and settle overlapping and tilted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import as_range, count_range
from .transforms import axis_angle_to_mat, mat_to_quat, random_quaternions, rpy_to_mat


@dataclass
class PartGeom:
    name: str
    extents: np.ndarray          # (3,) bbox size in the part frame, meters (model is centered)
    radius: float                # bounding-sphere radius, meters
    layer_R: np.ndarray = field(default_factory=lambda: np.eye(3))  # resting orientation for layers
    mesh: object = None          # trimesh in the part frame (meters), optional: stack mode rests parts on its shape
    _profiles: dict = field(default_factory=dict, repr=False)

    def layer_extents(self) -> np.ndarray:
        """Axis-aligned extents after applying layer_R."""
        return np.abs(self.layer_R) @ self.extents

    def profile(self, cell: float, flip: bool = False):
        """The part lying in its layer orientation (turned over about x if `flip`) as points on a `cell` grid of its
        xy plane: (xy (N, 2), underside height (N,), top height (N,), height of the part origin), heights above the
        part's lowest point. From the mesh surface (sampled once) if there is one, else from the bounding box."""
        key = (cell, flip)
        if key not in self._profiles:
            R = self.layer_R @ (axis_angle_to_mat([1, 0, 0], np.pi) if flip else np.eye(3))
            if self.mesh is not None:
                import trimesh
                pts = trimesh.sample.sample_surface(self.mesh, 600000, seed=0)[0] @ R.T
            else:
                e = np.abs(R) @ self.extents / 2
                gx, gy = np.meshgrid(np.arange(-e[0], e[0] + 1e-9, cell / 2), np.arange(-e[1], e[1] + 1e-9, cell / 2))
                xy = np.c_[gx.ravel(), gy.ravel()]
                pts = np.vstack([np.c_[xy, np.full(len(xy), -e[2])], np.c_[xy, np.full(len(xy), e[2])]])
            ij, inv = np.unique(np.floor(pts[:, :2] / cell).astype(int), axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            lo, hi = np.full(len(ij), np.inf), np.full(len(ij), -np.inf)
            np.minimum.at(lo, inv, pts[:, 2])
            np.maximum.at(hi, inv, pts[:, 2])
            z0 = lo.min()
            self._profiles[key] = ((ij + 0.5) * cell, lo - z0, hi - z0, -z0)
        return self._profiles[key]


@dataclass
class BinGeom:
    inner_size: np.ndarray       # (3,) usable inner volume, meters
    floor_z: float               # inner floor height in the bin frame


@dataclass
class Placement:
    type_idx: int
    pos: np.ndarray              # (3,) bin frame
    quat: np.ndarray             # (4,) wxyz, bin frame


def sample_counts(parts_cfg, rng: np.random.Generator, loading_cfg=None) -> list[int]:
    """Parts per type for one scene. loading.partial_fill: with probability `prob` the whole pile is scaled down to a
    random fraction in [min_fraction, 1] (a bin the robot has partly emptied)."""
    out = []
    for p in parts_cfg:
        lo, hi = count_range(p)
        out.append(int(rng.integers(lo, hi + 1)))
    pf = (loading_cfg or {}).get("partial_fill", {}) or {}
    if float(pf.get("prob", 0.0)) > 0 and rng.random() < float(pf["prob"]):
        f = rng.uniform(float(pf.get("min_fraction", 0.05)), 1.0)
        out = [max(1, int(round(c * f))) for c in out]
    return out


def choose_mode(loading_cfg, rng) -> str:
    mode = loading_cfg.get("mode", "random")
    if mode == "mixed":
        return "layers" if rng.random() < float(loading_cfg.get("mixed_layers_prob", 0.3)) else "random"
    return mode


def plan_scene(counts, geoms, bin_geom: BinGeom, loading_cfg, rng, mode: str | None = None):
    """Returns (placements, mode)."""
    mode = mode or choose_mode(loading_cfg, rng)
    order = np.concatenate([np.full(c, i, dtype=int) for i, c in enumerate(counts)]) if counts else np.zeros(0, int)
    rng.shuffle(order)
    if mode == "layers":
        placements, rest = _plan_layers(order, geoms, bin_geom, loading_cfg, rng)
        if len(rest) and loading_cfg.get("layers", {}).get("overflow", "drop") == "drop":
            placements += _plan_random(rest, geoms, bin_geom, loading_cfg, rng, existing=placements)
        return placements, mode
    if mode == "stack":
        return _plan_stack(order, geoms, bin_geom, loading_cfg, rng), mode
    return _plan_random(order, geoms, bin_geom, loading_cfg, rng), mode


def _yaw(rng, max_deg=180.0):
    return axis_angle_to_mat([0, 0, 1], np.deg2rad(rng.uniform(-max_deg, max_deg)))


def _bounds(g: PartGeom, R: np.ndarray):
    """Vertical-cylinder bound (radius_xy, half_height) of a part in orientation R."""
    ext = np.abs(R) @ g.extents
    return 0.5 * float(np.hypot(ext[0], ext[1])), 0.5 * float(ext[2])


def _plan_stack(order, geoms, bin_geom, loading_cfg, rng):
    """loading.stack: parts put down one by one onto the pile (a height map of the bin on a stack.cell_m grid), each
    in its layer orientation (e.g. front side up) with a random yaw (stack.yaw_deg), turned over with stack.flip_prob.
    Of stack.tries random spots inside the walls the lowest is taken, like filling a bin by hand; the part is put down
    flat stack.gap_m above the highest point under its real shape (PartGeom.mesh, else its box): placed, not dropped.
    Parts that bridge others tilt into the gaps when physics settles the pile."""
    scfg = loading_cfg.get("stack", {})
    margin = float(loading_cfg.get("wall_margin_m", 0.01))
    gap = float(scfg.get("gap_m", 0.003))
    tries = max(int(scfg.get("tries", 20)), 1)
    yaw_max = float(scfg.get("yaw_deg", 180.0))
    flip_p = float(scfg.get("flip_prob", 0.0))
    cell = float(scfg.get("cell_m", 0.003))
    ix, iy, _ = bin_geom.inner_size
    half_bin = np.array([ix / 2, iy / 2])
    pile = np.full((int(np.ceil(ix / cell)), int(np.ceil(iy / cell))), float(bin_geom.floor_z))  # top of the pile
    out = []
    for ti in order:
        g = geoms[ti]
        best = None
        for _ in range(tries):
            flip = bool(flip_p) and rng.random() < flip_p
            xy0, under, over, origin = g.profile(cell / 2, flip)
            a = rng.uniform(-np.deg2rad(yaw_max), np.deg2rad(yaw_max))
            pts = xy0 @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
            lo = -half_bin + margin - pts.min(axis=0)     # shift range that keeps the part inside the walls
            hi = half_bin - margin - pts.max(axis=0)
            xy = np.where(lo < hi, lo + rng.random(2) * (hi - lo), (lo + hi) / 2)
            idx = np.clip(np.floor((pts + xy + half_bin) / cell).astype(int), 0, np.array(pile.shape) - 1)
            z = float(np.max(pile[idx[:, 0], idx[:, 1]] - under)) + gap  # underside just clears the pile
            if best is None or z < best[0]:
                best = (z, a, flip, xy, idx, over, origin)
        z, a, flip, xy, idx, over, origin = best
        np.maximum.at(pile, (idx[:, 0], idx[:, 1]), z + over)
        R = axis_angle_to_mat([0, 0, 1], a) @ g.layer_R
        if flip:
            R = R @ axis_angle_to_mat([1, 0, 0], np.pi)
        out.append(Placement(int(ti), np.array([xy[0], xy[1], z + origin]), np.asarray(mat_to_quat(R), float)))
    return out


def _plan_random(order, geoms, bin_geom, loading_cfg, rng, existing=()):
    rcfg = loading_cfg.get("random", {})
    margin = float(loading_cfg.get("wall_margin_m", 0.01))
    h_lo, h_hi = as_range(rcfg.get("drop_height_m", [0.02, 0.15]), "drop_height_m")
    orient = rcfg.get("orientation", "uniform")
    ix, iy, _ = bin_geom.inner_size

    # already-placed items as (xy, z, r_xy, half_h)
    placed = []
    for p in existing:
        g = geoms[p.type_idx]
        r, h = g.radius, g.radius  # conservative for layer parts too
        placed.append((p.pos[:2], p.pos[2], r, h))

    out = []
    quats = random_quaternions(len(order), rng)
    for k, ti in enumerate(order):
        g = geoms[ti]
        if orient == "uniform":
            q = quats[k]
            r_xy = h = g.radius  # any orientation fits in the bounding sphere
        else:
            R = _yaw(rng) @ g.layer_R
            if orient == "flat" and rng.random() < 0.5:
                R = R @ axis_angle_to_mat([1, 0, 0], np.pi)
            q = mat_to_quat(R)
            r_xy, h = _bounds(g, R)
        hx = max(ix / 2 - margin - r_xy, 0.0)
        hy = max(iy / 2 - margin - r_xy, 0.0)
        xy = np.array([rng.uniform(-hx, hx), rng.uniform(-hy, hy)])
        z = bin_geom.floor_z + h + rng.uniform(h_lo, h_hi)
        for pxy, pz, pr, ph in placed:
            d = float(np.linalg.norm(xy - pxy))
            if d < r_xy + pr:
                if orient == "uniform":  # bounding spheres: rest on the other sphere (they just touch), much lower
                    z = max(z, pz + float(np.sqrt((r_xy + pr) ** 2 - d ** 2)) + 0.002)  # columns than stacking
                else:
                    z = max(z, pz + ph + h + 0.002)
        placed.append((xy, z, r_xy, h))
        out.append(Placement(int(ti), np.array([xy[0], xy[1], z]), np.asarray(q, float)))
    return out


def _plan_layers(order, geoms, bin_geom, loading_cfg, rng):
    lcfg = loading_cfg.get("layers", {})
    margin = float(loading_cfg.get("wall_margin_m", 0.01))
    spacing = float(lcfg.get("spacing_m", 0.004))
    jitter = float(lcfg.get("pos_jitter_m", 0.0))
    yaw_j = float(lcfg.get("yaw_jitter_deg", 0.0))
    flip_p = float(lcfg.get("flip_prob", 0.0))
    nl_lo, nl_hi = (int(v) for v in as_range(lcfg.get("num_layers", [1, 3]), "num_layers"))
    fr_lo, fr_hi = as_range(lcfg.get("fill_ratio", [1.0, 1.0]), "fill_ratio")
    num_layers = int(rng.integers(nl_lo, nl_hi + 1))
    if len(order) == 0:
        return [], order

    types = sorted(set(order.tolist()))
    # grid cell = largest footprint among present types, grown for worst-case yaw jitter
    j = np.deg2rad(yaw_j)
    cells = []
    for t in types:
        e = geoms[t].layer_extents()
        cells.append([e[0] * np.cos(j) + e[1] * np.sin(j), e[1] * np.cos(j) + e[0] * np.sin(j), e[2]])
    cell = np.max(np.array(cells), axis=0)
    ix, iy, _ = bin_geom.inner_size
    pitch = cell[:2] + spacing + 2 * jitter
    nx = max(int((ix - 2 * margin + spacing) // pitch[0]), 1)
    ny = max(int((iy - 2 * margin + spacing) // pitch[1]), 1)
    xs = (np.arange(nx) - (nx - 1) / 2) * pitch[0]
    ys = (np.arange(ny) - (ny - 1) / 2) * pitch[1]
    grid = np.array([(x, y) for x in xs for y in ys])
    layer_h = cell[2] + spacing

    out, k = [], 0
    for layer in range(num_layers):
        if k >= len(order):
            break
        n_cells = len(grid)
        n_use = min(max(int(round(rng.uniform(fr_lo, fr_hi) * n_cells)), 1), len(order) - k)
        idx = rng.choice(n_cells, size=n_use, replace=False)
        z_center = bin_geom.floor_z + layer * layer_h + cell[2] / 2 + spacing / 2
        for c in idx:
            ti = int(order[k])
            g = geoms[ti]
            R = axis_angle_to_mat([0, 0, 1], np.deg2rad(rng.uniform(-yaw_j, yaw_j))) @ g.layer_R
            if flip_p and rng.random() < flip_p:
                R = R @ axis_angle_to_mat([1, 0, 0], np.pi)
            xy = grid[c] + rng.uniform(-jitter, jitter, size=2)
            out.append(Placement(ti, np.array([xy[0], xy[1], z_center]), mat_to_quat(R)))
            k += 1
    return out, order[k:]


def part_geoms_from_cfg(parts_cfg, extents, radii, meshes=None) -> list[PartGeom]:
    meshes = meshes if meshes is not None else [None] * len(parts_cfg)
    return [PartGeom(p["name"], np.asarray(e, float), float(r), rpy_to_mat(p.get("layer_rpy_deg", [0, 0, 0])), m)
            for p, e, r, m in zip(parts_cfg, extents, radii, meshes)]


def bin_geom_from_cfg(bin_cfg) -> BinGeom:
    if bin_cfg.get("mesh"):
        inner = np.asarray(bin_cfg["inner_size_m"], float)
        floor = bin_cfg.get("floor_z_m")
        floor = float(bin_cfg.get("wall_thickness_m", 0.0)) if floor is None else float(floor)
    else:
        size = np.asarray(bin_cfg["size_m"], float)
        t = float(bin_cfg.get("wall_thickness_m", 0.01))
        inner = np.array([size[0] - 2 * t, size[1] - 2 * t, size[2] - t])
        floor = t
    return BinGeom(inner, floor)


def in_bin(pos_bin: np.ndarray, bin_geom: BinGeom, margin: float, rim_clearance: float | None = None) -> np.ndarray:
    """(N,3) bin-frame part centres -> bool mask of parts inside the inner volume (+margin).
    rim_clearance: parts whose centre is higher than (rim top - rim_clearance) do not count as inside (caught on the
    rim / lying on a wall); None = anything up to 0.5 m above the rim counts."""
    ix, iy, iz = bin_geom.inner_size
    p = np.atleast_2d(pos_bin)
    top = bin_geom.floor_z + iz + (0.5 if rim_clearance is None else -float(rim_clearance))
    return ((np.abs(p[:, 0]) <= ix / 2 + margin) & (np.abs(p[:, 1]) <= iy / 2 + margin)
            & (p[:, 2] >= bin_geom.floor_z - margin) & (p[:, 2] <= top))


def shake_schedule(loading_cfg, rng) -> list[int]:
    """Settle-step indices at which random velocity kicks are applied."""
    sh = loading_cfg.get("shake", {})
    n = int(sh.get("impulses", 0))
    if n <= 0:
        return []
    min_steps = int(loading_cfg.get("settle", {}).get("min_steps", 60))
    # kick early-ish so the pile has time to settle afterwards; spread over the first ~min_steps*2
    span = max(min_steps * 2, n * 10)
    steps = np.sort(rng.choice(np.arange(10, 10 + span), size=n, replace=False))
    return steps.tolist()


def bin_oscillation_offset(t: float, loading_cfg, phase: np.ndarray) -> np.ndarray:
    """(num_envs, 3) bin xy displacement at time t for the kinematic bin wiggle."""
    osc = loading_cfg.get("shake", {}).get("bin_oscillation", {}) or {}
    amp, freq, dur = float(osc.get("amp_m", 0.0)), float(osc.get("freq_hz", 3.0)), float(osc.get("duration_s", 0.0))
    out = np.zeros((len(phase), 3))
    if amp <= 0 or t > dur:
        return out
    fade = 0.5 * (1 - np.cos(2 * np.pi * min(t / dur, 1.0)))  # smooth in/out
    w = 2 * np.pi * freq
    out[:, 0] = amp * fade * np.sin(w * t + phase)
    out[:, 1] = amp * fade * np.sin(w * t + 1.7 * phase)
    return out
