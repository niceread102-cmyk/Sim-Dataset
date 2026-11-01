"""Grasp labels: per-part grasp library -> grasps of a scene with collision and approach checks (numpy only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .loading import BinGeom
from .transforms import axis_angle_to_mat, make_T, rpy_to_mat


@dataclass
class GraspLibrary:
    kind: str
    T: np.ndarray
    width: np.ndarray


def _frame_from_approach(p, approach) -> np.ndarray:
    z = np.asarray(approach, float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0, 0], z)
    x /= np.linalg.norm(x)
    return make_T(np.stack([x, np.cross(z, x), z], 1), p)


def sample_suction_grasps(mesh: trimesh.Trimesh, gripper_cfg, num: int, rng, dense: int = 20000) -> list[np.ndarray]:
    sc = gripper_cfg.get("suction", {}) or {}
    r = float(sc.get("cup_radius_m", 0.01))
    tol = float(sc.get("max_planarity_err_m", 0.0015))
    max_dev = np.cos(np.deg2rad(float(sc.get("max_normal_dev_deg", 12))))
    seed = int(rng.integers(1 << 31))
    pts, fid = trimesh.sample.sample_surface(mesh, dense, seed=seed)
    nrm = mesh.face_normals[fid]
    density = dense / mesh.area
    need = 0.75 * density * np.pi * r * r
    order = rng.permutation(dense)
    out = []
    for i in order:
        if len(out) >= num:
            break
        p, n = pts[i], nrm[i]
        d = pts - p
        near = np.einsum("ij,ij->i", d, d) < r * r
        near &= (nrm @ n) > max_dev
        if near.sum() < need:
            continue
        if np.abs(d[near] @ n).max() > tol:
            continue
        if any(np.linalg.norm(p - q[:3, 3]) < r for q in out):
            continue
        out.append(_frame_from_approach(p, -n))
    return out


def build_library(part_cfg, mesh: trimesh.Trimesh, gripper_cfg, rng) -> GraspLibrary:
    kind = gripper_cfg.get("type", "suction")
    gcfg = part_cfg.get("grasps", {}) or {}
    Ts, widths = [], []
    if kind == "suction":
        for g in gcfg.get("suction", []) or []:
            Ts.append(_frame_from_approach(np.asarray(g["pos"], float), -np.asarray(g["normal"], float)))
        auto = gcfg.get("suction_auto")
        if auto:
            Ts += sample_suction_grasps(mesh, gripper_cfg, int(auto.get("num", 32)),
                                        np.random.default_rng(int(auto.get("seed", 0))))
        widths = [0.0] * len(Ts)
    else:
        max_w = float(gripper_cfg.get("parallel_jaw", {}).get("max_width_m", 0.085))
        for g in gcfg.get("parallel_jaw", []) or []:
            Ts.append(make_T(rpy_to_mat(g.get("rpy_deg", [0, 0, 0])), np.asarray(g["pos"], float)))
            widths.append(min(float(g.get("width_m", max_w)), max_w))
    if not Ts:
        print(f"[WARN] part '{part_cfg.get('name')}' has no {kind} grasps — no grasp labels for it", flush=True)
    return GraspLibrary(kind, np.array(Ts).reshape(-1, 4, 4), np.asarray(widths, float))


def gripper_primitives(gripper_cfg, width: float, clearance: float) -> list[dict]:
    prims = [dict(p) for p in (gripper_cfg.get("primitives") or [])]
    if gripper_cfg.get("type") == "parallel_jaw":
        pj = gripper_cfg.get("parallel_jaw", {})
        fx, fy, fz = pj.get("finger_size_m", [0.02, 0.008, 0.04])
        px, py, pz = pj.get("palm_size_m", [0.06, 0.12, 0.05])
        off = width / 2 + fy / 2
        prims += [{"type": "box", "center_m": [0, s * off, 0], "size_m": [fx, fy, fz]} for s in (-1, 1)]
        prims.append({"type": "box", "center_m": [0, 0, -fz / 2 - pz / 2], "size_m": [px, py, pz]})
    for p in prims:
        if p["type"] == "cylinder":
            z0, z1 = p["z_m"]
            p["z_m"] = [z0 - clearance, z1]
        else:
            c, s = np.asarray(p["center_m"], float), np.asarray(p["size_m"], float)
            lo, hi = c[2] - s[2] / 2 - clearance, c[2] + s[2] / 2
            p["center_m"] = [c[0], c[1], (lo + hi) / 2]
            p["size_m"] = [s[0], s[1], hi - lo]
    return prims


def points_in_primitives(pts_g: np.ndarray, prims: list[dict]) -> np.ndarray:
    hit = np.zeros(len(pts_g), bool)
    for p in prims:
        if p["type"] == "cylinder":
            z0, z1 = p["z_m"]
            r = float(p["radius_m"])
            hit |= (pts_g[:, 0] ** 2 + pts_g[:, 1] ** 2 <= r * r) & (pts_g[:, 2] >= z0) & (pts_g[:, 2] <= z1)
        else:
            c, h = np.asarray(p["center_m"]), np.asarray(p["size_m"]) / 2
            hit |= np.all(np.abs(pts_g - c) <= h, axis=1)
    return hit


def primitive_surface_points(prims: list[dict], n_per: int = 200) -> np.ndarray:
    out = []
    for p in prims:
        if p["type"] == "cylinder":
            z0, z1 = p["z_m"]
            a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            zs = np.linspace(z0, z1, max(n_per // 24, 2))
            A, Z = np.meshgrid(a, zs)
            r = float(p["radius_m"])
            out.append(np.stack([r * np.cos(A).ravel(), r * np.sin(A).ravel(), Z.ravel()], 1))
            out.append(np.array([[0, 0, z0], [0, 0, z1]]))
        else:
            c, h = np.asarray(p["center_m"]), np.asarray(p["size_m"]) / 2
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
            g = np.random.default_rng(0).uniform(-1, 1, (n_per, 3)) * h + c
            out += [corners, g]
    return np.concatenate(out) if out else np.zeros((0, 3))


def _bin_collision(pts_bin: np.ndarray, bin_geom: BinGeom, rim_z: float) -> bool:
    ix, iy, _ = bin_geom.inner_size
    below_floor = pts_bin[:, 2] < bin_geom.floor_z
    in_wall = ((np.abs(pts_bin[:, 0]) > ix / 2) | (np.abs(pts_bin[:, 1]) > iy / 2)) & (pts_bin[:, 2] < rim_z)
    near = (np.abs(pts_bin[:, 0]) < ix / 2 + 0.1) & (np.abs(pts_bin[:, 1]) < iy / 2 + 0.1)
    return bool(np.any((below_floor | in_wall) & near))


@dataclass
class SceneGrasp:
    part: int
    T_w: np.ndarray
    width: float
    approach_angle_deg: float
    valid: bool
    reason: str


def evaluate_scene(part_T_bin: list[np.ndarray], part_types: list[int], libraries: list[GraspLibrary],
                   part_points: list[np.ndarray], bin_geom: BinGeom, rim_z: float, gripper_cfg, grasp_cfg,
                   candidate_mask: np.ndarray | None = None, rng=None) -> list[SceneGrasp]:
    rng = rng or np.random.default_rng(0)
    max_ang = float(grasp_cfg.get("max_approach_angle_deg", 45))
    clearance = float(grasp_cfg.get("approach_clearance_m", 0.05))
    max_per = int(grasp_cfg.get("max_grasps_per_part", 64))
    world_pts, owner = [], []
    for i, (T, ti) in enumerate(zip(part_T_bin, part_types)):
        world_pts.append(part_points[ti] @ T[:3, :3].T + T[:3, 3])
        owner.append(np.full(len(part_points[ti]), i))
    world_pts = np.concatenate(world_pts) if world_pts else np.zeros((0, 3))
    owner = np.concatenate(owner) if owner else np.zeros(0, int)
    prim_cache: dict[float, tuple[list, np.ndarray]] = {}
    out = []
    for i, (T, ti) in enumerate(zip(part_T_bin, part_types)):
        if candidate_mask is not None and not candidate_mask[i]:
            continue
        lib = libraries[ti]
        idx = np.arange(len(lib.T))
        if len(idx) > max_per:
            idx = rng.choice(idx, max_per, replace=False)
        for g in idx:
            Tg = T @ lib.T[g]
            width = float(lib.width[g])
            approach = Tg[:3, 2]
            ang = float(np.degrees(np.arccos(np.clip(-approach[2], -1, 1))))
            if ang > max_ang:
                out.append(SceneGrasp(i, Tg, width, ang, False, "approach_angle"))
                continue
            if width not in prim_cache:
                prims = gripper_primitives(gripper_cfg, width, clearance)
                prim_cache[width] = (prims, primitive_surface_points(prims))
            prims, prim_pts = prim_cache[width]
            R, t = Tg[:3, :3], Tg[:3, 3]
            reach = 0.3
            d = world_pts - t
            close = np.einsum("ij,ij->i", d, d) < reach * reach
            pg = d[close] @ R
            hit = points_in_primitives(pg, prims)
            if hit.any():
                who = owner[close][hit]
                other = who[who != i]
                out.append(SceneGrasp(i, Tg, width, ang, False,
                                      f"collision_part:{int(other[0])}" if len(other) else "collision_self"))
                continue
            if _bin_collision(prim_pts @ R.T + t, bin_geom, rim_z):
                out.append(SceneGrasp(i, Tg, width, ang, False, "collision_bin"))
                continue
            out.append(SceneGrasp(i, Tg, width, ang, True, "ok"))
    return out


def pick_topmost(part_T_bin, part_types, part_points, visib: np.ndarray, grasps: list[SceneGrasp], min_visib: float):
    valid_by_part: dict[int, list[SceneGrasp]] = {}
    for g in grasps:
        if g.valid:
            valid_by_part.setdefault(g.part, []).append(g)
    best, best_z = None, -np.inf
    for i, (T, ti) in enumerate(zip(part_T_bin, part_types)):
        if visib[i] < min_visib or i not in valid_by_part:
            continue
        z = float((part_points[ti] @ T[:3, :3].T + T[:3, 3])[:, 2].max())
        if z > best_z:
            best, best_z = i, z
    if best is None:
        return None, None
    return best, min(valid_by_part[best], key=lambda g: g.approach_angle_deg)


def sample_part_points(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts)


__all__ = ["GraspLibrary", "SceneGrasp", "build_library", "evaluate_scene", "pick_topmost",
           "sample_part_points", "gripper_primitives", "axis_angle_to_mat"]
