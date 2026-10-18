#!/usr/bin/env python3
"""Quick check: can the robot put the tool straight down into the crates of the top layer with this platform height and extension?

    python3 check_reach.py platform:=0.8 extension:=0.6 [robot:=m20_35_18d_mhzl2_20d] [gripper:=0.15] [--sweep]

No Isaac Sim, no ROS: only the URDF of the robot (kinematics, joint limits) and scene.pallets of configs/map/map.yaml.
  platform    height of the robot base (m): the platform of the map, the robot stands on it
  extension   length of the extension between the flange and the gripper (m)
  gripper     flange side of the gripper to its TCP (m): MHZL2-20D 0.15
For every distance r of the plates from the axis of the robot (smallest .. largest of the top layer) the tool goes straight down from the approach
pose to the plate with the tool vertical (J1, J4, J6 free). Printed: the smallest |J5| on the way (a wrist at J5 = 0 is singular: the straight line
breaks), the smallest distance of J2 / J3 to their limits, and a verdict. |J5| >= 30 deg is fine, 15..30 marginal, below it the line fails.
--sweep prints the verdict for many platform / extension pairs. The reachable tool-down poses of the arm are computed once (about a minute) and kept in
assets_cache/, then every check takes a second.
"""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree

import make_robot_config as m
from binsim.pallets import plan_pallets

HERE = Path(__file__).resolve().parent
J5_OK, J5_MARGINAL = 30.0, 15.0
HOME = np.radians([15.0, 0.0, -75.0])                       # J2 J3 J5 of the start pose: the branch the sequence stays on (elbow up, J5 < 0)


def arg(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(name + ":="):
            return a[len(name) + 2:]
    return default


def tool_down_poses(urdf, chain):
    """(r, z, J2, J3, J5) of the flange origin over a 1 deg grid of J2 J3 J5 (J1 = J4 = J6 = 0) where the flange z axis (tool) points down within 1 deg"""
    key = hashlib.sha1(urdf.path.read_bytes()).hexdigest()[:10]
    cache = HERE / "assets_cache" / f"reach_{urdf.path.stem}_{key}.npy"
    if cache.is_file():
        return np.load(cache)
    print("computing the tool-down poses of the arm (once, about a minute) ...", flush=True)
    mov = [j for j in chain if j.get("type") in m.MOVABLE]
    lo, hi = m.position_limits(mov)
    j2 = np.radians(np.arange(math.degrees(lo[1]), math.degrees(hi[1]) + 0.1, 1.0))
    j3 = np.radians(np.arange(math.degrees(lo[2]), math.degrees(hi[2]) + 0.1, 1.0))
    G2, G3 = (g.reshape(-1) for g in np.meshgrid(j2, j3, indexing="ij"))
    out = []
    for k in np.radians(np.arange(math.degrees(lo[4]), math.degrees(hi[4]) + 0.1, 1.0)):
        q = np.zeros((len(G2), 6))
        q[:, 1], q[:, 2], q[:, 4] = G2, G3, k
        T, _ = urdf.fk(chain, q)
        ok = T[:, 2, 2] < -math.cos(math.radians(1.0))
        if ok.any():
            p = T[ok, :3, 3]
            out.append(np.stack([np.abs(p[:, 0]), p[:, 2], G2[ok], G3[ok], np.full(ok.sum(), k)], 1))
    cache.parent.mkdir(exist_ok=True)
    F = np.concatenate(out).astype(np.float32)
    np.save(cache, F)
    return F


def insert_line(tree, F, r, h, tool, z_lo, z_hi, lim, tol=0.02):
    """Tool vertical at distance r from z_hi down to z_lo (world TCP z), base at height h, TCP `tool` below the flange.
    Returns (smallest |J5| deg, smallest distance of J2 / J3 to the limits deg), None if a pose on the way cannot be reached"""
    prev, j5, margin = None, 999.0, 999.0
    for z in np.linspace(z_hi, z_lo, 21):
        idx = tree.query_ball_point([r, z - h + tool], tol)
        if not idx:
            return None
        cand = F[idx]
        q = cand[np.argmin(np.abs(cand[:, 2:5] - (HOME if prev is None else prev)).max(1))][2:5]
        if prev is not None and np.degrees(np.abs(q - prev).max()) > 6.0:
            return None                                      # the joints jump: the straight line cannot follow
        prev = q
        j5 = min(j5, abs(math.degrees(q[2])))
        margin = min(margin, math.degrees(q[0] - lim[0][1]), math.degrees(lim[1][1] - q[0]), math.degrees(q[1] - lim[0][2]), math.degrees(lim[1][2] - q[1]))
    return j5, margin


def verdict(res):
    return "not reachable / line breaks" if res is None else "ok" if res[0] >= J5_OK else "marginal" if res[0] >= J5_MARGINAL else "singular: the line breaks"


def plates(cfg_map, robot_xy, robot_yaw_deg):
    """distance from the robot axis, azimuth in the robot frame and TCP z of the plates of the top layer (scene.pallets of map.yaml)"""
    pc = cfg_map["scene"]["pallets"]
    part = yaml.safe_load(open(HERE / "configs" / "objects" / f"{pc['part']}.yaml"))
    mesh = trimesh.load((HERE / "configs" / "objects" / part["mesh"]).resolve(), force="mesh")
    ext = mesh.extents * {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": 0.0254}[part.get("units", "mm")]
    crate = {**pc["crate"], "usd": None}
    r, az, z = [], [], []
    for seed in range(3):
        plan = plan_pallets({k: v for k, v in pc.items() if k not in ("part", "seed", "dynamic_layers")} | {"crate": crate}, ext, part.get("layer_rpy_deg"),
                            np.random.default_rng(seed))
        top = max(p["layer"] for p in plan["parts"])
        for p in plan["parts"]:
            if p["layer"] == top:
                d = p["pos"][:2] - robot_xy
                r.append(math.hypot(*d))
                a = math.degrees(math.atan2(d[1], d[0])) - robot_yaw_deg
                az.append((a + 180) % 360 - 180)
                z.append(p["pos"][2])
    return np.array(r), np.array(az), np.array(z)


def main():
    robot = arg("robot", "m20_35_18d_mhzl2_20d")
    cfg_map = yaml.safe_load(open(HERE / "configs" / "map" / "map.yaml"))
    ry = yaml.safe_load(open(HERE / "configs" / "robots" / f"{robot}.yaml"))
    pf = (cfg_map["scene"].get("platform") or {}).get("size_m", [0, 0, 0])
    h = float(arg("platform", ry.get("pos_m", [0, 0, 0])[2] + pf[2]))
    gripper = float(arg("gripper", 0.15))
    urdf = m.Urdf(m.find_urdf(robot))
    chain = urdf.chain(urdf.base_link(), "fanuc_flange")
    ext_urdf = float(arg("extension", 0.3))
    lim = m.position_limits([j for j in chain if j.get("type") in m.MOVABLE])
    F = tool_down_poses(urdf, chain)
    F = F[np.degrees(F[:, 3]) <= 20].astype(np.float64)
    tree = cKDTree(F[:, :2])
    r, az, z = plates(cfg_map, np.array(ry.get("pos_m", [0, 0])[:2], float), float((ry.get("rpy_deg") or [0, 0, 0])[2]))
    approach = float(yaml.safe_load(open(HERE / "configs" / "sequence" / "robot_sequence.yaml"))["approach"]["distance"])
    z_lo, z_hi = float(z.min()), float(z.max()) + approach
    j1 = (math.degrees(lim[0][0]), math.degrees(lim[1][0]))
    print(f"{robot}: plates of the top layer at r {r.min():.2f}..{r.max():.2f} m from the axis, azimuth {az.min():.0f}..{az.max():.0f} deg (J1 limit {j1[0]:.0f}..{j1[1]:.0f}), "
          f"TCP z {z_lo:.3f}..{z_hi:.3f} m (down to the plate, approach {approach:.2f} m above)")
    rr = np.linspace(r.min(), r.max(), 6)
    if "--sweep" in sys.argv:
        print("\nverdict by platform height (rows) and extension (columns): worst |J5| deg on the way over all distances, '-' = line breaks")
        exts = np.arange(0.0, 0.91, 0.1)
        print("            e = " + "  ".join(f"{e:4.1f}" for e in exts))
        for hh in np.arange(0.6, 1.41, 0.1):
            row = [min((x[0] if x else -1) for x in (insert_line(tree, F, q, hh, e + gripper, z_lo, z_hi, lim) for q in rr)) for e in exts]
            print(f"  platform {hh:.1f}:  " + "  ".join(f"{v:4.0f}" if v >= 0 else "   -" for v in row))
        return
    tool = ext_urdf + gripper
    print(f"\nplatform {h:.2f} m, extension {ext_urdf:.2f} m + gripper {gripper:.2f} m = tool {tool:.2f} m (flange to TCP)\n")
    print("  r (m)   smallest |J5| deg   J2/J3 limit margin deg   verdict")
    worst = 999.0
    for q in rr:
        res = insert_line(tree, F, q, h, tool, z_lo, z_hi, lim)
        worst = min(worst, res[0] if res else -1)
        print(f"  {q:5.2f}   " + (f"{res[0]:12.0f}   {res[1]:18.0f}" if res else f"{'-':>12s}   {'-':>18s}") + f"          {verdict(res)}")
    ok = worst >= J5_OK
    print("\n" + ("OK for every distance." if ok else "NOT enough: the tool needs a higher flange above the base (a longer extension or a lower platform): platform - extension <= ~0.2 m gives |J5| >= 40 deg."))
    if not ok:
        best = next((e for e in np.arange(ext_urdf, 1.21, 0.05) if min((x[0] if x else -1) for x in (insert_line(tree, F, q, h, e + gripper, z_lo, z_hi, lim) for q in rr)) >= J5_OK), None)
        lowest = next((hh for hh in np.arange(h, 0.39, -0.05) if min((x[0] if x else -1) for x in (insert_line(tree, F, q, hh, tool, z_lo, z_hi, lim) for q in rr)) >= J5_OK), None)
        print(f"  with this platform the extension has to be at least {best:.2f} m" if best else "  no extension up to 1.2 m is enough for this platform", end="")
        print(f"; with this extension the platform at most {lowest:.2f} m" if lowest else "; no platform height down to 0.4 m is enough for this extension")


if __name__ == "__main__":
    main()
