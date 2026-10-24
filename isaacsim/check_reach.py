#!/usr/bin/env python3
"""Quick check: can the robot put the tool straight down to the parts (or the crate floors) of the top layer of the pallets of a map?

    python3 check_reach.py map:=<map> [robot:=<name>] [platform:=<m>] [tool:=<m>] [--sweep]

No Isaac Sim, no ROS: only the URDF of the robot (kinematics, joint limits), its bringup/<robot>/config/ik_node.yaml (end frame, TCP), its start pose
(joint_pos of configs/robot/<robot>.yaml) and scene.pallets of the map (configs/map/<map>.yaml); the robot stands where the map puts it
(scene.robot_pose), robot:= defaults to scene.robot of the map.
  platform    height of the robot base (m); default: where it stands in the map
  tool        flange (ee_link of ik_node.yaml) to the TCP (m); default: the length of tcp_offset of ik_node.yaml
For every distance r of the targets from the axis of the robot (smallest .. largest of the top layer: the parts in the crates, or the crate floors
when the crates are empty) the tool goes straight down from the approach pose to the target with the tool vertical (J1, J4, J6 free). Printed: the
smallest |J5| on the way (a wrist at J5 = 0 is singular: the straight line breaks), the smallest distance of J2 / J3 to their limits, and a verdict.
|J5| >= 30 deg is fine, 15..30 marginal, below it the line fails. For a 6-axis arm with a wrist: J2, J3, J5 = the 2nd, 3rd and 5th joint of the chain.
--sweep prints the verdict for base heights around the one of the map (rows) and tool lengths (columns). The reachable tool-down poses of the arm are
computed once (about a minute) and kept in assets_cache/, then every check takes a second.
"""
import hashlib
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

import make_robot_config as m
import sim_config
from binsim.pallets import plan_pallets

HERE = Path(__file__).resolve().parent
J5_OK, J5_MARGINAL = 30.0, 15.0


def arg(name, default=None):
    return sim_config.option(sys.argv[1:], name) or default


def tool_down_poses(urdf, chain):
    """(r, z, J2, J3, J5) of the flange origin over a 1 deg grid of J2 J3 J5 (J1 = J4 = J6 = 0) where the flange z axis (tool) points down within 1 deg"""
    key = hashlib.sha1(urdf.path.read_bytes()).hexdigest()[:10]
    cache = HERE / "assets_cache" / f"reach_{urdf.path.stem}_{chain[-1].find('child').get('link')}_{key}.npy"
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


def insert_line(tree, F, r, h, tool, z_lo, z_hi, lim, home, tol=0.02):
    """Tool vertical at distance r from z_hi down to z_lo (world TCP z), base at height h, TCP `tool` below the flange; home = J2 J3 J5 of the start
    pose (the branch the sequence stays on). Returns (smallest |J5| deg, smallest distance of J2 / J3 to the limits deg), None if a pose on the way cannot be reached"""
    prev, j5, margin = None, 999.0, 999.0
    for z in np.linspace(z_hi, z_lo, 21):
        idx = tree.query_ball_point([r, z - h + tool], tol)
        if not idx:
            return None
        cand = F[idx]
        q = cand[np.argmin(np.abs(cand[:, 2:5] - (home if prev is None else prev)).max(1))][2:5]
        if prev is not None and np.degrees(np.abs(q - prev).max()) > 6.0:
            return None                                      # the joints jump: the straight line cannot follow
        prev = q
        j5 = min(j5, abs(math.degrees(q[2])))
        margin = min(margin, math.degrees(q[0] - lim[0][1]), math.degrees(lim[1][1] - q[0]), math.degrees(q[1] - lim[0][2]), math.degrees(lim[1][2] - q[1]))
    return j5, margin


def verdict(res):
    return "not reachable / line breaks" if res is None else "ok" if res[0] >= J5_OK else "marginal" if res[0] >= J5_MARGINAL else "singular: the line breaks"


def targets(scene, robot_xy, robot_yaw_deg):
    """distance from the robot axis, azimuth in the robot frame and TCP z of the targets of the top layer (scene.pallets of the map): the parts,
    or the floors of the crates when they are empty. Returns (r, az, z, what)"""
    pc = scene["pallets"]
    part = (yaml.safe_load(open(os.path.join(sim_config.OBJECTS_DIR, f"{pc['part']}.yaml"))) if pc.get("part") else None)
    if part:
        import trimesh                                       # only for the size of the part (the conda env `sim` has it)
        mesh = trimesh.load(os.path.normpath(os.path.join(sim_config.OBJECTS_DIR, part["mesh"])), force="mesh")
        ext = mesh.extents * {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": 0.0254}[part.get("units", "mm")]
    crate = {**pc["crate"], "usd": None}
    cfg = {k: v for k, v in pc.items() if k not in ("part", "seed", "dynamic_layers", "physics")} | {"crate": crate}
    if not part:
        cfg["parts_per_crate"] = 0
    r, az, z, what = [], [], [], "parts"
    for seed in range(3):
        plan = plan_pallets(cfg, ext if part else [0.01, 0.01, 0.01], (part or {}).get("layer_rpy_deg"), np.random.default_rng(seed))
        points = [(p["pos"], p["layer"]) for p in plan["parts"]]
        if not points:                                       # empty crates: down to the floor of every crate
            what = "crate floors"
            points = [(c["pos"] + np.array([0.0, 0.0, float(crate["floor_z_m"])]), c["layer"]) for c in plan["crates"]]
        top = max(layer for _, layer in points)
        for pos, layer in points:
            if layer == top:
                d = pos[:2] - robot_xy
                r.append(math.hypot(*d))
                a = math.degrees(math.atan2(d[1], d[0])) - robot_yaw_deg
                az.append((a + 180) % 360 - 180)
                z.append(pos[2])
    return np.array(r), np.array(az), np.array(z), what


def main():
    try:
        robot, map_name = sim_config.with_current(arg("robot"), arg("map"))   # without robot:= / map:= the ones of the robot_sim.py started last
        scene = sim_config.scene_of(map_name)
        robot = sim_config.pick_robot(robot, scene)
    except ValueError as e:
        sys.exit(str(e))
    if not scene.get("pallets"):
        sys.exit("no scene.pallets in the map: nothing to reach (map:=<map> of configs/map/ with pallets)")
    ry = sim_config.load_robot(robot)
    ik = yaml.safe_load(open(os.path.join(sim_config.BRINGUP_DIR, robot, "config", "ik_node.yaml")))
    pos, rpy = sim_config.robot_pose(ry, scene)
    h = float(arg("platform", pos[2]))
    tool = float(arg("tool", float(np.linalg.norm(ik.get("tcp_offset") or [0, 0, 0]))))
    urdf = m.Urdf(Path(sim_config.find_urdf(robot)))
    chain = urdf.chain(urdf.base_link(), ik["ee_link"])
    mov = [j for j in chain if j.get("type") in m.MOVABLE]
    if len(mov) != 6:
        sys.exit(f"{robot}: {len(mov)} joints from {urdf.base_link()} to {ik['ee_link']}: check_reach.py is for a 6-axis arm with a wrist")
    jp = ry.get("joint_pos") or {}                           # the start pose: the branch the sequence stays on (elbow, wrist)
    seed = ([next((float(v) for k, v in jp.items() if re.fullmatch(str(k), j.get("name"))), 0.0) for j in mov] if jp
            else [math.radians(float(v)) for v in ik["ik_seed_deg"]])
    home = np.array([seed[1], seed[2], seed[4]])
    lim = m.position_limits(mov)
    F = tool_down_poses(urdf, chain)
    F = F[np.degrees(F[:, 3]) <= 20].astype(np.float64)
    tree = cKDTree(F[:, :2])
    r, az, z, what = targets(scene, np.array(pos[:2], float), float(rpy[2]))
    approach = float(yaml.safe_load(open(os.path.join(sim_config.SEQUENCE_DIR, "robot_sequence.yaml")))["approach"]["distance"])
    z_lo, z_hi = float(z.min()), float(z.max()) + approach
    j1 = (math.degrees(lim[0][0]), math.degrees(lim[1][0]))
    print(f"{robot}{' in ' + map_name if map_name else ''}: {what} of the top layer at r {r.min():.2f}..{r.max():.2f} m from the axis, azimuth "
          f"{az.min():.0f}..{az.max():.0f} deg (J1 limit {j1[0]:.0f}..{j1[1]:.0f}), TCP z {z_lo:.3f}..{z_hi:.3f} m (down to them, approach {approach:.2f} m above)")
    rr = np.linspace(r.min(), r.max(), 6)
    if "--sweep" in sys.argv:
        tools = np.arange(0.1, 1.21, 0.1)
        print("\nverdict by base height (rows) and tool length flange -> TCP (columns): worst |J5| deg on the way over all distances, '-' = line breaks")
        print("            t = " + "  ".join(f"{t:4.1f}" for t in tools))
        for hh in np.arange(max(0.0, h - 0.4), h + 0.41, 0.1):
            row = [min((x[0] if x else -1) for x in (insert_line(tree, F, q, hh, t, z_lo, z_hi, lim, home) for q in rr)) for t in tools]
            print(f"  base {hh:.1f} m:  " + "  ".join(f"{v:4.0f}" if v >= 0 else "   -" for v in row))
        return
    print(f"\nbase at {h:.2f} m, tool (flange {ik['ee_link']} -> TCP) {tool:.2f} m\n")
    print("  r (m)   smallest |J5| deg   J2/J3 limit margin deg   verdict")
    worst = 999.0
    for q in rr:
        res = insert_line(tree, F, q, h, tool, z_lo, z_hi, lim, home)
        worst = min(worst, res[0] if res else -1)
        print(f"  {q:5.2f}   " + (f"{res[0]:12.0f}   {res[1]:18.0f}" if res else f"{'-':>12s}   {'-':>18s}") + f"          {verdict(res)}")
    ok = worst >= J5_OK
    print("\n" + ("OK for every distance." if ok else "NOT enough: the tool needs a higher flange above the base (a longer tool or a lower base): base - tool <= ~0.2 m gives |J5| >= 40 deg."))
    if not ok:
        best = next((t for t in np.arange(tool, 1.21, 0.05) if min((x[0] if x else -1) for x in (insert_line(tree, F, q, h, t, z_lo, z_hi, lim, home) for q in rr)) >= J5_OK), None)
        lowest = next((hh for hh in np.arange(h, -0.01, -0.05) if min((x[0] if x else -1) for x in (insert_line(tree, F, q, hh, tool, z_lo, z_hi, lim, home) for q in rr)) >= J5_OK), None)
        print(f"  with this base height the tool has to be at least {best:.2f} m" if best else "  no tool up to 1.2 m is enough at this base height", end="")
        print(f"; with this tool the base at most {lowest:.2f} m high" if lowest is not None else "; no base height down to 0 m is enough for this tool")


if __name__ == "__main__":
    main()
