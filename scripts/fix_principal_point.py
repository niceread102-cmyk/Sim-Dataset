"""Repair shards generated with an off-centre camera.intrinsics cx/cy (before generate_isaacsim.py used
binsim.camera.centred): the images were rendered with the principal point at the image centre, but camera.json,
scene_camera.json (cam_K) and scene_gt_info.json (bbox_obj, projected with K) used the config cx/cy.
Rewrites those three with the centred K. Images, depth, masks, poses and COCO are not touched (they are correct).
Only complete scenes (scene_info.json written) are changed; run it again after a run has finished. Idempotent.

    python scripts/fix_principal_point.py <shard_dir> [<shard_dir> ...] [--dry-run]
"""
import argparse
import glob
import json
import os

import numpy as np


def _json(path: str, obj):  # as binsim.writers.bop._json (atomic, indent 1)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def _projected_bbox(verts_cam: np.ndarray, K: np.ndarray, W: int, H: int):
    """= binsim.postprocess._projected_bbox (kept here so the script needs numpy only)."""
    v = verts_cam[verts_cam[:, 2] > 1e-4]
    if len(v) == 0:
        return [-1, -1, -1, -1]
    uv = (v[:, :2] / v[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    x0, y0 = np.floor(uv.min(axis=0)).astype(int)
    x1, y1 = np.ceil(uv.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
    if x1 < x0 or y1 < y0:
        return [-1, -1, -1, -1]
    return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]


def ply_vertices(path: str) -> np.ndarray:
    """Vertices of a binary little-endian / ascii PLY (the BOP models written by trimesh)."""
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"bad PLY header: {path}")
            hdr += line
        n = int(hdr.split(b"element vertex ")[1].split(b"\n")[0])
        if b"format ascii" in hdr:
            return np.array([f.readline().split()[:3] for _ in range(n)], float)
        props = hdr.split(b"element vertex ")[1].split(b"element")[0].count(b"property")
        return np.frombuffer(f.read(n * 4 * props), "<f4").reshape(n, props)[:, :3].astype(np.float64)


def fix_shard(shard: str, dry: bool) -> None:
    cam_path = os.path.join(shard, "camera.json")
    cam = json.load(open(cam_path))
    W, H = int(cam["width"]), int(cam["height"])
    ccx, ccy = (W - 1) / 2.0, (H - 1) / 2.0
    models = {int(os.path.basename(p)[4:10]): ply_vertices(p) for p in glob.glob(os.path.join(shard, "models", "obj_*.ply"))}
    n_sc = n_views = n_box = 0
    skipped = []
    for sd in sorted(glob.glob(os.path.join(shard, "train_pbr", "*"))):
        if not os.path.exists(os.path.join(sd, "scene_info.json")):
            skipped.append(os.path.basename(sd))
            continue
        sc = json.load(open(os.path.join(sd, "scene_camera.json")))
        if all(abs(v["cam_K"][2] - ccx) < 1e-6 and abs(v["cam_K"][5] - ccy) < 1e-6 for v in sc.values()):
            continue
        gt = json.load(open(os.path.join(sd, "scene_gt.json")))
        gti = json.load(open(os.path.join(sd, "scene_gt_info.json")))
        for k, v in sc.items():
            v["cam_K"][2], v["cam_K"][5] = ccx, ccy
            K = np.array(v["cam_K"], float).reshape(3, 3)
            for g, gi in zip(gt[k], gti[k]):
                R = np.array(g["cam_R_m2c"], float).reshape(3, 3)
                t = np.array(g["cam_t_m2c"], float)
                new = _projected_bbox(models[g["obj_id"]] @ R.T + t, K, W, H)
                n_box += new != gi["bbox_obj"]
                gi["bbox_obj"] = new
            n_views += 1
        if not dry:
            _json(os.path.join(sd, "scene_camera.json"), sc)
            _json(os.path.join(sd, "scene_gt_info.json"), gti)
        n_sc += 1
    if (abs(cam["cx"] - ccx) > 1e-6 or abs(cam["cy"] - ccy) > 1e-6) and not dry:
        cam["cx"], cam["cy"] = ccx, ccy
        _json(cam_path, cam)
    print(f"{shard}: {'would fix' if dry else 'fixed'} {n_sc} scenes / {n_views} views -> cx={ccx} cy={ccy}, "
          f"{n_box} bbox_obj changed" + (f"; skipped incomplete {skipped}" if skipped else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for s in a.shards:
        fix_shard(os.path.expanduser(s), a.dry_run)


if __name__ == "__main__":
    main()
