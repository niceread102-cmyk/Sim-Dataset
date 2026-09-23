"""Check that GT poses agree with the rendered images (the "project a 3D bbox before training"
step from the legacy script, automated).

For every visible instance: reproject the model with its GT pose and measure how much of the
RENDERED visible mask falls inside the reprojected silhouette (should be ~1.0), plus the depth
error between the rendered depth and the model's z-buffer on visible pixels. A wrong axis
convention or quaternion order shows up immediately as consistency << 1.

    python scripts/verify_poses.py <shard_dir> [--max-scenes 20] [--overlays 3]
Exit code 1 if the mean consistency is below --min (default 0.9).
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from binsim.bop_io import BopScene, list_scenes, load_models  # noqa: E402
from binsim.poses import project, rasterize_silhouette  # noqa: E402


def bbox_corners(m):
    lo, hi = m.bounds
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--max-scenes", type=int, default=20)
    ap.add_argument("--min-px", type=int, default=200)
    ap.add_argument("--overlays", type=int, default=3, help="write 3D-bbox overlay images for the first N views")
    ap.add_argument("--min", type=float, default=0.9)
    a = ap.parse_args()
    models, _ = load_models(a.shard_dir)
    cons_all, derr_all, n_overlay = [], [], 0
    out_dir = os.path.join(a.shard_dir, "viz", "verify")
    os.makedirs(out_dir, exist_ok=True)
    for sid in list_scenes(a.shard_dir)[:a.max_scenes]:
        sc = BopScene(a.shard_dir, sid)
        for im in sc.im_ids:
            K = sc.K(im)
            depth = sc.depth_m(im)
            H, W = depth.shape
            overlay = None
            if n_overlay < a.overlays:
                overlay = Image.fromarray(sc.rgb(im))
                draw = ImageDraw.Draw(overlay)
            cons = []
            for k, g in enumerate(sc.gt[str(im)]):
                m = models[g["obj_id"]]
                R = np.array(g["cam_R_m2c"]).reshape(3, 3)
                t = np.array(g["cam_t_m2c"]) / 1000
                vis = sc.mask(im, k)
                if vis.sum() < a.min_px:
                    continue
                verts = m.vertices @ R.T + t
                sil = rasterize_silhouette(verts, m.faces, K, W, H)
                cons.append((vis & sil).sum() / vis.sum())
                # depth sanity: rendered depth on visible pixels vs model depth range
                dv = depth[vis & (depth > 0)]
                if len(dv):
                    zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
                    derr_all.append(float(np.mean((dv < zmin - 0.005) | (dv > zmax + 0.005))))
                if overlay is not None:
                    uv = project(K, bbox_corners(m) @ R.T + t)
                    for i, j in EDGES:
                        draw.line([tuple(uv[i]), tuple(uv[j])], fill=(0, 255, 0), width=2)
            if overlay is not None:
                overlay.save(os.path.join(out_dir, f"{sid:06d}_{im:02d}_bbox.jpg"), quality=90)
                n_overlay += 1
            if cons:
                cons_all.append(np.mean(cons))
                if np.mean(cons) < a.min:
                    print(f"[WARN] scene {sid} view {im}: mask consistency {np.mean(cons):.3f}")
    if not cons_all:
        print("no visible instances found")
        sys.exit(1)
    mc = float(np.mean(cons_all))
    print(f"views checked: {len(cons_all)}  mean mask consistency: {mc:.4f}  "
          f"depth outside model z-range: {100 * np.mean(derr_all) if derr_all else float('nan'):.2f}% of visible px")
    print(f"bbox overlays: {out_dir}")
    sys.exit(0 if mc >= a.min else 1)


if __name__ == "__main__":
    main()
