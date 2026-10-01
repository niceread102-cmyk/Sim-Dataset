"""Write depth_vis/<im>.png previews for already generated data, from the stored depth PNGs (same looks as
outputs.depth_vis_style): phoxi = 16-bit per-image min-max like the PhoXi sample viewer (no depth = black, near =
dark) | fixed = 8-bit, near end of camera.depth_range_m = white ... far end = dark, no depth = black.

    python scripts/depth_preview.py <shard_dir | scene_dir> [--style phoxi] [--near 0.458 --far 1.118]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from binsim.viz import depth_preview, depth_preview_phoxi  # noqa: E402


def scene_dirs(path):
    if os.path.isdir(os.path.join(path, "depth")):
        return [path]
    tp = os.path.join(path, "train_pbr")
    return [os.path.join(tp, d) for d in sorted(os.listdir(tp)) if d.isdigit()] if os.path.isdir(tp) else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--near", type=float, default=0.458, help="m, white (camera.depth_range_m[0])")
    ap.add_argument("--far", type=float, default=1.118, help="m, dark (camera.depth_range_m[1])")
    ap.add_argument("--style", choices=["phoxi", "fixed"], default="phoxi")
    a = ap.parse_args()
    n = 0
    for sd in scene_dirs(os.path.expanduser(a.path)):
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        out = os.path.join(sd, "depth_vis")
        os.makedirs(out, exist_ok=True)
        for f in sorted(os.listdir(os.path.join(sd, "depth"))):
            if not f.endswith(".png"):
                continue
            im_id = str(int(f[:6]))
            d = np.array(Image.open(os.path.join(sd, "depth", f))).astype(np.float64) * cam[im_id]["depth_scale"] / 1000.0
            vis = depth_preview_phoxi(d) if a.style == "phoxi" else depth_preview(d, a.near, a.far)
            Image.fromarray(vis).save(os.path.join(out, f))
            n += 1
    print(f"wrote {n} previews")


if __name__ == "__main__":
    main()
