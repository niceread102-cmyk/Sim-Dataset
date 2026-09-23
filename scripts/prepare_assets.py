"""Convert/cache every asset in a config (CAD -> mesh -> USD) and print a summary.
Pure Python (needs trimesh + usd-core, or run with Isaac's python); no simulator required.

    python scripts/prepare_assets.py --config configs/default.yaml [--force] [key=value ...]
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from binsim.assets import prepare_all  # noqa: E402
from binsim.config import count_range, load_config  # noqa: E402
from binsim.grasps import build_library  # noqa: E402
from binsim.loading import bin_geom_from_cfg, part_geoms_from_cfg, plan_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    parts, bin_asset = prepare_all(cfg, force=a.force)
    for p, asset in zip(cfg.parts, parts):
        lib = build_library(p, asset.mesh, cfg.gripper, np.random.default_rng(0))
        print(f"part '{p['name']}': {len(asset.mesh.faces)} faces, extents {np.round(asset.extents * 1000, 1).tolist()} mm, "
              f"watertight={asset.mesh.is_watertight}, {len(lib.T)} {lib.kind} grasps\n  usd: {asset.usd_path}")
    print(f"bin: extents {np.round(bin_asset.extents * 1000, 1).tolist()} mm\n  usd: {bin_asset.usd_path}")
    # sanity-check capacity of the layer layout with the max counts
    geoms = part_geoms_from_cfg(cfg.parts, [x.extents for x in parts], [x.radius for x in parts])
    bg = bin_geom_from_cfg(cfg.bin)
    hi = [count_range(p)[1] for p in cfg.parts]
    pl, _ = plan_scene(hi, geoms, bg, cfg.loading, np.random.default_rng(0), mode="layers")
    top = max((x.pos[2] for x in pl), default=0)
    print(f"layers check with max counts {hi}: {len(pl)} placed, top layer center z={top * 1000:.0f} mm "
          f"(bin inner height {bg.inner_size[2] * 1000:.0f} mm)")


if __name__ == "__main__":
    main()
