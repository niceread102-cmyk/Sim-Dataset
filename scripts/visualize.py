"""Render the annotated bin view (6DOF axes, top-most part arrow, depth, masks, per-part table)
for any saved scene — no Isaac needed.

    python scripts/visualize.py <shard_dir> [--scene 0] [--view 0] [--out viz.jpg] [--show]
    python scripts/visualize.py <shard_dir> --all [--view 0]   # every scene -> <shard_dir>/viz/<scene>_<view>_bin.jpg
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from binsim.bop_io import BopScene, list_scenes, load_models  # noqa: E402
from binsim.transforms import inv_T  # noqa: E402
from binsim.viz import bin_panel, part_table_lines  # noqa: E402


def build_parts(scene: BopScene, im_id: int, models, info):
    parts = []
    Tw2c = scene.cam_T_w2c(im_id)
    Tc2w = inv_T(Tw2c)
    for k, (g, gi) in enumerate(zip(scene.gt[str(im_id)], scene.gt_info[str(im_id)])):
        R = np.array(g["cam_R_m2c"]).reshape(3, 3)
        t = np.array(g["cam_t_m2c"])
        T_cam = np.eye(4)
        T_cam[:3, :3], T_cam[:3, 3] = R, t / 1000
        T_bin = Tc2w @ T_cam
        ext = models[g["obj_id"]].extents
        parts.append({"label": k, "name": info[str(g["obj_id"])].get("name", f"obj{g['obj_id']}"), "obj_id": g["obj_id"],
                      "R": R, "t_mm": t, "R_bin": T_bin[:3, :3], "t_bin_mm": T_bin[:3, 3] * 1000,
                      "visib": gi["visib_fract"] if gi["visib_fract"] is not None else float("nan"), "axis_len_m": 0.6 * float(ext.max()), "n_valid_grasps": 0})
    return parts


def render_scene(shard_dir, sid, view, models, info, title):
    scene = BopScene(shard_dir, sid)
    parts = build_parts(scene, view, models, info)
    top, top_g = None, None
    gj = scene.grasps(view)
    if gj:
        top = gj.get("topmost_gt_idx")
        cands = [g for g in gj["grasps"] if g["valid"]]
        for g in cands:
            parts[g["gt_idx"]]["n_valid_grasps"] += 1
        cands = [g for g in cands if g["gt_idx"] == top]
        if cands:
            g = min(cands, key=lambda x: x["approach_angle_deg"])
            top_g = np.eye(4)
            top_g[:3, :3] = np.array(g["cam_R"]).reshape(3, 3)
            top_g[:3, 3] = np.array(g["cam_t_mm"]) / 1000
    img = bin_panel(scene.rgb(view), scene.depth_m(view), scene.instance_map(view), scene.K(view), parts, top, top_g, title)
    return img, parts


def render_all(shard_dir, view):
    """Annotated image of `view` for every finished scene (scene_info.json written), same name as the generator's."""
    models, info = load_models(shard_dir)
    out_dir = os.path.join(shard_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)
    done = skipped = 0
    for sid in list_scenes(shard_dir):
        out = os.path.join(out_dir, f"{sid:06d}_{view:02d}_bin.jpg")
        if os.path.exists(out) or not os.path.exists(os.path.join(shard_dir, "train_pbr", f"{sid:06d}", "scene_info.json")):
            skipped += 1
            continue
        img, _ = render_scene(shard_dir, sid, view, models, info, f"scene {sid:06d} view {view}")
        img.save(out, quality=90)
        done += 1
    print(f"{shard_dir}: wrote {done} images, skipped {skipped} (existing or unfinished) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--scene", type=int, default=None, help="scene id (default: first)")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--print-table", action="store_true")
    ap.add_argument("--all", action="store_true", help="every finished scene (skips existing images)")
    a = ap.parse_args()
    if a.all:
        render_all(a.shard_dir, a.view)
        return
    sid = a.scene if a.scene is not None else list_scenes(a.shard_dir)[0]
    scene = BopScene(a.shard_dir, sid)
    models, info = load_models(a.shard_dir)
    parts = build_parts(scene, a.view, models, info)
    top, top_g = None, None
    gj = scene.grasps(a.view)
    if gj:
        top = gj.get("topmost_gt_idx")
        cands = [g for g in gj["grasps"] if g["valid"]]
        for g in cands:
            parts[g["gt_idx"]]["n_valid_grasps"] += 1
        cands = [g for g in cands if g["gt_idx"] == top]
        if cands:
            g = min(cands, key=lambda x: x["approach_angle_deg"])
            top_g = np.eye(4)
            top_g[:3, :3] = np.array(g["cam_R"]).reshape(3, 3)
            top_g[:3, 3] = np.array(g["cam_t_mm"]) / 1000
    img = bin_panel(scene.rgb(a.view), scene.depth_m(a.view), scene.instance_map(a.view), scene.K(a.view), parts, top, top_g,
                    f"scene {sid:06d} view {a.view} (offline)")
    out = a.out or os.path.join(a.shard_dir, "viz", f"{sid:06d}_{a.view:02d}_offline.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out, quality=92)
    print(f"wrote {out}")
    if a.print_table:
        print("\n".join(part_table_lines(parts)))
    if a.show:
        img.show()


if __name__ == "__main__":
    main()
