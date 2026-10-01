"""Export generated data as ONE standard BOP dataset with per-instance masks, for 6D pose training
(bop_toolkit / GDRNPP / CosyPose / FFB6D-style loaders).

    python scripts/export_bop.py <output_dir> <bop_dir> [--workers 8] [--amodal] [--copy] [--scenes N]

<output_dir>: a run's output_dir (part_XX/shard_YY/... or shard_YY/...); all complete scenes are merged (scene ids are
unique across parts / shards). <bop_dir> gets the BOP layout
(https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md):

  camera.json, models/ (mm, models_info.json incl. symmetries)
  train_pbr/<scene_id>/
    rgb/, depth/                  links to the generated images (--copy: real copies)
    mask_visib/<im>_<gt>.png      one 8-bit PNG per ground-truth instance (the gt index in scene_gt.json):
                                  255 = visible pixels of that instance, 0 = everything else
    mask/<im>_<gt>.png            (--amodal) the instance's whole silhouette, occluded parts included; also fills
                                  px_count_all / visib_fract / bbox_obj of scene_gt_info.json (~1-6 s per image)
    scene_camera.json, scene_gt.json, scene_gt_info.json, scene_gt_coco.json

Instances without any visible pixel get their file too (BOP loaders expect one per gt), but as a hard link to one
empty mask per scene: no extra space and no extra inode (~45 % of the instances). Scenes already exported are
skipped, so it can be re-run while a run is still producing scenes. Depth stays in BOP units (depth PNG x
scene_camera depth_scale = mm).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from binsim.poses import rasterize_silhouette  # noqa: E402

DONE = ".bop_export.json"
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 9]  # mostly-black masks: 1.5 KB each (level 1: 6 KB), half the disk use


def _json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def load_ply(path):
    """(vertices (N,3) mm, faces (F,3)) of a binary little-endian / ascii PLY with triangle faces."""
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"bad PLY header: {path}")
            hdr += line
        head = hdr.decode("ascii", "replace")
        nv = int(head.split("element vertex ")[1].split()[0])
        nf = int(head.split("element face ")[1].split()[0])
        vprops = head.split("element vertex ")[1].split("element")[0].count("property")
        if "format ascii" in head:
            v = np.array([f.readline().split()[:3] for _ in range(nv)], float)
            fc = np.array([f.readline().split()[1:4] for _ in range(nf)], np.int64)
            return v, fc
        v = np.frombuffer(f.read(nv * 4 * vprops), "<f4").reshape(nv, vprops)[:, :3].astype(np.float64)
        fprops = head.split("element face ")[1]
        extra = [ln.split()[1] for ln in fprops.splitlines()[1:] if ln.startswith("property") and "list" not in ln]
        dt = [("n", "u1"), ("v", "<i4", 3)] + [(f"x{i}", {"uchar": "u1", "ushort": "<u2", "int": "<i4",
                                                          "uint": "<u4", "float": "<f4"}[t]) for i, t in enumerate(extra)]
        fc = np.frombuffer(f.read(nf * np.dtype(dt).itemsize), np.dtype(dt))["v"].astype(np.int64)
        return v, fc


def find_scenes(src):
    """Complete scenes of every shard under src -> {scene_name: (scene_dir, shard_dir)}."""
    shards = sorted(glob.glob(os.path.join(src, "shard_*"))) + sorted(glob.glob(os.path.join(src, "part_*", "shard_*")))
    scenes = {}
    for sh in shards:
        for sd in sorted(glob.glob(os.path.join(sh, "train_pbr", "*"))):
            name = os.path.basename(sd)
            if not name.isdigit() or not os.path.exists(os.path.join(sd, "scene_info.json")):
                continue
            if name in scenes:
                raise SystemExit(f"scene id {name} exists twice: {scenes[name][0]} and {sd}")
            scenes[name] = (sd, sh)
    return shards, scenes


def export_scene(job):
    sd, dst, models, amodal, copy = job
    t0 = time.monotonic()
    tmp = dst + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for sub in ("rgb", "depth"):
        if os.path.isdir(os.path.join(sd, sub)):
            if copy:
                shutil.copytree(os.path.join(sd, sub), os.path.join(tmp, sub))
            else:
                os.symlink(os.path.abspath(os.path.join(sd, sub)), os.path.join(tmp, sub))
    for name in ("scene_camera.json", "scene_gt.json", "scene_gt_coco.json"):
        if os.path.exists(os.path.join(sd, name)):
            shutil.copyfile(os.path.join(sd, name), os.path.join(tmp, name))
    cam = json.load(open(os.path.join(sd, "scene_camera.json")))
    gt = json.load(open(os.path.join(sd, "scene_gt.json")))
    gti = json.load(open(os.path.join(sd, "scene_gt_info.json")))
    subs = ["mask_visib"] + (["mask"] if amodal else [])
    for s in subs:
        os.makedirs(os.path.join(tmp, s))
    n_files = n_links = 0
    empty = None
    for k in sorted(cam, key=int):
        im = int(k)
        inst_path = os.path.join(sd, "mask_visib_inst", f"{im:06d}.png")
        inst = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
        if inst is None:
            raise RuntimeError(f"missing {inst_path}")
        H, W = inst.shape
        if empty is None:
            empty = os.path.join(tmp, ".empty_mask.png")  # link target of every empty mask
            cv2.imwrite(empty, np.zeros((H, W), np.uint8), PNG)
        objs = ndimage.find_objects(inst, max_label=len(gt[k]))
        K = np.array(cam[k]["cam_K"], float).reshape(3, 3)
        buf = np.zeros((H, W), np.uint8)
        for j, (g, info) in enumerate(zip(gt[k], gti[k])):
            path = os.path.join(tmp, "mask_visib", f"{im:06d}_{j:06d}.png")
            sl = objs[j] if j < len(objs) else None
            if sl is None:
                os.link(empty, path)
                n_links += 1
            else:
                buf[sl] = (inst[sl] == j + 1) * np.uint8(255)
                cv2.imwrite(path, buf, PNG)
                buf[sl] = 0
                n_files += 1
            if amodal:
                mpath = os.path.join(tmp, "mask", f"{im:06d}_{j:06d}.png")
                if sl is None:  # not visible: no silhouette needed for training, counted as outside the image
                    os.link(empty, mpath)
                    n_links += 1
                    info.update(px_count_all=0, visib_fract=0.0)
                    continue
                v, fc = models[int(g["obj_id"])]
                R = np.array(g["cam_R_m2c"], float).reshape(3, 3)
                t = np.array(g["cam_t_m2c"], float)
                full = rasterize_silhouette(v @ R.T + t, fc, K, W, H) | (inst == j + 1)  # visible px are inside
                cv2.imwrite(mpath, full.astype(np.uint8) * 255, PNG)
                n_files += 1
                ys, xs = np.nonzero(full)
                px_all = int(len(ys))
                info.update(px_count_all=px_all, visib_fract=min(info["px_count_visib"] / max(px_all, 1), 1.0),
                            bbox_obj=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)])
    _json(os.path.join(tmp, "scene_gt_info.json"), gti)
    _json(os.path.join(tmp, DONE), {"source": os.path.abspath(sd), "amodal": amodal, "copy": copy,
                                    "mask_files": n_files, "empty_links": n_links})
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return os.path.basename(dst), n_files, n_links, time.monotonic() - t0


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("bop_dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amodal", action="store_true", help="also mask/ + px_count_all / visib_fract / bbox_obj")
    ap.add_argument("--copy", action="store_true", help="copy rgb/ depth/ instead of linking them")
    ap.add_argument("--scenes", type=int, default=0, help="export at most this many scenes (0 = all)")
    a = ap.parse_args()
    src, dst = os.path.abspath(os.path.expanduser(a.output_dir)), os.path.abspath(os.path.expanduser(a.bop_dir))
    shards, scenes = find_scenes(src)
    if not scenes:
        raise SystemExit(f"no complete scenes under {src}")
    first = shards[0]
    for sh in shards[1:]:  # one dataset = one camera model / one set of models
        for f in ("camera.json", "models/models_info.json"):
            if os.path.exists(os.path.join(sh, f)) and md5(os.path.join(sh, f)) != md5(os.path.join(first, f)):
                raise SystemExit(f"{sh}/{f} differs from {first}/{f}: export these runs separately")
    os.makedirs(os.path.join(dst, "train_pbr"), exist_ok=True)
    shutil.copyfile(os.path.join(first, "camera.json"), os.path.join(dst, "camera.json"))
    if not os.path.isdir(os.path.join(dst, "models")):
        shutil.copytree(os.path.join(first, "models"), os.path.join(dst, "models"))
    models = {int(os.path.basename(p)[4:10]): load_ply(p) for p in glob.glob(os.path.join(dst, "models", "obj_*.ply"))}
    todo = []
    for name, (sd, _) in sorted(scenes.items()):
        out = os.path.join(dst, "train_pbr", name)
        if os.path.exists(os.path.join(out, DONE)):
            done = json.load(open(os.path.join(out, DONE)))
            if done.get("amodal") or not a.amodal:
                continue
        todo.append((sd, out, models, a.amodal, a.copy))
    if a.scenes:
        todo = todo[:a.scenes]
    print(f"{len(scenes)} complete scenes in {len(shards)} shard(s); exporting {len(todo)} -> {dst}", flush=True)
    t0, files, links = time.monotonic(), 0, 0
    with Pool(max(1, a.workers)) as pool:
        for i, (name, nf, nl, dt) in enumerate(pool.imap_unordered(export_scene, todo), 1):
            files, links = files + nf, links + nl
            el = time.monotonic() - t0
            print(f"[{i}/{len(todo)}] scene {name}: {nf} mask files, {nl} empty links, {dt:.0f}s | "
                  f"eta {el / i * (len(todo) - i) / 60:.0f} min", flush=True)
    print(f"done: {files} mask files + {links} links in {(time.monotonic() - t0) / 60:.1f} min -> {dst}")


if __name__ == "__main__":
    main()
