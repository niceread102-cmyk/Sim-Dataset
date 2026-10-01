"""Merge the per-scene COCO files of all shards into train / val sets split by PILE, for Mask R-CNN, mmdet, ...

    python scripts/export_coco.py <output_dir> [--val 0.1] [--seed 0] [--small-area 100 --small keep|ignore|drop]

All views of one pile show the same parts (camera moved a few cm, the pile shaken a little), so an image-level
split leaks near-duplicates into val: every scene goes entirely to train or to val.
Writes <output_dir>/coco/{train,val}.json with file_name relative to <output_dir> and pycocotools-style compressed
RLE, and <output_dir>/coco/split.json (scene lists - use the same split for the 6D-pose / BOP side).
--small: annotations with fewer than --small-area visible pixels are kept (default), marked iscrowd=1 (ignore:
detectron2 / mmdet leave them out of the loss, COCO eval ignores them) or dropped.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def rle_strings(count_lists) -> list[str]:
    """pycocotools' compressed RLE (maskApi.c rleToString) for many uncompressed count lists at once."""
    lens = np.array([len(c) for c in count_lists])
    cnt = np.concatenate([np.asarray(c, np.int64) for c in count_lists])
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    pos = np.arange(len(cnt)) - np.repeat(starts, lens)
    x = cnt.copy()
    d = np.flatnonzero(pos > 2)
    x[d] -= cnt[d - 2]  # counts after the 3rd are stored as the difference to the count two before
    codes = np.zeros((len(x), 8), np.uint8)
    nch = np.zeros(len(x), np.int64)
    active = np.ones(len(x), bool)
    for j in range(8):  # 5 bits per char, the last char carries the sign
        c = x & 0x1F
        x = x >> 5
        more = np.where(c & 0x10, x != -1, x != 0)
        codes[:, j] = np.where(active, (c | np.where(more, 0x20, 0)) + 48, 0)
        nch += active
        active &= more
        if not active.any():
            break
    chars = codes[np.arange(8)[None, :] < nch[:, None]].tobytes().decode("ascii")
    per = np.add.reduceat(nch, starts)
    cut = np.r_[0, np.cumsum(per)]
    return [chars[a:b] for a, b in zip(cut[:-1], cut[1:])]


def rle_counts_from_string(s: str) -> list[int]:
    """Inverse of rle_strings (maskApi.c rleFrString) - used by the self-test."""
    cnts, p = [], 0
    while p < len(s):
        x = k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def complete_scenes(out):
    scenes = []
    for sh in sorted(d for d in os.listdir(out) if d.startswith("shard_")):
        tp = os.path.join(out, sh, "train_pbr")
        for sc in sorted(os.listdir(tp)) if os.path.isdir(tp) else []:
            d = os.path.join(tp, sc)
            if os.path.exists(os.path.join(d, "scene_info.json")) and os.path.exists(os.path.join(d, "scene_gt_coco.json")):
                scenes.append(d)
    return scenes


def export(out, scenes, path, small_area, small, cat_holder):
    images, n_ann, n_small = [], 0, 0
    with open(path + ".tmp", "w") as f:
        f.write('{"info": {"description": "binsim synthetic bin-picking (merged)"}, "licenses": [], "annotations": [')
        first = True
        for sd in scenes:
            coco = json.load(open(os.path.join(sd, "scene_gt_coco.json")))
            cat_holder.setdefault("categories", coco["categories"])
            base_id = len(images)
            idmap = {}
            for im in coco["images"]:
                idmap[im["id"]] = base_id + len(idmap)
                rel = os.path.relpath(sd, out)
                images.append({"id": idmap[im["id"]], "file_name": os.path.join(rel, im["file_name"]),
                               "width": im["width"], "height": im["height"], "scene": rel, "im_id": im["id"],
                               "depth_file": os.path.join(rel, "depth", f"{im['id']:06d}.png"),
                               "inst_file": os.path.join(rel, "mask_visib_inst", f"{im['id']:06d}.png")})
            anns = coco["annotations"]
            if not anns:
                continue
            strs = rle_strings([a["segmentation"]["counts"] for a in anns])
            for a, s in zip(anns, strs):
                is_small = a["area"] < small_area
                n_small += is_small
                if is_small and small == "drop":
                    continue
                crowd = int(is_small and small == "ignore")
                rec = {"id": n_ann, "image_id": idmap[a["image_id"]], "category_id": a["category_id"], "iscrowd": crowd,
                       "ignore": crowd, "area": a["area"], "bbox": a["bbox"],
                       "segmentation": {"size": a["segmentation"]["size"], "counts": s}}
                if "gt_idx" in a:  # index into scene_gt.json of this image (the part's 6D pose)
                    rec["gt_idx"] = a["gt_idx"]
                f.write(("" if first else ",") + json.dumps(rec, separators=(",", ":")))
                first = False
                n_ann += 1
        f.write('], "images": ' + json.dumps(images, separators=(",", ":")))
        f.write(', "categories": ' + json.dumps(cat_holder.get("categories", [])) + "}")
    os.replace(path + ".tmp", path)
    return len(images), n_ann, n_small


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("--val", type=float, default=0.1, help="fraction of piles (scenes) for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--small-area", type=int, default=100)
    ap.add_argument("--small", choices=["keep", "ignore", "drop"], default="keep")
    a = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(a.output_dir))
    scenes = complete_scenes(out)
    if not scenes:
        sys.exit(f"no complete scenes under {out}")
    order = np.random.default_rng(a.seed).permutation(len(scenes))
    n_val = int(round(a.val * len(scenes))) if len(scenes) > 1 else 0
    if a.val > 0 and len(scenes) > 1:
        n_val = max(n_val, 1)
    val = sorted(scenes[i] for i in order[:n_val])
    train = sorted(scenes[i] for i in order[n_val:])
    d = os.path.join(out, "coco")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "split.json"), "w") as f:
        json.dump({"train": [os.path.relpath(s, out) for s in train], "val": [os.path.relpath(s, out) for s in val],
                   "seed": a.seed}, f, indent=1)
    cats = {}
    for name, sc in (("train", train), ("val", val)):
        n_im, n_ann, n_small = export(out, sc, os.path.join(d, f"{name}.json"), a.small_area, a.small, cats)
        print(f"{name}: {len(sc)} piles, {n_im} images, {n_ann} annotations "
              f"({n_small} under {a.small_area} px: {a.small}) -> {os.path.join(d, name + '.json')}")


if __name__ == "__main__":
    main()
