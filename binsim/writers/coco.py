"""COCO instance-segmentation JSON per scene (same layout as bop_toolkit's scene_gt_coco.json).
Masks are uncompressed RLE (column-major), readable by pycocotools / detectron2 / mmdet."""
from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray) -> dict:
    h, w = mask.shape
    flat = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    if flat.size == 0:
        return {"counts": [0], "size": [h, w]}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.r_[0, change, flat.size]).tolist()
    if flat[0] == 1:
        runs = [0] + runs  # COCO RLE always starts with a run of zeros
    return {"counts": runs, "size": [h, w]}


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in rle["counts"]:
        flat[pos:pos + c] = val
        pos += c
        val ^= 1
    return flat.reshape((w, h)).T.astype(bool)


class CocoScene:
    def __init__(self, categories: list[dict]):
        self.data = {"info": {"description": "binsim synthetic bin-picking"}, "licenses": [],
                     "categories": categories, "images": [], "annotations": []}
        self._ann = 0

    def add_image(self, im_id: int, width: int, height: int):
        self.data["images"].append({"id": im_id, "file_name": f"rgb/{im_id:06d}.png", "width": width, "height": height})

    def add_instance(self, im_id: int, obj_id: int, mask_visib: np.ndarray, bbox_visib, visib_fract: float,
                     ignore_below: float = 0.1):
        area = int(mask_visib.sum())
        if area == 0:
            return
        self.data["annotations"].append({
            "id": self._ann, "image_id": im_id, "category_id": obj_id, "iscrowd": 0, "area": area,
            "bbox": bbox_visib, "segmentation": rle_encode(mask_visib), "visib_fract": visib_fract,
            "ignore": bool(visib_fract < ignore_below),
        })
        self._ann += 1
