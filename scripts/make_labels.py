"""Simple per-image labels next to the BOP / COCO files: which parts the camera sees, which are in the bin but hidden,
each with x y z (mm) + roll pitch yaw (deg). Also writes README.md at the dataset root.

    python3 scripts/make_labels.py <output_dir> [--workers 8] [--force]

For every complete scene (any part_XX/shard_YY or shard_YY under <output_dir>):
  labels/<im>.csv     one row per part IN THE BIN (visible + fully hidden). Parts already picked (moved out of the bin)
                      are left out - they are not in the image any more.
                      visible = 1: the camera sees >= 1 pixel of it (visible_px, bbox, mask_value in mask_visib_inst)
                      visible = 0: in the bin but completely covered
                      x_mm..yaw_deg: part pose in the CAMERA frame (OpenCV: x right, y down, z forward)
                      bin_x_mm..bin_yaw_deg: the same in the BIN frame (origin = bin bottom centre, z up)
  labels/camera.csv   per image: intrinsics (fx fy cx cy, size), depth unit, camera pose in the bin frame
rpy = extrinsic XYZ (roll about x, then pitch about y, then yaw about z; R = Rz Ry Rx), as in URDF.
Scenes already labelled are skipped (--force: redo), so it can run while a run is still producing scenes.
"""
import argparse
import csv
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from binsim.transforms import mat_to_rpy  # noqa: E402

PART_COLS = ["gt_idx", "mask_value", "visible", "visible_px", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
             "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg",
             "bin_x_mm", "bin_y_mm", "bin_z_mm", "bin_roll_deg", "bin_pitch_deg", "bin_yaw_deg"]
CAM_COLS = ["image", "width", "height", "fx", "fy", "cx", "cy", "depth_unit_mm",
            "cam_x_mm", "cam_y_mm", "cam_z_mm", "cam_roll_deg", "cam_pitch_deg", "cam_yaw_deg"]


def complete_scenes(root):
    shards = sorted(glob.glob(os.path.join(root, "shard_*"))) + sorted(glob.glob(os.path.join(root, "part_*", "shard_*")))
    return [(sd, sh) for sh in shards for sd in sorted(glob.glob(os.path.join(sh, "train_pbr", "*")))
            if os.path.basename(sd).isdigit() and os.path.exists(os.path.join(sd, "scene_info.json"))]


def bin_box(shard):
    """Outer bin box (bin frame, mm) + 50 mm: centres outside it were picked / parked away."""
    b = (yaml.safe_load(open(os.path.join(shard, "config_resolved.yaml"))) or {}).get("bin", {})
    sx, sy, sz = [1000.0 * v for v in b.get("size_m", [0.6, 0.4, 0.25])]
    m = 50.0
    return np.array([-sx / 2 - m, -sy / 2 - m, -m]), np.array([sx / 2 + m, sy / 2 + m, sz + m])


def r5(v):
    return [round(float(x), 5) for x in v]


def label_scene(job):
    sd, shard, force = job
    out = os.path.join(sd, "labels")
    cam = json.load(open(os.path.join(sd, "scene_camera.json")))
    ims = sorted(cam, key=int)
    if not force and os.path.isfile(os.path.join(out, "camera.csv")) and \
            len(glob.glob(os.path.join(out, "[0-9]*.csv"))) == len(ims):
        return os.path.basename(sd), 0, 0, 0
    gt = json.load(open(os.path.join(sd, "scene_gt.json")))
    gti = json.load(open(os.path.join(sd, "scene_gt_info.json")))
    lo, hi = bin_box(shard)
    info = json.load(open(os.path.join(os.path.dirname(os.path.dirname(sd)), "camera.json")))
    os.makedirs(out, exist_ok=True)
    n_vis = n_hid = 0
    cam_rows = []
    for k in ims:
        c = cam[k]
        Rw = np.array(c["cam_R_w2c"], float).reshape(3, 3)  # bin -> camera
        tw = np.array(c["cam_t_w2c"], float)
        K = c["cam_K"]
        cam_pos = -Rw.T @ tw
        cam_rows.append([f"rgb/{int(k):06d}.png", info["width"], info["height"], K[0], K[4], K[2], K[5],
                         c.get("depth_scale", 0.1), *r5(cam_pos), *r5(mat_to_rpy(Rw.T))])
        rows = []
        for j, (g, x) in enumerate(zip(gt[k], gti[k])):
            R = np.array(g["cam_R_m2c"], float).reshape(3, 3)
            t = np.array(g["cam_t_m2c"], float)
            p_bin = Rw.T @ (t - tw)
            if np.any(p_bin < lo) or np.any(p_bin > hi):
                continue  # picked / parked outside the bin
            px = int(x["px_count_visib"])
            vis = px > 0
            n_vis += vis
            n_hid += not vis
            bb = x["bbox_visib"] if vis else ["", "", "", ""]
            rows.append([j, j + 1, int(vis), px, *bb, *r5(t), *r5(mat_to_rpy(R)), *r5(p_bin), *r5(mat_to_rpy(Rw.T @ R))])
        with open(os.path.join(out, f"{int(k):06d}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(PART_COLS)
            w.writerows(rows)
    with open(os.path.join(out, "camera.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CAM_COLS)
        w.writerows(cam_rows)
    return os.path.basename(sd), len(ims), n_vis, n_hid


README = """# 데이터셋 라벨 안내 (binsim, {name})

이미지 1장 = `train_pbr/<씬>/rgb/<이미지>.png`. 같은 번호의 파일끼리 한 세트입니다.
폴더 구조: `[part_XX/]shard_YY/train_pbr/<씬 6자리>/...` (씬 번호는 데이터셋 전체에서 겹치지 않음)

## 먼저 볼 것: `labels/` (사람이 읽기 쉬운 라벨, scripts/make_labels.py)

| 파일 | 내용 |
|---|---|
| `labels/<이미지>.csv` | 빈 안에 있는 부품 한 줄씩. `visible=1` 카메라에 보임, `visible=0` 빈 안에 있지만 완전히 가려짐. 이미 집어서 치운 부품은 없음 |
| `labels/camera.csv` | 이미지마다 카메라 내부값(fx fy cx cy), depth 단위, 빈 좌표계에서의 카메라 위치·자세 |

- 정면에 보이는 데이터 = `visible == 1` 인 줄 / 가려진 것 포함 전체 = 모든 줄
- `x_mm y_mm z_mm roll_deg pitch_deg yaw_deg` = **카메라 좌표계** 자세 (OpenCV: x 오른쪽, y 아래, z 앞)
- `bin_x_mm ... bin_yaw_deg` = **빈 좌표계** 자세 (원점 = 빈 바닥 중앙, z 위)
- rpy = 고정축 XYZ 순서 (roll: x축 → pitch: y축 → yaw: z축, R = Rz·Ry·Rx), 단위 도. URDF와 같은 규칙
- `mask_value` = `mask_visib_inst/<이미지>.png`에서 그 부품 픽셀의 값 (0 = 배경)
- `bbox_x bbox_y bbox_w bbox_h` = 보이는 부분의 박스 (픽셀), `visible_px` = 보이는 픽셀 수
- `gt_idx` = BOP `scene_gt.json`의 순번 (같은 부품)

## 이미지

| 폴더 | 내용 |
|---|---|
| `rgb/` | 컬러 이미지 (8비트 PNG) |
| `depth/` | depth, uint16. **값 × 0.1 = mm**, 0 = 측정 안 됨 (PhoXi 노이즈·구멍 포함) |
| `depth_vis/` | depth 미리보기 (보기용, 학습에 쓰지 않음) |
| `mask_visib_inst/` | 보이는 부품 마스크 한 장, uint16. 픽셀값 = `mask_value` (0 = 배경) |

## 표준 형식 (학습 도구용, 내용은 위와 같음)

| 파일 | 형식 |
|---|---|
| `scene_gt.json`, `scene_gt_info.json`, `scene_camera.json` | BOP 6D 포즈 형식 (씬에 넣은 부품 전부, 치운 부품은 `px_count_visib = 0`) |
| `scene_gt_coco.json` | COCO 인스턴스 분할 (보이는 부품만, `gt_idx` 포함) |
| `poses/<이미지>.csv` | 예전 포즈 표 (치운 부품 포함, 보임 여부 없음) → `labels/`를 쓰세요 |
| `../models/obj_000001.ply`, `models_info.json` | 부품 3D 모델 (mm), 지름, 대칭 (z축 180°) |
| `../camera.json` | 카메라 해상도와 내부값 |

BOP 표준 폴더(부품별 흑백 마스크)가 필요하면: `python3 scripts/export_bop.py <이 폴더> <출력 폴더>`
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="redo scenes that already have labels")
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.output_dir))
    scenes = complete_scenes(root)
    if not scenes:
        raise SystemExit(f"no complete scenes under {root}")
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write(README.format(name=os.path.basename(root)))
    done = vis = hid = 0
    with Pool(max(1, a.workers)) as pool:
        for i, (name, n, v, h) in enumerate(pool.imap_unordered(label_scene, [(sd, sh, a.force) for sd, sh in scenes]), 1):
            done += n > 0
            vis, hid = vis + v, hid + h
            if i % 50 == 0 or i == len(scenes):
                print(f"[{i}/{len(scenes)}] labelled {done} scenes (visible {vis}, hidden in bin {hid})", flush=True)
    print(f"README: {os.path.join(root, 'README.md')}")


if __name__ == "__main__":
    main()
