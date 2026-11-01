"""pick_filter.py: which part to pick: detected parts inside the floor of the crate under the camera, or the parts the simulation sends for the crate
(rank_world), ranked by flat (hub up or down), height and distance to the crate centre. Used by vision_pick.py; no ROS. The crate is `crate:` of the sequence yaml.
"""
import math

import cv2
import numpy as np


def roi_polygon(crate, T_world_cam, K, dist, margin=None):
    cx, cy = crate["centre"]
    margin = crate["margin"] if margin is None else margin
    w, h = crate["size"][0] / 2 - margin, crate["size"][1] / 2 - margin
    a = np.radians(crate["yaw"])
    c, s = np.cos(a), np.sin(a)
    local = np.array([[w, h], [-w, h], [-w, -h], [w, -h]])
    xy = local @ np.array([[c, s], [-s, c]]) + [cx, cy]
    pts_world = np.column_stack([xy, np.full(4, crate["floor_z"])])
    T_cam_world = np.linalg.inv(T_world_cam)
    rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
    uv, _ = cv2.projectPoints(pts_world, rvec, T_cam_world[:3, 3], K, np.array(dist, dtype=float))
    return uv.reshape(-1, 2).astype(np.int32)


def mask_outside(rgb, depth, poly):
    m = np.zeros(depth.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return cv2.bitwise_and(rgb, rgb, mask=m), np.where(m > 0, depth, 0).astype(depth.dtype)


def tilt_deg(R_cam_obj, T_world_cam):
    R_world = T_world_cam[:3, :3] @ np.array(R_cam_obj)
    return float(np.degrees(np.arccos(min(1.0, abs(R_world[2, 2])))))


def rank(poses, T_world_cam, K, dist, poly=None, crate=None):
    cfg = crate or {}
    keep = []
    for p in poses:
        if not p.get("confident"):
            continue
        t_cam = np.array(p["t"], dtype=float) / 1000.0
        if poly is not None:
            uv = cv2.projectPoints(t_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, np.array(dist, dtype=float))[0].ravel()
            if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(uv[0]), float(uv[1])), False) < 0:
                continue
        R_world = T_world_cam[:3, :3] @ np.array(p["R"])
        p["tilt_deg"] = tilt_deg(p["R"], T_world_cam)
        keep.append((p, abs(float(R_world[2, 2])), T_world_cam[:3, :3] @ t_cam + T_world_cam[:3, 3]))
    return order(keep, cfg, T_world_cam[:2, 3])


def rank_world(parts, crate):
    return order([(p, p["flat"], p["xyz_m"]) for p in parts], crate or {}, (crate or {}).get("centre"))


def order(keep, cfg, fallback_centre):
    if not keep:
        return []
    w = {"flat": 3.0, "height": 1.0, "centre": 0.5, **(cfg.get("rank_weights") or {})}
    top = max(pw[2] for _, _, pw in keep)
    centre = cfg["centre"] if cfg.get("centre") is not None else fallback_centre
    reach = math.hypot(*cfg["size"]) / 2.0 if cfg.get("size") else 0.3
    scale = float(cfg.get("height_scale_m", 0.04))
    for p, flat, pw in keep:
        height = max(0.0, 1.0 - (top - pw[2]) / scale)
        centred = max(0.0, 1.0 - float(np.hypot(pw[0] - centre[0], pw[1] - centre[1])) / reach)
        p["rank_score"] = float(w["flat"]) * flat + float(w["height"]) * height + float(w["centre"]) * centred
    return [p for p, _, _ in sorted(keep, key=lambda item: -item[0]["rank_score"])]
