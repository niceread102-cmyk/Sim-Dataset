"""Which detected part to pick: only the crate under the camera (an ROI), then the parts that lie flat, the highest, the nearest to the crate centre.

Used by vision_pick.py; no ROS. The crate is `crate:` of the sequence yaml (world, m):
    crate: {centre: [x, y], size: [x, y], yaw: deg, margin: m, floor_z: m}
size = the usable inner floor of the crate (its x along yaw), margin = how far from the walls a part must lie, floor_z = top of the crate floor.
The ROI is that rectangle projected into the image (the camera pose of the frame): outside the whole floor everything is blacked out before the pose network
sees it (a part at a wall stays whole), and a part is picked only when its centre lies inside the floor minus margin.
"""
import cv2
import numpy as np


def roi_polygon(crate, T_world_cam, K, dist, margin=None):
    """the inner floor of the crate shrunk by margin (m; None = crate["margin"]) -> its 4 corners in image pixels (int32, 4 x 2)"""
    cx, cy = crate["centre"]
    margin = crate["margin"] if margin is None else margin
    w, h = crate["size"][0] / 2 - margin, crate["size"][1] / 2 - margin
    a = np.radians(crate["yaw"])
    c, s = np.cos(a), np.sin(a)
    local = np.array([[w, h], [-w, h], [-w, -h], [w, -h]])
    xy = local @ np.array([[c, s], [-s, c]]) + [cx, cy]                   # turned by yaw, moved to the centre
    pts_world = np.column_stack([xy, np.full(4, crate["floor_z"])])
    T_cam_world = np.linalg.inv(T_world_cam)
    rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
    uv, _ = cv2.projectPoints(pts_world, rvec, T_cam_world[:3, 3], K, np.array(dist, dtype=float))
    return uv.reshape(-1, 2).astype(np.int32)


def mask_outside(rgb, depth, poly):
    """rgb and depth with everything outside the ROI = 0 (not cropped: K stays valid)"""
    m = np.zeros(depth.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return cv2.bitwise_and(rgb, rgb, mask=m), np.where(m > 0, depth, 0).astype(depth.dtype)


def tilt_deg(R_cam_obj, T_world_cam):
    """angle between the local z of the part (its axis) and world z: 0 = lying flat, hub up (standard) or down (turn_over)"""
    R_world = T_world_cam[:3, :3] @ np.array(R_cam_obj)
    return float(np.degrees(np.arccos(min(1.0, abs(R_world[2, 2])))))


def rank(poses, T_world_cam, K, dist, poly=None, crate=None, tilt_bins=(10.0, 30.0), z_band=0.01):
    """the parts to pick, best first.
    Gates: confident, its centre inside the ROI (poly; None = no ROI), tilt <= tilt_bins[-1].
    Order: tilt <= tilt_bins[0] first -> within z_band (m) of the highest part -> nearest to the crate centre (none: to the image centre).
    Every kept pose gets `tilt_deg` (shown in the picture)"""
    keep = []
    for p in poses:
        if not p.get("confident"):
            continue
        t_cam = np.array(p["t"], dtype=float) / 1000.0                    # mm -> m
        if poly is not None:
            uv = cv2.projectPoints(t_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, np.array(dist, dtype=float))[0].ravel()
            if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(uv[0]), float(uv[1])), False) < 0:
                continue
        p["tilt_deg"] = tilt_deg(p["R"], T_world_cam)
        if p["tilt_deg"] > tilt_bins[-1]:
            continue
        keep.append((p, T_world_cam[:3, :3] @ t_cam + T_world_cam[:3, 3]))   # the centre of the part in the world
    if not keep:
        return []
    top = max(pw[2] for _, pw in keep)
    centre = crate["centre"] if crate else T_world_cam[:2, 3]

    def key(item):
        p, pw = item
        return (int(p["tilt_deg"] > tilt_bins[0]),                         # 0 = flat enough
                int((top - pw[2]) / z_band),                               # 0 = within z_band of the highest part
                float(np.hypot(pw[0] - centre[0], pw[1] - centre[1])))     # then the one nearest to the centre
    return [p for p, _ in sorted(keep, key=key)]
