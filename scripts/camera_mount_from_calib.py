"""Camera pose in the BIN frame (camera.mount.T_bin_cam) from the real cell's calibration.

    python scripts/camera_mount_from_calib.py --T-base-cam calib.txt --units mm \
        --rim-corners  x0 y0 z0  x1 y1 z1  x2 y2 z2  [--bin-height 0.25]

--T-base-cam : camera pose in the robot base frame (hand-eye result, e.g. the Kabsch/SVD registration of PhoXi
               XYZ <-> base XYZ). A text/.npy file with 16 numbers (4x4, row-major) or 16 numbers on the command
               line. Camera axes as PhoXi / OpenCV: x right, y down, z forward.
--rim-corners: three inner corners of the bin's top rim touched with the robot TCP, in the base frame:
               corner 0, then the corner along the bin's LONG side (-> bin x), then the one along the SHORT side
               (-> bin y). The bin frame is the one the generator uses: origin at the bottom centre, z up.
--units      : units of the translations above (mm | m).

Prints the YAML to paste under `camera.mount` (configs/cameras/*.yaml or configs/default.yaml).
"""
import argparse
import os

import numpy as np


def read_matrix(values):
    if len(values) == 1 and os.path.exists(values[0]):
        p = values[0]
        v = np.load(p) if p.endswith(".npy") else np.loadtxt(p, delimiter=None if "," not in open(p).read() else ",")
    else:
        v = np.array([float(x) for x in values])
    v = np.asarray(v, float).reshape(4, 4)
    if np.abs(v[3] - [0, 0, 0, 1]).max() > 1e-6:
        raise ValueError(f"not a homogeneous transform (last row {v[3]})")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T-base-cam", nargs="+", required=True)
    ap.add_argument("--rim-corners", nargs=9, type=float, required=True, metavar="XYZ")
    ap.add_argument("--units", choices=["mm", "m"], default="mm")
    ap.add_argument("--bin-height", type=float, default=0.25, help="outer bin height in m (config bin.size_m[2])")
    a = ap.parse_args()
    s = 0.001 if a.units == "mm" else 1.0

    T_base_cam = read_matrix(a.T_base_cam)
    T_base_cam[:3, 3] *= s
    R = T_base_cam[:3, :3]
    if abs(np.linalg.det(R) - 1) > 1e-3 or np.abs(R @ R.T - np.eye(3)).max() > 1e-3:
        raise ValueError("rotation part of T-base-cam is not orthonormal")

    c = np.array(a.rim_corners, float).reshape(3, 3) * s
    x = c[1] - c[0]
    y = c[2] - c[0]
    long_m, short_m = np.linalg.norm(x), np.linalg.norm(y)
    x /= long_m
    y -= x * (x @ y)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    if z[2] < 0:  # corners given clockwise seen from above: keep z up
        y, z = -y, -z
    rim_centre = c[0] + (c[1] - c[0]) / 2 + (c[2] - c[0]) / 2
    T_base_bin = np.eye(4)
    T_base_bin[:3, :3] = np.stack([x, y, z], 1)
    T_base_bin[:3, 3] = rim_centre - z * a.bin_height

    T_bin_cam = np.linalg.inv(T_base_bin) @ T_base_cam
    eye = T_bin_cam[:3, 3]
    tilt = np.degrees(np.arccos(np.clip(-T_bin_cam[2, 2], -1, 1)))
    print(f"# bin inner rim measured: {long_m * 1000:.1f} x {short_m * 1000:.1f} mm "
          f"(config bin.size_m minus 2x wall should match)")
    print(f"# camera in bin frame: x {eye[0] * 1000:.1f}  y {eye[1] * 1000:.1f}  z {eye[2] * 1000:.1f} mm above the bin bottom, "
          f"optical axis {tilt:.1f} deg from straight down")
    print("  mount:")
    print("    T_bin_cam: [" + ", ".join(f"{v:.6f}" for v in T_bin_cam.flatten()) + "]")


if __name__ == "__main__":
    main()
