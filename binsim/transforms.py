"""Rigid-transform helpers (numpy only). Quaternions are (w, x, y, z) everywhere in binsim;
conversion to the simulator's order happens only in binsim/sim/compat.py."""
from __future__ import annotations

import numpy as np


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) wxyz with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.empty((R.shape[0], 4))
    for i, m in enumerate(R):
        t = np.trace(m)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(shape + (4,))


def quat_mul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def rpy_to_mat(rpy_deg) -> np.ndarray:
    """Extrinsic XYZ (roll about X, then pitch about Y, then yaw about Z): R = Rz @ Ry @ Rx."""
    r, p, y = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_to_rpy(R) -> np.ndarray:
    """Inverse of rpy_to_mat, degrees."""
    R = np.asarray(R, dtype=np.float64)
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.rad2deg([roll, pitch, yaw])


def axis_angle_to_mat(axis, angle_rad) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def make_T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def pose_to_T(pos, quat_wxyz) -> np.ndarray:
    return make_T(quat_to_mat(quat_wxyz), pos)


def inv_T(T) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T, pts) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def random_quaternions(n: int, rng: np.random.Generator) -> np.ndarray:
    """n uniformly random rotations (Shoemake), wxyz. Ported from the legacy capture script."""
    u1, u2, u3 = rng.uniform(0.0, 1.0, size=(3, n))
    q = np.empty((n, 4), dtype=np.float64)
    q[:, 0] = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    q[:, 1] = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q[:, 2] = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q[:, 3] = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    return q


def look_at_cv(eye, target, up_hint, roll_deg: float = 0.0) -> np.ndarray:
    """Camera-to-world rotation in the OpenCV convention (x right, y down, z forward).

    `up_hint` is the world direction that should appear "up" in the image. When looking straight
    down, world +Z is degenerate, so pass a horizontal direction (e.g. the bin's long axis).
    """
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up_hint))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:  # up_hint parallel to the view axis: pick any perpendicular
        alt = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = np.cross(z, alt)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    if roll_deg:
        R = R @ axis_angle_to_mat([0, 0, 1], np.deg2rad(roll_deg))
    return R
