"""Isolation layer for Isaac Lab API differences across versions (2.x / 3.x).

Everything version-sensitive the generator touches lives here: quaternion order, rigid-object
read/write method names, tensor types (torch vs warp), segmentation info layout.
If a first run fails inside Isaac, this is the file to adjust.
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch

_QUAT_ORDER = "wxyz"


def detect_quat_order(setting: str = "auto") -> str:
    """Ask Isaac Lab's own math utils which order it uses (robust to version changes)."""
    global _QUAT_ORDER
    if setting in ("wxyz", "xyzw"):
        _QUAT_ORDER = setting
        return _QUAT_ORDER
    try:
        from isaaclab.utils.math import quat_from_euler_xyz
        q = quat_from_euler_xyz(torch.tensor([math.pi / 2]), torch.zeros(1), torch.zeros(1))
        q = to_numpy(q)[0]
        s = math.sqrt(0.5)
        if abs(q[0] - s) < 1e-4 and abs(q[1] - s) < 1e-4:
            _QUAT_ORDER = "wxyz"
        elif abs(q[3] - s) < 1e-4 and abs(q[0] - s) < 1e-4:
            _QUAT_ORDER = "xyzw"
        else:
            print(f"[WARN] could not infer quaternion order from {q}; assuming wxyz", flush=True)
    except Exception as e:  # pragma: no cover - depends on Isaac Lab version
        print(f"[WARN] quaternion order detection failed ({e}); assuming wxyz", flush=True)
    return _QUAT_ORDER


def disable_isaacsim_physics_warm_start():
    """Isaac Lab 3.x loads physics and creates its tensor views itself in SimulationContext.reset(); Isaac Sim's
    SimulationManager reloading physics on timeline PLAY deletes those views (invalid views / GPU hang).
    isaaclab_physx rebinds `isaacsim.core.simulation_manager.SimulationManager` to its PhysxManager, but the
    original class has already subscribed to PLAY, so import it from its impl module."""
    try:
        from isaacsim.core.simulation_manager.impl.simulation_manager import SimulationManager
    except ImportError:
        return
    SimulationManager.enable_default_callbacks(enable_warm_start=False)


def to_sim_quat(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    return q if _QUAT_ORDER == "wxyz" else q[..., [1, 2, 3, 0]]


def from_sim_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q if _QUAT_ORDER == "wxyz" else q[..., [3, 0, 1, 2]]


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    try:  # warp array
        import warp as wp
        if isinstance(x, wp.array):
            return wp.to_torch(x).detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def to_torch(x, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    try:
        import warp as wp
        if isinstance(x, wp.array):
            return wp.to_torch(x).to(device)
    except ImportError:
        pass
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def _first_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


def write_pose(asset, pos: np.ndarray, quat_wxyz: np.ndarray, env_ids, device):
    """pos (n,3) world, quat (n,4) wxyz."""
    pose = np.concatenate([pos, to_sim_quat(quat_wxyz)], axis=-1)
    fn = _first_attr(asset, ["write_root_pose_to_sim", "write_root_link_pose_to_sim"])
    fn(torch.as_tensor(pose, dtype=torch.float32, device=device), env_ids=env_ids)


def write_velocity(asset, lin: np.ndarray, ang: np.ndarray, env_ids, device):
    vel = np.concatenate([lin, ang], axis=-1)
    fn = _first_attr(asset, ["write_root_velocity_to_sim", "write_root_com_velocity_to_sim"])
    fn(torch.as_tensor(vel, dtype=torch.float32, device=device), env_ids=env_ids)


def read_pose(asset):
    """-> (pos (N,3), quat_wxyz (N,4)) numpy, world frame."""
    d = asset.data
    pos = to_numpy(_first_attr(d, ["root_pos_w", "root_link_pos_w"]))
    quat = to_numpy(_first_attr(d, ["root_quat_w", "root_link_quat_w"]))
    return pos.astype(np.float64), from_sim_quat(quat)


def read_velocity_torch(asset, device):
    d = asset.data
    lin = to_torch(_first_attr(d, ["root_lin_vel_w", "root_com_lin_vel_w"]), device)
    ang = to_torch(_first_attr(d, ["root_ang_vel_w", "root_com_ang_vel_w"]), device)
    return lin, ang


def camera_pose_ros(cam):
    """Camera world poses in the ROS/OpenCV optical convention -> (pos (N,3), quat_wxyz (N,4))."""
    d = cam.data
    pos = to_numpy(d.pos_w)
    quat = to_numpy(_first_attr(d, ["quat_w_ros", "quat_w_opengl", "quat_w_world", "quat_w_opencv"]))
    return pos.astype(np.float64), from_sim_quat(quat)


def use_usd_camera_view(cam):
    """Make the camera's XformPrimView read/write poses through USD instead of Fabric. On the GB10 / Isaac Sim 6.0
    build, Fabric world-matrix writes don't take effect: the camera stays at its spawn pose in render and readback."""
    view = getattr(cam, "_view", None)
    if view is not None and getattr(view, "_use_fabric", False):
        view._use_fabric = False


def set_camera_poses(cam, pos: np.ndarray, quat_wxyz: np.ndarray, device):
    cam.set_world_poses(torch.as_tensor(pos, dtype=torch.float32, device=device),
                        torch.as_tensor(to_sim_quat(quat_wxyz), dtype=torch.float32, device=device),
                        convention="ros")


def _find_id_maps(info, out: list):
    if isinstance(info, dict):
        for key in ("idToLabels", "idToSemantics"):
            if key in info and isinstance(info[key], dict):
                out.append(info[key])
        for v in info.values():
            _find_id_maps(v, out)
    elif isinstance(info, (list, tuple)):
        for v in info:
            _find_id_maps(v, out)
    return out


_SLOT_RE = re.compile(r"/part_(\d+)(?:/|$)")


def instance_id_to_slot(info) -> dict[int, int]:
    """{instance id -> pool slot} from the camera's instance-segmentation info (prim paths).
    Key names moved between Isaac versions (the legacy script hit this too), so search broadly."""
    maps = _find_id_maps(info, [])
    lut: dict[int, int] = {}
    for m in maps:
        for k, v in m.items():
            label = v if isinstance(v, str) else (v.get("class") or v.get("semantic") or str(v)) if isinstance(v, dict) else str(v)
            match = _SLOT_RE.search(label)
            if match:
                try:
                    lut[int(k)] = int(match.group(1))
                except (TypeError, ValueError):
                    pass
    return lut


def map_instances(seg: np.ndarray, lut: dict[int, int]) -> np.ndarray:
    seg = seg.reshape(seg.shape[0], seg.shape[1])
    u, inv = np.unique(seg, return_inverse=True)
    mapped = np.array([lut.get(int(x), -1) for x in u], dtype=np.int32)
    return mapped[inv].reshape(seg.shape)


def resolve_nucleus(path: str) -> str:
    if not path or "{" not in path:
        return path
    fmt = {}
    try:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
        fmt.update(ISAAC_NUCLEUS_DIR=ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR=ISAACLAB_NUCLEUS_DIR)
    except ImportError:
        pass
    return path.format(**fmt)
