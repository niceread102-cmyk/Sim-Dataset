"""Isaac Lab scene: N cloned envs, each = floor + bin + robot + part pool + tiled camera + key light.

Env layout (env frame == world frame shifted by the env origin, no rotation):
    bin frame at bin.pos_m / bin.rpy_deg, robot at robot.pos_m, camera placed per view.
Part pool: for each part type, `max count` slots (prims part_000, part_001, ...). Every env has
the same prims (required for cloning); unused / escaped slots are parked on a hidden shelf below
the floor, out of every camera's view.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass

from ..config import count_range
from ..transforms import mat_to_quat, rpy_to_mat
from .compat import resolve_nucleus, to_sim_quat

FLOOR_SIZE = 3.6
PARK_Z = -3.0          # parking shelf height (below the visual floor)


@dataclass
class PoolLayout:
    slot_type: list[int]           # part type per slot
    type_slots: list[list[int]]    # slots per part type
    park_pos: np.ndarray           # (S,3) env-frame parking position per slot


def pool_layout(parts_cfg, radii) -> PoolLayout:
    slot_type, type_slots = [], []
    for ti, p in enumerate(parts_cfg):
        _, hi = count_range(p)
        type_slots.append(list(range(len(slot_type), len(slot_type) + hi)))
        slot_type += [ti] * hi
    S = len(slot_type)
    pitch = 2 * max(radii) + 0.03
    ncol = max(int(np.ceil(np.sqrt(S))), 1)
    park = np.array([[(i % ncol - (ncol - 1) / 2) * pitch, (i // ncol - (ncol - 1) / 2) * pitch,
                      PARK_Z + 0.06 + max(radii)] for i in range(S)])
    if ncol * pitch > FLOOR_SIZE:
        raise ValueError(f"part pool of {S} slots doesn't fit the parking shelf; reduce part counts")
    return PoolLayout(slot_type, type_slots, park)


def build_scene_cfg(cfg, part_assets, bin_asset, intr, layout: PoolLayout):
    @configclass
    class BinSceneCfg(InteractiveSceneCfg):
        pass

    run = cfg.run
    sc = BinSceneCfg(num_envs=int(run.num_envs), env_spacing=float(run.get("env_spacing_m", 4.0)),
                     replicate_physics=True)

    # visual/collision floor (a finite box, not an infinite ground plane: PhysX planes are
    # half-spaces and would push the parking shelf's parts up through the floor)
    sc.floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/floor",
        spawn=sim_utils.CuboidCfg(size=(FLOOR_SIZE, FLOOR_SIZE, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.36))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)))
    sc.parking = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/parking",
        spawn=sim_utils.CuboidCfg(size=(FLOOR_SIZE, FLOOR_SIZE, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, PARK_Z)))

    lights = cfg.get("randomization", {}).get("lights", {})
    sc.dome = AssetBaseCfg(prim_path="/World/dome_light",
                           spawn=sim_utils.DomeLightCfg(intensity=float(np.mean(lights.get("dome_intensity", [800, 800]))),
                                                        color=(1.0, 1.0, 1.0)))
    sc.key_light = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/key_light",
                                spawn=sim_utils.SphereLightCfg(intensity=float(np.mean(lights.get("key_intensity", [5000, 5000]))),
                                                               radius=0.15, color=(1.0, 1.0, 1.0)),
                                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.3, 1.6)))

    b = cfg.bin
    sc.bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bin",
        spawn=sim_utils.UsdFileCfg(usd_path=bin_asset.usd_path),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(b.get("pos_m", [0, 0, 0])),
                                                  rot=tuple(to_sim_quat(mat_to_quat(rpy_to_mat(b.get("rpy_deg", [0, 0, 0])))))))

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=8, solver_velocity_iteration_count=1,
        max_depenetration_velocity=1.0, max_linear_velocity=5.0, max_angular_velocity=100.0,
        sleep_threshold=0.0005, stabilization_threshold=0.0005)
    for s, ti in enumerate(layout.slot_type):
        setattr(sc, f"part_{s:03d}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/part_{s:03d}",
            spawn=sim_utils.UsdFileCfg(usd_path=part_assets[ti].usd_path, rigid_props=rigid_props),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(layout.park_pos[s]))))

    rob = cfg.get("robot", {}) or {}
    if rob.get("enabled", False) and rob.get("usd"):
        sc.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/robot",
            spawn=sim_utils.UsdFileCfg(usd_path=resolve_nucleus(rob["usd"]),
                                       rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(rob.get("pos_m", [-0.75, 0, 0])),
                rot=tuple(to_sim_quat(mat_to_quat(rpy_to_mat(rob.get("rpy_deg", [0, 0, 0]))))),
                joint_pos=dict(rob.get("joint_pos", {".*": 0.0}))),
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=float(rob.get("stiffness", 1e4)),
                                                  damping=float(rob.get("damping", 1e3)))})

    near, far = 0.05, max(5.0, float(cfg.camera.get("depth_range_m", [0, 3])[1]) * 1.5)
    sc.cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/phoxi",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0), rot=tuple(to_sim_quat([1.0, 0, 0, 0])), convention="ros"),
        data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intr.K.flatten().tolist(), width=intr.width, height=intr.height, clipping_range=(near, far)),
        width=intr.width, height=intr.height, colorize_instance_id_segmentation=False,
        update_latest_camera_pose=True)  # GT uses the pose read back after each move; default only reads it at init

    if cfg.run.get("save_overview_every", 0) is not None:
        sc.overview = CameraCfg(
            prim_path="/World/overview_cam", update_period=0, width=1280, height=720, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)))
    return sc


def attach_gripper_visual(stage, cfg, num_envs: int):
    """Reference an optional visual gripper USD under the robot's flange link in every env."""
    g = cfg.get("gripper", {}) or {}
    if not g.get("usd") or not (cfg.get("robot", {}) or {}).get("enabled"):
        return
    from pxr import Gf, UsdGeom
    usd = resolve_nucleus(g["usd"])
    q = mat_to_quat(rpy_to_mat(g.get("attach_rpy_deg", [0, 0, 0])))
    for i in range(num_envs):
        link = None
        for prim in stage.GetPrimAtPath(f"/World/envs/env_{i}/robot").GetAllChildren() if stage.GetPrimAtPath(f"/World/envs/env_{i}/robot") else []:
            if prim.GetName() == g.get("attach_link", "ee_link"):
                link = prim
        if link is None:
            print(f"[WARN] gripper attach_link '{g.get('attach_link')}' not found in env_{i} robot — gripper not attached", flush=True)
            return
        path = f"{link.GetPath()}/gripper_visual"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(usd)
        x = UsdGeom.Xformable(prim)
        x.ClearXformOpOrder()
        x.AddTranslateOp().Set(Gf.Vec3d(*g.get("attach_pos_m", [0, 0, 0])))
        x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
