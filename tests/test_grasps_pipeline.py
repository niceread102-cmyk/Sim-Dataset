"""Grasp logic + end-to-end post-processing on a CPU-rendered scene (no Isaac)."""
import json
import os
import subprocess
import sys

import numpy as np

from binsim import grasps as G
from binsim.bop_io import BopScene, list_scenes
from binsim.camera import intrinsics_from_cfg, sample_view
from binsim.config import load_config
from binsim.loading import bin_geom_from_cfg, part_geoms_from_cfg, plan_scene
from binsim.postprocess import SceneCapture, SceneContext, ViewCapture, process_scene
from binsim.camera import rim_height
from binsim.transforms import inv_T, make_T, quat_to_mat, rpy_to_mat
from binsim.writers.bop import write_camera_json, write_models
from conftest import CONFIG, REPO
from fake_render import render


def test_suction_sampling_on_gear(cfg, assets):
    parts, _ = assets
    lib = G.build_library(cfg.parts[0], parts[0].mesh, cfg.gripper, np.random.default_rng(0))
    assert len(lib.T) >= 10
    approach = lib.T[:, :3, 2]
    # flat patches of a spur gear are its two faces -> approach along ∓Z
    assert np.all(np.abs(np.abs(approach[:, 2]) - 1) < 0.05)
    # contact points lie on the faces, inside the annulus between bore and root
    r = np.linalg.norm(lib.T[:, :2, 3], axis=1)
    assert np.all((r > 0.006) & (r < 0.022))


def test_grasp_collision_detects_part_above(cfg, assets):
    parts, _ = assets
    lib = G.build_library(cfg.parts[0], parts[0].mesh, cfg.gripper, np.random.default_rng(0))
    pts = [G.sample_part_points(parts[0].mesh, 1500)]
    bg = bin_geom_from_cfg(cfg.bin)
    rim = rim_height(cfg.bin, bg)
    lying = make_T(t=[0, 0, bg.floor_z + 0.004])
    alone = G.evaluate_scene([lying], [0], [lib], pts, bg, rim, cfg.gripper, cfg.grasps)
    assert any(g.valid for g in alone)
    # identical gear hovering 2 cm above: every top-face grasp must now collide with it
    above = make_T(t=[0, 0, bg.floor_z + 0.024])
    both = G.evaluate_scene([lying, above], [0, 0], [lib], pts * 1, bg, rim, cfg.gripper, cfg.grasps)
    lower_valid = [g for g in both if g.part == 0 and g.valid]
    assert not lower_valid
    assert any(g.reason.startswith("collision_part:1") for g in both if g.part == 0)


def test_grasp_near_wall_collides_with_bin(cfg, assets):
    parts, _ = assets
    lib = G.build_library(cfg.parts[0], parts[0].mesh, cfg.gripper, np.random.default_rng(0))
    pts = [G.sample_part_points(parts[0].mesh, 1500)]
    bg = bin_geom_from_cfg(cfg.bin)
    # stood on edge against the +X wall: side-facing grasps must be rejected by angle or bin
    T = make_T(rpy_to_mat([0, 90, 0]), [bg.inner_size[0] / 2 - 0.005, 0, bg.floor_z + 0.027])
    res = G.evaluate_scene([T], [0], [lib], pts, bg, rim_height(cfg.bin, bg), cfg.gripper,
                           dict(cfg.grasps, max_approach_angle_deg=90))
    assert any(g.reason == "collision_bin" for g in res)


def test_end_to_end_scene(tmp_path, assets):
    cfg = load_config(CONFIG, [f"assets.cache_dir={tmp_path}/cache", "camera.render_scale=0.15",
                               "parts.0.count=[6,6]", "run.views_per_scene=2"])
    part_assets, bin_asset = assets
    intr = intrinsics_from_cfg(cfg.camera)
    bg = bin_geom_from_cfg(cfg.bin)
    geoms = part_geoms_from_cfg(cfg.parts, [a.extents for a in part_assets], [a.radius for a in part_assets])
    rng = np.random.default_rng(3)
    pl, mode = plan_scene([6], geoms, bg, cfg.loading, rng, mode="layers")
    T_parts = [make_T(quat_to_mat(p.quat), p.pos) for p in pl]
    libs = [G.build_library(p, a.mesh, cfg.gripper, np.random.default_rng(0)) for p, a in zip(cfg.parts, part_assets)]
    pts = [G.sample_part_points(a.mesh, 1500) for a in part_assets]
    out = str(tmp_path / "shard_00")
    write_models(out, cfg.parts, part_assets)
    write_camera_json(out, intr, 0.1)
    ctx = SceneContext(cfg, intr, list(cfg.parts), part_assets, pts, libs, bg, rim_height(cfg.bin, bg), out)

    views = []
    for v in range(2):
        T_bin_cam = sample_view(cfg.camera, cfg.bin, bg, rng, jitter=v > 0)
        T_cam_bin = inv_T(T_bin_cam)
        meshes = []
        for T in T_parts:
            Tc = T_cam_bin @ T
            meshes.append((part_assets[0].mesh.vertices @ Tc[:3, :3].T + Tc[:3, 3], part_assets[0].mesh.faces))
        n_cam = T_cam_bin[:3, :3] @ np.array([0, 0, 1.0])
        p_cam = T_cam_bin[:3, :3] @ np.array([0, 0, bg.floor_z]) + T_cam_bin[:3, 3]
        depth, inst, rgb = render(meshes, intr.K, intr.width, intr.height, floor=(n_cam, -n_cam @ p_cam))
        slots = [10 + k for k in range(len(T_parts))]        # arbitrary pool slots
        inst_slot = np.where(inst >= 0, np.take(slots, np.clip(inst, 0, None)), -1)
        views.append(ViewCapture(rgb, depth.astype(np.float32), inst_slot, T_bin_cam))

    sc = SceneCapture(scene_id=7, mode=mode, slots=[10 + k for k in range(len(T_parts))], types=[0] * len(T_parts),
                      T_bin_part=T_parts, views=views, save_viz=True, seed=1)
    summary = process_scene(ctx, sc)

    # GT poses agree with the rendered masks
    for v in summary["views"]:
        assert v["mask_consistency"] is not None and v["mask_consistency"] > 0.97, v
    assert summary["views"][0]["topmost"] is not None

    # BOP files are complete and readable
    assert list_scenes(out) == [7]
    s = BopScene(out, 7)
    assert s.im_ids == [0, 1] and len(s.gt["0"]) == 6
    assert all(0 <= i["visib_fract"] <= 1 for i in s.gt_info["0"])
    coco = json.load(open(os.path.join(s.dir, "scene_gt_coco.json")))
    assert len(coco["images"]) == 2 and coco["annotations"]
    g = s.grasps(0)
    assert g["topmost_gt_idx"] is not None and any(x["valid"] for x in g["grasps"])
    assert os.path.exists(os.path.join(s.dir, "poses", "000000.csv"))
    assert os.path.exists(os.path.join(out, "viz", "000007_00_bin.jpg"))
    # the sensor model left holes (range/shadow/edges) but kept most of the bin
    d = s.depth_m(0)
    assert 0.3 < (d > 0).mean() < 1.0

    # offline tools run on the output
    for script in ("verify_poses.py", "visualize.py"):
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", script), out], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
