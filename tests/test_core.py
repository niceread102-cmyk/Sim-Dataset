import numpy as np
import pytest

from binsim.camera import intrinsics_from_cfg, sample_view
from binsim.config import load_config
from binsim.loading import (BinGeom, bin_geom_from_cfg, in_bin, part_geoms_from_cfg, plan_scene,
                            sample_counts)
from binsim.noise import apply_depth_noise, projector_shadow
from binsim.poses import backproject, model_to_cam, symmetry_transforms
from binsim.transforms import (look_at_cv, make_T, mat_to_quat, mat_to_rpy, quat_to_mat, random_quaternions,
                               rpy_to_mat)
from binsim.writers.coco import rle_decode, rle_encode
from conftest import CONFIG


# ---------------------------------------------------------------- config
def test_config_defaults_and_overrides(cfg):
    assert cfg.camera.model.startswith("PhoXi")
    assert cfg.gripper.type == "suction"
    assert cfg.parts[0].name == "gear" and cfg.parts[0].mesh.endswith("example_gear.stl")
    c2 = load_config(CONFIG, ["run.num_envs=32", "loading.mode=layers", "parts.0.count=[3,4]", "parts.0.mass_kg=0.2"])
    assert c2.run.num_envs == 32 and c2.loading.mode == "layers"
    assert c2.parts[0].count == [3, 4] and c2.parts[0].mass_kg == 0.2


def test_config_validation():
    with pytest.raises(ValueError, match="loading.mode"):
        load_config(CONFIG, ["loading.mode=chaos"])
    with pytest.raises(ValueError, match="inner_size_m"):
        load_config(CONFIG, ["bin.mesh=/tmp/bin.stl"])


# ---------------------------------------------------------------- transforms
def test_quat_roundtrip():
    rng = np.random.default_rng(1)
    q = random_quaternions(50, rng)
    q *= np.where(q[:, :1] < 0, -1, 1)
    assert np.allclose(mat_to_quat(quat_to_mat(q)), q, atol=1e-9)
    for rpy in ([10, 20, 30], [-170, 45, 90], [0, 0, -120]):
        assert np.allclose(mat_to_rpy(rpy_to_mat(rpy)), rpy, atol=1e-6)


def test_look_at_top_down():
    R = look_at_cv([0, 0, 1], [0, 0, 0], [1, 0, 0])
    assert np.allclose(R[:, 2], [0, 0, -1])      # looking down
    assert np.allclose(R[:, 1], [-1, 0, 0])      # image down = -X  => image up = +X
    assert np.isclose(np.linalg.det(R), 1)


# ---------------------------------------------------------------- loading
def _geoms(cfg, assets):
    parts, _ = assets
    return part_geoms_from_cfg(cfg.parts, [a.extents for a in parts], [a.radius for a in parts])


@pytest.mark.parametrize("mode", ["random", "layers"])
def test_placement_inside_bin(cfg, assets, mode):
    geoms = _geoms(cfg, assets)
    bg = bin_geom_from_cfg(cfg.bin)
    rng = np.random.default_rng(0)
    for _ in range(10):
        counts = sample_counts(cfg.parts, rng)
        pl, m = plan_scene(counts, geoms, bg, cfg.loading, rng, mode=mode)
        assert m == mode and len(pl) == sum(counts)
        pos = np.array([p.pos for p in pl])
        ix, iy, _ = bg.inner_size
        r = geoms[0].radius
        assert np.all(np.abs(pos[:, 0]) <= ix / 2 + 1e-9) and np.all(np.abs(pos[:, 1]) <= iy / 2 + 1e-9)
        assert np.all(pos[:, 2] > bg.floor_z)
        if mode == "random":  # no initial overlap between bounding spheres
            d = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(len(pos)) * 10
            assert d.min() >= 2 * r - 1e-6


def test_layers_are_flat_grid(cfg, assets):
    geoms = _geoms(cfg, assets)
    bg = bin_geom_from_cfg(cfg.bin)
    lc = dict(cfg.loading)
    lc["layers"] = dict(lc["layers"], yaw_jitter_deg=0, pos_jitter_m=0, num_layers=[2, 2], fill_ratio=[1, 1])
    pl, _ = plan_scene([30], geoms, bg, lc, np.random.default_rng(0), mode="layers")
    z = np.array([p.pos[2] for p in pl])
    assert len(np.unique(np.round(z, 6))) <= 2
    R = quat_to_mat(np.array([p.quat for p in pl]))
    assert np.allclose(R[:, :, 2], [0, 0, 1], atol=1e-6)   # gear axis up


def test_in_bin():
    bg = BinGeom(np.array([0.5, 0.3, 0.2]), 0.01)
    p = np.array([[0, 0, 0.05], [0.4, 0, 0.05], [0, 0, -0.2]])
    assert in_bin(p, bg, 0.02).tolist() == [True, False, False]


# ---------------------------------------------------------------- camera / poses
def test_intrinsics_scaling(cfg):
    intr = intrinsics_from_cfg(cfg.camera)
    assert (intr.width, intr.height) == (1032, 772)
    assert np.isclose(intr.fx, 1250.0)


def test_model_to_cam_identity():
    T_cam = make_T(look_at_cv([0, 0, 1], [0, 0, 0], [1, 0, 0]), [0, 0, 1])
    R, t = model_to_cam(T_cam, make_T(t=[0.1, 0, 0]))
    # +X world appears "up" in the image => negative camera y; 1 m away along the optical axis
    assert np.allclose(t, [0, -100, 1000])


def test_view_jitter_stays_near_mount(cfg):
    bg = bin_geom_from_cfg(cfg.bin)
    T0 = sample_view(cfg.camera, cfg.bin, bg, np.random.default_rng(0), jitter=False)
    T1 = sample_view(cfg.camera, cfg.bin, bg, np.random.default_rng(1), jitter=True)
    assert np.linalg.norm(T0[:3, 3] - T1[:3, 3]) < 0.06
    assert T0[2, 3] > cfg.bin.size_m[2]


def test_symmetry_entries():
    s = symmetry_transforms({"axis": [0, 0, 1], "order": 4, "flip": False}, None)
    assert len(s["symmetries_discrete"]) == 3
    s = symmetry_transforms({"axis": [0, 0, 1], "order": "inf", "flip": True}, None)
    assert "symmetries_continuous" in s and len(s["symmetries_discrete"]) == 1


# ---------------------------------------------------------------- noise
def test_depth_range_and_dropout(cfg):
    K = intrinsics_from_cfg(cfg.camera).K
    d = np.full((200, 300), 0.8)
    d[:, :50] = 2.0  # beyond PhoXi M range
    out = apply_depth_noise(d, K, cfg.camera, np.random.default_rng(0))
    assert np.all(out[:, :40] == 0)
    valid = out[:, 60:] > 0
    assert valid.mean() > 0.95
    assert np.abs(out[:, 60:][valid] - 0.8).max() < 0.003


def test_projector_shadow_next_to_step():
    K = np.array([[500, 0, 160], [0, 500, 120], [0, 0, 1.0]])
    d = np.full((240, 320), 1.0)
    d[:, 150:170] = 0.8  # a raised bar 20 cm above the floor
    pts = backproject(d, K)
    sh = projector_shadow(pts, d > 0, K, [0.2, 0, 0], 0.002, downsample=1)
    # projector sits at +x: the floor just left of the bar (smaller u) is shadowed, right side isn't
    assert sh[:, 120:148].mean() > 0.5
    assert sh[:, 175:200].mean() < 0.05
    assert not sh[:, 150:170].any()


# ---------------------------------------------------------------- coco
def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(5):
        m = rng.random((37, 53)) > 0.6
        assert np.array_equal(rle_decode(rle_encode(m)), m)
    m = np.ones((4, 4), bool)
    assert rle_encode(m)["counts"][0] == 0
