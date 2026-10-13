"""Main generation loop (runs inside Isaac Sim, after AppLauncher has started the app).

Per batch (all envs in lock-step):
  reset/placement -> randomize visuals -> drop + shake + settle (physics only, no rendering)
  -> remove escaped parts -> V camera views (render only) -> hand off to writer threads.
"""
from __future__ import annotations

import json
import os
import platform
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from .. import grasps as G
from ..assets import git_hash, prepare_all
from ..camera import intrinsics_from_cfg, rim_height, sample_view
from ..config import dump_config, to_plain
from ..loading import (bin_geom_from_cfg, bin_oscillation_offset, in_bin, part_geoms_from_cfg, plan_scene,
                       sample_counts, shake_schedule)
from ..postprocess import SceneCapture, SceneContext, ViewCapture, process_scene
from ..transforms import inv_T, make_T, mat_to_quat, quat_to_mat, rpy_to_mat
from ..writers.async_io import AsyncWriter
from ..writers.bop import write_camera_json, write_models
from . import compat as C
from .randomize import Randomizer
from .scene import attach_gripper_visual, build_scene_cfg, pool_layout

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _T_batch(pos, quat_wxyz):
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = quat_to_mat(quat_wxyz)
    T[:, :3, 3] = pos
    return T


class Generator:
    def __init__(self, cfg, simulation_app):
        self.cfg, self.app = cfg, simulation_app
        run = cfg.run
        self.N = int(run.num_envs)
        self.V = int(run.views_per_scene)
        self.shard = int(run.get("shard_id", 0))
        self.out = os.path.join(os.path.expanduser(run.output_dir), f"shard_{self.shard:02d}")
        os.makedirs(os.path.join(self.out, "viz"), exist_ok=True)
        self.rng = np.random.default_rng([int(cfg.sim.get("seed", 0)), self.shard])

        # ---- assets & geometry (pure python; USD authored with pxr) ----
        t0 = time.monotonic()
        self.part_assets, self.bin_asset = prepare_all(cfg)
        print(f"[INFO] assets ready in {time.monotonic() - t0:.1f}s: "
              f"{[a.usd_path for a in self.part_assets]} bin={self.bin_asset.usd_path}", flush=True)
        self.geoms = part_geoms_from_cfg(cfg.parts, [a.extents for a in self.part_assets], [a.radius for a in self.part_assets],
                                         [a.mesh for a in self.part_assets])  # mesh: loading.mode stack
        self.bin_geom = bin_geom_from_cfg(cfg.bin)
        self.rim_z = rim_height(cfg.bin, self.bin_geom)
        self.layout = pool_layout(cfg.parts, [a.radius for a in self.part_assets])
        self.intr = intrinsics_from_cfg(cfg.camera)

        # ---- simulation ----
        self.device = cfg.sim.get("device", "cuda:0")
        C.detect_quat_order(cfg.sim.get("quat_order", "auto"))
        C.disable_isaacsim_physics_warm_start()
        dt = float(cfg.sim.get("physics_dt", 0.005))
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=self.device))
        self.dt = dt
        scene_cfg = build_scene_cfg(cfg, self.part_assets, self.bin_asset, self.intr, self.layout)
        self.scene = InteractiveScene(scene_cfg)
        import omni.usd
        self.stage = omni.usd.get_context().get_stage()
        attach_gripper_visual(self.stage, cfg, self.N)
        self.randomizer = Randomizer(self.stage, cfg, self.N, self.layout.slot_type)
        self.sim.reset()
        self.slots = [self.scene[f"part_{s:03d}"] for s in range(len(self.layout.slot_type))]
        self.bin = self.scene["bin"]
        self.cam = self.scene["cam"]
        C.use_usd_camera_view(self.cam)
        self.overview = self._entity("overview")
        self.robot = self._entity("robot")
        self.env_origins = C.to_numpy(self.scene.env_origins).astype(np.float64)
        self.T_env_bin = make_T(rpy_to_mat(cfg.bin.get("rpy_deg", [0, 0, 0])), cfg.bin.get("pos_m", [0, 0, 0]))
        self.T_w_bin = np.stack([make_T(t=o) @ self.T_env_bin for o in self.env_origins])
        render_settle = cfg.sim.get("render_during_settle")
        self.render_settle = (not cfg.sim.get("headless", True)) if render_settle is None else bool(render_settle)
        self._hold_robot()
        self._check_intrinsics()

        # ---- grasp libraries & collision points ----
        grng = np.random.default_rng(0)
        self.libraries = [G.build_library(p, a.mesh, cfg.gripper, grng) for p, a in zip(cfg.parts, self.part_assets)]
        npts = int(cfg.get("grasps", {}).get("point_samples", 1500))
        self.points = [G.sample_part_points(a.mesh, npts, seed=i) for i, a in enumerate(self.part_assets)]
        for p, lib in zip(cfg.parts, self.libraries):
            print(f"[INFO] grasp library '{p['name']}': {len(lib.T)} {lib.kind} grasps", flush=True)

        self.ctx = SceneContext(cfg, self.intr, list(cfg.parts), self.part_assets, self.points, self.libraries,
                                self.bin_geom, self.rim_z, self.out)
        self._write_run_files()
        self.writer = AsyncWriter(int(run.get("writer_threads", 8)))
        self.first_view = True

    # ------------------------------------------------------------------ setup helpers
    def _entity(self, name):
        try:
            return self.scene[name]
        except KeyError:
            return None

    def _hold_robot(self):
        if self.robot is None:
            return
        jp, jv = self.robot.data.default_joint_pos, self.robot.data.default_joint_vel
        self.robot.write_joint_state_to_sim(jp, jv)
        self.robot.set_joint_position_target(jp)

    def _check_intrinsics(self):
        try:
            K_act = C.to_numpy(self.cam.data.intrinsic_matrices)[0]
        except Exception:
            return
        d = np.abs(K_act - self.intr.K).max()
        if d > 0.5:
            print(f"[WARN] rendered camera K differs from config by {d:.2f}px — using the simulator's K for GT.\n"
                  f"  config K={self.intr.K.tolist()}\n  sim K={K_act.tolist()}", flush=True)
            self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy = K_act[0, 0], K_act[1, 1], K_act[0, 2], K_act[1, 2]

    def _write_run_files(self):
        dump_config(self.cfg, os.path.join(self.out, "config_resolved.yaml"))
        write_models(self.out, self.cfg.parts, self.part_assets)
        write_camera_json(self.out, self.intr, float(self.cfg.outputs.get("depth_scale", 0.1)))
        try:
            import isaaclab
            il_ver = getattr(isaaclab, "__version__", "unknown")
        except Exception:
            il_ver = "unknown"
        info = {"git": git_hash(REPO), "isaaclab": il_ver, "quat_order": C._QUAT_ORDER, "host": platform.node(),
                "shard": self.shard, "intrinsics": self.intr.to_json(), "slot_type": self.layout.slot_type,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(self.out, "run_info.json"), "w") as f:
            json.dump(info, f, indent=2)

    # ------------------------------------------------------------------ physics helpers
    def _step(self, render=False, n=1):
        for _ in range(n):
            self.scene.write_data_to_sim()
            self.sim.step(render=render)
            self.scene.update(self.dt)

    def _write_slots(self, pos, quat, env_ids=None, slots=None):
        """pos (S,N,3) world, quat (S,N,4) wxyz; writes zero velocity too."""
        slots = range(len(self.slots)) if slots is None else slots
        for s in slots:
            p, q = pos[s], quat[s]
            if env_ids is not None:
                p, q = p[env_ids], q[env_ids]
            ids = None if env_ids is None else torch.as_tensor(env_ids, device=self.device)
            C.write_pose(self.slots[s], p, q, ids, self.device)
            z = np.zeros((len(p), 3))
            C.write_velocity(self.slots[s], z, z, ids, self.device)

    def _read_slots(self):
        pos, quat = zip(*(C.read_pose(o) for o in self.slots))
        return np.stack(pos), np.stack(quat)   # (S,N,3), (S,N,4)

    def _max_speed(self, active_t):
        lins, angs = zip(*(C.read_velocity_torch(o, self.device) for o in self.slots))
        lin = torch.stack(lins).norm(dim=-1)   # (S,N)
        ang = torch.stack(angs).norm(dim=-1)
        lin = torch.where(active_t, lin, torch.zeros_like(lin))
        ang = torch.where(active_t, ang, torch.zeros_like(ang))
        return float(lin.max()), float(ang.max())

    # ------------------------------------------------------------------ one batch
    def _reset_batch(self):
        S = len(self.slots)
        pos = np.zeros((S, self.N, 3))
        quat = np.tile([1.0, 0, 0, 0], (S, self.N, 1))
        active = np.zeros((S, self.N), bool)
        modes = []
        for i in range(self.N):
            pos[:, i] = self.env_origins[i] + self.layout.park_pos
            counts = sample_counts(self.cfg.parts, self.rng, self.cfg.loading)
            placements, mode = plan_scene(counts, self.geoms, self.bin_geom, self.cfg.loading, self.rng)
            modes.append(mode)
            next_slot = [list(s) for s in self.layout.type_slots]
            for p in placements:
                s = next_slot[p.type_idx].pop(0)
                T = self.T_w_bin[i] @ make_T(quat_to_mat(p.quat), p.pos)
                pos[s, i], quat[s, i] = T[:3, 3], mat_to_quat(T[:3, :3])
                active[s, i] = True
        self._write_slots(pos, quat)
        bq = np.tile(mat_to_quat(self.T_w_bin[0][:3, :3]), (self.N, 1))
        C.write_pose(self.bin, self.T_w_bin[:, :3, 3], bq, None, self.device)
        return active, modes

    def _settle(self, active):
        lcfg = self.cfg.loading
        st = lcfg.get("settle", {})
        min_steps, max_steps = int(st.get("min_steps", 60)), int(st.get("max_steps", 1200))
        lin_thr, ang_thr = float(st.get("lin_vel_thresh", 0.005)), float(st.get("ang_vel_thresh", 0.05))
        hold_need = int(st.get("hold_steps", 10))
        kicks = set(shake_schedule(lcfg, self.rng))
        sh = lcfg.get("shake", {})
        phase = self.rng.uniform(0, 2 * np.pi, self.N)
        active_t = torch.as_tensor(active, device=self.device)
        bq = np.tile(mat_to_quat(self.T_w_bin[0][:3, :3]), (self.N, 1))
        osc = float((sh.get("bin_oscillation") or {}).get("amp_m", 0.0)) > 0
        hold, step = 0, 0
        for step in range(max_steps):
            if step in kicks:
                for s, obj in enumerate(self.slots):
                    m = active[s][:, None]
                    lin = self.rng.uniform(-1, 1, (self.N, 3)) * float(sh.get("lin_vel_mps", 0.15)) * m
                    lin[:, 2] = np.abs(lin[:, 2])   # kick up/sideways, never into the floor
                    ang = self.rng.uniform(-1, 1, (self.N, 3)) * float(sh.get("ang_vel_rps", 2.0)) * m
                    C.write_velocity(obj, lin, ang, None, self.device)
            if osc:
                off = bin_oscillation_offset(step * self.dt, lcfg, phase)
                C.write_pose(self.bin, self.T_w_bin[:, :3, 3] + off, bq, None, self.device)
            self._step(render=self.render_settle)
            if step >= min_steps and step % 5 == 0 and step > max(kicks, default=0):
                lin_m, ang_m = self._max_speed(active_t)
                hold = hold + 5 if (lin_m < lin_thr and ang_m < ang_thr) else 0
                if hold >= hold_need:
                    break
        if osc:
            C.write_pose(self.bin, self.T_w_bin[:, :3, 3], bq, None, self.device)
        return step + 1

    def _remove_escaped(self, active):
        pos, quat = self._read_slots()
        removed = np.zeros(self.N, int)
        margin = float(self.cfg.loading.get("settle", {}).get("escape_margin_m", 0.02))
        park_pos = self.env_origins[None] + self.layout.park_pos[:, None]
        park_q = np.tile([1.0, 0, 0, 0], pos.shape[:2] + (1,))
        for i in range(self.N):
            Tinv = inv_T(self.T_w_bin[i])
            pb = pos[:, i] @ Tinv[:3, :3].T + Tinv[:3, 3]
            esc = active[:, i] & ~in_bin(pb, self.bin_geom, margin)
            for s in np.flatnonzero(esc):
                self._write_slots(park_pos, park_q, env_ids=[i], slots=[s])
                active[s, i] = False
                removed[i] += 1
        if removed.any():
            self._step(render=False, n=2)
        return removed

    def _capture_views(self):
        views = [[] for _ in range(self.N)]
        for v in range(self.V):
            T_bin_cam = [sample_view(self.cfg.camera, self.cfg.bin, self.bin_geom, self.rng, jitter=(v > 0 or self.V == 1))
                         for _ in range(self.N)]
            T_w_cam = np.stack([self.T_w_bin[i] @ T_bin_cam[i] for i in range(self.N)])
            C.set_camera_poses(self.cam, T_w_cam[:, :3, 3], mat_to_quat(T_w_cam[:, :3, :3]), self.device)
            for _ in range(max(int(self.cfg.camera.get("render_subframes", 2)), 1)):
                self.sim.render()
            self.cam.update(dt=0.0, force_recompute=True)
            out = self.cam.data.output
            rgb = C.to_numpy(out["rgb"])
            depth = C.to_numpy(out["distance_to_image_plane"])
            seg = C.to_numpy(out["instance_id_segmentation_fast"])
            lut = C.instance_id_to_slot(self.cam.data.info)
            pos_r, quat_r = C.camera_pose_ros(self.cam)
            T_w_cam_read = _T_batch(pos_r, quat_r)
            if self.first_view:
                self._debug_first_view(T_w_cam, T_w_cam_read, lut, seg)
            for i in range(self.N):
                views[i].append(ViewCapture(rgb[i], depth[i].reshape(depth.shape[1], depth.shape[2]),
                                            C.map_instances(seg[i], lut), inv_T(self.T_w_bin[i]) @ T_w_cam_read[i]))
        return views

    def _debug_first_view(self, T_req, T_read, lut, seg):
        self.first_view = False
        dp = np.abs(T_req[:, :3, 3] - T_read[:, :3, 3]).max()
        dR = np.abs(T_req[:, :3, :3] - T_read[:, :3, :3]).max()
        print(f"[DEBUG] camera pose readback: max |dpos|={dp * 1000:.2f} mm, max |dR|={dR:.4f}", flush=True)
        if dp > 0.002 or dR > 0.01:
            print("[WARN] camera pose readback differs from the requested pose. Check sim.quat_order "
                  f"(detected {C._QUAT_ORDER}) — GT uses the READBACK pose, so data stays consistent, "
                  "but the view may not be where you expect.", flush=True)
        print(f"[DEBUG] instance-segmentation ids mapped to part slots: {len(lut)}; unique ids in env0: "
              f"{np.unique(seg[0])[:20].tolist()}", flush=True)
        if not lut:
            print(f"[WARN] no instance id -> part prim mapping found. raw info: {str(self.cam.data.info)[:2000]}", flush=True)

    def _save_overview(self, tag: str):
        if self.overview is None:
            return
        o = self.env_origins[0]
        c = self.T_w_bin[0][:3, 3] + np.array([0, 0, self.rim_z / 2])
        eye = o + np.array([1.6, -1.4, 1.5]) + self.T_env_bin[:3, 3]
        self.overview.set_world_poses_from_view(torch.tensor(eye[None], dtype=torch.float32, device=self.device),
                                                torch.tensor(c[None], dtype=torch.float32, device=self.device))
        for _ in range(3):
            self.sim.render()
        self.overview.update(dt=0.0, force_recompute=True)
        from PIL import Image
        img = C.to_numpy(self.overview.data.output["rgb"])[0][..., :3]
        Image.fromarray(img).save(os.path.join(self.out, "viz", f"overview_{tag}.jpg"), quality=92)

    def _on_summary(self, fut):
        s = fut.result()
        if not s:
            return
        for v in s["views"]:
            if v["mask_consistency"] is not None and v["mask_consistency"] < 0.9:
                print(f"[WARN] scene {s['scene_id']} view {v['im_id']}: GT pose/mask consistency "
                      f"{v['mask_consistency']:.2f} < 0.9 — poses may not match the images (convention bug?)", flush=True)
        if s["scene_id"] == self.shard * 100000:
            v0 = s["views"][0]
            print(f"[INFO] first scene written: {s['num_parts']} parts, mode={s['mode']}, topmost=#{v0['topmost']}, "
                  f"mask consistency={v0['mask_consistency']}, valid depth={v0['valid_depth_frac']:.2f}", flush=True)
            print("\n".join(v0["table"] or []), flush=True)

    # ------------------------------------------------------------------ main loop
    def run(self):
        run = self.cfg.run
        total = int(run.num_scenes)
        n_batches = int(np.ceil(total / self.N))
        viz_every = int(run.get("save_viz_every", 1) or 0)
        ov_every = run.get("save_overview_every", 0)
        t_start = t_log = time.monotonic()
        scenes = imgs = 0
        log_every = float(run.get("log_every_s", 5.0))
        print(f"[INFO] generating {n_batches * self.N} scenes x {self.V} views, {self.N} envs/batch -> {self.out}", flush=True)
        for b in range(n_batches):
            if not self.app.is_running():
                break
            active, modes = self._reset_batch()
            self.randomizer.apply(self.rng)
            steps = self._settle(active)
            removed = self._remove_escaped(active)
            pos, quat = self._read_slots()
            if ov_every is not None and (b == 0 or (ov_every and b % int(ov_every) == 0)):
                self._save_overview(f"{b:05d}")
            views = self._capture_views()
            for i in range(self.N):
                idx = b * self.N + i
                if idx >= total:
                    break
                slots = np.flatnonzero(active[:, i]).tolist()
                Tinv = inv_T(self.T_w_bin[i])
                T_bin_part = [Tinv @ make_T(quat_to_mat(quat[s, i]), pos[s, i]) for s in slots]
                sc = SceneCapture(scene_id=self.shard * 100000 + idx, mode=modes[i], slots=slots,
                                  types=[self.layout.slot_type[s] for s in slots], T_bin_part=T_bin_part,
                                  views=views[i], save_viz=bool(viz_every) and idx % viz_every == 0,
                                  seed=int(self.rng.integers(1 << 31)), removed=int(removed[i]))
                fut = self.writer.submit(process_scene, self.ctx, sc)
                fut.add_done_callback(self._on_summary)
                scenes += 1
                imgs += self.V
            now = time.monotonic()
            if now - t_log >= log_every or b == n_batches - 1:
                el = now - t_start
                print(f"[INFO] batch {b + 1}/{n_batches}: {scenes} scenes, {imgs} images in {el:.1f}s "
                      f"({scenes / el:.2f} scenes/s, {imgs / el:.2f} img/s) | settle {steps} steps, "
                      f"escaped {int(removed.sum())} | writer queue done={self.writer.done}", flush=True)
                t_log = now
        print("[INFO] waiting for writer threads ...", flush=True)
        self.writer.close()
        el = time.monotonic() - t_start
        print(f"[INFO] done: {scenes} scenes / {imgs} images in {el:.1f}s -> {self.out}", flush=True)
        with open(os.path.join(self.out, "run_info.json")) as f:
            info = json.load(f)
        info.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "scenes": scenes, "images": imgs,
                     "seconds": round(el, 1), "writer_errors": len(self.writer.errors)})
        with open(os.path.join(self.out, "run_info.json"), "w") as f:
            json.dump(to_plain(info), f, indent=2)
