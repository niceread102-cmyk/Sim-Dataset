"""Visual domain randomization per batch through USD attributes (lights, part and bin materials)."""
from __future__ import annotations

import numpy as np
from pxr import Gf, UsdGeom, UsdShade

from ..assets import define_omnipbr, set_omnipbr
from ..config import as_range


def _set_translate(prim, xyz):
    attr = prim.GetAttribute("xformOp:translate")
    if attr and attr.IsValid():
        attr.Set(Gf.Vec3d(*map(float, xyz)))
    else:
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))


class Randomizer:
    def __init__(self, stage, cfg, num_envs: int, slot_type: list[int]):
        self.stage, self.cfg, self.num_envs, self.slot_type = stage, cfg, num_envs, slot_type
        self.rcfg = cfg.get("randomization", {}) or {}
        self.shaders: dict[tuple, UsdShade.Shader] = {}
        self.bin_shaders: list = []
        if self.rcfg.get("part_materials", True):
            self._make_part_materials()
        if self.rcfg.get("bin_materials", True):
            self._make_bin_materials()

    def _material(self, path: str):
        return define_omnipbr(self.stage, path)

    def _bind(self, prim_path: str, mat):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False
        targets = [p for p in [prim] + list(prim.GetAllChildren()) if p.IsA(UsdGeom.Mesh)] or [prim]
        for p in targets:
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
        return True

    def _make_part_materials(self):
        for e in range(self.num_envs):
            for s in range(len(self.slot_type)):
                mat, sh = self._material(f"/World/Looks/rand/env{e}_part{s:03d}")
                if self._bind(f"/World/envs/env_{e}/part_{s:03d}", mat):
                    self.shaders[(e, s)] = sh

    def _make_bin_materials(self):
        for e in range(self.num_envs):
            mat, sh = self._material(f"/World/Looks/rand/env{e}_bin")
            if self._bind(f"/World/envs/env_{e}/bin", mat):
                self.bin_shaders.append(sh)

    def apply(self, rng: np.random.Generator):
        lights = self.rcfg.get("lights", {}) or {}
        dome = self.stage.GetPrimAtPath("/World/dome_light")
        if dome and dome.IsValid() and "dome_intensity" in lights:
            dome.GetAttribute("inputs:intensity").Set(float(rng.uniform(*as_range(lights["dome_intensity"]))))
        for e in range(self.num_envs):
            key = self.stage.GetPrimAtPath(f"/World/envs/env_{e}/key_light")
            if key and key.IsValid():
                if "key_intensity" in lights:
                    key.GetAttribute("inputs:intensity").Set(float(rng.uniform(*as_range(lights["key_intensity"]))))
                off = float(lights.get("key_offset_m", 0.8))
                h = rng.uniform(*as_range(lights.get("key_height_m", [1.2, 2.0])))
                _set_translate(key, [rng.uniform(-off, off), rng.uniform(-off, off), h])
        jit = float(self.rcfg.get("color_jitter", 0.08))
        for (e, s), sh in self.shaders.items():
            p = self.cfg.parts[self.slot_type[s]]
            base = np.asarray(p.get("color", [0.7, 0.7, 0.7]), float)
            col = np.clip(base * rng.uniform(1 - jit * 2, 1 + jit * 2) + rng.normal(0, jit / 2, 3), 0, 1)
            set_omnipbr(sh, col, p.get("color_tint", [1.0, 1.0, 1.0]),
                        rng.uniform(*as_range(p.get("roughness", 0.5))), rng.uniform(*as_range(p.get("metallic", 0.0))))
        for sh in self.bin_shaders:
            base = np.asarray(self.cfg.bin.get("color", [0.2, 0.3, 0.6]), float)
            set_omnipbr(sh, np.clip(base + rng.normal(0, jit, 3), 0, 1), [1.0, 1.0, 1.0], rng.uniform(0.4, 0.9), 0.0)
