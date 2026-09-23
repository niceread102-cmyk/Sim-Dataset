"""CAD -> simulation assets, cached by content hash.

    STEP/STP  --(cadquery or gmsh)-->  mesh  --(trimesh)-->  model (meters, optionally centered)
    STL/OBJ/PLY ---------------------->  mesh
    model -> <cache>/<name>_<hash>/{model.stl (m), model_mm.ply (BOP), part.usd, info.json}

USD files are authored directly with pxr (UsdGeom/UsdPhysics/UsdShade), so this module works in
plain Python with `usd-core` as well as inside Isaac Sim. Parts are dynamic rigid bodies with
convex-decomposition (or convex-hull / SDF) colliders; the bin is a kinematic rigid body.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import trimesh

from .config import to_plain

ASSET_VERSION = 3  # bump to invalidate caches when authoring changes
UNIT_SCALE = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": 0.0254}


@dataclass
class MeshAsset:
    name: str
    dir: str
    usd_path: str
    mesh: trimesh.Trimesh        # meters, in the (possibly recentered) model frame
    extents: np.ndarray          # (3,) m
    radius: float                # bounding-sphere radius around the model origin, m
    ply_mm_path: str


# ---------------------------------------------------------------- mesh loading

def _step_to_stl(step_path: str, out_stl: str, tol_mm: float):
    """Tessellate a STEP file. Tries cadquery (OCP), then gmsh. Output is in millimeters."""
    try:
        import cadquery as cq
        shape = cq.importers.importStep(step_path)
        cq.exporters.export(shape, out_stl, tolerance=tol_mm, angularTolerance=0.1)
        return
    except ImportError:
        pass
    try:
        import gmsh
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.occ.importShapes(step_path)
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
            gmsh.option.setNumber("Mesh.MeshSizeMax", max(tol_mm * 40, 1.0))
            gmsh.model.mesh.generate(2)
            gmsh.write(out_stl)
        finally:
            gmsh.finalize()
        return
    except ImportError:
        pass
    raise RuntimeError(
        f"Cannot read STEP '{step_path}': install `cadquery` (pip install cadquery) or `gmsh` "
        "(pip install gmsh), or export an STL from your CAD tool and point `mesh:` at it.")


def load_mesh(path: str, units: str = "mm", step_tol_mm: float = 0.05) -> trimesh.Trimesh:
    """Load any supported CAD/mesh file, returned in METERS."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"mesh file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".step", ".stp"):
        with tempfile.TemporaryDirectory() as td:
            stl = os.path.join(td, "tess.stl")
            _step_to_stl(path, stl, step_tol_mm)
            m = trimesh.load(stl, force="mesh")
        units = "mm"  # OCC/gmsh export STEP geometry in mm
    else:
        m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError(f"no triangles loaded from {path}")
    m = m.copy()
    m.apply_scale(UNIT_SCALE[units])
    m.merge_vertices()
    return m


def _hash(path: str | None, settings: dict) -> str:
    h = hashlib.sha1()
    h.update(str(ASSET_VERSION).encode())
    if path:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    h.update(json.dumps(to_plain(settings), sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- USD authoring

def _stage(path: str):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def _add_mesh(stage, prim_path: str, mesh: trimesh.Trimesh, approximation: str, extra: dict | None = None):
    from pxr import Gf, UsdGeom, UsdPhysics, Vt
    geom = UsdGeom.Mesh.Define(stage, prim_path)
    geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in mesh.vertices]))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray(mesh.faces.flatten().tolist()))
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = mesh.bounds
    geom.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    prim = geom.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
    extra = extra or {}
    if approximation == "convexDecomposition":
        prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
        prim.CreateAttribute("physxConvexDecompositionCollision:maxConvexHulls",
                             _sdf_type("int")).Set(int(extra.get("max_convex_hulls", 32)))
        prim.CreateAttribute("physxConvexDecompositionCollision:shrinkWrap", _sdf_type("bool")).Set(True)
    elif approximation == "sdf":
        prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
        prim.CreateAttribute("physxSDFMeshCollision:sdfResolution", _sdf_type("int")).Set(int(extra.get("sdf_resolution", 256)))
    return prim


def _sdf_type(name):
    from pxr import Sdf
    return {"int": Sdf.ValueTypeNames.Int, "bool": Sdf.ValueTypeNames.Bool,
            "float": Sdf.ValueTypeNames.Float}[name]


def _add_materials(stage, root: str, color, friction: float, restitution: float, roughness=0.5, metallic=0.0):
    from pxr import Gf, Sdf, UsdPhysics, UsdShade
    vis = UsdShade.Material.Define(stage, f"{root}/Looks/Visual")
    sh = UsdShade.Shader.Define(stage, f"{root}/Looks/Visual/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    vis.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    phys = UsdShade.Material.Define(stage, f"{root}/Looks/Physics")
    pm = UsdPhysics.MaterialAPI.Apply(phys.GetPrim())
    pm.CreateStaticFrictionAttr(float(friction))
    pm.CreateDynamicFrictionAttr(float(friction))
    pm.CreateRestitutionAttr(float(restitution))
    return vis, phys


def _bind(prim, vis, phys):
    from pxr import UsdShade
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(vis)
    api.Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def write_part_usd(path: str, mesh: trimesh.Trimesh, part_cfg):
    from pxr import UsdGeom, UsdPhysics
    stage = _stage(path)
    root = UsdGeom.Xform.Define(stage, "/Part").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(part_cfg.get("mass_kg", 0.05)))
    geom = _add_mesh(stage, "/Part/geometry", mesh, part_cfg.get("collision", "convexDecomposition"),
                     {"max_convex_hulls": part_cfg.get("max_convex_hulls", 32)})
    rough, metal = part_cfg.get("roughness", 0.5), part_cfg.get("metallic", 0.0)
    rough = float(np.mean(rough)) if isinstance(rough, (list, tuple)) else float(rough)
    metal = float(np.mean(metal)) if isinstance(metal, (list, tuple)) else float(metal)
    vis, phys = _add_materials(stage, "/Part", part_cfg.get("color", [0.7, 0.7, 0.7]),
                               part_cfg.get("friction", 0.5), part_cfg.get("restitution", 0.0), rough, metal)
    _bind(geom, vis, phys)
    stage.GetRootLayer().Save()


def procedural_bin_meshes(size, t) -> list[trimesh.Trimesh]:
    """Floor + 4 walls as separate boxes (each convex -> exact convexHull colliders).
    Bin frame: origin at bottom center, z up."""
    X, Y, Z = size
    boxes = [
        ([X, Y, t], [0, 0, t / 2]),                                   # floor
        ([t, Y, Z], [X / 2 - t / 2, 0, Z / 2]), ([t, Y, Z], [-X / 2 + t / 2, 0, Z / 2]),
        ([X - 2 * t, t, Z], [0, Y / 2 - t / 2, Z / 2]), ([X - 2 * t, t, Z], [0, -Y / 2 + t / 2, Z / 2]),
    ]
    out = []
    for ext, c in boxes:
        b = trimesh.creation.box(extents=ext)
        b.apply_translation(c)
        out.append(b)
    return out


def write_bin_usd(path: str, meshes: list[trimesh.Trimesh], bin_cfg, approximation: str):
    from pxr import UsdGeom, UsdPhysics
    stage = _stage(path)
    root = UsdGeom.Xform.Define(stage, "/Bin").GetPrim()
    stage.SetDefaultPrim(root)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)        # kinematic: never moves unless we drive it (shake)
    vis, phys = _add_materials(stage, "/Bin", bin_cfg.get("color", [0.2, 0.3, 0.6]),
                               bin_cfg.get("friction", 0.5), 0.0, 0.7, 0.0)
    for i, m in enumerate(meshes):
        _bind(_add_mesh(stage, f"/Bin/geometry_{i}", m, approximation), vis, phys)
    stage.GetRootLayer().Save()


# ---------------------------------------------------------------- public API

def prepare_part(part_cfg, cache_dir: str, force: bool = False) -> MeshAsset:
    settings = {k: part_cfg.get(k) for k in ("units", "recenter", "mass_kg", "friction", "restitution",
                                             "collision", "max_convex_hulls", "color", "roughness", "metallic",
                                             "step_tolerance_mm")}
    key = _hash(part_cfg["mesh"], settings)
    d = os.path.join(cache_dir, f"{part_cfg['name']}_{key}")
    usd, stl, ply = (os.path.join(d, f) for f in ("part.usd", "model.stl", "model_mm.ply"))
    if force and os.path.isdir(d):
        shutil.rmtree(d)
    if not all(os.path.isfile(p) for p in (usd, stl, ply)):
        os.makedirs(d, exist_ok=True)
        mesh = load_mesh(part_cfg["mesh"], part_cfg.get("units", "mm"), float(part_cfg.get("step_tolerance_mm", 0.05)))
        if part_cfg.get("recenter", True):
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
        if not mesh.is_watertight:
            print(f"[WARN] {part_cfg['name']}: mesh is not watertight — collisions/visibility may be off", flush=True)
        mesh.export(stl)
        mm = mesh.copy()
        mm.apply_scale(1000.0)
        mm.export(ply)
        write_part_usd(usd, mesh, part_cfg)
        with open(os.path.join(d, "info.json"), "w") as f:
            json.dump({"source": part_cfg["mesh"], "settings": to_plain(settings), "faces": len(mesh.faces),
                       "extents_m": mesh.extents.tolist()}, f, indent=2)
    mesh = trimesh.load(stl, force="mesh")
    return MeshAsset(part_cfg["name"], d, usd, mesh, mesh.extents.copy(),
                     float(np.linalg.norm(mesh.vertices, axis=1).max()), ply)


def prepare_bin(bin_cfg, cache_dir: str, force: bool = False) -> MeshAsset:
    settings = {k: bin_cfg.get(k) for k in ("size_m", "wall_thickness_m", "units", "collision", "color", "friction")}
    key = _hash(bin_cfg.get("mesh"), settings)
    d = os.path.join(cache_dir, f"bin_{key}")
    usd, stl = os.path.join(d, "bin.usd"), os.path.join(d, "bin.stl")
    if force and os.path.isdir(d):
        shutil.rmtree(d)
    if not (os.path.isfile(usd) and os.path.isfile(stl)):
        os.makedirs(d, exist_ok=True)
        if bin_cfg.get("mesh"):
            m = load_mesh(bin_cfg["mesh"], bin_cfg.get("units", "m"))
            meshes = [m]
            approx = bin_cfg.get("collision", "none")
            if approx == "convexHull":
                approx = "none"  # a hull would fill the bin; use the triangle mesh (fine for kinematic)
        else:
            meshes = procedural_bin_meshes(bin_cfg["size_m"], float(bin_cfg.get("wall_thickness_m", 0.01)))
            approx = "convexHull"
        trimesh.util.concatenate(meshes).export(stl)
        write_bin_usd(usd, meshes, bin_cfg, approx)
    mesh = trimesh.load(stl, force="mesh")
    return MeshAsset("bin", d, usd, mesh, mesh.extents.copy(), float(np.linalg.norm(mesh.vertices, axis=1).max()), stl)


def prepare_all(cfg, force: bool = False):
    cache = cfg.assets.cache_dir
    os.makedirs(cache, exist_ok=True)
    parts = [prepare_part(p, cache, force) for p in cfg.parts]
    bin_asset = prepare_bin(cfg.bin, cache, force)
    return parts, bin_asset


def git_hash(repo_dir: str) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None
