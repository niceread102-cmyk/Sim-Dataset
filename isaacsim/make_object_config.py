#!/usr/bin/env python3
"""Part config of a new object from its STL: the part yaml that configs/map/map.yaml lists under `parts:`.

    python3 make_object_config.py obj:=SUPPORT-INPUT_SHAFT [--units mm] [--mass KG | --density G_PER_CM3]
                                  [--collision convexDecomposition] [--color R G B] [--force]

Input   assets/object/stl/<name>.stl   (.step / .obj / .ply also work; <name> is matched without regard to case)
        (assets/object/usd/ is for USD assets made in the Isaac Sim GUI, e.g. the crate of the pallets)
Output  isaacsim/configs/object/<slug>.yaml    <slug> = <name> in lower case with `-` and spaces as `_` (SUPPORT-INPUT_SHAFT -> support_input_shaft)

What is chosen here (override with the options, or edit the yaml):
  units       mm (--units: mm | cm | m | in). The size is printed: check it against the real part.
  mass        the mesh volume x --density (default 7.85 g/cm^3, steel). --mass gives the mass in kg directly.
  collision   convexDecomposition (the part yaml of side_plate_bp does the same), 32 hulls
  material    steel-like metal, the same values as side_plate_bp (grey, metallic 1.0, roughness 0.5).
              --color R G B (0..1) sets a plain colour instead (no tint).
  count       parts per bin, estimated from the bin size (about 40 % of its volume): `lab generate_isaacsim.py obj:=<name>`
              uses it, `obj_num:=N` overrides it
  recenter    true (BOP models are centred), friction 0.5, restitution 0
  layer_rpy   the thinnest side of the bounding box faces up when parts are stacked in layers (loading.mode: layers)
  grasps      48 suction points sampled automatically
  symmetry    none (not written): add `symmetry: {axis: [0, 0, 1], order: 2, flip: false}` for a part that maps onto itself
A yaml that already exists is kept (--force overwrites it). map.yaml is not changed: run the part with `lab generate_isaacsim.py obj:=<slug>`.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml

HERE = Path(__file__).resolve().parent                    # Simulation/isaacsim
SIM_DIR = HERE.parent                                     # Simulation
OBJECTS_DIR = SIM_DIR / "assets" / "object" / "stl"
PARTS_DIR = HERE / "configs" / "object"
UNIT_SCALE = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": 0.0254}
EXTENSIONS = (".stl", ".step", ".stp", ".obj", ".ply")
METAL = {"color": [0.28571427, 0.2857114, 0.2857114], "color_tint": [0.66023165, 0.66022503, 0.66022503]}  # side_plate_bp


def _obj(a: str) -> str:
    return a[len("obj:="):] if a.startswith("obj:=") else a


def find_mesh(name: str) -> Path | None:
    """assets/object/stl/<name>.<ext>, the name matched without regard to case; a path to a file works too"""
    p = Path(name)
    if p.suffix.lower() in EXTENSIONS and p.is_file():
        return p.resolve()
    stem = p.stem if p.suffix.lower() in EXTENSIONS else name
    for f in sorted(OBJECTS_DIR.iterdir()) if OBJECTS_DIR.is_dir() else []:
        if f.suffix.lower() in EXTENSIONS and f.stem.lower() == stem.lower():
            return f
    return None


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def layer_rpy(extents) -> list:
    """rpy (deg) that turns the thinnest side of the bounding box to Z (up)"""
    return {0: [0, 90, 0], 1: [90, 0, 0], 2: [0, 0, 0]}[int(np.argmin(extents))]


def fit_count(ext) -> int:
    """rough number of parts that fit in the bin (about 40 % of the bin volume filled, bounding boxes): a starting point for `count`"""
    try:
        inner = yaml.safe_load((PARTS_DIR / "bin.yaml").read_text())["bin"]["inner_size_m"]
        return max(1, int(np.prod(inner) * 0.4 / np.prod(ext)))
    except Exception:  # noqa: BLE001
        return 1


def fmt(v) -> str:
    return "[" + ", ".join(f"{x:g}" for x in v) + "]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj", metavar="obj:=<name>", type=_obj, help="object name = assets/object/stl/<name>.stl")
    ap.add_argument("--units", choices=sorted(UNIT_SCALE), default="mm", help="length unit of the mesh (default mm)")
    ap.add_argument("--mass", type=float, help="mass in kg (default: mesh volume x --density)")
    ap.add_argument("--density", type=float, default=7.85, help="g/cm^3, used when --mass is not given (default 7.85, steel)")
    ap.add_argument("--collision", choices=("convexDecomposition", "convexHull", "sdf"), default="convexDecomposition")
    ap.add_argument("--hulls", type=int, default=32, help="max convex hulls of convexDecomposition")
    ap.add_argument("--color", nargs=3, type=float, metavar=("R", "G", "B"), help="plain colour 0..1 (default: steel-like metal like side_plate_bp)")
    ap.add_argument("--force", action="store_true", help="overwrite the yaml if it exists")
    args = ap.parse_args()

    mesh_path = find_mesh(args.obj)
    if mesh_path is None:
        sys.exit(f"mesh not found: assets/object/stl/{args.obj}.stl (also {', '.join(EXTENSIONS[1:])})")
    name = slug(mesh_path.stem)
    out = PARTS_DIR / f"{name}.yaml"

    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        sys.exit(f"no triangles loaded from {mesh_path}")
    mesh = mesh.copy()
    mesh.apply_scale(UNIT_SCALE[args.units])
    ext = np.asarray(mesh.extents)                        # m
    volume = abs(float(mesh.volume))                      # m^3
    watertight = bool(mesh.is_watertight)
    if args.mass is not None:
        mass, how = args.mass, "--mass"
    elif volume > 0:
        mass, how = volume * args.density * 1000.0, f"volume {volume * 1e6:.1f} cm^3 x {args.density:g} g/cm^3"
    else:
        mass, how = 1.0, "volume is 0: default 1 kg"
    material = f"color: {fmt(args.color)}\nroughness: 0.5\nmetallic: 0.0" if args.color else \
        f"color: {fmt(METAL['color'])}\ncolor_tint: {fmt(METAL['color_tint'])}\nroughness: 0.5\nmetallic: 1.0   # steel-like, as side_plate_bp"
    n = fit_count(ext)
    rel = os.path.relpath(mesh_path, PARTS_DIR).replace(os.sep, "/")

    text = f"""# {mesh_path.stem}: made by make_object_config.py from assets/object/stl/{mesh_path.name} ({args.units}).
name: {name}
mesh: {rel}   # {ext[0] * 1000:.1f} x {ext[1] * 1000:.1f} x {ext[2] * 1000:.1f} mm{'' if watertight else ', not watertight (PhysX warns)'}
units: {args.units}
recenter: true            # move the model origin to its bounding-box center (BOP models are centered)
mass_kg: {mass:.4g}       # {how}
friction: 0.5
restitution: 0.0
collision: {args.collision}   # convexDecomposition | convexHull | sdf (most accurate, slowest)
max_convex_hulls: {args.hulls}
{material}
count: [{max(1, n * 9 // 10)}, {n}]   # parts per bin (rough: 40 % of the bin volume). Used by `obj:=` of generate_isaacsim.py; map.yaml's count wins

# symmetry: {{axis: [0, 0, 1], order: 2, flip: false}}   # a part that maps onto itself (BOP symmetries); none = not symmetric

# Where the object stands in a map that lists it under scene.objects without its own pos_m / rpy_deg (robot_sim.py): the part centre in the
# world (m) and its orientation (deg, roll pitch yaw). The entry of the map wins: - {{part: <name>, pos_m: [x, y, z], rpy_deg: [r, p, y]}}
pos_m: [0.0, 0.0, 0.0]
rpy_deg: [0, 0, 0]

# Thinnest side of the bounding box up when the parts are stacked in layers (loading.mode: layers).
layer_rpy_deg: {fmt(layer_rpy(ext))}

grasps:
  # Suction points sampled automatically on flat patches of the mesh (stored in the asset cache).
  suction_auto: {{num: 48, seed: 0}}
  suction: []               # - {{pos: [0, 0, 0.005], normal: [0, 0, 1]}}
  parallel_jaw: []          # - {{pos: [0, 0, 0], rpy_deg: [0, 0, 0], width_m: 0.03}}
"""
    if out.is_file() and not args.force:
        written = False
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        written = True

    print(f"object {mesh_path.stem}: {len(mesh.faces)} triangles, {ext[0] * 1000:.1f} x {ext[1] * 1000:.1f} x {ext[2] * 1000:.1f} mm "
          f"(units {args.units}: is that the real size?)")
    if not watertight:
        print("[WARN] mesh is not watertight: the volume (mass) is approximate and PhysX may warn about the collision")
    if len(mesh.faces) > 200_000:
        print(f"[WARN] {len(mesh.faces)} triangles: convex decomposition and the first conversion are slow")
    print(f"mass {mass:.3g} kg ({how}), {'colour ' + fmt(args.color) if args.color else 'steel-like metal'}, collision {args.collision}")
    print(f"{'written' if written else 'kept   '}  {os.path.relpath(out, SIM_DIR)}" + ("" if written else "  (--force overwrites it)"))
    print(f"next:  lab generate_isaacsim.py obj:={name} [obj_num:=N] [out:=DIR]     (or list it under `parts:` of map.yaml)")


if __name__ == "__main__":
    main()
