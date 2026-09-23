"""Write assets/example_gear.stl (mm): a 20-tooth spur gear with a bore, so the pipeline runs
without your CAD. Pure numpy + trimesh."""
import argparse
import os

import numpy as np
import trimesh


def gear_mesh(teeth=20, r_root=22.0, tooth_h=4.0, r_bore=6.0, thickness=8.0, samples_per_tooth=16):
    n = teeth * samples_per_tooth
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    phase = (theta * teeth / (2 * np.pi)) % 1.0
    # trapezoidal tooth: rise 0.15, top 0.3, fall 0.15, root 0.4
    prof = np.interp(phase, [0.0, 0.15, 0.45, 0.6, 1.0], [0.0, 1.0, 1.0, 0.0, 0.0])
    r_out = r_root + tooth_h * prof
    outer = np.stack([r_out * np.cos(theta), r_out * np.sin(theta)], 1)
    inner = np.stack([r_bore * np.cos(theta), r_bore * np.sin(theta)], 1)
    z0, z1 = -thickness / 2, thickness / 2
    # vertex blocks: outer bottom, outer top, inner bottom, inner top
    V = np.concatenate([
        np.c_[outer, np.full(n, z0)], np.c_[outer, np.full(n, z1)],
        np.c_[inner, np.full(n, z0)], np.c_[inner, np.full(n, z1)],
    ])
    ob, ot, ib, it = 0, n, 2 * n, 3 * n
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[ot + i, ot + j, it + j], [ot + i, it + j, it + i]]   # top annulus (+Z)
        F += [[ob + i, ib + j, ob + j], [ob + i, ib + i, ib + j]]   # bottom annulus (-Z)
        F += [[ob + i, ob + j, ot + j], [ob + i, ot + j, ot + i]]   # outer wall
        F += [[ib + i, it + j, ib + j], [ib + i, it + i, it + j]]   # bore wall
    m = trimesh.Trimesh(V, np.array(F), process=True)
    assert m.volume > 0, "face winding should point outward"
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "example_gear.stl"))
    a = ap.parse_args()
    m = gear_mesh()
    assert m.is_watertight, "example gear mesh should be watertight"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    m.export(a.out)
    print(f"wrote {os.path.abspath(a.out)}  faces={len(m.faces)} extents_mm={m.extents.round(2).tolist()}")
