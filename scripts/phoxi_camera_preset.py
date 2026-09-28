"""Write configs/cameras/phoxi_m_calibrated.yaml with the REAL intrinsics of your PhoXi (replaces the datasheet
estimate in phoxi_m.yaml; everything else - range, noise model, mount - is inherited from phoxi_m.yaml).

Read from the device over GigE Vision (Photoneo GenTL producer + harvesters, see ~/photoneo_aarch64/GigE-V):
    python scripts/phoxi_camera_preset.py --device <serial>
or type the values shown in PhoXi Control (Calibration / camera matrix of the texture camera):
    python scripts/phoxi_camera_preset.py --fx 2250.1 --fy 2250.3 --cx 1030.2 --cy 771.8 --width 2064 --height 1544 \
        [--dist k1 k2 p1 p2 k3]

Then use it: configs/default.yaml -> defaults: [cameras/phoxi_m_calibrated, robots/ur10_suction]
The renderer is a pinhole camera: undistort real images with the saved `distortion` (cv2.undistort, same K) before
running a network trained on this data, and resize them by camera.render_scale.
"""
import argparse
import os
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIGE = os.path.expanduser("~/photoneo_aarch64/GigE-V/harvesters/basic")


def from_device(serial: str) -> dict:
    """Camera matrix / distortion of the main (texture) camera from the MainCameraCalibrationData chunk (same calls as
    ~/photoneo_aarch64/GigE-V/harvesters/basic/read_chunk_data.py)."""
    sys.path.insert(0, GIGE)
    from gentl_producer_loader import producer_path  # noqa: E402
    from harvesters.core import Harvester  # noqa: E402

    def chunk_values(features, name):
        sel, val = features.get_node(f"Chunk{name}Selector"), features.get_node(f"Chunk{name}Value")
        out = []
        for option in sel.symbolics:
            sel.value = option
            out.append(float(val.value))
        return out

    with Harvester() as h:
        h.add_file(str(producer_path), check_existence=True, check_validity=True)
        h.update()
        with h.create({"serial_number": serial}) as ia:
            f = ia.remote_device.node_map
            f.TriggerSelector.value = "FrameStart"
            f.TriggerMode.value = "On"
            f.TriggerSource.value = "Software"
            f.ChunkModeActive.value = True
            f.ChunkSelector.value = "MainCameraCalibrationData"
            f.ChunkEnable.value = True
            ia.start()
            f.TriggerSoftware.execute()
            with ia.fetch(timeout=20):
                K = chunk_values(f, "MainCameraCameraMatrix")
                dist = chunk_values(f, "MainCameraDistortionCoefficients")
            width, height = int(f.Width.value), int(f.Height.value)
    if len(K) != 9:
        raise RuntimeError(f"expected 9 camera-matrix values, got {len(K)}: {K}")
    return {"fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5], "width": width, "height": height, "dist": dist,
            "source": f"device {serial} (GigE chunk MainCameraCalibrationData)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="PhoXi serial number (read over GigE Vision)")
    for k in ("fx", "fy", "cx", "cy"):
        ap.add_argument(f"--{k}", type=float)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--dist", type=float, nargs="*", default=[])
    ap.add_argument("--out", default=os.path.join(REPO, "configs", "cameras", "phoxi_m_calibrated.yaml"))
    a = ap.parse_args()
    if a.device:
        c = from_device(a.device)
    elif None not in (a.fx, a.fy, a.cx, a.cy, a.width, a.height):
        c = {"fx": a.fx, "fy": a.fy, "cx": a.cx, "cy": a.cy, "width": a.width, "height": a.height, "dist": a.dist,
             "source": "entered by hand"}
    else:
        ap.error("give --device <serial>, or all of --fx --fy --cx --cy --width --height")

    base = yaml.safe_load(open(os.path.join(REPO, "configs", "cameras", "phoxi_m.yaml")))["camera"]
    old = base["intrinsics"]
    sx = c["width"] / base["resolution"][0]
    print(f"datasheet estimate (phoxi_m.yaml, {base['resolution'][0]}x{base['resolution'][1]}): fx {old['fx']} fy {old['fy']} "
          f"cx {old['cx']} cy {old['cy']}")
    print(f"calibrated ({c['width']}x{c['height']}): fx {c['fx']:.2f} fy {c['fy']:.2f} cx {c['cx']:.2f} cy {c['cy']:.2f}"
          f"  -> focal length differs by {100 * (c['fx'] / (old['fx'] * sx) - 1):+.1f} % (object size in the image)")
    cam = {"resolution": [c["width"], c["height"]],
           "intrinsics": {"fx": round(c["fx"], 4), "fy": round(c["fy"], 4), "cx": round(c["cx"], 4), "cy": round(c["cy"], 4)}}
    if c["dist"]:
        cam["distortion"] = [round(v, 8) for v in c["dist"]]
    with open(a.out, "w") as f:
        f.write(f"# PhoXi M with the CALIBRATED texture-camera intrinsics - {c['source']}, {time.strftime('%Y-%m-%d')}.\n"
                "# Written by scripts/phoxi_camera_preset.py; everything else comes from phoxi_m.yaml.\n"
                "# `distortion` is NOT rendered (pinhole): undistort real images with it before inference.\n"
                "defaults: [phoxi_m]\n")
        yaml.safe_dump({"camera": cam}, f, sort_keys=False, default_flow_style=None)
    print(f"wrote {a.out}\nuse it: configs/default.yaml -> defaults: [cameras/phoxi_m_calibrated, robots/ur10_suction]")


if __name__ == "__main__":
    main()
