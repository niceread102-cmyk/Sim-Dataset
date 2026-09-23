"""Visual reports (PIL + numpy only).

bin_panel():  [ RGB + per-part XYZ axes + top-most part arrow/label | colorized depth | instance masks ]
              with a per-part 6DOF table underneath.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .poses import project
from .transforms import mat_to_rpy

AXIS_COLORS = [(235, 60, 50), (60, 200, 70), (60, 120, 255)]


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def colorize_depth(depth_m: np.ndarray, rng_m=None) -> np.ndarray:
    """Turbo-like colormap; invalid (0) pixels black."""
    valid = depth_m > 0
    if not valid.any():
        return np.zeros(depth_m.shape + (3,), np.uint8)
    lo, hi = rng_m if rng_m else (np.percentile(depth_m[valid], 1), np.percentile(depth_m[valid], 99))
    x = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0, 1)
    stops = np.array([[48, 18, 59], [70, 130, 250], [30, 210, 190], [150, 250, 60], [250, 185, 40], [200, 30, 10]], float)
    pos = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(x, pos, stops[:, c]) for c in range(3)], -1).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def instance_overlay(rgb: np.ndarray, inst: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """inst: (H,W) int, -1 = background."""
    out = rgb.astype(np.float32).copy()
    ids = np.unique(inst[inst >= 0])
    rng = np.random.default_rng(7)
    palette = rng.integers(40, 255, size=(int(ids.max()) + 1 if len(ids) else 1, 3))
    for i in ids:
        m = inst == i
        out[m] = (1 - alpha) * out[m] + alpha * palette[i]
    return out.clip(0, 255).astype(np.uint8)


def _outline(mask: np.ndarray, width: int) -> np.ndarray:
    """Boundary pixels of a mask, `width` px thick (outside the mask)."""
    grown = mask.copy()
    for _ in range(max(width, 1)):
        g = grown.copy()
        g[1:] |= grown[:-1]
        g[:-1] |= grown[1:]
        g[:, 1:] |= grown[:, :-1]
        g[:, :-1] |= grown[:, 1:]
        grown = g
    return grown & ~mask


def _arrow(draw, p0, p1, color, width):
    draw.line([tuple(p0), tuple(p1)], fill=color, width=width)
    d = np.asarray(p1, float) - np.asarray(p0, float)
    n = np.linalg.norm(d)
    if n < 1e-6:
        return
    d /= n
    L = max(8.0, width * 3.5)
    perp = np.array([-d[1], d[0]])
    tip = np.asarray(p1, float)
    draw.polygon([tuple(tip), tuple(tip - d * L + perp * L * 0.5), tuple(tip - d * L - perp * L * 0.5)], fill=color)


def draw_part_axes(img: Image.Image, K, parts, topmost: int | None, top_grasp_cam=None, scale: float = 1.0):
    """parts: list of dicts with R (3x3), t_mm (3,), axis_len_m, label, visib."""
    draw = ImageDraw.Draw(img)
    f = _font(max(int(12 * scale), 10))
    w = max(int(2 * scale), 1)
    for i, p in enumerate(parts):
        if p["visib"] <= 0.05:
            continue
        R, t = np.asarray(p["R"]), np.asarray(p["t_mm"]) / 1000.0
        L = p["axis_len_m"] * (1.4 if i == topmost else 1.0)
        pts = np.vstack([t, t + R[:, 0] * L, t + R[:, 1] * L, t + R[:, 2] * L])
        if np.any(pts[:, 2] <= 0):
            continue
        uv = project(K, pts)
        for a in range(3):
            draw.line([tuple(uv[0]), tuple(uv[a + 1])], fill=AXIS_COLORS[a], width=w * (2 if i == topmost else 1))
        draw.text((uv[0][0] + 4, uv[0][1] + 2), str(p["label"]), fill=(255, 255, 255), font=f,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    if topmost is not None and top_grasp_cam is not None:
        T = top_grasp_cam                      # 4x4 grasp pose in camera frame (meters)
        tcp, approach = T[:3, 3], T[:3, 2]
        start = tcp - approach * 0.12
        if start[2] > 0 and tcp[2] > 0:
            uv = project(K, np.vstack([start, tcp]))
            _arrow(draw, uv[0], uv[1], (255, 225, 0), max(int(5 * scale), 3))
            # crosshair at the grasp point (the arrow collapses to a dot for a top-down approach)
            r = max(9 * scale, 7)
            x, y = uv[1]
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 225, 0), width=w + 1)
            draw.line([(x - 2 * r, y), (x + 2 * r, y)], fill=(255, 225, 0), width=w)
            draw.line([(x, y - 2 * r), (x, y + 2 * r)], fill=(255, 225, 0), width=w)


def top_text(p, grasp_cam) -> list[str]:
    t = np.asarray(p["t_mm"])
    rpy = mat_to_rpy(np.asarray(p["R"]))
    lines = [f"TOP-MOST pickable: #{p['label']} ({p['name']})  visib={p['visib']:.2f}",
             f"cam  xyz[mm]=({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f})  rpy[deg]=({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f})"]
    if p.get("t_bin_mm") is not None:
        tb = p["t_bin_mm"]
        rb = mat_to_rpy(np.asarray(p["R_bin"]))
        lines.append(f"bin  xyz[mm]=({tb[0]:.1f}, {tb[1]:.1f}, {tb[2]:.1f})  rpy[deg]=({rb[0]:.1f}, {rb[1]:.1f}, {rb[2]:.1f})")
    if grasp_cam is not None:
        g = grasp_cam[:3, 3] * 1000
        a = grasp_cam[:3, 2]
        lines.append(f"grasp TCP cam[mm]=({g[0]:.1f}, {g[1]:.1f}, {g[2]:.1f})  approach=({a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f})")
    return lines


def part_table_lines(parts, max_rows: int = 40) -> list[str]:
    hdr = f"{'id':>3} {'type':<10} {'x_mm':>8} {'y_mm':>8} {'z_mm':>8} {'roll':>7} {'pitch':>7} {'yaw':>7} {'visib':>5} {'grasps':>6}"
    lines = [hdr]
    for p in parts[:max_rows]:
        t = p["t_mm"]
        r = mat_to_rpy(np.asarray(p["R"]))
        lines.append(f"{p['label']:>3} {p['name'][:10]:<10} {t[0]:8.1f} {t[1]:8.1f} {t[2]:8.1f} "
                     f"{r[0]:7.1f} {r[1]:7.1f} {r[2]:7.1f} {p['visib']:5.2f} {p.get('n_valid_grasps', 0):6d}")
    if len(parts) > max_rows:
        lines.append(f"... {len(parts) - max_rows} more (see poses/*.csv)")
    return lines


def bin_panel(rgb, depth_m, inst, K, parts, topmost, top_grasp_cam, title: str = "", depth_range=None) -> Image.Image:
    H, W = rgb.shape[:2]
    scale = max(W / 640, 1.0)
    base = rgb[..., :3].copy()
    if topmost is not None:
        base[_outline(inst == topmost, max(int(2 * scale), 2))] = (255, 225, 0)
    ann = Image.fromarray(base)
    draw_part_axes(ann, K, parts, topmost, top_grasp_cam, scale)
    tiles = [ann, Image.fromarray(colorize_depth(depth_m, depth_range)), Image.fromarray(instance_overlay(rgb[..., :3], inst))]
    f = _font(int(13 * scale))
    lh = int(17 * scale)
    lines = [title] if title else []
    if topmost is not None:
        lines += top_text(parts[topmost], top_grasp_cam)
    else:
        lines.append("no pickable part (visibility / grasp constraints)")
    lines.append("")
    lines += part_table_lines(parts)
    text_h = lh * (len(lines) + 1)
    canvas = Image.new("RGB", (W * 3, H + text_h), (20, 20, 24))
    for k, t in enumerate(tiles):
        canvas.paste(t, (k * W, 0))
    d = ImageDraw.Draw(canvas)
    mono = None
    for name in ("DejaVuSansMono.ttf", "Menlo.ttc", "Courier New.ttf"):
        try:
            mono = ImageFont.truetype(name, int(13 * scale))
            break
        except OSError:
            continue
    mono = mono or f
    for k, line in enumerate(lines):
        color = (255, 225, 0) if (topmost is not None and 0 < k <= 4 and line.startswith(("TOP", "cam", "bin", "grasp"))) else (225, 225, 225)
        d.text((10, H + 6 + k * lh), line, fill=color, font=mono)
    for k, lab in enumerate(["RGB + 6DOF axes (x red, y green, z blue) / top-most arrow", "depth (sensor model)", "instance masks"]):
        d.text((k * W + 8, 6), lab, fill=(255, 255, 255), font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    return canvas
