"""Structured-light depth sensor artifacts (PhoXi-style), applied to the renderer's perfect depth.

Real scanners (unlike the clean `distance_to_image_plane` the legacy script saved) produce:
  - holes where the projector can't illuminate the surface (projector shadow, beside tall parts)
  - no return on surfaces seen at grazing angles
  - invalid / flying pixels along depth discontinuities (removed by the sensor's filters)
  - axial noise growing ~ z^2 and depth quantization
  - nothing outside the scanning range
Invalid pixels are 0, as in PhoXi/BOP depth maps. Every stage is toggled from `camera.noise`.
"""
from __future__ import annotations

import numpy as np

from .poses import backproject


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    out = mask.copy()
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            src = mask[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
            out[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)] |= src
    return out


def _smooth_diff(p: np.ndarray, axis: int) -> np.ndarray:
    """Forward or backward difference along axis, whichever changes depth less: next to a depth jump the normal comes
    from the surface the pixel is on (a central difference across the jump would make both sides look grazing)."""
    f = np.zeros_like(p)
    b = np.zeros_like(p)
    if axis == 1:
        f[:, :-1] = p[:, 1:] - p[:, :-1]
        b[:, 1:] = f[:, :-1]
        f[:, -1], b[:, 0] = b[:, -1], f[:, 0]  # image border: the one side there is
    else:
        f[:-1] = p[1:] - p[:-1]
        b[1:] = f[:-1]
        f[-1], b[0] = b[-1], f[0]
    return np.where((np.abs(f[..., 2]) <= np.abs(b[..., 2]))[..., None], f, b)


def surface_normals(pts: np.ndarray) -> np.ndarray:
    """(H,W,3) organized point cloud -> unit normals (camera-facing)."""
    dx = _smooth_diff(pts, 1)
    dy = _smooth_diff(pts, 0)
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    flip = (n * pts).sum(-1) > 0
    n[flip] *= -1
    return n


def projector_shadow(pts: np.ndarray, valid: np.ndarray, K: np.ndarray, baseline_m, tol_m: float,
                     downsample: int = 2, slope_px: float = 2.0, max_step_m: float = 0.02) -> np.ndarray:
    """True where the surface point is occluded as seen from the projector.

    The projector is modeled as a pinhole at `baseline_m` in the camera frame, same orientation
    and intrinsics as the camera. A z-buffer is built from the projector's viewpoint; points that
    are farther than the nearest surface along their projector ray are in shadow.
    The z-buffer is splatted from the camera pixels: surfaces stretched in the projector view get extra samples
    (solid shadows, no hatching), and where the projector sees a surface at a grazing angle several camera pixels with
    different depths share one projector cell: the tolerance grows by `slope_px` cells' worth of the surface's depth
    change per projector cell (no shadow acne / stripes on walls and tilted parts).
    """
    from scipy import ndimage
    H, W = valid.shape
    s = max(int(downsample), 1)
    P = pts - np.asarray(baseline_m, float)          # points in projector frame
    z = P[..., 2]
    ok = valid & (z > 1e-6)
    u = np.floor((P[..., 0] / np.where(ok, z, 1) * K[0, 0] + K[0, 2]) / s).astype(np.int64)
    v = np.floor((P[..., 1] / np.where(ok, z, 1) * K[1, 1] + K[1, 2]) / s).astype(np.int64)
    Wp, Hp = (W + s - 1) // s, (H + s - 1) // s
    ok &= (u >= 0) & (u < Wp) & (v >= 0) & (v < Hp)
    lin = np.where(ok, v * Wp + u, 0)
    zbuf = np.full(Hp * Wp, np.inf)
    np.minimum.at(zbuf, lin[ok], z[ok])
    # a surface steep in the camera view is stretched in the projector view: one sample per camera pixel leaves gaps in
    # the occluder, and the shadowed surface behind shows through them (hatched shadows). Add samples between
    # neighbouring pixels of one surface (depth step < max_step_m) wherever they land more than one cell apart.
    uf = P[..., 0] / np.where(ok, z, 1) * K[0, 0] / s
    vf = P[..., 1] / np.where(ok, z, 1) * K[1, 1] / s
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = (slice(0, H - dy), slice(max(-dx, 0), W - max(dx, 0)))
        b = (slice(dy, H), slice(max(dx, 0), W - max(-dx, 0)))
        cells = np.maximum(np.abs(uf[b] - uf[a]), np.abs(vf[b] - vf[a]))
        m = ok[a] & ok[b] & (cells > 1.0) & (np.abs(z[b] - z[a]) < max_step_m)
        if not m.any():
            continue
        pa, pb, n = P[a][m], P[b][m], np.minimum(np.ceil(cells[m]).astype(int), 16)
        rep = np.repeat(np.arange(len(n)), n - 1)
        t = (np.arange(len(rep)) - np.repeat(np.cumsum(n - 1) - (n - 1), n - 1) + 1) / np.repeat(n, n - 1)
        q = pa[rep] + t[:, None] * (pb[rep] - pa[rep])
        qu = np.floor((q[:, 0] / q[:, 2] * K[0, 0] + K[0, 2]) / s).astype(np.int64)
        qv = np.floor((q[:, 1] / q[:, 2] * K[1, 1] + K[1, 2]) / s).astype(np.int64)
        qk = (qu >= 0) & (qu < Wp) & (qv >= 0) & (qv < Hp)
        np.minimum.at(zbuf, (qv * Wp + qu)[qk], q[qk, 2])
    zb = zbuf.reshape(Hp, Wp)
    zb = np.where(np.isfinite(zb), zb, ndimage.grey_erosion(zb, size=3)).ravel()  # fill 1-cell gaps only
    tol = np.full(z.shape, tol_m)
    if slope_px > 0:
        # depth change per projector cell along the surface (one-sided differences, not across outlines): large where
        # the projector sees the surface at a grazing angle, where the splatted z-buffer is unreliable
        Q = np.stack([uf, vf, z], -1)
        per_cell = np.zeros(z.shape)
        for g in (_smooth_diff(Q, 1), _smooth_diff(Q, 0)):
            per_cell = np.maximum(per_cell, np.abs(g[..., 2]) / np.maximum(np.hypot(g[..., 0], g[..., 1]), 0.05))
        tol = tol + slope_px * np.minimum(per_cell, 0.02)
    shadow = np.zeros_like(valid)
    shadow[ok] = z[ok] > zb[lin[ok]] + tol[ok]
    return shadow


def _grey_open_round(z: np.ndarray, width: int) -> np.ndarray:
    """Grey opening with an octagon ~ disc of this diameter (square window = blocky, rectangular results).
    Octagon = square (half-side 0.41 r) + diamond (radius 0.59 r), each cheap: min/max filter + 3x3 cross steps."""
    from scipy import ndimage
    r = width / 2.0
    sq = 2 * int(round(0.414 * r)) + 1
    steps = int(round(0.586 * r))
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    e = ndimage.minimum_filter(z, size=sq)
    for _ in range(steps):
        e = ndimage.grey_erosion(e, footprint=cross)
    d = ndimage.maximum_filter(e, size=sq)
    for _ in range(steps):
        d = ndimage.grey_dilation(d, footprint=cross)
    return d


def _draw(v, rng, default=0.0) -> float:
    """Config value -> number: a value is used as is, [lo, hi] is drawn uniformly (per image)."""
    if v is None:
        return float(default)
    if isinstance(v, (list, tuple)):
        return float(rng.uniform(float(v[0]), float(v[1])))
    return float(v)


def _discs(shape, centers, radii) -> np.ndarray:
    out = np.zeros(shape, bool)
    H, W = shape
    for (y, x), r in zip(centers, radii):
        r = int(round(r))
        y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, H), max(x - r, 0), min(x + r + 1, W)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        out[y0:y1, x0:x1] |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return out


def smear_edges(depth: np.ndarray, blur_px: float, shift_px: float) -> np.ndarray:
    """Structured-light depth is not pixel-sharp at object outlines: outlines grow (+shift, nearer surface spreads) or
    shrink (-shift) by a few pixels, and the depth is low-pass filtered so foreground and background mix across every
    outline (smeared edges, 'flying' in-between values, small slots partly filled). Pixels without depth (0) are
    ignored by the filtering and stay 0."""
    from scipy import ndimage
    has = depth > 0
    d = depth.copy()
    s = int(round(shift_px))
    if s:
        size = 2 * abs(s) + 1
        if s > 0:
            m = ndimage.grey_erosion(np.where(has, d, np.inf), size=size)  # nearest value wins -> outlines grow
        else:
            m = ndimage.grey_dilation(np.where(has, d, 0.0), size=size)    # farthest value wins -> outlines shrink
        d = np.where(has & np.isfinite(m) & (m > 0), m, d)
    if blur_px > 0.3:
        w = has.astype(np.float64)
        num = ndimage.gaussian_filter(d * w, blur_px)
        den = ndimage.gaussian_filter(w, blur_px)
        d = np.where(has & (den > 1e-3), num / np.maximum(den, 1e-12), 0.0)
    return d


def apply_depth_noise(depth_m: np.ndarray, K: np.ndarray, cam_cfg, rng: np.random.Generator,
                      part_mask: np.ndarray | None = None) -> np.ndarray:
    """part_mask: pixels of the (metal) parts - specular and clustered holes are only applied there (none if None).
    Numeric noise settings may be [lo, hi] ranges, drawn per image."""
    depth = np.nan_to_num(depth_m.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = cam_cfg.get("depth_range_m", [0.0, 1e9])
    valid = (depth >= lo) & (depth <= hi)
    ncfg = cam_cfg.get("noise", {}) or {}
    if not ncfg.get("enabled", False):
        return np.where(valid, depth, 0.0)

    pts = backproject(depth, K)
    drop = ~valid

    spec = ncfg.get("specular", {}) or {}
    holes = ncfg.get("holes", {}) or {}
    proj = ncfg.get("projector", {}) or {}
    max_inc = ncfg.get("max_incidence_deg")
    n = surface_normals(pts) if (max_inc or (spec and part_mask is not None)) else None
    ray = pts / (np.linalg.norm(pts, axis=-1, keepdims=True) + 1e-12)
    if max_inc:
        from scipy import ndimage
        cos_inc = np.abs((n * ray).sum(-1))
        lim = np.full(depth.shape, np.cos(np.deg2rad(_draw(max_inc, rng))))
        inc_parts = ncfg.get("max_incidence_parts_deg")
        if inc_parts and part_mask is not None:  # metal parts lose steep surfaces sooner than the diffuse bin
            lim[part_mask] = np.cos(np.deg2rad(_draw(inc_parts, rng)))
        steep = (cos_inc < lim).astype(np.float32)
        drop |= ndimage.uniform_filter(steep, 3) > 0.5  # 3x3 majority: the sensor's normals span a neighbourhood,
                                                         # single steep mesh facets do not lose depth (no speckle)

    if spec and part_mask is not None:
        # mirror-like reflection of the projector into the camera saturates the sensor on metal -> no depth there
        to_proj = np.asarray(proj.get("baseline_m", [0.2, 0, 0]), float) - pts
        to_proj /= np.linalg.norm(to_proj, axis=-1, keepdims=True) + 1e-12
        half = to_proj - ray
        half /= np.linalg.norm(half, axis=-1, keepdims=True) + 1e-12
        from scipy import ndimage
        mirror = np.abs((n * half).sum(-1))
        sm = float(spec.get("smooth_px", 0.0))
        if sm > 0:  # judge on the surface around each pixel, not single mesh facets (those give mottled patches)
            w = (part_mask & valid).astype(np.float64)
            mirror = ndimage.gaussian_filter(mirror * w, sm) / np.maximum(ndimage.gaussian_filter(w, sm), 1e-6)
        hot = part_mask & valid & (mirror > np.cos(np.deg2rad(_draw(spec.get("angle_deg", 6.0), rng))))
        hot = _dilate(hot, int(spec.get("dilate_px", 1)))
        # a saturated highlight loses all its depth: each highlight patch is dropped whole with this probability
        # (per pixel would scatter speckle over the faces, which the real sensor does not show)
        lab, n_hot = ndimage.label(hot)
        if n_hot:
            big = np.bincount(lab.ravel())[1:] >= int(spec.get("min_px", 0))  # tiny glints (mesh facets) keep depth
            gone = np.r_[False, big & (rng.random(n_hot) < _draw(spec.get("prob", 0.8), rng))]
            drop |= gone[lab]

    if holes and part_mask is not None:
        # patchy missing regions on metal (dark / glossy spots, failed pattern decoding)
        ys, xs = np.nonzero(part_mask & valid)
        k = int(round(_draw(holes.get("count", 0), rng)))
        if k > 0 and len(ys):
            pick = rng.integers(0, len(ys), k)
            radii = [_draw(holes.get("radius_px", 4), rng) for _ in range(k)]
            drop |= _discs(depth.shape, zip(ys[pick], xs[pick]), radii) & part_mask
    if proj.get("enabled", False):
        drop |= projector_shadow(pts, valid, K, proj.get("baseline_m", [0.2, 0, 0]),
                                 float(proj.get("tolerance_mm", 2.0)) / 1000.0, int(proj.get("downsample", 2)),
                                 float(proj.get("slope_px", 2.0)))

    cav = ncfg.get("cavity", {}) or {}
    if cav:
        # narrow deep gaps between parts: little projector light reaches the bottom and the metal walls interreflect
        # -> no depth (real scans are black between stacked parts). Gap = pixels deeper than the grey opening
        # (surfaces narrower than window_px removed) by more than depth_mm.
        w = max(int(round(_draw(cav.get("window_px", 21), rng))), 3)
        zf = np.where(valid, depth, depth[valid].max() if valid.any() else 0.0)
        gap = depth - _grey_open_round(zf, w) > _draw(cav.get("depth_mm", 12.0), rng) / 1000.0
        if part_mask is not None:  # parts and the floor between them; not the bin walls (a band between rim and pile
            n_ = n if n is not None else surface_normals(pts)  # looks like a gap too, but the real walls are measured)
            gap &= part_mask | (np.abs((n_ * ray).sum(-1)) > 0.7)
        drop |= gap & valid

    grad_thr = ncfg.get("edge_grad_mm")
    jump = None
    if grad_thr:
        gx = np.zeros_like(depth)
        gy = np.zeros_like(depth)
        gx[:, 1:] = np.abs(np.diff(depth, axis=1))
        gy[1:] = np.abs(np.diff(depth, axis=0))
        jump = np.maximum(gx, gy) * 1000.0 > float(grad_thr)
        strength = ncfg.get("edge_strength")
        if strength is not None:
            # one knob 0..1 for how dark the outlines are: share of the outline without depth (0 = none -> faint,
            # 1 = all) and width of that gap on the far side of the jump (1 px + strength * edge_width_px)
            s = float(np.clip(_draw(strength, rng), 0.0, 1.0))
            p_edge = s
            w = int(round(s * float(ncfg.get("edge_width_px", 3))))
            edges = jump
            if w > 0:
                from scipy import ndimage
                near = ndimage.minimum_filter(np.where(valid, depth, np.inf), size=2 * w + 1)
                edges = jump | (_dilate(jump, w) & valid & (depth - near > float(grad_thr) / 1000.0))
        else:
            edges = _dilate(jump, int(round(_draw(ncfg.get("edge_dilate_px", 1), rng))))
            p_edge = _draw(ncfg.get("edge_dropout_prob", 1.0), rng)
        run = float(ncfg.get("edge_dropout_run_px", 0.0))
        if run > 0 and 0 < p_edge < 1 and edges.any():
            # outline pieces of ~run px drop out together (per pixel would leave dotted outlines = speckle)
            from scipy import ndimage
            field = ndimage.gaussian_filter(rng.standard_normal(depth.shape), run)
            drop |= edges & (field < np.quantile(field[edges], p_edge))
        else:
            drop |= edges & (rng.random(depth.shape) < p_edge)

    rag = ncfg.get("edge_ragged", {}) or {}
    if rag and jump is not None:
        # outlines are not clean curves: in a band around every depth jump, blob-shaped pieces are bitten out of the
        # surfaces or pushed out over the background (the nearer surface spreads) -> irregular, mushy outlines
        from scipy import ndimage
        band = _dilate(jump, max(int(round(_draw(rag.get("band_px", 3), rng))), 1)) & valid
        if band.any():
            field = ndimage.gaussian_filter(rng.standard_normal(depth.shape), max(_draw(rag.get("blob_px", 2.0), rng), 0.5))
            amount = float(np.clip(_draw(rag.get("amount", 0.4), rng), 0.0, 1.0))
            lo_q, hi_q = np.quantile(field[band], [amount / 2, 1 - amount / 2])
            bite = band & (field > hi_q)
            push = band & (field < lo_q)
            r = max(int(round(_draw(rag.get("band_px", 3), rng))), 1)
            near = ndimage.grey_erosion(np.where(valid, depth, np.inf), size=2 * r + 1)
            grow = push & np.isfinite(near) & (near < depth - 0.002)
            depth = np.where(grow, near, depth)
            drop = (drop | bite) & ~grow

    p = _draw(ncfg.get("random_dropout", 0.0), rng)
    if p > 0:
        drop |= rng.random(depth.shape) < p

    # outlines: smeared / grown or shrunk (decided on the clean geometry above; values changed here)
    blur, shift = _draw(ncfg.get("edge_blur_px", 0.0), rng), _draw(ncfg.get("edge_shift_px", 0.0), rng)
    if blur > 0.3 or round(shift):
        depth = smear_edges(np.where(valid, depth, 0.0), blur, shift)

    out = ncfg.get("outliers", {}) or {}
    if out:
        f = _draw(out.get("fraction", 0.0), rng)
        if f > 0:
            spike = valid & (rng.random(depth.shape) < f)
            depth = np.where(spike, depth + rng.normal(0.0, _draw(out.get("mm", 10.0), rng) / 1000.0, depth.shape), depth)
        far = out.get("far", {}) or {}
        k = int(round(_draw(far.get("count", 0), rng))) if far else 0
        if k > 0:
            # small patches inside holes (shadows, gaps) decoded far behind the scene (real scans: a few per image)
            from scipy import ndimage
            lab, n_h = ndimage.label(drop & valid)
            big = np.flatnonzero(np.bincount(lab.ravel())[1:] >= 20) + 1 if n_h else []
            for h in rng.choice(big, min(k, len(big)), replace=False) if len(big) else []:
                ys, xs = np.nonzero(lab == h)
                j = rng.integers(len(ys))
                patch = _discs(depth.shape, [(ys[j], xs[j])], [_draw(far.get("radius_px", 1.5), rng)]) & (lab == h)
                depth[patch] = depth[ys[j], xs[j]] + _draw(far.get("mm", 500.0), rng) / 1000.0
                drop &= ~patch

    sigma = _draw(ncfg.get("sigma_mm_at_1m", 0.0), rng) / 1000.0
    if sigma > 0:
        depth = depth + rng.standard_normal(depth.shape) * sigma * depth ** 2
    q = float(ncfg.get("quantize_mm", 0.0)) / 1000.0
    if q > 0:
        depth = np.round(depth / q) * q
    return np.where(drop, 0.0, depth)


def normalize_texture(img: np.ndarray, percentiles=99.0) -> np.ndarray:
    """8-bit texture from a rendered / raw image (uint8, 3 channels in -> 3 channels out).

    percentiles = p (one number): exposure - gain only, the p-th percentile of the brightness maps to 235 (black stays
    black, colours keep their ratios; like a camera's auto exposure). [lo, hi]: stretch lo..hi to 0..255 (much
    harder contrast; colours saturate).

    Raw PhoXi textures are floats with an arbitrary scale and must be converted to 8 bit for a network; apply this
    SAME function to real textures at inference as camera.texture_normalize does to the rendered ones, so both
    domains share one intensity mapping (absolute brightness of the simulation does not matter then)."""
    x = np.asarray(img, np.float32)
    ref = x[..., :3].mean(-1) if x.ndim == 3 else x  # colour: one mapping for all channels (keeps the hue)
    if np.ndim(percentiles) == 0:
        y = np.clip(x * (235.0 / max(float(np.percentile(ref, float(percentiles))), 1e-6)), 0, 255).astype(np.uint8)
    else:
        lo, hi = np.percentile(ref, percentiles)
        y = np.clip((x - lo) * (255.0 / max(float(hi - lo), 1e-6)), 0, 255).astype(np.uint8)
    return np.repeat(y[..., None], 3, axis=-1) if y.ndim == 2 else y


def texture_brightness(img: np.ndarray, factor_range, rng: np.random.Generator) -> np.ndarray:
    """Overall brightness of one image: x a factor drawn from factor_range (<1 darker, >1 brighter, clipped at 255).
    texture_normalize maps every image to the same 0..255 span, so without this all images end up equally bright."""
    from .config import as_range
    f = float(rng.uniform(*as_range(factor_range)))
    return np.clip(np.asarray(img, np.float32) * f, 0, 255).round().astype(np.uint8)


def texture_noise(img: np.ndarray, tcfg, rng: np.random.Generator) -> np.ndarray:
    """Lens blur + sensor noise on the 8-bit texture (camera.texture_noise), strength drawn per image:
    blur_px = gaussian sigma in pixels, gauss_std = noise in gray levels (values or [lo, hi] ranges)."""
    from scipy import ndimage

    from .config import as_range
    x = np.asarray(img, np.float32)
    gray = x.ndim == 3 and np.array_equal(x[..., 0], x[..., 1]) and np.array_equal(x[..., 1], x[..., 2])
    g = x[..., :1] if gray else (x if x.ndim == 3 else x[..., None])  # gray: one channel, copied back at the end
    b = float(rng.uniform(*as_range(tcfg.get("blur_px", 0.0))))
    if b > 0.05:
        g = ndimage.gaussian_filter(g, (b, b, 0))
    s = float(rng.uniform(*as_range(tcfg.get("gauss_std", 0.0))))
    if s > 0:
        g = g + rng.normal(0.0, s, g.shape)
    y = np.clip(np.round(g), 0, 255).astype(np.uint8)
    if gray:
        return np.repeat(y, x.shape[2], axis=-1)
    return y if x.ndim == 3 else y[..., 0]


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """PhoXi texture is a grayscale intensity image; keep 3 channels for tooling compatibility."""
    g = (rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).clip(0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)
