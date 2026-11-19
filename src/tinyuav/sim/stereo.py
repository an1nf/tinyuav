"""Simulated stereo pair and Semi-Global Matching depth (OpenCV StereoSGBM); invalid pixels are 0."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from tinyuav.sim.render import Z_MIN
from tinyuav.sim.render import Box, Cylinder, Sphere, World, render_view


@dataclass
class StereoRig:
    width: int = 320                 # VGA costs 4x the render time
    height: int = 240
    hfov_deg: float = 90.0
    baseline: float = 0.10           # m
    tilt_up_deg: float = 0.0         # deg
    offset: tuple = (0.10, 0.0, 0.0)  # m, left camera in the body frame
    max_range: float = 10.0          # m, farther depth is invalid
    render_range: float = 20.0       # m
    num_disp: int = 48               # multiple of 16; min depth f*B/(n-1)
    block: int = 5
    texture_win: int = 9             # px, texture filter window
    texture_min: float = 4.0         # gray levels of local standard deviation
    shadows: bool = False            # sun shadows

    @property
    def size(self):
        return (self.height, self.width)

    @property
    def hfov(self):
        return math.radians(self.hfov_deg)

    @property
    def focal(self):
        return self.width / 2 / math.tan(self.hfov / 2)

    def min_depth(self):
        return self.focal * self.baseline / (self.num_disp - 1)


def _post_pair(gl, gr, rng):
    """Shared exposure, gamma and blur; independent sensor noise."""
    gain, gamma = rng.uniform(0.8, 1.2), rng.uniform(0.85, 1.2)
    s = rng.uniform(0, 0.8)
    sig = rng.uniform(0.003, 0.015)
    out = []
    for g in (gl, gr):
        lum = np.clip(g.astype(np.float32) / 255.0 * gain, 0, 1) ** gamma
        if s > 0.3:
            lum = cv2.GaussianBlur(lum, (0, 0), s)
        lum = lum + rng.normal(0, sig, lum.shape)
        out.append(np.clip(lum * 255 + 0.5, 0, 255).astype(np.uint8))
    return out


def _bound(o):
    if isinstance(o, Box):
        return np.asarray(o.center, float), float(np.linalg.norm(o.half))
    if isinstance(o, Cylinder):
        return np.array([o.cx, o.cy, 0.5 * (o.z0 + o.z1)]), float(np.hypot(o.r, 0.5 * (o.z1 - o.z0)))
    if isinstance(o, Sphere):
        return np.asarray(o.center, float), float(o.r)
    return None, None


def cull(world, pos, yaw, rig, shadow_r=4.0):
    """World restricted to objects inside the field of view and render range, or casting a shadow nearby."""
    p = np.asarray(pos, float)
    half = rig.hfov / 2 + 0.05
    keep = []
    for o in world.objects:
        c, r = _bound(o)
        if c is None:
            keep.append(o)
            continue
        d = c[:2] - p[:2]
        dist = float(np.hypot(*d))
        if dist - r > rig.render_range:
            continue
        if dist < r + shadow_r:
            keep.append(o)
            continue
        ang = abs((np.arctan2(d[1], d[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
        if ang - np.arcsin(min(r / dist, 1.0)) < half:
            keep.append(o)
    return World(keep, world.ground, sun=world.sun, ambient=world.ambient, diffuse=world.diffuse, sky=world.sky)


def render_stereo(world, rig, pos, yaw, pitch=0.0, roll=0.0, rng=None, rgb=False):
    """Left and right gray images and left ground-truth depth (m); rgb=True adds the left color image."""
    rng = rng if rng is not None else np.random.default_rng()
    world = cull(world, pos, yaw, rig)
    kw = dict(cam_pitch=-math.radians(rig.tilt_up_deg), hfov=rig.hfov, size=rig.size,
              d_max=rig.render_range, shadows=rig.shadows)
    ox, oy, oz = rig.offset
    out = render_view(world, pos, yaw, pitch, roll, cam_offset=(ox, oy, oz), chroma=rgb, **kw)
    gl, d = out[:2]
    gr, _ = render_view(world, pos, yaw, pitch, roll, cam_offset=(ox, oy - rig.baseline, oz), **kw)
    gl, gr = _post_pair(gl, gr, rng)
    if not rgb:
        return gl, gr, d
    col = gl.astype(np.float32)[..., None] / 255.0 * out[2] + rng.normal(0, 0.01, out[2].shape)
    return gl, gr, d, np.clip(col * 255 + 0.5, 0, 255).astype(np.uint8)


_SGBM = {}


def sgm_depth(left, right, rig):
    """SGM depth (m) of the left camera, 0 where invalid."""
    key = (rig.num_disp, rig.block)
    if key not in _SGBM:
        b = rig.block
        _SGBM[key] = cv2.StereoSGBM_create(minDisparity=0, numDisparities=rig.num_disp, blockSize=b,
                                           P1=8 * b * b, P2=32 * b * b, disp12MaxDiff=1, uniquenessRatio=10,
                                           speckleWindowSize=60, speckleRange=2,
                                           mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = _SGBM[key].compute(left, right).astype(np.float32) / 16.0
    depth = np.zeros_like(disp)
    ok = disp > 0.5
    depth[ok] = rig.focal * rig.baseline / disp[ok]
    depth[(depth > rig.max_range) | (depth < Z_MIN)] = 0.0
    depth[:, :rig.num_disp] = 0.0                      # no match on the left band
    # untextured areas: invalid below a few gray levels of contrast
    lf = left.astype(np.float32)
    k = (rig.texture_win, rig.texture_win)
    std = np.sqrt(np.maximum(cv2.blur(lf * lf, k) - cv2.blur(lf, k) ** 2, 0))
    depth[std < rig.texture_min] = 0.0
    return depth


def observe_full(world, rig, pos, yaw, pitch=0.0, roll=0.0, rng=None, rgb=False):
    """Left image, SGM depth and ground-truth depth clipped to the same range; rgb=True appends color."""
    out = render_stereo(world, rig, pos, yaw, pitch, roll, rng, rgb=rgb)
    gl, gr, d = out[:3]
    dg = np.where(d < rig.max_range, d, 0.0).astype(np.float32)
    return (gl, sgm_depth(gl, gr, rig), dg) + tuple(out[3:])


def observe(world, rig, pos, yaw, pitch=0.0, roll=0.0, rng=None):
    """SGM depth and ground-truth depth clipped to the same range."""
    gl, gr, d = render_stereo(world, rig, pos, yaw, pitch, roll, rng)
    dg = np.where(d < rig.max_range, d, 0.0).astype(np.float32)
    return sgm_depth(gl, gr, rig), dg

