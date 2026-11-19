"""Map of a Flightmare scene from Unity's depth images: a grid of poses (positions, heights, yaws, pitches),
each exact depth image back-projected and voxelised. Unseen surfaces are missing; foliage is conservative.
"""
import math
import time

import numpy as np

from tinyuav.sim.render import body_rot


def backproject(depth, R_WB, p, T_BC, fov_v_deg, max_range=20.0, stride=2, colors=None):
    """Depth image (H, W) of a camera with T_BC on a body at (p, R_WB) to world points (N, 3) [, colours]."""
    H, W = depth.shape
    f = H / 2 / math.tan(math.radians(fov_v_deg) / 2)
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[v, u]
    k = (z > 0.05) & (z < max_range)
    u, v, z = u[k], v[k], z[k]
    col = colors[v, u] if colors is not None else None
    # Flightmare camera frame: x right, y optical axis, z up
    Pc = np.stack([(u + 0.5 - W / 2) / f * z, z, -(v + 0.5 - H / 2) / f * z], 1)
    T = np.asarray(T_BC, float)
    P = (Pc @ T[:3, :3].T + T[:3, 3]) @ np.asarray(R_WB, float).T + np.asarray(p, float)
    return (P, col) if colors is not None else P


def voxel_keys(P, lo, res):
    i = np.floor((P - lo) / res).astype(np.int64)
    return (i[:, 0] << 42) | (i[:, 1] << 21) | i[:, 2]


def key_centres(K, lo, res):
    i = np.stack([K >> 42, (K >> 21) & ((1 << 21) - 1), K & ((1 << 21) - 1)], 1)
    return lo + (i + 0.5) * res


def poses(origin, size, step=4.0, heights=(1.5, 3.0), yaws=6, pitches=(-30.0, 0.0, 30.0)):
    lo, hi = origin[:2] - size[:2] / 2, origin[:2] + size[:2] / 2
    xs = np.arange(lo[0] + step / 2, hi[0], step)
    ys = np.arange(lo[1] + step / 2, hi[1], step)
    for x in xs:
        for y in ys:
            for h in heights:
                for j in range(yaws):
                    for pt in pitches:
                        yield np.array([x, y, h]), 2 * math.pi * j / yaws, math.radians(pt)


def _merge(keys, hits, new):
    K = np.concatenate([keys] + new)
    H = np.concatenate([hits] + [np.ones(len(x), np.int32) for x in new])
    U, inv = np.unique(K, return_inverse=True)
    return U, np.bincount(inv, weights=H, minlength=len(U)).astype(np.int32)


def scan(client, origin, size, res=0.15, step=4.0, heights=(1.5, 3.0), yaws=6, pitches=(-30.0, 0.0, 30.0),
         max_range=20.0, stride=2, margin=2.0, min_hits=2, log=print):
    """(voxel centres (N, 3) inside the box, frames that saw each, stats); voxels seen < min_hits are dropped."""
    origin, size = np.asarray(origin, float), np.asarray(size, float)
    lo = origin - size / 2 - margin
    hi = origin + size / 2 + margin
    T = client.T_left
    keys, hits, buf, n, t0 = np.zeros(0, np.int64), np.zeros(0, np.int32), [], 0, time.time()
    todo = list(poses(origin, size, step, heights, yaws, pitches))
    for i, (p, yaw, pitch) in enumerate(todo):
        R = body_rot(yaw, pitch, 0.0)
        d = client.render(p, R)['depth_left']
        P = backproject(d, R, p, T, client.fov, max_range, stride)
        P = P[np.all((P >= lo) & (P < hi), 1)]
        buf.append(np.unique(voxel_keys(P, lo, res)))
        n += 1
        if len(buf) >= 100 or i == len(todo) - 1:
            keys, hits = _merge(keys, hits, buf)
            buf = []
            el = time.time() - t0
            log(f'scan {i + 1}/{len(todo)} · {len(keys)} voxels · {el:.0f} s · ~{el / (i + 1) * (len(todo) - i - 1):.0f} s left')
    C = key_centres(keys, lo, res)
    inside = np.all((C[:, :2] >= origin[:2] - size[:2] / 2) & (C[:, :2] <= origin[:2] + size[:2] / 2), 1)
    keep = inside & (hits >= min_hits)
    return C[keep], hits[keep], {'frames': n, 'voxels': int(keep.sum()), 'dropped_specks': int((inside & ~keep).sum()),
                                 'seconds': round(time.time() - t0, 1)}

