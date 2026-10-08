"""The privileged map of a Flightmare scene from Unity's own depth images (when the scene has no colliders).

Unity's `PointCloud` request only sees colliders (Physics.OverlapSphere in SavePointCloud.cs). The forest of
the 2022 binary has almost none: the probe got a 17 × 18 m flat sheet below the river and no trees or banks.
The scene was made for agile_flight, where obstacles are spawned objects and collisions are computed from
their list, not from the scene.

So the map is built from what Unity renders: the camera is placed on a grid of poses (positions × heights ×
yaws × pitches), each depth image (exact, along the optical axis) is back-projected to world points, and the
points are voxelised. This is the same information as the paper's point cloud (the geometry of the scene as
rendered), obtained without colliders. Two limits:
  · what no pose sees (behind a trunk from every grid point) is missing: a fine grid and several heights
    keep that small;
  · the depth pass draws foliage cards without their alpha cut-out, so tree crowns are larger, solid
    triangles: the map is conservative there.
"""
import math
import time

import numpy as np

from nanouav.scene.renderer import body_rot


def backproject(depth, R_WB, p, T_BC, fov_v_deg, max_range=20.0, stride=2, colors=None):
    """Depth image (H, W, metres along the optical axis, top row first) of a Flightmare camera with
    T_BC (camera → body) on a body at (p, R_WB) → world points (N, 3) [, their colours from `colors`]."""
    H, W = depth.shape
    f = H / 2 / math.tan(math.radians(fov_v_deg) / 2)
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[v, u]
    k = (z > 0.05) & (z < max_range)
    u, v, z = u[k], v[k], z[k]
    col = colors[v, u] if colors is not None else None
    # Flightmare camera frame: x right, y optical axis, z up (see bridge.R_FWD)
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
    """→ voxel centres (N, 3) of the surfaces seen inside the box (scene frame), the number of frames that
    saw each one, stats. Voxels seen by fewer than min_hits frames are dropped (isolated specks: a real
    surface is seen from many poses of the grid)."""
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


def survey(client, center, size, alt=60.0, step=60.0, cell=1.0, log=print):
    """Top-down views from `alt` m over a size × size m square → (RGB mosaic, height map) on a grid of
    `cell` m (the colour and height of the highest surface per cell), to choose where to scan and fly."""
    cx, cy = center
    n = int(round(size / cell))
    G = np.full((n, n), np.nan)
    RGB = np.zeros((n, n, 3), np.uint8)
    xs = np.arange(cx - size / 2 + step / 2, cx + size / 2, step)
    ys = np.arange(cy - size / 2 + step / 2, cy + size / 2, step)
    R = body_rot(0.0, math.pi / 2, 0.0)                     # nose down: the camera looks at the ground
    for k, (x, y) in enumerate([(x, y) for x in xs for y in ys]):
        p = np.array([x, y, alt])
        o = client.render(p, R)
        P, C = backproject(o['depth_left'], R, p, client.T_left, client.fov, 0.98 * client.depth_far, stride=1,
                           colors=o['left'])
        i = ((P[:, 0] - (cx - size / 2)) / cell).astype(int)
        j = ((P[:, 1] - (cy - size / 2)) / cell).astype(int)
        m = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        i, j, P, C = i[m], j[m], P[m], C[m]
        up = ~(G[i, j] > P[:, 2])                          # only points above what other views already saw
        i, j, P, C = i[up], j[up], P[up], C[up]
        order = np.argsort(P[:, 2])                         # highest written last
        G[i[order], j[order]] = np.fmax(G[i[order], j[order]], P[order, 2])
        RGB[i[order], j[order]] = C[order]
        log(f'survey {k + 1}/{len(xs) * len(ys)}')
    return RGB, G
