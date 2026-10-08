"""Vectorized signed distance to the primitives of a renderer World (the privileged expert's "point cloud").

The paper gives the expert the environment as a point cloud; here it gets the exact geometry, which is
the same information without sampling error. `distance(world, P)` is the distance from each point of
P (N×3) to the nearest obstacle surface (negative inside), including the ground z = 0.
"""
import numpy as np

from nanouav.scene.renderer import Box, Cylinder, Plane, Sphere


def _box(o, P):
    c, s = np.cos(-o.yaw), np.sin(-o.yaw)
    d = P - np.asarray(o.center, float)
    q = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], 1)
    q = np.abs(q) - np.asarray(o.half, float)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0.0)


def _cyl(o, P):
    dxy = np.hypot(P[:, 0] - o.cx, P[:, 1] - o.cy) - o.r
    dz = np.maximum(o.z0 - P[:, 2], P[:, 2] - o.z1)
    return np.hypot(np.maximum(dxy, 0), np.maximum(dz, 0)) + np.minimum(np.maximum(dxy, dz), 0.0)


def _sph(o, P):
    return np.linalg.norm(P - np.asarray(o.center, float), axis=1) - o.r


class Field:
    """Precomputed primitive arrays for fast batched queries (cylinders and boxes dominate the worlds)."""

    def __init__(self, world, ground=True):
        # a Flightmare scene (nanouav.flightmare.cloud.CloudWorld): distances to its point cloud
        self.cloud = world if hasattr(world, 'points') and hasattr(world, 'distance') else None
        self.ground = ground and world.ground is not None
        cyl = [o for o in world.objects if isinstance(o, Cylinder)]
        self.cyl = np.array([[o.cx, o.cy, o.r, o.z0, o.z1] for o in cyl], float).reshape(-1, 5)
        self.boxes = [o for o in world.objects if isinstance(o, Box)]
        self.spheres = [o for o in world.objects if isinstance(o, Sphere)]
        self.other = [o for o in world.objects if not isinstance(o, (Box, Cylinder, Sphere, Plane))]

    def local(self, center, radius):
        """Copy restricted to the primitives that can be within `radius` of `center` (faster queries)."""
        if self.cloud is not None:
            if not hasattr(self.cloud, 'local'):
                return self
            f = object.__new__(Field)
            f.__dict__.update(self.__dict__)
            f.cloud = self.cloud.local(center, radius)
            return f
        c = np.asarray(center, float)
        f = object.__new__(Field)
        f.cloud = None
        f.ground = self.ground
        if len(self.cyl):
            near = np.hypot(self.cyl[:, 0] - c[0], self.cyl[:, 1] - c[1]) - self.cyl[:, 2] < radius
            f.cyl = self.cyl[near]
        else:
            f.cyl = self.cyl
        f.boxes = [o for o in self.boxes if _box(o, c[None])[0] < radius]
        f.spheres = [o for o in self.spheres if _sph(o, c[None])[0] < radius]
        f.other = self.other
        return f

    def __call__(self, P, chunk=20000):
        P = np.asarray(P, float).reshape(-1, 3)
        out = np.full(len(P), np.inf)
        for i in range(0, len(P), chunk):
            out[i:i + chunk] = self._query(P[i:i + chunk])
        return out

    def _query(self, P):
        if self.cloud is not None:
            if getattr(self.cloud, 'split_ground', False):          # e.g. flightmare.arena.Arena
                return self.cloud.distance(P, ground=self.ground)
            return self.cloud.distance(P)
        d = P[:, 2].copy() if self.ground else np.full(len(P), np.inf)
        if len(self.cyl):
            C = self.cyl
            dxy = np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1]) - C[None, :, 2]
            dz = np.maximum(C[None, :, 3] - P[:, None, 2], P[:, None, 2] - C[None, :, 4])
            dc = np.hypot(np.maximum(dxy, 0), np.maximum(dz, 0)) + np.minimum(np.maximum(dxy, dz), 0.0)
            d = np.minimum(d, dc.min(1))
        for o in self.boxes:
            d = np.minimum(d, _box(o, P))
        for o in self.spheres:
            d = np.minimum(d, _sph(o, P))
        for o in self.other:
            d = np.minimum(d, np.array([o.sdf(p) for p in P]))
        return d


def distance(world, P, ground=True):
    return Field(world, ground)(P)
