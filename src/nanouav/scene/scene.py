import math
import numpy as np

W, H = 96, 96


def synthetic_scene(t: float, noise_std: float = 0.03):
    """
    Generate (gray uint8 96x96, depth float32 96x96).
    t: animation parameter (simulated seconds).
    """
    cx = W / 2 + math.sin(t * 0.3) * 18
    cy = H / 2 + math.cos(t * 0.2) * 10
    r = 22 + math.sin(t * 0.5) * 8
    base_depth = 0.5 + 1.2 * (1 + math.sin(t * 0.4))

    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = dist < r
    edge = np.clip((r - dist) / r, 0, 1)

    depth = np.where(inside,
                     np.clip(base_depth - edge * 1.4, 0.2, 4.0),
                     np.clip(base_depth + (dist - r) * 0.15, 0.2, 4.0)
                     ).astype(np.float32)
    depth += np.random.randn(H, W).astype(np.float32) * noise_std

    lum_obs = np.clip(40 + edge * 180, 0, 255).astype(np.uint8)
    lum_bg = np.clip(180 + (yy / H) * 40, 0, 255).astype(np.uint8)
    gray = np.where(inside, lum_obs, lum_bg).astype(np.uint8)
    gray = np.clip(gray.astype(np.int16) + np.random.randint(-8, 9, (H, W)), 0, 255).astype(np.uint8)

    return gray, np.clip(depth, 0.2, 4.0)


def multi_obstacle_scene(t: float):
    """Two obstacles: one near, one in the background."""
    gray1, depth1 = synthetic_scene(t, noise_std=0.02)
    gray2, depth2 = synthetic_scene(t * 0.7 + 2.1, noise_std=0.02)

    mask = depth1 <= depth2
    gray = np.where(mask, gray1, gray2).astype(np.uint8)
    depth = np.minimum(depth1, depth2)
    return gray, depth
