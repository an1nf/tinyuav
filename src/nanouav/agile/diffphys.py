"""R25 · Learning to fly through a forest by differentiable physics (Zhang, Hu, Song, Zou, Lin 2025,
"Learning vision-based agile flight via differentiable physics", Nature Machine Intelligence; arXiv
2407.10648), on the worlds of R18.

No expert and no labels: a small recurrent policy (CNN + GRU) sees a 16 × 12 inverse-depth image, its
velocity and attitude and the direction to the goal, and outputs an acceleration. It is trained by
back-propagating a loss through a differentiable point-mass simulator over 150 steps of 1/15 s:

* velocity tracking — smooth L1 of the gap between the desired velocity (toward the goal, capped at v_des) and
  the velocity averaged over the last ~2 s (so the drone may swerve locally);
* collision — softplus barrier on the clearance to the nearest obstacle, plus the approach speed toward it
  times max(1 − clearance, 0)²: a trunk being left behind costs nothing, one ahead does;
* acceleration and jerk (smoothness).

The obstacle distance is analytic (vertical cylinders and the ground), so its gradient is exact; the depth
image is rendered by casting rays against the same cylinders (no gradient through it, as in the paper).
"""
import math
import os

import numpy as np

H_IMG, W_IMG = 12, 16                 # network input (the paper: 16 × 12 after 2 × 2 max-pooling of 32 × 24)
RENDER = (24, 32)
HFOV = math.radians(90.0)
D_MAX = 10.0                          # m, as the stereo rig
DT = 1.0 / 15.0
G = 9.81
R_Q = 0.2                             # drone radius (as R18)
A_MAX = 2.5 * G                       # m/s², limit of the commanded net acceleration
LAMBDA_A = 12.0                       # 1/s, first-order response of the acceleration (the paper's λ)
DRAG = (0.01, 0.15)                   # quadratic and linear drag coefficients
N_OBS = 128                           # cylinders per environment (padded)
W = dict(v=1.0, c=2.0, a=0.01, j=0.001)
BETA = (2.5, 32.0)
GRAD_DECAY = 0.4                      # α of the temporal gradient decay e^{−α·dt} per step
AUG, AUG_HOLES = True, 0.1             # training images: 10 % of the 16 × 12 cells as holes, 5 % noise


# ── worlds ────────────────────────────────────────────────────────────────
def forest_batch(rng, B, density=(1 / 49, 1 / 16), r=(0.15, 0.4), length=60.0, width=30.0, clear=2.5):
    """B Poisson forests like the training forests of R18 (spacing 4–7 m), start at the origin (1.5 m up),
    goal 40 m along +x. → cx, cy, rr (B, N_OBS) float32 (padding: far away, radius 0)."""
    cx = np.full((B, N_OBS), 1e4, np.float32)
    cy = np.full((B, N_OBS), 1e4, np.float32)
    rr = np.zeros((B, N_OBS), np.float32)
    for b in range(B):
        dens = rng.uniform(*density)
        n = rng.poisson(dens * length * width)
        xy = np.stack([rng.uniform(-5, length - 5, n), rng.uniform(-width / 2, width / 2, n)], 1)
        xy = xy[np.hypot(xy[:, 0], xy[:, 1]) > clear]
        if len(xy) > N_OBS:                              # keep the ones nearest the line
            xy = xy[np.argsort(np.abs(xy[:, 1]))[:N_OBS]]
        k = len(xy)
        cx[b, :k], cy[b, :k] = xy[:, 0], xy[:, 1]
        rr[b, :k] = rng.uniform(*r, k)
    return cx, cy, rr


def env_obstacles(env):
    """The cylinders of an R18 environment → cx, cy, rr (1, N_OBS)."""
    C = [o for o in env.world.objects if hasattr(o, 'cx')]
    C = sorted(C, key=lambda o: abs(o.cy - env.start[1]))[:N_OBS]
    cx = np.full((1, N_OBS), 1e4, np.float32)
    cy = np.full((1, N_OBS), 1e4, np.float32)
    rr = np.zeros((1, N_OBS), np.float32)
    for k, o in enumerate(C):
        cx[0, k], cy[0, k], rr[0, k] = o.cx - env.start[0], o.cy - env.start[1], o.r
    return cx, cy, rr


# ── geometry (TensorFlow) ─────────────────────────────────────────────────
def _tf():
    import tensorflow as tf
    return tf


def clearance(p, cx, cy, rr):
    """Distance from p (B, 3) to the nearest obstacle surface (cylinders, ground) and the unit vector from
    that point toward p. → d (B,), n (B, 3)."""
    tf = _tf()
    dx = p[:, None, 0] - cx
    dy = p[:, None, 1] - cy
    rho = tf.sqrt(dx * dx + dy * dy + 1e-9)
    dc = rho - rr                                          # (B, N)
    k = tf.argmin(dc, axis=1)
    d_cyl = tf.reduce_min(dc, axis=1)
    ex = tf.gather(dx / rho, k, batch_dims=1)
    ey = tf.gather(dy / rho, k, batch_dims=1)
    n_cyl = tf.stack([ex, ey, tf.zeros_like(ex)], 1)
    d_gnd = p[:, 2]
    use_g = d_gnd < d_cyl
    d = tf.where(use_g, d_gnd, d_cyl)
    n = tf.where(use_g[:, None], tf.constant([[0.0, 0.0, 1.0]]) * tf.ones_like(n_cyl), n_cyl)
    return d, n


def ray_dirs(h=RENDER[0], w=RENDER[1], hfov=HFOV):
    """Unit ray directions of the camera (x forward, y left, z up) → (h·w, 3) and the forward cosine."""
    f = w / 2 / math.tan(hfov / 2)
    u = (np.arange(w) + 0.5 - w / 2)
    v = (np.arange(h) + 0.5 - h / 2)
    U, V = np.meshgrid(u, v)
    D = np.stack([np.ones_like(U) * f, -U, -V], -1).reshape(-1, 3)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    return D.astype(np.float32)


_RAYS = ray_dirs()


def camera_rot(yaw, thrust):
    """Camera → world rotation (B, 3, 3): yaw about z, then the body tilt that aligns z with the thrust."""
    tf = _tf()
    zb = thrust / tf.norm(thrust, axis=1, keepdims=True)
    xc = tf.stack([tf.cos(yaw), tf.sin(yaw), tf.zeros_like(yaw)], 1)
    yb = tf.linalg.cross(zb, xc)
    yb = yb / tf.norm(yb, axis=1, keepdims=True)
    xb = tf.linalg.cross(yb, zb)
    return tf.stack([xb, yb, zb], 2)


def render_depth(p, Rc, cx, cy, rr):
    """Inverse depth image (B, 12, 16, 1): rays against the cylinders and the ground, forward distance clipped
    at D_MAX, 2 × 2 min-pooled distance (nearest kept), then 1/d (far → 1/D_MAX, near → large)."""
    tf = _tf()
    D = tf.einsum('bij,rj->bri', Rc, tf.constant(_RAYS))  # (B, R, 3) world directions
    O = p[:, None, :]
    ox = O[..., 0:1] - cx[:, None, :]
    oy = O[..., 1:2] - cy[:, None, :]
    dx, dy = D[..., 0:1], D[..., 1:2]
    a = dx * dx + dy * dy + 1e-9
    bq = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - rr[:, None, :] ** 2
    disc = bq * bq - 4 * a * c
    t = (-bq - tf.sqrt(tf.maximum(disc, 0.0))) / (2 * a)
    t = tf.where((disc > 0) & (t > 0.05), t, 1e3)
    t_cyl = tf.reduce_min(t, axis=2)
    dz = D[..., 2]
    t_gnd = tf.where(dz < -1e-3, -O[..., 2] / tf.minimum(dz, -1e-3), 1e3)
    tt = tf.minimum(t_cyl, t_gnd)
    fwd = tf.constant(_RAYS[:, 0])[None]                  # forward cosine of each ray in the camera frame
    z = tf.clip_by_value(tt * fwd, 0.3, D_MAX)
    z = tf.reshape(z, (-1, RENDER[0], RENDER[1], 1))
    z = -tf.nn.max_pool2d(-z, 2, 2, 'VALID')             # min-pool: the nearest of each 2 × 2
    return tf.stop_gradient(1.0 / z)


# ── policy ────────────────────────────────────────────────────────────────
def build_policy(hidden=192):
    """CNN (kernels 2, 3, 3; 32, 64, 128 filters, LeakyReLU) + state embedding added + GRU + linear head.
    Inputs: inverse depth (12, 16, 1), state (9: velocity /10, thrust direction, desired velocity /10, all in
    the heading frame) and the GRU state. Output: acceleration (3, heading frame, × A_MAX via tanh)."""
    tf = _tf()
    L = tf.keras.layers
    img = tf.keras.Input((H_IMG, W_IMG, 1))
    st = tf.keras.Input((9,))
    h0 = tf.keras.Input((hidden,))
    x = L.Conv2D(32, 2, activation='leaky_relu')(img)
    x = L.MaxPool2D(2)(x)
    x = L.Conv2D(64, 3, activation='leaky_relu')(x)
    x = L.Conv2D(128, 3, activation='leaky_relu')(x)
    x = L.Flatten()(x)
    x = L.Dense(hidden)(x)
    s = L.Dense(hidden)(st)
    z = L.LeakyReLU()(L.Add()([x, s]))
    h = L.GRUCell(hidden)(z, [h0])[0]
    out = L.Dense(3, activation='tanh')(L.LeakyReLU()(h))
    return tf.keras.Model([img, st, h0], [out, h])


def heading_frame(vec, yaw):
    tf = _tf()
    c, s = tf.cos(yaw)[:, None], tf.sin(yaw)[:, None]
    x, y, z = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
    return tf.concat([c * x + s * y, -s * x + c * y, z], 1)


def from_heading(vec, yaw):
    tf = _tf()
    c, s = tf.cos(yaw)[:, None], tf.sin(yaw)[:, None]
    x, y, z = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
    return tf.concat([c * x - s * y, s * x + c * y, z], 1)


def state_input(v, thrust, v_set, yaw):
    """The 9 state numbers in the heading frame (the same at test time in `Rollout`): velocity, thrust
    direction and the desired velocity (toward the goal at v_des, as the paper's target velocity), /10."""
    tf = _tf()
    zb = thrust / tf.norm(thrust, axis=1, keepdims=True)
    return tf.concat([heading_frame(v, yaw) / 10.0, heading_frame(zb, yaw), heading_frame(v_set, yaw) / 10.0], 1)


def _decay(x, g):
    """Same value, gradient scaled by g (temporal gradient decay)."""
    tf = _tf()
    return g * x + (1 - g) * tf.stop_gradient(x)


def rollout_loss(policy, cx, cy, rr, v_des, T=150, z0=1.5, seed=0):
    """One batch of episodes through the differentiable simulator → (loss, metrics)."""
    tf = _tf()
    B = cx.shape[0]
    rng = np.random.default_rng(seed)
    goal = tf.constant(np.tile([[40.0, 0.0, z0]], (B, 1)).astype(np.float32))
    vd = tf.constant(np.asarray(v_des, np.float32).reshape(B, 1))
    p = tf.constant(np.c_[np.zeros((B, 2)), np.full(B, z0)].astype(np.float32))
    v = tf.constant(np.c_[np.asarray(v_des).reshape(B, 1) * rng.uniform(0.0, 1.0, (B, 1)),
                          np.zeros((B, 2))].astype(np.float32))
    loss, mt = _rollout(policy, cx, cy, rr, vd, goal, p, v, T)
    return loss, {k: float(x) for k, x in mt.items()}


_COMPILED = {}


def _rollout(policy, cx, cy, rr, vd, goal, p, v, T):
    tf = _tf()
    key = (id(policy), T)
    if key not in _COMPILED:
        _COMPILED[key] = tf.function(lambda *a: _rollout_body(policy, *a, T=T))
    return _COMPILED[key](cx, cy, rr, vd, goal, p, v)


def _rollout_body(policy, cx, cy, rr, vd, goal, p, v, T):
    tf = _tf()
    B = tf.shape(cx)[0]
    a = tf.zeros((B, 3))
    a_prev = tf.zeros((B, 3))
    h = tf.zeros((B, policy.inputs[2].shape[-1]))
    v_avg = v
    yaw = tf.zeros((B,))
    gdec = math.exp(-GRAD_DECAY * DT)
    k_a = 1 - math.exp(-LAMBDA_A * DT)
    a_cmd_delayed = tf.zeros((B, 3))
    Lv = Lc = La = Lj = tf.zeros((B,))
    alive = tf.ones((B,))
    min_clear = tf.fill([B], 99.0)
    for k in range(T):
        thrust = a + tf.constant([[0.0, 0.0, G]])
        Rc = camera_rot(tf.stop_gradient(yaw), tf.stop_gradient(thrust))
        img = render_depth(tf.stop_gradient(p), Rc, cx, cy, rr)
        if AUG:                                        # SGM-like: holes read as far, multiplicative noise
            hole = tf.random.uniform(tf.shape(img)) < AUG_HOLES
            img = tf.where(hole, 1.0 / D_MAX, img * (1.0 + 0.05 * tf.random.normal(tf.shape(img))))
        to_goal = goal - p
        dist = tf.norm(to_goal, axis=1, keepdims=True)
        gdir = to_goal / tf.maximum(dist, 1e-3)
        st = state_input(v, thrust, tf.stop_gradient(gdir * vd), tf.stop_gradient(yaw))
        out, h = policy([img, st, h])
        a_cmd = from_heading(out * A_MAX, tf.stop_gradient(yaw))
        # point mass: one-step latency, first-order acceleration response, drag
        a = a + (a_cmd_delayed - a) * k_a
        a_cmd_delayed = a_cmd
        sp = tf.norm(v, axis=1, keepdims=True)
        drag = -DRAG[0] * sp * v - DRAG[1] * v
        v = _decay(v + (a + drag) * DT, gdec)
        p = _decay(p + v * DT, gdec)
        sp_xy = tf.norm(v[:, :2], axis=1)
        yaw = tf.where(sp_xy > 0.5, tf.atan2(v[:, 1], v[:, 0]), yaw)
        # losses
        v_set = tf.stop_gradient(gdir * tf.minimum(vd, dist / 1.0))
        v_avg = v_avg + (v - v_avg) * (DT / 2.0)
        e = tf.norm(v_set - v_avg, axis=1)
        lv = tf.where(e < 1.0, 0.5 * e * e, e - 0.5)
        d, n = clearance(p, cx, cy, rr)
        cl = d - R_Q
        approach = tf.nn.relu(-tf.reduce_sum(v * tf.stop_gradient(n), axis=1))
        clb = tf.maximum(cl, -0.3)                     # the barrier saturates inside an obstacle
        lc = approach * tf.nn.relu(1.0 - clb) ** 2 + BETA[0] * tf.nn.softplus(-BETA[1] * clb)
        la = tf.reduce_sum(a * a, 1)
        lj = tf.reduce_sum(((a - a_prev) / DT) ** 2, 1)
        a_prev = a
        Lv += lv * alive
        Lc += lc * alive
        La += la * alive
        Lj += lj * alive
        min_clear = tf.minimum(min_clear, tf.stop_gradient(cl))
        # an episode ends at the goal or in a collision (no loss accumulates after it)
        alive = alive * tf.cast(tf.stop_gradient(dist[:, 0]) > 0.5, tf.float32) * \
            tf.cast(tf.stop_gradient(cl) > 0.0, tf.float32)
    loss = tf.reduce_mean(W['v'] * Lv + W['c'] * Lc + W['a'] * La + W['j'] * Lj) / T
    metrics = {'loss': loss, 'Lv': tf.reduce_mean(Lv) / T, 'Lc': tf.reduce_mean(Lc) / T,
               'crash': tf.reduce_mean(tf.cast(min_clear < 0, tf.float32)),
               'x_end': tf.reduce_mean(p[:, 0])}
    return loss, metrics


def train(out, iters=2000, batch=64, T=150, lr=1e-3, v_range=(3.0, 12.0), seed=0, log=print, init=None):
    """AdamW, cosine-decayed learning rate (the paper: 1e-3, batch 64, 150 steps). Saves <out>/diffphys.keras."""
    tf = _tf()
    rng = np.random.default_rng(seed)
    policy = tf.keras.models.load_model(init, compile=False) if init else build_policy()
    sched = tf.keras.optimizers.schedules.CosineDecay(lr, iters)
    opt = tf.keras.optimizers.AdamW(sched, weight_decay=1e-4)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'diffphys.keras')
    for it in range(iters):
        cx, cy, rr = forest_batch(rng, batch)
        vd = rng.uniform(*v_range, batch)
        with tf.GradientTape() as tape:
            loss, m = rollout_loss(policy, tf.constant(cx), tf.constant(cy), tf.constant(rr), vd, T=T,
                                   seed=int(rng.integers(1 << 30)))
        g = tape.gradient(loss, policy.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 5.0)
        opt.apply_gradients(zip(g, policy.trainable_variables))
        if it % 10 == 0 or it == iters - 1:
            log(f"it {it:5d}  loss {m['loss']:.3f}  Lv {m['Lv']:.3f}  Lc {m['Lc']:.3f}  crash {m['crash']:.2f}  "
                f"x_end {m['x_end']:.1f}")
        if (it + 1) % 100 == 0 or it == iters - 1:
            policy.save(path)
    return path


# ── deployment in the closed loop of R18/R22 ──────────────────────────────
class DiffPhysPilot:
    """The trained policy inside `Rollout` (pilot 'diffphys'): its own SGM depth min-pooled to 16 × 12 (holes
    → far), velocity, thrust direction from the attitude and the goal direction; keeps the GRU state between
    frames. → desired world acceleration."""

    def __init__(self, path):
        tf = _tf()
        self.tf = tf
        self.model = tf.keras.models.load_model(path, compile=False)
        self.h = np.zeros((1, self.model.inputs[2].shape[-1]), np.float32)

    def reset(self):
        self.h[:] = 0

    @staticmethod
    def depth_input(depth):
        """SGM depth (m, 0 = none) → inverse depth (12, 16): nearest valid of each block, holes far."""
        d = np.asarray(depth, np.float32)
        H, Wd = d.shape
        # crop to 4:3 is not needed (the rig is 4:3); blocks of H/12 × W/16
        bh, bw = H // H_IMG, Wd // W_IMG
        B = d[:bh * H_IMG, :bw * W_IMG].reshape(H_IMG, bh, W_IMG, bw).transpose(0, 2, 1, 3).reshape(H_IMG, W_IMG, -1)
        B = np.where(B > 0, B, np.inf)
        z = np.clip(B.min(-1), 0.3, D_MAX)
        z[~np.isfinite(z)] = D_MAX
        return (1.0 / z).astype(np.float32)

    def __call__(self, depth, p, v, R, goal_dir, yaw, v_des):
        tf = self.tf
        img = self.depth_input(depth)[None, ..., None]
        thrust = np.asarray(R, float).reshape(3, 3)[:, 2]          # body z in the world
        st = state_input(tf.constant(np.asarray(v, np.float32)[None]), tf.constant(thrust[None].astype(np.float32)),
                         tf.constant((np.asarray(goal_dir) * v_des).astype(np.float32)[None]), tf.constant([float(yaw)]))
        out, h = self.model([tf.constant(img), st, tf.constant(self.h)])
        self.h = h.numpy()
        a = from_heading(out * A_MAX, tf.constant([float(yaw)])).numpy()[0]
        return a


def main(argv=None):
    """nanouav run r25 train … (evaluate with: nanouav run r18 eval --pilots diffphys --model <out>/diffphys.keras)."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('train', help='train the recurrent policy through the differentiable point-mass simulator')
    p.add_argument('--out', default='results/r25')
    p.add_argument('--iters', type=int, default=20000, help='the paper: 50 000 at batch 64')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--steps', type=int, default=150, help='steps of 1/15 s per episode (10 s)')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--v', default='3,12', help='range of desired speeds (m/s)')
    p.add_argument('--init', default='', help='continue from this model')
    p.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == 'train':
        lo, hi = (float(x) for x in a.v.split(','))
        print('→', train(a.out, iters=a.iters, batch=a.batch, T=a.steps, lr=a.lr, v_range=(lo, hi), seed=a.seed,
                         init=a.init or None, log=lambda s: print(s, flush=True)))


if __name__ == '__main__':
    main()
