"""R25 · Learning to fly through a forest by differentiable physics (Zhang, Hu, Song, Zou, Lin 2025,
"Learning vision-based agile flight via differentiable physics", Nature Machine Intelligence; arXiv
2407.10648), on the worlds of R18.

No expert and no labels: a small recurrent policy (CNN + GRU) sees a 16 × 12 inverse-depth image, its
velocity and attitude and the direction to the goal, and outputs an acceleration. It is trained by
back-propagating a loss through a differentiable point-mass simulator over 150 steps of 1/15 s:

* velocity tracking — smooth L1 of the gap between the desired velocity (toward the goal at v_des) and the
  velocity averaged over the last ~2 s (so the drone may swerve locally). There is no slowing down at the goal:
  an episode ends when the drone crosses the goal plane, as success is decided in R18/R22, and the goal is
  20–100 m away (0.26.6). With a fixed 40 m and a target velocity that fell to zero at the goal, the GRU learned
  to count metres and slowed down after ~30 m whatever it was asked (R25, 0.26.6);
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
def forest_batch(rng, B, density=(1 / 49, 1 / 16), r=(0.15, 0.4), length=60.0, width=30.0, clear=2.5, n_obs=None):
    """B Poisson forests like the training forests of R18 (spacing 4–7 m), start at the origin (1.5 m up),
    goal 40 m along +x. → cx, cy, rr (B, N_OBS) float32 (padding: far away, radius 0)."""
    N = n_obs or N_OBS
    cx = np.full((B, N), 1e4, np.float32)
    cy = np.full((B, N), 1e4, np.float32)
    rr = np.zeros((B, N), np.float32)
    for b in range(B):
        dens = rng.uniform(*density)
        n = rng.poisson(dens * length * width)
        xy = np.stack([rng.uniform(-5, length - 5, n), rng.uniform(-width / 2, width / 2, n)], 1)
        xy = xy[np.hypot(xy[:, 0], xy[:, 1]) > clear]
        if len(xy) > N:                                  # keep the ones nearest the line
            xy = xy[np.argsort(np.abs(xy[:, 1]))[:N]]
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


YAW_FWD = False                       # True: the yaw follows the velocity only toward the goal (fly with
                                      # --yaw-fwd). A 400-iteration CPU fine-tune with it did not help (R25, 0.26.9)
LOSS_V = 'vec'                        # velocity tracking: 'vec' = |v_set − v_avg| (the paper); 'along' = the gap in
W_LAT = 0.2                           # speed toward the goal plus W_LAT × the sideways speed, so that going round a
TAU_V = 2.0                           # trunk costs no more than stopping in front of it (R25, 0.27.0). TAU_V: s of v_avg
BURN = (0.0, 0.0)                     # s: before the 150 steps with gradient, fly this long (drawn per batch) without
                                      # gradient, so that the GRU learns to fly after 10 s too (0.27.0: r25b slowed down
                                      # after ~10 s at any speed — its training episodes lasted 10 s)
Z_RANGE = None                        # (lo, hi): start and goal heights drawn independently (0.27.0); None: 1.5 m
GOAL_X = (20.0, 100.0)                # m, goal distance of the training episodes (0.26.6; before: always 40)
FOREST_LENGTH = 115.0


def rollout_loss(policy, cx, cy, rr, v_des, T=150, z0=1.5, seed=0, goal_x=40.0, z_goal=None, init=None):
    """One batch of episodes through the differentiable simulator → (loss, metrics). goal_x: m along +x; z0,
    z_goal: start and goal heights (one value or one per episode; z_goal defaults to z0)."""
    tf = _tf()
    B = cx.shape[0]
    rng = np.random.default_rng(seed)
    gx = np.broadcast_to(np.asarray(goal_x, np.float32), (B,))
    zs = np.broadcast_to(np.asarray(z0, np.float32), (B,))
    zg = zs if z_goal is None else np.broadcast_to(np.asarray(z_goal, np.float32), (B,))
    goal = tf.constant(np.c_[gx, np.zeros(B), zg].astype(np.float32))
    vd = tf.constant(np.asarray(v_des, np.float32).reshape(B, 1))
    p = tf.constant(np.c_[np.zeros((B, 2)), zs].astype(np.float32))
    v = tf.constant(np.c_[np.asarray(v_des).reshape(B, 1) * rng.uniform(0.0, 1.0, (B, 1)),
                          np.zeros((B, 2))].astype(np.float32))
    if init is None:                                   # at rest in the GRU and the actuators
        hid = policy.inputs[2].shape[-1]
        init = {'v': v, 'a': tf.zeros((B, 3)), 'acd': tf.zeros((B, 3)), 'h': tf.zeros((B, hid)),
                'yaw': tf.zeros((B,)), 'v_avg': v, 'alive': tf.ones((B,))}
    loss, mt = _rollout(policy, cx, cy, rr, vd, goal, p, init['v'], T, init)
    return loss, {k: float(x) for k, x in mt.items()}


_COMPILED = {}


def _rollout(policy, cx, cy, rr, vd, goal, p, v, T, init):
    tf = _tf()
    key = (id(policy), T)
    if key not in _COMPILED:
        _COMPILED[key] = tf.function(lambda *a: _rollout_body(policy, *a, T=T))
    return _COMPILED[key](cx, cy, rr, vd, goal, p, v, init['a'], init['acd'], init['h'], init['yaw'], init['v_avg'],
                          init['alive'])


def _rollout_body(policy, cx, cy, rr, vd, goal, p, v, a, a_cmd_delayed, h, yaw, v_avg, alive, T):
    tf = _tf()
    B = tf.shape(cx)[0]
    a_prev = a
    gdec = math.exp(-GRAD_DECAY * DT)
    k_a = 1 - math.exp(-LAMBDA_A * DT)
    Lv = Lc = La = Lj = tf.zeros((B,))
    fwd = n_alive = tf.zeros((B,))
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
        turn = sp_xy > 0.5
        if YAW_FWD:                                    # backing away from the goal the camera keeps its yaw
            turn = turn & (tf.reduce_sum(tf.stop_gradient(v[:, :2]) * gdir[:, :2], 1) > 0)
        yaw = tf.where(turn, tf.atan2(v[:, 1], v[:, 0]), yaw)
        # losses (the target velocity does not fall at the goal: the episode ends on crossing its plane)
        v_set = tf.stop_gradient(gdir * vd)
        v_avg = v_avg + (v - v_avg) * (DT / TAU_V)
        if LOSS_V == 'along':
            g_ = tf.stop_gradient(gdir)
            v_al = tf.reduce_sum(v_avg * g_, 1)
            lat = tf.norm(v_avg - v_al[:, None] * g_, axis=1)
            e = tf.sqrt((vd[:, 0] - v_al) ** 2 + (W_LAT * lat) ** 2 + 1e-6)
        else:
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
        fwd = fwd + alive * tf.reduce_sum(tf.stop_gradient(v) * tf.stop_gradient(gdir), 1) / vd[:, 0]
        n_alive = n_alive + alive
        # an episode ends past the goal plane or in a collision (no loss accumulates after it)
        alive = alive * tf.cast(tf.stop_gradient(p[:, 0]) < goal[:, 0] - 0.5, tf.float32) * \
            tf.cast(tf.stop_gradient(cl) > 0.0, tf.float32)
    loss = tf.reduce_mean(W['v'] * Lv + W['c'] * Lc + W['a'] * La + W['j'] * Lj) / T
    metrics = {'loss': loss, 'Lv': tf.reduce_mean(Lv) / T, 'Lc': tf.reduce_mean(Lc) / T,
               'crash': tf.reduce_mean(tf.cast(min_clear < 0, tf.float32)),
               'x_end': tf.reduce_mean(p[:, 0]),
               'v_ratio': tf.reduce_sum(fwd) / tf.maximum(tf.reduce_sum(n_alive), 1.0)}   # speed toward the goal / v_des
    return loss, metrics


_BURN = {}


def burn_in(policy, vd, z0, steps, rng):
    """Fly the policy `steps` steps without gradient through a training forest toward a goal 10 km away, moving
    the drone 35 m back whenever it passes x = 40 m (the forest spans −5…55 m), with the training image noise.
    → the state it ends in (velocity, acceleration, delayed command, GRU state, yaw, averaged velocity, alive),
    to start a training episode from. 0 steps: at v_des·U(0, 1) along x with everything else at rest."""
    tf = _tf()
    B = len(vd)
    hid = policy.inputs[2].shape[-1]
    v0 = np.c_[vd * rng.uniform(0.0, 1.0, B), np.zeros((B, 2))].astype(np.float32)
    st = {'v': tf.constant(v0), 'a': tf.zeros((B, 3)), 'acd': tf.zeros((B, 3)), 'h': tf.zeros((B, hid)),
          'yaw': tf.zeros((B,)), 'v_avg': tf.constant(v0), 'alive': tf.ones((B,))}
    if steps <= 0:
        return st
    key = id(policy)
    if key not in _BURN:
        k_a = 1 - math.exp(-LAMBDA_A * DT)

        @tf.function
        def run(cx, cy, rr, vdt, p, v, a, acd, h, yaw, v_avg, alive, n):
            goal = tf.concat([tf.fill([tf.shape(p)[0], 1], 1e4), tf.zeros_like(p[:, :1]), p[:, 2:3]], 1)
            for _ in tf.range(n):
                thrust = a + tf.constant([[0.0, 0.0, G]])
                img = render_depth(p, camera_rot(yaw, thrust), cx, cy, rr)
                if AUG:
                    hole = tf.random.uniform(tf.shape(img)) < AUG_HOLES
                    img = tf.where(hole, 1.0 / D_MAX, img * (1.0 + 0.05 * tf.random.normal(tf.shape(img))))
                tg = goal - p
                gdir = tg / tf.norm(tg, axis=1, keepdims=True)
                o, h = policy([img, state_input(v, thrust, gdir * vdt, yaw), h])
                a = a + (acd - a) * k_a
                acd = from_heading(o * A_MAX, yaw)
                sp = tf.norm(v, axis=1, keepdims=True)
                v = v + (a - DRAG[0] * sp * v - DRAG[1] * v) * DT
                p = p + v * DT
                p = tf.concat([tf.where(p[:, :1] > 40.0, p[:, :1] - 35.0, p[:, :1]), p[:, 1:]], 1)
                turn = tf.norm(v[:, :2], axis=1) > 0.5
                if YAW_FWD:
                    turn = turn & (tf.reduce_sum(v[:, :2] * gdir[:, :2], 1) > 0)
                yaw = tf.where(turn, tf.atan2(v[:, 1], v[:, 0]), yaw)
                v_avg = v_avg + (v - v_avg) * (DT / TAU_V)
                d, _ = clearance(p, cx, cy, rr)
                alive = alive * tf.cast(d - R_Q > 0, tf.float32)
            return v, a, acd, h, yaw, v_avg, alive
        _BURN[key] = run
    cx, cy, rr = forest_batch(rng, B)
    p0 = np.c_[np.zeros((B, 2)), np.broadcast_to(np.asarray(z0, np.float32), (B,))].astype(np.float32)
    out = _BURN[key](tf.constant(cx), tf.constant(cy), tf.constant(rr), tf.constant(vd.reshape(B, 1).astype(np.float32)),
                     tf.constant(p0), st['v'], st['a'], st['acd'], st['h'], st['yaw'], st['v_avg'], st['alive'],
                     tf.constant(int(steps)))
    return dict(zip(('v', 'a', 'acd', 'h', 'yaw', 'v_avg', 'alive'), out))


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
        cx, cy, rr = forest_batch(rng, batch, length=FOREST_LENGTH)
        vd = rng.uniform(*v_range, batch)
        gx = rng.uniform(*GOAL_X, batch)
        zs, zg = (rng.uniform(*Z_RANGE, (2, batch)) if Z_RANGE else (1.5, None))
        st = None
        if BURN[1] > 0:                                # a flight already under way (no gradient)
            st = burn_in(policy, vd, zs, int(rng.uniform(*BURN) / DT), rng)
        with tf.GradientTape() as tape:
            loss, m = rollout_loss(policy, tf.constant(cx), tf.constant(cy), tf.constant(rr), vd, T=T,
                                   seed=int(rng.integers(1 << 30)), goal_x=gx, z0=zs, z_goal=zg, init=st)
        g = tape.gradient(loss, policy.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 5.0)
        opt.apply_gradients(zip(g, policy.trainable_variables))
        if it % 10 == 0 or it == iters - 1:
            log(f"it {it:5d}  loss {m['loss']:.3f}  Lv {m['Lv']:.3f}  Lc {m['Lc']:.3f}  crash {m['crash']:.2f}  "
                f"x_end {m['x_end']:.1f}  v/v_des {m['v_ratio']:.2f}")
        if (it + 1) % 100 == 0 or it == iters - 1:
            policy.save(path)
    return path


# ── speed profile in its own simulator (`r25 profile`) ────────────────────
def speed_profile(path, speeds=(5.0, 7.0, 10.0, 12.0), B=64, T=150, seed=0, log=print, yaw_fwd=False):
    """Fly the policy (no gradient, ideal depth) through long forests toward a goal 1 km away, from rest and from
    v_des, and print the median forward speed of each second. A policy that slows down after a fixed distance
    learned the goal distance of its training (R25, 0.26.6). → {(v_des, start): [speed per second]}."""
    tf = _tf()
    pol = tf.keras.models.load_model(path, compile=False)
    hidden = pol.inputs[2].shape[-1]

    @tf.function
    def fly(cx, cy, rr, vd, v):
        Bn = tf.shape(cx)[0]
        p = tf.concat([tf.zeros((Bn, 2)), tf.fill([Bn, 1], 1.5)], 1)
        goal = tf.concat([tf.fill([Bn, 1], 1000.0), tf.zeros((Bn, 1)), tf.fill([Bn, 1], 1.5)], 1)
        a = acd = tf.zeros((Bn, 3))
        h = tf.zeros((Bn, hidden))
        yaw = tf.zeros((Bn,))
        alive = tf.ones((Bn,))
        k_a = 1 - math.exp(-LAMBDA_A * DT)
        out_v = tf.TensorArray(tf.float32, T)
        for k in tf.range(T):
            thrust = a + tf.constant([[0.0, 0.0, G]])
            img = render_depth(p, camera_rot(yaw, thrust), cx, cy, rr)
            to_goal = goal - p
            gdir = to_goal / tf.norm(to_goal, axis=1, keepdims=True)
            o, h = pol([img, state_input(v, thrust, gdir * vd, yaw), h])
            a = a + (acd - a) * k_a
            acd = from_heading(o * A_MAX, yaw)
            sp = tf.norm(v, axis=1, keepdims=True)
            v = v + (a - DRAG[0] * sp * v - DRAG[1] * v) * DT
            p = p + v * DT
            turn = tf.norm(v[:, :2], axis=1) > 0.5
            if yaw_fwd:
                turn = turn & (tf.reduce_sum(v[:, :2] * gdir[:, :2], 1) > 0)
            yaw = tf.where(turn, tf.atan2(v[:, 1], v[:, 0]), yaw)
            d, _ = clearance(p, cx, cy, rr)
            alive = alive * tf.cast(d - R_Q > 0, tf.float32)
            out_v = out_v.write(k, tf.where(alive > 0, v[:, 0], np.nan))
        return out_v.stack()

    rng = np.random.default_rng(seed)
    res = {}
    for vd in speeds:
        cx, cy, rr = forest_batch(rng, B, length=T * DT * max(speeds) * 1.2)
        for start in ('rest', 'v_des'):
            v0 = np.zeros((B, 3), np.float32)
            if start == 'v_des':
                v0[:, 0] = vd
            V = fly(tf.constant(cx), tf.constant(cy), tf.constant(rr), tf.constant(np.full((B, 1), vd, np.float32)),
                    tf.constant(v0)).numpy()
            per_s = [float(np.nanmedian(V[i:i + 15])) for i in range(0, T, 15)]
            res[(vd, start)] = per_s
            log(f'v_des {vd:4.1f} from {start:5s}: ' + ' '.join(f'{x:4.1f}' for x in per_s) +
                f'   crashed {np.mean(np.isnan(V[-1])):.2f}')
    return res


# ── stalls in its own simulator (`r25 stalls`) ────────────────────────────
def stall_eval(path, speeds=(2.0, 3.0, 5.0), B=256, seed=0, goal_x=40.0, z_start=(1.5, 1.5), z_goal=(1.5, 1.5),
               yaw_fwd=False, log=print, traces=None):
    """Fly the policy (no gradient, ideal depth) from v_des through B training-like forests to a goal plane
    goal_x away, for goal_x / v_des + 6 s (the time limit of R18). Per speed: arrived, crashed, stalled (≥ 1.5 s
    below 0.3·v_des while flying), median arrival time and the height error when it crosses the goal plane.
    Start and goal heights are drawn from z_start, z_goal. → {v_des: metrics}."""
    tf = _tf()
    pol = tf.keras.models.load_model(path, compile=False)
    hidden = pol.inputs[2].shape[-1]
    k_a = 1 - math.exp(-LAMBDA_A * DT)
    compiled = {}

    def make(T):
        @tf.function
        def fly(cx, cy, rr, vd, p, goal):
            Bn = tf.shape(cx)[0]
            v = tf.concat([vd, tf.zeros((Bn, 2))], 1)
            a = acd = tf.zeros((Bn, 3))
            h = tf.zeros((Bn, hidden))
            yaw = tf.zeros((Bn,))
            alive = tf.ones((Bn,))
            P = tf.TensorArray(tf.float32, T)
            S = tf.TensorArray(tf.float32, T)
            A = tf.TensorArray(tf.float32, T)
            C = tf.TensorArray(tf.float32, T)
            for k in tf.range(T):
                thrust = a + tf.constant([[0.0, 0.0, G]])
                img = render_depth(p, camera_rot(yaw, thrust), cx, cy, rr)
                tg = goal - p
                gdir = tg / tf.norm(tg, axis=1, keepdims=True)
                o, h = pol([img, state_input(v, thrust, gdir * vd, yaw), h])
                a = a + (acd - a) * k_a
                acd = from_heading(o * A_MAX, yaw)
                sp = tf.norm(v, axis=1, keepdims=True)
                v = v + (a - DRAG[0] * sp * v - DRAG[1] * v) * DT
                p = p + v * DT
                turn = tf.norm(v[:, :2], axis=1) > 0.5
                if yaw_fwd:
                    turn = turn & (tf.reduce_sum(v[:, :2] * gdir[:, :2], 1) > 0)
                yaw = tf.where(turn, tf.atan2(v[:, 1], v[:, 0]), yaw)
                d, _ = clearance(p, cx, cy, rr)
                crashed = tf.cast(d - R_Q <= 0, tf.float32)
                C = C.write(k, alive * crashed)
                alive = alive * (1 - crashed) * tf.cast(p[:, 0] < goal[:, 0] - 0.5, tf.float32)
                P = P.write(k, p)
                S = S.write(k, tf.norm(v, axis=1))
                A = A.write(k, alive)
            return P.stack(), S.stack(), A.stack(), C.stack()
        return fly

    rng = np.random.default_rng(seed)
    res = {}
    for vd in speeds:
        T = int(math.ceil((goal_x / vd + 6.0) / DT))
        if T not in compiled:
            compiled[T] = make(T)
        cx, cy, rr = forest_batch(rng, B)
        zs = rng.uniform(*z_start, B).astype(np.float32)
        zg = rng.uniform(*z_goal, B).astype(np.float32)
        p0 = np.c_[np.zeros((B, 2)), zs].astype(np.float32)
        goal = np.c_[np.full(B, goal_x), np.zeros(B), zg].astype(np.float32)
        P, S, A, C = [x.numpy() for x in compiled[T](tf.constant(cx), tf.constant(cy), tf.constant(rr),
                                                     tf.constant(np.full((B, 1), vd, np.float32)),
                                                     tf.constant(p0), tf.constant(goal))]
        if traces is not None:
            traces[vd] = {'P': P, 'S': S, 'A': A, 'C': C}
        crashed = C.max(0) > 0
        crossed = (P[:, :, 0] >= goal_x - 0.5).any(0)
        arrived = crossed & ~crashed
        k_arr = np.argmax(P[:, :, 0] >= goal_x - 0.5, 0)
        slow = (S < 0.3 * vd) & (A > 0)
        stalled = np.zeros(B, bool)
        for b in range(B):
            run = best = 0
            for x in slow[:, b]:
                run = run + 1 if x else 0
                best = max(best, run)
            stalled[b] = best * DT >= 1.5
        dz = np.abs(P[k_arr, np.arange(B), 2] - zg)[arrived]
        m = {'arrived': float(arrived.mean()), 'crashed': float(crashed.mean()), 'stalled': float(stalled.mean()),
             't_arrive': float(np.median(k_arr[arrived] * DT)) if arrived.any() else float('nan'),
             'dz_goal': float(np.median(dz)) if dz.size else float('nan')}
        res[vd] = m
        log(f"v_des {vd:4.1f}: arrived {m['arrived']:.3f}  crashed {m['crashed']:.3f}  stalled {m['stalled']:.3f}  "
            f"t_arrive {m['t_arrive']:.1f} s  |dz| at the goal {m['dz_goal']:.2f} m   (B {B})")
    return res


# ── deployment in the closed loop of R18/R22 ──────────────────────────────
class DiffPhysPilot:
    """The trained policy inside `Rollout` (pilot 'diffphys'): its own SGM depth min-pooled to 16 × 12 (holes
    → far), velocity, thrust direction from the attitude and the goal direction; keeps the GRU state between
    frames. → desired world acceleration."""

    _MODELS = {}                                      # loaded once per process (eval workers fly many episodes)

    def __init__(self, path):
        tf = _tf()
        self.tf = tf
        if path not in self._MODELS:
            n = int(os.environ.get('OMP_NUM_THREADS') or 0)
            if n:                                     # one core per eval worker: no thread oversubscription
                try:
                    tf.config.threading.set_intra_op_parallelism_threads(n)
                    tf.config.threading.set_inter_op_parallelism_threads(n)
                except RuntimeError:                  # TensorFlow already running in this process
                    pass
            self._MODELS.clear()
            self._MODELS[path] = tf.keras.models.load_model(path, compile=False)
        self.model = self._MODELS[path]
        self.h = np.zeros((1, self.model.inputs[2].shape[-1]), np.float32)

    def reset(self):
        self.h[:] = 0
        self._prev = None

    q = 0.0          # quantile of each block's valid depths (0: the nearest, as in training)
    edge = False     # drop SGM pixels on depth discontinuities first (rollout.edge_filter)
    persist = False  # a block counts as near only if it was near in the previous frame too (SGM flicker)

    @staticmethod
    def depth_input(depth, q=0.0, edge=False):
        """SGM depth (m, 0 = none) → inverse depth (12, 16): the q-quantile (0: nearest) of the valid depths of
        each block, holes far. With q = 0 one stray near pixel of SGM makes its whole block near (R25, 0.26.6)."""
        d = np.asarray(depth, np.float32)
        if edge:
            from nanouav.agile.rollout import edge_filter
            d = edge_filter(d)
        H, Wd = d.shape
        # crop to 4:3 is not needed (the rig is 4:3); blocks of H/12 × W/16
        bh, bw = H // H_IMG, Wd // W_IMG
        B = d[:bh * H_IMG, :bw * W_IMG].reshape(H_IMG, bh, W_IMG, bw).transpose(0, 2, 1, 3).reshape(H_IMG, W_IMG, -1)
        if q > 0:
            S = np.sort(np.where(B > 0, B, np.inf), axis=-1)
            n = (B > 0).sum(-1)
            k = np.clip(np.floor(q * n).astype(int), 0, B.shape[-1] - 1)
            z = np.take_along_axis(S, k[..., None], -1)[..., 0]
        else:
            z = np.where(B > 0, B, np.inf).min(-1)
        z = np.clip(z, 0.3, D_MAX)
        z[~np.isfinite(z)] = D_MAX
        return (1.0 / z).astype(np.float32)

    def __call__(self, depth, p, v, R, goal_dir, yaw, v_des):
        tf = self.tf
        x = self.depth_input(depth, self.q, self.edge)
        if self.persist:
            prev = getattr(self, '_prev', None)
            self._prev = x
            if prev is not None:
                x = np.minimum(x, prev)               # inverse depth: the farther of the two frames
        self._last_x = x
        img = x[None, ..., None]
        thrust = np.asarray(R, float).reshape(3, 3)[:, 2]          # body z in the world
        st = state_input(tf.constant(np.asarray(v, np.float32)[None]), tf.constant(thrust[None].astype(np.float32)),
                         tf.constant((np.asarray(goal_dir) * v_des).astype(np.float32)[None]), tf.constant([float(yaw)]))
        out, h = self.model([tf.constant(img), st, tf.constant(self.h)])
        self.h = h.numpy()
        a = from_heading(out * A_MAX, tf.constant([float(yaw)])).numpy()[0]
        return a


def bench(path, size=(320, 240), n=300, edge=True, log=print):
    """Time of one decision on one CPU thread (0.27.7, for the comparison with YOPOv2-Tracker): the depth
    preprocessing of the pilot (SGM image of `size` → 16 × 12 inverse depth, with the edge filter), the network
    as TFLite float32, and the pilot as flown in the evaluations (eager Keras). Also checks that TFLite gives
    the Keras outputs. → dict of ms (and the largest output difference)."""
    import os
    import time
    os.environ['OMP_NUM_THREADS'] = '1'
    tf = _tf()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:                       # TF already running (e.g. in the tests): the Keras time may use more
        pass
    rng = np.random.default_rng(0)
    pilot = DiffPhysPilot(path)
    pilot.edge = edge
    depth = rng.uniform(1.0, 20.0, (size[1], size[0])).astype(np.float32)
    args = (np.zeros(3), np.array([3.0, 0, 0]), np.eye(3), np.array([1.0, 0, 0]), 0.0, 5.0)

    def per_call(f, warm=20):
        for _ in range(warm):
            f()
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        return (time.perf_counter() - t0) / n * 1000

    m = pilot.model
    itp = tf.lite.Interpreter(model_content=tf.lite.TFLiteConverter.from_keras_model(m).convert(), num_threads=1)
    itp.allocate_tensors()
    ins = itp.get_input_details()
    xs = [rng.uniform(0, 1, [1] + list(t.shape[1:])).astype(np.float32) for t in m.inputs]
    feed = [(d['index'], xs[next(k for k, t in enumerate(m.inputs) if list(d['shape'][1:]) == list(t.shape[1:]))])
            for d in ins]

    def lite():
        for i, x in feed:
            itp.set_tensor(i, x)
        itp.invoke()
    out = {'preprocess_ms': per_call(lambda: DiffPhysPilot.depth_input(depth, 0.0, edge)),
           'network_tflite_ms': per_call(lite), 'pilot_keras_ms': per_call(lambda: pilot(depth, *args))}
    lite()
    ref = [np.asarray(o) for o in m(xs)]
    out['tflite_max_diff'] = max(min(float(np.abs(itp.get_tensor(d['index']) - r).max()) for r in ref
                                     if r.shape == tuple(d['shape'])) for d in itp.get_output_details())
    out['params'] = int(m.count_params())
    log(f"one decision on one CPU thread ({size[0]}×{size[1]} depth): preprocessing {out['preprocess_ms']:.2f} ms + "
        f"network (TFLite) {out['network_tflite_ms']:.3f} ms; as flown in the evaluations (eager Keras) "
        f"{out['pilot_keras_ms']:.1f} ms; {out['params']} parameters; TFLite = Keras to {out['tflite_max_diff']:.1e}")
    return out


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
    p.add_argument('--yaw-fwd', action='store_true', help='train with the yaw rule of --yaw-fwd (see YAW_FWD)')
    p.add_argument('--loss-v', default='vec', choices=['vec', 'along'], help='velocity tracking term (see LOSS_V)')
    p.add_argument('--w-lat', type=float, default=0.2, help="weight of the sideways speed in --loss-v along")
    p.add_argument('--tau-v', type=float, default=2.0, help='s of the averaged velocity that is tracked')
    p.add_argument('--z', default='', help='range of start and goal heights, e.g. 1,2.5 (default: always 1.5 m)')
    p.add_argument('--burn', default='', help='s of flight without gradient before each episode, e.g. 0,30 (see BURN)')
    q = sub.add_parser('profile', help='forward speed per second in its own simulator, 1 km goal (does it hold v_des?)')
    q.add_argument('--model', default='results/r25/diffphys.keras')
    q.add_argument('--v', default='5,7,10,12')
    q.add_argument('--yaw-fwd', action='store_true', help='the yaw rule of models trained from 0.26.9')
    q = sub.add_parser('stalls', help='arrivals, crashes and stalls (≥ 1.5 s nearly stopped) in its own simulator')
    q.add_argument('--model', default='results/r25/diffphys.keras')
    q.add_argument('--v', default='2,3,5')
    q.add_argument('--n', type=int, default=256, help='forests per speed')
    q.add_argument('--z', default='1.5,1.5', help='range of start and goal heights (m)')
    q.add_argument('--yaw-fwd', action='store_true')
    q.add_argument('--seed', type=int, default=0)
    q = sub.add_parser('bench', help='time of one decision on one CPU thread (preprocessing + TFLite network)')
    q.add_argument('--model', default='results/r25/diffphys.keras')
    q.add_argument('--size', default='320x240', help='SGM depth image, W×H (R18/R20: 320x240)')
    q.add_argument('--n', type=int, default=300)
    a = ap.parse_args(argv)
    if a.cmd == 'bench':
        bench(a.model, tuple(int(x) for x in a.size.split('x')), n=a.n)
        return
    if a.cmd == 'stalls':
        z = tuple(float(x) for x in a.z.split(','))
        stall_eval(a.model, [float(x) for x in a.v.split(',')], B=a.n, seed=a.seed, z_start=z, z_goal=z,
                   yaw_fwd=a.yaw_fwd)
        return
    if a.cmd == 'profile':
        speed_profile(a.model, [float(x) for x in a.v.split(',')], yaw_fwd=a.yaw_fwd)
        return
    if a.cmd == 'train':
        global YAW_FWD, LOSS_V, W_LAT, TAU_V
        global Z_RANGE
        YAW_FWD, LOSS_V, W_LAT, TAU_V = a.yaw_fwd, a.loss_v, a.w_lat, a.tau_v
        Z_RANGE = tuple(float(x) for x in a.z.split(',')) if a.z else None
        global BURN
        BURN = tuple(float(x) for x in a.burn.split(',')) if a.burn else (0.0, 0.0)
        lo, hi = (float(x) for x in a.v.split(','))
        print('→', train(a.out, iters=a.iters, batch=a.batch, T=a.steps, lr=a.lr, v_range=(lo, hi), seed=a.seed,
                         init=a.init or None, log=lambda s: print(s, flush=True)))


if __name__ == '__main__':
    main()
