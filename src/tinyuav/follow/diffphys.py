"""Follow policy: the navigation policy fine-tuned through the same differentiable simulator.

The target moves at 1-7 m/s along a smooth random path through a Poisson forest. The desired velocity comes
from the follow law of `tinyuav.follow.mission` (toward the point `standoff` m short of the target, at
|v_target| + k (distance - standoff), capped) and the camera faces the target, so the trained policy flies
the balloon mission unchanged. A softplus barrier on the line-of-sight clearance (distance from each trunk
axis to the drone-target segment, minus its radius) teaches it to keep the target in view; it is analytic
and costs nothing in flight.
"""
import math
import os

import numpy as np

from tinyuav.nav import diffphys as DP

STANDOFF, K_DIST, V_MAX = 4.0, 0.8, 8.0     # follow law of BalloonMission
Z_TARGET = (1.0, 3.0)                       # m, target height
W_VIS = 1.0                                 # line-of-sight barrier weight
VIS_MARGIN = 0.3                            # m, wanted line-of-sight clearance
BETA_VIS = 8.0
TAU_V = 1.0                                 # s, averaging time of the tracked velocity
CORRIDOR = 0.5                              # m free of trunks beside the target path
BURN = (0.0, 30.0)                          # s of flight without gradient before each episode


TURN_RATE = 60.0                            # deg/s, largest target turn rate


def target_paths(rng, B, T, v_range=(1.0, 4.0), z_range=Z_TARGET, x0=(4.0, 6.0)):
    """B target paths of T steps: heading random walk (<= TURN_RATE), speed ramp at 1 m/s^2 to v in v_range,
    slow height drift in z_range: Q, V (T, B, 3).
    """
    dt = DP.DT
    Q = np.zeros((T, B, 3), np.float32)
    V = np.zeros((T, B, 3), np.float32)
    vt = rng.uniform(*v_range, B)
    psi = rng.uniform(-0.3, 0.3, B)
    dpsi = np.zeros(B)
    z = rng.uniform(*z_range, B)
    vz_amp = rng.uniform(0.0, 0.3, B)
    vz_w = rng.uniform(0.2, 0.6, B)
    q = np.c_[rng.uniform(*x0, B), rng.uniform(-1.0, 1.0, B), z]
    for k in range(T):
        sp = np.minimum(vt, 1.0 * k * dt)
        dpsi = np.clip(dpsi + rng.normal(0, 0.2, B), -math.radians(TURN_RATE), math.radians(TURN_RATE))
        psi = psi + dpsi * dt - 0.3 * dt * (psi + np.arctan(q[:, 1] / 8.0))    # drift back toward y = 0
        psi = np.clip(psi, -1.2, 1.2)                         # broadly forward
        vz = vz_amp * np.sin(vz_w * k * dt * 2 * math.pi)
        v = np.c_[sp * np.cos(psi), sp * np.sin(psi), vz]
        q = q + v * dt
        q[:, 2] = np.clip(q[:, 2], *z_range)
        Q[k], V[k] = q, v
    return Q, V


def clear_corridor(cx, cy, rr, Q, margin=1.0):
    """Remove trunks within `margin` of the target paths."""
    cx, cy, rr = cx.copy(), cy.copy(), rr.copy()
    P = Q[::3, :, :2]
    for b in range(cx.shape[0]):
        d = np.hypot(cx[b][:, None] - P[:, b, 0][None], cy[b][:, None] - P[:, b, 1][None]).min(1)
        bad = d < rr[b] + margin
        cx[b, bad], cy[b, bad], rr[b, bad] = 1e4, 1e4, 0.0
    return cx, cy, rr


def sight_clearance(p, q, cx, cy, rr):
    """Line-of-sight clearance (B,): min over trunks of the horizontal distance from the axis to p-q, minus radius."""
    tf = DP._tf()
    px, py = p[:, 0:1], p[:, 1:2]
    dx, dy = q[:, 0:1] - px, q[:, 1:2] - py
    L2 = dx * dx + dy * dy + 1e-6
    t = tf.clip_by_value(((cx - px) * dx + (cy - py) * dy) / L2, 0.0, 1.0)
    ex, ey = px + t * dx - cx, py + t * dy - cy
    return tf.reduce_min(tf.sqrt(ex * ex + ey * ey + 1e-9) - rr, axis=1)


def follow_command(p, q, vq, tf):
    """Follow law: (unit direction, desired speed (B, 1), distance)."""
    hx = p[:, :2] - q[:, :2]
    dist = tf.norm(hx, axis=1, keepdims=True)
    u = hx / tf.maximum(dist, 1e-6)
    zf = tf.clip_by_value(q[:, 2:3], Z_TARGET[0], Z_TARGET[1])
    f = tf.concat([q[:, :2] + u * STANDOFF, zf], 1)
    e = f - p
    omega = e / tf.maximum(tf.norm(e, axis=1, keepdims=True), 1e-3)
    vt = tf.norm(vq[:, :2], axis=1, keepdims=True)
    vdes = tf.clip_by_value(vt + K_DIST * (dist - STANDOFF), 0.3, V_MAX)
    return omega, vdes, dist


def _step(policy, cx, cy, rr, q, vq, p, v, a, acd, h, aug, k_a, tf):
    """One control step of the follow loop."""
    tq = tf.stop_gradient(q - p)
    yaw = tf.atan2(tq[:, 1], tq[:, 0])                         # the camera faces the target
    thrust = a + tf.constant([[0.0, 0.0, DP.G]])
    img = DP.render_depth(tf.stop_gradient(p), DP.camera_rot(yaw, tf.stop_gradient(thrust)), cx, cy, rr)
    if aug:
        hole = tf.random.uniform(tf.shape(img)) < DP.AUG_HOLES
        img = tf.where(hole, 1.0 / DP.D_MAX, img * (1.0 + 0.05 * tf.random.normal(tf.shape(img))))
    omega, vdes, dist = follow_command(tf.stop_gradient(p), q, vq, tf)
    v_set = tf.stop_gradient(omega * vdes)
    out, h = policy([img, DP.state_input(v, thrust, v_set, yaw), h])
    a = a + (acd - a) * k_a
    acd = DP.from_heading(out * DP.A_MAX, yaw)
    return a, acd, h, v_set, dist


def _follow_burn(policy, cx, cy, rr, Q, Vq, p, n):
    """Follow n steps without gradient; returns the state to train from."""
    tf = DP._tf()
    B = tf.shape(cx)[0]
    v = a = acd = tf.zeros((B, 3))
    h = tf.zeros((B, policy.inputs[2].shape[-1]))
    v_avg = v
    alive = tf.ones((B,))
    k_a = 1 - math.exp(-DP.LAMBDA_A * DP.DT)
    for k in tf.range(n):
        q, vq = tf.gather(Q, k), tf.gather(Vq, k)
        a, acd, h, _, _ = _step(policy, cx, cy, rr, q, vq, p, v, a, acd, h, DP.AUG, k_a, tf)
        sp = tf.norm(v, axis=1, keepdims=True)
        v = v + (a - DP.DRAG[0] * sp * v - DP.DRAG[1] * v) * DP.DT
        p = p + v * DP.DT
        v_avg = v_avg + (v - v_avg) * (DP.DT / TAU_V)
        d, _ = DP.clearance(p, cx, cy, rr)
        alive = alive * tf.cast(d - DP.R_Q > 0, tf.float32)
    return p, v, a, acd, h, v_avg, alive


def _follow_body(policy, cx, cy, rr, Q, Vq, p, v, a, a_cmd_delayed, h, v_avg, alive, T):
    tf = DP._tf()
    B = tf.shape(cx)[0]
    a_prev = a
    gdec = math.exp(-DP.GRAD_DECAY * DP.DT)
    k_a = 1 - math.exp(-DP.LAMBDA_A * DP.DT)
    Lv = Lc = La = Lj = Ls = tf.zeros((B,))
    vis = derr = n_alive = tf.zeros((B,))
    min_clear = tf.fill([B], 99.0)
    for k in range(T):
        q, vq = Q[k], Vq[k]
        a, a_cmd_delayed, h, v_set, dist = _step(policy, cx, cy, rr, q, vq, p, v, a, a_cmd_delayed, h, DP.AUG, k_a, tf)
        sp = tf.norm(v, axis=1, keepdims=True)
        v = DP._decay(v + (a - DP.DRAG[0] * sp * v - DP.DRAG[1] * v) * DP.DT, gdec)
        p = DP._decay(p + v * DP.DT, gdec)
        # losses
        v_avg = v_avg + (v - v_avg) * (DP.DT / TAU_V)
        e = tf.norm(v_set - v_avg, axis=1)
        lv = tf.where(e < 1.0, 0.5 * e * e, e - 0.5)
        d, n = DP.clearance(p, cx, cy, rr)
        cl = d - DP.R_Q
        approach = tf.nn.relu(-tf.reduce_sum(v * tf.stop_gradient(n), axis=1))
        clb = tf.maximum(cl, -0.3)
        lc = approach * tf.nn.relu(1.0 - clb) ** 2 + DP.BETA[0] * tf.nn.softplus(-DP.BETA[1] * clb)
        sc = sight_clearance(p, q, cx, cy, rr)
        ls = tf.nn.softplus(BETA_VIS * (VIS_MARGIN - tf.maximum(sc, -1.0))) / BETA_VIS
        la = tf.reduce_sum(a * a, 1)
        lj = tf.reduce_sum(((a - a_prev) / DP.DT) ** 2, 1)
        a_prev = a
        Lv += lv * alive
        Lc += lc * alive
        Ls += ls * alive
        La += la * alive
        Lj += lj * alive
        min_clear = tf.minimum(min_clear, tf.stop_gradient(cl))
        vis += alive * tf.cast(tf.stop_gradient(sc) > 0, tf.float32)
        derr += alive * tf.abs(tf.stop_gradient(dist[:, 0]) - STANDOFF)
        n_alive += alive
        alive = alive * tf.cast(tf.stop_gradient(cl) > 0.0, tf.float32)
    W = DP.W
    loss = tf.reduce_mean(W['v'] * Lv + W['c'] * Lc + W_VIS * Ls + W['a'] * La + W['j'] * Lj) / T
    na = tf.maximum(tf.reduce_sum(n_alive), 1.0)
    return loss, {'loss': loss, 'Lv': tf.reduce_mean(Lv) / T, 'Lc': tf.reduce_mean(Lc) / T, 'Ls': tf.reduce_mean(Ls) / T,
                  'crash': tf.reduce_mean(tf.cast(min_clear < 0, tf.float32)),
                  'visible': tf.reduce_sum(vis) / na, 'dist_err': tf.reduce_sum(derr) / na}


_COMPILED = {}


def follow_batch(rng, B, T, v_range=(1.0, 4.0), density=None):
    """Forests with a clear corridor along each target path, the paths, and the drone start 1 m around the
    standoff behind the target, at rest. density: trees per m^2 (default 1/49-1/16).
    """
    Q, V = target_paths(rng, B, T, v_range)
    L = float(Q[-1, :, 0].max()) + 20.0                          # forest covers the whole path
    L_max = 6.0 + v_range[1] * T * DP.DT + 20.0                   # fixed size per T: no retracing
    kw = {} if density is None else {'density': (density, density)}
    cx, cy, rr = DP.forest_batch(rng, B, length=L, width=40.0,
                                 n_obs=int(np.clip(DP.N_OBS * L_max / 60.0 * 40 / 30, DP.N_OBS, 1536)), **kw)
    cx, cy, rr = clear_corridor(cx, cy, rr, Q, margin=CORRIDOR)
    p0 = np.c_[Q[0, :, 0] - rng.uniform(STANDOFF - 1, STANDOFF + 1, B), Q[0, :, 1],
               np.clip(Q[0, :, 2], *Z_TARGET)].astype(np.float32)
    cx, cy, rr = clear_corridor(cx, cy, rr, p0[None], margin=1.5)    # nothing on the drone at the start
    return cx, cy, rr, Q, V, p0


def follow_loss(policy, cx, cy, rr, Q, V, p0, T, burn=0):
    """`burn` steps without gradient, then T steps with gradient."""
    tf = DP._tf()
    key = (id(policy), T, W_VIS)
    if key not in _COMPILED:
        _COMPILED[key] = tf.function(lambda *a: _follow_body(policy, *a, T=T))
        _COMPILED[(key, 'burn')] = tf.function(lambda *a: _follow_burn(policy, *a))
    B = cx.shape[0]
    cx, cy, rr = tf.constant(cx), tf.constant(cy), tf.constant(rr)
    Q, V = tf.constant(Q), tf.constant(V)
    if burn > 0:
        p, v, a, acd, h, v_avg, alive = _COMPILED[(key, 'burn')](cx, cy, rr, Q, V, tf.constant(p0), tf.constant(burn))
    else:
        p, v, a, acd = tf.constant(p0), tf.zeros((B, 3)), tf.zeros((B, 3)), tf.zeros((B, 3))
        h, v_avg, alive = tf.zeros((B, policy.inputs[2].shape[-1])), tf.zeros((B, 3)), tf.ones((B,))
    # gradient part: only trunks near the reachable area
    cxs, cys, rrs = near_trunks(cx.numpy(), cy.numpy(), rr.numpy(), Q.numpy()[burn:burn + T], p.numpy())
    loss, mt = _COMPILED[key](tf.constant(cxs), tf.constant(cys), tf.constant(rrs), Q[burn:burn + T], V[burn:burn + T],
                              p, v, a, acd, h, v_avg, alive)
    return loss, {k: float(x) for k, x in mt.items()}


def near_trunks(cx, cy, rr, Qseg, p, n=DP.N_OBS, reach=15.0):
    """The n trunks nearest the target path segment and the drone, within `reach` m."""
    B = cx.shape[0]
    out = [np.full((B, n), 1e4, np.float32), np.full((B, n), 1e4, np.float32), np.zeros((B, n), np.float32)]
    pts = np.concatenate([Qseg[::5, :, :2], p[None, :, :2]], 0)
    for b in range(B):
        live = np.flatnonzero(rr[b] > 0)
        d = np.hypot(cx[b, live][:, None] - pts[:, b, 0][None], cy[b, live][:, None] - pts[:, b, 1][None]).min(1)
        keep = live[np.argsort(d)[:n]]
        keep = keep[np.sort(d)[:n] < reach]
        out[0][b, :len(keep)], out[1][b, :len(keep)], out[2][b, :len(keep)] = cx[b, keep], cy[b, keep], rr[b, keep]
    return out


def train(out, init, iters=1000, batch=32, T=150, lr=3e-4, v_range=(1.0, 4.0), seed=0, log=print,
          name='diffphys_follow.keras'):
    """Fine-tune a navigation policy to follow (AdamW, cosine rate); returns <out>/<name>."""
    tf = DP._tf()
    rng = np.random.default_rng(seed)
    policy = tf.keras.models.load_model(init, compile=False)
    opt = tf.keras.optimizers.AdamW(tf.keras.optimizers.schedules.CosineDecay(lr, iters), weight_decay=1e-4)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    k_max = int(BURN[1] / DP.DT)
    for it in range(iters):
        burn = int(rng.uniform(*BURN) / DP.DT)
        cx, cy, rr, Q, V, p0 = follow_batch(rng, batch, k_max + T, v_range)   # fixed shapes; [burn, burn + T) trains
        with tf.GradientTape() as tape:
            loss, m = follow_loss(policy, cx, cy, rr, Q, V, p0, T, burn=burn)
        g = tape.gradient(loss, policy.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 5.0)
        opt.apply_gradients(zip(g, policy.trainable_variables))
        if it % 10 == 0 or it == iters - 1:
            log(f"it {it:5d}  loss {m['loss']:.3f}  Lv {m['Lv']:.3f}  Lc {m['Lc']:.3f}  Ls {m['Ls']:.3f}  "
                f"crash {m['crash']:.2f}  visible {m['visible']:.2f}  dist_err {m['dist_err']:.2f}")
        if (it + 1) % 100 == 0 or it == iters - 1:
            policy.save(path)
    return path


def evaluate(path, B=128, T=225, v_range=(1.0, 4.0), seed=123, log=print, density=None):
    """Crash rate, visible fraction, distance and height errors, and time beyond 2 x standoff; ideal depth."""
    tf = DP._tf()
    policy = tf.keras.models.load_model(path, compile=False)
    rng = np.random.default_rng(seed)
    cx, cy, rr, Q, V, p0 = follow_batch(rng, B, T, v_range, density=density)
    aug, DP.AUG = DP.AUG, False
    try:
        body = tf.function(lambda *a: _follow_metrics(policy, *a, T=T))
        m = body(tf.constant(cx), tf.constant(cy), tf.constant(rr), tf.constant(Q), tf.constant(V), tf.constant(p0))
    finally:
        DP.AUG = aug
    m = {k: float(x) for k, x in m.items()}
    log(f"crashed {m['crashed']:.3f}  visible {m['visible']:.3f}  dist_err {m['dist_err']:.2f} m  "
        f"z_err {m['z_err']:.2f} m  far {m['far']:.3f}   (B {B}, {T * DP.DT:.0f} s, target {v_range[0]:g}–{v_range[1]:g} m/s"
        + ('' if density is None else f', 1/{1 / density:.0f} trees/m²') + ')')
    return m


def _follow_metrics(policy, cx, cy, rr, Q, Vq, p, T):
    tf = DP._tf()
    B = tf.shape(cx)[0]
    v = tf.zeros((B, 3))
    a = acd = tf.zeros((B, 3))
    h = tf.zeros((B, policy.inputs[2].shape[-1]))
    k_a = 1 - math.exp(-DP.LAMBDA_A * DP.DT)
    alive = tf.ones((B,))
    vis = derr = zerr = far = n = tf.zeros((B,))
    crashed = tf.zeros((B,))
    for k in tf.range(T):
        q, vq = Q[k], Vq[k]
        yaw = tf.atan2(q[:, 1] - p[:, 1], q[:, 0] - p[:, 0])
        thrust = a + tf.constant([[0.0, 0.0, DP.G]])
        img = DP.render_depth(p, DP.camera_rot(yaw, thrust), cx, cy, rr)
        omega, vdes, dist = follow_command(p, q, vq, tf)
        o, h = policy([img, DP.state_input(v, thrust, omega * vdes, yaw), h])
        a = a + (acd - a) * k_a
        acd = DP.from_heading(o * DP.A_MAX, yaw)
        sp = tf.norm(v, axis=1, keepdims=True)
        v = v + (a - DP.DRAG[0] * sp * v - DP.DRAG[1] * v) * DP.DT
        p = p + v * DP.DT
        d, _ = DP.clearance(p, cx, cy, rr)
        hit = tf.cast(d - DP.R_Q <= 0, tf.float32) * alive
        crashed = tf.maximum(crashed, hit)
        sc = sight_clearance(p, q, cx, cy, rr)
        vis += alive * tf.cast(sc > 0, tf.float32)
        derr += alive * tf.abs(dist[:, 0] - STANDOFF)
        zerr += alive * tf.abs(p[:, 2] - tf.clip_by_value(q[:, 2], Z_TARGET[0], Z_TARGET[1]))
        far += alive * tf.cast(dist[:, 0] > 2 * STANDOFF, tf.float32)
        n += alive
        alive = alive * (1 - hit)
    na = tf.maximum(tf.reduce_sum(n), 1.0)
    return {'crashed': tf.reduce_mean(crashed), 'visible': tf.reduce_sum(vis) / na,
            'dist_err': tf.reduce_sum(derr) / na, 'z_err': tf.reduce_sum(zerr) / na, 'far': tf.reduce_sum(far) / na}
