"""R18 · recreation of Loquercio et al. 2021 (nanouav.agile): geometry, stereo + SGM, expert, trajectories,
closed loop and the student's losses. Small sizes so the whole file runs in seconds."""
import math

import numpy as np
import pytest

from nanouav.agile import envs as EV
from nanouav.agile import trajectory as TR
from nanouav.agile.expert import Expert, collision_cost, global_plan
from nanouav.agile.geometry import Field
from nanouav.scene.renderer import Box, Cylinder, Plane, Sphere, Texture, World


def test_field_matches_scalar_sdf():
    tex = Texture('noise')
    w = World([Box((3, 1, 1), (0.5, 0.3, 1.0), 0.4, tex), Cylinder(5, -1, 0.3, 0, 4, tex),
               Sphere((2, -2, 1.5), 0.6, tex)], Plane(tex))
    P = np.random.default_rng(0).uniform([-1, -4, 0.1], [7, 4, 3], (300, 3))
    ref = np.array([min(min(o.sdf(p) for o in w.objects), p[2]) for p in P])
    assert np.allclose(Field(w)(P), ref, atol=1e-9)
    loc = Field(w).local((5, -1, 1), 1.5)                  # only the cylinder is near
    assert len(loc.cyl) == 1 and not loc.boxes and not loc.spheres


def test_collision_cost_eq3():
    d = np.array([-0.1, 0.0, 0.2, 0.4, 0.41, 2.0])
    assert collision_cost(d, 0.2) == pytest.approx([4, 4, 3, 0, 0, 0], abs=1e-9)


def test_poly5_projection_keeps_state_and_fits():
    p0, v0, a0 = np.array([1.0, 2.0, 1.5]), np.array([5.0, 0.5, 0.0]), np.array([0.0, 1.0, 0.0])
    pts = p0 + np.outer(TR.T_I, [5.0, 0.0, 0.0])
    c = TR.poly5_fit(pts, p0, v0, a0)
    assert TR.poly_eval(c, [0.0])[0] == pytest.approx(p0, abs=1e-9)
    assert TR.poly_eval(c, [0.0], 1)[0] == pytest.approx(v0, abs=1e-9)
    assert TR.poly_eval(c, [0.0], 2)[0] == pytest.approx(a0, abs=1e-9)
    assert np.abs(TR.poly_eval(c, TR.T_I) - pts).max() < 0.2
    assert TR.time_scale(c, 10.0) == pytest.approx(2.0, rel=0.05)    # plan at ~5 m/s → played 2× faster
    plan = TR.Plan(c, t0=3.0, beta=2.0)
    p, v, _ = plan.ref(3.25)                                         # τ = 0.5 s of the polynomial
    assert p == pytest.approx(TR.poly_eval(c, [0.5])[0]) and v == pytest.approx(2 * TR.poly_eval(c, [0.5], 1)[0])


def test_select_prefers_smooth_among_near_best():
    p0, v0, a0 = np.zeros(3), np.array([5.0, 0, 0]), np.zeros(3)
    straight = TR.poly5_fit(np.outer(TR.T_I, [5, 0, 0]), p0, v0, a0)
    wiggly = TR.poly5_fit(np.outer(TR.T_I, [5, 0, 0]) + np.c_[0 * TR.T_I, np.sin(9 * TR.T_I), 0 * TR.T_I], p0, v0, a0)
    assert TR.select([wiggly, straight], [0.10, 0.104]) == 1        # within 5 %: the smoother one
    assert TR.select([wiggly, straight], [0.10, 0.50]) == 0         # clearly riskier: excluded


def test_heading_frame_round_trip():
    P = np.random.default_rng(1).normal(size=(4, 10, 3))
    Q = TR.to_heading_frame(P, [1, 2, 3], 0.7)
    assert TR.from_heading_frame(Q, [1, 2, 3], 0.7) == pytest.approx(P)
    assert TR.to_heading_frame([[1 + math.cos(0.7), 2 + math.sin(0.7), 3]], [1, 2, 3], 0.7)[0] == pytest.approx([1, 0, 0])


def test_global_plan_goes_through_the_gap():
    e = EV.narrow_gap(3)
    g = global_plan(e, 5.0, straight=False)
    d = Field(e.world, ground=False)(g.pts)
    assert d.min() > 0.2 and g.pts[-1, 0] >= e.goal[0] - 0.5
    at_wall = g.pts[np.argmin(np.abs(g.pts[:, 0] - 10.0))]
    assert abs(at_wall[1] - e.meta['offset']) < e.meta['gap'] / 2
    assert global_plan(e, 5.0).length == pytest.approx(40.0, abs=0.2)   # test protocol: straight reference


def test_expert_labels_are_collision_free_and_ranked():
    e = EV.forest(4)
    g = global_plan(e, 7.0, straight=False)
    ex = Expert(e.world, n_samples=3000, chains=50, seed=0)
    L = ex.label(g.pos(6.0), g.ahead(6.0))
    assert L['trajs'].shape == (3, 10, 3) and L['n_free'] > 0
    assert np.all(np.diff(L['costs']) >= 0)
    d = Field(e.world)(L['trajs'].reshape(-1, 3))
    assert d.min() > 0.2


def test_sgm_depth_of_a_textured_wall():
    from nanouav.agile.stereo import StereoRig, observe
    tex = Texture('noise', albedo=0.7, albedo2=0.2, scale=0.05, seed=3)
    w = World([Box((4.0, 0, 2), (0.05, 6, 3), 0.0, tex)], Plane(Texture('noise', scale=0.1, seed=1)))
    rig = StereoRig(width=160, height=120, num_disp=32)
    d, g = observe(w, rig, (0, 0, 1.5), 0.0, rng=np.random.default_rng(0))
    m = (d > 0) & (np.abs(g - 3.85) < 0.05)                               # wall 3.95 m − camera offset 0.1 m
    assert m.mean() > 0.3
    assert np.median(np.abs(d[m] - g[m]) / g[m]) < 0.05


def test_rollout_tracks_global_reference_at_speed():
    from nanouav.agile.rollout import Config, Rollout
    e = EV.forest(1, density=1 / 64)
    r, samples, log = Rollout(e, Config(v_des=6.0, cam_hz=10.0), 'gbl').run()
    assert r['success'] and not samples and r['mean_speed'] > 5.0


def test_student_losses():
    tf = pytest.importorskip('tensorflow')
    from nanouav.agile.policy import build_model, collision_target, rwta
    te = np.random.default_rng(0).normal(size=(2, 3, 10, 3)).astype(np.float32)
    d = ((te[:, :, None] - te[:, None]) ** 2).sum((-1, -2))               # (B, K, M)
    off = d.sum((1, 2)) - np.trace(d, axis1=1, axis2=2)
    assert float(rwta(tf.constant(te), tf.constant(te))) == pytest.approx((0.05 / 2 * off).mean(), rel=1e-4)
    assert float(rwta(tf.constant(te), tf.constant(te[:, ::-1]))) == pytest.approx(float(rwta(tf.constant(te), tf.constant(te))), rel=1e-4)   # order-free
    esdf = np.full((1, 64, 64), 5.0, np.float32)
    esdf[0, 32, 8:] = 0.1                                                  # obstacle on the line y ∈ [0, 0.25)
    tn = np.zeros((1, 1, 10, 3), np.float32)
    tn[0, 0, :, 0] = np.linspace(0.5, 5, 10)
    tn[0, 0, :, 1] = 0.1
    c = float(collision_target(tf.constant(tn), tf.constant(esdf), tf.constant([1.5]))[0, 0])
    assert c == pytest.approx(10 * (4 - 0.25) * 0.1, rel=1e-4)            # every point at d = 0.1 m
    m = build_model((32, 48))
    tr, co = m([np.zeros((1, 32, 48, 1), np.float32), np.zeros((1, 15), np.float32)])
    assert tr.shape == (1, 3, 10, 3) and co.shape == (1, 3) and float(tf.reduce_min(co)) >= 0


def test_expert_bspline_starts_at_the_current_state():
    p0, v0 = np.array([1.0, -2.0, 1.5]), np.array([6.0, 1.0, 0.0])
    C = np.random.default_rng(2).normal(size=(4, 3, 3)) + p0 + [5, 0, 0]
    T = TR.bspline(p0, v0, C, t=[0.0, 1e-4, 1.0])
    assert T[:, 0] == pytest.approx(np.repeat(p0[None], 4, 0))
    assert (T[:, 1] - T[:, 0]) / 1e-4 == pytest.approx(np.repeat(v0[None], 4, 0), abs=2e-3)   # τ'(0) = v
    assert T[:, 2] == pytest.approx(C[:, 2])                                                # ends at the last control point
    assert TR._BS.sum(1) == pytest.approx(np.ones(10))


@pytest.mark.slow
def test_train_best_epoch_cosine_and_env_split(tmp_path):
    import json

    from nanouav.agile.dagger import save_shard
    from nanouav.agile.policy import train
    rng = np.random.default_rng(0)
    d = tmp_path / 'data_train'
    d.mkdir()
    for e in range(6):                                   # 6 tiny "environments" of 8 samples
        S = [{'depth': rng.integers(500, 9000, (120, 160)).astype(np.uint16), 'v': rng.normal(0, 1, 3).astype(np.float32),
              'R': np.eye(3, dtype=np.float32).ravel(), 'omega': np.array([1, 0, 0], np.float32),
              'trajs': rng.normal(0, 1, (3, 10, 3)).astype(np.float32), 'costs': rng.random(3).astype(np.float32),
              'esdf': rng.random((64, 64)).astype(np.float16), 'z': np.float32(1.5)} for _ in range(8)]
        save_shard(str(d / f'env_{e:04d}.npz'), S, {})
    out = str(tmp_path / 'm.keras')
    _, hist = train(str(d), out, epochs=3, batch=8, lr=1e-3, lr_end=1e-5, keep_best=True, val_by_env=True,
                    val_frac=0.34, log=lambda *a: None)
    meta = json.load(open(out + '.json'))
    assert meta['n_val'] == 16                           # 2 whole environments
    best = int(np.argmin([h['val'][1] for h in hist]))
    assert meta['kept_epoch'] == best


def test_projection_keeps_velocity_continuous():
    from nanouav.agile import trajectory as TR
    from nanouav.agile.envs import TEST
    from nanouav.agile.rollout import Config, Rollout
    R = Rollout(TEST['forest'](0), Config(v_des=3.0), 'gbl')
    p, v, a = np.zeros(3), np.array([3.0, 0, 0]), np.array([0.0, 0.5, 0])
    P = p + np.outer(TR.T_I, [4.3, 0.3, 0])          # a prediction too long for 3 m/s (β ≈ 0.7)
    pl = R._plan_points(P, p, v, a, 0.0, 'student', 3.0)
    p0, v0, a0 = pl.ref(0.0)
    assert pl.beta == pytest.approx(3.0 / np.hypot(4.3, 0.3), rel=1e-6)
    assert p0 == pytest.approx(p, abs=1e-9) and v0 == pytest.approx(v, abs=1e-9) and a0 == pytest.approx(a, abs=1e-9)
    R.PROJ = 'raw'                                   # ≤ 0.16.1: the commanded velocity jumped to β·v
    assert np.linalg.norm(R._plan_points(P, p, v, a, 0.0, 'student', 3.0).ref(0.0)[1] - v) > 0.5


def test_diagnose_classifies_a_crash():
    from nanouav.agile.diagnose import classify
    from nanouav.agile.envs import TEST
    env = TEST['forest'](0)
    tree = env.world.objects[0]
    p = np.array([tree.cx - tree.r - 0.1, tree.cy, 1.5])
    log = [(1.0, *p, 3.0, 0.0, 0.0, 'student')]
    base = {'p': p.tolist(), 'v': [3, 0, 0], 'costs': [0.1, 0.2, 0.3], 'lat': [1.0, -1.0, 0.0], 'beta': 0.7, 'W': []}
    late = dict(base, t=0.9, k=1, clear=[0.05, 0.1, 0.1])                       # unavoidable at the end
    tr = [dict(base, t=0.4, k=0, clear=[0.05, 0.6, 0.1]), late]                    # 0.6 s before: chose a colliding one
    d = classify(env, tr, log, 3.0)
    assert d['class'] == 'eligio_mal' and d['flips_2s'] == 1 and d['obstacle'] == 'Cylinder' and not d['overspeed']
    tr = [dict(base, t=0.4, k=1, clear=[0.05, 0.6, 0.1]), late]                    # chose the free one, then changed
    assert classify(env, tr, log, 3.0)['class'] == 'cambio_de_plan'
    tr = [dict(base, t=0.4, k=1, clear=[0.05, 0.1, 0.1]), late]                    # nothing free already
    assert classify(env, tr, log, 3.0)['class'] == 'todas_chocan'
    assert classify(env, tr, log, 2.0)['overspeed']                                # 3 m/s > 1.25 · 2 m/s


def test_arc_time_scale_and_speed_cap():
    from nanouav.agile import trajectory as TR
    from nanouav.agile.envs import TEST
    from nanouav.agile.rollout import Config, Rollout
    R = Rollout(TEST['forest'](0), Config(v_des=10.0), 'gbl')
    p, v, a = np.zeros(3), np.array([10.0, 0, 0]), np.zeros(3)
    ang = np.linspace(0, np.pi, 10)                    # a hooked prediction: half circle of radius 2 m
    P = np.stack([2 * np.sin(ang) + np.linspace(0.5, 1, 10), 2 - 2 * np.cos(ang), np.zeros(10)], 1)
    peak = lambda pl: pl.beta * np.linalg.norm(TR.poly_eval(pl.coef, np.linspace(0, 1, 41), 1), axis=1).max()
    chord = peak(R._plan_points(P, p, v, a, 0.0, 'student', 10.0))
    R.TSCALE = 'arc'
    arc = peak(R._plan_points(P, p, v, a, 0.0, 'student', 10.0))
    R.V_CAP = 1.3
    cap = peak(R._plan_points(P, p, v, a, 0.0, 'student', 10.0))
    assert chord > 1.5 * arc and cap <= 13.0 * 1.03


def test_episode_video_is_written(tmp_path):
    import cv2

    from nanouav.agile.envs import TEST
    from nanouav.agile.rollout import Config
    from nanouav.agile.video import render_episode
    env, cfg = TEST['forest'](0), Config()
    log = [(0.1 * i, 0.3 * i, 0.0, 1.5, 3.0, 0.0, 0.0, 'student') for i in range(5)]
    W = [[[0.3 * j, 0.1 * k * j, 1.5] for j in range(1, 11)] for k in (-1, 0, 1)]
    trace = [{'t': 0.1 * i, 'p': [0.3 * i, 0.0, 1.5], 'k': 1, 'costs': [0.2, 0.1, 0.3], 'clear': [0.5, 0.1, 0.6],
              'beta': 1.0, 'W': W, 'att': [0.0, 0.0, 0.0], 'depth': np.full((120, 160), 4000, np.uint16)}
             for i in range(5)]
    fn = render_episode(env, log, trace, cfg.rig, cfg.net_size, str(tmp_path / 'ep.mp4'), fps=15, chase=False,
                        result='CHOQUE (0.4 s)')
    cap = cv2.VideoCapture(fn)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    assert ok and frame.shape == (720, 1200, 3) and n >= 5


def test_perception_video_shows_only_onboard_data(tmp_path):
    """Default video: camera + tunnel HUD, full-resolution SGM, local map from the drone's own SGM."""
    import cv2

    from nanouav.agile.rollout import Config
    from nanouav.agile.video import render_episode
    cfg = Config()
    rng = np.random.default_rng(0)
    log = [(0.1 * i, 0.3 * i, 0.0, 1.5, 3.0, 0.0, 0.0, 'student') for i in range(5)]
    W = [[[0.3 + 0.4 * j, 0.1 * k * j, 1.5] for j in range(1, 11)] for k in (-1, 0, 1)]
    sgm = (rng.uniform(2, 9, cfg.rig.size) * 1000).astype(np.uint16)
    trace = [{'t': 0.1 * i, 'p': [0.3 * i, 0.0, 1.5], 'k': 1, 'costs': [0.2, 0.05, 0.3], 'beta': 1.0, 'W': W,
              'att': [0.0, 0.0, 0.0], 'depth': np.full((120, 160), 4000, np.uint16), 'sgm': sgm,
              'img': rng.integers(0, 255, cfg.rig.size, dtype=np.uint8)} for i in range(5)]
    fn = render_episode(None, log, trace, cfg.rig, cfg.net_size, str(tmp_path / 'ep.mp4'), fps=15, result='EXITO')
    cap = cv2.VideoCapture(fn)
    ok, frame = cap.read()
    assert ok and frame.shape == (720, 1280, 3) and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 5


def test_omega_points_along_the_reference_the_labels_follow():
    """R23: with omega_ref='gbl' the student's direction input points to τ_gbl 1 s ahead (where the expert's
    labels go); with 'line' (default) to the straight start→goal line."""
    from nanouav.agile.envs import training_env
    from nanouav.agile.rollout import GoalMission
    env = training_env(3, 0)
    ml, mg = GoalMission(env, 5.0), GoalMission(env, 5.0, omega_ref='gbl')
    mg.gref = ml.gref
    p = np.asarray(ml.gref.pos(12.0))
    ml.s_g = mg.s_g = 12.0                                  # the closest-point search follows the flight
    Fl, Fg = ml.frame(0.0, p, np.array([5.0, 0, 0]), 0.0), mg.frame(0.0, p, np.array([5.0, 0, 0]), 0.0)
    s, _ = mg.gref.closest(p)
    q = mg.gref.pos(s + 5.0) - p
    assert Fg['omega'] == pytest.approx(q / np.linalg.norm(q), abs=1e-6)
    assert np.linalg.norm(Fl['omega']) == pytest.approx(1.0) and np.allclose(Fl['ref'], Fg['ref'])
    mo = GoalMission(env, 5.0, gref=ml.gref, omega_ref='goal')
    q = np.asarray(env.goal, float) - p
    assert mo.frame(0.0, p, np.array([5.0, 0, 0]), 0.0)['omega'] == pytest.approx(q / np.linalg.norm(q))


def test_cost_to_go_reference_is_free_from_anywhere():
    """R23: the labels' reference replanned from the current position (descent of the cost-to-go to the goal)
    is collision-free from the start and from a point off τ_gbl, and passes through the goal. Also when the goal
    point lies in an obstacle's margin (env 16 of base seed 4242)."""
    from nanouav.agile.envs import training_env
    from nanouav.agile.expert import RQ, CostToGo
    from nanouav.agile.rollout import GoalMission
    env = training_env(3, 0)
    ctg, F = CostToGo(env), Field(env.world, ground=False)
    g = global_plan(env, 5.0)
    off = np.asarray(g.pos(15.0)) + [0.0, 2.5, 0.0]
    if F(off[None])[0] < 0.6:                       # keep the test point in free space
        off = off + [0.0, -5.0, 0.0]
    for p in (np.asarray(env.start, float), off):
        r = ctg.ref_from(p, 5.0)
        assert r is not None and np.linalg.norm(r.pts[0, :2] - p[:2]) < 1e-6
        assert F(r.pos(np.arange(0.3, r.length, 0.1))).min() > RQ
        assert np.linalg.norm(r.pts[:, :2] - np.asarray(env.goal)[:2], axis=1).min() < 0.8
        assert r.pts[-1, 0] > env.goal[0]                # and goes on past the goal line
    m = GoalMission(env, 5.0, label_ref='ctg')        # nominal route: the cost-to-go path from the start
    assert np.median(np.abs(m.gref.pts[:, 1] - env.start[1])) < np.median(np.abs(g.pts[:, 1] - env.start[1]))
    Fm = m.frame(0.0, off, np.array([5.0, 0, 0]), 0.0)
    assert Fm['label_ref'].shape == (10, 3) and np.allclose(Fm['label_ref'], Fm['ref'])
    assert np.linalg.norm(Fm['ref'][0, :2] - off[:2]) < 1.0                 # replanned from where the drone is
    # with the line penalty the reference keeps to the start→goal line and is still free
    r0, r1 = ctg.ref_from(off, 5.0), CostToGo(env, w_line=0.5).ref_from(off, 5.0)
    d_line = lambda r: np.median(np.abs(r.pts[:, 1] - env.start[1]))      # this env's line runs along +x
    assert d_line(r1) < d_line(r0) and F(r1.pos(np.arange(0.3, r1.length, 0.1))).min() > RQ
    env16 = training_env(16, 4242)
    assert CostToGo(env16).ref_from(np.asarray(env16.start, float), 5.0) is not None


def test_model_sidecar_carries_the_omega_convention(tmp_path):
    from nanouav.agile.policy import omega_of, write_meta
    m = tmp_path / 'student.keras'
    assert omega_of(m) == 'line'                    # R18–R22 models: no sidecar
    write_meta(m, omega_ref='goal', label_ref='ctg')
    assert omega_of(m) == 'goal'


def test_oracle_selection_picks_the_free_flown_plan():
    """--select oracle (diagnosis only): the hypothesis whose flown polynomial keeps clear of the obstacles,
    whatever the predicted costs say."""
    from nanouav.agile.rollout import fit_plan, oracle_index
    w = World([Cylinder(5.0, 0.0, 0.3, 0.0, 4.0, Texture("noise"))], Plane(Texture("noise")))
    F = Field(w, ground=False)
    p, v = np.array([0.0, 0.0, 1.5]), np.array([5.0, 0.0, 0.0])
    t = np.linspace(0.1, 1.0, 10)[:, None]
    hyps = [np.c_[5 * t, 0 * t, 0 * t + 1.5], np.c_[5 * t, 1.5 * t ** 2, 0 * t + 1.5], np.c_[5 * t, -2 * t ** 2, 0 * t + 1.5]]
    fits = [fit_plan(P, p, v, np.zeros(3), 5.0) for P in hyps]
    k = oracle_index(fits, F)
    assert k in (1, 2)                                  # never the straight one through the trunk
    assert oracle_index(fits[:1], F) == 0               # nothing free: the most clearance


def test_perceived_selection_uses_only_the_drones_own_sgm():
    """--select sgm: obstacles are the cells the drone's SGM hit in ≥ 2 recent frames (a single speckle is
    ignored); the plan through a perceived trunk is not chosen, whatever its predicted cost."""
    from nanouav.agile.rollout import PerceivedObstacles, fit_plan, perceived_index
    p, v = np.array([0.0, 0.0, 1.5]), np.array([5.0, 0.0, 0.0])
    t = np.linspace(0.1, 1.0, 10)[:, None]
    hyps = [np.c_[5 * t, 0 * t, 0 * t + 1.5], np.c_[5 * t, 1.5 * t ** 2, 0 * t + 1.5]]
    fits = [fit_plan(P, p, v, np.zeros(3), 5.0) for P in hyps]
    ang = np.linspace(0, 2 * np.pi, 40)
    trunk = np.c_[4.0 + 0.3 * np.cos(ang), 0.3 * np.sin(ang), np.full(40, 1.5)]
    obs = PerceivedObstacles(n_frames=5)
    obs.add(np.array([[2.0, 0.0, 1.5]]))                          # a speckle, seen once: ignored
    for _ in range(3):
        obs.add(trunk)
    assert obs.clearance(np.array([[2.0, 0.0, 1.5]])) > 1.0
    assert perceived_index(fits, [0.0, 0.5], obs) == 1            # the straight plan's low cost does not save it
    assert perceived_index(fits, None, PerceivedObstacles()) == 0          # nothing perceived: the least snap
    assert perceived_index(fits, [0.5, 0.0], PerceivedObstacles()) == 1   # … or the paper's cost rule (sgm_cost)


def test_perception_errors_compare_the_sgm_with_the_truth():
    from nanouav.agile.diagnose import perception_errors
    tr = [{'k': 0, 'clear': [0.1, 1.0], 'clear_seen': [0.8, 1.0]},     # saw free a plan that collides, chose it
          {'k': 1, 'clear': [0.1, 1.0], 'clear_seen': [0.1, 0.2]}]     # saw the colliding one; a false alarm
    e = perception_errors(tr)
    assert e == {'missed': 0.5, 'false_alarm': 0.5, 'chose_colliding': 0.5, 'n_plans': 4}
    assert perception_errors([{'k': 0, 'clear': [1.0]}]) is None       # not flown with --select sgm


def test_lookahead_rejects_a_plan_that_ends_pointing_into_a_trunk():
    """--lookahead: a plan free for its 1 s whose continuation at its final velocity hits a trunk is not free."""
    from nanouav.agile.rollout import PerceivedObstacles, fit_plan, perceived_index, perceived_clearance, with_lookahead
    p, v = np.array([0.0, 0.0, 1.5]), np.array([5.0, 0.0, 0.0])
    t = np.linspace(0.1, 1.0, 10)[:, None]
    hyps = [np.c_[5 * t, 0 * t, 0 * t + 1.5], np.c_[5 * t, 2.0 * t ** 2, 0 * t + 1.5]]
    fits = [fit_plan(P, p, v, np.zeros(3), 5.0) for P in hyps]
    ang = np.linspace(0, 2 * np.pi, 40)
    obs = PerceivedObstacles(n_frames=3)
    for _ in range(3):
        obs.add(np.c_[6.5 + 0.3 * np.cos(ang), 0.3 * np.sin(ang), np.full(40, 1.5)])   # 1.2 m past the straight plan
    seen = perceived_clearance(fits, obs)
    assert seen[0] > 0.35                                          # within 1 s, both plans look free
    assert perceived_index(fits, None, obs, clear=seen) == 0       # … and the straight one is the smoothest
    eff = with_lookahead(seen, fits, obs.clearance, 0.5)
    assert perceived_index(fits, None, obs, clear=eff) == 1        # its continuation hits the trunk


def test_keep_flies_on_a_free_plan_when_no_new_one_is():
    """--keep: when none of the new hypotheses is free but what is left of the plan being flown (≥ 0.3 s) still
    is, the drone goes on flying it instead of the new one with the most clearance."""
    from nanouav.agile.rollout import fit_plan, keep_previous, remainder
    p, v = np.array([0.0, 0.0, 1.5]), np.array([5.0, 0.0, 0.0])
    t = np.linspace(0.1, 1.0, 10)[:, None]
    coef, beta = fit_plan(np.c_[5 * t, 0 * t, 0 * t + 1.5], p, v, np.zeros(3), 5.0)
    old = TR.Plan(coef, 0.0, beta)
    X, left = remainder(old, 0.25)
    assert X[0] == pytest.approx(old.ref(0.25)[0]) and left == pytest.approx(1 / beta - 0.25)
    assert keep_previous([0.1, 0.2], 0.6, left)              # nothing new is free, the old plan still is
    assert not keep_previous([0.5, 0.1], 0.6, left)          # a new free plan: fly it
    assert not keep_previous([0.1], 0.6, 0.2)                # too little of the old plan left
    assert not keep_previous([0.1], 0.2, left)               # the old plan is not free either


def test_keep_in_the_closed_loop():
    """A stub student that alternates a plan around a trunk and plans straight into it: without --keep it
    takes the colliding ones (the diagnosis marks the old plan as still free); with --keep it flies on."""
    from nanouav.agile.envs import Env
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.agile.stereo import StereoRig
    tex = Texture('noise', albedo=0.7, albedo2=0.2, scale=0.05, seed=3)
    w = World([Cylinder(2.2, 0.0, 0.3, 0.0, 4.0, tex)], Plane(Texture('noise', scale=0.1, seed=1)))
    e = Env('keep', w, np.array([0.0, 0.0, 1.5]), np.array([20.0, 0.0, 1.5]), {}, True)
    t = np.linspace(0.1, 1.0, 10)[:, None]
    around = np.stack([np.c_[3 * t, 1.5 * t ** 2, 0 * t], np.c_[3 * t, 1.8 * t ** 2, 0 * t]])
    into = np.stack([np.c_[3 * t, 0 * t, 0 * t], np.c_[3 * t, 0.05 * t, 0 * t]])
    out = {}
    for keep in (False, True):
        n = [0]

        def policy(depth, v, R, omega):
            n[0] += 1
            return (around if n[0] % 2 else into), np.array([0.1, 0.1])
        R = Rollout(e, Config(v_des=3.0, seed=0, t_max=1.0, rig=StereoRig(width=160, height=120, num_disp=32)),
                    'student', policy)
        R.SELECT, R.KEEP, R.trace = 'oracle', keep, []
        R.run()
        out[keep] = R.trace
    assert not any(d.get('kept') for d in out[False])
    assert any(d.get('old_free') for d in out[False])        # the case --keep acts on happens
    kept = [d for d in out[True] if d.get('kept')]
    assert kept and all(d['none_free'] and d['old_free'] and d['clear_kept'] >= 0.2 for d in kept)


def test_compare_pairs_runs_by_environment(tmp_path):
    """r22 compare: arrivals per run, and the second run against the first paired by environment (McNemar)."""
    import json
    from nanouav.agile.stats import compare, mcnemar, rates
    assert mcnemar(0, 0) == 1.0 and mcnemar(29, 5) < 1e-4 and mcnemar(10, 15) == pytest.approx(0.424, abs=1e-3)
    mk = lambda ok: {'student_v3': [{'line': i, 'success': s, 'crashed': not s, 'time_s': 10.0, 'mean_speed': 3.0}
                                    for i, s in enumerate(ok)]}
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    json.dump(mk([1, 1, 0, 0]), open(a, 'w'))
    json.dump(mk([1, 0, 1, 1]), open(b, 'w'))
    lines = []
    s = compare([str(a), str(b)], out=lines.append)
    assert s['b.json']['student_v3'][:2] == (2, 1) and s['b.json']['total'][:2] == (2, 1)
    assert rates(mk([1, 0])['student_v3']) == pytest.approx((0.05, 1 / 60))
    assert len(lines) == 4


def test_perceived_clearance_kdtree_matches_brute_force():
    """The k-d tree gives the exact nearest cell: same clearance as brute force wherever it matters."""
    from nanouav.agile.rollout import PerceivedObstacles
    rng = np.random.default_rng(0)
    obs = PerceivedObstacles(n_frames=3)
    for _ in range(3):
        obs.add(rng.uniform([0, -3, 0], [10, 3, 3], (2000, 3)))
    X = rng.uniform([0, -3, 0], [10, 3, 3], (21, 3))
    C = obs.cells()
    brute = float(np.sqrt(((X[:, None] - C[None]) ** 2).sum(-1)).min()) - 0.5 * obs.cell
    assert obs.clearance(X) == pytest.approx(brute, abs=1e-9)
    d, q = obs.nearest(X[:1])
    assert d[0] == pytest.approx(np.linalg.norm(X[0] - q[0]))


def test_min_pool_keeps_the_nearest_pixel():
    """--pool min: a one-pixel-wide near obstacle survives the 4 × 4 reduction; plain striding skips it."""
    from nanouav.agile.rollout import _min_pool
    d = np.full((8, 8), 10.0, np.float32)
    d[1, 2] = 2.0                                   # a thin pole, off the stride grid
    d[5:, 5:] = 0                                   # a hole
    m, vv, uu = _min_pool(d, 4)
    assert m[0, 0] == 2.0 and (vv[0, 0], uu[0, 0]) == (1, 2)
    assert m[1, 1] == 10.0                          # the hole does not hide the valid pixels of its block
    assert d[::4, ::4].min() == 10.0


def test_fan_candidates_leave_along_the_velocity_and_fan_around_omega():
    from nanouav.agile.rollout import fan_cost, fan_points
    p, v, om = np.array([0.0, 0.0, 1.5]), np.array([4.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    W = fan_points(p, v, om, 5.0, z_lim=(0.7, 2.5))
    assert W.shape == (33, 10, 3)
    assert np.allclose(W[0, -1], [5.0, 0.0, 1.5])                 # 0°, no height change: straight ahead
    assert W[:, -1, 2].min() >= 0.7 and W[:, -1, 2].max() <= 2.5
    ref = p + np.outer(np.arange(1, 11) / 10, [5.0, 0, 0])
    assert int(np.argmin(fan_cost(W, ref))) == 0                  # nearest the line: the straight one


def test_reactive_pilot_goes_around_a_trunk_on_its_own_sgm():
    """Pilot 'reactive' (no network, nothing privileged): a trunk on the line is avoided from its own SGM."""
    from nanouav.agile.envs import Env
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.agile.stereo import StereoRig
    tex = Texture('noise', albedo=0.7, albedo2=0.2, scale=0.05, seed=3)
    w = World([Cylinder(6.0, 0.0, 0.3, 0.0, 4.0, tex)], Plane(Texture('noise', scale=0.1, seed=1)))
    e = Env('one_trunk', w, np.array([0.0, 0.0, 1.5]), np.array([12.0, 0.0, 1.5]), {}, True)
    R = Rollout(e, Config(v_des=3.0, seed=0, rig=StereoRig(width=160, height=120, num_disp=32)), 'reactive')
    R.KEEP = True
    r, _, log = R.run()
    assert r['success'] and not r['crashed'] and r['decide_ms'] is not None
    assert max(abs(row[2]) for row in log) > 0.3                  # it left the line to pass the trunk


def test_edge_filter_drops_the_pixels_smeared_across_a_depth_jump():
    """--edge: a trunk at 3 m in front of a wall at 8 m; SGM-like smear pixels at 5 m on its border are
    dropped, the trunk's inside and the wall's inside stay."""
    from nanouav.agile.rollout import edge_filter
    d = np.full((12, 12), 8.0, np.float32)
    d[:, 4:8] = 3.0
    d[:, 8] = 5.0                                   # the smear between the trunk and the wall
    d[0, 0] = 0.0                                   # a hole
    e = edge_filter(d)
    assert (e[:, 8] == 0).all()                     # smear gone
    assert (e[2:10, 5:7] == 3.0).all() and (e[2:10, 1:3] == 8.0).all()
    assert e[0, 1] == 0.0                           # next to a hole: not trusted either


def test_carving_removes_noise_that_later_frames_see_through():
    """--carve: a phantom cell hit in two frames but seen through in three later ones no longer counts; a real
    surface that every frame hits stays."""
    from nanouav.agile.rollout import PerceivedObstacles
    from nanouav.agile.stereo import StereoRig
    rig = StereoRig(width=160, height=120)
    pose = (np.array([0.0, 0.0, 1.5]), 0.0, 0.0, 0.0)
    wall = np.full((120, 160), 6.0, np.float32)            # a wall 6 m ahead (forward distance)
    obs = PerceivedObstacles(n_frames=6, carve=True)
    ghost = np.array([[3.0, 0.0, 1.5]])                     # noise floating 3 m ahead, straight on the axis
    real = np.array([[6.1, 0.0, 1.5]])
    for k in range(5):
        obs.add(np.vstack([real] + ([ghost] if k < 2 else [])), wall, rig, pose)
    C = obs.cells()
    assert len(C) and np.min(np.linalg.norm(C - real, axis=1)) < 0.2
    assert np.min(np.linalg.norm(C - ghost, axis=1)) > 0.5
    plain = PerceivedObstacles(n_frames=6)
    for k in range(5):
        plain.add(np.vstack([real] + ([ghost] if k < 2 else [])))
    assert np.min(np.linalg.norm(plain.cells() - ghost, axis=1)) < 0.2   # without carving the ghost stays


def test_soft_clearance_cost_prefers_the_wider_gap():
    """--w-clear: between two candidates equally near the line, the one that only just clears costs more."""
    from nanouav.agile.rollout import fan_cost
    ref = np.outer(np.arange(1, 11) / 10, [5.0, 0, 0])
    W = np.stack([ref + [0, 0.3, 0], ref + [0, -0.3, 0]])
    assert fan_cost(W, ref)[0] == pytest.approx(fan_cost(W, ref)[1])
    J = fan_cost(W, ref, clear=[0.36, 1.2], w_clear=2.0)
    assert J[0] > J[1] + 1.5
