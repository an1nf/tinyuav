"""R20 · Follow a colored balloon through a forest, avoiding the trees (on top of the R18 policy).

  nanouav run r20 frames --n 6                                                   # sample frames + detections
  nanouav run r20 sim --pilots gbl,direct,student --model results/r18/student_train.keras --n 20
  nanouav run r20 sim --pilots diffphys --model results/r26/diffphys_follow.keras --n 20 --edge   # R26
  nanouav run r20 dagger --envs 100 --init results/r18/student_train.keras --workers 7
  nanouav run r20 pick-color --source frame.jpg --box 300,200,360,260            # thresholds for real light
  nanouav run r20 video --source 0 --color red --diameter 0.3 --out tracked.mp4  # real camera / video

Results in $NANOUAV_RESULTS/r20 (`--results DIR` to change). See docs/research/R20_seguimiento_globo.md.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from nanouav import paths


def _out(a):
    d = a.out or os.path.join(str(paths.results_dir()), 'r20')
    os.makedirs(d, exist_ok=True)
    return d


def _env(job):
    from nanouav.follow.balloon import balloon_env
    return balloon_env(job['seed'], density=job.get('density'), n_distractors=job.get('distractors'),
                       speed=job.get('speed'), color=job.get('color', 'red'), p_same=job.get('p_same', 0.25),
                       z_lo=job.get('z_lo', 1.2), z_hi=job.get('z_hi', 2.6))


def _sim_job(job):
    from nanouav.agile import dagger
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.follow.balloon_mission import BalloonMission
    dagger._worker_init()
    try:
        env = _env(job)
        M = BalloonMission(env, standoff=job['standoff'], seed=job['seed'])
        for k in ('q_acc', 'yaw_ext', 'gate_v', 'relock_free'):     # 0.27.5: fast targets (None: as before)
            if job.get(k) is not None:
                setattr(M, k, job[k])
        if job.get('q_acc') is not None:
            M.flt = M._make_filter()
        pol = dagger._policy(job['model']) if job['pilot'] == 'student' else \
            (job['model'] if job['pilot'] == 'diffphys' else None)     # R26: the model path
        R = Rollout(env, Config(v_des=2.0, cam_hz=job['cam_hz'], seed=job['seed']), job['pilot'], pol, mission=M)
        R.EDGE = bool(job.get('edge'))                              # R26: the diffphys input as in R25 (--edge)
        r, _, _ = R.run()
    except RuntimeError as e:
        r = {'success': False, 'crashed': False, 'failed': False, 'error': str(e)}
    r.update(seed=job['seed'])
    return r


FAST_TARGET = {'q_acc': 5.0, 'yaw_ext': 0.5, 'gate_v': 1.0, 'relock_free': True}


def _lost_opts(a):
    """--fast-target and its parts → the BalloonMission attributes (only the ones given)."""
    o = dict(FAST_TARGET) if getattr(a, 'fast_target', False) else {}
    for k in ('q_acc', 'yaw_ext', 'gate_v'):
        if getattr(a, k, None) is not None:
            o[k] = getattr(a, k)
    if getattr(a, 'relock_free', False):
        o['relock_free'] = True
    return o


def cmd_sim(a):
    out = _out(a)
    pilots = a.pilots.split(',')
    jobs = [dict(pilot=p, seed=s, cam_hz=a.cam_hz, standoff=a.standoff, model=os.path.abspath(a.model) if a.model else None,
                 density=a.density, distractors=a.distractors, speed=a.speed, color=a.color, p_same=a.p_same,
                 z_lo=a.z[0], z_hi=a.z[1], edge=getattr(a, 'edge', False), **_lost_opts(a))
            for p in pilots for s in range(a.seed0, a.seed0 + a.n)]
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(k, '1')
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(_sim_job, jobs, chunksize=1)
    table = {}
    for j, r in zip(jobs, res):
        table.setdefault(j['pilot'], []).append(r)
    fn = os.path.join(out, f'balloon_{a.tag or "sim"}.json')
    json.dump(table, open(fn, 'w'), indent=1)
    print(f'{"pilot":8s} {"success":>8s} {"crash":>5s} {"lost":>4s} {"locked":>6s} {"true":>5s} {"d_err":>5s} {"relock":>6s}')
    for p in pilots:
        L = table[p]
        f = lambda k: np.nanmedian([r.get(k) if r.get(k) is not None else np.nan for r in L])
        print(f'{p:8s} {sum(r.get("success", 0) for r in L):3d}/{len(L):<4d} {sum(r.get("crashed", 0) for r in L):5d} '
              f'{sum(r.get("failed", 0) for r in L):4d} {f("locked_frac"):6.2f} {f("true_frac"):5.2f} '
              f'{f("dist_err_med"):5.2f} {f("relocks"):6.0f}')
    print(f'→ {fn}  ({time.time() - t0:.0f} s)')


def cmd_frames(a):
    """A few frames from behind the balloon at 4–12 m with the detections drawn (sanity check of the
    renderer and the detector) → frames_XX.png and a table of the range error."""
    import cv2
    from nanouav.agile.stereo import StereoRig, observe_full
    from nanouav.follow.blob import BlobDetector
    from nanouav.follow.estimate import blob_to_world
    out = _out(a)
    rig = StereoRig()
    rng = np.random.default_rng(0)
    rows = []
    for k in range(a.n):
        env = _env(dict(seed=a.seed0 + k, color=a.color, density=a.density))
        b = env.walkers[0]
        t = float(rng.uniform(1, b.path.length / b.speed))
        for w in env.walkers:
            w.move(t)
        c = np.array(b.body.center)
        d = float(rng.uniform(4, 12))
        heading = b.vel(t)[:2]
        heading = heading / max(np.linalg.norm(heading), 1e-6)
        pose = (np.r_[c[:2] - d * heading, c[2] + rng.uniform(-0.4, 0.4)], float(np.arctan2(*heading[::-1])), 0.0, 0.0)
        g, sgm, gt, rgb = observe_full(env.render_world(), rig, *pose, rng=rng, rgb=True)
        D = BlobDetector(a.color).detect(rgb)
        img = np.ascontiguousarray(rgb[..., ::-1])
        best = None
        for det in D:
            x1, y1, x2, y2 = map(int, det['box'])
            q, src = blob_to_world(rig, pose, sgm, det['box'], b.radius)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            if q is not None and (best is None or np.linalg.norm(q - c) < best[0]):
                best = (float(np.linalg.norm(q - c)), src)
        fn = os.path.join(out, f'frames_{k:02d}.png')
        cv2.imwrite(fn, cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        rows.append({'seed': a.seed0 + k, 'dist': round(d, 1), 'n_det': len(D), 'err_m': None if best is None else round(best[0], 2),
                     'src': None if best is None else best[1]})
        print(rows[-1], '→', fn)


def cmd_pick_color(a):
    """HSV statistics of the balloon in a real image (a box drawn around it) → thresholds for `video`."""
    import cv2
    img = cv2.imread(a.source)
    if img is None:
        cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f'cannot read {a.source}')
    x1, y1, x2, y2 = (int(v) for v in a.box.split(','))
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    hsv = hsv[hsv[:, 1] > 60]                         # the saturated pixels (the balloon, not the background)
    if len(hsv) < 10:
        raise SystemExit('too few saturated pixels in the box')
    h = hsv[:, 0]
    # circular statistics of the hue (red wraps around 0/180)
    ang = h / 180 * 2 * np.pi
    h0 = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2) % 180
    dh = np.abs(h - h0)
    dh = np.minimum(dh, 180 - dh)
    tol = float(np.percentile(dh, 95)) + 2
    s_min = float(np.percentile(hsv[:, 1], 5)) / 255
    v_min = float(np.percentile(hsv[:, 2], 2)) / 255
    print(f'hue {h0:.1f} ± {tol:.1f} (OpenCV 0–180) · saturation ≥ {s_min:.2f} · value ≥ {v_min:.2f}')
    print(f'→ nanouav run r20 video --source ... --hsv {h0:.1f},{tol:.1f},{max(s_min - 0.05, 0.2):.2f},{max(v_min - 0.03, 0.05):.2f}')


def cmd_video(a):
    """Color detector + tracker + target lock on a video file or a camera; range from the apparent size."""
    import cv2
    from nanouav.follow.blob import BlobDetector
    from nanouav.follow.estimate import TargetFilter, blob_to_world
    from nanouav.follow.tracker import TargetLock, Tracker
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f'cannot read {a.source}')
    H, W = frame.shape[:2]

    class Cam:
        width, height = W, H
        focal = W / 2 / np.tan(np.radians(a.hfov) / 2)
        offset, tilt_up_deg = (0.0, 0.0, 0.0), 0.0
    det = BlobDetector(a.color, min_area=a.min_area)
    if a.hsv:
        h0, tol, smin, vmin = (float(x) for x in a.hsv.split(','))
        det.h0, det.hue_tol, det.s_min, det.v_min = h0, tol, smin, vmin
    trk, lock, flt = Tracker(min_gate_px=12.0), TargetLock(), TargetFilter(dims=3)
    pose = (np.zeros(3), 0.0, 0.0, 0.0)
    zero = np.zeros((H, W), np.float32)
    pos_of = lambda tr: blob_to_world(Cam, pose, zero, tr.box, a.diameter / 2)[0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wr = cv2.VideoWriter(a.video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)) if a.video_out else None
    pick = tuple(float(x) for x in a.pick.split(',')) if a.pick else None
    n, t0, states = 0, time.time(), []
    while ok:
        t = n / fps
        rgb = np.ascontiguousarray(frame[..., ::-1])
        tracks = trk.update(det.detect(rgb), rgb)
        tr, how = (lock.select_initial(tracks, (W, H), pick), 'locked') if lock.track_id is None else lock.update(tracks, pos_of, flt, t)
        if tr is not None and lock.track_id is not None:
            q = pos_of(tr)
            if q is not None:
                flt.update(t, q, 0.2)
        for k in tracks:
            x1, y1, x2, y2 = map(int, k.box)
            col = (0, 0, 255) if tr is not None and k.id == tr.id else (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if col[2] == 255 else 1)
        if flt.pos is not None:
            x, y, z = flt.pos
            txt = f'{how} dist {np.linalg.norm(flt.pos):.1f} m  ahead {x:.1f} left {y:.1f} up {z:.1f}'
        else:
            txt = how or 'search'
        cv2.putText(frame, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        states.append(how)
        if wr:
            wr.write(frame)
        n += 1
        ok, frame = cap.read()
    if wr:
        wr.release()
    print(f'{n} frames · {n / max(time.time() - t0, 1e-6):.1f} fps · locked {states.count("locked") / max(n, 1):.0%} · '
          f'relocks {lock.relocks} → {a.video_out or "(no output video)"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('frames')
    p.add_argument('--n', type=int, default=6)
    p.add_argument('--seed0', type=int, default=3000)
    p.add_argument('--density', type=float, default=1 / 36)
    p = sub.add_parser('sim')
    p.add_argument('--pilots', default='gbl,direct')
    p.add_argument('--model', default='')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--seed0', type=int, default=3000)
    p.add_argument('--standoff', type=float, default=4.0)
    p.add_argument('--density', type=float, default=None)
    p.add_argument('--distractors', type=int, default=None)
    p.add_argument('--speed', type=float, default=None)
    p.add_argument('--p-same', type=float, default=0.25, help='probability that a distractor has the target color')
    p.add_argument('--edge', action='store_true', help='diffphys pilot: drop SGM pixels on depth edges (as in R25)')
    p.add_argument('--fast-target', action='store_true',
                   help='0.27.5: --q-acc 5 --yaw-ext 0.5 --gate-v 1 --relock-free (any of them can be overridden)')
    p.add_argument('--q-acc', type=float, default=None, help='process noise of the target filter (default 2 m/s²)')
    p.add_argument('--yaw-ext', type=float, default=None,
                   help='target lost: the camera looks at the last estimate moved on at most this many s')
    p.add_argument('--gate-v', type=float, default=None,
                   help='target lost: the re-lock gate grows at |v_target| + this (m/s), not 1.5 m/s')
    p.add_argument('--relock-free', action='store_true',
                   help='the first measurement after a re-lock skips the innovation gate')
    p.add_argument('--z', type=lambda x: tuple(float(v) for v in x.split(',')), default=(1.2, 2.6),
                   help='height range of the balloon, e.g. 1.5,1.5 for a constant height (the R18 student '
                        'has never been asked to change height)')
    p.add_argument('--tag', default='')
    p = sub.add_parser('dagger')
    p.add_argument('--envs', type=int, default=100)
    p.add_argument('--per-round', type=int, default=20)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--init', required=True)
    p.add_argument('--label-every', type=int, default=2)
    p.add_argument('--xi-start', type=float, default=1.0,
                   help='ξ of the first round: the R18 student has never changed height, so the expert flies '
                        'more at the start (R19 starts at 6)')
    p.add_argument('--xi-step', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=3e-4, help='fine-tuning: lower than the 1e-3 of R18')
    p.add_argument('--lr-end', type=float, default=1e-5)
    p.add_argument('--no-best', dest='best', action='store_false', help='keep the last epoch, not the best one')
    p = sub.add_parser('pick-color')
    p.add_argument('--source', required=True, help='image, video or camera index')
    p.add_argument('--box', required=True, help='x1,y1,x2,y2 around the balloon')
    p = sub.add_parser('video')
    p.add_argument('--source', required=True)
    p.add_argument('--out', dest='video_out', default='')
    p.add_argument('--hsv', default='', help='h0,tol,s_min,v_min from pick-color (overrides --color)')
    p.add_argument('--diameter', type=float, default=0.3, help='balloon diameter (m), for the range')
    p.add_argument('--min-area', type=int, default=20)
    p.add_argument('--hfov', type=float, default=70.0)
    p.add_argument('--pick', default='', help='x,y pixel of the balloon to follow (default: nearest to the center)')
    for q in sub.choices.values():
        q.add_argument('--results', dest='out', default='', help='results directory (default results/r20)')
        q.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
        q.add_argument('--cam-hz', type=float, default=10.0)
        q.add_argument('--color', default='red', help='red | orange | yellow | green | blue | magenta')
    a = ap.parse_args()
    cmd = {'frames': cmd_frames, 'sim': cmd_sim, 'pick-color': cmd_pick_color, 'video': cmd_video}.get(a.cmd)
    if cmd:
        return cmd(a)
    from nanouav.agile import dagger
    out = a.out or os.path.join(str(paths.results_dir()), 'r18')    # with R18's data_train
    os.makedirs(out, exist_ok=True)
    m = dagger.run(out, n_envs=a.envs, per_round=a.per_round, workers=a.workers, epochs=a.epochs, batch=a.batch,
                   cam_hz=a.cam_hz, kind='balloon', init_model=os.path.abspath(a.init), label_every=a.label_every,
                   xi_start=a.xi_start, xi_step=a.xi_step,
                   train_kw=dict(lr=a.lr, lr_end=a.lr_end, keep_best=a.best, val_by_env=True))
    print('student →', m)


if __name__ == '__main__':
    main()
