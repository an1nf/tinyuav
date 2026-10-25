"""R19 · Detect a person and follow them, avoiding obstacles (on top of the R18 policy).

  nanouav run r19 prep-visdrone --raw data/visdrone --out data/person --fetch     # VisDrone → YOLO person
  nanouav run r19 train-det --data data/person/person.yaml --model yolo11s.pt --imgsz 960 --epochs 100
  nanouav run r19 eval-det --model runs/detect/person/weights/best.pt --data data/person   # recall vs size
  nanouav run r19 video --source clip.mp4 --model best.pt --out tracked.mp4       # tracker + target lock
  nanouav run r19 sim --pilots gbl,direct,student --model results/r18/student_train.keras --n 10
  nanouav run r19 dagger --envs 100 --init results/r18/student_train.keras --workers 7

Results in $NANOUAV_RESULTS/r19 (`--results DIR` to change). See docs/research/R19_seguimiento_personas.md.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from nanouav import paths


def _out(a):
    d = a.out or os.path.join(str(paths.results_dir()), 'r19')
    os.makedirs(d, exist_ok=True)
    return d


def _detector(calib, seed):
    from nanouav.follow.detector import SimDetector
    kw = json.load(open(calib))['logistic'] if calib else {}
    return SimDetector(seed=seed, **{k: kw[k] for k in ('h50', 'h_scale', 'p_max') if k in kw})


def _sim_job(job):
    from nanouav.agile import dagger
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.follow.mission import FollowMission
    from nanouav.follow.people import people_env
    dagger._worker_init()
    env = people_env(job['seed'], density=job.get('density'), n_distractors=job.get('distractors'),
                     speed=job.get('speed'))
    M = FollowMission(env, standoff=job['standoff'], detector=_detector(job.get('calib'), job['seed']), seed=job['seed'])
    pol = dagger._policy(job['model']) if job['pilot'] == 'student' else \
        (job['model'] if job['pilot'] == 'diffphys' else None)         # R26: the model path
    try:
        r, _, _ = Rollout(env, Config(v_des=2.0, cam_hz=job['cam_hz'], seed=job['seed']), job['pilot'], pol,
                          mission=M).run()
    except RuntimeError as e:
        r = {'success': False, 'crashed': False, 'failed': False, 'error': str(e)}
    r.update(seed=job['seed'])
    return r


def cmd_sim(a):
    out = _out(a)
    pilots = a.pilots.split(',')
    seeds = list(range(a.seed0, a.seed0 + a.n))
    jobs = [dict(pilot=p, seed=s, cam_hz=a.cam_hz, standoff=a.standoff, model=os.path.abspath(a.model) if a.model else None,
                 calib=a.det_calib or None, density=a.density, distractors=a.distractors, speed=a.speed)
            for p in pilots for s in seeds]
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(k, '1')
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(_sim_job, jobs, chunksize=1)
    table = {}
    for j, r in zip(jobs, res):
        table.setdefault(j['pilot'], []).append(r)
    fn = os.path.join(out, f'follow_{a.tag or "sim"}.json')
    json.dump(table, open(fn, 'w'), indent=1)
    print(f'{"pilot":8s} {"success":>8s} {"crash":>5s} {"lost":>4s} {"locked":>6s} {"true":>5s} {"d_err":>5s} {"relock":>6s}')
    for p in pilots:
        L = table[p]
        f = lambda k: np.nanmedian([r.get(k) if r.get(k) is not None else np.nan for r in L])
        print(f'{p:8s} {sum(r.get("success", 0) for r in L):3d}/{len(L):<4d} {sum(r.get("crashed", 0) for r in L):5d} '
              f'{sum(r.get("failed", 0) for r in L):4d} {f("locked_frac"):6.2f} {f("true_frac"):5.2f} '
              f'{f("dist_err_med"):5.2f} {f("relocks"):6.0f}')
    print(f'→ {fn}  ({time.time() - t0:.0f} s)')


def cmd_prep(a):
    from nanouav.follow import visdrone
    y, st = visdrone.prepare(a.raw, a.out_data, splits=tuple(a.splits.split(',')), fetch=a.fetch)
    print('dataset →', y)


def cmd_train_det(a):
    from ultralytics import YOLO
    m = YOLO(a.model)
    data = a.data
    if not a.new_head and len(m.names) > 1 and m.names.get(0) == 'person':
        # keep the COCO head: same 80 names, our labels are all class 0 (= COCO person), so the person
        # classifier starts from COCO instead of a randomly initialized 1-class layer
        import yaml
        d = yaml.safe_load(open(a.data))
        d['path'] = d.get('path') or os.path.dirname(os.path.abspath(a.data))
        d['names'] = {int(k): v for k, v in m.names.items()}
        data = os.path.splitext(a.data)[0] + '_cocohead.yaml'
        yaml.safe_dump(d, open(data, 'w'))
    r = m.train(data=data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device, name=a.name,
                project=os.path.join(_out(a), 'detect'), fraction=a.fraction, workers=a.dl_workers,
                close_mosaic=10, patience=30)
    best = os.path.join(str(r.save_dir), 'weights', 'best.pt')
    print('best →', best)
    if a.export:
        print('export →', YOLO(best).export(format=a.export, imgsz=a.imgsz, half=a.export == 'engine'))


def cmd_eval_det(a):
    from nanouav.follow import visdrone
    R = visdrone.recall_by_height(a.model, a.data, a.split, conf=a.conf, imgsz=a.imgsz, max_images=a.max_images)
    R['logistic'] = visdrone.fit_logistic(R['bins'])
    R['logistic'] = {k: float(v) for k, v in R['logistic'].items()} if R['logistic'] else None
    R['fp_rate'] = R['fp_per_image']
    fn = os.path.join(_out(a), f'det_calib_{os.path.splitext(os.path.basename(a.model))[0]}.json')
    json.dump(R, open(fn, 'w'), indent=1)
    print(f'{"box height (px)":>16s} {"n":>6s} {"recall":>7s}')
    for b in R['bins']:
        rec = '—' if b['recall'] is None else f"{b['recall']:.2f}"
        print(f"{b['h_min']:>7.0f} – {min(b['h_max'], 9999):<6.0f} {b['n']:6d} {rec:>7s}")
    print('false positives / image', round(R['fp_per_image'], 2), '· logistic', R['logistic'], '→', fn)


def cmd_video(a):
    """Detector + tracker + target lock on a video file or camera; distance from the box height (no depth)."""
    import cv2
    from nanouav.follow.detector import YoloDetector
    from nanouav.follow.estimate import TargetFilter, box_to_world
    from nanouav.follow.tracker import TargetLock, Tracker
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f'cannot read {a.source}')
    H, W = frame.shape[:2]

    class Cam:                                         # pinhole with the given horizontal field of view
        width, height = W, H
        focal = W / 2 / np.tan(np.radians(a.hfov) / 2)
        offset, tilt_up_deg = (0.0, 0.0, 0.0), 0.0
    det, trk, lock, flt = YoloDetector(a.model, conf=a.conf, imgsz=a.imgsz), Tracker(), TargetLock(), TargetFilter()
    pose = (np.array([0.0, 0.0, a.cam_height]), 0.0, 0.0, 0.0)
    zero = np.zeros((H, W), np.float32)
    pos_of = lambda tr: box_to_world(Cam, pose, zero, tr.box)[0]
    wr = (cv2.VideoWriter(a.video_out, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS) or 30, (W, H))
          if a.video_out else None)
    pick = tuple(float(x) for x in a.pick.split(',')) if a.pick else None
    n, t0, log = 0, time.time(), []
    while ok:
        t = n / (cap.get(cv2.CAP_PROP_FPS) or 30)
        rgb = frame[..., ::-1]
        tracks = trk.update(det.detect(rgb), rgb)
        tr, how = (lock.select_initial(tracks, (W, H), pick), 'locked') if lock.track_id is None else lock.update(tracks, pos_of, flt, t)
        if tr is not None:
            q = pos_of(tr)
            if q is not None:
                flt.update(t, q, 0.3)
        for k in tracks:
            x1, y1, x2, y2 = map(int, k.box)
            col = (0, 0, 255) if tr is not None and k.id == tr.id else (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if col[2] == 255 else 1)
            cv2.putText(frame, str(k.id), (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        d = float(np.hypot(*flt.pos[:2])) if flt.pos is not None else float('nan')
        cv2.putText(frame, f'{how} target={lock.track_id} dist~{d:.1f} m relocks={lock.relocks}', (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        log.append({'frame': n, 'state': how, 'track': lock.track_id, 'dist': d})
        if wr:
            wr.write(frame)
        n += 1
        ok, frame = cap.read()
    if wr:
        wr.release()
    st = [r['state'] for r in log]
    print(f'{n} frames · {n / max(time.time() - t0, 1e-6):.1f} fps · locked {st.count("locked") / max(n, 1):.0%} · '
          f'relocks {lock.relocks} → {a.video_out or "(no output video)"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep-visdrone')
    p.add_argument('--raw', required=True)
    p.add_argument('--out', dest='out_data', required=True)
    p.add_argument('--splits', default='train,val')
    p.add_argument('--fetch', action='store_true', help='download the zips from the ultralytics GitHub releases')
    p = sub.add_parser('train-det')
    p.add_argument('--data', required=True)
    p.add_argument('--model', default='yolo11s.pt')
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default=None)
    p.add_argument('--name', default='person')
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--dl-workers', type=int, default=8)
    p.add_argument('--export', default='', help='onnx | engine (TensorRT, on the Jetson)')
    p.add_argument('--new-head', action='store_true', help='1-class head instead of keeping the COCO head')
    p = sub.add_parser('eval-det')
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True, help='directory written by prep-visdrone')
    p.add_argument('--split', default='val')
    p.add_argument('--conf', type=float, default=0.15)
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--max-images', type=int, default=None)
    p = sub.add_parser('video')
    p.add_argument('--source', required=True)
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--out', dest='video_out', default='', help='annotated output video (.mp4)')
    p.add_argument('--conf', type=float, default=0.15)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--hfov', type=float, default=70.0)
    p.add_argument('--cam-height', type=float, default=1.5)
    p.add_argument('--pick', default='', help='x,y pixel of the person to follow (default: nearest to the center)')
    p = sub.add_parser('sim')
    p.add_argument('--pilots', default='gbl,direct')
    p.add_argument('--model', default='')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--seed0', type=int, default=2000)
    p.add_argument('--standoff', type=float, default=4.0)
    p.add_argument('--density', type=float, default=None)
    p.add_argument('--distractors', type=int, default=None)
    p.add_argument('--speed', type=float, default=None)
    p.add_argument('--det-calib', default='', help='JSON from eval-det (recall curve of the real detector)')
    p.add_argument('--tag', default='')
    p = sub.add_parser('dagger')
    p.add_argument('--envs', type=int, default=100)
    p.add_argument('--per-round', type=int, default=20)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--init', required=True)
    p.add_argument('--label-every', type=int, default=2)
    p.add_argument('--imagenet', action='store_true')
    for q in sub.choices.values():
        q.add_argument('--results', dest='out', default='', help='results directory (default results/r19)')
        q.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
        q.add_argument('--cam-hz', type=float, default=10.0)
    a = ap.parse_args()
    if a.cmd == 'sim':
        return cmd_sim(a)
    if a.cmd == 'prep-visdrone':
        return cmd_prep(a)
    if a.cmd == 'train-det':
        return cmd_train_det(a)
    if a.cmd == 'eval-det':
        return cmd_eval_det(a)
    if a.cmd == 'video':
        return cmd_video(a)
    from nanouav.agile import dagger
    # same directory as R18: the follow data (data_follow) is trained together with R18's data_train
    out = a.out or os.path.join(str(paths.results_dir()), 'r18')
    os.makedirs(out, exist_ok=True)
    m = dagger.run(out, n_envs=a.envs, per_round=a.per_round, workers=a.workers, epochs=a.epochs, batch=a.batch,
                   cam_hz=a.cam_hz, weights='imagenet' if a.imagenet else None, kind='follow',
                   init_model=os.path.abspath(a.init), label_every=a.label_every)
    print('student →', m)


if __name__ == '__main__':
    main()
