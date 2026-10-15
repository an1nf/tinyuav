"""Perception variants for the 'sgm' clearance, judged against the truth on the reactive pilot's own flights.

The flight uses the baseline perception; at every decision, each variant judges the same 33 candidates.
FA: candidate truly free (>= 0.35) seen blocked; MISS: truly colliding (< 0.2) seen free (>= 0.35)."""
import dataclasses
import json
import sys

import numpy as np

from nanouav.agile import envs as EV
from nanouav.agile import rollout as RO
from nanouav.agile import trajectory as TR
from nanouav.agile.stereo import StereoRig


def edge_filter(depth, tau0=0.3, gamma=0.03, erode=True):
    d = np.asarray(depth, np.float32)
    valid = d > 0
    bad = np.zeros_like(valid)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        n = np.roll(d, (dy, dx), (0, 1))
        nv = n > 0
        tau = np.maximum(tau0, gamma * d * d)
        bad |= valid & nv & (np.abs(d - n) > tau)
        if erode:
            bad |= valid & ~nv
    return np.where(bad, 0.0, d)


VARIANTS = {
    'base': dict(pool='stride', edge=False, hits=2),
    'min': dict(pool='min', edge=False, hits=2),
    'edge': dict(pool='stride', edge=True, hits=2),
    'edge_min': dict(pool='min', edge=True, hits=2),
    'hits3': dict(pool='stride', edge=False, hits=3),
    'edge_hits3': dict(pool='stride', edge=True, hits=3),
}

orig_points = RO.sgm_points


def main(v, seeds, tilt=True):
    stats = {k: dict(fa=0, n_free=0, miss=0, n_col=0, blind=0, bad_pick=0, n_dec=0) for k in VARIANTS}
    results = []
    for seed in seeds:
        env = EV.TEST['forest'](seed)
        rig = dataclasses.replace(StereoRig(), tilt_up_deg=30.0 if (tilt and v >= 7) else 0.0)
        R = RO.Rollout(env, RO.Config(v_des=v, cam_hz=15.0, seed=seed, rig=rig), 'reactive')
        R.KEEP = True
        obs = {k: RO.PerceivedObstacles(n_frames=15, min_hits=c['hits']) for k, c in VARIANTS.items()}

        def points(depth, rig_, pose, stride=4, pool='stride'):
            for k, c in VARIANTS.items():
                dd = edge_filter(depth) if c['edge'] else depth
                obs[k].add(orig_points(dd, rig_, pose, pool=c['pool']))
            return orig_points(depth, rig_, pose, stride, pool)

        RO.sgm_points = points
        orig_fan = R._fan_choice

        def fan_choice(p, v_, a, omega, t, clear_of, source='student'):
            Wf = RO.fan_points(p, v_, omega, R._v_des, (R.z_ground + 0.7, float(R.env.start[2]) + 1.0))
            fits = [R._fit(P, p, v_, a, R._v_des) for P in Wf]
            tt = np.linspace(0, 1, 21)
            X = [TR.poly_eval(c, tt) for c, _ in fits]
            true = np.array([float(R.field(x).min()) for x in X])
            J = RO.fan_cost(Wf, R._blind, None)
            for k in VARIANTS:
                seen = np.array([obs[k].clearance(x) for x in X])
                s = stats[k]
                tf, tc, sf = true >= 0.35, true < 0.2, seen >= 0.35
                s['fa'] += int((tf & ~sf).sum()); s['n_free'] += int(tf.sum())
                s['miss'] += int((tc & sf).sum()); s['n_col'] += int(tc.sum())
                s['n_dec'] += 1
                if tf.any() and not sf.any():
                    s['blind'] += 1
                if sf.any():
                    kk = int(np.where(sf)[0][np.argmin(J[sf])])
                    s['bad_pick'] += int(true[kk] < 0.2)
            return orig_fan(p, v_, a, omega, t, clear_of, source)

        R._fan_choice = fan_choice
        r, _, _ = R.run()
        RO.sgm_points = orig_points
        results.append((seed, r['success'], r['time_s']))
    return stats, results


if __name__ == '__main__':
    v = float(sys.argv[1])
    seeds = [int(x) for x in sys.argv[2].split(',')]
    stats, res = main(v, seeds)
    print('flights', res)
    print(f'{"variant":12s} {"FA":>6s} {"MISS":>6s} {"blind":>6s} {"bad_pick":>8s} {"n_dec":>6s}')
    for k, s in stats.items():
        print(f'{k:12s} {s["fa"] / max(s["n_free"], 1):6.3f} {s["miss"] / max(s["n_col"], 1):6.3f} '
              f'{s["blind"] / max(s["n_dec"], 1):6.3f} {s["bad_pick"] / max(s["n_dec"], 1):8.4f} {s["n_dec"]:6d}')
    json.dump(stats, open(f'percep_v{v:g}.json', 'w'))
