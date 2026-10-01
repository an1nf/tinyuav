#!/usr/bin/env python3
"""
Resumen de episodios de Gazebo (assets/gz_runs/*.json) con IC95 de Wilson, y al lado el resultado del
simulador propio en el MISMO escenario si existe (assets/g1_scn-<mundo>_<planificador>_*.json).
  python3 gazebo/summarize_gz.py [--tag batch_0930_1800]
"""
import os, sys, json, glob, math, argparse
from collections import defaultdict

from nanouav import paths

ROOT = str(paths.repo_root())


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def med(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else float('nan')


def offline(world, planner, profile):
    out = []
    for f in glob.glob(os.path.join(str(paths.results_dir()), f'g1_scn-{world}_{planner}_{profile}_*.json')):
        for L in json.load(open(f)).values():
            out += [r['success'] for r in L]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(str(paths.results_dir()), 'gz_runs', f'*{a.tag}*.json')))
    G = defaultdict(list)
    for f in files:
        r = json.load(open(f))
        G[(r['world'], r['planner'], r['variant'], r['profile'])].append(r)
    if not G:
        print('sin resultados en assets/gz_runs'); return
    print(f"{'mundo':12s} {'plan':5s} {'CNN':6s} {'perfil':5s} {'éxito':>7s} {'IC95':>13s} {'col':>4s} {'t_med':>6s} "
          f"{'camino':>6s} {'holg_mín':>8s} {'RTF':>5s} │ {'offline':>9s}")
    for (w, p, v, pr), L in sorted(G.items()):
        n, k = len(L), sum(r['success'] for r in L)
        lo, hi = wilson(k, n)
        col = sum(r['collisions'] > 0 for r in L)
        tm = med([r['time_s'] for r in L if r['success']])
        pm = med([r['path_m'] for r in L if r['success']])
        cm = med([r['min_clearance_m'] for r in L])
        rtf = med([x for r in L for x in r.get('rtf', [])])
        off = offline(w, p, pr)
        off_s = f'{sum(off)}/{len(off)}' if off else '—'
        print(f'{w:12s} {p:5s} {v:6s} {pr:5s} {k:3d}/{n:<3d} [{lo:.2f}, {hi:.2f}] {col:4d} {tm:6.1f} {pm:6.1f} '
              f'{cm:8.2f} {rtf:5.2f} │ {off_s:>9s}')
    print('\nresultado por episodio: ' + ', '.join(sorted({r["result"] for L in G.values() for r in L})))


if __name__ == '__main__':
    main()
