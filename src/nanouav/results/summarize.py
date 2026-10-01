"""
Summarizes all experiment JSON files in results/ with success rates and Wilson 95% CIs.
Usage: nanouav summarize [filter]      e.g. nanouav summarize r6_
"""
import glob, json, math, os, sys
from nanouav import paths

ROOT = str(paths.repo_root())


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(c - h, 0), 100 * min(c + h, 1))


def runs_of(obj):
    if isinstance(obj, dict) and 'runs' in obj and isinstance(obj['runs'], dict):
        obj = obj['runs']
    if isinstance(obj, dict):
        out = {k: v for k, v in obj.items() if isinstance(v, list) and v and isinstance(v[0], dict) and 'success' in v[0]}
        return out
    return {}


flt = sys.argv[1] if len(sys.argv) > 1 else ''
for f in sorted(glob.glob(os.path.join(str(paths.results_dir()), '*.json'))):
    name = os.path.basename(f)
    if flt not in name:
        continue
    try:
        obj = json.load(open(f))
    except Exception:
        continue
    R = runs_of(obj)
    if R:
        print(f'\n{name}')
        tk = tn = 0
        for cond, L in R.items():
            k, n = sum(bool(r['success']) for r in L), len(L)
            col = sum(bool(r.get('collided')) for r in L)
            tk, tn = tk + k, tn + n
            lo, hi = wilson(k, n)
            ts = sorted(r['time_s'] for r in L if r['success'] and isinstance(r.get('time_s'), (int, float)))
            tm = f'  t_med {ts[len(ts) // 2]:.1f} s' if ts else ''        # median time of the successes
            print(f'   {cond:>18s}: {k:3d}/{n:<3d} ({100 * k / n:5.1f} %, CI95 {lo:5.1f}–{hi:5.1f})  collisions {col}{tm}')
        lo, hi = wilson(tk, tn)
        print(f'   {"TOTAL":>18s}: {tk:3d}/{tn:<3d} ({100 * tk / max(tn, 1):5.1f} %, CI95 {lo:5.1f}–{hi:5.1f})')
    elif name.startswith('r3_') and isinstance(obj, dict):
        print(f'\n{name}')
        for m, v in obj.items():
            if isinstance(v, dict) and 'int8_test' in v:
                t, g = v['int8_test'], v['int8_gazebo']
                print(f"   {m:14s} test AbsRel {t['abs_rel']:.3f} minErr {t['min_dist_mae']:.3f} rec/prec<1m "
                      f"{t['obs1m_recall']:.2f}/{t['obs1m_precision']:.2f} | gazebo AbsRel {g['abs_rel']:.3f} "
                      f"rec/prec {g['obs1m_recall']:.2f}/{g['obs1m_precision']:.2f}")
