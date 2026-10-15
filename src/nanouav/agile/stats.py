"""Paired comparison of evaluation runs over the same environments (`nanouav run r22 compare`).

Two runs of the same student over the same environments disagree on a third of them (R22, 0.23.0): one
run of 90 flights only shows differences of ~12 arrivals. The comparison is therefore paired by
environment (exact McNemar test on the discordant ones), and the crash rate is given per second and per
metre of flight (in R22 it is constant per second across speeds).
"""
import json
import math
import os


def mcnemar(b, c):
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(fn):
    """eval JSON ({'student_v3': [result, …], …}) → {(key, environment index): result}."""
    T = json.load(open(fn))
    return {(key, int(r.get('line', i))): r for key, R in T.items() for i, r in enumerate(R)}


def rates(results):
    """→ (crashes per second of flight, crashes per metre)."""
    k = sum(bool(r['crashed']) for r in results)
    T = sum(r['time_s'] for r in results)
    D = sum(r['time_s'] * r.get('mean_speed', 0.0) for r in results)
    return (k / T if T else float('nan')), (k / D if D else float('nan'))


def _speed(key):
    try:
        return float(key.rsplit('_v', 1)[1])
    except (IndexError, ValueError):
        return 0.0


def compare(fns, out=print):
    """Each run's arrivals and crash rates per key (pilot and speed), then every run against the first, paired
    by environment over the environments both have. → {name: {key or 'total': (only this, only first, p)}}."""
    runs = [load(f) for f in fns]
    names = [os.path.basename(f) for f in fns]
    keys = sorted({k for k, _ in runs[0]}, key=lambda k: (k.rsplit('_v', 1)[0], _speed(k)))
    w = max(len(n) for n in names)
    out(f'{"":{w + 3}s}' + ''.join(f'{k:>14s}' for k in keys) + f'{"total":>10s}   crash %/s · %/m')
    for j, (n, R) in enumerate(zip(names, runs)):
        cells, tot, rr = [], [0, 0], []
        for k in keys:
            res = [r for (kk, _), r in R.items() if kk == k]
            s = sum(bool(r['success']) for r in res)
            cells.append(f'{s:>9d}/{len(res):<4d}')
            tot[0] += s
            tot[1] += len(res)
            ps, pm = rates(res)
            rr.append(f'{100 * ps:.1f}·{100 * pm:.2f}')
        out(f'{chr(65 + j)}  {n:{w}s}' + ''.join(cells) + f'{tot[0]:>6d}/{tot[1]:<3d}   ' + '  '.join(rr))
    summary = {}
    for j in range(1, len(runs)):
        row, b_t, c_t = {}, 0, 0
        for k in keys:
            envs = sorted(e for kk, e in runs[0] if kk == k and (kk, e) in runs[j])
            b = sum(runs[j][(k, e)]['success'] and not runs[0][(k, e)]['success'] for e in envs)
            c = sum(runs[0][(k, e)]['success'] and not runs[j][(k, e)]['success'] for e in envs)
            row[k] = (b, c, mcnemar(b, c))
            b_t, c_t = b_t + b, c_t + c
        row['total'] = (b_t, c_t, mcnemar(b_t, c_t))
        summary[names[j]] = row
        out(f'{chr(65 + j)} vs A, environments only {chr(65 + j)} / only A arrives (McNemar p): ' +
            ' · '.join(f'{k} {b}/{c} ({p:.2f})' for k, (b, c, p) in row.items()))
    return summary
