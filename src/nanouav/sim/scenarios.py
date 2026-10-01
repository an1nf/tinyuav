"""
Escenarios compartidos Gazebo ↔ simulador propio (una sola fuente de verdad).

Cada escenario es un dict JSON-serializable:
  {name, seed, A:[x,y], B:[x,y], yaw0, z_cruise, ground:{kind, albedo, albedo2, scale},
   objects:[{type:'box', center:[x,y,z], half:[hx,hy,hz], yaw, tex:{...}} |
            {type:'cyl', c:[x,y], r, z0, z1, tex:{...}}],
   dynamic:[{name, r, p0:[x,y,z], axis:[ux,uy], amp, v}], t_max}
tex = {kind: uniform|checker|noise|stripes, albedo, albedo2, scale, seed} (misma semántica que
_00_scene.renderer.Texture → las imágenes de Gazebo se generan con el MISMO muestreador).

gen_world.py lo convierte a SDF (+ texturas PNG); to_world() lo convierte a World del renderer
propio para correr exactamente el mismo mundo offline (g1_gazebo_world.py --world json:…).
"""
import math
import numpy as np

A0 = (-1.5, 0.0)


def _tex(rng, kinds=('noise', 'stripes', 'uniform', 'checker'), p=(0.45, 0.2, 0.2, 0.15)):
    k = str(rng.choice(kinds, p=p))
    a = float(rng.uniform(0.25, 0.9))
    return {'kind': k, 'albedo': a, 'albedo2': float(a * rng.uniform(0.3, 0.7)),
            'scale': float(rng.uniform(0.15, 0.5)), 'seed': int(rng.integers(1 << 30))}


def _ground(rng, kind=None):
    k = kind or str(rng.choice(['noise', 'checker', 'noise'], p=[0.5, 0.3, 0.2]))
    a = float(rng.uniform(0.45, 0.85))
    return {'kind': k, 'albedo': a, 'albedo2': float(a * rng.uniform(0.3, 0.6)),
            'scale': float(rng.uniform(0.5, 1.2)) if k == 'checker' else float(rng.uniform(0.3, 0.9)),
            'seed': int(rng.integers(1 << 30))}


def _box(c, h, yaw=0.0, tex=None):
    return {'type': 'box', 'center': [float(v) for v in c], 'half': [float(v) for v in h],
            'yaw': float(yaw), 'tex': tex or {'kind': 'uniform', 'albedo': 0.5}}


def _cyl(x, y, r, z1, tex=None, z0=0.0):
    return {'type': 'cyl', 'c': [float(x), float(y)], 'r': float(r), 'z0': float(z0), 'z1': float(z1),
            'tex': tex or {'kind': 'uniform', 'albedo': 0.5}}


def _base(name, seed, B, rng, t_max=90.0, ground=None):
    B = np.asarray(B, float)
    return {'name': name, 'seed': int(seed), 'A': list(A0), 'B': [float(B[0]), float(B[1])],
            'yaw0': float(math.atan2(B[1] - A0[1], B[0] - A0[0])), 'z_cruise': 0.65,
            'ground': ground or _ground(rng), 'objects': [], 'dynamic': [], 't_max': t_max}


# ── escenarios ────────────────────────────────────────────────────────────
def obstaculos(seed=0, **kw):
    """El mundo original de Gazebo (pared de 3 cajas + cilindro), con texturas."""
    rng = np.random.default_rng(seed)
    s = _base('obstaculos', seed, (4.5, 0.0), rng, t_max=90)
    s['objects'] = [_box((1.8, 0.0, 0.3), (0.2, 0.2, 0.3), 0, _tex(rng)),
                    _box((1.2, -1.0, 0.25), (0.175, 0.175, 0.25), 0, _tex(rng)),
                    _box((1.0, 1.1, 0.4), (0.15, 0.15, 0.4), 0, _tex(rng)),
                    _cyl(3.0, 0.4, 0.18, 0.7, _tex(rng))]
    return s


def bosque(seed=0, density=0.35, gap=1.6, length=9.0, width=4.0, **kw):
    """Troncos (r 0.06–0.25 m, 2–3 m) por muestreo de disco de Poisson: separación mínima entre
    centros `gap` (1.6 m deja paso al X3; 1.0 al nano). density = troncos/m² objetivo."""
    rng = np.random.default_rng(seed)
    s = _base('bosque', seed, (length + 1.0, 0.0), rng, t_max=120)
    target = int(density * length * 2 * width)
    pts, tries = [], 0
    while len(pts) < target and tries < 20000:
        tries += 1
        x, y = rng.uniform(0.3, length), rng.uniform(-width, width)
        if math.hypot(x - A0[0], y - A0[1]) < 1.3 or math.hypot(x - s['B'][0], y - s['B'][1]) < 1.3:
            continue
        if all(math.hypot(x - a, y - b) >= gap for a, b in pts):
            pts.append((x, y))
    for x, y in pts:
        s['objects'].append(_cyl(x, y, rng.uniform(0.06, 0.25), rng.uniform(2.0, 3.0),
                                 _tex(rng, ('noise', 'stripes', 'uniform'), (0.6, 0.25, 0.15))))
    return s


def rendijas(seed=0, width=1.5, n_walls=2, **kw):
    """Slalom de paredes con una abertura cada una, alternando lado (el carril recto está cerrado)."""
    rng = np.random.default_rng(seed)
    s = _base('rendijas', seed, (2.5 * n_walls + 2.5, 0.0), rng, t_max=120)
    for i in range(n_walls):
        x = 1.5 + 2.5 * i
        yc = (1 if i % 2 == 0 else -1) * rng.uniform(0.9, 1.6)
        t = _tex(rng, ('noise', 'stripes', 'checker'), (0.5, 0.3, 0.2))
        lo, hi = yc - width / 2, yc + width / 2
        for a, b in ((-6.0, lo), (hi, 6.0)):
            if b - a > 0.05:
                s['objects'].append(_box((x, (a + b) / 2, 1.0), (0.1, (b - a) / 2, 1.0), 0, t))
    return s


def interior(seed=0, door=1.2, **kw):
    """Dos habitaciones (8 × 6 m) unidas por una puerta; muebles bajos y altos. Meta en la otra sala."""
    rng = np.random.default_rng(seed)
    s = _base('interior', seed, (7.5, rng.uniform(-1.5, 1.5)), rng, t_max=150,
              ground=_ground(rng, 'checker'))
    wall = _tex(rng, ('noise', 'stripes', 'uniform'), (0.5, 0.2, 0.3))
    X0, X1, Y = -3.0, 9.5, 3.0
    s['objects'] += [_box(((X0 + X1) / 2, Y + 0.1, 1.2), ((X1 - X0) / 2, 0.1, 1.2), 0, wall),
                     _box(((X0 + X1) / 2, -Y - 0.1, 1.2), ((X1 - X0) / 2, 0.1, 1.2), 0, wall),
                     _box((X0 - 0.1, 0, 1.2), (0.1, Y, 1.2), 0, wall),
                     _box((X1 + 0.1, 0, 1.2), (0.1, Y, 1.2), 0, wall)]
    xw, yd = 3.2, rng.uniform(-1.6, 1.6)                 # tabique con puerta
    lo, hi = yd - door / 2, yd + door / 2
    s['objects'] += [_box((xw, (-Y + lo) / 2, 1.2), (0.08, (lo + Y) / 2, 1.2), 0, wall),
                     _box((xw, (hi + Y) / 2, 1.2), (0.08, (Y - hi) / 2, 1.2), 0, wall)]
    furn = []
    for _ in range(int(rng.integers(3, 6))):              # mesas/estantes/cajas
        for _t in range(50):
            x = rng.uniform(X0 + 0.8, X1 - 0.8)
            y = rng.uniform(-Y + 0.6, Y - 0.6)
            if abs(x - xw) < 1.3 or math.hypot(x - A0[0], y) < 1.4 or math.hypot(x - s['B'][0], y - s['B'][1]) < 1.3:
                continue
            if all(math.hypot(x - a, y - b) > 1.6 for a, b in furn):
                break
        furn.append((x, y))
        hz = float(rng.choice([0.38, 0.9, 1.0]))          # mesa (se sobrevuela? no: 0.76 m) / estante
        s['objects'].append(_box((x, y, hz), (rng.uniform(0.2, 0.5), rng.uniform(0.2, 0.5), hz),
                                 rng.uniform(-0.5, 0.5), _tex(rng)))
    return s


def corredor(seed=0, **kw):
    """El mundo aleatorio de E3 (random_corridor_world) con la misma semilla → comparable con E3."""
    rng = np.random.default_rng(seed)
    s = _base('corredor', seed, (7.0, 0.0), rng, t_max=90)
    objs, placed = [], []
    n = int(rng.integers(2, 5))
    for k in range(n):
        for _ in range(40):
            x = rng.uniform(1.2, 5.5)
            y = rng.uniform(-0.2, 0.2) if k == 0 else rng.uniform(-1.6, 1.6)
            if all(abs(x - a) > 1.3 or abs(y - b) > 1.4 for a, b in placed):
                break
        placed.append((x, y))
        t = _tex(rng)
        if rng.random() < 0.5:
            hx, hy = rng.uniform(0.1, 0.3, 2)
            objs.append(_box((x, y, 0.75), (hx, hy, 0.75), rng.uniform(-1, 1), t))
        else:
            objs.append(_cyl(x, y, rng.uniform(0.08, 0.3), rng.uniform(1.0, 2.0), t))
    if rng.random() < 0.5:
        objs.append(_box((rng.uniform(1, 5), rng.uniform(-0.3, 0.3), 0.1), (0.2, 0.2, 0.1), 0.0, _tex(rng)))
    s['objects'] = objs
    return s


def dinamico(seed=0, v_ball=1.0, **kw):
    """Obstáculos estáticos ligeros + balón (r 0.15) que cruza el carril de lado a lado en x ≈ 3."""
    rng = np.random.default_rng(seed)
    s = _base('dinamico', seed, (6.5, 0.0), rng, t_max=90)
    s['objects'] = [_cyl(1.5, rng.uniform(-1.5, -0.8), 0.15, 2.0, _tex(rng)),
                    _cyl(5.0, rng.uniform(0.8, 1.5), 0.15, 2.0, _tex(rng))]
    s['dynamic'] = [{'name': 'balon', 'r': 0.15, 'p0': [3.0, -2.0, 0.65], 'axis': [0.0, 1.0],
                     'amp': 4.0, 'v': float(v_ball), 'tex': {'kind': 'uniform', 'albedo': 0.85}}]
    return s


def estrechos(seed=0, w=1.0, **kw):
    """Pasos estrechos SIN rodeo posible: puerta en una pared continua (±9 m), pasillo de 2.5 m de largo
    y segunda puerta desplazada. w = ancho libre (m). X3 (Ø 0.72 m): w = 1.0 deja 0.14 m por lado;
    nano (Ø 0.22 m): usar w = 0.45."""
    rng = np.random.default_rng(seed)
    s = _base('estrechos', seed, (9.0, 0.0), rng, t_max=150)
    t1 = _tex(rng, ('noise', 'stripes', 'checker'), (0.5, 0.3, 0.2))
    t2 = _tex(rng, ('noise', 'stripes', 'checker'), (0.5, 0.3, 0.2))
    H, Y = 1.1, 16.0                                                  # más allá de la grilla: sin rodeo

    def wall_with_gap(x, yc, width, t, thick=0.1):
        lo, hi = yc - width / 2, yc + width / 2
        for a, b in ((-Y, lo), (hi, Y)):
            s['objects'].append(_box((x, (a + b) / 2, H), (thick, (b - a) / 2, H), 0, t))
    y1 = float(rng.uniform(-0.3, 0.3))
    wall_with_gap(1.2, y1, w, t1)                                     # 1) puerta
    yc = float(rng.uniform(-0.4, 0.4))                                # 2) pasillo x ∈ [3.2, 5.7]
    x0, x1 = 3.2, 5.7
    wall_with_gap(x0, yc, w, t2)
    for sgn in (-1, 1):
        s['objects'].append(_box(((x0 + x1) / 2, yc + sgn * (w / 2 + 0.05), H), ((x1 - x0) / 2, 0.05, H), 0, t2))
    wall_with_gap(x1, yc, w, t2)
    wall_with_gap(7.5, -yc + float(rng.uniform(-0.3, 0.3)), w, t1)   # 3) puerta desplazada
    return s


SCENARIOS = {'obstaculos': obstaculos, 'bosque': bosque, 'rendijas': rendijas,
             'interior': interior, 'corredor': corredor, 'dinamico': dinamico,
             'estrechos': estrechos}


def make(name, seed=0, **kw):
    return SCENARIOS[name](seed=seed, **kw)


# ── al renderer propio ─────────────────────────────────────────────────────
def to_world(s, lighting=None):
    """Convierte el escenario en World del simulador propio (sin los dinámicos, que se animan aparte)."""
    from nanouav.scene.renderer import World, Box, Cylinder, Plane, Texture
    T = lambda d: Texture(d.get('kind', 'uniform'), albedo=d.get('albedo', 0.5), albedo2=d.get('albedo2', 0.3),
                          scale=d.get('scale', 1.0), seed=d.get('seed', 0))
    objs = []
    for o in s['objects']:
        if o['type'] == 'box':
            objs.append(Box(tuple(o['center']), tuple(o['half']), o.get('yaw', 0.0), T(o['tex'])))
        else:
            objs.append(Cylinder(o['c'][0], o['c'][1], o['r'], o['z0'], o['z1'], T(o['tex'])))
    return World(objs, Plane(T(s['ground'])), **(lighting or {}))


def clearance(s, p, dyn_pos=None):
    """Distancia (m) del punto p (3,) al obstáculo más cercano (estáticos + dinámicos)."""
    p = np.asarray(p, float)
    best = np.inf
    for o in s['objects']:
        if o['type'] == 'box':
            c, h, yaw = np.asarray(o['center']), np.asarray(o['half']), o.get('yaw', 0.0)
            cy, sy = math.cos(-yaw), math.sin(-yaw)
            d = p - c
            q = np.abs(np.array([cy * d[0] - sy * d[1], sy * d[0] + cy * d[1], d[2]])) - h
            dist = float(np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0.0))
        else:
            dr = math.hypot(p[0] - o['c'][0], p[1] - o['c'][1]) - o['r']
            dz = max(o['z0'] - p[2], p[2] - o['z1'], -1e9)
            dist = math.hypot(max(dr, 0), max(dz, 0)) + min(max(dr, dz), 0.0)
        best = min(best, dist)
    for d in (dyn_pos or []):
        best = min(best, float(np.linalg.norm(p - np.asarray(d[0]))) - d[1])
    return best
