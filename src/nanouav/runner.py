"""Suite runner: run many experiment jobs in parallel from a YAML file.

Suite file format (see ``configs/suites/``)::

    name: baseline                      # results go to results/<name>/ unless --out is given
    description: Paper C2 baseline, M vs M_ToF
    env: {OMP_NUM_THREADS: "1"}         # applied to every job (optional)
    matrix: {variant: [M, M_ToF]}       # optional: every job is expanded over the cartesian product
    jobs:
      - id: e3-nano-{variant}           # unique id; {placeholders} come from the matrix / job vars
        run: e3 --profile nano --n 30 --variant {variant}
      - id: r05-gt
        run: r05 --profile nano --law consistent --n 10
        env: {NANOUAV_FAKE_CNN: "1"}
        matrix: {}                      # a job-level matrix overrides the suite matrix ({} = no expansion)

Each job runs as ``python -m nanouav run <run>`` in its own process (one CPU core each by default,
GPU hidden unless the job sets ``gpu: true``), with stdout/stderr in ``<out>/logs/<id>.log`` and its
status in ``<out>/jobs.json``. ``--resume`` skips jobs that already finished with exit code 0.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from nanouav import paths
from nanouav.experiments import get as get_experiment

THREAD_ENV = {'OMP_NUM_THREADS': '1', 'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1',
              'TF_CPP_MIN_LOG_LEVEL': '3', 'MPLBACKEND': 'Agg'}


@dataclass
class Job:
    id: str
    run: str
    env: dict = field(default_factory=dict)
    gpu: bool = False

    @property
    def argv(self) -> list[str]:
        return shlex.split(self.run)


def _expand(template: dict, matrix: dict) -> list[dict]:
    if not matrix:
        return [dict(template)]
    keys = list(matrix)
    out = []
    for combo in itertools.product(*(matrix[k] if isinstance(matrix[k], list) else [matrix[k]] for k in keys)):
        vals = dict(zip(keys, map(str, combo)))
        j = {k: (v.format(**vals) if isinstance(v, str) else v) for k, v in template.items()}
        if 'env' in template:
            j['env'] = {k: str(v).format(**vals) for k, v in template['env'].items()}
        if j.get('id') == template.get('id'):            # id sin placeholders → sufijo automático
            j['id'] = template['id'] + '-' + '-'.join(vals.values())
        out.append(j)
    return out


def load_suite(path: str | Path) -> tuple[dict, list[Job]]:
    import yaml
    spec = yaml.safe_load(Path(path).read_text())
    if not isinstance(spec, dict) or 'jobs' not in spec:
        raise ValueError(f'{path}: a suite needs a top-level "jobs" list')
    jobs, seen = [], set()
    for raw in spec['jobs']:
        if isinstance(raw, str):
            raw = {'run': raw}
        raw.setdefault('id', raw['run'].split()[0])
        matrix = raw.pop('matrix', spec.get('matrix', {}))
        for j in _expand(raw, matrix):
            job = Job(id=re.sub(r'[^A-Za-z0-9_.-]+', '_', j['id']).strip('_'), run=j['run'], env={**spec.get('env', {}), **j.get('env', {})},
                      gpu=bool(j.get('gpu', False)))
            get_experiment(job.argv[0])                  # valida el id antes de lanzar nada
            if job.id in seen:
                raise ValueError(f'{path}: duplicated job id {job.id!r}')
            seen.add(job.id)
            jobs.append(job)
    return spec, jobs


def run_suite(path: str | Path, workers: int | None = None, out: str | None = None, resume: bool = False,
              dry_run: bool = False, only: str | None = None) -> int:
    spec, jobs = load_suite(path)
    if only:
        jobs = [j for j in jobs if only in j.id]
    out_dir = Path(out) if out else paths.results_dir() / spec.get('name', Path(path).stem)
    (out_dir / 'logs').mkdir(parents=True, exist_ok=True)
    state_f = out_dir / 'jobs.json'
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    todo = [j for j in jobs if not (resume and state.get(j.id, {}).get('rc') == 0)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f'[suite] {spec.get("name", path)}: {len(jobs)} jobs ({len(jobs) - len(todo)} already done) · '
          f'{workers} in parallel · results in {out_dir}')
    if dry_run:
        for j in todo:
            env = ' '.join(f'{k}={v}' for k, v in j.env.items())
            print(f'  {j.id:32s} {env} nanouav run {j.run}')
        return 0
    lock = threading.Lock()

    def _one(job: Job) -> int:
        env = {**os.environ, **THREAD_ENV, **job.env, 'NANOUAV_RESULTS': str(out_dir)}
        if not job.gpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        log = out_dir / 'logs' / f'{job.id}.log'
        t0 = time.time()
        with lock:
            print(f'[start] {job.id}', flush=True)
        with open(log, 'w') as fh:
            rc = subprocess.call([sys.executable, '-m', 'nanouav', 'run', *job.argv], env=env,
                                 stdout=fh, stderr=subprocess.STDOUT, cwd=str(paths.repo_root()))
        dt = time.time() - t0
        with lock:
            state[job.id] = {'run': job.run, 'env': job.env, 'rc': rc, 'seconds': round(dt, 1),
                             'finished': time.strftime('%Y-%m-%d %H:%M:%S')}
            state_f.write_text(json.dumps(state, indent=1))
            print(f'[{"done" if rc == 0 else "FAIL"}] {job.id} ({dt:.0f} s, rc={rc}) → {log.name}', flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rcs = list(ex.map(_one, todo))
    failed = [j.id for j, rc in zip(todo, rcs) if rc != 0]
    print(f'[suite] finished · {len(todo) - len(failed)} ok · {len(failed)} failed' +
          (f': {", ".join(failed)}' if failed else '') + f'\n        summary: nanouav summarize --dir {out_dir}')
    return 1 if failed else 0
