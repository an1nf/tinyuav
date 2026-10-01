"""Command-line interface: ``nanouav <command>`` (also ``python -m nanouav``).

    nanouav list [--tag sim]                     experiments in the registry
    nanouav run <id> [experiment args...]        run one experiment (``nanouav run r05 --help``)
    nanouav suite <file.yaml> [-j 7] [--resume]  run a suite of jobs in parallel
    nanouav summarize [--dir results/baseline]   success rates with Wilson 95% CI
    nanouav models list | check | import <dir>   manage trained networks (kept out of git)
    nanouav gazebo gen-world --scenario bosque   generate a Gazebo world from a shared scenario
    nanouav doctor                               check the environment (runtimes, GPU, models)
"""
from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path

from nanouav import __version__, paths
from nanouav.experiments import REGISTRY, get as get_experiment


def _run_module(module: str, argv: list[str]) -> int:
    old = sys.argv
    sys.argv = [module, *argv]
    try:
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    except SystemExit as e:                       # argparse --help / sys.exit dentro del experimento
        return int(e.code or 0) if not isinstance(e.code, str) else 1
    finally:
        sys.argv = old
    return 0


def cmd_list(a) -> int:
    rows = [e for e in REGISTRY.values() if not a.tag or a.tag in e.tags]
    w = max(len(e.id) for e in rows)
    for e in rows:
        needs = ','.join(n for n in e.needs if n != 'cpu')
        print(f'  {e.id:<{w}}  {e.summary}' + (f'  [{needs}]' if needs else ''))
    print(f'\n{len(rows)} experiments · `nanouav run <id> --help` for options')
    return 0


def cmd_run(a, rest) -> int:
    e = get_experiment(a.id)
    if a.results:
        os.environ['NANOUAV_RESULTS'] = str(Path(a.results).resolve())
    if a.fake_cnn:
        os.environ['NANOUAV_FAKE_CNN'] = '1'
    if a.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('MPLBACKEND', 'Agg')
    return _run_module(e.module, rest)


def cmd_suite(a) -> int:
    from nanouav.runner import run_suite
    return run_suite(a.file, workers=a.jobs, out=a.out, resume=a.resume, dry_run=a.dry_run, only=a.only)


def cmd_summarize(a) -> int:
    if a.dir:
        os.environ['NANOUAV_RESULTS'] = str(Path(a.dir).resolve())
    return _run_module('nanouav.results.summarize_gz' if a.gazebo else 'nanouav.results.summarize', a.rest)


MODEL_FILES = ('model_*.tflite', 'model_*.keras', 'model.tflite', 'metrics.json', 'r11_vel_*.keras')


def cmd_models(a) -> int:
    d = paths.models_dir()
    if a.action == 'import':
        src = Path(a.src).expanduser()
        if not src.is_dir():
            print(f'not a directory: {src}')
            return 1
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for pat in MODEL_FILES:
            for f in src.glob(pat):
                shutil.copy2(f, d / f.name)
                n += 1
                print(f'  {f.name}')
        if (src / 'r3').is_dir():
            shutil.copytree(src / 'r3', d / 'r3', dirs_exist_ok=True)
            print('  r3/')
        print(f'{n} files → {d}')
        return 0
    files = sorted(d.glob('*.tflite')) + sorted(d.glob('r3/*.tflite')) if d.exists() else []
    if not files:
        print(f'no models in {d}. Train them (`nanouav run train`) or import an existing folder:\n'
              f'  nanouav models import ~/Project_UAV/uav_v2/_03_cnn')
        return 1 if a.action == 'check' else 0
    for f in files:
        print(f'  {str(f.relative_to(d)):28s} {f.stat().st_size / 1024:7.1f} KB')
    if a.action == 'check':
        from nanouav.perception import cnn
        ok = cnn.is_real_mode('M')
        print('  M loads with the TFLite runtime' if ok else '  M could NOT be loaded (runtime missing?)')
        return 0 if ok else 1
    return 0


def cmd_gazebo(a, rest) -> int:
    tool = paths.gazebo_dir() / 'tools' / 'gen_world.py'
    old = sys.argv
    sys.argv = [str(tool), *rest]
    try:
        runpy.run_path(str(tool), run_name='__main__')
    finally:
        sys.argv = old
    return 0


def cmd_doctor(a) -> int:
    import importlib.util as iu
    print(f'nanouav {__version__} · python {sys.version.split()[0]}')
    print(f'  repo     {paths.repo_root()}')
    for name, fn in (('models', paths.models_dir), ('data', paths.data_dir), ('results', paths.results_dir)):
        p = fn(create=False)
        print(f'  {name:8s} {p} {"" if p.exists() else "(missing)"}')
    tfl = sorted(f.name for f in paths.models_dir().glob('model_*.tflite')) if paths.models_dir().exists() else []
    print(f'  networks {", ".join(tfl) if tfl else "none — `nanouav models import <dir>` or `nanouav run train`"}')
    for mod in ('numpy', 'cv2', 'matplotlib', 'yaml', 'ai_edge_litert', 'tflite_runtime', 'tensorflow'):
        print(f'  {mod:15s} {"ok" if iu.find_spec(mod) else "-"}')
    if iu.find_spec('tensorflow') and a.gpu:
        import tensorflow as tf
        print('  GPUs:', tf.config.list_physical_devices('GPU'))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog='nanouav', description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--version', action='version', version=__version__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('list', help='list experiments')
    p.add_argument('--tag')
    sub.add_parser('run', help='run one experiment: nanouav run [--results DIR] [--fake-cnn] [--cpu] <id> [args]')
    p = sub.add_parser('suite', help='run a YAML suite in parallel')
    p.add_argument('file')
    p.add_argument('-j', '--jobs', type=int)
    p.add_argument('--out')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--only', help='run only jobs whose id contains this text')
    p = sub.add_parser('summarize', help='summarise result files')
    p.add_argument('--dir')
    p.add_argument('--gazebo', action='store_true', help='Gazebo episodes (results/gz_runs)')
    p.add_argument('rest', nargs=argparse.REMAINDER)
    p = sub.add_parser('models', help='list / check / import trained models')
    p.add_argument('action', choices=['list', 'check', 'import'])
    p.add_argument('src', nargs='?')
    sub.add_parser('gazebo', help='Gazebo tools (gen-world)', add_help=False)
    p = sub.add_parser('doctor', help='environment check')
    p.add_argument('--gpu', action='store_true', help='also query TensorFlow GPUs')

    # `run` and `gazebo` forward every remaining argument untouched to the experiment / tool
    if argv and argv[0] == 'gazebo':
        return cmd_gazebo(None, argv[2:] if argv[1:2] == ['gen-world'] else argv[1:])
    if argv and argv[0] == 'run':
        if len(argv) < 2 or argv[1] in ('-h', '--help'):
            print('usage: nanouav run [--results DIR] [--fake-cnn] [--cpu] <id> [experiment args...]\n'
                  '       nanouav list   (available ids)')
            return 0
        ns = argparse.Namespace(results=None, fake_cnn=False, cpu=False, id=None)
        opts = argv[1:]
        while opts and opts[0] in ('--results', '--fake-cnn', '--cpu'):
            o = opts.pop(0)
            if o == '--results':
                ns.results = opts.pop(0)
            else:
                setattr(ns, o[2:].replace('-', '_'), True)
        ns.id = opts.pop(0)
        return cmd_run(ns, opts)
    a = ap.parse_args(argv)
    return {'list': cmd_list, 'suite': cmd_suite, 'summarize': cmd_summarize, 'models': cmd_models,
            'doctor': cmd_doctor}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
