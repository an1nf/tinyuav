# Experimentos y suites

## Registro

Cada experimento es un módulo ejecutable (`python -m <módulo>`) declarado en
`src/nanouav/experiments/__init__.py` con un id corto, un resumen, lo que necesita y el nombre que
tenía en la estructura plana:

```python
Experiment('r05', 'nanouav.experiments.r05_pole', 'Single-pole experiment: speed sweep, brake vs consistent law',
           tags=('sim', 'planning'), legacy='r5_pole.py'),
```

```bash
nanouav list                 # todos
nanouav list --tag sim       # por etiqueta (sim, planning, perception, dynamic, training, gazebo, ...)
nanouav run r05 --help       # opciones propias del experimento
```

Los ids `e*` corresponden al plan de evaluación original (E0–E4) y los `r*` a las iteraciones de
investigación R1–R15 (`docs/research/`). `nanouav run` también acepta el nombre antiguo del script
(`nanouav run r6_corridor_consistent.py ...`) o el del módulo.

## Ejecutar uno

```
nanouav run [--results DIR] [--fake-cnn] [--cpu] <id> [argumentos del experimento...]
```

| opción | efecto |
|---|---|
| `--results DIR` | escribe en `DIR` en lugar de `results/` (equivale a `NANOUAV_RESULTS=DIR`) |
| `--fake-cnn` | profundidad GT + ruido en vez de la red int8 (`NANOUAV_FAKE_CNN=1`) |
| `--cpu` | oculta la GPU (`CUDA_VISIBLE_DEVICES=`) |

Todo lo que va después del id se pasa sin tocar al experimento:

```bash
nanouav run e3 --profile nano --n 30 --variant M_ToF
nanouav run r12 --world "scn:estrechos;w=1.0" --goals 0:0 --planner mh,mh1 --n 10
nanouav run --fake-cnn r05 --profile nano --law consistent --n 10
NANOUAV_FUSE2=1 nanouav run r06 --profile x3 --law consistent --speeds 1.0,1.5 --n 30
```

## Suites

Una suite es un YAML en `configs/suites/` con una lista de trabajos. Cada trabajo es una línea de
`nanouav run`, con su id, variables de entorno propias y, opcionalmente, una matriz que lo expande
en el producto cartesiano de sus valores:

```yaml
name: baseline                      # resultados en results/<name>/ (o --out)
description: Paper C2 baseline, M vs M_ToF
env: {OMP_NUM_THREADS: "1"}         # para todos los trabajos (opcional)
matrix: {variant: [M, M_ToF]}       # se aplica a cada trabajo...
jobs:
  - id: e3-nano-{variant}           # id único; {placeholders} vienen de la matriz
    run: e3 --profile nano --n 30 --variant {variant}
  - id: r05-gt
    run: r05 --profile nano --law consistent --n 10
    env: {NANOUAV_FAKE_CNN: "1"}
    matrix: {}                      # ...salvo que el trabajo defina la suya ({} = sin expandir)
```

Si un id no tiene placeholders, la expansión le añade los valores como sufijo (`r06-brake-M`). Los
ids se validan antes de lanzar nada: un experimento inexistente o un id repetido detienen la suite.

```bash
nanouav suite configs/suites/baseline.yaml --dry-run        # ver qué se lanzaría
nanouav suite configs/suites/baseline.yaml -j 7             # 7 procesos en paralelo
nanouav suite configs/suites/baseline.yaml -j 7 --resume    # tras un corte: salta los terminados con rc=0
nanouav suite configs/suites/baseline.yaml --only r06       # solo los ids que contienen "r06"
nanouav suite configs/suites/baseline.yaml --out results/baseline_fuse2
```

Cada trabajo corre en su propio proceso con un hilo de BLAS/TF (`OMP_NUM_THREADS=1`, etc.) y sin
GPU salvo que declare `gpu: true`; por defecto se usan `nproc − 1` procesos. La salida queda así:

```
results/baseline/
  jobs.json                 estado de cada trabajo: run, env, rc, segundos, fecha
  logs/<id>.log             stdout+stderr del trabajo
  e3_results_nano.json      lo que escriba cada experimento (json, png, ...)
  e3_results_nano_M_ToF.json
  ...
```

Resumen con intervalos de confianza:

```bash
nanouav summarize --dir results/baseline
nanouav summarize --gazebo                      # episodios de Gazebo (results/gz_runs)
```

## Añadir un experimento

1. Crear `src/nanouav/experiments/rNN_nombre.py` con una CLI `argparse` y un `main()`:

   ```python
   """RNN · Qué pregunta responde (una línea).
   Usage: nanouav run rNN --n 10
   """
   import argparse, json
   import numpy as np
   from nanouav import paths, settings


   def main():
       ap = argparse.ArgumentParser(description=__doc__)
       ap.add_argument('--n', type=int, default=10)
       ap.add_argument('--seed', type=int, default=0)
       a = ap.parse_args()
       rng = np.random.default_rng(a.seed)
       out = {...}
       (paths.results_dir() / f'rNN_nombre_n{a.n}.json').write_text(json.dumps(out, indent=1))


   if __name__ == '__main__':
       main()
   ```

2. Registrarlo en `src/nanouav/experiments/__init__.py` (id, módulo, resumen, `needs`, `tags`).
3. Si tiene una configuración que se va a repetir, añadir un trabajo a una suite (o una suite nueva).
4. Añadir un test rápido si toca lógica del paquete (`tests/`), y una entrada en el `CHANGELOG.md`.
5. Documentar el resultado en `docs/research/RNN_*.md`: pregunta, montaje, comando exacto, tabla, conclusión.

Convenciones:

* Resultados siempre bajo `paths.results_dir()`, datasets bajo `paths.data_dir()`, modelos con
  `paths.models_dir()`. Nunca rutas relativas al archivo ni al directorio actual.
* El nombre del archivo de salida incluye los parámetros que lo distinguen (perfil, variante, ley,
  n…), para que varios trabajos de una suite no se pisen.
* Semillas explícitas (`np.random.default_rng(seed)`); nada de aleatoriedad sin semilla.
* Las banderas globales se leen con `settings.flag()` / `settings.number()`.
* Las variantes nuevas de un algoritmo se añaden detrás de un atributo apagado por defecto (como
  `RF_DYN` o `SQUEEZE`), para que los resultados publicados sigan siendo reproducibles; si cambian el
  comportamiento por defecto, el test de regresión lo detecta.
