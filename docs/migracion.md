# Migración desde la estructura plana (`uav_v2` / `uav_v3`)

La versión 0.13.0 reorganiza el proyecto sin cambiar su comportamiento numérico: con las mismas
semillas, escenarios, render, ToF, fusión, flujo óptico, ambos planificadores y un episodio en lazo
cerrado dan resultados **idénticos bit a bit** a los de `uav_v3` (verificado con SHA-1 de cada salida;
`NANOUAV_EXACT=1 pytest tests/test_regression.py`). Los resultados ya publicados siguen siendo válidos.

## Pasos para pasar una copia de trabajo

```bash
git clone <url> nanouav && cd nanouav          # o descomprimir el zip / clonar el bundle
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tflite,dev]"                 # + ".[train]" en la máquina con GPU
nanouav models import ~/Project_UAV/uav_v2/_03_cnn
cp -r ~/Project_UAV/uav_v2/data/* data/  2>/dev/null || true    # opcional: datasets ya generados
nanouav doctor && pytest -m "not slow"
```

Los resultados antiguos (`assets/`, `logs/`) no se migran: se pueden dejar en la copia vieja o
copiarse a `results/legacy/` como referencia.

## Comandos

| antes (desde la raíz) | ahora |
|---|---|
| `python e3_closed_loop.py --profile nano --n 30` | `nanouav run e3 --profile nano --n 30` |
| `python e4_missions.py --profile x3 --n 6` | `nanouav run e4 --profile x3 --n 6` |
| `python r5_pole.py --law consistent --n 10` | `nanouav run r05 --law consistent --n 10` |
| `python r6_corridor_consistent.py ...` | `nanouav run r06 ...` |
| `python g1_gazebo_world.py --world scn:bosque ...` | `nanouav run r12 --world scn:bosque ...` |
| `python r14_class_metrics.py` / `r15_dyn_egomotion.py` | `nanouav run r14` / `nanouav run r15` |
| `python train_cnn.py --variants XS,S,M,L ...` | `nanouav run train --variants XS,S,M,L ...` |
| `python train.py` (TartanAir) | `nanouav run train-tartanair` |
| `python r3_lnca_train.py ...` | `nanouav run r03 ...` |
| `python run_all.py` | `nanouav run figures` |
| `python summarize_results.py` | `nanouav summarize` |
| `python3 gazebo/summarize_gz.py` | `nanouav summarize --gazebo` |
| `python3 gazebo/gen_world.py --all --seeds 0,1,2` | `nanouav gazebo gen-world --all --seeds 0,1,2` |
| `bash run_jobs.sh jobs_A.txt 7` (+ `jobs_B.txt`) | `nanouav suite configs/suites/baseline.yaml -j 7` |
| `UAV_FAKE_CNN=1 python ...` | `nanouav run --fake-cnn ...` o `NANOUAV_FAKE_CNN=1` |
| `./pack.sh ../uav_v2.zip` | `git archive` / `git bundle` (el código está en git; los artefactos aparte) |
| `docker build -f Dockerfile.e2c -t nano_uav_sim:fixed .` | `docker build -f docker/Dockerfile.base -t nano_uav_sim:fixed .` |
| `docker build -f Dockerfile.sim -t nano_uav_sim:v12 .` | `docker build -f docker/Dockerfile.sim -t nano_uav_sim:v12 .` |
| `bash gazebo/gz_docker.sh`, `run_sim.sh`, `batch_eval.sh` | igual (mismas variables) |

`nanouav run` acepta también el nombre viejo del script: `nanouav run r6_corridor_consistent.py ...`.

## Archivos

| antes | ahora |
|---|---|
| `uav_config.py` | `src/nanouav/config.py` |
| `_00_scene/{renderer,scene}.py` | `src/nanouav/scene/` |
| `_01_tof/tof.py`, `_02_optical_flow/optical_flow.py` | `src/nanouav/sensors/` |
| `_03_cnn/{cnn,model_zoo,lnca_zoo,early_exit,tof_calibrator}.py` | `src/nanouav/perception/` |
| `_03_cnn/*.tflite`, `*.keras`, `metrics.json`, `r3/` | `models/` (fuera de git) |
| `_04_policy/*.py` | `src/nanouav/planning/` |
| `_04_policy/mh_planner_v1.py` | `src/nanouav/planning/legacy/mh_planner_r12.py` |
| `_05_benchmark/*.py` | `src/nanouav/benchmark/` |
| `_06_control/*.py` | `src/nanouav/control/` |
| `e3_closed_loop.py`, `e4_missions.py` | `src/nanouav/sim/{closed_loop,missions}.py` (+ envoltorios en `experiments/`) |
| `gazebo/scenarios.py` | `src/nanouav/sim/scenarios.py` |
| `train_cnn.py`, `train.py`, `r3_lnca_train.py` | `src/nanouav/training/{train_cnn,tartanair,r3_lnca_train}.py` |
| `summarize_results.py`, `gazebo/summarize_gz.py` | `src/nanouav/results/` |
| `e0c…e2d`, `r1…r15`, `g1_gazebo_world.py`, `run_all.py` | `src/nanouav/experiments/` (`r1` → `r01_…`, `g1` → `r12_scenarios`, `run_all` → `figures`) |
| `gazebo/scripts/*.py` | `gazebo/nodes/` |
| `gazebo/gen_world.py` | `gazebo/tools/gen_world.py` |
| `Dockerfile.e2c`, `Dockerfile.sim` | `docker/Dockerfile.base`, `docker/Dockerfile.sim` |
| `jobs_A.txt`, `jobs_B.txt`, `run_jobs.sh` | `configs/suites/baseline.yaml` + `nanouav suite` |
| `research/` | `docs/research/` |
| `README.md`, `RUN_LOCAL.md`, `LOCAL_GPU.md`, `_0*/README.md` | `docs/legacy/` (históricos; comandos de la estructura vieja) |
| `gazebo/README_GAZEBO.md` | `docs/gazebo.md` (actualizado) |
| `assets/` | `results/` |
| `data/` | `data/` (fuera de git) |
| `e2a_train_variants.py`, `generate_model.py`, `train_e0d.py` | retirados (ya eran atajos obsoletos a `train_cnn.py`) |
| `pack.sh` | retirado |

## Variables de entorno

| antes | ahora (el nombre viejo sigue funcionando) |
|---|---|
| `UAV_FAKE_CNN` | `NANOUAV_FAKE_CNN` |
| `UAV_FUSE2` | `NANOUAV_FUSE2` |
| `MH_GRID_RES` | `NANOUAV_MH_GRID_RES` |
| `NANO_UAV_ROOT` | `NANOUAV_HOME` |
| — | `NANOUAV_MODELS`, `NANOUAV_DATA`, `NANOUAV_RESULTS` |

## Código propio que importaba el proyecto

```python
# antes
sys.path.insert(0, ROOT)
from _04_policy.mh_planner import MHPlanner
from _03_cnn.cnn import predict_depth_norm
import uav_config as C
# ahora (con el paquete instalado, sin tocar sys.path)
from nanouav.planning.mh_planner import MHPlanner
from nanouav.perception.cnn import predict_depth_norm
from nanouav import config as C
```

## Otros cambios

* Código, comentarios, docstrings y mensajes en inglés; documentación en español. Se conservaron en
  español los **datos** que otros archivos leen: nombres de escenarios (`bosque`, `rendijas`…),
  acciones de la política (`libre`, `precaucion`, `evadir`), sectores del ToF, claves de algunos JSON
  de resultados (`suave`, `escalon`) y la línea `=== EPISODIO:` del nodo de Gazebo.
* `r08 --levels` usa ahora etiquetas en inglés (`'bias only 0.20'`…); las españolas antiguas
  (`'solo sesgo 0.20'`…) se siguen aceptando.
* Las claves de `results/r4_speed_limit.json` y `r7_camera_tilt.json` pasaron al inglés.
* `MHPlanner.set_segment` reinicia también el estado de margen, relajación y progreso; antes
  arrastraba el de un tramo al siguiente en misiones con varios tramos (Gazebo con `LOOP=true`). Los
  experimentos R12–R13 son de un solo tramo y no cambian.
* Los logs de `gazebo/batch_eval.sh` van a `results/gz_logs/` (antes `logs/gz/`).
