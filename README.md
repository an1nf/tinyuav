# nanouav

Evasión de obstáculos para nano-UAV con **una cámara monocular + un ToF 8×8**: CNN de profundidad
int8 de 96×96 pensada para un ESP32-S3, fusión CNN↔ToF, planificadores locales (carril consistente y
MH tipo Loquercio 2021), pila de control propia, simulador en lazo cerrado y evaluación en Gazebo
Harmonic / ROS 2 Jazzy.

El repositorio es un paquete de Python instalable con una única CLI (`nanouav`). Cada experimento de
la investigación (E0–E4, R1–R15) está declarado en un registro y se puede lanzar suelto o en lotes
paralelos descritos en YAML. Los modelos, datasets y resultados viven fuera de git.

```
nanouav list                                  # experimentos disponibles
nanouav run r05 --profile nano --n 10         # uno suelto
nanouav suite configs/suites/baseline.yaml -j 7 --resume
nanouav summarize --dir results/baseline      # tasas de éxito con IC 95 % de Wilson
```

## Instalación

Requiere Python ≥ 3.10.

```bash
git clone <url> nanouav && cd nanouav
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tflite,dev]"      # simulación + inferencia int8 + tests
pip install -e ".[train]"           # además TensorFlow, para entrenar (GPU: ver docs/gpu.md)
nanouav doctor                      # comprueba dependencias, rutas y modelos
```

| extra | contenido | para qué |
|---|---|---|
| (base) | numpy, opencv-headless, matplotlib, pyyaml, pillow | simulador, planificadores, experimentos con CNN *fake* |
| `tflite` | ai-edge-litert | inferencia de las redes int8 reales |
| `train` | tensorflow ≥ 2.16 | entrenamiento (`train`, `r03`, `r11`) y huella de LNCA (`r01`) |
| `dev` | pytest, ruff | tests y lint |

## Modelos

Las redes entrenadas no están en git. Se importan de una copia existente o se reentrenan:

```bash
nanouav models import ~/Project_UAV/uav_v2/_03_cnn   # copia model_*.tflite/.keras, metrics.json, r3/
nanouav models check                                  # carga M con el runtime TFLite
nanouav run train --variants XS,S,M,L --epochs 60     # o entrenar desde cero (GPU)
```

Sin modelos, los experimentos que lo permiten corren con profundidad *fake* (ground truth + ruido):
`nanouav run --fake-cnn r05 ...` o `NANOUAV_FAKE_CNN=1`. Detalles en [docs/artefactos.md](docs/artefactos.md).

## Estructura

```
src/nanouav/            paquete (código en inglés)
  config.py             constantes de cámara/ToF/profundidad (D_MIN, D_MAX, zonas ToF)
  paths.py settings.py  rutas de artefactos y banderas por variable de entorno
  scene/                renderer propio (mundo, texturas, iluminación)
  sensors/              ToF 8×8 y flujo óptico
  perception/           CNN int8, zoo de modelos (U-Net, LNCA), calibración y fusión ToF
  planning/             carril consistente (EvasionPlanner), MHPlanner, políticas, misiones
  control/              vehículo, simulador cuadricóptero, referencias, FlightStack, supervisor
  sim/                  lazo cerrado (E3), misiones (E4), escenarios compartidos con Gazebo
  training/             entrenamiento de la CNN (sintético, TartanAir, LNCA/M_ToF)
  benchmark/            latencia, jitter y energía en el ESP32-S3
  experiments/          un módulo por experimento + registro (`nanouav list`)
  results/              resúmenes con IC de Wilson
  cli.py runner.py      CLI y ejecutor de suites
configs/suites/         lotes de experimentos en YAML (smoke, baseline, scenarios, ...)
gazebo/                 mundo, modelos SDF, nodos ROS 2, scripts Docker y de lotes
docker/                 Dockerfile.base (ROS 2 Jazzy + Gazebo Harmonic) y Dockerfile.sim
tests/                  pytest: regresión numérica, escenarios, fusión, planificadores, nodos ROS
docs/                   documentación (español) e informes de investigación
models/ data/ results/  artefactos, fuera de git (cada uno con su README)
```

Arquitectura y flujo de datos: [docs/arquitectura.md](docs/arquitectura.md).

## Experimentos

| id | qué hace | requiere |
|---|---|---|
| `e0c` | Política P3: latencia del disparo por flujo óptico | — |
| `e1a` | Modelo físico de jitter del lazo de inferencia (ESP32-S3) | — |
| `e1b` | Presupuesto energético y compromiso velocidad–seguridad–autonomía | — |
| `e2b` | Calibración de escala de la CNN con el ToF | — |
| `e2d` | Salida temprana (flujo óptico + ToF) y ganancia de latencia | — |
| `e3` | Lazo cerrado: mundo tipo Gazebo, patadas de perturbación, Monte Carlo | — |
| `e4` | Misiones: waypoints, seguimiento de objetivo, obstáculos que aparecen | — |
| `figures` | Figuras de resumen (escena, flujo, políticas, benchmark) | — |
| `r01` | LNCA vs U-Net: parámetros, MACs, arena int8, latencia | tensorflow |
| `r02` | Seguimiento de objetivo con distintos conjuntos de sensores | — |
| `r03` | Entrenar/afinar M, M_ToF y variantes LNCA (fusión temprana del ToF) | GPU, tensorflow |
| `r04` | Velocidad máxima analítica vs alcance de percepción | — |
| `r05` | Poste aislado: barrido de velocidad, ley *brake* vs *consistent* | — |
| `r06` | Corredores aleatorios, *brake* vs *consistent* (paper C2) | — |
| `r07` | Inclinación de cámara vs distancia de frenado y cobertura ToF | — |
| `r08` | Éxito vs ruido/sesgo de estimación de estado (+ correcciones de deriva) | — |
| `r10` | Obstáculos dinámicos (balón de frente / cruzando), capa RF_DYN | — |
| `r11` | Estimación de velocidad aprendida desde imágenes (gen / train) | tensorflow |
| `r11b` | Velocidad tipo *flow deck* desde flujo óptico | — |
| `r12` | Escenarios compartidos con Gazebo; carril vs MH | — |
| `r14` | Métricas de profundidad por región/clase y tasas de seguridad (DDOS) | — |
| `r15` | Falsas detecciones dinámicas vs ego-movimiento (DroneMOT) | — |
| `train` | Entrenar las CNN XS/S/M/L y exportar .tflite int8 | GPU, tensorflow |
| `train-tartanair` | Entrenamiento mezclando imágenes TartanAir | GPU, tensorflow |
| `summarize` | Tasas de éxito con IC 95 % de Wilson de todos los resultados | — |
| `summarize-gz` | Tabla de episodios de Gazebo (+ el mismo escenario offline) | — |

`nanouav run <id> --help` muestra las opciones de cada uno. Las suites, el formato YAML y cómo añadir
un experimento están en [docs/experimentos.md](docs/experimentos.md).

### Suites incluidas

| suite | trabajos | contenido |
|---|---|---|
| `smoke.yaml` | 7 | arranque de cada familia (2–4 min); la usa CI en modo `--dry-run` |
| `baseline.yaml` | 27 | línea base del paper C2 (E3, poste, corredores, dinámicos, ruido de estado) × {M, M_ToF} |
| `scenarios.yaml` | 12 | los 6 escenarios de Gazebo offline, MH vs carril |
| `narrow_passages.yaml` | 19 | R13: ancho × planificador × CNN (X3 y nano), más profundidad GT como cota |
| `cables.yaml` | 10 | R16: cables a la altura de vuelo (nano), 2 y 5 cm, M / M_thin / M_ToF / M_ToF_thin + GT |
| `thin_structures.yaml` | 5 | R14 en lazo cerrado: bosque de troncos de 2–5 cm, M vs M_ToF × `fuse_tof2`, más GT |
| `perception.yaml` | 5 | R14 (métricas por región) y R15 (variantes del filtro dinámico) |

Cada suite escribe en `results/<nombre>/` (un log por trabajo y `jobs.json` con el estado);
`--resume` salta lo que ya terminó bien.

## Gazebo

```bash
docker build -f docker/Dockerfile.base -t nano_uav_sim:fixed .
docker build -f docker/Dockerfile.sim  -t nano_uav_sim:v12 .
bash gazebo/gz_docker.sh diag                      # GPU/EGL dentro del contenedor
WORLD=bosque SEED=2 bash gazebo/gz_docker.sh       # un episodio con ventanas
WORLDS=bosque,interior SEEDS=0,1,2 PLANNERS=mh,lane bash gazebo/batch_eval.sh
nanouav summarize --gazebo
```

Los mundos se generan desde los mismos escenarios JSON que usa el simulador propio
(`nanouav gazebo gen-world --scenario bosque --seed 2`). Guía completa: [docs/gazebo.md](docs/gazebo.md).

## Tests

```bash
pytest -m "not slow"        # ~15 s: regresión numérica, escenarios, fusión, planificadores, nodos ROS (con stubs)
pytest                      # además el episodio en lazo cerrado y, si hay modelos, la CNN real
```

`tests/test_regression.py` fija las salidas numéricas (escenarios, render, ToF, fusión, flujo,
ambos planificadores y un episodio corto). Se generaron justo después de reestructurar, una vez
comprobado que coincidían bit a bit con el proyecto plano anterior. Si un cambio las altera a
propósito: `python tests/test_regression.py --update` y anotarlo en el [CHANGELOG](CHANGELOG.md).

## Documentación

| documento | contenido |
|---|---|
| [docs/arquitectura.md](docs/arquitectura.md) | módulos, flujo percepción → planificación → control, banderas |
| [docs/experimentos.md](docs/experimentos.md) | registro, suites YAML, resultados, añadir un experimento |
| [docs/artefactos.md](docs/artefactos.md) | modelos, datasets y resultados fuera de git; cómo regenerarlos |
| [docs/gpu.md](docs/gpu.md) | entrenamiento con GPU (incl. Pascal / TITAN X) |
| [docs/gazebo.md](docs/gazebo.md) | entorno Gazebo v13: escenarios, Docker, lotes, monitor de episodio |
| [docs/migracion.md](docs/migracion.md) | equivalencias con la estructura plana anterior (`uav_v2`/`uav_v3`) |
| [docs/problemas_conocidos.md](docs/problemas_conocidos.md) | defectos detectados y aún no corregidos |
| [docs/research/](docs/research/) | informes de investigación R12–R15 y revisión de papers |
| [CONTRIBUTING.md](CONTRIBUTING.md) | convenciones de código, ramas y resultados |

## Licencia

Por definir.
