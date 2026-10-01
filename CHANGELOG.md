# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/); versiones según
[SemVer](https://semver.org/lang/es/). Cada entrada indica si cambia algún resultado numérico.

## [0.13.1] — 2026-10-01

Sin cambios en los resultados de simulación (el test de regresión pasa sin regenerar la referencia).

### Añadido
- `--thin P` en `nanouav run train` y `r03`: postes de 1–5 cm y cables en una fracción P de las
  escenas de entrenamiento y validación (R14), con caché de datos propia. `--suffix` da nombre propio
  a los modelos resultantes (`model_M_ToF_thin.tflite`) sin tocar los originales.
- `nanouav run train --eval-only`: evalúa modelos ya guardados y actualiza `metrics.json`.
- Escenario `bosque` con `r_min` / `r_max` (bosque de troncos finos), suite `thin_structures.yaml`,
  y trabajos del nano en `narrow_passages.yaml`.
- `r12` añade `_fuse2` al nombre del resultado cuando `NANOUAV_FUSE2=1`.

### Corregido
- La evaluación tras entrenar fallaba en la TITAN X: el modelo exportado sin compilar activaba XLA.
  XLA queda desactivado también en `train_cnn` y `r11`.
- Con un modelo .tflite pero sin runtime de TFLite, la CNN pasaba en silencio a profundidad *fake*;
  ahora lanza `TFLiteRuntimeMissing`. Pedir una variante que no está instalada ya no cae en silencio
  a `model.tflite` (= M).
- `metrics.json` se vuelve a leer antes de escribir, para no pisar evaluaciones hechas en paralelo.
- `docs/gpu.md`: `LD_LIBRARY_PATH` y `ptxas` cubren todas las carpetas `nvidia` de pip.

## [0.13.0] — 2026-10-01

Reestructuración completa en un paquete instalable. **Sin cambios numéricos**: las salidas de
escenarios, render, ToF, fusión, flujo óptico, planificadores y lazo cerrado coinciden bit a bit con
`uav_v3` (`NANOUAV_EXACT=1 pytest tests/test_regression.py`).

### Añadido
- Paquete `nanouav` (layout `src/`) con `pyproject.toml` y extras `tflite`, `train`, `dev`.
- CLI única `nanouav`: `list`, `run`, `suite`, `summarize`, `models`, `gazebo gen-world`, `doctor`.
- Registro de experimentos (`nanouav.experiments`) con ids estables (`e0c` … `r15`, `train`, …).
- Suites YAML con matrices, ejecución paralela, `--resume`, `--dry-run`, `--only`, estado en `jobs.json`
  (`smoke`, `baseline`, `scenarios`, `narrow_passages`, `perception`).
- Rutas de artefactos configurables (`NANOUAV_HOME`, `NANOUAV_MODELS`, `NANOUAV_DATA`, `NANOUAV_RESULTS`)
  y banderas con prefijo `NANOUAV_` (se aceptan los nombres antiguos).
- `nanouav models import|list|check` para traer las redes entrenadas sin versionarlas.
- Tests (pytest): regresión numérica, escenarios, fusión, planificadores en una puerta, CLI, runner,
  nodos de Gazebo con stubs de ROS 2, CNN real (si hay modelo). CI en GitHub Actions (3.10–3.12).
- Documentación en español: README, arquitectura, experimentos, artefactos, GPU, Gazebo, migración,
  problemas conocidos.

### Cambiado
- Código, comentarios, docstrings y mensajes traducidos al inglés (verificado por AST: solo cambió texto).
- Etiquetas de `r08 --levels` y claves de `r4_speed_limit.json` / `r7_camera_tilt.json` en inglés
  (las etiquetas antiguas de `r08` se siguen aceptando).
- Logs de `gazebo/batch_eval.sh` en `results/gz_logs/`; episodios de Gazebo en `results/gz_runs/`.
- `MHPlanner.set_segment` reinicia también margen, relajación, progreso e inspección (afecta solo a
  misiones de varios tramos, p. ej. Gazebo con `LOOP=true`).

### Retirado
- Scripts sueltos en la raíz, `run_jobs.sh` + `jobs_*.txt` (→ suites), `pack.sh`, y los atajos
  obsoletos `e2a_train_variants.py`, `generate_model.py`, `train_e0d.py`.

## [0.12.x] — 2026-09 (estructura plana `uav_v2` / `uav_v3`)
- R12: planificador MH (Loquercio 2021) y escenarios compartidos con Gazebo (v13).
- R13: pasos estrechos y anti-estancamiento.
- R14: métricas por clase y `fuse_tof2`. R15: falsas detecciones dinámicas vs ego-movimiento.
- Detalle en `docs/research/`.
