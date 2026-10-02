# Arquitectura

## Flujo de un paso de control

El mismo pipeline se ejecuta en el simulador propio (`nanouav.sim.closed_loop.run_episode`) y en
Gazebo (`gazebo/nodes/perception_node.py`). Solo cambia de dónde salen la imagen, el ToF y la odometría.

```
 cámara 96×96 gris ─┐
                    ├─► perception.cnn.predict_depth_norm ──► profundidad normalizada [0,1]
 ToF 8×8 (m) ───────┤        (int8 .tflite, o GT+ruido en modo fake)
                    │
                    ├─► perception.tof_calibrator.ToFCalibrator.update_and_scale   escala CNN ← ToF
                    └─► perception.tof_calibrator.fuse  (fuse_tof | fuse_tof2 con NANOUAV_FUSE2=1)
                                         │ profundidad métrica 96×96
                                         ▼
 estado (p, v, yaw, t) ─► planning.EvasionPlanner | planning.MHPlanner .update(state, depth, tof, pose)
                                         │ velocidad objetivo (mundo) + yaw objetivo + info
                                         ▼
                         control.flight_stack.ReferenceGenerator   (aceleración y jerk acotados)
                                         ▼
                         control.flight_stack.FlightStack          vel → actitud → tasas → mezclador
                                         ▼
                         control.quad_sim (simulador)  |  plugin de velocidad o motores (Gazebo)
                         control.supervisor.StabilityMonitor  → RECOVER si inclinación/tasas/caída
```

La percepción corre a 10 Hz (un fotograma nuevo = una inferencia); la referencia y el control a
50–250 Hz.

## Módulos

| módulo | responsabilidad | piezas principales |
|---|---|---|
| `nanouav.config` | convención de profundidad y geometría de sensores | `D_MIN`, `D_MAX`, `norm_to_m`, `m_to_norm`, `ZONE_BOXES`, `TOF_GRID` |
| `nanouav.paths` | dónde están modelos, datos y resultados | `models_dir()`, `data_dir()`, `results_dir()`, `repo_root()` |
| `nanouav.settings` | banderas globales por variable de entorno | `flag('FAKE_CNN')`, `number('MH_GRID_RES', 0.1)` |
| `nanouav.scene` | renderer propio por ray casting en numpy | `World`, `Box`, `Cylinder`, `Sphere`, `Plane`, `Texture`, `render_view` |
| `nanouav.sensors` | modelos de sensores | `tof_8x8` (VL53L5CX, modo objetivo más cercano), `optical_flow` |
| `nanouav.perception` | CNN y fusión | `cnn` (variantes XS/S/M/L/M_ToF), `model_zoo`, `lnca_zoo`, `tof_calibrator`, `early_exit` |
| `nanouav.planning` | planificadores locales | `evasion.EvasionPlanner` (carril + DP lateral + filtro de seguridad), `mh_planner.MHPlanner` (R12–R13), `policy`, `escape`, `mission` |
| `nanouav.planning.legacy` | versiones congeladas para comparar | `mh_planner_r12.MHPlannerV1` (`--planner mh1`) |
| `nanouav.control` | vehículo y control | `vehicle.PROFILES` (`nano`, `x3`), `quad_sim`, `reference`, `flight_stack`, `supervisor`, `controller` |
| `nanouav.sim` | simulación de episodios | `closed_loop` (E3), `missions` (E4), `scenarios` (fuente única de mundos para Gazebo y offline) |
| `nanouav.training` | entrenamiento | `train_cnn` (dataset sintético), `tartanair`, `r3_lnca_train` (M_ToF, LNCA) |
| `nanouav.agile` | recreación de Loquercio 2021 (R18), independiente del pipeline del nano | `stereo` (SGM), `envs`, `expert` (τ_gbl + M-H), `trajectory`, `policy` (MobileNet-V3 + R-WTA), `rollout`, `dagger`, `platform.AGILE` |
| `nanouav.benchmark` | hardware objetivo | latencia/arena, `jitter_model`, `energy_model` |
| `nanouav.experiments` | un módulo por experimento + registro | `REGISTRY`, `get()` |
| `nanouav.results` | informes | `summarize` (Wilson), `summarize_gz` |
| `nanouav.cli`, `nanouav.runner` | CLI y suites paralelas | `nanouav …`, `run_suite()` |

Las dependencias van en un solo sentido: `config/paths/settings` ← `scene, sensors` ← `perception` ←
`planning` ← `control` ← `sim` ← `experiments`. Los nodos de Gazebo importan el paquete igual que
cualquier experimento (con `PYTHONPATH=<repo>/src` dentro del contenedor).

## Planificadores

**`EvasionPlanner`** (E3–R11). Rejilla local 2D (celda 0.1 m) alimentada por la profundidad fusionada,
programación dinámica lateral sobre un carril de ±2.3 m y 4 m de alcance, y un filtro de seguridad
que limita la velocidad según la distancia de frenado (`A_BRAKE`, `T_REACT`). Con `CONSISTENT=True` y
`speed_law='brake'` aplica las restricciones consistentes de R5. Capas opcionales, apagadas por
defecto: `RF_DYN` (obstáculos dinámicos, R10), `VO`, `DRIFT_CORR` (R8–R9), `RF_NET_DISP`,
`RF_EDGE_INH`, `RF_EGO_REJ` (rechazo de falsas detecciones por ego-movimiento, R15).

**`MHPlanner`** (R12–R13, inspirado en Loquercio et al. 2021). Hereda la percepción y la rejilla, y
cambia la decisión: campo de coste hasta la meta por Dijkstra sobre la rejilla (`CTG_RES`, recalculado
cada `CTG_PERIOD`), 180 hipótesis de arcos (36 rumbos × 5 curvaturas) más una línea de corriente del
campo, margen dependiente de la velocidad, modo *squeeze* para pasos estrechos, relajación por
estancamiento, inspección de cruces dudosos (`INSPECT_*`) e histéresis de ruta. La resolución de su
rejilla se cambia con `NANOUAV_MH_GRID_RES` (0.1 m por defecto). Informe: `docs/research/R12_*`, `R13_*`.

## Banderas de entorno

| variable | efecto | antes |
|---|---|---|
| `NANOUAV_FAKE_CNN=1` | la CNN devuelve GT + ruido (sin .tflite) | `UAV_FAKE_CNN` |
| `NANOUAV_FUSE2=1` | fusión bidireccional `fuse_tof2` (R14) en vuelo | `UAV_FUSE2` |
| `NANOUAV_MH_GRID_RES=0.1` | resolución de la rejilla del MH | `MH_GRID_RES` |
| `NANOUAV_HOME` | raíz del repositorio (Docker, instalación no editable) | `NANO_UAV_ROOT` |
| `NANOUAV_MODELS` / `NANOUAV_DATA` / `NANOUAV_RESULTS` | reubicar artefactos | — |

Los nombres antiguos se siguen aceptando (`nanouav.settings._LEGACY`). En código nuevo, leer las
banderas siempre con `settings.flag()` / `settings.number()` y las rutas con `paths.*()`, nunca con
`os.environ` ni rutas relativas al archivo.

## Escenarios compartidos

`nanouav.sim.scenarios` genera cada mundo como un diccionario JSON (objetos, texturas, A, B,
obstáculos dinámicos). El mismo JSON produce el SDF de Gazebo (`nanouav gazebo gen-world`) y el
`World` del renderer propio (`scenarios.to_world`), con texturas muestreadas por el mismo
`Texture.sample`. Así un episodio de Gazebo y uno offline del mismo escenario y semilla son
comparables (`nanouav run r12 --world json:gazebo/worlds/gen/bosque_s2.json`).
