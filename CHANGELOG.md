# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/); versiones según
[SemVer](https://semver.org/lang/es/). Cada entrada indica si cambia algún resultado numérico.

## [0.19.2] — 2026-10-08

Sin cambios en los resultados de R18–R21.

### Añadido
- `r22 scan`: el mapa privilegiado a partir de las imágenes de profundidad de Unity en una malla de poses
  (`nanouav.flightmare.scan`). El bosque de 2022 casi no tiene colliders: la petición de nube solo devolvió
  una lámina bajo el río. Escribe el mismo `cloud_<escena>.npz` y comprueba el mapa con una vista nueva.
- `FlightmareClient(stereo=False)`: una sola cámara (para escanear). `CloudWorld(inflate=)`, que se guarda
  en el `.npz`.

### Cambiado
- `probe` recomienda `scan` cuando la nube sale plana o vacía.

## [0.19.1] — 2026-10-08

Sin cambios en los resultados de R18–R21. R22: corrige las convenciones de Unity a partir del código del
reproductor (flightmare_unity, `dev/agile_flight`) tras el primer probe real.

### Corregido
- La malla del dron tapaba las cámaras: `size` del vehículo 0.01 (como agile_autonomy), `--vehicle-size`.
- Profundidad de Unity = valor × plano lejano de la capa (100 m); antes se tomaba en metros.
- La petición de nube es absoluta (Unity no suma el `render_offset`): `FlightmareClient.cloud_points` pide
  en origen + desplazamiento y devuelve los puntos en el marco de la escena. El `.ply` está en orden ROS.
- `CloudWorld`: la distancia es la del centro ocupado más cercano (los vóxeles ya están inflados una
  resolución); el suelo sirve para terreno como lámina o como bloque.

### Cambiado
- `probe`: comprueba la base estéreo, la escala de la profundidad contra el SGM y la nube contra la
  profundidad de Unity. Escribe avisos en `calib.json`, un panel de 2 × 3 y `probe.npz`.
- El reproductor falso imita al real (profundidad / plano lejano, nube absoluta, ocupación por radio).

## [0.19.0] — 2026-10-07

Sin cambios en los resultados existentes.

### Añadido
- R22 · Flightmare como renderizador (`nanouav.flightmare`, `nanouav run r22`):
  - puente ZMQ en Python con el ejecutable de Unity (binarios 2020 y 2022);
  - par estéreo de `StereoRig`;
  - nube de puntos de la escena como mapa privilegiado (`CloudWorld`, que acepta `geometry.Field`);
  - líneas de vuelo encontradas en la nube;
  - `probe`, que comprueba las convenciones y escribe `calib.json`;
  - `eval`, `video` y un reproductor de Unity falso (`fake`) para probarlo todo sin Unity.
  Extra `flightmare` (pyzmq). Informe: `docs/research/R22_flightmare.md`.
- `Rollout.observer`: un renderizador externo en lugar del nuestro. `r18 video` dibuja la nube y la
  imagen del renderizador externo.

## [0.18.1] — 2026-10-05

Sin cambios en los resultados: los modelos guardados conservan su arquitectura.

### Añadido
- `policy.build_model(head='spatial')` (`r18 train|dagger --head spatial`): conserva la posición de las
  características de profundidad (convolución 1 × 1 + aplanado) en vez del promedio global, que
  pierde dónde está cada obstáculo. Por defecto sigue `gap`.
- R18: resultados de `--tscale arc --vcap 1.3` (16/40) y `--select consistent` (13/40) frente a 22/40.
  Ninguna ayuda.

## [0.18.0] — 2026-10-05

Sin cambios en los resultados existentes.

### Añadido
- R21 · el estudiante de R18 en Gazebo:
  - `gen_world.py --agile forest|gap` (el mundo de prueba de R18) y el modelo `agile_x3` (empuje/peso
    3.6, colisión de 0.3 m, par estéreo de `StereoRig`);
  - `gazebo/nodes/agile_node.py`, `gazebo/run_agile.sh` y `gz_docker.sh agile`;
  - `nanouav.agile.gz_core` (la lógica sin ROS, más un dron de velocidad con retraso para probarla
    sin Gazebo);
  - `r18 export` (Keras → TFLite, `policy.TFLitePolicy`, `load_policy`) y `r18 video --from-gz`.
  Informe: `docs/research/R21_gazebo_agile.md`.
- `rollout.fit_plan` y `rollout.choose`: la proyección y la elección de `Rollout` como funciones, para
  usarlas fuera del simulador.

## [0.17.2] — 2026-10-05

Sin cambios en los resultados: las opciones nuevas están apagadas por defecto.

### Añadido
- `Rollout.TSCALE` (`--tscale chord|arc`): β por la distancia recta (como antes) o por la longitud
  del camino predicho. Con la recta, una predicción curva se vuela más rápido que v_des: los choques
  del estudiante a 10 m/s fueron a 12–19 m/s.
- `Rollout.V_CAP` (`--vcap`): pico de velocidad de cada plan ≤ vcap · v_des.
- `Rollout.SELECT` (`--select snap|consistent`): mantener el lado de la elección anterior mientras su
  costo predicho esté dentro del 25 % del mejor.
- `r18 eval|diagnose|video` aceptan esas opciones, y los archivos de resultado llevan su sufijo.
- `diagnose`: la clase se decide 0.35–1 s antes del choque. Hay una clase nueva, `cambio_de_plan`,
  y una marca `sobrevelocidad`.

## [0.17.1] — 2026-10-05

Sin cambios en los resultados.

### Añadido
- `nanouav run r18 video`: vídeo de un episodio (vista cenital con las hipótesis, profundidad que ve
  la red, cámara de seguimiento), en H.264 si hay ffmpeg. `--pilot expert|gbl|blind` para comparar.
  `Rollout.trace` guarda también la actitud y, con `trace_depth`, la profundidad de entrada.
- `r18 diagnose`: comprueba si el obstáculo del choque estaba en el campo de visión (`fuera_vista`) y
  el ángulo al que estaba.
- R18: evaluación con la proyección corregida (22/40 y 16/40). La corrección no explica los choques a
  3 m/s. También se documentan qué cambio corresponde a cada clase de choque y la meta.

## [0.17.0] — 2026-10-05

### Corregido
- **Proyección de la predicción al polinomio que se vuela** (`Rollout.PROJ = 'scaled'`, ahora el valor
  por defecto). El polinomio se ajusta con v/β y a/β², así que tras el escalado de tiempo cada plan
  arranca en la velocidad actual. Antes la velocidad mandada saltaba a β·v en cada plan.
  - Para el estudiante, β ≈ 0.7 a 3 m/s: caídas del 30 % de la velocidad 15 veces por segundo.
  - τ_gbl y el experto no cambian (β ≈ 1; 6/6 a 3 y 10 m/s).
  - `--proj raw` reproduce lo anterior en `r18 pilots|eval|diagnose`.
  - Seguimiento (R19): con la cámara a 10 Hz el bloqueo no cambia. El test corto pasa de 5 a 10 Hz:
    a 5 Hz el bloqueo variaba con cambios mínimos de la trayectoria.

### Añadido
- `nanouav run r18 diagnose`: clasifica los choques del estudiante en `todas_chocan`, `eligio_mal`,
  `plan_justo` y `seguimiento`. Cuenta los cambios de lado y la β mediana, y guarda una vista cenital
  por choque. `Rollout.trace` guarda cada decisión.
- `r18 eval|pilots --tag`. Las trayectorias en `traj/` llevan el nombre del modelo, para no pisarse
  entre evaluaciones.
- R18: evaluación de `student_train` y `student_polish` en el PC de entrenamiento.

## [0.16.1] — 2026-10-05

Sin cambios en los resultados: las opciones nuevas están apagadas por defecto en `r18`. En `r20
dagger` están encendidas.

### Añadido
- `policy.train(..., lr_end=, keep_best=, val_by_env=)`. `lr_end` da un descenso coseno del lr;
  `keep_best` guarda la época con menor R-WTA de validación; `val_by_env` valida con entornos enteros.
  `load_shards` marca el entorno de cada muestra (`_env`).
- `nanouav run r18 train|dagger`: `--lr`, `--lr-end`, `--best` y `--val-by-env`. `r18 train`:
  `--data` (varios directorios) y `--name` (para no pisar `student_train.keras`).
- `r20 dagger`: lr 3e-4 → 1e-5, mejor época y validación por entornos por defecto.
- R18: resultados del DAgger completo de 850 entornos en el PC de entrenamiento
  (`docs/research/R18_agile_autonomy.md`).

## [0.16.0] — 2026-10-03

Sin cambios en los resultados existentes: las piezas compartidas con R18/R19 ganan opciones cuyos
valores por defecto reproducen el comportamiento anterior (los tests de R18, R19 y la regresión del
renderer pasan sin cambios).

### Añadido
- `nanouav run r20`: seguir un globo de color a través de un bosque, evitando los árboles, sobre la
  política de R18 (`docs/research/R20_seguimiento_globo.md`).
  - `follow.blob.BlobDetector`: detector de color (HSV + componentes conexas, une trozos cortados por
    un tronco); funciona igual en la imagen simulada y con una cámara real (`r20 video`, `r20 pick-color`).
  - `follow.balloon`: globos que flotan por rutas libres con altura variable, bosque en color con
    falsos positivos realistas, globos que cruzan (del mismo color con probabilidad 0.25).
  - `follow.balloon_mission.BalloonMission`: distancia por tamaño aparente, filtro 3D, punto de
    seguimiento a la altura del globo; DAgger `kind='balloon'`.
- Imagen en color: `Texture.color`, `render_view(..., chroma=True)`, `render_stereo/observe_full(..., rgb=True)`
  (la imagen gris es la misma); `Rollout` la pide si la misión tiene `needs_rgb`.
- Tracker: compensación del giro de la cámara (`Tracker.update(..., warp=)`, `estimate.rotation_warp`),
  `min_gate_px` y `drop()`. `TargetLock`: `gate_locked` (descarta una pista que salta en 3D), `w_dist` (al
  rebloquear prefiere la más cercana a la predicción) y `gate_growth`. `TargetFilter(dims=3)`.
- `FollowMission`: métodos por tipo de objetivo (`_detect`, `_pos_of`, `_goal_z`, …), `innov_gate` y
  métrica `switches`. `dagger.run(..., xi_start=)`.

## [0.15.0] — 2026-10-03

### Añadido
- `nanouav.follow` y `nanouav run r19`: detectar a una persona y seguirla evitando obstáculos.
  - Detector YOLO de personas (ultralytics; COCO o ajustado con VisDrone conservando la cabeza de COCO)
    y un detector simulado calibrable (`eval-det`).
  - Tracker estilo ByteTrack y bloqueo del objetivo con apariencia y compuerta 3D.
  - Estimación 3D con la profundidad estéreo y filtro del objetivo.
  - Personas que caminan en el simulador, misión de seguimiento para el lazo de R18 y DAgger de
    seguimiento (`kind='follow'`).
  - Conversión de VisDrone, entrenamiento y exportación del detector, y demo en vídeo.
  Informe: `docs/research/R19_seguimiento_personas.md`.
- `Rollout` con misiones (`GoalMission` = la tarea del paper, igual que antes; `FollowMission`) y
  `Config.label_every`; `stereo.observe_full` devuelve también la imagen;
  `global_plan(..., point_goal=True)`.

### Cambiado (R18)
- Cada polinomio arranca con la aceleración de referencia del plan en curso, no la medida (que cerraba
  un lazo a través de la vibración y hacía oscilar a baja velocidad). Pilotos de R18 revalidados con 6
  semillas por casilla: experto y τ_gbl 36/36 en el bosque (3, 7 y 10 m/s) y 36/36 en el hueco (3, 5 y
  7 m/s), igual que antes. `Rollout.A0_MODE` permite comparar.

## [0.14.0] — 2026-10-02

Sin cambios en los resultados existentes: el renderer gana dos parámetros con valores por defecto
idénticos (el test de regresión pasa sin regenerar la referencia).

### Añadido
- `nanouav.agile` y `nanouav run r18`: recreación de Loquercio et al. 2021 (*Learning high-speed flight
  in the wild*). Incluye:
  - cámara estéreo simulada con profundidad SGM (`cv2.StereoSGBM`), con filtro de textura;
  - entornos de entrenamiento (bosques y formas convexas por proceso de Poisson) y de prueba (bosque
    de 1/25 m⁻², hueco estrecho);
  - plataforma del paper (r_q = 0.2 m, c_max = 35.3 m/s², J = 0.007 kg·m²);
  - experto privilegiado (trayectoria global + Metropolis-Hastings sobre B-splines, Eq. 1–3);
  - estudiante (MobileNet-V3 + MLP, M = 3, pérdida R-WTA) y ejecución por polinomios de orden 5 con
    escala de tiempo y selección por coste;
  - DAgger reanudable y en paralelo, ajuste fino para el hueco, y evaluación contra los pilotos
    experto, τ_gbl y ciego.

  Informe y tabla de fidelidad en `docs/research/R18_agile_autonomy.md`.
- `render_view(..., size=(H, W), d_max=...)` y `camera_rays(hfov, size)`: resolución y alcance
  arbitrarios (por defecto 96×96 y `D_MAX`, como antes).

## [0.13.4] — 2026-10-02

Sin cambios en los resultados de simulación.

### Corregido
- Gazebo: el contenedor corre como root y dejaba a nombre de root los mundos generados, las texturas y
  `results/gz_runs`; después `nanouav gazebo gen-world` fallaba en el host con *Permission denied*.
  `gz_docker.sh` pasa `HOST_UID`/`HOST_GID` y `run_sim.sh` devuelve esos archivos al usuario al
  generar el mundo y al salir. `wait` sobre Gazebo ya no aborta la limpieza si Gazebo sale con error.

### Documentación
- R17 §5: validación con las redes completas (suite `speed_thin`, n = 10). `FLY_KEEP_NEAR` da casi
  toda la mejora con poco coste a 1 m/s, mientras que `MAP_DECAY_FOV` cuesta un 17–59 % de tiempo con
  la CNN. Con el M_ToF a 1 m/s + `FLY_KEEP_NEAR`, el mundo de Gazebo da 10/10 en 30.8 s.
- `speed_thin.yaml`: 12 trabajos `*-fly` nuevos (con `--resume` solo corren esos).

## [0.13.3] — 2026-10-01

Sin cambios en los resultados publicados: las opciones nuevas del planificador están apagadas por
defecto (el test de regresión pasa sin regenerar la referencia).

### Corregido
- Gazebo: reiniciar el mundo desde la interfaz (`/clock` vuelve atrás) empieza un episodio nuevo con
  un planificador nuevo y se guarda por separado (`*_epN.json`). Antes, el nodo seguía con el
  planificador anterior (contador de estancamiento y margen relajado heredados) y el RTF salía
  negativo. Tres saltos atrás en 10 s siguen indicando dos simulaciones en el mismo `/clock` y el
  monitor pasa a tiempo de pared.
- Gazebo con `V_CRUISE` > 0.8: el nodo usaba los límites de referencia normales en lugar de los
  `fast` del simulador propio. Ahora ambos usan `nanouav.planning.presets`.
- `r12 --v` distinto de 0.5 añade `_v<velocidad>` al nombre del resultado (los barridos se
  sobrescribían).
- `batch_eval.sh`: el log incluye la variante y la etiqueta (antes M y M_ToF se pisaban).

### Añadido
- `PLANNER_CFG` / parámetro `planner_cfg` en Gazebo, con la misma sintaxis que `r12 --cfg`.
- Opciones del MH para obstáculos finos (R17), apagadas por defecto: `FLY_KEEP_NEAR`,
  `MAP_DECAY_FOV` y `MAP_HIT_WINS`. Con profundidad perfecta en `bosque` pasan de 22/32 a 40/40
  episodios sin choque; con la CNN local (n = 4) no hay mejora clara. Suite `speed_thin.yaml` para
  decidir con las redes completas. Informe `docs/research/R17_velocidad_y_troncos_finos.md`.
- `MHPlanner.dbg` registra qué límite de velocidad manda en cada cuadro (`v_stop`, `v_clr`,
  `k_fov`, `v_plan`).
- `nanouav summarize --gazebo` separa filas por velocidad y `planner_cfg`; `nanouav summarize`
  muestra el tiempo mediano de los éxitos.

## [0.13.2] — 2026-10-01

### Documentación
- Informe `docs/research/R16_entrenamiento_y_lazo_cerrado.md`: resultados de la campaña del
  2026-10-01 (entrenamiento completo, M vs M_ToF, estructuras finas, cables, pasos estrechos).

### Corregido
- `r10`: `hit_ball` era `False` en choques con el balón (comparación estricta con muestras tomadas en
  otro instante); en ese mundo el único obstáculo es el balón, así que ahora `hit_ball` = colisión.
  Cambia la columna "hits" de los resúmenes, no el éxito ni las colisiones.

### Añadido
- Escenario `cables` (cables horizontales a la altura de vuelo, `gap`, `d`, `posts`) y suite
  `cables.yaml`. Con GT y d = 2 cm el nano choca en parte de los episodios (límite de resolución de
  96×96); con d = 5 cm la GT pasa.
- `r10 --scen cross` guarda `bearing_deg` e `in_fov` por velocidad: el balón llega por un rumbo
  constante atan(v_obs / v_dron); con la cámara de 60° solo es visible si v_obs < 0.58·v_dron. Por
  encima, el resultado depende de la suerte en la sincronización, no de la percepción (ver R16).

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
