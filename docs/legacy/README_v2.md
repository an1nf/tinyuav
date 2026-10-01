> **Documento histórico** de la estructura plana anterior (`uav_v2`/`uav_v3`). Los comandos y rutas
> ya no son válidos; ver [docs/migracion.md](../migracion.md) para su equivalente con `nanouav`.

# Nano-UAV — percepción monocular + ToF y control hasta motores (simulación)

Pipeline completo de un nano-cuadricóptero ESP32-S3 simulado en Python y en
Gazebo: cámara 96×96 → CNN de profundidad int8 → calibración/fusión con ToF 8×8
→ planificador de evasión → referencias suaves → control en cascada → motores.

**Los números son de simulación.** Nada aquí está medido en hardware ni probado
en Gazebo desde este entorno (ver "Límites").

## Estructura
```
uav_config.py       Geometría de sensores y convención de profundidad (única fuente)
_00_scene/          Renderer v2 (raycasting mundo, cámara = Gazebo) + escena legacy
_01_tof/            ToF 8×8 (FOV 40° dentro de la cámara de 60°)
_02_optical_flow/   Lucas-Kanade + divergencia real (traza del flujo afín)
_03_cnn/            Modelos XS/S/M/L, exportación int8, calibrador y fusión ToF
_04_policy/         Políticas P1/P2/P3 y planificador de evasión (evasion.py)
_05_benchmark/      Jitter, energía, distancia segura (estimaciones de literatura)
_06_control/        Dinámica 6-DOF, control en cascada, referencias, supervisor
gazebo/             World SDF, X3, nodos ROS 2 (perception_node, flight_controller_node)
train_cnn.py        Entrenamiento v2 (dataset sintético + opcional TartanAir)
e3_closed_loop.py   Lazo cerrado render→CNN→ToF→planificador→control→motores→6-DOF
run_all.py, e0c…e2d Experimentos anteriores (actualizados a la nueva convención)
```

## Uso
Pruebas en equipo local con GPU y pesos de v2: **[RUN_LOCAL.md](RUN_LOCAL.md)**.

```bash
pip install numpy opencv-python matplotlib tensorflow-cpu ai-edge-litert
python train_cnn.py                    # entrena XS,S,M,L → _03_cnn/model_*.tflite + metrics.json
python e3_closed_loop.py --profile x3  # lazo cerrado (también --profile nano)
python run_all.py                      # experimentos legacy
UAV_FAKE_CNN=1 python e3_closed_loop.py   # CNN = GT + ruido (depuración)
./pack.sh                              # zip sin venv, datos, cachés ni mallas
```
Gazebo (Docker `Dockerfile.sim`): `bash gazebo/run_sim.sh` (control por velocidad del
plugin) o `CONTROL_MODE=motors bash gazebo/run_sim.sh` (control propio hasta motores).

## Cambios v2 (correcciones)
1. Cuantización int8: datos representativos en [0,1] (antes img·255−128 → entrada casi binaria).
2. Profundidad: un solo par `norm_to_m/m_to_norm`; se eliminó el doble escalado ×D_MAX.
3. `benchmark.py`: imports corregidos.
4. Divergencia = tr(A) del flujo afín; umbral desde TTC (3 s @ 10 fps ≈ 0.067).
5. Variantes XS/S/M/L realmente distintas; `cnn.py` carga el modelo de cada variante.
6. Nodo ROS sin ground truth de obstáculos; CNN solo en frames nuevos.
7. Higiene: sin venv/mallas/archivos basura en el paquete (`.dockerignore`, `pack.sh`).

## Resultados en simulación (lazo cerrado E3, CNN M int8 real)

**x3** — world Gazebo: éxito en 26 s, holgura mín 0.56 m, inclinación máx 6.4°, jerk p99 3.3 m/s³, saturación 0.0 %.  
Golpe Δω≈6 rad/s + ráfaga 0.35 g: éxito; RECOVER 2.9 s, inclinación pico 28°.  
Envolvente de golpes: 3 rad/s → pico 9°, 6 rad/s → pico 28°, recupera 2.8 s, 9 rad/s → pico 62° (toca suelo), 12 rad/s → pico 104° (toca suelo).  
Monte Carlo 12 mundos: suavizado 11/12 (colisiones 1, timeouts 0) · sin suavizado 10/12 (colisiones 1). Medianas suavizado vs sin: jerk p99 3.2 vs 4.2 m/s³, acel. yaw 0.76 vs 1.16 rad/s², inclinación 6.0° vs 4.3°, holgura 0.53 vs 0.53 m.

**nano** — world Gazebo: éxito en 30 s, holgura mín 0.34 m, inclinación máx 3.4°, jerk p99 2.3 m/s³, saturación 0.0 %.  
Golpe Δω≈6 rad/s + ráfaga 0.35 g: éxito; RECOVER 1.6 s, inclinación pico 23°.  
Envolvente de golpes: 3 rad/s → pico 12°, 6 rad/s → pico 24°, recupera 1.6 s, 9 rad/s → pico 35°, recupera 1.9 s, 12 rad/s → pico 47°, recupera 2.0 s.  
Monte Carlo 12 mundos: suavizado 12/12 (colisiones 0, timeouts 0) · sin suavizado 12/12 (colisiones 0). Medianas suavizado vs sin: jerk p99 2.5 vs 1.3 m/s³, acel. yaw 2.10 vs 4.28 rad/s², inclinación 5.3° vs 3.0°, holgura 0.31 vs 0.31 m.

Gráficas: `assets/e3_*_{x3,nano}.png`, `assets/cnn_v2_*.png`. JSON: `assets/e3_results_*.json`, `_03_cnn/metrics.json`.

## Estado del entrenamiento y pendientes
M (30 épocas) es el modelo usado en todos los resultados. XS 16 épocas, S 12 (sub-entrenada),
L sin entrenar. Pasos con comandos para GPU local: **[LOCAL_GPU.md](LOCAL_GPU.md)**.

## Límites
- CNN entrenada solo con renders sintéticos: brecha de dominio hacia Gazebo (texturas/
  iluminación ogre2) y mucho mayor hacia una cámara real (OV2640). La fusión con ToF
  acota los fantasmas dentro de los 40° centrales; fuera de ese FOV manda la CNN.
- Gazebo no se ejecutó aquí. Modo `motors`: signo del torque de yaw de gz no verificado
  (`YAW_SIGN=-1` si el yaw diverge); marco del Twist del plugin configurable (`twist_frame`).
- El lazo cerrado usa estado verdadero (+ ruido de giróscopo), sin estimador ni vibración.
- Latencias/energía del ESP32-S3 siguen siendo estimaciones de literatura.
