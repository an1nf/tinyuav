> **Documento histórico** de la estructura plana anterior (`uav_v2`/`uav_v3`). Los comandos y rutas
> ya no son válidos; ver [docs/migracion.md](../migracion.md) para su equivalente con `nanouav`.

# Procesos pendientes (cómputo local, GPU 8 GB)

Todo lo de abajo corre desde la raíz del proyecto. La CNN usa ~1–2 GB de VRAM con
batch 128; 8 GB sobra. El renderer del dataset y la simulación E3 son **CPU** (numpy).

## 0. Entorno
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: WSL2 (TF-GPU no corre nativo)
pip install "tensorflow[and-cuda]" ai-edge-litert opencv-python matplotlib pillow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 1. Reentrenar las 4 variantes con presupuesto completo  (PENDIENTE — requerido)
Estado entregado: M 30 épocas (usable), XS 16, S 12 (sub-entrenada), **L sin entrenar**.
```bash
# dataset más grande (se genera 1 vez en CPU y queda en data/synth_v2.npz; ~5–10 min)
rm -f data/synth_v2.npz
python train_cnn.py --variants XS,S,M,L --epochs 60 --n-train 30000 --batch 128 --lr 4e-3
```
- Salida: `_03_cnn/model_{XS,S,M,L}.tflite`, `model.tflite` (= M), `metrics.json`, `assets/cnn_v2_*.png`.
- Criterios de aceptación: arena ≤ 180 KB, int8 ≈ float (ΔAbsRel < 0.01), recall obstáculo <1 m ≥ 0.85.
- Con TartanAir en `data/tartanair/moge_dataset_tartanair/`: añadir `--tartanair`.
- Si el lote de 30k no cabe en RAM (≈ 30000·96·96·3 B ≈ 830 MB, OK con 16 GB) bajar `--n-train`.

## 2. Validar en lazo cerrado con los modelos nuevos  (PENDIENTE — requerido)
```bash
python e3_closed_loop.py --profile x3   --n 30     # ~35 min CPU
python e3_closed_loop.py --profile nano --n 30
# comparar variantes (la que irá al ESP32):
python e3_closed_loop.py --profile nano --n 30 --only-mc --variant S
```
Salida: `assets/e3_results_*.json`, `assets/e3_{gazebo,kick,montecarlo}_*.png`.
Para paralelizar en varios núcleos: lanzar varios procesos con distinto `--profile/--variant`.

## 3. Inflado lateral del mapa  (OPCIONAL — cierra la colisión residual del X3)
La única colisión restante (X3, mundo MC 11) fue un roce con la esquina lateral de una caja
vista de canto. Probar:
```bash
python e3_closed_loop.py --profile x3 --n 30 --only-mc --inflate 0.1
```
Si mejora sin aumentar timeouts, fijar `inflate=0.1` por defecto en `EvasionPlanner`.

## 4. Gazebo (Docker, necesita GPU para ogre2)  (PENDIENTE — no probado aquí)
```bash
docker build -f Dockerfile.e2c -t nano_uav_sim:fixed .     # imagen base ROS 2 Jazzy + Gazebo Harmonic
docker build -f Dockerfile.sim -t nano_uav_sim:v12 .
xhost +local:docker
docker run --rm -it --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix nano_uav_sim:v12
# dentro: modo por defecto (plugin de velocidad)
bash gazebo/run_sim.sh
# control propio hasta motores (experimental)
CONTROL_MODE=motors bash gazebo/run_sim.sh
CONTROL_MODE=motors YAW_SIGN=-1 bash gazebo/run_sim.sh   # si el yaw diverge
```
Verificar, en este orden:
1. `ros2 topic hz /model/nano_uav/odometry` ≈ 250 Hz y `/nano_uav/imu` ≈ 250 Hz.
2. Modo velocity: si el dron se desvía al girar el yaw, el plugin espera Twist en marco
   mundo → `ros2 run ... perception_node.py --ros-args -p twist_frame:=world`.
3. Modo motors: primero en el suelo (motores deben igualarse ~660 rad/s al despegar),
   luego despegue. Si el yaw gira sin control → `YAW_SIGN=-1`.
4. Comparar la profundidad CNN del HUD contra `/nano_uav/depth/image` (brecha de dominio
   renderer → ogre2). Si es grande: capturar imágenes de Gazebo y afinar (ver 5).

## 5. Afinado con imágenes de Gazebo  (OPCIONAL — reduce brecha de dominio)
No hay script de captura aún: grabar `ros2 bag record /nano_uav/camera/image /nano_uav/depth/image`
y convertir a npz (gray uint8 96×96 + depth z en m). La cámara de profundidad del SDF es 8×8;
para GT denso cambiar temporalmente `depth_tof` a 96×96 con HFOV 1.047.

## 6. Empaquetar
```bash
./pack.sh ../uav_v2.zip
```

## 7. Misiones y alta velocidad (E4) — EN DESARROLLO
```bash
python e4_missions.py --profile x3 --n 6                                # todo
python e4_missions.py --profile x3 --only popup,speed --speeds 1.0,1.5,2.0 --n 10
python e4_missions.py --profile nano --only follow --n 6
UAV_FAKE_CNN=1 python e4_missions.py --only waypoints --speeds 1.0 --n 3   # depuración rápida
```
Salida: `assets/e4_results_{perfil}.json`, `assets/e4_missions_{perfil}.png`.
Estado al entregar: sin colisiones en las pruebas cortas, pero con timeouts (el dron es
conservador: chequeo de espacio desconocido + frenado físico). Ajustar en
`_04_policy/evasion.py`: `T_REACT`, `A_BRAKE`, `V_CREEP`, `unknown_along(L=…)`, `W_LANE`.

## 8. LNCA de profundidad (R1/R3, basado en Nunn & Nikoubin 2026) — NUEVO
Arquitectura en `_03_cnn/lnca_zoo.py`. Huella int8 sin entrenar: `python r1_lnca_footprint.py`.
Comparación a presupuesto completo (mismos datos/pérdida que M):
```bash
python r3_lnca_train.py --n-train 30000 --epochs 60 --batch 128 --lr 4e-3 \
       --models M,LNCA24,LNCA24_ToF,LNCA32ds_ToF
```
Salida: `assets/r3_lnca_train.json`, `_03_cnn/r3/*.tflite`. Criterio para adoptar LNCA+ToF:
recall<1 m ≥ M, min_dist_mae ≤ M+cal, arena ≤ 180 KB. Verificar en esp-tflite-micro que
TRANSPOSE_CONV int8 está registrado (se usa como PixelShuffle exacto, ver R1).

## 9. Seguimiento con detector realista (R2, basado en Su et al. 2026) — NUEVO
```bash
python r2_follow_sensors.py --profile nano --n 20 --vt 0.4,0.8
python r2_follow_sensors.py --profile x3   --n 20 --vt 0.4,0.8
```
Reportar TSR con IC95 Clopper-Pearson (criterio de fallo: colisión o pérdida > 5 s).

## 10. Vuelo rápido (R4–R8, basado en Loquercio et al. 2021) — NUEVO
Ver `research/R_2026-09_loquercio.md`. Modo `EvasionPlanner.CONSISTENT` (desactivado por defecto).
```bash
python r4_speed_limit.py                                       # cota teórica, segundos
python r5_pole.py --law consistent --n 10                      # poste, CNN real
UAV_FAKE_CNN=1 python r5_pole.py --law consistent --n 10       # poste, profundidad privilegiada
python r6_corridor_consistent.py --profile x3 --law consistent --speeds 1.0,1.5,2.0 --n 30
python r8_state_noise.py --law consistent --v 1.0 --n 20 --thrust
```

## 11. Obstáculos dinámicos y deriva (R9–R10, Shang et al. 2025) — NUEVO
Ver `research/R_2026-09_shang_neuromorphic.md`. Flags: `RF_DYN`, `EVADE_AGGR`, `VO`, `DRIFT_CORR` (apagados).
```bash
python r10_dynamic.py --scen head  --rf --aggr --n 30 --vobs 0.5,1,2,3
python r10_dynamic.py --scen cross --rf --aggr --n 30 --vobs 0.3,0.5,1,2
python r8_state_noise.py --law consistent --levels "solo sesgo 0.20" --vo --n 20 --thrust
```

## 12. Velocidad desde la cámara (R11, TinyNav) — NUEVO
```bash
python r11_vel_net.py gen --n 20000 --seed 1 --out data/r11_big.npz     # ~20 min CPU
python r11_vel_net.py train --data data/r11_big.npz --epochs 80        # redes con más datos (GPU)
python r11_vel_net.py genfull --n 800 --seed 33 --out data/r11_full_a.npz
python r11b_flow_deck.py data/r11_full_a.npz                            # flow deck frontal geométrico
```
