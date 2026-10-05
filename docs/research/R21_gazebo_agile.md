# R21 · El estudiante de R18 en Gazebo

*2026-10-05 · nanouav 0.18.0 · sobre R18.*

## Para qué

Ver y medir la política de R18 fuera del simulador donde se entrenó:
- otro renderizador (ogre2 de Gazebo, con la profundidad SGM calculada sobre esas imágenes);
- otra física (los motores y el controlador de velocidad del X3 de Gazebo);
- la misma geometría, para comparar episodio por episodio con `r18 eval`.

## Qué se montó

| pieza | qué hace |
|---|---|
| `gazebo/tools/gen_world.py --agile forest --seed 1000` | el mundo de prueba de R18: los mismos troncos, texturas y sol que `nanouav.agile.envs.TEST`, con el suelo cubriendo los 60 × 30 m |
| `gazebo/models/agile_x3` (lo genera `gen_world`) | X3 con la plataforma de R18: rotores hasta 1250 rad/s (empuje/peso ≈ 3.6, el X3 original tiene 1.5), caja de colisión de 0.3 m (el estudiante se entrenó para 0.2 m de radio; la del X3 mide 0.47 m), par estéreo de `StereoRig` (320 × 240, 90°, base 0.1 m, 15 Hz) |
| control de velocidad del X3 | límites 12 m/s y 15 m/s², ganancia 5 (respuesta de primer orden con τ = 0.2 s) |
| `nanouav.agile.gz_core.AgileCore` | la lógica del nodo sin ROS. Recibe el par estéreo y la odometría; calcula la profundidad SGM, ω y el plan con el mismo `fit_plan` / `choose` de `Rollout` y sus opciones. Manda v = v_ref + τ·a_ref + k_p·(p_ref − p); el término τ·a_ref compensa el retraso del controlador. La velocidad planeada sube como mucho 1.5 m/s sobre la actual: en Gazebo se parte de hover, no a v_des |
| `gazebo/nodes/agile_node.py` | despegue en A hasta 1.5 m, misión hasta la línea de meta y resultado + traza en `results/gz_runs/agile/*.json` |
| `gazebo/run_agile.sh`, `gz_docker.sh agile` | genera el mundo y lanza Gazebo, el puente y el nodo, un episodio por semilla, con un resumen al final |
| `nanouav run r18 export` | `.keras` → `.tflite` (el contenedor tiene LiteRT, no TensorFlow). Comprueba que la salida coincide; inferencia en ~1 ms en CPU |
| `nanouav run r18 video --from-gz` | el vídeo de un vuelo en Gazebo con las mismas vistas que `r18 video` |

## Validación (sin Gazebo)

- **El lazo completo de `AgileCore`.** Se probó con nuestro renderizador y un dron de velocidad con
  retraso de primer orden (`LagDrone`, la respuesta del controlador del X3). Con predicciones
  perfectas (el camino global como única hipótesis), llega a la meta a 3 y 7 m/s con más de 0.3 m de
  margen. Así un fallo en Gazebo será de la red o de la física, no del cableado.
- **El nodo** se ejecutó sobre rclpy falso (`tests/stubs`): despegue, misión, Twist hacia adelante,
  choque detectado y JSON escrito.
- **El TFLite** coincide con el modelo Keras hasta 1e-6.
- **No se probó dentro de Gazebo** desde este entorno. Las primeras corridas conviene hacerlas a
  3 m/s y con la interfaz abierta.

## Cómo correrlo

```bash
nanouav run r18 export --model results/r18/student_train.keras            # → student_train.tflite
AGILE_MODEL=results/r18/student_train.tflite AGILE_SEEDS=1000 AGILE_V=3 bash gazebo/gz_docker.sh agile
AGILE_MODEL=results/r18/student_train.tflite AGILE_SEEDS=1000,1001,1002,1003,1004 AGILE_V=3 \
    AGILE_TSCALE=arc AGILE_VCAP=1.3 HEADLESS=true bash gazebo/gz_docker.sh agile
nanouav run r18 video --from-gz results/gz_runs/agile/student_train_forest_s1000_v3.json
```

`AGILE_KEEP_DEPTH=true` guarda también la profundidad que vio la red, para el panel del vídeo.

## Qué esperar

- **Peor que en nuestro simulador.** Por el cambio de renderizador, el SGM sobre otras imágenes, la
  dinámica del X3 (su controlador no recibe la aceleración, solo la velocidad) y el arranque desde
  hover.
- **Lo que hay que mirar primero:**
  - que el despegue y el seguimiento del plan sean estables (sin oscilación en el vídeo);
  - la β y la velocidad máxima (`max_speed` en el JSON);
  - la tasa de éxito a 3 m/s comparada con `r18 eval` en las mismas semillas.
- **Flightmare**, el simulador del paper, necesita Ubuntu 20.04, ROS Noetic y su ejecutable de Unity,
  y ya no se mantiene. Iría en otro contenedor. Queda para después, si Gazebo no basta.
