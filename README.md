# tinyuav

Navegación rápida entre obstáculos y seguimiento de objetivos con percepción a bordo (estéreo + SGM),
políticas de ~0.4 M parámetros y evaluación en lazo cerrado en un simulador propio, en Flightmare (Unity)
y en Gazebo Harmonic / ROS 2 Jazzy.

<p align="center">
  <a href="docs/media/flightmare_warehouse_10ms.mp4">
    <img src="docs/media/flightmare_warehouse_10ms.webp" width="100%" alt="DiffPhys a 10 m/s en el almacén de Flightmare">
  </a>
</p>

*Flightmare, almacén, 10 m/s. Política DiffPhys (R25), que no se entrenó en este entorno. Izquierda:
cámara del dron y trayectoria elegida (marcas a 0.4 s y 1.0 s). Derecha: profundidad SGM y mapa local
con el SGM del último segundo. [MP4](docs/media/flightmare_warehouse_10ms.mp4)*

## Resultados

| tarea | entorno | resultado | referencia |
|---|---|---|---|
| navegar 40 m a 3–10 m/s | bosque de R18, 120 vuelos | **111/120**; 29/30 a 10 m/s | Loquercio et al. 2021: ~60 % a 10 m/s¹ |
| navegar 40 m a 3–10 m/s | almacén de Flightmare, 120 vuelos | **117/120**, 2 choques | piloto reactivo (R24): 104/120 |
| seguir un globo a 3–7 m/s | bosque de 1/16 m⁻², 150 vuelos | **147/150**, 0 choques | YOPOv2-Tracker: 88 %¹ (Fisher p = 0.008) |
| latencia de la política | 1 hilo de CPU | **1.1 ms** por decisión | — |

¹ Simuladores distintos; densidad y velocidades como en cada paper. Montaje, IC y pruebas emparejadas en
[R25](docs/research/R25_diffphys.md) y [R26](docs/research/R26_diffphys_seguimiento.md).

## Método

### Percepción

- Par estéreo de 320 × 240, base de 0.1 m y ~90° de campo. En el simulador propio lo dibuja el renderer;
  en Flightmare, Unity. Profundidad por SGM (`cv2.StereoSGBM`) con filtro de textura.
- Filtro de bordes (`--edge`): descarta los píxeles con un salto de profundidad hacia un vecino mayor que
  max(0.3, 0.03 z²) m, o con un vecino sin dato. Quita los puntos que el SGM deja en el espacio libre al
  mezclar un tronco con el fondo.
- Entrada de la red: profundidad inversa de 16 × 12, con el píxel válido más cercano de cada bloque; los
  huecos cuentan como lejos.

### Política de navegación (R25)

- Red: CNN de tres capas (núcleos 2, 3 y 3; 32, 64 y 128 filtros), suma del estado, GRU de 192 y salida de
  aceleración; 391 k parámetros. El estado son 9 valores en el marco de rumbo: velocidad, dirección del
  empuje y velocidad deseada.
- Entrenamiento por física diferenciable (DiffPhysDrone, Zhang et al. 2025), sin experto ni etiquetas. El
  gradiente atraviesa un simulador de masa puntual en TensorFlow con un paso de latencia, respuesta de
  primer orden de la aceleración (λ = 12 s⁻¹) y arrastre lineal y cuadrático.
- Episodios: Δt = 1/15 s, 150 pasos (10 s), lotes de 64 bosques de Poisson (separación de 4–7 m, troncos de
  0.15–0.4 m) y velocidad deseada de 3 a 12 m/s.
- Las distancias a troncos y suelo son analíticas, así que el gradiente del choque es exacto. La imagen se
  obtiene por rayos (32 × 24, mínimo de cada 2 × 2), sin gradiente, con 10 % de huecos y 5 % de ruido para
  parecerse al SGM.
- Pérdida, con los pesos del paper (1, 2, 0.01 y 0.001): error de velocidad (promediada en ~2 s), barrera
  softplus sobre la holgura más la velocidad de acercamiento, aceleración y jerk.
- Vuelo previo (`--burn 0,30`): antes de cada episodio, la red vuela de 0 a 30 s sin gradiente, y el tramo
  con gradiente empieza con su GRU, sus actuadores y su velocidad de ese vuelo. Sin él, frenaba pasados
  los 10 s del entrenamiento.
- Optimización: AdamW con tasa coseno; 20 000 iteraciones a lote 64 (lr 10⁻³) en una TITAN X y dos ajustes
  de 5000 (lr 3·10⁻⁴), el segundo con vuelo previo (r25c).
- En vuelo, la red recibe su propio SGM a 15 Hz con 50 ms de latencia y conserva el estado de la GRU entre
  cuadros. La aceleración se convierte en un plan de un cuadro que sigue un controlador en cascada.

### Seguimiento (R26 y R20)

- La velocidad deseada sigue la ley de R20: hacia el punto a 4 m del objetivo, a |v_objetivo| +
  0.8 · (distancia − 4 m), como mucho 8 m/s. La cámara apunta al objetivo.
- Se añade a la pérdida una barrera softplus de visibilidad: pide que la distancia, en planta, de cada eje
  de tronco al segmento entre el dron y el objetivo, menos su radio, supere 0.3 m. Es analítica, como la holgura
  del dron, así que evitar la oclusión se aprende al entrenar y en vuelo no cuesta nada.
- Ajuste desde R25 con objetivos que zigzaguean entre los troncos (giro ≤ 60°/s): 2000 iteraciones a
  1–4 m/s y luego 1500 a 1–7 m/s con lote 64 y lr 2·10⁻⁴.
- Detección del globo sobre la imagen en color: umbral HSV, cierre morfológico y componentes conexas. La
  distancia sale del tamaño aparente, medido en la altura de la caja, porque los troncos estrechan la caja
  pero no la acortan.
- Tracker con compensación del giro de la cámara y bloqueo del objetivo. Posición y velocidad del globo
  las estima un filtro de Kalman 3D de velocidad constante; si la pista bloqueada salta más de
  0.8 m + 5 m/s · Δt, se descarta (el tracker la cambió a otro globo).
- `--fast-target` (cinco cambios en la misión, ninguno en la red):
  - ruido de proceso de 5 m/s² en lugar de 2;
  - con el globo perdido, la cámara mira a la última estimación, movida como mucho 0.5 s;
  - la puerta de re-bloqueo crece a la velocidad del globo + 1 m/s;
  - la primera medida tras re-bloquear no pasa por la puerta de innovación;
  - la predicción sin medidas sigue girando al ritmo de los últimos 0.6 s, como mucho 90°/s y 90° en total.

### Simuladores y evaluación

- Simulador propio: bosques de Poisson con el protocolo de Loquercio et al. 2021 (R18).
- Flightmare (R22): puente ZMQ en Python con el binario de Unity, sin ROS ni flightlib. La escena del
  almacén se escanea una vez y en ella se siembran troncos cuadrados de 0.5–0.7 m de lado (1/25 m⁻²). El
  globo se pinta en las dos imágenes por intersección rayo–esfera contra la profundidad de Unity.
- Gazebo Harmonic / ROS 2 Jazzy (R21): el estudiante de R18 en un X3 con estéreo, como nodo ROS 2 y
  exportado a TFLite.
- Referencias implementadas en el mismo banco: el experto M-H y el estudiante por DAgger de Loquercio
  et al. 2021 (R18) y un piloto reactivo sin red que juzga candidatas sobre su propio SGM (R24).
- Todas las políticas vuelan las mismas semillas: IC de Wilson al 95 %, McNemar sobre los pares y Fisher o
  binomial contra los resultados publicados.
- Antes de R18 (E0–R17), el proyecto era una CNN monocular int8 + ToF 8 × 8 para un ESP32-S3.

## Instalación

```bash
git clone https://github.com/an1nf/tinyuav.git && cd tinyuav
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,tflite,flightmare,dev]"     # CLI: nanouav
nanouav doctor
```

Entrenar en GPU Pascal (TITAN X): [docs/gpu.md](docs/gpu.md). Modelos, datos y resultados quedan fuera de
git: [docs/artefactos.md](docs/artefactos.md).

## Uso

```bash
nanouav list                                                   # experimentos registrados
nanouav run r25 train --burn 0,30                              # guarda results/r25/diffphys.keras (receta: R25)
nanouav run r18 eval --pilots diffphys --model results/r25/diffphys.keras --v 3,5,7,10 --n 30 --edge
nanouav run r26 train --init results/r25/diffphys.keras --name follow_fast.keras --v 1,7   # receta: R26
nanouav run r20 sim --pilots diffphys --model results/r26/follow_fast.keras \
    --density 0.0625 --speed 7 --n 30 --edge --fast-target
nanouav run r25 bench --model results/r26/follow_fast.keras    # tiempo por decisión

export NANOUAV_FLIGHTMARE=/ruta/RPG_Flightmare.x86_64          # Flightmare
nanouav run r22 eval --arena --scene warehouse --pilots diffphys --model results/r25/diffphys.keras \
    --v 3,5,7,10 --n 30 --edge
nanouav run r22 follow-video --model results/r26/follow_fast.keras --speed 7 --fast-target
```

`nanouav run <id> --help` lista las opciones; las suites YAML, en [docs/experimentos.md](docs/experimentos.md).
Gazebo: [docs/gazebo.md](docs/gazebo.md).

## Estructura

```
src/nanouav/agile/       R18 (SGM, experto, DAgger), R25 diffphys.py, R26 diffphys_follow.py
src/nanouav/follow/      detección, tracker, filtro 3D y misiones de seguimiento (R19, R20)
src/nanouav/flightmare/  puente ZMQ, arenas y seguimiento en Flightmare (R22)
src/nanouav/…            simulador propio, percepción, planificación, control, experimentos
gazebo/ docker/          mundos, nodos ROS 2 e imágenes Docker
tests/                   pytest -m "not slow"
docs/research/           un informe por iteración (R1–R26)
```

## Licencia

Por definir.
