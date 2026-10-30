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

## Método

### Percepción

- Par estéreo de 320 × 240, base de 0.1 m y ~90° de campo. En el simulador propio lo dibuja el renderer;
  en Flightmare, Unity. Profundidad por SGM (`cv2.StereoSGBM`) con filtro de textura. Filtro de bordes (`--edge`): descarta los píxeles con un salto de profundidad hacia un vecino mayor que
  max(0.3, 0.03 z²) m, o con un vecino sin dato. Quita los puntos que el SGM deja en el espacio libre al
  mezclar un tronco con el fondo. Entrada de la red: profundidad inversa de 16 × 12, con el píxel válido más cercano de cada bloque.

### Política de navegación (R25)

- Red: CNN de tres capas. Entrenamiento por física diferenciable (DiffPhysDrone, Zhang et al. 2025), sin experto ni etiquetas. El
  gradiente atraviesa un simulador de masa puntual en TensorFlow con un paso de latencia, respuesta de
  primer orden de la aceleración (λ = 12 s⁻¹). Δt = 1/15 s, 150 pasos (10 s), lotes de 64 bosques de Poisson.

## Instalación

```bash
git clone https://github.com/an1nf/tinyuav.git && cd tinyuav
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,tflite,flightmare,dev]"     # CLI: nanouav
nanouav doctor
```
