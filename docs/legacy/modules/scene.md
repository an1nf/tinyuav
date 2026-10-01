# 00 · Generador de escenas sintéticas

## Qué es

Produce pares `(imagen_gris, mapa_de_profundidad)` que simulan lo que vería
la cámara OV2640 de 96×96 píxeles del dron volando hacia un obstáculo.

## Qué NO es

No es un renderizador 3D. El obstáculo es siempre un blob circular animado
por funciones seno. Suficiente para probar el pipeline; insuficiente para
evaluar la CNN con escenas reales.

## Por qué 96×96

El ESP32-S3 tiene ~512 KB de SRAM interna. Una imagen 96×96 en uint8 = 9.2 KB.
El tensor de entrada de la CNN a esa resolución cabe en SRAM; a 224×224 no.

## Salidas

- `gray`: array uint8 (96, 96) — imagen de cámara simulada
- `depth`: array float32 (96, 96) — profundidad en metros [0.2, 4.0]

## Cómo usar

```python
from scene import synthetic_scene
gray, depth = synthetic_scene(t=0.0)   # t controla la animación
```
