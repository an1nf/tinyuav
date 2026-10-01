> **Documento histórico** de la estructura plana anterior (`uav_v2`/`uav_v3`). Los comandos y rutas
> ya no son válidos; ver [docs/migracion.md](../../migracion.md) para su equivalente con `nanouav`.

# 03 · CNN de profundidad (v2)

## Interfaz
```python
from _03_cnn.cnn import predict_depth_norm, predict_depth
dn = predict_depth_norm(gray_u8, 'M')          # (96,96) normalizada [0,1]
dm = predict_depth(gray_u8, None, 'M')         # metros, sin calibrar
from _03_cnn.tof_calibrator import ToFCalibrator, fuse_tof
depth = fuse_tof(ToFCalibrator().update_and_scale(dn, tof), tof)
```
Modo fake (`UAV_FAKE_CNN=1` o sin .tflite): GT + ruido, requiere `depth_gt`.

## Entrenamiento (`train_cnn.py`)
- Dataset sintético v2: 8 000 train / 1 500 val / 1 000 test + 400 vistas del world Gazebo.
  Cámara igual a Gazebo (60°, pitch 0.15 rad), profundidad z, sombras, texturas, poses perturbadas.
- Aumentación: flip, brillo, contraste, gamma, ruido. Pérdida: L1 ponderada a lo cercano
  (1 + 3·(1−y)) + gradiente multi-escala. Adam + coseno, early stopping.
- Exportación int8 con datos representativos en [0,1] → escala de entrada 1/255, zp −128.
- `--tartanair`: mezcla TartanAir (recortado de 90° a 60° de FOV).

## Arquitectura
stem Conv3×3 s2 → DW-separables 48→24→12→6 → bottleneck → decoder (1×1 a baja
resolución → upsample bilineal → Add skip → DW) → 48×48×4 → upsample → 96×96 → sigmoide.
Solo ops TFLite-Micro (CONV_2D, DEPTHWISE_CONV_2D, ADD, RESIZE_BILINEAR, LOGISTIC).

## Resultados (int8, test sintético de 1 000 escenas; world Gazebo = 400 vistas)

| Variante | Params | int8 AbsRel | δ1 | Err. dist. mín (ToF) | Recall/Prec <1 m | AbsRel world Gazebo | Arena | .tflite | CPU x86 |
|---|---|---|---|---|---|---|---|---|---|
| XS | 6.7k | 0.270 | 0.58 | 0.35 m | 0.81/0.75 | 0.106 | 81 KB | 36 KB | 1.2 ms |
| S | 11.6k | 0.306 | 0.59 | 0.35 m | 0.81/0.74 | 0.156 | 81 KB | 43 KB | 1.1 ms |
| M | 25.9k | 0.222 | 0.66 | 0.29 m | 0.80/0.81 | 0.082 | 81 KB | 66 KB | 1.2 ms |

Float vs int8: diferencia < 0.005 en AbsRel en todas las variantes (la cuantización ya no degrada).

Arena = cota inferior por análisis de vida de tensores del grafo TFLite (sumar ~10–20 %
de overhead TFLM). Latencia medida en la CPU del entorno (x86, 1 hilo), no en el ESP32-S3.

## Fusión con ToF
El VL53L5CX reporta el objeto más cercano de cada zona → ningún píxel de la CNN dentro
de una zona puede estar mucho más cerca que esa lectura. `fuse_tof` eleva los píxeles
inconsistentes (fantasmas) a 0.8·ToF, o a 3 m si la zona ve "lejos".
