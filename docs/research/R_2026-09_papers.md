# Investigación · septiembre 2026 — qué se rescata de cada paper

Todo lo de aquí corrió en CPU (2 núcleos), sin GPU. Los números son de simulación.

## Papers
- **[P1]** Su, Zhang, Meng. *AI on the wing: fully on-board visual servoing for object tracking with
  autonomous nano-UAVs.* EAAI 177:114890 (2026). FeatherYOLO (20 k parámetros, 1.94 MMAC,
  224 kB) en GAP8 + guiado IBVS en Crazyflie 2.1; TSR 36/40.
- **[P2]** Nunn, Nikoubin. *Efficient monocular depth estimation on embedded systems with neural
  cellular automata.* SN Comput. Sci. 7:755 (2026). LNCA: encoder ×8 → NCA con pesos compartidos →
  PixelShuffle; 40.8 k parámetros; MidAir.

## R1 · Huella de LNCA a 96×96 int8 (`r1_lnca_footprint.py`)
P2 afirma que en el ESP32-S3 "todavía no es viable" (1.36 MiB en fp32 a 160×120). A nuestra
resolución y en int8 sí cabe: la arena es de 76–104 KB (M: 81–88 KB) y el límite es 180 KB.
Coste: 1.4–2.7× los MACs de M (5.9–11.6 frente a 4.3 MMAC). Se detectó un riesgo de
despliegue: el `DEPTH_TO_SPACE` int8 (v2) no está registrado en el runtime TFLite estándar. Se
reemplaza por una `TRANSPOSE_CONV` 8×8/s8 con pesos one-hot fijos, verificada idéntica (error 0.0).

## R3 · Entrenamiento comparativo, mismo presupuesto (`r3_lnca_train.py`)
4000 escenas, 15 épocas, misma pérdida y aumentación. Evaluación int8.

| modelo | parámetros | MMAC | arena | AbsRel test | minErr test | AbsRel Gazebo | rec/prec <1 m Gazebo |
|---|---|---|---|---|---|---|---|
| M (actual) | 24.4 k | 4.28 | 81 KB | 0.265 | 0.326 | 0.102 | 0.58 / 0.43 |
| M + cal. ToF (pipeline actual) | = | = | = | 0.191 | 0.193 | 0.104 | 0.80 / 0.50 |
| LNCA24 | 26.5 k | 7.26 | 76 KB | 0.296 | 0.369 | 0.124 | 0.67 / 0.44 |
| LNCA24 + ToF temprano | 26.6 k | 7.26 | 77 KB | 0.128 | 0.137 | 0.079 | 0.93 / 0.72 |
| **M + ToF temprano** | 24.6 k | 4.29 | 81 KB | **0.084** | **0.081** | **0.047** | **0.93 / 0.89** |

Periferia, fuera de los 40° del ToF (AbsRel test / Gazebo): M+cal 0.210 / 0.113 → M_ToF 0.103 / 0.052.
La escala sí se propaga fuera del FOV del ToF. Sin ToF en inferencia, M_ToF queda en 0.307 / 0.122,
cerca de M sin calibrar (0.262 / 0.107): degrada con suavidad.

**Conclusión:** la LNCA, a igualdad de parámetros, no supera a nuestra U-Net depthwise. Lo que sí
vale de P2 es la idea de la rejilla latente 12×12: sus 8 celdas centrales coinciden 1:1 con las 8
zonas del ToF (celda = 8 px ≈ 5° = zona). Inyectar el ToF ahí (conv 1×1, +144 parámetros) reduce
el AbsRel un 56 % frente al pipeline actual. Limitación: el ToF simulado sale del GT con ~2 % de
ruido y 15 % de abandono. El VL53L5CX real tiene multitrayecto, luz solar y zonas "sigma" inválidas.

## R2 · Seguimiento con detector realista (`r2_follow_sensors.py`, nano, n = 6 por celda)
El E4 original medía el objetivo con un oráculo (posición + 5 cm, sin FOV). `VisualTargetSensor`
replica la medida de P1: solo hay detección dentro del FOV y sin oclusión; el rumbo sale del
X_offset y el rango del tamaño aparente, con un sesgo por episodio del tamaño supuesto (±10 %);
hay falsos negativos. La variante `bbox_tof` (propia) toma el rango del ToF/CNN dentro de los 40°.
La capa `guidance='ibvs'` aplica la ablación G0→G6 de P1: histéresis, paso-bajo α = 0.3 y
retroceso acotado a 0.35 m/s (su única colisión ocurrió retrocediendo).

La medida realista expuso dos bugs que el oráculo ocultaba, y quedan corregidos:
1. Tras perder el objetivo, el filtro α-β extrapolaba indefinidamente y no había búsqueda.
   Ahora la extrapolación se acota a 0.6 s y sigue una reacquisición: hover + barrido de yaw hacia
   el lado de salida.
2. En `evasion.py` (ley 'brake'), el creep anti-bloqueo duraba un frame (el contador se reseteaba
   con lim = V_CREEP ≥ 0.1) y la huella del propio dron no se marcaba como vista. El resultado era
   un dron parado en espacio libre. Es probablemente la causa de los timeouts de E4 que quedaron
   pendientes en v3. Waypoints nano a 1 m/s: semilla 0 pasa de 47.5 → 39.0 s, el resto sin cambios.
   E3 no se ve afectado (usa la ley 'ramp').

Tras las correcciones: 48/48 éxitos. Error medio de seguimiento (v_obj 0.4 / 0.8 m/s):
- oráculo 0.15 / 0.34 m;
- bbox 0.23 / 0.37 m;
- bbox + ibvs 0.24 / 0.34 m (p95 1.07 → 0.82 m y jerk 6.1 → 5.6 m/s³ a 0.8 m/s);
- bbox_tof + ibvs 0.14 / 0.37 m: a baja velocidad recupera el nivel del oráculo; a 0.8 m/s domina
  el retardo, no el rango.

## Protocolo estadístico (de P1)
Reportar las tasas de éxito con IC95 de Clopper-Pearson:

| resultado | IC95 |
|---|---|
| 11/12 (x3 E3) | [61.5, 99.8] % |
| 12/12 (nano E3) | [73.5, 100] % |
| 36/40 (P1) | [76.3, 97.2] % |
| 30/30 | [88.4, 100] % |

Con n = 12 no se pueden separar variantes; hace falta n ≥ 30.
