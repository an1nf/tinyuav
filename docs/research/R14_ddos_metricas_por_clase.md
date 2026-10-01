# R14 · DDOS (Kolbeinsson & Mikolajczyk, arXiv 2312.12494v2, 2024) → métricas por clase y fusión ToF bidireccional

## El paper en breve
- Dataset sintético hecho en AirSim: 340 vuelos × 100 cuadros = 34 000 cuadros a 10 Hz, en 1280×720 RGB.
- Anotaciones: profundidad de 0 a 100 m, segmentación en 10 clases, flujo óptico, normales y pose/velocidad/aceleración.
- Clases "finas": *ultra thin* (cables), *thin* (postes, señales), *small mesh* (cercas, redes), *large mesh* (torres).
- Clima variado: lluvia, niebla, nieve, noche. Incluye encuentros cercanos y choques leves.
- Partición 300/20/20 vuelos, licencia CC BY-NC 4.0, disponible en Hugging Face.
- **Aporte metodológico:** AbsRel por clase. **Resultado clave:** DepthFormer tiene AbsRel global 0.136, que parece excelente, pero en la clase *ultra thin* da 0.998. Los tres modelos del estado del arte fallan por completo en cables (AbsRel 0.95–1.04).

## Lectura crítica frente a nuestro régimen
| | DDOS | Nosotros |
|---|---|---|
| Altitud / velocidad | 1–25 m / ~10 m/s | 0.65 m / 0.5–1 m/s |
| Rango de profundidad | 0–100 m | 0.2–4 m |
| Imagen | RGB 1280×720 | gris 96×96 |
| Modelos | transformers grandes | U-Net int8 de 26 k parámetros |

Las cifras de los dos trabajos no son comparables directamente. Además, la métrica de DDOS es simétrica: no distingue el error peligroso (ver el obstáculo más lejos de lo que está) del conservador. Tampoco se mide en lazo cerrado.

## Qué rescatamos: `r14_class_metrics.py` (600 escenas, CNN M int8)
Las máscaras son exactas: cada elemento se renderiza con y sin él. Las regiones son:
- cable: barra horizontal de Ø 2 cm;
- poste: r de 1.5 a 5 cm, más fino que lo visto en entrenamiento;
- hueco: píxeles vistos a través de una abertura;
- borde: banda de 2 px en discontinuidades de profundidad;
- resto de obstáculos y suelo.

Además del AbsRel se miden dos tasas de seguridad: **over** = P(pred > 1.25·gt), que es peligrosa, y **under** = P(pred < 0.8·gt), que da fantasmas o cierra huecos.

Resultados con la fusión ToF actual (en vuelo):

| Región | Rango | AbsRel | over | under |
|---|---|---|---|---|
| cable | < 1 m | 2.47 | **0.89** | 0.02 |
| poste | < 1 m | 1.03 | **0.87** | 0.01 |
| borde | < 1 m | 0.66 | 0.50 | 0.06 |
| obstáculo | < 1 m | 0.46 | **0.35** | 0.09 |
| hueco | 2–4 m | 0.19 | 0.05 | **0.31** |
| suelo | 1–2 m | 0.14 | 0.09 | 0.11 |
| **global** | | **0.30** | | |

1. **Se replica el hallazgo de DDOS en nuestro régimen.** Cables y postes finos son invisibles para M: en el 87–89 % de sus píxeles cercanos se predice el fondo. El AbsRel global lo esconde.
2. **Hallazgo propio (seguridad):** en el 35 % de los píxeles de obstáculos a menos de 1 m, la CNN los ve más de 25 % más lejos. Ese es el error que produce choques, y la fusión actual (`fuse_tof`) no lo corrige porque solo sube la profundidad.
3. **Confirmación cuantitativa de R13:** en huecos a 2–4 m, el 31 % de los píxeles se predicen demasiado cerca. Por eso las puertas aparecen "rellenas" en el mapa.

## Contribución nueva: `fuse_tof2`, consistencia ToF↔CNN en ambos sentidos
Está en `_03_cnn/tof_calibrator.py` y se activa con `UAV_FUSE2=1`. La idea: el ToF reporta el percentil bajo de cada zona. Si ese mismo percentil de la CNN en la zona queda más de 1.1× por encima de la lectura, la CNN no está viendo el objeto que el ToF sí ve, y la zona se reescala hacia la lectura. Cuesta 64 percentiles por cuadro, sin reentrenar y sin memoria extra.

| | fuse_tof | fuse_tof2 |
|---|---|---|
| AbsRel global | 0.301 | **0.252** (−16 %) |
| over, obstáculos < 1 m | 0.35 | **0.25** (−29 %) |
| over, postes < 1 m (AbsRel) | 0.87 (1.03) | **0.61** (0.69) |
| over, bordes < 1 m | 0.50 | **0.38** |
| under, huecos 2–4 m | 0.31 | 0.33 (sin cambio) |

**Robustez:** si el ToF simulado usa el percentil 10 o el 40 (y no el 25 que asume la fusión), el AbsRel global baja igual, de 0.30 a 0.25.

**En lazo cerrado** (MH, X3, mismas semillas):
- obstaculos: 4/4 con y sin fuse_tof2;
- corredor: 6/6 con y sin;
- estrechos w = 1.2: 1/3 con fuse_tof2 frente a 2/3 sin (las mismas semillas; sin choques en ningún caso).

Con estos tamaños de muestra no se ve diferencia en vuelo. Es la misma situación de M_ToF: la mejora fuera de vuelo no se traduce todavía en éxito en vuelo. Hacen falta n ≥ 50 y escenarios con postes finos.

## Cómo puede usarse DDOS en el proyecto
1. **Evaluación de estructuras finas en datos ajenos a nuestro renderer:** el subconjunto de baja altitud y encuentros cercanos, en gris 96×96 con la profundidad recortada a 4 m.
2. **Diversidad de dominio:** texturas de Unreal y clima para preentrenar o hacer ajuste fino. Debería reducir la brecha con Gazebo y con imágenes reales. La licencia NC permite uso académico.
3. **Sus etiquetas de segmentación** pueden alimentar una cabeza auxiliar de bordes y estructuras finas que solo exista durante el entrenamiento (sin costo en el microcontrolador).

## Siguientes pasos
- **Datos (barato):** agregar postes de r 1–5 cm y cables al generador de entrenamiento. Hoy los cilindros tienen r ≥ 0.05 y no hay cables. Medir con `r14_class_metrics.py`.
- **Pérdida (GPU):** agregar al entrenamiento de M un término de gradientes y otro ponderado en discontinuidades. Objetivo: bajar *under* en huecos y *over* en bordes.
- **Métrica para el paper:** reportar siempre over/under por región y rango, además del AbsRel global.
- **Lazo cerrado:** escenario de postes finos (bosque con r de 2–5 cm) con y sin fuse_tof2, n = 50.
