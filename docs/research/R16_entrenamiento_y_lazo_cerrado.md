# R16 · Entrenamiento a presupuesto completo, M_ToF en lazo cerrado y límites del sensor

*2026-10-01 · nanouav 0.13.x · TITAN X (Pascal) para entrenar, CPU para simular.*

## Pregunta

1. ¿Cuánto mejoran las CNN de profundidad con el presupuesto de entrenamiento completo
   (60 épocas × 30 000 imágenes)?
2. ¿La fusión temprana del ToF dentro de la red (**M_ToF**) mejora el vuelo en lazo cerrado o solo
   las métricas de profundidad?
3. ¿Entrenar con postes finos y cables (R14) resuelve el punto ciego de la red?

## 1 · Entrenamiento

Todas las variantes con 60 épocas, 30 000 escenas, batch 128, lr 4e-3. Métricas int8 sobre el
conjunto de prueba sintético (1 000 escenas, semillas fijas, comparable con las versiones anteriores).

| variante | AbsRel | recall obstáculo < 1 m | error dist. mínima | arena | .tflite |
|---|---|---|---|---|---|
| XS | 0.225 (antes 0.270) | 0.82 | 0.30 m | 81 KB | 36 KB |
| S | 0.227 (antes 0.306) | 0.81 | 0.29 m | 81 KB | 43 KB |
| **M** | **0.189** (antes 0.222) | **0.87** (antes 0.80) | 0.25 m | 81 KB | 66 KB |
| L | 0.180 | 0.81 | 0.24 m | 90 KB | 112 KB |
| M_thin (`--thin 0.3`) | 0.194 | 0.84 | 0.25 m | 81 KB | 66 KB |

M_ToF (R3, fusión temprana del ToF, iniciada desde el M nuevo), en el conjunto de prueba de R3:
AbsRel **0.057**, error de distancia mínima **0.058 m**, recall < 1 m **0.97** (con el presupuesto
corto de R3: 0.084 / 0.081 m / 0.93).

**Lectura.** M sigue siendo la variante para el ESP32: L mejora poco el AbsRel pero pierde recall
cercano y no cabe tan holgada; S no aporta sobre XS. Solo M (0.87) y M_ToF (0.97) cumplen el criterio
de recall ≥ 0.85.

## 2 · Percepción por regiones (R14, 600 escenas con postes, cables y huecos)

AbsRel global por pipeline:

| pipeline | M | M_thin | M_ToF | M_ToF_thin |
|---|---|---|---|---|
| red sola | 0.326 | 0.316 | 0.120 | 0.127 |
| + calibración ToF | 0.264 | 0.253 | 0.117 | 0.125 |
| + fusión bidireccional (`flight2`) | 0.223 | 0.213 | **0.114** | 0.122 |

Sobreestimación (`over`: la red ve el objeto más lejos de lo que está; el error peligroso), `flight2`:

| región | M | M_thin | M_ToF | M_ToF_thin |
|---|---|---|---|---|
| obstáculo < 1 m | 0.24 | — | **0.05** | — |
| borde < 1 m | 0.35 | — | 0.20 | — |
| poste < 1 m | 0.56 | 0.48 | 0.31 | **0.28** |
| poste 2–4 m | 0.27 | **0.15** | 0.30 | 0.19 |
| cable < 1 m | 0.86 | 0.78 | 0.85 | **0.77** |
| cable 1–2 m | 0.79 | 0.68 | 0.88 | 0.79 |
| hueco < 1 m (`under`: hueco "cerrado") | 0.22 | — | **0.05** | — |

**Lectura.**
- M_ToF reduce el error peligroso en obstáculos cercanos a la quinta parte y casi nunca "cierra" un
  hueco libre (lo que bloqueaba al planificador en las puertas, R13).
- Entrenar con estructuras finas mejora postes y cables en las dos redes, con un costo pequeño en el
  resto (+0.005–0.008 de AbsRel; −0.03 de recall en M). Los cables siguen invisibles en su mayoría
  (77–78 % de sobreestimación a < 1 m): un cable de 2 cm a 1 m ocupa ≈ 1.8 px en la imagen de 96×96 y
  menos de 1 px más allá de ≈ 1.3 m. Es un **límite de resolución del sensor**, no de entrenamiento.
  El ToF simulado tampoco lo detecta (percentil bajo por zona; un cable cubre < 5 % de la zona).

## 3 · Lazo cerrado: M vs M_ToF (suite `baseline`, 27 trabajos)

Éxito con IC 95 % de Wilson.

| prueba | M | M_ToF | diferencia |
|---|---|---|---|
| E3 X3 (mundo tipo Gazebo + Monte Carlo) | 51/60 (85 %, 74–92), 8 col. | **60/60 (100 %, 94–100)** | significativa |
| E3 nano | 60/60 | 60/60 | — |
| corredores X3, ley *brake*, 1.5 m/s | 9/50 (18 %, 10–31) | **23/50 (46 %, 33–60)** | significativa |
| corredores X3, ley *consistent*, 1.5 m/s | 45/50 (90 %) | 47/50 (94 %) | n. s. |
| corredores nano, *consistent*, 1.0–2.0 m/s | 108/110 | 109/110 | n. s. |
| poste nano, *consistent* (GT: 56/60) | 56/60 | **58/60** | M_ToF ≥ GT |
| balón de frente, RF + escape agresivo | 40/80 (50 %, 39–61) | **54/80 (68 %, 57–77)** | IC se tocan |
| ruido de estado (R8, 6 niveles) | 144/180 (80 %) | 146/180 (81 %) | n. s. |

**Lectura.** M_ToF mejora donde la percepción era el cuello de botella y empata donde limita la ley
de control o la estimación de estado. En R8 el que hunde el éxito es el sesgo (0.20 → 53 % con ambas
redes); el ruido blanco no afecta. La ley *consistent* domina a la *brake* con cualquier red.

## 4 · Dos "regresiones" de M_ToF que no lo son

### 4.1 Poste con ley *brake* a ≥ 1.5 m/s

| velocidad | M | M_ToF | GT |
|---|---|---|---|
| 0.5 / 1.0 | 8/10 · 8/10 | 10/10 · 10/10 | 10/10 · 10/10 |
| 1.5 | 8/10 | 2/10 | 5/10 |
| ≥ 1.73 (saturada) | 2/10 | 0/10 | 1–2/10 |

Con profundidad perfecta la ley *brake* falla igual: se congela delante del poste (`v_pass` ≈ 0.04,
`max_lat` < 0.2 m) o reacciona tarde (`dodge_start_d = nan`). M "funcionaba" porque subestima la
distancia y frena antes: su sesgo era un margen de seguridad accidental. Con la ley *brake* la
velocidad se satura en 1.73 m/s, así que las filas 2.0/2.5/3.0 m/s son el mismo experimento.

### 4.2 Balón que cruza a 1.0 m/s (M 7/20, M_ToF 0/20, GT 0/20)

El escenario `cross` es un rumbo de colisión: el balón se mantiene en un rumbo constante
atan(v_obs / v_dron) mientras la distancia se acorta.

| v balón | rumbo | dentro de ±30° | resultado (GT) |
|---|---|---|---|
| 0.5 m/s | 26.6° | sí | visto desde 3.3 m, esquiva con 0.35 m |
| 1.0 m/s | 45° | no (visible a 0.29 m) | roce, 0/20 |
| 2.0 m/s | 63° | no (visible a 0.29 m) | pasa por sincronización, margen 0.14 m |

El planificador solo recibe una detección confirmada a 0.27 m. Los éxitos y fracasos a ≥ 1.0 m/s
dependen de la sincronización (el dron va a ≈ 0.93 m/s), no de la percepción. **Límite de detección
de una cámara frontal de 60°: un obstáculo en rumbo de colisión solo es visible si
v_obs < tan(30°)·v_dron ≈ 0.58·v_dron.** Por encima hace falta más campo visual (ToF laterales,
barrido de yaw o una óptica más ancha).

## 5 · Estructuras finas en lazo cerrado

Bosque de troncos de 2–5 cm (`scn:bosque;r_min=0.02;r_max=0.05`), planificador MH, n = 20:

| percepción | X3, 0.5 m/s | X3 + `fuse_tof2` | nano, 1 m/s, `gap=1.0` |
|---|---|---|---|
| M | 16/20 (4 col.) | 19/20 (1 col.) | 19/20 (1 col.) |
| M_ToF | 20/20 | 20/20 | 20/20 |
| GT | 20/20 | — | — |

`fuse_tof2` reduce las colisiones de M (4 → 1), como predijo R14. M_ToF ya alcanza a la GT: este
escenario no separa M_ToF de M_ToF_thin.

### Cables (suite `cables`, nano, 1 m/s, hueco 0.7 m, n = 20)

Tres cables a la altura de vuelo con un hueco alternado y sin rodeo posible (`scn:cables;gap=0.7`).

| percepción | cable 2 cm | cable 5 cm |
|---|---|---|
| M | 6/20 (30 %, 15–52), 12 col. | 10/20 (50 %, 30–70), 8 col. |
| M_thin | 5/20 (25 %, 11–47), 15 col. | 13/20 (65 %, 43–82), 6 col. |
| M_ToF | 11/20 (55 %, 34–74), 9 col. | 15/20 (75 %, 53–89), 5 col. |
| M_ToF_thin | 10/20 (50 %, 30–70), 10 col. | 16/20 (80 %, 58–92), 4 col. |
| GT | 15/20 (75 %, 53–89), 5 col. | 19/20 (95 %, 76–99), 1 col. |

**Lectura.**
- Con cables de 2 cm **ni la profundidad perfecta basta** (25 % de choques): a 96×96 el cable se
  resuelve demasiado tarde. Es el límite de resolución del sensor medido en lazo cerrado.
- La fusión del ToF es lo que más ayuda: M → M_ToF sube de 30 a 55 % (2 cm) y de 50 a 75 % (5 cm).
- Entrenar con estructuras finas **no cambia el resultado en vuelo**: M vs M_thin y M_ToF vs
  M_ToF_thin difieren en 1–3 episodios de 20, dentro del ruido, y en sentidos opuestos según el
  diámetro. La mejora de R14 por píxel (sobreestimación 85 → 77 %) no alcanza para esquivar.
- Con 5 cm queda una brecha de percepción de 20 puntos entre M_ToF (75 %) y la GT (95 %).

## 6 · Pasos estrechos (suite `narrow_passages`)

X3, 0.5 m/s, n = 10 (`mh` = planificador R13; `mh1` = versión R12, sin los cambios para pasos estrechos):

| w | planificador | M | M_ToF | GT |
|---|---|---|---|---|
| 1.2 m | mh | 5/10 (1 col.) | **10/10** | 10/10 |
| 1.2 m | mh1 | 0/10 | 0/10 (1 col.) | 0/10 |
| 1.0 m | mh | 0/10 (2 col.) | 0/10 | 0/10 |
| 1.0 m | mh1 | 0/10 | 0/10 | 0/10 |

Nano, 1 m/s, n = 10:

| w | M | M_ToF | GT |
|---|---|---|---|
| 0.6 m | 2/10 (1 col.) | **6/10** | 10/10 |
| 0.5 m | 0/10 | 0/10 | 3/10 (1 col.) |
| 0.45 m | — | — | 3/10 |

**Lectura.**
- **X3, w = 1.2 m: M_ToF iguala a la GT** (10/10) y M se queda en 5/10. Se confirma la predicción de
  R13/R14: M "cierra" huecos libres (22 % de *under* a < 1 m) y M_ToF casi nunca (5 %).
- **Los cambios de R13 en el planificador son imprescindibles**: `mh1` no cruza ninguna puerta, ni con
  profundidad perfecta (0/40 entre las dos anchuras).
- **X3, w = 1.0 m: 0/10 incluso con GT**, con la rejilla por defecto de 10 cm. Concuerda con R13, donde
  w = 1.0 solo se cruzó con celda de 5 cm (`NANOUAV_MH_GRID_RES=0.05`). Queda por medir M_ToF con 5 cm,
  que es la combinación que R13 proponía.
- **Nano, w = 0.6 m: M 20 % → M_ToF 60 % → GT 100 %**: la percepción importa, y M_ToF cubre la mitad
  de la brecha. Por debajo de 0.6 m falla incluso la GT.

Con el nano y profundidad perfecta, `estrechos` con w = 0.45 no se cruza (ni con rejilla de 5 cm) y
w = 0.6 se cruza con 46 m de recorrido para un tramo de ≈ 10 m: los márgenes del MH son absolutos y
están ajustados para el X3 (ver `problemas_conocidos.md`). La equivalencia "w = 0.45 nano ≈ w = 1.2
X3" de R13 no se cumple.

## 7 · Conclusiones para el paper

1. **M_ToF es la red de referencia.** Mejora la profundidad (AbsRel 0.223 → 0.114 en el pipeline de
   vuelo) y el vuelo donde la percepción limita (E3 X3 85 → 100 %, corredores *brake* 18 → 46 %,
   balón de frente 50 → 68 %), sin empeorar en ningún escenario.
2. **Una percepción más exacta expone los límites del planificador**: la ley *brake* y el escape
   dinámico dependían del sesgo conservador de M. Las comparaciones de percepción deben incluir
   siempre la cota con GT.
3. **Límites del sensor, no del algoritmo**: cables de 2 cm (resolución de 96×96) y obstáculos que
   cruzan en rumbo de colisión con v_obs ≥ 0.58·v_dron (campo visual de 60°).
4. **Datos de estructuras finas**: mejoran la profundidad por píxel en postes y cables (R14), pero no
   el vuelo entre cables (suite `cables`) y cuestan un poco en el resto: **no se adoptan**. Lo que sí
   ayuda con los cables es el ToF dentro de la red (30 → 55 % a 2 cm, 50 → 75 % a 5 cm).
5. **Pasos estrechos**: con el X3 y w = 1.2 m, M_ToF iguala a la GT (10/10 frente a 5/10 de M), y sin
   los cambios de R13 en el MH no se cruza ninguna puerta ni con GT.
6. **Métrica**: reportar sobre/subestimación por región y rango además del AbsRel global; el global
   esconde justamente los errores que causan choques.

## 8 · Limitaciones

- Todo es simulación con un renderer propio y un ToF simulado; ninguna cifra está validada con el
  sensor real (VL53L5CX) ni en Gazebo con M_ToF.
- Un solo entrenamiento por modelo: diferencias de ±0.01 AbsRel o ±0.03 de recall pueden ser varianza
  entre semillas.
- n = 10–20 en varias pruebas: varios intervalos se solapan.

## 9 · Siguientes pasos

- Gazebo con M_ToF (`VARIANT=M_ToF bash gazebo/batch_eval.sh`) para medir el salto de dominio.
- Conjunto de evaluación con la cámara y el ToF reales (unos cientos de cuadros) — imprescindible
  antes de afirmar nada sobre el mundo real.
- Márgenes del MH proporcionales al radio del vehículo (nano en pasos estrechos: con GT, w = 0.5 da
  3/10).
- X3, w = 1.0 m con rejilla de 5 cm y M_ToF:
  `NANOUAV_MH_GRID_RES=0.05 nanouav run r12 --world "scn:estrechos;w=1.0" --goals 0:0 --planner mh --n 10 --variant M_ToF`
  (y lo mismo con `--fake-cnn`).
- Cables: probar `fuse_tof2` y un percentil más bajo del ToF por zona (el cable cubre < 5 % de la zona).
- Campo visual: evaluar ToF laterales o barrido de yaw para obstáculos que cruzan.
- Repetir M y M_thin con 2–3 semillas para separar el efecto de los datos de la varianza.

## Reproducir

```bash
nanouav run train --variants XS,S,M,L --epochs 60 --n-train 30000 --batch 128 --lr 4e-3
nanouav run r03 --models M_ToF --init-from models/model_M.keras --install --n-train 30000 --epochs 60 --batch 128 --lr 4e-3
nanouav run train --variants M --thin 0.3 --suffix _thin --epochs 60 --n-train 30000 --batch 128 --lr 4e-3
nanouav run r03 --models M_ToF --thin 0.3 --suffix _thin --init-from models/model_M_thin.keras --install --n-train 30000 --epochs 60 --batch 128 --lr 4e-3
nanouav run r14 --n 600 --variants M,M_ToF            # y M,M_thin · M_ToF,M_ToF_thin
nanouav suite configs/suites/baseline.yaml -j 5
nanouav suite configs/suites/thin_structures.yaml -j 2
nanouav suite configs/suites/cables.yaml -j 5
nanouav suite configs/suites/narrow_passages.yaml -j 4
nanouav run --fake-cnn r05 --profile nano --law brake --n 10                 # §4.1
nanouav run --fake-cnn r10 --scen cross --rf --aggr --vobs 1.0 --n 20         # §4.2
nanouav run r12 --profile nano --v 1.0 --world "scn:bosque;r_min=0.02;r_max=0.05;gap=1.0" --goals 0:0 --planner mh --n 20 --variant M_ToF
```
