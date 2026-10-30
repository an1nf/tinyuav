# R26 · DiffPhys siguiendo un objetivo que se mueve, con visibilidad

*2026-10-25 a 2026-10-30 · nanouav 0.27.0–0.27.12 · sobre R25 (política recurrente entrenada por física diferenciable) y R20
(seguir un globo).*

## Objetivo

Que la política de R25 siga a un objetivo que se mueve por el bosque (un globo en R20, una persona en R19)
sin chocar y sin perderlo de vista detrás de los troncos. Es lo que mostró el IDEAS Lab de Purdue con un
corredor en un bosque denso: odometría LiDAR-inercial, mapa local, predicción de oclusiones y
optimización de trayectoria consciente de la percepción, todo a bordo. Aquí, con una red de 390 k
parámetros y una cámara estéreo.

La referencia publicada más cercana es YOPOv2-Tracker (Lu et al. 2025, arXiv 2505.06923): una red que
también va de la imagen a la trayectoria y se entrena con el gradiente de los costos, sin experto. Sigue a
una persona a 6 m/s en un bosque real, en una Orin NX.

## Qué cambia respecto a R25

| pieza | R25 (navegar) | R26 (seguir) |
|---|---|---|
| velocidad deseada (entrada de la red) | hacia la meta, a v_des | la ley de seguimiento de R19/R20: hacia el punto a 4 m del objetivo, a \|v_objetivo\| + 0.8·(distancia − 4 m), como mucho 8 m/s |
| rumbo de la cámara | hacia donde vuela | hacia el objetivo, como en R19/R20 |
| mundo | bosque de Poisson | el mismo, sin troncos a menos de 0.5 m del camino del objetivo (como en R20) |
| objetivo | — | camino suave al azar a 1–4 m/s, giro ≤ 60°/s (zigzaguea entre los troncos), altura 1–3 m que cambia despacio |
| pérdida | velocidad, choque, aceleración, tirón | lo mismo y **visibilidad** |
| duración | 10 s con gradiente, tras 0–30 s sin él (0.27.0) | igual |

**Visibilidad.** La holgura de la línea de visión es la distancia de cada eje de tronco al segmento
dron → objetivo, en el plano, menos su radio. Es analítica, como la holgura del dron. La pérdida es una
barrera softplus que pide 0.3 m (`--w-vis`, 0 la apaga). En Purdue la oclusión se predice y se evita en
línea con un optimizador; aquí se aprende al entrenar y en vuelo no cuesta nada.

**Mismo contrato en vuelo.** La red recibe lo mismo que en R25 (profundidad, velocidad, empuje y velocidad
deseada), así que `r20 sim --pilots diffphys --model …` la vuela sin cambios. La misión de seguimiento da
la dirección y la velocidad, y el rumbo hacia el objetivo.

**Vuelo previo.** Igual que en R25 (0.27.0), cada episodio de entrenamiento sigue al objetivo 0–30 s sin
gradiente antes de los 10 s con gradiente, en el mismo mundo: seguir dura minutos. El tramo con gradiente
solo ve los 128 troncos más cercanos a su camino.

## Primeros resultados (CPU)

**Seguir sin ajuste.** La política de R25 con el vuelo previo (ajuste de CPU de 0.27.0), sin ningún
entrenamiento de seguimiento, en su simulador. 64 vuelos de 60 s, objetivo a 1–4 m/s que zigzaguea hasta
60°/s:

| política | choca en 60 s | objetivo a la vista | error de distancia | lejos (> 8 m) |
|---|---|---|---|---|
| r25b | 0 % | 73.8 % | 20.6 m | 56 % del tiempo |
| r25b + `--burn` | 15.6 % | 97.6 % | 0.80 m | 1.2 % |

La primera fila es con el objetivo a ≤ 25°/s y sin troncos a menos de 1 m de su camino. r25b se cansa a los
10 s y se queda atrás. Con el vuelo previo lo sigue, pero choca en 1 de cada 6 vuelos de un minuto.

**R20 (globo, imagen renderizada, detector de color, SGM, `--edge`), sin ajuste,** semillas 3000–3005:
5/6 éxitos, 1 choque, bloqueo el 99 % del tiempo y siempre en el globo correcto, error de distancia
mediano de 0.78 m. Incluye un vuelo de 37 s a 1.1 m/s. Como referencia, en R20 la línea recta al punto
estimado daba 8/8 con 0.84 m en bosques más ralos, y el estudiante mini de R18 1/4.

**Con el ajuste de seguimiento** (150 iteraciones en CPU, lote 16, vuelo previo de 0–20 s, desde el
modelo con vuelo previo; `r26/train_cpu150_w*.log`). Mismos 64 vuelos de 60 s (semilla 123):

| política | choca en 60 s | objetivo a la vista | error de distancia | error de altura | lejos (> 8 m) |
|---|---|---|---|---|---|
| sin ajuste de seguimiento | 15.6 % | 97.6 % | 0.80 m | — | 1.2 % |
| R26 sin visibilidad (`--w-vis 0`) | 1.6 % | 98.1 % | 0.40 m | 0.10 m | 0.7 % |
| **R26 con visibilidad (`--w-vis 1`)** | **1.6 %** | **99.0 %** | **0.32 m** | **0.09 m** | **0.2 %** |

- **El ajuste de seguimiento es lo que baja los choques:** de 15.6 % a 1.6 %, con o sin visibilidad.
- **La visibilidad parte a la mitad el tiempo con el objetivo tapado** (1.9 % → 1.0 %), y el error de
  distancia baja de 0.40 a 0.32 m. Son 64 vuelos y una sola semilla de entrenamiento; la comparación de la
  GPU (256 vuelos) lo confirmará o no.
- **Sigue los cambios de altura:** 9 cm de error medio.

**R20 con R26 (con visibilidad), semillas 3000–3005** (`r26/balloon_r26_w1_cpu.json`):

| política | éxito | choques | bloqueo | en el globo correcto | error de distancia |
|---|---|---|---|---|---|
| sin ajuste de seguimiento | 5/6 | 1 | 99 % | 100 % | 0.78 m |
| R26 | **6/6** | **0** | 99 % | 100 % | 1.04 m |

Con los globos más rápidos (2.5–3.5 m/s) se queda algo más atrás que sin el ajuste (1.1–1.75 m de error,
frente a 0.8–0.9 m). En su simulador no pasa (0.32 m), así que hay que mirarlo con el SGM y el tracker en la
evaluación grande.

## En la GPU

Desde r25c (R25 con vuelo previo), 2000 iteraciones a lote 64, con visibilidad (`--w-vis 1`) y sin ella
(`--w-vis 0`); registros en `r26/train_gpu2000_w*.log`.

**En su simulador** (`r26 eval`, 256 vuelos de 60 s, los mismos para los dos; objetivo a 1–4 m/s que
zigzaguea hasta 60°/s):

| política | choca en 60 s | objetivo tapado | error de distancia | error de altura | lejos (> 8 m) |
|---|---|---|---|---|---|
| R26 sin visibilidad | 2.3 % | 1.3 % del tiempo | 0.21 m | 0.03 m | 0.1 % |
| **R26 con visibilidad** | **1.2 %** | **0.7 %** | **0.18 m** | **0.03 m** | **0 %** |

- **La visibilidad vuelve a partir a la mitad el tiempo con el objetivo tapado** (1.3 % → 0.7 %), como en la
  CPU (1.9 % → 1.0 %).
- **Los choques también bajan** (6 → 3 de 256), aunque con tan pocos casos no es concluyente.
- **Sigue al objetivo a 18 cm de la distancia pedida y a 3 cm de su altura**, durante un minuto.

**R20: seguir un globo** (imagen renderizada, detector de color, tracker, SGM con `--edge`), bosques de
1/25 m⁻² (espaciado 5 m), 30 semillas (3000–3029), globo a 1–4 m/s con cambios de altura y 0–2 globos
distractores (a veces del mismo color):

| piloto | éxito | choques | bloqueo | en el globo correcto | error de distancia | rebloqueos |
|---|---|---|---|---|---|---|
| `gbl`: sigue el rastro real (privilegiado, cota superior) | 30/30 | 0 | 99 % | 100 % | 0.13 m | 0 |
| `direct`: línea recta al punto estimado | 26/30 | 4 | 82 % | 100 % | 0.69 m | 5 |
| R26 sin visibilidad | 30/30 | 0 | 99 % | 100 % | 0.93 m | 0 |
| **R26 con visibilidad** | **30/30** | **0** | **99 %** | **100 %** | **0.71 m** | **0** |

- **Ningún choque en 60 seguimientos**, con lo que el dron percibe por sí mismo: su propio SGM y el tracker,
  sin el rastro privilegiado. La línea recta choca en 4 de 30.
- **El bloqueo se mantiene el 99 % del tiempo**, igual que con el rastro real. La línea recta lo pierde el
  18 % del tiempo.
- **Con visibilidad, el error de distancia baja de 0.93 a 0.71 m.**
- **Queda más lejos del punto pedido que el rastro real** (0.71 frente a 0.13 m). En su simulador va a
  0.18 m, así que la diferencia viene de la estimación del objetivo (tracker y filtro), no de la política.

## Objetivos más rápidos (las velocidades de YOPOv2-Tracker, 3–7 m/s)

YOPOv2-Tracker sigue a 3–7 m/s en simulación y a 6 m/s en un bosque real. En su simulador, 64 vuelos de
30 s por velocidad (`r26 eval --v v,v --seconds 30`):

| política | objetivo a 3 m/s | 5 m/s | 7 m/s |
|---|---|---|---|
| R26 de la GPU (entrenada a 1–4 m/s) | 0 % · 0.21 m | 12.5 % · 0.31 m | 32.8 % · 0.59 m |
| **+ 200 iteraciones de CPU a 1–7 m/s** (`r26/train_cpu200_fast_v1-7.log`) | **0 % · 0.19 m** | **0 % · 0.25 m** | **21.9 % · 0.49 m** |

(choques en 30 s · error de distancia medio; el objetivo está a la vista el 98.8–99.5 % del tiempo en todos
los casos)

- **Por encima de lo que vio al entrenar, choca.** Lo sigue igual de cerca, pero no esquiva a tiempo.
- **Con objetivos de hasta 7 m/s al entrenar, a 5 m/s deja de chocar.** A 7 m/s baja de 33 % a 22 %, unos
  210 m de bosque por vuelo: 0.10 % de choques por metro, frente a 0.25 % de r25c navegando a 7 m/s en R18.
- **El ajuste en la GPU con `--v 1,7`** debería bajar más los choques a 7 m/s.

### En la GPU (R26 rápido)

Desde R26 con visibilidad, 1500 iteraciones a lote 64 con objetivos de 1–7 m/s y lr 2·10⁻⁴
(`r26/train_gpu1500_fast_v1-7.log`; evaluación en `r26/eval_gpu_fast.log`).

**En su simulador** (choques · error de distancia; 256 vuelos de 30 s por velocidad, y de 60 s a 1–4 m/s):

| política | objetivo a 3 m/s | 5 m/s | 7 m/s | 1–4 m/s, 60 s |
|---|---|---|---|---|
| R26 (entrenada a 1–4 m/s) | 0 % · 0.21 m | 12.5 % · 0.31 m | 32.8 % · 0.59 m | 1.2 % · 0.18 m |
| + 200 iteraciones de CPU a 1–7 m/s | 0 % · 0.19 m | 0 % · 0.25 m | 21.9 % · 0.49 m | — |
| **R26 rápido (GPU, 1–7 m/s)** | **0 % · 0.19 m** | **0 % · 0.25 m** | **5.1 % · 0.55 m** | **0 % · 0.18 m** |

(las dos primeras filas a 3–7 m/s, con 64 vuelos por velocidad; el objetivo está a la vista el 99.1–99.5 %
del tiempo)

- **A 7 m/s los choques bajan de 33 % a 5 %** en 30 s, unos 210 m de bosque: 0.024 % por metro.
- **No empeora a baja velocidad:** a 1–4 m/s, ningún choque en 256 vuelos de un minuto (antes 3), con el
  objetivo tapado el 0.6 % del tiempo y el mismo error de distancia (0.18 m).

**R20: globo a velocidad fija** (imagen renderizada, detector de color, tracker, SGM con `--edge`), bosques
de 1/25 m⁻², 30 semillas (3000–3029), 0–2 globos distractores. El globo acelera a 1 m/s² hasta su
velocidad y recorre 40 m:

| piloto | 3 m/s | 5 m/s | 7 m/s |
|---|---|---|---|
| `gbl`: sigue el rastro real (privilegiado) | 29/30 · 0 choques | 26/30 · 3 choques | 18/30 · 11 choques |
| `direct`: línea recta al punto estimado | 25/30 · 5 choques | 24/30 · 6 choques | 22/30 · 7 choques |
| **R26 rápido** | **30/30 · 0 choques** | **29/30 · 0 choques** | **26/30 · 0 choques** |

| piloto | error de distancia a 3 / 5 / 7 m/s | bloqueo a 3 / 5 / 7 m/s |
|---|---|---|
| `gbl` | 0.11 / 0.20 / 0.27 m | 99 / 96 / 93 % |
| `direct` | 0.87 / 1.23 / 1.47 m | 81 / 77 / 78 % |
| R26 rápido | 0.86 / 1.23 / 1.60 m | 99 / 99 / 99 % |

- **Ningún choque en 90 vuelos.** La línea recta choca en 18 y el rastro privilegiado en 14 (11 de ellos a
  7 m/s).
- **Emparejados por semilla** (`r26/balloon_gpu_fast_v*.json`), en los 90 vuelos:
  - frente a la línea recta, 15 frente a 1 ($p = 0.0005$);
  - frente al rastro privilegiado, 14 frente a 2 ($p = 0.004$); solo a 7 m/s, 9 frente a 1 ($p = 0.02$).
- **Guarda más distancia a los troncos:** holgura mínima mediana de 0.92–1.04 m, frente a 0.49–0.69 m del
  rastro y 0.54–0.63 m de la línea recta.
- **Los fallos de R26 son todos pérdidas** (1 a 5 m/s, 4 a 7 m/s): el globo se escapa.
- **El error de distancia crece con la velocidad** (0.86 → 1.60 m), mucho más que en su simulador
  (0.19 → 0.55 m), en el que la política conoce la posición y la velocidad exactas del objetivo. Como a
  1–4 m/s, lo que añade el error es la estimación del objetivo.

### Por qué se escapa el globo a 7 m/s (0.27.5–0.27.6)

Trazas cuadro a cuadro de las pérdidas a 7 m/s (semillas 3000 y 3010 con el modelo rápido de CPU, y 3000
con el de la GPU). Pasa lo mismo en todas:

1. **El globo gira en su punto intermedio justo cuando un tronco tapa la línea de visión.** Gira a
   50–70°/s: en la semilla 3000, de −24° a +40° en 1 s.
2. **El filtro de velocidad constante lo sigue prediciendo en línea recta.** En 1 s la predicción queda a
   5–8 m del globo.
3. **La cámara mira a la predicción**, así que el globo queda a 45–65° del eje, fuera del campo de visión
   (±45°). El dron pasa de largo, frena (sin ver el globo durante 1.5 s se queda quieto) y ya no lo vuelve
   a ver.
4. **Si lo ve, tampoco lo recupera:**
   - la puerta para volver a bloquearlo crece a 1.5 m/s mientras lo ha perdido, y el globo se aleja a 7;
   - cuando lo recupera, la primera medida queda a 8 m de la predicción y la puerta de innovación del
     filtro la descarta.
5. **Además, el filtro va retrasado.** Con el globo acelerando, estima 5.0 m/s cuando va a 6.5: el dron se
   queda 1.5–2.5 m más lejos de lo pedido, con más troncos en la línea de visión.

Con la profundidad real en lugar del SGM (`dp_gt`) la semilla 3000 falla igual: no es la percepción de los
troncos.

**Cinco cambios en la misión de seguimiento** (`r20 sim --fast-target`; por separado `--q-acc`,
`--yaw-ext`, `--gate-v`, `--relock-free`, `--turn-pred`):

| cambio | antes | `--fast-target` |
|---|---|---|
| ruido de proceso del filtro (menos retraso) | 2 m/s² | 5 m/s² |
| a dónde mira la cámara con el globo perdido | la predicción del filtro | la última estimación, movida como mucho 0.5 s |
| cuánto crece la puerta para volver a bloquearlo | 1.5 m/s | la velocidad del globo + 1 m/s |
| la primera medida tras volver a bloquearlo | pasa por la puerta de innovación | se acepta |
| la predicción con el globo perdido (0.27.6) | en línea recta | sigue girando al ritmo de los últimos 0.6 s (como mucho 90°/s y 90° en total) |

Ninguno toca la política, y valen igual para la línea recta (`direct`). El último es lo que YOPOv2-Tracker
echa en falta: según el artículo, sus fallos son sobre todo oclusiones, porque no predice la trayectoria del
objetivo.

**Predicción del giro, sin vuelo:** en las tres pérdidas trazadas, a 1–1.5 s de perder el globo, la línea
recta queda a 4.9–9.0 m de él, y la predicción con giro a 2.0–4.5 m.

**Con el modelo de la GPU**, R20 a 7 m/s, 30 semillas (3000–3029), corrido en otra máquina (sin cambios, a
1/25 m⁻², da los mismos fallos, semilla por semilla, que en la de la GPU):

| bosque | sin cambios | los cuatro primeros | **`--fast-target` (los cinco)** |
|---|---|---|---|
| 1/25 m⁻² | 26/30 · 0 choques · 4 pérdidas · 1.60 m | 26/30 · 1 · 3 · 1.31 m | **29/30 · 0 · 1 · 1.31 m** |
| 1/16 m⁻² (el de YOPOv2-Tracker) | 26/30 · 0 · 4 · 1.68 m | 29/30 · 0 · 1 · 1.40 m | **28/30 · 0 · 2 · 1.40 m** |

(éxitos · choques · pérdidas · error de distancia; `r26/r20_gpu_fast_target.log`)

- **Con `--fast-target`, 57/60 frente a 52/60, sin ningún choque.**
- **Emparejados, 5 frente a 0** ($p = 0.06$): todas las semillas que fallan con `--fast-target` también
  fallan sin él.
- **El error de distancia baja de 1.60–1.68 a 1.31–1.40 m.**

**Con el modelo rápido de CPU** (solo los cuatro primeros cambios; sirvió para elegirlos),
30 semillas (`r26/r20_cpu_fast_target.log`):

| bosque, globo | sin cambios | los cuatro primeros | emparejados: solo con / solo sin |
|---|---|---|---|
| 1/25 m⁻², 7 m/s | 26/30 · 2 choques · 2 pérdidas · 1.60 m | **28/30** · 1 · 1 · **1.39 m** | 3 / 1 |
| 1/16 m⁻², 7 m/s | 24/30 · 1 · 5 · 1.60 m | **27/30** · 2 · 1 · **1.38 m** | 4 / 1 |
| 1/16 m⁻², 5 m/s | 27/30 · 0 · 3 · 1.29 m | **30/30** · 0 · 0 · **1.08 m** | 3 / 0 |

- **Las pérdidas bajan de 10 a 2** en los 90 vuelos, sin más choques (3 y 3).
- **Emparejados, 10 frente a 2** ($p = 0.04$).

### R20 en Flightmare (0.27.10)

El seguimiento solo estaba probado en nuestro simulador. `nanouav run r22 follow` lo corre en Flightmare,
con Unity como YOPOv2-Tracker:

- **El mundo** es la arena de R22: la escena vacía del almacén, escaneada una vez, con un bosque de Poisson de
  cajas de madera de 0.3–0.7 m de ancho y 6 m de alto (por defecto 1/16 m⁻², la densidad de YOPOv2).
- **El globo se pinta en las imágenes de Unity.** El reproductor público no puede crear objetos nuevos, así
  que en las dos cámaras del par estéreo cada píxel cuyo rayo toca un globo antes que la profundidad de Unity
  toma su color sombreado y esa profundidad (`flightmare/follow.py`, `paint_spheres`). Las cajas lo tapan
  donde están delante, el detector de color lo ve y el SGM lo empareja como cualquier superficie.
- **Todo lo demás es R20:** el mismo camino del globo (40 m, punto intermedio al azar, globos distractores),
  la misión (detector, tracker, bloqueo, filtro, `--fast-target`), los pilotos y la regla de éxito.
- **`r22 follow-video`** graba un vuelo con la vista de percepción de R22 (0.27.12): la cámara del dron con
  el globo pintado y la trayectoria elegida, su SGM y su mapa local; el globo bloqueado va en un recuadro
  verde.

Probado contra el reproductor falso de los tests (nuestro renderer detrás del mismo protocolo): el piloto
privilegiado mantiene el globo bloqueado el 99 % del tiempo. Los resultados en Unity están pendientes.

### Comparación con YOPOv2-Tracker

Su prueba de seguimiento en simulación (sección IV-B y Fig. 10 del artículo) y la nuestra (R20) se parecen
mucho, pero no son la misma:

| | YOPOv2-Tracker | R26 en R20 |
|---|---|---|
| bosque | Poisson, 1/16 árboles/m² | Poisson, 1/16 árboles/m² (y 1/25) |
| objetivo | va con Fast-Planner (con el mapa real) hacia una meta a 40 m que cambia al azar a mitad de camino, sin reaccionar al dron | un globo; camino suave de 40 m con un punto intermedio al azar a ±6 m; 0–2 globos distractores |
| velocidades | 3, 4, 5, 6 y 7 m/s | 3, 5 y 7 m/s |
| vuelos | 10 por velocidad | 30 por velocidad |
| éxito | seguirlo hasta la meta sin chocar | lo mismo, más: bloqueado en el globo correcto ≥ 80 % del tiempo y nunca a más de 16 m |
| percepción | cámara RGB-D; la misma red detecta el objetivo y planea | estéreo con SGM, detector de color y tracker; la red solo vuela |
| odometría | la del simulador | la del simulador |

**Éxitos en el bosque de 1/16 árboles/m²:**

| velocidad | YOPOv2-Tracker (Fig. 10B, leída de la gráfica) | R26 rápido | **R26 rápido con `--fast-target`** |
|---|---|---|---|
| 3 m/s | 10/10 | 30/30 | **30/30** |
| 4 m/s | ~9/10 | — | **30/30** |
| 5 m/s | ~9/10 | 26/30 | **30/30** |
| 6 m/s | ~8/10 | — | **29/30** |
| 7 m/s | ~8/10 | 26/30 | **28/30** |
| 3–7 m/s | 44/50 (88 %) | 82/90 (91 %, sin 4 y 6) | **147/150 (98 %)** |
| choques | 1 de 50 («planning failed», 2 %) | 1 de 90 | **0 de 150** |

(R26 rápido: `r26/r20_gpu_d16.log` a 3 y 5 m/s, `r26/r20_gpu_fast_target.log` a 7 m/s; con `--fast-target`:
`r26/r20_gpu_ft.log`)

- **La lectura de la Fig. 10B es aproximada** (barras en 3D, ±1 vuelo). Cuadra con su resumen de la
  Fig. 10C: 88 % de éxitos, 10 % de objetivos perdidos y 2 % de choques o paradas en sus 50 vuelos.
- **Con `--fast-target`, R26 sigue al globo en 147 de 150 vuelos (98 %) sin un solo choque**, a las mismas
  cinco velocidades, frente al 88 % de YOPOv2-Tracker: $p = 0.008$ (Fisher).
- **Por velocidad, con sus 10 vuelos, ninguna diferencia es significativa por sí sola:** a 6 m/s, 29/30
  frente a ~8/10 ($p = 0.15$); a 7 m/s, 28/30 frente a ~8/10 ($p = 0.26$).
- **`--fast-target` también cuenta aquí:** sin él, 82/90 a 3, 5 y 7 m/s, con 1 choque; con él, 88/90.
  Emparejados, 6 frente a 0 ($p = 0.03$).
- **Nuestra condición de éxito es más estricta**, y R26 vuela tres veces más vuelos por velocidad.
- **Las tareas son casi iguales; los simuladores no.** Ninguno de los dos objetivos reacciona al dron:
  el suyo va a una meta que cambia a mitad de camino, el nuestro pasa por un punto intermedio al azar. Pero
  su simulador es Unity, su objetivo es otro (no un globo) y su detector, otro. Es una comparación
  indicativa, no un enfrentamiento directo.
- **Fallan por lo mismo:** según el artículo, YOPOv2-Tracker pierde el objetivo sobre todo por oclusiones
  porque no predice su trayectoria. Es lo que ataca `--turn-pred`; los 3 fallos de R26 son pérdidas.

**Todos los pilotos con `--fast-target`** (30 semillas; éxitos · choques):

| bosque | piloto | 3 m/s | 5 m/s | 7 m/s |
|---|---|---|---|---|
| 1/16 m⁻² | `gbl` (rastro privilegiado) | 29 · 1 | 25 · 5 | 16 · 14 |
| | `direct` (línea recta) | 14 · 16 | 19 · 9 | 19 · 7 |
| | **R26 rápido** | **30 · 0** | **30 · 0** | **28 · 0** |
| 1/25 m⁻² | `gbl` | 30 · 0 | 27 · 3 | 21 · 9 |
| | `direct` | 22 · 8 | 21 · 9 | 25 · 4 |
| | **R26 rápido** | **30 · 0** | **30 · 0** | **29 · 0** |

- **A 4 y 6 m/s en el bosque de 1/16 m⁻²:** `gbl` 27 · 3 y 16 · 14; `direct` 18 · 12 y 19 · 10; **R26 30 · 0
  y 29 · 0.**
- **R26: 236 de 240 sin un choque** (177 de 180 en la tabla, más 4 y 6 m/s). Los otros dos pilotos chocan
  124 veces en sus 480 vuelos.
- **En el bosque de 1/25 m⁻², `--fast-target` sube a R26 de 85/90 a 89/90** (30, 29, 26 → 30, 30, 29), y el
  error de distancia baja de 0.86 / 1.23 / 1.60 a 0.80 / 1.07 / 1.33 m.
- **A la línea recta no le sirve tanto:** de 71 a 68 éxitos en el bosque de 1/25 m⁻², con más choques a
  3 y 5 m/s (de 5 y 6 a 8 y 9); mejora a 7 m/s (de 22 a 25). El filtro más rápido es más ruidoso, y la línea
  recta no esquiva.
- **El rastro privilegiado sube de 73 a 78:** ya no pierde el globo (de 3 pérdidas a 0), pero sigue
  chocando a 7 m/s (9 de 30). No ve los troncos: choca al seguir su rastro a esa velocidad.

**Tiempo de decisión** (`nanouav run r25 bench`, un solo hilo de un Xeon a 2.8 GHz, sin GPU):

| | YOPOv2-Tracker | R26 |
|---|---|---|
| red | 2 ms (TensorRT, RTX 3060) | **0.08 ms** (TFLite float32, 391 k parámetros) |
| total | 3 ms en un i7-9700 con RTX 3060; 8.2 ms en una Orin NX | **1.1 ms**: 1.0 ms de preprocesar la profundidad + la red |

- **TFLite da las mismas salidas que Keras** (diferencia máxima de 4.5·10⁻⁷).
- **En las evaluaciones la red corre en Keras sin compilar:** 17–19 ms aquí y unos 48 ms en tu máquina con 8
  procesos a la vez. No es el tiempo de la red, es la sobrecarga de Keras.
- **No lo he medido en una Orin NX.** La estéreo (SGM) no está incluida, como tampoco lo está la cámara RGB-D
  en el tiempo de YOPOv2.

**Falta para un enfrentamiento directo:** el seguimiento en un simulador ajeno (R20 en Flightmare, Unity
como el suyo; preparado en 0.27.10) y medir el tiempo de decisión en una Orin NX.
