# R26 · DiffPhys siguiendo un objetivo que se mueve, con visibilidad

*2026-10-25 · nanouav 0.27.0 · sobre R25 (política recurrente entrenada por física diferenciable) y R20
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

**Frente a YOPOv2-Tracker:** ellos siguen a 3–7 m/s y aquí el objetivo va a 1–4 m/s, así que todavía no es
comparable. El siguiente paso es `r26 eval --v 3,7` y `r20 sim --speed` a esas velocidades, y una
persona (R19) en vez del globo.
