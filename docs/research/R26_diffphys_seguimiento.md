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

## Siguiente: entrenar en la GPU

1. R25 con el vuelo previo (r25c).
2. R26 desde r25c, con visibilidad (`--w-vis 1`) y sin ella (`--w-vis 0`), para medir lo que aporta.
3. R20 en bosques de 1/25 m⁻² con los dos, frente a `gbl` y `direct`.
