# R23 · Lo que ve el estudiante y lo que se le pide: dirección, referencia y etiquetas

*2026-10-11 · nanouav 0.21.0 · sobre R18 y R22. Todo en CPU (2 núcleos), sin GPU. Scripts: [`r23/`](r23/README.md).*

## Pregunta

El estudiante de R18 se queda en 22–23/40 en el bosque de prueba. El ciego (la recta, sin percepción) logra
16/40, el experto 40/40, y el paper reporta 100 % hasta 5 m/s. Casi todos sus choques son «todas chocan»:
ya 0.35–1 s antes, ninguna de sus tres hipótesis está libre. Su R-WTA de validación por entornos se estanca
en ~44, mientras la de entrenamiento sigue bajando a ~9. En Flightmare, tras dos rondas, **deriva hacia un
lado** a baja velocidad.

¿Le falta capacidad o datos al estudiante, o le estamos pidiendo algo que **no puede saber**?

## Resumen

1. **Las etiquetas de R18 no son función de lo que ve el estudiante.**
   - El experto sigue $\tau_\text{gbl}$, una ruta global planeada desde la salida con todo el mapa. La
     dirección $\omega$ del estudiante apunta a la recta.
   - Cambiar solo esa referencia cambia las etiquetas en R-WTA ~45: el valor en que se estanca la
     validación del estudiante (~44).
   - Los obstáculos que no ve casi no importan (0.2–0.6).
2. **El estudiante aprende a no fiarse de $\omega$, y en vuelo deriva.** Misma red pequeña, mismos 3 723
   estados, solo cambian etiquetas y $\omega$:
   - Con etiquetas que van a otro lado que $\omega$, **0/40** llegadas, a 3.7–5.5 m de la recta en
     promedio, con vuelos que se pierden a 9–11 m (percentil 90).
   - Con etiquetas que van a donde apunta $\omega$, 19/60 y a 0.2–0.4 m (Fisher $p = 2 \cdot 10^{-5}$; deriva
     $p < 10^{-12}$).
3. **Imitar a un experto privilegiado con una entrada de dirección exige cuatro condiciones.** Las
   correcciones obvias rompen cada una alguna de ellas:
   - (i) etiquetas aprendibles;
   - (ii) la misma $\omega$ en entrenamiento y en prueba;
   - (iii) un experto competente donde vaya el estudiante;
   - (iv) una $\omega$ que corrija a corto plazo.
4. **Una referencia que cumple las cuatro:** la ruta de costo mínimo a la meta **replaneada desde la
   posición actual**, con una penalización por alejarse de la recta, y $\omega$ hacia la recta 1 s adelante
   (como en R18). Sus etiquetas:
   - están libres de choques en el 98.9 %, también lejos de la recta;
   - forman con $\omega$ un ángulo mediano de 4.5° (R18: 12.8°; percentil 90: 19.6° contra 47°);
   - se aprenden mejor (validación 14.1 contra 22.6).

   En vuelo es la variante que más veces pasa donde la recta está bloqueada (3/12), pero a esta escala
   ningún estudiante esquiva mucho mejor que el ciego. Eso solo lo dirá el DAgger completo en la GPU.

## Idea

El estudiante aprende a imitar etiquetas $a^*$ a partir de su observación $o$ (profundidad, velocidad,
orientación, dirección $\omega$). Lo mejor que puede hacer cualquier red es reproducir la distribución
condicional $p(a^* \mid o)$; con R-WTA y $M$ hipótesis, sus salidas óptimas son los centroides de esa
distribución. Si $a^*$ depende de algo que no está en $o$, la distribución condicional se ensancha y **ningún
estudiante, por grande que sea o por muchos datos que tenga, baja de ese piso**. Y si la parte de $a^*$ que
debería explicar $\omega$ no la explica, la red aprende a darle poco peso a $\omega$.

En R18 hay dos cosas privilegiadas en las etiquetas: el mapa completo y la referencia $\tau_\text{gbl}$. La
dirección $\omega$ que recibe el estudiante, en cambio, apunta a la **línea recta** de salida a meta
(`GoalMission.frame`, en entrenamiento y en prueba). El paper define $\omega$ como el vector hacia «el punto
de referencia 1 s adelante, respecto del estado de referencia más cercano», y su experto se condiciona en
$\tau_\text{gbl}$; no dice explícitamente qué referencia usa $\omega$ en entrenamiento.

## 1 · Cuánto de las etiquetas no determina la observación

476 estados visitados por $\tau_\text{gbl}$ en 10 entornos de entrenamiento (a 3 y 7 m/s). En cada uno se
etiqueta con el experto M-H de R18 cambiando una sola cosa, y se mide el R-WTA entre conjuntos de etiquetas
(misma fórmula y escala que la pérdida del estudiante):

| se cambia… | R-WTA mediana (3 m/s) | (7 m/s) | RMS por punto |
|---|---|---|---|
| la semilla del muestreo M-H (ruido del experto) | 1.96 | 3.72 | 0.25–0.36 m |
| el mapa: solo los obstáculos visibles (campo de visión, ≤ 10 m, sin ocultos) | **0.22** | **0.61** | 0.10–0.14 m |
| la referencia: la recta en vez de $\tau_\text{gbl}$ (mapa completo) | **44.6** | **45.6** | ~2 m |
| ambas (lo que el estudiante puede saber) | 45.5 | 45.6 | ~2 m |

- El experto es **casi determinista** (piso de muestreo ~2) y **no usa lo que el estudiante no ve**.
- Lo que determina las etiquetas es **la referencia**.

La discrepancia crece con lo que $\tau_\text{gbl}$ aleja al dron de la recta (1.3 m en mediana, 4.5 m en el
percentil 90):

| distancia del dron a la recta | estados | R-WTA(etiquetas con $\tau_\text{gbl}$, etiquetas con la recta), mediana |
|---|---|---|
| < 0.5 m | 98 | 4.8 |
| 0.5–1.5 m | 190 | 32.9 |
| 1.5–3 m | 99 | 90.3 |
| > 3 m | 89 | 440.8 |

## 2 · Cuatro condiciones

| condición | si falla… |
|---|---|
| **(i) etiquetas aprendibles**: $a^*$ es (casi) función de $o$, incluida $\omega$ | piso irreducible; la red aprende a no fiarse de $\omega$ y nada la devuelve a la recta |
| **(ii) la misma $\omega$ en entrenamiento y en prueba** | $\omega$ es parte de la observación: cambiar cómo se genera es un cambio de distribución que DAgger no corrige (DAgger corrige los estados visitados, no el proceso que produce $\omega$) |
| **(iii) un experto competente donde vaya el estudiante** | se imitan etiquetas que chocan |
| **(iv) una $\omega$ que corrija a corto plazo** | una dirección a un punto lejano casi no cambia con el error lateral ni con la altura: el dron vaga y baja |

En prueba solo existen la salida, la meta y el estado del dron, así que (ii) y (iv) dejan una sola $\omega$
razonable: la de R18, la recta 1 s adelante. Lo que hay que cambiar es la referencia de las etiquetas.

| variante | etiquetas (referencia del experto) | $\omega$ entren. → prueba | (i) | (ii) | (iii) | (iv) |
|---|---|---|---|---|---|---|
| **A** = R18 | $\tau_\text{gbl}$ (planeada desde la salida) | recta → recta | ✗ | ✓ | ✓ | ✓ |
| B | $\tau_\text{gbl}$ | $\tau_\text{gbl}$ → recta | ✓ | ✗ | ✓ | ✓ |
| C | recta | recta → recta | ✓ | ✓ | ✗ | ✓ |
| D | replaneada desde aquí, a la meta | meta → meta | ✓ | ✓ | ✓ | ✗ |
| E | replaneada, a la meta | recta → recta | ~ | ✓ | ✓ | ✓ |
| F | $\tau_\text{gbl}$ | meta → meta | ✗ | ✓ | ✓ | ✗ |
| **G** | **replaneada desde aquí, pegada a la recta** | recta → recta | ✓ | ✓ | ✓ | ✓ |

**La referencia replaneada (`expert.CostToGo`).**
- **Cómo se calcula.** Un Dijkstra hacia atrás desde la meta da el costo hasta la meta $V(x)$ desde cualquier
  punto. Usa la misma rejilla y los mismos costos que $\tau_\text{gbl}$: celdas de 0.2 m, margen $r_q$ + 0.15 m y
  penalización por cercanía. G añade $w \, d^2$ por metro, con $d$ la distancia de la celda a la recta
  ($w$ = `W_LINE` = 0.5).
- **Qué da.** En cada estado, la referencia es el descenso de $V$ desde la posición actual, recortado y
  suavizado como $\tau_\text{gbl}$: **la mejor ruta a la meta desde aquí**. Con la penalización, vuelve a la
  recta y la sigue, y la deja solo donde hay obstáculos.
- **Por qué cumple las condiciones.** Es libre de choques desde cualquier punto (iii). No depende de una ruta
  elegida en la salida, solo de la posición, de la recta (que $\omega$ da) y de los obstáculos, sobre todo los
  cercanos (i).
- **Costo.** Un Dijkstra por entorno (~1 s) y ~70 ms por referencia.

## 3 · Prueba a escala reducida

**Datos.**
- 60 entornos de entrenamiento: bosques de Poisson como en R18, semilla base 4242.
- Dos pilotos: $\tau_\text{gbl}$ (estados sobre la ruta) y ciego (la recta hasta chocar: estados fuera de la
  ruta y cerca de obstáculos).
- $v \sim U(3, 8)$ m/s, una muestra cada 3 cuadros: **3 723 estados** con profundidad SGM.
- En cada estado, cuatro conjuntos de etiquetas del experto M-H de R18 (mapa completo), con referencia
  $\tau_\text{gbl}$, recta, replaneada a la meta o replaneada pegada a la recta.
- Validación: 12 entornos completos (804 estados).

**Estudiante.**
- Una CNN pequeña sobre profundidad de 60 × 80 (4 convoluciones + densas), con el estado de 15 números de
  R18 y $M = 3$ hipótesis con su costo.
- Las pérdidas de R18: $\lambda_1 = 10$ para R-WTA, $\lambda_2 = 0.1$ para el costo.
- 40 épocas, Adam con decaimiento coseno desde $10^{-3}$. Sin DAgger.
- Misma semilla, mismos estados y misma partición para todas las variantes: solo cambian las etiquetas y
  $\omega$.

### Las etiquetas

Para la mejor de las tres etiquetas de cada conjunto se mide:
- si está libre: proyectada al polinomio que se vuela, mantiene ≥ 0.2 m de los obstáculos durante 1 s;
- su ángulo con $\omega$ (la recta 1 s adelante), en mediana / percentil 90.

| etiquetas | libre | libre a > 3 m de la recta | ángulo con $\omega$ | ángulo a > 3 m |
|---|---|---|---|---|
| $\tau_\text{gbl}$ (A, B, F) | 0.986 | 1.000 | 12.8° / 47.0° | 49.0° / 77.4° |
| recta (C) | 0.918 | **0.500** | 0.0° / 5.4° | 2.2° / 9.4° |
| replaneada a la meta (D, E) | 0.994 | 0.990 | 8.5° / 34.9° | 29.6° / 63.0° |
| **replaneada pegada a la recta (G)** | **0.989** | **0.949** | **4.5° / 19.6°** | **15.1° / 33.8°** |

### Aprendizaje y seguridad en lazo abierto

«Elegida libre»: la hipótesis elegida como en vuelo (`fit_plan`, `choose`) está libre, con la $\omega$ de
prueba de cada variante.

| variante | R-WTA entren. | R-WTA valid. | elegida libre | < 0.5 m de la recta | 1.5–3 m | > 3 m |
|---|---|---|---|---|---|---|
| A = R18 | 2.6 | 22.6 | 0.943 | 0.949 | 0.969 | 0.867 |
| B | 1.8 | **6.4** | 0.813 | 0.925 | 0.615 | 0.357 |
| C | 3.1 | 13.4 | 0.786 | 0.915 | 0.615 | 0.265 |
| D | 1.9 | 16.4 | 0.920 | 0.947 | 0.923 | 0.724 |
| E | 1.9 | 17.4 | 0.924 | 0.947 | 0.892 | 0.765 |
| F | 2.6 | 21.4 | 0.939 | 0.939 | 0.969 | 0.888 |
| **G** | 1.9 | 14.1 | 0.854 | 0.923 | 0.754 | 0.531 |

### En vuelo

Bosques de prueba de R18 (semillas 1000–1009), 3 y 5 m/s, con la recta como referencia (lo único que existe
en prueba); cada variante con su $\omega$ de prueba. En esas 20 parejas el ciego llega 8 veces: la recta está
libre en 4 de los 10 bosques.

| variante | llega | choca | se agota el tiempo | distancia media a la recta (media / p90) | llega donde la recta está bloqueada | donde está libre |
|---|---|---|---|---|---|---|
| ciego | 8/20 | 12 | 0 | 0 | 0/12 | 8/8 |
| **A** = R18 | **0/20** | 16 | 4 | **3.65 / 8.82 m** | 0/12 | 0/8 |
| B | 7/20 | 13 | 0 | 0.20 / 0.33 m | 1/12 | 6/8 |
| C | 5/20 | 15 | 0 | 0.31 / 0.57 m | 2/12 | 3/8 |
| D | 2/20 | 18 | 0 | 1.95 / 3.43 m | 1/12 | 1/8 |
| E | 0/20 | 18 | 2 | 1.60 / 2.55 m | 0/12 | 0/8 |
| F | 0/20 | 15 | 5 | 5.54 / 11.30 m | 0/12 | 0/8 |
| **G** | **7/20** | 13 | 0 | 0.37 / 0.58 m | **3/12** | 4/8 |

## 4 · Lectura

- **(i) explica la deriva.**
  - A y F tienen etiquetas que no siguen $\omega$: 0/40 llegadas, a 3.7 y 5.5 m de la recta en promedio, y 9
    de sus 40 vuelos se pierden sin llegar ni chocar.
  - B, C y G tienen etiquetas que sí siguen $\omega$: 19/60, a 0.2–0.4 m.
  - Es la diferencia más robusta de la prueba (Fisher $p = 2 \cdot 10^{-5}$ en llegadas; Mann-Whitney
    $p < 10^{-12}$ en deriva). Es el mismo síntoma de Flightmare, aunque allí había además otra causa posible: la focal de las imágenes era 0.75 veces la nuestra ([R22, 0.21.1](R22_flightmare.md)).
  - El lazo abierto no lo ve: los planes de A casi siempre están libres (0.94), pero esa medida no dice si
    avanzan hacia la meta.
- **(ii), (iii) y (iv) también se ven.**
  - B aprende mejor que nadie (6.4) porque en entrenamiento $\omega$ apunta a donde van sus etiquetas. Pero
    aprende a seguir $\omega$, y en prueba la recta pasa por los árboles: en vuelo es un seguidor de la recta,
    6/8 donde está libre y 1/12 donde no.
  - C imita a un experto que, lejos de la recta, no encuentra trayectorias libres para volver a ella: la
    mitad de esas etiquetas chocan. Por esto el paper usa $\tau_\text{gbl}$.
  - D tiene etiquetas buenas y consistentes, pero su $\omega$ (la meta, a decenas de metros) apenas cambia
    cuando el dron se desvía: vaga a 2 m de la recta y pierde altura. En un vuelo pasó de 1.5 a 0.2 m y
    acabó contra el suelo.
- **G cumple las cuatro y es la mejor a igualdad de todo lo demás:**
  - etiquetas casi tan seguras como $\tau_\text{gbl}$ y alineadas con $\omega$;
  - mejor aprendizaje que A;
  - sigue la recta como B, pero la deja donde hay obstáculos: es la que más llega donde la recta está
    bloqueada (3/12).
- **Lo que esta prueba no puede decir.**
  - Ningún estudiante pequeño, sin DAgger, esquiva mucho mejor que el ciego (G 7/20, ciego 8/20).
  - Lejos de la recta, los planes de G chocan la mitad de las veces aunque sus etiquetas no: son estados poco
    frecuentes en estos datos, justo los que DAgger agrega.
  - 3/12 contra 1/12 no es significativo ($p = 0.59$).
  - Cuánto rinde G con la red completa y 850 entornos de DAgger solo lo dirá la GPU.

**Qué cambia respecto de R18.** El techo de R18 no venía de la percepción, ni de la capacidad de la red, ni
de los datos. Venía de pedirle al estudiante seguir una ruta que no puede ver, mientras se le daba una
dirección que no coincidía con ella. El paper resuelve (iii) con $\tau_\text{gbl}$; la referencia replaneada
resuelve (iii) sin romper (i).

## Cambio en el código

- `expert.CostToGo(env, w_line=)` y `ref_from(p, v)`: costo hasta la meta y ruta libre de choques desde
  cualquier punto. Si la meta cae en el margen de un obstáculo, la semilla crece hasta 5 m; la ruta sigue 3 m
  más allá de la meta. `expert.W_LINE = 0.5`.
- `Config.label_ref` y `GoalMission(label_ref=)`: `'gbl'` (por defecto, R18) o `'ctg'`.
  - Con `'ctg'`, el experto, la toma de control de DAgger y las etiquetas usan la referencia replaneada
    desde la posición actual.
  - La ruta nominal (de la que se mide ξ y hacia la que se orienta la salida) es la de costo mínimo planeada
    en la salida.
- `Config.omega_ref` y `GoalMission(omega_ref=)`:
  - `'line'` (por defecto, R18; la recomendada);
  - `'gbl'` ($\tau_\text{gbl}$ 1 s adelante, solo en entrenamiento: B);
  - `'goal'` (recto a la meta: D, F).
- `--label-ref {gbl,ctg}` y `--omega {line,gbl,goal}` en `r18 dagger` y `r22 dagger`.
  - Cada modelo guarda en `<modelo>.keras.json` con qué $\omega$ y qué etiquetas se entrenó, y `eval`,
    `video` y `diagnose` (R18 y R22) usan esa $\omega$ solos.
  - Los modelos sin ese archivo (R18–R22) siguen con la recta.
  - En `r22`, cada convención tiene su etiqueta de datos por defecto (`fm_<escena>_ctg`, …), para que no se
    mezclen.

## Qué correr en la GPU

```bash
# 1 · Simulador propio: la receta de R18 (6 épocas y batch 32, como sus rondas 8–29); solo cambian las
#     etiquetas. Comparar con student_train (22/40) y con el ciego (16/40). ~6–7 h en la TITAN X.
nanouav run r18 dagger --envs 850 --workers 7 --epochs 6 --batch 32 --label-ref ctg --out results/r23
nanouav run r18 eval --model results/r23/student_train.keras --out results/r23
nanouav run r18 eval --model results/r18/student_train.keras --out results/r18 --tag lat   # deriva de R18

# 2 · Flightmare: lo pendiente de R22 con las etiquetas nuevas
nanouav run r22 dagger --scene warehouse --init results/r18/student_spatial_in.keras --best --val-by-env \
    --envs 60 --label-ref ctg
nanouav run r22 eval --arena --scene warehouse --pilots student \
    --model results/r22/student_fm_warehouse_ctg.keras --v 3,5,7 --n 10
```

**Qué mirar.**
- El éxito de cada ronda de DAgger, en `results/r23/dagger_train.json`, contra la tabla de rondas de R18:
  19/30 en las rondas 24–25.
- El éxito en `r18 eval` contra 22/40 y contra el ciego (16/40).
- La distancia media a la recta: la columna `lat_m` de `r18 eval`, nueva en 0.21.0 (`lateral_m` en el JSON).
  La segunda orden de `eval` la mide también para el estudiante de R18.

Un resultado a favor:
- más llegadas donde la recta está bloqueada;
- menos deriva;
- validación por entornos bien por debajo de ~44.

**El arranque desde el modelo de R18.** En Flightmare, `--init` parte de un estudiante que aprendió a no
fiarse de $\omega$, y quizá tarde algunas rondas en desaprenderlo. Si dos rondas no mejoran, conviene empezar
de cero con `--head spatial` en lugar de `--init`.

## Límites

- **La etapa 2.**
  - Es una red pequeña (60 × 80, CPU) con 3 723 estados y sin DAgger. Dice qué diseño se comporta mejor a
    igualdad de todo lo demás, no cuánto rendirá el estudiante completo.
  - Con 10 vuelos por velocidad, los intervalos son anchos. Lo robusto es la deriva y la diferencia entre
    etiquetas consistentes y no consistentes.
- **La etapa 1** usa estados en la ruta de $\tau_\text{gbl}$. DAgger visita también estados alejados, donde
  la diferencia entre referencias es aún mayor.
- **La referencia replaneada sigue usando el mapa completo**, más allá de lo visible. La etapa 1 sugiere que
  eso pesa poco, pero no se midió para ella.
- **`W_LINE` = 0.5 no se ajustó.** Más alto, G se parece a C (vuelve a la recta aunque haya obstáculos); más
  bajo, se parece a E.
- **La lectura del paper sobre $\omega$ es una inferencia.** Las cuatro condiciones no dependen de ella.
- **La orientación de salida.** `GoalMission.start` orienta el dron en la salida hacia $\tau_\text{gbl}$
  también en prueba (sus primeros 0.5 m): un resto pequeño de información privilegiada, igual para todas las
  variantes.
