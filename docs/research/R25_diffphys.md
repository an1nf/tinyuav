# R25 — Aprender a volar por física diferenciable (Zhang et al. 2025)

*2026-10-15 · nanouav 0.26.0 · prueba de concepto en CPU; el entrenamiento real va en la GPU*

## La pregunta

DiffPhysDrone (Zhang, Hu, Song, Zou y Lin, *Nature Machine Intelligence* 2025; arXiv 2407.10648) es el estado
del arte por encima de Loquercio:
- 90 % de éxito frente al 60 % de Agile Autonomy en el montaje de este;
- 20 m/s en un bosque real;
- una red diminuta que corre en 3.7 ms en un ARM de 21 dólares.

No usa experto ni etiquetas: la política se entrena propagando el gradiente a través de un simulador
diferenciable de masa puntual. ¿Se reproduce en nuestro banco, y supera al piloto reactivo de R24 (37/40)?
Para el nano-UAV es lo más prometedor: una entrada de 16 × 12 de profundidad está a un paso del ToF 8 × 8.

## Implementación (`nanouav.agile.diffphys`, `nanouav run r25 train`)

**Simulador**, en TensorFlow:
- masa puntual con un paso de latencia, respuesta de primer orden de la aceleración (λ = 12 s⁻¹) y arrastre
  cuadrático y lineal;
- Δt = 1/15 s, 150 pasos (10 s), lotes de 64 bosques de Poisson como los de entrenamiento de R18 (espaciado
  4–7 m, troncos de 0.15–0.4 m);
- velocidad deseada entre 3 y 12 m/s;
- decaimiento temporal del gradiente $e^{-\alpha\,dt}$.

**Distancia:** analítica a los cilindros y al suelo, así que el gradiente es exacto.

**Imagen:** rayos de 32 × 24 contra los mismos cilindros, mínimo de cada 2 × 2 y luego inversa, en
16 × 12. No pasa gradiente por la imagen, como en el paper. Lleva aumento de datos tipo SGM: 10 % de huecos,
que se leen como lejos, y 5 % de ruido.

**Pérdida**, con los pesos del paper (1, 2, 0.01 y 0.001):
- seguir la velocidad deseada, con la velocidad promediada en ~2 s;
- choque: barrera softplus sobre la holgura, más la velocidad de acercamiento × $\max(1-c, 0)^2$;
- aceleración y jerk.

**Red:** CNN con núcleos de 2, 3 y 3 y 32, 64 y 128 filtros, más el estado sumado, más una GRU de 192 y una
salida de aceleración; unos 390 000 parámetros. El estado son 9 números en el marco de rumbo: velocidad,
dirección del empuje y velocidad deseada.

**En vuelo** (`--pilots diffphys` en `r18 eval`), dentro del mismo lazo de R18:
- la entrada es su propio SGM, con el mínimo de cada bloque y los huecos como lejos;
- la GRU conserva su estado entre cuadros;
- la aceleración se convierte en un plan de un cuadro que sigue el controlador en cascada.

## Prueba de concepto (CPU, 700 iteraciones a lote 32: el 0.7 % de lo que usa el paper)

Bosque de prueba de R18, 10 vuelos por velocidad:

| versión | 3 m/s | 5 m/s | 7 m/s | 10 m/s | velocidad media alcanzada |
|---|---|---|---|---|---|
| sin la velocidad deseada como entrada | 9 | 8 | 7 | 6 | 4.1–5.2 m/s, pida lo que se pida |
| con la velocidad deseada (como el paper) | 10 | 9 | 9 | 1 | 2.6 · 4.0 · 4.4 · 4.4 m/s |

**Lo que dicen estos números:**
- **La cadena funciona de punta a punta.** Una política entrenada sin experto, sin etiquetas y sin ver nunca
  un SGM vuela con su propio SGM y casi no choca: 2 choques en 40 vuelos.
- **Todavía no es comparable a velocidad alta.** A 7 y 10 m/s no pasa de ~4.4 m/s y a 10 m/s se queda sin
  tiempo, no choca. Con 700 iteraciones aprendió a esquivar antes que a correr.

No se puede afirmar que supere a nada hasta entrenarla como el paper.

## Entrenamiento en la GPU (20 000 iteraciones a lote 64, TITAN X Pascal)

Es el 40 % de lo que usa el paper (50 000 iteraciones a lote 64). Promedios de cada tramo de 1000 iteraciones
(registro en `r25/train_gpu20k.log`):

| iteraciones | pérdida | seguimiento de velocidad | choque (pérdida) | episodios con choque |
|---|---|---|---|---|
| 0–1 000 | 1.74 | 1.60 | 0.020 | 2.7 % |
| 1 000–2 000 | 1.19 | 1.05 | 0.017 | 0.8 % |
| 4 000–5 000 | 1.01 | 0.88 | 0.017 | 0.7 % |
| 9 000–10 000 | 0.94 | 0.80 | 0.017 | 0.4 % |
| 14 000–15 000 | 0.87 | 0.73 | 0.016 | 0.4 % |
| 19 000–20 000 | 0.86 | 0.73 | 0.016 | 0.4 % |

**Lo que dice la curva:**
- **Converge hacia las 15 000 iteraciones.** Después la pérdida ya no baja.
- **Lo que mejora es la velocidad.** La pérdida de seguimiento pasa de 1.60 a 0.73, la de choque apenas se
  mueve y los episodios con choque bajan de 2.7 % a 0.4 %. Es justo lo que le faltaba a la prueba de
  concepto, que esquivaba pero no corría.
- **El 0.4 % es en sus propios bosques**, con profundidad ideal más ruido. La prueba de verdad es su propio SGM
  en el lazo de R18 y las imágenes de Unity.

## Evaluación (30 entornos por velocidad, registro en `r25/eval_r18_y_flightmare.log`)

**Bosque de R18**, emparejado con el reactivo de R24 (`--keep --tilt --edge`) sobre las mismas semillas:

| piloto | 3 m/s | 5 m/s | 7 m/s | 10 m/s | choques | velocidad media real (3 · 5 · 7 · 10) |
|---|---|---|---|---|---|---|
| DiffPhys | 27/30 | 27/30 | 26/30 | 23/30 | 2 de 120 | 2.6 · 3.6 · 4.2 · 4.9 m/s |
| reactivo (R24) | 9/10 | 10/10 | 9/10 | 24/30 | | |

Emparejado: 3 entornos solo los pasa DiffPhys y 4 solo el reactivo ($p = 1$). Empatan en llegadas.

**Almacén de Flightmare**, sin reentrenar: la red nunca vio paredes, estanterías ni troncos cuadrados.

| piloto | 3 m/s | 5 m/s | 7 m/s | 10 m/s | total | choques por segundo a 10 m/s |
|---|---|---|---|---|---|---|
| DiffPhys | 26/30 | 28/30 | 28/30 | **27/30** | 109/120 | 1.0 % |
| reactivo (R24) | 29/30 | 29/30 | 26/30 | 20/30 | 104/120 | 9.3 % |

Emparejado a 10 m/s: 8 entornos solo los pasa DiffPhys y 1 solo el reactivo ($p = 0.04$). En total, 13 frente
a 8 ($p = 0.38$).

**Lo que dicen estos números:**
- **Casi no choca.** 2 choques en 120 vuelos en R18 y 5 en 120 en el almacén. Sus fallos son casi todos
  vuelos que se quedan sin tiempo.
- **Generaliza.** Entrenada solo con cilindros y suelo ideales, vuela con su propio SGM y con las imágenes de
  Unity mejor que el reactivo a 10 m/s, que es justo donde fallaba todo lo anterior.
- **Pero no vuela a la velocidad pedida.** Esto limita todo lo anterior: en R18, a 10 m/s, va a 4.9 m/s de
  media. En el almacén tarda 6.4 s en los 40 m (~6.2 m/s), y a 7 m/s tarda 7.3 s, frente a 5.8 s del
  reactivo. Parte de la ventaja a 10 m/s es ir más despacio. Todavía no se puede decir que bata a nada a
  10 m/s *reales*.

## Por qué iba lenta (0.26.6)

Hay dos causas, una de entrenamiento y otra de percepción. La dinámica no es el problema.

**1. Aprendió a contar metros.** En su propio simulador, con profundidad ideal y la meta a 1 km
(`nanouav run r25 profile`), la política entrenada en la GPU frena después de ~30 m pida lo que se le pida:

| v_des, empezando a esa velocidad | s 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 5 m/s | 4.9 | 4.8 | 4.8 | 4.8 | 4.9 | 5.0 | 4.7 |
| 7 m/s | 6.8 | 6.7 | 6.8 | 6.8 | 5.7 | 3.1 | 1.2 |
| 10 m/s | 9.4 | 9.2 | 8.6 | 5.6 | 2.2 | 1.4 | 1.1 |

En el entrenamiento la meta estaba siempre a 40 m y la velocidad objetivo caía a cero al llegar
(min(v_des, distancia)). Pero la red solo recibe la dirección de la meta, no la distancia. La GRU aprendió
entonces a integrar su velocidad y a frenar hacia los 30 m.

**Arreglo:**
- **Meta a 20–100 m al azar**, en bosques de 115 m.
- **La velocidad objetivo no cae al llegar.** El episodio termina al cruzar el plano de la meta, que es
  como se decide el éxito en R18 y R22.
- **Una métrica nueva en el registro:** `v/v_des`, la velocidad hacia la meta sobre la pedida.

Con 400 iteraciones de ajuste en CPU (lote 16, desde el modelo de la GPU), mantiene 5 y 7 m/s los 10 s y
10 m/s unos 6 s, hasta ~60 m.

**2. Un píxel cercano del SGM frena a la red.** En R18, a 10 m/s, sobre las semillas 1000–1009:

| política | entrada | llegadas | choques | tiempo mediano | velocidad media |
|---|---|---|---|---|---|
| GPU (original) | SGM | 7/10 | 0 | 8.3 s | 4.9 m/s |
| GPU (original) | SGM `--edge` | 9/10 | 0 | 7.8 s | 5.2 m/s |
| ajustada | SGM | 8/10 | 0 | 6.8 s | 5.9 m/s |
| ajustada | SGM, cuantil 0.1 por bloque | 9/10 | 1 | 6.2 s | 6.5 m/s |
| ajustada | SGM, cuantil 0.25 | 9/10 | 0 | 5.7 s | 7.0 m/s |
| ajustada | SGM, cuantil 0.1 y `--edge` | 8/10 | 2 | 5.3 s | 7.5 m/s |
| **ajustada** | **SGM `--edge`** | **10/10** | **0** | **5.5 s** | **7.3 m/s** |
| ajustada | profundidad exacta (`--dp-gt`) | 10/10 | 0 | 4.9 s | 8.2 m/s |

- **Con la profundidad exacta vuela a 8.2 m/s de media** (contando el arranque). La masa puntual
  representa bien al cuadrotor con su controlador.
- **Con el SGM frena.** La entrada toma el más cercano de cada bloque de 20 × 20 píxeles. Los píxeles que
  el SGM deja sueltos en los bordes de los troncos parecen obstáculos a pocos metros. En el entrenamiento
  los huecos se leían como lejos y nunca hubo puntos cercanos falsos.
- **El filtro de bordes (`--edge`, el de R24) los quita** sin perder troncos: 10/10 y 7.3 m/s.
- **Tomar un cuantil del bloque también los quita, pero pierde troncos finos** y aparecen choques.

**Las cuatro velocidades**, política ajustada con `--edge`, semillas 1000–1009, emparejada con el reactivo
de R24:

| piloto | 3 | 5 | 7 | 10 m/s | total | velocidad media de los que llegan |
|---|---|---|---|---|---|---|
| DiffPhys ajustada, `--edge` | 9/10 | 10/10 | 10/10 | 10/10 | 39/40 | 2.9 · 4.4 · 5.7 · 7.3 m/s |
| reactivo (R24) | 9/10 | 10/10 | 9/10 | 9/10 | 37/40 | 2.9 · 4.9 · 6.8 · 9.7 m/s |

- **No choca nunca a 7 y 10 m/s.** Hay 3 entornos que solo pasa DiffPhys y 1 que solo pasa el reactivo
  ($p = 0.62$, son 10 por velocidad).
- **Sigue siendo ~25 % más lenta que el reactivo a 10 m/s.** El ajuste en la GPU debe cerrar parte; la otra
  parte es el SGM: con la profundidad exacta va a 8.2 m/s.

Los JSON están en `r25/eval_forest_diffphys_cpuft400*.json`; el registro del ajuste, en
`r25/train_cpuft400.log`.

## Ajuste en la GPU (`results/r25b`, 5000 iteraciones a lote 64 desde el modelo de la GPU, lr 3·10⁻⁴)

**Entrenamiento** (`r25/train_gpu_r25b_5k.log`):
- `v/v_des` se mantiene en 0.93–0.94 en todo el ajuste;
- los episodios con choque bajan de 2.5 % a 2.0 %. Antes eran 0.4 %, pero a menos de la mitad de la velocidad.

**Perfil en su propio simulador** (`r25 profile`, `r25/profile_r25b.txt`):
- mantiene 5, 7 y ~9.3 m/s durante los 10 s, unos 90 m a 10 m/s;
- a 12 m/s va a ~10.5 m/s.

**Bosque de R18, 30 semillas por velocidad, `--edge`**, emparejado con el reactivo de R24 y con la versión
anterior (`r25/eval_forest_diffphys_r25b_edge.json`, reconstruido del registro):

| piloto | 3 | 5 | 7 | 10 m/s | total | velocidad media a 3 · 5 · 7 · 10 m/s |
|---|---|---|---|---|---|---|
| **DiffPhys r25b, `--edge`** | **28/30** | **29/30** | **29/30** | **28/30** | **114/120** | 2.8 · 4.4 · 5.7 · 6.7 m/s |
| DiffPhys de la GPU, sin `--edge` | 27/30 | 27/30 | 26/30 | 23/30 | 103/120 | 2.6 · 3.6 · 4.2 · 4.9 m/s |
| reactivo (R24) | 9/10 | 10/10 | 9/10 | 24/30 | | 2.9 · 4.9 · 6.8 · 9.7 m/s |

**Emparejados:**
- frente al reactivo: 6 entornos que solo pasa DiffPhys y **ninguno** que solo pase el reactivo ($p = 0.03$);
  a 10 m/s, 4 frente a 0;
- frente a la versión anterior: 13 frente a 2 ($p = 0.01$).

**Fallos:** 3 choques y 3 vuelos sin tiempo en 120. La semilla 1015 falla a todas las velocidades.

**Lo que falta:** a 10 m/s sigue más lenta que el reactivo (6.0 s frente a 4.1 s de mediana en los 40 m).
En su simulador mantiene ~9.3 m/s; con el SGM, aun filtrado, frena. Con la profundidad exacta, la versión
de CPU iba a 8.2 m/s.

**Almacén de Flightmare, 30 arenas por velocidad, `--edge`** (`r25/eval_arena_warehouse_diffphys_r25b_edge.json`,
reconstruido del registro):

| piloto | 3 | 5 | 7 | 10 m/s | total | tiempo mediano a 3 · 5 · 7 · 10 m/s |
|---|---|---|---|---|---|---|
| DiffPhys r25b, `--edge` | 25/30 | 28/30 | 27/30 | **28/30** | 108/120 | 15.0 · 9.0 · 6.6 · **4.8 s** |
| DiffPhys de la GPU, sin `--edge` | 26/30 | 28/30 | 28/30 | 27/30 | 109/120 | 14.1 · 9.1 · 7.3 · 6.4 s |
| reactivo (R24) | 29/30 | 29/30 | 26/30 | 20/30 | 104/120 | 13.6 · 8.2 · 5.8 · – s |

- **A 10 m/s:** 28/30 frente a 20/30 del reactivo; emparejado, 9 entornos frente a 1 ($p = 0.02$). Tarda
  4.8 s en los 40 m, frente a 6.4 s de la versión anterior, con las mismas llegadas (6 frente a 7, $p = 1$).
- **A 3 m/s es la peor:** 5 vuelos se quedan sin tiempo, sin chocar. Se detiene frente a los troncos.
  3 m/s es el borde inferior de las velocidades de entrenamiento (3–12 m/s).

## Persistencia de dos cuadros (`--dp-persist`, 0.26.8)

Los píxeles falsos que sobreviven al filtro de bordes parpadean: aparecen en un cuadro y no en el siguiente.
Un tronco de verdad persiste. Con `--dp-persist`, un bloque de 16 × 12 solo cuenta como cercano si también lo
era en el cuadro anterior: se toma el más lejano de los dos. Un tronco real llega a la red con un cuadro de
retraso, 67 ms.

Política ajustada en CPU, semillas 1000–1009, `--edge` (`r25/eval_forest_diffphys_cpuft400_edge*.json`):

| entrada | 3 | 5 | 7 | 10 m/s | velocidad media a 3 · 5 · 7 · 10 m/s | tiempo mediano a 10 m/s |
|---|---|---|---|---|---|---|
| `--edge` | 9/10 | 10/10 | 10/10 | 10/10 | 2.9 · 4.4 · 5.7 · 7.3 m/s | 5.5 s |
| `--edge --dp-persist` | 9/10 | 10/10 | 10/10 | 10/10 | 3.0 · 4.5 · 6.1 · **7.9 m/s** | **5.1 s** |
| profundidad exacta | | | | 10/10 | 8.2 m/s | 4.9 s |
| reactivo (R24) | 9/10 | 10/10 | 9/10 | 9/10 | 2.9 · 4.9 · 6.8 · 9.7 m/s | 4.1 s |

Las llegadas son las mismas y la velocidad sube: cierra dos tercios de lo que separaba al SGM de la
profundidad exacta, sin reentrenar.

**Con el modelo de la GPU (r25b), 30 entornos por velocidad, empeora a 10 m/s**
(`r25/eval_*_r25b_edge_persist.json`):

| r25b | 3 | 5 | 7 | 10 m/s | choques a 10 m/s | tiempo mediano a 10 m/s |
|---|---|---|---|---|---|---|
| R18, `--edge` | 28 | 29 | 29 | **28** | 0 | 6.0 s |
| R18, `--edge --dp-persist` | 28 | 29 | 29 | 24 | 5 | 5.0 s |
| almacén, `--edge` | 25 | 28 | 27 | **28** | 2 | 4.8 s |
| almacén, `--edge --dp-persist` | 26 | 27 | 28 | 24 | 6 | 4.65 s |

- **Va más rápido pero choca más.** r25b vuela más rápido que la política de CPU, y el cuadro de retraso con
  que la persistencia entrega un tronco real ya pesa a 10 m/s. Emparejado contra `--edge`: 1 frente a 5 en
  R18 y 0 frente a 4 en el almacén.
- **En el almacén aparecen choques contra el suelo** (5 de los 11) que sin la persistencia no había.
- **Conclusión:** `--dp-persist` queda como opción y no se usa. La prueba de CPU (10 semillas) no lo
  mostraba.

## Por qué se detiene a 3 m/s (0.26.9)

**El vídeo del almacén** (arena 3, 3 m/s): pasa bien los troncos durante 6 s. Hacia los 7–9 s llega a
menos de 1 m de un tronco, que le llena la imagen. Después se inclina mucho (~40° de alabeo, la cámara mira
al techo) y choca.

**La traza en R18** (semilla 1015, 3 m/s, `r25/decisions_r25b_edge_forest_v3_s1015.json`; `r18 video` y
`r22 video` la guardan ahora junto al vídeo, para el piloto diffphys):

| t | lo que pasa |
|---|---|
| 4.0–4.6 s | frena delante de un tronco a 0.35–0.6 m, hasta pararse |
| 4.8–5.2 s | retrocede: v_x = −0.7 → −1.7 m/s |
| 4.8–6.0 s | el rumbo sigue a la velocidad (la regla de R18 y del entrenamiento): gira de −17° a −153° en 1.2 s, con alabeo de hasta 53°, órdenes de 10 m/s² y caída de 1.6 a 0.6 m |
| 6.2 s | choque |

En su propio simulador, con profundidad ideal, también se queda quieta frente a grupos de troncos: 6 % de
los vuelos a 3 m/s y 11 % a 2 m/s. Ahí no retrocede ni hace maniobras bruscas. La velocidad objetivo apunta
a la meta, que está detrás del tronco, así que rodearlo cuesta más que quedarse parada.

**Lo que probé (semilla 1015 y semillas 1000–1009):**

| cambio | resultado |
|---|---|
| `--yaw-fwd` solo al volar: al retroceder la cámara mantiene el rumbo | semilla 1015: ya no choca a 3 y 5 m/s, pero se queda sin tiempo |
| ajuste en CPU con esa regla y velocidades desde 2 m/s, 400 iteraciones (`r25 train --yaw-fwd --v 2,12`) | 3 m/s: 8/10 (2 sin tiempo, en semillas que r25b pasa); 10 m/s: 9/10 |
| `--dp-amax 6`: limitar la aceleración horizontal ordenada | la semilla 1015 choca a 3, 5 y 7 m/s |

Ninguno lo arregla. Las opciones quedan para diagnóstico; los valores por defecto del entrenamiento no cambian.

## La red se cansa a los 10 s (0.27.0)

**La causa de fondo de la parada a baja velocidad no era la pérdida.** `r25 stalls` mide en su propio
simulador las llegadas y los vuelos parados (≥ 1.5 s por debajo de 0.3·v_des) con el límite de tiempo de
R18 (40 m / v_des + 6 s), 256 bosques por velocidad. r25b:

| v_des | llega | parados | choca |
|---|---|---|---|
| 2 m/s | 75.4 % | 20.7 % | 0 |
| 3 m/s | 98.8 % | 2.3 % | 0 |
| 5 m/s | 99.6 % | 0.4 % | 0 |

Las trazas muestran que frena hacia los 10 s de vuelo, no ante un tronco. El perfil de 20 s lo confirma
(`r25 profile` con T = 300; velocidad mediana de cada segundo, desde v_des):

| v_des | s 2 | s 8 | s 10 | s 12 | s 14 | s 16 | s 20 |
|---|---|---|---|---|---|---|---|
| 2 m/s | 2.4 | 2.2 | 2.0 | 1.9 | 1.7 | 1.6 | 1.2 |
| 5 m/s | 4.8 | 4.9 | 4.6 | 4.6 | 3.8 | 2.0 | 0.8 |
| 8 m/s | 7.5 | 7.9 | 6.8 | 4.8 | 2.2 | 1.5 | 0.8 |

Los episodios de entrenamiento duran 150 pasos (10 s) y empiezan con la GRU en cero. Pasado ese tiempo su
estado entra en una región que nunca vio, y la red frena a cualquier velocidad. A 3 m/s los 40 m duran
14–15 s, y por eso los vuelos sin tiempo del almacén. Seguir a alguien durante minutos sería imposible.

**Arreglo (`r25 train --burn 0,30`):** antes de cada episodio, la red vuela entre 0 y 30 s (al azar por
lote) sin gradiente por otro bosque. Cuando pasa x = 40 m se la retrasa 35 m. El episodio con gradiente
arranca con su velocidad, sus actuadores, su GRU y su rumbo, en un bosque nuevo. Cuesta ~50 % más por
iteración, porque el vuelo previo solo va hacia adelante.

**Ajuste en CPU desde r25b** (600 iteraciones, lote 16, `r25/train_cpuburn600.log`):

| | 2 m/s llega · parados | 3 m/s llega · parados | 5 m/s llega · parados |
|---|---|---|---|
| r25b | 75.4 % · 20.7 % | 98.8 % · 2.3 % | 99.6 % · 0.4 % |
| **r25b + `--burn 0,30`** | **99.6 % · 0.4 %** | **100 % · 0.8 %** | **100 % · 0.4 %** |

Mantiene 2.2–2.4, 4.8–4.9 y 7.4–7.8 m/s durante los 20 s del perfil.

**En el bosque de R18, con `--edge`, 30 semillas** (`r25/eval_forest_diffphys_cpuburn600_edge_v3_v10.json`):

| | 3 m/s | 10 m/s | choques | tiempo mediano a 10 m/s |
|---|---|---|---|---|
| r25b | 28/30 (corrida del PC) · 29/30 (aquí) | 28/30 | 3 en 120 | 6.0 s |
| **r25b + `--burn`** | **29/30** (solo la 1015, sin tiempo) | **30/30** | **0** | **5.7 s** |

Ni siquiera la semilla 1015, que fallaba a todas las velocidades, falla a 10 m/s.

**Lo que no sirvió:** entrenar también a 1–12 m/s sin el vuelo previo empeora: a 2 m/s, 0 % de llegadas,
porque a esa velocidad los 40 m duran 20 s. La pérdida «a lo largo de la meta» (`--loss-v along`) no
llegó a probarse a fondo: la causa era otra.

Después vienen la variante de entrada 8 × 8 (ToF) y la reparación por QP.
