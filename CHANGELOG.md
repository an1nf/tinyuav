# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/); versiones según
[SemVer](https://semver.org/lang/es/). Cada entrada indica si cambia algún resultado numérico.

## [0.26.9] — 2026-10-17

Sin cambios en los resultados por defecto.

### Añadido
- `r18 video` y `r22 video` con `--pilot diffphys` guardan junto al vídeo un JSON con cada decisión: posición,
  velocidad, actitud, aceleración ordenada y el bloque más cercano de la entrada. `r18 video --pilot diffphys`
  no cargaba el modelo y ahora funciona.
- Diagnóstico (todo desactivado por defecto):
  - `--yaw-fwd` en `r18` y `r22`: el rumbo sigue a la velocidad solo hacia la meta;
  - `r25 train --yaw-fwd` entrena con esa regla;
  - `--dp-amax` limita la aceleración horizontal de DiffPhys.

### Resultados
- **`--dp-persist` con r25b empeora a 10 m/s:** 24/30 frente a 28/30, tanto en R18 como en el almacén, con más
  choques. No se usa.
- **A 3 m/s se detiene así:** frena delante de un tronco, retrocede, y el rumbo, que sigue a la velocidad,
  gira 136° en 1.2 s; se inclina y choca. Ni `--yaw-fwd`, ni un ajuste con esa regla, ni limitar la
  aceleración lo arreglan. Detalle en `docs/research/R25_diffphys.md`.

## [0.26.8] — 2026-10-16

Sin cambios en los resultados por defecto.

### Añadido
- `--dp-persist` (en `r18 eval` y `r22 eval`): DiffPhys solo ve un bloque como cercano si también lo era en
  el cuadro anterior. Quita los píxeles falsos del SGM que parpadean; un tronco real llega con 67 ms de
  retraso.

### Resultados
- **DiffPhys ajustado (CPU) con `--edge --dp-persist`, semillas 1000–1009:** las mismas llegadas que con
  `--edge` (39/40) y más rápido. A 10 m/s va a 7.9 m/s de media (5.1 s), frente a 7.3 m/s (5.5 s) sin la
  persistencia y 8.2 m/s (4.9 s) con la profundidad exacta.
- **r25b en el almacén con `--edge`:** 108/120.
  - A 10 m/s, 28/30 frente a 20/30 del reactivo (9 frente a 1, $p = 0.02$), en 4.8 s frente a 6.4 s de la
    versión anterior.
  - A 3 m/s, 5 vuelos se quedan sin tiempo.

## [0.26.7] — 2026-10-16

Sin cambios en el código: resultados.
- **R25 ajustado en la GPU** (`r25b`, 5000 iteraciones): mantiene v_des en su simulador.
- **En el bosque de R18 con `--edge`:** 114/120 (28, 29, 29 y 28 de 30). Antes eran 103/120.
- **Emparejado con el reactivo:** 6 entornos que solo pasa DiffPhys y ninguno que solo pase el reactivo
  ($p = 0.03$). A 10 m/s, 28/30 frente a 24/30.
- **A 10 m/s sigue más lenta:** 6.7 m/s de media frente a 9.7 del reactivo.
- **Registros:** `docs/research/r25/` (`train_gpu_r25b_5k.log`, `profile_r25b.txt`,
  `eval_forest_diffphys_r25b_edge.json`).

## [0.26.6] — 2026-10-16

Cambia el entrenamiento de R25. Los modelos ya entrenados vuelan igual.

### Corregido
- **R25 aprendía a contar metros.** La meta estaba siempre a 40 m y la velocidad objetivo caía a cero al
  llegar, así que la GRU frenaba hacia los 30 m pidiera lo que se pidiera: a 10 m/s, 4.9 m/s de media en R18.
  Ahora:
  - la meta está a 20–100 m al azar, en bosques de 115 m;
  - la velocidad objetivo no cae al llegar;
  - el episodio termina al cruzar el plano de la meta, como en R18 y R22.

### Añadido
- `nanouav run r25 profile --model …`: la velocidad hacia adelante en cada segundo, en su propio simulador
  y con la meta a 1 km. Sirve para ver si mantiene v_des.
- El registro de entrenamiento muestra `v/v_des`.
- `r18 eval --pilots diffphys`, para el diagnóstico:
  - `--dp-gt` da la profundidad exacta en lugar del SGM;
  - `--dp-q` toma un cuantil de cada bloque de 16 × 12 en lugar del mínimo;
  - `--edge` ahora también filtra la entrada de DiffPhys.

### Resultados
- Con la profundidad exacta, DiffPhys llega 10/10 a 10 m/s, a 8.2 m/s de media. Con el SGM frena por los
  píxeles sueltos de los bordes; `--edge` los quita.
- 400 iteraciones de ajuste en CPU, evaluado con `--edge` en las semillas 1000–1009:
  - llega 39/40 (9, 10, 10 y 10), frente a 37/40 del reactivo;
  - no choca a 7 ni a 10 m/s;
  - a 10 m/s va a 7.3 m/s de media, frente a 4.9 antes y 9.7 del reactivo.

## [0.26.5] — 2026-10-16

Sin cambios en el código: resultados.
- **R25 evaluado, 30 entornos por velocidad:**
  - en el bosque de R18, 103/120; empata con el reactivo emparejado (3 frente a 4).
  - en el almacén de Flightmare, 109/120, sin reentrenar; a 10 m/s, 27/30 frente a 20/30 del reactivo
    (8 frente a 1, $p = 0.04$).
  - Solo 2 y 5 choques en 120 vuelos.
- No vuela a la velocidad pedida: a 10 m/s, 4.9 m/s de media en R18 y ~6.2 m/s en el almacén. Registro en
  `docs/research/r25/eval_r18_y_flightmare.log`.

## [0.26.4] — 2026-10-16

Sin cambios en los resultados.

### Corregido
- `r18 eval` no mostraba nada hasta terminar todos los vuelos. Ahora imprime una línea por vuelo, con el avance
  y el tiempo transcurrido.
- `--pilots diffphys` en `r18 eval`: cada proceso cargaba el modelo en cada vuelo, y TensorFlow abría un
  hilo por núcleo en cada uno de los 8 procesos. Ahora el modelo se carga una vez por proceso y TensorFlow
  usa los hilos de `OMP_NUM_THREADS` (1 por proceso).

## [0.26.3] — 2026-10-15

Sin cambios en los resultados existentes.

### Añadido
- `r22 eval|video --pilots diffphys --model …/diffphys.keras`: la política de R25 en Flightmare, con la cámara
  a nivel, como se entrenó.
- `r18 compare` (como `r22 compare`). En los dos, un piloto contra otro se empareja por velocidad, y
  `a.json+b.json` une corridas sobre semillas o entornos distintos. En R18 el entorno es la semilla.
- `docs/research/r24/flightmare/eval_arena_warehouse_reactive_edge_v3-7.json`: el reactivo en el almacén a
  3–7 m/s (84/90), reconstruido del registro porque la corrida a 10 m/s sobrescribió el JSON (0.26.2). Solo
  trae llegada, choque y tiempo.

### Resultados
- R25 entrenado en la GPU (20 000 iteraciones, lote 64): la pérdida baja de 1.74 a 0.86 y deja de bajar hacia
  las 15 000. Lo que mejora es el seguimiento de la velocidad (1.60 → 0.73). Los episodios de entrenamiento con
  choque bajan de 2.7 % a 0.4 %. La evaluación en R18 y Flightmare queda pendiente.

## [0.26.2] — 2026-10-15

### Corregido
- `r22 eval` sobrescribía el JSON de resultados cuando se corría otra velocidad con la misma configuración: la
  corrida a 10 m/s del reactivo borró la de 3, 5 y 7 m/s. Ahora se agregan las velocidades nuevas; el registro
  de 3–7 m/s está en `docs/research/r24/flightmare/`.

### Resultados
- R24 en Flightmare a 10 m/s: el reactivo llega en 20/30, con 5 de los 10 fallos en el primer 1.6 s.

## [0.26.1] — 2026-10-15

Sin cambios en el código: resultados y documentación.
- **R24 en Flightmare (almacén, 90 vuelos):** el piloto `reactive --edge`, sin red y sin nada privilegiado,
  llega en 84/90 (29, 29 y 26 de 30 a 3, 5 y 7 m/s).
  - Estudiante `sgm --keep`: 69/90; emparejado, 17 frente a 2 ($p = 7 \cdot 10^{-4}$).
  - Oráculo: 59/90.
  - Regla del paper: 38/90.
- `--fan` no mejora al estudiante: 68/90.
- `docs/research/R24`, `R22`, `fundamentos.md`; registro de la corrida en `docs/research/r24/flightmare/`.

## [0.26.0] — 2026-10-15

Sin cambios en los resultados por defecto.

### Añadido
- **R25, física diferenciable** (Zhang et al. 2025): `nanouav.agile.diffphys`, `nanouav run r25 train`.
  - simulador de masa puntual en TensorFlow con distancia analítica y profundidad de 16 × 12 por rayos;
  - pérdida de velocidad, choque (barrera más velocidad de acercamiento), aceleración y jerk;
  - política CNN + GRU;
  - piloto `diffphys` en `r18 eval`, con su propio SGM.
- **Prueba de concepto en CPU** (700 iteraciones): 10, 9, 9 y 1 de 10 a 3, 5, 7 y 10 m/s, sin pasar de
  ~4.4 m/s. Falta entrenarla como el paper en la GPU (ver R25).

## [0.25.0] — 2026-10-15

Sin cambios en los resultados por defecto (todo lo nuevo va tras banderas).
- **R24, piloto `reactive`:** sin red y sin nada privilegiado. Bosque de prueba de R18, 10 vuelos por
  velocidad (3, 5, 7 y 10 m/s):
  - `--keep --tilt --edge`: 37/40;
  - `--keep --tilt --edge --vga` (la cámara del paper): 39/40;
  - nuestro estudiante (`sgm`): 28/40;
  - el paper reporta 100 % hasta 5 m/s y ~60 % a 10 m/s para su estudiante, en su simulador.

  A 10 m/s con 30 entornos: 24/30 (`--edge`), 25/30 con VGA y 25/30 cobrando la latencia de decidir
  (`--latency 0.08`).
- **Probado y apagado:**
  - `--carve`: baja mucho las falsas alarmas por decisión, pero en vuelo da 34/40 y 20/30;
  - `--w-clear`: 22/30 a 10 m/s;
  - `--pool min` y `--slow`: sin mejora medible.

### Añadido
- **Piloto `reactive`** (`Rollout.plan_reactive`).
  - 33 candidatas en abanico alrededor de ω (`fan_points`), juzgadas sobre su propio SGM del último segundo.
  - Entre las libres elige la más cercana a la recta (`fan_cost`); con `--keep`, conserva el plan en curso.
  - En `r18 pilots/eval/video` y `r22 eval/video`.
- **`--fan`** (estudiante con `sgm` u `oracle`): si ninguna hipótesis está libre y `--keep` no puede actuar,
  vuela la mejor candidata libre del abanico.
- **Opciones de percepción:**
  - `--edge`: descarta los píxeles en saltos de profundidad (`rollout.edge_filter`);
  - `--pool min`: el píxel más cercano de cada bloque;
  - `--carve`: evidencia de espacio libre en `PerceivedObstacles`.
- **Opciones de R18:**
  - `--tilt`: la cámara a 30° hacia arriba a ≥ 7 m/s, como en R22;
  - `--vga`: el par estéreo de 640 × 480;
  - `--latency`: s de la captura al plan;
  - `--slow`: el abanico a menor velocidad si nada está libre;
  - `--w-clear`: costo de holgura suave.
- `decide_ms` / `decide_ms_p95` en cada vuelo y la columna `dec_ms` en `eval`: lo que tarda decidir.

### Cambiado
- `PerceivedObstacles.clearance` usa un k-d tree (scipy) en vez de fuerza bruta, con `nearest()` para el
  gradiente. Las decisiones son las mismas: solo cambian valores de diagnóstico por encima de 1.5 m.

## [0.24.1] — 2026-10-15

Sin cambios en los resultados: solo documentación.
- `docs/fundamentos.md` al día:
  - la elección con su propio SGM (`sgm`), `--keep` y lo que no funcionó (`--lookahead`);
  - la comparación emparejada (McNemar), el ruido entre corridas y la tasa de choque por segundo y por metro;
  - la focal de Unity;
  - el estado actual de R22.

## [0.24.0] — 2026-10-15

**Cambia el valor por defecto en R22.** `r22 eval`, `video`, `diagnose` y `dagger` usan `--keep`
(`--no-keep` para ≤ 0.23). Con 90 vuelos en el almacén y el mismo estudiante:
- `sgm --keep`: 21/30, 24/30 y 24/30 a 3, 5 y 7 m/s, 69/90;
- `sgm`: 58/90 (emparejados, 21 frente a 10 entornos, $p = 0.07$; a 3 m/s, 12 frente a 3, $p = 0.035$);
- frente a las dos corridas con `--lookahead`: 22/6 y 21/6 ($p = 0.004$ y $0.006$).

La tasa de choque por segundo pasa de 6.0, 3.7 y 3.6 %/s a 2.3, 2.6 y 3.8 %/s: ya no es constante, crece con
la velocidad. R18 sigue con `snap`.

### Cambiado
- `flightmare.dagger.run(keep=)`: el estudiante vuela sus rondas con la regla con la que se evalúa.

## [0.23.0] — 2026-10-14

Sin cambios en los resultados por defecto.
- **R22, `--lookahead` 0.5 y 1.0:** 53/90 y 54/90 frente a 58/90 sin él, con el mismo modelo (emparejados,
  $p = 0.42$ y $0.58$). Sin efecto: queda en 0.
- **R22, choques por segundo:** con cualquier selección, la tasa de choque por segundo de vuelo es casi la
  misma a 3, 5 y 7 m/s (`sgm` 5.4, 5.0 y 3.4 %/s; `snap` 8.5, 9.3 y 8.6 %/s); por metro cae con la velocidad.
  3 m/s es lo peor porque el vuelo dura más, no porque sea más difícil.

### Añadido
- `--keep` (`eval`, `video` y `diagnose` de R18 y R22) para `sgm` y `oracle`. Cuando ninguna hipótesis
  nueva está libre y lo que queda del plan en curso (≥ 0.3 s) sí, el dron sigue volándolo
  (`rollout.keep_previous`, `remainder`). Los archivos llevan `_keep` en el nombre. En el vídeo, el plan
  conservado se dibuja en blanco.
- `diagnose`: en el último segundo antes de cada choque, cuántas decisiones no tenían ninguna hipótesis libre
  y en cuántas el plan en curso seguía libre (`diagnose.keep_stats`); se mide también sin `--keep`.
- `r22 eval` imprime la tasa de choque por segundo y por metro de vuelo.
- `r22 compare A.json B.json …`: aciertos y tasas de cada corrida, y cada una contra la primera emparejada por
  entorno (McNemar exacto; `agile.stats`).

## [0.22.2] — 2026-10-14

Sin cambios en los resultados: solo el vídeo de percepción.

### Cambiado
- Etiquetas que ya no se pisan:
  - sobre la cámara, en DejaVu con borde oscuro;
  - «0.4 s» y «1.0 s» fuera de la esquina superior derecha de cada marco del túnel, la lejana solo si cabe;
  - el índice de cada hipótesis solo si no choca con otra etiqueta;
  - en el mapa local, el índice solo para planes de más de 1.2 m y lejos de ω y de los demás índices.
- Los anillos del mapa local se rotulan a la derecha del campo de visión, fuera de la trayectoria y de las
  flechas.

## [0.22.1] — 2026-10-14

Sin cambios en los resultados por defecto.
- **R22, 15 rondas de DAgger volando con `sgm`:** 58/90 (con 10 rondas, 62/90): sin cambio.
- **R18 con `--select sgm`:** 28/40, frente a `snap` 22/40 y el ciego 16/40.

### Añadido
- `--lookahead L` (`eval`, `video`, `diagnose` de R18 y R22) para `sgm` y `oracle`: el plan solo cuenta como
  libre si su continuación a su velocidad final durante L s también lo es (`rollout.extension`,
  `with_lookahead`). Los archivos llevan `_la<L>` en el nombre.

## [0.22.0] — 2026-10-14

**Cambia la selección por defecto en R22.** `r22 eval`, `video`, `diagnose` y `dagger` usan `--select sgm`:
el plan más suave entre los que el propio SGM del dron ve libres. Con 90 vuelos por modo en el almacén:
- `sgm` 62/90;
- `snap`, la regla del paper, 38/90 (McNemar $p = 4 \cdot 10^{-5}$);
- `oracle` 59/90.

R18 sigue con `snap`.

### Cambiado
- `r22 dagger --select {sgm,snap}`: el estudiante vuela sus rondas con la regla con la que se evalúa
  (`flightmare.dagger.run(select=)`).
- Los obstáculos percibidos (`PerceivedObstacles`) se actualizan en cada cuadro de cámara, también cuando el
  experto vuela en DAgger.

## [0.21.10] — 2026-10-14

Solo documentación: el 2 × 2 de selección en R22.
- `oracle` 24/30 y `oracle_cost` 19/30 (holgura real).
- `sgm` 20/30 y `sgm_cost` 16/30 (su propio SGM).
- `snap` 17/30 (la regla del paper sobre las 3 hipótesis).

Regla y percepción suman unas 4 llegadas cada una. Ninguna diferencia es significativa con 10 vuelos por
velocidad.

## [0.21.9] — 2026-10-14

**Cambia `--select sgm`.** Ahora elige como el oráculo: el plan más suave (menor snap) entre los que su
propio SGM ve libres. El de 0.21.7–0.21.8, que usaba la regla del costo de la red entre los libres (16/30 en
el almacén), pasa a llamarse `sgm_cost`. No cambia ningún otro resultado.

**`r22 diagnose --select sgm_cost`** (lo que era `sgm`):
- de los planes que chocan, vio libres el 2 %; de los libres, vio bloqueados el 7 %;
- solo en el 0.1 % de las decisiones voló uno con choque creyéndolo libre;
- 13 de sus 14 choques son `cambio_de_plan`.

### Añadido
- `--select oracle_cost` (la holgura real con la regla del costo) y `sgm_cost`. Junto con `oracle` y `sgm`
  forman un 2 × 2 que separa percepción (holgura real o su SGM) y regla (menor snap o costo predicho).

## [0.21.8] — 2026-10-14

Sin cambios en los resultados. En el almacén, `--select sgm` da 16/30: igual que `snap` (17/30), lejos del
oráculo (24/30).

### Añadido
- Con `--select sgm`, cada decisión guarda en `trace` la holgura que vio el propio SGM (`clear_seen`).
  `diagnose.perception_errors` y `r22 diagnose` resumen contra la verdad cuántos planes que chocan vio
  libres, cuántos libres vio bloqueados y cuántas veces voló uno con choque creyéndolo libre.

## [0.21.7] — 2026-10-14

Sin cambios en los resultados por defecto.

**R22, elegir mejor entre las mismas hipótesis.** Con el estudiante de 300 entornos, `--select oracle` da
24/30 (`snap`: 17/30; a 7 m/s, 10/10), y `--select consistent` da 14/30. El cuello de botella es la cabeza
de costo, no las hipótesis.

### Añadido
- `--select sgm` (R18 y R22): la regla del oráculo con los obstáculos del propio SGM del dron. Son celdas de
  0.15 m vistas en ≥ 2 cuadros del último segundo, con su propia pose; no es privilegiado. Lo implementan
  `rollout.sgm_points`, `PerceivedObstacles` y `perceived_index`. En nuestro bosque, con el modelo pequeño:
  `snap` 4/10, `sgm` 5/10, `oracle` 6/10.

## [0.21.6] — 2026-10-13

Sin cambios en los resultados; retoques del vídeo de percepción.

### Cambiado
- **Mapa local.** Vuelve la nube de puntos del SGM propio del último segundo, coloreada por distancia como el
  panel de profundidad y desvaneciéndose con la edad. El margen del dron es ahora un contorno tenue
  alrededor de las celdas vistas en ≥ 3 cuadros.
- **Bajo la cámara, solo el costo de choque predicho.** Una barra por hipótesis en su color, con su número
  y su valor, y la elegida enmarcada; sin título, tiempo, velocidad ni β. El resultado del vuelo es un borde
  verde o rojo en los últimos cuadros.
- **Tipografía.** DejaVu Sans vía Pillow (acentos, «ω»), con OpenCV como respaldo. Una sola paleta oscura
  y separadores finos entre paneles.

## [0.21.5] — 2026-10-13

Sin cambios en los resultados; cambia el diseño de los vídeos de percepción.

### Cambiado
- **Panel de profundidad.**
  - Colores inferno en escala logarítmica de 0.5 a 10 m: cerca, brillante; lejos, morado oscuro, como en
    las figuras del paper. Antes era TURBO lineal.
  - Donde no hay SGM se ve la imagen de la cámara atenuada, en vez de gris plano.
  - Barra de color en la misma escala.
- **Mapa local.**
  - Fondo oscuro y campo de visión sombreado.
  - 1 s de SGM propio, en celdas de 0.15 m que cuentan solo si se vieron en ≥ 3 cuadros (el ruido del SGM no
    se repite).
  - El margen del radio del dron alrededor de los obstáculos.
  - Las hipótesis con su número, la elegida del ancho del dron.
  - Flechas de la dirección a la meta (ω) y de la velocidad.
- Las entradas de `trace` guardan ω (`omega`).

## [0.21.4] — 2026-10-13

Sin cambios en los resultados por defecto.

**R22 tras 300 entornos** (etiquetas de R18, focal corregida): `eval` da 3/10, 8/10 y 6/10. En las rondas,
el éxito sube hasta 21/30 en la 5 (ξ = 3) y luego baja a 11–14/30. `diagnose` reparte los 14 choques así:
- ninguno `todas_chocan`: siempre había una hipótesis libre;
- 8 `cambio_de_plan`;
- 6 `eligio_mal`.

El cuello de botella es elegir y sostener el plan, no la percepción ni las hipótesis.

### Añadido
- `--select oracle` en `eval` y `diagnose` (R18 y R22): elige con la holgura real de cada polinomio volado
  (privilegiado, solo diagnóstico, `rollout.oracle_index`). Con las hipótesis del propio estudiante, da el
  techo de una cabeza de costo y una selección perfectas.
- Los resultados de `r22 eval` y `r22 diagnose` llevan el modo de selección en el nombre si no es `snap`, para
  no pisar los de `snap`.

## [0.21.3] — 2026-10-13

Solo documentación. R22 con la focal corregida: 20/30 con las etiquetas de R18 (antes 9/30). Las etiquetas
replaneadas de R23 dan 13/30 a la misma escala: se aprenden mejor (validación 7.8 contra 10.0) pero no
vuelan mejor todavía (R23 §5, R22).

## [0.21.2] — 2026-10-13

Sin cambios en los resultados. Con la focal corregida, `r22 check` en el almacén da las cajas a ≤ 2 px de lo
esperado y el SGM con un 4.5–6.6 % de error en ellas. Pero solo un 3.5–5 % de la imagen tiene profundidad
SGM, contra un 34–46 % en nuestro simulador.

### Añadido
- `r22 check` separa lo que falta por alcance de lo que falta por emparejamiento: fracción de SGM válida
  donde Unity da menos de `max_range` (10 m), y en cada caja. Guarda `check_geometry.png` con, por
  inclinación, imagen izquierda | SGM | profundidad de Unity.

## [0.21.1] — 2026-10-13

**Corrige un error de geometría en Flightmare que invalida lo que el estudiante vio en R22** (imágenes,
SGM y las dos rondas de DAgger). No cambia nada de R18–R21, ni τ_gbl ni el mapa de la arena: el escaneo
retroproyecta con el mismo campo de visión que envía.

### Corregido
- El campo de visión enviado a Unity salía de `calib.json`, donde el probe antiguo había escrito 90°. Lo
  decidió sobre una nube de puntos plana: la escena no tenía colisionadores. Unity lo toma como vertical:
  focal de 240 px a 640 × 480, contra los 320 px de nuestra cámara. `r22 check` lo mostró con las cajas
  fuera del eje: las tres aparecen al 0.74 de su distancia al centro, y el SGM da distancias 1.33 veces
  mayores (error relativo 0.31–0.42).
- Ahora el campo de visión sale de la cámara (`unity_fov`: 73.74° vertical para 90° horizontal en 4:3). Un
  valor en `calib.json` solo cuenta si lo midió `r22 check` (`fov_from: check`); el del probe se ignora y se
  avisa. El probe ya no escribe `fov_deg` (ahora `fov_deg_probe`).

### Añadido
- `r22 check` mide la focal de las imágenes respecto de la nuestra (`focal_ratio`: ajuste de las caras de
  las cajas sobre la profundidad de Unity). Si difiere más de un 3 %, corrige `fov_deg` en `calib.json` y
  pide repetir la comprobación. Una caja cuenta como bien puesta solo si está a ≤ 8 px (VGA) de donde la
  pone nuestro modelo.

## [0.21.0] — 2026-10-11

Sin cambios en los resultados con las opciones por defecto (`omega_ref='line'`, `label_ref='gbl'`: el
comportamiento de R18–R22).

### Añadido
- R23 (`docs/research/R23_direccion_y_etiquetas.md`): las etiquetas de R18 siguen τ_gbl mientras la dirección ω
  del estudiante apunta a la recta; las etiquetas no son función de la observación (piso de R-WTA ~45, igual
  a la validación del estudiante, ~44) y el estudiante aprende a no fiarse de ω y deriva. Cuatro condiciones
  para imitar a un experto privilegiado y una referencia que las cumple; prueba a escala reducida en CPU.
- `expert.CostToGo`: costo hasta la meta (Dijkstra hacia atrás en la rejilla de τ_gbl, con penalización por
  alejarse de la recta, `W_LINE`) y, desde cualquier posición, la ruta libre de choques que baja por él.
- `Config.label_ref` / `GoalMission(label_ref=)`: `'ctg'` usa esa ruta, replaneada desde la posición actual,
  para las etiquetas y la toma de control de DAgger; ξ se mide desde la ruta de costo mínimo planeada en
  la salida.
- `Config.omega_ref` / `GoalMission(omega_ref=)`: `'line'` (R18), `'gbl'` (τ_gbl 1 s adelante, solo
  entrenamiento) o `'goal'` (recto a la meta).
- `--omega {line,gbl,goal}` y `--label-ref {gbl,ctg}` en `r18 dagger` y `r22 dagger`. Cada modelo nuevo
  guarda su convención en `<modelo>.keras.json` (`policy.write_meta`, `policy.omega_of`), y `eval`, `video`,
  `diagnose` (R18 y R22) la usan solos. En `r22`, cada convención tiene su etiqueta de datos por defecto.
- `r18 eval`: distancia media a la recta por episodio (`lateral_m`, columna `lat_m`).

## [0.20.3] — 2026-10-10

Sin cambios en el código ni en los resultados.

### Añadido
- `docs/fundamentos.md`: qué se ha construido, las razones matemáticas que lo sostienen (escala
  monocular y ToF, estéreo y SGM, bosques de Poisson, experto M-H, R-WTA, proyección polinómica y escala
  de tiempo, DAgger, IC de Wilson, convenciones de Flightmare, arena) y la infraestructura del proyecto.

## [0.20.2] — 2026-10-09

Sin cambios en los resultados. R22, primeras dos rondas de DAgger en Flightmare: τ_gbl 10/10 a 3, 5 y 7 m/s;
estudiante 2/10, 2/10 y 5/10, con deriva hacia la izquierda a baja velocidad.

### Añadido
- `r22 check`: geometría de las imágenes contra nuestro modelo de cámara, con cajas fuera del eje, nivelada
  y a 30° (espejo, inclinación, SGM).
- `r22 diagnose`: clasificación de choques en la arena (como `r18 diagnose`), deriva lateral, lado de las
  hipótesis elegidas y calidad del SGM por decisión (`Rollout._trace_stats`). Un PNG por choque, con
  troncos y escena estática.

## [0.20.1] — 2026-10-09

Sin cambios en los resultados de R18–R21 (los videos cambian de diseño).

### Corregido
- `FlightmareClient`: el reproductor vuelve a su menú tras 10 s sin mensajes (entre rondas de DAgger,
  episodios de τ_gbl en `eval`). Ahora manda un cuadro cada 2 s si está ocioso, y cada pose lleva la
  configuración para recuperarse si el reproductor se reinicia igual (`reloads`). El falso lo imita
  (`idle_reset_s`).
- Cámara inclinada 30° hacia arriba en los vuelos a ≥ 7 m/s, como en el paper, en entrenamiento y
  evaluación en la arena (`set_tilt`, `dagger.tilted`).
- `z_min` del experto relativo al suelo de la escena.

### Cambiado
- Videos (`r18 video`, `r22 video`): solo lo que percibe el dron. Cámara con la trayectoria elegida como
  túnel (sección del dron cada 0.2 s) coloreado por el costo predicho, SGM a resolución completa con escala,
  y mapa local del propio SGM del último 0.5 s. `--debug-map` vuelve a los paneles anteriores.
- La traza guarda el SGM a resolución completa y la imagen de la cámara (`Rollout._trace_images`);
  `FlightmareObserver` devuelve también el RGB.

## [0.20.0] — 2026-10-08

Sin cambios en los resultados de R18–R21 (el campo `z` de las muestras sigue siendo la altura sobre el
suelo, que en nuestro simulador está en 0).

### Añadido
- Entrenamiento en Flightmare como el paper:
  - `r22 arena`: escaneo de la escena vacía, suelo, rectángulo libre y comprobación de la caja sembrada;
  - `flightmare.arena`: `GridSDF`, `Arena` (escena estática ∪ troncos exactos, suelo aparte) y `arena_env`
    (bosques de Poisson de troncos);
  - `FlightmareClient(n_objects=)` y `place(boxes)`: objetos dinámicos de Unity (`rpg_box02`);
  - `r22 dagger` (`flightmare.dagger`): DAgger con imágenes de Unity, etiquetas en paralelo
    (`Rollout.defer`), reanudable;
  - `r22 eval --arena`, `r22 video --arena`.
- Par estéreo VGA (`--vga`, por defecto en arena, dagger y `--arena`): 640 × 480 y 96 disparidades, como
  la cámara del paper.
- `Field` pasa `ground` a los mundos que separan el suelo y usa su `local`. El reproductor falso dibuja
  los objetos dinámicos y tiene un almacén de prueba (`fake --world arena`).

## [0.19.4] — 2026-10-08

Sin cambios en los resultados de R18–R21. R18: cabeza espacial (ImageNet) 7/9/6/1 de 10 a 3/5/7/10 m/s = 23/40
(22/30 entre 3 y 7 m/s, el mejor resultado hasta ahora en ese rango; 1/10 a 10 m/s).

### Añadido
- `Rollout.record_cam`: con traza, los pilotos privilegiados también observan y graban sus cuadros. `r22 video
  --pilot gbl|expert` muestra la cámara de Flightmare, el SGM y el plan del piloto.
- Choques en r22: punto del mapa golpeado, su altura sobre el suelo y cuántas vistas del escaneo lo vieron
  (`eval`, `video`; cruz roja en el video).
- `scan --min-hits` (2): descarta los vóxeles aislados; `CloudWorld.hits` se guarda en el `.npz`.
- `r22 survey`: mosaico cenital de la escena (color y altura), para elegir la zona de escaneo.

## [0.19.3] — 2026-10-08

Sin cambios en los resultados de R18–R21.

### Corregido
- `find_lines` aceptaba rutas globales de la búsqueda de respaldo, a 2 cm del mapa: τ_gbl chocó en la única
  línea del bosque real. Ahora exige inflado completo, salida y meta libres y que τ_gbl la vuele
  (`lines --check-v`, `--max-terrain`), e informa los rechazos.

### Añadido
- `--unity <ruta>` / `$NANOUAV_FLIGHTMARE`: cada comando de r22 abre su propio reproductor de Unity y lo
  cierra al terminar (el reproductor no siempre se reconecta después de volver al menú).

## [0.19.2] — 2026-10-08

Sin cambios en los resultados de R18–R21.

### Añadido
- `r22 scan`: el mapa privilegiado a partir de las imágenes de profundidad de Unity en una malla de poses
  (`nanouav.flightmare.scan`). El bosque de 2022 casi no tiene colliders: la petición de nube solo devolvió
  una lámina bajo el río. Escribe el mismo `cloud_<escena>.npz` y comprueba el mapa con una vista nueva.
- `FlightmareClient(stereo=False)`: una sola cámara (para escanear). `CloudWorld(inflate=)`, que se guarda
  en el `.npz`.

### Cambiado
- `probe` recomienda `scan` cuando la nube sale plana o vacía.

## [0.19.1] — 2026-10-08

Sin cambios en los resultados de R18–R21. R22: corrige las convenciones de Unity a partir del código del
reproductor (flightmare_unity, `dev/agile_flight`) tras el primer probe real.

### Corregido
- La malla del dron tapaba las cámaras: `size` del vehículo 0.01 (como agile_autonomy), `--vehicle-size`.
- Profundidad de Unity = valor × plano lejano de la capa (100 m); antes se tomaba en metros.
- La petición de nube es absoluta (Unity no suma el `render_offset`): `FlightmareClient.cloud_points` pide
  en origen + desplazamiento y devuelve los puntos en el marco de la escena. El `.ply` está en orden ROS.
- `CloudWorld`: la distancia es la del centro ocupado más cercano (los vóxeles ya están inflados una
  resolución); el suelo sirve para terreno como lámina o como bloque.

### Cambiado
- `probe`: comprueba la base estéreo, la escala de la profundidad contra el SGM y la nube contra la
  profundidad de Unity. Escribe avisos en `calib.json`, un panel de 2 × 3 y `probe.npz`.
- El reproductor falso imita al real (profundidad / plano lejano, nube absoluta, ocupación por radio).

## [0.19.0] — 2026-10-07

Sin cambios en los resultados existentes.

### Añadido
- R22 · Flightmare como renderizador (`nanouav.flightmare`, `nanouav run r22`):
  - puente ZMQ en Python con el ejecutable de Unity (binarios 2020 y 2022);
  - par estéreo de `StereoRig`;
  - nube de puntos de la escena como mapa privilegiado (`CloudWorld`, que acepta `geometry.Field`);
  - líneas de vuelo encontradas en la nube;
  - `probe`, que comprueba las convenciones y escribe `calib.json`;
  - `eval`, `video` y un reproductor de Unity falso (`fake`) para probarlo todo sin Unity.
  Extra `flightmare` (pyzmq). Informe: `docs/research/R22_flightmare.md`.
- `Rollout.observer`: un renderizador externo en lugar del nuestro. `r18 video` dibuja la nube y la
  imagen del renderizador externo.

## [0.18.1] — 2026-10-05

Sin cambios en los resultados: los modelos guardados conservan su arquitectura.

### Añadido
- `policy.build_model(head='spatial')` (`r18 train|dagger --head spatial`): conserva la posición de las
  características de profundidad (convolución 1 × 1 + aplanado) en vez del promedio global, que
  pierde dónde está cada obstáculo. Por defecto sigue `gap`.
- R18: resultados de `--tscale arc --vcap 1.3` (16/40) y `--select consistent` (13/40) frente a 22/40.
  Ninguna ayuda.

## [0.18.0] — 2026-10-05

Sin cambios en los resultados existentes.

### Añadido
- R21 · el estudiante de R18 en Gazebo:
  - `gen_world.py --agile forest|gap` (el mundo de prueba de R18) y el modelo `agile_x3` (empuje/peso
    3.6, colisión de 0.3 m, par estéreo de `StereoRig`);
  - `gazebo/nodes/agile_node.py`, `gazebo/run_agile.sh` y `gz_docker.sh agile`;
  - `nanouav.agile.gz_core` (la lógica sin ROS, más un dron de velocidad con retraso para probarla
    sin Gazebo);
  - `r18 export` (Keras → TFLite, `policy.TFLitePolicy`, `load_policy`) y `r18 video --from-gz`.
  Informe: `docs/research/R21_gazebo_agile.md`.
- `rollout.fit_plan` y `rollout.choose`: la proyección y la elección de `Rollout` como funciones, para
  usarlas fuera del simulador.

## [0.17.2] — 2026-10-05

Sin cambios en los resultados: las opciones nuevas están apagadas por defecto.

### Añadido
- `Rollout.TSCALE` (`--tscale chord|arc`): β por la distancia recta (como antes) o por la longitud
  del camino predicho. Con la recta, una predicción curva se vuela más rápido que v_des: los choques
  del estudiante a 10 m/s fueron a 12–19 m/s.
- `Rollout.V_CAP` (`--vcap`): pico de velocidad de cada plan ≤ vcap · v_des.
- `Rollout.SELECT` (`--select snap|consistent`): mantener el lado de la elección anterior mientras su
  costo predicho esté dentro del 25 % del mejor.
- `r18 eval|diagnose|video` aceptan esas opciones, y los archivos de resultado llevan su sufijo.
- `diagnose`: la clase se decide 0.35–1 s antes del choque. Hay una clase nueva, `cambio_de_plan`,
  y una marca `sobrevelocidad`.

## [0.17.1] — 2026-10-05

Sin cambios en los resultados.

### Añadido
- `nanouav run r18 video`: vídeo de un episodio (vista cenital con las hipótesis, profundidad que ve
  la red, cámara de seguimiento), en H.264 si hay ffmpeg. `--pilot expert|gbl|blind` para comparar.
  `Rollout.trace` guarda también la actitud y, con `trace_depth`, la profundidad de entrada.
- `r18 diagnose`: comprueba si el obstáculo del choque estaba en el campo de visión (`fuera_vista`) y
  el ángulo al que estaba.
- R18: evaluación con la proyección corregida (22/40 y 16/40). La corrección no explica los choques a
  3 m/s. También se documentan qué cambio corresponde a cada clase de choque y la meta.

## [0.17.0] — 2026-10-05

### Corregido
- **Proyección de la predicción al polinomio que se vuela** (`Rollout.PROJ = 'scaled'`, ahora el valor
  por defecto). El polinomio se ajusta con v/β y a/β², así que tras el escalado de tiempo cada plan
  arranca en la velocidad actual. Antes la velocidad mandada saltaba a β·v en cada plan.
  - Para el estudiante, β ≈ 0.7 a 3 m/s: caídas del 30 % de la velocidad 15 veces por segundo.
  - τ_gbl y el experto no cambian (β ≈ 1; 6/6 a 3 y 10 m/s).
  - `--proj raw` reproduce lo anterior en `r18 pilots|eval|diagnose`.
  - Seguimiento (R19): con la cámara a 10 Hz el bloqueo no cambia. El test corto pasa de 5 a 10 Hz:
    a 5 Hz el bloqueo variaba con cambios mínimos de la trayectoria.

### Añadido
- `nanouav run r18 diagnose`: clasifica los choques del estudiante en `todas_chocan`, `eligio_mal`,
  `plan_justo` y `seguimiento`. Cuenta los cambios de lado y la β mediana, y guarda una vista cenital
  por choque. `Rollout.trace` guarda cada decisión.
- `r18 eval|pilots --tag`. Las trayectorias en `traj/` llevan el nombre del modelo, para no pisarse
  entre evaluaciones.
- R18: evaluación de `student_train` y `student_polish` en el PC de entrenamiento.

## [0.16.1] — 2026-10-05

Sin cambios en los resultados: las opciones nuevas están apagadas por defecto en `r18`. En `r20
dagger` están encendidas.

### Añadido
- `policy.train(..., lr_end=, keep_best=, val_by_env=)`. `lr_end` da un descenso coseno del lr;
  `keep_best` guarda la época con menor R-WTA de validación; `val_by_env` valida con entornos enteros.
  `load_shards` marca el entorno de cada muestra (`_env`).
- `nanouav run r18 train|dagger`: `--lr`, `--lr-end`, `--best` y `--val-by-env`. `r18 train`:
  `--data` (varios directorios) y `--name` (para no pisar `student_train.keras`).
- `r20 dagger`: lr 3e-4 → 1e-5, mejor época y validación por entornos por defecto.
- R18: resultados del DAgger completo de 850 entornos en el PC de entrenamiento
  (`docs/research/R18_agile_autonomy.md`).

## [0.16.0] — 2026-10-03

Sin cambios en los resultados existentes: las piezas compartidas con R18/R19 ganan opciones cuyos
valores por defecto reproducen el comportamiento anterior (los tests de R18, R19 y la regresión del
renderer pasan sin cambios).

### Añadido
- `nanouav run r20`: seguir un globo de color a través de un bosque, evitando los árboles, sobre la
  política de R18 (`docs/research/R20_seguimiento_globo.md`).
  - `follow.blob.BlobDetector`: detector de color (HSV + componentes conexas, une trozos cortados por
    un tronco); funciona igual en la imagen simulada y con una cámara real (`r20 video`, `r20 pick-color`).
  - `follow.balloon`: globos que flotan por rutas libres con altura variable, bosque en color con
    falsos positivos realistas, globos que cruzan (del mismo color con probabilidad 0.25).
  - `follow.balloon_mission.BalloonMission`: distancia por tamaño aparente, filtro 3D, punto de
    seguimiento a la altura del globo; DAgger `kind='balloon'`.
- Imagen en color: `Texture.color`, `render_view(..., chroma=True)`, `render_stereo/observe_full(..., rgb=True)`
  (la imagen gris es la misma); `Rollout` la pide si la misión tiene `needs_rgb`.
- Tracker: compensación del giro de la cámara (`Tracker.update(..., warp=)`, `estimate.rotation_warp`),
  `min_gate_px` y `drop()`. `TargetLock`: `gate_locked` (descarta una pista que salta en 3D), `w_dist` (al
  rebloquear prefiere la más cercana a la predicción) y `gate_growth`. `TargetFilter(dims=3)`.
- `FollowMission`: métodos por tipo de objetivo (`_detect`, `_pos_of`, `_goal_z`, …), `innov_gate` y
  métrica `switches`. `dagger.run(..., xi_start=)`.

## [0.15.0] — 2026-10-03

### Añadido
- `nanouav.follow` y `nanouav run r19`: detectar a una persona y seguirla evitando obstáculos.
  - Detector YOLO de personas (ultralytics; COCO o ajustado con VisDrone conservando la cabeza de COCO)
    y un detector simulado calibrable (`eval-det`).
  - Tracker estilo ByteTrack y bloqueo del objetivo con apariencia y compuerta 3D.
  - Estimación 3D con la profundidad estéreo y filtro del objetivo.
  - Personas que caminan en el simulador, misión de seguimiento para el lazo de R18 y DAgger de
    seguimiento (`kind='follow'`).
  - Conversión de VisDrone, entrenamiento y exportación del detector, y demo en vídeo.
  Informe: `docs/research/R19_seguimiento_personas.md`.
- `Rollout` con misiones (`GoalMission` = la tarea del paper, igual que antes; `FollowMission`) y
  `Config.label_every`; `stereo.observe_full` devuelve también la imagen;
  `global_plan(..., point_goal=True)`.

### Cambiado (R18)
- Cada polinomio arranca con la aceleración de referencia del plan en curso, no la medida (que cerraba
  un lazo a través de la vibración y hacía oscilar a baja velocidad). Pilotos de R18 revalidados con 6
  semillas por casilla: experto y τ_gbl 36/36 en el bosque (3, 7 y 10 m/s) y 36/36 en el hueco (3, 5 y
  7 m/s), igual que antes. `Rollout.A0_MODE` permite comparar.

## [0.14.0] — 2026-10-02

Sin cambios en los resultados existentes: el renderer gana dos parámetros con valores por defecto
idénticos (el test de regresión pasa sin regenerar la referencia).

### Añadido
- `nanouav.agile` y `nanouav run r18`: recreación de Loquercio et al. 2021 (*Learning high-speed flight
  in the wild*). Incluye:
  - cámara estéreo simulada con profundidad SGM (`cv2.StereoSGBM`), con filtro de textura;
  - entornos de entrenamiento (bosques y formas convexas por proceso de Poisson) y de prueba (bosque
    de 1/25 m⁻², hueco estrecho);
  - plataforma del paper (r_q = 0.2 m, c_max = 35.3 m/s², J = 0.007 kg·m²);
  - experto privilegiado (trayectoria global + Metropolis-Hastings sobre B-splines, Eq. 1–3);
  - estudiante (MobileNet-V3 + MLP, M = 3, pérdida R-WTA) y ejecución por polinomios de orden 5 con
    escala de tiempo y selección por coste;
  - DAgger reanudable y en paralelo, ajuste fino para el hueco, y evaluación contra los pilotos
    experto, τ_gbl y ciego.

  Informe y tabla de fidelidad en `docs/research/R18_agile_autonomy.md`.
- `render_view(..., size=(H, W), d_max=...)` y `camera_rays(hfov, size)`: resolución y alcance
  arbitrarios (por defecto 96×96 y `D_MAX`, como antes).

## [0.13.4] — 2026-10-02

Sin cambios en los resultados de simulación.

### Corregido
- Gazebo: el contenedor corre como root y dejaba a nombre de root los mundos generados, las texturas y
  `results/gz_runs`; después `nanouav gazebo gen-world` fallaba en el host con *Permission denied*.
  `gz_docker.sh` pasa `HOST_UID`/`HOST_GID` y `run_sim.sh` devuelve esos archivos al usuario al
  generar el mundo y al salir. `wait` sobre Gazebo ya no aborta la limpieza si Gazebo sale con error.

### Documentación
- R17 §5: validación con las redes completas (suite `speed_thin`, n = 10). `FLY_KEEP_NEAR` da casi
  toda la mejora con poco coste a 1 m/s, mientras que `MAP_DECAY_FOV` cuesta un 17–59 % de tiempo con
  la CNN. Con el M_ToF a 1 m/s + `FLY_KEEP_NEAR`, el mundo de Gazebo da 10/10 en 30.8 s.
- `speed_thin.yaml`: 12 trabajos `*-fly` nuevos (con `--resume` solo corren esos).

## [0.13.3] — 2026-10-01

Sin cambios en los resultados publicados: las opciones nuevas del planificador están apagadas por
defecto (el test de regresión pasa sin regenerar la referencia).

### Corregido
- Gazebo: reiniciar el mundo desde la interfaz (`/clock` vuelve atrás) empieza un episodio nuevo con
  un planificador nuevo y se guarda por separado (`*_epN.json`). Antes, el nodo seguía con el
  planificador anterior (contador de estancamiento y margen relajado heredados) y el RTF salía
  negativo. Tres saltos atrás en 10 s siguen indicando dos simulaciones en el mismo `/clock` y el
  monitor pasa a tiempo de pared.
- Gazebo con `V_CRUISE` > 0.8: el nodo usaba los límites de referencia normales en lugar de los
  `fast` del simulador propio. Ahora ambos usan `nanouav.planning.presets`.
- `r12 --v` distinto de 0.5 añade `_v<velocidad>` al nombre del resultado (los barridos se
  sobrescribían).
- `batch_eval.sh`: el log incluye la variante y la etiqueta (antes M y M_ToF se pisaban).

### Añadido
- `PLANNER_CFG` / parámetro `planner_cfg` en Gazebo, con la misma sintaxis que `r12 --cfg`.
- Opciones del MH para obstáculos finos (R17), apagadas por defecto: `FLY_KEEP_NEAR`,
  `MAP_DECAY_FOV` y `MAP_HIT_WINS`. Con profundidad perfecta en `bosque` pasan de 22/32 a 40/40
  episodios sin choque; con la CNN local (n = 4) no hay mejora clara. Suite `speed_thin.yaml` para
  decidir con las redes completas. Informe `docs/research/R17_velocidad_y_troncos_finos.md`.
- `MHPlanner.dbg` registra qué límite de velocidad manda en cada cuadro (`v_stop`, `v_clr`,
  `k_fov`, `v_plan`).
- `nanouav summarize --gazebo` separa filas por velocidad y `planner_cfg`; `nanouav summarize`
  muestra el tiempo mediano de los éxitos.

## [0.13.2] — 2026-10-01

### Documentación
- Informe `docs/research/R16_entrenamiento_y_lazo_cerrado.md`: resultados de la campaña del
  2026-10-01 (entrenamiento completo, M vs M_ToF, estructuras finas, cables, pasos estrechos).

### Corregido
- `r10`: `hit_ball` era `False` en choques con el balón (comparación estricta con muestras tomadas en
  otro instante); en ese mundo el único obstáculo es el balón, así que ahora `hit_ball` = colisión.
  Cambia la columna "hits" de los resúmenes, no el éxito ni las colisiones.

### Añadido
- Escenario `cables` (cables horizontales a la altura de vuelo, `gap`, `d`, `posts`) y suite
  `cables.yaml`. Con GT y d = 2 cm el nano choca en parte de los episodios (límite de resolución de
  96×96); con d = 5 cm la GT pasa.
- `r10 --scen cross` guarda `bearing_deg` e `in_fov` por velocidad: el balón llega por un rumbo
  constante atan(v_obs / v_dron); con la cámara de 60° solo es visible si v_obs < 0.58·v_dron. Por
  encima, el resultado depende de la suerte en la sincronización, no de la percepción (ver R16).

## [0.13.1] — 2026-10-01

Sin cambios en los resultados de simulación (el test de regresión pasa sin regenerar la referencia).

### Añadido
- `--thin P` en `nanouav run train` y `r03`: postes de 1–5 cm y cables en una fracción P de las
  escenas de entrenamiento y validación (R14), con caché de datos propia. `--suffix` da nombre propio
  a los modelos resultantes (`model_M_ToF_thin.tflite`) sin tocar los originales.
- `nanouav run train --eval-only`: evalúa modelos ya guardados y actualiza `metrics.json`.
- Escenario `bosque` con `r_min` / `r_max` (bosque de troncos finos), suite `thin_structures.yaml`,
  y trabajos del nano en `narrow_passages.yaml`.
- `r12` añade `_fuse2` al nombre del resultado cuando `NANOUAV_FUSE2=1`.

### Corregido
- La evaluación tras entrenar fallaba en la TITAN X: el modelo exportado sin compilar activaba XLA.
  XLA queda desactivado también en `train_cnn` y `r11`.
- Con un modelo .tflite pero sin runtime de TFLite, la CNN pasaba en silencio a profundidad *fake*;
  ahora lanza `TFLiteRuntimeMissing`. Pedir una variante que no está instalada ya no cae en silencio
  a `model.tflite` (= M).
- `metrics.json` se vuelve a leer antes de escribir, para no pisar evaluaciones hechas en paralelo.
- `docs/gpu.md`: `LD_LIBRARY_PATH` y `ptxas` cubren todas las carpetas `nvidia` de pip.

## [0.13.0] — 2026-10-01

Reestructuración completa en un paquete instalable. **Sin cambios numéricos**: las salidas de
escenarios, render, ToF, fusión, flujo óptico, planificadores y lazo cerrado coinciden bit a bit con
`uav_v3` (`NANOUAV_EXACT=1 pytest tests/test_regression.py`).

### Añadido
- Paquete `nanouav` (layout `src/`) con `pyproject.toml` y extras `tflite`, `train`, `dev`.
- CLI única `nanouav`: `list`, `run`, `suite`, `summarize`, `models`, `gazebo gen-world`, `doctor`.
- Registro de experimentos (`nanouav.experiments`) con ids estables (`e0c` … `r15`, `train`, …).
- Suites YAML con matrices, ejecución paralela, `--resume`, `--dry-run`, `--only`, estado en `jobs.json`
  (`smoke`, `baseline`, `scenarios`, `narrow_passages`, `perception`).
- Rutas de artefactos configurables (`NANOUAV_HOME`, `NANOUAV_MODELS`, `NANOUAV_DATA`, `NANOUAV_RESULTS`)
  y banderas con prefijo `NANOUAV_` (se aceptan los nombres antiguos).
- `nanouav models import|list|check` para traer las redes entrenadas sin versionarlas.
- Tests (pytest): regresión numérica, escenarios, fusión, planificadores en una puerta, CLI, runner,
  nodos de Gazebo con stubs de ROS 2, CNN real (si hay modelo). CI en GitHub Actions (3.10–3.12).
- Documentación en español: README, arquitectura, experimentos, artefactos, GPU, Gazebo, migración,
  problemas conocidos.

### Cambiado
- Código, comentarios, docstrings y mensajes traducidos al inglés (verificado por AST: solo cambió texto).
- Etiquetas de `r08 --levels` y claves de `r4_speed_limit.json` / `r7_camera_tilt.json` en inglés
  (las etiquetas antiguas de `r08` se siguen aceptando).
- Logs de `gazebo/batch_eval.sh` en `results/gz_logs/`; episodios de Gazebo en `results/gz_runs/`.
- `MHPlanner.set_segment` reinicia también margen, relajación, progreso e inspección (afecta solo a
  misiones de varios tramos, p. ej. Gazebo con `LOOP=true`).

### Retirado
- Scripts sueltos en la raíz, `run_jobs.sh` + `jobs_*.txt` (→ suites), `pack.sh`, y los atajos
  obsoletos `e2a_train_variants.py`, `generate_model.py`, `train_e0d.py`.

## [0.12.x] — 2026-09 (estructura plana `uav_v2` / `uav_v3`)
- R12: planificador MH (Loquercio 2021) y escenarios compartidos con Gazebo (v13).
- R13: pasos estrechos y anti-estancamiento.
- R14: métricas por clase y `fuse_tof2`. R15: falsas detecciones dinámicas vs ego-movimiento.
- Detalle en `docs/research/`.
