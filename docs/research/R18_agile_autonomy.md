# R18 · Recreación de Loquercio et al. 2021 en nanouav (`nanouav.agile`)

*2026-10-02 · nanouav 0.14.0 · Loquercio, Kaufmann, Ranftl, Müller, Koltun, Scaramuzza, "Learning
high-speed flight in the wild", Science Robotics 6:abg5810 (2021).*

## Pregunta

¿Se puede reproducir el método completo del paper dentro de nanouav, sin las restricciones del nano
(una cámara estéreo, red grande, GPU para entrenar), como base del proyecto paralelo de seguimiento de
objetivos con evasión?

## Qué se implementó (paper → módulo)

| paper | nanouav | fidelidad |
|---|---|---|
| Flightmare (RotorS + Unity) | renderer propio por *ray casting* + `QuadSim` 6-DoF | **distinto**: mundos de primitivas texturizadas, no assets de Unity |
| Cámara estéreo simulada (VGA, base 0.1 m, ~90°) + profundidad SGM (Methods C, §S5) | `agile/stereo.py`: par izquierdo/derecho del renderer, mismo obturador, ruido independiente, `cv2.StereoSGBM` (SGM) + filtro de textura | **aproximado**: 320×240 por defecto (VGA posible con `StereoRig(640, 480)`, ~4× más lento) |
| Plataforma: esfera de 0.2 m, c_max = 35.3 m/s², J = 0.007 kg·m² (§S5) | `agile/platform.py` (`AGILE`) | **fiel** en esos tres valores; masa y motores elegidos (no están en el paper) |
| Bosques de Poisson y formas convexas, 850 entornos, δ ∈ {4…7} (Methods C) | `agile/envs.py`: `training_env(i)` | **fiel**; la intensidad se lee como 1/δ² m⁻², coherente con §S2 |
| Prueba: bosque 30×60 m, troncos de ~0.6 m, 1/25 m⁻², referencia recta de 40 m; hueco estrecho 0.8–1.0 m en una pared de 40 m a 10 m, desplazado ±5 m | `envs.forest`, `envs.narrow_gap` | **fiel** |
| Experto: trayectoria global libre de colisiones (Liu et al. 2017) | `expert.global_plan`: Dijkstra en grilla de 0.2 m con penalización de holgura, atajos y suavizado | **distinto en el método**, mismo papel |
| Experto: M-H sobre B-splines cúbicas de 3 puntos de control en esféricas, 50 000 muestras, varianzas 2/5/10 cada 16 000, coste Eq. 2–3 con λ_c = 1000, r_q = 0.2, dt 0.1 s, top-3 sin colisión | `expert.Expert.label` | **fiel** en todo lo especificado; elegido: Q = diag(10, 10, 20), σ base (0.15 m, 0.06 rad), 100 cadenas paralelas, separación mínima de 0.3 m entre las 3 etiquetas |
| Estudiante: MobileNet-V3 (preentrenada) + conv 1D → M×32; estado [v, R (9), ω] → MLP [64, 32, 32, 32] LeakyReLU → conv 1D; por modo MLP [64, 128, 128] → 10 puntos + coste; M = 3 | `agile/policy.py` (`build_model`) | **fiel**; MobileNet-V3 *Small*, pesos ImageNet opcionales (`--imagenet`), entrada 120×160 por defecto (paper 480×640) |
| Pérdida R-WTA (ε = 0.05) ×10 + coste de colisión ×0.1, Adam 1e-3, lote 8 | `policy.rwta`, `policy.train` | **aproximado** en el término de coste: C_collision de las predicciones se evalúa sobre un mapa de distancia 2D local guardado con cada muestra (±0.4 m de altura; el suelo es exacto) |
| Ejecución: proyección a polinomio de orden 5 desde el estado (Eq. 8), escala de tiempo β = v_des/v_μ, selección c*/c_k ≥ 0.95 y menor coste de entrada (*snap*), MPC | `agile/trajectory.py`, `rollout.Rollout` | **fiel** salvo el MPC: lo sustituye el controlador en cascada siguiendo p, v y a del polinomio |
| DAgger: el estudiante vuela si está a menos de ξ de τ_gbl (si no, MPC sobre τ_gbl); reentreno cada 30 entornos; ξ ← min(ξ + 0.25, 6) | `agile/dagger.py` | **fiel**; elegido: velocidad de cada entorno U(3, 10) m/s, épocas por reentreno |
| Ajuste fino del hueco: 100 entornos, pared de 50 m, hueco 0.7–1.2 m | `envs.gap_finetune_env`, `r18 dagger --kind gap` | **fiel** |
| Latencia a bordo 41.6 ms (~24 Hz) | `Config.latency = 0.05`, `cam_hz = 15` | aproximado (más lento para abaratar el render) |

El estudiante recibe solo una referencia **recta** del inicio a la meta (ω hacia el punto a 1 s), como
en el paper; τ_gbl solo lo ve el experto.

## Cómo correrlo (PC de entrenamiento)

```bash
nanouav run r18 sgm                                         # calidad del SGM frente a la profundidad real
nanouav run r18 pilots --world forest --v 3,5,7,10          # experto privilegiado, τ_gbl y ciego (sin red)
nanouav run r18 dagger --envs 850 --workers 7 --epochs 3    # recolección + reentrenos (reanudable)
nanouav run r18 eval --model results/r18/student_train.keras --world forest --v 3,5,7,10
nanouav run r18 dagger --kind gap --envs 100 --init results/r18/student_train.keras
nanouav run r18 eval --model results/r18/student_gap.keras --world gap --v 3,5,7
```

Costes medidos en el equipo de desarrollo (un núcleo por proceso): ~1 s por cuadro etiquetado (render
estéreo 0.4–0.8 s + experto 0.3–0.7 s), 50–130 muestras por entorno a 10 Hz. Con 7 procesos, los 850
entornos a 15 Hz son ~90 000 muestras (como el paper) y ~4 h de recolección; `--cam-hz 10` da ~60 000
en ~2.5 h. Los 29 reentrenos usan la GPU. Disco: ~40 KB por muestra (profundidad 120×160 en mm + mapa
local), ~3.5 GB en total. Todo es reanudable: si se corta, el mismo comando sigue donde quedó.

## Validación local (equipo de desarrollo, 2 núcleos, sin GPU)

**Profundidad SGM** (`r18 sgm --n 5`: bosque y hueco, 20 vistas, 320×240, base 0.1 m, 90°): mediana
de 39 % de píxeles válidos (la GT tiene 76 % dentro de 10 m; la banda izquierda sin correspondencia es
el 15 % de la imagen, y el muro liso del hueco pierde mucho), error relativo mediano 8 %, p90 24 %.
Huecos y bordes "pegajosos", como en una RealSense.

**Pilotos sin red** (`r18 pilots`, 6 semillas por fila; la plataforma del paper):

| mundo | piloto | 3 m/s | 5 m/s | 7 m/s | 10 m/s |
|---|---|---|---|---|---|
| bosque 1/25 m⁻² | experto privilegiado (M-H, horizonte deslizante) | 6/6 | 6/6 | 6/6 | 6/6 |
| bosque | τ_gbl (lo que vuela DAgger fuera de ξ) | 6/6 | 6/6 | 6/6 | 6/6 |
| bosque | ciego (referencia recta) | 2/6 | 2/6 | 2/6 | 2/6 |
| hueco 0.8–1.0 m | experto privilegiado | 6/6 | 6/6 | 6/6 | — |
| hueco | τ_gbl | 6/6 | 6/6 | 6/6 | — |
| hueco | ciego | 0/6 | 0/6 | 0/6 | — |

Tres correcciones salieron de esta validación (ya incluidas):
1. La B-spline del experto empezaba en la posición actual pero no con la velocidad actual; al
   proyectarla al polinomio de orden 5 (que sí respeta p, v, a) la trayectoria ejecutada se metía
   0.1–0.2 m en un tronco que la etiqueta esquivaba: 2/6 choques a 3 m/s. Ahora las dos primeras
   puntos de control fijan τ(0) = x (posición y velocidad), como dice el paper.
2. El piloto experto elige, entre sus 3 mejores, la primera cuyo polinomio *ejecutado* queda libre.
3. En huecos de 0.8 m (holgura de 0.2 m por lado) casi toda muestra paga coste de colisión y el M-H no
   siempre llega a la trayectoria centrada: el tramo de τ_gbl se añade como candidato más. La búsqueda
   global se reintenta con grilla más fina si la de 0.2 m no encuentra el hueco.

**Mini DAgger** (60 entornos, 2 rondas, ~4 000 muestras, CPU, *antes* de la corrección 1):
pérdida de validación R-WTA 21 → 11, y en el bosque de prueba el estudiante logra 7/24 frente a 8/24
del ciego. Es lo esperable con el 7 % de los datos y ξ ≤ 0.25 m (el estudiante voló el 33 % de los
cuadros en la segunda ronda y chocó en 9 de 30 entornos): **no dice nada todavía del método**, solo que
la tubería completa (recolección en paralelo → entrenamiento → vuelo del estudiante) funciona. La
corrida de referencia es la de 850 entornos en el PC de entrenamiento.

## DAgger completo en el PC de entrenamiento (850 entornos, 2026-10-05)

`nanouav run r18 dagger --envs 850 --workers 7` en una TITAN X (Pascal) y 8 núcleos. Las rondas 1–7 se
entrenaron con 3 épocas y batch 8; desde la ronda 8, con 6 épocas y batch 32. Total: **78 334 muestras**
(el paper tiene ~90 000). Cada ronda tardó 12–15 min.

| ronda | ξ | éxito del episodio | cuadros volados por el estudiante |
|---|---|---|---|
| 1 | 0 | 29/30 | 0 % |
| 5 | 1.0 | 8/30 | 61 % |
| 8 | 1.75 | 8/30 | 92 % |
| 10 | 2.25 | 15/30 | 89 % |
| 16 | 3.75 | 17/30 | 94 % |
| 22 | 5.25 | 18/30 | 98 % |
| 24–25 | 5.75–6 | 19/30 | 97–100 % |
| 26–29 | 6 | 16/30, 16/30, 16/30, 6/10 | 95–100 % |

Desde ξ ≈ 5 el estudiante vuela casi todo el episodio. Esos éxitos (17–19 de 30 en entornos de
entrenamiento al azar, con velocidad entre 3 y 10 m/s) son la mejor estimación previa de lo que dará
`r18 eval`.

**Pérdida de validación.** No sirve para comparar rondas ni para elegir el modelo:
- el conjunto de validación (5 % de las muestras al azar) cambia en cada ronda y se vuelve más difícil,
  porque las rondas nuevas son estados que visita el estudiante;
- dentro de una ronda salta de época en época (ronda 22: 16.5 → 33.1 → 19.3 → 17.0), con Adam a
  lr = 1e-3 constante, que se reinicia en cada ronda;
- se guardaba la **última** época. La ronda 29 terminó con val 22.8 cuando la primera época daba 16.8.

Por eso 0.16.1 añade a `train` y `dagger`:
- un descenso coseno del lr (`--lr-end`);
- guardar la mejor época (`--best`);
- validación con entornos enteros (`--val-by-env`), porque los cuadros seguidos de un vuelo casi se
  repiten y separar muestras sueltas da una validación optimista.

Con eso, un repaso del modelo final sobre las 78 000 muestras (`r18 train --init ... --name
student_polish`) pule los pesos sin recolectar nada nuevo.

## Evaluación del estudiante (bosque de prueba 1/25 m⁻², 10 semillas por velocidad)

| modelo | 3 m/s | 5 m/s | 7 m/s | 10 m/s | total |
|---|---|---|---|---|---|
| experto (referencia) | 10/10 | 10/10 | 10/10 | 10/10 | 40/40 |
| `student_train` (última ronda de DAgger) | 2/10 | 8/10 | 7/10 | 5/10 | 22/40 |
| `student_polish` (10 épocas más, lr 3e-4 → 1e-5, mejor época = 6) | 4/10 | 4/10 | 8/10 | 7/10 | 23/40 |
| `student_train`, proyección corregida (0.17.0) | 3/10 | 6/10 | 7/10 | 6/10 | 22/40 |
| `student_polish`, proyección corregida (0.17.0) | 4/10 | 5/10 | 3/10 | 4/10 | 16/40 |
| sin percepción (`blind`) | | | | | 16/40 |

- **Los dos modelos son equivalentes.** Con 10 episodios por casilla el intervalo es de ±15 puntos. El
  repaso bajó la R-WTA de validación (10.8, ahora con entornos enteros) pero no cambió el vuelo, así que
  el cuello de botella no es la pérdida de imitación.
- **Lo anómalo es 3 m/s.** Es lo más fácil para el experto (10/10) y lo peor para el estudiante.

**Una causa encontrada: la proyección al polinomio (corregida en 0.17.0).**
- El estudiante predice 10 puntos para 1 s. Esos puntos se ajustan a un polinomio de orden 5 que
  arranca en la posición, velocidad y aceleración actuales (Eq. 8). Después el tiempo se escala por
  β = v_des / v_μ, para que el plan se vuele a la velocidad pedida.
- Hasta 0.16.1, la velocidad inicial se imponía *antes* de escalar. Así, cada plan nuevo empezaba a
  β·v, no a v.
- Las predicciones regresan hacia la velocidad media del entrenamiento (U(3, 10) → 6.5 m/s). A 3 m/s
  el estudiante predice ~4.3 m en el segundo, así que β ≈ 0.7. Medido con la política de prueba: la
  velocidad mandada caía ~0.9 m/s (30 %) quince veces por segundo. A 10 m/s, β ≈ 0.95 y el salto
  es del 5 %.
- Ahora el ajuste usa v/β y a/β², y el plan escalado empieza exactamente en el estado actual
  (`Rollout.PROJ = 'scaled'`; `'raw'` reproduce lo anterior).
- τ_gbl y el experto no cambian (β ≈ 1): 6/6 a 3 y 10 m/s.
- El modelo no hay que reentrenarlo: la proyección solo se aplica al volar.

**Resultado de la corrección:** `student_train` queda igual (22/40) y `student_polish` baja (16/40;
la diferencia con 23/40 está en el borde del ruido). Las caídas de velocidad no eran la causa
principal de los choques a 3 m/s. La corrección se queda porque es la proyección correcta, pero la
causa hay que buscarla con el diagnóstico. El mejor modelo por ahora es `student_train`.

**Para saber el resto:** `nanouav run r18 diagnose` repite los episodios guardando cada decisión del
estudiante y clasifica cada choque según sus últimos 0.5 s:

| clase | qué pasó | qué apunta |
|---|---|---|
| `todas_chocan` | todas las hipótesis chocan según el mapa real | percepción o imitación |
| `eligio_mal` | había una hipótesis libre y se eligió una que choca | la cabeza de costo |
| `plan_justo` | el plan elegido pasaba a menos de 15 cm del límite | margen insuficiente |
| `seguimiento` | el plan tenía margen y aun así chocó | control o latencia |

También cuenta:
- los cambios de lado de la hipótesis elegida en los últimos 2 s (indecisión);
- si el obstáculo con que chocó estaba dentro del campo de visión (±45°) en los últimos 0.5 s
  (`fuera_vista`). La política no tiene memoria: un árbol que queda al lado al pasarlo ya no existe
  para ella, y a baja velocidad el dron pasa más tiempo junto a cada árbol.

Guarda una vista cenital de cada choque en `results/r18/diag_<modelo>_<proyección>/`.

`nanouav run r18 video --model ... --v 3 --seed 1000` hace un vídeo del episodio con tres vistas:
- la vista cenital, con las hipótesis de cada decisión coloreadas según el mapa real;
- la profundidad SGM que recibe la red, con las hipótesis proyectadas;
- una cámara de seguimiento.

`--pilot expert` da el mismo episodio volado por el experto, para comparar.

**Diagnóstico de `student_polish`** (`diagnose --v 3,10 --n 10`, 12 choques):
- **Sobrevelocidad.** A v_des = 10 m/s, cuatro de los seis choques fueron a 12–19 m/s. La mediana de
  β fue 1.15–1.54 y las hipótesis tenían ganchos y curvas. Causa: β se calculaba con la distancia
  recta al último punto, así que una predicción curva se volaba mucho más rápido que v_des. Esa
  velocidad, además, queda fuera de lo que la red vio al entrenar (3–10 m/s).
  - `--tscale arc` toma β de la longitud del camino.
  - `--vcap 1.3` limita el pico de velocidad de cada plan a 1.3 · v_des.
  - A 3 m/s, dos de los seis choques fueron a 4–6 m/s.
- **Cambio de plan.**
  - 0.4–0.5 s antes del choque, el plan elegido tenía margen en 7 de los 12 casos (0.5–1.8 m).
    Después la elección cambiaba a planes que chocaban.
  - A 3 m/s hubo de 1 a 9 cambios de lado en los últimos 2 s (mediana 3).
  - Es indecisión entre rodear el árbol por la izquierda o por la derecha.
  - `--select consistent` mantiene el lado mientras el costo predicho no empeore más de un 25 %.
- La primera versión del diagnóstico daba `todas_chocan` en 11 de 12 choques porque miraba los
  últimos 0.5 s, cuando el choque ya no se puede evitar. Ahora la clase se decide con las decisiones
  de 0.35–1 s antes y `cambio_de_plan` es una clase propia.

**Qué decide el siguiente cambio** (según la clase que domine en `diagnose`):

| domina | cambio |
|---|---|
| `todas_chocan` | más capacidad de percepción: profundidad a 240×320 en vez de 120×160 (`--net-size`), MobileNet preentrenada (`--imagenet`), más rondas a baja velocidad |
| `fuera_vista` | memoria: apilar 2–3 cuadros de profundidad o dar el plan anterior como entrada; o adelantar el yaw hacia donde va a girar |
| `eligio_mal` | la cabeza de costo: más peso (λ₂) o etiquetas de costo del mapa real |
| `plan_justo` | más margen en las etiquetas del experto (r_q 0.2 → 0.3 m) |
| `seguimiento` | control: latencia, ganancias, o seguir el plan con un MPC como el paper |

**Meta**:
- en el bosque de prueba: ≥ 90 % a 3–5 m/s, ≥ 70 % a 7 m/s y ≥ 50 % a 10 m/s, el orden de lo que el
  paper reporta en simulación;
- después, ≥ 80 % en el hueco estrecho;
- y la misma política siguiendo el globo de R20.

## Siguiente paso: seguimiento de objetivos

La política está condicionada por la dirección deseada ω. Para el proyecto paralelo: detector de 1–3
clases sobre el RGB → posición 3D del objetivo con la profundidad → tracker → ω hacia la posición
predicha a 1 s menos la distancia de seguimiento, y v_des según la velocidad del objetivo. En DAgger, el
experto recibe como referencia la trayectoria futura real del objetivo (privilegiada) y los entornos
incluyen objetivos móviles. El conflicto de orientación (la cámara mira hacia donde vuela, pero hay que
mantener el objetivo en vista) sigue abierto; ver R17 (punto ciego lateral).
