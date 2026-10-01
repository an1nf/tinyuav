# Problemas conocidos

Defectos encontrados al revisar todo el código durante la reestructuración (v0.13.0). **No se
corrigieron a propósito.** La reestructuración tenía que dejar los resultados numéricos idénticos, y
varios de estos defectos están en rutas que usan los experimentos publicados. Cada corrección debería
ir en su propio commit, con test y con entrada en el `CHANGELOG.md`, indicando si cambia algún
resultado.

Gravedad: **A** = puede dar resultados erróneos o una excepción en un camino que se usa;
**M** = camino poco usado o efecto acotado; **B** = cosmético o documentación.

## Planificación y control

| grav. | dónde | problema |
|---|---|---|
| A | `control/flight_stack.py` · `ReferenceGenerator.set_evade` | usa `self.sup`, `self.ref`, `self.ctrl` y `self.limits`, que `ReferenceGenerator` no tiene (copiado de `FlightStack.set_evade`). Llamarlo lanza `AttributeError`. Hoy solo se llama al de `FlightStack`. |
| M | `control/flight_stack.py` · `FlightStack` | si se entra en RECOVER con `_evade=True`, al salir se restauran los límites normales pero `_evade` sigue en `True`, así que los siguientes `set_evade(True)` no hacen nada: el resto de esa evasión vuela con el perfil normal. |
| M | `planning/escape.py` | en la maniobra de escape se mezclan los ejes de `vy`, y el centroide de espacio libre no se normaliza por el número de celdas. |
| M | `planning/evasion.py` · `LocalGrid.mark_seen` | si el dron sale de la rejilla por el lado negativo, `ix1`/`iy1` quedan negativos y el *slice* `max(ix0,0):ix1` marca como visto un trozo del lado opuesto. |
| M | `planning/mission.py` · `TargetTracker.update` | divide por `dt` sin comprobar que sea > 0. |
| M | `sim/closed_loop.py` · `MODE_IDX` | no incluye `'EVADE'` (ni los modos del MH: `SQUEEZE`, `INSPECT`, …); en el registro aparecen como 0 (= CRUISE). |
| B | `control/supervisor.py` | el docstring describe umbrales distintos de los que usa el código. |
| B | `sim/closed_loop.py` | el docstring de la patada de perturbación dice 3.5 y el código usa 6. |
| M | `planning/mh_planner.py` · márgenes | los márgenes (`GAP_BIAS`, `M_MIN`, `M_ABS`, `D_EMERG`…) son absolutos y están ajustados para el X3. Con el nano (Ø 0.22 m) y profundidad perfecta, `estrechos` con w = 0.6 se cruza pero dando muchas vueltas (46 m de recorrido), y con w = 0.45 no se cruza ni con rejilla de 5 cm. La equivalencia "w = 0.45 para el nano ≈ w = 1.2 para el X3" de R13 no se cumple. Probar márgenes proporcionales al radio. |
| — | `planning/mh_planner.py` (comportamiento) | ante una puerta de 1.0 m en una pared de 6 m con salida lateral, el MH prefiere rodear la pared en lugar de cruzar la puerta (con 1.4 m sí la cruza). Es coherente con su margen (`radius` 0.36 + `GAP_BIAS`), pero vale la pena revisarlo junto con R13. `tests/test_planners.py` fija ambos casos. |

## Escenarios de evaluación

| grav. | dónde | problema |
|---|---|---|
| A | `experiments/r10_dynamic.py` · `cross` | rumbo de colisión constante: con v_obs ≥ 0.58·v_dron el balón queda fuera del campo de la cámara (±30°) hasta ~0.3 m. Los éxitos a 1.0 y 2.0 m/s se deben a la sincronización (el dron va a ~0.93 m/s, no a 1.0), no a que se esquive: con GT, 0/20 a 1.0 m/s y 20/20 a 2.0 m/s, en ambos casos sin detección previa. No reportarlos como evasión. Se necesita más campo visual (o ToF laterales / barrido de yaw) para ese régimen. |
| M | `experiments/r05_pole.py` | con ley *brake* la velocidad real se satura en ≈ 1.73 m/s: las filas 2.0/2.5/3.0 m/s son el mismo experimento. |

## Percepción

| grav. | dónde | problema |
|---|---|---|
| A | `perception/early_exit.py` | selecciona una capa `re_lu` que no es la que pretende y trata el mapa de características como profundidad. Los números de E2d son un *proxy*, no una salida temprana real. |
| M | `perception/cnn.py` | `VARIANTS` no tiene entrada `M_ToF` (falla `jitter_sample('M_ToF')`); `predict_depth()` no pasa `tof`, así que con M_ToF solo funciona `predict_depth_norm(..., tof=...)`; `_real_mode` es global y refleja la última variante cargada. |
| M | `perception/tof_calibrator.py` · `ToFCalibrator.reset` | no limpia `last_quality`. |
| B | `experiments/e0c_p3_latency.py` | los colores se dan en RGB a funciones de OpenCV que esperan BGR. |

## Simulación, experimentos y entrenamiento

| grav. | dónde | problema |
|---|---|---|
| A | `sim/missions.py` · `plot_missions` | con `--only follow` no hay corridas de waypoints y `plt.subplots(1, 1)` devuelve un solo eje: falla al iterar. |
| M | `sim/scenarios.py` · `interior`, `corredor` | cuando el muestreo de posiciones agota los intentos, el objeto se coloca igualmente y puede solaparse con otro. `tests/test_scenarios.py` comprueba que A y B quedan libres para las semillas 0–2. |
| M | `experiments/r12_scenarios.py` · `done()` | la segunda condición (`abs(math.hypot(*[0, 0])) < 1`) es un marcador que siempre es verdadero: no exige velocidad baja al llegar. |
| M | `experiments/r11_vel_net.py` | `--out` y `--data` se resuelven relativos a la raíz del repo, no a `NANOUAV_DATA`; con esa variable definida, los archivos no van a donde se espera. `genfull` no aparece en la ayuda. |
| M | `experiments/r08_state_noise.py` | `--decay 0` da el mismo nombre de archivo que la configuración por defecto y la sobrescribe. |
| M | `training/tartanair.py` | el dataset representativo para la cuantización no aplica la misma escala que el entrenamiento. |
| M | `benchmark/benchmark.py` | el índice del p99 (`int(0.99 * N)`) depende de que `N` sea igual a la longitud de la lista. |
| B | `training/r3_lnca_train.py` | la ayuda de `--init-from` y el nombre `LNCA24+ToF` no coinciden con los ids reales (`LNCA24_ToF`). |
| B | `experiments/e1a_jitter_model.py`, `e1b_energy.py` | las conclusiones impresas están escritas a mano y no se recalculan con los parámetros. |
| B | `experiments/e2b_tof_calibration.py`, `e2d_early_exit.py`, `figures.py` | etiquetas ocultas por `axis('off')`, leyenda sobrescrita, tabla desalineada. |
| B | `experiments/r02_follow_sensors.py`, `r15_dyn_egomotion.py` | el docstring cita un archivo de salida / un umbral (0.4 m vs 0.6 m) distintos de los del código. |
| B | `benchmark/jitter_model.py` | el docstring del modelo WiFi no coincide con los parámetros. |

## Infraestructura

| grav. | dónde | problema |
|---|---|---|
| M | módulos con `ASSETS = str(paths.results_dir())` | la carpeta de resultados se fija al importar el módulo. `nanouav run` y las suites definen `NANOUAV_RESULTS` antes de importar, así que funciona; desde un notebook hay que definir la variable antes del primer `import`. |
| B | `gazebo/batch_eval.sh` | obtiene el resultado buscando el texto `=== EPISODIO:` en el log; si se cambia ese mensaje hay que cambiar el script (hay un test que lo vigila). |
| B | `gazebo/nodes/flight_controller_node.py` | modo `motors` sin probar en Gazebo. |
| B | lint | `ruff` solo revisa errores (sintaxis, nombres indefinidos). Hay imports sin usar en varios módulos; activar `F401` cuando se limpien. |
