# R13 · Pasos estrechos y anti-estancamiento en el planificador MH

## Pregunta
¿Puede el X3 (envolvente Ø 0.72 m) pasar por puertas y pasillos estrechos sin quedarse detenido?

## Escenario nuevo: `estrechos` (`gazebo/scenarios.py`, parámetro `w` = ancho libre)
Una puerta en una pared continua de 32 m (no se puede rodear), un pasillo de 2.5 m y una segunda puerta
desplazada. Con w = 1.0 m quedan 0.14 m por lado; con w = 1.2 m, 0.24 m.

Nota sobre el mundo `obstaculos`: sus huecos miden 0.63 m y 0.75 m. El X3 (hélices incluidas) mide
0.72 m, así que rodear la pared es lo correcto; ese mundo no sirve para probar pasos estrechos.

## Cambios en MHPlanner (`_04_policy/mh_planner.py`)
| Cambio | Motivo |
|---|---|
| Margen que depende de la velocidad: m(v) = 0.05 + 0.4·v; la velocidad se limita a (holgura − r − 0.05)/0.4 | Antes el margen era fijo en 0.15 m, así que una puerta de 1.0 m resultaba infactible. Ahora el dron pasa despacio cuando el espacio es justo |
| Penalización por holgura en las hipótesis: W_CLR 3.0 → 0.4 | Con 3.0, desplazarse de lado sin progresar era "más barato" que acercarse a una puerta. El dron vagaba frente a la pared |
| La línea de referencia (descenso por el costo-a-meta, centrada en el hueco) entra como hipótesis extra, con margen 0.015 m y a 0.12 m/s (modo SQUEEZE) | Los arcos tienen 10° de resolución y fallaban la abertura; la referencia pasa por el centro |
| Costo graduado: entre r y r + m vale 3; dentro de r vale 25 | El ruido del mapa no debe convertir un paso real en un rodeo de 15 m |
| Histéresis de ruta: las celdas de la ruta anterior cuestan ×0.6 | Evita alternar entre rutas cuando el mapa de la CNN fluctúa |
| Anti-estancamiento: si en 3 s no mejora ni la distancia a B ni el costo-a-meta, el margen baja a 0.03 m. Después entra en modo INSPECT: va hasta ~1.1 m antes del cruce dudoso, lo mira de frente 1 s y decide. Si se abrió, pasa; si no, lo rechaza (penalización que se olvida en ~20 s) y busca otra ruta | Garantiza que nunca queda detenido indefinidamente. La percepción activa resuelve las puertas que de lado o de lejos parecen cerradas |
| Prohibido moverse a ciegas (siempre hacia donde mira la cámara) | Al retroceder a ciegas chocó contra la cara trasera de una pared que el mapa había olvidado |

## Resultados (simulador propio, X3, 0.5 m/s)
| Configuración | Profundidad | Mundo | Éxito |
|---|---|---|---|
| MH R12 (anterior) | perfecta | estrechos w = 1.0 | **0/2** (vaga frente a la pared) |
| MH R13, celda de 5 cm | perfecta | estrechos w = 1.0 | **2/2** (68–75 s, holgura mínima 0.42–0.44 m) |
| MH R13, celda de 5 cm | CNN M | estrechos w = 1.2 | 2/3, 1 choque con el canto de la puerta |
| MH R13, celda de 5 cm | CNN M | estrechos w = 1.0 | 0/3 (cruza la primera puerta, no el pasillo) |
| MH R13, celda de 10 cm (por defecto) | CNN M | estrechos w = 1.2 | **2/3, sin choques** (una de ellas es la semilla que chocaba con 5 cm; el fallo es un timeout) |
| MH R13, celda de 10 cm (por defecto) | CNN M | obstaculos | 2/2 (22–38 s) |
| MH R13, celda de 5 cm | CNN M | obstaculos | 4/4 (sin regresión) |

## Hallazgo: con CNN, el límite es la percepción, no el planificador
Con profundidad perfecta, el MH R13 pasa una puerta de 1.0 m (14 cm por lado). Con la CNN M, la imagen de
mapa (`research/R13_mapa_cnn_puerta.png`) muestra que la pared queda como una banda de ~1 m de espesor. Las profundidades de
la CNN tienen ±30 % de error a 1–3 m y sus bordes son difusos, así que la puerta aparece "rellena" vista
de lado o desde lejos. De frente, a ~1.2 m, sí se ve abierta (holgura 0.6–1.2 m en el mapa). De ahí sale
el modo INSPECT.

Probé dos variantes que no sirvieron y quedan desactivadas con flags:
- Restar peso a la evidencia lejana de la CNN (`RANGE_W`): la pared, vista de lejos, parece abierta y el
  dron recorre la pared buscando un hueco que se mueve con él.
- Hacer más lento el borrado del mapa (`free_k` 0.8, memoria de 70 s): los fantasmas persisten.

La celda de 5 cm resuelve mejor la holgura, pero los rayos rasantes borran paredes con más facilidad.
Por eso el valor por defecto es 10 cm; `MH_GRID_RES=0.05` queda para profundidad buena (ToF denso,
Gazebo con profundidad, M_ToF).

## Siguiente paso (contribución)
1. Que el ToF tenga prioridad en el campo cercano: dentro del cono del ToF, la evidencia de la CNN que
   contradiga una lectura más lejana se descarta. La función `fuse_tof` ya acota esa evidencia a 0.8 de
   la lectura del ToF, pero solo en los 40° centrales.
2. Medir M_ToF (bordes más nítidos, AbsRel 0.081) en `estrechos`: debería acercar la CNN al resultado
   con profundidad perfecta.
3. Para el nano (Ø 0.22 m), `estrechos` con w = 0.45 equivale a w = 1.2 para el X3.
