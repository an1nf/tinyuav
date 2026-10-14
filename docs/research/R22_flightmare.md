# R22 · El estudiante de R18 en Flightmare

*2026-10-07 · nanouav 0.19.0 (convenciones de Unity corregidas en 0.19.1; mapa por escaneo en 0.19.2; líneas verificadas en 0.19.3; primera demo en 0.19.4; entrenamiento en Flightmare en 0.20.0; revisión antes de entrenar en 0.20.1; diagnóstico en la arena en 0.20.2) · sobre R18.*

## La idea

Flightmare es el simulador del paper. Tiene dos partes desacopladas:
- un **renderizador** en Unity: escenas fotorrealistas, entre ellas un bosque;
- una **física** en C++ (flightlib).

Aquí solo se usa el renderizador. Así se cambian las imágenes y el resto queda igual: nuestro modelo
de cuadricóptero, el controlador, el experto, el estudiante y la evaluación. Cualquier cambio de
resultado se debe a las imágenes.

No hace falta compilar flightlib, ni ROS, ni Ubuntu 20.04. `nanouav.flightmare.bridge` habla desde
Python el protocolo ZMQ con que flightlib conversa con Unity. Lo único externo es el ejecutable de
Unity.

| pieza | qué hace |
|---|---|
| `flightmare.bridge.FlightmareClient` | envía la configuración (escena, desplazamiento de la escena, un vehículo con dos cámaras: el par estéreo de `StereoRig`, 320 × 240, base 0.1 m) y luego una pose por cuadro. Recibe las dos imágenes y la profundidad de Unity; pide la nube de puntos de una caja de la escena. Sirve para el binario de 2020 (GitHub) y para el de 2022 (agile_flight) |
| `flightmare.cloud.CloudWorld` | la escena como mapa privilegiado: distancia al punto más cercano de la nube (árbol k-d), con paredes en los bordes de la caja. `geometry.Field` la acepta, así que el choque, la ruta global, el experto M-H y las etiquetas de DAgger funcionan sin cambios |
| `cloud.find_lines` | líneas de salida → meta de 40 m: salida despejada, terreno casi plano a lo largo de la línea (el estudiante vuela a altura fija) y una ruta global libre |
| `run.FlightmareObserver` | `Rollout.observer`: en cada cuadro pide a Unity el par estéreo en la pose del dron y calcula el SGM igual que en el entrenamiento |
| `nanouav run r22 probe` | comprueba en un cuadro real las convenciones: que hay base estéreo, que la profundidad de Unity × plano lejano coincide con el SGM, que la nube explica la profundidad (campo de visión vertical u horizontal, eje y). Escribe `calib.json`, `probe.png` (izquierda, derecha / SGM, profundidad de Unity, profundidad predicha por la nube) y `probe.npz` (los datos crudos) |
| `nanouav run r22 fake` | un reproductor de Unity falso: nuestro renderizador detrás del mismo protocolo, para probar todo sin Unity (lo usan los tests) |

## Lo que hace el reproductor de Unity (0.19.1)

El primer probe con el Unity real (bosque, binario 2022) falló en tres cosas. El código fuente del
reproductor es público (github.com/uzh-rpg/flightmare_unity, rama `dev/agile_flight`), y explica las tres:

| síntoma en el primer probe | causa (archivo de Unity) | corrección |
|---|---|---|
| los brazos del dron tapan la parte de arriba de las dos imágenes | el `size` del vehículo escala la malla del dron, y la cámara (a 0.1 m del centro) queda dentro de ella. Las cámaras se colocan con escala 1, así que el tamaño no cambia la base estéreo (`CameraController.cs`) | `size` 0.01, como en el puente del propio Loquercio (`agile_autonomy/flightmare_bridge.cpp`); `--vehicle-size` |
| la profundidad de Unity sale toda casi 0 | la capa de profundidad es la distancia sobre el eje óptico dividida por su plano lejano (`Linear01Depth`, modo `DepthRaw` de `UberReplacement.shader`) | metros = valor × `farClipPlane[1]` (100 m); `probe` lo comprueba contra el SGM |
| la nube es una lámina plana (15 cm de alto) | la petición de nube no suma el `render_offset`: la caja es absoluta. Pedimos (15, 0, 2) y Unity muestreó 94 m por debajo del dron, a la altura de una superficie plana (`SavePointCloud.cs`) | `cloud_points` pide la caja en origen + desplazamiento y resta el desplazamiento a los puntos |

Además, `SavePointCloud.cs` marca un vóxel como ocupado si `Physics.OverlapSphere(centro, resolución)`
toca un collider, y guarda los centros en orden ROS (x, y, z arriba), absolutos. Los árboles del bosque
tienen collider de malla (`terrainTreeManager.cs` los instancia como prefabs), así que salen en la nube.
Como el vóxel ya está inflado una resolución, la distancia al centro ocupado más cercano es una cota
inferior del despeje real: `CloudWorld` ya no resta media resolución. El suelo de cada celda es el techo de
la primera tira de vóxeles desde abajo, con una mediana 3 × 3 que quita las celdas con tronco; vale igual
si el terreno sale como lámina o como bloque.

El reproductor falso imita ahora todo esto: profundidad dividida por el plano lejano, la caja de la nube
absoluta y la ocupación por radio.

## El mapa por escaneo de profundidad (0.19.2)

El segundo probe con el Unity real confirmó la malla oculta (16 niveles de gris de diferencia
izquierda–derecha, 44 % de píxeles con SGM) y la escala de la profundidad (SGM / Unity = 96 frente a 100
esperado). Pero la nube, ya pedida en el sitio correcto, solo trae 3452 puntos: una lámina de 17 × 18 m a
z ≈ −1.8 m, bajo el río, sin árboles ni orillas. **El bosque de 2022 casi no tiene colliders**: se hizo para
agile_flight, donde los obstáculos son objetos que se agregan a la escena y los choques se calculan con la
lista de esos objetos, no con la escena.

`r22 scan` construye el mapa con lo que Unity dibuja. Pone la cámara en una malla de poses (cada 4 m, a 1.5
y 3 m, 6 rumbos, cabeceo −30°, 0° y 30°) y retroproyecta cada imagen de profundidad, que es exacta, a puntos
del mundo. Después agrupa los puntos en vóxeles de 15 cm. Escribe el mismo `cloud_<escena>.npz` que `r22 cloud`, así
que `lines`, `eval` y `video` no cambian. Al final predice desde el mapa la profundidad de una vista fuera
de la malla y la compara con la de Unity (con el falso: 4.9 %).

Hay dos límites:
- lo que ninguna pose ve falta en el mapa;
- la pasada de profundidad dibuja las hojas como triángulos sólidos (se ven en `probe.png`), así que las
  copas salen más grandes de lo que son y el experto las esquiva de más.

## Líneas verificadas y Unity por comando (0.19.3)

Con el mapa escaneado del bosque real, `lines` encontró una línea (a lo largo del río) y τ_gbl chocó en
ella a 3 m/s, a los 14.8 s. `find_lines` aceptaba la ruta global con la búsqueda de respaldo, que baja el
inflado hasta 2 cm del mapa. Ahora una línea se acepta solo si:
- la salida y la meta tienen 1.2 m libres;
- hay ruta con el inflado completo;
- **τ_gbl la vuela sin chocar** (`--check-v 3`; 0 lo desactiva).

`lines` informa por qué rechazó cada candidata.

El reproductor de Unity vuelve a su menú 10 s después de que un cliente se va, y no siempre se reconecta. Con
`--unity ~/flightmare_unity/RPG_Flightmare.x86_64` (o `export NANOUAV_FLIGHTMARE=…`), cada comando abre
su propio reproductor y lo cierra al terminar. El registro queda en `results/r22/unity.log`.

## Primera demo en el bosque real (0.19.4)

Con el mapa escaneado y la línea verificada:
- **τ_gbl** llega a la meta (13 s a 3 m/s, siguiendo el río);
- **el estudiante de R18** (`student_train`) choca a los ~4 s. Elige una hipótesis hacia la orilla: la red
  le daba el costo más bajo (0.05), pero en el mapa choca. La profundidad SGM de esas imágenes casi no tiene
  datos (agua, cielo y follaje oscuro sin textura que emparejar, y lo que está a más de 10 m queda fuera de
  rango). La red nunca vio nada así: se entrenó con nuestro renderizador, donde el SGM es denso.

Para la demo y el diagnóstico:
- `r22 video` con un piloto privilegiado (gbl, expert) ahora también graba las imágenes de Flightmare y su
  SGM, con el plan del piloto proyectado (`Rollout.record_cam`);
- cuando hay choque, `eval` y `video` dicen qué punto del mapa se golpeó, a qué altura sobre el suelo
  estaba y cuántas vistas del escaneo lo vieron. El video lo marca con una cruz roja;
- `scan` descarta los vóxeles que vieron menos de 2 vistas (`--min-hits`): una superficie real se ve desde
  muchas poses, y un punto aislado no;
- `r22 survey` arma un mosaico cenital de la escena (color y altura de la superficie más alta por metro),
  con una malla de 10 m, la caja escaneada y las líneas. Sirve para elegir una zona de bosque en tierra
  firme, donde el SGM tiene textura con qué trabajar.

## Entrenar en Flightmare como el paper (0.20.0)

El estudiante de R18 se entrenó con nuestro renderizador. En Unity ve otras imágenes y otro SGM, y no vuela
bien. El paper entrena dentro de Flightmare; esta versión hace lo mismo.

| pieza | el paper | aquí |
|---|---|---|
| escena | un bosque vacío con árboles sembrados por procedimiento en posiciones conocidas (agile_autonomy: `spawn_trees`, separación media 4 m) | la escena vacía (por defecto el almacén de agile_flight) escaneada una vez, más troncos sembrados: el prefab `rpg_box02` (caja de madera de 0.81 m con collider) escalado a 0.3–0.8 × 0.3–0.8 × 6 m, como objeto dinámico que Unity mueve en cada cuadro |
| mapa del experto | la nube de los colliders de esos árboles | la malla de distancias de la escena escaneada ∪ las cajas (exactas) |
| cámara | estéreo VGA, base 0.1 m, ~90° | igual: 640 × 480, SGM con 96 disparidades (33 ms en CPU; error 2.8 % frente a 4.2 % a media resolución, en nuestro bosque) |
| DAgger | 850 entornos, ξ creciente, reentrenar cada 30 | igual, empezando desde `student_spatial_in`; las etiquetas del experto (~0.4 s cada una) se calculan en paralelo mientras vuela el siguiente entorno |

`r22 arena`:
- escanea la escena vacía;
- encuentra el suelo y el rectángulo libre más grande, donde de 0.5 a 4 m sobre el suelo hay al menos
  1.2 m libres;
- comprueba con la profundidad de Unity que una caja sembrada aparece donde y del tamaño que debe. Si no,
  corrige `box_unit` en `calib.json`.

Cada entorno de entrenamiento es un bosque de Poisson de troncos con separación δ ∈ {4, 5, 6, 7} m. Los de
prueba tienen densidad 1/25 m⁻² y troncos de 0.5–0.7 m, el protocolo de R18. `r22 eval --arena` y
`r22 video --arena` los usan.

Validado con el reproductor falso (un almacén de prueba): la caja sale en su sitio y la malla de distancias es
exacta; DAgger de 2 rondas vuela, etiqueta en paralelo, entrena y evalúa. τ_gbl llega en todos los entornos.

## Revisión antes del primer entrenamiento (0.20.1)

Tres fallos habrían arruinado el entrenamiento o lo habrían alejado del paper. 0.20.1 los corrige:
- **El reproductor se habría reiniciado entre rondas.** Vuelve a su menú tras 10 s sin mensajes
  (CameraController.cs), y entre rondas el estudiante entrena varios minutos. En `eval --arena`, los
  episodios de τ_gbl tampoco piden imágenes. El cliente ahora manda un cuadro cada 2 s cuando está ocioso.
  Además, cada pose lleva la configuración: si aun así el reproductor se reinicia, la toma como tal,
  recarga la escena y el cliente reenvía la pose. El reproductor falso imita ese reinicio y hay una prueba.
- **La cámara no se inclinaba a alta velocidad.** El paper inclina el sensor de profundidad 30° hacia arriba
  para vuelos a ≥ 7 m/s, para que mire al frente mientras el dron cabecea (Materials and Methods). R18
  nunca lo hizo: a 10 m/s la cámara mira al suelo, lo que explica en parte el 1/10. En Flightmare se aplica
  por episodio (`FlightmareClient.set_tilt`, `flightmare.dagger.tilted`), en entrenamiento y en evaluación.
- **La altura mínima del experto** (`z_min`, 0.4 m) era absoluta. Ahora es relativa al suelo de la escena.

El video ahora muestra solo lo que percibe el dron:
- la cámara con la trayectoria elegida como un túnel: la sección del dron (0.4 × 0.24 m) cada 0.2 s en
  perspectiva, coloreada por el costo de choque que predice la red; las otras hipótesis van como líneas finas;
- el SGM a resolución completa con escala de 0 a 10 m;
- un mapa local hecho solo con su propio SGM del último medio segundo, a su altura.

Ya no aparece el mapa verdadero ni la cámara de seguimiento (`--debug-map` los recupera). El mapa verdadero
sigue existiendo solo para el experto que etiqueta y para decidir si hubo choque, como en el paper.

## Primeras dos rondas en Flightmare (0.20.2)

60 entornos, ξ de 1.0 a 1.5, 4668 muestras (el 6 % de las del simulador propio). Prueba en 10 bosques del
almacén:

| piloto | 3 m/s | 5 m/s | 7 m/s |
|---|---|---|---|
| τ_gbl | 10/10 | 10/10 | 10/10 |
| estudiante tras 2 rondas | 2/10 | 2/10 | 5/10 |

τ_gbl confirma que el entorno, el mapa y el juez de choques están bien. El estudiante todavía no vuela, y
hay dos señales que revisar antes de seguir:
- **a 3 y 5 m/s le va peor que a 7 m/s;**
- **a 3 m/s, todos los choques son a la izquierda** de la línea (y = +2 a +11 m; uno contra la pared del
  almacén). A 7 m/s, con la cámara inclinada, no hay deriva.

La explicación más probable es la de DAgger con ξ chico: el estudiante solo vuela a menos de 1–1.5 m de la
trayectoria de τ_gbl. Si se aleja, τ_gbl lo devuelve, así que nunca ve los estados alejados. En la prueba
nadie lo devuelve, y a baja velocidad (vuelos de 13 s) la deriva se acumula. ξ crece con las rondas
justamente para eso.

Antes de gastar horas hay que descartar otras dos causas: un error geométrico (un espejo izquierda/derecha,
que la comprobación de la caja centrada de `arena` no podía ver) o la banda izquierda sin SGM. Para eso:
- `r22 check`: tres cajas fuera del eje (izquierda a 6 m, derecha a 9 m, arriba a 7.5 m), nivelada y a 30°.
  Comprueba que aparecen en la imagen de Unity y en el SGM donde las pone nuestro modelo de cámara, y que no
  hay espejo;
- `r22 diagnose --model M --v 3,5`: clasifica cada choque como `r18 diagnose`. Las clases son: todas chocan
  (percepción), cambio de plan, eligió mal (cabeza de costo), plan justo y seguimiento. Por episodio da la
  deriva lateral, a qué lado van las hipótesis elegidas y la calidad del SGM de los cuadros con que decidió
  (válido total, mitad izquierda y derecha, error frente a la profundidad verdadera). Hace un PNG por
  choque.

## La focal de Unity no era la nuestra (0.21.1)

`r22 check` (2026-10-13, VGA, almacén) encontró las tres cajas fuera del eje al **0.74 de su distancia
al centro** de la imagen, tanto en horizontal como en vertical:

| caja | esperado (px) | medido (px) | razón |
|---|---|---|---|
| izquierda, 6 m | u = 206.2 | 235.5 | 0.741 |
| derecha, 9 m | u = 431.1 | 402.4 | 0.743 |
| arriba, 7.5 m | v = 144.9 | 169.2 | 0.743 |

**Qué pasaba.**
- `calib.json` tenía `fov_deg: 90`, escrito por el probe antiguo sobre una nube de puntos plana (la escena
  no tiene colisionadores).
- Unity toma ese valor como vertical: focal de 240 px a 640 × 480, contra los 320 px de nuestra cámara
  (90° horizontales). La razón esperada es 240/320 = 0.75.
- Con esas imágenes, el SGM da distancias 1.33 veces mayores: el error relativo de 0.31–0.42 que reportó la
  comprobación. Los obstáculos, además, aparecen más centrados de lo que están.

**Qué invalida.** Todo lo que el estudiante vio en Flightmare: los vídeos de percepción, las dos rondas de
DAgger (2/10, 2/10, 5/10) y su deriva. No afecta a τ_gbl ni a la arena: el escaneo retroproyecta con el
mismo campo de visión que envía, y los troncos son exactos.

**Corrección.**
- El campo de visión sale ahora de la cámara: 73.74° vertical.
- `r22 check` mide la razón de focales y, si difiere más de un 3 %, corrige `calib.json`.
- Los datos de DAgger nuevos van a otro directorio (`fm_<escena>_ctg`, R23), así que no se mezclan con los
  viejos.

## Con la focal corregida (2026-10-13)

`r22 check`: cajas a ≤ 2 px de lo esperado, focal 0.990, SGM con 4.5–6.6 % de error sobre las cajas y un
65–92 % de cobertura en ellas. El suelo del almacén, liso, casi no tiene SGM.

60 entornos (2 rondas), arranque desde `student_spatial_in`, mismo `eval` (10 entornos de prueba):

| | 3 m/s | 5 m/s | 7 m/s | total |
|---|---|---|---|---|
| focal mal, etiquetas τ_gbl (0.20.2) | 2/10 | 2/10 | 5/10 | 9/30 |
| **focal bien, etiquetas τ_gbl** (`fm_warehouse_fov`) | 5/10 | 7/10 | 8/10 | **20/30** |
| focal bien, etiquetas replaneadas (`fm_warehouse_ctg`, R23) | 4/10 | 5/10 | 4/10 | 13/30 |

- La focal explica la mayor parte de lo malo de R22 (9 → 20, $p = 0.009$).
- Los choques que quedan son contra troncos, a 1.3–1.7 m de altura y sin deriva. Varios ocurren en los
  primeros 1.5–2.3 s, contra un tronco cerca de la salida.

## 300 entornos con las etiquetas de R18 (2026-10-13)

`--label-ref gbl --tag fm_warehouse_fov`: 10 rondas, 25 856 muestras, ξ de 1 a 5.5 m.

| ronda | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ξ (m) | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 | 5.5 |
| éxito | 10/30 | 18/30 | 14/30 | 18/30 | **21/30** | 12/30 | 12/30 | 11/30 | 14/30 | 14/30 |
| estudiante | 80 % | 84 % | 92 % | 95 % | 97 % | 96 % | 96 % | 98 % | 99 % | 98 % |

En las rondas, el éxito cae cuando ξ pasa de 3 m y el estudiante vuela casi todo el episodio: con menos
tomas de control, cada error suyo cuenta.

**`eval`.** 3/10 · 8/10 · 6/10 = 17/30. Tras 2 rondas había dado 20/30: ocho rondas más no mejoraron.

**`diagnose`** (otra corrida del mismo `eval`: 16/30):

| v | éxito | todas chocan | cambio de plan | eligió mal | deriva media |
|---|---|---|---|---|---|
| 3 | 4/10 | 0 | 2 | 4 | −0.33 m |
| 5 | 6/10 | 0 | 2 | 2 | +0.95 m |
| 7 | 6/10 | 0 | 4 | 0 | +0.61 m |

- **En ningún choque estaban todas las hipótesis en colisión.** La red propone una salida libre: la
  percepción y la imitación bastan para eso. Es al revés de R18 en nuestro simulador, donde dominaba
  «todas chocan».
- **Falla la decisión:**
  - en 6 choques eligió una hipótesis con choque habiendo una libre (la cabeza de costo);
  - en 8 volaba planes libres y cambió a uno con choque (indecisión).
- La deriva es pequeña (−0.3 a +1 m de media) y el lado elegido está equilibrado.

**Siguiente.** `--select oracle` (techo con las mismas hipótesis) y `--select consistent` (no cambiar de
plan sin motivo) dicen, sin entrenar, cuánto vale arreglar cada una.

## Elegir mejor entre las mismas hipótesis (2026-10-14)

Mismo modelo (`student_fm_warehouse_fov`, 300 entornos) y mismos 10 entornos de prueba; solo cambia la regla
para elegir entre las 3 hipótesis:

| selección | 3 m/s | 5 m/s | 7 m/s | total |
|---|---|---|---|---|
| `snap` (paper: menor costo predicho, luego menor snap) | 3/10 | 8/10 | 6/10 | 17/30 |
| **`oracle`** (holgura real del polinomio volado; privilegiado) | 7/10 | 7/10 | **10/10** | **24/30** |
| `consistent` (mantener el plan anterior si su costo es parecido) | 2/10 | 5/10 | 7/10 | 14/30 |

- **Con una elección perfecta, el mismo estudiante pasa de 17 a 24/30** ($p = 0.09$), y a 7 m/s no choca.
  Las hipótesis son en su mayoría buenas: el cuello de botella es la cabeza de costo.
- **Mantener el plan no ayuda** (14/30, $p = 0.61$). Los cambios de plan que `diagnose` marca no son
  indecisión sin más, sino cambios hacia la hipótesis que la cabeza de costo puntúa mejor por error.
- **Lo que queda con el oráculo** (6 choques) es el techo de estas hipótesis.

**Versión no privilegiada: `--select sgm`.** Aplica la misma regla que el oráculo, pero con los obstáculos que
el propio dron percibe. Son celdas de 0.15 m que su SGM tocó en ≥ 2 cuadros del último segundo, calculadas con
su propia pose. Entre los planes con ≥ radio + 0.15 m de holgura percibida, elige con la regla del paper; si
ninguno la tiene, el de más holgura.

Es un híbrido: la red propone, la geometría propia elige. Se aparta del estudiante puramente de extremo a
extremo del paper, pero no usa nada que el dron no tenga. El SGM sobre los obstáculos tiene un 5 % de error
y un 65–92 % de cobertura (`r22 check`).

**Comprobación en nuestro bosque** (CPU, el modelo pequeño G de R23, 10 bosques de prueba a 3 m/s):

| selección | llega |
|---|---|
| `snap` | 4/10 |
| `sgm` | 5/10 |
| `oracle` | 6/10 |

`sgm` queda entre los dos, como se esperaba. Son pocos vuelos y no es significativo: la prueba que cuenta es
`r22 eval --select sgm` con el estudiante de 300 entornos.

**En el almacén, `--select sgm` da 5/10 · 5/10 · 6/10 = 16/30**: igual que `snap` (17/30), lejos del oráculo
(24/30). Con su propio SGM no elige mejor que con la cabeza de costo. Para saber por qué, con `--select sgm`
cada decisión guarda la holgura que vio su SGM junto a la real (`clear_seen`). `r22 diagnose --select sgm`
resume tres medidas:
- **perdidos:** cuántos planes que de verdad chocan vio libres;
- **falsas alarmas:** cuántos libres vio bloqueados;
- **elección con choque:** en cuántas decisiones voló uno con choque creyéndolo libre.

En nuestro bosque, con el modelo pequeño, casi no pierde ninguno (0–2.6 %) y da 5–14 % de falsas alarmas.

**`r22 diagnose --select sgm` en el almacén** (4/10 · 7/10 · 5/10 = 16/30):
- Su SGM coincide casi siempre con la verdad. De los planes que de verdad chocan vio libres el 2 %; de los
  libres vio bloqueados el 7 %. Solo en el 0.1 % de las decisiones voló un plan con choque creyéndolo libre.
- De los 14 choques, 13 son `cambio_de_plan`. Volaba planes libres y, en el último medio segundo, ya no le
  quedaba ninguno: se metió en una trampa plan a plan, no por no ver.
- Sin embargo, el oráculo elige el más suave (menor snap) entre los libres, mientras que `sgm` usaba la regla
  del paper, es decir, la cabeza de costo, la misma que puntúa mal. Para separar percepción y regla:
  - `sgm` pasa a elegir como el oráculo (el más suave entre los que su SGM ve libres);
  - `sgm_cost` es el de antes (16/30);
  - `oracle_cost` es la holgura real con la regla del costo.

**El 2 × 2** (mismo estudiante, mismos 10 entornos; llegadas a 3 · 5 · 7 m/s):

| | el más suave entre los libres | costo de la red entre los libres |
|---|---|---|
| holgura real (privilegiado) | `oracle`: 7 · 7 · 10 = **24/30** | `oracle_cost`: 6 · 7 · 6 = 19/30 |
| su propio SGM | `sgm`: 5 · 7 · 8 = **20/30** | `sgm_cost`: 5 · 5 · 6 = 16/30 |
| sin filtro (la regla del paper sobre las 3) | `snap`: 3 · 8 · 6 = 17/30 | |

- **Los dos efectos suman y pesan parecido.** Elegir el más suave en vez de usar el costo de la red suma 4–5
  llegadas: juntando las dos filas, 44/60 contra 35/60 ($p = 0.12$). Ver con la verdad en vez de con su SGM suma
  otras 4: juntando las dos columnas, 43/60 contra 36/60 ($p = 0.25$).
- **`sgm` es la mejor variante sin información privilegiada:** 20/30 contra 17/30 de la regla del paper, sin
  reentrenar.
- **Ninguna diferencia es significativa todavía.** Con 10 vuelos por velocidad, el intervalo de 95 % de cada
  total es ±0.17. Además, el mismo modo repetido varía en ±1–2 llegadas (`snap`: 17 y 16). Hace falta
  `--n 30` (90 vuelos por modo, ±0.10) para decidir.
 El cálculo cuesta unos 80 ms por cuadro en este
CPU (búsqueda por fuerza bruta; un k-d tree lo bajaría si hiciera falta en tiempo real).

## Cómo correrlo

```bash
pip install -e ".[flightmare]"                    # pyzmq (scipy ya está)

# el ejecutable de Unity de agile_flight (2022: escenas 0 almacén, 1 calle, 2 bosque, 3 páramo)
wget https://download.ifi.uzh.ch/rpg/Flightmare/RPG_Flightmare.zip
unzip RPG_Flightmare.zip -d ~/flightmare_unity && chmod +x ~/flightmare_unity/RPG_Flightmare.x86_64
~/flightmare_unity/RPG_Flightmare.x86_64 &        # se conecta a nosotros; dejarlo abierto

nanouav run r22 probe                             # → results/r22/probe.png, calib.json
# entrenamiento en Flightmare (almacén + troncos sembrados, VGA)
nanouav run r22 arena --scene warehouse
nanouav run r22 dagger --scene warehouse --init results/r18/student_spatial_in.keras --best --val-by-env
nanouav run r22 eval --arena --scene warehouse --pilots gbl,student --model results/r22/student_fm_warehouse.keras --v 3,5,7,10
nanouav run r22 video --arena --scene warehouse --pilot student --model results/r22/student_fm_warehouse.keras --v 5 --line 0

nanouav run r22 survey --size 300                 # mosaico cenital: dónde hay bosque en tierra firme
nanouav run r22 scan --origin 30,0,4 --size 80,40,10    # el mapa de la zona de vuelo (bosque de 2022)
# nanouav run r22 cloud ...                       # solo en escenas con colliders
nanouav run r22 lines --n 10                      # → lines_forest.json, lines_forest.png
nanouav run r22 eval --pilots gbl --v 3 --n 10    # control: con el mapa correcto, τ_gbl ≈ 100 %
nanouav run r22 eval --model results/r18/student_train.keras --pilots student --v 3,5,7 --n 10
nanouav run r22 video --model results/r18/student_train.keras --v 3 --line 0
```

El binario de la release de GitHub (2020, `RPG_Flightmare.tar.xz`) tiene otra numeración, con el
bosque en la escena 3: `--binary 2020`.

## Validación (sin Unity)

Con el reproductor falso:
- **El probe** (0.19.1) no avisa de nada y elige campo de visión vertical, sin espejo y profundidad × 100.
  Error frente a la nube: 8.6 % el SGM y 4.8 % la profundidad de Unity, con la escena desplazada
  (21, 217, 94) como el bosque real.
- **La nube** pone los troncos donde están.
- **τ_gbl** llega a la meta en las líneas encontradas en la nube.
- **El estudiante** vuela con el par estéreo que llega por el protocolo.

El primer probe real (0.19.0) mostró los tres fallos de la tabla de arriba. Falta repetirlo con 0.19.1.

## Qué esperar

- **El salto de dominio es grande.** El estudiante se entrenó con nuestras texturas y aquí verá
  texturas fotorrealistas. Lo que la red recibe es la profundidad SGM, no la imagen, y eso suaviza el
  salto, pero el bosque de Unity tiene ramas, hojas y troncos irregulares que nunca vio.
- **El paso siguiente es entrenar con Flightmare**, como el paper. DAgger con estas mismas piezas,
  con una sola instancia de Unity, en serie.
