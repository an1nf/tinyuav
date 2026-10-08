# R22 · El estudiante de R18 en Flightmare

*2026-10-07 · nanouav 0.19.0 (convenciones de Unity corregidas en 0.19.1; mapa por escaneo en 0.19.2) · sobre R18.*

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

## Cómo correrlo

```bash
pip install -e ".[flightmare]"                    # pyzmq (scipy ya está)

# el ejecutable de Unity de agile_flight (2022: escenas 0 almacén, 1 calle, 2 bosque, 3 páramo)
wget https://download.ifi.uzh.ch/rpg/Flightmare/RPG_Flightmare.zip
unzip RPG_Flightmare.zip -d ~/flightmare_unity && chmod +x ~/flightmare_unity/RPG_Flightmare.x86_64
~/flightmare_unity/RPG_Flightmare.x86_64 &        # se conecta a nosotros; dejarlo abierto

nanouav run r22 probe                             # → results/r22/probe.png, calib.json
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
