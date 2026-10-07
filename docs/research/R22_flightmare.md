# R22 · El estudiante de R18 en Flightmare

*2026-10-07 · nanouav 0.19.0 · sobre R18.*

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
| `nanouav run r22 probe` | comprueba las convenciones que no están documentadas, comparando la nube con la profundidad SGM y la de Unity en un mismo cuadro: los ejes y el origen del `.ply`, si el campo de visión de Unity es vertical u horizontal, la escala de la profundidad y la orientación del eje y. Escribe `calib.json` y `probe.png` |
| `nanouav run r22 fake` | un reproductor de Unity falso: nuestro renderizador detrás del mismo protocolo, para probar todo sin Unity (lo usan los tests) |

## Cómo correrlo

```bash
pip install -e ".[flightmare]"                    # pyzmq (scipy ya está)

# el ejecutable de Unity de agile_flight (2022: escenas 0 almacén, 1 calle, 2 bosque, 3 páramo)
wget https://download.ifi.uzh.ch/rpg/Flightmare/RPG_Flightmare.zip
unzip RPG_Flightmare.zip -d ~/flightmare_unity && chmod +x ~/flightmare_unity/RPG_Flightmare.x86_64
~/flightmare_unity/RPG_Flightmare.x86_64 &        # se conecta a nosotros; dejarlo abierto

nanouav run r22 probe                             # → results/r22/probe.png, calib.json
nanouav run r22 cloud --origin 30,0,4 --size 80,30,10   # el mapa de la zona de vuelo
nanouav run r22 lines --n 10                      # → lines_forest.json, lines_forest.png
nanouav run r22 eval --pilots gbl --v 3 --n 10    # control: con el mapa correcto, τ_gbl ≈ 100 %
nanouav run r22 eval --model results/r18/student_train.keras --pilots student --v 3,5,7 --n 10
nanouav run r22 video --model results/r18/student_train.keras --v 3 --line 0
```

El binario de la release de GitHub (2020, `RPG_Flightmare.tar.xz`) tiene otra numeración, con el
bosque en la escena 3: `--binary 2020`.

## Validación (sin Unity)

Con el reproductor falso:
- **El probe** identifica la convención correcta: nube en ejes de Unity, campo de visión vertical y
  profundidad en metros. Error SGM frente a la nube: 6 %; profundidad de Unity frente a la nube: 0.9 %.
- **La nube** pone los troncos donde están.
- **τ_gbl** llega a la meta en las líneas encontradas en la nube.
- **El estudiante** vuela con el par estéreo que llega por el protocolo.

No se ha probado con el Unity real desde este entorno. `probe` existe justamente para comprobar
allí lo que aquí se supuso.

## Qué esperar

- **El salto de dominio es grande.** El estudiante se entrenó con nuestras texturas y aquí verá
  texturas fotorrealistas. Lo que la red recibe es la profundidad SGM, no la imagen, y eso suaviza el
  salto, pero el bosque de Unity tiene ramas, hojas y troncos irregulares que nunca vio.
- **El paso siguiente es entrenar con Flightmare**, como el paper. DAgger con estas mismas piezas,
  con una sola instancia de Unity, en serie.
