# R20 · Seguir un globo a través de un bosque

*2026-10-03 · nanouav 0.16.0 · sobre R18 (política de Loquercio et al. 2021) y R19 (cadena de seguimiento).*

## Objetivo

Paso intermedio antes de seguir personas: el dron sigue un **globo de color** que flota entre los
árboles, a 4 m de distancia y a su misma altura, sin chocar y sin cambiarse a otro globo. A diferencia
de las personas de R19 (cilindro + esfera, que un detector real no reconoce), el globo se detecta
**sobre la imagen renderizada** con el mismo código que funcionaría con la cámara real: no hay detector
simulado.

## Qué cambia respecto a R19

| pieza | R19 (persona) | R20 (globo) |
|---|---|---|
| imagen | gris | color (RGB): el renderer da el color de cada superficie (`Texture.color`), `observe_full(..., rgb=True)`; la imagen gris y la profundidad SGM no cambian |
| detector | YOLO (real) / `SimDetector` (simulado) | `blob.BlobDetector`: umbral HSV alrededor del tono del globo, cierre morfológico y componentes conexas. Une los trozos que corta un tronco delgado (si juntos forman una caja cuadrada) y separa dos globos del mismo color que se tocan en la imagen (dos picos de la transformada de distancia con una cintura entre ellos). Mismo código en simulación y con cámara real |
| distancia | profundidad SGM en la caja; si no, altura de la persona | **tamaño aparente** con el diámetro conocido, medido en la altura de la caja: los troncos son verticales y estrechan la caja, pero no la acortan. SGM solo si la caja toca el borde: un globo no tiene textura y el SGM ve el fondo a través de él |
| filtro | (x, y) en el suelo | 3D (x, y, z) |
| punto de seguimiento | 4 m detrás, a 2 m de altura | 4 m detrás, a la altura del globo (limitada a 1–3 m) |
| tracker | ByteTrack | igual + **compensación del giro de la cámara** (`estimate.rotation_warp`: con la actitud de dos cuadros se predice cuánto se desplaza la caja en la imagen) y compuerta mínima de 12 px para cajas pequeñas |
| bloqueo | apariencia + compuerta 3D al rebloquear | además: si la pista bloqueada **salta** en 3D (> 0.8 m + 5 m/s·Δt) se la descarta, porque el tracker la cambió a otro globo; al rebloquear gana el más cercano a la predicción (dos globos del mismo color se ven iguales); la compuerta crece 1.5 m por segundo perdido |

El experto, DAgger y la política de R18 no cambian: el globo vuela rutas sin colisión (el mismo
planificador global, con 0.5 m de margen y las esquinas redondeadas), así que su rastro es una ruta libre
para el dron.

## El mundo

- **Bosque en color**: cortezas marrones y grises, suelo verde, ocre o arena, y de 3 a 9 arbustos.
  Algunos arbustos tienen colores cálidos poco saturados (hojas secas, flores) y algunas cortezas son
  rojizas: son falsos positivos realistas para un detector de color. Con saturación mínima 0.65
  desaparecen; con 0.55 aparecían en los troncos rojizos.
- **Globo objetivo**: esfera de 0.3 m de radio. Arranca a 1 m/s² hasta 1–4 m/s y recorre ~45 m. Sube y
  baja entre 1.2 y 2.6 m (dos senos de 8–20 m de longitud de onda).
- **De 0 a 2 globos más** que cruzan su camino. Con probabilidad 0.25 son del mismo color: solo el
  bloqueo 3D los distingue. Vuelan más alto (3.3–4 m): cruzan la imagen y pueden tapar al objetivo, pero
  no la ruta del dron. Los obstáculos móviles quedan fuera de R20: el experto solo conoce el mundo
  estático, y un globo sin textura casi no aparece en la profundidad SGM.
- Pasillo libre entre la salida del dron y la del globo.

## Cómo correrlo

```bash
nanouav run r20 frames --n 6                     # cuadros de muestra con las detecciones → results/r20/frames_XX.png
nanouav run r20 sim --pilots gbl,direct,student --model results/r18/student_train.keras --n 20
nanouav run r20 sim --pilots student --model results/r18/student_train.keras --n 20 \
        --distractors 2 --p-same 1.0 --tag same  # caso difícil: dos globos del mismo color
nanouav run r20 sim --pilots student --model results/r18/student_train.keras --n 20 --z 1.5,1.5 --tag z15
nanouav run r20 dagger --envs 100 --init results/r18/student_train.keras --workers 7 --epochs 6 --batch 32   # ξ = 1, 2, …, 5
nanouav run r20 sim --pilots student --model results/r18/student_balloon.keras --n 20 --tag balloon_ft
```

`--color` elige el globo (`red`, `orange`, `yellow`, `green`, `blue`, `magenta`), `--density` el bosque
y `--speed` la velocidad del globo. `r20 dagger` guarda sus datos en `results/r18/data_balloon` y
entrena junto con `data_train`, igual que `r19 dagger`.

Con un globo real y una cámara:

```bash
nanouav run r20 pick-color --source foto.jpg --box 300,200,360,260     # mide el tono con esa luz
nanouav run r20 video --source 0 --hsv 2.0,9.0,0.55,0.15 --diameter 0.30 --hfov 70 --out globo.mp4
```

`video` dibuja la caja del globo bloqueado y su posición 3D, calculada con el diámetro: adelante,
izquierda y arriba de la cámara.

## Validación local

**Detector y distancia** (`r20 frames`, cámara 320×240, 90°, por el tamaño aparente):

| distancia | 3 m | 4 m | 6 m | 10 m | 15 m |
|---|---|---|---|---|---|
| error de posición 3D | 0.08 m | 0.06 m | 0.10 m | 0.30 m | 1.1 m |

A 15 m el globo mide 6 px; un píxel de error en el diámetro ya es ~15 % de la distancia. Por eso el
filtro pesa menos las medidas lejanas (σ = 0.15 + 0.05·d).

A 320×240 el detector corre a ~760 fps en un núcleo de CPU (~250 fps a 640×480).

**Lazo cerrado** (`r20 sim`, cámara a 10 Hz, bosques de 0 a 1/25 m⁻², globo a 1–4 m/s):

| caso | piloto | éxito | choques | perdido | cuadros con bloqueo | en el globo correcto | error de distancia | rebloqueos |
|---|---|---|---|---|---|---|---|---|
| normal (8 semillas, 3000–3007) | `gbl` (rastro real) | 8/8 | 0 | 0 | 99 % | 100 % | 0.08 m | 0 |
| | `direct` (línea recta al punto estimado) | 8/8 | 0 | 0 | 83 % | 100 % | 0.84 m | 5 |
| 2 globos más, **del mismo color** (6 semillas, 3100–3105) | `gbl` | 6/6 | 0 | 0 | 99 % | 100 % | 0.10 m | 0.5 |
| | `direct` | 6/6 | 0 | 0 | 84 % | 100 % | 0.61 m | 5 |
| normal, 4 semillas | `student` (mini corrida de R18) | 1/4 | 3 | 0 | 98 % | 100 % | 1.50 m | 0 |
| altura fija 1.5 m (`--z 1.5,1.5`) | `student` (mini) | 3/4 | 1 | 0 | 99 % | 100 % | 0.75 m | 0 |

Valores medianos por episodio. Ningún cuadro bloqueado en un globo equivocado en los 28 episodios de
`gbl` y `direct`.

**El estudiante no sigue cambios de altura.** Los tres choques con altura variable son de altitud: el
globo sube y baja entre 1.2 y 2.6 m y el estudiante se queda cerca de 1.65 m. Cuando el globo baja,
el estudiante desciende tarde hasta 0.4 m y choca con un tronco. La causa está en los datos de R18: la
meta siempre está a 1.5 m y se vuela a esa altura, así que la componente vertical de ω vale casi cero en
todo el entrenamiento y la red no aprendió a responder a ella. Con altura fija la mini política funciona.
Por eso `r20 dagger` empieza con ξ = 1 y no con 6 como `r19 dagger`: así el experto, que sigue el rastro
con su altura, vuela más al principio. Es la primera prueba que hay que repetir con el estudiante de
850 entornos (`--z 1.5,1.5` y sin `--z`).

Problemas encontrados y corregidos en el camino:
1. **Sin compensación del giro**, el bloqueo estaba el 25–80 % del tiempo y había decenas de
   rebloqueos: al acelerar, el dron cabecea y la caja (pequeña) salta en la imagen más que su tamaño.
   Con la compensación sube a 92–99 %.
2. La ruta global une sus tramos con esquinas en los puntos intermedios. A 3–4 m/s el dron se salía del
   rastro en la esquina y chocaba con un árbol. Se redondean las esquinas (filtro gaussiano de 1.2 m,
   solo donde sigue habiendo margen).
3. **Dos globos del mismo color que se cruzan**:
   - el tracker les intercambiaba la identidad y el bloqueo seguía al otro globo; ahora se detecta el
     salto en 3D y se termina esa pista;
   - si los dos globos formaban una sola mancha, la distancia por tamaño salía mal y arrastraba al filtro;
     ahora la mancha se separa en dos y la distancia se toma de la altura de la caja;
   - una pista descartada por error ya no queda vetada para siempre: se termina y el mismo globo vuelve
     como pista nueva por la compuerta 3D.
4. El globo arrancaba de golpe a 4 m/s y el dron lo perdía de vista al cabecear para alcanzarlo (la
   cámara va fija al cuerpo). Ahora acelera a 1 m/s².
5. Los globos que cruzaban a la altura del dron lo chocaban: ni τ_gbl ni el experto conocen obstáculos
   móviles. Ahora vuelan por encima (ver «El mundo»).

## Qué esperar y qué falta

- **Estudiante de 850 entornos sin ajuste.**
  - Con altura fija debería acercarse a τ_gbl en bosques poco densos y perder algo en los densos
    (1/25 m⁻²).
  - Con altura variable fallará como la mini política (ver arriba).
  - `r20 dagger` (100 entornos, ~10 000 muestras) entrena las dos cosas que R18 no vio: mirar al
    objetivo en vez de hacia donde vuela, y cambiar de altura.
  - Tras el ajuste espero: ≥ 80 % de éxito con altura variable a 1–4 m/s en los bosques de 0–1/36 m⁻²,
    algo menos en 1/25 m⁻².
- **Obstáculos móviles**: fuera de R20, como los peatones en R19 (allí el dron vuela por encima). Para
  esquivarlos haría falta que el experto los conozca y que la profundidad los vea; un globo sin
  textura casi no aparece en el SGM.
- **Luz real**: el tono del globo cambia con el sol y la sombra, y el detector de color depende de los
  umbrales. `pick-color` los mide en una foto. Un globo de color poco común en el bosque (magenta,
  azul) da menos falsos positivos que el rojo o el naranja.
- **Hardware**: el detector de color corre a cientos de fps en CPU, así que el globo no necesita GPU.
  La política de R18 sí (Jetson Orin NX o similar).
- **Siguiente paso** (personas): cambiar `BlobDetector` por el YOLO de R19. El resto de la cadena (3D
  con profundidad, filtro, bloqueo, misión) ya es la de R19.
