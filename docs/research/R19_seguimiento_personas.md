# R19 · Detectar a una persona y seguirla, evitando obstáculos

*2026-10-03 · nanouav 0.15.0 · sobre R18 (recreación de Loquercio et al. 2021).*

## Objetivo

Que el dron identifique a **una** persona (la que el operador elige), la siga a una distancia fija en
espacios abiertos o con árboles, y no se cambie a otra persona que se cruce. La navegación entre
obstáculos la hace la política de R18; este informe añade lo que le dice hacia dónde volar.

## Cadena (cada cuadro de cámara)

```
imagen izquierda ─► detector de personas ─► tracker (ByteTrack) ─► bloqueo del objetivo (apariencia + compuerta 3D)
                                                                      │ caja del objetivo
profundidad SGM ───────────────────────────────────────────────────► posición 3D (cuartil bajo de la profundidad
                                                                      │ dentro de la caja; si no hay, altura de la caja)
                                                          filtro de Kalman (x, y, vx, vy) en el mundo
                                                                      │
             punto de seguimiento = objetivo + 4 m hacia el dron, a 2 m de altura
                                                                      │
       ω (dirección deseada) + v_des = v_objetivo + 0.8·(distancia − 4 m)  ──►  política R18 (evita obstáculos)
       yaw → hacia la persona (que no salga de la imagen)
```

| módulo (`nanouav.follow`) | qué hace |
|---|---|
| `detector.YoloDetector` | YOLO (ultralytics) restringido a personas: COCO (clase 0) o el modelo ajustado con VisDrone |
| `detector.SimDetector` | en el simulador las personas son cilindro + esfera y un detector real no las reconoce: proyecta la persona real y la degrada como un detector (recall logístico en la altura de la caja, oclusión con la profundidad, *jitter*, falsos positivos); su curva se calibra con `eval-det` |
| `tracker.Tracker` | ByteTrack: Kalman por caja, asociación en dos etapas (confiables y de baja confianza), IoU + distancia de centros (la imagen salta cuando el dron gira) + apariencia, húngaro |
| `tracker.TargetLock` | sigue a *esa* persona: si su pista se pierde, predice dónde está y solo vuelve a bloquear una pista que se le parezca (histogramas de torso y piernas) **y** esté donde puede estar el objetivo |
| `estimate` | geometría de la cámara, caja + profundidad → posición en el suelo, filtro del objetivo |
| `people` | personas que caminan por rutas libres de colisión (el mismo planificador global), con ropa distinta; entornos con 0–2 personas que se cruzan |
| `mission.FollowMission` | misión para el lazo cerrado de R18: ω, velocidad y yaw para el estudiante; referencia privilegiada (el rastro de la persona) para el experto y DAgger; métricas |
| `visdrone` | VisDrone2019 → YOLO, una clase *persona* (peatón + gente), regiones ignoradas pintadas |

El experto y DAgger usan como referencia el **rastro** de la persona elevado a 2 m: las personas caminan
por donde no hay obstáculos, así que su rastro es una ruta libre para el dron. Avanza desde el punto
más cercano del dron a la velocidad que cierra la distancia hasta el punto de seguimiento, sin pasarlo.

## Datos

| dato | para qué | cómo se consigue |
|---|---|---|
| COCO (clase persona) | vistas a ras de suelo: es lo que ve un dron a 2 m de altura y 4 m detrás de alguien | ya viene en los pesos de YOLO11 (`yolo11n.pt`, `yolo11s.pt`); para reentrenar, `coco.yaml` de ultralytics (descarga desde images.cocodataset.org, **en tu PC**) |
| VisDrone2019-DET | vistas aéreas, personas pequeñas; el dron sube o se aleja | `nanouav run r19 prep-visdrone --fetch` (releases de ultralytics en GitHub) |
| (opcional) vídeos propios | validar el tracker y el bloqueo con tu cámara | `nanouav run r19 video --source clip.mp4` |

**Medida en este equipo:** YOLO11n de COCO, sin ajustar, sobre 120 imágenes de VisDrone val a 640 px,
tiene este recall (IoU 0.5) según la altura de la persona en la imagen original:

| altura (px) | 8–12 | 12–16 | 16–24 | 24–32 | 32–48 | 48–96 | ≥ 96 |
|---|---|---|---|---|---|---|---|
| recall | 0.00 | 0.01 | 0.08 | 0.12 | 0.24 | 0.50 | 0.92 |

2.7 falsos positivos por imagen. Para seguir a alguien a 4–8 m con una cámara de 640×480, la persona mide
~60–150 px y el modelo de COCO ya sirve. Para alturas mayores o distancias largas hay que ajustar con
VisDrone a 960–1280 px. El ajuste conserva la cabeza de COCO: las etiquetas son la clase 0, que ya es
*persona*. En una época de prueba sobre 27 imágenes, mAP50 0.031 conservándola frente a 0.0001 con una
cabeza nueva de una clase.

## Cómo correrlo (PC de entrenamiento)

```bash
pip install -e ".[follow]"                                    # ultralytics + PyTorch
nanouav run r19 prep-visdrone --raw data/visdrone --out data/person --fetch
nanouav run r19 train-det --data data/person/person.yaml --model yolo11s.pt --imgsz 960 --epochs 100 --batch 16
nanouav run r19 eval-det --model results/r19/detect/person/weights/best.pt --data data/person   # → det_calib_best.json
nanouav run r19 video --source clip.mp4 --model results/r19/detect/person/weights/best.pt --out tracked.mp4
nanouav run r19 sim --pilots gbl,direct,student --model results/r18/student_train.keras --n 20 \
        --det-calib results/r19/det_calib_best.json
nanouav run r19 dagger --envs 100 --init results/r18/student_train.keras --workers 7   # ajuste fino (datos de R18 + seguimiento)
nanouav run r19 sim --pilots student --model results/r18/student_follow.keras --n 20 --tag follow_ft
```

Pilotos de `r19 sim`: `gbl` (sigue el rastro real, cota superior de navegación), `expert` (M-H con el
rastro), `direct` (vuela en línea recta hacia el punto estimado por el tracker, sin evitar
obstáculos; es la línea base) y `student` (la política R18 con ω del tracker).

Métricas: éxito (la persona termina su recorrido, sin choque, con el bloqueo en la persona correcta
≥ 80 % del tiempo), fracción de cuadros con bloqueo, fracción de esos en la persona correcta, error
mediano de distancia, rebloqueos y tiempo máximo perdido.

## Validación local

**Tracker y bloqueo en vídeo** (`r19 video` sobre un paneo de `bus.jpg`, cuatro personas, YOLO11n
COCO en CPU): 100 % de cuadros con la persona elegida, 0 rebloqueos, 6.6 fps en CPU.

**Lazo cerrado** (rastro real, `gbl`; bosques de 1/49–1/25 m⁻² o campo abierto, 0–2 personas
cruzándose, la persona camina a 1–3 m/s):

| semilla | resultado | cuadros con bloqueo | en la persona correcta | error de distancia mediano | rebloqueos |
|---|---|---|---|---|---|
| 1 | éxito | 89 % | 99.7 % | 0.07 m | 10 |
| 3 | éxito | 85 % | 100 % | 0.08 m | 9 |

`r19 sim`, 4 semillas por piloto (2000–2003), detector simulado por defecto:

| piloto | éxito | choques | bloqueo | correcta | error de distancia | rebloqueos |
|---|---|---|---|---|---|---|
| `gbl` (rastro real) | 4/4 | 0 | 82 % | 100 % | 0.11 m | 10 |
| `direct` (línea recta al punto estimado) | 4/4 | 0 | 50 % | 100 % | 0.86 m | 22 |
| `student` (política de la mini corrida de R18) | 4/4 | 0 | 85 % | 100 % | 0.74 m | 9 |

Son bosques poco densos y solo 4 episodios: dice que la cadena completa funciona (detector → tracker →
bloqueo → 3D → ω → política), no cuánto mejor es el estudiante. La cifra que importa sale con el
estudiante de 850 entornos, el detector calibrado y bosques más densos (`--density 0.04`).
`direct` pierde el bloqueo la mitad del tiempo porque no gira con suavidad: su yaw salta con cada
estimación.

DAgger de seguimiento probado de punta a punta (2 entornos, 55 muestras, el estudiante vuela el 100 %
de los cuadros con ξ = 6).

Durante esta validación salieron dos errores del lazo de R18 que solo aparecen a baja velocidad
(corregidos):
1. La referencia del seguimiento se recortaba al punto de seguimiento *actual*; con el horizonte de 1 s,
   la escala de tiempo β = v_des/v_μ se disparaba (×4) y el dron oscilaba de 1 a 6 m/s. Ahora se recorta
   a dónde estará ese punto.
2. La aceleración inicial de cada polinomio es la de referencia del plan en curso, no la medida (que
   cierra un lazo a través de la vibración); `Rollout.A0_MODE` permite comparar.

## Qué esperar y qué falta

- Con el detector real calibrado (`--det-calib`) el simulador da una cifra realista de pérdidas de
  bloqueo por distancia y oclusión. Con el detector por defecto (optimista para personas pequeñas) las
  pérdidas salen sobre todo de oclusiones por troncos.
- La apariencia es un histograma: distingue ropa distinta, no a dos personas vestidas igual. El reemplazo
  natural es un *embedding* de re-identificación (OSNet o las características ReID de BoT-SORT) en
  `tracker.appearance()`.
- La política de R18 se entrenó mirando hacia donde vuela; al seguir, el yaw apunta a la persona. Casi
  siempre coinciden (el punto de seguimiento está en la línea dron → persona), pero en giros bruscos la
  cámara deja de mirar a donde va el dron. DAgger de seguimiento (`r19 dagger`) entrena con esa
  situación.
- En el dron real: Jetson Orin NX, D435i/D455 (RGB para el detector y estéreo para la profundidad),
  YOLO exportado a TensorRT (`train-det --export engine` en la propia Jetson).
