# R12 · Planificador MH (Loquercio et al. 2021) — del bloqueo en Gazebo a 16/16

## Síntoma (Gazebo, X3, modo consistente)
Despega en A (−1.5, 0), avanza y se queda quieto en x ≈ 0.4–0.5, modo AVOID y v = 0, indefinidamente.
Con B = (4, 1.5) llega a 0.9 m de la meta y se queda ahí.

## Diagnóstico (réplica del mundo en el simulador propio: `g1_gazebo_world.py`)
- Se reproduce igual: se detiene en x ≈ 0.44 con d_B ≈ 4.1 m (Gazebo: 3.99–4.12). Con profundidad
  **perfecta** (`UAV_FAKE_CNN=1`) también se bloquea → **no es la CNN, es la planificación**.
- Las tres cajas forman una pared para el X3 (r = 0.36 m): huecos de 0.75 m y 0.63 m < 2r + márgenes.
  Hay que rodear por fuera (|y| > 1.8 m).
- El planificador de carril tiene tres fallas estructurales:
  1. solo busca en ±2.3 m de la recta A→B con pendiente ≤ tan 28°, así que no existe plan que rodee la pared;
  2. el movimiento lateral depende de la velocidad de avance (v_lat = v_fwd·pendiente + K·error), y con
     v_fwd = 0 no hay desplazamiento lateral;
  3. el limitador de FOV multiplica la velocidad por k = 0 cuando se mueve a 90° del rumbo
     (desplazarse de lado anula el comando). Con B = (4, 1.5) es lo mismo: al pasar la perpendicular por B
     v_fwd = 0 y el error lateral de 0.9 m ya no se corrige.
- La regla "consistente" de R5 (Loquercio) sí estaba activa (`consistent=True`), pero era un parche sobre
  el planificador de carril: acotaba pendientes y latencias. No incorporaba la arquitectura del paper.

## Qué se tomó ahora del paper (`_04_policy/mh_planner.py`)
| Loquercio 2021 | MHPlanner |
|---|---|
| Política condicionada a una **trayectoria de referencia** global | Campo de costo-a-meta (Dijkstra vectorizado en la grilla local); desconocido = transitable con costo (optimista) |
| Experto privilegiado: **muestreo MH de muchas trayectorias** libres de colisión; red **multimodal** que predice varias con costo | Abanico de 180 hipótesis (36 rumbos × 5 curvaturas, 2.5 s), costo = costo-a-meta del extremo alcanzable + holgura + giro + histéresis; se elige la de menor costo en cada frame (receding horizon) |
| Rumbo alineado con la dirección de vuelo (la cámara mira hacia donde se va) | yaw = dirección de la hipótesis elegida; se avanza solo lo que la cámara ya ve (primero girar, después avanzar) |
| Velocidad máxima limitada por el alcance de percepción | v·τ + v²/2a ≤ tramo del arco **libre y observado** |
Además: aproximación final holonómica a B (corrige el error lateral), SCAN (girar para observar) si no hay
ninguna hipótesis factible, y el mismo filtro de barrera y freno de emergencia que el planificador de carril.

**Hallazgo lateral:** la CNN deja fantasmas a 2–4 m (fuera del alcance del ToF; con giros de exploración
se acumulan más de 1 500 celdas). Si el mapa es restricción dura en la referencia global, el dron queda
encerrado (costo = ∞). La solución es separar confianza por alcance: **costo alto pero finito en la
referencia global** y **restricción dura solo en los arcos (≤ 1.25 m)**, donde CNN y ToF son fiables. A
medida que el dron se acerca, los fantasmas se vuelven a observar y se borran.

## Resultados (X3, v = 0.5 m/s, mundo de Gazebo con ±0.15 m de jitter por obstáculo, n = 8 por meta)
| Planificador | Profundidad | B = (4.5, 0) | B = (4, 1.5) |
|---|---|---|---|
| carril (consistente) | CNN M | **0/8** (6 bloqueos, 2 colisiones) | 7/8 |
| MH | CNN M | **8/8** (t 19–42 s) | **8/8** |
| MH | GT (perfecta) | 8/8 (t ≈ 16–22 s) | 8/8 (t ≈ 14–15 s) |

Regresión en corredores aleatorios de E3 (meta x = 7 m, n = 12): ver `logs/g1c_*.log` y
`assets/g1_corridor_*.json`.

## Uso
- Gazebo: `PLANNER=mh` (por defecto) o `PLANNER=lane` (anterior), variables en `gazebo/run_sim.sh`.
- Offline: `python g1_gazebo_world.py --planner lane,mh --goals 4.5:0,4:1.5 --n 8 --jitter 0.15`
  y `--world corridor --goals 7:0`.
- Costo: unos 25 ms por actualización en CPU (Python/numpy), dentro del presupuesto de 10 Hz del nodo.
  Para el nano-UAV real haría falta portarlo a C (el Dijkstra y los 180 arcos son vectorizables).

## Pendiente / siguiente paso natural
- El paper aprende la política (imitación del experto MH). Aquí el MH es el experto ejecutado en línea:
  es el "profesor". El paso de contribución es destilarlo en la CNN (salida multimodal de trayectorias
  desde la profundidad 12×12 latente de M_ToF), como hace el paper, para quitar el Dijkstra del MCU.
