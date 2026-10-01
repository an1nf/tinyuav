# Gazebo (v13)

Gazebo Harmonic + ROS 2 Jazzy dentro de Docker. El repositorio se monta en `/sim/nano_uav`, así que
los modelos (`models/`) y los resultados (`results/gz_runs/`) son los mismos del host.

## Imágenes

```bash
docker build -f docker/Dockerfile.base -t nano_uav_sim:fixed .   # ROS 2 Jazzy + Gazebo Harmonic + ros_gz (una vez)
docker build -f docker/Dockerfile.sim  -t nano_uav_sim:v12 .     # el proyecto encima (PYTHONPATH=/sim/nano_uav/src)
```

Como el repositorio se monta en tiempo de ejecución, los cambios de código no requieren
reconstruir `nano_uav_sim:v12`.

## Uso (desde la raíz del repositorio, en el host)

```bash
bash gazebo/gz_docker.sh diag                       # 1) comprobar GPU/EGL dentro del contenedor
bash gazebo/gz_docker.sh                            # obstaculos, semilla 0, MH
WORLD=bosque SEED=2 bash gazebo/gz_docker.sh
WORLD=bosque WORLD_KW="gap=1.0,density=0.5" PROFILE=nano bash gazebo/gz_docker.sh   # bosque denso para el nano
WORLD=rendijas WORLD_KW="width=1.2" bash gazebo/gz_docker.sh
WORLD=interior SEED=1 bash gazebo/gz_docker.sh
WORLD=dinamico bash gazebo/gz_docker.sh
WORLD=estrechos WORLD_KW="w=1.2" bash gazebo/gz_docker.sh   # pasos estrechos (X3: w ≥ 1.0)
WORLD=base bash gazebo/gz_docker.sh                 # el mundo original (gazebo/worlds/nano_uav_world.sdf)
VARIANT=M_ToF NANOUAV_FUSE2=1 bash gazebo/gz_docker.sh       # otra CNN / fusión bidireccional
DOCKER="sudo docker" bash gazebo/gz_docker.sh       # si docker pide sudo
bash gazebo/gz_docker.sh bash                       # shell dentro del contenedor
```

Variables de `run_sim.sh` (todas opcionales): `WORLD`, `SEED`, `WORLD_KW`, `REGEN=1`,
`VARIANT=M|M_ToF`, `PROFILE=x3|nano`, `PLANNER=mh|lane`, `CONSISTENT`, `RF_DYN`, `V_CRUISE`,
`GOAL_X/GOAL_Y` (sustituyen a B), `LOOP=true`, `T_MAX`, `HEADLESS=true`, `EXIT_ON_DONE=true`,
`RUN_TAG`, `CONTROL_MODE=velocity|motors`, `YAW_SIGN=-1`, y las banderas `NANOUAV_*`.

### Lotes sin ventanas

```bash
WORLDS=obstaculos,bosque,rendijas,interior SEEDS=0,1,2,3,4 PLANNERS=mh,lane bash gazebo/batch_eval.sh
PAR=2 ...                                           # contenedores en paralelo (~1–2 núcleos + GPU cada uno)
nanouav summarize --gazebo                          # tabla con IC 95 % de Wilson + columna offline
```

Cada episodio deja `results/gz_runs/<mundo>_s<semilla>_<planificador>_<variante>_<tag>.json` y su log
en `results/gz_logs/`.

### El mismo escenario offline (CPU, mucho más rápido)

```bash
nanouav run r12 --world scn:bosque --goals 0:0 --planner mh,lane --n 10
nanouav run r12 --world json:gazebo/worlds/gen/bosque_s2.json --goals 0:0 --planner mh   # el JSON exacto de Gazebo
nanouav suite configs/suites/scenarios.yaml -j 7                                         # todos, comparable con batch_eval
```

### Generar mundos sin lanzar Gazebo

```bash
nanouav gazebo gen-world --scenario bosque --seed 2
nanouav gazebo gen-world --all --seeds 0,1,2
```

Los SDF y JSON quedan en `gazebo/worlds/gen/` y las texturas en `gazebo/models/nano_textures/`
(ambos fuera de git; `run_sim.sh` los regenera si faltan o con `REGEN=1`).

## Qué cambió respecto a v12

| Antes | Ahora |
|---|---|
| Un único mundo de 4 obstáculos con colores planos; suelo con tablero solo hasta 3.5 m | 7 escenarios generados con semilla (`nanouav.sim.scenarios`) |
| La apariencia de Gazebo no se parece a la del entrenamiento | Texturas en gris muestreadas con el **mismo** `Texture.sample` del renderer con el que se entrenó la CNN; suelo texturizado de 40 × 40 m |
| Mundo de Gazebo ≠ mundo offline | El mismo escenario JSON se carga en Gazebo (SDF) y en el simulador propio (`nanouav run r12 --world json:…`) |
| Sin métrica: había que mirar la ventana | Monitor de episodio en el nodo: holgura real (primitivas del JSON), colisiones, llegada, tiempo de simulación, recorrido, modos y RTF → `results/gz_runs/*.json` |
| Render por software (`libEGL … dri2`), arranque de ~40–70 s | `gz_docker.sh` da al contenedor las bibliotecas gráficas del driver NVIDIA (`NVIDIA_DRIVER_CAPABILITIES=all`) + `diag` para comprobarlo |
| `sleep 5` fijo | Espera activa hasta que el mundo carga; reloj `/clock` puenteado; aviso si RTF < 0.8 |
| Solo obstáculos estáticos | Balón que cruza el carril (VelocityControl + odometría; `gazebo/nodes/dyn_obstacles_node.py`) |
| Sensores ideales | Ruido gaussiano: cámara σ = 0.007 (≈ 2 niveles de gris), ToF σ = 1.5 cm |
| Una corrida a mano | `batch_eval.sh` sin ventanas + `nanouav summarize --gazebo` |

## Escenarios (`src/nanouav/sim/scenarios.py`)

| nombre | qué prueba | parámetros (`WORLD_KW`) |
|---|---|---|
| obstaculos | la "pared" de 3 cajas del mundo original (el X3 no cabe por los huecos) | — |
| bosque | troncos r 0.06–0.25 m por disco de Poisson; B a 11.5 m | `gap` (separación mín., 1.6), `density` |
| rendijas | slalom de paredes con una abertura cada una, alternando lado | `width` (1.5), `n_walls` (2) |
| interior | dos salas unidas por una puerta, muebles; B en la otra sala | `door` (1.2) |
| corredor | el mismo generador aleatorio de E3 | — |
| dinamico | balón r 0.15 m que cruza el carril en x = 3 m | `v_ball` (1.0) |
| estrechos | puerta + pasillo de 2.5 m + puerta desplazada en paredes de 32 m (sin rodeo) | `w` (ancho libre, 1.0) |

## Nodos (`gazebo/nodes/`)

| nodo | función |
|---|---|
| `perception_node.py` | cámara + ToF → CNN → fusión → planificador → referencias suaves → Twist (modo `velocity`) o setpoints (modo `motors`); misión A→B; monitor de episodio; HUD |
| `dyn_obstacles_node.py` | mueve los obstáculos dinámicos del JSON (onda triangular, igual que offline) |
| `flight_controller_node.py` | modo `motors`: FlightStack propio hasta velocidades de motor (experimental, no probado en Gazebo) |

Los nodos importan el paquete desde `$NANOUAV_HOME/src`, sin instalarlo. `tests/test_gazebo_nodes.py`
los ejercita con *stubs* de rclpy y de los mensajes, de modo que un cambio en el paquete que rompa
los nodos se detecta sin ROS. `batch_eval.sh` extrae el resultado de cada episodio de la línea
`=== EPISODIO: …` del log; ese texto está cubierto por un test.

## Validación offline de los escenarios (X3, MH, CNN M, n = 3 por escenario, semillas 0–2)

Todos los escenarios estáticos se pueden resolver: obstaculos 3/3, bosque 3/3, rendijas 3/3,
interior 3/3 y corredor 3/3. dinamico da 1/3 (dos golpes del balón): el MH aún no integra la capa
RF_DYN. En bosque, el recorrido es entre 1.7 y 3.1 veces la distancia en línea recta, porque el
planificador duda entre huecos y aparecen fantasmas de la CNN. Los pasos estrechos se analizan en
`docs/research/R13_pasos_estrechos.md`.

## Verificación pendiente en Gazebo

Del plan original (sigue vigente): comprobar `ros2 topic hz /model/nano_uav/odometry` ≈ 250 Hz; si el
dron se desvía al girar en modo `velocity`, probar `twist_frame:=world`; en modo `motors`, primero en
el suelo y con `YAW_SIGN=-1` si el yaw diverge; comparar la profundidad CNN del HUD con
`/nano_uav/depth/image` para medir la brecha de dominio renderer → ogre2.
