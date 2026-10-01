# Entorno Gazebo v13

## Qué cambió respecto a v12
| Antes | Ahora |
|---|---|
| Un único mundo de 4 obstáculos con colores planos; suelo con tablero solo hasta 3.5 m | 6 escenarios generados con semilla (`scenarios.py`): obstaculos, bosque, rendijas, interior, corredor y dinamico |
| La apariencia de Gazebo no se parece a la del entrenamiento | Texturas en gris muestreadas con el **mismo** `Texture.sample` del renderer con el que se entrenó la CNN (ruido, franjas, tablero); suelo texturizado de 40 × 40 m |
| Mundo de Gazebo ≠ mundo offline | El mismo escenario JSON se carga en Gazebo (SDF) y en el simulador propio (`g1_gazebo_world.py --world scn:bosque`): comparación directa Gazebo ↔ offline |
| Sin métrica: había que mirar la ventana | Monitor de episodio en el nodo: holgura real (primitivas del JSON), colisiones, llegada, tiempo en reloj de simulación, recorrido, modos y RTF → `assets/gz_runs/*.json` |
| Render por software (`libEGL … dri2`), arranque de ~40–70 s | `gz_docker.sh` da al contenedor las bibliotecas gráficas del driver NVIDIA (`NVIDIA_DRIVER_CAPABILITIES=all`) + `diag` para comprobarlo |
| `sleep 5` fijo | Espera activa hasta que el mundo carga; reloj `/clock` puenteado; aviso si RTF < 0.8 |
| Solo obstáculos estáticos | Balón que cruza el carril (VelocityControl + odometría; `dyn_obstacles_node.py`) |
| Sensores ideales | Ruido gaussiano: cámara σ = 0.007 (≈ 2 niveles de gris), ToF σ = 1.5 cm |
| Una corrida a mano | `batch_eval.sh` sin ventanas + `summarize_gz.py` (IC95 de Wilson, columna offline) |

## Uso (desde la raíz del proyecto, en el host)
```bash
bash gazebo/gz_docker.sh diag                       # 1) comprobar GPU/EGL dentro del contenedor
bash gazebo/gz_docker.sh                            # obstaculos, semilla 0, MH
WORLD=bosque SEED=2 bash gazebo/gz_docker.sh
WORLD=bosque WORLD_KW="gap=1.0,density=0.5" PROFILE=nano bash gazebo/gz_docker.sh   # bosque denso para el nano
WORLD=rendijas WORLD_KW="width=1.2" bash gazebo/gz_docker.sh
WORLD=interior SEED=1 bash gazebo/gz_docker.sh
WORLD=dinamico bash gazebo/gz_docker.sh
WORLD=estrechos WORLD_KW="w=1.2" bash gazebo/gz_docker.sh   # pasos estrechos (X3: w ≥ 1.0)
WORLD=base bash gazebo/gz_docker.sh                 # el mundo original
DOCKER="sudo docker" ...                            # si docker pide sudo

# lotes sin ventanas → tabla
WORLDS=obstaculos,bosque,rendijas,interior SEEDS=0,1,2,3,4 PLANNERS=mh,lane bash gazebo/batch_eval.sh
python3 gazebo/summarize_gz.py

# el mismo escenario en el simulador propio (CPU, rápido)
python g1_gazebo_world.py --world scn:bosque --goals 0:0 --planner mh,lane --n 10
```
Generar mundos sin lanzar Gazebo: `python3 gazebo/gen_world.py --all --seeds 0,1,2` (se regeneran si falta el SDF,
o con `REGEN=1`). Los SDF quedan en `gazebo/worlds/gen/`, las texturas en `gazebo/models/nano_textures/`.

## Escenarios (`gazebo/scenarios.py`)
| nombre | qué prueba | parámetros |
|---|---|---|
| obstaculos | la "pared" de 3 cajas del mundo original (el X3 no cabe por los huecos) | — |
| bosque | troncos r 0.06–0.25 m por disco de Poisson; B a 11.5 m | gap (separación mín., 1.6), density |
| rendijas | slalom de paredes con una abertura cada una, alternando lado | width (1.5), n_walls (2) |
| interior | dos salas unidas por una puerta, muebles; B en la otra sala | door (1.2) |
| corredor | el mismo generador aleatorio de E3 | — |
| dinamico | balón r 0.15 m que cruza el carril en x = 3 m | v_ball (1.0) |
| estrechos | puerta + pasillo de 2.5 m + puerta desplazada en paredes de 32 m (sin rodeo) | w (ancho libre, 1.0) |

## Validación offline de los escenarios (X3, MH, CNN M, n = 3 por escenario, semillas 0–2)
Todos los escenarios estáticos se pueden resolver: obstaculos 3/3, bosque 3/3, rendijas 3/3,
interior 3/3 y corredor 3/3. dinamico da 1/3 (dos golpes del balón): el MH aún no integra la capa
RF_DYN. En bosque, el recorrido es entre 1.7 y 3.1 veces la distancia en línea recta, porque el
planificador duda entre huecos y aparecen fantasmas de la CNN. Los mundos nuevos exponen así dos puntos
débiles del planificador.
