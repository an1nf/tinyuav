# R17 · Velocidad de crucero y troncos finos fuera del campo visual

*2026-10-01 · nanouav 0.13.3 · X3, planificador MH, simulador propio (CPU) y Gazebo.*

## Pregunta

En Gazebo, `bosque` semilla 2 con M_ToF, el dron llega sin chocar, pero tarda 40.5 s en recorrer
11.5 m: 0.28 m/s de media, con `V_CRUISE` = 0.5. ¿Cómo hacerlo más rápido sin perder seguridad?

## 1 · Qué dice el log de Gazebo

- **Episodio 1** (el único válido): 40.5 s, recorrido de 12.8 m, holgura mínima de 0.62 m. Se
  pierde tiempo en el despegue (~2 s), en unos 10 s entre x = 3.4 y 5.0 m pasando entre troncos a
  0.09–0.19 m/s (holgura de 0.21–0.30 m), y en un giro sobre el sitio (TURN, t = 28.8 s). El resto
  del tiempo va a 0.35–0.46 m/s.
- **Los "RTF negativos" eran reinicios del mundo** hechos desde la interfaz de Gazebo: `/clock`
  vuelve a 0, la simulación queda en pausa (RTF 0.00 durante 15 s) y el dron reaparece en A. El nodo
  seguía con el planificador del episodio anterior: el contador de estancamiento venía de 35 s y
  llegó a 350 s, y el margen quedó relajado a 0.03 m todo el tiempo. Los episodios 2 y 3 (53.8 s y
  45.5 s) se midieron con un planificador contaminado. **Corregido:** un reinicio del mundo empieza un
  episodio nuevo con un planificador nuevo, y cada episodio se guarda por separado (`*_epN.json`).
- Con `V_CRUISE` > 0.8, el nodo usaba los límites de referencia normales (aceleración 1.2 m/s²,
  giro 0.8 rad/s), mientras que offline se usan los `fast` (2.2 m/s², 1.2 rad/s). **Corregido:**
  Gazebo y `r12` comparten `nanouav.planning.presets`.

## 2 · Barrido de velocidad offline: aparece un fallo de seguridad

Primero se midió con profundidad perfecta (GT + ruido de 8 cm), que debería ser la cota superior.

```bash
NANOUAV_FAKE_CNN=1 nanouav run r12 --world json:gazebo/worlds/gen/bosque_s2.json --goals 0:0 --planner mh --n 8 --v 1.0
NANOUAV_FAKE_CNN=1 nanouav run r12 --world scn:bosque --goals 0:0 --planner mh --n 8 --v 1.0      # 8 bosques distintos
```

| mundo | v (m/s) | mapa | éxito | choques | t mediano (s) |
|---|---|---|---|---|---|
| bosques aleatorios (gap 1.6) | 0.5 | actual | 8/8 | 0 | 28.8 |
| bosques aleatorios | 1.0 | actual | 7/8 | 1 | 24.0 |
| **bosque_s2** (el de Gazebo) | 0.5 | actual | **4/8** | 4 | 33.7 |
| **bosque_s2** | 1.0 | actual | **3/8** | 5 | 27.0 |

Con profundidad perfecta el dron choca contra un tronco fino (r = 0.097 m en (2.13, −0.48)) en la
mitad de los episodios. El registro cuadro a cuadro muestra dos mecanismos que se suman:

1. **El filtro de píxeles voladores borra los troncos finos.** `flying_pixels` descarta todo píxel a
   menos de 2 px de un salto de profundidad (ventana de 5×5). Un tronco de 0.2 m mide menos de 5–6 px
   a más de 2.5–3 m, así que desaparece entero; además solo se usa 1 de cada 3 píxeles. A 1 m/s el
   tronco no entró nunca en el mapa (evidencia 0 en todo el episodio).
2. **Lo que sale del campo visual se olvida y pasa a "libre conocido".** A 0.5 m/s el tronco se vio
   una vez a 3.56 m. Los rayos libres que pasan rozándolo lo dejan en g ≈ 2 (2·0.5 + 1), apenas por
   encima del umbral de 1.5. Después quedó a −33…−46° del eje de la cámara, fuera del campo de ±30°, y
   el decaimiento global (×0.995 por cuadro) lo bajó del umbral en ~6 s. Las celdas seguían marcadas
   como vistas, así que el planificador curvó hacia el tronco y lo barrió con el costado.

La geometría hace que esto no sea mala suerte. Con una cámara de 60°, un obstáculo al lado del
cuerpo (separación lateral < r + r_obs ≈ 0.46 m) solo es visible a más de 0.46 / tan 30° ≈ 0.8 m por
delante. Más cerca, lo único que protege al dron es la memoria del mapa.

**Cambios** (atributos del MH, **apagados por defecto**, `--cfg` / `PLANNER_CFG`; "r17" = los tres):

| opción | qué hace |
|---|---|
| `FLY_KEEP_NEAR=1` | el filtro conserva el lado cercano del salto: el tronco se queda, el halo del fondo y las profundidades intermedias de la CNN se siguen quitando |
| `MAP_DECAY_FOV=1` | solo decaen las celdas dentro del sector de la cámara: sin medida no hay actualización |
| `MAP_HIT_WINS=1` | en un mismo cuadro, un acierto no se borra por los rayos libres que pasan rozando la celda |

| mundo | v | mapa | éxito | choques | t mediano (s) |
|---|---|---|---|---|---|
| bosques aleatorios | 0.5 | r17 | 8/8 | 0 | 31.9 |
| bosques aleatorios | 1.0 | r17 | 8/8 | 0 | 26.3 |
| bosque_s2 | 0.5 | r17 | **8/8** | 0 | 32.9 |
| bosque_s2 | 1.0 | r17 | **8/8** | 0 | 33.8 |
| bosque_s2 | 1.5 | r17 | 8/8 | 0 | 34.3 |
| bosque_s2 | 1.0 | solo `FLY_KEEP_NEAR` | 8/8 | 0 | 28.3 |
| bosque_s2 | 0.5 | solo `FLY_KEEP_NEAR` | 7/8 | 1 | 31.6 |
| bosque_s2 | 1.0 | solo `MAP_DECAY_FOV` + `MAP_HIT_WINS` | 2/8 | 6 | 25.1 |
| bosque disperso (gap 2.2) | 0.5 / 1.0 / 1.5 | r17 | 6/6 · 6/6 · 6/6 | 0 | 25.7 / 17.2 / 16.1 |

Con GT, r17 pasa de 22/32 a 40/40 (más 18/18 en el bosque disperso). La pieza principal es
`FLY_KEEP_NEAR`; el decaimiento por sector cubre el caso del tronco visto una sola vez. El precio es
algo de tiempo: el dron ve más troncos y los rodea (31.9 frente a 28.8 s a 0.5 m/s).

## 3 · Qué limita la velocidad

En cada cuadro del planificador se registró qué límite manda (`MHPlanner.dbg`, GT, r17, 4 episodios
por fila; "holgura" incluye "holgura + FOV"):

| mundo | V_CRUISE | v media pedida | holgura | crucero | girar primero | aproximación | despegue |
|---|---|---|---|---|---|---|---|
| bosques aleatorios | 0.5 | 0.41 m/s | 40 % | 51 % | 0 % | 6 % | 3 % |
| bosques aleatorios | 1.0 | 0.53 m/s | 74 % | 10 % | 7 % | 4 % | 4 % |
| bosque_s2 | 1.0 | 0.44 m/s | 80 % | 7 % | 8 % | 3 % | 3 % |
| bosque disperso (gap 2.2) | 1.0 | 0.73 m/s | 36 % | 41 % | 12 % | 6 % | 5 % |

A 1 m/s, en el bosque de Gazebo, el 80 % del tiempo la velocidad la fija la ley de holgura
v ≤ (c − r − m) / M_V con M_V = 0.4 s. Con troncos cada 1.6 m y el X3 (Ø 0.72 m), la holgura típica
es c ≈ 0.6 m, lo que da ~0.5 m/s pida lo que pida `V_CRUISE`. Por eso subir a 1.0 m/s ahorra un 18 %
en los bosques aleatorios y nada en bosque_s2, y 1.5 m/s no añade nada. En el bosque disperso sí
rinde: 25.7 → 17.2 s (−33 %).

**Bajar M_V** (menos margen por m/s) es la palanca directa, pero no es segura:

| mundo | v | M_V | éxito | t mediano |
|---|---|---|---|---|
| bosques aleatorios | 1.0 | 0.4 / 0.25 / 0.15 | 8/8 · 8/8 · 8/8 | 26.3 / 22.4 / 18.2 s |
| bosque_s2 | 1.0 | 0.4 / 0.25 / 0.15 | **8/8 · 5/8 · 4/8** | 33.8 / 23.0 / 22.5 s |

## 4 · Con la CNN (M_ToF local de presupuesto corto, n = 4: indicativo)

| v | mapa | éxito | t mediano | recorrido |
|---|---|---|---|---|
| 0.5 | actual | 4/4 | 77.3 s | 18.6 m |
| 0.5 | r17 | 2/4 | 61.0 s | 18.4 m |
| 0.5 | `FLY_KEEP_NEAR` | 4/4 | 86.2 s | 24.1 m |
| 1.0 | actual | 4/4 | 75.3 s | 25.6 m |
| 1.0 | r17 | 4/4 | 64.8 s | 20.3 m |
| 1.0 | `FLY_KEEP_NEAR` | 3/4 | 39.5 s | 16.5 m |

Con la CNN las opciones no mejoran de forma clara: 13/16 frente a 8/8. Con n = 4 la diferencia no
es significativa, pero tampoco hay evidencia a favor. Los dos choques con r17 a 0.5 m/s son contra
el mismo tronco (r = 0.13 m), por otro mecanismo: la CNN no lo detectó en el borde de la imagen (a
−21…−28°, fuera del ToF de ±20°) y los rayos libres borraron sus celdas *mientras estaba a la vista*.
El dron avanzaba 27° desviado de su cámara, lo que permite `FOV_OK` = 28°, y lo barrió con el
costado. Es el mismo punto ciego lateral.

El M_ToF del PC de entrenamiento (presupuesto completo, el que voló en Gazebo) es mejor que el local,
y en Gazebo tardó 40.5 s donde el local tarda 77 s offline. **La decisión de activarlas por defecto
depende de la suite `speed_thin` con esa red:**

```bash
nanouav gazebo gen-world --scenario bosque --seed 2
nanouav gazebo gen-world --scenario estrechos --seed 0 --kw w=1.2
nanouav suite configs/suites/speed_thin.yaml -j 7          # 36 trabajos, ~1–1.5 h; GT, M y M_ToF, base / r17 / FLY
nanouav summarize --dir results/speed_thin                 # éxito, IC 95 %, choques y tiempo mediano
```

## Conclusiones

1. El mapa del MH pierde troncos finos por dos defectos: el filtro de píxeles voladores los borra y el
   decaimiento global los olvida en cuanto salen del campo visual. Con profundidad perfecta eso causa
   choques en la mitad de los episodios del mundo de Gazebo. Las opciones r17 lo corrigen (40/40) y
   quedan como opción hasta validarlas con la CNN completa. Los resultados publicados no cambian.
2. En un bosque como el de Gazebo, el tiempo se va en pasar entre troncos con poca holgura, no en el
   límite de crucero. `V_CRUISE` = 1.0 ahorra 0–18 %, y un 33 % si el bosque es menos denso. Ir más
   rápido entre troncos exige menos margen, y eso hoy choca (M_V = 0.25: 5/8).
3. El límite de fondo es el **punto ciego lateral**: la cámara de 60° no ve el costado del corredor
   que barre el cuerpo a menos de ~0.8 m, y el MH permite avanzar hasta 28° desviado del eje de la
   cámara. Siguiente paso: exigir que el corredor completo (centro ± (r + margen)) quede dentro del
   campo visual, o bien ToF laterales.

### Para Gazebo, hoy

```bash
DOCKER="sudo docker" VARIANT=M_ToF WORLD=bosque SEED=2 V_CRUISE=1.0 bash gazebo/gz_docker.sh
DOCKER="sudo docker" VARIANT=M_ToF WORLD=bosque SEED=2 V_CRUISE=1.0 PLANNER_CFG="FLY_KEEP_NEAR=1,MAP_DECAY_FOV=1,MAP_HIT_WINS=1" bash gazebo/gz_docker.sh
DOCKER="sudo docker" VARIANT=M_ToF WORLD=bosque SEED=2 WORLD_KW="gap=2.2" V_CRUISE=1.0 bash gazebo/gz_docker.sh   # bosque menos denso
```
