# Informes de investigación

Cada informe documenta una iteración: qué se tomó de un paper, qué se implementó, el experimento,
los números y la conclusión. Son **registros históricos**: los comandos y rutas que citan son los de
la estructura plana en la que se escribieron (`python r5_pole.py`, `_04_policy/…`, `UAV_FAKE_CNN`).
Su equivalente actual está en [../migracion.md](../migracion.md); por ejemplo
`python g1_gazebo_world.py …` → `nanouav run r12 …`, `_04_policy/mh_planner.py` →
`src/nanouav/planning/mh_planner.py`, `UAV_FUSE2=1` → `NANOUAV_FUSE2=1`.

| informe | iteraciones | experimentos (`nanouav run …`) |
|---|---|---|
| [R_2026-09_papers.md](R_2026-09_papers.md) | R1–R3: LNCA (Nunn & Nikoubin 2026), seguimiento (Su et al. 2026) | `r01`, `r02`, `r03` |
| [R_2026-09_loquercio.md](R_2026-09_loquercio.md) | R4–R8: vuelo rápido, ley consistente, cámara, ruido de estado | `r04`, `r05`, `r06`, `r07`, `r08` |
| [R_2026-09_shang_neuromorphic.md](R_2026-09_shang_neuromorphic.md) | R9–R10: obstáculos dinámicos, deriva | `r08 --vo`, `r10` |
| [R_2026-09_tinynav.md](R_2026-09_tinynav.md) | R11: velocidad desde la cámara | `r11`, `r11b` |
| [R12_mh_planner_loquercio.md](R12_mh_planner_loquercio.md) | R12: planificador MH, bloqueo en Gazebo | `r12` |
| [R13_pasos_estrechos.md](R13_pasos_estrechos.md) | R13: pasos estrechos, anti-estancamiento | `r12 --world scn:estrechos…`, suite `narrow_passages` |
| [R14_ddos_metricas_por_clase.md](R14_ddos_metricas_por_clase.md) | R14: métricas por clase, `fuse_tof2` | `r14`, suite `perception` |
| [R15_tracking_drones.md](R15_tracking_drones.md) | R15: falsas detecciones dinámicas vs ego-movimiento | `r15`, suite `perception` |
| [R16_entrenamiento_y_lazo_cerrado.md](R16_entrenamiento_y_lazo_cerrado.md) | R16: entrenamiento completo, M vs M_ToF en lazo cerrado, límites del sensor (cables, campo visual) | `train`, `r03`, `r14`, suites `baseline`, `thin_structures`, `cables`, `narrow_passages` |
| [R17_velocidad_y_troncos_finos.md](R17_velocidad_y_troncos_finos.md) | R17: velocidad de crucero en bosque, troncos finos que el mapa pierde, reinicios en Gazebo | `r12 --v … --cfg FLY_KEEP_NEAR=1,…`, suite `speed_thin` |
| [R18_agile_autonomy.md](R18_agile_autonomy.md) | R18: recreación completa de Loquercio et al. 2021 (estéreo + SGM, experto M-H privilegiado, estudiante multimodal, DAgger) | `r18 sgm/pilots/dagger/eval` |
| [R19_seguimiento_personas.md](R19_seguimiento_personas.md) | R19: detectar y seguir a una persona (YOLO + ByteTrack + bloqueo del objetivo + política R18) | `r19 prep-visdrone/train-det/eval-det/video/sim/dagger` |
| [R20_seguimiento_globo.md](R20_seguimiento_globo.md) | R20: seguir un globo de color por un bosque (detector de color, distancia por tamaño, filtro 3D, compensación del giro) | `r20 frames/sim/dagger/pick-color/video` |
| [R21_gazebo_agile.md](R21_gazebo_agile.md) | R21: el estudiante de R18 en Gazebo (mundo de prueba, X3 ágil con estéreo, nodo ROS 2, export a TFLite) | `gazebo/gz_docker.sh agile`, `r18 export`, `r18 video --from-gz` |
| [R23_direccion_y_etiquetas.md](R23_direccion_y_etiquetas.md) | R23: el techo del estudiante — sus etiquetas siguen una ruta que no ve (τ_gbl) mientras ω apunta a la recta; aprende a no fiarse de ω y deriva. Cuatro condiciones y una referencia replaneada que las cumple | `r18 dagger --label-ref ctg`, `r22 dagger --label-ref ctg` |
| [R22_flightmare.md](R22_flightmare.md) | R22: el estudiante de R18 en Flightmare (Unity del paper) por un puente ZMQ en Python, con la nube de puntos como mapa privilegiado | `r22 probe/cloud/lines/eval/video/fake` |
| [R24_reactivo_sin_red.md](R24_reactivo_sin_red.md) | R24: un piloto reactivo sin red ni nada privilegiado (abanico de candidatas juzgadas sobre su propio SGM, `--keep`) supera a nuestro estudiante y llega a lo que el paper reporta para el suyo; filtro de bordes, costo de holgura suave, VGA | `r18 pilots --pilots reactive`, `r22 eval --pilots reactive` |
| [R25_diffphys.md](R25_diffphys.md) | R25: política recurrente de 16 × 12 entrenada por física diferenciable (Zhang et al. 2025), sin experto ni etiquetas; en R18 empata con el reactivo (103/120) y en el almacén lo supera a 10 m/s (27/30 frente a 20/30), pero vuela más lento de lo pedido; con el arreglo de 0.26.6 y `--edge`, 114/120 en R18 y emparejada 6 frente a 0 sobre el reactivo; 0.27.0: frenaba tras 10 s de vuelo, `--burn` lo arregla; r25c: 117/120 en el almacén | `r25 train`, `r18 eval --pilots diffphys`, `r22 eval --pilots diffphys` |

| [R26_diffphys_seguimiento.md](R26_diffphys_seguimiento.md) | R26: la política de R25 ajustada para seguir un objetivo que se mueve, con una barrera de visibilidad (la línea de visión contra los troncos) en la pérdida; vuelo previo sin gradiente para que no se canse a los 10 s; en R20 (globo, bosque de 1/25 m⁻²) 30/30 sin choques, frente a 26/30 de la línea recta; con objetivos de 3–7 m/s (las velocidades de YOPOv2-Tracker), ningún choque en 90 vuelos y 26/30 a 7 m/s, frente a 18/30 del rastro privilegiado; `--fast-target` (filtro más rápido y predicción del giro) para no perder el globo cuando gira detrás de un tronco: 29/30 y 28/30 a 7 m/s en bosques de 1/25 y 1/16 m⁻², sin choques; frente a YOPOv2-Tracker (~8/10 a 7 m/s en su bosque de 1/16 m⁻²), con 1.1 ms por decisión en un hilo de CPU | `r26 train/eval`, `r20 sim --pilots diffphys --fast-target`, `r25 bench` |

Plantilla para un informe nuevo (`RNN_tema.md`): pregunta · qué se toma de la fuente · cambios en el
código (archivo y bandera, apagada por defecto) · montaje y **comando exacto** (`nanouav run …` o
suite) con el commit · tabla de resultados con n e IC · conclusión y siguiente paso.
