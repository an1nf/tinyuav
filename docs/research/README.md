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

Plantilla para un informe nuevo (`RNN_tema.md`): pregunta · qué se toma de la fuente · cambios en el
código (archivo y bandera, apagada por defecto) · montaje y **comando exacto** (`nanouav run …` o
suite) con el commit · tabla de resultados con n e IC · conclusión y siguiente paso.
