# R15 · DroneMOT, VisDrone2018 y DroneVis → falsas detecciones dinámicas según el ego-movimiento

## Los tres papers, en una línea cada uno
| Paper | Qué es | Utilidad para nosotros |
|---|---|---|
| **DroneMOT** (Wang, Wang & Li, arXiv 2407.09051, 2024) | Seguimiento multi-objeto desde un dron en movimiento. DLA34 con atención espacial y temporal guiada por el heatmap del cuadro anterior; asociación que descompone el movimiento del dron en hover / traslación / rotación | **Media-alta**: la premisa (la rotación propia es lo que más rompe la asociación) es comprobable en nuestra capa dinámica, y las métricas por pista (IDs, MT/ML) dan rigor |
| **VisDrone2018** (Zhu et al., arXiv 1804.07437) | Benchmark de detección y seguimiento: 263 videos (179 k cuadros), 10 209 imágenes, 2.5 M cajas, 10 clases, 14 ciudades, vistas altas o cenitales | **Baja** para evasión (otro régimen). Útil para el protocolo de SOT (éxito AUC, precisión a 20 px) si se evalúa el seguidor de objetivo de E4 |
| **DroneVis** (Heakl et al., arXiv 2406.00447, 2024) | Biblioteca Python para el Parrot AR.Drone 2: envuelve YOLOv8, BoT-SORT, MediaPipe y GLPDepth, con GUI, CLI y control por gestos | **Baja** científicamente: las métricas se copian de los papers originales (COCO, KITTI), no se miden con datos de dron. Útil para posicionarnos: su profundidad (GLPDepth) pesa **245 MB y corre a 4 fps en CPU**; la nuestra pesa **65 KB int8** |

### DroneMOT con más detalle
- **Red:** atención espacial (3 capas de self-attention sobre rasgos DLA34 + codificación posicional 2D). Atención temporal con el heatmap del cuadro t−1 como filtro, que recupera objetos borrosos u ocluidos.
- **AFS:** EMA de apariencia con α adaptativo a la confianza de la detección, más 10 "rasgos clave" históricos (LRU).
- **DMP:** en hover usa Kalman como cámara fija. En traslación compensa con una afín entre cuadros (como BoT-SORT). En rotación usa un "triángulo" objeto–vecino más lejano–vecino más cercano (dos ángulos + un lado, invariantes a la rotación).
- **Costo de asociación:** C = IoU + 0.5·apariencia + 0.1·rotación, resuelto con Hungarian.
- **Resultados:** VisDrone2019-MOT IDF1 58.6 / MOTA 43.7; UAVDT IDF1 69.6 / MOTA 50.1 con 129 IDs. La ablación es clara: MDA baja los IDs de 1509 a 406, y DMP solo, de 1407 a 229.
- **Lectura crítica:** entrenado en 6 RTX 2080 Ti y sin tiempo de inferencia reportado. Vistas altas, objetos pequeños. El rasgo de rotación es ad hoc. En la tabla III, IDP baja a 43.0 al combinar DMP y AFS sin que se comente. Nada de esto es ejecutable en un nano-dron; la idea sí es transferible.

## Experimento R15 (`r15_dyn_egomotion.py`)
**Pregunta:** ¿cuántas pistas dinámicas falsas confirma nuestra capa dinámica (excitación = celda vista libre que ahora aparece ocupada) en un mundo estático, según el tipo de ego-movimiento? ¿Y detecta igual un balón real?

**Montaje:** lazo abierto con trayectoria prescrita (el carril está libre), CNN M + ToF como en vuelo, 20 s por episodio, n = 6 por celda. Los regímenes, como en DroneMOT, son H (hover), T (traslación a 0.5 m/s), R (rotación ±57°, |ω| ≤ 1.05 rad/s) y TR (traslación + rotación). El balón cruza a 1 m/s y 2.2 m por delante.

### Línea base
| Régimen | Pistas falsas/min | Tiempo en EVADE sin motivo (de 20 s) | Recall del balón | Balón detectado |
|---|---|---|---|---|
| H | 10.5 | 0.4 s (2 %) | 0.46 | 6/6 |
| T | 22.0 | 1.8 s (9 %) | 0.41 | 6/6 |
| R | **43.5** | 1.3 s (7 %) | 0.49 | 6/6 |
| TR | 35.0 | **2.9 s (15 %)** | 0.38 | 5/6 |

1. **Se confirma la premisa de DroneMOT en nuestro sistema:** la rotación propia multiplica por ~4 las pistas falsas respecto al hover, y la traslación por ~2.
2. **Hallazgo nuevo:** incluso en hover, sin movimiento, aparecen ~10 pistas falsas por minuto. Es parpadeo cuadro a cuadro de la CNN. La odometría ya compensa el ego-movimiento en el marco del mundo, así que estas falsas no vienen de mala compensación (el problema que resuelve DroneMOT): vienen de que la profundidad de la CNN cambia con el punto de vista.
3. Con traslación + rotación, el dron pasa **hasta el 15 % del tiempo esquivando fantasmas**.

### Variantes probadas
| Variante | H FP / recall | T FP / recall | R FP / recall | Veredicto |
|---|---|---|---|---|
| No crear pistas con \|ω\| > 0.35 (análogo directo de DMP) | 10.5 / 0.46 | 22 / 0.41 | 38.5 / **0.13** | ✗ el recall se desploma |
| Exigir desplazamiento neto ≥ 0.4 m (`RF_NET_DISP`) | 8.5 / 0.30 | 10.5 / 0.27 | 37.5 / 0.33 | ± baja las falsas en T a la mitad y EVADE en H a 0, pero pierde ~⅓ del recall |
| + ≥ 5 puntos por grupo (`RF_MIN_PTS`) | 1.5 / **0.00** | 8.5 / 0.03 | — | ✗ deja de ver el balón |
| Rechazar velocidades que explica ω × r (`RF_EGO_REJ`) | — | — | 43.5 / — | ✗ sin efecto. Las pistas falsas se mueven en *contra* de la rotación (cos ≈ −0.6 a −1), no con ella |
| Inhibición junto a contornos estáticos (`RF_EDGE_INH` = 0.3 m; estático = ocupado de forma continua > 1 s) | **3.5** / 0.33 | 17 / 0.38 | 33 / 0.32 | ✓ el mejor compromiso: las falsas en hover caen 67 % y EVADE sin motivo cae a la mitad o menos |
| Inhibición + desplazamiento neto | **1.5** / 0.19 | **5.0** / 0.29 | 28 / 0.21 | ✓/± las falsas bajan 77–86 % en H y T; el recall baja ~45 % |

Con TR (traslación + rotación), EVADE sin motivo pasa de 2.9 s a 1.7 s (inhibición) y a 1.0 s (inhibición + desplazamiento neto), con recall de 0.32 y 0.20.

La primera versión de la inhibición usaba "celda con evidencia ≥ 3" y apagaba también al balón (recall 0), porque su propia estela (ToF ×2 + CNN) se volvía "estática". Exigir ocupación continua > 1 s lo arregla.

Todas las variantes están en `_04_policy/evasion.py`, apagadas por defecto, así que los resultados publicados no cambian.

## Conclusiones y siguiente paso
- El cuello de botella de la capa dinámica no es la asociación (lo que ataca DroneMOT) sino la **consistencia de la profundidad entre puntos de vista**. Es el mismo problema que R13 y R14 encontraron para puertas y bordes.
- La única variante que se apoya en una idea de percepción y no de asociación, la inhibición junto a contornos estáticos (análoga al IRF de Shang 2025), es la que mejor separa: los fantasmas nacen pegados a los obstáculos, donde el borde difuso de la CNN cambia con el punto de vista. Bajo rotación sigue habiendo ~30 falsas/min: falta el ToF.
- El dato más fiable para separar objeto real de fantasma es el **ToF**: un objeto real en su campo cambia la lectura de la zona; un fantasma de la CNN no. Propuesta: excitación con dos niveles, confirmada si el ToF la corrobora y con peso reducido fuera de su campo. Se mide con `r15_dyn_egomotion.py`.
- **Métricas para el paper:** reportar falsas/min, tiempo en EVADE sin motivo, recall y latencia por régimen de ego-movimiento. Hoy solo se reporta el éxito en el escenario dinámico.
- De DroneVis sale una sugerencia de práctica: pruebas automáticas (pytest) para el planificador y el nodo de Gazebo. Hoy hubo regresiones que esas pruebas habrían detectado al momento.
