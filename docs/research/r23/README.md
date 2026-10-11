# R23 · scripts de la prueba en CPU

Los números de [R23](../R23_direccion_y_etiquetas.md) salen de estos scripts.

- **Entorno.** Necesitan `nanouav` instalado (`pip install -e .`) y TensorFlow.
- **Salida.** Escriben en `results/r23_cpu/` (o en `$R23_DIR`).
- **Tiempos.** Medidos en 2 núcleos, sin GPU.

| paso | script | salida | tiempo |
|---|---|---|---|
| 1 · etapa 1: estados de τ_gbl y etiquetas con referencia / mapa cambiados | `s1_labels.py` | `s1_states.npy`, `s1_labels.npy` | ~15 min |
| 1 · tablas de la etapa 1 | `s1_analyze.py` | (texto) | s |
| 2 · estados (τ_gbl y ciego, 60 entornos), profundidad SGM, etiquetas τ_gbl y recta | `s2_data.py 60` | `s2/` | ~1 h 10 min |
| 2 · etiquetas replaneadas: a la meta (D, E) y pegadas a la recta (G) | `s2_label.py d`, `s2_label.py g` | `s2d/`, `s2g/` | ~40 min cada una |
| 2 · entrenar A, B, C | `s2_train.py` | `s2_model_{A,B,C}.keras`, `s2_results.json` | ~25 min |
| 2 · entrenar D, E, F, G (y A2 = A) | `s2_train_d.py D,E,F,G` | `s2_model_*.keras`, `s2_results_d.json` | ~8 min cada una |
| 2 · lazo abierto por distancia a la recta y seguridad de las etiquetas | `s2_analyze.py` | `s2_perstate_all.json` | ~10 min |
| 2 · ángulo entre ω y las etiquetas | `s2_consistency.py` | (texto) | ~1 min |
| 2 · vuelos en los bosques de prueba | `s2_eval.py A,B,C,D,E,F,G 3,5 10` | `s2_closed_loop.json` | ~1 min por vuelo |
| un vuelo con su línea de tiempo | `s2_one.py G 3 1002` | (texto) | ~1 min |

**Diferencias con la nota.**
- **Etiquetas D de 110 de los 120 episodios.** En la nota salieron de una versión previa de `CostToGo`: la
  ruta terminaba en la meta, sin seguir 3 m más allá, y la semilla no crecía si la meta caía en el margen de
  un obstáculo. Con esa versión fallaban 5 entornos enteros, que se reetiquetaron con la actual. La
  diferencia está solo en el último segundo antes de la meta.
- **Columnas `mean_abs_dz_m` y `progress` de `s2_eval.py`.** Se añadieron después de los vuelos de A–F; G sí
  las tiene.

**Resultados.** En `datos/` están los resúmenes que dan las tablas de la nota:
- `s2_results.json` (A–C) y `s2_results_d.json` (D–G, A2): R-WTA y lazo abierto;
- `s2_closed_loop.json`: cada vuelo.

Los datos (~56 MB) y los modelos quedan fuera de git, como el resto de `results/`.
