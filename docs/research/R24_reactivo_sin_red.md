# R24 — Un piloto reactivo sin red con la misma percepción

*2026-10-15 · nanouav 0.25.0*

## La pregunta

Desde R18 comparamos al estudiante del paper con el experto privilegiado y con el piloto ciego. Faltaba la
línea base que importa: **un planificador clásico, sin aprendizaje y sin nada privilegiado, con el mismo
sensor**. Freißmuth et al. (ETH, arXiv 2610.13158) la sugieren: consultan la distancia directamente sobre
un buffer de imágenes de profundidad y dicen que eso puede servir de filtro de seguridad bajo una política
aprendida. Si un planificador así iguala al estudiante, la red aporta poco. Si lo supera, es lo que hay que
batir.

## El piloto `reactive` (`rollout.Rollout.plan_reactive`)

En cada cuadro (15 Hz) se hace lo mismo, sin red y sin mapa privilegiado:

1. **Candidatas** (`fan_points`): 33 trayectorias de 1 s. Cada una es una curva de Bézier cuadrática que
   sale en la dirección de la velocidad actual y termina a $v_{des} \cdot 1$ s. Los finales se abren en
   abanico alrededor de ω: 0°, ±10°, ±20°, ±30°, ±45° y ±60°, cada uno a la misma altura o ±0.5 m.
2. **Proyección.** Cada candidata pasa por la misma proyección sobre un polinomio de grado 5 que el
   estudiante (Eq. 8, con la misma escala de tiempo β).
3. **Holgura en su propio SGM.** Es la misma de `--select sgm`: celdas de 0.15 m vistas en ≥ 2 de los
   ~15 cuadros del último segundo. Ahora se consulta con un k-d tree.
4. **Elección.** Entre las candidatas libres (holgura ≥ radio + 0.15 m) gana la más cercana a la recta del
   próximo segundo, más 0.3 × la distancia al plan en curso para no cambiar de lado en cada cuadro
   (`fan_cost`).
5. **Si ninguna está libre**, sigue con el plan en curso si lo que queda de él lo está (`--keep`). Si
   tampoco, vuela la candidata de más holgura.

Decidir cuesta una mediana de 17–29 ms por cuadro en este CPU (`dec_ms`, nueva columna de `eval`).

## Resultados en nuestro simulador

Bosque de prueba de R18 ($\delta = 1/25$, semillas 1000–1009, 10 vuelos por velocidad):

| piloto | 3 m/s | 5 m/s | 7 m/s | 10 m/s | total (IC 95 %) |
|---|---|---|---|---|---|
| experto M-H (privilegiado) | 10 | 10 | 10 | 10 | 40/40 |
| estudiante `student_train`, `snap` (R18) | 3 | 6 | 7 | 6 | 22/40 [0.40, 0.69] |
| estudiante `student_train`, `sgm` (R22) | 6 | 8 | 7 | 7 | 28/40 [0.55, 0.82] |
| ciego | | | | | 16/40 [0.26, 0.55] |
| **reactivo** `--keep` | 10 | 10 | 9 | 6 | 35/40 [0.74, 0.95] |
| **reactivo** `--keep --tilt` | 10 | 10 | 9 | 7 | 36/40 [0.77, 0.96] |
| **reactivo** `--keep --tilt --edge` | 9 | 10 | 9 | 9 | **37/40** [0.80, 0.97] |
| **reactivo** `--keep --tilt --edge --vga` (la cámara del paper) | 10 | 10 | 9 | 10 | **39/40** [0.87, 0.99] |
| paper, estudiante en su simulador | 100 % | 100 % | — | ~60 % | |

**A 10 m/s con 30 entornos** (semillas 1000–1029):
- `--tilt`: 20/30;
- `--tilt --edge`: 24/30 [0.63, 0.90]; emparejado contra el anterior, 5 contra 1 ($p = 0.22$);
- `--tilt --edge` cobrando la latencia de la decisión (`--latency 0.08`: 50 ms + ~30 ms): 25/30;
- `--tilt --edge --vga`: 25/30 [0.66, 0.93]. Fallan otros entornos que con 320 × 240, pero no más.

Frente al 60 % del paper, 24 de 30 tiene $p = 0.017$ (binomial). Pero son simuladores distintos: la
comparación limpia es la de nuestra recreación, donde el reactivo supera a nuestro estudiante (37 frente a 28
de 40).

**Lo que significa.**
- **Sin aprendizaje, el mismo sensor llega a 37 de 40.** Iguala o supera lo que el paper reporta para su
  estudiante, y supera claramente a nuestra recreación de ese estudiante.
- **La red no es necesaria para esquivar en estos bosques.** Lo que hacía falta era juzgar los planes con lo
  que el dron mide y no soltar un plan libre. Es el mismo hallazgo de R22, ahora sin red.
- **La latencia de decidir no cambia el resultado:** 25 de 30 con 80 ms frente a 24 de 30 con 50 ms.

## Qué se probó para la percepción

**`--tilt`.** El sensor del paper inclinado 30° hacia arriba a ≥ 7 m/s, como ya se hacía en R22. Sin él, a
10 m/s el dron se inclina hacia adelante y la cámara mira al suelo.

**`--edge` (`rollout.edge_filter`).** Se descartan los píxeles SGM con un salto de profundidad hacia un
vecino mayor que $\max(0.3, 0.03 z^2)$ m, o con un vecino sin dato. Es la prueba de saltos de Choudhury y
Tiwari (arXiv 2609.06547), que crece con $z^2$ como el error del estéreo. Quita los puntos que el SGM deja en
el espacio libre al «embarrar» un tronco contra el fondo.

**Falsas alarmas y omisiones por decisión.** Medidas contra la verdad sobre las 33 candidatas en vuelos del
propio piloto:
- *falsa alarma*: candidata realmente libre (≥ 0.35 m) que el dron ve bloqueada;
- *omisión*: candidata que choca (< 0.2 m) y el dron ve libre.

| variante | 7 m/s: falsa alarma / omisión | 10 m/s: falsa alarma / omisión |
|---|---|---|
| base (1 píxel de cada bloque de 4 × 4, ≥ 2 cuadros) | 29 % / 5 % | 30 % / 13 % |
| `--pool min` (el píxel más cercano del bloque) | 51 % / 4 % | 52 % / 8 % |
| `--edge` | 19 % / 7 % | 19 % / 19 % |
| ≥ 3 cuadros | 6 % / 23 % | 8 % / 41 % |

En vuelo, a 7 y 10 m/s con 10 entornos cada una:

| variante | aciertos |
|---|---|
| base | 16/20 |
| `--pool min` | 17/20 |
| `--pool min --edge` | 17/20 |
| `--edge` | 18/20 |

Con 10 vuelos por velocidad, ninguna diferencia es significativa. Se queda `--edge` por la falsa alarma
más baja con la omisión casi igual, y por los 24/30 a 10 m/s.

**`--carve` (espacio libre visto a través).** Una celda deja de contar si los cuadros que vieron más allá de
ella superan a los que la tocaron. Es el modelo inverso del sensor de los mapas de ocupación, y el «libre si
algún cuadro lo ve libre» de Freißmuth et al.
- **Por decisión ayuda mucho:** a 3 m/s la falsa alarma baja del 21 % al 6 % (al 3 % con `--edge`) sin subir
  la omisión.
- **En vuelo, empeora:** 34/40 contra 37/40, con 5 fallos nuevos y 2 corregidos; a 10 m/s con 30
  entornos, 20/30 contra 24/30. Lo más probable es que las celdas del borde de un tronco, 7.5 cm fuera de su
  superficie, queden atravesadas por rayos tangentes.

Queda apagado. Para que funcione, el radio de la vecindad debería escalar con $0.075 f / z$ píxeles.

**`--w-clear` (costo de holgura suave).** Se suma $w \, e^{-(c - 0.35)/0.3}$ al costo de cada candidata. Es
la barrera de Gil et al. (arXiv 2509.20084), para preferir el hueco más ancho. Con $w = 2$ arregla dos de los
entornos que fallaban a 10 m/s, donde el dron se metía por un pasillo que se cerraba. Rompe otros cuatro:
22/30 contra 24/30. Queda apagado.

**El resultado que queda a 10 m/s.** Con 320 × 240, `--edge` o VGA, la tasa se estanca en 24–25 de 30. La
mitad de los fallos restantes cae en los mismos entornos con cualquier variante de percepción, al mismo
instante (1014, 1015, 1016 y 1024). Ahí el dron vuela recto por la línea, sus opciones libres se cierran en
~0.2 s y no queda ninguna: a 10 m/s, el alcance de 10 m del estéreo es 1 s de vuelo, justo el horizonte del
plan. Es el límite de R4: la velocidad la limita el alcance del sensor.

## Lo que sigue

- **En Flightmare, la misma comparación en el almacén:** el reactivo contra el estudiante con `--keep`
  (69/90), y el híbrido estudiante + `--fan`. Si ninguna de sus 3 hipótesis está libre y `--keep` no puede
  actuar, `--fan` prueba el abanico del reactivo.
- **El único entorno que sigue fallando a 3, 7 y 10 m/s (semilla 1006):** un grupo denso donde los huecos
  libres de verdad tienen 0.4 m de holgura y el dron los ve con 0.1–0.25 m. Ninguna candidata parece libre y
  vuela la de más holgura hacia un callejón. Frenar (`--slow 0.5`) no lo arregla, porque a media velocidad
  tampoco ve ninguna libre.

## Comandos

```bash
nanouav run r18 pilots --pilots reactive --keep --tilt --edge --v 3,5,7,10 --n 10
nanouav run r18 pilots --pilots reactive --keep --tilt --edge --v 10 --n 30 --latency 0.08
nanouav run r22 eval --arena --scene warehouse --pilots reactive --v 3,5,7 --n 30 --edge --tag n30
```

Los resultados de cada vuelo están en `docs/research/r24/eval_*.json`. El estudio de falsas alarmas y
omisiones por decisión está en `docs/research/r24/percepcion_por_decision.py`
(`python percepcion_por_decision.py 10 1000,1003,1008`).
