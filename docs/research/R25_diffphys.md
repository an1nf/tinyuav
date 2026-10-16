# R25 — Aprender a volar por física diferenciable (Zhang et al. 2025)

*2026-10-15 · nanouav 0.26.0 · prueba de concepto en CPU; el entrenamiento real va en la GPU*

## La pregunta

DiffPhysDrone (Zhang, Hu, Song, Zou y Lin, *Nature Machine Intelligence* 2025; arXiv 2407.10648) es el estado
del arte por encima de Loquercio:
- 90 % de éxito frente al 60 % de Agile Autonomy en el montaje de este;
- 20 m/s en un bosque real;
- una red diminuta que corre en 3.7 ms en un ARM de 21 dólares.

No usa experto ni etiquetas: la política se entrena propagando el gradiente a través de un simulador
diferenciable de masa puntual. ¿Se reproduce en nuestro banco, y supera al piloto reactivo de R24 (37/40)?
Para el nano-UAV es lo más prometedor: una entrada de 16 × 12 de profundidad está a un paso del ToF 8 × 8.

## Implementación (`nanouav.agile.diffphys`, `nanouav run r25 train`)

**Simulador**, en TensorFlow:
- masa puntual con un paso de latencia, respuesta de primer orden de la aceleración (λ = 12 s⁻¹) y arrastre
  cuadrático y lineal;
- Δt = 1/15 s, 150 pasos (10 s), lotes de 64 bosques de Poisson como los de entrenamiento de R18 (espaciado
  4–7 m, troncos de 0.15–0.4 m);
- velocidad deseada entre 3 y 12 m/s;
- decaimiento temporal del gradiente $e^{-\alpha\,dt}$.

**Distancia:** analítica a los cilindros y al suelo, así que el gradiente es exacto.

**Imagen:** rayos de 32 × 24 contra los mismos cilindros, mínimo de cada 2 × 2 y luego inversa, en
16 × 12. No pasa gradiente por la imagen, como en el paper. Lleva aumento de datos tipo SGM: 10 % de huecos,
que se leen como lejos, y 5 % de ruido.

**Pérdida**, con los pesos del paper (1, 2, 0.01 y 0.001):
- seguir la velocidad deseada, con la velocidad promediada en ~2 s;
- choque: barrera softplus sobre la holgura, más la velocidad de acercamiento × $\max(1-c, 0)^2$;
- aceleración y jerk.

**Red:** CNN con núcleos de 2, 3 y 3 y 32, 64 y 128 filtros, más el estado sumado, más una GRU de 192 y una
salida de aceleración; unos 390 000 parámetros. El estado son 9 números en el marco de rumbo: velocidad,
dirección del empuje y velocidad deseada.

**En vuelo** (`--pilots diffphys` en `r18 eval`), dentro del mismo lazo de R18:
- la entrada es su propio SGM, con el mínimo de cada bloque y los huecos como lejos;
- la GRU conserva su estado entre cuadros;
- la aceleración se convierte en un plan de un cuadro que sigue el controlador en cascada.

## Prueba de concepto (CPU, 700 iteraciones a lote 32: el 0.7 % de lo que usa el paper)

Bosque de prueba de R18, 10 vuelos por velocidad:

| versión | 3 m/s | 5 m/s | 7 m/s | 10 m/s | velocidad media alcanzada |
|---|---|---|---|---|---|
| sin la velocidad deseada como entrada | 9 | 8 | 7 | 6 | 4.1–5.2 m/s, pida lo que se pida |
| con la velocidad deseada (como el paper) | 10 | 9 | 9 | 1 | 2.6 · 4.0 · 4.4 · 4.4 m/s |

**Lo que dicen estos números:**
- **La cadena funciona de punta a punta.** Una política entrenada sin experto, sin etiquetas y sin ver nunca
  un SGM vuela con su propio SGM y casi no choca: 2 choques en 40 vuelos.
- **Todavía no es comparable a velocidad alta.** A 7 y 10 m/s no pasa de ~4.4 m/s y a 10 m/s se queda sin
  tiempo, no choca. Con 700 iteraciones aprendió a esquivar antes que a correr.

No se puede afirmar que supere a nada hasta entrenarla como el paper.

## Entrenamiento en la GPU (20 000 iteraciones a lote 64, TITAN X Pascal)

Es el 40 % de lo que usa el paper (50 000 iteraciones a lote 64). Promedios de cada tramo de 1000 iteraciones
(registro en `r25/train_gpu20k.log`):

| iteraciones | pérdida | seguimiento de velocidad | choque (pérdida) | episodios con choque |
|---|---|---|---|---|
| 0–1 000 | 1.74 | 1.60 | 0.020 | 2.7 % |
| 1 000–2 000 | 1.19 | 1.05 | 0.017 | 0.8 % |
| 4 000–5 000 | 1.01 | 0.88 | 0.017 | 0.7 % |
| 9 000–10 000 | 0.94 | 0.80 | 0.017 | 0.4 % |
| 14 000–15 000 | 0.87 | 0.73 | 0.016 | 0.4 % |
| 19 000–20 000 | 0.86 | 0.73 | 0.016 | 0.4 % |

**Lo que dice la curva:**
- **Converge hacia las 15 000 iteraciones.** Después la pérdida ya no baja.
- **Lo que mejora es la velocidad.** La pérdida de seguimiento pasa de 1.60 a 0.73, la de choque apenas se
  mueve y los episodios con choque bajan de 2.7 % a 0.4 %. Es justo lo que le faltaba a la prueba de
  concepto, que esquivaba pero no corría.
- **El 0.4 % es en sus propios bosques**, con profundidad ideal más ruido. La prueba de verdad es su propio SGM
  en el lazo de R18 y las imágenes de Unity.

## Evaluación (30 entornos por velocidad, registro en `r25/eval_r18_y_flightmare.log`)

**Bosque de R18**, emparejado con el reactivo de R24 (`--keep --tilt --edge`) sobre las mismas semillas:

| piloto | 3 m/s | 5 m/s | 7 m/s | 10 m/s | choques | velocidad media real (3 · 5 · 7 · 10) |
|---|---|---|---|---|---|---|
| DiffPhys | 27/30 | 27/30 | 26/30 | 23/30 | 2 de 120 | 2.6 · 3.6 · 4.2 · 4.9 m/s |
| reactivo (R24) | 9/10 | 10/10 | 9/10 | 24/30 | | |

Emparejado: 3 entornos solo los pasa DiffPhys y 4 solo el reactivo ($p = 1$). Empatan en llegadas.

**Almacén de Flightmare**, sin reentrenar: la red nunca vio paredes, estanterías ni troncos cuadrados.

| piloto | 3 m/s | 5 m/s | 7 m/s | 10 m/s | total | choques por segundo a 10 m/s |
|---|---|---|---|---|---|---|
| DiffPhys | 26/30 | 28/30 | 28/30 | **27/30** | 109/120 | 1.0 % |
| reactivo (R24) | 29/30 | 29/30 | 26/30 | 20/30 | 104/120 | 9.3 % |

Emparejado a 10 m/s: 8 entornos solo los pasa DiffPhys y 1 solo el reactivo ($p = 0.04$). En total, 13 frente
a 8 ($p = 0.38$).

**Lo que dicen estos números:**
- **Casi no choca.** 2 choques en 120 vuelos en R18 y 5 en 120 en el almacén. Sus fallos son casi todos
  vuelos que se quedan sin tiempo.
- **Generaliza.** Entrenada solo con cilindros y suelo ideales, vuela con su propio SGM y con las imágenes de
  Unity mejor que el reactivo a 10 m/s, que es justo donde fallaba todo lo anterior.
- **Pero no vuela a la velocidad pedida.** Esto limita todo lo anterior: en R18, a 10 m/s, va a 4.9 m/s de
  media. En el almacén tarda 6.4 s en los 40 m (~6.2 m/s), y a 7 m/s tarda 7.3 s, frente a 5.8 s del
  reactivo. Parte de la ventaja a 10 m/s es ir más despacio. Todavía no se puede decir que bata a nada a
  10 m/s *reales*.

**Siguiente: por qué va lenta.** Hay tres sospechas, y cada una se separa de las otras con un experimento:
1. **Ya es lenta en su propio simulador.** Es un problema de entrenamiento: el peso del choque frente al
   seguimiento, o la velocidad promediada en 2 s.
2. **Es lenta solo con el SGM.** Los bordes y el ruido de la profundidad le parecen obstáculos cercanos.
3. **Es lenta en R18 aun con la profundidad exacta.** Es la diferencia entre la masa puntual y el cuadrotor
   con su controlador.

Después vienen la variante de entrada 8 × 8 (ToF) y la reparación por QP.
