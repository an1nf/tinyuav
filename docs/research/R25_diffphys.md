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

## Siguiente: entrenar en la GPU

```bash
nanouav run r25 train --out results/r25 --iters 20000 --batch 64
nanouav run r18 eval --pilots diffphys --model results/r25/diffphys.keras --v 3,5,7,10 --n 10
```

Hay que batir:
- el reactivo de R24 (37/40, 24/30 a 10 m/s);
- nuestro estudiante de Loquercio (28/40).

Después vienen la variante de entrada 8 × 8 (ToF) y Flightmare.
