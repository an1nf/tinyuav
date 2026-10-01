> **Documento histórico** de la estructura plana anterior (`uav_v2`/`uav_v3`). Los comandos y rutas
> ya no son válidos; ver [docs/migracion.md](../../migracion.md) para su equivalente con `nanouav`.

# 04 · Política de decisión (P1/P2/P3)

## Qué es

Define **cuándo** ejecutar la CNN y **qué hacer** con su salida.
Es la lógica de control de percepción, independiente del algoritmo de vuelo.

## Las tres políticas

### P1 — Siempre activa
La CNN corre en cada frame, sin importar la situación.
- Pro: máxima información
- Contra: consume batería constante, jitter alto siempre

### P2 — Basada en ToF
La CNN se activa cuando el ToF detecta algo a menos de `tof_threshold` metros.
- Pro: simple, determinista
- Contra: el ToF tiene 8×8 de resolución; puede perder obstáculos pequeños

### P3 — Basada en flujo óptico *(la más interesante del proyecto)*
La CNN se activa cuando la divergencia del flujo óptico supera un umbral.
- Pro: detecta movimiento relativo antes que el ToF; ahorra energía en vuelo libre
- Contra: requiere un frame previo; ruido en flujo óptico puede causar falsos positivos

## Por qué P3 es la contribución original

En la literatura, las CNN en nano-drones corren siempre (PULP-Dronet familia).
La idea de **disparar condicionalmente la CNN según divergencia de flujo óptico**
no está publicada para el régimen de MCU de <10 USD. Es el experimento E0-C.

## Salida de decisión

```python
Decision(action, confidence, triggered_by)
action: 'libre' | 'precaucion' | 'evadir'
triggered_by: 'none' | 'tof' | 'flow' | 'cnn'
```

## Umbral de divergencia (v2)
La divergencia ahora es `tr(A)` de un ajuste afín del flujo (no `mean|vx+vy|`).
Para acercamiento frontal `tr(A) ≈ 2/TTC` [1/frame] → el umbral sale del
tiempo a colisión: `DIV_THRESHOLD = 2/(3 s · 10 fps) ≈ 0.067`.

## evasion.py — planificador reactivo (reemplaza la semielipse de v11)
1. Retro-proyecta profundidad calibrada (96×96, cada 3 px) y ToF (8×8, peso doble) a 3D
   con la pose del instante de captura.
2. Descarta suelo (z < 0.15 m) y lo que queda fuera de la banda de altura del dron (±0.30 m).
3. Grilla 2D local (0.1 m) con memoria: los obstáculos que salen del campo visual al
   pasar a su lado NO se olvidan; los fantasmas se borran con los rayos libres.
4. Perfil lateral y(x) por programación dinámica (4 m adelante, carril ±2.3 m,
   pendiente ≤ 0.8, histéresis con el plan anterior, paso lateral puro si está bloqueado).
5. Filtro de seguridad tipo barrera: la velocidad hacia cada celda ocupada cercana no
   puede superar la que permite frenar considerando 0.4 s de reacción:
   `v·τ + v²/(2a) ≤ holgura`.
6. Solo entrega objetivos `(v, yaw)`; la suavidad la impone `_06_control.ReferenceGenerator`.
