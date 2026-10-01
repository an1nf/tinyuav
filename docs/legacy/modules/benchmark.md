> **Documento histórico** de la estructura plana anterior (`uav_v2`/`uav_v3`). Los comandos y rutas
> ya no son válidos; ver [docs/migracion.md](../../migracion.md) para su equivalente con `nanouav`.

# 05 · Benchmark: jitter, energía y distancia segura

## Qué mide

Corre N frames del pipeline completo con cada variante de modelo y cada política,
y produce métricas estadísticas comparativas.

## Métricas

### Jitter PID (µs)
Variación en el tiempo de ejecución del bucle de control.
- **p50**: mediana (comportamiento típico)
- **p99**: percentil 99 (peor caso casi-siempre)
- **límite**: 200 µs → encima de esto el dron puede perder estabilidad

*Estos valores son estimaciones de literatura. El E3-A los medirá con hardware real.*

### Distancia de detección segura (cm)
A velocidad `v` m/s, con `fps` frames/s y latencia de reacción mecánica `t_r`:

```
d_safe = v × (1/fps + t_r)
```

A 0.5 m/s, con 9 FPS y 50 ms de reacción → d_safe ≈ 8 cm.
Si el obstáculo está a menos de 8 cm cuando se detecta, no hay tiempo de frenar.

### Autonomía de vuelo (min)
```
E_batería = mAh × V × 3600   [Joules]
P_cnn = E_frame × fps
P_total = P_base + P_cnn
t = E_batería / P_total / 60
```

### CNN activations %
Con P3: qué porcentaje de frames disparó realmente la CNN.
Objetivo: <40% en entornos abiertos → ahorro energético significativo.

## Archivos generados

- `assets/benchmark_jitter.png`
- `assets/benchmark_tradeoff.png`
- `assets/benchmark_policy.png`
- `assets/benchmark_energy.png`
