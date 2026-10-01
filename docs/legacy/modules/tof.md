> **Documento histórico** de la estructura plana anterior (`uav_v2`/`uav_v3`). Los comandos y rutas
> ya no son válidos; ver [docs/migracion.md](../../migracion.md) para su equivalente con `nanouav`.

# 01 · Sensor ToF 8×8 (VL53L5CX)

## Qué es

Simula la salida del VL53L5CX: una grilla de 8×8 mediciones de distancia.
El sensor real divide su campo visual en 64 zonas y mide el tiempo que tarda
un pulso infrarrojo en rebotar en cada una.

## Qué NO es

El sensor real tiene ruido no-gaussiano, zonas ciegas en superficies
reflectantes/transparentes, y una frecuencia máxima de ~15 Hz en modo 8×8.
Esta simulación usa ruido gaussiano simple.

## Por qué importa

El ToF da distancias absolutas (en metros) pero con resolución espacial muy baja.
La CNN da resolución espacial alta pero distancias relativas (sin escala).
La idea central del proyecto: **usar el ToF como "maestro" para calibrar la CNN**.

## Modelo de ruido

```
d_medido = d_real + N(0, sigma) + bias_zona
sigma ≈ 0.02 m en zona cercana, 0.08 m a 3+ m
```

## Salida

Array float32 (8, 8) con distancias en metros.
