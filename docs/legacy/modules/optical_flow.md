> **Documento histórico** de la estructura plana anterior (`uav_v2`/`uav_v3`). Los comandos y rutas
> ya no son válidos; ver [docs/migracion.md](../../migracion.md) para su equivalente con `nanouav`.

# 02 · Flujo óptico (Lucas-Kanade)

## Qué es

Calcula vectores de movimiento entre dos frames consecutivos usando el algoritmo
de Lucas-Kanade con pirámides (implementación real de OpenCV, no simulada).

## Qué NO es

En el ESP32-S3 real no correrías Lucas-Kanade completo: es demasiado costoso.
Usarías un sensor de flujo óptico dedicado (PMW3901) o una versión muy reducida.
Este módulo sirve para validar la **lógica de decisión** basada en divergencia.

## Concepto: divergencia

Imagina que vuelas hacia una pared. Los píxeles en el centro del frame
parecen "expandirse" hacia los bordes: eso es divergencia positiva.

```
Frame 1:    Frame 2 (más cerca):
  [ · ]       [  ·  ]
  [ · ]      [ · · · ]
  [ · ]       [  ·  ]
```

Divergencia alta → obstáculo cercano → activa la CNN.
Divergencia baja → espacio libre → CNN en reposo (ahorra energía).

## Umbral de activación

`DIV_THRESHOLD = 0.15` (empírico, ajustable).
Por debajo: la CNN no se ejecuta. Por encima: se ejecuta la CNN.

## Salidas

- `flow_pts`: lista de (x0,y0,x1,y1) — vectores de flujo
- `divergence`: float — magnitud promedio de movimiento
