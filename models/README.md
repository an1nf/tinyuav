# models/

Redes entrenadas. **No se versionan** (ver `docs/artefactos.md`).

```bash
nanouav models import ~/Project_UAV/uav_v2/_03_cnn    # traer una copia existente
nanouav run train --variants XS,S,M,L --epochs 60     # o reentrenar
nanouav models check
```

| archivo | qué es |
|---|---|
| `model_{XS,S,M,L}.tflite` | CNN de profundidad int8 96×96 (la que corre en el ESP32-S3) |
| `model_{XS,S,M,L}.keras` | pesos en float (punto de partida para afinar) |
| `model.tflite` | alias de M (compatibilidad) |
| `model_M_ToF.tflite` | M con fusión temprana del ToF (`nanouav run r03 ... --install`) |
| `metrics.json` | AbsRel, recall < 1 m, arena, latencia de cada variante |
| `r3/` | salidas del experimento R3 (LNCA, M_ToF) |
| `r11_vel_*.keras` | redes de estimación de velocidad (R11) |
