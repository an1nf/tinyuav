# Artefactos: modelos, datos y resultados

En git solo van código, configuraciones, documentación y la referencia numérica de los tests
(`tests/data/`). Todo lo que se genera o se descarga vive en tres carpetas ignoradas por git, cada
una con su README:

| carpeta | contenido | variable para reubicarla |
|---|---|---|
| `models/` | redes entrenadas: `model_{XS,S,M,L,M_ToF}.tflite`, `.keras`, `metrics.json`, `r3/`, `r11_vel_*.keras` | `NANOUAV_MODELS` |
| `data/` | datasets y cachés: `synth_v2.npz`, `synth_r3.npz`, `r11_*.npz`, `tartanair/` | `NANOUAV_DATA` |
| `results/` | salidas de experimentos y suites (`json`, `png`, `logs/`, `gz_runs/`, `gz_logs/`) | `NANOUAV_RESULTS` |

`nanouav doctor` muestra las rutas efectivas. Con las variables se puede, por ejemplo, tener los
datasets en otro disco (`export NANOUAV_DATA=/mnt/datos/nanouav`) o separar los resultados de una
campaña (`nanouav run --results results/campaña_x ...`).

## Modelos

### Importar los existentes

```bash
nanouav models import ~/Project_UAV/uav_v2/_03_cnn
nanouav models list
nanouav models check            # carga model_M.tflite con el runtime TFLite (pip install -e ".[tflite]")
```

`import` copia `model_*.tflite`, `model_*.keras`, `model.tflite`, `metrics.json`, `r11_vel_*.keras`
y la carpeta `r3/`. No borra nada de origen.

### Reentrenar

| modelo | comando | salida |
|---|---|---|
| XS, S, M, L | `nanouav run train --variants XS,S,M,L --epochs 60 --n-train 30000 --batch 128 --lr 4e-3` | `models/model_*.tflite`, `.keras`, `metrics.json`; figuras en `results/` |
| con TartanAir | añadir `--tartanair` (datos en `data/tartanair/`, ver abajo) | ídem |
| M_ToF (fusión temprana) | `nanouav run r03 --models M_ToF --init-from models/model_M.keras --install --n-train 30000 --epochs 60 --batch 128 --lr 4e-3` | `models/r3/M_ToF.tflite` y, con `--install`, `models/model_M_ToF.tflite` |
| con estructuras finas (R14) | `nanouav run train --variants M --thin 0.3 --suffix _thin ...` y luego `nanouav run r03 --models M_ToF --thin 0.3 --suffix _thin --init-from models/model_M_thin.keras --install ...` | `models/model_M_thin.*`, `models/model_M_ToF_thin.*` (los originales no se tocan) |
| LNCA | `nanouav run r03 --models M,LNCA24,LNCA24_ToF,LNCA32ds_ToF ...` | `models/r3/*.tflite`, `results/r3_lnca_train.json` |
| velocidad (R11) | `nanouav run r11 train --data data/r11_big.npz --epochs 80` | `models/r11_vel_*.keras`, `results/r11_vel_net.json` |

Criterios de aceptación de una CNN para el ESP32-S3: arena ≤ 180 KB, int8 ≈ float
(ΔAbsRel < 0.01), recall de obstáculo a < 1 m ≥ 0.85. `metrics.json` guarda estas cifras.
Entrenamiento con GPU: [gpu.md](gpu.md).

`--thin` solo cambia los datos de entrenamiento y validación; el conjunto de prueba es siempre el
original, así que las cifras de `metrics.json` siguen siendo comparables. El rendimiento sobre
postes y cables se mide con `nanouav run r14 --variants M_ToF,M_ToF_thin`. Para evaluar un modelo
ya entrenado sin reentrenarlo: `nanouav run train --variants M --eval-only` (con `--suffix` si aplica).

### Sin modelos

Los experimentos de lazo cerrado pueden correr con profundidad *fake* (ground truth + ruido
gaussiano por variante) con `--fake-cnn` / `NANOUAV_FAKE_CNN=1`. Sirve como cota superior de
percepción y para depurar el planificador, no para resultados de percepción.

## Datos

Todos los datasets sintéticos se regeneran de forma determinista desde la semilla:

| archivo | lo genera | tiempo aprox. |
|---|---|---|
| `data/synth_v2.npz` | `nanouav run train` la primera vez (borrarlo para regenerar con otro `--n-train`) | 5–10 min CPU para 30 k |
| `data/synth_r3.npz` | `nanouav run r03` la primera vez | ídem |
| `data/synth_v2_thin0.3.npz`, `data/synth_r3_thin0.3.npz` | `--thin 0.3` en `train` / `r03` (30 % de las escenas con postes de 1–5 cm y cables) | ídem |
| `data/r11_*.npz` | `nanouav run r11 gen --n 20000 --seed 1 --out data/r11_big.npz` (paralelizable por semilla) | ~20 min CPU |
| `data/r11_full_*.npz` | `nanouav run r11 genfull --n 800 --seed 33 --out data/r11_full_a.npz` | minutos |
| `data/tartanair/` | descarga manual de `moge-dataset-tartanair.zip` a `data/tartanair/`; se extrae solo | — |

## Resultados

Cada experimento escribe en `results/` (o en la carpeta de su suite). Para conservar o compartir
resultados de una campaña sin meterlos en git:

* los resúmenes que se citan en un informe se copian como tabla al `docs/research/*.md` correspondiente,
  junto con el comando exacto y el commit;
* las carpetas completas se archivan fuera del repositorio (`tar czf baseline_2026-10.tgz results/baseline`).

`jobs.json` de cada suite registra la línea de comando, el entorno y el código de salida de cada
trabajo, lo que basta para reproducir cualquier archivo con el mismo commit.
