> **Documento histórico** de la estructura plana anterior (`uav_v2`/`uav_v3`). Los comandos y rutas
> ya no son válidos; ver [docs/migracion.md](../migracion.md) para su equivalente con `nanouav`.

# Pruebas en el equipo local (GPU 12 GB + pesos entrenados de v2)

Todo corre desde la raíz del proyecto. **Solo el entrenamiento usa GPU**; las simulaciones de
lazo cerrado son CPU (una por núcleo) y pueden correr mientras la GPU entrena.
En Windows usar WSL2 (TensorFlow con GPU no corre nativo en Windows).

## 0. Preparación (una vez)

```bash
unzip uav_v3r5.zip -d uav_v3r && cd uav_v3r
python -m venv .venv && source .venv/bin/activate
pip install "tensorflow[and-cuda]" ai-edge-litert opencv-python matplotlib scipy
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # debe listar la GPU

# Traer los pesos entrenados de v2 (misma arquitectura XS/S/M/L; ajustar la ruta)
V2=~/ruta/a/uav_v2
cp $V2/_03_cnn/model_*.tflite $V2/_03_cnn/model_*.keras $V2/_03_cnn/metrics.json _03_cnn/
cp $V2/_03_cnn/model.tflite _03_cnn/ 2>/dev/null
cp $V2/data/*.npz data/ 2>/dev/null          # cachés de dataset, si existen (ahorra tiempo)

export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3
python e3_closed_loop.py --profile nano --n 2 --only-mc      # humo: debe decir "Modo REAL · model_M.tflite"
```

## 1. GPU — entrenar M_ToF (fusión temprana ToF, R3) partiendo de tu M

M_ToF = tu M + una capa 1×1 que inyecta el ToF 8×8 en el nivel 12×12. La capa arranca en cero,
así que el modelo parte **idéntico a tu M**; el ajuste fino solo aprende a usar el ToF.
El dataset R3 (30 k escenas con ToF simulado) se genera en CPU la primera vez (~10 min).

```bash
# 1a. Control justo: tu M con el mismo ajuste fino pero SIN ToF (no instala nada)
python r3_lnca_train.py --models M --init-from _03_cnn/model_M.keras \
       --n-train 30000 --epochs 30 --batch 128 --lr 1e-3 --out-json assets/r3_ft.json

# 1b. M_ToF desde tu M; --install lo deja en _03_cnn/model_M_ToF.tflite para el lazo cerrado
python r3_lnca_train.py --models M_ToF --init-from _03_cnn/model_M.keras \
       --n-train 30000 --epochs 30 --batch 128 --lr 1e-3 --install --out-json assets/r3_ft.json

python summarize_results.py r3_ft        # M vs M+cal vs M_ToF (test y Gazebo)
```
Criterio para adoptar M_ToF: recall < 1 m ≥ M+cal, error de distancia mínima menor, arena ≤ 180 KB
(el log de R3 imprime la arena).

Opcional (paper C1): comparación desde cero con el mismo presupuesto, incluida la LNCA.
```bash
python r3_lnca_train.py --models M,M_ToF,LNCA24,LNCA24_ToF,LNCA32ds_ToF \
       --n-train 30000 --epochs 60 --batch 128 --lr 4e-3 --out-json assets/r3_scratch.json
```

## 2. CPU — simulaciones con n grande (en paralelo, mientras la GPU entrena)

`run_jobs.sh` reparte las líneas de un archivo de trabajos entre los núcleos (por defecto,
núcleos − 1), con la GPU oculta a las simulaciones. Logs en `logs/`.

```bash
bash run_jobs.sh jobs_A.txt          # con tu M de v2: E3 n=30, poste, corredores n=50, dinámicos, ruido
# cuando termine el paso 1b (existe _03_cnn/model_M_ToF.tflite):
bash run_jobs.sh jobs_B.txt          # las mismas pruebas con M_ToF
python summarize_results.py          # tabla de todo con IC95 de Wilson
```

Qué contiene cada lista (editable):

| Trabajo | Pregunta |
|---|---|
| `e3_closed_loop.py --n 30` (nano, x3) | Re-validar E3 con tus pesos definitivos |
| `r5_pole.py` brake / consistent / consistent+GT | ¿Se sostiene 30/30 del modo consistente con n = 60? |
| `r6_corridor_consistent.py` n = 50 (nano, x3) | Resultado principal del paper C2 (restricciones consistentes) |
| `r10_dynamic.py` head/cross | Dinámicos con capa ERF/IRF, n = 20 por velocidad |
| `r8_state_noise.py --n 30` | Curva éxito frente a ruido/sesgo de estimación |
| `jobs_B.txt` | Lo mismo con M_ToF → ¿mejora la percepción en vuelo, no solo offline? |

Tiempo aproximado: cada episodio tarda 15–40 s en un núcleo. `jobs_A` ≈ 1 000 episodios
≈ 6–10 h en un núcleo → con 8 núcleos, ~1 h.

## 3. Opcional — velocidad aprendida con más datos (R11)

```bash
for s in 1 2 3 4; do python r11_vel_net.py gen --n 5000 --seed $s --out data/r11_$s.npz & done; wait
python r11_vel_net.py train --data data/r11_1.npz,data/r11_2.npz,data/r11_3.npz,data/r11_4.npz --epochs 80
```
Con 5.8 k secuencias las redes no aprendían (R11). Esto dice si era falta de datos.

## 4. Qué devolverme

`assets/*.json` (o la salida de `python summarize_results.py` completa) y, si algo falla,
el log correspondiente de `logs/`.
