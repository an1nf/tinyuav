# Entrenamiento con GPU

Solo el entrenamiento (`train`, `train-tartanair`, `r03`, `r11 train`) usa GPU. La generación de
datasets, el renderer, la simulación en lazo cerrado y la inferencia int8 corren en CPU (numpy +
TFLite). La CNN usa 1–2 GB de VRAM con batch 128, así que 8 GB sobran.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: usar WSL2 (TF-GPU no corre nativo)
pip install -e ".[train,tflite,dev]"
pip install "tensorflow[and-cuda]"                     # trae CUDA/cuDNN como paquetes nvidia-* de pip
nanouav doctor --gpu                                   # debe listar la GPU
```

### Si TensorFlow no encuentra las bibliotecas de CUDA

Síntoma: `Cannot dlopen some GPU libraries` y `nanouav doctor --gpu` muestra `GPUs: []`. Las ruedas
`nvidia-*` de pip traen CUDA/cuDNN, pero TensorFlow no siempre las encuentra (en particular con
`--system-site-packages`, donde `nvidia` puede estar repartido en varias carpetas). Lo más cómodo es
añadirlo al `activate` del entorno, una sola vez:

```bash
cat >> .venv/bin/activate <<'EOF'
# CUDA/cuDNN de los paquetes nvidia-* de pip → visibles para TensorFlow
export LD_LIBRARY_PATH="$(python -c "import nvidia,glob,os; print(':'.join(sorted(p for d in nvidia.__path__ for p in glob.glob(os.path.join(d,'*','lib')))))"):$LD_LIBRARY_PATH"
# ptxas de CUDA 12 (el de /usr/local/cuda 13 ya no compila para Pascal; sin esto solo hay avisos)
export PATH="$(python -c "import nvidia,glob; print(':'.join(g for p in nvidia.__path__ for g in glob.glob(p + '/cuda_nvcc/bin')))"):$PATH"
EOF
deactivate; source .venv/bin/activate
python -c "import tensorflow as tf; g = tf.config.list_physical_devices('GPU'); print(g); assert g"
```

### GPU Pascal (TITAN X, GTX 10xx)

En la TITAN X (Pascal, *compute capability* 6.1) el entrenamiento solo funcionó con esta
combinación:

1. **cuDNN fijado** a la versión con la que funcionó:
   `pip install nvidia-cudnn-cu12==9.3.0.75`. Las versiones posteriores dieron errores de cuDNN en
   esta GPU.
2. **Sin XLA**: los modelos se compilan con `jit_compile=False` (en `train_cnn.py`,
   `r3_lnca_train.py` y `r11_vel_net.py`, también para `predict()`), porque el autotuner de XLA no
   encuentra kernels de cuDNN fusionados en Pascal (`Autotuner could not find any supported configs`).
   Keras 3 activa XLA por defecto en GPU, incluso en un modelo sin compilar.

Probado en la TITAN X: TF 2.21.0, cuDNN 9.3.0.75, driver 580; M con 60 épocas × 30 000 imágenes
tarda ≈ 14 s por época (≈ 15 min).

Si un entrenamiento termina pero falla al evaluar, los pesos ya están guardados
(`models/model_<v>.keras` y `.tflite`); se evalúa sin reentrenar con
`nanouav run train --variants M --eval-only`.

En GPUs más recientes no debería hacer falta ninguna de las dos cosas.

## Lanzar

```bash
nanouav run train --variants XS,S,M,L --epochs 60 --n-train 30000 --batch 128 --lr 4e-3
nanouav run r03 --models M_ToF --init-from models/model_M.keras --install \
                --n-train 30000 --epochs 60 --batch 128 --lr 4e-3
```

En una suite, los trabajos de entrenamiento tienen que declarar `gpu: true` (el runner oculta la
GPU por defecto para que los trabajos de simulación en paralelo no reserven VRAM):

```yaml
jobs:
  - id: train-m-tof
    run: r03 --models M_ToF --init-from models/model_M.keras --install --epochs 60
    gpu: true
```
